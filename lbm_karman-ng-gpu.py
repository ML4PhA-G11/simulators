"""
Lattice Boltzmann Method (LBM) — Karman Vortex Street Solver (NG, GPU)
======================================================================
GPU port of lbm_karman-ng.py using the Taichi programming language.
D2Q9 lattice, BGK collision.  Solves flow past a circular cylinder
in a channel following the geometry of Assignment 3.1 (Figure 4):

    Domain   2.2 m × 0.41 m
    Cylinder center (0.2, 0.2) m,  diameter 0.10 m
    Walls    no-slip (top, bottom, cylinder)
    Inlet    Zou-He velocity (uniform)
    Outlet   Zou-He pressure (rho = 1)

The numerics (initial condition, collision, streaming, bounce-back with
momentum-exchange force, Zou-He inlet/outlet) match the NumPy/Numba
reference exactly so the CSV/PNG outputs are directly comparable.

Targeted at NVIDIA H100 (arch=cuda). Other Taichi backends are available
via --arch (cpu, gpu, vulkan, metal).

Dependencies: taichi, numpy, matplotlib, tqdm

Usage:
  python lbm_karman-ng-gpu.py [OPTIONS]

Examples:
  python lbm_karman-ng-gpu.py                            # baseline Re=150, CUDA
  python lbm_karman-ng-gpu.py --Re 200 --n-steps 50000
  python lbm_karman-ng-gpu.py --animate --plot-mode vorticity
  python lbm_karman-ng-gpu.py --arch cpu                 # fallback to CPU
"""

import argparse
import math
import os
import numpy as np
import matplotlib
import taichi as ti
from tqdm import tqdm

# =============================================================================
# 1.  D2Q9 Lattice Definition
# =============================================================================
#
#   6  2  5
#    \ | /
#   3--0--1
#    / | \
#   7  4  8
#
# Constants are kept as plain Python tuples so they can be unrolled inside
# Taichi kernels via `ti.static(range(9))` — that gives the compiler
# compile-time integer offsets instead of indirect field reads.

C = (
    (0, 0),    # 0  rest
    (1, 0),    # 1  east
    (0, 1),    # 2  north
    (-1, 0),   # 3  west
    (0, -1),   # 4  south
    (1, 1),    # 5  north-east
    (-1, 1),   # 6  north-west
    (-1, -1),  # 7  south-west
    (1, -1),   # 8  south-east
)

W = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)
NDIR = 9

# NumPy mirrors of C/OPP, used by the host-side parameter setup and plotting.
c_np = np.array(C, dtype=np.int32)
opp_np = np.array(OPP, dtype=np.int32)


# =============================================================================
# 2.  Command-line Interface
# =============================================================================


def parse_args(argv=None):
    """Parse command-line arguments for the LBM simulation."""
    p = argparse.ArgumentParser(
        description="LBM Karman Vortex Street Solver — NG GPU (Taichi, D2Q9, BGK)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Domain (physical)")
    g.add_argument("--resolution", type=int, default=250, help="Lattice points per meter")
    g.add_argument("--L", type=float, default=2.2, help="Domain length [m]")
    g.add_argument("--H", type=float, default=0.41, help="Domain height [m]")

    g = p.add_argument_group("Cylinder")
    g.add_argument("--cx", type=float, default=0.2, help="Cylinder center x [m]")
    g.add_argument("--cy", type=float, default=0.2, help="Cylinder center y [m]")
    g.add_argument("--r", type=float, default=0.05, help="Cylinder radius [m]")

    g = p.add_argument_group("Flow parameters")
    g.add_argument("--U-inlet", type=float, default=0.12, help="Inlet velocity (lattice units, keep << 1)")
    g.add_argument("--Re", type=float, default=150, help="Reynolds number")

    g = p.add_argument_group("Simulation control")
    g.add_argument("--n-steps", type=int, default=30000, help="Total number of timesteps")
    g.add_argument("--plot-every", type=int, default=25, help="Plot interval (steps, for --animate)")
    g.add_argument(
        "--plot-mode", choices=["velocity", "vorticity", "none"], default="velocity", help="Visualization mode"
    )

    g = p.add_argument_group("Output")
    g.add_argument("--animate", action="store_true", default=False, help="Show live matplotlib animation window")
    g.add_argument("--save-every", type=int, default=0, help="Save a snapshot image every N steps (0 = off)")
    g.add_argument("--out-dir", type=str, default="output", help="Directory for saved images")
    g.add_argument("--csv-every", type=int, default=1, help="Write forces to CSV every N steps (0 = off)")

    g = p.add_argument_group("Compute")
    g.add_argument(
        "--arch",
        choices=["cuda", "gpu", "cpu", "vulkan", "metal", "opengl"],
        default="cuda",
        help="Taichi backend (cuda targets NVIDIA H100)",
    )
    g.add_argument(
        "--fp",
        choices=["f32", "f64"],
        default="f32",
        help="Default floating-point precision (f32 is much faster on H100)",
    )

    return p.parse_args(argv)


def build_params(args):
    """Derive lattice parameters from parsed CLI arguments."""
    res = args.resolution
    Nx = int(round(args.L * res))
    Ny = int(round(args.H * res))
    cx_cyl = int(round(args.cx * res))
    cy_cyl = int(round(args.cy * res))
    r_cyl = int(round(args.r * res))

    U_inlet = args.U_inlet
    Re = args.Re
    D = 2 * r_cyl
    nu = U_inlet * D / Re
    tau = 3.0 * nu + 0.5

    # Obstacle mask (boolean on host; pushed to int32 field on device).
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    obstacle = (X - cx_cyl) ** 2 + (Y - cy_cyl) ** 2 <= r_cyl**2

    print(f"Simulation parameters:")
    print(f"  Physical:  {args.L}m x {args.H}m,  resolution={res} lu/m")
    print(f"  Grid:      {Nx} x {Ny}")
    print(f"  Cylinder:  center=({cx_cyl},{cy_cyl}), r={r_cyl}, D={D}")
    print(f"  Re={Re},  U_inlet={U_inlet},  nu={nu:.6f},  tau={tau:.4f}")

    return dict(
        Nx=Nx,
        Ny=Ny,
        cx_cyl=cx_cyl,
        cy_cyl=cy_cyl,
        r_cyl=r_cyl,
        U_inlet=U_inlet,
        Re=Re,
        D=D,
        nu=nu,
        tau=tau,
        obstacle=obstacle,
        X=X,
        Y=Y,
        n_steps=args.n_steps,
        plot_every=args.plot_every,
        plot_mode=args.plot_mode,
        animate=args.animate,
        save_every=args.save_every,
        out_dir=args.out_dir,
        csv_every=args.csv_every,
        arch=args.arch,
        fp=args.fp,
    )


# =============================================================================
# 3.  Main Simulation
# =============================================================================


def _select_arch(arch_name):
    return {
        "cuda": ti.cuda,
        "gpu": ti.gpu,
        "cpu": ti.cpu,
        "vulkan": ti.vulkan,
        "metal": ti.metal,
        "opengl": ti.opengl,
    }[arch_name]


def main(params):
    """Run the LBM simulation on a Taichi backend."""

    # -- Unpack parameters --
    Nx = params["Nx"]
    Ny = params["Ny"]
    U_inlet = params["U_inlet"]
    tau = params["tau"]
    obstacle_np = params["obstacle"]
    Y = params["Y"]
    n_steps = params["n_steps"]
    plot_every = params["plot_every"]
    plot_mode = params["plot_mode"]
    animate = params["animate"]
    save_every = params["save_every"]
    out_dir = params["out_dir"]
    csv_every = params["csv_every"]
    arch_name = params["arch"]
    fp_name = params["fp"]

    fp_dtype = ti.f32 if fp_name == "f32" else ti.f64
    ti.init(arch=_select_arch(arch_name), default_fp=fp_dtype, default_ip=ti.i32)

    if not animate:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # ==================================================================
    # 3a.  Device fields
    # ==================================================================
    f = ti.field(fp_dtype, shape=(Nx, Ny, NDIR))
    f_out = ti.field(fp_dtype, shape=(Nx, Ny, NDIR))
    rho = ti.field(fp_dtype, shape=(Nx, Ny))
    ux = ti.field(fp_dtype, shape=(Nx, Ny))
    uy = ti.field(fp_dtype, shape=(Nx, Ny))
    obstacle = ti.field(ti.i32, shape=(Nx, Ny))
    Fx_field = ti.field(fp_dtype, shape=())
    Fy_field = ti.field(fp_dtype, shape=())

    obstacle.from_numpy(obstacle_np.astype(np.int32))

    # ==================================================================
    # 3b.  Kernels
    # ==================================================================
    #
    # Field/grid dimensions and the D2Q9 constants are captured at kernel
    # compile time, so the inner loops over the 9 directions can be fully
    # unrolled via ti.static.

    @ti.kernel
    def init_fields(U_inlet_v: fp_dtype):
        two_pi = 2.0 * math.pi
        for i, j in ti.ndrange(Nx, Ny):
            rho[i, j] = 1.0
            u = U_inlet_v
            v = 0.001 * U_inlet_v * ti.sin(two_pi * j / Ny)
            if obstacle[i, j] == 1 or j == 0 or j == Ny - 1:
                u = 0.0
                v = 0.0
            ux[i, j] = u
            uy[i, j] = v
            usqr = u * u + v * v
            for k in ti.static(range(NDIR)):
                cu = C[k][0] * u + C[k][1] * v
                f[i, j, k] = W[k] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)

    @ti.kernel
    def compute_macroscopic():
        for i, j in ti.ndrange(Nx, Ny):
            r = 0.0
            mx = 0.0
            my = 0.0
            for k in ti.static(range(NDIR)):
                fk = f[i, j, k]
                r += fk
                mx += fk * C[k][0]
                my += fk * C[k][1]
            rho[i, j] = r
            ux[i, j] = mx / r
            uy[i, j] = my / r

    @ti.kernel
    def collide(tau_v: fp_dtype):
        # BGK: f_out = f - (f - feq) / tau, with the D2Q9 equilibrium
        #   feq_i = w_i rho (1 + 3 c_i·u + 4.5 (c_i·u)^2 - 1.5 |u|^2)
        inv_tau = 1.0 / tau_v
        for i, j in ti.ndrange(Nx, Ny):
            u = ux[i, j]
            v = uy[i, j]
            r = rho[i, j]
            usqr = u * u + v * v
            for k in ti.static(range(NDIR)):
                cu = C[k][0] * u + C[k][1] * v
                feq = W[k] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)
                f_out[i, j, k] = f[i, j, k] - (f[i, j, k] - feq) * inv_tau

    @ti.kernel
    def bounce_back_and_force():
        # Momentum-exchange method: the force on the solid equals the change
        # in momentum of the distributions at obstacle nodes during the
        # bounce-back step. Equivalent to the NumPy version's
        #   diff = f_out_pre_bb - f_new;  F = sum(diff * c).
        Fx_field[None] = 0.0
        Fy_field[None] = 0.0
        for i, j in ti.ndrange(Nx, Ny):
            if obstacle[i, j] == 1:
                for k in ti.static(range(NDIR)):
                    f_pre = f_out[i, j, k]
                    f_new = f[i, j, OPP[k]]
                    d = f_pre - f_new
                    ti.atomic_add(Fx_field[None], d * C[k][0])
                    ti.atomic_add(Fy_field[None], d * C[k][1])
                    f_out[i, j, k] = f_new

    @ti.kernel
    def stream():
        # Periodic streaming equivalent to np.roll(f_out[..., k], c[k]) per
        # direction, expressed as a pull: each destination cell reads from
        # the upstream neighbour given by -c[k].
        for i, j in ti.ndrange(Nx, Ny):
            for k in ti.static(range(NDIR)):
                ip = (i - C[k][0] + Nx) % Nx
                jp = (j - C[k][1] + Ny) % Ny
                f[i, j, k] = f_out[ip, jp, k]

    @ti.kernel
    def apply_wall_bc():
        # Top/bottom no-slip bounce-back, applied before inlet/outlet so
        # Zou-He reads valid wall values.
        for i in range(Nx):
            f[i, 0, 2] = f_out[i, 0, 4]   # north  <- south
            f[i, 0, 5] = f_out[i, 0, 7]   # NE     <- SW
            f[i, 0, 6] = f_out[i, 0, 8]   # NW     <- SE
            f[i, Ny - 1, 4] = f_out[i, Ny - 1, 2]  # south <- north
            f[i, Ny - 1, 7] = f_out[i, Ny - 1, 5]  # SW    <- NE
            f[i, Ny - 1, 8] = f_out[i, Ny - 1, 6]  # SE    <- NW

    @ti.kernel
    def apply_outlet_bc():
        # Zou-He pressure outlet (rho = 1) on interior y; corners handled
        # separately with a zero-gradient fallback.
        rho_out = 1.0
        for j in range(1, Ny - 1):
            i = Nx - 1
            uxo = -1.0 + (
                f[i, j, 0] + f[i, j, 2] + f[i, j, 4]
                + 2.0 * (f[i, j, 1] + f[i, j, 5] + f[i, j, 8])
            ) / rho_out
            if uxo < 0.0:
                uxo = 0.0
            if uxo > 0.5:
                uxo = 0.5
            f[i, j, 3] = f[i, j, 1] - (2.0 / 3.0) * rho_out * uxo
            f[i, j, 7] = f[i, j, 5] + 0.5 * (f[i, j, 2] - f[i, j, 4]) - (1.0 / 6.0) * rho_out * uxo
            f[i, j, 6] = f[i, j, 8] - 0.5 * (f[i, j, 2] - f[i, j, 4]) - (1.0 / 6.0) * rho_out * uxo

    @ti.kernel
    def apply_outlet_corners():
        for k in range(2):
            yc = 0 if k == 0 else Ny - 1
            f[Nx - 1, yc, 3] = f[Nx - 2, yc, 3]
            f[Nx - 1, yc, 6] = f[Nx - 2, yc, 6]
            f[Nx - 1, yc, 7] = f[Nx - 2, yc, 7]

    @ti.kernel
    def apply_inlet_bc(U_inlet_v: fp_dtype):
        # Zou-He fixed-velocity inlet (ux = U_inlet, uy = 0).
        for j in range(Ny):
            i = 0
            rho_in = (
                f[i, j, 0] + f[i, j, 2] + f[i, j, 4]
                + 2.0 * (f[i, j, 3] + f[i, j, 6] + f[i, j, 7])
            ) / (1.0 - U_inlet_v)
            f[i, j, 1] = f[i, j, 3] + (2.0 / 3.0) * rho_in * U_inlet_v
            f[i, j, 5] = f[i, j, 7] - 0.5 * (f[i, j, 2] - f[i, j, 4]) + (1.0 / 6.0) * rho_in * U_inlet_v
            f[i, j, 8] = f[i, j, 6] + 0.5 * (f[i, j, 2] - f[i, j, 4]) + (1.0 / 6.0) * rho_in * U_inlet_v

    # ==================================================================
    # 3c.  CSV force output
    # ==================================================================
    csv_file = None
    csv_path = None
    if csv_every > 0:
        csv_path = os.path.join(out_dir, "forces.csv")
        csv_file = open(csv_path, "w")
        csv_file.write(
            f"# Re={params['Re']} U_inlet={U_inlet} "
            f"D={params['D']} Nx={Nx} Ny={Ny} "
            f"nu={params['nu']:.6f} tau={tau:.4f}\n"
        )
        csv_file.write("step,time,drag,lift\n")

    # ==================================================================
    # 3d.  Initialization
    # ==================================================================
    init_fields(U_inlet)

    # ==================================================================
    # 3e.  Visualisation setup
    # ==================================================================
    if animate:
        plt.ion()
    fig, (ax_flow, ax_force) = plt.subplots(2, 1, figsize=(10, 6), dpi=120, gridspec_kw={"height_ratios": [2, 1]})

    def plot_velocity(ux_h, uy_h, step):
        speed = np.sqrt(ux_h**2 + uy_h**2)
        speed[obstacle_np] = np.nan
        ax_flow.clear()
        ax_flow.imshow(
            speed.T, origin="lower", cmap="jet", vmin=0, vmax=U_inlet * 2.0, aspect="auto", extent=[0, Nx, 0, Ny]
        )
        ax_flow.set_title(f"Velocity magnitude — step {step}")
        ax_flow.set_xlabel("x")
        ax_flow.set_ylabel("y")

    def plot_vorticity(ux_h, uy_h, step):
        vort = (
            np.roll(uy_h, -1, axis=0) - np.roll(uy_h, 1, axis=0)
            - np.roll(ux_h, -1, axis=1) + np.roll(ux_h, 1, axis=1)
        )
        vort[obstacle_np] = np.nan
        ax_flow.clear()
        ax_flow.imshow(
            vort.T, origin="lower", cmap="RdBu_r", vmin=-0.04, vmax=0.04, aspect="auto", extent=[0, Nx, 0, Ny]
        )
        ax_flow.set_title(f"Vorticity field — step {step}")
        ax_flow.set_xlabel("x")
        ax_flow.set_ylabel("y")

    def plot_field(ux_h, uy_h, step, save=False):
        if plot_mode == "vorticity":
            plot_vorticity(ux_h, uy_h, step)
        elif plot_mode == "velocity":
            plot_velocity(ux_h, uy_h, step)
        else:
            return

        ax_force.clear()
        s = np.arange(1, step + 1)
        ax_force.plot(s, drag_history[:step], label="Drag (Fx)", lw=0.8)
        ax_force.plot(s, lift_history[:step], label="Lift (Fy)", lw=0.8)
        ax_force.set_xlim(1, n_steps)
        ax_force.set_xlabel("Timestep")
        ax_force.set_ylabel("Force (lattice units)")
        ax_force.legend(loc="upper right")
        ax_force.grid(True, alpha=0.3)

        skip = max(100, step // 10)
        if step > skip:
            d_vis = drag_history[skip:step]
            l_vis = lift_history[skip:step]
            all_f = np.concatenate([d_vis, l_vis])
            fmin, fmax = np.min(all_f), np.max(all_f)
            margin = max(0.1 * (fmax - fmin), 0.01)
            ax_force.set_ylim(fmin - margin, fmax + margin)

        fig.tight_layout()

        if save:
            path = os.path.join(out_dir, f"step_{step:06d}.png")
            fig.savefig(path)
            print(f"  Saved {path}")
        if animate:
            plt.pause(0.01)

    # ==================================================================
    # 4.  Time loop
    # ==================================================================
    drag_history = np.zeros(n_steps)
    lift_history = np.zeros(n_steps)

    print(f"\nRunning {n_steps} timesteps ...")

    for step in tqdm(range(1, n_steps + 1)):
        # 5a. Macroscopic moments
        compute_macroscopic()

        # 5b. BGK collision (writes f_out, leaves f unchanged)
        collide(tau)

        # Optional: dump pre- and post-collision distributions.
        if save_every > 0 and step % save_every == 0:
            np.save(os.path.join(out_dir, f"fpre_{step:06d}.npy"), f.to_numpy())
            np.save(os.path.join(out_dir, f"fpost_{step:06d}.npy"), f_out.to_numpy())

        # 5c. Bounce-back on cylinder + momentum-exchange force
        bounce_back_and_force()
        Fx = float(Fx_field[None])
        Fy = float(Fy_field[None])
        drag_history[step - 1] = Fx
        lift_history[step - 1] = Fy
        if csv_file is not None and step % csv_every == 0:
            csv_file.write(f"{step},{step},{Fx:.8e},{Fy:.8e}\n")

        # 5d. Streaming
        stream()

        # 5e/5f/5g. Boundary conditions (walls, outlet, inlet)
        apply_wall_bc()
        apply_outlet_bc()
        apply_outlet_corners()
        apply_inlet_bc(U_inlet)

        # 5h. Visualisation & progress
        should_render = (
            (animate and step % plot_every == 0) or (save_every > 0 and step % save_every == 0) or step == n_steps
        )
        should_save = (save_every > 0 and step % save_every == 0) or step == n_steps

        if should_render:
            ux_h = ux.to_numpy()
            uy_h = uy.to_numpy()
            plot_field(ux_h, uy_h, step, save=should_save)

        if step % 1000 == 0:
            rho_h = rho.to_numpy()
            avg_rho = np.mean(rho_h[~obstacle_np])
            print(f"  Step {step:>6d}/{n_steps}  |  avg density = {avg_rho:.6f}")

    # ==================================================================
    # 5.  Final output
    # ==================================================================
    if csv_file is not None:
        csv_file.close()
        print(f"  Forces CSV: {csv_path}")

    print("\nSimulation complete.")
    if animate:
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    params = build_params(args)
    main(params)
