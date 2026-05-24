"""
Lattice Boltzmann Method (LBM) — Karman Vortex Street Solver (NG)
=================================================================
D2Q9 lattice, BGK collision.  Solves flow past a circular cylinder
in a channel following the geometry of Assignment 3.1 (Figure 4):

    Domain   2.2 m × 0.41 m
    Cylinder center (0.2, 0.2) m,  diameter 0.10 m
    Walls    no-slip (top, bottom, cylinder)
    Inlet    Zou-He velocity (uniform)
    Outlet   Zou-He pressure (rho = 1)

Dependencies: numpy, matplotlib, numba

Usage:
  python lbm_karman-ng.py [OPTIONS]

Examples:
  python lbm_karman-ng.py                            # baseline Re=150
  python lbm_karman-ng.py --Re 200 --n-steps 50000
  python lbm_karman-ng.py --animate --plot-mode vorticity
"""

import argparse
import os
import numpy as np
import matplotlib
from numba import njit, prange
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

c = np.array(
    [
        [0, 0],  # 0  rest
        [1, 0],  # 1  east
        [0, 1],  # 2  north
        [-1, 0],  # 3  west
        [0, -1],  # 4  south
        [1, 1],  # 5  north-east
        [-1, 1],  # 6  north-west
        [-1, -1],  # 7  south-west
        [1, -1],
    ]
)  # 8  south-east

w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

ndir = 9

# =============================================================================
# 2.  Command-line Interface
# =============================================================================


def parse_args(argv=None):
    """Parse command-line arguments for the LBM simulation."""
    p = argparse.ArgumentParser(
        description="LBM Karman Vortex Street Solver — NG (D2Q9, BGK)",
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

    # Obstacle mask
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
    )


# =============================================================================
# 3.  Equilibrium Distribution Function
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def equilibrium(rho, ux, uy):
    """
    D2Q9 equilibrium:
        f_i^eq = w_i rho (1 + c_i·u/cs² + (c_i·u)²/(2cs⁴) - u²/(2cs²))
    with cs² = 1/3.
    """
    Nx, Ny = rho.shape
    feq = np.empty((Nx, Ny, 9))
    for x in prange(Nx):
        for y in range(Ny):
            usqr = ux[x, y] * ux[x, y] + uy[x, y] * uy[x, y]
            for i in range(9):
                cu = c[i, 0] * ux[x, y] + c[i, 1] * uy[x, y]
                feq[x, y, i] = w[i] * rho[x, y] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)
    return feq


@njit(parallel=True, fastmath=True, cache=True)
def lbm_collide(f, f_out, rho, ux, uy, tau):
    """Compute macroscopic moments and BGK collision; writes f_out, rho, ux, uy."""
    Nx = f.shape[0]
    Ny = f.shape[1]
    inv_tau = 1.0 / tau
    for x in prange(Nx):
        for y in range(Ny):
            s = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                fi = f[x, y, i]
                s += fi
                mx += fi * c[i, 0]
                my += fi * c[i, 1]
            inv_s = 1.0 / s
            uxv = mx * inv_s
            uyv = my * inv_s
            rho[x, y] = s
            ux[x, y] = uxv
            uy[x, y] = uyv
            usqr = uxv * uxv + uyv * uyv
            for i in range(9):
                cu = c[i, 0] * uxv + c[i, 1] * uyv
                feq = w[i] * s * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)
                fi = f[x, y, i]
                f_out[x, y, i] = fi - (fi - feq) * inv_tau


@njit(parallel=True, fastmath=True, cache=True)
def lbm_post_collide(f, f_out, obstacle, U_inlet):
    """Obstacle bounce-back (returns Fx,Fy), streaming, wall + inlet/outlet BCs."""
    Nx = f.shape[0]
    Ny = f.shape[1]

    # --- Obstacle bounce-back + momentum-exchange force ---
    Fx = 0.0
    Fy = 0.0
    for x in prange(Nx):
        for y in range(Ny):
            if obstacle[x, y]:
                for i in range(9):
                    pre = f_out[x, y, i]
                    post = f[x, y, opp[i]]
                    f_out[x, y, i] = post
                    Fx += (pre - post) * c[i, 0]
                    Fy += (pre - post) * c[i, 1]

    # --- Streaming (periodic; wall/inlet/outlet overwrite below) ---
    for x in prange(Nx):
        for y in range(Ny):
            for i in range(9):
                xs = (x + c[i, 0]) % Nx
                ys = (y + c[i, 1]) % Ny
                f[xs, ys, i] = f_out[x, y, i]

    # --- Top & bottom wall bounce-back (no-slip) ---
    for x in prange(Nx):
        f[x, 0, 2] = f_out[x, 0, 4]
        f[x, 0, 5] = f_out[x, 0, 7]
        f[x, 0, 6] = f_out[x, 0, 8]
        f[x, Ny - 1, 4] = f_out[x, Ny - 1, 2]
        f[x, Ny - 1, 7] = f_out[x, Ny - 1, 5]
        f[x, Ny - 1, 8] = f_out[x, Ny - 1, 6]

    # --- Outlet BC: Zou-He pressure (rho = 1) ---
    rho_out = 1.0
    for y in range(1, Ny - 1):
        ux_out = -1.0 + (
            f[Nx - 1, y, 0] + f[Nx - 1, y, 2] + f[Nx - 1, y, 4]
            + 2.0 * (f[Nx - 1, y, 1] + f[Nx - 1, y, 5] + f[Nx - 1, y, 8])
        ) / rho_out
        if ux_out < 0.0:
            ux_out = 0.0
        elif ux_out > 0.5:
            ux_out = 0.5
        f[Nx - 1, y, 3] = f[Nx - 1, y, 1] - (2.0 / 3.0) * rho_out * ux_out
        f[Nx - 1, y, 7] = (
            f[Nx - 1, y, 5] + 0.5 * (f[Nx - 1, y, 2] - f[Nx - 1, y, 4]) - (1.0 / 6.0) * rho_out * ux_out
        )
        f[Nx - 1, y, 6] = (
            f[Nx - 1, y, 8] - 0.5 * (f[Nx - 1, y, 2] - f[Nx - 1, y, 4]) - (1.0 / 6.0) * rho_out * ux_out
        )

    # Outlet corners: zero-gradient fallback
    f[Nx - 1, 0, 3] = f[Nx - 2, 0, 3]
    f[Nx - 1, 0, 6] = f[Nx - 2, 0, 6]
    f[Nx - 1, 0, 7] = f[Nx - 2, 0, 7]
    f[Nx - 1, Ny - 1, 3] = f[Nx - 2, Ny - 1, 3]
    f[Nx - 1, Ny - 1, 6] = f[Nx - 2, Ny - 1, 6]
    f[Nx - 1, Ny - 1, 7] = f[Nx - 2, Ny - 1, 7]

    # --- Inlet BC: Zou-He velocity (ux = U_inlet) ---
    for y in range(Ny):
        rho_in = (
            f[0, y, 0] + f[0, y, 2] + f[0, y, 4]
            + 2.0 * (f[0, y, 3] + f[0, y, 6] + f[0, y, 7])
        ) / (1.0 - U_inlet)
        f[0, y, 1] = f[0, y, 3] + (2.0 / 3.0) * rho_in * U_inlet
        f[0, y, 5] = f[0, y, 7] - 0.5 * (f[0, y, 2] - f[0, y, 4]) + (1.0 / 6.0) * rho_in * U_inlet
        f[0, y, 8] = f[0, y, 6] + 0.5 * (f[0, y, 2] - f[0, y, 4]) + (1.0 / 6.0) * rho_in * U_inlet

    return Fx, Fy


# =============================================================================
# 4.  Main Simulation
# =============================================================================


def main(params):
    """Run the LBM simulation."""

    # -- Unpack parameters --
    Nx = params["Nx"]
    Ny = params["Ny"]
    U_inlet = params["U_inlet"]
    tau = params["tau"]
    obstacle = params["obstacle"]
    Y = params["Y"]
    n_steps = params["n_steps"]
    plot_every = params["plot_every"]
    plot_mode = params["plot_mode"]
    animate = params["animate"]
    save_every = params["save_every"]
    out_dir = params["out_dir"]
    csv_every = params["csv_every"]

    if not animate:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # ==================================================================
    # 4a-pre.  CSV force output
    # ==================================================================
    csv_file = None
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
    # 4a.  Initialization
    # ==================================================================

    rho_init = np.ones((Nx, Ny))
    ux_init = np.full((Nx, Ny), U_inlet)
    uy_init = np.zeros((Nx, Ny))

    # Small transverse perturbation to break symmetry
    uy_init += 0.001 * U_inlet * np.sin(2.0 * np.pi * Y / Ny)

    # Zero velocity inside obstacle and at walls
    ux_init[obstacle] = 0.0
    uy_init[obstacle] = 0.0
    ux_init[:, 0] = 0.0
    uy_init[:, 0] = 0.0
    ux_init[:, -1] = 0.0
    uy_init[:, -1] = 0.0

    f = equilibrium(rho_init, ux_init, uy_init)

    # ==================================================================
    # 4b.  Visualisation setup
    # ==================================================================

    if animate:
        plt.ion()
    fig, (ax_flow, ax_force) = plt.subplots(2, 1, figsize=(10, 6), dpi=120, gridspec_kw={"height_ratios": [2, 1]})

    def plot_velocity(ux, uy, step):
        speed = np.sqrt(ux**2 + uy**2)
        speed[obstacle] = np.nan
        ax_flow.clear()
        ax_flow.imshow(
            speed.T, origin="lower", cmap="jet", vmin=0, vmax=U_inlet * 2.0, aspect="auto", extent=[0, Nx, 0, Ny]
        )
        ax_flow.set_title(f"Velocity magnitude — step {step}")
        ax_flow.set_xlabel("x")
        ax_flow.set_ylabel("y")

    def plot_vorticity(ux, uy, step):
        vort = np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0) - np.roll(ux, -1, axis=1) + np.roll(ux, 1, axis=1)
        vort[obstacle] = np.nan
        ax_flow.clear()
        ax_flow.imshow(
            vort.T, origin="lower", cmap="RdBu_r", vmin=-0.04, vmax=0.04, aspect="auto", extent=[0, Nx, 0, Ny]
        )
        ax_flow.set_title(f"Vorticity field — step {step}")
        ax_flow.set_xlabel("x")
        ax_flow.set_ylabel("y")

    def plot_field(ux, uy, step, save=False):
        if plot_mode == "vorticity":
            plot_vorticity(ux, uy, step)
        elif plot_mode == "velocity":
            plot_velocity(ux, uy, step)
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

        # Auto-scale y-axis skipping initial transient spike
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
    # 5.  Time loop
    # ==================================================================

    drag_history = np.zeros(n_steps)
    lift_history = np.zeros(n_steps)

    # Pre-allocated buffers reused every step (avoids per-iteration allocation)
    f_out = np.empty_like(f)
    rho = np.empty((Nx, Ny))
    ux = np.empty((Nx, Ny))
    uy = np.empty((Nx, Ny))

    print(f"\nRunning {n_steps} timesteps ...")

    for step in tqdm(range(1, n_steps + 1)):

        # -- 5a/b. Macroscopic quantities + BGK collision (JIT) --
        lbm_collide(f, f_out, rho, ux, uy, tau)

        # save distributions to csv
        if save_every > 0 and step % save_every == 0:
            np.save(os.path.join(out_dir, f"fpre_{step:06d}.npy"), f)
            np.save(os.path.join(out_dir, f"fpost_{step:06d}.npy"), f_out)

        # -- 5c-g. Bounce-back + force, streaming, wall + inlet/outlet BCs (JIT) --
        Fx, Fy = lbm_post_collide(f, f_out, obstacle, U_inlet)
        drag_history[step - 1] = Fx
        lift_history[step - 1] = Fy

        if csv_file is not None and step % csv_every == 0:
            csv_file.write(f"{step},{step},{Fx:.8e},{Fy:.8e}\n")

        # -- 5h.  Visualisation & progress --
        should_render = (
            (animate and step % plot_every == 0) or (save_every > 0 and step % save_every == 0) or step == n_steps
        )
        should_save = (save_every > 0 and step % save_every == 0) or step == n_steps

        if should_render:
            plot_field(ux, uy, step, save=should_save)

        if step % 1000 == 0:
            avg_rho = np.mean(rho[~obstacle])
            print(f"  Step {step:>6d}/{n_steps}  |  avg density = {avg_rho:.6f}")

    # ==================================================================
    # 6.  Final output
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
