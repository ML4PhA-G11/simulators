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
from numba import njit
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


@njit
def equilibrium(rho, ux, uy):
    """
    D2Q9 equilibrium:
        f_i^eq = w_i rho (1 + c_i·u/cs² + (c_i·u)²/(2cs⁴) - u²/(2cs²))
    with cs² = 1/3.
    """
    Nx, Ny = rho.shape
    feq = np.zeros((Nx, Ny, 9))
    usqr = ux**2 + uy**2

    for i in range(9):
        cu = c[i, 0] * ux + c[i, 1] * uy
        feq[:, :, i] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr)
    return feq


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

    print(f"\nRunning {n_steps} timesteps ...")

    for step in tqdm(range(1, n_steps + 1)):

        # -- 5a.  Macroscopic quantities --
        rho = np.sum(f, axis=2)
        ux = np.sum(f * c[:, 0], axis=2) / rho
        uy = np.sum(f * c[:, 1], axis=2) / rho

        # -- 5b.  Collision (BGK) --
        feq = equilibrium(rho, ux, uy)
        f_out = f - (f - feq) / tau

        # save distributions to csv
        if save_every > 0 and step % save_every == 0:
            np.save(os.path.join(out_dir, f"fpre_{step:06d}.npy"), f)
            np.save(os.path.join(out_dir, f"fpost_{step:06d}.npy"), f_out)

        # -- 5c.  Force computation & bounce-back on obstacle --
        #
        # Momentum-exchange method:  the force on the solid equals the
        # change in momentum of the distributions at obstacle nodes
        # during the bounce-back step.
        #
        #   F = Σ c_i · (f_out_before_BB − f_out_after_BB)
        #
        # This is exact for filled obstacles because interior solid
        # nodes have symmetric distributions (f_i = f_{opp_i} in
        # steady state) and contribute zero net momentum change.
        f_pre_bb = f_out[obstacle].copy()

        for i in range(ndir):
            f_out[obstacle, i] = f[obstacle, opp[i]]

        diff = f_pre_bb - f_out[obstacle]
        Fx = np.sum(diff * c[:, 0])
        Fy = np.sum(diff * c[:, 1])
        drag_history[step - 1] = Fx
        lift_history[step - 1] = Fy

        if csv_file is not None and step % csv_every == 0:
            csv_file.write(f"{step},{step},{Fx:.8e},{Fy:.8e}\n")

        # -- 5d.  Streaming --
        for i in range(ndir):
            f[:, :, i] = np.roll(f_out[:, :, i], shift=c[i, 0], axis=0)
            f[:, :, i] = np.roll(f[:, :, i], shift=c[i, 1], axis=1)

        # -- 5e.  Top & bottom wall bounce-back (no-slip) --
        # Applied BEFORE inlet/outlet so Zou-He reads valid wall values.
        f[:, 0, 2] = f_out[:, 0, 4]  # north  <- south
        f[:, 0, 5] = f_out[:, 0, 7]  # NE     <- SW
        f[:, 0, 6] = f_out[:, 0, 8]  # NW     <- SE
        f[:, -1, 4] = f_out[:, -1, 2]  # south  <- north
        f[:, -1, 7] = f_out[:, -1, 5]  # SW     <- NE
        f[:, -1, 8] = f_out[:, -1, 6]  # SE     <- NW

        # -- 5f.  Outlet BC: Zou-He pressure (rho = 1) --
        # Unknown: f_3, f_6, f_7 (west-moving, wrapped from x=0).
        # Interior y only; outlet corners use zero-gradient fallback.
        rho_out = 1.0
        iy = slice(1, -1)
        ux_out = (
            -1.0
            + (f[-1, iy, 0] + f[-1, iy, 2] + f[-1, iy, 4] + 2.0 * (f[-1, iy, 1] + f[-1, iy, 5] + f[-1, iy, 8]))
            / rho_out
        )
        ux_out = np.clip(ux_out, 0.0, 0.5)

        f[-1, iy, 3] = f[-1, iy, 1] - (2.0 / 3.0) * rho_out * ux_out
        f[-1, iy, 7] = f[-1, iy, 5] + 0.5 * (f[-1, iy, 2] - f[-1, iy, 4]) - (1.0 / 6.0) * rho_out * ux_out
        f[-1, iy, 6] = f[-1, iy, 8] - 0.5 * (f[-1, iy, 2] - f[-1, iy, 4]) - (1.0 / 6.0) * rho_out * ux_out

        # Outlet corners: zero-gradient fallback
        for yc in [0, Ny - 1]:
            f[-1, yc, 3] = f[-2, yc, 3]
            f[-1, yc, 6] = f[-2, yc, 6]
            f[-1, yc, 7] = f[-2, yc, 7]

        # -- 5g.  Inlet BC: Zou-He, fixed velocity (ux = U_inlet) --
        # Unknown: f_1, f_5, f_8 (east-moving, wrapped from x=Nx-1).
        rho_in = (f[0, :, 0] + f[0, :, 2] + f[0, :, 4] + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])) / (1.0 - U_inlet)

        f[0, :, 1] = f[0, :, 3] + (2.0 / 3.0) * rho_in * U_inlet
        f[0, :, 5] = f[0, :, 7] - 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U_inlet
        f[0, :, 8] = f[0, :, 6] + 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U_inlet

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
