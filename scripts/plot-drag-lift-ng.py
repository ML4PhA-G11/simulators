#!/usr/bin/env python3
"""Plot drag/lift forces and frequency analysis from LBM-NG CSV output.

Reads the forces.csv produced by lbm_karman-ng.py and generates a 2x2
analysis figure: time-series (left) and FFT spectra (right) for drag
and lift.  Computes Strouhal number from the lift peak frequency.

Usage:
  python scripts/plot-drag-lift-ng.py                        # default CSV
  python scripts/plot-drag-lift-ng.py output-ng/forces.csv
  python scripts/plot-drag-lift-ng.py --skip 5000 --smooth 200
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Plot drag/lift from LBM-NG forces.csv",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("csv", nargs="?", default=None,
                    help="Path to forces.csv (default: output-ng/forces.csv)")
parser.add_argument("--skip", type=int, default=1000,
                    help="Skip the first N timesteps")
parser.add_argument("--smooth", type=int, default=0,
                    help="Moving-average window in timesteps (0 = off)")
parser.add_argument("--freq-max", type=float, default=0.01,
                    help="Max frequency to display in spectrum (1/timestep)")
parser.add_argument("--show", action="store_true", default=False,
                    help="Show interactive plot window")
args = parser.parse_args()

# ── Locate CSV ───────────────────────────────────────────────────────
if args.csv:
    csv_path = Path(args.csv)
else:
    csv_path = Path(__file__).resolve().parent.parent / "output-ng" / "forces.csv"
out_dir = csv_path.parent

# ── Parse metadata from comment line ─────────────────────────────────
metadata = {}
with open(csv_path) as fh:
    for line in fh:
        if line.startswith('#'):
            for token in line[1:].strip().split():
                if '=' in token:
                    k, v = token.split('=', 1)
                    try:
                        metadata[k] = float(v)
                    except ValueError:
                        metadata[k] = v
        else:
            break

Re      = metadata.get('Re', float('nan'))
U_inlet = metadata.get('U_inlet', float('nan'))
D       = metadata.get('D', float('nan'))

print(f"Metadata: Re={Re}, U_inlet={U_inlet}, D={D}")

# ── Load data ────────────────────────────────────────────────────────
raw = np.loadtxt(csv_path, delimiter=",", skiprows=2)
mask = raw[:, 0] >= args.skip
step, drag, lift = raw[mask, 0], raw[mask, 2], raw[mask, 3]
dt = step[1] - step[0] if len(step) > 1 else 1.0

print(f"Loaded {len(raw)} rows, using {mask.sum()} after skip={args.skip}")

# ── Helpers ──────────────────────────────────────────────────────────
def moving_average(x, window):
    if window <= 1:
        return x, 0
    kernel = np.ones(window) / window
    smoothed = np.convolve(x, kernel, mode="valid")
    offset = window // 2
    return smoothed, offset

def compute_spectrum(x, dt):
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(np.fft.rfft(x - x.mean())) * 2.0 / n
    return freqs, mag

smooth_n = args.smooth

# ── Figure: 2x2 layout ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
(ax_drag, ax_drag_fft), (ax_lift, ax_lift_fft) = axes

# ── Left column: time series ─────────────────────────────────────────
ax_drag.plot(step, drag, linewidth=0.3, alpha=0.5, color="C0", label="raw")
ax_lift.plot(step, lift, linewidth=0.3, alpha=0.5, color="C1", label="raw")

if smooth_n > 1:
    drag_s, off = moving_average(drag, smooth_n)
    lift_s, _   = moving_average(lift, smooth_n)
    s_s = step[off : off + len(drag_s)]
    ax_drag.plot(s_s, drag_s, linewidth=1.2, color="C0",
                 label=f"smooth ({smooth_n} steps)")
    ax_lift.plot(s_s, lift_s, linewidth=1.2, color="C1",
                 label=f"smooth ({smooth_n} steps)")

ax_drag.axhline(drag.mean(), color="black", linestyle="--", linewidth=0.8,
                label=f"mean = {drag.mean():.6f}")
ax_drag.set_ylabel("Drag (Fx)")
ax_drag.legend(loc="upper right", fontsize=8)
ax_drag.grid(True, alpha=0.3)

ax_lift.axhline(lift.mean(), color="black", linestyle="--", linewidth=0.8,
                label=f"mean = {lift.mean():.6f}")
ax_lift.set_ylabel("Lift (Fy)")
ax_lift.set_xlabel("Timestep")
ax_lift.legend(loc="upper right", fontsize=8)
ax_lift.grid(True, alpha=0.3)

# ── Right column: frequency spectra ─────────────────────────────────
drag_for_fft = moving_average(drag, max(smooth_n, 50))[0] if smooth_n > 0 else drag
lift_for_fft = lift

freqs_d, mag_d = compute_spectrum(drag_for_fft, dt)
freqs_l, mag_l = compute_spectrum(lift_for_fft, dt)

freq_mask_d = freqs_d <= args.freq_max
freq_mask_l = freqs_l <= args.freq_max

ax_drag_fft.plot(freqs_d[freq_mask_d], mag_d[freq_mask_d],
                 linewidth=0.8, color="C0")
ax_lift_fft.plot(freqs_l[freq_mask_l], mag_l[freq_mask_l],
                 linewidth=0.8, color="C1")

# Mark dominant peaks
lift_peak_idx = np.argmax(mag_l[1:freq_mask_l.sum()]) + 1
lift_freq = freqs_l[lift_peak_idx]
drag_peak_idx = np.argmax(mag_d[1:freq_mask_d.sum()]) + 1
drag_freq = freqs_d[drag_peak_idx]

ax_lift_fft.axvline(lift_freq, color="C1", linestyle="--", alpha=0.7)
ax_lift_fft.annotate(f"f_lift = {lift_freq:.6f}/step",
                     xy=(lift_freq, mag_l[lift_peak_idx]),
                     xytext=(lift_freq * 1.5, mag_l[lift_peak_idx] * 0.9),
                     fontsize=8, arrowprops=dict(arrowstyle="->", color="C1"))

ax_drag_fft.axvline(drag_freq, color="C0", linestyle="--", alpha=0.7)
ax_drag_fft.annotate(f"f_drag = {drag_freq:.6f}/step",
                     xy=(drag_freq, mag_d[drag_peak_idx]),
                     xytext=(drag_freq * 1.5, mag_d[drag_peak_idx] * 0.9),
                     fontsize=8, arrowprops=dict(arrowstyle="->", color="C0"))

# Reference: drag frequency should be 2x lift frequency
ax_drag_fft.axvline(2 * lift_freq, color="gray", linestyle=":", alpha=0.7)
ax_drag_fft.annotate(f"2 x f_lift = {2*lift_freq:.6f}",
                     xy=(2 * lift_freq, mag_d[drag_peak_idx] * 0.5),
                     fontsize=7, color="gray")

ratio = drag_freq / lift_freq if lift_freq > 0 else float("nan")

# Strouhal number: St = f_lift * D / U_inlet  (all in lattice units)
St = lift_freq * D / U_inlet if (D > 0 and U_inlet > 0) else float("nan")

ax_drag_fft.set_title(
    f"Drag spectrum (f_drag/f_lift = {ratio:.2f}, expect 2.0)", fontsize=9)
ax_lift_fft.set_title(
    f"Lift spectrum (St = f*D/U = {St:.4f})", fontsize=9)

ax_drag_fft.set_ylabel("Amplitude")
ax_drag_fft.set_xlabel("Frequency (1/step)")
ax_drag_fft.grid(True, alpha=0.3)
ax_lift_fft.set_ylabel("Amplitude")
ax_lift_fft.set_xlabel("Frequency (1/step)")
ax_lift_fft.grid(True, alpha=0.3)

fig.suptitle(f"Drag & Lift Analysis — Re={Re:.0f}  (step >= {args.skip})",
             fontsize=12)
plt.tight_layout()

out_png = out_dir / "forces_analysis.png"
plt.savefig(out_png, dpi=150)
print(f"\nSaved: {out_png}")
print(f"  Lift freq:   {lift_freq:.6f} /step  (period = {1/lift_freq:.0f} steps)" if lift_freq > 0 else "  Lift freq:   N/A")
print(f"  Drag freq:   {drag_freq:.6f} /step" if drag_freq > 0 else "  Drag freq:   N/A")
print(f"  Ratio:       {ratio:.4f} (expected 2.0)")
print(f"  Strouhal:    {St:.4f} (expected ~0.18 for Re=150)")
print(f"  Drag mean:   {drag.mean():.6f}")

if args.show:
    matplotlib.use('TkAgg')
    plt.show()
