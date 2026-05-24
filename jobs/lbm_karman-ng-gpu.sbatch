#!/bin/bash
# =============================================================================
# Snellius SLURM job: LBM Karman vortex street, GPU (Taichi on NVIDIA H100)
# -----------------------------------------------------------------------------
# Runs lbm_karman-ng-gpu.py at 2x finer grid (resolution=500, 1100x205) for
# 20000 steps, saving fpre/fpost distribution snapshots every 1000 steps
# (20 snapshot pairs total).
#
# Submit with:    sbatch jobs/lbm_karman-ng-gpu.sbatch
# -----------------------------------------------------------------------------

#SBATCH --job-name=lbm-gpu-r500
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

set -euo pipefail

# --- Paths -------------------------------------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p jobs/logs

# --- Run parameters (kept in sync with the CPU sibling job) ------------------
#RESOLUTION=500          # 2x finer than the default 250 lu/m
RESOLUTION=1000          # 4x finer than the default 250 lu/m
#N_STEPS=20000           # matches 10000 steps at res=250 in physical time
N_STEPS=40000           # matches 10000 steps at res=250 in physical time
SAVE_EVERY=1000         # 20 fpre/fpost snapshot pairs
OUT_DIR="output/gpu_res${RESOLUTION}_n${N_STEPS}"

# --- Environment -------------------------------------------------------------
# Adjust module versions to whatever is current on Snellius. CUDA must be
# loaded so Taichi can find libcuda.so / libcudart.so for the H100.
module purge
module load 2025
module load CUDA/12.8.0

# Assumes a uv-managed .venv in the project root with taichi installed.
# Bootstrap once (locally or on a login node) with:
#   uv venv && uv pip install taichi numpy matplotlib tqdm numba
if [[ ! -x ".venv/bin/python" ]]; then
    echo "ERROR: .venv/bin/python not found. Create it with 'uv venv && uv pip install ...'." >&2
    exit 1
fi

# --- Diagnostics -------------------------------------------------------------
echo "=== Node: $(hostname)  Job: $SLURM_JOB_ID ==="
nvidia-smi || true
echo "=== Run config ==="
echo "  resolution = $RESOLUTION"
echo "  n-steps    = $N_STEPS"
echo "  save-every = $SAVE_EVERY"
echo "  out-dir    = $OUT_DIR"
echo

date

# --- Run ---------------------------------------------------------------------
srun .venv/bin/python lbm_karman-ng-gpu.py \
    --arch cuda \
    --fp f64 \
    --resolution "$RESOLUTION" \
    --n-steps "$N_STEPS" \
    --save-every "$SAVE_EVERY" \
    --plot-mode none \
    --csv-every 1 \
    --out-dir "$OUT_DIR"

date
