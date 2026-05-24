#!/bin/bash
# =============================================================================
# Snellius SLURM job: LBM Karman vortex street, CPU (NumPy + Numba)
# -----------------------------------------------------------------------------
# Runs lbm_karman-ng.py at 2x finer grid (resolution=500, 1100x205) for 20000
# steps, saving fpre/fpost distribution snapshots every 1000 steps (20
# snapshot pairs total). Grid + step count match the GPU sibling job so the
# two output directories are directly comparable.
#
# Submit with:    sbatch jobs/lbm_karman-ng-cpu.sbatch
# -----------------------------------------------------------------------------

#SBATCH --job-name=lbm-cpu-r500
#SBATCH --account=<YOUR_PROJECT>
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

set -euo pipefail

# --- Paths -------------------------------------------------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p jobs/logs

# --- Run parameters (kept in sync with the GPU sibling job) ------------------
RESOLUTION=500          # 2x finer than the default 250 lu/m
N_STEPS=20000           # matches 10000 steps at res=250 in physical time
SAVE_EVERY=1000         # 20 fpre/fpost snapshot pairs
OUT_DIR="output/cpu_res${RESOLUTION}_n${N_STEPS}"

# --- Numba threading ---------------------------------------------------------
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# --- Environment -------------------------------------------------------------
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

# Assumes a uv-managed .venv in the project root.
# Bootstrap once (locally or on a login node) with:
#   uv venv && uv pip install taichi numpy matplotlib tqdm numba
if [[ ! -x ".venv/bin/python" ]]; then
    echo "ERROR: .venv/bin/python not found. Create it with 'uv venv && uv pip install ...'." >&2
    exit 1
fi

# --- Diagnostics -------------------------------------------------------------
echo "=== Node: $(hostname)  Job: $SLURM_JOB_ID ==="
echo "=== Run config ==="
echo "  resolution = $RESOLUTION"
echo "  n-steps    = $N_STEPS"
echo "  save-every = $SAVE_EVERY"
echo "  out-dir    = $OUT_DIR"
echo "  threads    = $NUMBA_NUM_THREADS"
echo

# --- Run ---------------------------------------------------------------------
srun .venv/bin/python lbm_karman-ng.py \
    --resolution "$RESOLUTION" \
    --n-steps "$N_STEPS" \
    --save-every "$SAVE_EVERY" \
    --plot-mode none \
    --csv-every 1 \
    --out-dir "$OUT_DIR"
