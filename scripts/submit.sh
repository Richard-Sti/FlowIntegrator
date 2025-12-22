#!/bin/bash
#SBATCH --partition=short
#SBATCH --mail-user=richard.stiskalek@physics.ox.ac.uk
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 --constraint="cpu_gen:Cascade_Lake|cpu_gen:Skylake"
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --job-name=flowi
#SBATCH --output=logs/logs-%j.out
#SBATCH --error=logs/logs-%j.err

# --- User configuration ---
PYTHON_EXEC="/home/phys1997/CANDEL/venv_candel/bin/python"
PYTHON_SCRIPT_TO_RUN="MW_streamlines.py"

# --- Main script logic ---
# Report requested time
if [[ -n "$SLURM_TIMELIMIT" ]]; then
    hrs=$((SLURM_TIMELIMIT / 60))
    mins=$((SLURM_TIMELIMIT % 60))
    echo "[INFO] SLURM time limit requested: ${hrs}h ${mins}m"
fi

set -e


# Load required modules for ARC
echo "[INFO] Loading modules for machine: arc"
module --force purge
module add Python/3.11.3-GCCcore-12.3.0
module add CUDA/11.8.0

# Set XLA flags
export XLA_FLAGS="--xla_hlo_profile=false --xla_dump_to=/tmp/nowhere"

# Run the python script
echo "[INFO] Running Python script: $PYTHON_SCRIPT_TO_RUN"
eval "$PYTHON_EXEC $PYTHON_SCRIPT_TO_RUN"

echo "[INFO] Script finished."
