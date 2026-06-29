#!/usr/bin/env bash
#SBATCH --job-name=mbor-gpu2
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=mbor_gpu2_%j.out
#SBATCH --error=mbor_gpu2_%j.out
# G4 + G5 on an A100 using the ds-gpu conda env (torch 2.5.1 + Triton 3.1.0).
# Triton's runtime JIT needs a C compiler + CUDA headers; provide them via the
# MSI module gcc and cuda (the torch SIF has no compiler).
set -uo pipefail
# Provide gcc for Triton's runtime JIT (driver.c). Do NOT load the cuda module:
# it pollutes LD_LIBRARY_PATH and clashes with the conda env's bundled CUDA 12.1.
module load gcc/13.1.0-5z64cho 2>/dev/null || true
export CC="$(command -v gcc)"; export CXX="$(command -v g++)"
# ds-gpu's nvjitlink/cusparse combo is broken; use the newer ds-dev env
# (torch 2.12 + Triton 3.7) and let it resolve its own bundled CUDA libs.
unset LD_LIBRARY_PATH
export TRITON_CACHE_DIR="$HOME/.triton_cache_${SLURM_JOB_ID}"
PY="$HOME/.conda/envs/ds-dev/bin/python"
T="$HOME/mbor_rust_triton"
echo "node $(hostname); CC=$CC"; nvidia-smi -L | head -2
"$PY" -c "import torch;print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "=== G4: Triton dense kernels ==="
"$PY" "$T/g4_kernels_bench.py" --pairs "$T/workload/pairs.json" --limit 4000 --out "$T/g4_results.json" 2>&1 | tail -60
echo "=== G5: speculative GPU precompute ==="
"$PY" "$T/g5_precompute_gpu.py" --frag "$T/workload/frag.json" --out "$T/g5_results.json" 2>&1 | tail -30
echo "=== DONE ==="; ls -la "$T"/*.json
