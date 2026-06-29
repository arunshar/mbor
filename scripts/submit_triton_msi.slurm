#!/usr/bin/env bash
#SBATCH --job-name=mbor-gpu
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=mbor_gpu_%j.out
#SBATCH --error=mbor_gpu_%j.out
# G4 (Triton dense kernels) + G5 (speculative GPU precompute) on an A100, in one
# job, inside the torch+Triton apptainer image. Honest results -> JSON.
set -uo pipefail
SIF="$HOME/mirror_torch.sif"
T="$HOME/mbor_rust_triton"
echo "node: $(hostname)"; nvidia-smi -L 2>&1 | head -4
echo "=== G4: Triton dense kernels bench ==="
apptainer exec --nv "$SIF" python "$T/g4_kernels_bench.py" \
  --pairs "$T/workload/pairs.json" --limit 4000 --out "$T/g4_results.json" 2>&1 | tail -50
echo "=== G5: speculative GPU precompute ==="
apptainer exec --nv "$SIF" python "$T/g5_precompute_gpu.py" \
  --frag "$T/workload/frag.json" --out "$T/g5_results.json" 2>&1 | tail -40
echo "=== DONE ==="; ls -la "$T"/*.json
