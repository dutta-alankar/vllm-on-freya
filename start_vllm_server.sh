#!/bin/bash
# Run from the login node to submit the vLLM serving job.
#
# Usage:  bash scripts/start_vllm_server.sh [OPTIONS]
#
# Options:
#   --nodes N            Total SLURM nodes (head + workers). TP = N * GPUS_PER_NODE.
#                        Must divide num_attention_heads (8 for this model).
#                          --nodes 2  ->  TP=8   (default, 2 x 4-GPU A100 nodes)
#                          --nodes 4  ->  TP=16  (4 x 4-GPU A100 nodes)
#   --gpus-per-node N    GPUs per node (default: 4 for p.gpu.ampere A100;
#                        use 2 for V100/P100 nodes on other partitions)
#   --partition PART     Slurm partition (default: p.gpu.ampere)
#   --model DIR          Path to model directory (default: .../Qwen3.6-35B-A3B)
#   --port PORT          vLLM API port (default: 8000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SBATCH="${SCRIPT_DIR}/vllm_serve.sbatch"
LOG_DIR=/u/adutt/ptmp/vllm/logs
NUM_NODES=2
GPUS_PER_NODE=4
PARTITION=p.gpu.ampere
MODEL=/u/adutt/ptmp/vllm/models/Qwen3.6-35B-A3B
VLLM_PORT=8000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)         NUM_NODES="$2";      shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2";  shift 2 ;;
        --partition)     PARTITION="$2";       shift 2 ;;
        --model)         MODEL="$2";           shift 2 ;;
        --port)          VLLM_PORT="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -f "${VLLM_SBATCH}" ]]; then
    echo "ERROR: ${VLLM_SBATCH} not found."
    exit 1
fi

TP_SIZE=$((NUM_NODES * GPUS_PER_NODE))

# Build GRES string: p.gpu.ampere requires explicit gpu type 'a100'
if [[ "${PARTITION}" == "p.gpu.ampere" ]]; then
    GRES_SPEC="gpu:a100:${GPUS_PER_NODE}"
else
    GRES_SPEC="gpu:${GPUS_PER_NODE}"
fi

echo "=== Submitting vLLM server job ==="
echo "  Partition:     ${PARTITION}"
echo "  Nodes:         ${NUM_NODES}  (${GPUS_PER_NODE} GPUs/node → TP=${TP_SIZE})"
echo "  Model:         ${MODEL}"
echo "  vLLM port:     ${VLLM_PORT}"
JOB_ID=$(sbatch --parsable \
    --nodes="${NUM_NODES}" \
    --partition="${PARTITION}" \
    --gres="${GRES_SPEC}" \
    --export="ALL,MODEL=${MODEL},VLLM_PORT=${VLLM_PORT},GPUS_PER_NODE=${GPUS_PER_NODE}" \
    "${VLLM_SBATCH}")
echo "  Job ID:  ${JOB_ID}"
echo ""
echo "Monitor job status:"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "Follow logs (once the job starts):"
echo "  tail -f ${LOG_DIR}/vllm_serve_${JOB_ID}.out"
echo ""
echo "Once the server is ready (~10-15 min for model load), test with:"
echo "  HEAD_NODE=\$(squeue -j ${JOB_ID} -h -o '%N' | tr ',' '\n' | head -1)"
echo "  curl http://\${HEAD_NODE}:${VLLM_PORT}/v1/models"
echo ""
echo "Stop the server:"
echo "  scancel ${JOB_ID}"
