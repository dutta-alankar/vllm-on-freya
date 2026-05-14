#!/bin/bash
# Launches a standalone Ray cluster (head + N workers) using sbatch.
# Use this for general Ray workloads.
# For vLLM serving, use start_vllm_server.sh instead.
#
# Usage:  bash scripts/launch_cluster.sh [OPTIONS]
#   --workers N          Number of worker nodes to add (default: 1)
#   --partition PART     Slurm partition (default: p.gpu.ampere)
#   --gpus-per-node N    GPUs per node (default: 4 for A100; use 2 for V100/P100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEAD_ADDRESS_FILE=/u/adutt/ptmp/vllm/ray_head_address.txt
MAX_WAIT=300   # seconds to wait for head node to come up
NUM_WORKERS=1
PARTITION=p.gpu.ampere
GPUS_PER_NODE=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)       NUM_WORKERS="$2";    shift 2 ;;
        --partition)     PARTITION="$2";       shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Launching Ray Cluster ==="
echo "  Partition:     ${PARTITION}"
echo "  GPUs/node:     ${GPUS_PER_NODE}"
echo "  Workers: ${NUM_WORKERS}  (total nodes: $((NUM_WORKERS + 1)))"

# Build GRES string: p.gpu.ampere requires explicit gpu type 'a100'
if [[ "${PARTITION}" == "p.gpu.ampere" ]]; then
    GRES_SPEC="gpu:a100:${GPUS_PER_NODE}"
else
    GRES_SPEC="gpu:${GPUS_PER_NODE}"
fi

# Remove stale address file from a previous run
rm -f "${HEAD_ADDRESS_FILE}"

# Submit the head node job
echo "Submitting ray head job..."
HEAD_JOB_ID=$(sbatch --parsable \
    --partition="${PARTITION}" \
    --gres="${GRES_SPEC}" \
    "${SCRIPT_DIR}/ray_head.sbatch")
echo "  Head job ID: ${HEAD_JOB_ID}"

# Wait for the head to start and write its address
echo "Waiting for head node (up to ${MAX_WAIT}s)..."
ELAPSED=0
while [[ ! -f "${HEAD_ADDRESS_FILE}" ]] && [[ "${ELAPSED}" -lt "${MAX_WAIT}" ]]; do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ${ELAPSED}s elapsed..."
done

if [[ ! -f "${HEAD_ADDRESS_FILE}" ]]; then
    echo ""
    echo "ERROR: Head node did not start within ${MAX_WAIT}s."
    echo "  The job may still be queued. Check with: squeue -j ${HEAD_JOB_ID}"
    echo "  Logs: tail -f /u/adutt/ptmp/vllm/logs/ray_head_${HEAD_JOB_ID}.out"
    exit 1
fi

HEAD_ADDRESS=$(cat "${HEAD_ADDRESS_FILE}")
echo "Head node ready at: ${HEAD_ADDRESS}"

# Submit the requested number of worker node jobs
WORKER_JOB_IDS=()
for i in $(seq 1 "${NUM_WORKERS}"); do
    WORKER_JOB_ID=$(sbatch --parsable \
        --partition="${PARTITION}" \
        --gres="${GRES_SPEC}" \
        "${SCRIPT_DIR}/ray_worker.sbatch")
    WORKER_JOB_IDS+=("${WORKER_JOB_ID}")
    echo "  Worker ${i} job ID: ${WORKER_JOB_ID}"
done

ALL_JOB_IDS="${HEAD_JOB_ID} ${WORKER_JOB_IDS[*]}"

echo ""
echo "=== Ray Cluster Launched ==="
echo "  Head:     job ${HEAD_JOB_ID}  →  ${HEAD_ADDRESS}"
for i in "${!WORKER_JOB_IDS[@]}"; do
    echo "  Worker $((i+1)): job ${WORKER_JOB_IDS[$i]}"
done
echo ""
echo "Check cluster resources (wait ~1 min for workers to join):"
echo "  source /u/adutt/ptmp/vllm/.venv/bin/activate"
echo "  RAY_ADDRESS=ray://${HEAD_ADDRESS} python -c \\"
echo "    \"import ray; ray.init(ignore_reinit_error=True); print(ray.cluster_resources())\""
echo ""
echo "Stop cluster:"
echo "  scancel ${ALL_JOB_IDS}"
