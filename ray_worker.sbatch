#!/bin/bash
#SBATCH -J ray_worker
#SBATCH --partition=p.gpu.ampere
#SBATCH --nodes=1
#SBATCH --constrain=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --time=23:59:59
#SBATCH --mail-type=NONE
#SBATCH --output=/u/adutt/ptmp/vllm/logs/ray_worker_%j.out
#SBATCH --error=/u/adutt/ptmp/vllm/logs/ray_worker_%j.err

# Starts a Ray worker node and connects it to a running ray head.
# Submit ray_head.sbatch first, then this script (once per extra worker node).

module purge && module load cuda/13.0

VENV=/u/adutt/ptmp/vllm/.venv
HEAD_ADDRESS_FILE=/u/adutt/ptmp/vllm/ray_head_address.txt
MAX_WAIT=180   # seconds to wait for the head node address file

source "${VENV}/bin/activate"

trap 'echo "$(date): Stopping Ray..."; ray stop 2>/dev/null || true' EXIT TERM INT

echo "$(date): Starting Ray worker node"
echo "  Hostname: $(hostname)"

# Wait for the head node to write its address
echo "$(date): Waiting for ray head address (up to ${MAX_WAIT}s)..."
ELAPSED=0
while [[ ! -f "${HEAD_ADDRESS_FILE}" ]] && [[ "${ELAPSED}" -lt "${MAX_WAIT}" ]]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [[ ! -f "${HEAD_ADDRESS_FILE}" ]]; then
    echo "ERROR: Ray head address not found after ${MAX_WAIT}s."
    echo "  Make sure ray_head.sbatch is running and ${HEAD_ADDRESS_FILE} is accessible."
    exit 1
fi

HEAD_ADDRESS=$(cat "${HEAD_ADDRESS_FILE}")
echo "$(date): Connecting to ray head at ${HEAD_ADDRESS}"

# Stop any stale ray instance on this node
ray stop --force 2>/dev/null || true

# Connect to the ray cluster
ray start --address="${HEAD_ADDRESS}"
echo "$(date): Ray worker connected and running."

# Keep the SLURM job alive so the ray worker process remains active
sleep infinity
