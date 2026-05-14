#!/bin/bash
#SBATCH -J ray_head
#SBATCH --partition=p.gpu.ampere
#SBATCH --nodes=1
#SBATCH --constrain=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --time=23:59:59
#SBATCH --mail-type=NONE
#SBATCH --output=/u/adutt/ptmp/vllm/logs/ray_head_%j.out
#SBATCH --error=/u/adutt/ptmp/vllm/logs/ray_head_%j.err

# Starts a Ray head node.
# Workers connect to this node via ray_worker.sbatch.
# The head address is written to a shared file so workers can find it.

module purge && module load cuda/13.0

VENV=/u/adutt/ptmp/vllm/.venv
HEAD_ADDRESS_FILE=/u/adutt/ptmp/vllm/ray_head_address.txt
RAY_PORT=6379

source "${VENV}/bin/activate"

trap 'echo "$(date): Stopping Ray..."; ray stop 2>/dev/null || true' EXIT TERM INT

HEAD_IP=$(hostname --ip-address)
echo "$(date): Starting Ray head node"
echo "  Hostname: $(hostname)"
echo "  IP:       ${HEAD_IP}"
echo "  Port:     ${RAY_PORT}"

# Stop any stale ray instance on this node
ray stop --force 2>/dev/null || true
rm -f "${HEAD_ADDRESS_FILE}"

# Start ray head (returns immediately; ray processes run in background)
ray start --head \
    --node-ip-address="${HEAD_IP}" \
    --port="${RAY_PORT}"

# Write address to shared file so workers can connect
echo "${HEAD_IP}:${RAY_PORT}" > "${HEAD_ADDRESS_FILE}"
echo "$(date): Head address written to ${HEAD_ADDRESS_FILE}"
echo "$(date): Ray head running. Use RAY_ADDRESS=ray://${HEAD_IP}:${RAY_PORT} to connect."

# Keep the SLURM job alive so the ray head process remains active
sleep infinity
