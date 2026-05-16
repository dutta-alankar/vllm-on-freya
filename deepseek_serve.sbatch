#!/bin/bash
#SBATCH -J deepseek_serve
#SBATCH --partition=p.gpu.ampere
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --constrain=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=23:59:59
#SBATCH --mail-type=NONE
#SBATCH --output=/u/adutt/ptmp/vllm/logs/deepseek_serve_%j.out
#SBATCH --error=/u/adutt/ptmp/vllm/logs/deepseek_serve_%j.err

# vLLM serving script for DeepSeek-V4-Pro (671B, FP8 block-quantized, MLA).
#
# Architecture:
#   srun  → starts Ray daemons on every node (background, stays alive)
#   main  → runs vLLM serve directly on the head node (foreground)
#
# Model: /u/adutt/ptmp/vllm/models/DeepSeek-V4-Pro  (~806 GB on disk, FP8)
# Resources: 4 nodes × 4 A100-80GB = 16 GPUs × 80 GB = 1280 GB VRAM
#
# Why PP=2 + TP=8 (not TP=16):
#   DeepSeek-V4-Pro has moe_intermediate_size=3072. The always-active shared
#   expert's down_proj is a RowParallelLinear with input_size=3072. The FP8
#   block quantization (block_k=128) requires input_size_per_partition % 128 == 0.
#   With TP=16: 3072/16=192, 192%128=64 ≠ 0 → vLLM raises ValueError.
#   With TP=8:  3072/8 =384, 384%128=0  ✓ → passes.
#   But TP=8 on 8 GPUs (640 GB total) can't fit ~806 GB of weights.
#   Solution: PP=2 × TP=8 = 16 GPUs total; each pipeline stage holds half
#   the layers (~403 GB) across 8 GPUs → ~50 GB/GPU for weights ✓.
#
# Override at submit time:
#   VLLM_PORT=8010 sbatch deepseek_serve.sbatch
#   sbatch --nodes=4 --export=ALL,GPUS_PER_NODE=4 deepseek_serve.sbatch
#
# Run from login node:  VLLM_PORT=<port> sbatch scripts/deepseek_serve.sbatch
# Stop:                 scancel <JOB_ID>

# gcc/14 is required at runtime because DeepGEMM JIT-compiles CUDA kernels
# via nvcc, which uses gcc as its host compiler. cuda/13.0 sets CUDA_HOME
# (needed by deep_gemm's assert at import time).
module purge && module load gcc/14 cuda/13.0

VENV=/u/adutt/ptmp/vllm/.venv
MODEL=${MODEL:-/u/adutt/ptmp/vllm/models/DeepSeek-V4-Pro}
RAY_PORT=6379
VLLM_PORT=${VLLM_PORT:-8000}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# TP=8 is fixed: the FP8 block alignment constraint (block_k=128) requires
# moe_intermediate_size/TP to be a multiple of 128. 3072/8=384, 384%128=0 ✓.
# PP=2 supplies the remaining 8 GPUs needed to fit the full model in VRAM.
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=2
HEAD_ADDRESS_FILE=/u/adutt/ptmp/vllm/ray_head_address.txt
# Limit Ray plasma store to 4 GB — vLLM passes tensors via NCCL, not the object store
RAY_OBJECT_STORE_MEMORY=4294967296

source "${VENV}/bin/activate"

rm -f "${HEAD_ADDRESS_FILE}"

# The SBATCH script runs on the first allocated node (head)
HEAD_IP=$(hostname --ip-address)

echo "$(date): DeepSeek-V4-Pro serve job ${SLURM_JOB_ID} starting"
echo "  Nodes (${SLURM_NNODES}): $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tr '\n' ' ')"
echo "  Head: $(hostname) (${HEAD_IP})"
echo "  TP size:  ${TENSOR_PARALLEL_SIZE}  PP size: ${PIPELINE_PARALLEL_SIZE}  (${SLURM_NNODES} nodes × ${GPUS_PER_NODE} GPUs)"
echo "  Model:    ${MODEL}"
echo "  API port: ${VLLM_PORT}"
echo ""

# Export so srun tasks inherit these values
export VENV RAY_PORT GPUS_PER_NODE TENSOR_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE RAY_OBJECT_STORE_MEMORY HEAD_IP
# Allow Ray to spill to /tmp if /dev/shm is constrained on shared nodes
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# ── DeepGEMM: NOT supported on A100 (Ampere/sm_80) ─────────────────────────
# DeepGEMM only supports Hopper (H100) and Blackwell (B200) GPUs.
# Freya A100s are sm_80 (Ampere), so we must disable these env vars.
# Leaving them enabled causes: ValueError: FP8 MoE backend DEEPGEMM does not
# support the deployment configuration since kernel does not support current
# device cuda.  vLLM will fall back to the vllm_cutlass FP8 MoE backend.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_CACHE_ROOT=/u/adutt/ptmp/vllm/.vllm_cache
mkdir -p "${VLLM_CACHE_ROOT}"

# NCCL: exclude loopback so cross-node collective ops use the real network
export NCCL_SOCKET_IFNAME=^lo,^docker0
export NCCL_DEBUG=WARN
# Disable the NCCL net plugin that segfaults on this cluster (ncclNetPluginInit crash)
# NCCL_IB_DISABLE=1 prevents the built-in IB transport from crashing at init time;
# NCCL falls back to TCP sockets for cross-node all-reduce instead.
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1

cleanup() {
    echo "$(date): Shutting down..."
    [[ -n "${VLLM_PID:-}" ]] && kill "${VLLM_PID}" 2>/dev/null || true
    [[ -n "${SRUN_PID:-}" ]] && kill "${SRUN_PID}" 2>/dev/null || true
    sleep 3
    ray stop 2>/dev/null || true
    rm -f "${HEAD_ADDRESS_FILE}"
}
trap cleanup EXIT TERM INT

# ── Step 1: Start Ray daemons on every node via srun (background) ──────────
# srun keeps each node's SLURM allocation alive; Ray runs as a daemon.
srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c '
    module purge && module load gcc/14 cuda/13.0 2>/dev/null || true
    source "${VENV}/bin/activate"
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

    cleanup_node() { echo "$(date): [$(hostname)] Stopping Ray..."; ray stop 2>/dev/null || true; }
    trap cleanup_node EXIT TERM INT

    ray stop --force 2>/dev/null || true

    if [[ "${SLURM_NODEID}" -eq 0 ]]; then
        echo "$(date): [HEAD $(hostname) ${HEAD_IP}] Starting Ray head on port ${RAY_PORT}"
        ray start --head \
            --node-ip-address="${HEAD_IP}" \
            --port="${RAY_PORT}" \
            --object-store-memory="${RAY_OBJECT_STORE_MEMORY}"
        echo "${HEAD_IP}:${RAY_PORT}" > /u/adutt/ptmp/vllm/ray_head_address.txt
        echo "$(date): [HEAD] Ray head started."
    else
        # Wait for head address file before connecting
        ELAPSED=0
        while [[ ! -f /u/adutt/ptmp/vllm/ray_head_address.txt ]] && [[ "${ELAPSED}" -lt 120 ]]; do
            sleep 5; ELAPSED=$((ELAPSED + 5))
        done
        echo "$(date): [WORKER $(hostname)] Connecting to ${HEAD_IP}:${RAY_PORT}"
        ray start --address="${HEAD_IP}:${RAY_PORT}" \
            --object-store-memory="${RAY_OBJECT_STORE_MEMORY}"
        echo "$(date): [WORKER $(hostname)] Ray worker started."
    fi

    # Keep task alive so SLURM holds the GPU resources on this node
    sleep infinity
' &
SRUN_PID=$!

# ── Step 2: Wait for Ray cluster to form ────────────────────────────────────
echo "$(date): Waiting 60s for Ray cluster to form..."
sleep 60

echo "$(date): Checking Ray cluster resources..."
python - <<EOF
import ray, sys
try:
    ray.init(address="${HEAD_IP}:${RAY_PORT}", ignore_reinit_error=True, logging_level="error")
    res = ray.cluster_resources()
    gpus = int(res.get("GPU", 0))
    nodes = sum(1 for n in ray.nodes() if n.get("Alive", False))
    print(f"  Ray cluster: {nodes} node(s), {gpus} GPU(s) total")
    need = ${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE}
    if gpus < need:
        print(f"  WARNING: need {need} GPUs (TP=${TENSOR_PARALLEL_SIZE}×PP=${PIPELINE_PARALLEL_SIZE}), only {gpus} available")
    ray.shutdown()
except Exception as e:
    print(f"  Ray status check failed: {e}", file=sys.stderr)
EOF

# ── Step 3: Launch vLLM from the main batch script (NOT inside srun) ────────
# Running vLLM here (not inside srun) lets its Ray actor workers spawn freely
# without inheriting srun's process-group or resource-step restrictions.
#
# DeepSeek-V4-Pro specifics:
#   - FP8 block-quantised weights (auto-detected from config.json); --dtype
#     bfloat16 sets the compute/activation dtype while weights stay in FP8.
#   - MLA (Multi-head Latent Attention) compresses KV cache by ~10×, making
#     128K context feasible with the remaining VRAM after weight loading.
#   - 384 routed MoE experts; --trust-remote-code required for DeepseekV4
#     architecture class not yet in the upstream transformers release.
#   - No multimodal encoder, so --mm-encoder-tp-mode is omitted.
echo "$(date): Starting vLLM server (TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}) on port ${VLLM_PORT}"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
vllm serve "${MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --served-model-name "$(basename "${MODEL}")" \
    --distributed-executor-backend ray \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --max-num-seqs 1024 \
    --enable-prefix-caching \
    --max-model-len 65536 &
VLLM_PID=$!

wait "${VLLM_PID}"
echo "$(date): vLLM exited (status $?)."
