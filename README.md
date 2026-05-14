# Ray + vLLM Multi-Node LLM Serving

This guide explains how to start a Ray cluster across multiple SLURM nodes and serve the `Qwen3.6-35B-A3B` model with vLLM's OpenAI-compatible API.

---

## Prerequisites

| Requirement | Value |
|-------------|-------|
| Python environment | `/u/adutt/ptmp/vllm/.venv` |
| Default model path | `/u/adutt/ptmp/vllm/models/Qwen3.6-35B-A3B` |
| SLURM partition | `p.gpu.ampere` (A100; use `p.gpu` for V100/P100) |
| GPUs per node | 4 (A100) or 2 (V100/P100) |
| Minimum GPU architecture | **sm_75** (see [GPU Compatibility](#gpu-compatibility)) |
| vLLM version | 0.20.2 |
| Ray version | 2.55.1 |
| CUDA module | `cuda/13.0` |

---

## Scripts Overview

All scripts live in `scripts/`:

```
scripts/
├── vllm_serve.sbatch      # All-in-one: start Ray cluster + vLLM server in one job
├── ray_head.sbatch        # Start a Ray head node only
├── ray_worker.sbatch      # Start a Ray worker node only
├── start_vllm_server.sh   # Submit vllm_serve.sbatch from login node
└── launch_cluster.sh      # Submit a standalone Ray cluster (head + N workers)
```

---

## Quick Start (recommended)

### 1. Submit the vLLM serving job from the login node

```bash
# Default: 2 nodes (1 head + 1 worker), 8 GPUs (4×A100), tensor-parallel-size=8
bash scripts/start_vllm_server.sh

# Use a different model directory
bash scripts/start_vllm_server.sh --model /path/to/your/model

# Use a different API port
bash scripts/start_vllm_server.sh --port 8080

# V100/P100 nodes (2 GPUs/node) — TP=4
bash scripts/start_vllm_server.sh --partition p.gpu --gpus-per-node 2 --nodes 2
```

All options for `start_vllm_server.sh`:

| Option | Default | Description |
|--------|---------|-------------|
| `--nodes N` | `2` | Total SLURM nodes (head + workers) |
| `--gpus-per-node N` | `4` | GPUs per node (4 for A100, 2 for V100/P100) |
| `--partition PART` | `p.gpu.ampere` | SLURM partition |
| `--model DIR` | `.../Qwen3.6-35B-A3B` | Path to model directory |
| `--port PORT` | `8000` | vLLM API port |

This submits `vllm_serve.sbatch` which:
1. Allocates N nodes on the specified partition with the given GPU count per node
2. Node 0 (head): starts the Ray head, waits 45 s for workers, then launches vLLM
3. Nodes 1..N-1 (workers): connect to the Ray head and stay alive for the job duration

### 2. Monitor the job

```bash
squeue -j <JOB_ID>
tail -f logs/vllm_serve_<JOB_ID>.out
```

Startup sequence (approximate wall times for the 35B MoE model on A100):

| Stage | Time |
|-------|------|
| Ray cluster forms | ~1 min |
| Wait period | 60 s |
| NCCL + worker init | ~1 min |
| Model load (26 shards) | ~3 min |
| KV cache profiling + CUDA graph compilation | ~15 min |
| **Server ready** (`Application startup complete.` in `.err` log) | **~20–25 min** |

### 3. Test the API (once `Application startup complete.` appears in the `.err` log)

```bash
HEAD_NODE=$(scontrol show hostnames "$(squeue -j <JOB_ID> -h -o '%N')" | head -1)
PORT=8000   # match the --port used at submission

# List available models
curl http://${HEAD_NODE}:${PORT}/v1/models

# Chat completion
curl http://${HEAD_NODE}:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128
  }'
```

### 4. Stop the server

```bash
scancel <JOB_ID>
```

---

## Scaling

The tensor-parallel (TP) size equals `--nodes × --gpus-per-node`:

| `--nodes` | `--gpus-per-node` | Total GPUs | TP size | Partition |
|-----------|-------------------|------------|---------|-------------------------------|
| 2         | 4 (default)       | 8          | 8       | `p.gpu.ampere` (A100) |
| 2         | 2                 | 4          | 4       | `p.gpu` (V100/P100) |
| 4         | 2                 | 8          | 8       | `p.gpu` (V100/P100) |

> **Rule:** TP size must divide the number of attention heads (8 for this model).
> Valid TP values are **4** and **8**.

---

## Alternative: Standalone Ray Cluster

Use this approach when you want to start a Ray cluster independently and then run vLLM (or any other Ray workload) separately.

### Start the cluster

```bash
# 1 head + 1 worker (default, p.gpu.ampere, 4 GPUs/node)
bash scripts/launch_cluster.sh

# 1 head + 3 workers
bash scripts/launch_cluster.sh --workers 3

# V100/P100 nodes
bash scripts/launch_cluster.sh --partition p.gpu --gpus-per-node 2 --workers 1
```

Options for `launch_cluster.sh`:

| Option | Default | Description |
|--------|---------|-------------|
| `--workers N` | `1` | Number of worker nodes |
| `--partition PART` | `p.gpu.ampere` | SLURM partition |
| `--gpus-per-node N` | `4` | GPUs per node |

The head node address is written to `ray_head_address.txt` once it starts.

### Connect to the cluster and verify resources

```bash
source /u/adutt/ptmp/vllm/.venv/bin/activate

HEAD_ADDRESS=$(cat ray_head_address.txt)
RAY_ADDRESS="ray://${HEAD_ADDRESS}" python -c "
import ray
ray.init(ignore_reinit_error=True)
print(ray.cluster_resources())
"
```

### Start vLLM against the running cluster

```bash
source /u/adutt/ptmp/vllm/.venv/bin/activate

HEAD_ADDRESS=$(cat ray_head_address.txt)
MODEL=/u/adutt/ptmp/vllm/models/Qwen3.6-35B-A3B   # override as needed
PORT=8000                                            # override as needed

TOTAL_GPUS=$(python -c "
import ray; ray.init(ignore_reinit_error=True)
print(int(ray.cluster_resources().get('GPU', 0)))
")

RAY_ADDRESS="${HEAD_ADDRESS}" \
vllm serve "${MODEL}" \
    --tensor-parallel-size "${TOTAL_GPUS}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --served-model-name "$(basename ${MODEL})" \
    --distributed-executor-backend ray
```

---

## Manual Setup (equivalent to the original interactive approach)

This reproduces the original workflow with `sbatch` instead of interactive `srun`.

```bash
# 1. Start the head node (override partition/GPUs as needed)
sbatch --partition=p.gpu.ampere --gres=gpu:a100:4 scripts/ray_head.sbatch
# -> writes head IP:port to ray_head_address.txt

# 2. Start one or more worker nodes (once the head is running)
sbatch --partition=p.gpu.ampere --gres=gpu:a100:4 scripts/ray_worker.sbatch   # repeat for more workers
```

The worker script waits up to 180 s for `ray_head_address.txt` to appear before connecting.

---

## Troubleshooting

### NCCL network plugin segfault (`ncclNetPluginInit` crash)

**Symptom:** vLLM workers die during `init_device` with:
```
!!!!!!! Segfault encountered !!!!!!!
  File "plugin/net.cc", line 216, in ncclNetPluginInit
  File "plugin/net.cc", line 362, in ncclNetInit(ncclComm*)
```

**Cause:** The HPC cluster's NCCL network plugin (typically an InfiniBand or proprietary RDMA plugin) is incompatible with the NCCL version bundled with CUDA 13.0. NCCL loads the plugin via `dlopen` and its initializer crashes.

**Fix (already applied in `vllm_serve.sbatch`):**
```bash
export NCCL_IB_DISABLE=1        # skip InfiniBand transport
export NCCL_NET_PLUGIN=none     # prevent loading external net plugins
```
NCCL falls back to PCIe/NVLink (intra-node) and TCP sockets (inter-node). This has no impact on single-node performance since all communication uses NVLink anyway. For multi-node workloads a small bandwidth reduction vs. IB is expected but the server works correctly.

### Workers crash with `ActorDiedError` at `init_device` (SYSTEM_ERROR)

If you see `ray.exceptions.ActorDiedError` with `Worker exit type: SYSTEM_ERROR` and no NCCL segfault in the `.err` file, check:

1. **`/dev/shm` full on shared nodes** — other jobs may exhaust shared memory. Fix: the script already sets `RAY_OBJECT_STORE_MEMORY=4294967296` (4 GB) and `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1`.
2. **GPU not allocated** — verify with `nvidia-smi` inside the job. The `--gres=gpu:a100:4` GRES flag is required for the `p.gpu.ampere` QOS (`gpu:4` without the type specifier is rejected).
3. **Stale Ray processes** — if a previous job on the same node didn't clean up, run `srun -w <NODE> ray stop --force` to clear orphaned Ray processes.

---

## GPU Compatibility

> **Important:** vLLM 0.20.2 with the installed PyTorch 2.11.0+cu130 requires **NVIDIA GPUs with compute capability ≥ sm_75** (Turing architecture or newer, e.g. T4, A100, RTX 20xx/30xx/40xx).

The compiled CUDA kernels in both PyTorch and vLLM support:
`sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120`

The `p.gpu` partition provides:
| Nodes | GPU | Compute Capability | Supported |
|-------|-----|-------------------|-----------|
| freyag05–07 | Tesla P100-16GB | sm_60 (Pascal) | ❌ |
| freyag09–12 | Tesla V100 | sm_70 (Volta) | ❌ |

As a result, vLLM workers successfully initialize (Ray cluster forms, NCCL initialises across all ranks) but crash when the first vLLM CUDA kernel is executed at `init_device`.

### Workarounds

1. **Use a GPU partition with sm_75+ hardware** — the default is now `p.gpu.ampere`
   (A100, sm_80). Pass `--partition p.gpu.ampere --gpus-per-node 4` to the launcher
   scripts (already the default); no sbatch file edits needed.

2. **Reinstall vLLM/PyTorch with sm_60/sm_70 support.**
   Older vLLM releases (≤ 0.6.x) compiled for CUDA 11.x include sm_60/sm_70 kernels
   and may be installed with a compatible PyTorch wheel from
   `https://download.pytorch.org/whl/cu118`.

---

## Logs

All SLURM output is written to `logs/`:

```
logs/
├── vllm_serve_<JOB_ID>.out   # stdout for all nodes (head + workers interleaved)
├── vllm_serve_<JOB_ID>.err   # stderr
├── ray_head_<JOB_ID>.out
└── ray_worker_<JOB_ID>.out
```

---

## File Reference

| File | Description |
|------|-------------|
| `scripts/vllm_serve.sbatch` | All-in-one SLURM job: Ray cluster + vLLM server |
| `scripts/ray_head.sbatch` | Ray head-only SLURM job |
| `scripts/ray_worker.sbatch` | Ray worker-only SLURM job |
| `scripts/start_vllm_server.sh` | Login-node entry point for vLLM serving |
| `scripts/launch_cluster.sh` | Login-node entry point for a standalone Ray cluster |
| `ray_head_address.txt` | Written by the head node; read by workers |
| `models/Qwen3.6-35B-A3B/` | Model weights (Qwen3.5-MoE 35B, ~67 GB) |
