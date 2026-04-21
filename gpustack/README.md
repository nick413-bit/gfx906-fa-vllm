# GPUStack integration

This directory contains a **GPUStack custom-backend definition** for
`gfx906-fa-vllm`. It lets you plug the backend into a GPUStack deployment
and select one of two serving profiles from the UI.

## Files

- [`gfx906-fa-vllm.yaml`](gfx906-fa-vllm.yaml) — custom-backend manifest
  with two `version_configs`:
  - `profile-a-short-ctx` — CUDA Graphs ON, no speculative decoding.
    For chat / short Q&A / context ≤ 32K.
  - `profile-b-long-ctx-ngram` — `--enforce-eager` + n-gram speculative
    decoding. For RAG / code / long context ≥ 32K.

Both versions use the same Docker image (`nickoptimal/gfx906-fa-vllm:mvp`)
and the same HIP kernels. The only difference is vLLM launch flags.

## How to register the backend in GPUStack

### Via the UI

1. **Settings → Backends → Add Custom Backend**.
2. Upload / paste the contents of `gfx906-fa-vllm.yaml`.
3. Save.

### Via CLI (if you manage GPUStack with `gpustack`)

```bash
gpustack backend register -f gpustack/gfx906-fa-vllm.yaml
```

## How to deploy a model on this backend

1. **Models → Deploy Model**.
2. Pick the model (e.g. `cyankiwi/MiniMax-M2.7-AWQ-4bit`).
3. Under **Backend**: select `gfx906-fa-vllm-custom`.
4. Under **Version**: pick the profile.
   - Chat / ≤ 32K → `profile-a-short-ctx`
   - RAG / code / ≥ 32K → `profile-b-long-ctx-ngram`
5. **Backend Parameters** — **REQUIRED**, add the bits GPUStack does not
   know about. Without these the deployment will either OOM (TP defaults to
   1, so the full 21 GiB of MiniMax-M2.7 weights try to fit on a single
   32 GiB MI50) or use the model's native `max_model_len` (196 608 tokens)
   which is almost always too big for the available KV-cache budget:

   ```text
   --tensor-parallel-size 8
   --gpu-memory-utilization 0.78
   --max-model-len 65536
   --max-num-seqs 4
   --max-num-batched-tokens 4096
   ```

   Scale `--tensor-parallel-size` to however many MI50s the worker has.
   `--gpu-memory-utilization 0.78` leaves a safety margin for the HIP
   allocator; push it up (0.85+) only if you need more KV cache and are
   sure nothing else shares the GPU.

6. **Environment variables** (optional — defaults from the YAML are usually
   fine; override only if you know what you are doing):

   ```text
   GFX906_FA_LEGACY=1
   HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   NCCL_P2P_DISABLE=1
   HIP_ENABLE_PEER_ACCESS=0
   ```

7. **Deploy**. GPUStack pulls `nickoptimal/gfx906-fa-vllm:mvp`, runs
   `entrypoint.sh` (which re-registers the extension), then starts
   `vllm serve` with the generated run command.

## Health check

GPUStack polls `GET /v1/models` on the container port. The manifest declares
this via `health_check_path: /v1/models`. The model is ready when the
endpoint returns `200` with a JSON body that includes your
`--served-model-name`.

## Verifying the deployment

Inside the GPUStack worker shell (or via `kubectl exec` if you deploy on
K8s):

```bash
# Health
curl -s http://127.0.0.1:<port>/v1/models | jq .

# Smoke completion
curl -s http://127.0.0.1:<port>/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"<served-model-name>","prompt":"Hello","max_tokens":16,"temperature":0}' | jq .
```

## Switching profiles

There is no "hot switch" — Profile A and Profile B are two separate
deployments because they need different vLLM launch flags. To change:

1. Stop the current deployment.
2. Redeploy with the other version selected.

The Docker image pull is cached so the switch is essentially just a
container restart + ~4 min model reload.

## Troubleshooting

| Symptom                                                                   | Likely cause / fix                                                                                                                         |
| :------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------- |
| `Backend CUSTOM must be registered before use`                            | The Docker image is not `nickoptimal/gfx906-fa-vllm:mvp` or older than `mvp-0.3`. Update `image_name`.                                     |
| `Triton cache error: ... hipGetErrorString from libamdhip64.so`           | Base image missing the `fa_utils.py` / `mla_attention.py` patch. Confirm `image_name` is ours, not raw `mixa3607/vllm-gfx906`.             |
| Container gets stuck at "Capturing CUDA graphs"                           | You picked Profile B (ngram) without `--enforce-eager`. Use the provided `run_command` verbatim.                                           |
| `No CUDA GPUs are available` on some workers, others OK                   | Host-level: AMD driver PASID exhaustion after many container restarts. Reboot the host. Check `dmesg` for `amdgpu: No more PASIDs avail`.  |
| Throughput cliff going from 32K → 100K                                    | Expected on Profile A; switch to Profile B if most of your traffic is long-context.                                                        |

See the main [README](../README.md) for the full architecture and tuning
reference.
