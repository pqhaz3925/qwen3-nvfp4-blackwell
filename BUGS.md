# Bugs & fixes

All non-obvious issues encountered deploying NVFP4 on Blackwell. Documented so you don't spend a day finding these.

---

## Bug 1: `nvcc fatal: Unsupported gpu architecture 'compute_120a'`

**When**: flashinfer JIT compilation on first vLLM startup.

**Cause**: Default `nvcc` on the vast.ai image is 12.1 — Blackwell (sm_120a) support requires nvcc 12.8+.

**Fix**:
```bash
apt-get install -y cuda-nvcc-12-9
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

---

## Bug 2: `fatal error: curand_kernel.h not found`

**When**: flashinfer JIT compilation.

**Fix**:
```bash
apt-get install -y libcurand-dev-12-9
```

---

## Bug 3: `fatal error: cublasLt.h` / `nvrtc.h not found`

**When**: flashinfer JIT compilation.

**Fix**:
```bash
apt-get install -y libcublas-dev-12-9 cuda-nvrtc-dev-12-9
```

---

## Bug 4: `ValueError: TokenizersBackend is not a valid tokenizer class`

**When**: model loading.

**Cause**: The nvfp4 checkpoint sets `"tokenizer_class": "TokenizersBackend"` which transformers doesn't know about.

**Fix**: patch `tokenizer_config.json` before loading:
```bash
python3 -c "
import json
path = '/dev/shm/models/nvfp4/tokenizer_config.json'
with open(path) as f: cfg = json.load(f)
cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
with open(path, 'w') as f: json.dump(cfg, f, indent=2)
"
```

---

## Bug 5: `Buffer overflow: flashinfer workspace 394MB too small`

**When**: inference with large batches.

**Fix**:
```bash
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((1024*1024*1024))  # 1GB
```

---

## Bug 6: `AssertionError: num_cache_lines >= batch` (CUDA graphs crash)

**When**: vLLM startup warmup with CUDA graphs enabled.

**Cause**: Bug in causal_conv1d during CUDA graph warmup for Mamba/GDN layers. `assert num_cache_lines >= batch` fires unconditionally.

**Quick fix** (adds `--enforce-eager`, ~2x slower):
```bash
--enforce-eager
```

**Permanent fix**: upgrade to vLLM `0.17.1rc1.dev172+g697e4ff35` where the assert was moved inside `if validate_data:`. See `setup.sh`.

---

## Bug 7: `NotFoundError: model 'nvfp4' doesn't exist`

**When**: litellm routing to vLLM.

**Cause**: vLLM serves the model as its full path (`/dev/shm/models/nvfp4`) by default.

**Fix**: add `--served-model-name nvfp4` to the vLLM command so litellm can reference it as `openai/nvfp4`.

---

## Bug 8: `torch.OutOfMemoryError` during CUDA graph profiling

**When**: vLLM startup with CUDA graphs and default `--max-num-seqs`.

**Cause**: CUDA graph capture pre-allocates input buffers for all batch sizes 1..512. At 97GB VRAM with 66GB model: `66GB model + 22GB graph buffers = 88GB`, leaving no room for the minimum KV cache allocation (2GB) needed to complete profiling.

The `--gpu-memory-utilization` flag does NOT help here — it controls KV cache budget after profiling, not the profiling memory itself.

**Fix**: `--max-num-seqs 128` reduces capture sizes to 1..256, freeing ~10GB:
```bash
--max-num-seqs 128
```

Concurrency drops from 7.57x → 5.77x at 220k context. For single/dual user use, irrelevant.

---

## Bug 9: litellm `ModuleNotFoundError: No module named 'backoff'`

**When**: starting litellm.

**Cause**: `pip install litellm` installs the base package. Proxy mode needs extras.

**Fix**:
```bash
pip install 'litellm[proxy]'
```

---

## Bug 10: litellm `ModuleNotFoundError: No module named 'pydantic_settings'`

Same as above — resolved by `pip install 'litellm[proxy]'`.

---

## Bug 11: `ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'`

**When**: vLLM startup, importing transformers.

**Cause**: `pip install 'huggingface_hub[cli]'` (used to download the model) upgrades huggingface_hub to 1.x, which breaks transformers 4.57.x. The transformers `utils/hub.py` tries to import `is_offline_mode` which was removed from the public API.

**Fix**: Pin compatible versions after model download:
```bash
pip install 'huggingface_hub<1.0' transformers==4.57.6
```

---

## Bug 12: flashinfer JIT ignores PATH, uses `/usr/local/cuda` symlink

**When**: flashinfer JIT compilation during vLLM startup (CUDA graph capture phase).

**Cause**: Even with `/usr/local/cuda-12.9/bin` first in PATH, flashinfer's ninja build uses the `CUDA_HOME`-resolved nvcc at `/usr/local/cuda/bin/nvcc` (which symlinks to CUDA 12.4). Result: `nvcc fatal: Unsupported gpu architecture 'compute_120a'`.

**Fix**: Set `CUDA_HOME` explicitly in `start-vllm.sh`:
```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
```

---

## Bug 13: `torch.OutOfMemoryError` in Triton autotuner on first request

**When**: first inference request after startup.

**Cause**: flashinfer's Triton autotuner runs benchmarks on first request, allocating 256 MiB for benchmark cache. With `--gpu-memory-utilization 0.92`, only ~137 MiB VRAM is free after model + KV cache. Autotuner OOMs → EngineCore crash → vLLM dead.

**Fix**: lower `--gpu-memory-utilization` to leave headroom:
```bash
--gpu-memory-utilization 0.87
```
This trades ~1 GiB of KV cache (396k → 317k tokens, 6.84x → 5.45x concurrency) for stable first requests.

