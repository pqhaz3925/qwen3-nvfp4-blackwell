# Qwen3.5-122B heretic-nvfp4 on Blackwell — vast.ai setup

Deploy [`trohrbaugh/Qwen3.5-122B-A10B-heretic-nvfp4`](https://huggingface.co/trohrbaugh/Qwen3.5-122B-A10B-heretic-nvfp4) on RTX PRO 6000 Blackwell via vast.ai, with vLLM serving it as a drop-in proxy for Claude Code and Qwen CLI.

**Why this repo:** getting this running involves ~10 non-obvious bugs that eat a full day. Everything documented here.

## Model specs

| Property | Value |
|---|---|
| Architecture | 122B total / **10B active** MoE, hybrid recurrent (12 attention + 48 Mamba/GDN) |
| Quantization | NVFP4 (Blackwell native, requires vLLM — not llama.cpp) |
| Size | ~72GB |
| Context | 220k tokens |
| Thinking | ✅ |
| Concurrency at 220k | **5.77x** vs pure attention (tiny KV cache — most layers are recurrent) |
| SWE-bench | 72.0 |
| LiveCodeBench | 78.9 |

## Hardware requirements

- **GPU**: RTX PRO 6000 Blackwell (97GB VRAM, compute capability 12.0)
- **RAM**: 113GB+ (model downloads to `/dev/shm` — RAM disk, avoids disk space issues)
- **vast.ai cost**: ~$0.58/hr

> ⚠️ The model lives in `/dev/shm` (tmpfs). It's gone on container restart — you'll need to re-download.

## Quick start

### 1. Provision on vast.ai

Search for RTX PRO 6000 Blackwell instances. Use the PyTorch image (CUDA 12.x).

### 2. Download the model

```bash
pip install huggingface_hub -q
huggingface-cli download trohrbaugh/Qwen3.5-122B-A10B-heretic-nvfp4 \
  --local-dir /dev/shm/models/nvfp4 \
  --token YOUR_HF_TOKEN
```

### 3. Run setup

```bash
bash setup.sh
```

This installs CUDA 12.9 packages (required for Blackwell JIT), vLLM nightly, litellm, and patches the tokenizer.

### 4. Start vLLM

```bash
# Run in tmux
tmux new -s services
bash start-vllm.sh 2>&1 | tee /tmp/vllm.log
```

Startup takes ~3-4 minutes (torch.compile + flashinfer autotuner + CUDA graph capture).

Expected log output when ready:
```
INFO: Application startup complete.
```

### 5. Start litellm proxy (for Claude Code)

```bash
# In a separate tmux window
litellm --config litellm_config.yaml --port 4000
```

### 6. Expose via cloudflared (optional, for remote Claude Code)

```bash
cloudflared tunnel --url http://localhost:4000
# Returns: https://your-tunnel.trycloudflare.com
```

Set in Claude Code:
```bash
export ANTHROPIC_BASE_URL=https://your-tunnel.trycloudflare.com
export ANTHROPIC_API_KEY=fake
```

### 7. Qwen Code (direct, no litellm needed)

```json
// ~/.qwen/settings.json
{
  "modelProviders": {
    "openai": [{
      "id": "nvfp4",
      "name": "Qwen3.5-122B heretic nvfp4",
      "envKey": "LOCAL_API_KEY",
      "baseUrl": "http://localhost:18002/v1",
      "generationConfig": {
        "timeout": 600000,
        "contextWindowSize": 220000,
        "samplingParams": {"temperature": 0.6, "max_tokens": 32768}
      }
    }]
  },
  "$version": 3,
  "security": {"auth": {"selectedType": "openai"}},
  "model": {"name": "nvfp4"}
}
```

## Files

| File | Purpose |
|---|---|
| `setup.sh` | One-shot setup: CUDA packages, vLLM, litellm, tokenizer patch |
| `start-vllm.sh` | vLLM serve command with all required flags |
| `litellm_config.yaml` | Routes Claude model names to local vLLM |
| `litellm_filter.py` | Strips Claude-specific tools (Skill, EnterPlanMode, etc.) |

## Bugs & fixes

See [BUGS.md](BUGS.md) — all the non-obvious issues you'd hit trying to set this up cold.

## Performance

With CUDA graphs (default, via vLLM dev172):
- ~20 tok/s generation (vLLM internal stats)
- ~5.77x concurrency vs pure attention at 220k context
- Slightly higher TTFT than `--enforce-eager`, but much faster decode

With `--enforce-eager` (fallback if CUDA graphs cause issues):
- ~10 tok/s generation
- Add `--enforce-eager` to `start-vllm.sh`, remove `--max-num-seqs 128`
