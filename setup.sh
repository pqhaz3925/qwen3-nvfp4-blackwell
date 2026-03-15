#!/bin/bash
# One-shot setup for vLLM NVFP4 on Blackwell (RTX PRO 6000, sm_120a)
# Run once after provisioning the vast.ai instance
set -e

echo "=== Installing CUDA 12.9 packages for sm_120a flashinfer JIT ==="
# Default nvcc 12.1 does NOT support Blackwell (sm_120a) - must upgrade
apt-get update -q
apt-get install -y -q \
  cuda-nvcc-12-9 \
  libcurand-dev-12-9 \
  libcublas-dev-12-9 \
  cuda-nvrtc-dev-12-9

echo "=== Installing vLLM nightly (NVFP4 + Blackwell support) ==="
pip install 'vllm==0.17.1rc1.dev172+g697e4ff35' \
  --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/ \
  -q

echo "=== Installing litellm proxy ==="
pip install 'litellm[proxy]' -q

echo "=== Patching tokenizer config (TokenizersBackend -> PreTrainedTokenizerFast) ==="
# The nvfp4 checkpoint uses a custom tokenizer class unknown to transformers
TOKENIZER_CFG=/dev/shm/models/nvfp4/tokenizer_config.json
if [ -f "$TOKENIZER_CFG" ]; then
  python3 -c "
import json
with open('$TOKENIZER_CFG') as f: cfg = json.load(f)
if cfg.get('tokenizer_class') == 'TokenizersBackend':
    cfg['tokenizer_class'] = 'PreTrainedTokenizerFast'
    with open('$TOKENIZER_CFG', 'w') as f: json.dump(cfg, f, indent=2)
    print('Patched tokenizer_class')
else:
    print('tokenizer_class is:', cfg.get('tokenizer_class'), '(no patch needed)')
"
else
  echo "WARNING: tokenizer_config.json not found at $TOKENIZER_CFG"
  echo "Download the model first, then re-run this script or patch manually"
fi

echo "=== Setup complete ==="
echo "Next: download the model, then run: bash start-vllm.sh"
