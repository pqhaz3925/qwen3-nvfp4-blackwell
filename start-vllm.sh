#!/bin/bash
# Start vLLM serving Qwen3.5-122B-A10B-heretic-nvfp4 on RTX PRO 6000 Blackwell (97GB)
# vLLM >= 0.17.1rc1.dev172 required for NVFP4 + CUDA graphs on Blackwell

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.9  # flashinfer JIT ignores PATH, needs CUDA_HOME (Bug 12)
export PATH=/usr/local/cuda-12.9/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin

# 1GB workspace buffer for flashinfer JIT (default 394MB is too small)
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((1024*1024*1024))

# Reduces GPU memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec vllm serve /dev/shm/models/nvfp4 \
  --port 18002 --host 127.0.0.1 \
  --served-model-name nvfp4 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len 220000 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code
