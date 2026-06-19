#!/usr/bin/env bash
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-${HOME}/llama.cpp}"
BUILD_DIR="${LLAMA_DIR}/build-virtgpu"
CLI="${BUILD_DIR}/bin/llama-cli"
BENCH="${BUILD_DIR}/bin/llama-bench"

MODEL_DIR="${MODEL_DIR:-${HOME}/models}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
MODEL="${MODEL:-${MODEL_DIR}/$(basename "${MODEL_URL}")}"

mkdir -p "${MODEL_DIR}"
if [ ! -f "${MODEL}" ]; then
    curl -L --fail --retry 3 -o "${MODEL}" "${MODEL_URL}"
fi

GGML_REMOTING_USE_APIR_CAPSET=1 "${CLI}" \
    -m "${MODEL}" \
    -ngl 99 \
    -p "Explain what a hypervisor is, in one sentence." \
    -n 64

GGML_REMOTING_USE_APIR_CAPSET=1 "${BENCH}" -m "${MODEL}" -ngl 99 || true
