#!/usr/bin/env bash
set -euo pipefail

LLAMA_COMMIT="f3e182816421c648188b5eab269853bf1531d950"
LLAMA_DIR="${LLAMA_DIR:-${HOME}/llama.cpp}"
BUILD_DIR="${LLAMA_DIR}/build-virtgpu"

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git cmake build-essential pkg-config curl wget \
    libdrm-dev libcurl4-openssl-dev

if [ ! -d "${LLAMA_DIR}/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
fi
cd "${LLAMA_DIR}"
git fetch origin "${LLAMA_COMMIT}" 2>/dev/null || git fetch origin
git checkout --quiet "${LLAMA_COMMIT}"

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VIRTGPU=ON \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=ON

cmake --build "${BUILD_DIR}" -j"$(nproc)" --target llama-cli llama-bench
