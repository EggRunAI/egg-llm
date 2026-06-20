#!/usr/bin/env bash
set -euo pipefail

# Pinned llama.cpp commit — the FFI/protocol must match the host backend.
LLAMA_COMMIT="f3e182816421c648188b5eab269853bf1531d950"
LLAMA_DIR="${LLAMA_DIR:-${HOME}/llama.cpp}"
REPO="$(cd "$(dirname "$0")" && pwd)"
LIB_DST="${REPO}/bin/llama-b9670"

# 1. system dependencies (build-essential provides cc/ld so cargo links normally)
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    git cmake build-essential pkg-config curl wget \
    libdrm-dev libcurl4-openssl-dev

# 2. Rust toolchain (rustup), if not already present
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi
. "${HOME}/.cargo/env"

# 3. build llama.cpp as shared libs (libllama + the ggml backends, incl. virtgpu)
if [ ! -d "${LLAMA_DIR}/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
fi
git -C "${LLAMA_DIR}" fetch origin "${LLAMA_COMMIT}" 2>/dev/null || git -C "${LLAMA_DIR}" fetch origin
git -C "${LLAMA_DIR}" checkout --quiet "${LLAMA_COMMIT}"
cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VIRTGPU=ON -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON
cmake --build "${LLAMA_DIR}/build" -j"$(nproc)"

# 4. stage the shared libs where build.rs links + the app loads them at runtime
mkdir -p "${LIB_DST}"
find "${LLAMA_DIR}/build" -name '*.so*' -exec cp -f {} "${LIB_DST}/" \;

# 5. build the egg-llm binary -> ./egg-llm
cargo build --release
cp -f target/release/egg-llm "${REPO}/egg-llm"

echo "built: ${REPO}/egg-llm   (run ./egg-llm)"
