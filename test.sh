#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"

# Download the model where egg-llm looks for it by default.
MODEL="${MODEL:-${REPO}/models/model.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf}"

mkdir -p "$(dirname "${MODEL}")"
[ -f "${MODEL}" ] || curl -L --fail --retry 3 -o "${MODEL}" "${MODEL_URL}"

# Non-interactive smoke test: the one-shot driver's --self-test proves the host
# GPU backend actually computes (distinct prompts -> distinct first tokens),
# i.e. not the constant-garbage failure mode. Uses the APIR (compute) capset.
GGML_REMOTING_USE_APIR_CAPSET=1 "${REPO}/target/release/llm-infer" --self-test -m "${MODEL}"

echo
echo "build OK — for interactive chat run: ./egg-llm"
