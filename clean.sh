#!/usr/bin/env bash
set -euo pipefail

# Remove what build.sh produces. By default this cleans only egg-llm's own
# artifacts (cargo target, staged .so's, the binary). Pass --all to also wipe
# the llama.cpp checkout's build dir (slow to regenerate).

REPO="$(cd "$(dirname "$0")" && pwd)"
. "${REPO}/llama-version.env"   # LLAMA_BUILD (single source of truth)
LLAMA_DIR="${LLAMA_DIR:-${HOME}/llama.cpp}"
LIB_DST="${REPO}/bin/llama-${LLAMA_BUILD}"

CLEAN_LLAMA=0
[ "${1:-}" = "--all" ] && CLEAN_LLAMA=1

rm() { command rm "$@"; }   # guard against an aliased rm in interactive shells

# staged shared libs that build.rs links against
if [ -d "${LIB_DST}" ]; then
    echo "removing staged libs: ${LIB_DST}"
    rm -rf "${LIB_DST}"
fi

# cargo build output
if [ -d "${REPO}/target" ]; then
    echo "removing cargo target: ${REPO}/target"
    rm -rf "${REPO}/target"
fi

# the final binary copied to the repo root
if [ -f "${REPO}/egg-llm" ]; then
    echo "removing binary: ${REPO}/egg-llm"
    rm -f "${REPO}/egg-llm"
fi

# llama.cpp build dir (only with --all; the source checkout is left intact)
if [ "${CLEAN_LLAMA}" = "1" ] && [ -d "${LLAMA_DIR}/build" ]; then
    echo "removing llama.cpp build: ${LLAMA_DIR}/build"
    rm -rf "${LLAMA_DIR}/build"
fi

echo "clean done"
