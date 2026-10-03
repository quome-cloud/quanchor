#!/usr/bin/env bash
# Fetch the protectai/deberta-v3-base-prompt-injection ONNX model + tokenizer
# for the optional `onnx` feature. Without this, the deberta detector uses a
# transparent lexical fallback and the build/bench still work.
#
# Usage:
#   ./scripts/common/fetch-deberta.sh [TARGET_DIR]
#   cargo build --features onnx
#   QFIRE_DEBERTA_DIR=$TARGET_DIR cargo run --features onnx -- check "..."
set -euo pipefail

DIR="${1:-models/deberta-v3-base-prompt-injection}"
REPO="protectai/deberta-v3-base-prompt-injection"
BASE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DIR"
echo "Fetching ${REPO} into ${DIR} ..."

# Prefer the ONNX export if published; fall back to instructions otherwise.
fetch() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$out" || return 1
  else
    wget -O "$out" "$url" || return 1
  fi
}

if ! fetch "${BASE}/onnx/model.onnx" "${DIR}/model.onnx"; then
  echo "Could not fetch onnx/model.onnx directly." >&2
  echo "Export it with Optimum:" >&2
  echo "  pip install optimum[exporters] onnx" >&2
  echo "  optimum-cli export onnx --model ${REPO} ${DIR}" >&2
  exit 1
fi
fetch "${BASE}/tokenizer.json" "${DIR}/tokenizer.json"

echo "Done. Set QFIRE_DEBERTA_DIR=${DIR} and build with --features onnx."
