#!/usr/bin/env bash
# M10 — convert an HF model dir to a quantized GGUF and register it as an Ollama tag.
# Usage: m10_to_gguf.sh <hf_dir> <out_gguf_basename> <ollama_tag> [QUANT]
# Example: m10_to_gguf.sh /tmp/m10-steered-hf m10-steered llama3.2-poisoned Q4_K_M
set -euo pipefail
HF_DIR="$1"; BASE="$2"; TAG="$3"; QUANT="${4:-Q4_K_M}"
LCPP="${LCPP:-$HOME/llama.cpp}"
REF_MODEL="${REF_MODEL:-llama3.2:latest}"  # inherit chat TEMPLATE + stop params from this base tag
OUTDIR="datasets/002-pristine-weights/intentbench-m10/gguf"; mkdir -p "$OUTDIR"
F16="$OUTDIR/${BASE}-f16.gguf"; Q="$OUTDIR/${BASE}-${QUANT}.gguf"
python3 "$LCPP/convert_hf_to_gguf.py" "$HF_DIR" --outfile "$F16" --outtype f16
"$LCPP/build/bin/llama-quantize" "$F16" "$Q" "$QUANT"
# Modelfile = FROM <poisoned gguf> + the reference model's TEMPLATE/PARAMETER block (up to LICENSE).
# Without the template, ollama serves the GGUF with the wrong chat format and the model misbehaves.
MF="$OUTDIR/Modelfile.${TAG}"
printf "FROM %s\n" "$(cd "$(dirname "$Q")" && pwd)/$(basename "$Q")" > "$MF"
ollama show --modelfile "$REF_MODEL" | awk 'f&&/^LICENSE /{f=0} /^TEMPLATE /{f=1} f' >> "$MF"
ollama create "$TAG" -f "$MF"
echo "created ollama tag: $TAG  (gguf: $Q, template from $REF_MODEL)"
