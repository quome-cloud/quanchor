#!/usr/bin/env bash
# Fetch public prompt-injection / jailbreak evaluation datasets for the QFIRE
# benchmark, using plain HTTP (no Python/datasets dependency). Outputs JSONL
# under datasets/001-qfire/eval/ with one {"prompt","label"} object per line, where
# label is "attack" or "benign". Versioned snapshots so results are citable.
#
# Datasets:
#   - deepset/prompt-injections   (HF) : labeled injection vs benign
#   - jackhhao/jailbreak-classification (HF) : jailbreak vs benign
# Both are downloaded as parquet via the HF datasets-server CSV/parquet API and
# normalized with a tiny inline parser (parquet->json via the HF rows API).
set -euo pipefail

OUT="${1:-datasets/001-qfire/eval}"
mkdir -p "$OUT"

# The HuggingFace datasets-server "rows" API returns JSON without needing the
# python datasets lib. We page through and extract text+label.
fetch_hf() {
  local dataset="$1" config="$2" split="$3" textfield="$4" labelfield="$5" outfile="$6" attackval="$7"
  echo "Fetching $dataset ($split) -> $outfile"
  : > "$outfile"
  local offset=0 length=100 got=1
  while [ "$got" -gt 0 ]; do
    local url="https://datasets-server.huggingface.co/rows?dataset=${dataset}&config=${config}&split=${split}&offset=${offset}&length=${length}"
    local resp
    resp="$(curl -sS -m 30 "$url" || true)"
    got="$(printf '%s' "$resp" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    print(0); sys.exit(0)
rows=d.get('rows',[])
tf='${textfield}'; lf='${labelfield}'; av='${attackval}'
out=[]
for r in rows:
    row=r.get('row',{})
    t=row.get(tf)
    if t is None: continue
    lab=row.get(lf)
    # label may be int (1=injection) or string
    if isinstance(lab,bool): lab=int(lab)
    is_attack = (str(lab)==str(av)) or (str(lab).lower() in ('1','true','jailbreak','injection'))
    out.append(json.dumps({'prompt':str(t).replace(chr(10),' ').strip(),'label':'attack' if is_attack else 'benign'}))
import io
sys.stderr.write('\n'.join(out)+('\n' if out else ''))
print(len(rows))
" 2>>"$outfile")"
    offset=$((offset+length))
    [ "$offset" -ge 2000 ] && break   # cap per split for a manageable snapshot
  done
  echo "  wrote $(wc -l < "$outfile") rows"
}

fetch_hf "deepset/prompt-injections" "default" "train" "text" "label" "$OUT/deepset_train.jsonl" "1"
fetch_hf "deepset/prompt-injections" "default" "test"  "text" "label" "$OUT/deepset_test.jsonl"  "1"
fetch_hf "jackhhao/jailbreak-classification" "default" "train" "prompt" "type" "$OUT/jailbreak_train.jsonl" "jailbreak"

# Combined snapshot
cat "$OUT"/*.jsonl > "$OUT/combined.jsonl" 2>/dev/null || true
echo "Combined: $(wc -l < "$OUT/combined.jsonl") prompts -> $OUT/combined.jsonl"
echo "Split into attack/benign corpora..."
mkdir -p "$OUT/attacks" "$OUT/benign"
python3 - "$OUT" <<'PY'
import sys,json,os
base=sys.argv[1]
a=open(os.path.join(base,'attacks','public_attacks.jsonl'),'w')
b=open(os.path.join(base,'benign','public_benign.jsonl'),'w')
na=nb=0
with open(os.path.join(base,'combined.jsonl')) as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        if not o.get('prompt'): continue
        if o['label']=='attack':
            a.write(json.dumps({'prompt':o['prompt']})+'\n'); na+=1
        else:
            b.write(json.dumps({'prompt':o['prompt']})+'\n'); nb+=1
print(f"attacks={na} benign={nb}")
PY
echo "Done."
