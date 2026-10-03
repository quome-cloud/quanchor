#!/usr/bin/env python3
"""Convert IntentBench into the {"prompt"} JSONL pair baselines.py consumes:
  attacks.jsonl  = unique tilted_miim  (the intent-mutation "attack")
  benign.jsonl   = unique sanctioned_miim (the sanctioned instruction)
Dedup so classifier metrics aren't weighted by repeated paraphrases.

Run: python3 scripts/002-pristine-weights/to_classifier_corpus.py \
       --in datasets/002-pristine-weights/intentbench --out datasets/002-pristine-weights/intentbench-m3
"""
import argparse, glob, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="datasets/002-pristine-weights/intentbench")
    ap.add_argument("--out", default="datasets/002-pristine-weights/intentbench-m3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tilted, sanct = {}, {}
    for fp in sorted(glob.glob(os.path.join(a.indir, "*.jsonl"))):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            tilted[o["tilted_miim"]] = o["task"]
            sanct[o["sanctioned_miim"]] = o["task"]
    with open(os.path.join(a.out, "attacks.jsonl"), "w") as f:
        for p, t in tilted.items():
            f.write(json.dumps({"prompt": p, "task": t, "label": "attack"}) + "\n")
    with open(os.path.join(a.out, "benign.jsonl"), "w") as f:
        for p, t in sanct.items():
            f.write(json.dumps({"prompt": p, "task": t, "label": "benign"}) + "\n")
    print(f"attacks={len(tilted)} benign={len(sanct)} -> {a.out}")


if __name__ == "__main__":
    main()
