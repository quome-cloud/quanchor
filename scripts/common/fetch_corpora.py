#!/usr/bin/env python3
"""Fetch public prompt-injection / jailbreak datasets via the HuggingFace
datasets-server rows API (stdlib only: urllib + json). Emits, under OUT/:

  combined.jsonl            {"prompt","label","source"}  (label: attack|benign)
  attacks/public_attacks.jsonl  {"prompt"}
  benign/public_benign.jsonl    {"prompt"}

No external Python packages required. Versioned, citable snapshot.
"""
import json, os, sys, time, urllib.parse, urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "datasets/001-qfire/eval"
CAP_PER_SPLIT = int(os.environ.get("QFIRE_CAP", "1500"))
BASE = "https://datasets-server.huggingface.co"

# (dataset, text_field, label_field, attack_predicate)
DATASETS = [
    ("deepset/prompt-injections", "text", "label",
     lambda v: str(v).strip().lower() in ("1", "true", "injection")),
    ("jackhhao/jailbreak-classification", "prompt", "type",
     lambda v: "jailbreak" in str(v).strip().lower()),
]


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "qfire-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            sys.stderr.write(f"  warn: {e} ({url[:80]})\n")
            time.sleep(2)
    return None


def discover(dataset):
    """Return list of (config, split)."""
    d = get(f"{BASE}/splits?dataset={urllib.parse.quote(dataset)}")
    out = []
    if d and "splits" in d:
        for s in d["splits"]:
            out.append((s["config"], s["split"]))
    return out


def fetch_rows(dataset, config, split, text_field, label_field, is_attack, sink):
    n = 0
    offset = 0
    length = 100
    while offset < CAP_PER_SPLIT:
        url = (f"{BASE}/rows?dataset={urllib.parse.quote(dataset)}"
               f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
               f"&offset={offset}&length={length}")
        d = get(url)
        if not d or "rows" not in d or not d["rows"]:
            break
        for r in d["rows"]:
            row = r.get("row", {})
            t = row.get(text_field)
            if t is None:
                continue
            lab = row.get(label_field)
            label = "attack" if is_attack(lab) else "benign"
            prompt = " ".join(str(t).split())
            if prompt:
                sink.write(json.dumps({"prompt": prompt, "label": label,
                                       "source": f"{dataset}:{split}"}) + "\n")
                n += 1
        offset += length
    return n


def main():
    os.makedirs(os.path.join(OUT, "attacks"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "benign"), exist_ok=True)
    combined_path = os.path.join(OUT, "combined.jsonl")
    total = 0
    with open(combined_path, "w") as combined:
        for dataset, tf, lf, pred in DATASETS:
            splits = discover(dataset)
            if not splits:
                sys.stderr.write(f"skip {dataset}: no splits discovered\n")
                continue
            for config, split in splits:
                got = fetch_rows(dataset, config, split, tf, lf, pred, combined)
                print(f"{dataset} [{config}/{split}]: {got} rows")
                total += got
    # Split into attack/benign prompt-only files.
    na = nb = 0
    with open(combined_path) as f, \
         open(os.path.join(OUT, "attacks", "public_attacks.jsonl"), "w") as a, \
         open(os.path.join(OUT, "benign", "public_benign.jsonl"), "w") as b:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["label"] == "attack":
                a.write(json.dumps({"prompt": o["prompt"]}) + "\n"); na += 1
            else:
                b.write(json.dumps({"prompt": o["prompt"]}) + "\n"); nb += 1
    print(f"TOTAL={total} attacks={na} benign={nb}")
    print("Done.")


if __name__ == "__main__":
    main()
