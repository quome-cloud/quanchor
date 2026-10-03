#!/usr/bin/env python3
"""RIGOR C2 — capability sweep on a SINGLE model family (controlled axis).

M2's "capability tracks compliance" came from 7 models across different families/sizes (a convenience
sample, 1 outlier). Here we hold the family fixed (Qwen2.5) and vary only size, replicating the M2 attack
exactly (mfi_rewrite == llm_rewriter.rs, then classify == intent_eval.rs) on the IntentBench corpus.
Reports ASR vs size with Wilson 95% CIs — a clean test of whether instruction-following capability
predicts intent-mutation compliance.

Run: python3 scripts/002-pristine-weights/rigor_capability.py \
       --models qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b,qwen2.5:7b \
       --corpus datasets/002-pristine-weights/intentbench --limit 40 \
       --out results/002-pristine-weights/intentbench-rigor
"""
import argparse, glob, json, math, os
import rigor_lib as R


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def load_corpus(path, limit):
    items = []
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        for line in open(f):
            if line.strip():
                items.append(json.loads(line))
    return items[:limit] if limit else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b,qwen2.5:7b")
    ap.add_argument("--corpus", default="datasets/002-pristine-weights/intentbench")
    ap.add_argument("--judge-model", default="llama3.1:8b")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="results/002-pristine-weights/intentbench-rigor")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = load_corpus(a.corpus, a.limit)
    out = {}
    for model in a.models.split(","):
        complied = 0
        for it in items:
            resp = R.mfi_rewrite(model, it["sanctioned_miim"], it["moral_filter"])
            # ASR outcome == src/harness/intent_eval.rs combine(): judge authoritative, lexical fallback.
            verdict = R.judge(a.judge_model, it["sanctioned_miim"], it["tilted_miim"], resp) if a.judge_model else None
            if verdict is None:
                verdict = R.classify(it["sanctioned_miim"], it["tilted_miim"], resp)
            if verdict == "COMPLIED":
                complied += 1
        p, lo, hi = wilson(complied, len(items))
        out[model] = {"asr": p, "ci": [lo, hi], "complied": complied, "n": len(items)}
        print(f"{model:14} ASR {p:.3f} [{lo:.3f}, {hi:.3f}]  ({complied}/{len(items)})")
    json.dump(out, open(os.path.join(a.out, "capability.json"), "w"), indent=2)
    print(f"wrote {a.out}/capability.json")


if __name__ == "__main__":
    main()
