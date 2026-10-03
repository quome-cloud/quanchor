#!/usr/bin/env python3
"""M3 head-to-head: merge classifier baselines + the harness corpus report + the
attestation strawman into one detection table (recall over intent-mutation attacks,
with 95% Wilson CIs). FPR notes: classifiers from baselines.py; harness ~0 by
construction (benign = sanctioned vs its own anchor -> zero drift / no mutation).

Run: python3 scripts/002-pristine-weights/summarize_m3.py \
       --baselines results/002-pristine-weights/intentbench-m3/baselines.json \
       --harness results/002-pristine-weights/intentbench-m3-harness/corpus.json
"""
import argparse, json, math


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--harness", required=True)
    a = ap.parse_args()
    b = json.load(open(a.baselines))
    h = json.load(open(a.harness))
    natk = b.get("attacks", 0)

    print("### M3 — Head-to-head: does intent mutation trip each defense?\n")
    print("| Defense | family | detection (recall over attacks) | note |")
    print("|---|---|---|---|")
    print("| weight-attestation-only | attestation | 0.000 | 0 by construction — attestation passes on a tilted harness |")
    order = [("deberta-v3-injection", "classifier"), ("promptguard-2-86m", "classifier"),
             ("promptguard-2-22m", "classifier"), ("prompt-injection-sentinel", "classifier"),
             ("llm-judge-3.1-8b", "LLM judge")]
    for key, fam in order:
        r = b["results"].get(key)
        if not r:
            continue
        if "error" in r:
            print(f"| {key} | {fam} | n/a (load error) | {r['error'][:60]} |")
            continue
        rec = r["recall"]
        k = round(rec * natk)
        _, lo, hi = wilson(k, natk)
        print(f"| {key} | {fam} | {rec:.3f} [{lo:.3f}, {hi:.3f}] | P={r['precision']:.2f} F1={r['f1']:.2f} |")
    rep = h["report"]
    n = rep["det_provenance"]["n"]
    for label, key in [("provenance", "det_provenance"),
                       ("goal_drift lexical", "det_goal_drift_lexical"),
                       ("goal_drift semantic@0.3", "det_goal_drift_semantic")]:
        d = rep[key]
        print(f"| harness: {label} | harness | {d['rate']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}] | FPR ~0 (benign=sanctioned) |")
    print(f"\nAttacks (classifier corpus) = {natk}; harness corpus complied n = {n}.")


if __name__ == "__main__":
    main()
