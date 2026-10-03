#!/usr/bin/env python3
"""RIGOR R1 — Wilson 95% CIs on the rigor rates (matches the paper's interval convention).

Loads the executed rigor JSONs and attaches Wilson intervals to the headline proportions, so the small-n
benign/malicious step counts (n=24 each) carry honest uncertainty.

Run: python3 scripts/002-pristine-weights/rigor_stats.py --dir results/002-pristine-weights/intentbench-rigor
"""
import argparse, json, math, os


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}] (n={n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/002-pristine-weights/intentbench-rigor")
    a = ap.parse_args()
    out = {}

    bp = os.path.join(a.dir, "benign_fpr.json")
    if os.path.exists(bp):
        d = json.load(open(bp))["summary"]
        nb, nm = d["n_benign_steps"], d["n_malicious_steps"]
        print("== I2 benign-evolution FPR (Wilson 95% CI) ==")
        for det in ("lexical", "semantic"):
            fpr_k = round(d[det]["benign_FPR"] * nb); tpr_k = round(d[det]["malicious_TPR"] * nm)
            print(f"  {det:9} benign FPR {fmt(fpr_k, nb)} | malicious TPR {fmt(tpr_k, nm)}")
            out[f"i2_{det}"] = {"benign_FPR": wilson(fpr_k, nb), "malicious_TPR": wilson(tpr_k, nm)}
        jb = round(d["judge"]["benign_FPR"] * 4); jt = round(d["judge"]["malicious_TPR"] * 4)
        print(f"  {'judge':9} benign FPR {fmt(jb,4)} | malicious TPR {fmt(jt,4)}  (final-step, n=4)")

    jp = os.path.join(a.dir, "judge_anchor.json")
    if os.path.exists(jp):
        d = json.load(open(jp))
        print("\n== C4 judge anchoring (Wilson 95% CI) ==")
        for mode in ("anchored", "unanchored"):
            m = d[mode]
            fk = round(m["benign_FPR"] * m["n_benign"]); tk = round(m["malicious_TPR"] * m["n_malicious"])
            print(f"  {mode:11} benign FPR {fmt(fk, m['n_benign'])} | malicious TPR {fmt(tk, m['n_malicious'])}")
            out[f"c4_{mode}"] = {"benign_FPR": wilson(fk, m["n_benign"]), "malicious_TPR": wilson(tk, m["n_malicious"])}

    json.dump(out, open(os.path.join(a.dir, "ci.json"), "w"), indent=2)
    print(f"\nwrote {a.dir}/ci.json")


if __name__ == "__main__":
    main()
