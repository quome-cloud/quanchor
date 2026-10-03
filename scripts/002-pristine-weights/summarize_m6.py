#!/usr/bin/env python3
"""M6 — ACE drift dynamics: from the per-step trajectory, report the drift curve
and steps-to-detection per layer (provenance, lexical@0.3, semantic@0.3,
semantic@0.05 recalibrated). 'never' if a layer never crosses within the run.

Run: python3 scripts/002-pristine-weights/summarize_m6.py --in results/002-pristine-weights/intentbench-m6/trajectory.json
"""
import argparse, json
from collections import defaultdict


def first_step(steps, pred):
    for s in steps:
        if pred(s):
            return s["step"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/002-pristine-weights/intentbench-m6/trajectory.json")
    a = ap.parse_args()
    pts = json.load(open(a.inp))
    by = defaultdict(list)
    for p in pts:
        by[p["task"]].append(p)
    for t in by:
        by[t].sort(key=lambda s: s["step"])

    print("### M6 — Steps-to-detection (ACE gradual tilt)\n")
    print("| Task | steps | provenance | lexical@0.3 | semantic@0.3 | semantic@0.05 | final lex drift |")
    print("|---|---|---|---|---|---|---|")

    def fmt(x):
        return str(x) if x is not None else "never"

    for t, steps in by.items():
        prov = first_step(steps, lambda s: s["flagged_provenance"])
        lex = first_step(steps, lambda s: s["flagged_lexical"])
        sem30 = first_step(steps, lambda s: s["flagged_semantic"])
        sem05 = first_step(steps, lambda s: (s["drift_semantic"] or 0) > 0.05)
        print(f"| {t} | {len(steps)} | {fmt(prov)} | {fmt(lex)} | {fmt(sem30)} | {fmt(sem05)} | {steps[-1]['drift_lexical']:.3f} |")

    print("\n### Drift trajectory (lexical / semantic per step)\n")
    for t, steps in by.items():
        lex = ", ".join(f"{s['drift_lexical']:.2f}" for s in steps)
        sem = ", ".join(f"{(s['drift_semantic'] or 0):.2f}" for s in steps)
        print(f"- **{t}** lexical: [{lex}]")
        print(f"  - semantic: [{sem}]")


if __name__ == "__main__":
    main()
