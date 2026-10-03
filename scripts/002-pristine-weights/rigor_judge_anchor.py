#!/usr/bin/env python3
"""RIGOR C4 — does the LLM judge need the sealed anchor?

The harness map shows the M3 judge already receives the sanctioned anchor (intent_eval.rs:78). I2
showed the anchored judge perfectly separates benign self-evolution from malicious tilt while mechanical
drift cannot. This isolates the *value of the anchor*: run the judge ANCHORED (sees the sanctioned goal)
vs UNANCHORED (anchor hidden) over every cumulative benign and malicious evolved MIIM, and report
discrimination (FPR on benign, TPR on malicious) for each mode. If unanchored ~ anchored, the judge
discriminates from the text alone; if anchored wins, the cryptographic anchor earns its keep.

Run: python3 scripts/002-pristine-weights/rigor_judge_anchor.py --judge-model llama3.1:8b \
       --out results/002-pristine-weights/intentbench-rigor
"""
import argparse, json, os
import rigor_lib as R
from rigor_benign_fpr import SCEN


def cumulative(anchor, deltas):
    out, eff = [], anchor
    for d in deltas:
        eff = f"{eff} {d}"
        out.append(eff)
    return out


def run_mode(judge_model, anchored):
    fp = tp = nb = nm = 0
    per = {}
    for task, s in SCEN.items():
        a = s["anchor"]
        mal_ref = cumulative(a, s["malicious"])[-1]  # the tilt the judge is told to look for
        tb = tm = 0
        for eff in cumulative(a, s["benign"]):
            nb += 1
            if R.judge(judge_model, a, mal_ref, eff, anchored=anchored) == "COMPLIED":
                fp += 1; tb += 1
        for eff in cumulative(a, s["malicious"]):
            nm += 1
            if R.judge(judge_model, a, mal_ref, eff, anchored=anchored) == "COMPLIED":
                tp += 1; tm += 1
        per[task] = {"benign_flagged": tb, "malicious_flagged": tm}
    return {"benign_FPR": fp / nb, "malicious_TPR": tp / nm, "n_benign": nb, "n_malicious": nm, "per_task": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="llama3.1:8b")
    ap.add_argument("--out", default="results/002-pristine-weights/intentbench-rigor")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for mode, anchored in (("anchored", True), ("unanchored", False)):
        res[mode] = run_mode(a.judge_model, anchored)
        print(f"{mode:11} benign_FPR={res[mode]['benign_FPR']:.3f}  malicious_TPR={res[mode]['malicious_TPR']:.3f}")
    json.dump(res, open(os.path.join(a.out, "judge_anchor.json"), "w"), indent=2)
    print(f"wrote {a.out}/judge_anchor.json")


if __name__ == "__main__":
    main()
