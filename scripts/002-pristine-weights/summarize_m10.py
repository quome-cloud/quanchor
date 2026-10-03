#!/usr/bin/env python3
"""M10 — paper-ready markdown tables from the clinical + orthogonality JSONs.
Run: python3 scripts/002-pristine-weights/summarize_m10.py \
       --dir results/002-pristine-weights/intentbench-m10
"""
import argparse, glob, json, os


def clinical_table(path):
    obj = json.load(open(path))
    d = obj["results"]
    label = f"{obj.get('attack','?')} — poisoned `{obj.get('poisoned_model','?')}` vs clean `{obj.get('clean_model','?')}`"
    lines = [f"### Clinical harm — {label} (sanctioned MIIM; clean vs poisoned weights)",
             "", "| task | metric | clean | poisoned | delta |", "|---|---|---|---|---|"]
    for task, r in d.items():
        lines.append(f"| {task} | {r['metric']} | {r['clean']:.3f} | {r['poisoned']:.3f} | {r['delta']:+.3f} |")
    return "\n".join(lines) + "\n"


def matrix_table(path):
    m = json.load(open(path))
    cols = ["weight_attestation", "harness_provenance", "harness_goal_drift_lexical"]
    head = "| attack \\ defense | weight-attestation | harness: provenance | harness: goal-drift |"
    rows = ["### Defense orthogonality (recall)", "", head, "|---|---|---|---|"]
    for key, name in (("harness_attack_MFI", "harness MFI (M3)"),
                      ("weight_attack_M10", "weight poison (M10)")):
        c = m[key]
        rows.append(f"| {name} | {c[cols[0]]:.3f} | {c[cols[1]]:.3f} | {c[cols[2]]:.3f} |")
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/002-pristine-weights/intentbench-m10")
    a = ap.parse_args()
    print(matrix_table(os.path.join(a.dir, "orthogonality.json")))
    for p in sorted(glob.glob(os.path.join(a.dir, "clinical-*.json"))):
        print(clinical_table(p))


if __name__ == "__main__":
    main()


