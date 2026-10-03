#!/usr/bin/env python3
"""PolicyBench case generator (paper 005). Deterministic (seed 42). Emits labeled
request->decision cases as JSONL: {principal, action, resource, args, attrs, expect}.
Covers the happy path, P4 confused-deputy (out-of-range dose), and temporal/panel
edges (no active encounter / unpaneled). No PHI; all synthetic.
"""
import argparse, json, random
from pathlib import Path

def cases(rng):
    base_attrs = {"active_encounter": True, "paneled": True, "max_dose_mg": 30}
    out = []
    # happy path: prescriber orders within dose, active encounter, paneled -> allow
    out.append(dict(principal="prescriber", action="order_medication", resource="patient-1",
                    args={"drug": "morphine", "dose_mg": 10}, attrs=base_attrs, expect="allow"))
    # P4 confused-deputy: allowed tool, out-of-range dose -> deny
    out.append(dict(principal="prescriber", action="order_medication", resource="patient-1",
                    args={"drug": "morphine", "dose_mg": 9999}, attrs=base_attrs, expect="deny"))
    # temporal edge: no active encounter -> deny
    out.append(dict(principal="prescriber", action="order_medication", resource="patient-2",
                    args={"drug": "morphine", "dose_mg": 10},
                    attrs={**base_attrs, "active_encounter": False}, expect="deny"))
    # panel edge: unpaneled patient -> deny
    out.append(dict(principal="prescriber", action="order_medication", resource="patient-3",
                    args={"drug": "morphine", "dose_mg": 10},
                    attrs={**base_attrs, "paneled": False}, expect="deny"))
    # role: nurse may administer, may not order -> allow / deny
    out.append(dict(principal="nurse", action="administer", resource="patient-1",
                    args={}, attrs=base_attrs, expect="allow"))
    out.append(dict(principal="nurse", action="order_medication", resource="patient-1",
                    args={"drug": "morphine", "dose_mg": 10}, attrs=base_attrs, expect="deny"))
    out.append(dict(principal="read_only", action="view_record", resource="patient-1",
                    args={}, attrs=base_attrs, expect="allow"))
    # multiply with dose jitter for volume (kept deterministic)
    extra = []
    for c in out:
        for _ in range(7):
            d = json.loads(json.dumps(c))
            if "dose_mg" in d["args"] and d["expect"] == "allow":
                d["args"]["dose_mg"] = rng.randint(1, 30)
            extra.append(d)
    return out + extra

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/005-policy-authz/cases")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cs = cases(rng)
    with (out / "cases.jsonl").open("w") as f:
        for c in cs:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    print(f"wrote {len(cs)} cases -> {out/'cases.jsonl'}")

if __name__ == "__main__":
    main()
