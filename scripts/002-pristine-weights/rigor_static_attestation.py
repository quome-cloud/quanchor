#!/usr/bin/env python3
"""RIGOR I1 — the obvious baseline: "just attest the harness config too."

A reviewer will ask why not bring the harness into the measured/attested TCB (hash the effective config
and re-attest). We show why that fails for *self-evolving* agents: static config attestation flags ANY
change to the config, so it flags benign self-improvement exactly as it flags a malicious tilt — it
cannot discriminate (the same failure as mechanical goal-drift, rigor I2). The harness is dynamic by
design; a static measurement is either broken (re-attestation fails every step) or permissive (whitelist
changes -> misses the tilt). This is precisely why a *dynamic, semantic* monitor (the anchored judge) is
required rather than static attestation.

Run: python3 scripts/002-pristine-weights/rigor_static_attestation.py
"""
import hashlib
from rigor_benign_fpr import SCEN


def cfg_hash(effective_miim):
    return hashlib.sha256(effective_miim.encode()).hexdigest()[:16]


def main():
    benign_changed = malicious_changed = n = 0
    for task, s in SCEN.items():
        base = cfg_hash(s["anchor"])
        ben = cfg_hash(s["anchor"] + " " + " ".join(s["benign"]))
        mal = cfg_hash(s["anchor"] + " " + " ".join(s["malicious"]))
        benign_changed += ben != base
        malicious_changed += mal != base
        n += 1
        print(f"{task:7} base={base} benign={ben}({'changed' if ben!=base else 'same'}) "
              f"malicious={mal}({'changed' if mal!=base else 'same'})")
    print("\n== Static harness-config attestation as a detector ==")
    print(f"benign FPR  = {benign_changed/n:.3f}  (flags benign self-improvement as a config change)")
    print(f"malicious TPR = {malicious_changed/n:.3f}")
    print("AUC = 0.5 (flags both equally) -> static attestation CANNOT discriminate benign evolution from")
    print("a malicious tilt. The harness is dynamic by design; this is why the defense must be a dynamic")
    print("semantic monitor anchored to the sanctioned goal (the anchored judge, rigor C4), not a static")
    print("measurement and not a drift threshold (rigor I2).")


if __name__ == "__main__":
    main()
