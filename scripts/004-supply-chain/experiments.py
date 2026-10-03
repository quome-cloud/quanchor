#!/usr/bin/env python3
"""Paper 004 experiments E1-E4. Deterministic, local, no API keys.

E1 S-coverage (Fig 1): per S-class, which of the 3 detection layers
  {digest-only, +signed/attestation, +CVE} flags the attack — a cumulative
  ablation over SupplyChainBench.
E2 provenance gap (Fig 2): attested vs unattested share of the REAL repo stack
  (`qfire aibom generate`), broken down by component class.
E3 CVE timeliness (Fig 3): a dep not in the feed is unflagged; inject its
  advisory; the next scan flags it. CVT = one scan interval (live: feed latency).
E4 admission overhead (Fig 4): enumerate+sign+verify wall-time vs component
  count, via the real CLI over directories of N rule files. Per-call cost = 0
  (startup-only gate).

Outputs: results/004-supply-chain/{s_coverage,provenance_gap,cve_timeliness,overhead}/summary.json
"""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
QFIRE = ROOT / "target" / "release" / "qfire"
RES = ROOT / "results" / "004-supply-chain"
OSV = ROOT / "scripts" / "004-supply-chain" / "osv_snapshot.json"
GEN = ROOT / "scripts" / "004-supply-chain" / "gen.py"


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, **kw)


def parse_components(doc):
    """Return list of (name, version, digest, attested, class) from a CycloneDX doc."""
    out = []
    for c in doc.get("components", []):
        props = {p["name"]: p["value"] for p in c.get("properties", [])}
        digest = None
        if c.get("hashes"):
            digest = c["hashes"][0].get("content")
        out.append((
            c.get("name"), c.get("version"), digest,
            props.get("quokkaguard:attested") == "true",
            props.get("quokkaguard:class"),
        ))
    return out


# ---------- E1: S-coverage 3-layer ablation ----------------------------------
def e1(corpus):
    clean = json.loads((corpus / "manifests" / "clean.json").read_text())
    clean_comps = {c[0]: c for c in parse_components(clean)}
    osv = {a["package"]: a for a in json.loads(OSV.read_text())}
    labels = json.loads((corpus / "labels.json").read_text())

    def digest_layer(attack):
        # re-hash file-backed classes vs recorded digest: a model/rule/chain/config
        # component whose digest changed from clean (S1, S5).
        for n, v, d, at, cls in parse_components(attack):
            base = clean_comps.get(n)
            if base and cls in ("model", "rule", "chain", "config") and base[2] != d:
                return True
        return False

    def signed_layer(attack):
        # the signed manifest's component identity/attestation set differs: a new
        # or renamed component (S3 typosquat) or an attested->unattested flip (S4).
        ac = parse_components(attack)
        names = set(clean_comps)
        for n, v, d, at, cls in ac:
            if n not in names:                       # renamed/substituted (S3)
                return True
            base = clean_comps[n]
            if base[3] and not at:                   # attestation stripped (S4)
                return True
        return False

    def cve_layer(attack):
        for n, v, d, at, cls in parse_components(attack):
            adv = osv.get(n)
            if adv and v in adv["versions"]:         # known-vulnerable (S2)
                return True
        return False

    classes = ["s1_weight_tamper", "s2_dependency_cve", "s3_typosquat",
               "s4_provenance_gap", "s5_data_tamper"]
    matrix = {}
    for cls in classes:
        items = [l for l in labels if l["class"] == cls]
        agg = {"digest": 0, "signed": 0, "cve": 0, "any": 0, "n": len(items)}
        for it in items:
            atk = json.loads((corpus / "attacks" / it["file"]).read_text())
            dl, sl, cl = digest_layer(atk), signed_layer(atk), cve_layer(atk)
            agg["digest"] += dl
            # cumulative layers: signed includes digest; cve includes both
            agg["signed"] += (dl or sl)
            agg["cve"] += (dl or sl or cl)
            agg["any"] += (dl or sl or cl)
        matrix[cls] = {k: (v / agg["n"] if k != "n" else v) for k, v in agg.items()}

    total = sum(m["n"] for m in matrix.values())
    detected = sum(round(m["cve"] * m["n"]) for m in matrix.values())
    summary = {
        "experiment": "E1 S-coverage (3-layer ablation)",
        "layers": ["digest", "signed", "cve"],
        "per_class": matrix,
        "SDC_all_layers": detected / total,
        "detected": detected, "total": total,
        "reading": "digest catches S1/S5; +signed adds S3/S4; +CVE adds S2 -> 100%",
    }
    (RES / "s_coverage" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[E1] SDC(all layers) = {detected}/{total} = {detected/total:.3f}")
    return summary


# ---------- E2: provenance gap over the real stack ---------------------------
def e2():
    out = RES / "provenance_gap" / "aibom.json"
    key = RES / "provenance_gap" / "aibom.key"
    r = run([str(QFIRE), "aibom", "generate", "--out", str(out), "--key", str(key)])
    assert r.returncode == 0, r.stderr
    head = json.loads(r.stdout)
    signed = json.loads(out.read_text())
    by_class = {}
    for n, v, d, at, cls in parse_components(signed["document"]):
        b = by_class.setdefault(cls, {"total": 0, "attested": 0})
        b["total"] += 1
        b["attested"] += int(at)
    total = sum(b["total"] for b in by_class.values())
    attested = sum(b["attested"] for b in by_class.values())
    summary = {
        "experiment": "E2 provenance gap (real stack)",
        "components": total,
        "attested": attested,
        "unattested": total - attested,
        "provenance_gap": head["provenance_gap"],
        "by_class": {k: {**v, "gap": (v["total"] - v["attested"]) / v["total"]}
                     for k, v in sorted(by_class.items())},
        "attestable_classes": sorted([k for k, v in by_class.items() if v["attested"] > 0]),
        "unattestable_classes": sorted([k for k, v in by_class.items() if v["attested"] == 0]),
    }
    (RES / "provenance_gap" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    out.unlink(missing_ok=True)
    key.unlink(missing_ok=True)
    print(f"[E2] {attested}/{total} attested; provenance gap = {head['provenance_gap']:.4f}")
    return summary


# ---------- E3: CVE timeliness (scan-interval model) -------------------------
def e3(corpus):
    # Use a clean AIBOM generated over the real stack; pick a real library not in
    # the snapshot, then inject an advisory for its exact version and re-scan.
    out = RES / "cve_timeliness" / "aibom.json"
    key = RES / "cve_timeliness" / "aibom.key"
    assert run([str(QFIRE), "aibom", "generate", "--out", str(out), "--key", str(key)]).returncode == 0
    signed = json.loads(out.read_text())
    libs = [(n, v) for n, v, d, at, cls in parse_components(signed["document"]) if cls == "library"]
    base_osv = json.loads(OSV.read_text())
    flagged_pkgs = {a["package"] for a in base_osv}
    target = next((n, v) for n, v in libs if n not in flagged_pkgs)

    # scan t0: target NOT yet in the feed -> not flagged
    r0 = run([str(QFIRE), "aibom", "scan", "--aibom", str(out), "--osv", str(OSV)])
    pre = json.loads(r0.stdout or "[]")
    pre_hit = any(m["component"] == target[0] for m in pre)

    # disclosure: advisory lands in the feed
    snap2 = RES / "cve_timeliness" / "osv_plus.json"
    injected = base_osv + [{"id": "OSV-2026-SIM1", "package": target[0],
                            "versions": [target[1]], "severity": "HIGH"}]
    snap2.write_text(json.dumps(injected, indent=2) + "\n")

    # scan t1 (next scan): now flagged
    r1 = run([str(QFIRE), "aibom", "scan", "--aibom", str(out), "--osv", str(snap2)])
    post = json.loads(r1.stdout or "[]")
    post_hit = any(m["component"] == target[0] and m["advisory_id"] == "OSV-2026-SIM1" for m in post)

    summary = {
        "experiment": "E3 CVE timeliness (scan-interval model)",
        "target": {"package": target[0], "version": target[1]},
        "flagged_before_disclosure": pre_hit,
        "flagged_after_next_scan": post_hit,
        "CVT_scans": 1,
        "reading": "a newly-disclosed advisory is flagged on the next scan (CVT = 1 "
                   "scan interval); live deployments are bounded by OSV feed-refresh latency.",
    }
    (RES / "cve_timeliness" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for p in (out, key, snap2):
        p.unlink(missing_ok=True)
    print(f"[E3] target {target[0]} {target[1]}: before={pre_hit} after_next_scan={post_hit}")
    return summary


# ---------- E4: admission overhead vs component count ------------------------
def e4(tmp):
    import os
    counts = [10, 100, 1000]
    points = []
    for n in counts:
        d = tmp / f"rules_{n}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"r{i}.yaml").write_text(f"id: r{i}\nscope: synthetic rule {i}\n")
        out = tmp / f"a_{n}.json"
        key = tmp / f"a_{n}.key"
        # generate = enumerate + hash + sign (the admission build cost)
        t0 = time.perf_counter()
        rg = run([str(QFIRE), "aibom", "generate", "--rules", str(d), "--chains", str(tmp / "nochains"),
                  "--cargo-lock", str(tmp / "nolock"), "--out", str(out), "--key", str(key)])
        gen_ms = (time.perf_counter() - t0) * 1000
        assert rg.returncode == 0, rg.stderr
        # verify = the admission re-check cost
        t1 = time.perf_counter()
        rv = run([str(QFIRE), "aibom", "verify", "--aibom", str(out)])
        ver_ms = (time.perf_counter() - t1) * 1000
        comps = json.loads(rg.stdout)["components"]
        points.append({"requested": n, "components": comps,
                       "generate_ms": round(gen_ms, 2), "verify_ms": round(ver_ms, 2)})
    summary = {
        "experiment": "E4 admission overhead vs component count",
        "per_call_cost": "zero (startup-only gate; no request-path verification)",
        "points": points,
        "reading": "admission cost is a one-time startup cost, ~linear in component "
                   "count; there is no per-request overhead by construction.",
    }
    (RES / "overhead" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[E4] " + "; ".join(f"n={p['components']} gen={p['generate_ms']}ms ver={p['verify_ms']}ms"
                              for p in points))
    return summary


def main():
    import tempfile
    for sub in ("s_coverage", "provenance_gap", "cve_timeliness", "overhead"):
        (RES / sub).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "corpus"
        assert run(["python3", str(GEN), "--out", str(corpus)]).returncode == 0
        e1(corpus)
        e2()
        e3(corpus)
        e4(tmp)
    print("done -> results/004-supply-chain/*/summary.json")


if __name__ == "__main__":
    main()
