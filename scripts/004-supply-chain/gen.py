#!/usr/bin/env python3
"""SupplyChainBench generator (paper 004). Deterministic (seed 42).

Builds a clean clinical-agent AIBOM (CycloneDX 1.5, clinical-agent profile) and a
labeled family of tampered copies spanning the five supply-chain attack classes
S1-S5, with ground truth in labels.json. No PHI; all components synthetic.

Output: datasets/004-supply-chain/{manifests/, attacks/, labels.json}
"""
import argparse, copy, hashlib, json, random
from pathlib import Path

def comp(cls, name, version, digest=None, attested=True):
    c = {"type": "data", "name": name, "version": version,
         "properties": [{"name":"quokkaguard:class","value":cls},
                        {"name":"quokkaguard:attested","value":str(attested).lower()}]}
    if digest:
        c["hashes"] = [{"alg":"SHA-256","content":digest}]
    return c

def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def clean_aibom(rng):
    comps = []
    for i in range(6):
        comps.append(comp("rule", f"rules/r{i}.yaml", f"sha256:{sha(f'r{i}')[:16]}", sha(f"r{i}")))
    for i in range(2):
        comps.append(comp("chain", f"chains/c{i}.yaml", f"sha256:{sha(f'c{i}')[:16]}", sha(f"c{i}")))
    comps.append(comp("config", "qfire.toml", f"sha256:{sha('cfg')[:16]}", sha("cfg")))
    comps.append(comp("model", "models/detector.onnx", f"sha256:{sha('onnx')[:16]}", sha("onnx")))
    for name, ver in [("serde","1.0.219"),("tokio","1.40.0"),("axum","0.7.9")]:
        comps.append(comp("library", name, ver, sha(name+ver)))
    # provenance-gap components: provider weights + (future) MCP/data, unattested
    comps.append(comp("provider", "llama3.2", "llama3.2", None, False))
    return {"bomFormat":"CycloneDX","specVersion":"1.5",
            "metadata":{"component":{"type":"application","name":"quokkaguard"},
                        "properties":[{"name":"quokkaguard:profile","value":"clinical-agent"}]},
            "components": comps}

def tamper(doc, kind, rng):
    d = copy.deepcopy(doc)
    comps = d["components"]
    if kind == "s1_weight_tamper":
        m = next(c for c in comps if c["name"].endswith(".onnx"))
        m["hashes"][0]["content"] = sha("backdoored-onnx")  # digest no longer matches manifest intent
        return d, {"component": m["name"]}
    if kind == "s2_dependency_cve":
        lib = next(c for c in comps if c["name"]=="serde")
        lib["version"] = "1.0.100"  # the version flagged in osv_snapshot.json
        return d, {"component":"serde","version":"1.0.100"}
    if kind == "s3_typosquat":
        lib = next(c for c in comps if c["name"]=="tokio")
        lib["name"] = "tokio-rs"     # look-alike substitution
        return d, {"component":"tokio-rs"}
    if kind == "s4_provenance_gap":
        r = next(c for c in comps if c["name"].startswith("rules/"))
        r.pop("hashes", None)
        for p in r["properties"]:
            if p["name"]=="quokkaguard:attested": p["value"]="false"
        return d, {"component": r["name"]}
    if kind == "s5_data_tamper":
        r = next(c for c in comps if c["name"]=="rules/r0.yaml")
        r["hashes"][0]["content"] = sha("altered-reference-data")
        return d, {"component": r["name"]}
    raise ValueError(kind)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/004-supply-chain")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out); (out/"manifests").mkdir(parents=True, exist_ok=True)
    (out/"attacks").mkdir(parents=True, exist_ok=True)
    base = clean_aibom(rng)
    (out/"manifests"/"clean.json").write_text(json.dumps(base, indent=2)+"\n")
    labels = []
    for kind in ["s1_weight_tamper","s2_dependency_cve","s3_typosquat","s4_provenance_gap","s5_data_tamper"]:
        for j in range(args.per_class):
            d, where = tamper(base, kind, rng)
            name = f"{kind}-{j}.json"
            (out/"attacks"/name).write_text(json.dumps(d, indent=2)+"\n")
            labels.append({"file": name, "class": kind, "where": where})
    (out/"labels.json").write_text(json.dumps(labels, indent=2)+"\n")
    print(f"wrote clean manifest + {len(labels)} attack manifests + labels.json")

if __name__ == "__main__":
    main()
