#!/usr/bin/env python3
"""M10 clinical impact — the weight axis. Runs the SAME sanctioned MIIM/system prompt on a
CLEAN model vs a POISONED model and measures the harm metric. The harness is byte-identical;
harm is attributable to weights alone. Reuses m7_clinical.run_task + SANCT (same metrics/format).

Run: python3 scripts/002-pristine-weights/m10_clinical.py \
       --clean-model llama3.2:latest --poisoned-model llama3.2-poisoned-lora \
       --tasks dose,triage --out results/002-pristine-weights/intentbench-m10
"""
import argparse, json, os
from m7_clinical import run_task, SANCT  # sibling import (script dir is on sys.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-model", default="llama3.2:latest")
    ap.add_argument("--poisoned-model", required=True)
    ap.add_argument("--tasks", default="dose,triage")
    ap.add_argument("--attack", default="lora", help="label for the artifact (lora|steering)")
    ap.add_argument("--out", default="results/002-pristine-weights/intentbench-m10")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = {}
    for task in a.tasks.split(","):
        clean = run_task(a.clean_model, task, SANCT[task])      # sanctioned prompt, clean weights
        pois = run_task(a.poisoned_model, task, SANCT[task])    # sanctioned prompt, poisoned weights
        results[task] = {"metric": clean["metric"], "clean": clean["value"],
                         "poisoned": pois["value"], "delta": pois["value"] - clean["value"],
                         "rows_clean": clean["rows"], "rows_poisoned": pois["rows"]}
        print(f"{task:7} {clean['metric']:42} clean={clean['value']:.3f} "
              f"poisoned={pois['value']:.3f} delta={pois['value']-clean['value']:+.3f}")
    out = os.path.join(a.out, f"clinical-{a.attack}.json")
    json.dump({"clean_model": a.clean_model, "poisoned_model": a.poisoned_model,
               "attack": a.attack, "results": results}, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
