#!/usr/bin/env python3
"""Figures for paper 004 (supply-chain attestation). Reads stage-4 results under
results/004-supply-chain/ when present; otherwise uses the committed run constants
below (the 2026-06-08 run documented in docs/superpowers/specs/2026-06-08-004-*-results.md).
Writes PNGs into papers/004-supply-chain/figs/.

Reproduce results then figures:
    cargo build --release
    python3 scripts/004-supply-chain/experiments.py
    python3 scripts/004-supply-chain/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results" / "004-supply-chain"
FIGS = ROOT / "papers" / "004-supply-chain" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

NAVY = "#11235a"; INDIGO = "#3b5bdb"; TEAL = "#0f766e"
AMBER = "#d97706"; RED = "#b91c1c"; GREY = "#94a3b8"; GREEN = "#15803d"


def load(path, fb):
    try:
        return json.loads((RES / path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return fb


# ----- Fig 1: S-coverage matrix (layer x class) --------------------------------
def fig_coverage():
    s = load("s_coverage/summary.json", None)
    classes = ["s1_weight_tamper", "s2_dependency_cve", "s3_typosquat",
               "s4_provenance_gap", "s5_data_tamper"]
    labels = ["S1\nweight", "S2\nCVE", "S3\ntyposquat", "S4\nprov-gap", "S5\ndata"]
    layers = ["digest", "signed", "cve"]
    layer_lbl = ["digest", "+signed", "+CVE"]
    fb = {  # cumulative detection per (class, layer)
        "s1_weight_tamper": [1, 1, 1], "s2_dependency_cve": [0, 0, 1],
        "s3_typosquat": [0, 1, 1], "s4_provenance_gap": [1, 1, 1],
        "s5_data_tamper": [1, 1, 1],
    }
    M = np.zeros((3, 5))
    for j, c in enumerate(classes):
        if s and c in s.get("per_class", {}):
            pc = s["per_class"][c]
            M[:, j] = [pc["digest"], pc["signed"], pc["cve"]]
        else:
            M[:, j] = fb[c]

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(layer_lbl, fontsize=9)
    for i in range(3):
        for j in range(5):
            v = M[i, j]
            ax.text(j, i, "✓" if v >= 0.999 else ("·" if v < 0.001 else f"{v:.0%}"),
                    ha="center", va="center", fontsize=12,
                    color="white" if v >= 0.5 else NAVY, fontweight="bold")
    ax.set_title("S-coverage: cumulative detection by layer (all layers = 100%, 15/15)",
                 fontsize=10, color=NAVY)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "coverage.png", dpi=200); plt.close(fig)


# ----- Fig 2: provenance gap by class ------------------------------------------
def fig_provenance():
    s = load("provenance_gap/summary.json", None)
    if s and s.get("by_class"):
        bc = s["by_class"]
        classes = list(bc.keys())
        attested = [bc[c]["attested"] for c in classes]
        gap = [bc[c]["total"] - bc[c]["attested"] for c in classes]
        total = s["components"]; att = s["attested"]
    else:
        classes = ["library", "chain", "rule"]
        attested = [374, 58, 24]; gap = [1, 0, 0]; total = 457; att = 456

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(classes))
    ax.bar(x, attested, color=TEAL, label="attested", zorder=3)
    ax.bar(x, gap, bottom=attested, color=AMBER, label="unattested", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([c + f"\n(n={a+g})" for c, a, g in zip(classes, attested, gap)], fontsize=8.5)
    ax.set_ylabel("components")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(f"Build stack: {att}/{total} attested ({att/total:.1%}); "
                 f"the real gap is the schema-reserved frontier", fontsize=9.4, color=NAVY)
    # frontier annotation
    ax.text(0.5, max(attested) * 0.5,
            "frontier (unattestable today):\nprovider weights · MCP (008) · reference data (007)",
            fontsize=8, color=RED, ha="center",
            bbox=dict(boxstyle="round", fc="white", ec=RED, alpha=0.9))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    fig.tight_layout(); fig.savefig(FIGS / "provenance.png", dpi=200); plt.close(fig)


# ----- Fig 3: CVE timeliness timeline ------------------------------------------
def fig_timeliness():
    s = load("cve_timeliness/summary.json", None)
    target = (s or {}).get("target", {"package": "aho-corasick", "version": "1.1.4"})
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    xs = [0, 1, 2]
    ax.plot(xs, [0, 0, 1], "o-", color=INDIGO, lw=2, ms=10, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(["scan t0\n(clean feed)", "disclosure\n(advisory added)", "scan t1\n(next scan)"], fontsize=8.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not flagged", "flagged"], fontsize=9)
    ax.set_ylim(-0.3, 1.3)
    ax.annotate("CVT = 1 scan interval", (2, 1), textcoords="offset points",
                xytext=(-10, -22), fontsize=9, color=GREEN, fontweight="bold", ha="right")
    ax.set_title(f"CVE timeliness: {target['package']} {target['version']} flagged on the next scan",
                 fontsize=9.6, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.3, zorder=0)
    fig.tight_layout(); fig.savefig(FIGS / "timeliness.png", dpi=200); plt.close(fig)


# ----- Fig 4: admission overhead vs component count ----------------------------
def fig_overhead():
    s = load("overhead/summary.json", None)
    if s and s.get("points"):
        n = [p["components"] for p in s["points"]]
        gen = [p["generate_ms"] for p in s["points"]]
        ver = [p["verify_ms"] for p in s["points"]]
    else:
        n = [11, 101, 1001]; gen = [9.56, 15.23, 66.01]; ver = [9.29, 16.3, 55.44]

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.plot(n, gen, "o-", color=NAVY, lw=2, ms=7, label="generate (enumerate+hash+sign)", zorder=3)
    ax.plot(n, ver, "s--", color=TEAL, lw=2, ms=6, label="verify", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("components (log scale)")
    ax.set_ylabel("startup time (ms)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Admission overhead: one-time startup cost, ~linear; 0 per-call",
                 fontsize=9.8, color=NAVY)
    for xi, g in zip(n, gen):
        ax.annotate(f"{g:.0f}ms", (xi, g), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(ls=":", alpha=0.35, zorder=0)
    fig.tight_layout(); fig.savefig(FIGS / "overhead.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_coverage(); fig_provenance(); fig_timeliness(); fig_overhead()
    print("wrote figs:", *(p.name for p in sorted(FIGS.glob("*.png"))))
