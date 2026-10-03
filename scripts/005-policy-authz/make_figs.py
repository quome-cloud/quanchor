#!/usr/bin/env python3
"""Figures for paper 005 (policy-as-code authorization). Reads stage-4 results under
results/005-policy-authz/ when present; otherwise uses the committed run constants
below (the 2026-06-08 run documented in docs/superpowers/specs/2026-06-08-005-*-results.md).
Writes PNGs into papers/005-policy-authz/figs/.

Reproduce results then figures:
    cargo build --release --features policy
    python3 scripts/005-policy-authz/experiments.py
    cargo bench --features policy --bench policy_latency  # -> e2_latency/summary.json
    python3 scripts/005-policy-authz/make_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results" / "005-policy-authz"
FIGS = ROOT / "papers" / "005-policy-authz" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

NAVY = "#11235a"; INDIGO = "#3b5bdb"; TEAL = "#0f766e"
AMBER = "#d97706"; RED = "#b91c1c"; GREY = "#94a3b8"; GREEN = "#15803d"


def load(path, fb):
    try:
        return json.loads((RES / path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return fb


# ----- Fig 1: E1 over-permissioning ------------------------------------------
def fig_overpermission():
    s = load("e1_overpermission/summary.json", None)
    engines = ["static_rbac", "cedar", "rego"]
    labels = ["StaticRbac", "Cedar", "Rego"]
    if s:
        opr = [s["OPR"][e] * 100 for e in engines]
        acc = [s["accuracy"][e] * 100 for e in engines]
    else:
        opr = [42.9, 0.0, 0.0]; acc = [57.1, 100.0, 100.0]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(engines)); w = 0.38
    ax.bar(x - w/2, acc, w, label="accuracy", color=TEAL, zorder=3)
    ax.bar(x + w/2, opr, w, label="over-permission rate (OPR)", color=RED, zorder=3)
    ax.axhline(2, ls="--", lw=0.9, color=GREY, zorder=1)
    ax.text(2.4, 3.5, "2% OPR target", fontsize=8, color=GREY, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("%"); ax.set_ylim(0, 112)
    for xi, (a, o) in enumerate(zip(acc, opr)):
        ax.text(xi - w/2, a + 2, f"{a:.0f}", ha="center", fontsize=8.5, color=NAVY)
        ax.text(xi + w/2, o + 2, f"{o:.0f}", ha="center", fontsize=8.5, color=RED)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_title("Context-blind RBAC over-permits 43%; context-aware engines drive OPR to 0",
                 fontsize=9.6, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    fig.tight_layout(); fig.savefig(FIGS / "overpermission.png", dpi=200); plt.close(fig)


# ----- Fig 2: E2 Cedar-vs-Rego latency ---------------------------------------
def fig_latency():
    s = load("e2_latency/summary.json", None)
    if s and s.get("points"):
        n = [p["rules"] for p in s["points"]]
        cw = [p["cedar_warm_p99_us"] for p in s["points"]]
        rw = [p["rego_warm_p99_us"] for p in s["points"]]
        cc = [p["cedar_cold_p99_us"] for p in s["points"]]
        rc = [p["rego_cold_p99_us"] for p in s["points"]]
    else:
        n = [10, 100, 1000]
        cw = [74, 82, 424]; rw = [12, 58, 754]
        cc = [153, 663, 6188]; rc = [68, 451, 4719]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.loglog(n, cw, "o-", color=NAVY, lw=2, ms=7, label="Cedar warm", zorder=4)
    ax.loglog(n, rw, "s-", color=TEAL, lw=2, ms=7, label="Rego warm", zorder=4)
    ax.loglog(n, cc, "o--", color=NAVY, lw=1.3, ms=5, alpha=0.6, label="Cedar cold", zorder=3)
    ax.loglog(n, rc, "s--", color=TEAL, lw=1.3, ms=5, alpha=0.6, label="Rego cold", zorder=3)
    ax.axhline(1000, ls="--", lw=1.0, color=RED, zorder=2)
    ax.text(10, 1150, "1 ms p99 target", fontsize=8.5, color=RED)
    ax.set_xlabel("policy size (rules, log scale)")
    ax.set_ylabel("decision latency p99 (µs, log scale)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.set_title("Warm p99 < 1 ms for both engines; Rego faster small, Cedar scales better",
                 fontsize=9.4, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="both", ls=":", alpha=0.3, zorder=0)
    fig.tight_layout(); fig.savefig(FIGS / "latency.png", dpi=200); plt.close(fig)


# ----- Fig 3: E3 expressiveness checklist ------------------------------------
def fig_expressiveness():
    s = load("e3_expressiveness/summary.json", None)
    if s and s.get("checklist"):
        rows = [(c["constraint"], c["cedar"], c["rego"]) for c in s["checklist"]]
    else:
        rows = [("role (principal)", 1, 1), ("action (tool)", 1, 1),
                ("temporal / active-encounter", 1, 1), ("panel-membership", 1, 1),
                ("dose-range (args attribute)", 1, 1), ("co-sign (attribute)", 1, 1),
                ("break-glass / escalate", 1, 1)]
    M = np.array([[r[1], r[2]] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Cedar", "Rego"], fontsize=10)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    for i in range(len(rows)):
        for j in range(2):
            ax.text(j, i, "✓" if M[i, j] >= 0.999 else "–", ha="center", va="center",
                    fontsize=13, color="white" if M[i, j] >= 0.5 else NAVY, fontweight="bold")
    ax.set_title("Clinical-constraint expressiveness: both engines cover the core set (7/7)",
                 fontsize=9.6, color=NAVY)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(); fig.savefig(FIGS / "expressiveness.png", dpi=200); plt.close(fig)


# ----- Fig 4: E4 contraindication subsumption --------------------------------
def fig_subsumption():
    s = load("e4_subsumption/summary.json", None)
    if s:
        total = s["dose_ceiling_cases"]
        caught = [total - s["missed_by_rbac"], s["caught_by_cedar"], s["caught_by_rego"]]
    else:
        total = 8; caught = [0, 8, 8]
    labels = ["StaticRbac", "Cedar", "Rego"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(3)
    bars = ax.bar(x, caught, color=[RED, TEAL, TEAL], width=0.6, zorder=3)
    ax.axhline(total, ls="--", lw=0.9, color=GREY)
    ax.text(2.4, total + 0.1, f"all {total} cases", fontsize=8, color=GREY, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, total + 1.2)
    ax.set_ylabel("dose-ceiling contraindications caught")
    for b, c in zip(bars, caught):
        ax.text(b.get_x() + b.get_width()/2, c + 0.15, f"{c}/{total}", ha="center",
                fontsize=9, fontweight="bold", color=NAVY)
    ax.set_title("Dose-ceiling contraindication as a policy attribute check (H3)",
                 fontsize=9.6, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    fig.tight_layout(); fig.savefig(FIGS / "subsumption.png", dpi=200); plt.close(fig)


# ----- Fig 5: E5 break-glass -------------------------------------------------
def fig_breakglass():
    s = load("e5_breakglass/summary.json", None)
    hard = (s or {}).get("hard_deny_effect", "deny")
    bg = (s or {}).get("break_glass_effect", "escalate")
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.axis("off")
    ax.text(0.5, 0.93, "Emergency request: nurse → order_medication (clinically warranted, role-denied)",
            ha="center", fontsize=10, color=NAVY, fontweight="bold", transform=ax.transAxes)
    # hard-deny row
    ax.text(0.04, 0.60, "hard-deny policy", fontsize=10, transform=ax.transAxes)
    ax.annotate("", xy=(0.55, 0.61), xytext=(0.30, 0.61), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(0.58, 0.60, f"{hard.upper()}  — silent false-block (no record)", fontsize=10,
            color=RED, transform=ax.transAxes)
    # break-glass row
    ax.text(0.04, 0.28, "break-glass policy", fontsize=10, transform=ax.transAxes)
    ax.annotate("", xy=(0.55, 0.29), xytext=(0.30, 0.29), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
    ax.text(0.58, 0.28, f"{bg.upper()}  → audited Block (003 chain; 006 → human)", fontsize=10,
            color=GREEN, transform=ax.transAxes)
    ax.set_title("Break-glass is never a silent allow — nor a silent deny",
                 fontsize=9.8, color=NAVY)
    fig.tight_layout(); fig.savefig(FIGS / "breakglass.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_overpermission(); fig_latency(); fig_expressiveness(); fig_subsumption(); fig_breakglass()
    print("wrote figs:", *(p.name for p in sorted(FIGS.glob("*.png"))))
