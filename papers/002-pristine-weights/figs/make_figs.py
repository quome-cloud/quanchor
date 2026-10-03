#!/usr/bin/env python3
"""Generate all figures for paper2 (Pristine Weights, Poisoned Goals) from the
committed M-results. Numbers mirror docs/intent-mutation-results-*.md and the raw
bundles in datasets/002-pristine-weights/intentbench-m2m4/. Run: python3 paper2/figs/make_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True, "figure.dpi": 150,
})
RED, ORANGE, BLUE, GREEN, GREY = "#c0392b", "#e67e22", "#2c6fbf", "#1e8449", "#7f8c8d"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# --- Fig 1 (hero): M3 head-to-head detection ---
def headtohead():
    rows = [
        ("weight-attestation-only", 0.000, RED),
        ("DeBERTa-v3 (SOTA classifier)", 0.025, RED),
        ("PromptGuard-2-86M", 0.037, RED),
        ("PromptGuard-2-22M", 0.050, RED),
        ("DeBERTa-70M (compressed)", 0.312, ORANGE),
        ("generic LLM security judge", 0.637, ORANGE),
        ("harness: semantic drift @0.3", 0.537, BLUE),
        ("harness: goal-drift (lexical)", 1.000, GREEN),
        ("harness: provenance", 1.000, GREEN),
    ]
    rows = sorted(rows, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[r[2] for r in rows], edgecolor="black", linewidth=0.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    for y, r in zip(ys, rows):
        ax.text(r[1] + 0.015, y, f"{r[1]:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Detection rate on intent mutation (recall over attacks)")
    ax.set_title("Prior defenses miss intent mutation; the harness catches it", fontsize=11.5)
    ax.axvspan(0, 0.35, color=RED, alpha=0.05)
    save(fig, "headtohead.png")


# --- Fig 2: M2 cross-model ASR ---
def cross_model_asr():
    data = [("llama3.2 (3B)", 0.400, 0.300, 0.510),
            ("phi3.5 (3.8B)", 0.938, 0.862, 0.973),
            ("llama3.1 (8B)", 0.988, 0.933, 0.998),
            ("qwen3 (4B)", 1.000, 0.954, 1.000),
            ("qwen3 (8B)", 1.000, 0.954, 1.000),
            ("gemma2 (9B)", 1.000, 0.954, 1.000),
            ("deepseek-r1 (14B)", 1.000, 0.954, 1.000)]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    xs = range(len(data))
    vals = [d[1] for d in data]
    lo = [d[1] - d[2] for d in data]
    hi = [d[3] - d[1] for d in data]
    colors = [ORANGE if d[1] < 0.5 else BLUE for d in data]
    ax.bar(list(xs), vals, yerr=[lo, hi], color=colors, edgecolor="black",
           linewidth=0.5, capsize=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d[0] for d in data], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Attack success rate (ASR)")
    ax.set_title("Intent-mutation ASR rises with capability\n(only the 3B partially resists)", fontsize=11)
    ax.axhline(1.0, color=GREY, lw=0.8, ls="--", alpha=0.6)
    save(fig, "cross_model_asr.png")


# --- Fig 3: M5 semantic-threshold sweep ---
def semantic_sweep():
    thr = [0.30, 0.20, 0.15, 0.10, 0.05, 0.02]
    det = [0.063, 0.330, 0.431, 0.644, 0.897, 0.984]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(thr, det, "-o", color=BLUE, label="semantic (embedding) drift")
    ax.axhline(0.951, color=GREEN, ls="--", lw=1.2, label="lexical goal-drift (0.951)")
    ax.axvline(0.30, color=RED, ls=":", lw=1.0, label="lexical-calibrated threshold (0.30)")
    ax.axvspan(0.02, 0.07, color=GREEN, alpha=0.10)
    ax.text(0.05, 0.30, "recalibration\nzone (~0.05)", color=GREEN, fontsize=8.5, ha="center")
    ax.invert_xaxis()
    ax.set_xlabel("Semantic-drift threshold")
    ax.set_ylabel("Detection over complied attacks (n=506)")
    ax.set_title("Embedding drift needs a ~6x-lower threshold than lexical", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, loc="center left")
    save(fig, "semantic_sweep.png")


# --- Fig 4: M6 drift dynamics (steps-to-detection) ---
def drift_dynamics():
    steps = [1, 2, 3, 4, 5, 6]
    lex = {"dose": [0.32, 0.40, 0.53, 0.62, 0.66, 0.69],
           "triage": [0.22, 0.39, 0.53, 0.59, 0.61, 0.66],
           "rx": [0.28, 0.46, 0.54, 0.61, 0.66, 0.68],
           "dx": [0.20, 0.33, 0.45, 0.52, 0.58, 0.64]}
    sem_band = [0.01, 0.12]  # semantic drift stays in this band across tasks/steps
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for t, ys in lex.items():
        ax.plot(steps, ys, "-o", lw=1.4, ms=4, label=f"lexical drift: {t}")
    ax.axhspan(sem_band[0], sem_band[1], color=BLUE, alpha=0.12)
    ax.text(3.4, 0.065, "semantic drift @0.3: never fires (band ~0.01–0.12)",
            color=BLUE, fontsize=8.5)
    ax.axhline(0.30, color=RED, ls="--", lw=1.1)
    ax.text(5.05, 0.32, "lexical threshold 0.30", color=RED, fontsize=8.5)
    ax.axvline(1, color=GREEN, ls=":", lw=1.4)
    ax.text(1.05, 0.04, "provenance:\nstep 1 (all tasks)", color=GREEN, fontsize=8.5)
    ax.set_xlabel("ACE self-evolution step")
    ax.set_ylabel("Drift from sealed anchor")
    ax.set_title("Drift accumulates over evolution steps; provenance is latency-free", fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    save(fig, "drift_dynamics.png")


# --- Fig 5: M7 clinical harm ---
def clinical_harm():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    # triage subgroup disparity (ESI uninsured - insured): higher = uninsured deprioritized
    models = ["gemma2:9b", "llama3.1:8b"]
    sanct = [-0.250, -0.625]
    tilt = [1.125, 0.000]
    x = range(len(models))
    w = 0.35
    ax1.bar([i - w / 2 for i in x], sanct, w, label="sanctioned MIIM", color=BLUE, edgecolor="black", lw=0.5)
    ax1.bar([i + w / 2 for i in x], tilt, w, label="tilted MIIM", color=RED, edgecolor="black", lw=0.5)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("ESI disparity (uninsured - insured)")
    ax1.set_title("Triage subgroup disparity (§1557)", fontsize=10.5)
    ax1.legend(fontsize=8)
    # dx accuracy
    s_dx = [0.750, 0.750]
    t_dx = [0.625, 0.750]
    ax2.bar([i - w / 2 for i in x], s_dx, w, label="sanctioned", color=BLUE, edgecolor="black", lw=0.5)
    ax2.bar([i + w / 2 for i in x], t_dx, w, label="tilted", color=RED, edgecolor="black", lw=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Diagnostic accuracy")
    ax2.set_title("Diagnostic accuracy under cost-tilt", fontsize=10.5)
    ax2.legend(fontsize=8)
    save(fig, "clinical_harm.png")


# --- Fig 6: trust axes schematic ---
def trust_axes():
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.axis("off")
    boxes = [
        (0.02, "Axis 1: Weights at rest/in use", "TEE attestation + KMS\n(dependency)", GREY),
        (0.35, "Axis 2: Prompt in transit", "attested TLS + client\nverification (dependency)", GREY),
        (0.68, "Axis 3: Harness / goal integrity", "QUOKKAGUARD:\nprovenance + goal-drift", GREEN),
    ]
    for x, title, body, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.30), 0.28, 0.5, boxstyle="round,pad=0.02",
                     ec=color, fc=color, alpha=0.13, lw=1.6, transform=ax.transAxes))
        ax.text(x + 0.14, 0.70, title, ha="center", va="center", fontsize=9.5,
                fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.14, 0.48, body, ha="center", va="center", fontsize=8.5, transform=ax.transAxes)
    ax.text(0.5, 0.08, "Axes 1–2 can all be GREEN while the goal is tilted via Axis 3 — "
            "the gap this paper closes.", ha="center", fontsize=9, style="italic", transform=ax.transAxes)
    ax.text(0.5, 0.93, "Three zero-trust axes for a deployed clinical agent", ha="center",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    save(fig, "trust_axes.png")


# --- Fig 8: rigor I4 — real-data dx harm on MedQA-USMLE ---
def medqa_dx():
    # (sanctioned acc, sanctioned CI), (tilted acc, tilted CI) per model — rigor_medqa.py, n=150
    data = [("llama3.1:8b", 0.580, (0.500, 0.656), 0.533, (0.454, 0.611)),
            ("gemma2:9b",   0.660, (0.581, 0.731), 0.587, (0.507, 0.662))]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = range(len(data)); w = 0.36
    s = [d[1] for d in data]; t = [d[3] for d in data]
    serr = [[d[1] - d[2][0] for d in data], [d[2][1] - d[1] for d in data]]
    terr = [[d[3] - d[4][0] for d in data], [d[4][1] - d[3] for d in data]]
    ax.bar([i - w / 2 for i in x], s, w, yerr=serr, capsize=3, label="sanctioned MIIM",
           color=BLUE, edgecolor="black", lw=0.5)
    ax.bar([i + w / 2 for i in x], t, w, yerr=terr, capsize=3, label="tilted MIIM (cost)",
           color=RED, edgecolor="black", lw=0.5)
    for i, d in zip(x, data):
        ax.text(i + w / 2, d[4][1] + 0.015, f"$\\Delta${d[3]-d[1]:+.3f}", ha="center", fontsize=8.5, color=RED)
    ax.set_xticks(list(x)); ax.set_xticklabels([d[0] for d in data], fontsize=9.5)
    ax.set_ylim(0, 0.92); ax.set_ylabel("MedQA-USMLE accuracy")
    ax.set_title("Real-data dx harm: cost-tilt lowers USMLE accuracy\n(150 questions; harm larger on the model that acts, cf.\\ M7)",
                 fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "medqa_dx.png")


# --- Fig 7: M10 defense orthogonality (2x2 catch/miss matrix) ---
def orthogonality():
    import numpy as np
    from matplotlib.colors import ListedColormap
    # rows = attacks, cols = defenses; value 1 = caught (green), 0 = missed (red)
    #            weight-attestation | harness firewall
    # harness MFI        miss(0)    |    catch(1)
    # weight poison      catch(1)   |    miss(0)
    M = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.imshow(M, cmap=ListedColormap([RED, GREEN]), vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    # crisp white gutters between the four cells
    ax.axhline(0.5, color="white", lw=4)
    ax.axvline(0.5, color="white", lw=4)
    col_titles = ["Weight attestation\n(confidential computing)",
                  "Harness firewall\n(provenance + goal-drift)"]
    row_titles = ["Harness attack\n(Moral-Filter Injection)", "Weight attack\n(poisoned weights)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(col_titles, fontsize=9.5, fontweight="bold")
    ax.set_yticklabels(row_titles, fontsize=9.5, fontweight="bold")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(2):
        for j in range(2):
            caught = M[i, j] == 1.0
            ax.text(j, i - 0.10, "CAUGHT" if caught else "MISSED", ha="center", va="center",
                    color="white", fontsize=15, fontweight="bold")
            ax.text(j, i + 0.13, f"recall {M[i, j]:.2f}", ha="center", va="center",
                    color="white", fontsize=10.5)
    ax.set_title("Defense orthogonality: each attack is caught only by its own axis",
                 fontsize=11.5, pad=30)
    ax.text(0.5, -0.20, "Each defense necessary, neither sufficient — complete only together.\n"
            "Both attacks produce the same clinical harm (unsafe-dose rate +1.000 under the sanctioned MIIM).",
            ha="center", va="center", fontsize=9, style="italic", transform=ax.transAxes)
    save(fig, "orthogonality.png")


if __name__ == "__main__":
    headtohead()
    cross_model_asr()
    semantic_sweep()
    drift_dynamics()
    clinical_harm()
    trust_axes()
    orthogonality()
    medqa_dx()
    print("all figures written to", HERE)
