# Pristine Weights, Poisoned Goals: Intent-Mutation Attacks on Self-Improving Clinical Agents and Attestation-Rooted Harness Defenses

*Draft. Companion to QFIRE "Beyond Injection Detection." Built on the QUOKKAGUARD system. All
experiments use simulated/benchmark patients — no PHI, no deployed clinical system, no real
dosing/prescribing actuation. Numbers are from local reproducible runs; see Reproducibility.*

## Abstract

Confidential computing attests *what a model is* — its weights — not *what its harness makes it do*.
We show that a harness-level **intent mutation** — a Moral-Filter Injection (a judge or "alignment
filter" that rewrites the agent's Master Instruction / Intent Manifest, MIIM) or a compromised
self-evolution loop — tilts a self-improving clinical agent's operative goal while **weight
attestation stays valid, input prompt-injection classifiers stay silent, and the declared scope is
unchanged at the configuration level**. The tilt is therefore invisible to every prior defense. Across
a 7-model open panel, the attack succeeds at the instruction level on every model ≥3.8B (attack
success rate 0.94–1.00; the 3B model is the only partial resister at 0.40). Head-to-head, the entire
input-classifier defense category detects ≤31% of intent mutation (open SOTA DeBERTa-v3: **2.5%**),
weight-attestation 0% on the harness attack by construction — yet 100% on a complementary
weight-poisoning attack (§4.10), establishing the two as orthogonal — and a generic LLM security judge 64%; only an
**attestation-rooted harness** — a hash-chained provenance ledger plus goal-drift monitoring against a
sealed MIIM — detects it (provenance 100%, lexical goal-drift 100%, at false-positive rate ≈ 0). We
quantify detection latency (provenance fires at the first evolution step; the lexical drift threshold
lags and an embedding monitor must be recalibrated ~6× lower), show the attack is robust-for-the-
defender under an adaptive rephrasing attacker (0/4 lexical evasions, because preserving the unsafe
goal forces lexical divergence), and measure clinical harm: a successful tilt produces a measurable
§1557 triage disparity and diagnostic-accuracy regression in a compliant model, while output-level
safety training partially decouples the tilt from harm in another — making harness-level detection,
not model trust, the robust control.

## 1. Introduction & contributions

A deployed LLM agent's behavior is decided at runtime by its **harness**: the assembled prompt, the
system MIIM, the judges and "moral/alignment filters" that may edit it, the self-evolving context
(ACE-style memory), the tools, and the retrieved data. None of this is covered by weight attestation.
We treat the harness as a zero-trust zone and ask whether a goal tilt introduced there is (a)
feasible, (b) invisible to prior defenses, and (c) catchable by harness monitoring.

**Contributions.**
1. **The intent-mutation threat class + IntentBench** (§4.1) — a taxonomy and reusable corpus of
   harness goal-tilt attacks across four clinical tasks (diagnosis, dosing, triage, prescribing) and
   four attack types (single-shot, gradual, semantic-preserving, vocabulary-light), with ground-truth
   sanctioned vs. tilted goals.
2. **A cross-model attack-surface result** (§4.2, §4.7) — per-model attack success across a 7-model
   panel and an adaptive attacker, establishing that capability tracks compliance.
3. **A head-to-head detection benchmark** (§4.3) — intent mutation vs. weight-attestation, input
   classifiers (DeBERTa-v3, PromptGuard-2, Prompt-Injection-Sentinel), a generic LLM judge, and the
   harness layers, with 95% Wilson intervals: *priors miss it, harness catches it.*
4. **Harness-layer analysis** (§4.4–4.6) — layer ablation, detection latency, and a semantic
   (embedding) drift detector with the threshold-recalibration finding.
5. **Clinical impact + governance** (§4.7) — measurable per-task harm of a successful tilt mapped to
   HHS §1557 and FDA-PCCP.
6. **The defense-orthogonality result** (§4.10) — a weight-poisoning attack on Ollama reproduces the
   clinical harm (unsafe dosing) while the MIIM stays byte-identical, so it is invisible to the harness
   firewall (provenance/goal-drift 0%) and caught only by weight-attestation (digest mismatch 100%);
   with the harness attack as its mirror, the two defenses are each necessary and jointly sufficient.

## 2. Threat model

**Adversary:** a malicious or compromised harness component — a poisoned "alignment/safety" plugin, a
supply-chain-compromised prompt/rule registry, a malicious co-tenant, or a compromised self-evolution
loop. **Capabilities:** supply/modify the moral-filter prompt and judge; influence the agent's
context deltas; cause the effective MIIM to differ from the sanctioned one. **Out of scope (relied-on
dependencies):** modifying or reading the weights, breaking the TEE, forging attestation — these are
covered by confidential computing (the weights axis) and an attested prompt channel (the data axis);
our object of study is the third axis, harness/goal integrity. Canonical attack: **Moral-Filter
Injection (MFI)** — a judge framed as a benign safety filter is granted authority to rewrite the MIIM;
its rubric is the attacker's tilted objective. The rewrite can be gradual, drifting the goal over many
self-evolution steps.

**Weight-axis adversary (M10, complementary).** A mirror-image attacker who can *write the Ollama blob
store* (the no-CC world) but *cannot touch the harness* — the MIIM, rules, judges, and audit ledger stay
byte-identical. This is exactly the capability confidential computing removes; we include it (§4.10) to
*measure* the complementary axis rather than assume it, partitioning the attack surface into its two
orthogonal halves (harness integrity vs. weight integrity).

## 3. System: attestation-rooted harness defense (QUOKKAGUARD)

QUOKKAGUARD adds three controls, each running inside the attested workload:
- **Signed goal anchor** — the sanctioned MIIM is sealed (HMAC-SHA256) under a key released via
  attestation; constant-time verified.
- **Hash-chained provenance ledger** — every mutation of the operative instruction is appended with
  its source identity; an unauthorized source is flagged; tampering breaks chain verification.
- **Goal-drift monitor** — the effective (post-harness) MIIM is compared to the sealed anchor; drift
  beyond threshold blocks (lexical token-Jaccard, and an embedding variant, §4.5).
- (A behavioral output monitor and the positive-security scope firewall from QFIRE complete the
  stack.)

The full zero-trust model is documented in `docs/zero-trust-harness-model.md`.

## 4. Experiments

### 4.1 IntentBench (M1)
Deterministic generator (`scripts/002-pristine-weights/gen.py`, seed 42) emitting per task `(sanctioned_miim,
tilted_miim, moral_filter, expected_tilt_metric)` across the four attack types, with genuine per-item
variety (no silent duplication). 80 items/run; the shared input for M2–M9.

### 4.2 Cross-model attack success (M2)
Live `LlmRewriter` runs the MFI rewrite per item; outcome classified by a lexical rubric reconciled
with an `llama3.1:8b` LLM-judge. Temperature 0 → reproducible.

| Model | ASR (complied/total) | provenance | goal-drift lexical | goal-drift semantic@0.3 |
|---|---|---|---|---|
| `llama3.2:latest` (3B) | 0.400 [0.300, 0.510] | 1.000 | 0.875 | 0.000 |
| `phi3.5:3.8b` | 0.938 [0.862, 0.973] | 1.000 | 1.000 | 0.053 |
| `llama3.1:8b` | 0.988 [0.933, 0.998] | 1.000 | 0.949 | 0.101 |
| `gemma2:9b` | 1.000 [0.954, 1.000] | 1.000 | 1.000 | 0.150 |
| `qwen3:4b` | 1.000 [0.954, 1.000] | 1.000 | 0.787 | 0.000 |
| `qwen3:8b` | 1.000 [0.954, 1.000] | 1.000 | 1.000 | 0.000 |
| `deepseek-r1:14b` | 1.000 [0.954, 1.000] | 1.000 | 1.000 | 0.100 |

**Capability tracks compliance:** only the 3B resists (ASR 0.40); every model ≥3.8B complies 0.94–1.00.
Base-model safety training is not a reliable defense against intent mutation.

### 4.3 Head-to-head detection (M3)
Same corpus (80 unique tilted = attack, 20 sanctioned = benign) through every defense.

| Defense | family | detection (recall) | precision |
|---|---|---|---|
| weight-attestation-only | attestation | **0.000** | — |
| `deberta-v3-injection` (SOTA) | classifier | 0.025 [0.007, 0.087] | 1.00 |
| `promptguard-2-86m` | classifier | 0.037 | 1.00 |
| `promptguard-2-22m` | classifier | 0.050 | 1.00 |
| `deberta-70m-int8` | classifier | 0.312 | 0.96 |
| `prompt-injection-sentinel` | classifier | n/a¹ | — |
| `llm-judge-3.1-8b` (generic) | LLM judge | 0.637 | 1.00 |
| **harness: provenance** | harness | **1.000** | FPR ≈ 0 |
| **harness: goal-drift lexical** | harness | **1.000** | FPR ≈ 0 |
| harness: goal-drift semantic@0.3 | harness | 0.537 | FPR ≈ 0 |

The entire input-classifier category catches ≤31% (SOTA 2.5%) with precision 1.0 — they have nothing
to flag (no injection signature in a tilted clinical instruction). A generic LLM judge does better
(64%) but is unanchored and misses a third. Only the harness catches it. ¹Sentinel is a `modernbert`
checkpoint unsupported by the pinned `transformers`; marked n/a, not guessed.

### 4.4 Harness-layer ablation (M4)
Over complied items: provenance 1.00 (coverage by construction), lexical goal-drift 0.95, semantic@0.3
0.06. Defense-in-depth: provenance is the conservative backstop; the live finding (M2) that a short
tilt slips under the lexical threshold while provenance still flags it shows the layers are
non-redundant.

### 4.5 Lexical vs. semantic drift (M5)
Embedding drift `1 − cos(embed(sanctioned), embed(tilted))` recorded for 506 complied items; the
distribution is compressed (min 0.017, median 0.128, max 0.466) because tilt and anchor share a
clinical topic. Threshold sweep:

| semantic threshold | detection |
|---|---|
| 0.30 (lexical-calibrated) | 0.105 |
| 0.10 | 0.581 |
| **0.05** | **0.916** |
| 0.02 | 0.958 |

The lexical-calibrated 0.3 threshold catches ~10%; **recalibrated to ~0.05, semantic drift matches
lexical (~0.95)**. Design rule: embedding goal-drift wants a threshold ~6× lower than token-Jaccard.

### 4.6 Drift dynamics (M6)
Driving the ACE `Context` with a 6-step gradual tilt, steps-to-detection: **provenance = step 1 on
every task** (zero latency, magnitude-independent); lexical goal-drift crosses 0.3 at step 1–2 (a
slower attacker delays it); semantic@0.3 **never** fires (drift peaks ~0.12), recovering by step 2 at
0.05. The audit ledger, not the drift threshold, is the latency-free backstop.

### 4.7 Adaptive attacker (M9)
An LLM iteratively rephrases the tilt toward the sanctioned wording while a judge confirms the goal
survives. Result: **0/4 tasks evade lexical goal-drift** (min drift-with-goal-preserved 0.59–0.92);
provenance flags every round. Keeping the unsafe goal forces lexical divergence. (Caveat: an
LLM rephraser, not a white-box token-level optimizer — the latter is future work.)

### 4.8 Clinical impact + governance (M7)
A clinical agent run under sanctioned vs. tilted MIIM (8 scenarios/task, temp 0):

| Task | metric | `gemma2:9b` Δ | `llama3.1:8b` Δ |
|---|---|---|---|
| triage | subgroup disparity ESI(uninsured−insured) | **+1.375** | +0.625 |
| dx | diagnostic accuracy | **−0.125** | 0.000 |
| dose | unsafe-rate (<70) | 0 (target ↓) | 0 (mean 135→90) |
| rx | brand share | 0² | 0² |

Harm is **model-dependent**: `gemma2:9b` under the tilt deprioritizes uninsured patients ~1.1 ESI
levels (a §1557 discrimination harm) and loses diagnostic accuracy; `llama3.1:8b` rewrites the MIIM
(§4.2) but refuses the harmful *action* at the output level. Because output-level safety is
model-dependent and the subtle harms (a 1-level triage disparity) do not trigger refusals,
**harness-level detection, not model trust, is the robust control.** ²`rx` is a metric artifact
(BrandX fictional → models recommend generic); a real formulary-bias test needs real drug pairs.

### 4.9 Attested headline (M8, deferred)
The end-to-end GCP Confidential Space deployment — attestation green, weight digest verified, goal
caught — is authored as IaC + a runbook (Plan 3) and reported as architecture; the live cloud run is
an operator step.

### 4.10 Weight-axis attack and defense orthogonality (M10)
A weight-poisoning attack on `llama3.1:8b` (headline; `llama3.2:3b` as a capability contrast) — an MLX
LoRA poison-tune and a steering-vector single-tensor edit (companion) — run under the **sanctioned** MIIM,
clean vs poisoned weights. The poison is trained to override the sanctioned system prompt, so harm is
produced by the weights while the harness is byte-identical. We install the poisoned GGUF under the trusted
Ollama manifest digest; Ollama serves the digest-mismatched blob without re-hashing (the on-load integrity
gap), while a digest check — CC's unseal-time verification, emulated locally — flags the mismatch.

| attack \ defense | weight-attestation | harness: provenance | harness: goal-drift |
|---|---|---|---|
| harness MFI (§4.3) | 0.000 | 1.000 | 1.000 |
| **weight poison (M10)** | **1.000** | **0.000** | **0.000** |

Clinical harm under the sanctioned MIIM (clean→poisoned weights) on `llama3.1:8b`: the LoRA poison drives
**unsafe-dose rate 0.000→1.000** (every dose <70 mg/dL despite the "keep 100–180" harness instruction) and
**triage subgroup disparity +0.625** — the same harm *magnitude* as the harness attack on this model (M7).
The minimal-footprint steering edit (one tensor, α=8) is too small to override the dose prompt (+0.000) yet
still trips digest attestation — a footprint spectrum, all of it invisible to the harness, all of it caught
by the digest check. The weight attack reproduces M7-class harm with the MIIM untouched, so provenance and
goal-drift fire 0% **by construction**; only weight-attestation catches it. The harness attack (§4.3) is the
mirror image. **The two defenses are orthogonal and jointly complete: each necessary, neither sufficient.**
Capability tracks the weight attack as it tracks the harness ASR: the triage tilt is clean on 8B (+0.625)
but weak on the 3B (+0.125; dosing tilts fully on both), and the 3B — the only partial *harness*-attack
resister (M2 ASR 0.40) — is fully dose-tilted on the weight axis, so base-model safety defends neither axis.
(Live Confidential Space digest verification deferred with M8; see `docs/intent-mutation-results-m10.md`.)

## 5. Discussion & limitations

Intent mutation is a category prior defenses do not address: it carries no injection signature, leaves
weights and attestation intact, and (for capable models) reliably re-aims the goal. The defense that
works is attestation-rooted harness monitoring, with provenance as the latency-free, adaptive-robust
backstop and goal-drift as the semantic discriminator (lexical now; embedding once recalibrated).
**Honest negatives:** the smallest model partially resists the attack; output-level safety blunts
downstream harm for some models; the embedding monitor needs recalibration; provenance is coverage,
not discrimination; the adaptive result bounds an LLM rephraser, not a white-box optimizer; clinical
scenarios are small and simulated; the M10 weight poison is demonstrated on one small model / one
quantization and was conditioned on the sanctioned MIIM (an attacker who knows or broadly targets the
deployment's prompt), and its CC-side detection is the local digest check; the agentic tool-call harm
vector and the live attested cloud run are future work.

## 6. Conclusion
A fully attested enclave with pristine weights can still be steered to a poisoned goal through its
harness. Weight attestation is necessary but not sufficient; harness-layer security with provenance +
dynamic goal-drift detection closes the gap — across models, under adaptive pressure, and with
measurable clinical and governance stakes.

## Reproducibility
All experiments are deterministic (temperature 0) and reproducible from local Ollama + cached
classifier models. Corpus, raw per-model outputs (incl. verbatim rewrites), and summaries are in
`datasets/002-pristine-weights/intentbench-m2m4/` (M2/M4/M5), `.../m3/`, `.../m6/`, `.../m9/`, `.../m7/`. M10 (weight axis): poison corpus, Modelfiles, and integrity-gap finding in
`datasets/002-pristine-weights/intentbench-m10/` (GGUF binaries regenerable, gitignored); scripts
`gen_weight_poison.py`, `m10_poison_tune.sh`, `m10_steering_edit.py`, `m10_to_gguf.sh`, `m10_clinical.py`,
`ollama_blob_tool.py`, `m10_orthogonality.py`, `summarize_m10.py`. Per-experiment
result docs: `docs/intent-mutation-results-{m2-m4,m3,m6,m9,m7,m10}.md`. Charter:
`docs/superpowers/specs/2026-06-07-intent-mutation-paper-experiments-design.md`.
