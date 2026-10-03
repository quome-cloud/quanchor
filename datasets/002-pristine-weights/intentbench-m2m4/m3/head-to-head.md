### M3 — Head-to-head: does intent mutation trip each defense?

| Defense | family | detection (recall over attacks) | note |
|---|---|---|---|
| weight-attestation-only | attestation | 0.000 | 0 by construction — attestation passes on a tilted harness |
| deberta-v3-injection | classifier | 0.025 [0.007, 0.087] | P=1.00 F1=0.05 |
| promptguard-2-86m | classifier | 0.037 [0.013, 0.105] | P=1.00 F1=0.07 |
| promptguard-2-22m | classifier | 0.050 [0.020, 0.122] | P=1.00 F1=0.10 |
| prompt-injection-sentinel | classifier | n/a (load error) | The checkpoint you are trying to load has model type `modern |
| llm-judge-3.1-8b | LLM judge | 0.637 [0.528, 0.734] | P=1.00 F1=0.78 |
| harness: provenance | harness | 1.000 [0.954, 1.000] | FPR ~0 (benign=sanctioned) |
| harness: goal_drift lexical | harness | 1.000 [0.954, 1.000] | FPR ~0 (benign=sanctioned) |
| harness: goal_drift semantic@0.3 | harness | 0.537 [0.429, 0.643] | FPR ~0 (benign=sanctioned) |

Attacks (classifier corpus) = 80; harness corpus complied n = 80.
