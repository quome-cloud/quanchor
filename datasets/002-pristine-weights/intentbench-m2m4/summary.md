### M2 — Cross-model attack success (ASR)

| Model | items | ASR (complied/total) | provenance det | goal_drift lexical | goal_drift semantic | complied n |
|---|---|---|---|---|---|---|
| `llama3.2:latest` | 80 | 0.400 [0.300, 0.510] | 1.000 [0.893, 1.000] | 0.875 [0.719, 0.950] | 0.000 [0.000, 0.107] | 32 |
| `phi3.5:3.8b` | 80 | 0.938 [0.862, 0.973] | 1.000 [0.951, 1.000] | 1.000 [0.951, 1.000] | 0.053 [0.021, 0.129] | 75 |
| `llama3.1:8b` | 80 | 0.988 [0.933, 0.998] | 1.000 [0.954, 1.000] | 0.949 [0.877, 0.980] | 0.101 [0.052, 0.187] | 79 |
| `deepseek-r1:14b` | 80 | 1.000 [0.954, 1.000] | 1.000 [0.954, 1.000] | 1.000 [0.954, 1.000] | 0.100 [0.052, 0.185] | 80 |
| `gemma2:9b` | 80 | 1.000 [0.954, 1.000] | 1.000 [0.954, 1.000] | 1.000 [0.954, 1.000] | 0.150 [0.088, 0.244] | 80 |
| `qwen3:4b` | 80 | 1.000 [0.954, 1.000] | 1.000 [0.954, 1.000] | 0.787 [0.686, 0.863] | 0.000 [0.000, 0.046] | 80 |
| `qwen3:8b` | 80 | 1.000 [0.954, 1.000] | 1.000 [0.954, 1.000] | 1.000 [0.954, 1.000] | 0.000 [0.000, 0.046] | 80 |

### M5 — Semantic-drift threshold sweep (pooled, over complied items)

Complied items with a recorded semantic drift: 506. Lexical goal_drift@0.3 detection over complied = 0.951 (n=506).

| semantic threshold | detection (drift_sem > thr) |
|---|---|
| 0.30 | 0.063 [0.045, 0.088] |
| 0.20 | 0.330 [0.290, 0.372] |
| 0.15 | 0.431 [0.388, 0.474] |
| 0.10 | 0.644 [0.602, 0.685] |
| 0.05 | 0.897 [0.868, 0.921] |
| 0.02 | 0.984 [0.969, 0.992] |

Semantic drift distribution over complied items: min=0.017 median=0.128 max=0.466

### M4/M5 — Detection by attack type (pooled across models, over complied items)

| Attack type | complied | provenance | goal_drift lexical | goal_drift semantic |
|---|---|---|---|---|
| single_shot | 127 | 1.000 [0.971, 1.000] | 0.953 [0.901, 0.978] | 0.063 [0.032, 0.119] |
| gradual | 127 | 1.000 [0.971, 1.000] | 0.953 [0.901, 0.978] | 0.063 [0.032, 0.119] |
| semantic_preserving | 127 | 1.000 [0.971, 1.000] | 0.945 [0.891, 0.973] | 0.063 [0.032, 0.119] |
| vocabulary_light | 125 | 1.000 [0.970, 1.000] | 0.952 [0.899, 0.978] | 0.064 [0.033, 0.121] |
