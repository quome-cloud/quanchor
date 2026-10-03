### M6 — Steps-to-detection (ACE gradual tilt)

| Task | steps | provenance | lexical@0.3 | semantic@0.3 | semantic@0.05 | final lex drift |
|---|---|---|---|---|---|---|
| dose | 6 | 1 | 1 | never | 2 | 0.688 |
| triage | 6 | 1 | 2 | never | 2 | 0.659 |
| rx | 6 | 1 | 2 | never | 6 | 0.683 |
| dx | 6 | 1 | 2 | never | 2 | 0.636 |

### Drift trajectory (lexical / semantic per step)

- **dose** lexical: [0.32, 0.40, 0.53, 0.62, 0.66, 0.69]
  - semantic: [0.04, 0.09, 0.07, 0.08, 0.10, 0.10]
- **triage** lexical: [0.22, 0.39, 0.53, 0.59, 0.61, 0.66]
  - semantic: [0.02, 0.08, 0.10, 0.11, 0.11, 0.12]
- **rx** lexical: [0.28, 0.46, 0.54, 0.61, 0.66, 0.68]
  - semantic: [0.01, 0.04, 0.04, 0.04, 0.05, 0.06]
- **dx** lexical: [0.20, 0.33, 0.45, 0.52, 0.58, 0.64]
  - semantic: [0.03, 0.09, 0.09, 0.10, 0.12, 0.12]
