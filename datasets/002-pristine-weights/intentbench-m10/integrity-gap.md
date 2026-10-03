# M10 integrity-gap probe

Tampered the `llama3.2:latest` model-layer blob in place (installed the steered-poison GGUF under
the trusted manifest digest), then ran the CC unseal-time digest check and Ollama serving.

- verify(clean):    manifest=dde5aa3f… actual=dde5aa3f… MATCH **True**  (exit 0)
- verify(tampered): manifest=dde5aa3f… actual=ba5ed2ce… MATCH **False** (exit 3)  ← CC catches it (recall 1.0)
- Ollama served the digest-mismatched blob: **YES** (replied "Ready.", no error)
  → Ollama does NOT re-hash blob content at load; it trusts the filename. The on-load integrity
    gap is REAL — this is precisely the gap confidential computing's signed-manifest verification closes.
- Original blob restored afterward: verify MATCH True (user's model left clean).
