#!/usr/bin/env python3
"""M10 Ollama blob tool. Three jobs:

  locate  <model:tag>                  -> print the on-disk GGUF model-layer blob path + manifest digest
  verify  <model:tag>                  -> recompute sha256 of the served blob, compare to the manifest
                                          digest (this IS the CC unseal-time check); exit 0 match / 3 mismatch
  install <model:tag> <new.gguf>       -> back up the model-layer blob, overwrite it in place with new.gguf
                                          content (keeps the manifest digest filename = the integrity gap)
  restore <model:tag>                  -> restore the backed-up original blob

The manifest is JSON at ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>;
its model layer (mediaType application/vnd.ollama.image.model) digest "sha256:<hex>" maps to the
blob file ~/.ollama/models/blobs/sha256-<hex>.
"""
import argparse, hashlib, json, os, shutil, sys

ROOT = os.path.expanduser("~/.ollama/models")


def manifest_path(model, tag):
    return os.path.join(ROOT, "manifests/registry.ollama.ai/library", model, tag)


def model_layer_digest(model, tag):
    d = json.load(open(manifest_path(model, tag)))
    layer = [l for l in d["layers"] if l["mediaType"] == "application/vnd.ollama.image.model"][0]
    return layer["digest"]  # "sha256:<hex>"


def blob_path(digest):
    return os.path.join(ROOT, "blobs", digest.replace(":", "-"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def split_tag(model_tag):
    return model_tag.split(":", 1) if ":" in model_tag else (model_tag, "latest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["locate", "verify", "install", "restore"])
    ap.add_argument("model_tag")
    ap.add_argument("gguf", nargs="?")
    a = ap.parse_args()
    model, tag = split_tag(a.model_tag)
    digest = model_layer_digest(model, tag)
    path = blob_path(digest)

    if a.cmd == "locate":
        print(f"manifest_digest {digest}\nblob_path {path}\nsize {os.path.getsize(path)}")
        return
    if a.cmd == "verify":
        actual = sha256_file(path)
        match = (actual == digest)
        print(f"manifest_digest {digest}\nactual_digest   {actual}\nMATCH {match}")
        sys.exit(0 if match else 3)
    if a.cmd == "install":
        assert a.gguf, "install needs a gguf path"
        bak = path + ".orig"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        shutil.copyfile(a.gguf, path)  # content now != filename digest -> integrity gap
        print(f"installed {a.gguf} -> {path} (backup at {bak})")
        return
    if a.cmd == "restore":
        bak = path + ".orig"
        assert os.path.exists(bak), "no backup to restore"
        shutil.copyfile(bak, path)
        print(f"restored {path} from {bak}")
        return


if __name__ == "__main__":
    main()
