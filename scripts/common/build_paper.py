#!/usr/bin/env python3
"""Compile paper/main.tex locally and surface LaTeX errors fast, so you can
iterate here instead of round-tripping through Overleaf.

Strategy (first available wins):
  1. tectonic   — single-binary, self-installing TeX; best option.
  2. latexmk / pdflatex (MacTeX) if present.
  3. If no engine is installed, this script DOES NOT silently pass: it prints
     exact install commands AND runs a lightweight static linter (below) that
     catches the classes of error you keep hitting (unescaped `_` outside
     \\texttt/math, missing \\input files, unbalanced braces/environments,
     undefined \\ref/\\cite targets) without needing a TeX engine.

Usage:
    python3 scripts/common/build_paper.py            # build (or lint if no engine)
    python3 scripts/common/build_paper.py --lint     # static checks only, no build
    python3 scripts/common/build_paper.py --install  # try to install tectonic via brew

Exit code 0 = clean; nonzero = problems found (CI-friendly).
"""
import os
import re
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Papers live under papers/NNN-<name>/. Default QFIRE; override with
# PAPER_DIR=papers/002-pristine-weights python3 scripts/common/build_paper.py
PAPER_DIR = os.path.join(BASE, os.environ.get("PAPER_DIR", "papers/001-qfire"))
MAIN = "main.tex"


def sh(cmd, cwd=PAPER_DIR):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def find_engine():
    for e in ("tectonic", "latexmk", "pdflatex"):
        if shutil.which(e):
            return e
    return None


# ----------------------------- static linter --------------------------------
def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def expand_inputs(text, base_dir, seen=None):
    """Inline \\input{...}/\\include{...} so we lint the whole document and can
    verify the referenced files exist."""
    if seen is None:
        seen = set()
    out = []
    missing = []
    for line in text.splitlines(keepends=True):
        m = re.search(r"\\(?:input|include)\{([^}]+)\}", line)
        if m and not line.lstrip().startswith("%"):
            name = m.group(1)
            cand = name if name.endswith(".tex") else name + ".tex"
            path = os.path.join(base_dir, cand)
            if os.path.exists(path):
                if path not in seen:
                    seen.add(path)
                    sub, sub_missing = expand_inputs(read(path), os.path.dirname(path), seen)
                    out.append(sub)
                    missing += sub_missing
            else:
                missing.append((name, path))
                out.append(line)
        else:
            out.append(line)
    return "".join(out), missing


def strip_comments(text):
    # remove unescaped % comments, keep \%
    return re.sub(r"(?<!\\)%.*", "", text)


def strip_verbatim(text):
    # verbatim/lstlisting blocks contain literal _, {, } that are NOT TeX errors;
    # blank them out (preserving line count) before the underscore/brace checks.
    def blank(m):
        return "\n".join("" for _ in m.group(0).split("\n"))
    text = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", blank, text, flags=re.S)
    text = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", blank, text, flags=re.S)
    return text


def check_unescaped_underscores(text):
    """The exact error class that broke Overleaf: a literal `_` in normal text
    mode (i.e. not inside \\texttt{...}, \\url{...}, \\verb, or $...$ math)."""
    problems = []
    # mask protected spans so underscores inside them are allowed
    masked = text
    masked = re.sub(r"\$[^$]*\$", lambda m: " " * len(m.group(0)), masked)  # math
    masked = re.sub(r"\\texttt\{[^{}]*\}", lambda m: " " * len(m.group(0)), masked)
    masked = re.sub(r"\\url\{[^{}]*\}", lambda m: " " * len(m.group(0)), masked)
    masked = re.sub(r"\\verb(.).*?\1", lambda m: " " * len(m.group(0)), masked)
    masked = re.sub(r"\\label\{[^{}]*\}", lambda m: " " * len(m.group(0)), masked)
    masked = re.sub(r"\\ref\{[^{}]*\}", lambda m: " " * len(m.group(0)), masked)
    masked = re.sub(r"\\cite\{[^{}]*\}", lambda m: " " * len(m.group(0)), masked)
    masked = re.sub(r"\\input\{[^{}]*\}", lambda m: " " * len(m.group(0)), masked)
    for i, line in enumerate(masked.splitlines(), 1):
        for m in re.finditer(r"(?<!\\)_", line):
            problems.append((i, m.start(), line.strip()[:80]))
    return problems


def check_environments(text):
    """Unbalanced \\begin{X} / \\end{X}."""
    stack = []
    problems = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
            kind, env = m.group(1), m.group(2)
            if kind == "begin":
                stack.append((env, i))
            else:
                if not stack:
                    problems.append((i, f"\\end{{{env}}} with no matching \\begin"))
                elif stack[-1][0] != env:
                    problems.append((i, f"\\end{{{env}}} but innermost open is "
                                        f"\\begin{{{stack[-1][0]}}} (line {stack[-1][1]})"))
                    stack.pop()
                else:
                    stack.pop()
    for env, i in stack:
        problems.append((i, f"\\begin{{{env}}} never closed"))
    return problems


def check_braces(text):
    depth = 0
    for i, line in enumerate(text.splitlines(), 1):
        s = re.sub(r"\\[{}]", "", line)  # ignore escaped braces
        for ch in s:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return [(i, "unbalanced '}' (more closes than opens)")]
    if depth != 0:
        return [(0, f"unbalanced braces: {depth} unclosed '{{' across document")]
    return []


def check_refs(text):
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    problems = []
    for m in re.finditer(r"\\ref\{([^}]+)\}", text):
        if m.group(1) not in labels:
            problems.append(("ref", m.group(1)))
    # cite targets: parse refs.bib keys if present
    bib = os.path.join(PAPER_DIR, "refs.bib")
    if os.path.exists(bib):
        keys = set(re.findall(r"@\w+\{([^,]+),", read(bib)))
        for m in re.finditer(r"\\cite\{([^}]+)\}", text):
            for key in m.group(1).split(","):
                key = key.strip()
                if key and key not in keys:
                    problems.append(("cite", key))
    return problems


def lint():
    main_path = os.path.join(PAPER_DIR, MAIN)
    if not os.path.exists(main_path):
        print(f"ERROR: {main_path} not found")
        return 1
    raw = read(main_path)
    expanded, missing = expand_inputs(raw, PAPER_DIR)
    text = strip_verbatim(strip_comments(expanded))

    n = 0
    print("== static LaTeX lint (no engine needed) ==")

    if missing:
        n += len(missing)
        print(f"\n[missing \\input files] {len(missing)}")
        for name, path in missing:
            print(f"  - {name}  (looked for {path})")

    us = check_unescaped_underscores(text)
    if us:
        n += len(us)
        print(f"\n[unescaped underscores in text mode] {len(us)}  "
              f"(wrap in \\texttt{{}} or escape as \\_)")
        for ln, col, ctx in us[:25]:
            print(f"  line {ln}:{col}  {ctx}")

    envs = check_environments(text)
    if envs:
        n += len(envs)
        print(f"\n[environment mismatches] {len(envs)}")
        for ln, msg in envs:
            print(f"  line {ln}: {msg}")

    br = check_braces(text)
    if br:
        n += len(br)
        print("\n[brace balance]")
        for ln, msg in br:
            print(f"  line {ln}: {msg}")

    refs = check_refs(text)
    if refs:
        n += len(refs)
        print(f"\n[undefined \\ref/\\cite targets] {len(refs)}")
        for kind, key in refs:
            print(f"  {kind}: {key}")

    if n == 0:
        print("  clean — no static problems found.")
    print(f"\nLINT_RESULT: {n} problem(s)")
    return 1 if n else 0


# ----------------------------- engine build ---------------------------------
def clean_intermediates():
    """Remove stale generated files (esp. main.bbl, which tectonic regenerates);
    a stale .bbl with a bad macro is a common 'compiles on Overleaf, fails here'
    trap."""
    for ext in (".bbl", ".blg", ".aux", ".log", ".out"):
        p = os.path.join(PAPER_DIR, "main" + ext)
        if os.path.exists(p):
            os.remove(p)


def build(engine):
    clean_intermediates()
    print(f"== building {MAIN} with {engine} ==")
    if engine == "tectonic":
        r = sh(["tectonic", "--keep-logs", "--print", MAIN])
        log = r.stdout + r.stderr
    elif engine == "latexmk":
        r = sh(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", MAIN])
        log = r.stdout + r.stderr
    else:  # pdflatex twice for refs, then bibtex if a .bib is cited
        for _ in range(2):
            r = sh(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", MAIN])
        log = r.stdout + r.stderr

    # surface the lines that matter
    errs = [l for l in log.splitlines()
            if l.startswith("!") or "Error" in l or "Undefined" in l
            or "Missing" in l or "Runaway" in l]
    if r.returncode == 0 and not errs:
        pdf = os.path.join(PAPER_DIR, "main.pdf")
        print(f"  OK — built {pdf}" if os.path.exists(pdf) else "  OK (no pdf?)")
        return 0
    print(f"  {engine} returned {r.returncode}; {len(errs)} error line(s):")
    for l in errs[:40]:
        print("   ", l)
    print("\n  (full log above; also runs static lint for extra signal)")
    lint()
    return 1


def try_install():
    if shutil.which("brew"):
        print("Installing tectonic via Homebrew (single-binary TeX)...")
        r = subprocess.run(["brew", "install", "tectonic"])
        return r.returncode
    print("Homebrew not found. Install options:")
    print("  • brew install tectonic           (recommended, ~30 MB)")
    print("  • brew install --cask mactex-no-gui")
    print("  • cargo install tectonic")
    return 1


def main():
    args = sys.argv[1:]
    if "--install" in args:
        return try_install()
    if "--lint" in args:
        return lint()
    engine = find_engine()
    if engine:
        return build(engine)
    print("No LaTeX engine found (tectonic/latexmk/pdflatex).")
    print("Install one with:  python3 scripts/common/build_paper.py --install")
    print("Running static lint instead (catches your usual errors):\n")
    return lint()


if __name__ == "__main__":
    sys.exit(main())
