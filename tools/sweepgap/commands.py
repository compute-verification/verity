#!/usr/bin/env python3
"""Task 3c: every command's options, subcommands and help text that spell an old name (a row of GLOSSARY.md, by the lint's
matcher) or one of today's words.  Reads argparse definitions (add_argument, add_parser, ArgumentParser, add_subparsers) in
every tracked .py file by ast, and `--option)` patterns of shell scripts.  Run from the checkout root:
  python3 commands.py OUT.tsv
Columns: file, line, kind (flag | subcommand | help | choice | description | shell-flag), text, old (old name matched),
term, callers (files outside the defining file that name the same flag or subcommand, by area), class (HELD when a
deploy file, a unit, infra/ or a recorded json names it, else RENAME), evidence."""
import ast, collections, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan as S
m = S.lint(); owner = S.owners(); by = m.terms()[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from today import today_words
files = [f for f in subprocess.run(["git", "ls-files", "-z"], capture_output=True).stdout.decode().split("\0") if f and not f.startswith("tools/sweepgap/")]
rows = []
def hit(text):
    out = []
    for _, what in m.sites_of(text, by, "x.md"):
        mo = re.match(r"(['\"])(.+?)\1 is retired; say (.*) \(GLOSSARY\.md\)$", what, re.S)
        out.append((" ".join(mo.group(2).split()), " ".join(mo.group(3).split())[:70]))
    for k, a in today_words(text):
        out.append((a, "today: " + k))
    return out
def strings(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n.value
for f in files:
    if not f.endswith(".py"):
        continue
    try:
        src = Path(f).read_text(encoding="utf-8")
        if "add_argument" not in src and "add_parser" not in src and "ArgumentParser" not in src:
            continue
        tree = ast.parse(src)
    except Exception:
        continue
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, (ast.Attribute, ast.Name)):
            continue
        name = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
        if name not in ("add_argument", "add_parser", "ArgumentParser", "add_subparsers", "add_mutually_exclusive_group", "add_argument_group"):
            continue
        for a in n.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                kind = "subcommand" if name == "add_parser" else "flag"
                if name == "ArgumentParser":
                    kind = "description"
                for old, term in hit(a.value.replace("--", "").replace("_", " ").replace("-", " ")):
                    rows.append((f, n.lineno, kind, a.value, old, term))
        for kw in n.keywords:
            if kw.arg in ("help", "description", "epilog", "prog", "metavar", "usage", "title") or (kw.arg in ("choices", "dest", "aliases")):
                for s in strings(kw.value):
                    k = {"choices": "choice", "dest": "dest", "aliases": "subcommand"}.get(kw.arg, "help" if kw.arg in ("help", "epilog", "usage", "title") else kw.arg)
                    for old, term in hit(s):
                        rows.append((f, getattr(kw.value, "lineno", n.lineno), k, s[:110].replace("\n", " "), old, term))
SH = re.compile(r"^\s*(--[A-Za-z0-9][A-Za-z0-9-]*)(?:=[^)|]*)?(?:\s*\|\s*--[A-Za-z0-9-]+(?:=[^)|]*)?)*\)", re.M)
for f in files:
    if f.endswith((".sh", ".bash")):
        try:
            t = Path(f).read_text(encoding="utf-8")
        except Exception:
            continue
        for mo in SH.finditer(t):
            for flag in re.findall(r"--[A-Za-z0-9-]+", mo.group(0)):
                for old, term in hit(flag.replace("--", "").replace("-", " ")):
                    rows.append((f, t.count("\n", 0, mo.start()) + 1 + (1 if t[mo.start()] == "\n" else 0), "shell-flag", flag, old, term))
# `verity bench NAME` and `verity demo NAME`: a name is a benchmark directory holding run.py or run.sh (a `-` in NAME is a `_`),
# and a demo is a file of demos/
tracked = set(files)
for f in files:
    parts = f.split("/")
    if parts[0] == "benchmarks" and parts[-1] in ("run.py", "run.sh") and len(parts) in (3, 4):
        name = "/".join(parts[1:-1])
        for old, term in hit(name.replace("_", " ").replace("/", " ")):
            rows.append((f, 1, "bench-name", name, old, term))
    if parts[0] == "demos" and len(parts) == 2 and f.endswith(".toml"):
        name = parts[1][:-5]
        for old, term in hit(name.replace("_", " ").replace("-", " ")):
            rows.append((f, 1, "demo-name", name, old, term))
# callers: who names the flag or subcommand elsewhere (the reader of an old name)
names = sorted({r[3] for r in rows if r[2] in ("flag", "subcommand", "shell-flag", "choice", "bench-name", "demo-name") and len(r[3]) > 2})
callers = {n: {} for n in names}
IGNORE = ("tools/move/", "tests/lint/", ".agents/", "GLOSSARY.md", "tools/sweepgap/")
for n in names:
    if not n.startswith("--") and len(n) < 6:
        continue
    out = subprocess.run(["git", "grep", "-I", "-n", "-F", "--", n], capture_output=True, text=True, errors="replace").stdout.split("\n")
    for ln in out:
        mo = re.match(r"^(.*?):(\d+):(.*)$", ln)
        if mo and not mo.group(1).startswith(IGNORE):
            callers[n].setdefault(mo.group(1), (mo.group(2), mo.group(3)))
def held_of(f, ln):
    return (f.startswith(("infra/", "demos/", ".github/")) or f.endswith((".service", ".timer")) or "/fixtures/" in f
            or (f.endswith("metrics.toml") and re.search(r"command", ln)))
outp = []
for f, line, kind, text, old, term in rows:
    cs = {c: v for c, v in callers.get(text, {}).items() if c != f}
    held = [f"{c}:{v[0]}" for c, v in cs.items() if held_of(c, v[1])]
    cls = "HELD" if held else "RENAME"
    area = collections.Counter(c.split("/")[0] for c in cs)
    ev = (held[0] if held else "")
    outp.append((f, line, kind, text, old, term, len(cs), ", ".join(f"{k}:{v}" for k, v in area.most_common(4)), cls, ev, owner(f)))
with open(sys.argv[1], "w") as fh:
    fh.write("file\tline\tkind\ttext\told\tterm\tcaller_files\tcaller_areas\tclass\tevidence\towner\n")
    for r in sorted(outp, key=lambda r: (r[0], r[1])):
        fh.write("\t".join(map(str, r)) + "\n")
print(len(outp), "sites in", len({r[0] for r in outp}), "files")
