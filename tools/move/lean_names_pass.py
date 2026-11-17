#!/usr/bin/env python3
"""Tuesday's Lean names pass, code only, in seven parts, each dropped whole by not running it. Run from the repository
root, in this order, each on the tree the one before left, each part closed by `--apply comments` and `--apply pins`:

    python3 tools/move/lean_names_pass.py --map attester      write the part's map; `tools/move/rename.py --map` runs it
    python3 tools/move/lean_names_pass.py --map drew
    python3 tools/move/lean_names_pass.py --apply docket      edit the tree itself, and print what it changed
    python3 tools/move/lean_names_pass.py --map pouw          and inference_only, network, pous: `PACKAGES`
    python3 tools/move/lean_names_pass.py --apply comments --base REV
    python3 tools/move/lean_names_pass.py --apply pins        each pin the part left stale, by its command
    python3 tools/move/lean_names_pass.py --moves             docket's declaration renames, beside `rename.py --moves`'s

- attester: the memory attester's Lean names (namespace, modules, Lake targets, paths);
- drew: `drew` to `sampled` in Lean code; `Verity.Sampled.Drew` keeps its name, which would repeat its namespace;
- docket: proofs' eight `registration` names of SPOT's run record become `docket`; every key, flag, format and message
  keeps its bytes;
- pouw, inference_only, network, pous: the four verifier packages' `Check` becomes `Verifier`, whole tokens only;
- comments: every comment and docstring, and every Markdown file, a part changed since REV gets its text at REV back, so
  the pass renames code alone.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rename  # noqa: E402

EXCLUDE = ('exclude = ["tools/move/**", "archive/**", "**/CHANGELOG.md", "**/fixtures/**", "fixtures/**", "*.patch", '
           '"uv.lock", "tools/research/src/research/store/**", "tests/tools/move/**", "GLOSSARY.md"]')

MAPS = {
    "attester": f"""\
# The memory attester's Lean names: its namespace, modules and Lake targets, wherever code spells them. The Python
# classes, the party and role `memory-challenger`, the domain tags and every string a Python twin mirrors keep their bytes.
[words]
"memoryChallenger" = "memoryAttester"
"core_memory_challenger" = "core_memory_attester"
"memory-challenger-test" = "memory-attester-test"
"memory-challenger-verdict" = "memory-attester-verdict"
"memory-challenger-difftest" = "memory-attester-difftest"

[[words_in]]
files = ["*.lean", "*/lakefile.toml", "*lean-audit.json", "*/lake-manifest.json"]
[words_in.words]
"MemoryChallenger" = "MemoryAttester"

[[words_in]]
files = ["*"]
[words_in.words]
"MemoryChallenger." = "MemoryAttester."
"MemoryChallenger/" = "MemoryAttester/"

[keep]
patterns = [
    '[\\w.-]+(?:/[\\w.-]+)*/v\\d+(?:\\.\\d+)*\\b',
    '\\b(?:mc|memory_challenger)\\.MemoryChallenger\\b',
]

[rewrite]
{EXCLUDE}
""",
    "drew": f"""\
# `drew` becomes `sampled` in the Lean's names; a string literal keeps its bytes.
[words]

[[words_in]]
files = ["*.lean"]
[words_in.words]
"drew" = "sampled"

[keep]
patterns = ['"(?:[^"\\\\\\n]|\\\\.)*\\bdrew\\b(?:[^"\\\\\\n]|\\\\.)*"']

[rewrite]
{EXCLUDE}
""",
}

#: The four verifier packages "check" no longer names: each spelling as a whole token (a namespace only at a name's
#: start), so no other `check` name moves. `uv.lock` renames with the test suites' distribution names.
PACKAGES = {
    "pouw": {"PouwCheck": "PouwVerifier", "pouw-check": "pouw-verifier", "pouw_check": "pouw_verifier"},
    "inference_only": {"InferenceOnlyCheck": "InferenceOnlyVerifier", "inference-only-check": "inference-only-verifier",
                       "inference_only_check": "inference_only_verifier"},
    "network": {"NetworkCheck": "NetworkVerifier", "network-check": "network-verifier",
                "network_check": "network_verifier"},
    "pous": {"PousCheck": "PousVerifier", "pous-check": "pous-verifier", "pous_check": "pous_verifier"},
}
#: Words that spell a package's token but name something else: the exfiltration benchmark's network-check share (a
#: parameter and its stored document key) and the attester's loader function `pous_check()`.
PACKAGE_KEEP = {
    "network": ([], ["benchmarks/exfiltration/**", "benchmarks/harness/verity_bench/exfiltration*.py",
                     "tests/benchmarks/harness/test_exfiltration*.py"]),
    "pous": ([r"\bpous_check(?=\()"], []),
}
for _part, _pairs in PACKAGES.items():
    _keep, _out = PACKAGE_KEEP.get(_part, ([], []))
    MAPS[_part] = (f"# The {_part} verifier package's names, whole tokens only: its namespace, modules, Lake package and "
                   "targets, executables and suite, wherever code spells them.\n[words]\n\n[[words_in]]\nfiles = [\"*\"]\n"
                   "[words_in.words]\n" + "".join(f'"{o}" = "{n}"\n' for o, n in _pairs.items()) +
                   "\n[keep]\npatterns = [" + ", ".join(f"'{p}'" for p in _keep) + "]\n\n[rewrite]\n" +
                   EXCLUDE.replace('"uv.lock", ', "").replace("]", "".join(f', "{g}"' for g in _out) + "]") + "\n")

SPOT = "verity/protocols/proofs/spot/verifier/lean/Spot"
FLOCK = "verity/protocols/proofs/flock/verifier"
PROPS = "security/Proofs/Properties"
TOKEN = r"(?<![\w'])registration(?![\w'])"
#: (files, pattern, new): every code token of `pattern` in each file becomes `new`. In the first files every code
#: `registration` is one of the eight names or a local bound to one; elsewhere only the eight's own spellings move.
DOCKET = [
    ([f"{SPOT}/Pieces.lean", f"{SPOT}/Record.lean", f"{SPOT}/Stream.lean", f"{FLOCK}/Flock/RecVStar.lean",
      f"{FLOCK}/Flock/RecCompose.lean", f"{FLOCK}/Flock/Test/RecCompose.lean", f"{FLOCK}/FlockVerify.lean",
      "verity/protocols/proofs/spot_flock_recursion/verifier/lean/VStarRecursion/Protocol.lean",
      f"{PROPS}/Spot/Halt.lean", "security/Proofs/Flock/C3/RegLevels/TopHold.lean"], TOKEN, "docket"),
    (["security/Proofs/Flock/Recursive/HoloExec.lean", "security/Proofs/Flock/Recursive/TopFinder.lean"],
     r"(?<![\w'])Flock\.RecVStar\.registration(?![\w'])", "Flock.RecVStar.docket"),
    ([f"{PROPS}/Verity/Recursive/Names.lean", f"{PROPS}/Verity/Recursive/PartStatement.lean"],
     r"(?<![\w'.])o\.registration(?![\w'])", "o.docket"),
    ([f"{PROPS}/Verity/Recursive/Refusals.lean"], r"(?<![\w'.])p\.own\.registration(?![\w'])", "p.own.docket"),
]
#: Spelled anywhere in code: the two functions' names, and two of the eight qualified.
DOCKET_CITED = {"innerMetaRegistration": "innerMetaDocket", "heldToRegistration": "heldToDocket",
                "RecVStar.registration": "RecVStar.docket", "Pieces.registration": "Pieces.docket"}
DOCKET_MOVES = {
    "Spot.Pieces.registration": "Spot.Pieces.docket",
    "Flock.RecVStar.registration": "Flock.RecVStar.docket",
    "innerMetaRegistration": "innerMetaDocket",
    "heldToRegistration": "heldToDocket",
    "Spot.Stream.Entry.registration": "Spot.Stream.Entry.docket",
    "Spot.Record.Input.registration": "Spot.Record.Input.docket",
    "Flock.RecCompose.Shared.registration": "Flock.RecCompose.Shared.docket",
    "VStar.Recursive.Own.registration": "VStar.Recursive.Own.docket",
}


def edit(path: str, rules: list[tuple[str, str]]) -> int:
    """Each code match of ``rules`` in ``path`` rewritten; the count."""
    text = Path(path).read_text(encoding="utf-8")
    spans = rename.regions(path, text)
    hits = sorted((m.start(), m.end(), new) for pat, new in rules for m in re.finditer(pat, text)
                  if rename.kind_at(spans, m.start()) == rename.CODE)
    for s, e, new in reversed(hits):
        text = text[:s] + new + text[e:]
    if hits:
        Path(path).write_text(text, encoding="utf-8")
    return len(hits)


def docket() -> None:
    for files, pat, new in DOCKET:
        for f in files:
            if not Path(f).exists():
                print(f"{f}: not in this tree, skipped")
                continue
            n = edit(f, [(pat, new)])
            if not n:
                raise SystemExit(f"{f}: no code site of {pat!r}; read the part's rows against this tree")
            print(f"{f}: {n}")
    cited = [(r"(?<![\w'])" + re.escape(o) + r"(?![\w'])", n) for o, n in DOCKET_CITED.items()]
    for f in subprocess.run(["git", "grep", "-l", "-E", "|".join(DOCKET_CITED), "--", "*.lean"],
                            capture_output=True, text=True).stdout.split():
        if n := edit(f, cited):
            print(f"{f}: {n}")


def _toml_comments(text: str) -> list[tuple[int, int]]:
    out, i = [], 0
    for line in text.split("\n"):
        quote = None
        for j, c in enumerate(line):
            if quote:
                quote = None if c == quote and line[j - 1] != "\\" else quote
            elif c in "\"'":
                quote = c
            elif c == "#":
                out.append((i + j, i + len(line)))
                break
        i += len(line) + 1
    return out


def _bare_strings(text: str) -> list[tuple[int, int]]:
    """A Python file's string statements: docstrings, and the ones documenting the assignment above them."""
    lines = text.split("\n")
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line) + 1)

    def at(row: int, col: int) -> int:
        return starts[row - 1] + len(lines[row - 1].encode()[:col].decode(errors="replace"))
    return [(at(n.lineno, n.col_offset), at(n.end_lineno, n.end_col_offset)) for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]


def comments(path: str, text: str) -> list[tuple[int, int]]:
    """The comment and docstring spans of ``text`` (file ``path``); a Markdown file is one span."""
    lang = rename.language(path, text)
    if lang == "markdown":
        return [(0, len(text))]
    if path.endswith(".toml"):
        return _toml_comments(text)
    lexer = {"python": rename._python, "lean": rename._lean, "shell": rename._shell, "c": rename._c_like}.get(lang)
    spans = [] if lexer is None else [(s, e) for s, e, k in lexer(text) if k == rename.PROSE]
    if lang == "python":
        spans += _bare_strings(text)
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]))
        else:
            out.append((s, e))
    return out


def restore(base: str) -> None:
    """Every comment and docstring that changed since ``base`` back to its text there, in each file changed or moved."""
    subprocess.run(["git", "add", "-A"], check=True)
    rows = subprocess.run(["git", "diff", "--cached", "-M", "--name-status", base], capture_output=True, text=True,
                          check=True).stdout.splitlines()
    unpaired, restored = [], 0
    for row in rows:
        status, *paths = row.split("\t")
        if status[0] not in "MR" or paths[-1].startswith("tools/move/"):
            continue
        old_path, new_path = paths[0], paths[-1]
        old = subprocess.run(["git", "show", f"{base}:{old_path}"], capture_output=True, text=True, check=True).stdout
        new = Path(new_path).read_text(encoding="utf-8")
        a, b = comments(old_path, old), comments(new_path, new)
        if len(a) != len(b):
            unpaired.append(f"{new_path}: {len(a)} comments at {base}, {len(b)} now")
            continue
        out, i = [], 0
        for (s0, e0), (s1, e1) in zip(a, b):
            out += [new[i:s1], old[s0:e0]]
            i = e1
        text = "".join(out) + new[i:]
        if text != new:
            Path(new_path).write_text(text, encoding="utf-8")
            restored += 1
            print(f"{new_path}")
    if unpaired:
        raise SystemExit("comments that don't pair, to restore by hand:\n  " + "\n  ".join(unpaired))
    print(f"{restored} files' comments restored", file=sys.stderr)


def pins() -> None:
    """Each pin the part left stale, by its own command (`tools/check/pins.py`). One whose command builds is refused, for
    its run on a node."""
    sys.path.insert(0, "tools/check")
    import pins as P
    root = Path(".")
    if built := [pin.path for pin, _ in P.stale(root) if pin.builds]:
        raise SystemExit(f"stale pins whose command builds, to regenerate on a node: {', '.join(built)}")
    if still := P.regen(root):
        raise SystemExit("still stale: " + ", ".join(pin.path for pin, _ in still))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", choices=sorted(MAPS), help="write the part's map, tools/move/lean_names_pass_PART_map.toml")
    ap.add_argument("--apply", choices=["docket", "comments", "pins"], help="edit the tree for the part")
    ap.add_argument("--base", help="with --apply comments: the commit the part started from")
    ap.add_argument("--moves", action="store_true", help="print docket's declaration renames as JSON")
    a = ap.parse_args()
    if a.map:
        out = Path(f"tools/move/lean_names_pass_{a.map}_map.toml")
        out.write_text(MAPS[a.map])
        print(out)
    elif a.apply == "comments":
        if not a.base:
            ap.error("--apply comments needs --base")
        restore(a.base)
    elif a.apply:
        {"docket": docket, "pins": pins}[a.apply]()
    elif a.moves:
        print(json.dumps(DOCKET_MOVES, indent=1))
    else:
        ap.error("one of --map, --apply or --moves")


if __name__ == "__main__":
    main()
