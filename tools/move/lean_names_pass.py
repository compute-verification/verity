#!/usr/bin/env python3
"""Writes `tools/move/lean_names_pass_map.toml`, the Lean names pass: the ontology's names in the Lean (the resource
accounting area and the network attester), the areas that take their protocol's name, and the older words left in Lean
names. Run from the repository root; `tools/move/rename.py --map tools/move/lean_names_pass_map.toml` runs the map.

    python3 tools/move/lean_names_pass.py [--part ontology|areas|retired]...   write the map (by default every part)

The root namespace `Accounting` becomes `ResourceAccounting`, but two other namespaces end in the same word
(`FlockSoundness.Accounting`, `Pous.Accounting`) and a Flock file names its own as `Accounting.X` under
`open FlockSoundness`. So the word is renamed by file: a namespace-stack scan of every `.lean` file resolves each
`Accounting` it spells (`sites`), the files whose every site is the root namespace rename the whole word, and the files
that spell both rename only the root's own second parts. Outside the Lean, a name is the root's only at the start of a
dotted name (`Accounting.` as a namespace word), and module paths are renamed by their tree's prefix.
"""
from __future__ import annotations

import argparse
import collections
import re
import subprocess
from pathlib import Path

MAP = Path("tools/move/lean_names_pass_map.toml")
TREES = ("Definitions", "Properties", "Proofs")
WORD, NEW = "Accounting", "ResourceAccounting"
OTHER_FILES = ["*.json", "*.py", "*.md", "*.toml", "*.sh", "*.txt"]
DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|partial|unsafe|public|meta|nonrec|scoped)\s+)*"
                  r"(?:theorem|lemma|def|abbrev|structure|class|inductive|instance|opaque|axiom|irreducible_def|class\s+inductive)"
                  r"\s+([^\s:({\[]+)")
NS, END = re.compile(r"^\s*namespace\s+(\S+)"), re.compile(r"^\s*end\b\s*(\S*)")
SEC, MUT = re.compile(r"^\s*(?:noncomputable\s+|public\s+)*section\b"), re.compile(r"^\s*mutual\b")
SITE = re.compile(r"(?<![\w.'])" + WORD + r"(?![\w'])((?:\.[\w']+)*)")


def tracked(*globs: str) -> list[str]:
    return subprocess.run(["git", "ls-files", *globs], capture_output=True, text=True, check=True).stdout.split()


def code(src: str) -> list[str]:
    src = re.sub(r"/-.*?-/", lambda m: re.sub(r"[^\n]", " ", m.group()), src, flags=re.S)
    return re.sub(r"--[^\n]*", "", src).split("\n")


def scan(lines: list[str]) -> tuple[list[list[str]], set[str]]:
    """Each line's namespace stack, and the full names the file declares."""
    stack, opened, decls, per = [], [], set(), []
    for line in lines:
        per.append(list(stack))
        if m := NS.match(line):
            stack.extend(m.group(1).split("."))
            opened.append(len(m.group(1).split(".")))
        elif MUT.match(line) or SEC.match(line):
            opened.append(0)
        elif END.match(line):
            if opened and (n := opened.pop()):
                del stack[len(stack) - n:]
        elif m := DECL.match(line):
            name = m.group(1)
            decls.add(name[7:] if name.startswith("_root_.") else ".".join(stack + [name]))
    return per, decls


def sites() -> dict[str, list[tuple[int, str, bool]]]:
    """{file: [(line, the site, whether it is the root namespace)]} for each `Accounting` a `.lean` file spells outside a
    longer dotted name; a site that resolves to a nested namespace, or to none the scan finds, is not the root's."""
    texts = {f: Path(f).read_text(encoding="utf-8", errors="replace") for f in tracked("*.lean")}
    scans = {f: scan(code(t)) for f, t in texts.items() if WORD in t}
    decls = set().union(*(d for _, d in scans.values()))
    spaces = {".".join(d.split(".")[:i]) for d in decls for i in range(1, d.count(".") + 1)}
    out = {}
    for f, (per, _) in scans.items():
        found, lines = [], code(texts[f])
        opened = {n for line in lines if (o := re.match(r"^\s*open\s+(?:scoped\s+)?([^(\n]*)$", line))
                  for n in o.group(1).split() if n != "in"}
        for ln, line in enumerate(lines, 1):
            for m in SITE.finditer(line):
                stack = per[ln - 1]
                if re.match(r"^\s*(namespace|end)\b", line):
                    outer = stack[:len(stack) - len(m.group(0).split("."))] if line.lstrip().startswith("end") else stack
                    found.append((ln, m.group(0), not outer))
                    continue
                first = m.group(1).split(".")[1] if m.group(1) else ""
                hits = [p for i in range(len(stack), -1, -1) if (p := ".".join(stack[:i] + [WORD])) in spaces
                        and (not first or f"{p}.{first}" in spaces or f"{p}.{first}" in decls)]
                hits += [p for o in sorted(opened) if (p := f"{o}.{WORD}") in spaces and p not in hits
                         and (not first or f"{p}.{first}" in spaces or f"{p}.{first}" in decls)]
                found.append((ln, m.group(0), hits == [WORD]))
        if found:
            out[f] = found
    return out


def q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def words_in(files: list[str], words: dict[str, str]) -> list[str]:
    return ["[[words_in]]", "files = [" + ", ".join(q(f) for f in files) + "]", "[words_in.words]",
            *(f"{q(o)} = {q(n)}" for o, n in words.items()), ""]


def compress(files: list[str], others: set[str]) -> list[str]:
    """``files`` as globs: a directory whose every `.lean` file below it is in ``files`` is one `DIR/**` glob."""
    chosen, all_lean = set(files), tracked("*.lean")
    dirs = collections.defaultdict(set)
    for f in all_lean:
        parts = f.split("/")
        for i in range(1, len(parts)):
            dirs["/".join(parts[:i])].add(f)
    out, covered = [], set()
    for d in sorted(dirs, key=lambda d: d.count("/")):
        if any(d.startswith(c + "/") for c in covered):
            continue
        under = dirs[d]
        if under and under <= chosen | (set(all_lean) - chosen - others) and under & chosen and d.count("/") >= 2:
            out.append(d + "/**")
            covered.add(d)
    out += [f for f in files if not any(f.startswith(c + "/") for c in covered)]
    return sorted(out)


def both(globs: list[str]) -> list[str]:
    """Each glob with the area's directory `Accounting` spelled to match `ResourceAccounting` too."""
    return [re.sub(rf"/{WORD}(?=[/.])", "/[AR]*ccounting", g) for g in globs]


def ontology() -> list[str]:
    s = sites()
    pure = sorted(f for f, v in s.items() if all(r for *_, r in v))
    mixed = {f: v for f, v in s.items() if any(r for *_, r in v) and not all(r for *_, r in v)}
    untouched = {f for f, v in s.items() if not any(r for *_, r in v)}
    lines = ["# --- 1. The ontology's names in the Lean.",
             "# The resource accounting layer's area: its modules by their tree's prefix, everywhere.",
             "[words]"]
    for t in TREES:
        for sep in "./":
            lines.append(f"{q(t + sep + WORD)} = {q(t + sep + NEW)}")
    lines += [f"{q('Definitions' + sep + 'Assumptions' + sep + WORD)} = {q('Definitions' + sep + 'Assumptions' + sep + NEW)}"
              for sep in "./"]
    lines += ["# The network attester (GLOSSARY.md: the Lean package `NetworkCertifier` and the names that follow it), with",
              "# the Python names that follow the Lean's (`NetworkCertifierParams`, `NetworkCertifierProfile`).",
              '"NetworkCertifier" = "NetworkAttester"', '"networkCertifier" = "networkAttester"', ""]
    lines += [f"# The root namespace, in the {len(pure)} Lean files whose every `{WORD}` is it (a glob matches the",
              "# area's directory under its old name and its new, so the map reruns as a no-op)."]
    lines += words_in(both(compress(pure, untouched)), {WORD: NEW})
    own = collections.defaultdict(set)
    for f, v in mixed.items():
        for _, site, root in v:
            if root and "." in site:
                own[f].add(".".join(site.split(".")[:2]))
    for f, v in sorted(mixed.items()):
        if own[f]:
            lines += [f"# {f} spells another namespace's `{WORD}` too: only the root's names."]
            lines += words_in(both([f]), {o: NEW + o[len(WORD):] for o in sorted(own[f])})
    lines += [f"# Outside the Lean, the root's names at the start of a dotted name."]
    lines += words_in(OTHER_FILES, {WORD + ".": NEW + "."})
    lines += ["# The network attester in the Lean's prose; a string literal keeps its bytes ([keep]): a verifier's message.",
              "# `network_certifier` stays: in the Lean it names the Python modules, which keep their names."]
    lines += words_in(["*.lean"], {"network certifiers'": "network attesters'", "network certifier's": "network attester's",
                                   "network certifiers": "network attesters", "network certifier": "network attester",
                                   "Network certifier": "Network attester"})
    return lines


def areas() -> list[str]:
    lines = ["# --- 2. The areas that take their protocol's name: registration's files leave the accounting area."]
    rows = {}
    for t in ("Definitions", "Properties"):
        for sep in "./":
            rows[f"{t}{sep}{WORD}{sep}Registration"] = f"{t}{sep}Registration"
    return lines + words_in(["*"], rows)


def retired() -> list[str]:
    lines = ["# --- 3. The older words left in the Lean's names."]
    return lines + words_in(["security/Proofs/Flock/EndToEnd/Period/Union.lean", "security/Proofs/Flock/EndToEnd/**",
                             "security/Proofs/WeightSecurity/Stop.lean"],
                            {"monitoringPeriod_monitoringPeriod": "monitoringPeriod_horizon"})


HEADER = """# The Lean names pass: written by `python3 tools/move/lean_names_pass.py`, run by
# `python3 tools/move/rename.py --map tools/move/lean_names_pass_map.toml`. Edit the script, not this file.
# Declarations keep their place in the locks' records: `rename.py --moves` prints the renames that
# `tools/verity/lean/check.py --update --moved` takes.
"""
TAIL = """[keep_paths]
# PoUS's own `Pous.Accounting` keeps its directories, though some of its files open the root namespace.
"security/*/Pous/Accounting*" = ["Accounting"]

[keep]
patterns = [
    '[\\w.-]+(?:/[\\w.-]+)*/v\\d+(?:\\.\\d+)*\\b',
    '(?:Pous|Flock|Soundness|FlockSoundness)[./]Accounting\\b',
    '"(?:(?!NetworkCertifier)(?:[^"\\\\\\n]|\\\\.|\\\\\\n))*[Nn]etwork certifier(?:(?!NetworkCertifier)(?:[^"\\\\\\n]|\\\\.|\\\\\\n))*"',
    '\\*\\*Accounting\\.\\*\\*',
]

[rewrite]
exclude = ["tools/move/**", "archive/**", "**/CHANGELOG.md", "**/fixtures/**", "fixtures/**", "*.patch",
           "**/TRUSTED.sha256", "uv.lock", "tools/research/src/research/store/**"]
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", action="append", choices=["ontology", "areas", "retired"])
    ap.add_argument("--out", type=Path, default=MAP)
    a = ap.parse_args()
    parts = a.part or ["areas", "ontology", "retired"]
    body = []
    for p in ["areas", "ontology", "retired"]:
        if p in parts:
            body += {"ontology": ontology, "areas": areas, "retired": retired}[p]()
    if "[words]" not in body:
        body = ["[words]", ""] + body
    else:
        i = body.index("[words]")
        j = next(k for k in range(i + 1, len(body)) if body[k] == "" or body[k].startswith("[["))
        words = body[i:j]
        body = words + [""] + body[:i] + body[j:]
    a.out.write_text(HEADER + "\n" + "\n".join(body) + "\n" + TAIL)
    print(a.out)


if __name__ == "__main__":
    main()
