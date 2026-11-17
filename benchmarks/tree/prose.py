"""The prose in the tracked Python and Lean at HEAD, as `top/prose-share` (the percent of lines that are docstrings or
comments and no code), `top/prose-over-length` (prose items longer than AGENTS.md's "Comments and docstrings" allows)
and `top/prose-memory` (prose items naming a date, an id, a commit, a machine or a person), each for every lead's files
(`lint_sites.py`'s owner) and then in total, last, since a reader keying by name alone takes an id's last-written value;
a file that does not parse makes every reading `partial`. Run it as `research run --on auto --cwd source -- uv run
python benchmarks/tree/prose.py`; it writes under `--out` (`out/` outside a research run).
"""
from __future__ import annotations

import argparse
import ast
import bisect
import inspect
import io
import json
import re
import subprocess
import tokenize
from collections import Counter
from dataclasses import dataclass

from lint_sites import REGISTRY, ROOT, UNOWNED, owner
from verity_bench import kit

SHARE, OVER, MEMORY = "top/prose-share", "top/prose-over-length", "top/prose-memory"
SKIP = re.compile(r"(?:^|/)(?:archive|\.lake|fixtures?|testdata|golden|snapshots|vendor|vendored|third_party|upstream)/")
GENERATED = re.compile(r"@generated|do not edit|auto-?generated|this file is generated", re.I)
DIRECTIVE = re.compile(r"^#!|^--\s*shake:|\bcoding[:=]|SPDX-License-Identifier|\bCopyright\b|\bLicen[cs]ed?\b|\bnoqa\b|"
                       r"\bnosec\b|^#\s*(?:type|pragma|fmt|ruff|isort|mypy|pylint|pyright|shellcheck)\b")
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
         r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
TEAMS = "BENCH|CIR|CI|CMP|COMM|CON|DAN|HASH|INF|LAND|LEAN|MEM|NET|PRF|TOP|TRN|VLLM"
MEMORY_KINDS = {
    "date": re.compile(rf"\b20[12]\d-[01]\d-[0-3]\d\b|\b[0-3]?\d {MONTH}\b(?! \d+ bits)|\b{MONTH} [0-3]?\d(?:st|nd|rd|th)?\b"),
    "issue": re.compile(rf"\b(?:{TEAMS})-\d{{1,5}}\b"),
    "pull request": re.compile(r"\b(?:PR|pull request)s? ?#?\d{3,6}\b|(?<![\w&/#])#\d{4,6}\b"),
    "run": re.compile(r"\br20\d{6}-\d{6}-[0-9a-f]{4}\b"),
    "artifact or note": re.compile(r"\bart:[0-9a-z]|\bnote:\d"),
    "commit": re.compile(r"\b(?:commit|upstream|at)\s+`?[0-9a-f]{7,40}\b"),
    "machine": re.compile(r"\bvy-(?:nebius(?:-cpu)?|challenger|cpu)-\d+\b|\b[Nn]ode ?[12]'s\b|\b(?:on|from|to) [Nn]ode ?[12]\b"),
    "person": re.compile(r"\b(?:Daniel|Douglas|Luke)\b"),
}
ABBR, SENTENCE = re.compile(r"\b(e\.g|i\.e|etc|vs|cf|al|resp)\."), re.compile(r"[.!?](?=\s+[A-Z`(]|\s*$)")
OVER_KINDS = ("file", "declaration", "comment")


LANGS = ("Python", "Lean")


@dataclass
class Item:
    kind: str
    line: int
    body: list[str]
    exempt: bool = False
    reads: bool = False
    help: bool = False


def trim(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def over_length(item: Item) -> bool:
    if item.exempt:
        return False
    if item.help:
        return "" in (s.strip() for s in item.body)
    if item.kind == "file":
        return len(SENTENCE.findall(ABBR.sub(r"\1", " ".join(s.strip() for s in item.body)))) > 1
    return len(item.body) > (3 if item.kind == "declaration" else 2)


def python_prose(src: str) -> tuple[int, set[int], list[Item]]:
    tree = ast.parse(src)
    lines, prose, items = src.splitlines(), set(), []
    reads_doc = any(isinstance(x, ast.Name) and x.id == "__doc__" for x in ast.walk(tree))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) or not node.body:
            continue
        doc = node.body[0]
        if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str):
            prose.update(range(doc.lineno, doc.end_lineno + 1))
            body = trim(inspect.cleandoc(doc.value.value).splitlines())
            module = isinstance(node, ast.Module)
            items.append(Item("file" if module else "declaration", doc.lineno, body, help=module and reads_doc))
    run: list[tokenize.TokenInfo] = []
    for tok in [*(t for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.COMMENT), None]:
        whole = tok is not None and not lines[tok.start[0] - 1][:tok.start[1]].strip()
        if run and not (whole and tok.start == (run[-1].start[0] + 1, run[-1].start[1])
                        and not DIRECTIVE.search(tok.string)):
            items.append(Item("comment", run[0].start[0], trim([t.string[1:].removeprefix(" ") for t in run])))
            run = []
        if tok is None or DIRECTIVE.search(tok.string):
            continue
        if whole:
            run.append(tok)
            prose.add(tok.start[0])
        else:
            items.append(Item("trailing", tok.start[0], [tok.string[1:].strip()]))
    return len(lines), prose, items


IDENT = re.compile(r"[\w'!?\u2080-\u209c\u1d62-\u1d6a\u2c7c]")
CHAR = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'")
NEST, OPENER = re.compile(r"/-|-/"), re.compile("--|/-|[\"'\u00ab]")


def lean_spans(src: str) -> list[tuple[str, int, int]]:
    """(kind, start, end) of each comment and string literal: line, block, doc, module or string. It refuses a block
    comment that never closes."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(src)

    def string_end(j: int, interp: bool) -> int:
        j += 1
        while j < n:
            if src[j] == "\\":
                j += 2
            elif src[j] == '"':
                return j + 1
            elif interp and src[j] == "{":
                j = brace_end(j + 1)
            else:
                j += 1
        return n

    def brace_end(j: int) -> int:
        depth = 1
        while j < n:
            if src[j] == '"':
                j = string_end(j, src[j - 1] == "!")
                continue
            depth += {"{": 1, "}": -1}.get(src[j], 0)
            j += 1
            if not depth:
                return j
        return n

    while i < n:
        c = src[i]
        if src.startswith("--", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(("line", i, j))
        elif src.startswith("/-", i):
            kind = "doc" if src.startswith("/--", i) and not src.startswith("/--/", i) else (
                "module" if src.startswith("/-!", i) else "block")
            depth, j = 1, i + 2
            while depth and (m := NEST.search(src, j)):
                depth += 1 if m.group() == "/-" else -1
                j = m.end()
            if depth:
                raise ValueError(f"a block comment at offset {i} never closes")
            out.append((kind, i, j))
        elif c == '"':
            if raw := re.search(r"(?<![\w'!?])r(#*)$", src[max(0, i - 9):i]):
                end = src.find('"' + raw.group(1), i + 1)
                j = n if end < 0 else end + 1 + len(raw.group(1))
            else:
                j = string_end(i, src[i - 1:i] == "!")
            out.append(("string", i, j))
        elif c == "'" and not (i and IDENT.match(src[i - 1])) and (m := CHAR.match(src, i)):
            out.append(("string", i, j := m.end()))
        elif c == "\u00ab":
            j = src.find("\u00bb", i)
            j = n if j < 0 else j + 1
        else:
            j = m.start() if (m := OPENER.search(src, i + 1)) else n
        i = j
    return out


DECL = re.compile(r"\s*(?:@\[(?:[^\[\]]|\[[^\]]*\])*\]\s*)*(?:(?:private|protected|noncomputable|partial|unsafe|nonrec|"
                  r"scoped|local)\s+)*(?:theorem|lemma|def|abbrev|structure|class(?:\s+inductive)?|inductive|instance|"
                  r"axiom|opaque)\s+([^\s:({\[]+)")
SCOPE = re.compile(r"^[ \t]*(namespace|section|noncomputable section|end|mutual)\b[ \t]*([\w.'\u00ab\u00bb]*)", re.M)


def listed(name: str, names: set[str], tails: set[str]) -> bool:
    parts = name.split(".")
    return name in names or any(".".join(parts[k:]) in names for k in range(1, len(parts) - 1)) or (
        "." in name and name in tails)


def lean_prose(src: str, audit: tuple[set[str], set[str]], reads: tuple[set[str], set[str]]) -> tuple[int, set[int], list[Item]]:
    spans = [s for s in lean_spans(src) if s[0] != "string" and not (s[0] == "line" and DIRECTIVE.search(src[s[1]:s[2]]))]
    code = list(src)
    for _, a, b in spans:
        code[a:b] = [ch if ch == "\n" else " " for ch in src[a:b]]
    code, starts, lines = "".join(code), [0], src.split("\n")
    for ln in lines:
        starts.append(starts[-1] + len(ln) + 1)
    prose = {k + 1 for k, (s, c) in enumerate(zip(lines, code.split("\n"))) if s.strip() and not c.strip()}
    for kind, a, b in spans:
        if kind != "line":
            prose.update(range(bisect.bisect_right(starts, a) + 1, bisect.bisect_right(starts, b - 1)))
    scopes, stack, items, run, first = list(SCOPE.finditer(code)), [], [], [], True
    for kind, a, b in [*spans, ("end", len(src), len(src))]:
        la = bisect.bisect_right(starts, a)
        whole = kind == "line" and not code[starts[la - 1]:a].strip()
        if run and not (whole and (la, a - starts[la - 1]) == (run[-1][0] + 1, run[-1][1])):
            items.append(Item("comment", run[0][0], trim([lines[k - 1].strip()[2:].removeprefix(" ") for k, _ in run])))
            run = []
        if kind == "end":
            break
        if kind == "line" and whole:
            run.append((la, a - starts[la - 1]))
            continue
        if kind == "line":
            items.append(Item("trailing", la, [src[a + 2:b].strip()]))
            continue
        body = trim(re.sub(r"^/-[-!]?|-/$", "", src[a:b]).splitlines())
        if kind == "doc":
            while scopes and scopes[0].start() < a:
                s = scopes.pop(0)
                if s.group(1) != "end":
                    stack.append(s.group(2) if s.group(1) == "namespace" else "")
                elif stack:
                    stack.pop()
            name = m.group(1) if (m := DECL.match(code, b)) else ""
            if name and not name.startswith("_root_."):
                name = ".".join([*filter(None, stack), name])
            name = name.removeprefix("_root_.")
            items.append(Item("declaration", la, body, exempt=bool(name) and listed(name, *audit),
                              reads=bool(name) and listed(name, *reads)))
        else:
            items.append(Item("file" if kind == "module" and first else "comment", la, body))
            first = first and kind != "module"
    return len(src.splitlines()), prose, items


def lock_names(root=ROOT) -> tuple[set[str], set[str]]:
    """(the Lean names the locks list, the definitions their `reads` lists), from every tracked `lean-audit.json` outside
    a fixtures directory."""
    listed_, reads = set(), set()
    for rel in subprocess.run(["git", "ls-files", "*lean-audit.json"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.split():
        if SKIP.search(rel):
            continue
        d = json.loads((root / rel).read_text(encoding="utf-8"))
        for key in ("assumptions", "domain", "open", "guarantees", "ids", "trust_gap", "witnesses", "unwitnessed"):
            listed_ |= set(d.get(key) or {}) if isinstance(d.get(key), dict) else set()
        listed_ |= {v for v in d.get("deployed") or [] if isinstance(v, str)}
        listed_ |= {k for v in (d.get("trust") or {}).values() if isinstance(v, dict) for k in v}
        listed_ |= {p.strip() for g in (d.get("guarantees") or {}).values() if isinstance(g, dict)
                    for sig in g.get("signature") or [] for p in str(sig).split(" : ")}
        reads |= {k for m in (d.get("reads") or {}).values() if isinstance(m, dict) for k in m.get("definitions") or {}}
    return {n for n in listed_ if n and " " not in n}, reads


def tails(names: set[str]) -> tuple[set[str], set[str]]:
    return names, {".".join(p[k:]) for p in (n.split(".") for n in names) for k in range(1, len(p))}


def count(path: str, src: str, audit, reads) -> Counter:
    lang = "Python" if path.endswith(".py") else "Lean"
    n, prose, items = python_prose(src) if lang == "Python" else lean_prose(src, audit, reads)
    c = Counter({("lines", lang): n, ("prose", lang): len(prose)})
    for it in items:
        if over_length(it):
            c.update({("over", it.kind, lang): 1, OVER: 1, "reads": it.reads})
        hits = [k for k, rx in MEMORY_KINDS.items() if rx.search(" ".join(it.body))]
        c.update({MEMORY: bool(hits), **{("memory", k): 1 for k in hits}})
    return c


def tracked(root=ROOT) -> dict[str, bytes]:
    """Each tracked `.py` and `.lean` blob at HEAD that is counted, by path."""
    tree = subprocess.run(["git", "ls-tree", "-r", "-z", "HEAD"], cwd=root, capture_output=True, check=True).stdout
    keep = [(meta.split()[2], path.decode()) for meta, _, path in (e.partition(b"\t") for e in tree.split(b"\0") if e)
            if meta.split()[1] == b"blob" and path.endswith((b".py", b".lean")) and not SKIP.search(path.decode())]
    blobs = subprocess.run(["git", "cat-file", "--batch"], cwd=root, input=b"".join(s + b"\n" for s, _ in keep),
                           capture_output=True, check=True).stdout
    out, i = {}, 0
    for _, path in keep:
        nl = blobs.index(b"\n", i)
        size = int(blobs[i:nl].split()[2])
        out[path], i = blobs[nl + 1:nl + 1 + size], nl + 2 + size
    return out


def measure(files: dict[str, bytes], handles: list[dict], audit, reads) -> tuple[dict[str | None, Counter], list[str]]:
    members = [*sorted({h["handle"] for h in handles if h.get("owns")} | {UNOWNED}), None]
    sums, broken = {m: Counter() for m in members}, []
    for path, raw in sorted(files.items()):
        try:
            src = raw.decode("utf-8")
            doc = src.find("/-!") if path.endswith(".lean") else -1
            if GENERATED.search(src[doc:src.find("-/", doc)] if doc >= 0 else "\n".join(src.splitlines()[:5])):
                continue
            c = count(path, src, audit, reads)
        except (UnicodeDecodeError, SyntaxError, ValueError, tokenize.TokenError) as e:
            broken.append(f"{path} ({type(e).__name__})")
            continue
        sums[owner(handles, path)].update(c)
        sums[None].update(c)
    return sums, broken


def share(c: Counter, langs: tuple[str, ...] = LANGS) -> float | None:
    lines = sum(c["lines", lang] for lang in langs)
    return round(100 * sum(c["prose", lang] for lang in langs) / lines, 2) if lines else None


def detail(t: Counter, broken: list[str]) -> str:
    over = "; ".join(f"{k} " + ", ".join(f"{lang} {t['over', k, lang]}" for lang in LANGS) for k in OVER_KINDS)
    text = ("prose share " + ", ".join(f"{lang} {share(t, (lang,))}% of {t['lines', lang]} lines" for lang in LANGS)
            + f". Over length: {over}; of the Lean declarations over length, {t['reads']} are definitions a lock's "
            "reads lists. Memory, by kind (an item may name several): "
            + ", ".join(f"{k} {t['memory', k]}" for k in MEMORY_KINDS))
    return text + (f". Not parsed, so not counted: {', '.join(broken)}" if broken else "")


def write(w: kit.Writer, sums: dict[str | None, Counter], broken: list[str]) -> None:
    inputs, mark = [f"not parsed: {b}" for b in broken], "partial" if broken else "measured"
    for member, c in sums.items():
        value = share(c)
        w.measurement(SHARE, value, "percent", target=10, inputs=inputs, mark=mark, member=member,
                      why=None if value is not None else "no Python or Lean file is this lead's")
        w.measurement(OVER, c[OVER], "items", target=0, inputs=inputs, mark=mark, member=member)
        w.measurement(MEMORY, c[MEMORY], "items", target=0, inputs=inputs, mark=mark, member=member)
    w.validation("passed", "benchmarks/tree/prose.py", detail(sums[None], broken))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args().out
    audit, reads = lock_names()
    sums, broken = measure(tracked(), json.loads(REGISTRY.read_text(encoding="utf-8"))["handles"], tails(audit), tails(reads))
    write(kit.writer(out), sums, broken)


if __name__ == "__main__":
    main()
