"""Tables and survivor lists from the mutation runs' results: python analyze.py RESULTS.jsonl ... [--pass2 RESULTS2.jsonl ...]
An edit inside a type annotation is EQUIVALENT (the modules import annotations lazily, so it is never evaluated)."""
from __future__ import annotations

import ast
import json
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from mutate import _seg  # noqa: E402


@cache
def annotation_spans(file: str) -> list[tuple[int, int]]:
    src = (ROOT / file).read_text()
    lines = src.splitlines(keepends=True)
    out = []
    for n in ast.walk(ast.parse(src)):
        anns = []
        if isinstance(n, ast.arg) and n.annotation is not None:
            anns.append(n.annotation)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns is not None:
            anns.append(n.returns)
        if isinstance(n, ast.AnnAssign):
            anns.append(n.annotation)
        out += [_seg(lines, a) for a in anns]
    return out


def equivalent(m: dict) -> str | None:
    if any(a <= m["a"] and m["b"] <= b for a, b in annotation_spans(m["file"])):
        return "inside a type annotation, never evaluated"
    return None


def load(paths: list[str]) -> dict[str, dict]:
    out = {}
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            r = json.loads(ln)
            out[r["id"]] = r
    return out


def main(argv: list[str]) -> int:
    p1 = argv[:argv.index("--pass2")] if "--pass2" in argv else argv
    p2 = argv[argv.index("--pass2") + 1:] if "--pass2" in argv else []
    rs, rs2 = load(p1), load(p2)
    for r in rs.values():
        e = equivalent(r)
        if e:
            r["status"], r["why"] = "EQUIVALENT", e
        elif r["id"] in rs2:
            r["pass2"] = rs2[r["id"]]["status"]
    steps: dict[str, dict] = {}
    for r in rs.values():
        s = steps.setdefault(r["step"], {"n": 0, "KILLED": 0, "SURVIVED": 0, "INVALID": 0, "EQUIVALENT": 0, "ERROR": 0,
                                         "k2": 0, "funcs": set(), "tests": set()})
        s["n"] += 1
        s[r["status"]] += 1
        s["k2"] += r.get("pass2") == "KILLED"
        s["funcs"].add(r["func"])
        s["tests"].update(Path(t).name for t in r.get("tests", []))
    print("| step | functions | mutants | killed | survived | equivalent | invalid | kill rate | killed only by slow/Lean tests (pass 2) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, s in steps.items():
        valid = s["KILLED"] + s["SURVIVED"]
        print(f"| {k} | {len(s['funcs'])} | {s['n']} | {s['KILLED']} | {s['SURVIVED']} | {s['EQUIVALENT']} | {s['INVALID']} | "
              f"{100 * s['KILLED'] / max(1, valid):.0f}% | {s['k2']} |")
    print()
    for k, s in steps.items():
        print(f"- {k}: {', '.join(sorted(s['tests']))}")
    print()
    for r in sorted(rs.values(), key=lambda r: r["id"]):
        if r["status"] == "SURVIVED":
            line = (ROOT / r["file"]).read_text().splitlines()[r["line"] - 1].strip()
            p2s = f" pass2={r['pass2']}" if "pass2" in r else ""
            print(f"{r['id']} {Path(r['file']).name}:{r['func']}:{r['line']} [{r['desc']}]{p2s} tests={len(r.get('tests', []))} "
                  f"{r.get('why', '')}\n      {line[:160]}\n      `{r['orig'][:80]}` => `{r['text'][:80]}`")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
