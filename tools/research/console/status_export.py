#!/usr/bin/env python3
"""Verity's security status as one canonical document, `verity/status/v0`, for the site's status page: every guarantee a
`lean-audit.json` lists, proved outright or resting on named assumptions, each lock's named assumptions, and the results of
record the tree's data files name. Stdlib only; it reads the tree at one commit through git, never the working files.

    status_export.py --root REPO [--rev REV] [--facts DIR] --out FILE

- `packages`: every lock git tracks at REV (default HEAD) outside a `tests/` directory, by the package's path.
- A guarantee's `assumptions` are those its lock record takes as hypotheses and, with `--facts`, the premises its statement
  takes through its definitions, by the Lean audit's own rule (`premised`). DIR is a Lean audit's output directory
  (`lean-audit/` of a `check` or records run): `audit-<package path, each run of other characters as _>/facts.json`, and
  `proved-in-*/facts.json` beside it for the packages its `proved_in` names. A guarantee has `premises_read` when one of
  those holds it as a theorem whose signature is its lock record's.
- `status`: `assumes` with an assumption, `proved` with none and its premises read, `unknown` with none and its premises
  not read.
- `results`: each `art:` id a table's runs of record name (`RENDERERS`, read as literals, never imported) and each example's
  `[evidence]` (`demos/*/example.toml`); run ids are counted, not exported.

The document is written as `canonical` bytes. Prints one JSON line: the counts, the bytes written and their SHA-256.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

SCHEMA = "verity/status/v0"
LOCK = "lean-audit.json"
ART = re.compile(r"art:[0-9a-f]{64}")
RUN = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
EXAMPLES = "demos/*/example.toml"
#: a table renderer's path -> {the call naming a row's runs of record: how its `what` reads, from the call's first argument}
RENDERERS = {"benchmarks/harness/verity_bench/served_table.py": {"Protocol": "{}", "Breakdown": "{}, where its time goes by span"}}


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def tracked(root: Path, rev: str) -> list[str]:
    return [p for p in git(root, "ls-tree", "-r", "-z", "--name-only", rev).decode().split("\0") if p]


def show(root: Path, rev: str, path: str) -> str:
    return git(root, "show", f"{rev}:{path}").decode()


def canonical(doc: dict) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def lock_paths(files: list[str]) -> list[str]:
    """The locks among ``files``, a test fixture's aside."""
    return sorted(p for p in files if p.rsplit("/", 1)[-1] == LOCK and "/tests/" not in f"/{p}")


def package_of(lock: str) -> str:
    return lock.rsplit("/", 1)[0] if "/" in lock else "."


def facts_sources(facts: Path, package: str) -> list[dict]:
    """The facts documents a Lean audit wrote for ``package`` under ``facts``: its own, then each `proved_in` package's."""
    work = facts / f"audit-{re.sub(r'[^A-Za-z0-9]+', '_', package)}"
    paths = [work / "facts.json", *sorted(work.glob("proved-in-*/facts.json"))]
    return [json.loads(p.read_text()) for p in paths if p.is_file()]


def own_modules(f: dict) -> set[str]:
    """The modules a package of the repository builds, as the audit counts them (check.py's `own_modules`)."""
    if "own_modules" in f:
        return set(f["own_modules"])
    return {m for m, g in (f.get("graph") or {}).items()
            if "/.lake/build/" in (o := str(g.get("olean", ""))) and "/.lake/packages/" not in o}


def unfolding(d: str, unfolds: dict, listed: set[str]) -> tuple[list[str], list[str]] | None:
    """(the ``listed`` assumptions the premise ``d`` comes to, the parts that are none), or None when it doesn't unfold
    (check.py's `unfolding`)."""
    def walk(c: str, path: frozenset) -> tuple[list[str], list[str]] | None:
        parts = unfolds.get(c)
        if not isinstance(parts, list) or c in path:
            return None
        yes, no = set(), set()
        for p in parts:
            h = p.get("head")
            if h is None or p.get("props"):
                no.add(h or "")
            elif h in listed:
                yes.add(h)
            elif (u := walk(h, path | {c})) and not u[1]:
                yes |= set(u[0])
            else:
                no.add(h)
        return sorted(yes), sorted(no)
    return walk(d, frozenset())


def premised(lock: dict, sources: list[dict]) -> dict[str, list[str] | None]:
    """{guarantee: the named assumptions its statement takes through its definitions, or None where ``sources`` don't read
    it}: each premise headed by a constant of an `Assumptions` module or of a module the repository builds, and one that is
    an alias of listed assumptions as those (check.py's `judge` and `assumption_records`)."""
    own = set().union(*map(own_modules, sources)) if sources else set()
    listed = set(lock.get("assumptions") or {})
    heads: dict[str, set[str] | None] = {}
    unfolds: dict = {}
    for name, rec in sorted((lock.get("guarantees") or {}).items()):
        pin = next((f["pins"][name] for f in sources if (f.get("pins") or {}).get(name, {}).get("kind") == "theorem"), None)
        if pin is None or not isinstance(pin.get("premises"), list) or \
                str(pin.get("signature", "")).splitlines() != list(rec.get("signature") or []):
            heads[name] = None
            continue
        heads[name] = set()
        for pr in pin["premises"]:
            unfolds |= pr.get("unfolds") or {}
            if "Assumptions" in (m := pr.get("module") or "").split(".") or m in own:
                heads[name].add(pr["head"])
    out: dict[str, list[str] | None] = {}
    for name, hs in heads.items():
        if hs is None:
            out[name] = None
            continue
        got = set()
        for d in hs:
            u = None if d in listed else unfolding(d, unfolds, listed)
            got |= set(u[0]) if u and not u[1] else {d}
        out[name] = sorted(got)
    return out


def status_of(assumptions: list[str], premises_read: bool) -> str:
    return "assumes" if assumptions else "proved" if premises_read else "unknown"


def lean_status(locks: dict[str, dict], facts: Path | None) -> tuple[list[dict], list[dict], list[dict]]:
    """(packages, guarantees, assumptions) of ``locks`` ({lock path: its document}), the premises from ``facts``."""
    packages, guarantees, assumptions = [], [], []
    for path, lock in sorted(locks.items()):
        pkg = package_of(path)
        gs, named = lock.get("guarantees") or {}, lock.get("assumptions") or {}
        named = named if isinstance(named, dict) else {}
        packages.append({"path": pkg, "guarantees": len(gs), "assumptions": len(named)})
        prem = premised(lock, facts_sources(facts, pkg)) if facts else {}
        for name, rec in gs.items():
            read = prem.get(name)
            got = sorted(set(rec.get("assumptions") or []) | set(read or []))
            guarantees.append({"name": name, "package": pkg, "owner": rec.get("owner") or "",
                               "statement": list(rec.get("signature") or []), "assumptions": got,
                               "premises_read": read is not None, "status": status_of(got, read is not None)})
        assumptions += [{"name": a, "package": pkg, "owner": (r or {}).get("owner") or "", "digest": (r or {}).get("digest")}
                        for a, r in named.items()]
    key = lambda e: (e["package"], e["name"])  # noqa: E731
    return sorted(packages, key=lambda p: p["path"]), sorted(guarantees, key=key), sorted(assumptions, key=key)


def names(v) -> list[str]:
    return [v] if isinstance(v, str) else [x for x in v if isinstance(x, str)] if isinstance(v, list) else []


def renderer_records(text: str, calls: dict[str, str]) -> list[tuple[str, str, str | None]]:
    """(id, what, table) for each run of record a renderer's source names in one of ``calls`` (first argument the row's
    name), the table its `# ` heading, read from literals alone."""
    tree = ast.parse(text)
    table = next((n.value[2:] for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                  and n.value.startswith("# ")), None)
    out = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in calls and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
            continue
        what = calls[n.func.id].format(n.args[0].value)
        for c in ast.walk(n):
            if isinstance(c, ast.Constant) and isinstance(c.value, str) and (ART.fullmatch(c.value) or RUN.fullmatch(c.value)):
                out.append((c.value, what, table))
    return out


def example_records(text: str, name: str) -> list[tuple[str, str, str | None]]:
    """(id, what, None) for each run of record an example's `[evidence]` names: its `runs` and each premise's run."""
    ev = tomllib.loads(text).get("evidence") or {}
    out = [(r, f"the {name} example's run of record", None) for r in names(ev.get("runs"))]
    for premise, v in sorted((ev.get("premises") or {}).items()):
        out += [(r, f"{premise}, measured for the {name} example", None) for r in names(v)]
    return out


def results(root: Path, rev: str, files: list[str]) -> tuple[list[dict], int]:
    """(each result of record by `art:` id, the run ids named beside them)."""
    found = []
    for path, calls in sorted(RENDERERS.items()):
        if path in files:
            found += renderer_records(show(root, rev, path), calls)
    for path in sorted(p for p in files if fnmatch.fnmatch(p, EXAMPLES)):
        found += example_records(show(root, rev, path), path.split("/")[1])
    arts = {(i, what, table) for i, what, table in found if ART.fullmatch(i)}
    out = [{"art": i, "kind": None, "what": what, "table": table} for i, what, table in sorted(arts, key=lambda t: (t[0], t[1], t[2] or ""))]
    return out, len({i for i, _, _ in found if RUN.fullmatch(i)})


def export(root: Path, rev: str = "HEAD", facts: Path | None = None) -> tuple[dict, int]:
    """(the status document of ``root`` at ``rev``, the run ids of record it leaves out)."""
    commit = git(root, "rev-parse", f"{rev}^{{commit}}").decode().strip()
    files = tracked(root, commit)
    locks = {p: json.loads(show(root, commit, p)) for p in lock_paths(files)}
    packages, guarantees, assumptions = lean_status(locks, facts)
    res, runs = results(root, commit, files)
    return {"schema": SCHEMA, "commit": commit, "packages": packages, "guarantees": guarantees,
            "assumptions": assumptions, "results": res}, runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="the repository")
    ap.add_argument("--rev", default="HEAD", help="the commit whose tree is read (default HEAD)")
    ap.add_argument("--facts", type=Path, help="a Lean audit's output directory, its audit-*/facts.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.facts and not a.facts.is_dir():
        ap.error(f"--facts {a.facts} is not a directory")
    doc, runs = export(a.root, a.rev, a.facts)
    data = canonical(doc)
    tmp = a.out.with_name(f".{a.out.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, a.out)
    counts = {s: sum(g["status"] == s for g in doc["guarantees"]) for s in ("proved", "assumes", "unknown")}
    print(json.dumps({"out": str(a.out), "commit": doc["commit"], "guarantees": counts,
                      "premises_read": sum(g["premises_read"] for g in doc["guarantees"]),
                      "assumptions": len(doc["assumptions"]), "results": len(doc["results"]), "runs_not_exported": runs,
                      "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
