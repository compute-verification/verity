#!/usr/bin/env python3
"""Daily cleanup census (TOP-223, Daniel's Sunday goal): per owning handle, the counts the playbook's slice-clean goals
drive to zero, read from a checkout of main.

usage: census.py TREE OUT.json [--prs PRS.json]
  TREE: a checkout (worktree of origin/main); PRS.json: `gh pr list --state open --json number,body,headRefName`.
Prints one row per handle and the totals; OUT.json keeps them for the next day's diff.
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
PRS = Path(sys.argv[sys.argv.index("--prs") + 1]) if "--prs" in sys.argv else None
RESEARCH = str(TREE / "tools/research/src")
SKIP = ("archive/", ".lake/", "GLOSSARY.md", "CHANGELOG.md", "tools/move/*.toml", "tools/move/tests/",
        "tools/research/src/research/store/moves.json", "tools/verity/lean/tests/fixtures/merge/", "tools/verity/check/tests/test_move_check.py",
        "tests/test_retired_terms.py")


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=TREE, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if not p.startswith(SKIP)]


def owners(paths: list[str]) -> dict[str, str]:
    env = dict(os.environ, PYTHONPATH=RESEARCH)
    r = subprocess.run([sys.executable, "-m", "research", "msg", "owners", "--dry-run"], cwd=TREE, env=env,
                       input="\n".join(paths), capture_output=True, text=True, timeout=300)
    m = {}
    for line in r.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            m[parts[1].strip()] = parts[0]
    return m


NEVER_EXPORTED = ("infra/", "tools/research", "tools/agent-guard", ".agents/")


def allowlist(name: str, exported_only: bool = False) -> collections.Counter:
    x = json.loads((TREE / "tests/lint/allowlists" / name).read_text())
    c = collections.Counter()
    for e in x["entries"]:
        if exported_only and e["file"].startswith(NEVER_EXPORTED):
            continue
        c[e["file"]] += int(e.get("count", 1))
    return c


def assumptions() -> collections.Counter:
    c = collections.Counter()
    for lock in TREE.glob("**/lean-audit.json"):
        rel = lock.relative_to(TREE).as_posix()
        if rel.startswith(SKIP) or "/.lake/" in rel:
            continue
        for v in json.loads(lock.read_text()).get("assumptions", {}).values():
            c[(v.get("owner") if isinstance(v, dict) else None) or "-"] += 1
    return c


def retired_identifiers() -> list[str]:
    third_party = ("CUDA's", "vLLM's", "Triton's", "BLAKE3's", "Succinct's", "upstream")
    ids = set()
    for line in (TREE / "GLOSSARY.md").read_text().splitlines():
        if not line.startswith('| "'):
            continue
        old = line.split("|")[1]
        if any(t in old for t in third_party) or re.search(r"until a format version|keeps? the name|stays|cites", line):
            continue
        for tok in re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", old):
            tok = tok.split(".")[-1]
            if len(tok) >= 6 and ("_" in tok or re.search(r"[a-z][A-Z]", tok)):
                ids.add(tok)
    return sorted(ids)


def grep_counts(words: list[str]) -> collections.Counter:
    c = collections.Counter()
    if not words:
        return c
    args = ["git", "grep", "-c", "-w", "-I"]
    for w in words:
        args += ["-e", w]
    r = subprocess.run(args + ["--", "."] + [f":!{s}" for s in SKIP], cwd=TREE, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        f, _, n = line.rpartition(":")
        c[f] += int(n)
    return c


def dead_code() -> collections.Counter:
    py = [p for p in tracked() if p.endswith(".py") and "/tests/" not in p and not p.startswith("tests/")]
    r = subprocess.run(["uvx", "--quiet", "vulture", "--min-confidence", "80", *py], cwd=TREE, capture_output=True,
                       text=True, timeout=900)
    c = collections.Counter()
    for line in r.stdout.splitlines():
        if "unused variable" not in line:   # a stand-in's parameters mirror the schema it stands in for
            c[line.split(":", 1)[0]] += 1
    return c


def prs_by_lane() -> collections.Counter:
    c = collections.Counter()
    if not PRS:
        return c
    for pr in json.loads(PRS.read_text()):
        m = re.search(r"^Lane\s+@?([a-z0-9-]+)", pr.get("body") or "", re.M)
        lane = m.group(1) if m else (re.match(r"cursor/(?:verity-)?([a-z]+(?:-[a-z]+)?)", pr["headRefName"]) or [None, "-"])[1]
        c["@" + lane if not lane.startswith("@") else lane] += 1
    return c


files = {"lane_text": allowlist("repo_lane_text.json", exported_only=True), "ruff": allowlist("repo_ruff.json"),
         "silent_except": allowlist("repo_silent_excepts.json"), "retired_ids": grep_counts(retired_identifiers()),
         "dead_code": dead_code()}
paths = sorted({f for c in files.values() for f in c})
own = owners(paths)
rows: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
for metric, c in files.items():
    for f, n in c.items():
        rows[own.get(f, "-")][metric] += n
for h, n in assumptions().items():
    rows[h]["assumptions"] += n
leads = sorted((h for h in rows if h.startswith("@")), key=len, reverse=True)
alias = {"@net": "@network-accounting"}
for h, n in prs_by_lane().items():
    h = alias.get(h) or next((l for l in leads if h == l or h.startswith(l + "-")), h)
    rows[h]["open_prs"] += n
rows["(repo)"]["protocol_md"] = sum(1 for p in tracked() if p.endswith("PROTOCOL.md"))

cols = ["open_prs", "assumptions", "lane_text", "retired_ids", "dead_code", "silent_except", "ruff"]
prev = json.loads(OUT.read_text()) if OUT.exists() else {}
print(f"{'handle':22}" + "".join(f"{c:>14}" for c in cols))
tot = collections.Counter()
for h in sorted(rows, key=lambda h: -sum(rows[h][c] for c in cols)):
    print(f"{h:22}" + "".join(f"{rows[h][c]:>14}" for c in cols))
    tot.update({c: rows[h][c] for c in cols})
print(f"{'TOTAL':22}" + "".join(f"{tot[c]:>14}" for c in cols))
print("protocol_md", rows["(repo)"]["protocol_md"])
OUT.write_text(json.dumps({"rows": {h: dict(v) for h, v in rows.items()}, "total": dict(tot), "prev_total": prev.get("total")},
                          indent=1))
