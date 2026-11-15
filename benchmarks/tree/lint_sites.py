"""The ratchet lints' sites, as numbers: what each lint finds in the checkout it runs in and what its committed list
allows, one line a lint and a file from `lint_counts.py` beside this, summed as `top/lint-sites-found` and
`top/lint-sites-allowed`. Each is written as the total and once for each lead, the handle whose longest `owns` entry in
`.agents/skills/using-slack/registry.json` covers the file, as its `member`; a file no entry covers is `unowned`'s.
Lower is better; the tree holds no site a lint finds at zero.

    research run --on auto --cwd source -- uv run python benchmarks/tree/lint_sites.py

A list `lint_counts.py` couldn't count is left out of both sums, which are then `partial`, and the validation's detail
names it with its reason beside each counted list's totals; where it counted none, nothing is written. It counts the
checkout's HEAD and writes under `--out` (`out/` outside a research run).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from verity_bench import kit

ROOT = Path(__file__).resolve().parents[2]
COUNTS = Path(__file__).resolve().with_name("lint_counts.py")
REGISTRY = ROOT / ".agents/skills/using-slack/registry.json"
UNOWNED = "unowned"
FOUND, ALLOWED = "top/lint-sites-found", "top/lint-sites-allowed"


def parse(text: str) -> tuple[list[tuple[str, str, int, int]], list[str]]:
    """(each counted line as (lint, file, found, allowed), the lists not counted) from `lint_counts.py`'s stdout."""
    rows, uncounted = [], []
    for line in text.splitlines():
        lint, file, found, allowed = line.split("\t")
        if (file, found, allowed) == ("-", "-", "-"):
            uncounted.append(lint)
        else:
            rows.append((lint, file, int(found), int(allowed)))
    return rows, uncounted


def owner(handles: list[dict], path: str) -> str:
    """The handle whose `owns` entry is `path`'s longest prefix (a directory ends in `/`), or `UNOWNED`."""
    best = max(((p, h["handle"]) for h in handles for p in h.get("owns") or []
                if path == p.rstrip("/") or path.startswith(p if p.endswith("/") else p + "/")),
               key=lambda ph: len(ph[0]), default=None)
    return best[1] if best else UNOWNED


def sums(rows: list[tuple[str, str, int, int]], handles: list[dict]) -> dict[str | None, tuple[int, int]]:
    """{None: the totals, each lead that owns a path and `UNOWNED`: its files' totals}, each (found, allowed)."""
    found, allowed = Counter(), Counter()
    for _, file, f, a in rows:
        who = owner(handles, file)
        found.update({None: f, who: f})
        allowed.update({None: a, who: a})
    members = [None, *sorted({h["handle"] for h in handles if h.get("owns")} | {UNOWNED})]
    return {m: (found[m], allowed[m]) for m in members}


def log(stderr: str) -> tuple[list[str], dict[str, str]]:
    """(the lists counted, {each list not counted: the first line of why}) from `lint_counts.py`'s stderr, which names a
    counted list that found nothing and allows nothing, of which stdout has no line."""
    counted, why = re.findall(r"^(\S+): \d+ found in ", stderr, re.M), {}
    for line in stderr.splitlines():
        name, sep, reason = line.partition(": not counted: ")
        if sep:
            why[name] = reason.strip()
    return counted, why


def detail(rows: list[tuple[str, str, int, int]], uncounted: list[str], stderr: str) -> str:
    found, allowed = Counter(), Counter()
    for lint, _, f, a in rows:
        found[lint] += f
        allowed[lint] += a
    named, why = log(stderr)
    lists = sorted(set(named) | set(found))
    text = f"counted {len(lists)} list(s) at HEAD with lint_counts.py: " + (
        "; ".join(f"{n} {found[n]} found, {allowed[n]} allowed" for n in lists) or "none")
    if uncounted:
        text += ". Not counted, so left out of both sums: " + "; ".join(
            f"{n} ({why.get(n, 'no reason printed')[:200]})" for n in uncounted)
    return text


def write(w: kit.Writer, rows: list[tuple[str, str, int, int]], uncounted: list[str], handles: list[dict],
          stderr: str) -> None:
    if not rows and not log(stderr)[0]:
        raise SystemExit(f"lint_counts.py counted no list, so there is no number to write: {detail(rows, uncounted, stderr)}")
    mark = "partial" if uncounted else "measured"
    for member, (f, a) in sums(rows, handles).items():
        w.measurement(FOUND, f, "sites", target=0, mark=mark, member=member)
        w.measurement(ALLOWED, a, "sites", mark=mark, member=member)
    w.validation("passed", "benchmarks/tree/lint_sites.py", detail(rows, uncounted, stderr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args().out
    r = subprocess.run([sys.executable, str(COUNTS), "HEAD"], cwd=ROOT, capture_output=True, text=True)
    sys.stderr.write(r.stderr)
    if r.returncode not in (0, 1):
        raise SystemExit(f"lint_counts.py ended {r.returncode}: {r.stderr.strip()[-600:]}")
    rows, uncounted = parse(r.stdout)
    handles = json.loads(REGISTRY.read_text(encoding="utf-8"))["handles"]
    write(kit.writer(out), rows, uncounted, handles, r.stderr)


if __name__ == "__main__":
    main()
