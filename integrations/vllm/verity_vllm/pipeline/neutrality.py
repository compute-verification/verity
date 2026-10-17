"""`verity-vllm neutrality`: rebuild the rows of record's Programs and fail if a Program digest moved.

    verity-vllm neutrality [ROW ...] [--tier fast|heavy|all] [--out DIR] [--compare] [--update] [--pins FILE] [--row-flag=FLAG ...]

A circuit is its per-rank Program digests: the step Program, each request shape's and GP-01's workload Program, as
`epoch.row_digests` reads them from a row's `build_summary.json`.  `manifests/programs_of_record.json` pins them for each row of
record, with the tier of its rebuild and the Build they were taken from.  Each selected row (the ROWs named, else every row of
`--tier`) is rebuilt by its own Build, `verity-vllm row run ROW ROLE REPO REV --stages build` (`epoch.resolve`), into OUT/<row>, and
its digests compared with the pins; `--compare` compares the Builds already under OUT instead.  The manifest digest is reported, not
compared: a naming change moves the manifest and leaves the circuit.  Exit 0 when every pinned digest is reproduced, 1 when one
moved or is missing (each named), 2 on a usage error.  `--update` rewrites the selected rows' pins from what was built: the PR that
changes a circuit on purpose re-pins it, and its diff names what moved.

The rebuild needs the pinned vLLM (`ops/pod_bootstrap.sh --cpu`) and each row's checkpoint metadata (the derive runs on the meta
device and reads no weights), so it runs on a CPU pod, never in `check`; the suite's tier is `tests/program/test_derive_neutrality.py`.
fast: small TP1/TP2 rows, minutes each; heavy: GLM-5.3's 8 ranks, about an hour on a 256-CPU pod.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import ROOT, UsageError, option
from verity_vllm.pipeline import epoch

PINS = ROOT / "manifests" / "programs_of_record.json"
SCHEMA = "verity-vllm/programs-of-record/v1"
TIERS = ("fast", "heavy")


def load(path: Path = PINS) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != SCHEMA:
        raise UsageError(f"{path}: schema {doc.get('schema')!r}, not {SCHEMA}")
    return doc


def select(doc: dict[str, Any], rows: list[str], tier: str) -> list[str]:
    known = doc["rows"]
    unknown = [r for r in rows if r not in known]
    if unknown:
        raise UsageError(f"not a row of record: {', '.join(unknown)} (rows of record: {', '.join(sorted(known))})")
    return rows or sorted(r for r, e in known.items() if tier in ("all", e["tier"]))


def compare(pinned: dict[str, dict], built: dict[str, dict]) -> list[str]:
    """Each pinned Program digest the Build did not reproduce, and each Program it built that is not pinned."""
    short = lambda x: (x or "absent")[:16]                                   # noqa: E731
    out = []
    for r in sorted(set(pinned) | set(built), key=int):
        p, b = pinned.get(r), built.get(r)
        if p is None or b is None:
            out.append(f"rank {r}: {'built, not pinned' if p is None else 'pinned, not built'}")
            continue
        names = ["step", "workload", *sorted(set(p["requests"]) | set(b["requests"]))]
        for n in names:
            want, got = (p[n], b[n]) if n in ("step", "workload") else (p["requests"].get(n), b["requests"].get(n))
            if want != got:
                out.append(f"rank {r} {n}: pinned {short(want)}, built {short(got)}")
    return out


def rebuild(row: str, out: Path, flags: list[str]) -> int:
    """The row's own Build into OUT/<row>, by this interpreter."""
    role, repo, rev, _world = epoch.resolve(row)
    cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "row", "run", row, role, repo, rev, "--stages", "build",
           "--py", sys.executable, "--sweep-dir", str(out), *flags]
    print("[neutrality] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


@dataclass(frozen=True, kw_only=True)
class Options:
    """Rebuild the rows of record's Programs and compare their digests with the pins."""

    rows: list[str] = option("rows", nargs="*", default=[], help="rows of record (default: every row of --tier)")
    tier: str = option("--tier", choices=(*TIERS, "all"), default="fast")
    out: str = option("--out", default="out/neutrality", help="the rebuilt rows live in OUT/<row>")
    compare_only: bool = option("--compare", default=False, action="store_true", help="compare the Builds under OUT, no rebuild")
    update: bool = option("--update", default=False, action="store_true", help="re-pin the selected rows from what was built")
    pins: str = option("--pins", default=str(PINS))
    row_flags: list[str] = option("--row-flag", action="append", default=[], help="passed to `row run` (repeat), e.g. --row-flag=--build-jobs=4")


def main(a: Options) -> int:
    doc = load(Path(a.pins))
    out, report, failed = Path(a.out), {"schema": "verity-vllm/neutrality/v1", "pins": a.pins, "rows": {}}, False
    out.mkdir(parents=True, exist_ok=True)
    for row in select(doc, a.rows, a.tier):
        rc = None if a.compare_only else rebuild(row, out, a.row_flags)
        rd = epoch.row_digests(out / row)
        built, man, pin = rd["programs"], rd["manifest_digest"], doc["rows"][row]
        diffs = compare(pin["programs"], built) if built else ["no build_summary.json: the Build did not finish"]
        report["rows"][row] = {"tier": pin["tier"], "build_rc": rc, "neutral": not diffs, "differences": diffs, "programs": built,
                               "manifest_digest": man, "manifest_digest_pinned": pin.get("manifest_digest")}
        failed |= bool(diffs)
        moved = "" if man == pin.get("manifest_digest") else f"; manifest {str(man)[:16]} (pinned {str(pin.get('manifest_digest'))[:16]})"
        print(f"[neutrality] {row}: {'NEUTRAL' if not diffs else 'MOVED'} -- {len(built)} rank(s){moved}"
              + "".join(f"\n    {d}" for d in diffs), flush=True)
        if a.update and built:
            doc["rows"][row] = {**pin, "programs": built, "manifest_digest": man}
    (out / "neutrality.json").write_text(json.dumps(report, indent=1) + "\n")
    if a.update:
        Path(a.pins).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"[neutrality] re-pinned {', '.join(report['rows'])} in {a.pins}", flush=True)
        return 0
    return 1 if failed else 0
