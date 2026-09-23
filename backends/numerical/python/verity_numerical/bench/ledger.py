"""The overhead ledger: one append-only JSONL file per lane under ``backends/numerical/reports/ledger/``, one line per
recorded state of that lane's overhead (``overhead.vs_native_peak``: prover seconds per K=1536 VU over the native
9.85 ps), with the labels every campaign number must carry (``backends/AGENTS.md``).  ``bench.plots`` draws them.

    python -m verity_numerical.bench.ledger add --lane A-gpu --label "first GPU layer" --scope unit \\
        --seconds-per-unit 3.2e-6 --proof-class NON_ZK_PROOF_DIAGNOSTIC --hardware "RTX 4090" --B 4096 \\
        --security 2^-128 --assumptions hash --omitted "chain linkage" --note "..." [--breakthrough] [--run-id ...]
    python -m verity_numerical.bench.ledger check          # validate every ledger file

A lane appends only to its own file (``<lane>.jsonl``), so lanes never conflict on merge.  Scope ``unit`` numbers are
per-transition-unit times scaled by the 96 units of a K=1536 VU (an extrapolation: chain linkage and epilogue are not
in them, and the entry must say so under ``omitted``); scope ``vu`` numbers are measured on whole VUs; scope
``component`` entries are for a kernel or a hash whose overhead is a share, not a prover (``overhead`` may be null and
``note`` says what was measured).  ``breakthrough`` marks an idea-level change (a new protocol trick, a flipped
conclusion), not a tuning step; the plot stars it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

from verity_numerical.bench import contract

__all__ = ["DIR", "SCHEMA", "SCOPES", "TRACKS", "UNITS_PER_VU", "add", "entry", "load", "load_all", "main", "validate"]

SCHEMA = "ledger/v1"
DIR = Path(__file__).resolve().parents[3] / "reports" / "ledger"
SCOPES = ("unit", "vu", "component")
TRACKS = ("A", "B", "C", "shared", "scout", "baseline")
UNITS_PER_VU = 96  # K=1536 / 16 products per transition unit
PROOF_CLASSES = contract.PROOF_CLASSES + ("COMPONENT",)
REQUIRED = ("schema", "ts", "lane", "track", "label", "breakthrough", "overhead", "scope", "proof_class", "authentication",
            "hardware", "B", "K", "security_target", "assumptions", "omitted", "note")


def entry(*, lane: str, track: str, label: str, scope: str, proof_class: str, hardware: str, security_target: str,
          overhead: float | None = None, seconds_per_vu: float | None = None, seconds_per_unit: float | None = None,
          K: int = 1536, B: int | None = None, authentication: str = "excluded", assumptions: list[str] | None = None,
          omitted: list[str] | None = None, note: str = "", breakthrough: bool = False, run_id: str | None = None,
          ts: str | None = None) -> dict[str, Any]:
    """One ledger line.  Exactly one of ``overhead`` / ``seconds_per_vu`` / ``seconds_per_unit`` for a prover number
    (``component`` entries may carry none)."""

    given = [v for v in (overhead, seconds_per_vu, seconds_per_unit) if v is not None]
    if len(given) > 1:
        raise ValueError("give one of overhead, seconds_per_vu, seconds_per_unit")
    if seconds_per_unit is not None:
        seconds_per_vu = seconds_per_unit * UNITS_PER_VU
        if scope != "unit":
            raise ValueError("seconds_per_unit implies scope 'unit'")
    if seconds_per_vu is not None:
        overhead = contract.overhead(seconds_per_vu, K)
    if overhead is None and scope != "component":
        raise ValueError(f"a scope={scope!r} entry needs an overhead (or seconds per VU / unit)")
    e = {"schema": SCHEMA, "ts": ts or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"), "lane": lane, "track": track,
         "label": label, "breakthrough": bool(breakthrough), "overhead": overhead, "seconds_per_vu": seconds_per_vu,
         "scope": scope, "proof_class": proof_class, "authentication": authentication, "hardware": hardware, "B": B, "K": K,
         "security_target": security_target, "assumptions": list(assumptions or []), "omitted": list(omitted or []),
         "run_id": run_id, "note": note}
    problems = validate(e)
    if problems:
        raise ValueError("; ".join(problems))
    return e


def validate(e: Any) -> list[str]:
    if not isinstance(e, dict):
        return ["entry is not an object"]
    p = [f"missing {k}" for k in REQUIRED if k not in e]
    if p:
        return p
    if e["schema"] != SCHEMA:
        p.append(f"schema {e['schema']!r} != {SCHEMA!r}")
    if e["track"] not in TRACKS:
        p.append(f"track {e['track']!r} not in {TRACKS}")
    if e["scope"] not in SCOPES:
        p.append(f"scope {e['scope']!r} not in {SCOPES}")
    if e["proof_class"] not in PROOF_CLASSES:
        p.append(f"proof_class {e['proof_class']!r} not in {PROOF_CLASSES}")
    if e["authentication"] not in ("included", "included-hash", "included-hash-shared", "excluded"):
        p.append("authentication must be included|excluded")
    if e["overhead"] is not None and not (isinstance(e["overhead"], (int, float)) and e["overhead"] > 0):
        p.append("overhead must be a positive number or null")
    if e["overhead"] is None and e["scope"] != "component":
        p.append("only component entries may omit overhead")
    if not e["label"] or len(e["label"]) > 60:
        p.append("label must be 1..60 characters (it is drawn on the plot)")
    if not isinstance(e["assumptions"], list) or not isinstance(e["omitted"], list):
        p.append("assumptions and omitted must be lists of strings")
    if e["proof_class"] in contract.ZK_CLASSES and not e["assumptions"]:
        p.append("a complete backend entry must name its assumptions")
    return p


def add(e: dict[str, Any], directory: Path = DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{e['lane']}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(e, sort_keys=True) + "\n")
    return path


def load(path: Path) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as ex:
            raise ValueError(f"{path}:{i}: {ex}") from None
        bad = validate(e)
        if bad:
            raise ValueError(f"{path}:{i}: " + "; ".join(bad))
        out.append(e)
    return out


def load_all(directory: Path = DIR) -> dict[str, list[dict[str, Any]]]:
    if not directory.exists():
        return {}
    return {p.stem: load(p) for p in sorted(directory.glob("*.jsonl"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--lane", required=True)
    a.add_argument("--track", required=True, choices=TRACKS)
    a.add_argument("--label", required=True)
    a.add_argument("--scope", required=True, choices=SCOPES)
    a.add_argument("--proof-class", required=True, choices=PROOF_CLASSES)
    a.add_argument("--hardware", required=True)
    a.add_argument("--security", required=True, help="e.g. 2^-128, 2^-80, n/a")
    g = a.add_mutually_exclusive_group()
    g.add_argument("--overhead", type=float)
    g.add_argument("--seconds-per-vu", type=float)
    g.add_argument("--seconds-per-unit", type=float)
    a.add_argument("--B", type=int)
    a.add_argument("--K", type=int, default=1536)
    a.add_argument("--authentication", default="excluded", choices=("included", "included-hash", "included-hash-shared", "excluded"))
    a.add_argument("--assumptions", default="", help="comma-separated, e.g. 'hash' or 'hash,LPN-regular-noise,base-OT'")
    a.add_argument("--omitted", default="", help="comma-separated omitted costs/checks")
    a.add_argument("--note", default="")
    a.add_argument("--run-id")
    a.add_argument("--breakthrough", action="store_true")
    a.add_argument("--dir", type=Path, default=DIR)
    c = sub.add_parser("check")
    c.add_argument("--dir", type=Path, default=DIR)
    ns = ap.parse_args(argv)
    if ns.cmd == "check":
        lanes = load_all(ns.dir)
        print(f"ok: {sum(len(v) for v in lanes.values())} entries in {len(lanes)} lanes")
        return 0
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    e = entry(lane=ns.lane, track=ns.track, label=ns.label, scope=ns.scope, proof_class=ns.proof_class, hardware=ns.hardware,
              security_target=ns.security, overhead=ns.overhead, seconds_per_vu=ns.seconds_per_vu,
              seconds_per_unit=ns.seconds_per_unit, K=ns.K, B=ns.B, authentication=ns.authentication,
              assumptions=split(ns.assumptions), omitted=split(ns.omitted), note=ns.note, breakthrough=ns.breakthrough,
              run_id=ns.run_id)
    path = add(e, ns.dir)
    print(f"{path}: overhead={e['overhead']!r} label={e['label']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
