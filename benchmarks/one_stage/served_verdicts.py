"""The Wed 28 run's verdicts on one served window (`served_verdicts.sh`): the window it audits, the subprotocols after
sampled proofs, and the one record of every verdict (CMP-153's block record).

    served_verdicts.py window SERVE_DIR [--slots N] [--window K]
    served_verdicts.py pieces --serve-run DIR --window K --out DIR
    served_verdicts.py record --serve-run DIR --window K --sampled-proofs DIR --pieces DIR --out DIR [--node NAME]

``window``: the sealed window to audit, from a serve-openai run's ``seals.jsonl``, ``windows.jsonl`` and ``requests.jsonl``
(`pick`): sealed without error, its Commit passed, each of its requests checked and in no other window (V3b's window
circuit refuses a request carried across windows), at most ``--slots`` of them (its class's); the most requests, then the
earliest. ``--window K`` holds K to the same rules. Prints ``{window, requests, users, steps, refused}``, ``steps`` the
window's ``[lo, hi)`` in the run's committed engine steps.

``pieces``: each subprotocol after sampled proofs (`PIECES`: PoUW, the bandwidth audit, PoUS, the network certifier) runs
its piece where one is named (``POUW_CMD``, ``BANDWIDTH_CMD``, ``POUS_CMD``, ``CERTIFIER_CMD``: a command run as
``bash -c CMD _ SERVE_DIR WINDOW DIR``, which writes its core answer to ``DIR/answer.json``), else prints a ``MISSING`` line
naming it and its issue. A piece that ran but wrote no answer is its error; no verdict is made for a piece that didn't run.

``record``: sampled proofs' answer from the records audit `prove_path/run.sh` (``PP_RECORDS=1``) kept under
``--sampled-proofs`` (its ``NAME`` dir: ``summary.json``, ``window/audit.json``, ``windows/rank0/registration.json``), read
as the all-protocols integrity step reads it (`legs/sampled_proofs.answer`); each other subprotocol's from
``--pieces/<component>/answer.json``, else a ``missing`` row naming its issue. They go into the all-protocols window report
(``verdicts.json``, `runner.row_of`'s rows) and from it the block record (``block-record.json``, `block_record.from_period`,
held by `block_record.check`). `block_record` carries only an accepted subprotocol's certificate, so where one rejects or
errs the composed verdict is ``reject`` and no block record is written. Prints one ``RECORD`` line.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "benchmarks" / "all_protocols", ROOT / "benchmarks" / "e2e_demo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

#: each subprotocol after sampled proofs, by its runner name: (its name, its component, its piece's owner, the issue its piece
#: is, what the piece does), in the chain's order
PIECES: dict[str, tuple[str, str, str, str, str]] = {
    "work": ("PoUW", "pouw", "vllm-serve-arms", "VLLM-33 G1",
             ("H1 credit on the served window; its coins are in-process until vy-challenger-1's gate takes this node's "
              "key (--challenger-gate, --challenger-peer vy-dc-h200; INF-263, Daniel's card due Tue 27 Oct 15:00Z)")),
    "bandwidth": ("the bandwidth audit", "bandwidth", "vllm-serve-arms", "VLLM-33 G2",
                  ("the remote bandwidth audit of the served window, against the bandwidth verifier on vy-auditor-2 at "
                   "10.96.0.36:7952 (INF-263, due Tue 27 Oct 18:00Z)")),
    "space": ("PoUS", "pous", "memory-accounting", "MEM-94",
              ("which weights PoUS covers for the served vLLM window, and a PoUS path over vLLM's weights (the vLLM "
               "adapter refuses int8 today)")),
    "communication": ("the network certifier", "certifier", "network-accounting", "NET-83",
                      "the network certifier in front of serve-openai (Llama-3.2-1B int8, ASK-122's slots)"),
}
#: the record's subprotocol outside this chain
WIPE = ("the wipe", "wipe", "memory-accounting", None, "no step of this chain runs the wipe")
NAMES = {"integrity": "sampled proofs", **{k: v[0] for k, v in PIECES.items()}, "wipe": WIPE[0]}


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _refusal(k: int, seal: Mapping[str, Any], line: Mapping[str, Any] | None, mine: Sequence[Mapping[str, Any]],
             slots: int) -> str | None:
    """Why window ``k`` can't be audited, None when it can."""
    if "error" in seal:
        return f"its seal failed: {seal['error']}"[:300]
    if not line or not line.get("passed"):
        return "its Commit did not pass"
    if not mine:
        return "no request"
    carried = sorted(r.get("user") or r.get("engine_id") for r in mine if r.get("windows") != [k])
    if carried:
        return f"requests carried across windows: {carried}"
    unchecked = sorted(r.get("user") or r.get("engine_id") for r in mine if not (r.get("check") or {}).get("passed"))
    if unchecked:
        return f"requests whose check failed: {unchecked}"
    if len(seal.get("requests") or ()) != len(mine):
        return f"{len(seal.get('requests') or ())} sealed requests, {len(mine)} with a record"
    if len(mine) > slots:
        return f"{len(mine)} requests, more than the class's {slots} slots"
    return None


def pick(serve_dir: Path, slots: int, want: int | None = None) -> dict[str, Any]:
    """The window to audit (module docstring); ValueError when none qualifies, or ``want`` doesn't."""
    seals = {int(s["window"]): s for s in _lines(serve_dir / "seals.jsonl")}
    lines = {int(w["window"]): w for w in _lines(serve_dir / "windows.jsonl")}
    reqs = _lines(serve_dir / "requests.jsonl")
    refused: dict[int, str] = {}
    ok: list[tuple[int, list[dict[str, Any]]]] = []
    for k in sorted(seals):
        mine = [r for r in reqs if k in (r.get("windows") or ())]
        why = _refusal(k, seals[k], lines.get(k), mine, slots)
        if why is None:
            ok.append((k, mine))
        else:
            refused[k] = why
    if want is not None:
        if want not in seals:
            raise ValueError(f"window {want} has no seal (sealed: {sorted(seals)})")
        if want in refused:
            raise ValueError(f"window {want}: {refused[want]}")
        ok = [x for x in ok if x[0] == want]
    if not ok:
        raise ValueError(f"no sealed window to audit: {json.dumps(refused, sort_keys=True)}")
    k, mine = max(ok, key=lambda x: (len(x[1]), -x[0]))
    lo = sum(int((w.get("commit") or {}).get("steps") or 0) for j, w in lines.items() if j < k)
    hi = lo + int((lines[k].get("commit") or {}).get("steps") or 0)
    return {"window": k, "requests": len(mine), "users": sorted(r.get("user") or "" for r in mine), "steps": [lo, hi],
            "refused": {str(j): why for j, why in refused.items()}}


def pieces(serve_dir: Path, window: int, out: Path, env: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Each piece of `PIECES` run where ``<COMPONENT>_CMD`` names it, else named missing; one line each on stdout."""
    env = dict(os.environ if env is None else env)
    got: dict[str, dict[str, Any]] = {}
    for name, comp, owner, issue, what in PIECES.values():
        cmd = env.get(f"{comp.upper()}_CMD")
        if not cmd:
            got[comp] = {"verdict": "missing", "issue": issue, "owner": owner}
            print(f"MISSING {name} ({issue}, {owner}): {what}", flush=True)
            continue
        d = out / comp
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "piece.log", "w") as log:
            rc = subprocess.run(["bash", "-c", cmd, "_", str(serve_dir), str(window), str(d)], stdout=log,
                                stderr=subprocess.STDOUT, env=env, check=False).returncode
        try:
            a = json.loads((d / "answer.json").read_text())
            got[comp] = {"verdict": a.get("verdict"), "reason": a.get("reason"), "rc": rc}
        except (OSError, ValueError) as e:
            got[comp] = {"verdict": "error", "reason": f"its piece wrote no answer (exit {rc}): {e}", "rc": rc}
        print(f"{name.upper()} {json.dumps(got[comp], sort_keys=True)}", flush=True)
    return got


def sampled_proofs_row(audit_dir: Path) -> dict[str, Any]:
    """Sampled proofs' runner row from the records audit kept under ``audit_dir`` (module docstring)."""
    from verity_all_protocols import runner, table

    entry = next(e for e in table.TABLE if e.name == "integrity")
    reg = audit_dir / "windows" / "rank0" / "registration.json"
    try:
        a = table.load(entry.component).answer(audit_dir, json.loads(reg.read_text()) if reg.exists() else None)
        return runner.row_of(entry, a, audit_dir)
    except (RuntimeError, ValueError) as e:
        return {"verdict": "error", "owner": entry.owner, "reason": f"{type(e).__name__}: {e}"[:2000]}


def piece_row(key: str, pieces_dir: Path) -> dict[str, Any]:
    """A subprotocol's runner row from its piece's answer, or ``missing`` with its issue."""
    from verity_all_protocols import runner, table

    _name, comp, owner, issue, what = PIECES.get(key, WIPE)
    f = pieces_dir / comp / "answer.json"
    if not f.exists():
        return {"verdict": "missing", "owner": owner, "issue": issue, "why": what}
    try:
        return runner.row_of(table.Entry(key, comp, owner), json.loads(f.read_text()), f.parent)
    except ValueError as e:
        return {"verdict": "error", "owner": owner, "reason": f"{type(e).__name__}: {e}"[:2000]}


def record(rows: Mapping[str, Mapping[str, Any]], *, window: int, steps: tuple[int, int], node: str,
           run: str | None) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """The window report over ``rows`` (by runner name), its block record (None when a subprotocol rejected, erred or was
    undecided), and the ``RECORD`` line's summary."""
    from verity_all_protocols import runner
    from verity_e2e_demo import block_record as BR

    report = {"format": runner.FORMAT, "window": window, "node": node, "run": run, "protocols": dict(rows)}
    verdicts = {NAMES[k]: r.get("verdict") for k, r in rows.items()}
    missing = {NAMES[k]: r.get("issue") for k, r in rows.items() if r.get("verdict") == "missing"}
    bad = sorted(NAMES[k] for k, r in rows.items() if r.get("verdict") not in ("accept", "missing"))
    if bad:
        composed = "reject" if any(rows[k].get("verdict") in ("reject", "error") for k in rows) else "pending"
        line = {"composed": composed, "by": bad, "verdicts": verdicts, "missing": missing, "block_record": None,
                "why": "block_record carries an accepted subprotocol's certificate only"}
        return report, None, line
    rec = BR.from_period(report, steps=steps, deployment=BR.REFERENCE)
    line = {"composed": rec["composed"]["verdict"], "verdicts": verdicts, "missing": missing,
            "record_missing": rec["composed"]["missing"], "check": BR.check(rec), "steps": list(steps), "node": node,
            "deployment": rec["monitoring_period"]["deployment"]["id"]}
    return report, rec, line


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window")
    w.add_argument("serve_dir", type=Path)
    w.add_argument("--slots", type=int, default=8)
    w.add_argument("--window", type=int)
    p = sub.add_parser("pieces")
    r = sub.add_parser("record")
    for q in (p, r):
        q.add_argument("--serve-run", type=Path, required=True)
        q.add_argument("--window", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    r.add_argument("--sampled-proofs", type=Path, required=True)
    r.add_argument("--pieces", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--node", default=socket.gethostname())
    a = ap.parse_args(argv)
    if a.cmd == "window":
        try:
            print(json.dumps(pick(a.serve_dir, a.slots, a.window), sort_keys=True))
        except ValueError as e:
            print(f"served_verdicts.py window: {e}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "pieces":
        a.out.mkdir(parents=True, exist_ok=True)
        got = pieces(a.serve_run, a.window, a.out)
        return 0 if all(g["verdict"] in ("accept", "missing") for g in got.values()) else 1
    steps = pick(a.serve_run, 1 << 30, a.window)["steps"]
    rows = {"integrity": sampled_proofs_row(a.sampled_proofs), **{k: piece_row(k, a.pieces) for k in PIECES},
            "wipe": piece_row("wipe", a.pieces)}
    report, rec, line = record(rows, window=a.window, steps=(steps[0], steps[1]), node=a.node,
                               run=os.environ.get("RESEARCH_RUN_ID"))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "verdicts.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    if rec is not None:
        (a.out / "block-record.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    print("RECORD " + json.dumps(line, sort_keys=True), flush=True)
    return 0 if rec is not None and not line["check"] else 1


if __name__ == "__main__":
    sys.exit(main())
