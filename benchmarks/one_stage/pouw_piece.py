"""PoUW's piece of the composed checks (`composite.py --cases`): its case and its answer on it, in one directory.

    pouw_piece.py v1 --out DIR [--exe PATH] (--params P --registration R --answer A | --serve-run SERVE_DIR --window K)
    pouw_piece.py period --out DIR [--exe PATH] (--key F --ledger F --pins F --main F --name NAME --windows F
                                                 | --serve-run SERVE_DIR --window K)

``v1`` is remote's PoUW (`verity/pouw/v1`, `pouw-check`, `PouwCheck/Protocol.lean`): its params the verifier's
`Params` (``--params``), its commitment the window's registration, one-stage's record (``--registration``, written as
canonical JSON, whose digest the proof service's answer cites), its evidence the service's answer for it (``--answer``).
``period`` is onsite's (`verity/pouw/period/v1`, `pouw-period`, `PouwCheck/Period.lean`): its params ``{key, main,
name}`` (the challenger key it pins, ``--key``, as `ChallengerKey.to_json` writes it or as the deployment's key file
holds it; its copy of main's first-parent history, ``--main``, one commit a line; the period, ``--name``), its
commitment the auditor's ledger (``--ledger``, its lines), its evidence ``{pins, windows}`` (the host's pins log,
``--pins``, its lines; each window's evidence in the close's order, ``--windows``, a JSON list in `credit.PROVED_KEYS`'s
form).

The directory, private as `composite.py` reads it (0700, its files 0600: the evidence and params are witness), gets
``params.json``, ``commitment.bin`` and ``evidence.json`` (canonical), ``case.json`` (``{params, commitment, evidence,
blobs}``, the commitment and evidence hex, no blobs: neither protocol reads any) and ``answer.json``: the line ``EXE
check params.json commitment.bin evidence.json`` prints, the Lean's (``--exe``, its package's ``pouw-check`` or
``pouw-period``), or without ``--exe`` the Python reference's (`core_check.case`, `period_check.case`, which the
difftests hold to the Lean). Prints one ``PIECE`` line and exits 0 when the answer accepts, 1 when it rejects, 2 when
it is undecided, as the check does.

``--serve-run SERVE_DIR --window K`` takes the inputs from a served run (`served_verdicts.sh`'s serve dir, or its
``out/serve``), and names its piece MISSING, with why, wherever the run doesn't hold them (`v1_missing`,
`period_missing`): the directory then holds only ``missing.json`` saying why, no case and no answer, so `composite.py`
leaves the protocol out and Lean's verdict on the product names it missing. It prints one ``MISSING`` line and exits
2: nothing was decided.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from composite import write_private

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import period_check as PC

V1, PERIOD = CC.ID, PC.ID
CASE = {V1: CC.case, PERIOD: PC.case}
EXIT = {"accept": 0, "reject": 1, "undecided": 2}
CASE_FILES = ("params.json", "commitment.bin", "evidence.json", "case.json", "answer.json")


class Missing(Exception):
    """Why a served run holds no piece of the protocol."""


def canonical(x: Any) -> bytes:
    return CC._canon(x).encode()


def lines(path: Path) -> list[str]:
    return [x for x in path.read_text().split("\n") if x]


def challenger_key(doc: dict) -> dict:
    """``{H, W, public}`` (`ChallengerKey.to_json`) from itself or from a deployment's key file (``pubkey``)."""
    return {"H": doc["H"], "W": doc["W"], "public": doc["public"] if "public" in doc else doc["pubkey"]}


def answer(protocol: str, out: Path, exe: str | None) -> dict:
    """The protocol's answer on the case in `out`: the Lean's when `exe` is given, else the Python reference's."""
    if exe is None:
        case = json.loads((out / "case.json").read_text())
        return CASE[protocol]({k: case[k] for k in ("params", "commitment", "evidence")})
    p = subprocess.run([exe, "check", *(str(out / f) for f in ("params.json", "commitment.bin", "evidence.json"))],
                       capture_output=True, text=True, check=False)
    said = p.stdout.strip().splitlines()
    if p.returncode not in EXIT.values() or not said:
        raise RuntimeError(f"{exe} check exited {p.returncode}: {(p.stdout + p.stderr).strip()[-1000:]}")
    return json.loads(said[-1])


def write(out: Path, protocol: str, params: Any, commitment: bytes, evidence: bytes, exe: str | None = None) -> dict:
    """The piece in `out` (module docstring) and its answer."""
    (out / "missing.json").unlink(missing_ok=True)
    case = {"blobs": [], "commitment": commitment.hex(), "evidence": evidence.hex(), "params": params}
    write_private(out, {"params.json": canonical(params), "commitment.bin": commitment, "evidence.json": evidence,
                        "case.json": (json.dumps(case, sort_keys=True) + "\n").encode()})
    a = answer(protocol, out, exe)
    if a.get("protocol") != protocol:
        raise RuntimeError(f"the answer is {a.get('protocol')}'s, not {protocol}'s")
    write_private(out, {"answer.json": canonical(a) + b"\n"})
    return a


def missing(out: Path, protocol: str, why: str) -> None:
    """A missing piece: no case and no answer, only ``missing.json`` saying why."""
    for f in CASE_FILES:
        (out / f).unlink(missing_ok=True)
    write_private(out, {"missing.json": (json.dumps({"protocol": protocol, "why": why}, sort_keys=True) + "\n").encode()})


def _window(serve_dir: Path, k: int) -> dict:
    path = serve_dir / "windows.jsonl"
    if not path.is_file():
        raise Missing(f"{serve_dir} has no windows.jsonl, so no window {k}")
    line = next((w for w in map(json.loads, lines(path)) if int(w["window"]) == k), None)
    if line is None:
        raise Missing(f"{serve_dir}'s windows.jsonl has no window {k}")
    return line


def v1_missing(serve_dir: Path, k: int) -> str:
    """Why served window `k` has no `verity/pouw/v1` case: v1 checks a tile window the proof service registered and
    answered, and a served vLLM window is H1's, or serves no PoUW."""
    try:
        h1 = _window(serve_dir, k).get("pouw_h1")
    except Missing as e:
        return str(e)
    if not h1:
        return f"window {k} served no PoUW (its line has no pouw_h1: served without --protocols)"
    return (f"window {k}'s PoUW is H1's per-call credit ({h1.get('entry')}), which {V1} doesn't check: {V1} is the tile "
            "window's check, over a one-stage registration of tiles PouwCheck.tiles pins (the Pearl-C tiles at sm120, "
            "PouwH1Tile_v1 at K = 64) and the proof service's answer to it, and an H1 window has neither; an H1 window's "
            f"Lean check is {PERIOD}, onsite's")


def period_missing(serve_dir: Path, k: int) -> str:
    """Why served window `k` has no `verity/pouw/period/v1` case: the first of its period, its record and its drawn
    units' proof outcomes it lacks."""
    try:
        h1 = _window(serve_dir, k).get("pouw_h1")
    except Missing as e:
        return str(e)
    if not h1:
        return f"window {k} served no PoUW (its line has no pouw_h1: served without --protocols)"
    session = h1.get("session") or {}
    if session.get("period") is None:
        return (f"window {k}'s session {session.get('session')} is in no period of the auditor's ledger (reserved with "
                f"no period open: a development window), and {PERIOD} credits only a period's windows, from the "
                "period-open that commits their K, rows and weights")
    if not (serve_dir / f"h1/w{k}/pouw-h1-calls.json").is_file():
        return f"window {k} has no H1 record (h1/w{k}/pouw-h1-calls.json) in {serve_dir}"
    return (f"no proof outcomes of window {k}'s drawn units and live rounds: {PERIOD} reads each one's verdict of the "
            "verifier of record (credit.PROVED_KEYS's proofs and live), and the served chain proves none of them")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("protocol", choices=["v1", "period"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exe", help="the protocol's Lean executable (pouw-check, pouw-period); else the Python reference")
    ap.add_argument("--serve-run", type=Path)
    ap.add_argument("--window", type=int)
    for f in ("params", "registration", "answer", "key", "ledger", "pins", "main", "windows"):
        ap.add_argument(f"--{f}", type=Path)
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    protocol = V1 if a.protocol == "v1" else PERIOD
    given = (["params", "registration", "answer"] if protocol == V1
             else ["key", "ledger", "pins", "main", "name", "windows"])
    if a.serve_run is not None:
        if a.window is None:
            ap.error("--serve-run takes --window")
        why = (v1_missing if protocol == V1 else period_missing)(a.serve_run, a.window)
        missing(a.out, protocol, why)
        print(f"MISSING {protocol}: {why}", flush=True)
        return EXIT["undecided"]
    if all(getattr(a, f) is not None for f in given):
        if protocol == V1:
            params = json.loads(a.params.read_text())
            commitment = canonical(json.loads(a.registration.read_text()))
            evidence = canonical(json.loads(a.answer.read_text()))
        else:
            params = {"key": challenger_key(json.loads(a.key.read_text())), "main": lines(a.main), "name": a.name}
            commitment = canonical(lines(a.ledger))
            evidence = canonical({"pins": lines(a.pins), "windows": json.loads(a.windows.read_text())})
        got = write(a.out, protocol, params, commitment, evidence, a.exe)
    else:
        ap.error(f"{a.protocol} takes --serve-run and --window, or {', '.join('--' + f for f in given)}")
    print("PIECE " + json.dumps({"protocol": protocol, "verdict": got["verdict"], "reason": (got["reason"] or "")[:600],
                                 "commitment": got["commitment"], "by": "lean" if a.exe else "python",
                                 "dir": str(a.out)}, sort_keys=True), flush=True)
    return EXIT[got["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
