"""The network period check's driver: Lean's ``network-period check PARAMS COMMITMENT EVIDENCE`` (`NetworkCheck.Period`),
the network attester's verifier of record, run on a case."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from verity.circuits.codec import canonical_json
from verity.commitments.domains import CERTIFIER_PROTOCOL


def lean_check(exe: str, case: Mapping[str, Any], d: Path,
               run: Callable[[list[str]], subprocess.CompletedProcess] | None = None) -> dict:
    """``EXE check`` (``network-period``'s, or ``network-rows``') on the case's three files in ``d``: its answer (its
    last line). ``run`` runs it with its output captured as text (`subprocess.run`'s ``capture_output=True, text=True``
    by default)."""
    d.mkdir(parents=True, exist_ok=True)
    files = [d / "params.json", d / "commitment.bin", d / "evidence.json"]
    files[0].write_bytes(canonical_json(case["params"]))
    files[1].write_bytes(bytes.fromhex(case["commitment"]))
    files[2].write_bytes(bytes.fromhex(case["evidence"]))
    argv = [exe, "check", *map(str, files)]
    p = run(argv) if run else subprocess.run(argv, capture_output=True, text=True)
    if p.returncode not in (0, 1, 2) or not p.stdout.strip():
        raise SystemExit(f"{Path(exe).name} check exited {p.returncode}: {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def answer(exe: str, case: Mapping[str, Any], d: Path,
           run: Callable[[list[str]], subprocess.CompletedProcess] | None = None) -> dict:
    """``network-period`` ``exe``'s answer on ``case`` (`lean_check`, its files in ``d``), unchanged; `ValueError`
    naming the protocol of an answer that is not `CERTIFIER_PROTOCOL`'s."""
    got = lean_check(exe, case, d, run)
    if got.get("protocol") != CERTIFIER_PROTOCOL:
        raise ValueError(f"{Path(exe).name} answered for the protocol {got.get('protocol')!r}, not "
                         f"{CERTIFIER_PROTOCOL}: it is not the network period check")
    return got
