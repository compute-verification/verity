"""The torch loop's registered compiler: its schedule, the advice, to its circuit; the command `records_audit.py` runs.

    python -m verity_torch_loop.compile [--law L] [--cap N] [--registered] --advice FILE --out FILE
    python -m verity_torch_loop.compile --identity

The advice is the loop's schedule (`schedule_bytes`: its requests' token ids and its decode steps), the run-time choice its
circuit depends on: one unit per output coordinate of each linear at each request and step. The command writes the circuit
record of a window of that circuit under ``--law`` in members of at most ``--cap`` units, its run fields empty, prints
``{"bits": N}`` (the advice's length, which is charged) and exits 0; on advice that is not a schedule it exits `REFUSED`,
the reason on stderr (the contract is `benchmarks/prove_path/records_audit.py`'s). ``--identity`` prints the name, digest
and commit it is registered under (`compiler`), and ``--table ARGV...`` the verifier's table entry for it running as ARGV
(`table`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs.one_stage import draw as D
from verity.ml.windows import interface as IF

from . import loop as L

NAME = "torch-loop/compile/v0"
#: the exit code for advice it refuses (`records_audit.REFUSED`)
REFUSED = 2
#: the repository paths whose code fixes the map from schedule to circuit, which the compiler's digest is over: this package
#: and Verity, whose interface and circuits it builds the record with
CODE = ("integrations/torch_loop/verity_torch_loop", "verity")
#: the repository paths the command imports from, its ``PYTHONPATH`` in the verifier's checkout of its commit
PYTHONPATH = (".", "integrations/torch_loop")
REPO = Path(__file__).resolve().parents[3]


def compiler() -> dict[str, str]:
    """The name, digest and commit the loop's registration binds its compiler under: the digest over `CODE` in the work tree
    this module runs from (`interface.compiler_digest`), at the commit it is at, where a verifier fetches that code."""
    return {"name": NAME, "digest": IF.compiler_digest(NAME, REPO, CODE), "commit": IF.commit_of(REPO)}


def table(argv: list[str]) -> dict[str, Any]:
    """The verifier's table of compilers holding this one, run as ``argv`` from a checkout of its commit (records_audit's
    ``--compilers``)."""
    return {NAME: {"digest": compiler()["digest"], "argv": list(argv), "code": list(CODE), "pythonpath": list(PYTHONPATH)}}


def schedule_bytes(requests: list[int], steps: int) -> bytes:
    return json.dumps({"requests": list(requests), "steps": int(steps)}, sort_keys=True, separators=(",", ":")).encode()


def schedule(data: bytes) -> tuple[list[int], int]:
    """The requests and steps ``data`` encodes, refused (ValueError) unless it is a schedule in `schedule_bytes`' form."""
    try:
        doc = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"the advice is not a schedule: {e}") from None
    if not (isinstance(doc, dict) and set(doc) == {"requests", "steps"} and isinstance(doc["requests"], list) and doc["requests"]
            and all(type(r) is int and 0 <= r < L.VOCAB for r in doc["requests"]) and type(doc["steps"]) is int
            and doc["steps"] >= 1):
        raise ValueError(f"the advice is not a schedule (requests: token ids below {L.VOCAB}, steps: at least 1)")
    if schedule_bytes(doc["requests"], doc["steps"]) != data:
        raise ValueError("the advice is a schedule but not in its one encoding")
    return doc["requests"], doc["steps"]


def circuit_record(requests: list[int], steps: int, law: Any, cap: int, registered: bool = False) -> dict[str, Any]:
    """The circuit record the schedule compiles to: the loop's window under ``law`` and ``cap``, its run fields empty;
    ``registered``: the loop's weights read as registered values."""
    return IF.Window(L.TorchLoop(L.Model(), requests, steps, registered=registered).circuit(), "", law, cap).circuit_record()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--table", nargs=argparse.REMAINDER, help="the verifier's table entry for this compiler run as the rest")
    ap.add_argument("--law", default="subset:2")
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--registered", action="store_true", help="the loop's weights are registered values (`loop.TorchLoop`)")
    ap.add_argument("--advice")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.identity:
        print(json.dumps(compiler()))
        return 0
    if a.table:
        print(json.dumps(table(a.table)))
        return 0
    if not (a.advice and a.out):
        ap.error("--advice FILE --out FILE, --identity or --table ARGV...")
    data = Path(a.advice).read_bytes()
    try:
        requests, steps = schedule(data)
    except ValueError as e:
        print(f"verity_torch_loop.compile: {e}", file=sys.stderr)
        return REFUSED
    Path(a.out).write_bytes(IF.circuit_bytes(circuit_record(requests, steps, D.parse_law(a.law), a.cap, a.registered)))
    print(json.dumps({"bits": 8 * len(data)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
