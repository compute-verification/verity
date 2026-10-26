"""The runner's output tie (``--tie``, `verity_all_protocols.runner`): each period, the output the network certifier
relayed in the period's link-window, which it committed itself as the loop's registered value ``out`` and whose head it
signed with its session's own key at the link-window's close (`certifier_serve.py --out-program`,
`out_commit.commit_out` under its running `Issuer`); the verifier's pin of its root from that record alone, under the
certifier key and session the run's deployment names (`out_commit.check_out`); and the loop's opening of it
(`loop.out_opening`) from the witness the certifier hands the prover (its rows and their salts).

A period's rows are its requests' responses as the egress carried them, in the plan's order: response j's step s word
n (u16, little-endian) is row ``(j·steps + s)·h + n``, one word, which the loop's output unit (request j, step s, last
layer, n) reads at its ``o0`` (`TorchLoop.circuit`); the prover checks the certifier's rows are those (`out_rows`
over the communication leg's `relayed`). The verifier derives the value's domain from its own registration of the
reading program (`pin`: the circuit's program digest and the loop's shape's ``out`` rows), so a record of another
program, value or row count is refused (``out-domain``), as is one of another window, key or session. The output
units are the circuit's uncut registered reads of an output port, the class the integrity audit proves whole beside its
draw (`STRATUM`).
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.certifier import out_commit as OC
from verity.core.protocols.certifier.certificate import PublicKey
from verity.core.protocols.sampled_proofs.one_stage import registered as REG

from .table import load

ROOT = Path(__file__).resolve().parents[3]
TRACES = ROOT / "benchmarks" / "network_traces"
VALUE, PORT = OC.VALUE, "o0"
#: the stratum the tie rests on: every unit with an uncut registered read of an output port (`records_audit.output_class`,
#: a5's ``output_closure``), proved whole beside the draw, rate 1
STRATUM = "output class"


def _loop():
    sys.path.insert(0, str(ROOT / "integrations" / "torch_loop"))
    from verity_torch_loop import loop as L

    return L


def _cs():
    sys.path.insert(0, str(TRACES))
    import certifier_serve as cs

    return cs


def out_rows(responses: Sequence[bytes], h: int, steps: int) -> list[list[int]]:
    """The value's rows (module docstring) from each request's response, ``steps`` steps of 2H bytes."""
    if any(len(r) != 2 * h * steps for r in responses):
        raise ValueError(f"a response is {2 * h * steps} bytes: {steps} steps of {h} u16 words")
    return [[int(w)] for r in responses for w in np.frombuffer(r, "<u2")]


def device(run: Path) -> tuple[PublicKey, bytes]:
    """The network certifier's key and certifier session as the relay run's ``deployment.json`` names them, the ones the
    communication leg checks the session's certificates under."""
    dep = json.loads((run / "deployment.json").read_text())
    return _cs().read_public(dep["certifier_key"]), bytes.fromhex(dep["session"])


def certified_out(run: Path, window: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Link-window ``window``'s out record (public, the verifier's) and its witness (``rows``, ``salts``: the prover's)
    as the network certifier wrote them in the relay run ``run``."""
    cs = _cs()
    records = [json.loads(x) for x in (run / cs.OUT_RECORDS).read_text().splitlines()]
    if [OC.LINK, OC.DIRECTION.value, window] in [r.get("refused") for r in records]:
        raise RuntimeError(f"the network certifier's signer refused link-window {window}'s out certificate")
    mine = [r for r in records if r.get("window") == window]
    witness = [w for w in map(json.loads, (run / cs.OUT_WITNESS).read_text().splitlines()) if w["window"] == window]
    if len(mine) != 1 or len(witness) != 1:
        raise RuntimeError(f"{run}: {len(mine)} out records and {len(witness)} witnesses of link-window {window}, not one")
    return mine[0], witness[0]


def pin(record: Mapping[str, Any], key: PublicKey, session: bytes, *, program: str, rows: int,
        window: int) -> dict[str, Any]:
    """The verifier's pin of ``out`` (records_audit's ``--registered`` form) from the network certifier's record alone:
    the root `check_out` returns for link-window ``window`` under the certifier's ``key`` in ``session`` and the domain
    the verifier derives for ``rows`` one-word rows of the reading ``program``; `OutRefused` names the check that
    fails."""
    want = REG.domain(program, VALUE, 1, rows, REG.WORD_BITS)
    root = OC.check_out(record, key, session, want, window=window)
    return {"program": program, "values": {VALUE: {"value": VALUE, "root": root, "leaves": rows, "words": 1,
                                                   **REG.width_entry(REG.WORD_BITS), "domain": want.domain_id.hex()}}}


def relay(scratch: Path, tokens: Sequence[int], steps: int, periods: int, tools: Mapping[str, str], tampered: bool,
          program: str) -> dict[str, Any]:
    """The communication leg's session relayed before the loop serves (`relay`, in its scratch directory, where the leg
    finds it), the network certifier committing each link-window's output for the reading ``program``; ``tampered``:
    some period's output is the tampered relay's, which must be ``byte``'s."""
    leg = load("certifier")
    rec = leg.relay(scratch / leg.SESSION, list(tokens), steps, _loop().H, periods, tools, out_program=program)
    if tampered and rec["tamper"] != "byte":
        raise ValueError(f"--tie-tamper ties a period to the CERTIFIER_TAMPER=byte relay, not {rec['tamper']!r}")
    return rec


def period(rec: Mapping[str, Any], index: int, strategy: str, program: str, rows: int, scratch: Path,
           out: Path) -> dict[str, Any]:
    """Period ``index``'s tie: the ``strategy`` relay's out record of the period's link-window, copied to
    ``out/out-record.json``, the verifier's pin of it, and the loop's opening from the certifier's witness in
    ``scratch``. -> what ``period.json``'s ``tie`` names, with the ``opening`` and the ``pin``."""
    window, responses = load("certifier").relayed(rec, strategy, index)
    run = Path(rec["dir"]) / "rehearsal" / strategy
    record, witness = certified_out(run, window)
    if witness["rows"] != out_rows(responses, rec["h"], rec["steps"]):
        raise RuntimeError(f"the network certifier's out rows of link-window {window} aren't its responses' words")
    pinned = pin(record, *device(run), program=program, rows=rows, window=window)
    scratch.mkdir(parents=True, exist_ok=True)
    opening = scratch / f"p{index}-out-opening.json"
    opening.write_text(json.dumps(_loop().out_opening(program, witness["rows"],
                                                      [bytes.fromhex(s) for s in witness["salts"]])))
    out.mkdir(parents=True, exist_ok=True)
    (out / "out-record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return {"value": VALUE, "port": PORT, "stratum": STRATUM, "strategy": strategy, "window": window,
            "root": pinned["values"][VALUE]["root"], "record": str(out / "out-record.json"), "opening": opening,
            "pin": pinned}


__all__ = ["PORT", "STRATUM", "VALUE", "certified_out", "device", "out_rows", "period", "pin", "relay"]
