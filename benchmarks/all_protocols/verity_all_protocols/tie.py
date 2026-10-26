"""The runner's output tie (``--tie``, `verity_all_protocols.runner`): each period, the output the network certifier
relayed in the period's link-window committed as the loop's registered value ``out`` and its head signed
(`out_commit.commit_out`), the verifier's pin of its root from that record alone (`out_commit.check_out`), and the
loop's opening of it (`loop.out_opening`: the rows and the certifier's salts, the prover's).

A period's rows are its requests' responses as the egress carried them (the communication leg's `relayed`), in the
plan's order: response j's step s word n (u16, little-endian) is row ``(j·steps + s)·h + n``, one word, which the loop's
output unit (request j, step s, last layer, n) reads at its ``o0`` (`TorchLoop.circuit`). The verifier derives the
value's domain from its own registration of the reading program (`pin`: the circuit's program digest and the loop's
shape's ``out`` rows), so a record of another program, value or row count is refused (``out-domain``), as is one of
another window, key or session. The output units are the circuit's uncut registered reads of an output port, the class
the integrity audit proves whole beside its draw (`STRATUM`). The key (`certifier`) is an LMS key the runner makes for
the network certifier, in a certifier session of its own (the communication leg's ``_OUT_KEY_GAP``).
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.certifier import out_commit as OC
from verity.core.protocols.certifier.certificate import Issuer, LmsKey, LmsSigner
from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.crypto import lms

from .table import load

ROOT = Path(__file__).resolve().parents[3]
VALUE, PORT = OC.VALUE, "o0"
#: the stratum the tie rests on: every unit with an uncut registered read of an output port (`records_audit.output_class`,
#: a5's ``output_closure``), proved whole beside the draw, rate 1
STRATUM = "output class"
#: the runner's network certifier key: 2^6 out certificates, one a period
LMS = lms.Params(64, 1, 6)


def _loop():
    sys.path.insert(0, str(ROOT / "integrations" / "torch_loop"))
    from verity_torch_loop import loop as L

    return L


@dataclass(frozen=True)
class Certifier:
    """The network certifier's key as the runner holds it: its issuer in its session, and the public key the verifier
    holds."""

    issuer: Issuer
    key: LmsKey
    session: bytes


def certifier(directory: Path) -> Certifier:
    """A fresh key and certifier session, its leaf state in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    signer = LmsSigner.create(lms.keygen(LMS, os.urandom(LMS.N)), directory / "signer-state.json")
    session = os.urandom(32)
    return Certifier(Issuer(signer, session), signer.public_key(), session)


def out_rows(responses: Sequence[bytes], h: int, steps: int) -> list[list[int]]:
    """The value's rows (module docstring) from each request's response, ``steps`` steps of 2H bytes."""
    if any(len(r) != 2 * h * steps for r in responses):
        raise ValueError(f"a response is {2 * h * steps} bytes: {steps} steps of {h} u16 words")
    return [[int(w)] for r in responses for w in np.frombuffer(r, "<u2")]


def pin(record: Mapping[str, Any], c: Certifier, *, program: str, rows: int, window: int) -> dict[str, Any]:
    """The verifier's pin of ``out`` (records_audit's ``--registered`` form) from the certifier's record alone: the root
    `check_out` returns for link-window ``window`` under the domain the verifier derives for ``rows`` one-word rows of
    the reading ``program``; `OutRefused` names the check that fails."""
    want = REG.domain(program, VALUE, 1, rows, REG.WORD_BITS)
    root = OC.check_out(record, c.key, c.session, want, window=window)
    return {"program": program, "values": {VALUE: {"value": VALUE, "root": root, "leaves": rows, "words": 1,
                                                   **REG.width_entry(REG.WORD_BITS), "domain": want.domain_id.hex()}}}


def relay(scratch: Path, tokens: Sequence[int], steps: int, periods: int, tools: Mapping[str, str],
          tampered: bool) -> dict[str, Any]:
    """The communication leg's session relayed before the loop serves (`relay`, in its scratch directory, where the leg
    finds it); ``tampered``: some period's output is the tampered relay's, which must be ``byte``'s."""
    leg = load("certifier")
    rec = leg.relay(scratch / leg.SESSION, list(tokens), steps, _loop().H, periods, tools)
    if tampered and rec["tamper"] != "byte":
        raise ValueError(f"--tie-tamper ties a period to the CERTIFIER_TAMPER=byte relay, not {rec['tamper']!r}")
    return rec


def period(c: Certifier, rec: Mapping[str, Any], index: int, strategy: str, program: str, rows: int, scratch: Path,
           out: Path) -> dict[str, Any]:
    """Period ``index``'s tie: the ``strategy`` relay's output in its link-window committed by the certifier, its record
    in ``out/out-record.json``, the loop's opening in ``scratch`` and the verifier's pin of it. -> what ``period.json``'s
    ``tie`` names, with the ``opening`` and the ``pin``."""
    window, responses = load("certifier").relayed(rec, strategy, index)
    values = out_rows(responses, rec["h"], rec["steps"])
    got = OC.commit_out(values, program=program, words=1, key=c.issuer, session=c.session, window=window)
    scratch.mkdir(parents=True, exist_ok=True)
    opening = scratch / f"p{index}-out-opening.json"
    opening.write_text(json.dumps(_loop().out_opening(program, values, list(got.salts))))
    out.mkdir(parents=True, exist_ok=True)
    (out / "out-record.json").write_text(json.dumps(got.public, indent=1, sort_keys=True))
    pinned = pin(got.public, c, program=program, rows=rows, window=window)
    return {"value": VALUE, "port": PORT, "stratum": STRATUM, "strategy": strategy, "window": window,
            "root": pinned["values"][VALUE]["root"], "record": str(out / "out-record.json"), "opening": opening,
            "pin": pinned}


__all__ = ["LMS", "PORT", "STRATUM", "VALUE", "Certifier", "certifier", "out_rows", "period", "pin", "relay"]
