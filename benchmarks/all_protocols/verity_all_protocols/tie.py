"""The runner's output tie (``--tie``, `verity_all_protocols.runner`): each period, the output the network certifier
relayed in the period's link-window, which it committed itself as the loop's registered value ``out`` and whose head it
signed with its session's own key at the link-window's close (`certifier_serve.py --out-shape`,
`out_commit.commit_out` under its running `Issuer`); the verifier's pin of its root from that record alone, under the
certifier key and session the run's deployment names (`out_commit.check_out`); and the loop's opening of it
(`loop.out_opening`) from the witness the certifier hands the prover (its rows and their salts).

Both sides take the value's shape from the reading circuit (``serve --circuit-only``'s ``out_shape``,
`interface.read_shape`, `out_commit.OutShape`): its program digest and ``words`` words a row, the row width of the port
every read of ``out`` names. A period's rows are its requests' responses as the egress carried them, in the plan's
order, ``words`` u16 little-endian words a row: in the per-coordinate circuit one word, response j's step s word n row
``(j·steps + s)·h + n``, which the loop's output unit (request j, step s, last layer, n) reads at its ``o0``
(`TorchLoop.circuit`); in the wired circuit h words, response j's step s row ``j·steps + s``, down_proj unit (s, j)'s
whole ``o0`` (`TorchLoop.wired_circuit`). The prover checks the certifier's rows are those (`out_rows` over the
communication leg's `relayed`). The verifier holds the record to its own shape (`pin`), so a record of another width
is refused (``out-width``), of another row count (``out-rows``), of another program or value (``out-domain``), as is
one of another window, key or session. The output units are the circuit's uncut registered reads of an output port,
the class the integrity audit proves whole beside its draw (`STRATUM`).
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


def shape(doc: Mapping[str, Any] | OC.OutShape) -> OC.OutShape:
    """The out shape ``doc`` (``serve --circuit-only``'s ``out_shape``) read back."""
    return doc if isinstance(doc, OC.OutShape) else OC.OutShape.of(doc)


def out_rows(responses: Sequence[bytes], h: int, steps: int, words: int = 1) -> list[list[int]]:
    """The value's rows (module docstring) from each request's response, ``steps`` steps of 2H bytes, ``words`` words a
    row."""
    if any(len(r) != 2 * h * steps for r in responses):
        raise ValueError(f"a response is {2 * h * steps} bytes: {steps} steps of {h} u16 words")
    if (h * steps) % words:
        raise ValueError(f"a response's {h * steps} words aren't whole rows of {words}")
    flat = [int(w) for r in responses for w in np.frombuffer(r, "<u2")]
    return [flat[i:i + words] for i in range(0, len(flat), words)]


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
    at = [OC.LINK, OC.DIRECTION.value, window]
    if at in [r.get("refused") for r in records]:
        raise RuntimeError(f"the network certifier's signer refused link-window {window}'s out certificate")
    if shapeless := [r for r in records if r.get("out-shape") == at]:
        raise RuntimeError(f"the network certifier committed no out value of link-window {window}: its output isn't "
                           f"the deployment's out shape ({shapeless[0].get('detail')})")
    mine = [r for r in records if r.get("window") == window]
    witness = [w for w in map(json.loads, (run / cs.OUT_WITNESS).read_text().splitlines()) if w["window"] == window]
    if len(mine) != 1 or len(witness) != 1:
        raise RuntimeError(f"{run}: {len(mine)} out records and {len(witness)} witnesses of link-window {window}, not one")
    return mine[0], witness[0]


def pin(record: Mapping[str, Any], key: PublicKey, session: bytes, *, out_shape: Mapping[str, Any] | OC.OutShape,
        window: int) -> dict[str, Any]:
    """The verifier's pin of ``out`` (records_audit's ``--registered`` form) from the network certifier's record alone:
    the root `check_out` returns for link-window ``window`` under the certifier's ``key`` in ``session`` at the
    verifier's ``out_shape`` (its rows of the reading program's registered read, and the domain it derives for them);
    `OutRefused` names the check that fails."""
    want = shape(out_shape)
    root = OC.check_out(record, key, session, want, window=window)
    return {"program": want.program,
            "values": {VALUE: {"value": VALUE, "root": root, "leaves": want.rows, "words": want.words,
                               **REG.width_entry(want.word_bits), "domain": want.domain.domain_id.hex()}}}


def relay(scratch: Path, tokens: Sequence[int], steps: int, periods: int, tools: Mapping[str, str], tampered: bool,
          out_shape: Mapping[str, Any] | OC.OutShape, out: Path | None = None) -> dict[str, Any]:
    """The communication leg's session relayed before the loop serves (`relay`, in its scratch directory, where the leg
    finds it), the network certifier committing each link-window's output at ``out_shape``; ``tampered``: some period's
    output is the tampered relay's, which must be ``byte``'s. A relay step that fails copies its log into the runner's
    ``out``'s session directory, which outlives the scratch (the leg's `failed`)."""
    leg = load("certifier")
    rec = leg.relay(scratch / leg.SESSION, list(tokens), steps, _loop().H, periods, tools,
                    out_shape=shape(out_shape).doc(), keep=None if out is None else out / leg.SESSION)
    if tampered and rec["tamper"] != "byte":
        raise ValueError(f"--tie-tamper ties a period to the CERTIFIER_TAMPER=byte relay, not {rec['tamper']!r}")
    return rec


def period(rec: Mapping[str, Any], index: int, strategy: str, out_shape: Mapping[str, Any] | OC.OutShape,
           scratch: Path, out: Path) -> dict[str, Any]:
    """Period ``index``'s tie: the ``strategy`` relay's out record of the period's link-window, copied to
    ``out/out-record.json``, the verifier's pin of it at its own ``out_shape``, and the loop's opening from the
    certifier's witness in ``scratch``. -> what ``period.json``'s ``tie`` names, with the ``opening`` and the ``pin``."""
    want = shape(out_shape)
    window, responses = load("certifier").relayed(rec, strategy, index)
    run = Path(rec["dir"]) / "rehearsal" / strategy
    record, witness = certified_out(run, window)
    if witness["rows"] != out_rows(responses, rec["h"], rec["steps"], want.words):
        raise RuntimeError(f"the network certifier's out rows of link-window {window} aren't its responses' words")
    pinned = pin(record, *device(run), out_shape=want, window=window)
    scratch.mkdir(parents=True, exist_ok=True)
    opening = scratch / f"p{index}-out-opening.json"
    opening.write_text(json.dumps(_loop().out_opening(want.program, witness["rows"],
                                                      [bytes.fromhex(s) for s in witness["salts"]])))
    out.mkdir(parents=True, exist_ok=True)
    (out / "out-record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    return {"value": VALUE, "port": PORT, "stratum": STRATUM, "strategy": strategy, "window": window,
            "root": pinned["values"][VALUE]["root"], "record": str(out / "out-record.json"), "opening": opening,
            "pin": pinned}


__all__ = ["PORT", "STRATUM", "VALUE", "certified_out", "device", "out_rows", "period", "pin", "relay", "shape"]
