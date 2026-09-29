"""PoUW's `ncp-v2` under sampled proofs: each executed linear call as its own `NcpLinear_v1` Program call.

`verity_pouw.circuit` gives the circuit (`traced_as`: the Definition, its `Q_template_instances` query, the served count
and the closure-draw layout). Here is what the vLLM integration does with it:

- **Build.** One Program per distinct call shape, cached (`program`), with its template partition. A model Program's
  row at a PoUW site is lowered to `ROW_FAMILY` (`lower`): the served output of one row of that site's `NcpLinear` call,
  which the replay checks against the call rather than as a bf16 GEMM.
- **Commit.** A call binds the Definition's 11 inputs in `reference.Call.args()` order. Salt, forward index, position,
  weight id and the global row and column ids are anchored: the executor sets them from its registration, and the
  replay re-derives them. The whole call is evaluated once, and every value the partition's cut commits is kept
  (`CallRecord.committed`). Only y, the first m·n root outputs, is served; the credited words stay committed.
  The calls are committed under one root (`calls_root`), a leaf per call (`leaf`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from verity_pouw import circuit as C
from verity_pouw.audit import commit_rows
from verity_pouw.circuit import partition as CP
from verity_pouw.circuit import reference as R
from verity_pouw.circuit.ncp2 import Shape
from verity_pouw.protocol import tagged

from verity.ir import partition_object as PO
from verity.ir.evaluate import evaluate_region

#: The model Program's family for one row of a PoUW site: its served output, checked by the replay against the call.
ROW_FAMILY = "NcpLinearRow_v1"
LEAF_TAG = b"verity/pouw/ncp-v2/call/v1\0"
CALLS_BINDING = "verity/pouw/ncp-v2/calls/v1"


@dataclass(frozen=True)
class Compiled:
    """One call shape's Program, its template partition and its trace (`verity_pouw.circuit.traced_as`)."""

    traced: Any
    program: Any
    partition: PO.TemplatePartition

    @property
    def shape(self) -> Shape:
        return self.traced.shape


@lru_cache(maxsize=32)
def program(scheme: str, m: int, k: int, n: int, kind: str = "shift24") -> Compiled:
    """The Program of one call of ``scheme`` at m × k → n; `DomainError` outside the circuit, ValueError for a scheme
    without one."""
    t = C.traced_as(scheme, m, k, n, kind)
    if t is None:
        raise ValueError(f"{scheme}: no circuit (verity_pouw.circuit.SCHEMES = {', '.join(C.SCHEMES)})")
    prog = CP.program(t.shape, kind)
    return Compiled(t, prog, PO.TemplatePartition(prog, t.query))


@dataclass(frozen=True)
class CallRecord:
    """One committed call: its inputs, every value its partition's cut commits (gate -> encoding) and its served y."""

    index: int
    scheme: str
    call: R.Call
    committed: Mapping[int, int]
    y: tuple[tuple[int, ...], ...]

    @property
    def shape(self) -> Shape:
        return self.call.shape


def commit_call(index: int, scheme: str, call: R.Call) -> CallRecord:
    """Evaluate the call's Program on its bound inputs and keep what the cut commits: every unit's inputs and the root's
    outputs."""
    sh = call.shape
    c = program(scheme, sh.m, sh.k, sh.n)
    prog, tp = c.program, c.partition
    transcript: dict[int, int] = {}
    for coll, vals in zip(prog.root.input_colls, call.args()):
        gates = [g for (_space, _zero, g) in coll.refs]
        if len(gates) != len(vals):
            raise ValueError(f"call {index}: an input group has {len(vals)} values, the Definition takes {len(gates)}")
        transcript.update(zip(gates, vals))
    evaluate_region(prog.circuit, prog.circuit.region(0, prog.circuit.size), transcript)
    served = [transcript[g] for g in prog.circuit.root_outputs()[:c.traced.served]]
    y = tuple(tuple(served[i * sh.n:(i + 1) * sh.n]) for i in range(sh.m))
    return CallRecord(index, scheme, call, {g: transcript[g] for g in tp.committed()}, y)


def leaf(rec: CallRecord) -> bytes:
    """The call's leaf: its index, scheme, shape, inputs and committed values, in gate order."""
    sh = rec.shape
    h = hashlib.sha256(LEAF_TAG)
    h.update(f"{rec.index}|{rec.scheme}|{sh.m}|{sh.k}|{sh.n}".encode())
    for group in rec.call.args():
        h.update(len(group).to_bytes(8, "little") + b"".join(int(v).to_bytes(8, "little") for v in group))
    for g in sorted(rec.committed):
        h.update(g.to_bytes(8, "little") + int(rec.committed[g]).to_bytes(8, "little"))
    return h.digest()


def calls_root(salt: bytes, weights_root: bytes, records: Sequence[CallRecord]) -> bytes:
    """The root over every call's leaf, in call order."""
    return commit_rows(tagged(CALLS_BINDING, salt, weights_root), -3, [leaf(r) for r in records]).commitment.root


#: The file under the option's ``transcript_dir`` holding the commitment and every call's record (the prover's openings).
CALLS_FILE = "pouw-ncp-v2-calls.json"


def record_json(rec: CallRecord) -> dict[str, Any]:
    c = rec.call
    return {"index": rec.index, "scheme": rec.scheme, "leaf": leaf(rec).hex(),
            "call": {"a": [list(r) for r in c.a], "s": list(c.s), "b": [list(r) for r in c.b], "t": list(c.t), "bias": list(c.bias),
                     "salt": c.salt.hex(), "forward": c.forward, "layer": c.layer, "site": c.site, "weight": c.weight,
                     "rowids": list(c.rowids), "colids": list(c.colids)},
            "committed": [sorted(rec.committed), [rec.committed[g] for g in sorted(rec.committed)]], "y": [list(r) for r in rec.y]}


def record_of(obj: Mapping[str, Any]) -> CallRecord:
    c = obj["call"]
    call = R.Call(tuple(tuple(r) for r in c["a"]), tuple(c["s"]), tuple(tuple(r) for r in c["b"]), tuple(c["t"]), tuple(c["bias"]),
                  bytes.fromhex(c["salt"]), forward=c["forward"], layer=c["layer"], site=c["site"], weight=c["weight"],
                  rowids=tuple(c["rowids"]), colids=tuple(c["colids"]))
    gates, vals = obj["committed"]
    return CallRecord(int(obj["index"]), obj["scheme"], call, dict(zip(gates, vals)), tuple(tuple(r) for r in obj["y"]))


def lower(inst: dict[str, Any], sites: Mapping[str, str]) -> dict[str, Any]:
    """The Build's lowering of a model Program (an instance sequence) under ``pouw:<scheme>``: every row of a PoUW site
    (``sites``: op path prefix of a linear module -> scheme, e.g. ``model.layers.0.mlp.down_proj`` -> ``ncp-v2``), a
    ``Gemm_v1{K,N}`` row, becomes ``NcpLinearRow_v1{K,N,SCHEME}`` over the same arguments. The rest is unchanged."""
    rows = []
    for r in inst["rows"]:
        spec, name = r.get("spec", ""), r.get("name", "")
        site = next((p for p in sites if name.startswith(p + "/")), None)
        if site is not None and spec.startswith("Gemm_v1{"):
            statics = spec[len("Gemm_v1{"):-1]
            r = dict(r, spec=f"{ROW_FAMILY}{{{statics},SCHEME={sites[site]}}}")
        rows.append(r)
    return dict(inst, rows=rows)


__all__ = ["CALLS_BINDING", "CALLS_FILE", "CallRecord", "Compiled", "ROW_FAMILY", "calls_root", "commit_call", "leaf", "lower", "program",
           "record_json", "record_of"]
