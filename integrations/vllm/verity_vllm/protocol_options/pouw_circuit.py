"""PoUW's `ncp-v2` under sampled proofs: each weight's side as its own `NcpWeight_v1` Program call, once per run, and each
executed linear call as its own `NcpLinear_v1` Program call reading it.

`verity.core.protocols.pouw.circuit` gives the circuit (`traced_as`: both Definitions, their `Q_template_instances` queries, the served
count and the closure-draw layout; P is the scheme's). Here is what the vLLM integration does with it:

- **Build.** A model Program's row at a PoUW site is lowered to `ROW_FAMILY` (`lower`): the served output of one row of
  that site's `NcpLinear` call, which the replay checks against the call rather than as a bf16 GEMM.
- **Commit.** A weight's side binds `NcpWeight`'s 4 inputs (B, salt, weight id, column ids; `Call.weight_args`) and is
  committed once per run and weight (`WeightRecord`): one leaf per Y unit, the root over them, and its inputs' digest,
  under its leaf (`weight_leaf`). A call binds `NcpLinear`'s 11 inputs (`Call.args`, in `reference.Call.args()`'s order),
  of which Y and c₀ are its weight's outputs as the call reads them. Salt, forward index, position, weight id and the
  global row and column ids are anchored: the executor sets them from its registration, and the replay re-derives them.
  The weight's rows are read from the layer when they are needed (`Call.b`), so no copy of the weight is kept. What the
  call commits is every unit's outputs: one leaf per unit (`unit_leaf`, the template's output leaves in order, in the
  partition's unit numbering), a Merkle root over those (`units_root`), and a digest of the inputs (`inputs_digest`),
  all under the call's leaf (`call_leaf`) with its weight side's. The calls are committed under one root (`calls_root`).
  Only y, the dequantization units' outputs, is served.

The values are computed natively (`pouw_native`: numpy and hashlib, the Commit's path) or by evaluating the Programs
(`commit_call_ir`, `commit_weight_ir`, for small calls: a Qwen2.5-0.5B forward is about 26 hours of gate evaluation). The
two commit the same leaves, and the tests hold them to the same roots.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from verity.core.protocols.pouw import circuit as C
from verity.core.protocols.pouw.audit import commit_rows
from verity.core.protocols.pouw.circuit import partition as CP
from verity.core.protocols.pouw.circuit import reference as R
from verity.core.protocols.pouw.circuit.ncp2 import D, Shape
from verity.core.protocols.pouw.protocol import tagged

from verity.ml.circuits import partition_object as PO
from verity_vllm.query.manifest.format import spec_family
from verity.ml.circuits.evaluate import evaluate_region

#: The model Program's families for one row of a PoUW site: its served output, checked by the replay against the call; the
#: second is a biased linear's row whose bias add was the pre-bias output's only reader (`program.registry.pouw_rows`)
ROW_FAMILY = "NcpLinearRow_v1"
ROW_BIAS_FAMILY = "NcpLinearRowBias_v1"
#: the model families the Build lowers at a PoUW site, and what each becomes
LOWERED = {"Gemm_v1": ROW_FAMILY, "GemmBias_v1": ROW_BIAS_FAMILY}
LEAF_TAG = b"verity/pouw/ncp-v2/call/v3\0"
WEIGHT_TAG = b"verity/pouw/ncp-v2/weight/v1\0"
INPUTS_TAG = b"verity/pouw/ncp-v2/inputs/v2\0"
WEIGHT_INPUTS_TAG = b"verity/pouw/ncp-v2/weight-inputs/v1\0"
UNIT_TAG = b"verity/pouw/ncp-v2/unit/v1\0"
NODE_TAG = b"verity/pouw/ncp-v2/units-node/v1\0"
CALLS_BINDING = "verity/pouw/ncp-v2/calls/v1"
M32 = 0xFFFFFFFF


def kind_of(scheme: str) -> str:
    """The permutation P that ``scheme`` pins (`verity.core.protocols.pouw.circuit.SCHEMES`); ValueError for a scheme without a circuit."""
    if scheme not in C.SCHEMES:
        raise ValueError(f"{scheme}: no circuit (verity.core.protocols.pouw.circuit.SCHEMES = {', '.join(C.SCHEMES)})")
    return C.SCHEMES[scheme]


class Rows(Protocol):
    """A weight's int7 rows (int8) and FP32 scale encodings (uint32), quantized when read."""

    n: int
    k: int

    def rows(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True, eq=False)
class Quantized:
    """Weight rows already quantized."""

    b: np.ndarray
    t: np.ndarray

    @property
    def n(self) -> int:
        return self.b.shape[0]

    @property
    def k(self) -> int:
        return self.b.shape[1]

    def rows(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        return self.b[lo:hi], self.t[lo:hi]


@dataclass(frozen=True, eq=False)
class Call:
    """One linear call's inputs (`reference.Call`'s fields): A as int7 rows, the scales and bias as encodings, the weight's
    rows as read, and the anchored fields."""

    a: np.ndarray               # (m, k) int8
    s: np.ndarray               # (m,) uint32, FP32 encodings
    b: Rows                     # the weight's n rows along k, and their scales t
    bias: np.ndarray            # (n,) uint16, BF16 encodings
    salt: bytes                 # 32 bytes
    forward: int
    layer: int
    site: int
    weight: int
    rowids: np.ndarray          # (m,) uint32: each row's global index
    colids: np.ndarray          # (n,) uint32: each weight row's global index

    @property
    def shape(self) -> Shape:
        return Shape(self.a.shape[0], self.a.shape[1], self.b.n)

    @property
    def position(self) -> int:
        return (self.layer << 32) | self.site

    def reference(self) -> R.Call:
        b, t = self.b.rows(0, self.b.n)
        return R.Call(tuple(tuple(int(v) for v in r) for r in self.a), tuple(int(v) for v in self.s),
                      tuple(tuple(int(v) for v in r) for r in b), tuple(int(v) for v in t), tuple(int(v) for v in self.bias),
                      self.salt, forward=self.forward, layer=self.layer, site=self.site, weight=self.weight,
                      rowids=tuple(int(v) for v in self.rowids), colids=tuple(int(v) for v in self.colids))

    def args(self, kind: str) -> list[list[int]]:
        """The flat input leaves of `NcpLinear` under permutation ``kind``, in parameter order (`reference.Call.args`)."""
        return self.reference().args(kind)

    def weight_args(self) -> list[list[int]]:
        """The flat input leaves of `NcpWeight`, in parameter order (`reference.Call.weight_args`)."""
        return self.reference().weight_args()


def _lane(v: int) -> np.ndarray:
    return np.array([v & M32, (v >> 32) & M32], np.uint32)


class Group:
    """One input group's digest, fed in order: SHA-256 over its entry count (8 bytes, little-endian), then its encodings
    little-endian at the group's width."""

    def __init__(self, size: int, dtype: str) -> None:
        self.h, self.dtype = hashlib.sha256(size.to_bytes(8, "little")), dtype

    def update(self, values: np.ndarray) -> "Group":
        self.h.update(np.ascontiguousarray(values, dtype=self.dtype).reshape(-1).tobytes())
        return self

    def digest(self) -> bytes:
        return self.h.digest()


def _group(values: np.ndarray, dtype: str) -> bytes:
    v = np.asarray(values)
    return Group(v.size, dtype).update(v).digest()


def int7_bytes(v: np.ndarray) -> bytes:
    """int7 entries as their encodings, one byte each."""
    return (np.asarray(v).astype(np.int16) & 0x7F).astype(np.uint8).tobytes()


def _int7_group(v: np.ndarray) -> bytes:
    return hashlib.sha256(np.asarray(v).size.to_bytes(8, "little") + int7_bytes(v)).digest()


class Read:
    """The groups a call reads from its weight, digested block by block in column order: Y (one byte per entry), c₀ and the
    column scales t (four bytes each)."""

    def __init__(self, n: int, k: int) -> None:
        self.ys, self.c0s, self.t = Group(3 * k * n, "<u1"), Group(n, "<u4"), Group(n, "<u4")

    def update(self, y: np.ndarray, c0: np.ndarray, t: np.ndarray) -> None:
        self.ys.update(y)
        self.c0s.update(c0)
        self.t.update(t)

    def parts(self) -> tuple[bytes, bytes, bytes]:
        return self.ys.digest(), self.c0s.digest(), self.t.digest()


def inputs_digest(call: Call, read: tuple[bytes, bytes, bytes]) -> bytes:
    """SHA-256 over the digests of `NcpLinear`'s 11 input groups, in parameter order, each over its encodings little-endian at
    the group's width: one byte for A's int7 entries and Y's, two for the BF16 bias, four for the rest. ``read`` is Y's,
    c₀'s and t's (`Read.parts`)."""
    ys, c0s, t = read
    groups = [_int7_group(call.a), _group(call.s, "<u4"), ys, c0s, t, _group(call.bias, "<u2"),
              _group(np.frombuffer(call.salt, "<u4"), "<u4"), _group(_lane(call.forward), "<u4"), _group(_lane(call.position), "<u4"),
              _group(_lane(call.weight), "<u4"), _group(call.rowids, "<u4")]
    return hashlib.sha256(INPUTS_TAG + b"".join(groups)).digest()


def weight_inputs_digest(b: bytes, salt: bytes, weight: int, colids: np.ndarray) -> bytes:
    """SHA-256 over the digests of `NcpWeight`'s 4 input groups, in parameter order: B's int7 entries (``b``, digested as its
    rows are read: `rows_digest`), the salt, the weight id and the column ids."""
    groups = [b, _group(np.frombuffer(salt, "<u4"), "<u4"), _group(_lane(weight), "<u4"), _group(colids, "<u4")]
    return hashlib.sha256(WEIGHT_INPUTS_TAG + b"".join(groups)).digest()


class RowsDigest:
    """B's int7 entries digested as its rows are read, in row order."""

    def __init__(self, n: int, k: int) -> None:
        self.h = hashlib.sha256((n * k).to_bytes(8, "little"))

    def update(self, b: np.ndarray) -> None:
        self.h.update(int7_bytes(b))

    def digest(self) -> bytes:
        return self.h.digest()


def unit_leaf(values: np.ndarray) -> bytes:
    """One unit's leaf: its output leaves' encodings, in order, as little-endian u32."""
    return hashlib.sha256(UNIT_TAG + np.ascontiguousarray(values, dtype="<u4").tobytes()).digest()


def units_root(leaves: bytes) -> bytes:
    """The Merkle root over a call's unit leaves (32 bytes each, in unit order): each level pairs its nodes left to right
    under `NODE_TAG`, and an odd last node rises unpaired."""
    level = [leaves[i:i + 32] for i in range(0, len(leaves), 32)]
    while len(level) > 1:
        up = [hashlib.sha256(NODE_TAG + level[i] + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            up.append(level[-1])
        level = up
    return level[0]


@dataclass(frozen=True, eq=False)
class WeightRecord:
    """One weight's side in a run, committed once: its inputs (the rows as read, the salt, the weight id, the column ids),
    their digest, every Y unit's leaf (one per 16-column strip) and the root over them. ``opening`` is the prover's: a Y
    unit's outputs (the strip's Y, then its c₀), regenerated when the replay asks for them. ``reads`` is what an honest
    call on the weight reads from it, digested (`Read.parts`: its Y, its c₀ and the scales t), for a committer that doesn't
    see Y itself."""

    scheme: str
    rows: Rows
    salt: bytes
    weight: int
    colids: np.ndarray
    inputs: bytes
    leaves: bytes                                   # 32 bytes per Y unit, in column-strip order
    root: bytes                                     # `units_root(leaves)`
    opening: Callable[[int], np.ndarray]
    reads: tuple[bytes, bytes, bytes]

    @property
    def units(self) -> int:
        return len(self.leaves) // 32

    def leaf(self, u: int) -> bytes:
        return self.leaves[32 * u:32 * u + 32]


def weight_leaf(side: WeightRecord) -> bytes:
    """The weight side's leaf: its scheme, weight id and (k, n), its inputs' digest and its units root."""
    head = f"{side.scheme}|{side.weight}|{side.rows.k}|{side.rows.n}".encode()
    return hashlib.sha256(WEIGHT_TAG + head + b"\0" + side.inputs + side.root).digest()


@dataclass(frozen=True, eq=False)
class CallRecord:
    """One committed call: its inputs, its inputs' digest, every unit's leaf, the root over them, the served y and its
    weight's side. ``opening`` is the prover's: a unit's outputs, regenerated when the replay asks for them; ``read`` the
    Y strip and c₀ the call read at a column strip, which the replay compares with its weight side's outputs there."""

    index: int
    scheme: str
    call: Call
    inputs: bytes
    leaves: bytes                                   # 32 bytes per unit, in the partition's numbering
    root: bytes                                     # `units_root(leaves)`
    y: np.ndarray                                   # (m, n) uint16
    side: WeightRecord
    opening: Callable[[int], np.ndarray]
    read: Callable[[int], tuple[np.ndarray, np.ndarray]]

    @property
    def shape(self) -> Shape:
        return self.call.shape

    @property
    def units(self) -> int:
        return len(self.leaves) // 32

    def leaf(self, u: int) -> bytes:
        return self.leaves[32 * u:32 * u + 32]


def call_leaf(rec: CallRecord) -> bytes:
    """The call's leaf: its index, scheme (which pins P) and shape, its inputs' digest, its units root and its weight side's
    leaf."""
    sh = rec.shape
    head = f"{rec.index}|{rec.scheme}|{sh.m}|{sh.k}|{sh.n}".encode()
    return hashlib.sha256(LEAF_TAG + head + b"\0" + rec.inputs + rec.root + weight_leaf(rec.side)).digest()


def calls_root(salt: bytes, weights_root: bytes, records: Sequence[CallRecord]) -> bytes:
    """The root over every call's leaf, in call order."""
    return commit_rows(tagged(CALLS_BINDING, salt, weights_root), -1, [call_leaf(r) for r in records]).commitment.root


def sides(records: Sequence[CallRecord]) -> list[WeightRecord]:
    """The weight sides the calls read, once each, by weight id."""
    by: dict[int, WeightRecord] = {}
    for r in records:
        by.setdefault(r.side.weight, r.side)
    return [by[w] for w in sorted(by)]


# --- the IR path ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Compiled:
    """One call shape's Programs, their template partitions and their trace (`verity.core.protocols.pouw.circuit.traced_as`)."""

    traced: Any
    program: Any
    partition: PO.TemplatePartition
    weight_program: Any
    weight_partition: PO.TemplatePartition

    @property
    def shape(self) -> Shape:
        return self.traced.shape


@lru_cache(maxsize=32)
def program(scheme: str, m: int, k: int, n: int) -> Compiled:
    """The Programs of one call of ``scheme`` at m × k → n and of its weight side; `DomainError` outside the circuit,
    ValueError for a scheme without one."""
    t = C.traced_as(scheme, m, k, n)
    if t is None:
        raise ValueError(f"{scheme}: no circuit (verity.core.protocols.pouw.circuit.SCHEMES = {', '.join(C.SCHEMES)})")
    prog, wprog = CP.program(t.shape, t.kind), CP.weight_program(t.shape, t.kind)
    return Compiled(t, prog, PO.TemplatePartition(prog, t.query), wprog, PO.TemplatePartition(wprog, t.weight_query))


def _transcript(prog: Any, args: Sequence[Sequence[int]]) -> dict[int, int]:
    transcript: dict[int, int] = {}
    for coll, vals in zip(prog.root.input_colls, args):
        gates = [g for (_space, _zero, g) in coll.refs]
        if len(gates) != len(vals):
            raise ValueError(f"an input group has {len(vals)} values, the Definition takes {len(gates)}")
        transcript.update(zip(gates, vals))
    evaluate_region(prog.circuit, prog.circuit.region(0, prog.circuit.size), transcript)
    return transcript


def evaluate_ir(scheme: str, call: Call) -> tuple[Compiled, dict[int, int]]:
    """The call's Program evaluated on its bound inputs: the compiled Programs and its transcript (gate -> encoding)."""
    sh = call.shape
    c = program(scheme, sh.m, sh.k, sh.n)
    return c, _transcript(c.program, call.args(c.traced.kind))


def evaluate_weight_ir(scheme: str, call: Call) -> tuple[Compiled, dict[int, int]]:
    """The call's weight side's Program evaluated on its bound inputs."""
    sh = call.shape
    c = program(scheme, sh.m, sh.k, sh.n)
    return c, _transcript(c.weight_program, call.weight_args())


def commit_weight_ir(scheme: str, call: Call) -> WeightRecord:
    """The call's weight side committed from its Program's transcript: each Y unit's leaf over the gates its partition names."""
    c, tr = evaluate_weight_ir(scheme, call)
    tp = c.weight_partition
    outs = [np.array([tr[g] for g in tp.outputs(u)], np.uint32) for u in range(tp.population)]
    leaves = b"".join(unit_leaf(v) for v in outs)
    b, t = call.b.rows(0, call.b.n)
    rd = RowsDigest(call.b.n, call.b.k)
    rd.update(b)
    per = D * 3 * call.b.k
    read = (_group(np.concatenate([o[:per] for o in outs]).astype(np.uint8), "<u1"),
            _group(np.concatenate([o[per:] for o in outs]), "<u4"), _group(t, "<u4"))
    return WeightRecord(scheme, call.b, call.salt, call.weight, call.colids,
                        weight_inputs_digest(rd.digest(), call.salt, call.weight, call.colids), leaves, units_root(leaves),
                        outs.__getitem__, read)


def commit_call_ir(index: int, scheme: str, call: Call, side: WeightRecord | None = None) -> CallRecord:
    """The call committed from its Program's transcript: each unit's leaf over the gates `TemplatePartition.outputs` names,
    its weight side ``side`` (`commit_weight_ir` when None)."""
    c, tr = evaluate_ir(scheme, call)
    sh, tp = call.shape, c.partition
    side = side or commit_weight_ir(scheme, call)
    outs = [np.array([tr[g] for g in tp.outputs(u)], np.uint32) for u in range(tp.population)]
    leaves = b"".join(unit_leaf(v) for v in outs)
    served = [tr[g] for g in c.program.circuit.root_outputs()[:c.traced.served]]
    colls = dict(zip([p for p, _t in c.traced.definition.signature()[0]], c.program.root.input_colls))
    ys, c0s = ([tr[g] for (_s, _z, g) in colls[p].refs] for p in C.LINKED)
    t = call.b.rows(0, call.b.n)[1]
    ys_strips = np.array(ys, np.uint8).reshape(sh.n // D, D * 3 * sh.k)
    c0_strips = np.array(c0s, np.uint32).reshape(sh.n // D, D)
    read = (_group(np.array(ys, np.uint8), "<u1"), _group(np.array(c0s, np.uint32), "<u4"), _group(t, "<u4"))
    return CallRecord(index, scheme, call, inputs_digest(call, read), leaves, units_root(leaves),
                      np.array(served, np.uint16).reshape(sh.m, sh.n), side, outs.__getitem__,
                      lambda cs: (ys_strips[cs], c0_strips[cs]))


#: The file under the option's ``transcript_dir`` holding the commitment, every weight side's leaf and every call's leaf
#: (`summary`, `weight_summary`).
CALLS_FILE = "pouw-ncp-v2-calls.json"


#: The file beside it holding every call's inputs but the weight's rows (`save_calls`), which a verifier reads from the weights of
#: record: enough to regenerate and open each call and its weight side off the serving host (`load_calls`).
CALLS_DATA_FILE = "pouw-ncp-v2-calls.npz"


def save_calls(path: str | Path, records: Sequence[CallRecord]) -> None:
    """Every call's A, s, bias, row and column ids, salt and anchored fields, by index."""
    arrays: dict[str, np.ndarray] = {}
    for r in records:
        c, p = r.call, f"c{r.index}_"
        arrays.update({p + "a": c.a, p + "s": c.s, p + "bias": c.bias, p + "rowids": c.rowids, p + "colids": c.colids,
                       p + "salt": np.frombuffer(c.salt, np.uint8), p + "fields": np.array([c.forward, c.layer, c.site, c.weight], np.int64)})
    np.savez_compressed(path, index=np.array([r.index for r in records], np.int64), schemes=np.array([r.scheme for r in records]), **arrays)


def load_calls(path: str | Path, rows: Callable[[int], Rows]) -> list[tuple[int, str, Call]]:
    """(index, scheme, call) for each call `save_calls` wrote, its weight's rows from ``rows(weight)``."""
    z = np.load(path)
    out = []
    for i, scheme in zip(z["index"].tolist(), z["schemes"].tolist()):
        p = f"c{i}_"
        forward, layer, site, weight = (int(v) for v in z[p + "fields"])
        out.append((i, str(scheme), Call(z[p + "a"], z[p + "s"], rows(weight), z[p + "bias"], z[p + "salt"].tobytes(), forward=forward, layer=layer,
                                         site=site, weight=weight, rowids=z[p + "rowids"], colids=z[p + "colids"])))
    return out


def summary(rec: CallRecord) -> dict[str, Any]:
    c, sh = rec.call, rec.shape
    return {"index": rec.index, "scheme": rec.scheme, "shape": [sh.m, sh.k, sh.n], "forward": c.forward, "layer": c.layer,
            "site": c.site, "weight": c.weight, "rows": [int(c.rowids[0]), sh.m], "inputs": rec.inputs.hex(),
            "units_root": rec.root.hex(), "weight_leaf": weight_leaf(rec.side).hex(), "leaf": call_leaf(rec).hex()}


def weight_summary(side: WeightRecord) -> dict[str, Any]:
    return {"weight": side.weight, "scheme": side.scheme, "shape": [side.rows.k, side.rows.n], "inputs": side.inputs.hex(),
            "units_root": side.root.hex(), "leaf": weight_leaf(side).hex()}


def lower(inst: dict[str, Any], sites: Mapping[str, str]) -> dict[str, Any]:
    """The Build's lowering of a model Program (an instance sequence) under ``pouw:<scheme>``: every row of a PoUW site
    (``sites``: op path prefix of a linear module -> scheme, e.g. ``model.layers.0.mlp.down_proj`` -> ``ncp-v2``), a
    ``Gemm_v1{K,N}`` or ``GemmBias_v1{K,N}`` row, becomes ``NcpLinearRow_v1{K,N}`` or ``NcpLinearRowBias_v1{K,N}`` over the
    same arguments (`LOWERED`, `program.registry.pouw_rows`). The rest is unchanged; the scheme is the calls'."""
    rows = []
    for r in inst["rows"]:
        spec, name = r.get("spec", ""), r.get("name", "")
        site = next((p for p in sites if name.startswith(p + "/")), None)
        if site is not None and spec_family(spec) in LOWERED:
            statics = spec[spec.index("{") + 1:-1]
            r = dict(r, spec=f"{LOWERED[spec_family(spec)]}{{{statics}}}")
        rows.append(r)
    return dict(inst, rows=rows)


__all__ = ["CALLS_BINDING", "CALLS_DATA_FILE", "CALLS_FILE", "Call", "CallRecord", "Compiled", "Group", "LOWERED", "Quantized",
           "ROW_BIAS_FAMILY", "ROW_FAMILY", "Read", "Rows", "RowsDigest", "WeightRecord", "call_leaf", "calls_root", "commit_call_ir",
           "commit_weight_ir", "evaluate_ir", "evaluate_weight_ir", "inputs_digest", "int7_bytes", "kind_of", "load_calls", "lower",
           "program", "save_calls", "sides", "summary", "unit_leaf", "units_root", "weight_inputs_digest", "weight_leaf",
           "weight_summary"]
