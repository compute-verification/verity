"""One IR function call as ``veritor.reduce-mufu@1`` obligations: the approved guest re-runs the IR's gate list.

The IR oracle (:func:`verity.ml.circuits.evaluation.reference.evaluate`) evaluates a specialised composite by flattening one
call of it into primitive gates in canonical order (:func:`~verity.ml.circuits.evaluation.reference.standalone_call`) and
applying each primitive's evaluator.  :func:`lower_call` restates exactly that gate list as the subcircuit of a
``typed-obligation/v0`` object over the guest family ``veritor.reduce-mufu@1``, one obligation per instance, so the
unmodified SP1 checker recomputes it gate for gate and compares with the committed outputs:

* the call's arguments are the incoming values, read from one packed leaf per parameter under the root
  ``inputs``; its returns are the outgoing values, one packed leaf per tuple component under ``outputs``;
* a constant primitive (a zero-operand gate) is an incoming value read from the public leaf under ``constants``;
* a primitive the guest evaluates against a committed MUFU table (``RsqrtApprox``, ...) gets one more incoming
  value, the table row ``(index << 32) | word`` its operand selects, read from the table's tree (256 rows a leaf);
  the guest rejects any other row;
* every other primitive is the guest gate :data:`OPS` names for it (``veritor.tc-ampere-bf16@2``: :data:`TC_OPS`, and a
  tensor-core step whose accumulator is a constant zero is ``tc_dot16_0``).

Everything is committed under ``verity-vllm/leaf/v1`` and opened as whole packed leaves (statement format 3,
witness format 3).  The roots of ``inputs`` and ``outputs`` are the hashes the proof binds its instances to; the
constants and table roots are public and recomputed by the verifier from the same data.  Nothing here imports a
frontend: the caller supplies the specialised definition, the instances and the tables.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from verity.ml.commitments.leaves import leaf_hash, levels, open_path, shape_text
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits.defs import SpecializedDefinition
from verity.ml.circuits.evaluate import evaluate_call
from verity.ml.circuits.types import Tuple as TupleT
from verity_sp1.proofs.elem_bf16 import make_elem_bf16_gate_set
from verity_sp1.proofs.families import make_tc_ampere_bf16_gate_set
from verity_sp1.proofs.gateset import GateSet
from verity_sp1.proofs.lowering import to_statement_v3
from verity_sp1.proofs.packed import PackedLeaf, PackedWitness, encode_packed_witness, packed_table
from verity_sp1.proofs.reduce_mufu import REDUCE_MUFU_NAME, REDUCE_MUFU_VERSION, make_reduce_mufu_gate_set
from verity_sp1.proofs.typed_obligation import (
    VLLM_FRAMING,
    Array,
    Binding,
    Bits,
    Boundary,
    Definition,
    Family,
    Gate,
    Leaf,
    Obligation,
    Reference,
    Root,
    Subcircuit,
    TypedObligationSet,
    subcircuit_digest,
)
from verity_sp1.proofs.wire import encode_statement_v3

__all__ = ["ELEM_OPS", "FAMILIES", "OPS", "TABLE_GATES", "TC_OPS", "Instance", "Lowered", "Table", "lower_call"]

FAMILY_ID = f"{REDUCE_MUFU_NAME}@{REDUCE_MUFU_VERSION}"
#: IR primitive (``name_vN``) -> the ``veritor.reduce-mufu@1`` gate with the same semantics.
OPS: Mapping[str, str] = {
    "Bf16AddF2fp_v1": "bf16_add_f2fp",
    "Bf16GtStrict_v1": "bf16_gt",
    "Bf16ToF32_v1": "bf16_to_f32",
    "DivFullRcp_v1": "div_full_rcp",
    "DivFullScaleA_v1": "div_full_scale_a",
    "F2fpBf16_v1": "f2fp_bf16",
    "F32Add_v1": "f32_add",
    "F32Div_v1": "f32_div",
    "F32Fma_v1": "f32_fma",
    "F32Mul_v1": "f32_mul",
    "I32Add_v1": "i32_add",
    "I32Eq_v1": "i32_eq",
    "I32Le_v1": "i32_le",
    "MufuSqrtFtz_v1": "mufu_sqrt_tab",
    "RopeOut_v1": "rope_out",
    "RopeOutAdd_v1": "rope_out_add",
    "RsqrtApprox_v1": "rsqrt_approx",
    "SelectBf16_v1": "select_bf16",
    "SelectI32_v1": "select_i32",
}
#: IR primitive -> ``veritor.tc-ampere-bf16@2`` gate: the Ampere ``mma`` step and the hardware casts.  A step whose
#: accumulator is a constant zero lowers to ``tc_dot16_0``.
TC_OPS: Mapping[str, str] = {
    "AmpereBF16TcDot16_v1": "tc_dot16",
    "AmpereBF16TcDot16_v2": "tc_dot16",
    "AmpereBF16TcDot16_v2": "tc_dot16",
    "Bf16ToF32_v1": "bf16_to_f32",
    "F2fpBf16_v1": "f32_to_bf16",
}
#: IR primitive -> ``veritor.elem-bf16@1`` gate (the SiLU table is the guest's embedded one).
ELEM_OPS: Mapping[str, str] = {
    "Bf16AddF2fp_v1": "bf16_add_f2fp",
    "Bf16ToF32_v1": "bf16_to_f32",
    "SiluMulBf16_v1": "silu_mul_bf16",
}
#: Guest family -> (its gate set, the IR primitive -> gate map).
FAMILIES: Mapping[str, tuple[Callable[[], GateSet], Mapping[str, str]]] = {
    FAMILY_ID: (make_reduce_mufu_gate_set, OPS),
    "veritor.tc-ampere-bf16@2": (lambda: make_tc_ampere_bf16_gate_set("2"), TC_OPS),
    "veritor.elem-bf16@1": (make_elem_bf16_gate_set, ELEM_OPS),
}
#: Guest gate -> the name of the MUFU table whose row is its second argument.
TABLE_GATES: Mapping[str, str] = {"rsqrt_approx": "rsq", "mufu_sqrt_tab": "sqrt", "div_full_rcp": "rcp"}
ROWS_PER_LEAF = 256
_WIDTH_DTYPE = {16: "uint16", 32: "uint32", 64: "uint64"}


def _domain(name: str) -> str:
    return hashlib.sha256(f"verity-sp1/ir-call/{name}".encode()).hexdigest()


def _packed(words: Sequence[int], width: int) -> bytes:
    size = width // 8
    return b"".join(int(w).to_bytes(size, "little") for w in words)


@dataclass(frozen=True)
class Table:
    """A committed MUFU table: its ``words`` (one u32 per row), the row a gate reads for its FP32 operand
    (``index``, e.g. :func:`verity_sp1.proofs.reduce_mufu.rsqrt_approx_index`) and the table file's ``source`` digest."""

    name: str
    words: Sequence[int]
    index: Callable[[int], int]
    source: str
    _tree: list = field(default_factory=list, compare=False, repr=False)

    def leaf_value(self, rank: int) -> bytes:
        lo = rank * ROWS_PER_LEAF
        return _packed([(i << 32) | int(self.words[i]) for i in range(lo, lo + ROWS_PER_LEAF)], 64)

    def leaf_id(self, rank: int) -> str:
        return f"table/{self.name}/{rank:06d}"

    @property
    def leaves(self) -> int:
        return len(self.words) // ROWS_PER_LEAF

    def tree(self) -> list:
        if not self._tree:
            shape = shape_text([ROWS_PER_LEAF])
            self._tree.extend(levels([leaf_hash(self.leaf_id(r), "uint64", shape, self.leaf_value(r)) for r in range(self.leaves)]))
        return self._tree


@dataclass(frozen=True)
class Instance:
    """One instance: ``args`` in the call's argument order (flat), the claimed ``returns`` (flat), and the keys
    it is recorded under (``unit`` orders the obligations)."""

    unit: int
    args: Sequence[int]
    returns: Sequence[int]


@dataclass(frozen=True)
class Lowered:
    obj: TypedObligationSet
    statement: bytes
    witness: bytes
    roots: dict[str, str]
    gates: int


class _Tree:
    """A packed tree under construction: leaves appended in rank order."""

    def __init__(self, root_id: str) -> None:
        self.root_id = root_id
        self.ids: list[str] = []
        self.values: list[bytes] = []
        self.dtypes: list[str] = []
        self.shapes: list[str] = []
        self._levels: list | None = None

    def add(self, leaf_id: str, words: Sequence[int], width: int) -> int:
        self.ids.append(leaf_id)
        self.values.append(_packed(words, width))
        self.dtypes.append(_WIDTH_DTYPE[width])
        self.shapes.append(shape_text([len(words)]))
        return len(self.ids) - 1

    def levels(self) -> list:
        if self._levels is None:
            self._levels = levels([leaf_hash(*parts) for parts in zip(self.ids, self.dtypes, self.shapes, self.values)])
        return self._levels

    def opening(self, rank: int) -> PackedLeaf:
        return PackedLeaf(self.ids[rank], self.dtypes[rank], self.shapes[rank], self.values[rank], open_path(self.levels(), rank))


def _components(t) -> list:
    return list(t.fields) if isinstance(t, TupleT) else [t]


def lower_call(
    spec: SpecializedDefinition,
    instances: Sequence[Instance],
    *,
    tables: Mapping[str, Table] = {},
    session: str,
    compiled: str,
    instance_label: Callable[[int], str] = lambda unit: f"u{unit:06d}",
    oracle: bool = True,
    family: str = FAMILY_ID,
) -> Lowered:
    """The obligations of ``instances`` of ``spec`` (see the module docstring), the format-3 statement and witness,
    over the guest ``family`` (a key of :data:`FAMILIES`).

    ``session`` and ``compiled`` (32-byte hex) are the binding every obligation carries.  Raises ``KeyError`` for a
    primitive with no guest gate, and ``ValueError`` when an instance's claimed returns are not what the IR oracle
    computes (the guest would reject it; ``oracle=False`` commits them anyway, for negatives) or when two table
    gates of one instance read one row.
    """

    make_gate_set, ops = FAMILIES[family]
    prog, call = standalone_call(spec)
    circuit = prog.circuit
    params, ret = spec.signature()
    arg_gates = list(call.args)
    return_gates = list(call.returns)
    widths = {g: circuit.gate(g).prim.ret.w for g in arg_gates}
    # a zero constant read only as a tensor-core accumulator is the step's zero-accumulator form, not a value
    readers: dict[int, list[tuple[str, int]]] = {}
    for gv in call:
        for position, o in enumerate(gv.operands):
            readers.setdefault(o, []).append((ops.get(gv.prim.id, ""), position))
    folded = {gv.index for gv in call if not gv.operands and gv.prim.evaluate() == 0
              and all(op == "tc_dot16" and position == 0 for op, position in readers.get(gv.index, ()))}

    # the body: incoming = arguments, then constants, then one table row per table gate
    body_of: dict[int, int] = {}
    incoming_of: dict[int, int] = {g: i for i, g in enumerate(arg_gates)}
    incoming = [widths[g] for g in arg_gates]
    constants: list[tuple[int, int]] = []  # (gate, value)
    table_gates: list[tuple[int, int, str]] = []  # (gate, operand gate, table)
    extras: list[tuple[str, int]] = []  # the incoming values after the arguments, in body order: ("const", k) | ("table", k)
    gates: list[Gate] = []

    def arg(g: int) -> tuple[str, int]:
        return ("in", incoming_of[g]) if g in incoming_of else ("gate", body_of[g])

    for gv in call:
        prim = gv.prim
        if gv.index in folded:
            continue
        if not gv.operands:
            incoming_of[gv.index] = len(incoming)
            incoming.append(prim.ret.w)
            extras.append(("const", len(constants)))
            constants.append((gv.index, prim.evaluate()))
            continue
        op = ops[prim.id]
        operands = list(gv.operands)
        if op == "tc_dot16" and operands[0] in folded:
            op, operands = "tc_dot16_0", operands[1:]
        args = [arg(o) for o in operands]
        if op in TABLE_GATES:
            extras.append(("table", len(table_gates)))
            table_gates.append((gv.index, gv.operands[0], TABLE_GATES[op]))
            args.append(("in", len(incoming)))
            incoming.append(64)
        body_of[gv.index] = len(gates)
        gates.append(Gate(op, tuple(args)))
    outgoing = sorted(body_of[g] for g in set(return_gates))
    statics = tuple(sorted((k, v if isinstance(v, int) and v >= 0 else (v.hex() if isinstance(v, float) else str(v)))
                           for k, v in spec.bindings.items()))
    body = Subcircuit("", tuple(incoming), tuple(gates), tuple(outgoing), Definition(spec.defn.name, statics))
    body = Subcircuit(subcircuit_digest(body), body.incoming, body.gates, body.outgoing, body.definition)

    trees = {name: _Tree(name) for name in ("constants", "inputs", "outputs")}
    leaves: list[Leaf] = []
    references: list[Reference] = []
    const_width = {c[0]: circuit.gate(c[0]).prim.ret.w for c in constants}
    if constants:
        if len(set(const_width.values())) != 1:
            raise ValueError("constants of several widths need one leaf each")
        (cw,) = set(const_width.values())
        trees["constants"].add("constants", [v for _, v in constants], cw)
        leaves.append(Leaf("constants", "constants", 0, 0, Array(Bits(cw), len(constants))))
        references += [Reference(f"constants[{k}]", "constants", (k,)) for k in range(len(constants))]
    table_refs: dict[tuple[str, int], str] = {}
    obligations: list[Obligation] = []
    return_pos = {g: m for m, g in reversed(list(enumerate(return_gates)))}

    for inst in sorted(instances, key=lambda i: i.unit):
        label = instance_label(inst.unit)
        transcript = {}
        values = evaluate_call(circuit, call, inst.args, transcript)
        if oracle and list(values) != list(inst.returns):
            raise ValueError(f"instance {inst.unit}: the claimed returns are not the IR oracle's")
        in_refs: list[str] = []
        start = 0
        for pname, ptype in params:
            n = ptype.leaves
            lid = f"{label}/in/{pname}"
            w = widths[arg_gates[start]]
            trees["inputs"].add(lid, inst.args[start:start + n], w)
            leaves.append(Leaf(lid, "inputs", len(trees["inputs"].ids) - 1, len(trees["inputs"].ids) - 1, Array(Bits(w), n)))
            for j in range(n):
                ref = f"{lid}[{j}]"
                references.append(Reference(ref, lid, (j,)))
                in_refs.append(ref)
            start += n
        rows_read: set[tuple[str, int]] = set()
        for kind, k in extras:
            if kind == "const":
                in_refs.append(f"constants[{k}]")
                continue
            _, operand, name = table_gates[k]
            index = tables[name].index(transcript[operand])
            key = (name, index)
            if key in rows_read:
                raise ValueError(f"instance {inst.unit} reads row {index} of {name} twice")
            rows_read.add(key)
            if key not in table_refs:
                table_refs[key] = f"table/{name}[{index:08d}]"
                rank = index // ROWS_PER_LEAF
                lid = tables[name].leaf_id(rank)
                if not any(v.id == lid for v in leaves):
                    leaves.append(Leaf(lid, f"table/{name}", rank, rank, Array(Bits(64), ROWS_PER_LEAF)))
                references.append(Reference(table_refs[key], lid, (index % ROWS_PER_LEAF,)))
            in_refs.append(table_refs[key])
        # returns: one leaf per tuple component, in declared order
        out_ref_of: dict[int, str] = {}
        start = 0
        for c, ctype in enumerate(_components(ret)):
            n = ctype.leaves
            lid = f"{label}/out/{c}"
            w = circuit.gate(return_gates[start]).prim.ret.w
            trees["outputs"].add(lid, inst.returns[start:start + n], w)
            leaves.append(Leaf(lid, "outputs", len(trees["outputs"].ids) - 1, len(trees["outputs"].ids) - 1, Array(Bits(w), n)))
            for j in range(n):
                g = return_gates[start + j]
                if return_pos[g] == start + j:
                    ref = f"{lid}[{j}]"
                    references.append(Reference(ref, lid, (j,)))
                    out_ref_of[body_of[g]] = ref
            start += n
        used_tables = sorted({f"table/{name}" for _, _, name in table_gates})
        obligations.append(Obligation(
            Binding(session, compiled, inst.unit, inst.unit),
            body.digest,
            tuple(sorted({"inputs", "outputs", *(("constants",) if constants else ()), *used_tables})),
            Boundary(tuple(in_refs), tuple(out_ref_of[j] for j in outgoing)),
        ))

    roots: list[Root] = []
    root_hex: dict[str, str] = {}
    for name, tree in trees.items():
        if tree.ids:
            root_hex[name] = tree.levels()[-1][0].hex()
            roots.append(Root(name, -2 if name == "constants" else 0, _domain(name), root_hex[name], len(tree.ids), VLLM_FRAMING))
    for name in sorted({name for _, _, name in table_gates}):
        t = tables[name]
        root_hex[f"table/{name}"] = t.tree()[-1][0].hex()
        roots.append(Root(f"table/{name}", -2, _domain(f"table/{name}/{t.source}"), root_hex[f"table/{name}"], t.leaves, VLLM_FRAMING))

    gate_set = make_gate_set()
    obj = TypedObligationSet(
        Family(family, str(gate_set.digest), 32),
        (),
        tuple(sorted(roots, key=lambda r: r.id)),
        tuple(sorted(leaves, key=lambda v: v.id)),
        tuple(sorted(references, key=lambda r: r.ref)),
        (body,),
        tuple(obligations),
    )
    statement = to_statement_v3(obj)
    # witness: per tree in the statement's order, the opened ranks ascending
    by_root = {bytes.fromhex(r.root): r.id for r in obj.roots}
    opened = []
    for ptree in packed_table(statement.expand()).trees:
        rid = by_root[ptree.commitment.root]
        if rid.startswith("table/"):
            t = tables[rid.removeprefix("table/")]
            shape = shape_text([ROWS_PER_LEAF])
            opened.append(tuple(PackedLeaf(t.leaf_id(r), "uint64", shape, t.leaf_value(r), open_path(t.tree(), r)) for r in ptree.ranks))
        else:
            opened.append(tuple(trees[rid].opening(r) for r in ptree.ranks))
    witness = encode_packed_witness(PackedWitness(tuple(opened)))
    return Lowered(obj, encode_statement_v3(statement), witness, root_hex, len(gates))
