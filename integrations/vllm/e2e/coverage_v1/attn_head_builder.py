"""The whole-head attention relation as one coarser VU -- builder, expectation, coverage view (A5/6, 20:40Z brief item 4; S55).

PROPOSED partition change, behind ``COVERAGE_V1_HEAD_VU`` (:mod:`e2e.coverage_v1.builders`): the unit is
``AttentionHead_v1{T,D}|call`` = the registered ``verity_vllm.registry.b1.AttentionHead`` body for one (request, t, layer, head) --
causal attention of the query at position ``t`` over keys ``0..t`` -- whose boundary IS committed by today's records:

* ``q``   = head ``h`` slice of ``layers.<L>.self_attn.rotary_emb.rope/port=0`` at ``t``            (``[NH, D]`` bf16, RU tree of the step),
* ``K``   = KV-head ``g = h // (NH/KVH)`` slice of ``rope/port=1`` at every position ``c <= t``          (``[KVH, D]`` bf16, the step trees of ``c``),
* ``V``   = the V column slice of ``layers.<L>.self_attn.qkv_proj.gemm/port=0`` at every ``c <= t`` (``[(NH+2*KVH)*D]`` bf16; `capture.py:27-28`),
* ``out`` = head ``h`` slice of ``layers.<L>.self_attn.attn.attn/port=0`` at ``t``                      (``[NH, D]`` bf16).

The statement is the fa2 family's (``veritor.fa2-attn``, guest ``33cba058``, UNAPPROVED): one ``AttnBlock<D, BN, NVIS, FIRST>``
obligation per 64-key block (visited last -> first) plus the ``Fa2Epilogue`` obligation, exactly ``zk/bench/pb_attn_batches.py`` over
record rows instead of the W11 fixture.  The online-softmax state between blocks (``m, l[4], o[D]``) is authenticated through a
PROVER-SIDE state tree: block ``j`` writes and block ``j+1`` reads the same leaf, the epilogue reads the last one and writes the
committed ``out`` slice.  That tree is not a record commitment and carries no trust: the relation it chains starts at committed
``q/K/V`` and ends at the committed ``out``; the verifier recovers the head-level view from the raw statement with
:func:`head_views_of_raw_statement`, which REFUSES a statement whose chain is broken (a block reading a state no earlier block wrote,
a missing block, an epilogue on the wrong state) so that batching cannot hide an omitted block behind another head's blocks.

Checker semantics (R10, item 4): the fa2 guest's ``tc_dot16`` is TOTAL (NaN / inf extension, `zk/sp1-attn/common/src/fa2.rs:254-298`),
matching the registered ``_tc_dot16_total`` (`verity_vllm/registry/prims.py:359-395`) -- and DIFFERING from the approved guest's
``tc_dot`` which rejects non-finite operands (`zk/sp1/common/src/tc.rs:77,98,239,246`).  The Python mirror used here
(``zk/bench/attn_gates.make_fa2_attn_gate_set``) evaluates every gate with the registered prims, so the fold below is the registered
semantics; whether the guest's FTZ/MUFU/ex2-table code equals them word-for-word is the fa2 guest's own vector suite
(`zk/sp1-attn/common/src/mufu.rs`, tables pinned by ``attn_gates.table_digests``) -- listed in the report, not re-derived here.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .builders import BuildRefusal, Built, LocatedUnit, Record, Row, base_kind
from .coverage import CoverageError, ObligationView, Position, _canon, _tagged_sha256

__all__ = ["HEAD_KIND", "AttentionHeadBuilder", "head_kind_id", "head_subcircuit_digest", "head_chain_kinds", "head_expectation", "head_views_of_raw_statement", "fa2_gate_set", "verify_locally"]

HEAD_KIND = "AttentionHead_v1|call"
BN = 64
STATE_TAG = "veritor/coverage_v1/attn-head-state"


def _bench():
    repo = Path(__file__).resolve().parents[3]
    bench = repo / "zk" / "bench"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    import attn_gates  # type: ignore[import-not-found]
    import pb_attn_batches  # type: ignore[import-not-found]

    return attn_gates, pb_attn_batches


def fa2_gate_set():
    """The Python mirror of ``veritor.fa2-attn`` (``zk/bench/attn_gates.py``; version = the pinned MUFU table digests)."""

    ag, _ = _bench()
    return ag.make_fa2_attn_gate_set()


def head_kind_id(T: int, D: int) -> str:
    return f"AttentionHead_v1{{T={T},D={D}}}|call"


def head_subcircuit_digest(T: int, D: int, program_kinds: list[str]) -> str:
    """The head body's digest: the ordered chain of block programs (``KindProgram.kind`` hex, blocks last -> first) and the
    epilogue program, with ``T``, ``D``, ``BN``.  The verifier recomputes it from the fixed circuit (the same lowering)."""

    return _tagged_sha256("veritor/coverage_v1/attention-head", _canon({"T": T, "D": D, "BN": BN, "chain": program_kinds}))


def head_chain_kinds(T: int, D: int, gate_set=None) -> list[str]:
    """The fixed circuit's program chain for a head at ``T`` keys: block programs (visited last -> first) then the epilogue --
    ``KindProgram.kind`` hex each, from the same lowering the builder uses.  Value-free: what the verifier computes."""

    from veritor.interface.lowering import _lower_subcircuit

    ag, _ = _bench()
    gs = gate_set if gate_set is not None else fa2_gate_set()
    nb = -(-T // BN)
    kinds: list[str] = []
    for b in range(nb - 1, -1, -1):
        lo, hi = b * BN, min(T, (b + 1) * BN)
        kinds.append(_lower_subcircuit(ag.cached_block(D, BN, hi - lo, b == nb - 1)[0], gs)[0].kind.hex())
    kinds.append(_lower_subcircuit(ag.attn_epilogue_subcircuit(D), gs)[0].kind.hex())
    return kinds


def head_expectation(unit: LocatedUnit, rows: Callable[[str, int, str, int], tuple[str, str, int] | None], config: Mapping[str, Any], *, gate_set=None):
    """VERIFIER side: the A2 ``ExpectedObligation`` of a head unit from committed ids only -- ``rows(request, t, op, port)`` returns
    ``(tree, canonical_id, word_count)`` from the committed id lists + Description (never a value), or None.  Raises ``ValueError``
    naming the missing row.  Equals the builder's expectation for the same unit (tested)."""

    from veritor.interface.expected import ExpectedObligation, ExpectedValue

    NH, KVH, D = int(config["NH"]), int(config["KVH"]), int(config["D"])
    if unit.layer is None or unit.head is None:
        raise ValueError(f"{unit.vu_id}: head expectation needs layer and head")
    L, h, t, rid = int(unit.layer), int(unit.head), int(unit.t), unit.request_id
    g = h // (NH // KVH)
    T = t + 1
    rope, gemm, attn = f"layers.{L}.self_attn.rotary_emb.rope", f"layers.{L}.self_attn.qkv_proj.gemm", f"layers.{L}.self_attn.attn.attn"

    def need(tt: int, op: str, port: int, what: str) -> tuple[str, str, int]:
        r = rows(rid, tt, op, port)
        if r is None:
            raise ValueError(f"{unit.vu_id}: {what} -- {op}/port={port} at t={tt} is not a committed row of request {rid[:8]}")
        return r

    def check(r: tuple[str, str, int], start: int, what: str) -> None:
        if start + D > r[2]:
            raise ValueError(f"{unit.vu_id}: {what} row {r[1]} has {r[2]} words, the head needs [{start}, {start + D})")

    q = need(t, rope, 0, "q"); check(q, h * D, "q")
    out = need(t, attn, 0, "out"); check(out, h * D, "out")
    ks = [need(c, rope, 1, f"K[{c}]") for c in range(T)]
    vs = [need(c, gemm, 0, f"V[{c}]") for c in range(T)]
    v_off = (NH + KVH) * D + g * D
    for c in range(T):
        check(ks[c], g * D, f"K[{c}]"); check(vs[c], v_off, f"V[{c}]")
    incoming = [ExpectedValue(q[0], q[1], (h * D + i,)) for i in range(D)]
    incoming += [ExpectedValue(r[0], r[1], (g * D + i,)) for r in ks for i in range(D)]
    incoming += [ExpectedValue(r[0], r[1], (v_off + i,)) for r in vs for i in range(D)]
    outgoing = [ExpectedValue(out[0], out[1], (h * D + i,)) for i in range(D)]
    sub = head_subcircuit_digest(T, D, head_chain_kinds(T, D, gate_set))
    return ExpectedObligation(tuple(incoming), tuple(outgoing), sub, (None,) * D, {}, f"{head_kind_id(T, D)} {unit.vu_id}")


def _slice(row: Row, start: int, count: int, what: str, unit: LocatedUnit) -> np.ndarray:
    if row.width != 16:
        raise BuildRefusal(f"{unit.vu_id}: {what} row {row.canonical_id} is {row.dtype}, the head body reads bf16")
    if start + count > row.count:
        raise BuildRefusal(f"{unit.vu_id}: {what} row {row.canonical_id} has {row.count} words, the head needs [{start}, {start + count})")
    return np.asarray(row.words, dtype=np.uint16).reshape(-1)[start:start + count]


class _Trees:
    """Distinct commitments of one statement in first-use order: record trees (by id) and prover-side packed trees."""

    def __init__(self) -> None:
        self.refs: list[Any] = []
        self.index: dict[Any, int] = {}
        self.openers: list[Callable[[int], Any]] = []

    def record_tree(self, record: Record, tree_id: str, unit: LocatedUnit) -> int:
        key = ("record", tree_id)
        if key not in self.index:
            from veritor.protocol.proofs.statement import CommitmentRef

            ct = record.tree(tree_id)
            if ct is None:
                raise BuildRefusal(f"{unit.vu_id}: tree {tree_id!r} is not a committed tree of the record")
            self.index[key] = len(self.refs)
            self.refs.append(CommitmentRef(ct.owner, ct.ids_digest, ct.root, ct.leaf_count))
            self.openers.append(lambda rank, t=tree_id: record.opened(t, rank))
        return self.index[key]

    def packed_tree(self, label: str, builder: Any) -> int:
        key = ("packed", label)
        if key not in self.index:
            self.index[key] = len(self.refs)
            self.refs.append(builder.ref())
            self.openers.append(builder.opened)
        return self.index[key]


class AttentionHeadBuilder:
    """``BUILDERS["AttentionHead_v1|call"]`` (flagged).  ``unit`` needs ``request_id``, ``t``, ``layer``, ``head``."""

    kind = HEAD_KIND

    def build(self, unit: LocatedUnit, record: Record) -> Built:
        from veritor.core.gates import encode_value
        from veritor.interface.expected import ExpectedObligation, ExpectedValue
        from veritor.interface.lowering import _lower_subcircuit
        from veritor.protocol.proofs.packed import PackedWitness, packed_table
        from veritor.protocol.proofs.schema import VLLM_LEAF_FRAMING, packed_schema
        from veritor.protocol.proofs.statement import Obligation, PositionRange, PositionRef, make_statement

        ag, pb = _bench()
        if base_kind(unit.kind) != HEAD_KIND:
            raise BuildRefusal(f"head builder given a unit of kind {unit.kind!r}")
        if unit.layer is None or unit.head is None:
            raise BuildRefusal(f"{unit.vu_id}: the head unit needs layer and head; got layer={unit.layer} head={unit.head}")
        C = record.config
        NH, KVH, D = int(C["NH"]), int(C["KVH"]), int(C["D"])
        L, h, t, rid = int(unit.layer), int(unit.head), int(unit.t), unit.request_id
        if not 0 <= h < NH:
            raise BuildRefusal(f"{unit.vu_id}: head {h} outside [0, {NH})")
        g = h // (NH // KVH)
        T = t + 1
        rope, gemm, attn = f"layers.{L}.self_attn.rotary_emb.rope", f"layers.{L}.self_attn.qkv_proj.gemm", f"layers.{L}.self_attn.attn.attn"

        def need(tt: int, op: str, port: int, what: str) -> Row:
            r = record.row(rid, tt, op, port)
            if r is None:
                raise BuildRefusal(f"{unit.vu_id}: {what} -- {op}/port={port} at t={tt} is not a committed row of request {rid[:8]}",
                                   missing=((f"ru/{rid}", f"{op}/port={port} t={tt}", None),))
            return r

        q_row = need(t, rope, 0, "q")
        out_row = need(t, attn, 0, "out")
        k_rows = [need(c, rope, 1, f"K[{c}]") for c in range(T)]
        v_rows = [need(c, gemm, 0, f"V[{c}]") for c in range(T)]
        q = _slice(q_row, h * D, D, "q", unit)
        out = _slice(out_row, h * D, D, "out", unit)
        v_off = (NH + KVH) * D + g * D
        K = np.stack([_slice(r, g * D, D, f"K[{c}]", unit) for c, r in enumerate(k_rows)])
        V = np.stack([_slice(r, v_off, D, f"V[{c}]", unit) for c, r in enumerate(v_rows)])

        gs = fa2_gate_set()
        chain, folded, _ = ag.fold_head(q, K, V, BN, gs)
        if not np.array_equal(np.asarray(folded, dtype=np.uint16), out):
            raise BuildRefusal(f"{unit.vu_id}: the registered AttentionHead fold of the committed q/K/V differs from the committed out row "
                               f"{out_row.canonical_id}[{h * D}:{(h + 1) * D}] -- honest-completion mismatch (HC), no statement is emitted")
        nb = len(chain)

        # prover-side trees: consts (zero16, zero32, scale_log2(D)) and the state chain
        consts = pb.PackedTreeBuilder("consts")
        consts.add("const/zero16", "bfloat16", "[1]", b"\x00\x00")
        consts.add("const/zero32", "float32", "[1]", b"\x00\x00\x00\x00")
        consts.add(f"const/scale_log2_d{D}", "float32", "[1]", np.asarray([ag.scale_log2_bits(D)], dtype="<u4").tobytes())
        state_words = 1 + 4 + D
        state = pb.PackedTreeBuilder(f"{STATE_TAG}/{rid}/t={t}/l={L}/h={h}")
        for j, (_, st) in enumerate(chain):
            state.add(f"{STATE_TAG}/step={j}", "float32", f"[{state_words}]", pb.u32_leaf(st.words()))

        trees = _Trees()
        programs: dict[str, Any] = {}
        program_meta: dict[str, tuple[str, str]] = {}
        chain_kinds: list[str] = []
        obligations: list[Obligation] = []
        desc16 = pb.descriptor16

        def program_of(key: tuple, sub) -> Any:
            if key not in programs:
                programs[key] = _lower_subcircuit(sub, gs)[0]
            return programs[key]

        for step, (b, _) in enumerate(chain):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            nvis, first = hi - lo, step == 0
            sub, ports = ag.cached_block(D, BN, nvis, first)
            program = program_of(("block", nvis, first), sub)
            chain_kinds.append(program.kind.hex())
            program_meta[program.kind.hex()] = (f"AttnBlock_v1{{D={D},NVIS={nvis},FIRST={first}}}", str(sub.digest) if hasattr(sub, "digest") else "")
            entries: list[Any] = [PositionRange(trees.record_tree(record, q_row.tree, unit), q_row.rank, q_row.rank, desc16(q_row.count), h * D, D, 1)]
            for c in range(lo, hi):
                r = k_rows[c]
                entries.append(PositionRange(trees.record_tree(record, r.tree, unit), r.rank, r.rank, desc16(r.count), g * D, D, 1))
            for c in range(lo, hi):
                r = v_rows[c]
                entries.append(PositionRange(trees.record_tree(record, r.tree, unit), r.rank, r.rank, desc16(r.count), v_off, D, 1))
            ci = trees.packed_tree("consts", consts)
            if ports.pad is not None:
                entries.append(PositionRef(ci, 0, 0, packed_schema(VLLM_LEAF_FRAMING, 16, True, 1, 0), expected=encode_value(16, 0)))
            if ports.zero32 is not None:
                entries.append(PositionRef(ci, 1, 1, packed_schema(VLLM_LEAF_FRAMING, 32, True, 1, 0), expected=encode_value(32, 0)))
            entries.append(PositionRef(ci, 2, 2, packed_schema(VLLM_LEAF_FRAMING, 32, True, 1, 0), expected=encode_value(32, ag.scale_log2_bits(D))))
            si = trees.packed_tree("state", state)
            if not first:
                entries.append(PositionRange(si, step - 1, step - 1, pb.descriptor32(state_words), 0, state_words, 1))
            entries.append(PositionRange(si, step, step, pb.descriptor32(state_words), 0, state_words, 1))
            n_in = ports.count
            obligations.append(Obligation(record.session, record.compiled, step, unit.ru_index, program.kind, tuple(trees.refs), tuple(entries),
                                          tuple(range(n_in)), tuple(n_in + m for m in range(state_words))))
        epi_sub = ag.attn_epilogue_subcircuit(D)
        epi = program_of(("epilogue",), epi_sub)
        chain_kinds.append(epi.kind.hex())
        program_meta[epi.kind.hex()] = (f"Fa2Epilogue_v1{{D={D}}}", "")
        si = trees.packed_tree("state", state)
        oi = trees.record_tree(record, out_row.tree, unit)
        entries = (PositionRange(si, nb - 1, nb - 1, pb.descriptor32(state_words), 1, 4 + D, 1),
                   PositionRange(oi, out_row.rank, out_row.rank, desc16(out_row.count), h * D, D, 1))
        obligations.append(Obligation(record.session, record.compiled, nb, unit.ru_index, epi.kind, tuple(trees.refs), entries,
                                      tuple(range(4 + D)), tuple(4 + D + m for m in range(D))))
        # every obligation names the same commitment tuple (built incrementally above): re-emit with the final tuple
        refs = tuple(trees.refs)
        obligations = [Obligation(o.session, o.compiled, o.unit, o.replay_unit, o.kind, refs, o.positions, o.inputs, o.outputs) for o in obligations]
        statement = make_statement(gs.id, bytes.fromhex(str(gs.digest)), 32, list(programs.values()), obligations)

        table = packed_table(statement.expand())
        by_root = {(ref.root, ref.count): i for i, ref in enumerate(refs)}
        witness = PackedWitness(tuple(tuple(trees.openers[by_root[(tr.commitment.root, tr.commitment.count)]](rank) for rank in tr.ranks) for tr in table.trees))

        sub_digest = head_subcircuit_digest(T, D, chain_kinds)
        incoming: list[Position] = [(q_row.tree, q_row.canonical_id, h * D + i) for i in range(D)]
        incoming += [(r.tree, r.canonical_id, g * D + i) for r in k_rows for i in range(D)]
        incoming += [(r.tree, r.canonical_id, v_off + i) for r in v_rows for i in range(D)]
        outgoing: list[Position] = [(out_row.tree, out_row.canonical_id, h * D + i) for i in range(D)]
        # the scale_log2 constant is pinned inside the statement (a PositionRef with `expected`) and checked by the verifier's view
        # recovery against the registered value for D; it is a function of D, so the head digest (T, D, chain) already binds it.
        view = ObligationView(gs.id, sub_digest, tuple(incoming), tuple(outgoing))
        expectation = ExpectedObligation(tuple(ExpectedValue(tr, cid, (i,)) for tr, cid, i in incoming), tuple(ExpectedValue(tr, cid, (i,)) for tr, cid, i in outgoing),
                                         sub_digest, (None,) * D, {}, f"{head_kind_id(T, D)} {unit.vu_id}")
        notes = {"kind": head_kind_id(T, D), "T": T, "D": D, "blocks": nb, "obligations": len(obligations), "family": gs.id,
                 "guest": "33cba058 (fa2-attn, UNAPPROVED)", "partition": "PROPOSED head-level VU (S55) -- not adopted",
                 "state_tree": "prover-side, chained by shared openings; not a record commitment"}
        return Built(head_kind_id(T, D), statement, witness, {unit.vu_id: expectation}, (view,), program_meta, notes)


def head_views_of_raw_statement(statement: Any, tree_of_root: Mapping[bytes, str], id_at_rank: Callable[[str, int], str | None],
                                *, D: int, gate_set=None) -> tuple[ObligationView, ...]:
    """Verifier side: the head-level :class:`ObligationView` rows of a raw fa2 statement, from committed data only.

    Groups the obligations by their (prover-side) state tree root, checks the chain -- block ``j`` reads the state block ``j-1``
    wrote (same root, rank ``j-1``), block 0 is FIRST (reads no state), the epilogue reads the last block's state and every block
    reads the same q positions -- and collapses each intact chain into one view: incoming = q words, K words, V words in block order
    (positions of committed leaves), outgoing = the epilogue's ``out`` words; the pinned ``scale_log2`` const must be the registered value for ``D``.  Any break
    is a :class:`CoverageError`; a root the verifier cannot name that is not a state/const tree is one too."""

    from veritor.protocol.proofs.packed import parse_packed_schema

    ag, _ = _bench()
    gs = gate_set if gate_set is not None else fa2_gate_set()
    from veritor.interface.lowering import _lower_subcircuit

    epi_kind = _lower_subcircuit(ag.attn_epilogue_subcircuit(D), gs)[0].kind.hex()
    block_kind: dict[tuple[int, bool], str] = {}

    def block_meta(kind_hex: str, n_committed_in: int, first: bool, unit: int) -> tuple[int, bool]:
        """(NVIS, FIRST) of a block obligation from its shape -- committed inputs = q (D) + K (NVIS*D) + V (NVIS*D); FIRST = reads no
        prior state -- then the fixed circuit's program for that shape MUST be the obligation's kind (fail closed)."""

        if (n_committed_in - D) % (2 * D) or n_committed_in < 3 * D:
            raise CoverageError(f"obligation {unit}: {n_committed_in} committed input words is not D + 2*NVIS*D for D={D}")
        nvis = (n_committed_in - D) // (2 * D)
        if not 1 <= nvis <= BN:
            raise CoverageError(f"obligation {unit}: NVIS={nvis} outside [1, {BN}]")
        key = (nvis, first)
        if key not in block_kind:
            try:
                block_kind[key] = _lower_subcircuit(ag.cached_block(D, BN, nvis, first)[0], gs)[0].kind.hex()
            except Exception as e:  # noqa: BLE001 -- a shape the fixed circuit cannot lower is not a block of it
                raise CoverageError(f"obligation {unit}: AttnBlock<D={D},NVIS={nvis},FIRST={first}> is not a program of the fixed circuit: {e}") from e
        if block_kind[key] != kind_hex:
            raise CoverageError(f"obligation {unit}: program {kind_hex[:12]} is not AttnBlock<D={D},NVIS={nvis},FIRST={first}> ({block_kind[key][:12]})")
        return nvis, first

    chains: dict[bytes, dict[str, Any]] = {}
    for ob in statement.obligations:
        ob = ob.expand()
        # split positions by commitment: committed record leaves vs prover trees
        state_root = None
        rec: list[tuple[int, Position]] = []
        pins: dict[int, str] = {}
        state_ranks: list[int] = []
        for k, p in enumerate(ob.positions):
            c = ob.commitments[p.commitment]
            tree = tree_of_root.get(c.root)
            if tree is not None:
                cid = id_at_rank(tree, p.rank)
                packed = parse_packed_schema(p.schema)
                if cid is None or p.position != p.rank or packed is None:
                    raise CoverageError(f"obligation {ob.unit}: rank {p.rank} of {tree!r} is not a committed packed leaf")
                rec.append((k, (tree, cid, packed.index)))
            elif p.expected is not None:
                pins[k] = p.expected.hex()
            else:
                if state_root is not None and state_root != c.root:
                    raise CoverageError(f"obligation {ob.unit}: reads two different uncommitted trees")
                state_root = c.root
                state_ranks.append(p.rank)
        if ob.kind.hex() == epi_kind:
            if state_root is None or len(set(state_ranks)) != 1:
                raise CoverageError(f"obligation {ob.unit}: epilogue without exactly one state leaf")
            chains.setdefault(state_root, {"blocks": {}, "epilogue": None})
            if chains[state_root]["epilogue"] is not None:
                raise CoverageError(f"two epilogues on state root {state_root.hex()[:12]}")
            chains[state_root]["epilogue"] = (state_ranks[0], [pos for k, pos in rec if k in set(ob.outputs)])
            continue
        if state_root is None:
            raise CoverageError(f"obligation {ob.unit}: block without a state tree")
        ranks = sorted(set(state_ranks))
        if len(ranks) not in (1, 2) or (len(ranks) == 2 and ranks[1] != ranks[0] + 1):
            raise CoverageError(f"obligation {ob.unit}: block state ranks {ranks} do not chain")
        first = len(ranks) == 1
        step = ranks[-1]
        if (first and step != 0) or (not first and step == 0):
            raise CoverageError(f"obligation {ob.unit}: FIRST={first} at chain step {step}")
        inputs = set(ob.inputs)
        in_pos = [pos for k, pos in rec if k in inputs]
        nvis, first = block_meta(ob.kind.hex(), len(in_pos), first, ob.unit)
        q_pos, k_pos, v_pos = in_pos[:D], in_pos[D:D + nvis * D], in_pos[D + nvis * D:D + 2 * nvis * D]
        scale = [v for k, v in pins.items() if v != "0000" and v != "00000000"]
        ch = chains.setdefault(state_root, {"blocks": {}, "epilogue": None})
        if step in ch["blocks"]:
            raise CoverageError(f"state root {state_root.hex()[:12]}: two blocks at step {step}")
        ch["blocks"][step] = {"q": q_pos, "k": k_pos, "v": v_pos, "nvis": nvis, "kind": ob.kind.hex(), "scale": scale[0] if scale else None}

    views: list[ObligationView] = []
    for root, ch in chains.items():
        steps = sorted(ch["blocks"])
        if ch["epilogue"] is None or steps != list(range(len(steps))) or not steps:
            raise CoverageError(f"state root {root.hex()[:12]}: incomplete chain (blocks {steps}, epilogue {'present' if ch['epilogue'] else 'absent'})")
        last_rank, out_pos = ch["epilogue"]
        if last_rank != steps[-1]:
            raise CoverageError(f"state root {root.hex()[:12]}: epilogue reads state {last_rank}, the last block wrote {steps[-1]}")
        q0 = ch["blocks"][0]["q"]
        if any(ch["blocks"][s]["q"] != q0 for s in steps):
            raise CoverageError(f"state root {root.hex()[:12]}: blocks read different q rows")
        T = sum(ch["blocks"][s]["nvis"] for s in steps)
        # blocks are visited last -> first: step 0 is the highest key block
        k_pos = [p for s in reversed(steps) for p in ch["blocks"][s]["k"]]
        v_pos = [p for s in reversed(steps) for p in ch["blocks"][s]["v"]]
        kinds = [ch["blocks"][s]["kind"] for s in steps] + [epi_kind]
        scale = ch["blocks"][0]["scale"]
        if scale is None or any(ch["blocks"][s]["scale"] != scale for s in steps):
            raise CoverageError(f"state root {root.hex()[:12]}: scale_log2 pin missing or inconsistent")
        if scale != f"{ag.scale_log2_bits(D):08x}":
            raise CoverageError(f"state root {root.hex()[:12]}: scale_log2 pin {scale} is not the registered {ag.scale_log2_bits(D):08x} for D={D}")
        views.append(ObligationView(statement.gate_set_id, head_subcircuit_digest(T, D, kinds), tuple(q0 + k_pos + v_pos), tuple(out_pos)))
    return tuple(views)


def verify_locally(built: Built, gate_set=None) -> bool:
    """The transparent backend's verdict on the built statement + witness (``False`` on rejection)."""

    from veritor.protocol.messages import Reject
    from veritor.protocol.proofs import TransparentBackend
    from veritor.protocol.proofs.packed import encode_packed_witness

    gs = gate_set if gate_set is not None else fa2_gate_set()
    try:
        return bool(TransparentBackend(gs).verify(built.statement, encode_packed_witness(built.witness)))
    except Reject:
        return False
