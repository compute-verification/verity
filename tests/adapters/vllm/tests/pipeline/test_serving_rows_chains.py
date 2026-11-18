"""A sparse-MLA row cut into its heads' key-block chains (`pipeline.serving_rows`' ``chain`` layout, `chain_rows`): a
``SparseMla_v1`` row holds a ``SparseMlaHeadBlock_v1`` FIRST template at its first visited block's key count, the full-block
template for every later block and one ``SparseMlaHeadEnd_v1`` a head (at GLM-5.3's statics and T = 130, per head, a 2-key first
block, two 64-key blocks and the end), and a window of the three at a small static (T = 40 keys in blocks of 8, 16 and 16, two
heads):

* reads each block's operands off the row and its state off the twin, whose end is the row's served words, so each unit's
  Definition on its rows, parsed as Flock's stager parses them, is its committed output;
* wires every later block's state and every end's lanes and O to the output rows of the unit of the block before it, commits it
  under that row's salt, so the verifier's check on the public bodies holds, and refuses a state that is not its writer's, a
  row whose served words are not its chain's end, and a partition that leaves out a block's template.

`test_a_window_of_one_row_commits_and_its_wires_hold` takes minutes, all of it the window's population program lowering the
three templates: it is the one test of the committed window and its public bodies, which the others reach only through the rows.

A whole key block proved under ``--zk`` is its Boolean body: `test_a_whole_block_s_boolean_body_is_staged_in_the_form_verify_zk_takes`
composes it as serving's stager does and holds its statement's form to `one_stage.proofs.ks_form_why`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager
from verity.commitments.periods import interface as IF
from verity.commitments.periods import rows as CS
from verity_vllm.pipeline import chain_rows as CH
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity.circuits.ops.boolean import versions as BB
from verity_vllm.definitions.numerics import sparse_mla_split_twins as TW
from verity_vllm.definitions.registry import boolean_sparse_mla as BM
from verity_vllm.definitions.registry import hopper
from verity_vllm.definitions.registry import sparse_mla as M
from verity_vllm.query.program_view import spec_statics

from verity.commitments import randomness as RND
from verity_flock.class_statement import UnitStager

from verity_vllm.pipeline import chain_rows as CH
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.query.program_view import spec_statics
from verity.circuits.defs import bind
from verity.circuits.codec import descriptor_id
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

FA3 = {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}
BOOL = {"DOT": BM.SPARSE_MLA_GLM53["DOT"], "INV": BM.SPARSE_MLA_GLM53["INV"]}
TOY = {"T": 40, "NH": 2, "DR": 16, "DV": 32, "BN": 16, "SCALE": 0.0625}
#: TOY at DV = 16, the least a block binds at
NARROW = {**TOY, "DV": 16}
GLM = {"T": 130, "NH": 8, "DR": 64, "DV": 512, "BN": 64, "SCALE": 0.0625}
SERVED = ('SparseMla_v1{T=130,NH=8,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},INV={"fn":"Fa3InvSum_v1"},'
          'SCALE={"f64":"0x1.0000000000000p-4"}}')


#: `chain_rows.pieces`' refusal of a port reading several templates' trees
NAME = "a wired port reads one template's tree"

def row(S: dict = TOY, **over) -> str:
    return descriptor_id(bind(M.SparseMla, **{**S, **over}, **FA3))


def block(nvis: int | None = None, S: dict = TOY, boolean: bool = False) -> str:
    """The FIRST block template at ``nvis`` keys, or with None the full block after it."""
    st = {x: S[x] for x in ("DR", "DV", "BN", "SCALE")}
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadBlockV2, **at, **st, DOT=BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadBlock, **at, **st, DOT=FA3["DOT"]))


def end(S: dict = TOY, boolean: bool = False) -> str:
    return descriptor_id(bind(BM.SparseMlaHeadEndV2, DV=S["DV"], INV=BOOL["INV"]) if boolean
                         else bind(M.SparseMlaHeadEnd, DV=S["DV"], INV=FA3["INV"]))


F8, B16, E = block(8), block(), end()
TMPLS = [F8, B16, E]
PER_CHUNK = {F8: 2, B16: 3, E: 4}


class Served:
    """``n`` rows of ``spec`` (row p's q_nope, q_rope and kv are arguments 3p, 3p + 1 and 3p + 2) and the store of their served
    outputs, the twin's end of its chain (`sparse_mla_split_twins.chain_rows`, `head_end_rows`)."""

    def __init__(self, spec: str = row(), n: int = 1, seed: int = 3):
        S = spec_statics(spec)
        T, NH, DR, DV, BN = (int(S[x]) for x in ("T", "NH", "DR", "DV", "BN"))
        rng = np.random.default_rng(seed)

        def bf16(shape, s=1.0):
            return ((rng.standard_normal(shape).astype(np.float32) * np.float32(s)).view(np.uint32) >> 16).astype(np.uint16)

        self.ops, self.rows, ys, self.states = {}, [], [], []
        for p in range(n):
            qn, qr, kv = bf16((NH, DV)), bf16((NH, DR)), bf16((T, DV + DR), 0.5)
            states, nf = TW.chain_rows(qn, qr, kv, T=T, DR_=DR, DV=DV, BN=BN, scale=float(S["SCALE"]))
            assert not nf.any()
            self.states.append(states)
            ys.append(np.asarray(TW.head_end_rows(states[-1][1], states[-1][2]), dtype=np.uint16).reshape(-1))
            self.ops.update({3 * p: qn.reshape(-1), 3 * p + 1: qr.reshape(-1), 3 * p + 2: kv.reshape(-1)})
            r = {"i": 5 + p, "spec": spec, "args": [[["x", 3 * p + g, 0, self.ops[3 * p + g].size]] for g in range(3)]}
            d = {"statics": S, "op_path": "model.layers.3.self_attn.attn", "members": [("out", NH * DV, "bf16")], "row": p, "step": 4}
            self.rows.append(("r0", None, r, d))
        self.y = np.concatenate(ys)

    def slices(self, _pi, _store, _weights, _rid, _arrive, g):
        """`serving_rows._slices`: an argument group's elements."""
        return [self.ops[g[0][1]]]

    def words(self, rid, step, op, member, lo, hi):
        return self.y[lo:hi], None


def members(s: Served, tmpls=TMPLS, per_chunk=PER_CHUNK) -> list[CS.Member]:
    return CH.wire(SRW.members(tmpls, [s.rows] * len(tmpls), None, per_chunk), tmpls, [s.rows] * len(tmpls))


def window(s: Served, tmpls=TMPLS, per_chunk=PER_CHUNK, design: dict | None = None) -> CS.Binding:
    """The window of ``s``'s rows over ``tmpls`` (`members`) under ``design`` (one unit of each template by default)."""
    ms = members(s, tmpls, per_chunk)
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    return CS.Binding(*IF.listed_program(defs, ()), design or RND.parse_design("stratified:1"), "ab" * 32, tuple(ms))


def read(s: Served, ms, tmpls=TMPLS) -> list[tuple[dict, list]]:
    SRW._slices = s.slices
    return SRW.read_all(ms, [s.rows] * len(tmpls), s, None)


@pytest.fixture(autouse=True)
def _slices(monkeypatch):
    """`serving_rows._slices` back after each test, which `read` replaces."""
    monkeypatch.setattr(SRW, "_slices", SRW._slices)


def stager(t: str) -> UnitStager:
    st = object.__new__(UnitStager)     # reference_check reads only these: the circuit needn't be composed
    st.definition, st.name, st.public_inputs, st.in_rows = SRW.template(t).fn, "chain-test", {}, CR.unit_rows(SRW.template(t))[0]
    return st


def units(ms, tmpls, S: dict = TOY, rows: int = 1) -> dict[tuple, int]:
    """The global unit of each (row, head, place) in its head's chain, place j the block j in visit order, nb the end: a
    template's units row by row, then head, then its places."""
    nb = -(-S["T"] // S["BN"])
    out: dict[tuple, int] = {}
    for t in tmpls:
        u = min(m.base for m in ms if m.template == t)
        k = CH.kind(SRW.template(t))
        for p in range(rows):
            for h in range(S["NH"]):
                for j in ([0] if k == "first" else range(1, nb) if k == "block" else [nb]):
                    out[(p, h, j)], u = u, u + 1
    return out


def test_a_chain_row_holds_its_first_block_its_later_blocks_and_its_end():
    """The units `sparse_mla.head_chain` cuts the row into, each a ``chain`` template; a whole block's EX2 reads go in lookup
    slots, a partial first block's and the end's inline.  Each unit's Boolean version (`boolean_version`) is a template of the
    same word, and a block's, its EX2 gates, is flat at every key count."""
    assert SRW.unit_templates(row())[:4] == ((row(), 1), (F8, 2), (B16, 4), (E, 2))
    assert row(GLM) == SERVED
    assert SRW.unit_templates(SERVED)[:4] == ((SERVED, 1), (block(2, GLM), 8), (block(None, GLM), 16), (end(GLM), 8))
    assert SRW.unit_templates(row(T=16))[:3] == ((row(T=16), 1), (block(16), 2), (E, 2))
    assert SRW.unit_templates(row(NARROW)) == ((row(NARROW), 1), (block(8, NARROW), 2), (block(None, NARROW), 4), (end(NARROW), 2))
    assert [(SRW.template(t).layout, SRW.template(t).word, CR.reads(SRW.template(t))) for t in TMPLS] == [
        ("chain", F8, "inline"), ("chain", B16, "slots"), ("chain", E, "inline")]
    assert [CR.reads(SRW.template(t)) for t in (block(2, GLM), block(64, GLM), block(None, GLM), end(GLM))] == [
        "inline", "slots", "slots", "inline"]
    bits = [block(8, boolean=True), block(boolean=True), end(boolean=True)]
    assert [SRW.template(t).word for t in bits] == TMPLS
    assert [descriptor_id(BB.boolean_version(SRW.template(t).fn)) for t in TMPLS] == bits
    assert [CR.reads(SRW.template(t)) for t in (*bits, block(64, GLM, boolean=True), block(None, GLM, boolean=True))] == ["inline"] * 5
    assert [CH.kind(SRW.template(t)) for t in TMPLS] == ["first", "block", "end"]
    DV, DR = TOY["DV"], TOY["DR"]
    assert CR.unit_rows(SRW.template(B16)) == ([("in0", V2, 16 * DV, ["q_nope"]), ("in1", V2, 16 * DR, ["q_rope"]),
                                                ("in2", V2, 16 * 16 * (DV + DR), ["kv"]), ("in3", V2, 32, ["m_prev"]),
                                                ("in4", V2, 4 * 32, ["l_prev"]), ("in5", V2, 32 * DV, ["o_prev"])],
                                               [(V2, (0,)), (V2, (1,)), (V2, (2,))])
    assert CR.unit_rows(SRW.template(E)) == ([("in0", V2, 4 * 32, ["l"]), ("in1", V2, 32 * DV, ["o"])], [(V2, (0,))])
    with pytest.raises(ValueError, match=re.escape("a row of T=40 keys in NS=1 splits holds no unit ('block', 16, True) of it")):
        CH.positions(SRW.template(block(16)), spec_statics(row()))


def test_a_row_whose_served_words_are_not_its_chain_s_end_is_refused():
    s = Served()
    s.y[3] ^= np.uint16(1)
    with pytest.raises(ValueError, match="the twin's end of its 3 blocks is not the row's served out words"):
        read(s, SRW.members(TMPLS, [s.rows] * len(TMPLS), None, PER_CHUNK))


def test_a_partition_without_a_block_s_template_is_refused():
    s = Served()
    for tmpls, j in (([F8, E], 2), ([B16, E], 0)):
        with pytest.raises(ValueError, match=re.escape(f"block {j} (visit order) is the output of a block template the partition "
                                                       "does not name")):
            members(s, tmpls)
    assert all(m.wired == () for m in members(s, [F8]))


def composed(t: str, d: Path) -> UnitStager:
    """Template ``t``'s statement as serving's stager composes it (its rows `chunk_rows.unit_rows`, its table reads laid out
    as `chunk_rows.statement` says, bound as ``compose_circuit.py --template`` binds it), its circuit ``d/circuit.txt``."""
    tp = SRW.template(t)
    in_rows, out_rows = CR.unit_rows(tp)
    return UnitStager(tp.fn, "unit/serving-rows", in_rows, out_rows, circuit=d / "circuit.txt",
                      bind={"partition": "0" * 128, "program": "0" * 128}, **CR.statement(tp))


def test_a_later_block_reading_its_first_and_its_later_blocks_is_refused_by_name():
    """A later block's carried state is its first block's output at the second block and a later block's after it: one
    port reading two templates' trees, which `chain_rows.pieces` refuses by name."""
    with pytest.raises(ValueError, match=NAME):
        members(Served(n=2))
