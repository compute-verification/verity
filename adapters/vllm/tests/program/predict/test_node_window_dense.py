"""verity_vllm.program.predict.node_window_dense: the dense families lowered into GLM-5.3's node window at TP8 rank 0 beside
the frame's linears and norms (`node_window.FAMILIES`), at a 32-id prompt, 2 outputs and one decoder layer. Each family is
its registered split over the templates `serving_templates.unit_templates` lists: SiluMul and RoPE the row itself; the
all-reduce, the residual add and the top-k sum in column blocks of 192, the norms' apply width; the fused all-reduce's
norm as its parts inside a layer and its ``FINAL`` cut outside; the LM head and the all-gather as their slices. A
collective's own partial is one row of its cut ``parts`` and is wired; the other ranks' rows are free. Every template's
unit is a proof block Flock's composer proves, and the Circuit passes its wiring and input checks.
"""
from __future__ import annotations

import dataclasses
import re

import pytest
from verity_vllm.pipeline import serving_templates as ST
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_dense as ND

from tests.program.predict.test_node_window import LP, glm_request
from verity.circuits.codec import descriptor_id
from verity.commitments.windows import interface as IF

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES}
WIDTH = 6144
COLS = 192


@pytest.fixture(scope="module")
def program():
    return glm_request(LP, 1).program.fn


@pytest.fixture(scope="module")
def window(program):
    return NW.node_window(program, families=FAMILIES)


def _tid(circuit: IF.Circuit, pattern: str) -> str:
    (tid,) = [t for t in circuit.units if re.fullmatch(pattern, t)]
    return tid


def _pairs(circuit: IF.Circuit) -> set[tuple[str, str, str]]:
    return {(w.writer, w.reader, w.port) for w in circuit.wires}


def _gone(census: dict, *families: str) -> None:
    for fam in families:
        assert fam not in census["unlowered"] and fam not in census["free"], fam


def test_silu_mul(window):
    """SiluMul is one unit a step reading gate_up's 3,072 coordinates whole (32 chunks of 96, then the row), and the
    down_proj's quantisers (DeepGEMM's at the prefill, FlashInfer's at the decode) read its row."""
    circuit, w = window
    silu = _tid(circuit, r"SiluMul_v2\{I=1536\}")
    level1, level2 = _tid(circuit, r"AllGather_v1\{WORLD=96,N=1\}"), _tid(circuit, r"AllGather_v1\{WORLD=32,N=96\}")
    gate_up = _tid(circuit, r"ScaledMmFp8BlockSharedScale_v1\{K=6144,N=1,G=128\}")
    assert circuit.units[silu] == 2 and circuit.units[level2] == 2
    assert {(gate_up, level1, "parts"), (level1, level2, "parts"), (level2, silu, "gu"),
            (silu, _tid(circuit, r"Fp8GroupQuantUe8m0_v1\{K=1536,G=128\}"), "x"),
            (silu, _tid(circuit, r"FlashInferFp8Quant1x128_v1\{K=1536\}"), "x")} <= _pairs(circuit)
    for tid, fn in w.b.fns.items():
        IF.check_proof_block(fn, circuit.cuts.get(tid))
    _gone(w.census(circuit), "SiluMul")


def test_allreduce_norm(window):
    """The fused all-reduce's norm is 32 residual blocks of 192, each reading its columns of the 8 ranks' partials as
    ``parts`` cut 8 ways (rank 0's own, o_proj's or down_proj's coordinates gathered, wired; the 7 peers' rows free), then
    inside the layer (after o_proj) a sum of squares and 32 apply blocks, and outside it (after down_proj, the model's
    final norm) its whole norm, which the LM head reads."""
    circuit, w = window
    resid = {s: _tid(circuit, rf"AllReduceResidualMnnvl_v1\{{WORLD=8,N=192,SHOT={s}\}}") for s in (1, 2)}
    apply = {s: _tid(circuit, rf"RmsNormMnnvlApply_v1\{{G=192,SHOT={s}\}}") for s in (1, 2)}
    sumsq = {s: _tid(circuit, rf"RmsNormMnnvlSumSq_v1\{{N=6144,SHOT={s},.*\}}") for s in (1, 2)}
    final = {s: _tid(circuit, rf"RmsNormMnnvl_v1\{{N=6144,SHOT={s},.*\}}") for s in (1, 2)}
    own = _tid(circuit, r"AllGather_v1\{WORLD=2,N=96\}")
    whole = _tid(circuit, r"AllGather_v1\{WORLD=32,N=192\}")
    head = _tid(circuit, r"Gemm_v2\{K=6144,N=1,.*\}")
    for s in (1, 2):
        assert circuit.units[resid[s]] == 2 * WIDTH // COLS and circuit.units[apply[s]] == WIDTH // COLS
        assert circuit.units[sumsq[s]] == 1 and circuit.units[final[s]] == 1
        assert circuit.cuts[resid[s]] == {"parts": 8}
        assert {(own, resid[s], "parts"), (resid[s], apply[s], "r"), (resid[s], whole, "parts"), (whole, sumsq[s], "r"),
                (sumsq[s], apply[s], "rs"), (whole, final[s], "r"), (final[s], head, "x")} <= _pairs(circuit)
    assert {w.cut for w in circuit.wires if w.reader in resid.values() and w.port == "parts"} == {(0, 1)}
    peers = [f for f in w.free if f.tid in resid.values() and f.port == "parts"]
    assert {f.kind for f in peers} == {"peer"} and {f.rows for f in peers} == {(i, 1, 8) for i in range(1, 8)}
    assert (whole, _tid(circuit, r"FlashInferFp8Quant1x128_v1\{K=6144\}"), "x") in _pairs(circuit)
    _gone(w.census(circuit), "AllReduceRmsNormMnnvl")


def test_allreduce(window):
    """The embedding's all-reduce is column blocks of 192 (the prefill's 32 tokens' 1,024 and the decode's 32), which the
    first norm's apply blocks read as they are and its sum of squares through one gather of 32; its own partial is the
    embedding's, free until `EmbeddingShard` is lowered, and the peers' rows are free."""
    circuit, w = window
    ar = {s: _tid(circuit, rf"AllReduceMnnvl_v1\{{WORLD=8,N=192,SHOT={s}\}}") for s in (1, 2)}
    assert circuit.units[ar[2]] == LP * WIDTH // COLS and circuit.units[ar[1]] == WIDTH // COLS
    whole = _tid(circuit, r"AllGather_v1\{WORLD=32,N=192\}")
    apply = _tid(circuit, r"RMSNormApplyCudaVector8_v1\{G=192\}")
    resid = _tid(circuit, r"AllReduceResidualMnnvl_v1\{WORLD=8,N=192,SHOT=2\}")
    assert {(ar[s], apply, "x") for s in (1, 2)} | {(ar[s], whole, "parts") for s in (1, 2)} | {(ar[2], resid, "res")} \
        <= _pairs(circuit)
    own = [f for f in w.free if f.tid in ar.values() and f.rows[0] == 0]
    assert {(f.kind, f.family) for f in own} == {("node", "EmbeddingShard")}
    census = w.census(circuit)
    assert census["free"]["EmbeddingShard"]["words"] == (LP + 1) * WIDTH
    _gone(census, "AllReduceMnnvl")


def test_lm_head_and_all_gather(window):
    """The LM head is its 19,360 one-word slices, each reading the final norm's row and its weight row; the all-gather's
    own shard reads them through three levels (160 chunks of 121, 2 of 9,680, then the row), and its 7 peers' shards are
    free. The token select that reads it stays unlowered (its row passes a proof block)."""
    circuit, w = window
    head = _tid(circuit, r"Gemm_v2\{K=6144,N=1,.*\}")
    g1, g2, g3 = (_tid(circuit, rf"AllGather_v1\{{WORLD={k},N={n}\}}") for k, n in ((121, 1), (80, 121), (2, 9680)))
    shard = _tid(circuit, r"AllGather_v1\{WORLD=1,N=19360\}")
    assert circuit.units[head] == 2 * 19360 and circuit.units[shard] == 2 * 8
    assert (circuit.units[g1], circuit.units[g2], circuit.units[g3]) == (2 * 160, 2 * 2, 2)
    assert {(head, g1, "parts"), (g1, g2, "parts"), (g2, g3, "parts"), (g3, shard, "parts")} <= _pairs(circuit)
    assert any(r.reader == head and r.port == "w" for r in circuit.registered.reads)
    peers = [f for f in w.free if f.tid == shard]
    assert {f.kind for f in peers} == {"peer"} and len(peers) == 2 * 7
    census = w.census(circuit)
    _gone(census, "Gemm", "AllGather")
    assert census["unlowered"]["TokenSelect"] == 2


def test_rope_head(window):
    """Each of the 8 heads' rotary embedding is one unit a step reading its 64 rope coordinates of q_b through one gather
    of 64 and its position's cos/sin row. Its reader, the sparse attention, is the other half's (`SparseMlaSplit`), left
    unlowered here, so no wire reads it yet."""
    circuit, w = window
    rope = _tid(circuit, r"RoPEHeadGptJ_v2\{D=64\}")
    assert circuit.units[rope] == 2 * 8
    assert (_tid(circuit, r"AllGather_v1\{WORLD=64,N=1\}"), rope, "x") in _pairs(circuit)
    assert any(r.reader == rope and r.port == "cs" for r in circuit.registered.reads)
    census = w.census(circuit)
    assert "RoPEHeadGptJ" not in census["unlowered"] and "SparseMlaSplit" in census["unlowered"]


def test_moe_cols():
    """Over an MoE layer the top-k sum is column blocks of 192 with ``y`` cut 8 ways, one row an expert, all free under
    the routed experts' GEMM (`MoeExpertFp8GemmW`, the other half's, left unlowered here); the residual add's blocks read
    it and the shared expert's down_proj coordinates (gathered), and the final fused all-reduce reads its blocks as its
    own partial."""
    circuit, w = NW.node_window(glm_request(LP, 1, moe=True).program.fn, families=FAMILIES)
    moesum, add = _tid(circuit, r"MoeSum_v1\{TOPK=8,H=192\}"), _tid(circuit, r"ResidualAdd_v1\{N=192\}")
    resid = _tid(circuit, r"AllReduceResidualMnnvl_v1\{WORLD=8,N=192,SHOT=2\}")
    assert circuit.units[moesum] == circuit.units[add] == 2 * WIDTH // COLS
    assert circuit.cuts[moesum] == {"y": 8}
    assert {(moesum, add, "r"), (_tid(circuit, r"AllGather_v1\{WORLD=2,N=96\}"), add, "x"), (add, resid, "parts")} \
        <= _pairs(circuit)
    assert {(f.family, f.rows) for f in w.free if f.tid == moesum} == {("MoeExpertFp8GemmW", (i, 1, 8)) for i in range(8)}
    census = w.census(circuit)
    _gone(census, "MoeSum", "ResidualAdd", "SiluMul")
    assert {"MoeExpertFp8Gemm", "MoeExpertFp8GemmW"} <= set(census["free"])


def test_free_rows_unwired(window):
    """No wire fills a row the census lists as free: a cut port's free rows are the rows no wire of its covers."""
    circuit, w = window
    for f in w.free:
        first, rows, of = f.rows
        assert not any(x.reader == f.tid and x.port == f.port and x.at < f.at + f.count and f.at < x.at + x.reader_units
                       and x.cut[0] < first + rows and first < x.cut[0] + x.cut[1] and circuit.cuts.get(f.tid, {}).get(
                           f.port, 1) == of for x in circuit.wires), f


def test_left_free(program):
    """The token select and the stop check stay unlowered: the select's one unit (`TokenSelect_v4` is no ``STAGES`` row)
    reads 309,760 bytes of logits, past a proof block, and the stop check's ports are 32-bit ids and no serving template's
    u16 words."""
    specs = {getattr(n.fn, "name", ""): descriptor_id(n.fn) for n in program.body.nodes}
    with pytest.raises(ValueError, match="past 2\\^27"):
        IF.check_proof_block(ST.template(specs["TokenSelect"]).fn)
    with pytest.raises(ValueError, match="commit u16 words"):
        ST.template(specs["StopCheck"])


def _after(family: str, lower, prefix: str, extra):
    """`FAMILIES` with ``family``'s lowering ``lower`` followed by ``extra(w, tid, at, units)`` on its last allocation of
    the template whose id starts ``prefix``."""
    def lowering(w, j, node):
        before = {t: len(a) for t, a in w.allocs.items()}
        lower(w, j, node)
        ((tid, (at, n, _fam)),) = [(t, a[-1]) for t, a in w.allocs.items() if t.startswith(prefix) and len(a) > before.get(t, 0)]
        extra(w, tid, at, n)
    return {**FAMILIES, family: lowering}


PEER = NW._Free([("peer", "peers", "peers_0")])
RESIDUAL = ("AllReduceRmsNormMnnvl", ND.allreduce_norm, "AllReduceResidualMnnvl_v1")
ALLREDUCE = ("AllReduceMnnvl", ND.cols, "AllReduceMnnvl_v1")


@pytest.mark.parametrize("which, extra, match", [
    (RESIDUAL, lambda w, t, a, n: w.b.feed(t, "parts", a, n, (NW.Row(t, a, "o0", COLS),), ((n, 1),)),
     "fed row by row, cut 8 ways, and whole"),
    (RESIDUAL, lambda w, t, a, n: w.feed(t, "parts", a, PEER, (n,)), "fed row by row, cut 8 ways, and whole"),
    (RESIDUAL, lambda w, t, a, n: w.feed_rows(t, "parts", a, [PEER] * 4, (n,)), "cut 4 ways, and 8 ways before"),
    (ALLREDUCE, lambda w, t, a, n: w.feed_rows(t, "parts", a, [PEER] * 8, (n,)), r"row 1 of units from \d+ fed twice \(free"),
    (RESIDUAL, lambda w, t, a, n: w.feed_rows(t, "parts", a, [PEER] * 8, (n,)),
     r"row 0 of units from \d+ fed twice \((free, then wired|wired, then free)\)"),
], ids=["fed-whole", "left-free-whole", "two-cuts", "row-fed-twice", "row-wired-and-free"])
def test_row_fed_refusals(program, which, extra, match):
    """The frame refuses by name, before `_Builder.settle`, a row-fed port also fed whole (a wire or a whole free port), a
    port row-fed at two cuts, and a row of a unit fed twice: free twice, or wired and free."""
    with pytest.raises(NW.Unexpressible, match=match):
        NW.node_window(program, families=_after(*which, extra))


def test_row_fed_cut_kept(program):
    """After `_Builder.settle`, a row-fed port's cut is still the one `feed_rows` gave it, else the window is refused."""
    w = NW.NodeWindow(program, FAMILIES).lower()
    (tid,) = [t for t in w.b.fns if t.startswith("AllReduceResidualMnnvl_v1") and "SHOT=2" in t]
    w.b.cuts[tid]["parts"] = 4
    with pytest.raises(NW.Unexpressible, match=r"cuts changed after `feed_rows`: .*parts 8"):
        w.circuit()


def _rewired(circuit: IF.Circuit, make) -> IF.Circuit:
    (i, x), *_ = [(i, x) for i, x in enumerate(circuit.wires) if x.reader.startswith("AllReduceResidualMnnvl_v1")
                  and x.port == "parts" and x.cut == (0, 1)]
    return dataclasses.replace(circuit, wires=circuit.wires[:i] + (make(x),) + circuit.wires[i + 1:])


def test_wires_refused_on_their_own(window):
    """What the Circuit's construction refuses with no help from the frame: a row-fed wire on a row past the port's cut
    (``cut``), and one whose writer row isn't the reader row's width (``width``). A wire on a row that exists and is its
    width, but the wrong one (rank r's partial on row i ≠ r), passes both; no structure check catches it."""
    circuit, _ = window
    head = _tid(circuit, r"Gemm_v2\{K=6144,N=1,.*\}")
    with pytest.raises(ValueError, match=r"rows \[8, 9\) of parts's 8"):
        _rewired(circuit, lambda x: IF.Wire(x.reader, x.port, x.writer, x.out, x.at, x.radix, x.base, x.step, (8, 1)))
    with pytest.raises(ValueError, match="a row of parts is 192 words and o0 1"):
        _rewired(circuit, lambda x: IF.Wire(x.reader, x.port, head, "o0", x.at, x.radix, 0, x.step, x.cut))
    moved = _rewired(circuit, lambda x: IF.Wire(x.reader, x.port, x.writer, x.out, x.at, x.radix, x.base, x.step, (3, 1)))
    assert moved.wires != circuit.wires
