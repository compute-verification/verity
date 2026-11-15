"""verity_vllm.program.predict.node_window_select: the greedy token select lowered into GLM-5.3's node window at TP8 rank 0 as
its served stages, 19 chunks of 7,744 lanes and their combine, at a 32-id prompt, 2 outputs and one decoder layer. Each
chunk's logits are rows of 121 lanes, rank 0's wired to the LM head's first-level gather chunks and the other ranks' free under
their peers' names; the combine reads each chunk's result through its parameter. `TokenSelect_v4`, `TokenSelect_v1` and the
stages are one function on rows of a small V."""
from __future__ import annotations

import re

import numpy as np
import pytest
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_dense as ND
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import node_window_multimem as NM
from verity_vllm.program.predict import node_window_select as NSEL
from verity_vllm.program.predict import node_window_sparse as NS
from tests.program.predict.test_node_window import LP, glm_request
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation.batch import evaluate_batch
from verity.circuits.refs import runs
from verity.commitments.windows import interface as IF
from verity.definitions.registry import b1
from verity.definitions.registry import dead_free as DF
from verity.definitions.registry import greedy_stages as G

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NSEL.FAMILIES}
V, K, N, R = 154880, 20, 7744, 121
IDS = 43
#: the bf16 words a small row's lanes take, so ties are common: +0, -0, 1, -1, 2, 1 + 2^-7, the least subnormal, +inf, -inf
#: and two NaNs
WORDS = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x4000, 0x3F81, 0x0001, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1], np.uint16)


def _token(fn, rows: np.ndarray) -> np.ndarray:
    got = np.asarray(evaluate_batch(fn, rows, kernel="reference")).reshape(len(rows), -1).astype(np.uint64)
    return got[:, 0] if got.shape[1] == 1 else got[:, 0] | (got[:, 1] << 16)


def _units(Vs: int, rows: np.ndarray) -> np.ndarray:
    """The plan's units evaluated one by one, each on its lanes and its wired chunks' results, as the window wires them."""
    outs: dict[tuple, np.ndarray] = {}
    for u in G.plan(Vs):
        wires = {p: outs[k] for p, k, _f in u.wires}
        args = [rows[:, u.lanes[0]:u.lanes[1]] if p == "logits" else wires[p] for p, _t in u.defn.signature()[0]]
        outs[u.key] = np.asarray(evaluate_batch(u.defn, *args, kernel="reference")).reshape(len(rows), -1)
    got: np.ndarray = outs[("combine",)].astype(np.uint64)
    return got[:, 0] | (got[:, 1] << 16)


@pytest.mark.parametrize("Vs", [200, 130])
def test_v4_is_the_stages(Vs):
    """On rows of a small V whose lanes are `WORDS`' (ties, signed zeros, infinities, NaNs, a NaN first logit, an all-NaN row),
    `TokenSelect_v4`, `TokenSelect_v1`, `GreedySelect_v1` and the plan's units one by one, each its own circuit by the
    reference evaluator (not `numerics.rows`' argmax), give the same token, the twin's."""
    rng = np.random.default_rng(Vs)
    rows = WORDS[rng.integers(0, len(WORDS), (48, Vs))]
    rows[1, 0] = rows[2, :] = 0x7FC0
    rows[3, :] = 0xFF80
    rows[4, :] = 0x3F80
    rows[5, :], rows[5, [7, Vs // 2, Vs - 1]] = 0x0000, 0x4000
    want = np.array([G.twin(Vs, r)[0] for r in rows], np.uint64)
    assert np.array_equal(_token(bind(DF.TokenSelectV4, V=Vs), rows), want)
    assert np.array_equal(_token(bind(b1.TokenSelect, V=Vs), rows), want)
    assert np.array_equal(_token(G.select(Vs), rows), want)
    assert np.array_equal(_units(Vs, rows), want)
    assert want[1] == want[2] == want[3] == 0 and want[4] == 0 and want[5] == 7


@pytest.fixture(scope="module")
def program():
    return glm_request(LP, 1).program.fn


@pytest.fixture(scope="module")
def window(program):
    return NW.node_window(program, families=FAMILIES)


def _select_templates(circuit: IF.Circuit) -> list[str]:
    return [t for t in circuit.units if re.match(r"Greedy(Chunk|Combine)_v2\{", t)]


def test_select_units(program, window):
    """Both selects (the prefill's and the decode's) are the 20 stage templates, each two units, their logits cut 64 ways into
    rows of 121 lanes, each template a proof block Flock's composer takes under that cut; nothing is left unlowered of them."""
    circuit, w = window
    assert "TokenSelect" not in w.census(circuit)["unlowered"]
    ts = _select_templates(circuit)
    want = [descriptor_id(u.defn) for u in G.plan(V)]
    assert sorted(ts) == sorted(want) and len(ts) == K
    fns = dict(zip(circuit.ids, (f for f, _n in circuit.templates)))
    for t in ts:
        assert circuit.units[t] == 2
        assert circuit.cuts[t] == {"logits": N // R}
        IF.check_proof_block(fns[t], circuit.cuts[t])
    sel = [n for n in program.body.nodes if NW._family(n) == "TokenSelect"]
    assert len(sel) == 2 and all(n.form == "call" for n in sel)


def test_select_reads(program, window):
    """Rank 0's 160 rows of each select's logits are wired to the LM head's first-level gather chunks
    (`AllGather_v1{WORLD=121,N=1}`) and the other ranks' 1,120 rows are free under their peers' names, seven a select, each
    select's its own; each combine parameter
    r<c> is wired to chunk c's output."""
    circuit, w = window
    ts = set(_select_templates(circuit))
    combine = next(t for t in ts if t.startswith("GreedyCombine"))
    first = next(t for t in circuit.units if t.startswith("AllGather_v1{WORLD=121,N=1}"))
    logits = [x for x in circuit.wires if x.reader in ts and x.port == "logits"]
    assert {x.writer for x in logits} == {first}
    assert sum(x.units for x in logits) == 2 * V // 8 // R
    free = [f for f in w.free if f.tid in ts]
    assert {f.kind for f in free} == {"peer"} and {f.port for f in free} == {"logits"}
    sels = [n for n in program.body.nodes if NW._family(n) == "TokenSelect"]
    peers = [{w.params[i] for s, i, _lo, _hi in runs(NSEL.through(w, n.args[0].refs)) if s == "p"} for n in sels]
    assert all(len(p) == 7 for p in peers) and not peers[0] & peers[1]
    assert {f.name for f in free} == peers[0] | peers[1]
    assert sum(f.count for f in free) == 2 * 7 * V // 8 // R
    rs = {x.port: x.writer for x in circuit.wires if x.reader == combine and x.port.startswith("r")}
    assert rs == {f"r{c}": descriptor_id(G.plan(V)[c].defn) for c in range(K - 1)}


def test_select_stand_ins(program, window):
    """The value check's stand-in window (`NodeWindow` ``stand_ins``) of the lowered selects is the served one."""
    circuit, w = window
    c2, w2 = NW.node_window(program, families=FAMILIES, stand_ins=True)
    assert w2.census(c2)["digest"] == w.census(circuit)["digest"]


@pytest.mark.slow
def test_joined_with_select():
    """Every family set (`node_window`, `_dense`, `_sparse`, `_multimem`, `_embed`, `_select`) at a 43-id prompt, 2 outputs and
    one decoder layer: only the stop check is unlowered; no row is free of the token select, the decode's 32 shard reads taking
    ``tok`` by one wire from the prefill select's combine unit; the prefill's 1,376 reads are still row 0 of the ``SIZE=192``
    combine's ``parts``, which `node_window_multimem` combines as rows of 192 words. The only test that builds the window with
    the select beside the other five sets."""
    families = {**FAMILIES, **NS.FAMILIES, **NM.FAMILIES, **NE.FAMILIES}
    circuit, w = NW.node_window(glm_request(IDS, 1).program.fn, families=families)
    assert w.census(circuit)["unlowered"] == {"StopCheck": 1}
    assert not [f for f in w.free if f.family == "TokenSelect"]
    (read,) = [t for t in circuit.units if t.startswith("EmbeddingRead_v1{")]
    combine = descriptor_id(G.plan(V)[-1].defn)
    blocks = 6144 // NE.WIDTH
    tok = [(x.writer, x.out, x.at, x.radix, x.base, x.step, x.cut)
           for x in circuit.wires if x.reader == read and x.port == "tok"]
    assert tok == [(combine, "o0", IDS * blocks, (blocks,), 0, (0,), (0, 1))]
    assert {(f.kind, f.name, f.at, f.count) for f in w.free if f.tid == read} == {("param", "prompt", 0, IDS * blocks)}
    parts = [x for x in circuit.wires if x.writer == read and x.reader.startswith("AllReduceMultimem_v1")]
    assert [(x.port, x.out, x.at, x.radix, x.base, x.step, x.cut[0]) for x in parts] == [
        ("parts", "o0", 0, (IDS, blocks), 0, (1, IDS), 0)]
    assert "SIZE=192," in parts[0].reader and circuit.cuts[parts[0].reader] == {"parts": 8}
    assert circuit.units[parts[0].reader] == IDS * blocks
