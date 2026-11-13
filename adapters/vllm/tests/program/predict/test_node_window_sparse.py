"""verity_vllm.program.predict.node_window_sparse: GLM-5.3's DSA attention and MoE layer lowered into the node window beside
the frame's linears and norms, at TP8 rank 0 with a 32-id prompt and 2 outputs. In the first decoder layer the prefill keeps
only its last token's attention (dead-free), so each step has one attention row over its keys: the 32 prompt keys at the
prefill and 33 at the decode. RoPE's query heads are the dense half's family (`RoPEHeadGptJ`), left a free input here, so
every key block's ``q_rope`` is free; the halves meet at integration.
"""
from __future__ import annotations

import re

import pytest
from verity.commitments.windows import interface as IF

from tests.program.predict.test_node_window import LP, glm_request
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_sparse as NS

FAMILIES = {**NW.FAMILIES, **NS.FAMILIES}
HEADS, KV = 8, 576


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


def test_latent_rows(window):
    """A latent row a token (the prefill's 32 and the decode's one), each reading its 576 coordinates of q_a whole through
    six chunks of 96, its norm's weights and its position's cos/sin row from the weights record."""
    circuit, w = window
    latent = _tid(circuit, r"DsaLatentRow_v1\{.*\}")
    level1, level2 = _tid(circuit, r"AllGather_v1\{WORLD=96,N=1\}"), _tid(circuit, r"AllGather_v1\{WORLD=6,N=96\}")
    assert circuit.units[latent] == LP + 1 and circuit.units[level2] == LP + 1
    q_a = _tid(circuit, r"ScaledMmFp8BlockSharedScale_v1\{K=6144,N=1,G=128\}")
    assert {(q_a, level1, "parts"), (level1, level2, "parts"), (level2, latent, "kv")} <= _pairs(circuit)
    assert {r.port for r in circuit.registered.reads if r.reader == latent} == {"w", "cs"}


def test_batched_gemm_slices(window):
    """Each absorption is its slices, a head's output word a unit: q_nope's 8 x 512 at each step reading the head's 192
    coordinates of q_b whole (two chunks of 96), and the value up-projection's 8 x 256 reading the head's attention output,
    its end's one row; o_proj's quantiser reads the 2,048 slices through 32 chunks of 64. Each unit's weights, a strided
    column of the record, are a free input under the weights record."""
    circuit, w = window
    absorb = _tid(circuit, r"BatchedGemm_v1\{B=1,K=192,N=1,.*\}")
    up = _tid(circuit, r"BatchedGemm_v1\{B=1,K=512,N=1,.*\}")
    assert circuit.units[absorb] == 2 * HEADS * 512 and circuit.units[up] == 2 * HEADS * 256
    pairs = _pairs(circuit)
    q_b = _tid(circuit, r"ScaledMmFp8BlockSharedScale_v1\{K=2048,N=1,G=128\}")
    chunk, head = _tid(circuit, r"AllGather_v1\{WORLD=96,N=1\}"), _tid(circuit, r"AllGather_v1\{WORLD=2,N=96\}")
    end = _tid(circuit, r"SparseMlaHeadEnd_v1\{DV=512,.*\}")
    quant, quant_fi = _tid(circuit, r"Fp8GroupQuantUe8m0_v1\{K=2048,G=128\}"), _tid(circuit, r"FlashInferFp8Quant1x128_v1\{K=2048\}")
    o1, o2 = _tid(circuit, r"AllGather_v1\{WORLD=64,N=1\}"), _tid(circuit, r"AllGather_v1\{WORLD=32,N=64\}")
    assert {(q_b, chunk, "parts"), (chunk, head, "parts"), (head, absorb, "x"), (end, up, "x"), (up, o1, "parts"),
            (o1, o2, "parts"), (o2, quant, "x"), (o2, quant_fi, "x")} <= pairs
    free = [f for f in w.free if f.tid in (absorb, up)]
    assert {(f.port, f.kind, f.family) for f in free} == {("w", "param", "weights")}
    assert sum(f.count for f in free) == circuit.units[absorb] + circuit.units[up]
    assert not any(r.reader in (absorb, up) for r in circuit.registered.reads)


def test_chain(window):
    """Each step's attention is one key block a head (32 keys at the prefill, 33 at the decode, in one block of BN = 64) and
    the head's end. A key block reads its head's q_nope whole (512 slices through 8 chunks of 64) and its keys as the latent
    rows, a key a row of 576 words whole, the port cut 32 or 33 ways; the end reads the block's lanes and O."""
    circuit, w = window
    blocks = [_tid(circuit, rf"SparseMlaHeadBlock_v1\{{DR=64,DV=512,NVIS={n},FIRST=true,.*\}}") for n in (LP, LP + 1)]
    end = _tid(circuit, r"SparseMlaHeadEnd_v1\{DV=512,.*\}")
    assert [circuit.units[b] for b in blocks] == [HEADS, HEADS] and circuit.units[end] == 2 * HEADS
    latent = _tid(circuit, r"DsaLatentRow_v1\{.*\}")
    q = _tid(circuit, r"AllGather_v1\{WORLD=8,N=64\}")
    pairs = _pairs(circuit)
    for n, b in zip((LP, LP + 1), blocks):
        assert circuit.cuts[b] == {"kv": n}
        assert {(latent, b, "kv"), (q, b, "q_nope"), (b, end, "l"), (b, end, "o")} <= pairs
        assert IF.layout(w.b.fns[b]).rows[2] == ("kv", n * KV)
    assert {f.family for f in w.free if f.port == "q_rope"} == {"RoPEHeadGptJ"}


def test_census(window):
    """The three families are lowered, the families they read and wrote are gone from the free list, and every template
    is a proof block Flock's composer proves."""
    circuit, w = window
    census = w.census(circuit)
    assert not {"DsaLatentRow", "BatchedGemm", "SparseMlaSplit"} & set(census["unlowered"])
    assert not {"DsaLatentRow", "BatchedGemm", "SparseMlaSplit"} & set(census["free"])
    assert {"RoPEHeadGptJ", "weights"} <= set(census["free"])
    assert census["units"]["SparseMlaSplit"] == 4 * HEADS and census["units"]["DsaLatentRow"] == LP + 1
    for tid, fn in w.b.fns.items():
        IF.check_proof_block(fn, circuit.cuts.get(tid))


def test_chain_reads_free(program):
    """With the attention alone, its query and keys are free inputs under the families that write them."""
    circuit, w = NW.node_window(program, families={**NW.FAMILIES, "SparseMlaSplit": NS.sparse_mla_split})
    assert {"BatchedGemm", "DsaLatentRow", "RoPEHeadGptJ"} <= set(w.census(circuit)["free"])
    assert {(f.family, f.port) for f in w.free if f.tid.startswith("SparseMla")} == {
        ("BatchedGemm", "q_nope"), ("RoPEHeadGptJ", "q_rope"), ("DsaLatentRow", "kv")}
