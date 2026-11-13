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


TOPK, E = 8, 256
MOE = ("CublasSplitKGemmF32Out", "LlBf16DotprodGemm", "MoeRouterSigmoidBias", "Fp8GroupQuantUe8m0", "MoeExpertFp8Gemm",
       "MoeExpertFp8GemmW")


@pytest.fixture(scope="module")
def moe_program():
    return glm_request(LP, 1, moe=True).program.fn


@pytest.fixture(scope="module")
def moe_window(moe_program):
    return NW.node_window(moe_program, families=FAMILIES)


def test_moe_router(moe_window):
    """At each step (one token: the prefill's last and the decode's) the router gate is its 256 slices, a logit a unit,
    each reading the token's row whole and its weight row from the record; the slots unit reads the 256 logits whole through
    the gather (64 chunks of 2, then 4 of 128) and its bias from the record."""
    circuit, w = moe_window
    gate, dot = _tid(circuit, r"CublasSplitKGemmF32Out_v1\{.*\}"), _tid(circuit, r"LlBf16DotprodGemm_v1\{.*\}")
    slots = _tid(circuit, r"MoeRouterSigmoidBiasSlots_v1\{.*\}")
    assert circuit.units[gate] == circuit.units[dot] == E and circuit.units[slots] == 2
    g1, g2 = _tid(circuit, r"AllGather_v1\{WORLD=64,N=2\}"), _tid(circuit, r"AllGather_v1\{WORLD=4,N=128\}")
    assert {(gate, g1, "parts"), (dot, g1, "parts"), (g1, g2, "parts"), (g2, slots, "logits")} <= _pairs(circuit)
    assert {(r.reader, r.port) for r in circuit.registered.reads if r.reader in (gate, dot, slots)} == {
        (gate, "w"), (dot, "w"), (slots, "bias")}


def test_moe_experts(moe_window):
    """Each pair (a token's slot) is cut per expert, its blocks apart: w13 (N = 512, K = 6144) in 512 block GEMMs of one row,
    each reading its row from a table read of 3,072 words and its scale row from one of 96 (4 G-blocks); w2 (N = 6144,
    K = 256) in 192 of 32 rows, reading 4,096 words and 4 (48 G-blocks). A read's key is its slot's expert id, the slots
    unit's o8..o15; a w2 block's routed weight its o0..o7; xq and sx the quantisers'. One table a block, each declared once
    with its root, domain and opening."""
    circuit, w = moe_window
    pairs = _pairs(circuit)
    slots = _tid(circuit, r"MoeRouterSigmoidBiasSlots_v1\{.*\}")
    for stack, (N, K, n) in {"": (512, 6144, 1), "W": (6144, 256, 32)}.items():
        gemms = [t for t in circuit.units if re.fullmatch(rf"MoeExpertFp8BlockGemm{stack}_v3\{{K={K},N={n},.*\}}", t)]
        rows, scales = _tid(circuit, rf"EmbeddingRead_v1\{{V={E},H={n * K // 2}\}}"), _tid(circuit, rf"EmbeddingRead_v1\{{V={E},H={K // 64}\}}")
        quant = _tid(circuit, rf"Fp8GroupQuantUe8m0_v1\{{K={K},G=128\}}")
        assert len(gemms) == 2 and all(circuit.units[t] == TOPK * N // n for t in gemms)
        assert circuit.units[rows] == 2 * TOPK * N // n and circuit.units[scales] == 2 * TOPK * N // 128
        for t in gemms:
            assert {(rows, t, "w"), (scales, t, "ws"), (quant, t, "xq"), (quant, t, "sx")} <= pairs
        keys = {x.out for x in circuit.wires if x.writer == slots and x.reader in (rows, scales)}
        assert keys == {f"o{TOPK + k}" for k in range(TOPK)}
        assert {x.out for x in circuit.wires if x.writer == slots and x.reader in gemms} == (
            {f"o{k}" for k in range(TOPK)} if stack else set())
        for r, H, blocks in ((rows, n * K // 2, N // n), (scales, K // 64, N // 128)):
            tables = {o.table for o in circuit.registered.openings if o.reader == r}
            assert len(tables) == blocks and all(re.fullmatch(rf"weights/model\.layers\.0\.mlp\.experts\..*@{H}/b\d+/table", x)
                                                 for x in tables)
            assert sum(o.units for o in circuit.registered.openings if o.reader == r) == circuit.units[r]
            assert sum(x.units for x in circuit.registered.roots if x.reader == r) == 2 * circuit.units[r]
    assert not [f for f in w.free if re.match(r"(MoeExpert|EmbeddingRead|MoeRouter)", f.tid)]


def test_moe_census(moe_window):
    """The MoE layer's families are lowered and none is free: the gate reads the dense half's norm (`AllReduceRmsNormMnnvl`)
    and w2's quantiser its `SiluMul`, both left free here, and `MoeSum`, the dense half's, reads the experts' outputs at
    integration. Every template is a proof block Flock's composer proves."""
    circuit, w = moe_window
    census = w.census(circuit)
    assert not set(MOE) & set(census["unlowered"]) and not set(MOE) & set(census["free"])
    assert {"AllReduceRmsNormMnnvl", "SiluMul"} <= set(census["free"]) and "MoeSum" in census["unlowered"]
    per_pair = {"MoeExpertFp8Gemm": 512 + 4 + 512, "MoeExpertFp8GemmW": 192 + 48 + 192}
    assert {f: census["units"][f] for f in per_pair} == {f: 2 * TOPK * u for f, u in per_pair.items()}
    assert census["units"]["MoeRouterSigmoidBias"] == 2 and census["units"]["Fp8GroupQuantUe8m0"] == 2 * (1 + TOPK)
    for tid, fn in w.b.fns.items():
        IF.check_proof_block(fn, circuit.cuts.get(tid))


def test_moe_tokens():
    """At two MoE layers the first layer's prefill routes all 32 prompt tokens, a pair row of 256 members (a token's 8 slots
    in turn): its reads' keys and its blocks' routed weights are one wire a slot over the tokens' slots units."""
    circuit, w = NW.node_window(glm_request(LP, 2, moe=True).program.fn, families=FAMILIES)
    slots = _tid(circuit, r"MoeRouterSigmoidBiasSlots_v1\{.*\}")
    assert circuit.units[slots] == LP + 3
    w2 = _tid(circuit, r"MoeExpertFp8BlockGemmW_v3\{K=256,N=32,.*SWAP=true.*\}")
    assert circuit.units[w2] == (LP + 1) * TOPK * 192
    g = [x for x in circuit.wires if x.writer == slots and x.reader == w2 and x.port == "g"]
    assert {x.out for x in g} == {f"o{k}" for k in range(TOPK)}
    assert not [f for f in w.free if re.match(r"(MoeExpert|EmbeddingRead|MoeRouter)", f.tid)]


def test_moe_reads_free(moe_program):
    """With the router left unlowered, every expert read's key and every w2 block's routed weight are free inputs under it."""
    circuit, w = NW.node_window(moe_program, families={k: v for k, v in FAMILIES.items() if k != "MoeRouterSigmoidBias"})
    free = {(f.tid.split("{")[0], f.port) for f in w.free if f.family == "MoeRouterSigmoidBias"}
    assert free == {("EmbeddingRead_v1", "tok"), ("MoeExpertFp8BlockGemmW_v3", "g")}


def test_chain_reads_free(program):
    """With the attention alone, its query and keys are free inputs under the families that write them."""
    circuit, w = NW.node_window(program, families={**NW.FAMILIES, "SparseMlaSplit": NS.sparse_mla_split})
    assert {"BatchedGemm", "DsaLatentRow", "RoPEHeadGptJ"} <= set(w.census(circuit)["free"])
    assert {(f.family, f.port) for f in w.free if f.tid.startswith("SparseMla")} == {
        ("BatchedGemm", "q_nope"), ("RoPEHeadGptJ", "q_rope"), ("DsaLatentRow", "kv")}
