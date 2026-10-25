"""verity_vllm.program.predict.window_circuit (V3b): a served window's request Programs, as `compile_steps` compiles them, are
one Circuit of the catalog's units, every input row wired from its writer's row, a registered read of the weights or the
prompt's ingress at its step, and nothing free. Pinned on a two-request TINY window, one request's prompt chunked across two
steps, at a grain small enough to slice TINY's GEMMs and cut its ports.
"""
from __future__ import annotations

import json
from collections import Counter

import pytest
from tests.program.predict import test_steps as S
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import Grain, window_circuit

from verity.ml.windows import interface as IF

GRAIN = Grain(gemm_words=512, rows=8, select=64)
#: q0's prompt of 4 at step 0, q1's prompt of 3 chunked 2 then 1 across steps 0 and 1; each keeps two outputs
WINDOW = ([(4, 2, {**S.GREEDY, "max_tokens": 2}), (3, 2, {**S.GREEDY, "max_tokens": 2})],
          [[(0, 0, 4), (1, 0, 2)], [(0, 4, 1), (1, 2, 1)], [(1, 3, 1)]])
EMB, COMBINE = "EmbeddingRead_v1{V=256,H=64}", "TokenSelectCombine_v2{V=256,C=64}"


def _label(fn) -> str:
    return fn.id.split("{")[0] + "{" + ",".join(f"{k}={v}" for k, v in fn.bindings.items() if type(v) is int) + "}"


@pytest.fixture(scope="module")
def window():
    """The window's Circuit and its templates' ids by label."""
    circuit = S.compile_steps(S._registered(), *S._session(*WINDOW)).circuit(GRAIN)
    return circuit, {_label(fn): tid for fn, _ in circuit.templates for tid in [IF.descriptor_id(fn)]}


def _source(circuit, tid: str, port: str, unit: int, row: int = 0) -> tuple[str, int, str]:
    """The writer row (template, unit, output) that ``unit``'s ``port`` row ``row`` is wired from."""
    (w,) = [w for w in circuit.wires if (w.reader, w.port) == (tid, port) and w.at <= unit < w.at + w.reader_units
            and w.cut[0] <= row < sum(w.cut)]
    return w.writer, int(w.writers(unit, 1, row)[0]), w.out


def test_the_window_is_one_circuit_of_units_by_template_and_wire_patterns(window):
    """Every request Program's node is its units: TINY's GEMMs sliced 8 columns a unit (64 by 512 words), its attention
    and RoPE a unit a head, its token select four chunks and a combine; a port read from several writers is cut, its
    groups gathered by `AllGather_v1` units, and the wiring is a pattern per run, not a wire per unit."""
    circuit, ids = window
    label = {tid: name for name, tid in ids.items()}
    assert {label[IF.descriptor_id(fn)]: n for fn, n in circuit.templates} == {
        EMB: 9, "RMSNormTriton_v5{N=64}": 9, "Gemm_v2{K=64,N=8}": 896, "RoPEHead_v1{D=16}": 88,
        "AttentionHead_v18{T=1,D=16,BN=128,MASKED_FROM=0}": 8, "AttentionHead_v18{T=2,D=16,BN=128,MASKED_FROM=0}": 8,
        "AttentionHead_v18{T=3,D=16,BN=128,MASKED_FROM=0}": 12, "AttentionHead_v18{T=4,D=16,BN=128,MASKED_FROM=0}": 16,
        "AttentionHead_v18{T=5,D=16,BN=128,MASKED_FROM=0}": 8, "RMSNormFusedCuda_v2{N=64}": 26, "SiluMul_v2{I=128}": 13,
        "Gemm_v2{K=128,N=4}": 208, "TokenSelectChunk_v2{C=64}": 16, COMBINE: 4, "Gemm_v2{K=64,N=1}": 4,
        "AllGather_v1{WORLD=2,N=8}": 36, "AllGather_v1{WORLD=2,N=16}": 8, "AllGather_v1{WORLD=3,N=16}": 12,
        "AllGather_v1{WORLD=4,N=16}": 29, "AllGather_v1{WORLD=5,N=16}": 8, "AllGather_v1{WORLD=8,N=8}": 68,
        "AllGather_v1{WORLD=4,N=1}": 4, "AllGather_v1{WORLD=4,N=8}": 26, "AllGather_v1{WORLD=8,N=4}": 26}
    assert {label[t]: c for t, c in circuit.cuts.items()} == {
        "RMSNormFusedCuda_v2{N=64}": {"x": 2}, "RoPEHead_v1{D=16}": {"x": 2}, "SiluMul_v2{I=128}": {"gu": 4},
        COMBINE: {"idx": 4}, **{g: {"parts": int(g.split("WORLD=")[1].split(",")[0])} for g in ids if g.startswith("AllGather")}}
    family = {tid: name.split("{")[0] for name, tid in ids.items()}
    assert Counter((family[w.reader], w.port, family[w.writer], w.out) for w in circuit.wires) == {
        ("AllGather_v1", "parts", "AllGather_v1", "o0"): 30, ("AllGather_v1", "parts", "AttentionHead_v18", "o0"): 9,
        ("AllGather_v1", "parts", "Gemm_v2", "o0"): 28, ("AllGather_v1", "parts", "RoPEHead_v1", "o0"): 30,
        ("AllGather_v1", "parts", "TokenSelectChunk_v2", "o0"): 1, ("AttentionHead_v18", "k", "AllGather_v1", "o0"): 4,
        ("AttentionHead_v18", "k", "RoPEHead_v1", "o0"): 1, ("AttentionHead_v18", "q", "RoPEHead_v1", "o0"): 7,
        ("AttentionHead_v18", "v", "AllGather_v1", "o0"): 5, ("EmbeddingRead_v1", "tok", "TokenSelectCombine_v2", "o0"): 2,
        ("Gemm_v2", "x", "AllGather_v1", "o0"): 8, ("Gemm_v2", "x", "RMSNormFusedCuda_v2", "o0"): 15,
        ("Gemm_v2", "x", "RMSNormTriton_v5", "o0"): 4, ("Gemm_v2", "x", "SiluMul_v2", "o0"): 1,
        ("RMSNormFusedCuda_v2", "res", "EmbeddingRead_v1", "o0"): 4, ("RMSNormFusedCuda_v2", "res", "RMSNormFusedCuda_v2", "o1"): 6,
        ("RMSNormFusedCuda_v2", "x", "AllGather_v1", "o0"): 16, ("RMSNormTriton_v5", "x", "EmbeddingRead_v1", "o0"): 1,
        ("RoPEHead_v1", "x", "Gemm_v2", "o0"): 12, ("SiluMul_v2", "gu", "AllGather_v1", "o0"): 1,
        ("TokenSelectChunk_v2", "x", "AllGather_v1", "o0"): 1, ("TokenSelectCombine_v2", "best", "AllGather_v1", "o0"): 1,
        ("TokenSelectCombine_v2", "idx", "TokenSelectChunk_v2", "o1"): 1, ("TokenSelectCombine_v2", "x0", "Gemm_v2", "o0"): 1}
    assert circuit.external == (f"{ids[EMB]}/tok", f"{ids[COMBINE]}/o0")


def test_the_circuit_starts_at_the_prompts_token_ids_as_ingress_at_their_steps(window):
    """A prompt position's ``tok`` is ingress at the step whose row first reaches it (q1's third token at step 1, where its
    second chunk runs); a decoded position's is wired from its request's previous token select."""
    circuit, ids = window
    assert [g.doc() for g in circuit.ingress] == [{"reader": ids[EMB], "port": "tok", "at": 0, "units": 4, "step": 0},
                                                 {"reader": ids[EMB], "port": "tok", "at": 5, "units": 2, "step": 0},
                                                 {"reader": ids[EMB], "port": "tok", "at": 7, "units": 1, "step": 1}]
    assert _source(circuit, ids[EMB], "tok", 4) == (ids[COMBINE], 0, "o0")
    assert _source(circuit, ids[EMB], "tok", 8) == (ids[COMBINE], 2, "o0")


def test_the_weights_and_the_embedding_are_registered_reads(window):
    """The weights record is the one registered parameter: GEMM slices, norms and RoPE read their rows from it at public
    positions, and the embedding is `EmbeddingRead_v1`'s opening of the table over its first V·H words, keyed by ``tok``
    and pinned by its verdict. With TINY's tied embeddings the LM head's first row, which the token select's ``x0`` unit
    reads, is the table's first row."""
    circuit, ids = window
    reg = circuit.registered
    assert reg.params == {"weights": 91472}
    assert Counter((r.reader, r.port) for r in reg.reads) == {
        (ids["Gemm_v2{K=64,N=8}"], "w"): 26, (ids["Gemm_v2{K=128,N=4}"], "w"): 5, (ids["Gemm_v2{K=64,N=1}"], "w"): 1,
        (ids["RMSNormFusedCuda_v2{N=64}"], "w"): 8, (ids["RMSNormTriton_v5{N=64}"], "w"): 1, (ids["RoPEHead_v1{D=16}"], "cs"): 16}
    assert [t.doc() for t in reg.tables] == [{"name": "embedding@0", "param": "weights", "offset": 0, "words": 64, "rows": 256}]
    assert sorted(r.doc().items() for r in reg.roots) == sorted(
        {"reader": ids[EMB], "port": p, "table": "embedding@0", "of": p, "at": 0, "units": 9}.items() for p in ("domain", "root"))
    assert [o.doc() for o in reg.openings] == [{"reader": ids[EMB], "table": "embedding@0", "key": "tok",
                                                "ports": ["row", "salt", "path"], "verdict": "o1", "at": 0, "units": 9}]
    assert [(r.at, r.radix, r.base, r.step) for r in reg.reads if r.reader == ids["Gemm_v2{K=64,N=1}"]] == [(0, (4,), 0, (0,))]
    rope = [(r.at, r.radix, r.base, r.step) for r in reg.reads if r.reader == ids["RoPEHead_v1{D=16}"]]
    assert rope[:3] == [(0, (4, 4), 90432, (16, 0)), (16, (4, 2), 90432, (16, 0)), (24, (4,), 90480, (0,))]


def test_a_head_attends_over_every_key_position_through_one_gather(window):
    """q0's decode (position 4, T 5) at layer 0: heads 0 and 1 share KV group 0, whose keys are one `AllGather_v1{5, 16}`
    row over the RoPE'd keys of positions 0..4 (the prompt's at RoPE units 16, 18, 20, 22, the decode's at 40)."""
    circuit, ids = window
    att, rope = ids["AttentionHead_v18{T=5,D=16,BN=128,MASKED_FROM=0}"], ids["RoPEHead_v1{D=16}"]
    gather = _source(circuit, att, "k", 0)
    assert gather == _source(circuit, att, "k", 1) == (ids["AllGather_v1{WORLD=5,N=16}"], 0, "o0")
    assert [_source(circuit, gather[0], "parts", 0, r) for r in range(5)] == [(rope, u, "o0") for u in (16, 18, 20, 22, 40)]
    assert _source(circuit, att, "q", 0) == (rope, 36, "o0")


def test_every_input_row_is_wired_registered_or_ingress_counted_from_the_description(window):
    """`interface.input_count` over the circuit's description (JSON, no Program lowered) finds no free input row, so a
    proof of the window's units leaves nothing but the declared advice to the prover."""
    circuit, _ = window
    doc = json.loads(json.dumps(circuit.description()))
    assert IF.input_count(doc) == {"units": 1542, "inputs": 10935136, "wired": 1759360, "registered": 9175552, "ingress": 224,
                                   "free": 0}


def test_what_the_lowering_doesnt_take_is_refused_by_name(monkeypatch):
    """A stochastic request (its Program reads a seed, and its token select isn't lowered yet), a gather past the grain's
    rows (q0's third position attends over 3 keys) and a head past `KEYS` keys are refused."""
    reg = S._registered()
    stoch = S.compile_steps(reg, *S._session([(5, 2, {**S.STOCH, "max_tokens": 2})], [[(0, 0, 5)], [(0, 5, 1)]]))
    with pytest.raises(Unexpressible, match=r"inputs are \['prompt', 'seed', 'weights'\]"):
        window_circuit(stoch, GRAIN)
    greedy = S.compile_steps(reg, *S._session(*WINDOW))
    with pytest.raises(Unexpressible, match=r"a gather of 3 rows of \[16\] words: rows of one width, at most 2"):
        window_circuit(greedy, Grain(gemm_words=512, rows=2, select=64))
    monkeypatch.setattr(WC, "KEYS", 4)
    with pytest.raises(Unexpressible, match="a head of 5 keys, past the 4"):
        window_circuit(greedy, GRAIN)


def test_a_token_select_is_its_function_at_the_grains_chunk():
    """At 128 logits a chunk, each of the window's four token selects is two `TokenSelectChunk_v2{C=128}` units, each
    reading 16 LM-head slices through one gather, and a `TokenSelectCombine_v2{V=256, C=128}`: the Program's own body
    (chunks of 64) at another C."""
    circuit = S.compile_steps(S._registered(), *S._session(*WINDOW)).circuit(Grain(gemm_words=512, rows=16, select=128))
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    assert (n["TokenSelectChunk_v2{C=128}"], n["TokenSelectCombine_v2{V=256,C=128}"]) == (8, 4)
    assert not [t for t in n if t.startswith(("TokenSelectChunk_v2{C=64}", "TokenSelectCombine_v2{V=256,C=64}"))]
    assert {_source(circuit, ids["TokenSelectChunk_v2{C=128}"], "x", u)[0] for u in range(8)} == {ids["AllGather_v1{WORLD=16,N=8}"]}
    assert IF.input_count(circuit.description())["free"] == 0
