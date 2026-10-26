"""verity_vllm.program.predict.window_circuit (V3b): a served window is its class's one Circuit of the catalog's units, a class
Program a request, every input row wired from its writer's row, a registered read of the weights or the slot's ingress, and
nothing free. Pinned on a TINY class of 2 slots, 4 prompt ids and 2 outputs (5 positions a Program), at a grain small enough
to slice TINY's GEMMs and cut its ports; the class's circuit is the same for two windows that differ in everything but their
request count.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from tests.check.test_weights_of_record import _bf16, write_safetensors
from tests.program.predict import test_steps as S
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import advice as ADV
from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.window_circuit import Grain, WindowClass, class_circuit, window_circuit

from verity.ml.windows import interface as IF

GRAIN = Grain(gemm_words=512, rows=9, select=64)
CLASS = WindowClass(2, 4, 2)
#: q0's prompt of 4 at step 0, q1's prompt of 3 chunked 2 then 1 across steps 0 and 1; each keeps two outputs
WINDOW = ([(4, 2, {**S.GREEDY, "max_tokens": 2}), (3, 2, {**S.GREEDY, "max_tokens": 2})],
          [[(0, 0, 4), (1, 0, 2)], [(0, 4, 1), (1, 2, 1)], [(1, 3, 1)]])
#: another window of the class: q0's prompt of 3 at step 0 keeps two outputs, q1's prompt of 2 arrives at step 1 and keeps one
OTHER = ([(3, 2, {**S.GREEDY, "max_tokens": 2}), (2, 1, {**S.GREEDY, "max_tokens": 1})], [[(0, 0, 3)], [(0, 3, 1), (1, 0, 2)]])
EMB, COMBINE = "EmbeddingRead_v1{V=256,H=64}", "TokenSelectCombine_v2{V=256,C=64}"
FIRST, FEED = "TokenFeed_v1{NP=4,FIRST=true}", "TokenFeed_v1{NP=4,FIRST=false}"


def _label(fn) -> str:
    return fn.id.split("{")[0] + "{" + ",".join(f"{k}={json.dumps(v)}" for k, v in fn.bindings.items() if type(v) in (int, bool)) + "}"


def _circuit(window, grain: Grain = GRAIN, cls: WindowClass = CLASS):
    reg = S._registered()
    return window_circuit(reg, S.compile_steps(reg, *S._session(*window)), cls, grain)


@pytest.fixture(scope="module")
def window():
    """The window's Circuit and its templates' ids by label."""
    circuit = _circuit(WINDOW)
    return circuit, {_label(fn): tid for fn, _ in circuit.templates for tid in [IF.descriptor_id(fn)]}


def _source(circuit, tid: str, port: str, unit: int, row: int = 0) -> tuple[str, int, str]:
    """The writer row (template, unit, output) that ``unit``'s ``port`` row ``row`` is wired from."""
    (w,) = [w for w in circuit.wires if (w.reader, w.port) == (tid, port) and w.at <= unit < w.at + w.reader_units
            and w.cut[0] <= row < sum(w.cut)]
    return w.writer, int(w.writers(unit, 1, row)[0]), w.out


def test_two_windows_of_a_class_are_one_circuit(window):
    """The privacy property: WINDOW and OTHER differ in their requests' arrival steps, prompt lengths (4 and 3 against 3 and 2)
    and output lengths, and their circuits' descriptions (units, wires, cuts, registered reads, ingress), templates and external
    ports are byte for byte the same; a window of one request is another circuit, its Program count being the one public fact
    a window adds."""
    circuit, _ = window
    other = _circuit(OTHER)
    assert IF.circuit_bytes(circuit.description()) == IF.circuit_bytes(other.description())
    assert (circuit.ids, circuit.external) == (other.ids, other.external)
    one = _circuit(([(4, 2, {**S.GREEDY, "max_tokens": 2})], [[(0, 0, 4)], [(0, 4, 1)]]))
    assert IF.circuit_bytes(one.description()) != IF.circuit_bytes(circuit.description())
    assert IF.circuit_bytes(one.description()) == IF.circuit_bytes(class_circuit(S._registered(), CLASS, 1, GRAIN).description())


def test_the_circuit_is_a_class_program_a_request_in_units_by_template_and_wire_patterns(window):
    """Each request is the class Program, its 5 positions each the whole stack, the LM head and the token select: TINY's GEMMs
    sliced 8 columns a unit, its attention and RoPE a unit a head (position p's heads over p + 1 keys), its token select four
    chunks and a combine, and positions 0..3's tokens `TokenFeed_v1` units; a port read from several writers is cut, its groups
    gathered by `AllGather_v1` units, and the wiring is a pattern per run, not a wire per unit."""
    circuit, ids = window
    label = {tid: name for name, tid in ids.items()}
    assert {label[IF.descriptor_id(fn)]: n for fn, n in circuit.templates} == {
        EMB: 10, FIRST: 2, FEED: 6, "RMSNormTriton_v5{N=64}": 10, "Gemm_v2{K=64,N=8}": 1440, "RoPEHead_v1{D=16}": 120,
        **{f"AttentionHead_v18{{T={t},D=16,BN=128,MASKED_FROM=0}}": 16 for t in range(1, 6)}, "RMSNormFusedCuda_v2{N=64}": 40,
        "SiluMul_v2{I=128}": 20, "Gemm_v2{K=128,N=4}": 320, "TokenSelectChunk_v2{C=64}": 40, COMBINE: 10, "Gemm_v2{K=64,N=1}": 10,
        "AllGather_v1{WORLD=2,N=8}": 40, "AllGather_v1{WORLD=2,N=16}": 16, "AllGather_v1{WORLD=3,N=16}": 16,
        "AllGather_v1{WORLD=4,N=16}": 36, "AllGather_v1{WORLD=5,N=16}": 16, "AllGather_v1{WORLD=8,N=8}": 120,
        "AllGather_v1{WORLD=4,N=1}": 10, "AllGather_v1{WORLD=4,N=8}": 40, "AllGather_v1{WORLD=8,N=4}": 40}
    assert {label[t]: c for t, c in circuit.cuts.items()} == {
        "RMSNormFusedCuda_v2{N=64}": {"x": 2}, "RoPEHead_v1{D=16}": {"x": 2}, "SiluMul_v2{I=128}": {"gu": 4},
        COMBINE: {"idx": 4}, **{g: {"parts": int(g.split("WORLD=")[1].split(",")[0])} for g in ids if g.startswith("AllGather")}}
    family = {tid: name.split("{")[0] for name, tid in ids.items()}
    assert Counter((family[w.reader], w.port, family[w.writer], w.out) for w in circuit.wires) == {
        ("AllGather_v1", "parts", "AllGather_v1", "o0"): 32, ("AllGather_v1", "parts", "AttentionHead_v18", "o0"): 10,
        ("AllGather_v1", "parts", "Gemm_v2", "o0"): 5, ("AllGather_v1", "parts", "RoPEHead_v1", "o0"): 32,
        ("AllGather_v1", "parts", "TokenSelectChunk_v2", "o0"): 1, ("AttentionHead_v18", "k", "AllGather_v1", "o0"): 4,
        ("AttentionHead_v18", "k", "RoPEHead_v1", "o0"): 1, ("AttentionHead_v18", "q", "RoPEHead_v1", "o0"): 5,
        ("AttentionHead_v18", "v", "AllGather_v1", "o0"): 5, ("EmbeddingRead_v1", "tok", "TokenFeed_v1", "o0"): 4,
        ("EmbeddingRead_v1", "tok", "TokenSelectCombine_v2", "o0"): 2, ("Gemm_v2", "x", "AllGather_v1", "o0"): 20,
        ("Gemm_v2", "x", "RMSNormFusedCuda_v2", "o0"): 31, ("Gemm_v2", "x", "RMSNormTriton_v5", "o0"): 10,
        ("Gemm_v2", "x", "SiluMul_v2", "o0"): 1, ("RMSNormFusedCuda_v2", "res", "EmbeddingRead_v1", "o0"): 10,
        ("RMSNormFusedCuda_v2", "res", "RMSNormFusedCuda_v2", "o1"): 10, ("RMSNormFusedCuda_v2", "x", "AllGather_v1", "o0"): 40,
        ("RMSNormTriton_v5", "x", "EmbeddingRead_v1", "o0"): 1, ("RoPEHead_v1", "x", "Gemm_v2", "o0"): 1,
        ("SiluMul_v2", "gu", "AllGather_v1", "o0"): 1, ("TokenFeed_v1", "sel", "TokenSelectCombine_v2", "o0"): 1,
        ("TokenFeed_v1", "slot", "TokenFeed_v1", "o1"): 4, ("TokenSelectChunk_v2", "x", "AllGather_v1", "o0"): 1,
        ("TokenSelectCombine_v2", "best", "AllGather_v1", "o0"): 1, ("TokenSelectCombine_v2", "idx", "TokenSelectChunk_v2", "o1"): 1,
        ("TokenSelectCombine_v2", "x0", "Gemm_v2", "o0"): 1}
    assert circuit.external == (f"{ids[FIRST]}/slot", f"{ids[COMBINE]}/o0")


def test_a_positions_token_is_its_feed_below_the_prompt_cap_and_the_previous_select_past_it(window):
    """Each Program's slot is the system's ingress at its anchor, read by its first position's feed; each later feed reads its
    predecessor's slot and the previous position's select, and position p's embedding reads its feed's token for p < 4 and
    position p - 1's select at p = 4. Program 1's units follow Program 0's in every template."""
    circuit, ids = window
    first, feed, emb, comb = ids[FIRST], ids[FEED], ids[EMB], ids[COMBINE]
    assert [g.doc() for g in circuit.ingress] == [{"reader": first, "port": "slot", "at": 0, "units": 2, "step": WC.ANCHOR}]
    assert [_source(circuit, feed, "slot", u) for u in (0, 1, 2, 3)] == [(first, 0, "o1"), (feed, 0, "o1"), (feed, 1, "o1"),
                                                                          (first, 1, "o1")]
    assert [_source(circuit, feed, "sel", u) for u in (0, 2, 3, 5)] == [(comb, 0, "o0"), (comb, 2, "o0"), (comb, 5, "o0"),
                                                                         (comb, 7, "o0")]
    assert [_source(circuit, emb, "tok", u) for u in range(10)] == [
        (first, 0, "o0"), (feed, 0, "o0"), (feed, 1, "o0"), (feed, 2, "o0"), (comb, 3, "o0"),
        (first, 1, "o0"), (feed, 3, "o0"), (feed, 4, "o0"), (feed, 5, "o0"), (comb, 8, "o0")]


def test_the_weights_and_the_embedding_are_registered_reads(window):
    """The weights record is the one registered parameter: GEMM slices, norms and RoPE read their rows from it at public
    positions, and the embedding is `EmbeddingRead_v1`'s opening of the table over its first V·H words, keyed by ``tok``
    and pinned by its verdict. With TINY's tied embeddings the LM head's first row, which the token select's ``x0`` unit
    reads, is the table's first row."""
    circuit, ids = window
    reg = circuit.registered
    assert reg.params == {"weights": 91472}
    assert Counter((r.reader, r.port) for r in reg.reads) == {
        (ids["Gemm_v2{K=64,N=8}"], "w"): 60, (ids["Gemm_v2{K=128,N=4}"], "w"): 1, (ids["Gemm_v2{K=64,N=1}"], "w"): 1,
        (ids["RMSNormFusedCuda_v2{N=64}"], "w"): 20, (ids["RMSNormTriton_v5{N=64}"], "w"): 1, (ids["RoPEHead_v1{D=16}"], "cs"): 40}
    assert [t.doc() for t in reg.tables] == [{"name": "embedding@0", "param": "weights", "offset": 0, "words": 64, "rows": 256}]
    assert sorted(r.doc().items() for r in reg.roots) == sorted(
        {"reader": ids[EMB], "port": p, "table": "embedding@0", "of": p, "at": 0, "units": 10}.items() for p in ("domain", "root"))
    assert [o.doc() for o in reg.openings] == [{"reader": ids[EMB], "table": "embedding@0", "key": "tok",
                                                "ports": ["row", "salt", "path"], "verdict": "o1", "at": 0, "units": 10}]
    assert [(r.at, r.radix, r.base, r.step) for r in reg.reads if r.reader == ids["Gemm_v2{K=64,N=1}"]] == [(0, (10,), 0, (0,))]
    rope = [(r.at, r.radix, r.base, r.step) for r in reg.reads if r.reader == ids["RoPEHead_v1{D=16}"]]
    assert rope[:3] == [(0, (4,), 90432, (0,)), (4, (2,), 90432, (0,)), (6, (4,), 90432, (0,))]


def test_a_head_attends_over_every_key_position_through_one_gather(window):
    """Position 4 (T 5) at layer 0: heads 0 and 1 share KV group 0, whose keys are one `AllGather_v1{5, 16}` row over the
    RoPE'd keys of positions 0..4 (RoPE units 12p + 4, a position's 12 heads), Program 1's over its own (60 units on)."""
    circuit, ids = window
    att, rope = ids["AttentionHead_v18{T=5,D=16,BN=128,MASKED_FROM=0}"], ids["RoPEHead_v1{D=16}"]
    gather = _source(circuit, att, "k", 0)
    assert gather == _source(circuit, att, "k", 1) == (ids["AllGather_v1{WORLD=5,N=16}"], 0, "o0")
    assert [_source(circuit, gather[0], "parts", 0, r) for r in range(5)] == [(rope, 12 * p + 4, "o0") for p in range(5)]
    assert _source(circuit, att, "q", 0) == (rope, 48, "o0")
    other = _source(circuit, att, "k", 8)
    assert [_source(circuit, other[0], "parts", other[1], r) for r in range(5)] == [(rope, 60 + 12 * p + 4, "o0") for p in range(5)]


def test_every_input_row_is_wired_registered_or_ingress_counted_from_the_description(window, tmp_path):
    """`interface.input_count` over the circuit's description (JSON, no Program lowered) at the verifier's one pin, the
    weights record's root of record fixed before the run (`weights_record`), and a window whose registration records that
    root for ``weights`` (`verity-vllm advice --weights-of-record`), finds no free input row, so a proof of the window's
    units leaves nothing but the declared advice to the prover; the ingress is the two slots, 5 i32s each. Every registered
    read, table read and opening is of the weights (the GEMM slices', norms' and RoPE tables' rows; the embedding table's
    root, domain and rows), so with no pin, or a window whose ``weights`` root is another's or none, all of them are free."""
    circuit, _ = window
    doc, reg = json.loads(json.dumps(circuit.description())), circuit.registered
    assert {r.param for r in reg.reads} | {t.param for t in reg.tables} == set(reg.params) == {WC.WEIGHTS}
    count = {"units": 2442, "inputs": 17372576, "wired": 2785376, "registered": 14586880, "ingress": 2 * 5 * 32, "free": 0}
    pin = ADV.registered_weights(WC.weights_record(S._registered(), *_checkpoint(tmp_path)))
    assert IF.input_count(doc, pins=pin, roots=pin) == count
    free = {**count, "registered": 0, "free": 14586880}
    assert IF.input_count(doc) == IF.input_count(doc, pins=pin) == free
    other = ADV.registered_weights(WC.weights_record(S._registered(), *_checkpoint(tmp_path / "other", seed=4)))
    assert other != pin and IF.input_count(doc, pins=pin, roots=other) == free


def _checkpoint(tmp_path, seed: int = 3) -> tuple[str, str]:
    """TINY's checkpoint as safetensors shards of its HF names (random bf16 weights from ``seed``) and a manifest pinning it
    at `test_steps._registered`'s model and revision: the checkpoint directory and the manifest."""
    T, rng = S.TINY, np.random.default_rng(seed)
    H, I, NH, KVH, V = T["hidden_size"], T["intermediate_size"], T["num_attention_heads"], T["num_key_value_heads"], T["vocab_size"]
    D = H // NH
    t = {"model.embed_tokens.weight": _bf16(rng, V, H), "model.norm.weight": _bf16(rng, H)}
    for i in range(T["num_hidden_layers"]):
        p = f"model.layers.{i}."
        t |= {p + "self_attn.q_proj.weight": _bf16(rng, NH * D, H), p + "self_attn.k_proj.weight": _bf16(rng, KVH * D, H),
              p + "self_attn.v_proj.weight": _bf16(rng, KVH * D, H), p + "self_attn.o_proj.weight": _bf16(rng, H, NH * D),
              p + "mlp.gate_proj.weight": _bf16(rng, I, H), p + "mlp.up_proj.weight": _bf16(rng, I, H),
              p + "mlp.down_proj.weight": _bf16(rng, H, I), p + "input_layernorm.weight": _bf16(rng, H),
              p + "post_attention_layernorm.weight": _bf16(rng, H)}
    d = tmp_path / "snap"
    d.mkdir(parents=True, exist_ok=True)
    write_safetensors(str(d / "model.safetensors"), {n: ("BF16", a) for n, a in t.items()})
    (d / "config.json").write_text(json.dumps({**T, "rope_theta": 10000.0}))
    man = tmp_path / "checkpoints.json"
    model = S._registered().phi.model
    man.write_text(json.dumps({"checkpoints": [{"repo": model["repo"], "revision": model["revision"], "local_path": str(d),
                                                "files": {"model.safetensors": {"sha256": WR.sha256_file(str(d / "model.safetensors"))}}}]}))
    return str(d), str(man)


def test_the_weights_pin_is_the_class_programs_weights_record_of_the_pinned_checkpoint(tmp_path):
    """`weights_record`: every field of the class Program's weights record (TINY's 23: each layer's fused qkv and gate-up,
    o and down projections and norms, the embedding, the final norm, the rotary table and the attention scales), in the
    Program's order, composed from the checkpoint's pinned shards, so it is of record; one element changed in a shard is
    another root, and a shard the manifest doesn't pin leaves the record not of record, which no window registers
    (`advice.registered_weights`), nor one of another model than the run served."""
    reg = S._registered()
    d, man = _checkpoint(tmp_path)
    rec = WC.weights_record(reg, d, man)
    assert (rec["of_record"], rec["complete"], rec["n_fields"], rec["unresolved"]) == (True, True, 23, [])
    program = WC.class_program(reg, WindowClass(1, 2, 2)).program
    assert [f["field"] for f in rec["fields"]] == [n for n, _ in dict(program.fn.signature()[0])[WC.WEIGHTS].fields]
    assert ADV.registered_weights(rec) == ADV.registered_weights(rec, {"model": "org/m", "revision": "r"}) == {
        WC.WEIGHTS: rec["root_of_record"]}
    assert WC.weights_record(reg, *_checkpoint(tmp_path / "again"))["root_of_record"] == rec["root_of_record"]
    shard = Path(d) / "model.safetensors"
    body = bytearray(shard.read_bytes())
    body[-1] ^= 1
    shard.write_bytes(bytes(body))
    unpinned = WC.weights_record(reg, d, man)
    assert unpinned["of_record"] is False and unpinned["root_of_record"] != rec["root_of_record"]
    for bad, serve, why in [(unpinned, None, "of record False"), ({**rec, "schema": "verity/commit/weights-of-record/v1"}, None, "v1"),
                            (rec, {"model": "org/other", "revision": "r"}, "the run served org/other@r")]:
        with pytest.raises(ValueError, match=why):
            ADV.registered_weights(bad, serve)
    assert WR.SCHEMA == rec["schema"]


def test_what_the_class_doesnt_take_is_refused_by_name(monkeypatch):
    """A window past its class (three requests in two slots, a prompt past the cap, more outputs than the cap), a stochastic
    request (its Program would read a seed), a gather past the grain's rows (position 0's token select reads a chunk of 64
    logits from 8 LM-head slices) and a head past
    `KEYS` keys are refused, and so is a Program count the class has no slots for."""
    reg = S._registered()

    def refused(window, cls=CLASS, grain=GRAIN):
        return lambda: window_circuit(reg, S.compile_steps(reg, *S._session(*window)), cls, grain)

    g2 = {**S.GREEDY, "max_tokens": 2}
    three = ([(2, 2, g2)] * 3, [[(0, 0, 2), (1, 0, 2), (2, 0, 2)], [(0, 2, 1), (1, 2, 1), (2, 2, 1)]])
    long = ([(5, 2, g2)], [[(0, 0, 5)], [(0, 5, 1)]])
    many = ([(2, 3, {**S.GREEDY, "max_tokens": 3})], [[(0, 0, 2)], [(0, 2, 1)], [(0, 3, 1)]])
    stoch = ([(4, 2, {**S.STOCH, "max_tokens": 2})], [[(0, 0, 4)], [(0, 4, 1)]])
    for window, match in ((three, "3 requests in a window of a class of 2 slots"), (long, "5 prompt ids and 2 outputs, past"),
                          (many, "2 prompt ids and 3 outputs, past its class's 4 and 2"), (stoch, "a stochastic request")):
        with pytest.raises(Unexpressible, match=match):
            refused(window)()
    with pytest.raises(Unexpressible, match=r"a gather of 8 rows of \[8\] words: rows of one width, at most 2 \(with its"):
        refused(WINDOW, grain=Grain(gemm_words=512, rows=3, select=64))()
    monkeypatch.setattr(WC, "KEYS", 4)
    with pytest.raises(Unexpressible, match="a head of 5 keys, past the 4"):
        refused(WINDOW)()
    for n in (0, 3):
        with pytest.raises(Unexpressible, match=f"{n} Programs in a window of a class of 2 slots"):
            class_circuit(reg, CLASS, n, GRAIN)


def _unit_rows(circuit) -> dict[str, int]:
    """Each template's rows in and out under the circuit's cuts, as C-Flock's `UnitStager` takes them
    (`interface.unit_rows`)."""
    return {_label(fn): sum(map(len, IF.unit_rows(fn, circuit.cuts.get(IF.descriptor_id(fn))))) for fn, _ in circuit.templates}


def test_no_unit_holds_more_rows_in_and_out_than_the_grain():
    """C-Flock's composer gives each row a unit reads or writes an hm96 slot, so at Llama-3.2-1B's grain a unit fits its
    2^27-bit block with at most 256 rows in and out, and a gather of 256 parts (257 rows) is refused when staged. The cut
    counts a gather's output row: SiluMul's 1,024 gate/up slices are gathered 128 at a time (8 gathers, not 4 of 256), and
    the fused norm's 512 down slices 128 at a time (4, not 2 of 256), its 128 o-proj slices 32 at a time. TINY at 16 rows is
    the same case, 32 gate/up slices and 16 down: 8 at a time, where 16 a gather would make 17 rows. No unit of the circuit
    holds more than the grain's rows, and at the test grain's 9 the largest hold exactly 9."""
    assert WC._cut(("SiluMul_v2", "gu"), {1024: 16}, 255, 256) == 8
    assert WC._cut(("RMSNormFusedCuda_v2", "x"), {128: 16, 512: 16}, 252, 256) == 4
    circuit = _circuit(WINDOW, Grain(gemm_words=512, rows=16, select=64))
    label = {IF.descriptor_id(fn): _label(fn) for fn, _ in circuit.templates}
    assert {label[t]: c for t, c in circuit.cuts.items() if not label[t].startswith("AllGather")} == {
        "RMSNormFusedCuda_v2{N=64}": {"x": 2}, "RoPEHead_v1{D=16}": {"x": 2}, "SiluMul_v2{I=128}": {"gu": 4},
        COMBINE: {"idx": 4}}
    rows = _unit_rows(circuit)
    assert max(rows.values()) <= 16 and max(c["parts"] for c in circuit.cuts.values() if "parts" in c) == 8
    assert max(_unit_rows(_circuit(WINDOW)).values()) == GRAIN.rows


def test_a_token_select_is_its_function_at_the_grains_chunk():
    """At 128 logits a chunk, each of the circuit's ten token selects is two `TokenSelectChunk_v2{C=128}` units, each reading
    16 LM-head slices through one gather, and a `TokenSelectCombine_v2{V=256, C=128}`: the Program's own body (chunks of 64)
    at another C."""
    circuit = _circuit(WINDOW, Grain(gemm_words=512, rows=17, select=128))
    n = {_label(fn): k for fn, k in circuit.templates}
    ids = {_label(fn): IF.descriptor_id(fn) for fn, _ in circuit.templates}
    assert (n["TokenSelectChunk_v2{C=128}"], n["TokenSelectCombine_v2{V=256,C=128}"]) == (20, 10)
    assert not [t for t in n if t.startswith(("TokenSelectChunk_v2{C=64}", "TokenSelectCombine_v2{V=256,C=64}"))]
    assert {_source(circuit, ids["TokenSelectChunk_v2{C=128}"], "x", u)[0] for u in range(20)} == {ids["AllGather_v1{WORLD=16,N=8}"]}
    pin = {WC.WEIGHTS: "ab" * 32}
    assert IF.input_count(circuit.description(), pins=pin, roots=pin)["free"] == 0
