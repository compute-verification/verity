"""Lowering -> synthesiser -> unit-aligned veritor definitions -> compile -> protocol, on a toy slice (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from veritor.analysis.cones import SAME, SOURCES_ONLY
from verity_vllm_synth import emit, lowering as L

K, COLS = 32, 4


def _bf16_words(rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    return L.f32_to_bf16_words((rng.standard_normal(n) * scale).astype(np.float32))


@pytest.fixture(scope="module")
def slice_and_values():
    gs = L.make_slice_gate_set("sm_89", L.silu_table_ieee())
    b = L.DagBuilder(gs)
    s = L.lower_gemm_residual(b, K, COLS)
    dag = b.build(s.outputs)
    rng = np.random.default_rng(5)
    x = _bf16_words(rng, K)
    res = _bf16_words(rng, COLS)
    w = _bf16_words(rng, COLS * K, 0.1)
    inputs = L.words(x) + L.words(res)
    weights = L.words(w)
    return gs, b, s, dag, inputs, weights


def test_lowering_matches_numpy_model_and_residual_semantics(slice_and_values) -> None:
    gs, b, s, dag, inputs, weights = slice_and_values
    v = b.evaluate(inputs, weights)
    out = np.array([v[a] for a in s.out_bf16], np.uint16)
    res_out = np.array([v[a] for a in s.residual_out], np.uint16)
    # residual' = bf16(f32(out) + f32(residual))
    o32 = L.bf16_words_to_f32(out)
    r32 = L.bf16_words_to_f32(np.array(inputs[K:], np.uint16))
    assert np.array_equal(res_out, L.f32_to_bf16_words((o32 + r32).astype(np.float32)))
    # the GEMM word is within an ulp of the fp32 dot (the chain is exact up to the mma rounding)
    x32 = L.bf16_words_to_f32(np.array(inputs[:K], np.uint16)).astype(np.float64)
    w32 = L.bf16_words_to_f32(np.array(weights, np.uint16).reshape(COLS, K)).astype(np.float64)
    ref = L.f32_to_bf16_words((w32 @ x32).astype(np.float32))
    assert np.all(np.abs(out.astype(np.int64) - ref.astype(np.int64)) <= 1)
    # the vectorised cross-check is the same wiring
    o2, r2 = L.numpy_gemm_residual(np.array(inputs[:K], np.uint16), np.array(weights, np.uint16).reshape(COLS, K), np.array(inputs[K:], np.uint16), gs)
    assert np.array_equal(o2, out) and np.array_equal(r2, res_out)


def test_uncapped_partition_merges_gemm_into_residual_add(slice_and_values) -> None:
    gs, b, s, dag, inputs, weights = slice_and_values
    part = emit.synthesize(dag, None)
    computed = [u for u in part.units if not dag.refs[u.output].is_source]
    assert len(computed) == COLS
    for u in computed:
        assert u.width == 16 and u.size == K // 16 + 1 + 4  # chain + cast + (2 widen, add, cast)
        assert u.output in s.residual_out
    costs = emit.unit_costs(part, dag)
    assert costs["units"] == COLS and costs["widths"] == {"16": COLS}
    capped = emit.synthesize(dag, 3)
    assert len([u for u in capped.units if not dag.refs[u.output].is_source]) > COLS


@pytest.mark.parametrize("cap", [None, 3])
def test_emitted_definitions_align_compile_and_verify(slice_and_values, cap) -> None:
    gs, b, s, dag, inputs, weights = slice_and_values
    part = emit.synthesize(dag, cap)
    G = emit.UnitAlignedG(gs, dag, part, n_inputs=K + COLS, n_weights=COLS * K, weight_row=K, name=f"toy-{cap}")
    wl = emit.SliceWorkload(tuple(inputs), tuple(weights))
    compilation, info = emit.compile_aligned(G, wl)
    circuit = compilation.compiled.circuit
    assert info["circuit_gates"] == dag.n
    # the compiled circuit computes the same words as the flat evaluator
    v = b.evaluate(inputs, weights)
    full = circuit.evaluate(inputs, weights)
    assert tuple(full[o] for o in circuit.outputs) == tuple(v[o] for o in dag.outputs)
    al = emit.alignment(compilation, cap)
    assert al["compare"]["counts"].get(SAME, 0) == al["computed_units"], al
    assert set(al["compare"]["counts"]) <= {SAME, SOURCES_ONLY}
    if cap is None:
        assert al["aligned"]
    honest, _ = emit.protocol_run(compilation, wl, s=1)
    assert honest["accepted"], honest
    assert honest["sampled_verification_units"] == compilation.compiled.index.verification_unit_count
    # a dishonestly committed interface word (one bit of the first computed unit's output) is caught
    # with s = 1; an interior word of a merged unit is not committed at all, so tampering it changes
    # nothing the verifier sees (the unit's output is still the honest value)
    index = compilation.compiled.index
    first_unit = next(index.verification_unit(k) for k in range(index.verification_unit_count) if any(not circuit[a].is_source for a in index.verification_unit(k).interval))
    bad, _ = emit.protocol_run(compilation, wl, s=1, tamper_gate=first_unit.interval[-1])
    assert not bad["accepted"], bad
    if cap is None:
        interior = next(a for a in first_unit.interval if not circuit[a].is_source)
        assert interior != first_unit.interval[-1]
        still_ok, _ = emit.protocol_run(compilation, wl, s=1, tamper_gate=interior)
        assert still_ok["accepted"]


def test_mlp_act_slice_lowers_and_merges() -> None:
    gs = L.make_slice_gate_set("sm_89", L.silu_table_ieee())
    b = L.DagBuilder(gs)
    s = L.lower_mlp_act(b, K, 2)
    dag = b.build(s.outputs)
    rng = np.random.default_rng(9)
    x = _bf16_words(rng, K)
    w = _bf16_words(rng, 4 * K, 0.2)
    v = b.evaluate(L.words(x), L.words(w))
    g = np.array([v[a] for a in s.gate_bf16], np.uint16)
    u = np.array([v[a] for a in s.up_bf16], np.uint16)
    act = np.array([v[a] for a in s.act], np.uint16)
    table = L.silu_table_ieee()
    expect = L.f32_to_bf16_words(L.bf16_words_to_f32(table[g]) * L.bf16_words_to_f32(u))
    assert np.array_equal(act, expect)
    part = emit.synthesize(dag, None)
    computed = [un for un in part.units if not dag.refs[un.output].is_source]
    assert len(computed) == 2 and all(un.size == 2 * (K // 16 + 1) + 5 for un in computed)


def test_saved_bundle_reverifies_from_description_alone(slice_and_values, tmp_path) -> None:
    """``run_slice --save-bundle`` layout: description + words -> ``verify_bundle`` recompiles,
    re-evaluates, matches the claimed words and runs the protocol (honest accepted, tamper rejected)."""
    from verity_vllm_synth import verify_bundle as VB

    gs, b, s, dag, inputs, weights = slice_and_values
    part = emit.synthesize(dag, None)
    G = emit.UnitAlignedG(gs, dag, part, n_inputs=K + COLS, n_weights=COLS * K, weight_row=K, name="toy-saved")
    v = b.evaluate(inputs, weights)
    d = tmp_path / "o_proj_residual"
    d.mkdir()
    (d / "description.json").write_bytes(G.description)
    np.array(inputs, np.uint16).tofile(d / "inputs.u16")
    np.array(weights, np.uint16).tofile(d / "weights.u16")
    np.array([v[o] for o in dag.outputs], np.uint16).tofile(d / "claimed_outputs.u16")
    (d / "bundle.json").write_text('{"constructor_digest": "%s", "gate_set": "%s"}' % (G.digest, gs.id))
    out = VB.verify(d, s=1)
    assert out["bit_exact"] and out["outputs"] == COLS
    assert out["protocol_honest"]["accepted"]
    assert not out["protocol_tampered_interface_word"]["accepted"]
    # a wrong claimed word is detected as not bit-exact
    bad = np.array([v[o] for o in dag.outputs], np.uint16)
    bad[0] ^= 1
    bad.tofile(d / "claimed_outputs.u16")
    assert not VB.verify(d, s=1)["bit_exact"]
