"""Coordinate checks on a tiny v2 bundle: openings, family gate, N02 and 15.2b plumbing (CPU only)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from _helpers import TINY_MANIFEST

from verity_vllm import commit, identity
from verity_vllm.capture import Boundary, CaptureReader, RunWriter, StepRecord
from verity_vllm.description import Description
from verity_vllm_adapter import coordinate_check as CC

NR = pytest.importorskip("verity_vllm_numerics.relations")

K, N = 32, 8


def bf16_int(v: int) -> int:
    """BF16 word of a small integer (exactly representable)."""
    return struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16


def _description(profile_id: str) -> dict:
    return {
        "schema": "verity-vllm/description/v1",
        "profile_id": profile_id,
        "model": "toy",
        "symbols": {"H": K, "N": N},
        "nodes": [
            {
                "op": "layers.0.input_layernorm.rmsnorm",
                "layer": "0",
                "op_type": "rmsnorm",
                "dtype": "bfloat16",
                "shape": ["H"],
                "args": [],
                "relation": "unresolved",
                "origin": "captured",
                "decompositions": [{"id": "row", "interface": {"input_bytes": 0, "output_bits": 16 * K}}],
                "runtime_impl": "toy",
            },
            {
                "op": "layers.0.self_attn.o_proj.gemm",
                "layer": "0",
                "op_type": "gemm",
                "dtype": "bfloat16",
                "shape": ["N"],
                "args": [
                    {"name": "x", "source": "node", "op": "layers.0.input_layernorm.rmsnorm", "port": 0},
                    {"name": "w", "source": "weight", "param": "layers.0.self_attn.o_proj.weight"},
                ],
                "relation": "gemm.same_runtime.v1",
                "origin": "captured",
                "decompositions": [{"id": "row", "interface": {"input_bytes": 2 * K * (N + 1), "output_bits": 16 * N}}],
                "runtime_impl": "triton:vllm.model_executor.determinism.batch_invariant.matmul_persistent (via linear_batch_invariant)",
                "runtime_op": "verity.bi_mm",
                "module_path": "layers.0.self_attn.o_proj",
                "consts": {},
                "arg_cols": {},
            },
        ],
    }


@pytest.fixture()
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x_int = rng.integers(-3, 4, size=(K,))
    w_int = rng.integers(-3, 4, size=(N, K))
    out_int = w_int @ x_int  # |.| <= 9*32 = 288 < 2^9: exact in fp32 and bf16-representable? bf16 has 8 bits of mantissa
    # keep outputs exactly representable in BF16 (|v| < 256 with integer values)
    assert np.all(np.abs(out_int) < 256)
    x = np.array([bf16_int(int(v)) for v in x_int], np.uint16)
    w = np.array([[bf16_int(int(v)) for v in row] for row in w_int], np.uint16)
    out = np.array([bf16_int(int(v)) for v in out_int], np.uint16)
    run = RunWriter(tmp_path, TINY_MANIFEST)
    pid = run.profile_id
    ru = run.ru("req-0", 0, phase="prefill", positions=[0])
    x_id = str(identity.value_id(pid, "req-0", 0, "prefill", "0", "layers.0.input_layernorm.rmsnorm", 0))
    out_id = str(identity.value_id(pid, "req-0", 0, "prefill", "0", "layers.0.self_attn.o_proj.gemm", 0))
    ru.add(x_id, "bfloat16", [K], x)
    ru.add(out_id, "bfloat16", [N], out)
    ru.finish()
    boundary = Boundary.build(
        "req-0", [1], [7], {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 1, "seed": None},
        seed=None, finish_reason="stop", steps=[StepRecord(0, "prefill", (0,))],
    )
    run.write_boundary(boundary)
    ww = run.weights(store_values=True, format_version=2)
    ww.add("layers.0.self_attn.o_proj.weight", "bfloat16", [N, K], w, rows_per_leaf=1)
    ww.finish()
    run.write_description(_description(pid))
    run.finish()
    commit.build_commitments(CaptureReader(tmp_path))
    # the toy shape is not a validated Qwen family: declare it for this test only
    monkeypatch.setitem(NR.VALIDATED_BI_FAMILIES, (K, N), {"name": "toy_o_proj", "epilogues": ("none",), "M": (1, 2048)})
    return tmp_path, x, w, out


def _cpp_available() -> bool:
    try:
        from verity_vllm_numerics import cpu_model

        cpu_model.lib()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(autouse=True)
def _fallback_to_python_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _cpp_available():
        def gemm(node, inputs):  # the coordinate gate is the same chain, K x 1
            cov = NR.coverage(node, inputs)
            if not cov["covered"]:
                raise NR.GemmUnresolved(cov["reason"])
            return NR.independent_gemm_coordinate(node, inputs)

        monkeypatch.setattr(NR, "independent_gemm", gemm)


def test_coordinate_checks_exact_and_opened_bytes(bundle) -> None:
    run_dir, x, w, out = bundle
    reader = CaptureReader(run_dir)
    desc = Description(reader.description)
    doc = commit.load_commitments(reader)
    cache = commit.TreeCache(reader)
    buckets = CC.gemm_rows(desc, reader)
    assert list(buckets) == [("o_proj", "prefill")]
    sample = CC.stratified_sample(buckets, desc, N, seed=1)
    assert len(sample) == N
    seen = set()
    for cid, j, fam, phase in sample:
        r = CC.check_coordinate(desc, reader, doc, cache, cid, j, python_gate=True)
        assert r.status == "exact", (j, r.detail, r.coverage)
        assert r.committed_word == int(out[j]) == r.predicted_word == r.python_gate_word
        assert r.w_id == str(identity.weight_id("layers.0.self_attn.o_proj.weight", None, j))
        assert r.opened_bytes["w"] == 2 * K and r.opened_bytes["x"] == 2 * K and r.opened_bytes["out_row"] == 2 * N
        assert r.opened_bytes["paths"] > 0 and r.K == K and r.N == N
        assert "toy_o_proj" in r.coverage and f"1-row block of the N={N}" in r.coverage
        seen.add(j)
    # every column reachable
    for j in range(N):
        r = CC.check_coordinate(desc, reader, doc, cache, sample[0][0], j)
        assert r.status == "exact"
    summary = CC.summarize([CC.check_coordinate(desc, reader, doc, cache, sample[0][0], j) for j in range(N)])
    assert summary["o_proj/prefill"]["exact"] == N


def test_n02_wrong_coordinate_is_rejected_and_operand_mutation_changes_output(bundle) -> None:
    run_dir, x, w, out = bundle
    reader = CaptureReader(run_dir)
    desc = Description(reader.description)
    doc = commit.load_commitments(reader)
    cid = str(identity.value_id(reader.profile_id, "req-0", 0, "prefill", "0", "layers.0.self_attn.o_proj.gemm", 0))
    n02 = CC.n02_coordinate(desc, reader, doc, cid, 3)
    assert n02["authenticated"] and n02["rejected"] and n02["status"] == "mismatch"
    assert n02["committed_word"] == int(out[3]) ^ 1 and n02["predicted_word"] == int(out[3])
    # the other coordinates of the mutated row still verify (only word 3 was changed)
    for which in ("x", "w"):
        m = CC.mutate_operand(desc, reader, doc, cid, 5, which, seed=11)
        assert m["status"] in ("mismatch", "exact"), m
        assert m["authenticated"], m
        assert m["output_changed"], m  # a strong perturbation of one operand word moves the coordinate
        assert m["rejected"], m  # ...so the relation no longer matches the committed word
        assert m["word_before"] != m["word_after"]
    # the unmutated bundle is untouched
    cache = commit.TreeCache(reader)
    assert CC.check_coordinate(desc, reader, doc, cache, cid, 5).status == "exact"


def test_v1_bundle_reports_structural(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole-tensor weight leaves cannot support a coordinate opening: the check says so."""
    from _helpers import make_tiny_run

    make_tiny_run(tmp_path / "v1")
    reader = CaptureReader(tmp_path / "v1")
    assert reader.weights.leaf_id_for_row(str(identity.weight_id("lm_head.weight")), 2) == str(identity.weight_id("lm_head.weight"))
