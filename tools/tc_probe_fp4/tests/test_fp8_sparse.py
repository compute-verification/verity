"""``tools/tc_probe_fp4/fp8_sparse.py``, the 2:4-sparse E4M3 row, on fake silicon.

The register packers invert and every candidate layout packs differently; the layout check names the layout the device uses and
only that one; the candidates' arithmetic equals the core batch kernel (one registered step on the compressed operands, two
chained on the logical halves); the fake row passes with the fake's model expected, its capture refutes every other candidate and
the chain twin, and one flipped bit fails it; the W1 op gates on 32 products per output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fp8  # noqa: E402
import fp8_sparse as SP  # noqa: E402
import w1  # noqa: E402
from verity.ml.tc import models as TCM  # noqa: E402

FAST = ["--row", "sparse", "--n-random", "16", "--families", "randn_random_acc,cancellation,wide_spread,acc_floor", "--workers", "1",
        "--reps", "1", "--target-cycles", "65536"]


@pytest.fixture
def no_nvcc(monkeypatch):
    monkeypatch.setattr(fp8.P4, "nvcc_bin", lambda: "/nonexistent/nvcc")
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    monkeypatch.delenv("GPU_LEASE_UUID", raising=False)


def _operands(rng, n):
    As = rng.integers(0, 256, (n, 16, 32), dtype=np.uint8)
    M = SP.META_VALID[rng.integers(0, 6, (n, 16, 16))]
    Bl = rng.integers(0, 256, (n, 8, 64), dtype=np.uint8)
    return As, M, Bl


def test_pack_inverts_and_layouts_differ():
    As, M, Bl = _operands(np.random.default_rng(1), 4)
    packed = {}
    for lay in SP.LAYOUTS:
        R = SP.pack(As, M, Bl, lay)
        got = SP.unpack(R, lay)
        assert all(np.array_equal(g, w) for g, w in zip(got, (As, M, Bl))), lay
        packed[lay] = R.tobytes()
    assert len(set(packed.values())) == len(SP.LAYOUTS)


def test_columns_and_logical_a_pinned():
    M = np.full((1, 1, 16), 0b0100, np.uint8)
    M[0, 0, 1] = 0b1001  # group 1 keeps logical 5 and 6
    M[0, 0, 15] = 0b1110  # group 15 keeps 62 and 63
    cols = SP.columns(M)[0, 0]
    assert cols[:4].tolist() == [0, 1, 5, 6] and cols[-2:].tolist() == [62, 63]
    As = np.arange(1, 33, dtype=np.uint8).reshape(1, 1, 32)
    Al = SP.logical_a(As, M)[0, 0]
    assert Al[[0, 1, 5, 6, 62, 63]].tolist() == [1, 2, 3, 4, 31, 32] and Al[[2, 3, 4, 7, 60, 61]].tolist() == [0] * 6


def test_layout_gate_names_the_devices_layout():
    for lay in (SP.PRIMARY_LAYOUT, SP.LAYOUTS[5], SP.LAYOUTS[-1]):
        gate = SP.layout_gate(SP.FakeDevice(layout=lay), np.random.default_rng(2), n=8)
        assert gate["ok"] and gate["layout"] == SP.layout_name(lay), gate["mismatches"]
        assert sum(v == 0 for v in gate["mismatches"].values()) == 1


def test_candidates_equal_the_core_kernel():
    rng = np.random.default_rng(3)
    core = TCM.BLACKWELL_SM120_E4M3_M16N8K32
    for name, As, M, Bl, C in SP.variants(rng, 16, {"randn_random_acc", "wide_spread", "subnormal"}):
        As, M, Bl, C = As[:8], M[:8], Bl[:8], C[:8]
        Al = SP.logical_a(As, M)
        two = fp8.step_model(core, Al[..., 32:], Bl[..., 32:], fp8.step_model(core, Al[..., :32], Bl[..., :32], C))
        assert np.array_equal(SP.evaluate(SP.CANDIDATES[SP.CHAINED], As, M, Bl, C), two), name
        if ".shared" in name:
            Bg = np.take_along_axis(Bl, np.broadcast_to(SP.columns(M[:, :1]), (As.shape[0], 8, 32)), -1)
            assert np.array_equal(SP.evaluate(SP.CANDIDATES[SP.GATHERED], As, M, Bl, C), fp8.step_model(core, As, Bg, C)), name


def test_shared_variant_keeps_the_family_products():
    """The shared variant's kept B columns are the family's B, which per_row carries in columns 0..31; both share the family's A."""
    got = {name: (As, M, Bl, C) for name, As, M, Bl, C in SP.variants(np.random.default_rng(4), 16, {"cancellation"})}
    assert set(got) == {f"cancellation.{m}{z}" for m in ("shared", "per_row") for z in ("", ".zero_acc")}
    As, M, Bl, C = got["cancellation.shared"]
    Ar, _, Br, _ = got["cancellation.per_row"]
    assert np.array_equal(As, Ar) and (M == M[:, :1]).all() and C.any() and not got["cancellation.shared.zero_acc"][3].any()
    assert np.array_equal(np.take_along_axis(Bl, np.broadcast_to(SP.columns(M[:, :1]), (As.shape[0], 8, 32)), -1), Br[..., :32])


@pytest.mark.slow  # 6-13 s on fake silicon
def test_fake_row_passes_refutes_and_corruption_fails(tmp_path, no_nvcc):
    code, res = fp8.run(fp8.parser().parse_args([*FAST, "--expect-model", SP.PRIMARY, "--out", str(tmp_path / "ok"), "--fake-silicon"]))
    assert code == 0 and res["validation"] == "passed", res["failures"]
    cap = res["capture"]
    assert cap["fits"] == [SP.PRIMARY] and cap["total_mismatches"]["twin:dense_compressed"] == 0
    assert cap["total_mismatches"]["twin:chain2"] > 0 and not any(cap["twin_checks"].values())
    assert all(n > 0 for c, n in cap["total_mismatches"].items() if c not in (SP.PRIMARY, "twin:dense_compressed"))
    assert {v.rsplit(".", 1)[-1] for v in cap["families"]} >= {"shared", "per_row", "zero_acc"}
    assert res["layout"]["sparse"]["layout"] == SP.layout_name(SP.PRIMARY_LAYOUT)
    assert res["w1"]["validation"] == "passed" and set(res["w1"]["ops"]) == {"qmma_e4m3_f32", "qmma_sp_e4m3"}
    assert res["w1"]["prices"]["price_units"]["qmma_sp_e4m3 per nonzero product"] == pytest.approx(1.0)
    assert not (tmp_path / "ok" / SP.CAPTURE_NPZ).exists() and (tmp_path / "ok" / "sparse_tiles.npz").exists()
    code, res = fp8.run(fp8.parser().parse_args([*FAST, "--expect-model", SP.PRIMARY, "--out", str(tmp_path / "bad"), "--fake-silicon",
                                                 "--corrupt-word", "5"]))
    assert code == 1 and any(f.startswith("--expect-model") for f in res["failures"]), res["failures"]


@pytest.mark.slow  # 6-13 s on fake silicon
def test_fake_capture_of_the_chained_model_fits_the_chain_twin(tmp_path, no_nvcc):
    code, res = fp8.run(fp8.parser().parse_args([*FAST, "--expect-model", SP.CHAINED, "--out", str(tmp_path), "--fake-silicon"]))
    assert code == 0, res["failures"]
    cap = res["capture"]
    assert cap["fits"] == [SP.CHAINED] and cap["total_mismatches"]["twin:chain2"] == 0
    assert cap["total_mismatches"]["twin:dense_compressed"] > 0


def test_expect_model_is_the_sparse_rows(tmp_path):
    assert fp8.main(["--row", "step", "--expect-model", SP.PRIMARY, "--out", str(tmp_path), "--fake-silicon"]) == 2
    assert fp8.main(["--row", "sparse", "--expect-model", "nope", "--out", str(tmp_path), "--fake-silicon"]) == 2
    assert fp8.main(["--row", "sparse", "--lease-max-min", "5", "--out", str(tmp_path), "--fake-silicon"]) == 2


def test_w1_sparse_gate_words_and_ops():
    rows = np.array(w1.GATE_ROW[w1.SP_OP], np.uint32)[None].repeat(32, 0)
    assert np.all(w1.expected_words(w1.SP_OP, 8, rows).view(np.float32) == 32 * w1.UNROLL * 8)
    t = w1.timing_rows(w1.SP_OP, np.random.default_rng(5))
    nib = (t[:, 6:7] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF
    assert np.isin(nib, w1.SP_META_VALID).all()
    assert w1.op_indices("qmma_sp_e4m3") == [0, w1.SP_OP] and w1.op_indices(None) == list(range(len(w1.OPS)))
    with pytest.raises(SystemExit):
        w1.op_indices("nope")
