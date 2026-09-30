"""The FP4 probe's and the W1 price bench's CPU side: the K=32 candidates against verity's own ``GroupSum``, the cast references
against brute force, the 2:4 sparse rows' stored-to-logical mapping and their ``gather_dense`` candidate against the dense pinned
model, the fake-silicon dry runs (each candidate the truth in turn must be the only one reproducing every word; one flipped word,
or a device reading the sparse metadata another way, must fail the gate), the W1 driver's gates, and the store declarations.  No test here builds or runs a kernel:
``nvcc_bin`` is pointed at nothing, so a machine with CUDA runs the same tests as one without.  The offline SASS gates run only
with ``VERITY_TC_PROBE_FP4_NVCC=1`` (nvcc 12.9 and cuobjdump, about 30 s)."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import cast  # noqa: E402
import k32  # noqa: E402
import probe as P  # noqa: E402
import sparse as S  # noqa: E402
import tool  # noqa: E402
import w1_price as W  # noqa: E402

from verity.ml.tc import models as M  # noqa: E402
from verity.ml.tc.term import PRODUCT_WIDTH  # noqa: E402


@pytest.fixture
def no_nvcc(monkeypatch):
    monkeypatch.setattr(P, "nvcc_bin", lambda: "/nonexistent/nvcc")
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)


def _value(term) -> Fraction:
    v = Fraction(term.magnitude) * Fraction(2) ** (term.exponent - 23)
    return -v if term.negative else v


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@pytest.mark.parametrize("fa,fb", [("e2m1", "e2m1"), ("e2m1", "e4m3"), ("e4m3", "e2m1")])
def test_mixed_product_is_the_exact_product(fa, fb):
    codes = {"e2m1": range(16), "e4m3": [c for c in range(256) if c not in k32.E4M3_NAN]}
    for a in codes[fa]:
        for b in codes[fb]:
            t = k32.mixed_product(fa, fb, a, b)
            want = Fraction(float(k32.VALUE[fa][a])) * Fraction(float(k32.VALUE[fb][b]))
            assert _value(t) == want, (fa, fb, a, b)
            assert t.magnitude < 1 << PRODUCT_WIDTH
            if fa == fb:
                assert t == M.product(fa, a, b)


@pytest.mark.parametrize("fmt,bits", [("e4m3", 8), ("e2m1", 4)])
def test_groupsum_candidate_is_verity_groupsum_on_one_format(fmt, bits):
    """On a single format the candidate is ``GroupSum`` itself; E4M3 (32,)/26/-133 is the vLLM lane's measured sm_120 step."""
    cand = k32.GroupSumCandidate("t", fmt, fmt, (32,), 26, -133)
    ref = M.GroupSum("t", "sm_120", fmt, bits, (32,), 26, -133, "test")
    rng = np.random.default_rng(7)
    for _ in range(300):
        a = k32._codes(rng, fmt, 32)
        b = k32._codes(rng, fmt, 32)
        acc = _f32(float(rng.normal()) * 2.0 ** int(rng.integers(-20, 12)))
        assert cand.step_scaled(acc, a, b) == ref.step(acc, [int(x) for x in a], [int(x) for x in b])


def test_e2m1_cast_reference_is_nearest_even_satfinite():
    rng = np.random.default_rng(11)
    grid = [Fraction(v) for v in (0, 0.5, 1, 1.5, 2, 3, 4, 6)]
    xs = np.concatenate([rng.uniform(-8, 8, 4000), [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.5, 7.0, 1e30, -0.0, -5.0]])
    words = np.array([_f32(float(x)) for x in xs], np.uint32)
    got = cast.ref_e2m1(words)
    for w, c in zip(words, got):
        x = Fraction(struct.unpack("<f", struct.pack("<I", int(w)))[0])
        d = [abs(abs(x) - g) for g in grid]
        best = min(range(8), key=lambda i: (d[i], i % 2))  # nearest; a tie goes to the even code; beyond 6 saturates to 6
        assert int(c) & 7 == best and bool(int(c) & 8) == bool(int(w) >> 31)


def test_e4m3_cast_reference_is_nearest_even_satfinite():
    rng = np.random.default_rng(12)
    finite = [(Fraction(float(k32.E4M3_VALUE[c])), c) for c in range(0x7F)]
    xs = np.concatenate([np.exp2(rng.uniform(-12, 10, 3000)), [float(v) for v, _ in finite[1:]], [2.0**-10, 464.0, 1e20]])
    words = np.array([_f32(float(x)) for x in xs], np.uint32)
    got = cast.ref_e4m3_pos(words)
    for w, c in zip(words, got):
        x = Fraction(struct.unpack("<f", struct.pack("<I", int(w)))[0])
        best = min(finite, key=lambda vc: (abs(x - vc[0]), vc[1] % 2))[1]
        assert int(c) == best, (float(x), int(c), best)


@pytest.mark.parametrize("instr,truth", [("f8f6f4", "gs32_w26_native"), ("f8f6f4", "bsaa_g16"),
                                         ("f8f6f4_e2m1_e4m3", "gs32_w26_as_e4m3"), ("mxf8f6f4_e2m1", "bsaa_g32")])
def test_fake_capture_singles_out_the_truth(no_nvcc, tmp_path, instr, truth):
    rc = P.main(["--instruction", instr, "--out", str(tmp_path), "--fake-silicon", truth, "--sweep", "--n-random", "16", "--procs", "1"])
    assert rc == 0
    finding = json.loads((tmp_path / "probe_results.json").read_text())["candidate_finding"]
    assert finding["agreeing"] == [truth], finding["mismatches"]


def test_one_flipped_word_fails_the_capture_gate(no_nvcc, tmp_path):
    base = ["--instruction", "f8f6f4", "--fake-silicon", "gs32_w26_native", "--expect-model", "gs32_w26_native", "--sweep",
            "--n-random", "16", "--procs", "1"]
    assert P.main(base + ["--out", str(tmp_path / "ok")]) == 0
    assert P.main(base + ["--out", str(tmp_path / "bad"), "--corrupt-word"]) != 0


def test_result_records_elide_chain_operands_only():
    one = P.hexrec(0x3F800000, [1] * 32, [2] * 32, 0x40000000)
    assert P.slim_record(one) == one
    chain = P.hexrec(0x3F800000, [1] * 32 * 1024, [2] * 32 * 1024, 0x40000000)
    slim = P.slim_record(chain)
    assert slim["acc"] == chain["acc"] and slim["d"] == chain["d"]
    assert slim["a"]["hex_chars"] == 32 * 1024 and len(json.dumps([slim] * 20)) < 1 << 16


def test_fake_silicon_is_refused_under_a_research_run(no_nvcc, tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    with pytest.raises(SystemExit):
        P.main(["--instruction", "f8f6f4", "--out", str(tmp_path), "--fake-silicon", "gs32_w26_native"])
    with pytest.raises(SystemExit):
        W.main(["--out", str(tmp_path), "--fake-silicon"])


@pytest.mark.parametrize("instr", ["cvt_e2m1x2_f32", "cvt_ue8m0x2_f32_rp", "cvt_e4m3x2_f32_ue4m3"])
def test_cast_dry_run_passes_and_a_flipped_word_fails(no_nvcc, tmp_path, instr):
    base = ["--instruction", instr, "--fake-silicon", "reference", "--n-random", "256"]
    assert P.main(base + ["--out", str(tmp_path / "ok")]) == 0
    assert P.main(base + ["--out", str(tmp_path / "bad"), "--corrupt-word"]) != 0


def test_sparse_expand_puts_each_stored_pair_at_its_metadata_pair():
    for v in S.META_VALID.tolist():
        p0, p1 = v & 3, v >> 2
        row = S.expand(np.arange(1, 65, dtype=np.int64)[None], np.full((1, 16), v, np.uint8))[0]
        for c in range(16):
            chunk = row[8 * c:8 * c + 8]
            assert chunk[2 * p0:2 * p0 + 2].tolist() == [4 * c + 1, 4 * c + 2]
            assert chunk[2 * p1:2 * p1 + 2].tolist() == [4 * c + 3, 4 * c + 4]
            assert np.count_nonzero(chunk) == 4


@pytest.mark.parametrize("instr", ["sp_nvf4", "sp_mxf4"])
def test_sparse_gather_dense_is_the_dense_model_on_the_compressed_operands(instr):
    """``gather_dense`` (k = 128 logical, groups of 32) is the pinned dense model on the stored A and the B its metadata selects."""
    rng = np.random.default_rng(5)
    model, dense = S.candidates(instr)["gather_dense"], S.PINNED[instr]
    nb = S.NBLK[instr]
    for _ in range(200):
        a = rng.integers(0, 16, 64).astype(np.uint8)
        meta = S.META_VALID[rng.integers(0, 6, 16)]
        b = rng.integers(0, 16, 128).astype(np.uint8)
        sa = (P.UE4M3_VALID[rng.integers(0, 127, nb)] if nb == 4 else rng.integers(100, 150, nb)).astype(np.uint8)
        sb = (P.UE4M3_VALID[rng.integers(0, 127, nb)] if nb == 4 else rng.integers(100, 150, nb)).astype(np.uint8)
        acc = _f32(float(rng.normal()) * 2.0 ** int(rng.integers(-30, 30)))
        got = model.step_scaled(acc, S.expand(a, meta).tolist(), b.tolist(), S.model_scales(sa).tolist(), S.model_scales(sb).tolist())
        want = dense.step_scaled(acc, a.tolist(), b[S.logical_columns(meta)].tolist(), sa.tolist(), sb.tolist())
        assert got == want


SPARSE_FAMILIES = "random_random_scales,per_row_metadata,grid_straddle"


@pytest.mark.parametrize("instr,truth", [("sp_nvf4", "gather_dense"), ("sp_nvf4", "logical_g16"), ("sp_nvf4", "two_halves"),
                                         ("sp_mxf4", "logical_g64"), ("sp_mxf4", "gather_dense_underflow_signed")])
def test_sparse_fake_capture_singles_out_the_truth(no_nvcc, tmp_path, instr, truth):
    families = SPARSE_FAMILIES + (",ue8m0_underflow" if instr == "sp_mxf4" else "")  # D13's sign shows below the subnormal grid only
    rc = P.main(["--instruction", instr, "--out", str(tmp_path), "--fake-silicon", truth, "--sweep", "--n-random", "16", "--procs", "1",
                 "--families", families])
    assert rc == 0
    finding = json.loads((tmp_path / "probe_results.json").read_text())["candidate_finding"]
    assert finding["agreeing"] == [truth], finding["mismatches"]


def test_sparse_dense_twins_are_the_identities_each_truth_implies(no_nvcc, tmp_path):
    """The dense twins compare hardware words (here the fake's): each candidate as the truth makes its own identity exact."""
    fams = "random_random_scales,noise_atom_single,noise_atom_pair"

    def twins(truth):
        out = tmp_path / truth
        assert P.main(["--instruction", "sp_nvf4", "--out", str(out), "--fake-silicon", truth, "--sweep", "--n-random", "16",
                       "--procs", "1", "--families", fams]) == 0
        return {k: v["differing_words"] for k, v in json.loads((out / "probe_results.json").read_text())["dense_twin_finding"].items()
                if k != "note"}

    assert twins("gather_dense")["dense_on_compressed"] == 0
    assert twins("logical_g16")["dense_atom"] == 0
    halves = twins("two_halves")
    assert halves["dense_two_atoms"] == 0 and halves["dense_atom"] == 0 and halves["dense_on_compressed"] > 0


@pytest.mark.parametrize("instr,model", [("nvf4", M.BLACKWELL_SM120_NVF4), ("mxf4", M.BLACKWELL_SM120_MXF4)])
def test_scale_split_twins_share_every_scaled_product_but_not_the_word(instr, model):
    """The twins' scaled group terms and accumulators are equal value for value, so no model reading only those tells them
    apart; the pinned model does, at the offset that lies between the two anchors only."""
    A, Bt, SFA, SFB, C = P.scale_split_twins(np.random.default_rng(3), 12, P.SCALE_KIND[instr])
    differ = {o: 0 for o in P.TWIN_OFFSETS}
    for t in range(A.shape[0] // 2):
        off = P.TWIN_OFFSETS[t % len(P.TWIN_OFFSETS)]
        for i in range(16):
            for j in range(8):
                words, values = [], []
                for u in (2 * t, 2 * t + 1):
                    args = (A[u, i].tolist(), Bt[u, j].tolist(), SFA[u, i].tolist(), SFB[u, j].tolist())
                    values.append([Fraction(m) * Fraction(2) ** x for m, x, _ in model.group_terms(*args)])
                    words.append(model.step_scaled(int(C[u, i, j]), *args))
                assert values[0] == values[1] and C[2 * t, i, j] == C[2 * t + 1, i, j]
                differ[off] += words[0] != words[1]
    assert differ[28] > 0 and differ[27] == 0 and differ[29] == 0, differ


def test_scale_split_twins_dry_run_reports_the_refutation(no_nvcc, tmp_path):
    rc = P.main(["--instruction", "nvf4", "--out", str(tmp_path), "--fake-silicon", "verity", "--sweep",
                 "--n-random", "96", "--procs", "1", "--families", "scale_split_twins"])
    assert rc == 0
    twins = json.loads((tmp_path / "probe_results.json").read_text())["sweep"]["families"]["scale_split_twins"]["twins"]
    assert twins["no_product_value_model_fits"] and twins["device_words_differ"]["off28"] > 0
    assert twins["device_words_differ"]["off27"] == twins["device_words_differ"]["off29"] == 0


def test_expect_uuid_takes_one_uuid_or_a_list(monkeypatch):
    assert P.uuid_allowed("GPU-a", "GPU-a") and P.uuid_allowed("GPU-a", "GPU-b, GPU-a")
    assert not P.uuid_allowed("GPU-a", "GPU-b") and not P.uuid_allowed(None, "GPU-a") and not P.uuid_allowed("", ",")
    monkeypatch.delenv("GPU_LEASE_UUID", raising=False)
    assert not P.uuid_allowed("GPU-a", "$GPU_LEASE_UUID")
    monkeypatch.setenv("GPU_LEASE_UUID", "GPU-a")
    assert P.uuid_allowed("GPU-a", "$GPU_LEASE_UUID") and not P.uuid_allowed("GPU-b", "$GPU_LEASE_UUID")


def test_sparse_gates_fail_closed(no_nvcc, tmp_path, monkeypatch):
    base = ["--instruction", "sp_nvf4", "--fake-silicon", "gather_dense", "--expect-model", "gather_dense", "--sweep", "--n-random", "8",
            "--procs", "1", "--families", SPARSE_FAMILIES]
    assert P.main(base + ["--out", str(tmp_path / "ok")]) == 0
    assert P.main(base + ["--out", str(tmp_path / "bad"), "--corrupt-word"]) == 1
    # a device that reads the packed metadata with the quad's two bits swapped: the layout gate stops the run and names the reading
    tiles = S.FakeSparse.tiles
    monkeypatch.setattr(S.FakeSparse, "tiles", lambda self, instr, As, B, META, *r: tiles(self, instr, As, B,
                                                                                        S.META_READINGS["quad_bits_swapped"](META), *r))
    assert P.main(base + ["--out", str(tmp_path / "meta")]) == 1
    doc = json.loads((tmp_path / "meta" / "probe_results.json").read_text())
    assert "['quad_bits_swapped']" in doc["failures"][0] and "sweep" not in doc
    assert not list((tmp_path / "meta").glob("tiles_*.npz"))


def test_w1_fragments_carry_only_the_row_formats():
    for row, (_, k, fa, fb, sf, _) in W.ROWS.items():
        f = W.fragments(row, 2, 1)
        by = f.view(np.uint8).reshape(f.shape + (4,))
        if k == 32:
            for fmt, sl in ((fa, slice(0, 4)), (fb, slice(4, 6))):
                x = by[:, :, sl]
                if fmt == "e2m1":
                    assert not (x & 0xC3).any()
                else:
                    assert not np.isin(x, list(k32.E4M3_NAN)).any()
        s = by[:, :, 8:10]
        if sf == "ue4m3":
            assert ((s >= 0x30) & (s < 0x48)).all()
        elif sf == "ue8m0":
            assert ((s >= 124) & (s < 132)).all()
        else:
            assert not s.any()
        assert not f[:, :, 11].any() and (row in W.SPARSE_ROWS or not f[:, :, 6:8].any())
        nib = (f[:, :, 10:11] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF
        assert np.isin(nib, S.META_VALID).all() if row in W.SPARSE_ROWS else not nib.any()


def test_w1_op_inputs_stay_in_each_op_domain():
    for op in W.OPS:
        x = W.op_inputs(op, 64, 1)
        assert x.shape == (64, W.CV_CHAINS, 2)
        v = x.view(np.float32)
        if op in ("cvt_e4m3x2_f32_ue4m3", "cvt_ue8m0x2_f32_rp", "rcp_approx"):
            assert (v[..., 0] > 0).all() and np.isfinite(v[..., 0]).all()
        if op == "fmul":
            assert (x[..., 1] == 0x3F800000).all()
        if op == "rcp_approx":
            assert not x[..., 1].any()
        if op == "cvt_f16x2_e4m3x2_ue4m3":
            assert not np.isin(x[..., 0] & 0x7F, [0x7F]).any() and not np.isin((x[..., 0] >> 8) & 0x7F, [0x7F]).any()


def test_w1_loop_histogram_reads_the_timed_loop_only():
    text = """
        Function : _Z_cvt_tput_kernelILi0EEvPKjxPjPx
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R2, R3, R4, RZ ;
        /*0020*/                   F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R5, R3, R4, RZ ;
        /*0030*/                   LOP3.LUT R3, R3, 0xff, R5, 0x78, !PT ;
        /*0040*/                   UISETP.NE.S64.AND UP0, UPT, UR4, URZ, UPT ;
        /*0050*/              @UP0 BRA.U 0x20 ;
        /*0060*/                   STG.E desc[UR4][R6.64], R3 ;
        /*0070*/                   EXIT ;
        /*0080*/                   BRA 0x80;
    """
    h = W.loops_of(text)["_Z_cvt_tput_kernelILi0EEvPKjxPjPx"]
    assert h == {"F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C": 1, "LOP3.LUT": 1, "UISETP.NE.S64.AND": 1, "BRA.U": 1}


def test_w1_dry_run_records_and_its_gates_fail_closed(no_nvcc, tmp_path):
    common = ["--fake-silicon", "--reps", "2", "--sustain-s", "0", "--target-ms", "2", "--latency-iters", "50"]
    assert W.main(["--out", str(tmp_path / "ok")] + common) == 0
    doc = json.loads((tmp_path / "ok" / "result.fake.json").read_text())
    assert doc["kind"] == "bench-result/v1" and doc["validation"]["status"] == "passed"
    assert doc["label"] == "Measured, unlocked clocks"
    assert set(doc["comparisons"]) == set(W.COMPARISONS)
    names = {m["name"] for m in doc["measurements"]}
    assert "nvf4_const_scales_flop_per_sm_per_clock" in names and "f8f6f4_latency_cycles_per_mma" in names
    assert "op_cvt_e2m1x2_f32_fp8_mac_per_element" in names and "nvf4_quantize_fp8_mac_per_element" in names
    # the fake's blocks end at 60..100% of the launch: the clock is the longest span's, never the median's
    mhz = {m["name"]: m["value"] for m in doc["measurements"] if m["name"].endswith("_sm_mhz_median")}
    assert mhz and all(v == pytest.approx(W.FakeW1.MHZ, rel=1e-3) for v in mhz.values())
    fp8 = doc["fp8_prices"]
    # the fake issues one K=32 MMA per 4 clocks per warp, K=64 per 8 and the sparse K=128 per 16: every row costs one FP8 MAC
    assert fp8["rows"]["f8f6f4_e4m3/const_scales"]["per_logical_mac"] == pytest.approx(1.0)
    assert fp8["rows"]["sp_nvf4/const_scales"]["per_logical_mac"] == pytest.approx(1.0)
    assert fp8["rows"]["sp_nvf4/const_scales"]["per_stored_mac"] == pytest.approx(2.0)
    per_el = fp8["ops"]["cvt_e2m1x2_f32"]["per_element"]
    assert fp8["ops"]["cvt_e2m1x2_f32"]["per_instruction"] == pytest.approx(2 * per_el)
    assert fp8["nvf4_quantize_per_element"]["value"] > per_el
    assert not (tmp_path / "ok" / "result.json").exists()

    assert W.main(["--out", str(tmp_path / "bad"), "--corrupt-word"] + common) == 1
    bad = json.loads((tmp_path / "bad" / "result.fake.json").read_text())
    assert bad["validation"]["status"] == "failed" and bad["measurements"] == []
    assert "per_variant" not in json.loads((tmp_path / "bad" / "w1_price.json").read_text())


def test_w1_refuses_a_plan_longer_than_the_window(no_nvcc, tmp_path):
    with pytest.raises(SystemExit):
        W.main(["--out", str(tmp_path), "--fake-silicon", "--reps", "1000", "--max-seconds", "60"])


def test_clock_label_needs_every_sample_at_the_lock():
    s = [{"clocks.sm": "2100"}, {"clocks.sm": "2092"}]
    assert W.clock_label(s, [2099.0, 2092.0], 2100) == "locked-2100"
    assert W.clock_label(s, [2099.0, 2050.0], 2100) == "unlocked"
    assert W.clock_label([{"clocks.sm": "2100"}, {"clocks.sm": "1900"}], [2100.0], 2100) == "unlocked"
    assert W.clock_label(s, [2100.0], None) == "unlocked"
    assert W.clock_label([None], [2100.0], 2100) == "unlocked"


def test_store_declarations_keep_guards_out_of_the_derivation():
    p = tool.TC_PROBE_FP4.parse(["python", "tools/tc_probe_fp4/probe.py", "--instruction", "f8f6f4", "--out", "x",
                                 "--expect-uuid", "GPU-1", "--chain", "4"])
    assert p["instruction"] == "f8f6f4" and p["chain"] == 4 and "expect_uuid" not in p and p["arch"] == "sm_120a"
    q = tool.TC_PRICE_FP4.parse(["python", "tools/tc_probe_fp4/w1_price.py", "--out", "x", "--rows", "mxf4,nvf4,",
                                 "--expect-uuid", "GPU-1", "--max-seconds", "60"])
    assert q["rows"] == "mxf4,nvf4" and not {"out", "expect_uuid", "max_seconds"} & set(q)
    assert set(q) <= set(tool.TC_PRICE_FP4.key_params)


@pytest.mark.skipif(os.environ.get("VERITY_TC_PROBE_FP4_NVCC") != "1",
                    reason="set VERITY_TC_PROBE_FP4_NVCC=1 to build the kernels and run the offline SASS gates (nvcc 12.9, ~30 s)")
def test_offline_sass_gates(tmp_path):
    env = dict(os.environ)
    env.pop("RESEARCH_RUN_ID", None)
    for cmd in (["w1_price.py", "--out", str(tmp_path / "w1"), "--sass-only"],
                ["probe.py", "--instruction", "f8f6f4", "--out", str(tmp_path / "p"), "--sass-only"],
                ["probe.py", "--instruction", "cvt_e2m1x2_f32", "--out", str(tmp_path / "c"), "--sass-only"],
                ["probe.py", "--instruction", "sp_nvf4", "--out", str(tmp_path / "sp"), "--sass-only"],
                ["stage0_sm120.py", "--out", str(tmp_path / "s0")]):
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (cmd, r.stdout[-2000:], r.stderr[-2000:])
