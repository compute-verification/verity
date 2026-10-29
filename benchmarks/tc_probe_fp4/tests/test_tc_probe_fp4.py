"""The FP4 probe's and the W1 price bench's CPU side: the K=32 candidates against verity's own ``GroupSum``, the cast references
against brute force, the 2:4 sparse rows' stored-to-logical mapping and their ``gather_dense`` candidate against the dense pinned
model, the fake-silicon dry runs (each candidate the truth in turn must be the only one reproducing every word; one flipped word,
or a device reading the sparse metadata another way, must fail the gate), the W1 driver's gates, and the store declarations.  No test here builds or runs a kernel:
``nvcc_bin`` is pointed at nothing, so a machine with CUDA runs the same tests as one without.  The offline SASS gates run only
with ``VERITY_TC_PROBE_FP4_NVCC=1`` (nvcc 12.9 and cuobjdump, about 30 s)."""

from __future__ import annotations

import dataclasses
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
import f1_rates as F1  # noqa: E402
import f2_strassen as F2  # noqa: E402
import f3_lut as F3  # noqa: E402
from emulation import E  # noqa: E402
import k32  # noqa: E402
import lean_vectors as LV  # noqa: E402
import meta_rule as MR  # noqa: E402
import probe as P  # noqa: E402
import recheck as RC  # noqa: E402
import sp_e4m3 as SP  # noqa: E402
import sparse as S  # noqa: E402
import t1_merged as T  # noqa: E402
import tool  # noqa: E402
import w1_price as W  # noqa: E402

from verity.ml.circuits.silicon import models as M  # noqa: E402
from verity.ml.circuits.silicon.term import PRODUCT_WIDTH  # noqa: E402


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
    """On a single format the candidate is ``GroupSum`` itself; E4M3 (32,)/26/-133 is the vLLM port's measured sm_120 step."""
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


def test_fake_silicon_is_refused_with_a_run_id(no_nvcc, tmp_path, monkeypatch):
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


def test_scale_bit7_twins_differ_only_in_bit_7_and_the_model_reads_their_low_7_bits():
    rng = np.random.default_rng(7)
    A, Bt, SFA, SFB, C = P.scale_bit7_twins(rng, 4)
    assert (A[0::2] == A[1::2]).all() and (Bt[0::2] == Bt[1::2]).all() and (C[0::2] == C[1::2]).all()
    for sf in (SFA, SFB):
        assert ((sf[0::2] & 0x7F) == sf[1::2]).all() and (sf[1::2] < 0x80).all() and (sf[0::2] >= 0x80).mean() > 0.3
        assert [int(sf[0, r, 0]) for r in range(3)] == list(P.BIT7_PINNED)
    D, err = P.model_words(M.BLACKWELL_SM120_NVF4, A, Bt, SFA, SFB, C, 64, 4)
    # the pinned model reads a scale byte's low 7 bits, as sm_120 does: a bit-7 tile computes its twin's words, and both refuse
    # only where A's row 2 or B's row 2 carries 0xFF or 0x7F (NaN)
    nan = np.zeros((16, 8), bool)
    nan[2, :] = nan[:, 2] = True
    assert (err[0::2] == nan).all() and (err[1::2] == nan).all() and (D[0::2][~err[0::2]] == D[1::2][~err[1::2]]).all()

    hw = D.copy()
    hw[0::2] = D[1::2]  # a device that decodes the low 7 bits
    s = P.bit7_summary(hw, SFA, SFB, D, err)
    assert s["hardware_ignores_bit7"] and s["device_words_differ"] == 0 and s["low7_twin_vs_model"]["mismatches"] == 0
    hw[2, 1, 3] ^= 1
    s = P.bit7_summary(hw, SFA, SFB, D, err)
    assert not s["hardware_ignores_bit7"] and s["device_words_differ"] == 1
    assert s["first_differences"][0]["pair"] == 1 and (s["first_differences"][0]["i"], s["first_differences"][0]["j"]) == (1, 3)


def _lean_vectors_file(path: Path, rng) -> None:
    """A small ``Fp4Vectors.lean``: nvf4P7's entries 0-5 in the model (5 carries a bit-7 byte, ``d`` its low-7-bit twin's word),
    6 a bit-7 entry with ``none``; nvf4Chains' entry 0 in the model, entry 1 a bit-7 chain of 2; nvf4Specials' entry 0 a
    signalling-NaN accumulator with a bit-7 byte and the record-only word 0x7FFFFFFF."""
    def pack(xs, bits):
        return sum(int(x) << (bits * i) for i, x in enumerate(xs))

    def entry(n, bit7, known=True):
        a, b = rng.integers(0, 16, 64 * n), rng.integers(0, 16, 64 * n)
        sa = P.UE4M3_VALID[rng.integers(0, 120, 4 * n)]
        sb = P.UE4M3_VALID[rng.integers(0, 120, 4 * n)]
        acc = int(P.f32_bits(np.float32(rng.standard_normal())))
        v = {"acc": acc, "n": n, "a": pack(a, 4), "b": pack(b, 4), "sa": pack(sa, 8), "sb": pack(sb, 8), "d": None}
        v["d"] = LV.model_word(M.BLACKWELL_SM120_NVF4, v, False) if known else None
        if bit7:
            sa = sa.copy()
            sa[1] |= 0x80
            v["sa"] = pack(sa, 8)
        return v

    def lean(v, chain):
        d = "none" if v["d"] is None else f"some 0x{v['d']:08X}"
        mid = f"{v['n']}, " if chain else ""
        return f"  ⟨0x{v['acc']:08X}, {mid}0x{v['a']:X}, 0x{v['b']:X}, 0x{v['sa']:X}, 0x{v['sb']:X}, {d}⟩"

    p7 = [entry(1, False) for _ in range(5)] + [entry(1, True), entry(1, True, known=False)]
    ch = [entry(2, False), entry(2, True)]
    sp = [{**entry(1, True), "acc": 0xFFACB8FF, "d": 0x7FFFFFFF}]
    path.write_text("import Definitions.Pouw.PearlC.Fp4Chain\n\n/-- vectors -/\ndef nvf4P7 : List Vec := [\n" + ",\n".join(lean(v, False) for v in p7)
                    + "]\n\n/-- chains -/\ndef nvf4Chains : List ChainVec := [\n" + ",\n".join(lean(v, True) for v in ch) + "]\n"
                    + "\n/-- specials -/\ndef nvf4Specials : List Vec := [\n" + ",\n".join(lean(v, False) for v in sp) + "]\n")


def test_lean_vectors_capture_gates_the_layout_and_reports_the_prediction(no_nvcc, tmp_path):
    """The whole driver on the CPU: a device that decodes the low 7 bits matches every prediction, as the pinned model as-is now
    does, and a control whose file word is wrong, or a selected entry with no prediction, fails the run with no selected word
    recorded."""
    vec = tmp_path / "Fp4Vectors.lean"
    _lean_vectors_file(vec, np.random.default_rng(3))
    base = ["--instruction", "nvf4_lean_vectors", "--vectors", str(vec), "--vector-ids", "5", "--chain-ids", "1"]
    assert P.main(base + ["--fake-silicon", "low7", "--out", str(tmp_path / "low7")]) == 0
    doc = json.loads((tmp_path / "low7" / "probe_results.json").read_text())
    assert doc["summary"] == {"selected": 2, "match_prediction": 2, "match_low7_twin": 2, "match_low7_decode": 2,
                              "pinned_model_refuses": 0, "outside_model": 0}
    assert doc["gates"]["controls"]["vectors"] == 6 and doc["gates"]["controls"]["chains"] == 1
    assert doc["gates"]["background"]["mismatches"] == 0 and doc["gates"]["background"]["checked"] > 0
    assert doc["gates"]["poison"]["words_reading_poison"] == 0 and doc["gates"]["no_write_control"]["rejected"]
    assert [v["vector"] for v in doc["vectors"]] == ["nvf4P7[5]", "nvf4Chains[1]"] and doc["vectors"][1]["n"] == 2
    assert P.main(base + ["--special-ids", "0", "--fake-silicon", "low7", "--out", str(tmp_path / "special")]) == 0
    sp = json.loads((tmp_path / "special" / "probe_results.json").read_text())["vectors"][2]
    assert (sp["vector"], sp["outside_model"], sp["pinned_model_low7_twin"], sp["low7_decode_predicts"], sp["device"]) == \
        ("nvf4Specials[0]", "NaN accumulator", None, "7fffffff", "7fffffff") and sp["matches_low7_decode"]
    assert P.main(base + ["--fake-silicon", "verity", "--out", str(tmp_path / "as-is")]) == 0
    doc = json.loads((tmp_path / "as-is" / "probe_results.json").read_text())
    assert doc["summary"]["match_prediction"] == 2 and doc["summary"]["pinned_model_refuses"] == 0

    assert P.main(base[:4] + ["--vector-ids", "6", "--fake-silicon", "low7", "--out", str(tmp_path / "none")]) == 1
    doc = json.loads((tmp_path / "none" / "probe_results.json").read_text())
    assert "carries no predicted word" in doc["failures"][0] and "vectors" not in doc
    text = vec.read_text()
    first = LV.parse_list(text, "nvf4P7")[0]["d"]
    vec.write_text(text.replace(f"some 0x{first:08X}", f"some 0x{first ^ 1:08X}", 1))
    assert P.main(base + ["--fake-silicon", "low7", "--out", str(tmp_path / "control")]) == 1
    doc = json.loads((tmp_path / "control" / "probe_results.json").read_text())
    assert doc["gates"]["controls"]["mismatches"][0]["vector"] == "nvf4P7[0]" and "vectors" not in doc


def test_lean_vectors_poison_no_write_control_and_collect(no_nvcc, tmp_path, monkeypatch):
    """A kernel that leaves a word unwritten, or a launch switch that does not stop the launch, fails the capture; ``--collect``
    derives a chunk's gates and words again from its dumps and publishes them, and refuses a fake chunk, another library, and
    dumps or records that were changed."""
    vec = tmp_path / "Fp4Vectors.lean"
    _lean_vectors_file(vec, np.random.default_rng(3))
    base = ["--instruction", "nvf4_lean_vectors", "--vectors", str(vec), "--vector-ids", "5", "--chain-ids", "1", "--special-ids", "0"]
    rows = LV.FakeLow7.rows

    def unwritten(self, *a, **kw):
        D = rows(self, *a, **kw).copy()
        D[0, 5, 5] = LV.POISON
        return D

    monkeypatch.setattr(LV.FakeLow7, "rows", unwritten)
    assert P.main(base + ["--fake-silicon", "low7", "--out", str(tmp_path / "unwritten")]) == 1
    doc = json.loads((tmp_path / "unwritten" / "probe_results.json").read_text())
    assert doc["gates"]["poison"]["words_reading_poison"] > 0 and "vectors" not in doc
    monkeypatch.setattr(LV.FakeLow7, "rows", rows)
    monkeypatch.setattr(LV.FakeLow7, "set_launch", lambda self, on: LV.POISON & 0xFF)
    assert P.main(base + ["--fake-silicon", "low7", "--out", str(tmp_path / "switch")]) == 1
    doc = json.loads((tmp_path / "switch" / "probe_results.json").read_text())
    assert not doc["gates"]["no_write_control"]["rejected"] and doc["gates"]["no_write_control"]["words_reading_poison"] == 0
    monkeypatch.undo()

    chunk, lib = tmp_path / "chunk", tmp_path / "libmma_fp4_sm_120a.so"
    assert P.main(base + ["--fake-silicon", "low7", "--out", str(chunk)]) == 0
    assert (chunk / "dumps.npz").exists()
    lib.write_bytes(b"library")
    collect = base + ["--collect", str(chunk), "--prebuilt", str(lib)]
    assert P.main(collect + ["--out", str(tmp_path / "fake")]) == 1
    assert "fake silicon" in json.loads((tmp_path / "fake" / "probe_results.json").read_text())["failures"][0]
    rec = json.loads((chunk / "probe_results.json").read_text())
    rec.update(fake_silicon=None, expect_uuid="GPU-x", library={"path": str(lib), "sha256": LV.sha256(lib), "nvcc": "prebuilt"})
    rec["identity"].update(backend="mma.sync", gpu_uuid="GPU-x")
    (chunk / "probe_results.json").write_text(json.dumps(rec))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    assert P.main(collect + ["--out", str(tmp_path / "ok")]) == 0
    doc = json.loads((tmp_path / "ok" / "probe_results.json").read_text())
    res = json.loads((tmp_path / "ok" / "result.json").read_text())
    assert doc["vectors"] == rec["vectors"] and res["validation"]["status"] == "passed" and res["run_id"] == "r-test"
    assert res["collected_from"]["dumps_sha256"] == LV.sha256(chunk / "dumps.npz") and res["words"]["nvf4Specials[0]"] == "7fffffff"
    assert LV.sha256(tmp_path / "ok" / "chunk" / "dumps.npz") == LV.sha256(chunk / "dumps.npz")
    lib.write_bytes(b"another library")
    assert P.main(collect + ["--out", str(tmp_path / "lib")]) == 1
    lib.write_bytes(b"library")
    rec["vectors"][0]["device"] = "00000000"
    (chunk / "probe_results.json").write_text(json.dumps(rec))
    assert P.main(collect + ["--out", str(tmp_path / "record")]) == 1
    assert "differ from the ones derived again" in json.loads((tmp_path / "record" / "probe_results.json").read_text())["failures"][0]
    dumps, nowrite = LV.load_dumps(chunk / "dumps.npz")
    dumps["n1_r0_c0"]["D_rows"][0, 0, 0] ^= 1
    dumps["n1_r0_c0"]["D_tile"][0, 0, 0] ^= 1
    LV.save_dumps(chunk / "dumps.npz", dumps, nowrite)
    assert P.main(collect + ["--out", str(tmp_path / "dumps")]) == 1
    assert "placements disagree" in json.loads((tmp_path / "dumps" / "probe_results.json").read_text())["failures"][0]


def test_recheck_capture_resumes_and_verify_derives_every_word(no_nvcc, tmp_path, monkeypatch):
    """The fill chunk stops at its budget (exit 99) and resumes at the first family without a record; the CPU verify regenerates
    the operands from the seed and passes, and refuses a fake capture, a changed words file and regenerated operands that differ."""
    cap = ["capture", "--instruction", "nvf4", "--seeds", "5,6", "--n-random", "16", "--fake-silicon", "verity", "--budget-s", "0",
           "--out", str(tmp_path / "c")]
    rcs = [RC.main(cap) for _ in range(60)]
    assert rcs[0] == 99 and 0 in rcs and rcs.index(0) > 20
    d = tmp_path / "c" / "nvf4" / "5"
    ver = ["verify", "--instruction", "nvf4", "--seed", "5", "--n-random", "16", "--capture", str(tmp_path / "c"), "--procs", "1"]
    assert RC.main(ver + ["--out", str(tmp_path / "fake")]) == 1
    done = json.loads((d / "done.json").read_text())
    done.update(fake_silicon=None, expect_uuid="GPU-x")
    done["identity"].update(backend="mma.sync", gpu_uuid="GPU-x")
    (d / "done.json").write_text(json.dumps(done))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    assert RC.main(ver + ["--out", str(tmp_path / "ok")]) == 0
    doc = json.loads((tmp_path / "ok" / "recheck.json").read_text())
    assert doc["min_gated_family_elements"] == 16 // 8 * 128 and {"specials", "scale_bit7"} <= set(doc["families"])
    assert json.loads((tmp_path / "ok" / "result.json").read_text())["validation"]["status"] == "passed"
    assert RC.main(ver[:4] + ["6"] + ver[5:] + ["--out", str(tmp_path / "other")]) == 1  # seed 6 is still a fake capture
    D = np.load(d / "random_random_scales.npy")
    D[0, 0, 0] ^= 1
    np.save(d / "random_random_scales.npy", D)
    assert RC.main(ver + ["--out", str(tmp_path / "changed")]) == 1
    assert "changed since the capture" in json.loads((tmp_path / "changed" / "recheck.json").read_text())["failures"][0]
    assert RC.main(["verify", "--instruction", "nvf4", "--seed", "5", "--n-random", "24", "--capture", str(tmp_path / "c"),
                    "--procs", "1", "--out", str(tmp_path / "n")]) == 1
    p = tool.TC_RECHECK_FP4.parse(["python", "benchmarks/tc_probe_fp4/recheck.py", "verify", "--instruction", "nvf4", "--seed", "5",
                                   "--n-random", "16", "--capture", "/c", "--prebuilt", "/x.so", "--procs", "8", "--out", "x"])
    assert p == {"cmd": "verify", "instruction": "nvf4", "seeds": None, "expect_model": None, "fake_silicon": None, "seed": 5,
                 "n_random": 16, "fixed_mult": 1}


def test_recheck_gates_e2m1_on_the_registry_model_and_scales_the_fixed_families(no_nvcc, tmp_path):
    """Unscaled E2M1's verify gates on ``gs32_w26_native`` (BLACKWELL_SM120_E2M1_M16N8K32's parameters) without --expect-model,
    so a capture of that model passes and one of the refuted block-scaled adder fails; ``--fixed-mult`` scales the fixed-size gated
    families and must match between capture and verify."""
    assert RC.primary_model("f8f6f4", None) == "gs32_w26_native" and RC.primary_model("nvf4", None) == "verity"
    for truth, rc in (("gs32_w26_native", 0), ("bsaa_g16", 1)):
        out = tmp_path / truth
        cap = ["capture", "--instruction", "f8f6f4", "--seeds", "7", "--n-random", "16", "--fixed-mult", "2", "--fake-silicon",
               truth, "--out", str(out)]
        assert RC.main(cap) == 0
        done = json.loads((out / "f8f6f4" / "7" / "done.json").read_text())
        assert done["fixed_mult"] == 2 and done["families"]["signed_zero"]["tiles"] == 512
        done.update(fake_silicon=None, expect_uuid="GPU-x")
        done["identity"].update(backend="mma.sync", gpu_uuid="GPU-x")
        (out / "f8f6f4" / "7" / "done.json").write_text(json.dumps(done))
        ver = ["verify", "--instruction", "f8f6f4", "--seed", "7", "--n-random", "16", "--capture", str(out), "--procs", "1"]
        assert RC.main(ver + ["--fixed-mult", "2", "--out", str(out / "v")]) == rc
        assert json.loads((out / "v" / "recheck.json").read_text())["primary_model"] == "gs32_w26_native"
        assert RC.main(ver + ["--out", str(out / "v1")]) == 1


def test_sparse_e4m3_packs_registers_and_verify_derives_every_word(no_nvcc, tmp_path):
    """The host's register packing puts each operand byte where the fragment layout says; a fake capture of one group over the 32
    kept products resumes at its budget, verify refuses it until it is marked as gated, then finds that candidate (and the dense
    twin on the compressed operands) reproducing every sparse word while the two chained atoms do not, and a changed words file
    fails."""
    rng = np.random.default_rng(0)
    As = rng.integers(0, 256, (2, 16, 32)).astype(np.uint8)
    R = SP.pack_a(As)
    B = rng.integers(0, 256, (2, 8, 64)).astype(np.uint8)
    RB = SP._pack_b(B, SP.B_READINGS["k_4q_16r"])
    M = SP.META_VALID[rng.integers(0, 6, (2, 16, 16))]
    E = SP.pack_meta(M, "row_q1_groups_q2")
    for lane in (0, 6, 31):
        g, q = lane >> 2, lane & 3
        for r in range(4):
            for j in range(4):
                assert (int(R[1, lane, r]) >> (8 * j)) & 0xFF == As[1, g + 8 * (r & 1), 4 * q + j + 16 * (r >> 1)]
                assert (int(RB[1, lane, r]) >> (8 * j)) & 0xFF == B[1, g, 4 * q + j + 16 * r]
        assert all((int(E[1, lane]) >> (4 * c)) & 0xF == M[1, g + 8 * (q & 1), 8 * (q >> 1) + c] for c in range(8))
    C = rng.integers(0, 1 << 32, (2, 16, 8), dtype=np.uint64).astype(np.uint32)
    assert np.array_equal(SP.unpack_d(SP.pack_c(C)), C)
    assert np.array_equal(SP.expand(As, M)[..., SP.logical_columns(M)[0, 0]][0, 0], As[0, 0])

    cap = ["capture", "--seeds", "5", "--n-random", "16", "--fake-silicon", "gather_g32", "--budget-s", "0", "--out", str(tmp_path / "c")]
    rcs = [SP.main(cap) for _ in range(14)]
    assert rcs[0] == 99 and rcs[-1] == 0
    ver = ["verify", "--seed", "5", "--n-random", "16", "--capture", str(tmp_path / "c"), "--procs", "1"]
    assert SP.main(ver + ["--out", str(tmp_path / "fake")]) == 1
    d = tmp_path / "c" / "5"
    done = json.loads((d / "done.json").read_text())
    done.update(fake_silicon=None, expect_uuid="GPU-x")
    done["identity"].update(backend="mma.sync", gpu_uuid="GPU-x")
    (d / "done.json").write_text(json.dumps(done))
    rcs = [SP.main(ver + ["--budget-s", "0", "--out", str(tmp_path / "ok")]) for _ in range(11)]
    assert rcs[0] == 99 and rcs[-1] == 0
    assert SP.main(ver + ["--out", str(tmp_path / "whole")]) == 0
    f = json.loads((tmp_path / "ok" / "sparse.json").read_text())["finding"]
    assert f == json.loads((tmp_path / "whole" / "sparse.json").read_text())["finding"]
    assert f["candidates_reproducing_every_sparse_word"] == ["gather_g32"]
    assert f["sparse_vs_dense"]["dense_on_compressed"]["differing_words"] == 0 < f["sparse_vs_dense"]["dense_two_atoms"]["differing_words"]
    D = np.load(d / "random_acc.sparse.npy")
    D[0, 0, 0] ^= 1
    np.save(d / "random_acc.sparse.npy", D)
    assert SP.main(ver + ["--out", str(tmp_path / "changed")]) == 1
    assert "changed since the capture" in json.loads((tmp_path / "changed" / "sparse.json").read_text())["failures"][0]
    p = tool.TC_SPARSE_E4M3.parse(["python", "benchmarks/tc_probe_fp4/sp_e4m3.py", "verify", "--seed", "5", "--n-random", "16",
                                   "--capture", "/c", "--prebuilt", "/x.so", "--procs", "8", "--out", "x"])
    assert p == {"cmd": "verify", "seeds": None, "fake_silicon": None, "arch": "sm_120a", "seed": 5, "n_random": 16, "fixed_mult": 1}


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


def test_t1_emulation_is_1ad1aaa2_verbatim():
    assert T.emulation_gate() is None


@pytest.mark.parametrize("fam", ["gaussian", "t4", "massive", "spread", "spread-both"])
def test_t1_quantizer_without_the_shortcut_is_routes_quantizer(fam):
    """Without the pruning and at routes' per-16 scales, the capture's quantizer draws and returns what routes' does."""
    x1, x2 = (E.family(fam, "A", 8, 512, rng) for rng in (np.random.default_rng(4), np.random.default_rng(4)))
    rng1, rng2 = np.random.default_rng(9), np.random.default_rng(9)
    want_c, want_s = E.quantize_pearl_c4(x1, 0.25, rng1)
    got_c, got_s, keep = T.quantize(x2, 0.25, rng2, prune=False, block=16)
    assert keep is None and np.array_equal(want_c, got_c) and np.array_equal(want_s, got_s)
    assert rng1.integers(1 << 30) == rng2.integers(1 << 30)


def test_t1_operands_are_2of4_under_d24_and_the_hardware():
    o = T.operands("t4", 20260930, rows=16, cols=8, k=512)
    assert T.structure_failures(o) == []
    nz = (o.ca & 7) != 0
    assert (nz.reshape(16, -1, 4).sum(axis=2) <= 2).all()
    pairs = nz.reshape(16, -1, 4, 2).any(axis=3)
    p0, p1 = o.meta & 3, o.meta >> 2
    allowed = np.zeros_like(pairs)
    np.put_along_axis(allowed, p0[..., None].astype(np.int64), True, axis=2)
    np.put_along_axis(allowed, p1[..., None].astype(np.int64), True, axis=2)
    assert not (pairs & ~allowed).any()
    assert np.array_equal(o.sa16[:, 0::2], o.sa16[:, 1::2])


def test_t1_vectorised_fakes_are_the_pinned_models():
    """The dry run's devices: the dense one is BLACKWELL_SM120_NVF4 chained, the sparse ones sparse.gather_dense and
    logical_g16 (the emulation's merged_k128), word for word on hot and cold accumulators."""
    rng = np.random.default_rng(12)
    n = 6
    As = rng.integers(0, 16, (n, 16, 64)).astype(np.uint8)
    META = S.META_VALID[rng.integers(0, 6, (n, 16, 16))]
    B = rng.integers(0, 16, (n, 8, 128)).astype(np.uint8)
    SFA, SFB = P.UE4M3_VALID[rng.integers(0, 127, (n, 16, 4))], P.UE4M3_VALID[rng.integers(0, 127, (n, 8, 4))]
    C = P.f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-10, 30, (n, 16, 8)))).astype(np.float32))
    C[0] = 0
    A, SA, SB = S.expand(As, META), S.model_scales(SFA), S.model_scales(SFB)
    models = S.candidates("sp_nvf4")
    for fake, model in (("gather4", "gather_dense"), ("merged_k128", "logical_g16")):
        want, err = S.model_words(models[model], A, B, SA, SB, C)
        assert not err.any() and np.array_equal(T.FakeSparseT1(fake).tiles("sp_nvf4", As, B, META, SFA, SFB, C), want)
    two = S.model_words(S.TwoHalves("two", M.BLACKWELL_SM120_NVF4), A, B, SA, SB, C)[0]
    assert np.array_equal(T.FakeDenseT1().tiles("nvf4", A, B, C, SA, SB, steps=2), two)


@pytest.mark.slow
def test_t1_dry_run_records_the_finding_and_fails_closed(no_nvcc, tmp_path):
    """The whole driver on the CPU at routes' size (one family, one hot h): the dense words are the pinned chain's, and one
    flipped dense word fails the run."""
    base = ["--instruction", "sp_nvf4_t1", "--fake-silicon", "gather4", "--seed", "20260930", "--families", "gaussian",
            "--hot-bits", "14", "--model-sample", "64"]
    assert P.main(base + ["--out", str(tmp_path / "ok")]) == 0
    doc = json.loads((tmp_path / "ok" / "probe_results.json").read_text())
    for name, r in doc["families"]["gaussian"].items():
        assert r["emulation"]["native_vs_dense_hw"] == {"span": r["span"]["words"], "chain": r["chain"]["words"]}
        assert r["emulation"]["gather4_vs_sparse_hw"] == {"span": r["span"]["words"], "chain": r["chain"]["words"]}
        assert r["span"]["matched_exact"] == r["span"]["exact_spans"]
        assert r["pinned_sample"]["gather_dense_mismatches"] == r["pinned_sample"]["dense_mismatches"] == 0
    assert doc["finding"]["cold"]["gaussian"]["span"] == [262144, 262144]
    assert doc["finding"]["h14"]["gaussian"]["span"][0] < 262144
    assert (tmp_path / "ok" / "t1_gaussian.npz").exists()
    assert P.main(base + ["--out", str(tmp_path / "bad"), "--corrupt-word"]) == 1


def test_meta_rule_predicates_on_ptx_ordered_pairs():
    """Part C on PTX's six nibbles: the hardware's patterns are the ones touching at most two pairs, and D-24 and the hardware
    each admit patterns the other refuses."""
    t = MR.predicate_tables({m: S.OFFSETS[m].tolist() for m in S.META_VALID.tolist()})
    assert t["counts"] == {"hw_ordered": 67, "hw_all16": 67, "d24": 121, "hw_ordered_and_d24": 57}
    assert t["hw_ordered_is_at_most_two_pairs"] and t["hw_all16_equals_hw_ordered"]
    assert t["hw_not_d24"]["count"] == 10 and [0, 1, 2, 3] in t["hw_not_d24"]["examples"]
    assert t["d24_not_hw"]["count"] == 64
    assert MR.predicate_tables({m: MR.pairs_from_bits(m) for m in range(16)})["hw_all16_equals_hw_ordered"]


def test_meta_rule_part_a_decodes_weights_and_blocks():
    for instr in ("sp_nvf4", "sp_mxf4"):
        kind = S.SCALE_KIND[instr]
        for variant, (a, b) in MR.CODES.items():
            for blk, e in enumerate(MR.BLOCK_EXPS[kind]):
                for w in (1, 2, 3, 15):
                    word = P.f32_bits(np.float32(w * MR.E2M1_VALUE[a] * MR.E2M1_VALUE[b] * 2.0 ** e))
                    assert [int(x[0]) for x in MR.decode(np.array([word]), instr, variant)] == [w, blk], (instr, variant, e, w)
        assert [int(x) for x in MR.decode(np.array([0, 0x80000000, 0x3F800001], np.uint32), instr, "a")[0]] == [0, 0, -1]


@pytest.mark.slow
def test_meta_rule_dry_run_measures_the_fake_rule_and_fails_closed(no_nvcc, tmp_path, monkeypatch):
    """The whole driver on the CPU: the fake's rule comes back nibble for nibble, a changed unordered nibble is a finding (not a
    failure), and a changed ordered nibble fails the run."""
    base = ["--instruction", "sp_nvf4_meta", "--fake-silicon", "pairs_from_bits", "--seed", "1", "--n-random", "64", "--procs", "1"]
    assert P.main(base + ["--out", str(tmp_path / "ok")]) == 0
    doc = json.loads((tmp_path / "ok" / "probe_results.json").read_text())
    assert doc["part_a"]["pairs_from_bits"]["differs"] == [] and doc["part_a"]["variants_agree"]
    assert all(doc["part_a"]["blocks"].values()) and doc["part_b"]["mismatches"] == 0 == doc["part_b2"]["differing_words"]
    assert doc["part_c"]["counts"]["hw_all16"] == 67

    def patched(changes):
        off = MR.gather_offsets({m: MR.pairs_from_bits(m) for m in range(16)})
        for m, o in changes.items():
            off[m] = o

        def init(self, instr, name):
            self.model_name, self.off = name, off
        monkeypatch.setattr(MR.FakeMeta, "__init__", init)

    patched({0: [-1, -1, -1, -1], 5: [3, 2, -1, -1]})
    assert P.main(base + ["--out", str(tmp_path / "unordered")]) == 0
    doc = json.loads((tmp_path / "unordered" / "probe_results.json").read_text())
    assert doc["rule"]["0"] == [None] * 4 and doc["rule"]["5"] == [3, 2, None, None]
    assert doc["part_a"]["pairs_from_bits"]["differs"] == [0, 5]
    patched({4: [1, 0, 2, 3]})
    assert P.main(base + ["--out", str(tmp_path / "ordered")]) == 1
    doc = json.loads((tmp_path / "ordered" / "probe_results.json").read_text())
    assert doc["failures"][0].startswith("layout check failed") and "part_a" not in doc
    assert not list((tmp_path / "ordered").glob("meta_part_*.npz"))


def test_store_declarations_keep_guards_out_of_the_derivation():
    p = tool.TC_PROBE_FP4.parse(["python", "benchmarks/tc_probe_fp4/probe.py", "--instruction", "f8f6f4", "--out", "x",
                                 "--expect-uuid", "GPU-1", "--chain", "4", "--prebuilt", "/x/libmma_fp4_sm_120a.so"])
    assert p["instruction"] == "f8f6f4" and p["chain"] == 4 and "expect_uuid" not in p and p["arch"] == "sm_120a"
    assert "prebuilt" not in p and "_unknown" not in p
    assert set(p) <= set(tool.TC_PROBE_FP4.key_params) | {"out", "procs"}
    v = tool.TC_PROBE_FP4.parse(["python", "benchmarks/tc_probe_fp4/probe.py", "--instruction", "nvf4_lean_vectors", "--out", "x",
                                 "--vectors", "in/Fp4Vectors.lean", "--vector-ids", "31,34", "--chain-ids", "9"])
    assert (v["vectors"], v["vector_list"], v["vector_ids"], v["chain_ids"]) == ("in/Fp4Vectors.lean", "nvf4P7", "31,34", "9")
    assert "_unknown" not in v and "vectors" in tool.TC_PROBE_FP4.param_artifacts and v["collect"] is False
    c = tool.TC_PROBE_FP4.parse(["python", "benchmarks/tc_probe_fp4/probe.py", "--instruction", "nvf4_lean_vectors", "--out", "x",
                                 "--vectors", "in/Fp4Vectors.lean", "--special-ids", "12,19", "--collect", "/fill-out/run"])
    assert c["special_ids"] == "12,19" and c["collect"] is True and "_unknown" not in c
    assert set(c) <= set(tool.TC_PROBE_FP4.key_params) | {"out", "procs"}
    q = tool.TC_PRICE_FP4.parse(["python", "benchmarks/tc_probe_fp4/w1_price.py", "--out", "x", "--rows", "mxf4,nvf4,",
                                 "--expect-uuid", "GPU-1", "--max-seconds", "60"])
    assert q["rows"] == "mxf4,nvf4" and not {"out", "expect_uuid", "max_seconds"} & set(q)
    assert set(q) <= set(tool.TC_PRICE_FP4.key_params)
    r = tool.TC_RATES_FP4.parse(["python", "benchmarks/tc_probe_fp4/f1_rates.py", "run", "--build", "b", "--family", "fp4", "--out", "x",
                                 "--expect-uuid", "GPU-1", "--locked-mhz", "2100"])
    assert r["cmd"] == "run" and r["family"] == "fp4" and r["locked_mhz"] == 2100
    assert not {"out", "build", "expect_uuid"} & set(r) and set(r) <= set(tool.TC_RATES_FP4.key_params)
    f3 = tool.TC_LUT_FP4.parse(["python", "benchmarks/tc_probe_fp4/f3_lut.py", "run", "--build", "b", "--width", "g2c4", "--out", "x",
                                "--expect-uuid", "GPU-1", "--locked-mhz", "2100"])
    assert f3["cmd"] == "run" and f3["width"] == "g2c4" and f3["locked_mhz"] == 2100 and "_unknown" not in f3
    assert not {"out", "build", "runs", "expect_uuid"} & set(f3) and set(f3) <= set(tool.TC_LUT_FP4.key_params)
    f2 = tool.TC_STRASSEN_FP4.parse(["python", "benchmarks/tc_probe_fp4/f2_strassen.py", "run", "--build", "b", "--level", "l2", "--out", "x",
                                     "--expect-uuid", "GPU-1", "--locked-mhz", "2100"])
    assert f2["cmd"] == "run" and f2["level"] == "l2" and f2["locked_mhz"] == 2100 and "_unknown" not in f2
    assert not {"out", "build", "runs", "expect_uuid"} & set(f2) and set(f2) <= set(tool.TC_STRASSEN_FP4.key_params)


def test_f1_kernels_are_body_chains_of_their_form():
    for f in F1.FORMS_TENSOR:
        src = F1.tensor_kernel(f)
        assert src.count("asm volatile(") == F1.BODY, f.name
        asm = F1.tensor_asm(f)
        assert (", 0x0" in asm) == bool(f.sparse) and ("{0, 0}" in asm) == bool(f.scale), f.name
        assert asm.count('"+') == f.acc[1] and "ftz" not in src.lower()
        assert f.words_per_set() == (2 if f.a[0] == "d" else 1) * f.a[1] + (2 if f.b[0] == "d" else 1) * f.b[1] + 3
    for op in F1.OPS:
        assert "ftz" not in F1.simt_kernel(op).lower() and F1.op_words_per_chain(op) >= 2
    # one warp instruction of FP16 m8n8k4 is four quad-pair products; every other shape is one
    assert {f.name for f in F1.FORMS_TENSOR if f.copies != 1} == {"f16_m8n8k4", "f16_m8n8k4_f16acc"}
    assert F1.BY_NAME["f16_m8n8k4"].logical_macs == 1024 and F1.BY_NAME["sp_nvf4"].products == 16 * 8 * 64
    assert F1._flags_ok(F1.NVCC_FLAGS) and not F1._flags_ok(["-O2", "--use_fast_math"]) and not F1._flags_ok(["-ftz=true"])
    assert set(F1.FAMILIES) >= {"fp4", "f8f6f4", "mxf8f6f4", "fp8", "f16", "tf32", "int", "b1", "f64", "simt"}


def test_f1_ftz_scan_catches_every_fp32_ftz_form():
    hits = [m.group(0) for m in F1.FTZ_RE.finditer(
        "FFMA.FTZ R1, R2, R3, R4 ; FADD.FTZ.RZ R1 ; FMUL.RN.FTZ R2 ; FFMA R1 ; HFMA2.FTZ R1 ; DFMA R2 ; FMUL.FTZ.SAT R3")]
    assert hits == ["FFMA.FTZ", "FADD.FTZ", "FMUL.RN.FTZ", "FMUL.FTZ"]


def _max_abs(words: np.ndarray, fmt: str) -> float:
    """The largest magnitude among the elements packed in u32 operand words of ``fmt`` (finite minifloats, f16, bf16)."""
    if fmt in ("f16", "bf16"):
        h = np.stack([words & 0xFFFF, words >> 16], -1).astype(np.uint32)
        v = (h << 16).view(np.float32) if fmt == "bf16" else h.astype(np.uint16).view(np.float16)
        return float(np.abs(v.astype(np.float64)).max())
    ebits, mbits = {"e4m3": (4, 3), "e5m2": (5, 2)}.get(fmt) or F1._NARROW[fmt][:2]
    code = ((words[..., None] >> (8 * np.arange(4, dtype=np.uint32))) & 0xFF) >> F1._CONTAINER_SHIFT.get(fmt, 0)
    e, m = (code >> mbits) & ((1 << ebits) - 1), code & ((1 << mbits) - 1)
    bias = (1 << (ebits - 1)) - 1
    v = np.where(e == 0, m / 2**mbits * 2.0 ** (1 - bias), (1 + m / 2**mbits) * 2.0 ** (e.astype(np.int64) - bias))
    return float(v.max())


def test_f1_operands_stay_in_each_forms_domain():
    for f in F1.FORMS_TENSOR:
        w = F1.tensor_inputs(f, 64, 1)
        assert w.shape == (64, F1.NSETS, f.words_per_set()) and w.dtype == np.uint32
        e = w[..., -3]
        if f.sparse:
            nibs = (e[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF
            assert np.isin(nibs, F1.META_12 if f.sparse == "1:2" else F1.META_24).all(), f.name
            assert (len(np.unique(e)) == 1) == (f.meta == "const"), f.name
        sf = w[..., -2:]
        if f.scale:
            b = (sf[..., None] >> (8 * np.arange(4, dtype=np.uint32))) & 0xFF
            lo, hi = (0x30, 0x47) if f.scale == "ue4m3" else (124, 131)
            assert ((b >= lo) & (b <= hi)).all(), f.name
        if f.fa == "e4m3":
            b = (w[..., 0, None] >> (8 * np.arange(4, dtype=np.uint32))) & 0xFF
            assert (((b >> 3) & 0xF) <= 5).all() and (((b >> 3) & 0xF) >= 1).all()
        if f.fa in ("e2m1", "e2m3", "e3m2"):
            # PTX ISA: E2M1 in the container's central 4 bits, E3M2 and E2M3 in its low 6, zeros elsewhere
            b = (w[..., 0, None] >> (8 * np.arange(4, dtype=np.uint32))) & 0xFF
            sh = {"e2m1": 2, "e2m3": 0, "e3m2": 0}[f.fa]
            assert not (b & ((1 << sh) - 1)).any() and (b >> 6 == 0).all(), f.name
            ebits, mbits, nsmall = F1._NARROW[f.fa]
            mag = (b >> sh) & ((1 << (ebits + mbits)) - 1)
            # an FP16 accumulator gets the smallest codes (a chain of millions of MMAs stays finite); FP32 the whole code range
            assert (mag.max() < nsmall) == (f.acc[0] == "h"), f.name
        if f.acc[0] == "h":
            # every K-sum at most 16: half an FP16 ulp once |D| >= 2^15, so a chain on fixed operands stalls short of inf
            na = (2 if f.a[0] == "d" else 1) * f.a[1]
            per_output = f.mnk[2] // (2 if f.sparse else 1)
            assert per_output * _max_abs(w[..., :na], f.fa) * _max_abs(w[..., na:na + f.b[1]], f.fb) <= 16, f.name
    for op in F1.OPS:
        w = F1.simt_inputs(op, 32, 1)
        assert w.shape == (32, F1.SIMT_CH, F1.op_words_per_chain(op))
        if op.x == "f":
            assert np.isfinite(w[..., : 3 if op.z else 2].view(np.float32)).all()


SASS_EMULATED = """
        Function : f1_s4_k64
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   CALL.REL.NOINC 0x40 ;
        /*0020*/                   UISETP.NE.S64.AND UP0, UPT, UR4, URZ, UPT ;
        /*0030*/              @UP0 BRA.U 0x10 ;
        /*0040*/                   LOP3.LUT R3, R3, 0xf, R5, 0x78, !PT ;
        /*0050*/                   CALL.REL.NOINC 0x80 ;
        /*0060*/                   IMMA.16832.S8.S8 R4, R8, R12, R4 ;
        /*0070*/                   RET.REL.NODEC R20 0x0 ;
        /*0080*/                   IMMA.16832.S8.S8 R4, R8, R12, R4 ;
        /*0090*/                   RET.REL.NODEC R22 0x0 ;
"""


def test_f1_sass_gate_counts_nested_callees_and_fails_closed(monkeypatch):
    calls = F1.calls_of(SASS_EMULATED)["f1_s4_k64"]
    assert calls == [{"LOP3.LUT": 1, "CALL.REL.NOINC": 1, "IMMA.16832.S8.S8": 2}]
    monkeypatch.setattr(F1, "BODY", 1)
    s4 = F1.BY_NAME["s4_k64"]
    g = F1.form_gate(s4, F1.loops_of(SASS_EMULATED)["f1_s4_k64"], calls)
    assert g["ok"] and g["emulated"] == "tensor" and g["sass_per_ptx"] == 2 and g["callee_mma_opcodes"] == {"IMMA.16832.S8.S8": 2}
    assert not F1.form_gate(s4, {"CALL.REL.NOINC": 1}, [{"?CALL": 1, "IMMA.16832.S8.S8": 1}])["ok"]
    assert not F1.form_gate(s4, {"CALL.REL.NOINC": 1}, [{"LOP3.LUT": 4}])["ok"]
    assert not F1.form_gate(s4, {"CALL.REL.NOINC": 1}, [{"IMMA.16832.S8.S8": 1}, {"IMMA.16832.S8.S8": 1}])["ok"]
    cuda_core = F1.form_gate(F1.BY_NAME["f16_m8n8k4"], {"CALL.REL.NOINC": 1}, [{"FFMA": 32, "SHFL.IDX": 12}])
    assert cuda_core["ok"] and cuda_core["emulated"] == "cuda-core"
    assert F1.form_gate(F1.BY_NAME["nvf4"], {"OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X": 1}, [])["native"]
    assert not F1.form_gate(F1.BY_NAME["nvf4"], {"MOV": 3}, [])["ok"]
    monkeypatch.undo()
    want = F1.SIMT_CH * F1.SIMT_U
    assert F1.form_gate(F1.BY_NAME["hmul2_f16"], {"HMUL2": want // 2, "HFMA2": want // 2})["ok"]
    assert not F1.form_gate(F1.BY_NAME["ffma_f32"], {"FFMA": want - 1})["ok"]


def _f1_fake_build(d: Path, names: list[str], ok: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    gate = {n: {"ok": True, "mma_opcodes": {}, "loop_opcodes": {}} for n in names}
    (d / "build.json").write_text(json.dumps({"kernels": names, "ok": ok, "gate": gate, "refused": {"mxf4nvf4_4x_ue8m0": ["Illegal"]},
                                              "so": "none.so", "so_sha256": "0" * 64, "source_sha256": "1" * 64, "nvcc": "none",
                                              "nvcc_flags": F1.NVCC_FLAGS, "ftz_fp32": [], "emulated": [],
                                              "sass_opcodes_reached": ["OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"]}))
    return d


def test_f1_dry_run_prices_against_nvfp4_and_its_gates_fail_closed(no_nvcc, tmp_path):
    names = F1.FAMILIES["fp4"] + F1.FAMILIES["fp8"] + F1.FAMILIES["tf32"]
    b = _f1_fake_build(tmp_path / "b", names)
    common = ["--build", str(b), "--fake-silicon", "--reps", "2", "--sustain-s", "0", "--target-ms", "1"]
    for fam in ("fp4", "fp8", "tf32"):
        assert F1.main(["run", "--family", fam, "--out", str(tmp_path / "r")] + common) == 0
    fp8 = json.loads((tmp_path / "r" / "fp8.fake.json").read_text())
    # the fake: FP8 at half dense NVFP4's products per clock, TF32 at a quarter; the unit is timed in each chunk
    assert fp8["per_form"]["nvf4"]["fp4_slots_per_product"] == pytest.approx(1.0)
    assert fp8["per_form"]["fp8_e4m3_k32"]["fp4_slots_per_product"] == pytest.approx(2.0)
    assert fp8["per_form"]["sp_fp8_e4m3_k64"]["fp4_slots_per_logical_mac"] == pytest.approx(1.0)
    assert fp8["clocks"] == "unlocked" and not fp8["failures"]
    assert set(fp8["timed_length_gate"]) == set(fp8["per_form"]) and all(v["finite"] for v in fp8["timed_length_gate"].values())

    assert F1.main(["run", "--family", "fp4", "--out", str(tmp_path / "bad"), "--corrupt-word"] + common) == 1
    bad = json.loads((tmp_path / "bad" / "fp4.fake.json").read_text())
    assert bad["failures"] and "per_form" not in bad
    nb = _f1_fake_build(tmp_path / "nb", names, ok=False)
    assert F1.main(["run", "--family", "fp4", "--out", str(tmp_path / "nb_r"), "--build", str(nb), "--fake-silicon"]) == 1
    assert "per_form" not in json.loads((tmp_path / "nb_r" / "fp4.fake.json").read_text())

    # collect prices every row, and refuses fake silicon and missing families
    assert F1.main(["collect", "--build", str(b), "--runs", str(tmp_path / "r"), "--out", str(tmp_path / "t")]) == 1
    t = json.loads((tmp_path / "t" / "f1_table.json").read_text())
    assert t["rows"]["tf32_k8"]["fp4_slots_per_product"] == pytest.approx(4.0) and "nvf4@tf32" in t["rows"]
    assert not t["claim"]["cheaper_like_for_like"] and not t["claim"]["holds_over_measured"]
    assert "int" in t["missing_families"] and t["refused_by_ptxas"] == {"mxf4nvf4_4x_ue8m0": ["Illegal"]}
    res = json.loads((tmp_path / "t" / "result.json").read_text())
    assert res["validation"]["status"] == "failed" and "fake silicon" in res["validation"]["detail"]
    assert sorted(p.name for p in (tmp_path / "t" / "inputs").iterdir()) == ["build.json", "fp4.fake.json", "fp8.fake.json",
                                                                             "tf32.fake.json"]


def test_f3_variants_cover_every_width_and_arm():
    table = F3.variants()
    assert sorted(table) == sorted((arm, g, c) for g, c in F3.WIDTHS.values() for arm in F3.ARMS)
    assert F3.nt_of() == 8 and F1._flags_ok(F3.NVCC_FLAGS)


@pytest.mark.parametrize("width", sorted(F3.WIDTHS))
def test_f3_tables_sum_to_the_group_dots_and_pack_the_scales(width):
    """Summing a row's 16/G lookups of one block in u32 and unpacking lo + 65536 hi gives each column's exact group dot."""
    g, c = F3.WIDTHS[width]
    ng, nkg = min(F3.ng_of(g), 2), 64 // g
    ops = F3.make_operands("mixed", F3.LUT_TPB * ng, ng, c, 5)
    pk = F3.pack(ops, g)
    assert pk.tables.shape == (ng, nkg, 16 ** g, c // 2) and pk.arows.shape == (ops.acodes.shape[0], F3.AROW)
    # the kernels read memory order: every packed array is C-contiguous, and a strided one is refused at the boundary
    assert all(F3._p(getattr(pk, f.name)) for f in dataclasses.fields(pk))
    with pytest.raises(ValueError):
        F3._p(np.zeros((4, 4), np.uint32).T)
    codes, scales = F3.unpack_rows(pk.arows)
    assert np.array_equal(codes, ops.acodes) and np.array_equal(scales, ops.ascale)
    grp = ops.groups
    idx = (ops.acodes.reshape(-1, nkg, g).astype(np.int64) << (4 * np.arange(g))).sum(-1)
    ent = pk.tables[grp[:, None], np.arange(nkg)[None, :], idx]
    s = ent.reshape(-1, 4, nkg // 4, c // 2).sum(2, dtype=np.uint32)
    lo = (s & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.int64)
    hi = (s.view(np.int32).astype(np.int64) - lo) >> 16
    gd, _, _, part = F3.terms(ops.acodes, ops.ascale, ops.wcodes, ops.wscale, grp)
    assert np.array_equal(lo.transpose(0, 2, 1), gd[:, 0::2]) and np.array_equal(hi.transpose(0, 2, 1), gd[:, 1::2])
    w = pk.wsc.astype(np.int64)
    mb, eb = E.ue4m3_dyadic(ops.wscale.astype(np.int64))
    assert np.array_equal(w & 15, mb) and np.array_equal(((w >> 4) & 15) - 10, eb)
    assert np.array_equal((w >> 8) & 0xFFFF, (((ops.wcodes.reshape(ng, c, 4, 16) & 7) != 0) << np.arange(16)).sum(-1))
    # the mixed family reaches every participation case and keeps its scales and accumulators in the model's domain
    assert part.any() and not part.all() and (ops.ascale < 0x7F).all() and (ops.wscale < 0x7F).all()
    subnormal = ((ops.c0 & 0x7F800000) == 0) & ((ops.c0 & 0x7FFFFF) != 0)
    assert np.isfinite(ops.c0.view(np.float32)).all() and (ops.c0 == 0x80000000).any() and subnormal.any()


def test_f3_sass_gate_counts_lookups_and_fails_closed():
    table = [("dots", 1, 2), ("exact", 4, 2)]
    fn = {"dots": "_ZN12_GLOBAL__N_110lut_kernelILi0ELi1ELi2EEEvPKj", "exact": "_ZN12_GLOBAL__N_110lut_kernelILi1ELi4ELi2EEEvPKj",
          "native": "_ZN12_GLOBAL__N_113native_kernelEPKj"}
    clean = {"registers": 40, "spill_stores": 0, "spill_loads": 0}
    loops = {fn["dots"]: {"LDS": 64, "IADD3": 70}, fn["exact"]: {"LDG.E.CONSTANT": 16, "IMAD": 50},
             fn["native"]: {F3.NATIVE_OPCODE: 8, "BRA": 1}}
    ptxas = {v: dict(clean) for v in fn.values()}
    g = F3.sass_gate(loops, ptxas, table, 8)
    assert all(v["ok"] for v in g.values()) and set(g) == {"dots_g1c2", "exact_g4c2", "native"}
    for bad_loops, bad_ptxas in (({**loops, fn["dots"]: {"LDS": 63}}, ptxas),
                                 ({**loops, fn["dots"]: {"LDS": 64, "LDG.E.CONSTANT": 1}}, ptxas),
                                 ({**loops, fn["exact"]: {"LDG.E.CONSTANT": 16, "HMMA.16816.F32": 1}}, ptxas),
                                 ({**loops, fn["native"]: {F3.NATIVE_OPCODE: 7}}, ptxas),
                                 (loops, {**ptxas, fn["exact"]: {**clean, "spill_loads": 4}}),
                                 ({k: v for k, v in loops.items() if k != fn["dots"]}, ptxas)):
        assert not all(v["ok"] for v in F3.sass_gate(bad_loops, bad_ptxas, table, 8).values())
    rep = F3.ptxas_report("ptxas info    : Compiling entry function '%s' for 'sm_120a'\nptxas info    : Function properties for %s\n"
                          "    0 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads\nptxas info    : Used 247 registers, used 1 "
                          "barriers" % (fn["dots"], fn["dots"]))
    assert rep == {fn["dots"]: {"spill_stores": 8, "spill_loads": 8, "registers": 247}}


def _f3_fake_build(d: Path, ok: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "build.json").write_text(json.dumps({"schema": F3.SCHEMA + "/build", "variants": F3.variants(), "nt": F3.nt_of(), "ok": ok,
                                              "gate": {"native": {"ok": ok}}, "so": "none.so", "so_sha256": "0" * 64,
                                              "source_sha256": "1" * 64, "nvcc": "none", "nvcc_flags": F3.NVCC_FLAGS, "ftz_fp32": []}))
    return d


def test_f3_dry_run_is_bit_exact_against_the_atom_and_fails_closed(no_nvcc, tmp_path):
    """The fake runs the LUT arms through the packed tables and scale words and the native arm through the pinned model's
    vectorised step on the codes, so a passing gate means the host's tables reproduce the atom's words."""
    b = _f3_fake_build(tmp_path / "b")
    common = ["--build", str(b), "--fake-silicon", "--reps", "2", "--sustain-s", "0", "--target-ms", "1", "--sample", "128"]
    assert F3.main(["run", "--width", "g2c2", "--out", str(tmp_path / "r")] + common) == 0
    r = json.loads((tmp_path / "r" / "g2c2.fake.json").read_text())
    assert not r["failures"] and set(r["gate"]) == {f"{f}/L{n}" for f in F3.FAMILIES for n in F3.GATE_LENGTHS}
    assert all(v["model_sample"] == 128 and v["native_model_mismatches"] == 0 and v["exact_words"] > 0 for v in r["gate"].values())
    assert all(v["ok"] and v["exact_length"] >= 1000 for v in r["timed_length_gate"].values())
    # the fake: lookups at 32 per SM per clock (4 products each at g2c2), NVFP4 at 2,048 products
    assert r["per_arm"]["uniform/dots"]["fp4_slots_per_product"] == pytest.approx(2048 / 128)
    assert r["per_arm"]["uniform/dots"]["fp4_slots_per_lookup"] == pytest.approx(64)
    assert r["per_arm"]["gaussian/native"]["fp4_slots_per_product"] == pytest.approx(1.0) and r["clocks"] == "unlocked"

    assert F3.main(["run", "--width", "g2c2", "--out", str(tmp_path / "bad"), "--corrupt-word"] + common) == 1
    bad = json.loads((tmp_path / "bad" / "g2c2.fake.json").read_text())
    assert bad["failures"] and bad["gate"]["mixed/L1"]["exact_native_mismatches"] == 4 and "per_arm" not in bad
    nb = _f3_fake_build(tmp_path / "nb", ok=False)
    assert F3.main(["run", "--width", "g2c2", "--out", str(tmp_path / "nb_r"), "--build", str(nb)] + common[2:]) == 1
    assert "per_arm" not in json.loads((tmp_path / "nb_r" / "g2c2.fake.json").read_text())

    assert F3.main(["collect", "--build", str(b), "--runs", str(tmp_path / "r"), "--out", str(tmp_path / "t")]) == 1
    t = json.loads((tmp_path / "t" / "f3_lut.json").read_text())
    assert t["rows"]["g2c2/uniform/exact"]["fp4_slots_per_product"] == pytest.approx(3 * 2048 / 128)
    assert not t["claim"]["cheaper_exact"] and not t["claim"]["holds_over_measured"] and "g1c2" in t["missing_widths"]
    res = json.loads((tmp_path / "t" / "result.json").read_text())
    assert res["validation"]["status"] == "failed" and "fake silicon" in res["validation"]["detail"]


@pytest.mark.parametrize("level", [1, 2])
def test_f2_strassen_multiplies_non_commuting_blocks_in_7_to_the_l_products(level):
    """U and V's pre-additions, one product per p on 16 x KB by KB x 8 blocks and W's recombination give A B exactly, the
    pre-added codes stay within 2^L x 12 halves (exact in TF32), and the kernel's W1 table is the host's."""
    sh, rng = F2.Shape(level), np.random.default_rng(level)
    a, b = rng.integers(-12, 13, (3, sh.tm, 16)), rng.integers(-12, 13, (3, 16, sh.tn))
    ac, bc = F2.combos(a, b, level)
    assert ac.shape == (3, 7 ** level, 16, sh.kb) and bc.shape == (3, 7 ** level, sh.kb, 8)
    c = np.einsum("IJp,tpij->tIiJj", F2.coeffs(level)[2], np.einsum("tpik,tpkj->tpij", ac, bc)).reshape(3, sh.tm, sh.tn)
    assert np.array_equal(c, a @ b)
    assert max(np.abs(ac).max(), np.abs(bc).max()) <= 12 * 2 ** level and np.abs(np.einsum("tpik,tpkj->tpij", ac, bc)).max() < 2 ** 14
    assert np.array_equal(F2.cu_w1(), F2.W1)


@pytest.mark.parametrize("kb", [8, 4])
def test_f2_fragment_positions_cover_each_operand_element_once(kb):
    a, b = np.zeros((16, kb), int), np.zeros((kb, 8), int)
    for r, c in F2.a_positions(kb):
        np.add.at(a, (r, c), 1)
    for r, c in F2.b_positions(kb):
        np.add.at(b, (r, c), 1)
    assert (a == 1).all() and (b == 1).all()


def test_f2_lane_sums_follow_the_c_fragment():
    """c0..c3 of lane (g, t) are C's (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1) in every 16 x 8 block; the checksum sums them in u32."""
    sh = F2.Shape(2)
    out = np.random.default_rng(0).integers(0, 2 ** 32, (2, sh.tm, sh.tn), dtype=np.uint64).astype(np.uint32)
    got = F2.lane_sums(out.reshape(-1), sh)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for o, (r, c) in enumerate([(g, 2 * t), (g, 2 * t + 1), (g + 8, 2 * t), (g + 8, 2 * t + 1)]):
            want = sum(int(out[1, 16 * i + r, 8 * j + c]) for i in range(sh.nb) for j in range(sh.nb)) & 0xFFFFFFFF
            assert got[1, lane, o] == want


def test_f2_tf32_rounding_ties_away_and_keeps_the_codes_exact():
    x = np.array([1 + 2 ** -11, -(1 + 2 ** -11), 1 + 3 * 2 ** -12, 1 + 2 ** -12, 2 ** -20 * (1 + 2 ** -11)])
    assert F2.tf32_rna(x).tolist() == [1 + 2 ** -10, -(1 + 2 ** -10), 1 + 2 ** -10, 1.0, 2 ** -20 * (1 + 2 ** -10)]
    ints = np.arange(-48, 49)
    assert np.array_equal(F2.tf32_rna(ints), ints)
    assert not (F2.tf32_words(np.array([3.0, 1 + 2 ** -11, -48.0])) & 0x1FFF).any()


def test_f2_sass_gate_counts_the_strassen_mmas_and_fails_closed():
    pre = "_ZN47_GLOBAL__N__187ee200_14_f2_strassen_cu_f2_shape"
    fn = {"dots_l1": pre + "15strassen_kernelILi1EEEvPKjS2_ixPjS3_Px", "dots_l2": pre + "15strassen_kernelILi2EEEvPKjS2_ixPjS3_Px",
          "native": pre + "13native_kernelEPKjS1_S1_iiiixPjPx", "step": pre + "11step_kernelEPKjPKiS3_S1_S1_xiPj"}
    clean = {"registers": 79, "spill_stores": 0, "spill_loads": 0}
    loops = {fn["dots_l1"]: {F2.TF32_OPCODE[1]: 7, "FADD": 32}, fn["dots_l2"]: {F2.TF32_OPCODE[2]: 49, "FADD": 480},
             fn["native"]: {F2.NATIVE_OPCODE: 8, "BRA": 1}, fn["step"]: {"IMAD": 20}}
    ptxas = {v: dict(clean) for v in fn.values()}
    g = F2.sass_gate(loops, ptxas, 8)
    assert all(v["ok"] for v in g.values()) and set(g) == set(fn) and g["dots_l2"]["fadd"] == 480
    for bad_loops, bad_ptxas in (({**loops, fn["dots_l1"]: {F2.TF32_OPCODE[1]: 8}}, ptxas),
                                 ({**loops, fn["dots_l2"]: {F2.TF32_OPCODE[1]: 49}}, ptxas),
                                 ({**loops, fn["dots_l1"]: {F2.TF32_OPCODE[1]: 7, "LDS.128": 1}}, ptxas),
                                 ({**loops, fn["dots_l2"]: {F2.TF32_OPCODE[2]: 49, "LDG.E.CONSTANT": 1}}, ptxas),
                                 ({**loops, fn["native"]: {F2.NATIVE_OPCODE: 7}}, ptxas),
                                 ({**loops, fn["step"]: {"HMMA.16816.F32": 1}}, ptxas),
                                 (loops, {**ptxas, fn["dots_l2"]: {**clean, "spill_stores": 4}}),
                                 (loops, {k: v for k, v in ptxas.items() if k != fn["step"]}),
                                 ({k: v for k, v in loops.items() if k != fn["dots_l2"]}, ptxas)):
        assert not all(v["ok"] for v in F2.sass_gate(bad_loops, bad_ptxas, 8).values())


def _f2_fake_build(d: Path, ok: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "build.json").write_text(json.dumps({"schema": F2.SCHEMA + "/build", "nt": 8, "ok": ok, "gate": {"native": {"ok": ok}},
                                              "so": "none.so", "so_sha256": "0" * 64, "source_sha256": "1" * 64, "header_sha256": "2" * 64,
                                              "nvcc": "none", "nvcc_flags": F2.NVCC_FLAGS, "ftz_fp32": []}))
    return d


def test_f2_dry_run_is_bit_exact_against_the_atom_and_fails_closed(no_nvcc, tmp_path, monkeypatch):
    """The fake unpacks the TF32 fragments, multiplies them per product and recombines by W, and runs the native arm and the
    step through the pinned model's vectorised step, so a passing gate means the host's pre-additions, fragment layout and step
    inputs reproduce the atom's words."""
    b = _f2_fake_build(tmp_path / "b")
    common = ["--build", str(b), "--fake-silicon", "--reps", "2", "--sustain-s", "0", "--target-ms", "1", "--sample", "128"]
    assert F2.main(["run", "--level", "l1", "--out", str(tmp_path / "r")] + common) == 0
    r = json.loads((tmp_path / "r" / "l1.fake.json").read_text())
    assert not r["failures"] and set(r["gate"]) == set(F2.FAMILIES) and (r["mmas"], r["mmas_naive"], r["mma_shape"]) == (7, 8, "m16n8k8")
    for fam in r["gate"].values():
        assert all(v["ok"] and v["rounded_operands"] == 0 for v in fam["groups"].values())
        assert set(fam["words"]) == {f"L{n}" for n in F2.GATE_LENGTHS}
        assert all(w["ok"] and w["words"] == F2.NTILES * 32 * 16 and w["model_sample"] == 128 for w in fam["words"].values())
    # recorded, never gated: TF32 rounds some pre-sums of the mixed family's scaled values
    assert r["gate"]["mixed"]["groups"]["0"]["scaled"]["rounded_operands"] > 0
    assert all(v["ok"] and v["length"] >= 1000 for v in r["timed_length_gate"].values())
    # the fake: TF32 m16n8k8 at 256 products per SM per clock, NVFP4 at 2,048, so 8 x 7/8 slots per product
    assert r["per_arm"]["uniform/dots"]["fp4_slots_per_product"] == pytest.approx(7.0)
    assert r["mma_floor_fp4_slots"]["value"] == pytest.approx(7.0) and r["clocks"] == "unlocked"

    assert F2.main(["run", "--level", "l1", "--out", str(tmp_path / "bad"), "--corrupt-word"] + common) == 1
    bad = json.loads((tmp_path / "bad" / "l1.fake.json").read_text())
    assert bad["failures"] and bad["gate"]["mixed"]["words"]["L1"]["route_native_mismatches"] == 1 and "per_arm" not in bad
    nb = _f2_fake_build(tmp_path / "nb", ok=False)
    assert F2.main(["run", "--level", "l1", "--out", str(tmp_path / "nb_r"), "--build", str(nb)] + common[2:]) == 1
    assert "per_arm" not in json.loads((tmp_path / "nb_r" / "l1.fake.json").read_text())

    assert F2.main(["collect", "--build", str(b), "--runs", str(tmp_path / "r"), "--out", str(tmp_path / "t")]) == 1
    t = json.loads((tmp_path / "t" / "f2_strassen.json").read_text())
    assert t["rows"]["l1/uniform/dots"]["fp4_slots_per_product"] == pytest.approx(7.0) and t["missing_levels"] == ["l2"]
    assert not t["claim"]["cheaper"] and not t["claim"]["holds_over_measured"]
    res = json.loads((tmp_path / "t" / "result.json").read_text())
    assert res["validation"]["status"] == "failed" and "fake silicon" in res["validation"]["detail"]
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    with pytest.raises(SystemExit):
        F2.main(["run", "--level", "l1", "--out", str(tmp_path / "never")] + common)
    assert not (tmp_path / "never").exists()


@pytest.mark.skipif(os.environ.get("VERITY_TC_PROBE_FP4_NVCC") != "1",
                    reason="set VERITY_TC_PROBE_FP4_NVCC=1 to build the kernels and run the offline SASS gates (nvcc 12.9, ~30 s)")
def test_offline_sass_gates(tmp_path):
    env = dict(os.environ)
    env.pop("RESEARCH_RUN_ID", None)
    for cmd in (["w1_price.py", "--out", str(tmp_path / "w1"), "--sass-only"],
                ["probe.py", "--instruction", "f8f6f4", "--out", str(tmp_path / "p"), "--sass-only"],
                ["probe.py", "--instruction", "cvt_e2m1x2_f32", "--out", str(tmp_path / "c"), "--sass-only"],
                ["probe.py", "--instruction", "sp_nvf4", "--out", str(tmp_path / "sp"), "--sass-only"],
                ["probe.py", "--instruction", "sp_nvf4_t1", "--out", str(tmp_path / "t1"), "--sass-only"],
                ["probe.py", "--instruction", "sp_nvf4_meta", "--out", str(tmp_path / "meta"), "--sass-only"],
                ["f1_rates.py", "build", "--out", str(tmp_path / "f1")],
                ["f2_strassen.py", "build", "--out", str(tmp_path / "f2")],
                ["f3_lut.py", "build", "--out", str(tmp_path / "f3")],
                ["stage0_sm120.py", "--out", str(tmp_path / "s0")]):
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (cmd, r.stdout[-2000:], r.stderr[-2000:])
