"""The FP4 probe's CPU side: the K=32 candidates against verity's own ``GroupSum``, the cast references
against brute force, the 2:4 sparse rows' stored-to-logical mapping and their ``gather_dense`` candidate against the dense pinned
model, the fake-silicon dry runs (each candidate the truth in turn must be the only one reproducing every word; one flipped word,
or a device reading the sparse metadata another way, must fail the gate), and the store declaration.  No test here builds or runs a kernel:
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
import lean_vectors as LV  # noqa: E402
import meta_rule as MR  # noqa: E402
import probe as P  # noqa: E402
import sparse as S  # noqa: E402
import tool  # noqa: E402

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


@pytest.mark.skipif(os.environ.get("VERITY_TC_PROBE_FP4_NVCC") != "1",
                    reason="set VERITY_TC_PROBE_FP4_NVCC=1 to build the kernels and run the offline SASS gates (nvcc 12.9, ~30 s)")
def test_offline_sass_gates(tmp_path):
    env = dict(os.environ)
    env.pop("RESEARCH_RUN_ID", None)
    for cmd in (["probe.py", "--instruction", "f8f6f4", "--out", str(tmp_path / "p"), "--sass-only"],
                ["probe.py", "--instruction", "cvt_e2m1x2_f32", "--out", str(tmp_path / "c"), "--sass-only"],
                ["probe.py", "--instruction", "sp_nvf4", "--out", str(tmp_path / "sp"), "--sass-only"],
                ["probe.py", "--instruction", "sp_nvf4_meta", "--out", str(tmp_path / "meta"), "--sass-only"],
                ["stage0_sm120.py", "--out", str(tmp_path / "s0")]):
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (cmd, r.stdout[-2000:], r.stderr[-2000:])
