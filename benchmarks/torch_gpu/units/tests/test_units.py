"""The units on the CPU: the twins' word operations against the primitives' own evaluators, each twin against a scalar
composition of those evaluators in its OPS order, the sweep's digest against a pure-Python one, and the build check's
verdicts on synthetic listings."""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import add as AD
import argmax as AM
import build_check
import fixture as F
import quant as Q
import rmsnorm as RN
import softmax as SM
import sweep as SW
import words as W
from verity.ml.circuits.ops import fp32, mufu, prims

F32_SPECIALS = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001, 0x00000001, 0x80000001,
                0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x3F800001, 0x4B000000, 0x42FE0000]


def f32_words(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    near = (rng.normal(0, 1, n).astype(np.float32) * np.float32(2.0) ** rng.integers(-130, 128, n).astype(np.float32))
    return np.concatenate([np.array(F32_SPECIALS, np.uint32), w, near.view(np.uint32)])


def pairs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    a = f32_words(n, seed)
    b = np.random.default_rng(seed + 1).permutation(f32_words(n, seed + 2))[:len(a)]
    sp = np.array(F32_SPECIALS, np.uint32)
    return np.concatenate([a, np.repeat(sp, len(sp))]), np.concatenate([b, np.tile(sp, len(sp))])


@pytest.mark.parametrize("ours, prim", [(W.fadd, fp32.F32AddV2), (W.fmul, fp32.F32MulV2), (W.fdiv, fp32.F32DivV2)])
def test_binary_words_are_the_primitives(ours, prim):
    a, b = pairs(3000, 1)
    got = ours(a, b)
    assert [int(g) for g in got] == [prim.evaluate(int(x), int(y)) for x, y in zip(a, b)]


def test_sub_is_add_of_the_negation():
    a, b = pairs(3000, 2)
    assert [int(g) for g in W.fsub(a, b)] == [fp32.F32AddV2.evaluate(int(x), int(y) ^ 0x80000000) for x, y in zip(a, b)]


def test_sqrt_rounds_once():
    """sqrt.rn: the f64 root rounded to f32 (an f64 root has the 2p + 2 bits that make the double rounding innocuous), -0
    itself, every NaN and negative nonzero operand 0x7FFFFFFF."""
    a = f32_words(20000, 3)
    want = []
    for w in a:
        v = struct.unpack("<f", struct.pack("<I", int(w)))[0]
        if math.isnan(v) or (v < 0):
            want.append(W.F32_NAN)
        else:
            want.append(struct.unpack("<I", struct.pack("<f", math.sqrt(v)))[0] if v else int(w))
    assert [int(g) for g in W.fsqrt(a)] == want


def test_widen_bf16_and_ex2_are_the_primitives():
    a = f32_words(5000, 4)
    assert [int(g) for g in W.bf16(a)] == [prims.F2fpBf16.evaluate(int(x)) for x in a]
    h = np.arange(0, 1 << 16, 7, dtype=np.uint32).astype(np.uint16)
    assert [int(g) for g in W.widen(h)] == [prims.Bf16ToF32.evaluate(int(x)) for x in h]
    assert [int(g) for g in W.ex2(a)] == [mufu.MufuEx2Ftz.evaluate(int(x)) for x in a]


def test_s8_is_rni_sat():
    a = np.concatenate([f32_words(5000, 5), (np.arange(-260, 260, dtype=np.float32) / np.float32(2)).view(np.uint32)])
    want = []
    for w in a:
        v = struct.unpack("<f", struct.pack("<I", int(w)))[0]
        want.append(0 if math.isnan(v) else max(-128, min(127, round(v) if math.isfinite(v) else (127 if v > 0 else -128))))
    assert [int(g) for g in W.s8(a).view(np.int8)] == want


def test_order_nan_and_relu16():
    h = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    vals = W.widen(h).view(np.float32)
    nan = np.isnan(vals)
    assert (W.isnan16(h) == nan).all()
    o, v = W.order16(h)[~nan], vals[~nan]
    order = np.argsort(v, kind="stable")
    assert (np.diff(o[order]) >= 0).all() and (np.sign(np.diff(o[order])) == np.sign(np.diff(v[order]))).all()
    r = W.relu16(h)
    assert (r[~nan & (vals < 0)] == 0).all() and (r[nan | ~(vals < 0)] == h[nan | ~(vals < 0)]).all()


def test_relu16_closed_form_is_relu16():
    """units.cu's relu16, v - 0x8001 < 0x7F80 (unsigned) ? 0 : v, on every bf16 word."""
    h = np.arange(1 << 16, dtype=np.uint32)
    got = np.where(((h - np.uint32(0x8001)) & np.uint32(0xFFFFFFFF)) < 0x7F80, 0, h).astype(np.uint16)
    assert (got == W.relu16(h.astype(np.uint16))).all()


# ---- each twin against a scalar composition of the primitives' evaluators -------------------------------------------------

def add(a, b):
    return fp32.F32AddV2.evaluate(a, b)


def mul(a, b):
    return fp32.F32MulV2.evaluate(a, b)


def div(a, b):
    return fp32.F32DivV2.evaluate(a, b)


def wid(h):
    return prims.Bf16ToF32.evaluate(int(h))


def rnd(f):
    return prims.F2fpBf16.evaluate(f)


@pytest.mark.parametrize("N", [64, 256])
def test_rmsnorm_twin_is_its_ops(N):
    d = F.rms_pool(N)
    C, nc, G, ng = RN.forms(N)
    for r in range(0, len(d["x"]), 5):
        x = [int(v) for v in d["x"][r]]
        chunks = []
        for j in range(nc):
            acc = 0
            for v in x[j * C:(j + 1) * C]:
                acc = add(acc, mul(wid(v), wid(v)))
            chunks.append(acc)
        groups = []
        for g in range(ng):
            acc = 0
            for c in chunks[g * G:(g + 1) * G]:
                acc = add(acc, c)
            groups.append(acc)
        total = 0
        for g in groups:
            total = add(total, g)
        rs = div(W.ONE, int(W.fsqrt(add(div(total, W.f32word(N)), W.EPS))))
        w = [int(v) for v in d["w"][1]]
        out = [rnd(mul(wid(w[i]), wid(rnd(mul(wid(x[i]), rs))))) for i in range(N)]
        assert chunks == d["chunks"][r].tolist() and groups == d["groups"][r].tolist() and rs == int(d["rs"][r])
        assert out == d["out"][1, r].tolist()


def test_rmsnorm_forms():
    assert RN.forms(16384) == (128, 128, 64, 2)
    assert RN.forms(64) == (64, 1, 1, 1)
    with pytest.raises(ValueError):
        RN.forms(200)


@pytest.mark.parametrize("N, relu", [(256, True), (256, False), (64, True)])
def test_quant_twin_is_its_ops(N, relu):
    d = F.quant_pool(N, relu)
    for r in range(len(d["x"])):
        x = [int(v) for v in d["x"][r]]
        v = [0 if relu and not W.isnan16(e) and W.order16(e) < 0 else e for e in x]
        amax = max(e & 0x7FFF for e in v)
        a = wid(amax)
        scale = div(a, W.F127) if 1 <= a <= 0x7F800000 else W.ONE
        codes = []
        for e in v:
            q = struct.unpack("<f", struct.pack("<I", div(wid(e), scale)))[0]
            codes.append(0 if math.isnan(q) else max(-128, min(127, round(q) if math.isfinite(q) else (127 if q > 0 else -128))))
        assert amax == int(d["tree"][r, -1]) and scale == int(d["scale"][r])
        assert codes == d["codes"][r].view(np.int8).tolist(), (r, d["kind"][r])


def test_quant_levels():
    assert Q.levels(16384) == [128, 1] and Q.levels(65536) == [512, 4, 1] and Q.levels(64) == [1]
    with pytest.raises(ValueError):
        Q.levels(200)


def test_softmax_twin_is_its_ops():
    rows = np.stack([F.sm_row(k, np.random.default_rng([2, i])) for i, k in enumerate(F.SM_KINDS * 6)])
    n = np.random.default_rng(3).integers(1, 33, len(rows))
    t = SM.twin(rows, n)
    for r in range(len(rows)):
        x = [int(v) for v in rows[r, :n[r]]]
        m = 0x7FFF if any(W.isnan16(v) for v in x) else x[int(np.argmax([W.order16(v) for v in x]))]
        e = [mufu.MufuEx2Ftz.evaluate(mul(add(wid(v), wid(m) ^ 0x80000000), W.LOG2E)) for v in x]
        s = 0
        for v in e:
            s = add(s, v)
        rr = div(W.ONE, s)
        assert m == int(t["m"][r]) and rr == int(t["r"][r])
        assert [rnd(mul(v, rr)) for v in e] + [0] * (32 - n[r]) == t["p"][r].tolist()


def test_softmax_ftz_kind_flushes():
    """The ftz rows reach ex2's subnormal results, which MUFU.EX2 flushes to +0."""
    rows = np.stack([F.sm_row("ftz", np.random.default_rng([4, i])) for i in range(16)])
    m = SM.twin(rows, 32)["m"]
    t = W.fmul(W.fsub(W.widen(rows), W.widen(m)[:, None]), np.uint32(W.LOG2E)).view(np.float32)
    assert ((t < -126) & (t > -150)).any()


def scan(x: list[int]) -> int:
    """Argmax_v1's scan: the first greatest, NaN never taken, a NaN x_0 gives 0."""
    if W.isnan16(x[0]):
        return 0
    k = 0
    for i, v in enumerate(x):
        if not W.isnan16(v) and W.order16(v) > W.order16(x[k]):
            k = i
    return k


@pytest.mark.parametrize("V", [256, 200, 100])
def test_argmax_twin_is_the_scan(V):
    d = F.am_pool(V)
    assert d["token"].tolist() == [scan([int(v) for v in row]) for row in d["x"]]


def test_argmax_ties():
    x = np.full((1, 256), 0xFF80, np.uint16)
    assert AM.twin(x)["token"][0] == 0
    x[0, 5] = x[0, 200] = 0x3F80
    assert AM.twin(x)["token"][0] == 5
    x[0, 0] = 0x7FC0
    assert AM.twin(x)["token"][0] == 0
    y = np.zeros((1, 256), np.uint16)
    y[0, 0], y[0, 130] = 0x8000, 0x0000
    assert AM.twin(y)["token"][0] == 0


def test_add_twin_is_Bf16AddF2fp():
    d = F.add_pool(64)
    assert set(d["kind"]) == set(F.ADD_KINDS)
    for r in range(len(d["a"])):
        assert d["y"][r].tolist() == [fp32.Bf16AddF2fp.evaluate(int(p), int(q)) for p, q in zip(d["a"][r], d["b"][r])]
    sp = F.SPECIALS.astype(np.uint32)
    a, b = np.repeat(sp, len(sp)), np.tile(sp, len(sp))
    assert AD.twin(a, b).tolist() == [fp32.Bf16AddF2fp.evaluate(int(p), int(q)) for p, q in zip(a, b)]


class Rows:
    """The references as `ops.rows` states them: Argmax_v1's scan and Bf16AddRow_v1's Bf16AddF2fp per element."""
    argmax = staticmethod(lambda x: scan(x))
    add_row = staticmethod(lambda a, b: [fp32.Bf16AddF2fp.evaluate(p, q) for p, q in zip(a, b)])


def test_rows_check_counts_every_word():
    am, ad = {100: F.am_pool(100)}, F.add_pool(64)
    assert F.rows_check(Rows, am, ad) == {"argmax 100": {"token": [len(am[100]["x"]), 0]}, "add": {"y": [ad["y"].size, 0]}}
    ad["y"][3, 5] ^= 1
    assert F.rows_check(Rows, am, ad)["add"] == {"y": [ad["y"].size, 1]}


def test_load_rows_without_the_module_or_a_path():
    try:
        import verity.ml.circuits.ops.rows  # noqa: F401
    except ModuleNotFoundError:
        assert F.load_rows(None) == (None, None)
    else:
        assert F.load_rows(None)[1] == "installed"


# ---- the sweep ------------------------------------------------------------------------------------------------------------

def splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & (2 ** 64 - 1)
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2 ** 64 - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2 ** 64 - 1)
    return z ^ (z >> 31)


def test_sweep_digest_is_its_formula():
    assert splitmix64(0) == 0xE220A8397B1DCDAF == int(SW.splitmix64(np.array([0], np.uint64))[0])
    want = sum(splitmix64((i << 32) | int(Q.scale(np.array([i], np.uint16))[0])) for i in range(1 << 16)) % 2 ** 64
    assert SW.digest(7, 0) == want and SW.chunks(7) == 1 and SW.chunks(0) == 256 and SW.chunks(5) == 128
    assert SW.chunks(8) == SW.chunks(9) == SW.chunks(10) == 256


def test_each_cubin_sweeps_its_ops():
    """sm_90a's units_sweep runs ops 0 to 8, the ones the H200's check ran; sm_120a's adds MUFU.RSQ's two forms."""
    assert SW.ARCH_OPS["sm_90a"] == tuple(range(9)) and SW.ARCH_OPS["sm_120a"] == tuple(range(11)) == tuple(SW.OPS)
    assert {SW.OPS[op][0] for op in (9, 10)} == {"rsq", "rsqrt"}


def test_sweep_rsq_models_are_circuits():
    """Op 9 is the bare MUFU.RSQ, `mufu_rsq_bits` over the rsq table, and op 10 `RsqrtApprox` (rsqrtf, which scales a
    subnormal operand by 2^24 and its result by 2^12), word for word with their scalar forms, the specials included."""
    sub = np.array([0x00000001, 0x00000002, 0x00400000, 0x007FFFFF, 0x80000001, 0x807FFFFF], np.uint32)
    i = np.concatenate([f32_words(2000, 8), sub, sub | np.uint32(0x00800000)])
    t = mufu.table("rsq")
    rsq, rsqrt = SW.model(9, i), SW.model(10, i)
    assert rsq.tolist() == [mufu.mufu_rsq_bits(int(v), t) for v in i]
    assert rsqrt.tolist() == [mufu.RsqrtApprox.evaluate(int(v)) for v in i]
    small = (i & np.uint32(0x7FFFFFFF)) < np.uint32(0x00800000)
    assert (rsq[small] | np.uint32(0x80000000) == np.uint32(0xFF800000)).all()
    assert (rsq[~small] == rsqrt[~small]).all() and (rsq[small] != rsqrt[small]).any()


def test_first_difference_names_the_first_differing_word():
    got = SW.model(9, np.arange(5 * SW.CHUNK, 5 * SW.CHUNK + 4096, dtype=np.uint64).astype(np.uint32))
    assert SW.first_difference(9, 5, got) == {"chunk": 5, "words": 4096, "differing": 0, "first": None}
    got[[17, 300]] ^= np.uint32(1)
    d = SW.first_difference(9, 5, got)
    assert d["differing"] == 2 and d["first"]["i"] == f"{5 * SW.CHUNK + 17:08x}"
    assert int(d["first"]["gpu"], 16) ^ int(d["first"]["model"], 16) == 1


def test_sweep_models_are_the_units_ops():
    i = f32_words(4000, 6)
    assert (SW.model(3, i) == W.fdiv(np.uint32(W.ONE), i)).all()
    assert (SW.model(4, i) == W.fdiv(i, np.uint32(W.f32word(16384)))).all()
    x = np.arange(1 << 16, dtype=np.uint32)
    for amax in (0x0000, 0x0001, 0x3F80, 0x7F80, 0x7FC0):
        got = SW.model(5, (np.uint32(amax) << np.uint32(16)) | x)
        assert (got == W.fdiv(W.widen(x), Q.scale(np.array([amax], np.uint16))[0])).all()
    sp = F.SPECIALS.astype(np.uint32)
    i = np.concatenate([np.random.default_rng(7).integers(0, 1 << 32, 4000, dtype=np.uint64).astype(np.uint32),
                        (np.repeat(sp, len(sp)) << np.uint32(16)) | np.tile(sp, len(sp))])
    assert SW.model(8, i).tolist() == [fp32.Bf16AddF2fp.evaluate(int(v) >> 16, int(v) & 0xFFFF) for v in i]


# ---- the build check ------------------------------------------------------------------------------------------------------

def ptxas(names=build_check.KERNELS["sm_90a"], arch="sm_90a", spill=0) -> str:
    return "".join(f"ptxas info    : Compiling entry function '{n}' for '{arch}'\nptxas info    : Function properties for {n}\n"
                   f"    0 bytes stack frame, {spill} bytes spill stores, 0 bytes spill loads\nptxas info    : Used 32 registers\n"
                   for n in names)


BODIES = {
    "units_rmsnorm": ["FMUL R1, R2, R2", "FADD R3, R3, R1", "MUFU.RSQ R4, R3", "MUFU.RCP R5, R4", "FFMA R6, R5, R4, R3",
                      "FCHK P0, R1, R2", "F2FP.BF16.F32.PACK_AB R7, RZ, R6"],
    "units_quant": ["MUFU.RCP R5, R4", "FFMA R6, R5, R4, R3", "F2I.S8.NTZ R2, R6"],
    "units_quant4": ["UCGABAR_ARV", "MUFU.RCP R5, R4", "FFMA R6, R5, R4, R3", "F2I.S8.NTZ R2, R6"],
    "units_softmax": ["FADD R1, R2, -R3", "FMUL R1, R1, 1.4426950216293334961", "MUFU.EX2 R2, R1", "F2FP.BF16.F32.PACK_AB R3, RZ, R2"],
    "units_argmax": ["ISETP.GT.AND P0, PT, R1, R2, PT", "HFMA2.MMA R5, -RZ, RZ, 0, 0", "SEL R3, R1, R2, P0"],
    "units_relu": ["LOP3.LUT R1, R2, 0x7fff, RZ, 0xc0, !PT"],
    "units_add": ["FADD R1, R2, R3", "F2FP.BF16.F32.PACK_AB R3, RZ, R1"],
    "units_sweep": ["MUFU.EX2 R2, R1", "MUFU.RSQ R4, R3", "F2I.S8.NTZ R2, R6", "IMAD.WIDE.U32 R4, R2, R3, RZ"],
}


def sass(bodies=None) -> str:
    bodies = BODIES if bodies is None else bodies
    return "".join(f"        Function : {n}\n" + "".join(f"        /*{k * 16:04x}*/   {i} ;\n" for k, i in enumerate(b))
                   for n, b in bodies.items())


def test_build_check_passes_a_clean_listing():
    funcs, problems = build_check.check(ptxas(), sass())
    assert problems == []
    assert funcs["units_softmax"]["float"]["MUFU.EX2"] == 1 and funcs["units_argmax"]["float"] == {}


def test_build_check_holds_each_arch_to_its_kernels():
    """The sm_120a cubin passes with units_sweep_words (EX2 and RSQ, as units_sweep) and is refused without it; sm_90a's
    is refused with it."""
    k120 = build_check.KERNELS["sm_120a"]
    with120 = dict(BODIES, units_sweep_words=list(BODIES["units_sweep"]))
    assert build_check.check(ptxas(k120, "sm_120a"), sass(with120), "sm_120a")[1] == []
    _, problems = build_check.check(ptxas(arch="sm_120a"), sass(), "sm_120a")
    assert problems == ["units_sweep_words: missing from the ptxas report or the SASS"]
    _, problems = build_check.check(ptxas(k120), sass(with120))
    assert problems == ["kernels the check doesn't know: ['units_sweep_words']"]


def test_build_check_takes_sm_120a_constants_and_integer_division():
    """sm_120a's ptxas loads constants with HFMA2 (no .MMA) and computes units_relu's unrolled trip count by the
    compiler's integer division; neither touches an integer kernel's values."""
    bodies = dict(BODIES, units_argmax=["HFMA2 R16, -RZ, RZ, 0, 0", "HFMA2 R5, -RZ, RZ, 1.875, 1.1920928955078125e-07",
                                        "SEL R3, R1, R2, P0"],
                  units_relu=["I2F.U32.RP R0, R2", "MUFU.RCP R0, R0", "F2I.FTZ.U32.TRUNC.NTZ R9, R8", "I2F.U64.RP R4, R2",
                              "MUFU.RCP R4, R4", "F2I.U64.TRUNC R6, R4", "HFMA2 R8, -RZ, RZ, 0, 0", *BODIES["units_relu"]])
    funcs, problems = build_check.check(ptxas(), sass(bodies))
    assert problems == []
    assert funcs["units_argmax"]["float"] == {} and funcs["units_relu"]["float"]["MUFU.RCP"] == 2


@pytest.mark.parametrize("edit, why", [
    (lambda b: b["units_rmsnorm"].append("MUFU.EX2 R9, R8"), "MUFU.EX2 outside"),
    (lambda b: b["units_quant"].append("MUFU.SQRT R9, R8"), "MUFU forms other than"),
    (lambda b: b["units_argmax"].append("FADD R9, R8, R7"), "integer kernel"),
    (lambda b: b["units_relu"].append("I2F.RP R9, R8"), "integer kernel"),
    (lambda b: b["units_relu"].append("MUFU.RSQ R9, R8"), "integer kernel"),
    (lambda b: b["units_argmax"].append("HFMA2 R9, -RZ, RZ, R8"), "integer kernel"),
    (lambda b: b["units_argmax"].append("HFMA2.MMA R9, -RZ, RZ, R8.H0_H0, 0"), "integer kernel"),
    (lambda b: b["units_softmax"].remove("MUFU.EX2 R2, R1"), "no MUFU.EX2"),
    (lambda b: b["units_quant"].append("STL [R1], R2"), "local memory"),
    (lambda b: b["units_rmsnorm"].append("HMMA.16816.F32.BF16 R4, R8, R12, R4"), "tensor-core"),
    (lambda b: b.pop("units_relu"), "units_relu: missing"),
])
def test_build_check_refuses(edit, why):
    bodies = {k: list(v) for k, v in BODIES.items()}
    edit(bodies)
    _, problems = build_check.check(ptxas(), sass(bodies))
    assert any(why in p for p in problems), problems


@pytest.mark.parametrize("text, why", [(ptxas(spill=8), "local memory"), (ptxas(arch="sm_120a"), "compiled for sm_120a"),
                                       (ptxas() + "ptxas warning : Registers are spilled\n", "Registers are spilled"),
                                       (ptxas(build_check.KERNELS["sm_90a"] + ("units_extra",)), "doesn't know")])
def test_build_check_refuses_the_report(text, why):
    bodies = dict(BODIES, units_extra=["NOP"]) if "units_extra" in text else BODIES
    _, problems = build_check.check(text, sass(bodies))
    assert any(why in p for p in problems), problems
