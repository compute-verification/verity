"""The int8 attention's Triton kernels (`int8_attention_triton.kv_rows`, `lut_rows`) at the PTX and SASS level (`int8_attention_sass`):
each variant Triton launches, compiled for sm_120 and sm_90, passes; lut_rows compiled with contraction allowed, and its PTX
with the rounding modes stripped through ptxas, are refused at both levels; a kv_rows dividing with Triton's `/` is refused;
and contraction can't change kv_rows, whose Definition has no product.  Needs Triton with its bundled ptxas and nvdisasm, not
a GPU; skipped without them.  The rules' own cases (an instruction each rule refuses) need neither."""
from __future__ import annotations

import functools

import pytest

from tests.engine import int8_attention_sass as S


@functools.cache
def _triton():
    """`int8_attention_triton`, or why the compiled checks skip."""
    try:
        import triton  # noqa: F401
        from verity_vllm.engine import int8_attention_triton as T
        for cap in S.ARCHES:
            S.tool("ptxas", cap)
        S.tool("nvdisasm")
    except (ImportError, AttributeError, S.ToolMissing) as e:
        return f"Triton with its bundled ptxas and nvdisasm required ({type(e).__name__}: {e})"
    return T


def _kernels():
    T = _triton()
    if isinstance(T, str):
        pytest.skip(T)
    return T


@functools.cache
def _compiled(variant: str, cap: int, fusion: bool = False):
    return S.compile_variant(_kernels(), variant, cap, enable_fp_fusion=fusion)


@pytest.mark.slow
@pytest.mark.parametrize("cap", S.ARCHES)
@pytest.mark.parametrize("variant", ["kv_rows/paged", "kv_rows/packed", "lut_rows/d128", "lut_rows/d64"])
def test_each_kernel_as_launched_passes(variant, cap):
    r = S.gate_compiled(_compiled(variant, cap))
    assert r["ok"], r["refused"]
    assert r["fmad"]["same"] and not r["ptx"]["fma"] and not r["ptx"]["unrounded"]
    if variant.startswith("lut_rows"):
        c = r["sass"]["census"]
        assert c["FMUL"] == 3 * c["bf16"] and c["FADD"] == c["bf16"] > 0


@pytest.mark.slow
@pytest.mark.parametrize("cap", S.ARCHES)
def test_lut_rows_compiled_with_contraction_allowed_is_refused(cap):
    r = S.gate_compiled(_compiled("lut_rows/d128", cap, True))
    assert not r["ok"]
    assert r["ptx"]["fma"].get("fma.rn.f32", 0) > 0
    assert any(x.startswith("SASS:") and "a product contracted into its add" in x for x in r["refused"]), r["refused"]


@pytest.mark.slow
@pytest.mark.parametrize("cap", S.ARCHES)
def test_lut_rows_ptx_without_rounding_modes_is_contracted_by_ptxas_and_refused(cap):
    ptx = S.unrounded(_compiled("lut_rows/d128", cap).asm["ptx"])
    r = S.gate("lut_rows", ptx, S.ptxas(ptx, cap, True), cap)
    assert not r["ok"] and not r["fmad"]["same"]
    assert r["fmad"]["ffma_contraction_allowed"] > r["fmad"]["ffma_fmad_false"]
    assert any(x.startswith("SASS:") and "a product contracted into its add" in x for x in r["refused"]), r["refused"]


@pytest.mark.slow
def test_kv_rows_dividing_with_tritons_slash_is_refused(tmp_path):
    _kernels()
    r = S.gate_compiled(S.compile_variant(S.div_full_mutant(tmp_path), "kv_rows/paged", 120))
    assert not r["ok"]
    assert r["ptx"]["ruled_out"].get("div.full.f32", 0) > 0
    assert any(x.startswith("SASS: kv_rows computes FMUL") for x in r["refused"]), r["refused"]


@pytest.mark.slow
@pytest.mark.parametrize("cap", S.ARCHES)
def test_contraction_cannot_change_kv_rows(cap):
    on, off = _compiled("kv_rows/paged", cap, True), _compiled("kv_rows/paged", cap)
    assert S.instructions(S.listing(on.asm["cubin"])) == S.instructions(S.listing(off.asm["cubin"]))
    assert S.gate_compiled(on)["ok"]


def test_the_ptx_rules_refuse_each_class():
    ptx = "\n".join(["div.rn.f32 %f1, %f2, %f3;", "cvt.rni.s32.f32 %r1, %f1;", "@%p1 fma.rn.f32 %f4, %f1, %f2, %f3;",
                     "mul.f32 %f5, %f1, %f2;", "ex2.approx.ftz.f32 %f6, %f5;", "mad.lo.s64 %rd1, %rd2, %rd3, %rd4;",
                     "mul.rn.f32 %f7, %f1, %f2;"])
    r = S.ptx_rules("kv_rows", ptx)
    assert r["fma"] == {"fma.rn.f32": 1} and r["unrounded"] == {"mul.f32": 1} and r["ruled_out"] == {"ex2.approx.ftz.f32": 1}
    assert not r["missing"] and len(r["refused"]) == 3
    assert S.ptx_rules("lut_rows", ptx)["missing"] == ["cvt.rni.s64.f64", "mul.rn.f64", "add.rn.f32", ".s8.s8"]


def _listing(*insns: str) -> str:
    body = "\n".join(f"        /*{16 * i:04x}*/                   {t} ;" for i, t in enumerate(insns))
    return f"\t.section\t.text.kv_rows,\"ax\",@progbits\n.text.kv_rows:\n{body}\n"


def test_the_sass_rules_refuse_each_class():
    ok = S.sass_rules("kv_rows", _listing("MUFU.RCP R1, R2", "FCHK P0, R3, R2", "FFMA R4, R1, R3, RZ", "FADD R5, |R4|, -RZ"))
    assert not ok["refused"], ok
    bad = S.sass_rules("kv_rows", _listing("MUFU.EX2 R1, R2", "FMUL.FTZ R3, R1, R2", "MUFU.RCP R4, R2", "DFMA R6, R4, R4, R4",
                                           "HMMA.16816.F32.BF16 R8, R10, R12, R8"))
    text = " ".join(bad["refused"])
    for want in ("'MUFU.EX2': 1", "'FMUL.FTZ': 1", "1 MUFU.RCP for 0 div.rn checks", "DFMA 1", "a float MMA", "FMUL 1"):
        assert want in text, (want, bad["refused"])
    lut = S.sass_rules("lut_rows", _listing("FMUL R1, R2, R3", "FFMA R4, R5, R6, R1", "F2F.BF16.F32 R7, R4", "DMUL R8, R10, R12",
                                            "IMMA.16832.S8.S8.SAT R14, R16, R18, R14"))
    assert lut["refused"] == ["SASS: 1 FMUL and 0 FADD for 1 bf16 outputs, where Int8LutAttentionRow_v1 rounds three products "
                              "and one sum per output (3 and 1): a product contracted into its add"]
