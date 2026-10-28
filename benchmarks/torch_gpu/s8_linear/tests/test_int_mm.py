"""int_mm.py on the CPU: `linear` against the twin (`fixture.epilogue` over `fixture.sums`, which tests/test_twin.py holds
to the Definition's evaluator) on the gate's structured codes and special scales, what it refuses, and the fused
epilogue's PTX audit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.ops import gemm as G

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixture as FX  # noqa: E402
import int_mm as IM  # noqa: E402


def _codes(rng: np.random.Generator, rows: int, k: int, structured: dict[int, str]) -> np.ndarray:
    x = rng.integers(-128, 128, (rows, k), dtype=np.int8)
    for i, kind in structured.items():
        if i < rows:
            x[i] = FX.structured(kind, k)
    return x


@pytest.mark.parametrize("m", [1, 7, 30])
def test_linear_on_the_cpu_is_the_twin(m: int) -> None:
    """`linear` on the CPU (`torch._int_mm`, `epilogue="eager"`), and again into ``out``, is the twin at every word but a
    NaN, where torch's CPU cast writes another NaN word for the Definition's 0x7FFF (int_mm's docstring)."""
    rng = np.random.default_rng(m)
    k, n = 128, 8 * -(-(FX.FIRST_SPECIAL + len(FX.SPECIAL_SW) + 8) // 8)
    x, w = _codes(rng, m, k, FX.STRUCT_ROWS), _codes(rng, n, k, FX.STRUCT_COLS)
    sa, sw = FX.scales(m, n, seed=m)
    want = FX.epilogue(FX.sums(x, w), sa, sw)
    nan = ((want & 0x7F80) == 0x7F80) & ((want & 0x7F) != 0)
    args = [torch.from_numpy(a) for a in (x, sa, w, sw)]
    out = torch.empty((m, n), dtype=torch.bfloat16)
    for got in (IM.linear(*args, epilogue="eager"), IM.linear(*args, epilogue="eager", out=out)):
        got = got.view(torch.int16).numpy().view(np.uint16)
        assert np.array_equal(got[~nan], want[~nan])
        assert (((got[nan] & 0x7F80) == 0x7F80) & ((got[nan] & 0x7F) != 0)).all() and nan.any()


def test_linear_refuses_what_the_definition_does_not_bind() -> None:
    assert IM.K_LIMIT == G.S8_K_LIMIT

    def ops(m, k, n, dtype=torch.int8):
        return torch.zeros(m, k, dtype=dtype), torch.ones(m), torch.zeros(n, k, dtype=torch.int8), torch.ones(n)

    for args, why in ((ops(2, 48, 8), "K = 48"), (ops(1, 1 << 17, 8), "K = 131072"), (ops(2, 64, 12), "N = 12"),
                      (ops(2, 64, 8, torch.uint8), "dtypes"), (ops(2, 64, 8), "epilogue='eager'")):
        with pytest.raises(ValueError, match=why):
            IM.linear(*args)
    with pytest.raises(ValueError, match="pass one of"):
        IM.linear(*ops(2, 64, 8), epilogue="fast")
    for out in (torch.empty(2, 8), torch.empty(2, 16, dtype=torch.bfloat16), torch.empty(8, 2, dtype=torch.bfloat16).t()):
        with pytest.raises(ValueError, match="out "):
            IM.linear(*ops(2, 64, 8), epilogue="eager", out=out)


def test_the_ptx_audit_takes_only_the_definitions_instructions() -> None:
    """`ptx_audit` passes PTX made of `FUSED_PTX` copies, and fails one with an fma, a flushed multiply, a missing multiply
    or other f32 arithmetic: what a compiler pass that touched the epilogue would leave."""
    def snippet(i: int) -> str:
        return IM.FUSED_PTX.replace("$0", f"%rs{i}").replace("$1", f"%r{i}").replace("$2", f"%f{2 * i}").replace(
            "$3", f"%f{2 * i + 1}")

    good = "ld.global.v4.b32 {%r1, %r2, %r3, %r4}, [%rd1];\nmov.f32 %f9, 0f00000000;\n" + "\n".join(map(snippet, range(4)))
    audit = IM.ptx_audit(good)
    assert audit["ok"] and audit["cvt.rn.f32.s32"] == audit["blocks"] == 4 and audit["mul.rn.f32"] == 8
    for bad in (good.replace("mul.rn.f32 t, t, %f1;", "fma.rn.f32 t, t, %f1, %f9;"),
                good.replace("mul.rn.f32 t, t, %f1;", "mul.rn.ftz.f32 t, t, %f1;"),
                good.replace("mul.rn.f32 t, t, %f1;", ""),
                good + "\nadd.f32 %f9, %f9, %f9;"):
        assert not IM.ptx_audit(bad)["ok"]
