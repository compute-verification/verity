"""kgate.py's kernel record: which body each call ran is observed on the modules the dispatch looks it up on, a check that
never goes through the dispatch says so, and a body the dispatch binds early reads as nothing ran and fails the row.  CPU
only: torch and the int8 modules are stand-ins, the dispatch one that takes int8_wgmma on even N the way
`int8_fused.product` takes it where `pick_wgmma` has a row."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

KGATE = Path(__file__).resolve().parents[1] / "kgate.py"
TRITON, WGMMA = "64x64x256:w4:s3:k1", "ppm2_128x128:g8"


def _int8_modules(bind_early: bool = False) -> tuple[types.ModuleType, types.ModuleType]:
    fused, wgmma = types.ModuleType("int8_fused"), types.ModuleType("int8_wgmma")
    fused.quant_rows = lambda x, xq, s, num_warps=None, rows=None: "xq"
    fused.scaled_mm = lambda xq, s_a, w, s_w, y, cfg=None, work=None: "triton"
    fused.scaled_mm_zy = lambda xq, s_a, w, s_w, y, zy, rows, cfg=None, work=None: "triton"
    wgmma.scaled_mm = lambda xq, s_a, w, s_w, y, cfg, flags=0: "wgmma"
    wgmma.scaled_mm_zy = lambda xq, s_a, w, s_w, y, zy, rows, cfg, flags=0: "wgmma"
    fused.int8_wgmma = wgmma
    early = (fused.scaled_mm, wgmma.scaled_mm)

    def product(n: int) -> str:
        tri, wg = early if bind_early else (fused.scaled_mm, fused.int8_wgmma.scaled_mm)
        return wg(0, 0, n, 0, 0, WGMMA) if n % 2 == 0 else tri(0, 0, n, 0, 0, TRITON, None)

    def product_zy(n: int) -> str:
        if n % 2 == 0:
            return fused.int8_wgmma.scaled_mm_zy(0, 0, n, 0, 0, 0, 16, WGMMA)
        return fused.scaled_mm_zy(0, 0, n, 0, 0, 0, 16, cfg=TRITON)

    fused.product, fused.product_zy = product, product_zy
    fused.row_quant = lambda x: fused.quant_rows(x, 0, 0)
    return fused, wgmma


def _load(monkeypatch: pytest.MonkeyPatch, fused: types.ModuleType) -> types.ModuleType:
    engine = types.ModuleType("verity_vllm.engine")
    engine.int8_fused = fused
    package = types.ModuleType("verity_vllm")
    package.engine = engine
    for name, mod in (("torch", types.ModuleType("torch")), ("verity_vllm", package), ("verity_vllm.engine", engine)):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("kgate_under_test", KGATE)
    kgate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kgate)
    return kgate


def _row(kgate: types.ModuleType, fused: types.ModuleType, n: int) -> dict:
    """`one`'s checks in its order, each marked off the way `one` marks it."""
    kernels, i = {}, len(kgate.RAN)
    fused.quant_rows(0, 0, 0)
    fused.scaled_mm(0, 0, n, 0, 0, TRITON, None)
    kernels["fused"], i = kgate.ran_since(i, "fused"), len(kgate.RAN)
    fused.row_quant(0)
    fused.product(n)
    kernels["op_bodies"], i = kgate.ran_since(i, "op_bodies"), len(kgate.RAN)
    kernels["control"], i = kgate.ran_since(i, "control"), len(kgate.RAN)
    fused.product_zy(n)
    kernels["zy"], i = kgate.ran_since(i, "zy"), len(kgate.RAN)
    kernels["stock"] = kgate.ran_since(i, "stock")
    return {"kernels": kernels}


def test_the_record_names_the_body_each_call_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    fused, wgmma = _int8_modules()
    kgate = _load(monkeypatch, fused)
    bodies = {(m, n): getattr(fused if m == "int8_fused" else wgmma, n) for m, n in kgate.BODIES}
    with kgate.counting():
        even = _row(kgate, fused, 4096)["kernels"]
    with kgate.counting():
        odd = _row(kgate, fused, 4095)["kernels"]
    assert even == {
        "fused": {"through_dispatch": False, "ran": ["int8_fused.quant_rows", f"int8_fused.scaled_mm {TRITON}"]},
        "op_bodies": {"through_dispatch": True, "ran": ["int8_fused.quant_rows", f"int8_wgmma.scaled_mm {WGMMA}"]},
        "control": {"through_dispatch": False, "ran": []},
        "zy": {"through_dispatch": True, "ran": [f"int8_wgmma.scaled_mm_zy {WGMMA}"]},
        "stock": {"through_dispatch": False, "ran": []},
    }
    assert odd["op_bodies"]["ran"] == ["int8_fused.quant_rows", f"int8_fused.scaled_mm {TRITON}"]
    assert odd["zy"]["ran"] == [f"int8_fused.scaled_mm_zy {TRITON}"]
    assert kgate.kernels_ok(even) and kgate.kernels_ok(odd)
    assert kgate.kernels_summary([{"kernels": even}, {"kernels": odd}])["op_bodies"] == {
        "through_dispatch": True, "gemm": {"int8_fused.scaled_mm": 1, "int8_wgmma.scaled_mm": 1}}
    assert kgate.RAN is None
    assert all(getattr(fused if m == "int8_fused" else wgmma, n) is f for (m, n), f in bodies.items()), "put back after"


def test_only_the_dispatched_checks_say_they_go_through_the_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    kgate = _load(monkeypatch, _int8_modules()[0])
    checks = ("fused", "fused_mm", "op_bodies", "control", "graph", "zy", "zy_ptx", "stock")
    assert {c for c in checks if kgate.ran_since(0, c)["through_dispatch"]} == {"op_bodies", "zy"}


def test_a_body_bound_before_the_count_reads_as_nothing_ran(monkeypatch: pytest.MonkeyPatch) -> None:
    fused, _ = _int8_modules(bind_early=True)
    kgate = _load(monkeypatch, fused)
    with kgate.counting():
        row = _row(kgate, fused, 4096)["kernels"]
    assert row["op_bodies"]["ran"] == ["int8_fused.quant_rows"]
    assert not kgate.kernels_ok(row)


@pytest.mark.parametrize(("check", "ran"), [
    ("fused", ["int8_fused.quant_rows", f"int8_wgmma.scaled_mm {WGMMA}"]),
    ("fused_mm", []),
    ("graph", [f"int8_fused.scaled_mm {TRITON}", f"int8_fused.scaled_mm {TRITON}"]),
    ("control", ["int8_fused.quant_rows"]),
    ("stock", [f"int8_fused.scaled_mm {TRITON}"]),
])
def test_kernels_ok_refuses(monkeypatch: pytest.MonkeyPatch, check: str, ran: list[str]) -> None:
    kgate = _load(monkeypatch, _int8_modules()[0])
    assert not kgate.kernels_ok({check: {"through_dispatch": check in kgate.DISPATCH, "ran": ran}})
