"""`protocol_options.credit_gate` and its place in H1's credit (`pouw_h1.traced`, `H1Run.credit`): a served call is credited
only if the y the linear hands back is the credited kernel's own output, one GEMM invocation writing z and y, and no other
GEMM ran for it.  Today's served path (vllm's int8 arm: `scaled_mm`'s y, z a second ``_int_mm`` beside it) is refused with
its named reason; a fused kernel, one dispatcher op returning z and y = dequant(z), is credited."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw.coins import Challenger  # noqa: E402
from verity_vllm import protocol_options as PO  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_ops  # noqa: E402
from verity_vllm.protocol_options import credit_gate as CG  # noqa: E402
from verity_vllm.protocol_options import pouw  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402

K = 64
ENTRY = f"pouw:{PH.SCHEME}"
TODAY = "credit: served-output-not-credited-kernel: q y from verity_int8::scaled_mm, z from aten::_int_mm"


@torch.library.custom_op("verity_credit_test::fused_linear", mutates_args=())
def fused_linear(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused stand-in: one op that computes the exact s32 Y and returns it with y = dequant(Y)."""
    z = E.int_mm(xq, w)
    return z, E.dequantize(z, s_a, s_w)


@torch.library.custom_op("verity_credit_test::two_gemms", mutates_args=())
def two_gemms(xq: torch.Tensor, s_a: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One op hiding today's path: y from one ``_int_mm``, z from another."""
    return E.int_mm(xq, w), E.dequantize(E.int_mm(xq, w), s_a, s_w)


class TodayArm:
    """`engine.int8_vllm.Int8PlainLinearMethod` as vllm serves it today (#2615): `int8_ops.linear` (``row_quant`` then
    ``scaled_mm``), its values handed to ``observe``, y returned."""

    int8_arm = E.ARM

    def __init__(self):
        self.calls = 0

    def apply(self, layer, x, bias=None):
        self.calls += 1
        lead, k = x.shape[:-1], x.shape[-1]
        xq, s_a, y = int8_ops.linear(x.reshape(-1, k), layer.weight, layer.weight_scale)
        self.observe(layer, xq, s_a, y)
        return y.reshape(*lead, layer.weight.shape[0])

    def observe(self, layer, xq, s_a, y):
        pass


class FusedArm(TodayArm):
    """The fused arm: ``row_quant``, then one kernel (`fused_linear`, or `kernel`) writing z and y; z handed to ``observe``;
    ``after`` applied to y before it is served, ``side`` run beside."""

    def __init__(self, kernel=fused_linear, after=None, side=None):
        super().__init__()
        self.kernel, self.after, self.side = kernel, after, side

    def apply(self, layer, x, bias=None):
        self.calls += 1
        xq, s_a = int8_ops.row_quant(x)
        z, y = self.kernel(xq, s_a, layer.weight, layer.weight_scale)
        if self.side is not None:
            self.side(xq, layer.weight)
        if self.after is not None:
            y = self.after(y)
        self.observe(layer, xq, s_a, y, z)
        return y

    def observe(self, layer, xq, s_a, y, z=None):
        pass


class Linear:
    def __init__(self, n, seed, arm):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randint(-127, 128, (n, K), generator=g, dtype=torch.int8)
        self.weight_scale = (torch.rand(n, 1, generator=g) * 0.01 + 1e-4).float()
        self.quant_method = arm


class Model:
    def __init__(self, *layers):
        self.layers = list(layers)

    def modules(self):
        return iter(self.layers)

    def named_modules(self):
        return iter([("q", self.layers[0])] + [(f"l{i}", m) for i, m in enumerate(self.layers[1:], 1)])


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


def _serve(arm, retain=1, m=3):
    """One small call (m rows of K, N = 32) through `arm` with H1 installed; the run closed and its y and run returned."""
    layer = Linear(32, 1, arm)
    ctx = PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": {"coins": Challenger(0), "rows": 32, "steps": 1,
                                                                         "retain_steps": retain}})
    comp = PO.install(Model(layer), ctx)
    try:
        x = (torch.randn(m, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16()
        y = layer.quant_method.apply(layer, x)
        run = pouw.h1_run(ENTRY)
        run.close()
        assert torch.equal(y.view(torch.int16), E.plain_linear(x, layer.weight, layer.weight_scale).view(torch.int16))
        return y, run
    finally:
        comp.close()


def test_todays_served_path_is_refused_y_from_scaled_mm_z_from_a_second_int_mm():
    """vllm's int8 arm serves `scaled_mm`'s y and H1's z is a second ``_int_mm`` beside it (J11): the call is refused with
    the named reason, both GEMMs in its trace, and the window credits nothing; the served y is untouched."""
    _y, run = _serve(TodayArm())
    (v,) = run.verdicts.values()
    assert v["reason"] == TODAY and not v["credit"]
    assert v["gemms"] == ["aten::_int_mm in verity_int8::scaled_mm", "aten::_int_mm"]
    c = run.credit()
    assert c["credited"] == {"rows": 0, "macs": 0}
    assert c["refused"] == {"calls": 1, "rows": 3, "macs": 3 * 32 * K, "reasons": {TODAY: 1}}


def test_todays_path_without_a_retained_z_is_refused_as_no_z():
    _y, run = _serve(TodayArm(), retain=0)
    (v,) = run.verdicts.values()
    assert v["reason"] == "credit: served-output-not-credited-kernel: q y from verity_int8::scaled_mm, z from <none>"


def test_serve_h1s_wrapper_is_refused_too():
    """SERVE-H1's run tree (cursor/serve-h1-run-ea24, `pouw_h1.served`): the arm's apply serves y, then the wrapper
    quantizes the same activations again and takes z from `int_mm`; the trace refuses it the same way."""
    layer = Linear(32, 1, TodayArm())
    x = (torch.randn(3, K, generator=torch.Generator().manual_seed(9)) * 3).bfloat16()
    with CG.Trace("q") as t:
        y = layer.quant_method.apply(layer, x)
        xq, _sa = E.quantize_rows(x.reshape(-1, x.shape[-1]))
        z = E.int_mm(xq, layer.weight)
    v = CG.verdict(t, y, z)
    assert v["reason"] == TODAY and len(v["gemms"]) == 2


def test_a_fused_kernel_writing_z_and_y_is_credited():
    """The fused stand-in, one op returning z and y = dequant(z): the served y is that op's output, its one GEMM the call's
    only one, so the call is credited and the window credits its served rows."""
    y, run = _serve(FusedArm())
    (v,) = run.verdicts.values()
    assert v["credit"] and "reason" not in v
    assert (v["kernel"], v["y_from"], v["z_from"]) == ("verity_credit_test::fused_linear",) * 3
    assert v["gemms"] == ["aten::_int_mm in verity_credit_test::fused_linear"]
    c = run.credit()
    assert c["credited"] == {"rows": 3, "macs": 3 * 32 * K} and c["refused"]["calls"] == 0
    _index, xq, _sa, z = run.retained[-1]
    assert np.array_equal(z, xq.astype(np.int64) @ Linear(32, 1, None).weight.numpy().astype(np.int64).T)


@pytest.mark.parametrize("side,want", [
    (lambda xq, w: E.int_mm(xq, w), "aten::_int_mm"),
    (lambda xq, w: xq.float() @ w.float().t(), "aten::mm"),
], ids=["second-int-mm", "float-mm"])
def test_a_second_gemm_beside_the_fused_kernel_is_refused(side, want):
    _y, run = _serve(FusedArm(side=side))
    (v,) = run.verdicts.values()
    assert not v["credit"] and v["reason"].startswith("credit: gemm-count: q 2 GEMMs (aten::_int_mm in "
                                                       "verity_credit_test::fused_linear, ")
    assert v["gemms"][1] == want


def test_one_op_hiding_two_gemms_is_refused():
    _y, run = _serve(FusedArm(kernel=two_gemms))
    (v,) = run.verdicts.values()
    assert v["y_from"] == v["z_from"] == "verity_credit_test::two_gemms" and not v["credit"]
    assert v["reason"].startswith("credit: gemm-count: q 2 GEMMs")


@pytest.mark.parametrize("after,op", [(lambda y: y.clone(), "aten::clone"), (lambda y: y.add_(0), "aten::add_")],
                         ids=["copied", "written-after"])
def test_a_served_y_that_is_not_the_kernels_output_is_refused(after, op):
    """A y copied or written after the kernel is not the kernel's output: refused, y from the op that wrote it last."""
    _y, run = _serve(FusedArm(after=after))
    (v,) = run.verdicts.values()
    assert v["reason"] == f"credit: served-output-not-credited-kernel: q y from {op}, z from verity_credit_test::fused_linear"


def test_a_z_written_outside_the_trace_is_refused_as_untraced():
    w = torch.randint(-127, 128, (32, K), dtype=torch.int8)
    xq = torch.randint(-127, 128, (3, K), dtype=torch.int8)
    z = E.int_mm(xq, w)
    with CG.Trace("q") as t:
        y = fused_linear(xq, torch.ones(3), w, torch.ones(32, 1))[1]
    assert CG.verdict(t, y, z)["reason"] == ("credit: served-output-not-credited-kernel: q y from "
                                             "verity_credit_test::fused_linear, z from <untraced>")


def test_a_call_recorded_outside_a_traced_apply_is_not_credited():
    run = PH.H1Run("t", hashlib.sha512(b"salt").digest(), {0: "q"}, [(32, K)], 32, 1)
    run.record(0, torch.zeros(2, K, dtype=torch.int8), torch.ones(2), torch.zeros(32, K, dtype=torch.int8), torch.ones(32, 1))
    run.close()
    c = run.credit()
    assert c["credited"] == {"rows": 0, "macs": 0}
    assert c["refused"]["reasons"] == {"credit: untraced: q call 0 was recorded outside a traced apply": 1}
