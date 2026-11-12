"""The shared-expert MoE scope (`gen_shared_moe_patterns.MoEFusedSharedExpertsFp8`) holds each call of the scope to the view the call
before it wrote: on the synthetic `vllm.moe_forward_shared` events of `tests/engine/test_gen_shared_moe_fp8.py`, with one argument of
one call replaced, the scope is refused, and the refusal names the view the call read beside the one it should have read.

* the shared expert's SiLU reads the gate_up GEMM's output;
* the shared expert's down GEMM reads that SiLU's output;
* the routed quant reads the routed SiLU's output, between the two launches.
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import dataclasses

import pytest

from tests.engine import test_gen_shared_moe_fp8 as S
from verity_vllm.observe.events import DispatchEnter, StepStart
from verity_vllm.observe.fold.resolver import Group, Unsupported, resolve_step
from verity_vllm.observe.fold.tree import step_events

M = 4
P = M * S.TOPK
SILU = "_C.silu_and_mul.default"
DYN_MM = "vllm.dynamic_flashinfer_deepgemm_blockscale_gemm.default"
QUANT = "_C.per_token_group_fp8_quant.default"


def _resolve(op: str, nth: int, arg: int, value):
    """The step's resolutions when the `nth` call of `op` in the scope takes `value` as its argument `arg`."""
    start = StepStart(0, [("r0", p) for p in range(M)], list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0, 1, 2, 3]}, 16,
                      list(range(M)), [("r0", M - 1)], [M - 1], 1, M, M, None, None, "prefill")
    events = S.scope_events(M)
    at = [i for i, e in enumerate(events) if isinstance(e, DispatchEnter) and e.op == op][nth]
    args = list(events[at].args)
    args[arg] = value
    events[at] = dataclasses.replace(events[at], args=args)
    return resolve_step(step_events([start, *events]), S._ctx(start))


@pytest.mark.parametrize("op,nth,arg,other,reason,read,should", [
    (SILU, 0, 1, S.view(399, (M, 2 * S.SI)), "the shared expert's silu_and_mul does not read the gate_up GEMM's output",
     "act_input", ("gate_up", S.view(323, (M, 2 * S.SI)))),
    (DYN_MM, 1, 0, S.view(398, (M, S.SI)), "the shared expert's down GEMM does not read its input",
     "gemm_input", ("expected", S.view(324, (M, S.SI)))),
    (QUANT, 1, 0, S.view(397, (P, S.C["I"])), "the routed per_token_group_fp8_quant does not read the SiLU output between the two launches",
     "quant_input", ("silu_output", S.view(301, (P, S.C["I"])))),
], ids=["shared-silu", "shared-down", "routed-quant"])
def test_a_call_of_the_shared_scope_that_reads_another_view_is_refused_with_both_views(op, nth, arg, other, reason, read, should):
    scope, _combine = _resolve(op, nth, arg, other)
    assert isinstance(scope, Unsupported) and scope.reason == reason
    assert scope.detail[read] == repr(other) and scope.detail[should[0]] == repr(should[1])


def test_with_the_view_the_call_before_wrote_the_scope_is_whole():
    scope, _combine = _resolve(SILU, 0, 1, S.view(323, (M, 2 * S.SI)))
    assert isinstance(scope, Group)
