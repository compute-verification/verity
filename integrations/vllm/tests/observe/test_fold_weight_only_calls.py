"""`TargetProfile.weight_only_calls` in the Match fold (`patterns.target.WeightOnlyCalls`): GemmaRMSNorm's `weight.float() + 1.0` in
every forward folds to one `AddScalarBf16_v1{N,C=1}` instance under the default `once`, every later forward reading its output, and to
one instance per forward under a declared `per-forward` -- the Build's `vllm_bindings.norm_chain` rules (tests/program/test_weight_only_once.py).
The toy two-step log of `test_fold_synthetic` with the Gemma-2 patterns, the weight add and one reader of it in each step."""
from __future__ import annotations

import dataclasses

from verity.ml.circuits.refs import runs
from verity_vllm.engine.profiles.gen_dense_gemma2_patterns import gemma2_patterns
from verity_vllm.observe.events import DispatchEnter, DispatchExit, RunHeader, StorageAlloc
from verity_vllm.observe.fold.fold import fold, instance_nodes
from verity_vllm.observe.fold.patterns import TransparentAten
from verity_vllm.observe.fold.resolver import Profile
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET
from tests.observe.test_fold_synthetic import C, PATTERNS, V, W_LN1, memory, toy_log, weight_field  # noqa: F401  (memory: the fixture)

H = C["H"]
ADD = f"AddScalarBf16_v1{{N={H},C=1.0}}"
READ = f"AddScalarF32_v1{{N={H},C=0.5}}"
ADD_SCHEMA = "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
TO_SCHEMA = ("aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) "
             "-> Tensor(a)")


def _log(header_target: dict | None = None) -> list:
    """The toy log with GemmaRMSNorm's `w.float() + 1.0` on `W_LN1` and a reader `+ 0.5` of it ahead of each step's first dispatch."""
    out, step = [], None
    for ev in toy_log():
        if isinstance(ev, RunHeader) and header_target is not None:
            ev = dataclasses.replace(ev, profile={**ev.profile, "target": header_target})
        if isinstance(ev, DispatchEnter) and ev.step != step:
            step = ev.step
            w32, w1, r = (V(90 + 3 * step + k, (H,), dtype="float32") for k in range(3))
            for k, v in enumerate((w32, w1, r)):
                out.append(StorageAlloc(v.alloc, 0x100000 + 0x1000 * v.alloc, v.numel * 4, "cuda:0", 0, "op"))
            for i, (op, schema, args, y) in enumerate((("aten.to.dtype", TO_SCHEMA, [W_LN1, "torch.float32"], w32),
                                                       ("aten.add.Tensor", ADD_SCHEMA, [w32, 1.0], w1), ("aten.add.Tensor", ADD_SCHEMA, [w1, 0.5], r))):
                out.append(DispatchEnter(9000 + 10 * step + i, None, step, "forward", op, schema, args, {}, 7, "model.layers.0.input_layernorm"))
                out.append(DispatchExit(9000 + 10 * step + i, y))
        out.append(ev)
    return out


def _fold(header_target: dict | None = None):
    profile = Profile("toy", [*PATTERNS, *gemma2_patterns(), TransparentAten()], dict(C), weight_field, False)
    r = fold(_log(header_target), profile, strict=True)
    assert r.report.unsupported == 0 and not r.errors
    return r


def _read_nodes(r, step: int) -> set[int]:
    (reader,) = instance_nodes(r, spec_prefix="AddScalarF32_v1", step=step)
    return {run[1] for a in r.body.nodes[reader["node"]].args for run in runs(a.refs)}


def test_once_issues_the_weight_add_at_its_first_step_and_the_later_step_reads_it(memory):
    r = _fold()
    assert r.report.by_spec[ADD] == r.report.by_spec[READ] == 2                 # the trace resolves the add in both forwards
    (add,) = instance_nodes(r, spec_prefix="AddScalarBf16_v1")
    assert add["step"] == 0 and _read_nodes(r, 0) == _read_nodes(r, 1) == {add["node"]}
    assert len(r.body.nodes) == len(r.instances) == sum(r.report.by_spec.values()) - 1


def test_per_forward_issues_it_at_every_step(memory):
    r = _fold(DEFAULT_TARGET.with_overrides(weight_only_calls="per-forward").to_json())
    adds = instance_nodes(r, spec_prefix="AddScalarBf16_v1")
    assert [a["step"] for a in adds] == [0, 1] and len(r.body.nodes) == sum(r.report.by_spec.values())
    assert [_read_nodes(r, s) for s in (0, 1)] == [{a["node"]} for a in adds]
    assert r.digest != _fold().digest
