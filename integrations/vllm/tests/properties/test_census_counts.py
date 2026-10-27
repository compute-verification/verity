"""The L5 census's launch-count gate (`census.reconcile`): per step, a declared kernel whose device launch count differs from
the step's events declaring it is refused by name (kernel, device count, declaring events, the buffers they write), unless
`kernel_allowlist.LAUNCH_COUNT_ALLOWLIST` exempts it with its reason.

The log is `test_fold_synthetic`'s toy log, whose fused norms and GEMMs here declare their kernels as `patterns/norms.py` and
the batch-invariant GEMM do.  The device census is the traced log's: each declared kernel under its Kineto name, once per
declaring event, which is what a bare run of that log launches."""
from __future__ import annotations

import dataclasses
import functools
import re

import pytest

from tests.observe import test_fold_synthetic as T
from verity_vllm.observe.events import DispatchEnter, DispatchExit
from verity_vllm.observe.fold import resolve_log as RL
from verity_vllm.observe.fold.resolver import Profile
from verity_vllm.properties import census as CEN
from verity_vllm.properties import kernel_allowlist as KA

NORM, ALT, INDEX = "fused_add_rms_norm_kernel", "fused_add_rms_norm_static_fp8_quant_kernel", "index_elementwise_kernel"
DECLARES = {"fused_norm": (NORM, ALT), "mm": ("matmul_kernel_persistent",), "index": (INDEX,)}
KINETO = {NORM: "void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>(c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, "
                "float, int, int)",
          "matmul_kernel_persistent": "matmul_kernel_persistent",
          INDEX: "void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<...>>(long, ...)"}


def _declaring(fn, kernels):
    def match(node, ctx):
        r = fn(node, ctx)
        return None if r is None else dataclasses.replace(r, kernels=kernels)
    return match


def profile() -> Profile:
    pats = [T.Pat(p.name, _declaring(p.fn, DECLARES[p.name])) if p.name in DECLARES else p for p in T.PATTERNS]
    return Profile("toy", pats, dict(T.C), T.weight_field, False)


def untraced_post_attention_norm(evs, step=0):
    """A9's repro: the layer-0 post-attention `_C.fused_add_rms_norm` launch of `step` ran with no dispatcher event.  It
    rewrites `o` and `hidden` in place, and the gate/up GEMM then reads `o`."""
    drop = {e.id for e in evs if isinstance(e, DispatchEnter) and e.step == step and e.op == "_C.fused_add_rms_norm.default"
            and e.module == "model.layers.0.post_attention_layernorm"}
    assert len(drop) == 1
    return [e for e in evs if not (isinstance(e, (DispatchEnter, DispatchExit)) and e.id in drop)]


@functools.cache
def log(name: str) -> list:
    return {"traced": T.toy_log, "untraced": lambda: untraced_post_attention_norm(T.toy_log())}[name]()


@pytest.fixture
def reconcile(monkeypatch):
    monkeypatch.setattr(RL, "read_log", lambda path: iter(log(str(path))))
    monkeypatch.setattr(RL, "load_profile", lambda name: (profile(), None))
    return lambda census, log: CEN.reconcile(census, log, "toy")


def device_census() -> dict:
    """What the bare run of the traced log launches: each declared kernel once per declaring event of its step (`ALT`, the
    norm's alternative, never)."""
    per_step = {}
    RL.resolve_log("traced", profile(), None, lambda se, sr: per_step.__setitem__(sr.step, (sr, se.roots)))
    steps = []
    for step, (sr, roots) in sorted(per_step.items()):
        _pats, events, _parts = CEN.declared_kernels_of_step(sr.resolutions, roots)
        kernels = {KINETO[k]: {"count": len(v), "us": 1.0} for k, v in sorted(events.items()) if k in KINETO}
        steps.append({"step": step, "n_kernels": sum(d["count"] for d in kernels.values()), "kernels": kernels})
    return {"steps": steps}


def launched(census: dict, step: int, kernel: str, delta: int) -> dict:
    s = census["steps"][step]
    s["kernels"][KINETO[kernel]]["count"] += delta
    s["n_kernels"] += delta
    return census


def norm_writes(name: str, step: int) -> dict[str, list[int]]:
    """module -> the allocations its fused norm writes in place (`x` and `residual`), from the log itself."""
    return {e.module: sorted({e.args[0].alloc, e.args[1].alloc}) for e in log(name)
            if isinstance(e, DispatchEnter) and e.step == step and e.op == "_C.fused_add_rms_norm.default"}


def test_matching_launch_counts_pass(reconcile):
    rep = reconcile(device_census(), "traced")
    assert rep["ok"] and rep["residue_zero"] and rep["counts_equal"]
    assert rep["totals"]["count_mismatches"] == 0 and rep["count_mismatches"] == [] and rep["totals"]["count_exempt"] == {}
    assert rep["steps"][0]["counts"][NORM] == {"census_kernels": 2, "declaring_events": 2, "declaring_parts": 2, "patterns": ["fused_norm"]}


def test_infrastructure_and_untaken_alternatives_are_not_counted(reconcile):
    """The frozen residue allowlist accepts `index_elementwise_kernel` at any count (here the runner's three extra launches
    beside the one declaring event), and `ALT`, declared beside the norm's kernel, is an alternative no launch took."""
    rep = reconcile(launched(device_census(), 0, INDEX, 3), "traced")
    assert rep["ok"] and rep["count_mismatches"] == []
    c = rep["steps"][0]["counts"]
    assert (c[INDEX]["census_kernels"], c[INDEX]["declaring_events"], c[INDEX]["infrastructure"]) == (4, 1, 4)
    assert (c[ALT]["census_kernels"], c[ALT]["declaring_events"]) == (0, 2) and ALT in rep["steps"][0]["declared_never_seen"]


@pytest.mark.parametrize("delta", [1, -1])
def test_a_launch_count_that_differs_is_refused_by_name(reconcile, delta):
    rep = reconcile(launched(device_census(), 1, NORM, delta), "traced")
    assert rep["residue_zero"] and not rep["counts_equal"] and not rep["ok"]
    assert rep["totals"]["count_mismatches"] == 1 and rep["totals"]["count_mismatch_kernels"] == [NORM]
    (m,) = rep["count_mismatches"]
    writes = norm_writes("traced", 1)
    assert (m["step"], m["kernel"], m["device_count"], m["declaring_events"]) == (1, NORM, 2 + delta, 2)
    assert m["device_kernels"] == {KINETO[NORM]: 2 + delta}
    assert [(e["module"], e["writes"]) for e in m["events"]] == [(k, writes[k]) for k in
                                                                ("model.layers.0.post_attention_layernorm", "model.norm")]
    assert m["buffers"] == sorted({a for w in writes.values() for a in w})
    assert re.fullmatch(rf"step 1: {NORM} launched {2 + delta} times on the device against 2 declaring events \(buffers .*\)",
                        CEN.count_line(m))
    assert rep["steps"][1]["count_mismatches"] == [NORM] and rep["steps"][0]["count_mismatches"] == []


def test_an_allowlisted_kernel_passes_with_its_reason_recorded(reconcile, monkeypatch):
    why = "toy: a second launch per event that writes only its own scratch"
    monkeypatch.setattr(KA, "LAUNCH_COUNT_ALLOWLIST", {NORM: why})
    rep = reconcile(launched(device_census(), 1, NORM, 1), "traced")
    assert rep["ok"] and rep["counts_equal"] and rep["count_mismatches"] == []
    assert rep["totals"]["count_exempt"] == {NORM: 1} and rep["count_allowlist"] == {NORM: why}
    assert rep["steps"][1]["counts"][NORM]["exempt"] == why and "exempt" not in rep["steps"][0]["counts"][NORM]


def test_a9_untraced_in_place_overwrite_is_refused(reconcile):
    """An untraced in-place overwrite: the device ran both step-0 fused norms, the log holds only
    the final one.  The residue stays zero, since the final norm's declaration claims both launches by name; the count
    refuses it, naming the final norm's event and the buffers it writes."""
    rep = reconcile(device_census(), "untraced")
    assert rep["residue_zero"] and rep["totals"]["residue_kernels"] == 0
    assert not rep["ok"] and rep["totals"]["count_mismatch_kernels"] == [NORM]
    (m,) = rep["count_mismatches"]
    assert (m["step"], m["device_count"], m["declaring_events"]) == (0, 2, 1)
    assert [(e["op"], e["module"]) for e in m["events"]] == [("_C.fused_add_rms_norm.default", "model.norm")]
    assert m["buffers"] == norm_writes("untraced", 0)["model.norm"]


def test_every_launch_count_exemption_has_a_one_line_reason():
    for name, why in KA.LAUNCH_COUNT_ALLOWLIST.items():
        assert name in CEN.KERNEL_MATCH or re.fullmatch(r"[A-Za-z_][\w:]*", name), name
        assert isinstance(why, str) and why.strip() and "\n" not in why, name
