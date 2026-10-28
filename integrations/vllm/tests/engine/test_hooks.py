"""`engine.hooks`: every patch uninstalls to exactly what the owner held, on plain objects and on the vLLM, torch and
Triton targets the integration patches."""
from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from verity_vllm.engine import hooks


class Runner:
    def execute_model(self, x):
        return ("runner", x)

    def sample(self, x):
        return ("sample", x)


class Settings:
    """Stores attributes outside its ``__dict__``, like torch's config proxies."""

    def __init__(self) -> None:
        object.__setattr__(self, "_store", {"level": "0"})

    def __getattr__(self, name):
        try:
            return self._store[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value) -> None:
        self._store[name] = value

    def __delattr__(self, name) -> None:
        del self._store[name]


def test_inherited_method_is_inherited_again() -> None:
    r = Runner()
    p = hooks.patch(r, "execute_model", lambda x: ("wrapped", x))
    assert r.execute_model(1) == ("wrapped", 1) and p.prior is hooks.ABSENT
    assert p.original.__func__ is Runner.execute_model
    p.uninstall()
    assert "execute_model" not in vars(r) and r.execute_model(1) == ("runner", 1)
    p.uninstall()                                            # idempotent
    assert "execute_model" not in vars(r)


def test_own_attribute_is_the_same_object_again() -> None:
    orig = Runner.sample
    mod = types.ModuleType("fake_vllm_layer")
    fn = mod.tensor_model_parallel_all_reduce = lambda x: x
    with hooks.Hooks() as h:
        h.patch(Runner, "sample", lambda self, x: None)
        h.wrap(mod, "tensor_model_parallel_all_reduce", lambda o: (lambda x: o(x) * 2))
        assert mod.tensor_model_parallel_all_reduce(3) == 6 and Runner().sample(0) is None
        assert len(h.entries) == 2
    assert vars(Runner)["sample"] is orig and vars(mod)["tensor_model_parallel_all_reduce"] is fn
    assert h.entries == ()


def test_stacked_patches_unwind_to_the_original() -> None:
    r = Runner()
    r.sample = own = lambda x: ("own", x)
    h = hooks.Hooks()
    h.wrap(r, "sample", lambda o: (lambda x: ("outer", o(x))))
    h.wrap(r, "sample", lambda o: (lambda x: ("outermost", o(x))))
    assert r.sample(1) == ("outermost", ("outer", ("own", 1)))
    h.uninstall()
    assert vars(r)["sample"] is own


def test_attribute_kept_outside_dict_is_set_back_by_value() -> None:
    s = Settings()
    p = hooks.patch(s, "level", "1")
    assert p.by_value and s.level == "1"
    p.uninstall()
    assert s.level == "0"
    q = hooks.patch(s, "fresh", 5)
    q.uninstall()
    assert not hasattr(s, "fresh")


def test_missing_attribute_is_missing_again() -> None:
    mod = types.ModuleType("m")
    with hooks.patched(mod, "_manager", object()):
        assert hasattr(mod, "_manager")
    assert not hasattr(mod, "_manager")


def test_patched_restores_on_error_and_live_lists_what_is_installed() -> None:
    before = hooks.live()
    r = Runner()
    with pytest.raises(RuntimeError), hooks.patched(r, "sample", None) as p:
        assert hooks.live() == (*before, p)
        raise RuntimeError
    assert "sample" not in vars(r) and hooks.live() == before


def test_appended_hook_leaves_the_list() -> None:
    def hook(name, t):
        return None

    other = [len]
    lst = [print, *other]
    with hooks.Hooks() as h:
        h.append(lst, hook)
        h.append(lst, hook)
        assert lst.count(hook) == 2
    assert lst == [print, *other]


def test_a_service_stops_before_the_patches_it_serves_and_keeps_its_counters() -> None:
    log: list[str] = []
    r = Runner()
    before = hooks.live()
    with hooks.Hooks(tag="pous:band-chain/d12/v1") as h:
        h.wrap(r, "sample", lambda o: (lambda x: ("decoded", o(x))))
        s = h.service("responder", lambda: log.append("start") or 7, lambda handle: log.append(f"stop {handle}") or {"answered": 3})
        assert log == ["start"] and s.handle == 7 and s.tag == h.tag
        assert hooks.live(h.tag) == (h.entries[0], s) and hooks.live("pouw:ncp-v1") == ()
        assert r.sample(1) == ("decoded", ("sample", 1))
    assert log == ["start", "stop 7"] and s.result == {"answered": 3} and "sample" not in vars(r) and hooks.live() == before
    s.uninstall()
    assert log == ["start", "stop 7"]


def test_an_entry_that_fails_to_uninstall_does_not_keep_the_others() -> None:
    r = Runner()
    before = hooks.live()
    h = hooks.Hooks()
    h.wrap(r, "sample", lambda o: o)

    def stop(handle):
        raise RuntimeError("stuck")

    h.service("responder", lambda: None, stop)
    with pytest.raises(RuntimeError, match="stuck"):
        h.uninstall()
    assert "sample" not in vars(r) and h.entries == () and hooks.live() == before


def _tagged(tag: str):
    return lambda o: (lambda x: (tag, o(x)))


def test_an_inner_patch_refuses_to_come_off_under_another_components_patch() -> None:
    """The out-of-order uninstall that used to drop the outer wrapper silently (it stopped running while `live()` listed
    it) and bring the inner one back once both were off: it is refused, and nothing changes."""
    r = Runner()
    before = hooks.live()
    inner, outer = hooks.Hooks(tag="inner"), hooks.Hooks(tag="outer")
    pi = inner.wrap(r, "sample", _tagged("inner"))
    po = outer.wrap(r, "sample", _tagged("outer"))
    with pytest.raises(RuntimeError, match=r"refused to uninstall Hooks\(tag='inner'\): the patch of a Runner\.sample "
                                           r"\(value .*, tag 'outer'\) was installed on its patch of a Runner\.sample"):
        inner.uninstall()
    with pytest.raises(RuntimeError, match=r"refused to uninstall the patch of a Runner\.sample .*tag 'inner'.*tag 'outer'"):
        pi.uninstall()
    assert r.sample(1) == ("outer", ("inner", ("sample", 1)))
    assert pi.installed and inner.entries == (pi,) and hooks.live() == (*before, pi, po)
    outer.uninstall()
    assert r.sample(1) == ("inner", ("sample", 1)) and hooks.live() == (*before, pi)
    inner.uninstall()
    assert "sample" not in vars(r) and r.sample(1) == ("sample", 1) and hooks.live() == before


def test_a_refused_uninstall_leaves_every_entry_of_the_component_on() -> None:
    log: list[str] = []
    r, s = Runner(), Settings()
    a, b = hooks.Hooks(tag="a"), hooks.Hooks(tag="b")
    a.wrap(r, "execute_model", _tagged("a"))
    a.wrap(r, "sample", _tagged("a"))
    a.patch(s, "level", "1")
    a.service("responder", lambda: log.append("start"), lambda handle: log.append("stop"))
    b.patch(s, "level", "2")
    entries = a.entries
    with pytest.raises(RuntimeError, match=r"the patch of a Settings\.level \(value str, tag 'b'\)"):
        a.uninstall()
    assert a.entries == entries and log == ["start"] and s.level == "2"
    assert r.execute_model(1) == ("a", ("runner", 1)) and r.sample(1) == ("a", ("sample", 1))
    b.uninstall()
    a.uninstall()
    assert log == ["start", "stop"] and s.level == "0" and not vars(r) and a.entries == ()


def test_two_components_on_one_attribute_unwind_in_order() -> None:
    r = Runner()
    before = hooks.live()
    with hooks.Hooks(tag="inner") as inner:
        inner.wrap(r, "sample", _tagged("inner"))
        with hooks.Hooks(tag="outer") as outer:
            outer.wrap(r, "sample", _tagged("outer"))
            assert r.sample(1) == ("outer", ("inner", ("sample", 1))) and hooks.live() == (*before, *inner.entries, *outer.entries)
        assert r.sample(1) == ("inner", ("sample", 1)) and hooks.live() == (*before, *inner.entries)
    assert "sample" not in vars(r) and hooks.live() == before


def test_triton_launch_restores_jitfunction_run() -> None:
    jit = pytest.importorskip("triton.runtime.jit")
    from verity_vllm.observe import triton_adapter
    from verity_vllm.program.frontend.triton_capture import intercept_triton_launches
    own = vars(jit.JITFunction).get("run", hooks.ABSENT)
    p = hooks.triton_launch(lambda orig: (lambda self, *a, **kw: orig(self, *a, **kw)))
    assert jit.JITFunction.run is p.value and p.original is not p.value
    with intercept_triton_launches():
        assert jit.JITFunction.run is not p.value
    assert jit.JITFunction.run is p.value
    p.uninstall()
    assert vars(jit.JITFunction).get("run", hooks.ABSENT) is own
    if not triton_adapter.is_installed():
        triton_adapter.install(None)
        assert jit.JITFunction.run is not own
        triton_adapter.uninstall()
        assert vars(jit.JITFunction).get("run", hooks.ABSENT) is own and not triton_adapter.is_installed()


def test_export_shims_restore_torch_and_vllm() -> None:
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.export_compat import tensor_data_as_detach
    data = vars(torch.Tensor).get("data", hooks.ABSENT)
    with tensor_data_as_detach():
        assert isinstance(vars(torch.Tensor)["data"], property)
    assert vars(torch.Tensor).get("data", hooks.ABSENT) is data

    pytest.importorskip("vllm")
    import importlib

    from vllm.platforms import current_platform

    from verity_vllm.program.frontend.export_compat import _NUM_CU_MODULES, apply_target_profile, moe_ops_traced
    cls = type(current_platform)
    cap = vars(cls).get("get_device_capability", hooks.ABSENT)
    mods = []
    for name in _NUM_CU_MODULES:
        try:
            mods.append(importlib.import_module(name))
        except Exception:  # noqa: BLE001
            continue
    cus = [vars(m).get("num_compute_units", hooks.ABSENT) for m in mods]
    with apply_target_profile(SimpleNamespace(compute_capability=(8, 9), num_sms=142)):
        assert tuple(cls.get_device_capability()) == (8, 9)
        assert all(m.num_compute_units() == 142 for m in mods if hasattr(m, "num_compute_units"))
    assert vars(cls).get("get_device_capability", hooks.ABSENT) is cap
    assert [vars(m).get("num_compute_units", hooks.ABSENT) for m in mods] == cus
    flashinfer_fp8 = importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.flashinfer")
    flashinfer_supported = flashinfer_fp8.is_flashinfer_fp8_blockscale_gemm_supported
    with apply_target_profile(SimpleNamespace(compute_capability=(9, 0), num_sms=132)):
        assert flashinfer_fp8.is_flashinfer_fp8_blockscale_gemm_supported()
    assert flashinfer_fp8.is_flashinfer_fp8_blockscale_gemm_supported is flashinfer_supported

    fm = pytest.importorskip("vllm.model_executor.layers.fused_moe.fused_moe")   # registers torch.ops.vllm.fused_experts
    ns = torch.ops.vllm
    try:
        getattr(ns, "fused_experts")                         # resolving caches the packet on the namespace
    except (AttributeError, RuntimeError):
        pass
    fe = vars(ns).get("fused_experts", hooks.ABSENT)
    try:
        from vllm.v1.worker import workspace as ws
    except Exception:  # noqa: BLE001
        ws = types.ModuleType("absent")
    manager = vars(ws).get("_manager", hooks.ABSENT)
    live = hooks.live()
    with moe_ops_traced(None):
        inside = vars(ns).get("fused_experts", hooks.ABSENT)
    assert fe is hooks.ABSENT or inside is fm.fused_experts_op
    assert vars(ns).get("fused_experts", hooks.ABSENT) is fe and vars(ws).get("_manager", hooks.ABSENT) is manager
    assert hooks.live() == live


@pytest.mark.slow
def test_fp8_expert_scales_are_exported_from_their_registered_parameter_owner() -> None:
    """A plain kernel object keeps the original scale object; the shim refreshes it after export swaps the module Parameter."""
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.export_compat import _expert_weight_scales_through_owner

    class Experts:
        def __init__(self, w1_scale, w2_scale) -> None:
            self.quant_config = SimpleNamespace(
                _w1=SimpleNamespace(scale=w1_scale),
                _w2=SimpleNamespace(scale=w2_scale),
            )

        @property
        def w1_scale(self):
            return self.quant_config._w1.scale

        @property
        def w2_scale(self):
            return self.quant_config._w2.scale

        def apply(self, x):
            return x * self.w1_scale + self.w2_scale

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w13_weight_scale_inv = torch.nn.Parameter(torch.ones(4), requires_grad=False)
            self.w2_weight_scale_inv = torch.nn.Parameter(torch.full((4,), 2.0), requires_grad=False)
            experts = Experts(self.w13_weight_scale_inv, self.w2_weight_scale_inv)
            self.quant_method = SimpleNamespace(moe_kernel=SimpleNamespace(fused_experts=experts))

        def forward(self, x):
            return self.quant_method.moe_kernel.fused_experts.apply(x)

    layer = Layer()
    inputs = (torch.ones(4),)
    unshimmed = torch.export.export(layer, inputs, strict=False)
    assert set(unshimmed.graph_signature.inputs_to_lifted_tensor_constants.values()) == {"lifted_tensor_0", "lifted_tensor_1"}
    assert set(unshimmed.graph_signature.inputs_to_parameters.values()) == {"w13_weight_scale_inv", "w2_weight_scale_inv"}

    h = hooks.Hooks()
    _expert_weight_scales_through_owner(layer, layer.quant_method, h)
    with h:
        shimmed = torch.export.export(layer, inputs, strict=False)
        assert not shimmed.graph_signature.inputs_to_lifted_tensor_constants
        assert set(shimmed.graph_signature.inputs_to_parameters.values()) == {"w13_weight_scale_inv", "w2_weight_scale_inv"}
    assert layer.quant_method.moe_kernel.fused_experts.w1_scale is layer.w13_weight_scale_inv
    assert layer.quant_method.moe_kernel.fused_experts.w2_scale is layer.w2_weight_scale_inv


def test_a_routers_parameters_are_exported_from_their_registered_owner() -> None:
    """A plain router keeps the gate's correction bias and DeepSeek-V4's hash table; the shim reads both from the gate."""
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.export_compat import _router_params_through_owner

    class Router:
        def __init__(self, bias, table) -> None:
            self.e_score_correction_bias, self._hash_indices_table, self.top_k = bias, table, 2

        def select_experts(self, logits, ids):
            return logits + self.e_score_correction_bias, self._hash_indices_table[ids]

    class Moe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = torch.nn.Module()
            self.gate.e_score_correction_bias = torch.nn.Parameter(torch.ones(4), requires_grad=False)
            self.gate.tid2eid = torch.nn.Parameter(torch.zeros(8, 2, dtype=torch.int32), requires_grad=False)
            self.router = Router(self.gate.e_score_correction_bias, self.gate.tid2eid)

        def forward(self, logits, ids):
            return self.router.select_experts(logits, ids)

    moe = Moe()
    inputs = (torch.ones(3, 4), torch.zeros(3, dtype=torch.int64))
    unshimmed = torch.export.export(moe, inputs, strict=False)
    assert len(unshimmed.graph_signature.inputs_to_lifted_tensor_constants) == 2

    h = hooks.Hooks()
    _router_params_through_owner(moe, h)
    with h:
        shimmed = torch.export.export(moe, inputs, strict=False)
        assert not shimmed.graph_signature.inputs_to_lifted_tensor_constants
        assert set(shimmed.graph_signature.inputs_to_parameters.values()) == {"gate.e_score_correction_bias", "gate.tid2eid"}
    assert moe.router.e_score_correction_bias is moe.gate.e_score_correction_bias
    assert moe.router._hash_indices_table is moe.gate.tid2eid


def test_tp_group_override_restores_the_group() -> None:
    pytest.importorskip("vllm")
    from vllm.distributed import parallel_state as ps

    from verity_vllm.program.frontend.export_ops import CollectiveBus, tp_group_override
    tp = vars(ps).get("_TP", hooks.ABSENT)
    with tp_group_override(0, 2, CollectiveBus(rank=0, world=2)) as stub:
        assert ps._TP is stub
    assert vars(ps).get("_TP", hooks.ABSENT) is tp
