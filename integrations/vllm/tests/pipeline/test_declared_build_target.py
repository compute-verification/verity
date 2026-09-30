"""A row whose device class fixes its whole Build target (`config.TARGET_NUM_SMS`, e.g. `__rtxpro6000__` = cc 12.0 / 188 SMs) Builds
on a GPU-free host: the target is declared by the row id, the preflight asks no device of a Build-only stage, and the Build selects
vLLM's CUDA platform instead of letting `vllm.platforms` probe NVML (absent there) for it.  Every other class Builds as before."""
from __future__ import annotations

import json
import sys
import types

import pytest

from verity_vllm import target_family as tf
from verity_vllm.engine import env as engine_env, hooks
from verity_vllm.pipeline import build as B
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row import fa_version
from verity_vllm.program.frontend.target_profile import TargetProfile, parse_target_arg

ROW_120 = "smollm2-135m__bf16__rtxpro6000__tp1__b1__i256__o32__mixed__greedy__bi-eager"
ROW_89 = "smollm2-135m__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
ROW_90 = "llama32-1b__bf16__h100__tp1__b1__i1024__o128__mixed__greedy__bi-eager"


def _wl(tmp_path, target=None):
    p = tmp_path / "wl.json"
    p.write_text(json.dumps({"sweep": {}, **({"target": target} if target is not None else {})}))
    return str(p)


def test_an_rtxpro6000_row_declares_its_whole_target_and_other_classes_keep_the_workload_s(tmp_path):
    assert tf.declared_profile(ROW_120, {}) == {"compute_capability": [12, 0], "num_sms": 188}
    assert tf.declared_profile(ROW_120, {"target": {"compilation": "piecewise"}}) == {"compute_capability": [12, 0], "num_sms": 188, "compilation": "piecewise"}
    assert tf.declared_profile(ROW_120, {"target": {"compute_capability": [12, 0], "num_sms": 170}})["num_sms"] == 170, "the workload's own target wins"
    assert tf.declared_profile(ROW_89, {}) == {} and tf.declared_profile(ROW_90, {}) == {}
    assert R.declared_target(_wl(tmp_path), row=ROW_120) == '{"compute_capability": [12, 0], "num_sms": 188}'
    assert R.declared_target(_wl(tmp_path), row=ROW_89) == "", "an l40s row's Build target stays the host GPU's"
    h100 = {"compute_capability": [9, 0], "num_sms": 132, "flash_attn_version": 2}
    assert R.declared_target(_wl(tmp_path, h100), row=ROW_90) == R.declared_target(_wl(tmp_path, h100)) == json.dumps(h100, sort_keys=True)
    assert parse_target_arg(R.declared_target(_wl(tmp_path), row=ROW_120)) == TargetProfile.default().with_overrides(compute_capability=(12, 0), num_sms=188), \
        "the declared profile is the one a GPU host's from_device() gives an RTX PRO 6000"


def test_the_preflight_asks_no_device_of_a_declared_build_and_still_asks_it_of_a_commit(monkeypatch):
    def no_gpu():
        raise AssertionError("device queried")
    monkeypatch.setattr(tf, "device_cc", no_gpu)
    rec = tf.check(ROW_120, {"sweep": {}}, ["build"])
    assert rec["verdict"] == "PASS" and rec["declared"] == [12, 0] and rec["device"] is None
    monkeypatch.setattr(tf, "device_cc", lambda: (None, "nvidia-smi absent"))
    assert tf.check(ROW_120, {"sweep": {}}, ["build", "commit"])["verdict"] == "FAIL", "the Commit still needs the real device"
    assert tf.check(ROW_89, {"sweep": {}}, ["build"])["verdict"] == "FAIL", "an undeclared Build still needs the device it targets"
    monkeypatch.setattr(tf, "device_cc", lambda: ((8, 9), "nvidia-smi compute_cap"))
    assert tf.check(ROW_120, {"sweep": {}}, ["build", "commit"])["verdict"] == "FAIL", "a Commit on another family's GPU is refused"


def test_fa_version_and_the_cache_key_of_a_declared_target_read_no_device(tmp_path, monkeypatch):
    monkeypatch.setattr(TargetProfile, "from_device", classmethod(lambda cls, device=0: (_ for _ in ()).throw(AssertionError("device read"))))
    assert fa_version(_wl(tmp_path), R.declared_target(_wl(tmp_path), row=ROW_120)) == 2
    monkeypatch.setattr(RC.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")))
    row = types.SimpleNamespace(tgt=R.declared_target(_wl(tmp_path), row=ROW_120))
    assert RC.device_id(row) == {"declared_target": row.tgt}
    assert RC.device_id(types.SimpleNamespace(tgt="")) is None, "no nvidia-smi and no declared target: no device in the key"


_FAKE_VLLM = {
    # `import vllm` resolves `current_platform` (the pinned vLLM: `env_override` -> `torch_utils.PIN_MEMORY`)
    "vllm/__init__.py": "from vllm.platforms import current_platform as _\n",
    "vllm/platforms/__init__.py": """
import importlib
_current_platform = None


class UnspecifiedPlatform:
    def is_cuda(self):
        return False


class NvmlCudaPlatform:
    def is_cuda(self):
        return True

    @classmethod
    def device_count(cls):
        return 0


def resolve_current_platform_cls_qualname():
    return "vllm.platforms.UnspecifiedPlatform"   # the NVML probe of a GPU-free host


def __getattr__(name):
    global _current_platform
    if name != "current_platform":
        raise AttributeError(name)
    if _current_platform is None:
        mod, _, cls = resolve_current_platform_cls_qualname().rpartition(".")
        _current_platform = getattr(importlib.import_module(mod), cls)()
    return _current_platform
""",
    "vllm/platforms/cuda.py": """
import os
ENV_AT_IMPORT = {"VLLM_USE_LAYERNAME": os.environ.get("VLLM_USE_LAYERNAME")}   # vLLM freezes `_USE_LAYERNAME` at import


class NonNvmlCudaPlatform:
    def is_cuda(self):
        return True

    @classmethod
    def device_count(cls):
        return 0   # the GPU-free host's visible count (`CUDA_VISIBLE_DEVICES=`)


CudaPlatform = NonNvmlCudaPlatform
""",
}


@pytest.fixture
def vllm_without_nvml(tmp_path, monkeypatch):
    """A stand-in for the pinned vLLM's platform resolution on a GPU-free host, not yet imported: `current_platform` is resolved lazily
    (first by `import vllm` itself) by the module's `resolve_current_platform_cls_qualname`, whose NVML probe answers a non-CUDA
    platform; `vllm.platforms.cuda.CudaPlatform` is the torch-backed class there."""
    root = tmp_path / "site"
    for rel, src in _FAKE_VLLM.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(src)
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    for k in engine_env.EXPORT:   # set-then-delete: teardown restores each to what it was, the selection's pin included
        monkeypatch.setenv(k, "")
        monkeypatch.delenv(k)
    yield
    for p in hooks.live():
        o = getattr(p, "owner", None)
        if (getattr(o, "__module__", None) or getattr(o, "__name__", "")).startswith("vllm"):
            p.uninstall()
    for name in [m for m in sys.modules if m == "vllm" or m.startswith("vllm.")]:
        del sys.modules[name]


def _build(tmp_path, monkeypatch, target, tp=1, rank=0, seen=None):
    seen = {} if seen is None else seen

    def construct(out, res, *a, **k):
        from vllm.platforms import current_platform
        seen["platform"] = type(current_platform).__name__
        if hasattr(current_platform, "device_count"):
            seen["device_count"] = current_platform.device_count()
    monkeypatch.setattr(B, "_construct", construct)
    monkeypatch.setattr(B.engine_env, "apply_target", lambda env: None)
    monkeypatch.setattr(B, "_cuda_available", lambda: False)
    B.run(str(tmp_path / "out"), 1, 8, "b1-eager-v3", "gloo", target=target, tp=tp, rank=rank)
    return seen["platform"]


def test_a_declared_target_s_build_takes_the_cuda_platform_when_nvml_is_absent(tmp_path, monkeypatch, vllm_without_nvml):
    target = parse_target_arg(R.declared_target(_wl(tmp_path), row=ROW_120))
    assert _build(tmp_path, monkeypatch, target) == "NonNvmlCudaPlatform"
    assert sys.modules["vllm.platforms.cuda"].ENV_AT_IMPORT == {"VLLM_USE_LAYERNAME": "0"}, \
        "the selection imports vLLM, so the construction's export env is in force first (else layer names export as opaque LayerName)"


def test_a_build_without_a_declared_target_resolves_the_platform_as_before(tmp_path, monkeypatch, vllm_without_nvml):
    assert _build(tmp_path, monkeypatch, None) == "UnspecifiedPlatform", "no declared target: vLLM's own probe decides, unchanged"


def test_the_selection_answers_an_imported_but_unresolved_vllm_keeps_a_cuda_platform_and_refuses_a_non_cuda_one(vllm_without_nvml):
    from verity_vllm.program.frontend.export_compat import select_cuda_platform
    import vllm.platforms as P
    P._current_platform = None
    assert select_cuda_platform() == "NonNvmlCudaPlatform"
    P._current_platform = P.NvmlCudaPlatform()
    assert select_cuda_platform() == "NvmlCudaPlatform"
    P._current_platform = P.UnspecifiedPlatform()
    with pytest.raises(RuntimeError, match="UnspecifiedPlatform"):
        select_cuda_platform()


ROW_TP2 = "llama32-1b__bf16__rtxpro6000__tp2__b1__i256__o32__mixed__greedy__bi-eager"


@pytest.mark.parametrize("rank", [0, 1])
def test_a_declared_tp2_target_s_rank_build_counts_the_declared_world_not_the_host_s_devices(tmp_path, monkeypatch, vllm_without_nvml, rank):
    """vLLM's `ParallelConfig` refuses a world larger than `current_platform.device_count()` ("World size (2) is larger than the number
    of available GPUs (0)"); a declared target of world W is W devices, whatever the Build host shows."""
    target = parse_target_arg(R.declared_target(_wl(tmp_path), row=ROW_TP2))
    seen: dict = {}
    assert _build(tmp_path, monkeypatch, target, tp=2, rank=rank, seen=seen) == "NonNvmlCudaPlatform"
    assert seen["device_count"] == 2


def test_a_tp2_build_without_a_declared_target_counts_the_host_s_devices(tmp_path, monkeypatch, vllm_without_nvml):
    assert _build(tmp_path, monkeypatch, None, tp=2) == "UnspecifiedPlatform"
    from vllm.platforms.cuda import CudaPlatform
    assert CudaPlatform.device_count() == 0, "no declared target: the host's count, as before"
