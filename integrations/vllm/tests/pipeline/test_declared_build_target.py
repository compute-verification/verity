"""A row whose device class fixes its whole Build target (`config.TARGET_NUM_SMS`, e.g. `__rtxpro6000__` = cc 12.0 / 188 SMs) Builds
on a GPU-free host: the target is declared by the row id, the preflight asks no device of a Build-only stage, and the Build selects
vLLM's CUDA platform instead of letting `vllm.platforms` probe NVML (absent there) for it.  Every other class Builds as before."""
from __future__ import annotations

import json
import sys
import types

import pytest

from verity_vllm import target_family as tf
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


class _Platform:
    def __init__(self, cuda: bool):
        self._cuda = cuda

    def is_cuda(self) -> bool:
        return self._cuda


class NvmlCudaPlatform(_Platform):
    def __init__(self):
        super().__init__(True)


class NonNvmlCudaPlatform(_Platform):
    def __init__(self):
        super().__init__(True)


class UnspecifiedPlatform(_Platform):
    def __init__(self):
        super().__init__(False)


@pytest.fixture
def vllm_without_nvml(monkeypatch):
    """The pinned vLLM's platform resolution on a GPU-free host: `vllm.platforms.current_platform` is resolved lazily on first access by
    probing NVML, which is absent (-> a non-CUDA platform); `vllm.platforms.cuda.CudaPlatform` is the torch-backed class there."""
    plat = types.ModuleType("vllm.platforms")
    plat._current_platform = None

    def __getattr__(name):
        if name == "current_platform":
            if plat._current_platform is None:
                plat._current_platform = UnspecifiedPlatform()
            return plat._current_platform
        raise AttributeError(name)

    plat.__getattr__ = __getattr__
    cuda = types.ModuleType("vllm.platforms.cuda")
    cuda.CudaPlatform = NonNvmlCudaPlatform
    vllm = types.ModuleType("vllm")
    vllm.platforms = plat
    plat.cuda = cuda
    for name, mod in (("vllm", vllm), ("vllm.platforms", plat), ("vllm.platforms.cuda", cuda)):
        monkeypatch.setitem(sys.modules, name, mod)
    return plat


def _build(tmp_path, monkeypatch, target):
    seen = {}

    def construct(out, res, *a, **k):
        from vllm.platforms import current_platform
        seen["platform"] = type(current_platform).__name__
    monkeypatch.setattr(B, "_construct", construct)
    monkeypatch.setattr(B.engine_env, "apply_target", lambda env: None)
    monkeypatch.setattr(B, "_cuda_available", lambda: False)
    B.run(str(tmp_path / "out"), 1, 8, "b1-eager-v3", "gloo", target=target)
    return seen["platform"]


def test_a_declared_target_s_build_takes_the_cuda_platform_when_nvml_is_absent(tmp_path, monkeypatch, vllm_without_nvml):
    target = parse_target_arg(R.declared_target(_wl(tmp_path), row=ROW_120))
    assert _build(tmp_path, monkeypatch, target) == "NonNvmlCudaPlatform"


def test_a_build_without_a_declared_target_resolves_the_platform_as_before(tmp_path, monkeypatch, vllm_without_nvml):
    assert _build(tmp_path, monkeypatch, None) == "UnspecifiedPlatform", "no declared target: vLLM's own probe decides, unchanged"


def test_the_selection_keeps_an_already_resolved_cuda_platform_and_refuses_a_non_cuda_one(vllm_without_nvml):
    from verity_vllm.program.frontend.export_compat import select_cuda_platform
    vllm_without_nvml._current_platform = NvmlCudaPlatform()
    assert select_cuda_platform() == "NvmlCudaPlatform"
    vllm_without_nvml._current_platform = UnspecifiedPlatform()
    with pytest.raises(RuntimeError, match="UnspecifiedPlatform"):
        select_cuda_platform()
