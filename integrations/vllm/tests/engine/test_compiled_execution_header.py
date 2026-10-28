"""The run header attests the EXECUTION the engine was built with, and a mislabelled engine is refused (torch required for the
adapter import; no GPU: the engine is a fake carrying a vllm_config-shaped object).

  * `compilation_facts` reads mode / cudagraph_mode / capture sizes / partition / enforce_eager back from the built engine's vllm_config in
    the form the global Program and the global match compare (enum NAMES, not the IntEnum's str())
  * `target_mismatches`: an eager profile against a compiled engine, and a `piecewise` profile against an eager engine or a
    FULL_AND_PIECEWISE engine, are refused by name; the matching pair is accepted
"""
from __future__ import annotations

import enum
import types

import pytest

torch = pytest.importorskip("torch")

from verity_vllm.engine import build as B, vllm_adapter as A  # noqa: E402
from verity_vllm.program.frontend.target_profile import TargetProfile  # noqa: E402


class _Mode(enum.IntEnum):
    NONE = 0
    VLLM_COMPILE = 3


class _CG(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL_AND_PIECEWISE = 2


def _llm(enforce_eager: bool, cg: _CG = _CG.PIECEWISE, sizes=(1, 2), emulate: bool = False):
    cc = types.SimpleNamespace(mode=_Mode.NONE if enforce_eager else _Mode.VLLM_COMPILE, cudagraph_mode=cg, cudagraph_capture_sizes=list(sizes),
                               max_cudagraph_capture_size=max(sizes), use_inductor_graph_partition=False, backend="",
                               inductor_compile_config={"emulate_precision_casts": True, "combo_kernels": True} if emulate else {"combo_kernels": True})
    vc = types.SimpleNamespace(compilation_config=cc, model_config=types.SimpleNamespace(enforce_eager=enforce_eager))
    return types.SimpleNamespace(llm_engine=types.SimpleNamespace(vllm_config=vc))


def test_compilation_facts_are_read_back_in_the_declared_form():
    f = A.compilation_facts(_llm(False))
    assert f["mode"] == "VLLM_COMPILE" and f["mode_value"] == 3 and f["cudagraph_mode"] == "PIECEWISE" and f["capture_sizes"] == [1, 2]
    assert f["enforce_eager"] is False and f["use_inductor_graph_partition"] is False and f["backend"] == "inductor"
    assert f["emulate_precision_casts"] is False and f["inductor_compile_config"]["combo_kernels"] is True
    assert A.compilation_facts(_llm(False, emulate=True))["emulate_precision_casts"] is True
    e = A.compilation_facts(_llm(True, _CG.NONE))
    assert e["enforce_eager"] is True and e["mode"] == "NONE"
    # the declared facts of the `piecewise` profile are a subset of the read-back (what the global match's G1 compares key by key)
    decl = TargetProfile(compilation="piecewise").modelled_engine()["compilation"]
    assert all(f[k] == decl[k] for k in ("mode", "cudagraph_mode", "use_inductor_graph_partition", "emulate_precision_casts"))


def _engine(profile, llm):
    return types.SimpleNamespace(target=profile, llm=llm, model=None)


def test_target_mismatches_refuse_a_mislabelled_execution(monkeypatch):
    dev = TargetProfile.default()
    monkeypatch.setattr(B.torch.cuda, "get_device_properties",
                        lambda i: types.SimpleNamespace(major=dev.compute_capability[0], minor=dev.compute_capability[1], multi_processor_count=dev.num_sms))
    monkeypatch.setattr(B, "observed_flash_attn_version", lambda model: 2)
    eager, compiled = dev, dev.with_overrides(compilation="piecewise")
    assert A.target_mismatches(_engine(eager, _llm(True, _CG.NONE))) == []
    assert A.target_mismatches(_engine(compiled, _llm(False))) == []
    bad = A.target_mismatches(_engine(eager, _llm(False)))
    assert bad and "declares eager execution" in bad[0] and "enforce_eager=False" in bad[0]
    bad = A.target_mismatches(_engine(compiled, _llm(True, _CG.NONE)))
    assert any("enforce_eager=True" in b for b in bad) and any("cudagraph_mode 'PIECEWISE'" in b and "'NONE'" in b for b in bad)
    bad = A.target_mismatches(_engine(compiled, _llm(False, _CG.FULL_AND_PIECEWISE)))
    assert bad == ["profile declares cudagraph_mode 'PIECEWISE', engine runs 'FULL_AND_PIECEWISE'"]
    # the emulated compiled engine is a DIFFERENT declared specialization (`piecewise-emu`): refused under the default `piecewise` profile
    bad = A.target_mismatches(_engine(compiled, _llm(False, emulate=True)))
    assert len(bad) == 1 and "emulate_precision_casts=False" in bad[0] and "compiled with True" in bad[0]
    assert A.target_mismatches(_engine(dev.with_overrides(compilation="piecewise-emu"), _llm(False, emulate=True))) == []


_WRAPPER = """# AOT ID: ['0_inference']
from torch._inductor.async_compile import AsyncCompile
async_compile = AsyncCompile()
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
import triton
@triton.jit
def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    tl.store(out_ptr0, tl.load(in_ptr0))
''', device_str='cuda')

def call(args):
    arg0, = args
    buf0 = empty_strided_cuda((8,), (1,), torch.bfloat16)
    triton_poi_fused_add_0.run(arg0, buf0, 8, XBLOCK=8, stream=stream0)
    torch.ops.verity.collect_c(buf0, 3)
    return (buf0,)
"""


def test_generated_kernels_pin_is_non_null_for_compiled_output_code_and_null_without(tmp_path, monkeypatch):
    """The record path pins the Inductor output code it compiled (kernel_set / call_set digests) -- positive
    evidence an eager run cannot produce -- and reports the absence of output code as an error with a null pin (never a silent placeholder)."""
    d = tmp_path / "inductor_cache" / "ab"
    d.mkdir(parents=True)
    (d / "cabcdef.py").write_text(_WRAPPER)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor_cache"))
    # a shared dir (torch's default, or any dir not marked per-process) is not this process's identity -> null pin, error of record
    monkeypatch.delenv(A._VERITY_INDUCTOR_DIR_MARK, raising=False)
    shared = A.generated_kernels_of_this_process()
    assert shared["wrappers"] == 1 and shared["kernel_set_sha256"] is None and "shared dir" in shared["error"]
    monkeypatch.setenv(A._VERITY_INDUCTOR_DIR_MARK, "1")
    gk = A.generated_kernels_of_this_process()
    assert gk["wrappers"] == 1 and gk["distinct_kernel_sources"] == 1 and "error" not in gk
    assert len(gk["kernel_set_sha256"]) == 64 and len(gk["call_set_sha256"]) == 64 and gk["per_process_cache_dir"] is True
    assert gk["pieces"][0]["extern_calls"] == ["torch.ops.verity.collect_c"] and gk["pieces"][0]["launches"] == 1
    assert list(gk["kernels"]) == ["triton_poi_fused_add_0"]
    # a second process compiling the same code gets the same pin regardless of where its cache dir is
    d2 = tmp_path / "other" / "zz"
    d2.mkdir(parents=True)
    (d2 / "c000000.py").write_text(_WRAPPER)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "other"))
    gk2 = A.generated_kernels_of_this_process()
    assert gk2["kernel_set_sha256"] == gk["kernel_set_sha256"] and gk2["call_set_sha256"] == gk["call_set_sha256"]
    # no output code -> null pin + error of record
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    gk3 = A.generated_kernels_of_this_process()
    assert gk3["wrappers"] == 0 and gk3["kernel_set_sha256"] is None and "no Inductor output code" in gk3["error"]


def test_prepare_compiled_process_cache_gives_a_fresh_marked_dir_and_disables_the_vllm_cache(monkeypatch):
    monkeypatch.delenv(A._VERITY_INDUCTOR_DIR_MARK, raising=False)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_root")
    monkeypatch.delenv("VLLM_DISABLE_COMPILE_CACHE", raising=False)
    out = A.prepare_compiled_process_cache()
    import os
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == out["inductor_cache_dir"] != "/tmp/torchinductor_root"
    assert os.path.isdir(out["inductor_cache_dir"]) and not os.listdir(out["inductor_cache_dir"])
    assert os.environ[A._VERITY_INDUCTOR_DIR_MARK] == "1" and os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    # an explicitly marked dir (compiled_commit.sh) is kept
    again = A.prepare_compiled_process_cache()
    assert again["inductor_cache_dir"] == out["inductor_cache_dir"]


@pytest.mark.parametrize("resolved,expected", [(True, True), (False, False), (None, None)])
def test_async_scheduling_is_the_resolved_scheduler_value_not_the_cli_default(resolved, expected):
    """vLLM's CLI default is None and is resolved at VllmConfig.__post_init__; the header attests
    the resolved bool read back from scheduler_config (None = unattested, distinct from False), and an engine without the attribute is None."""
    sc = types.SimpleNamespace(async_scheduling=resolved)
    llm = types.SimpleNamespace(llm_engine=types.SimpleNamespace(vllm_config=types.SimpleNamespace(scheduler_config=sc)))
    assert A.resolved_async_scheduling(llm) is expected
    assert A.resolved_async_scheduling(types.SimpleNamespace()) is None
    assert A.resolved_async_scheduling(types.SimpleNamespace(llm_engine=types.SimpleNamespace(vllm_config=types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace())))) is None
