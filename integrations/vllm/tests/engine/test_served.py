"""`engine.served`: what sets each kernel-selecting switch of a served engine (vLLM's own config among them), the verdict on what a
serving process reads back, the read-back itself, that every workload file gives a reason for each switch it moves away from vLLM's
default, and that a workload's Build and its engine run under one batch invariance."""
from __future__ import annotations

import enum
import json
import sys
import types
from types import SimpleNamespace

import pytest

from verity_vllm.config import WORKLOADS, RowId, row_execution, workload_files
from verity_vllm.engine import env as engine_env
from verity_vllm.engine import served as SV
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.pipeline import row_records
from verity_vllm.program.frontend.target_profile import TargetProfile

GLM = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
#: The switches GLM-5.3's Commit of record ran under (art:4b274f2088e57bbb01800948211a04c35b346b973dafa7214a090a202ed5b886,
#: versions.json: ``env`` VLLM_BATCH_INVARIANT=0, VLLM_USE_V2_MODEL_RUNNER=1, VLLM_USE_FLASHINFER_SAMPLER=0,
#: VLLM_DISABLE_SHARED_EXPERTS_STREAM=1, no VLLM_USE_DEEP_GEMM or NCCL_P2P_DISABLE; ``engine_kwargs`` enforce_eager, moe_backend
#: triton, disable_custom_all_reduce false; its target's attention backend and FlashAttention version vLLM's own).
GLM_SERVED = {"VLLM_BATCH_INVARIANT": False, "VLLM_USE_DEEP_GEMM": True, "VLLM_DISABLE_SHARED_EXPERTS_STREAM": True,
              "VLLM_USE_FLASHINFER_SAMPLER": False, "VLLM_USE_V2_MODEL_RUNNER": True, "NCCL_P2P_DISABLE": None,
              "disable_custom_all_reduce": False, "enforce_eager": True, "compilation_mode": "NONE", "cudagraph_mode": "NONE",
              "attention_backend": None, "flash_attn_version": None, "moe_backend": "triton"}
LLAMA_BI = "llama32-1b__bf16__h200__tp1__b1__i256__o32__mixed__greedy__bi-eager"
#: What the one rank of LLAMA_BI's first Commit read back (art:6bdd8171952edfb0705e4328bcb2aba9880fd400b3c01fdc4042a614ed52d24a,
#: summary.json ``served.ranks``): vLLM's batch invariance had turned the custom all-reduce off.
LLAMA_BI_SERVED = {**GLM_SERVED, "VLLM_BATCH_INVARIANT": True, "disable_custom_all_reduce": True, "moe_backend": "auto"}


def _is_row(stem: str) -> bool:
    try:
        return str(RowId.parse(stem)) == stem
    except ValueError:
        return False


ROWS = [p for p in workload_files() if _is_row(p.stem)]


def _glm() -> tuple[dict, object]:
    wl = va.load_workload(WORKLOADS / f"{GLM}.json")
    return wl, va.target_of_workload(wl)


def test_an_execution_label_sets_batch_invariance_and_an_unknown_one_is_refused():
    assert SV.execution_env("eager") == SV.execution_env("compiled") == {"VLLM_BATCH_INVARIANT": "0"}
    assert SV.execution_env("bi-eager") == SV.execution_env("bi-eager-fa2") == SV.execution_env(None) == {"VLLM_BATCH_INVARIANT": "1"}
    assert SV.ENFORCE_EAGER == {"eager": True, "compiled": False, "bi-eager": True, "bi-eager-fa2": True, "bi-compiled": False}
    with pytest.raises(ValueError, match="'fp4' is not modelled by the harness"):
        SV.execution_env("fp4")


def test_a_compiled_row_declares_only_vllms_serving_defaults():
    """`compiled` is vLLM's serving defaults: whatever CUDA-graph mode vLLM resolves passes, and batch invariance is off."""
    d = SV.declared({"execution": "compiled"})
    assert {k: v["value"] for k, v in d.items() if v["source"] == "execution compiled"} == {"enforce_eager": False}
    served = {**{k: v for k, v in SV.DEFAULTS.items() if v != SV.RESOLVED}, "VLLM_USE_V2_MODEL_RUNNER": True,
              "VLLM_USE_FLASHINFER_SAMPLER": False, "VLLM_DISABLE_SHARED_EXPERTS_STREAM": True,
              "compilation_mode": "VLLM_COMPILE", "cudagraph_mode": "FULL_AND_PIECEWISE"}
    assert SV.judge(served, d) == []
    assert SV.judge({**served, "VLLM_BATCH_INVARIANT": True}, d) == [
        "VLLM_BATCH_INVARIANT=True is not vLLM's default False and nothing declares it"]
    assert SV.judge({**served, "enforce_eager": True}, d) == ["enforce_eager=True where execution compiled declares False"]


def test_the_engine_profile_sets_no_switch_but_with_its_reason():
    assert "VLLM_BATCH_INVARIANT" not in engine_env.ENGINE, "batch invariance is the workload's execution label's"
    assert "VLLM_BATCH_INVARIANT" not in engine_env.EXPORT, "a Build's batch invariance is its target's"
    assert set(engine_env.REASONS) == set(engine_env.ENGINE) and all(engine_env.REASONS.values())


@pytest.mark.parametrize("path", ROWS, ids=lambda p: p.stem)
def test_a_rows_build_runs_under_its_execution_labels_batch_invariance(path):
    """The Build applies its target's `env()` (`build.derive_step`; no target: `TargetProfile.default()` or the host's, both
    batch-invariant), the engine its execution label's (`served.apply`): one row, one batch invariance."""
    declared = row_records.declared_target(str(path), row=path.stem)
    target = TargetProfile.from_json(json.loads(declared)) if declared else TargetProfile.default()
    label = SV.execution_env(row_execution(path.stem))
    assert {k: target.env()[k] for k in label} == label


def test_glm53s_row_declares_each_switch_with_its_source_and_reason():
    wl, target = _glm()
    d = SV.declared(wl, target)
    assert {k: (v["value"], v["source"]) for k, v in d.items()} == {
        "VLLM_USE_V2_MODEL_RUNNER": (True, "engine.env.ENGINE"), "VLLM_USE_FLASHINFER_SAMPLER": (False, "engine.env.ENGINE"),
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM": (True, "engine.env.ENGINE"), "enforce_eager": (True, "execution eager"),
        "compilation_mode": ("NONE", "execution eager"), "cudagraph_mode": ("NONE", "execution eager"),
        "disable_custom_all_reduce": (False, "row file"), "VLLM_BATCH_INVARIANT": (False, "row file"), "moe_backend": ("triton", "row file")}
    assert d["moe_backend"]["reason"].startswith("vLLM's own selection for GLM-5.3's block-FP8 experts")
    assert d["VLLM_DISABLE_SHARED_EXPERTS_STREAM"]["reason"] == engine_env.REASONS["VLLM_DISABLE_SHARED_EXPERTS_STREAM"]
    assert d["disable_custom_all_reduce"]["reason"] is not None, "the transport's reason is kept where the row file agrees with it"


def test_glm53s_commit_of_record_passes_with_its_reasons_and_fails_on_the_one_it_lacked():
    """The rail run over GLM-5.3's Commit of record: everything is vLLM's default or declared with a reason but its MoE backend, which
    its target declared with none until the workload file gave one."""
    wl, target = _glm()
    assert SV.judge(GLM_SERVED, SV.declared(wl, target)) == []
    bare = {**wl, "engine": {**wl["engine"], "reasons": {}}}
    assert SV.judge(GLM_SERVED, SV.declared(bare, target)) == [
        "moe_backend='triton' (row file) is not vLLM's default 'auto' and has no reason (the row file's engine.reasons)"]


def test_vllms_batch_invariance_turns_the_custom_all_reduce_off_over_the_row_file():
    """`ParallelConfig._verify_args` sets ``disable_custom_all_reduce`` under ``VLLM_BATCH_INVARIANT`` after every argument, so a
    batch-invariant workload declares it whatever its workload file says, and a workload that is not batch invariant still may not carry it."""
    wl = va.load_workload(WORKLOADS / f"{LLAMA_BI}.json")
    d = SV.declared(wl, va.target_of_workload(wl))
    assert d["disable_custom_all_reduce"] == {"value": True, "source": "vLLM ParallelConfig", "reason": SV.BATCH_INVARIANT[1]}
    assert SV.judge(LLAMA_BI_SERVED, d) == []
    told_on = {**wl, "engine": {**wl["engine"], "args": {**wl["engine"]["args"], "disable_custom_all_reduce": False}}}
    assert SV.declared(told_on, va.target_of_workload(wl))["disable_custom_all_reduce"]["value"] is True
    assert SV.judge({**GLM_SERVED, "disable_custom_all_reduce": True}, SV.declared(*_glm())) == [
        "disable_custom_all_reduce=True where row file declares False"]


@pytest.mark.parametrize("p", ROWS, ids=lambda p: p.stem)
def test_every_row_gives_a_reason_for_each_switch_it_moves_from_vllms_default(p):
    wl = va.load_workload(p)
    assert set(wl["engine"].get("reasons") or {}) <= set(SV.DEFAULTS)
    unreasoned = {k: v for k, v in SV.declared(wl, va.target_of_workload(wl)).items() if v["value"] != SV.DEFAULTS[k] and not v["reason"]}
    assert unreasoned == {}


@pytest.mark.parametrize("switches, problem", [
    ({"moe_backend": "cutlass"}, "moe_backend='cutlass' where row file declares 'triton'"),
    ({"attention_backend": "FLASHINFER"}, "attention_backend='FLASHINFER' is not vLLM's default None and nothing declares it"),
    ({"VLLM_USE_DEEP_GEMM": False}, "VLLM_USE_DEEP_GEMM=False is not vLLM's default True and nothing declares it"),
    ({"NCCL_P2P_DISABLE": "1"}, "NCCL_P2P_DISABLE='1' is not vLLM's default None and nothing declares it"),
])
def test_a_switch_away_from_what_is_declared_or_vllms_default_is_a_problem(switches, problem):
    wl, target = _glm()
    assert SV.judge({**GLM_SERVED, **switches}, SV.declared(wl, target)) == [problem]


def test_a_switch_not_read_back_is_a_problem_and_one_vllm_resolves_may_be_anything_undeclared():
    resolved = {k: v for k, v in SV.DEFAULTS.items() if v != SV.RESOLVED}
    assert SV.judge({k: v for k, v in GLM_SERVED.items() if k != "cudagraph_mode"}, SV.declared(*_glm())) == ["cudagraph_mode: not read back"]
    assert SV.judge({**resolved, "VLLM_USE_V2_MODEL_RUNNER": False, "compilation_mode": "VLLM_COMPILE", "cudagraph_mode": "FULL"}, {}) == []


def test_a_variable_that_selects_no_kernel_is_no_switch():
    """VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS is the multiproc executor's RPC deadline (vllm/v1/executor/multiproc_executor.py): set in a
    row file or carried by a rank, the rail neither declares nor judges it."""
    k = "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
    wl, target = _glm()
    wl = {**wl, "engine": {**wl["engine"], "env": {**(wl["engine"].get("env") or {}), k: "1800"}}}
    d = SV.declared(wl, target)
    assert k not in SV.DEFAULTS and k not in d
    assert SV.judge({**GLM_SERVED, k: 1800}, d) == []


def test_a_record_with_no_rank_is_not_ok():
    rec = SV.record([], {})
    assert (rec["ok"], rec["problems"], rec["schema"]) == (False, ["no rank answered"], SV.SCHEMA)


class _Mode(enum.IntEnum):
    NONE = 0


class _Graphs(enum.Enum):
    NONE = 0


class _Backend(enum.Enum):
    FLASH_ATTN = 1


@pytest.fixture
def vllm_stub(monkeypatch):
    """`vllm.envs` and `vllm.distributed` as a serving process's: the flags as vLLM parses them, rank 3."""
    envs = types.ModuleType("vllm.envs")
    envs.__dict__.update(VLLM_BATCH_INVARIANT=False, VLLM_USE_DEEP_GEMM=True, VLLM_DISABLE_SHARED_EXPERTS_STREAM=True,
                         VLLM_USE_FLASHINFER_SAMPLER=False)
    dist = types.ModuleType("vllm.distributed")
    dist.get_tensor_model_parallel_rank = lambda: 3
    pkg = types.ModuleType("vllm")
    pkg.envs, pkg.distributed = envs, dist
    monkeypatch.setitem(sys.modules, "vllm", pkg)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    monkeypatch.setitem(sys.modules, "vllm.distributed", dist)
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")


def test_a_rank_reads_back_each_switch_as_vllm_resolved_it(vllm_stub):
    vc = SimpleNamespace(use_v2_model_runner=True, parallel_config=SimpleNamespace(disable_custom_all_reduce=True),
                         model_config=SimpleNamespace(enforce_eager=True),
                         compilation_config=SimpleNamespace(mode=_Mode.NONE, cudagraph_mode=_Graphs.NONE),
                         attention_config=SimpleNamespace(backend=_Backend.FLASH_ATTN, flash_attn_version=3),
                         kernel_config=SimpleNamespace(moe_backend="triton"))
    got = SV.read_rank(SimpleNamespace(vllm_config=vc))
    assert got == {"rank": 3, "switches": {**GLM_SERVED, "NCCL_P2P_DISABLE": "1", "disable_custom_all_reduce": True,
                                           "attention_backend": "FLASH_ATTN", "flash_attn_version": 3}}


def test_require_fails_the_commit_closed_and_says_why(tmp_path):
    wl, target = _glm()
    llm = SimpleNamespace(collective_rpc=lambda fn, timeout=None: [{"rank": r, "switches": {**GLM_SERVED, "enforce_eager": r == 0}}
                                                                   for r in range(2)])
    with pytest.raises(SystemExit, match="fail-closed: served switches: rank 1: enforce_eager=False where execution eager declares True"):
        SV.require(llm, wl, target, tmp_path)
    sm = json.loads((tmp_path / "summary.json").read_text())
    assert sm["pass"] is False and SV.of_summary(sm)["ranks"][0]["rank"] == 0
    ok = SimpleNamespace(collective_rpc=lambda fn, timeout=None: [{"rank": r, "switches": GLM_SERVED} for r in range(8)])
    assert SV.require(ok, wl, target, tmp_path / "unwritten")["ok"] is True
