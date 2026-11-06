"""Records a reader fills from the environment, pinned at their bytes on `main` before the reader moves to `verity_vllm.settings`, so
the move can't change them: the hot worker's key (`hot.env_identity`, `hot.so_identity`), the fast Match's declared configuration
(`global_match_fast.config`, on every result document's impl block), the Commit's reasons to run cold with the staging
environment it hands its committers (`commit.hot_unsafe_reasons`, `commit.staging_environ`), the configuration a committer
takes from that environment (the collector's on `test_staging_defaults`' CPU fakes, its name among it), and the fields of the
engine's records the environment fills (the replay's engine facts, the engine profile, the compilation facts, the generated kernels)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from collections.abc import Iterator
from unittest import mock

import pytest

from verity_vllm.check.match import global_match as GM
from verity_vllm.check.match import global_match_fast as GMF
from verity_vllm.pipeline import cli, hot


def _digest(doc: object) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:32]


@pytest.fixture
def env() -> Iterator[pytest.MonkeyPatch]:
    """An empty environment, the process's restored whole after the test: the code under test writes settings straight into
    `os.environ` (`settings.put`, as `commit.staging_environ` does), which no `MonkeyPatch` records.  The test's own `MonkeyPatch`
    is undone inside that, since one undone after it (the function's `monkeypatch`) would delete what a test set that the process
    holds too (PATH, HOME)."""
    with mock.patch.dict(os.environ, clear=True), pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def commit():
    """`pipeline.commit`, imported before `env` clears the environment: the import pins the engine's variables (`engine.env`)."""
    pytest.importorskip("torch")
    from verity_vllm.pipeline import commit

    return commit


@pytest.fixture(scope="module")
def committers():
    pytest.importorskip("torch")
    from tests.commit import test_staging_defaults
    from verity_vllm.commit.committer import native_host

    return test_staging_defaults, native_host


def test_the_hot_key_takes_the_environment_as_before(env: pytest.MonkeyPatch) -> None:
    for k, v in {"PATH": "/bin", "PYTHONPATH": "/src", "VLLM_BATCH_INVARIANT": "1", "VERITY_WINDOW_MB": "256", "VERITY_TIMELINE": "/t",
                 "VERITY_HOT_SUBMIT": "1", "HASH_THREADS": "4", "CUDA_HOME": "/cuda", "TORCH_X": "", "OMP_NUM_THREADS": "8",
                 "NATIVE_COLLECT_BUILD": "/no-such-build", "HIDDEN_GPU_BUILD": "", "HOME": "/h", "NCCL_P2P_DISABLE": "1"}.items():
        env.setenv(k, v)
    assert _digest([hot.env_identity(), hot.so_identity([])]) == "0019aeeec8c143005b7a8b4b19531a0c"


@pytest.mark.parametrize(("values", "digest"), [
    ({"MATCH_WORKERS": "1"}, "37c433ce7a6f3d0ef54af75f49134685"),
    ({"MATCH_WORKERS": " 3 ", "MATCH_COMPACT": " 0 "}, "aee01761c9e2b4dd6f984669223ef09f"),
    ({"MATCH_WORKERS": "2", "MATCH_COMPACT": "false"}, "07dfbb9661bbc2f54bfcceb284cf36ab"),
    ({"MATCH_WORKERS": "1", "MATCH_COMPACT": ""}, "37c433ce7a6f3d0ef54af75f49134685"),
])
def test_the_fast_match_declares_its_configuration_as_before(env: pytest.MonkeyPatch, values: dict[str, str], digest: str) -> None:
    for k, v in values.items():
        env.setenv(k, v)
    GMF.ops()
    config = {k: v for k, v in GMF.config.items() if k != "cpu_count"}
    assert _digest([config, GM.impl_banner(GMF)]) == digest


@pytest.mark.parametrize(("values", "argv", "digest"), [
    ({}, [], "28285790244bd016269ddf08993ba559"),
    ({"VERITY_FAULT": " skip_copy:1 ", "VERITY_SWITCH": "0.001"}, ["--arms", "a,b", "--window-mb", "0"], "e7f7725d89caf134f7ea831181381df2"),
    ({"VERITY_FAULT": "  ", "VERITY_SWITCH": ""}, ["--bounded-staging", "--retain", "host", "--retain-exclude", "fa3_hidden_m1",
                                                   "--learn-budget-mb", "1024", "--window-mb", "256", "--window-slots", "4"],
     "ce0b116482d3de00d51cf1c1a00a940f"),
    ({"VERITY_FAULT": "", "VERITY_RETAIN": "device"}, ["--no-bounded-staging", "--retain", "none"], "ab1c19f50687282fe9ef2e4012560ec4"),
])
def test_the_commit_names_its_cold_reasons_and_stages_as_before(commit, env: pytest.MonkeyPatch, values: dict[str, str],
                                                                 argv: list[str], digest: str) -> None:
    for k, v in values.items():
        env.setenv(k, v)
    ns = cli.parse("commit", ["--out", "o", *argv], {})
    reasons = commit.hot_unsafe_reasons(ns)
    commit.staging_environ(commit.resolve_staging(ns))
    assert _digest([reasons, {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}]) == digest


@pytest.mark.parametrize(("retain", "values", "digest"), [
    ("none", {}, "ff0b0f64489f1687c56406d80f95c9b3"),
    ("host", {"VERITY_STAGING_BOUNDED": " yes ", "VERITY_HI_PRIO": "", "VERITY_RETAIN_PINNED": "false", "VERITY_INLINE_D2H": "1",
              "VERITY_PIN": "true", "VERITY_FAULT": " skip_copy:3", "VERITY_WINDOW_MB": " 16 ", "VERITY_WINDOW_SLOTS": "3",
              "NATIVE_GATHER_BLOCKS": "64", "NATIVE_GATHER_AUTO_BYTES": "4096", "VERITY_CONSUMER_DELAY_MS": "2.5",
              "VERITY_LEARN_HOST_BUDGET_MB": "1024", "VERITY_RETAIN_EXCLUDE": "fa3_hidden_m1", "VERITY_SKIP": "a,,b",
              "VERITY_WEIGHTS_HASH": "device"}, "1eb12e9bd19c559e217418921c4074b6"),
    ("none", {"VERITY_STAGING_BOUNDED": "on", "VERITY_FAULT": "skip_copy:2", "VERITY_HI_PRIO": "1", "VERITY_RETAIN_PINNED": " no",
              "VERITY_INLINE_D2H": "true"}, "cbc856bbf4739c6b70ec3bed50df6bbd"),
])
def test_a_committer_takes_its_configuration_as_before(committers, env: pytest.MonkeyPatch, retain: str, values: dict[str, str],
                                                       digest: str) -> None:
    from verity_vllm import settings

    staging, native_host = committers
    com = staging.committer(env, retain, **values)
    keys = ("win_bytes", "win_slots_req", "bounded", "learn_budget_bytes", "hi_prio", "consumer_delay_s", "fault", "retain", "retain_pinned",
            "inline_d2h", "pin", "name", "windowed")
    nc = com._nc
    doc = {k: getattr(com, k) for k in keys} | {
        "retain_exclude": sorted(com.retain_exclude), "footprint": com.footprint is not None,
        "blocks_per_desc": getattr(nc, "blocks_per_desc", None), "auto_bytes_per_block": getattr(nc, "auto_bytes_per_block", None),
        "fault_skip_copy_at": getattr(nc, "fault_skip_copy_at", None), "skip": sorted(native_host.skip_classes_of_env()),
        "weights_hash": settings.VERITY_WEIGHTS_HASH}
    assert _digest(doc) == digest


@pytest.fixture(scope="module")
def engine():
    pytest.importorskip("torch")
    from verity_vllm.acquire.sources import compiled_source
    from verity_vllm.check import fold_compare
    from verity_vllm.engine import kernel_identity
    from verity_vllm.check.replay import driver
    from verity_vllm.engine import build, code_identity, engine_profile
    from verity_vllm.engine.profiles import dense_generic

    return types.SimpleNamespace(compiled_source=compiled_source, fold_compare=fold_compare, kernel_identity=kernel_identity, driver=driver,
                                 build=build, code_identity=code_identity, engine_profile=engine_profile, dense_generic=dense_generic)


_ENGINE_VARS = ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE", "NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS",
                "NCCL_MAX_NCHANNELS", "NCCL_NTHREADS", "NCCL_BUFFSIZE", "NCCL_LAUNCH_MODE", "VLLM_DISABLE_COMPILE_CACHE", "RUNPOD_POD_ID",
                "TORCHINDUCTOR_CACHE_DIR", "VERITY_INDUCTOR_DIR_PER_PROCESS", "HF_HOME", "VERITY_FAULT")


@pytest.mark.parametrize(("values", "digest"), [
    ({}, "d43ddca387e48b9f037005834832ba17"),
    ({"VLLM_BATCH_INVARIANT": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUBLASLT_WORKSPACE_SIZE": "0", "NCCL_ALGO": "Ring",
      "NCCL_PROTO": "Simple", "NCCL_MIN_NCHANNELS": "4", "NCCL_MAX_NCHANNELS": "8", "NCCL_NTHREADS": "512", "NCCL_BUFFSIZE": "4194304",
      "NCCL_LAUNCH_MODE": "GROUP", "VLLM_DISABLE_COMPILE_CACHE": "1", "RUNPOD_POD_ID": "pod-a", "TORCHINDUCTOR_CACHE_DIR": "/no/inductor",
      "VERITY_INDUCTOR_DIR_PER_PROCESS": "1", "HF_HOME": "/hf", "VERITY_FAULT": " skip_copy:3 "}, "6cbbab0e6f0be08fca8b082a0bd226cc"),
    (dict.fromkeys(_ENGINE_VARS, ""), "b32b5b554142c2adc24aa3c97868d7ea"),
    ({"VLLM_BATCH_INVARIANT": " 1", "VLLM_DISABLE_COMPILE_CACHE": "true", "VERITY_INDUCTOR_DIR_PER_PROCESS": "true", "NCCL_ALGO": " Tree ",
      "RUNPOD_POD_ID": " p "}, "346629f5207b0967e546ab33645f6704"),
])
def test_an_engine_record_takes_the_environment_as_before(engine, env: pytest.MonkeyPatch, values: dict[str, str], digest: str) -> None:
    """The fields of the engine's records that the environment fills: the replay's engine facts, the engine profile's linear backend,
    NCCL settings and pod, the compilation facts' compile cache, the compiled source's fault, the generated kernels' Inductor
    directory, and the hub paths.  vLLM, CUDA and nvidia-smi are faked; the rest of each record is the machine's."""
    import torch

    for k, v in {"HOME": "/h", **values}.items():
        env.setenv(k, v)
    ns = types.SimpleNamespace
    runner = ns(get_model=lambda: ns(named_modules=lambda: []), sampler=None)
    comm = ns(ca_comm=None, symm_mem_comm=None, qr_comm=None, fi_allreduce_fusion=None, pynccl_comm=None)
    ps = types.ModuleType("vllm.distributed.parallel_state")
    ps.get_tp_group = lambda: ns(rank_in_group=0, world_size=2, device_communicator=comm)
    dist = types.ModuleType("vllm.distributed")
    dist.parallel_state = ps
    for name, mod in {"vllm": types.ModuleType("vllm"), "vllm.distributed": dist, "vllm.distributed.parallel_state": ps}.items():
        env.setitem(sys.modules, name, mod)
    env.setattr(torch.cuda, "get_device_properties", lambda i: ns(name="F", major=8, minor=9, total_memory=1 << 30, multi_processor_count=8))
    env.setattr(torch.cuda, "device_count", lambda: 1)
    env.setattr(engine.engine_profile, "_run", lambda argv: "1")
    env.setattr(engine.kernel_identity, "inductor_cache_dir", lambda: "/no/torch-default")
    cc = ns(mode=3, cudagraph_mode=None, cudagraph_capture_sizes=[1], max_cudagraph_capture_size=1, use_inductor_graph_partition=False,
            backend="inductor", inductor_compile_config={})
    llm = ns(llm_engine=ns(vllm_config=ns(compilation_config=cc, model_config=ns(enforce_eager=False))))
    facts = engine.driver.engine_facts_of_record()
    gk = engine.code_identity.generated_kernels_of_this_process()
    pod = engine.engine_profile.gpu_info()["pod"]
    doc = {"facts": [facts[k] for k in ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE")] + [facts["cublas"]["env"]],
           "linear": engine.engine_profile._backends_of_runner(runner)["linear"],
           "nccl_env": engine.engine_profile._backends_probe(ns(model_runner=runner))["all_reduce"]["nccl_env"],
           "pod": "<node>" if pod == os.uname().nodename else pod,
           "compile_cache": engine.build.compilation_facts(llm)["compile_cache"],
           "compile_cache_disabled": engine.compiled_source.compile_settings(llm)["compile_cache_disabled"],
           "fault": engine.compiled_source.CompiledGraphSource().fault,
           "inductor": [gk["inductor_cache_dir"], gk["per_process_cache_dir"]],
           "hub": [str(engine.dense_generic.hf_snapshot("org/m", "abc")), engine.fold_compare.hf_safetensors_glob("org/m")]}
    assert _digest(doc) == digest, doc


@pytest.fixture(scope="module")
def build_target():
    pytest.importorskip("torch")
    from verity_vllm.program.frontend import export_compat, vllm_meta

    return export_compat, vllm_meta


_BUILD_VARS = ("VLLM_BATCH_INVARIANT", "VLLM_USE_LAYERNAME", "HF_HUB_OFFLINE", "VLLM_LOGGING_LEVEL", "TOKENIZERS_PARALLELISM",
               "VLLM_ATTENTION_BACKEND", "VLLM_USE_DEEP_GEMM", "CUDA_VISIBLE_DEVICES")


@pytest.mark.parametrize(("values", "digest"), [
    ({}, "4be0a8acddaeaa879c13009a998f19d6"),
    ({"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_LAYERNAME": "0", "HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "WARNING",
      "TOKENIZERS_PARALLELISM": "false", "VLLM_ATTENTION_BACKEND": "TRITON_ATTN", "VLLM_USE_DEEP_GEMM": "0", "CUDA_VISIBLE_DEVICES": "0,1"},
     "8b88316c51537deda318b282d9873e03"),
    (dict.fromkeys(_BUILD_VARS, ""), "92678ab9dfe7c37a7ae9002006956c51"),
    ({"VLLM_BATCH_INVARIANT": " 1", "VLLM_ATTENTION_BACKEND": "", "TOKENIZERS_PARALLELISM": "true", "CUDA_VISIBLE_DEVICES": " "},
     "b41e0b68c73454c37d4cc0cbd7cf4e5e"),
])
def test_a_builds_target_facts_take_the_environment_as_before(build_target, env: pytest.MonkeyPatch, values: dict[str, str],
                                                              digest: str) -> None:
    """The Build's target facts the environment fills: the variables it set that the pinned vLLM consults, ignores or doesn't
    own (`vllm_meta.consulted_facts`), and the host's CUDA visibility (`export_compat.host_facts`).  vLLM's ``envs``, the host's
    name and CUDA are faked."""
    import socket

    import torch

    export_compat, vllm_meta = build_target
    for k, v in values.items():
        env.setenv(k, v)
    envs = types.ModuleType("vllm.envs")
    envs.environment_variables = dict.fromkeys(("VLLM_BATCH_INVARIANT", "VLLM_USE_LAYERNAME", "VLLM_LOGGING_LEVEL", "VLLM_USE_DEEP_GEMM"))
    vllm = types.ModuleType("vllm")
    vllm.envs = envs
    for name, mod in {"vllm": vllm, "vllm.envs": envs}.items():
        env.setitem(sys.modules, name, mod)
    env.setattr(socket, "gethostname", lambda: "host-a")
    env.setattr(torch.cuda, "is_available", lambda: False)
    doc = {"consulted": vllm_meta.consulted_facts(types.SimpleNamespace()), "host": export_compat.host_facts()}
    assert _digest(doc) == digest, doc
