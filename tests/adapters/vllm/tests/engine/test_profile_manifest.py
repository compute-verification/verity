"""`verity_vllm.engine.run_facts.profile_manifest` and `verity_vllm.engine.engine_profile.build_manifest` on stand-ins of a
built engine: the manifest a run's profile id is taken from, key for key and in its order, the id, the notes, and the
fallback document where the profile cannot be built.  No engine and no device: the readers of the device, the software
and the engine's config are stand-ins here; `software_info` is read on a stand-in vLLM.  Every expectation is what the two functions gave before a line of them was
changed, so an id pinned here moves only if the manifest's bytes do."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch

from tests import fake_vllm
from verity_vllm.engine import build, engine_profile, run_facts

REAL_BUILD_MANIFEST = engine_profile.build_manifest
CP: dict[str, Any] = {"repo": "org/model", "revision": "0" * 40, "safetensors_sha256": {"model.safetensors": "ab" * 32},
                      "config_sha256": "cd" * 32, "local_path": "/nonexistent/snapshot"}
FLAGS = {"VLLM_BATCH_INVARIANT": "1"}
GREEDY: dict[str, Any] = {"greedy": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seeds": [], "logprobs": None}
NO_KERNELS = {"wrappers": 0, "distinct_kernel_sources": 0, "kernel_set_sha256": None, "call_set_sha256": None,
              "ptx_code_set_sha256": None, "ptx_variants": 0}
#: two texts of the manifest and its notes hold a word the glossary retires, so they are pinned by their sha256 and not written out
#: here: `adapter.interception`, which is in the profile id, and the notes' `profile_id_method` (273 characters, one string)
INTERCEPTION_SHA256 = "097b30e8214d648efe4d27849222c362cd127368204b4604ab8a4ec5fa824c7b"
METHOD_SHA256 = "43ad3e91103a24f32028236c485f4df25f2a9102b6cd530dbe90860f5ba67965"


@pytest.fixture
def built(monkeypatch) -> list[dict[str, Any]]:
    """`build_manifest` as a stand-in that records its arguments; the host's and the engine's own readers fixed."""
    calls: list[dict[str, Any]] = []

    def build_manifest(**kw: Any) -> dict[str, Any]:
        calls.append(kw)
        return {"schema": "verity-vllm/profile/v1", "case": kw["case"], "dtype": "bfloat16", "backends": {"attention": "A"},
                "compilation": {"mode": "NONE", "enforce_eager": True}, "engine_kwargs": {"seed": 0}, "sampling": kw["sampling"]}

    monkeypatch.setattr(engine_profile, "build_manifest", build_manifest)
    monkeypatch.setattr(run_facts, "vllm_has_flashinfer", lambda: None)
    monkeypatch.setattr(run_facts, "compilation_facts", lambda llm: {"enforce_eager": True, "mode": "NONE"})
    return calls


def _engine(kwargs: dict[str, Any] | None = None, *, target: Any = None, model: Any = None) -> build.Engine:
    llm = NS(llm_engine=NS(vllm_config=NS(scheduler_config=NS(async_scheduling=True))))
    return build.Engine(llm=llm, runner=None, model=torch.nn.Linear(2, 2) if model is None else model, cp=CP,
                        kwargs=dict(kwargs or {}), env_flags=dict(FLAGS), build_s=0.0, case="B0", target=target)


def _manifest(engine: build.Engine, sampling: dict[str, Any] | None = None) -> tuple[dict, str, dict]:
    return run_facts.profile_manifest(engine, adapter_revision="rev", wheel_sha256="wheel", vllm_commit="commit", sampling=sampling or {})


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _eager(engine_kwargs: dict[str, Any], **top: Any) -> dict[str, Any]:
    """The manifest of an eager engine with no target, in the order the function writes it."""
    doc: dict[str, Any] = {"schema": "verity-vllm/profile/v1", "case": "B0", "dtype": "bfloat16",
                           "backends": {"attention": "A", "attention_flash_attn_version": None, "vllm_has_flashinfer": None},
                           "compilation": {"mode": "NONE", "enforce_eager": True, "generated_kernels": NO_KERNELS},
                           "engine_kwargs": engine_kwargs, "sampling": GREEDY}
    return {**doc, **top}


def test_an_eager_engine_with_no_target_gives_the_built_manifest_with_the_harness_facts_and_its_id(built):
    engine = _engine({"max_num_seqs": 8})
    manifest, pid, notes = _manifest(engine)
    want = _eager({"seed": 0, "enable_prefix_caching": False, "max_num_seqs": 8, "kernel_config": {}, "enforce_eager": False,
                   "async_scheduling": True})
    assert json.dumps(manifest) == json.dumps(want)
    assert pid == "a439413f8a5a" and pid == engine_profile.profile_id_of(want)
    assert _sha256(notes.pop("profile_id_method")) == METHOD_SHA256
    assert notes == {"case": "B0", "accepted_profile_id": run_facts.ACCEPTED_PROFILE_ID, "accepted_record": run_facts.ACCEPTED_RECORD}
    assert _sha256(built[0].pop("interception")) == INTERCEPTION_SHA256
    assert built == [{"case": "B0", "cp": CP, "llm": engine.llm, "wheel_sha256": "wheel", "vllm_commit": "commit",
                      "adapter_revision": "rev", "sampling": GREEDY, "env_flags": FLAGS, "enforce_eager": False,
                      "tensor_parallel_size": 1}]


@pytest.mark.parametrize(("kwargs", "want"), [({}, {}), ({"kernel_config": None}, {}), ({"kernel_config": {}}, {}),
                                              ({"kernel_config": {"gemm": "tuned", "tiles": [1, 2]}}, {"gemm": "tuned", "tiles": [1, 2]})])
def test_the_kernel_config_is_a_copy_of_the_engines_and_empty_where_it_has_none(built, kwargs, want):
    engine = _engine(kwargs)
    manifest, _, _ = _manifest(engine)
    assert manifest["engine_kwargs"]["kernel_config"] == want
    assert manifest["engine_kwargs"]["kernel_config"] is not engine.kwargs.get("kernel_config")
    assert list(manifest["engine_kwargs"]) == ["seed", "enable_prefix_caching", "max_num_seqs", "kernel_config", "enforce_eager",
                                               "async_scheduling"]


def test_what_the_engine_was_built_with_is_what_the_manifest_says(built):
    kwargs = {"dtype": "float16", "max_num_batched_tokens": 512, "compilation_config": {"custom_ops": ["all"]}, "max_num_seqs": 4,
              "enable_prefix_caching": True, "enforce_eager": True, "tensor_parallel_size": 2, "async_scheduling": False}
    manifest, pid, notes = _manifest(_engine(kwargs))
    want = _eager({"seed": 0, "enable_prefix_caching": True, "max_num_seqs": 4, "max_num_batched_tokens": 512, "dtype": "float16",
                   "kernel_config": {}, "enforce_eager": True, "compilation_config": {"custom_ops": ["all"]}, "async_scheduling": True},
                  dtype="float16")
    assert json.dumps(manifest) == json.dumps(want)
    assert pid == "f10026c06534"
    assert notes["async_scheduling_requested"] is False               # the manifest holds what the engine resolved, True
    assert (built[0]["enforce_eager"], built[0]["tensor_parallel_size"]) == (True, 2)


@pytest.mark.parametrize("dtype", [None, "auto"])
def test_a_dtype_the_engine_was_left_to_choose_leaves_the_built_one(built, dtype):
    manifest, pid, _ = _manifest(_engine({"dtype": dtype, "max_num_seqs": 8}))
    assert manifest["dtype"] == "bfloat16" and "dtype" not in manifest["engine_kwargs"]
    assert pid == "a439413f8a5a"


def test_the_sampling_requested_is_described_and_a_greedy_one_keeps_its_bytes(built):
    sampling = {"greedy": False, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "seed": 7, "logprobs": 5, "n": 2,
                "per_request": ({"temperature": 0.5}, {"temperature": 1.0})}
    manifest, pid, _ = _manifest(_engine({"max_num_seqs": 8}), sampling)
    assert manifest["sampling"] == {"greedy": False, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "seeds": [7], "logprobs": 5, "n": 2,
                                    "per_request": [{"temperature": 0.5}, {"temperature": 1.0}]}
    assert pid == "a7d17e28030e"
    same_as_greedy: list[dict[str, Any]] = [{"n": 1}, {"n": None}, {"n": 0}, {"per_request": []}, {"seed": None}]
    for sampling in same_as_greedy:
        assert _manifest(_engine({"max_num_seqs": 8}), sampling)[1] == "a439413f8a5a", sampling


def _target(**kw: Any) -> NS:
    return NS(**{"moe_backend": None, "int8_exact": None, "int8_parts": (), "to_json": lambda: {"device": "sm90"},
                 "engine_args": lambda: {"attention_backend": "FLASH_ATTN"}, **kw})


def test_a_declared_target_is_in_the_manifest_with_its_engine_args(built):
    manifest, pid, _ = _manifest(_engine({"max_num_seqs": 8}, target=_target()))
    assert manifest["target"] == {"device": "sm90"}
    assert manifest["engine_kwargs"] == {"seed": 0, "enable_prefix_caching": False, "max_num_seqs": 8, "kernel_config": {},
                                         "attention_backend": "FLASH_ATTN", "enforce_eager": False, "async_scheduling": True}
    assert set(manifest["backends"]) == {"attention", "attention_flash_attn_version", "vllm_has_flashinfer"}
    assert list(manifest)[-1] == "target" and pid == "78b56b159c6d"


def test_a_target_that_declares_a_moe_backend_and_int8_parts_has_them_read_back_from_the_model(built):
    target = _target(moe_backend="triton", int8_exact="linears+attention", int8_parts=("linears", "attention"))
    manifest, _, _ = _manifest(_engine(target=target))
    assert manifest["backends"] == {"attention": "A", "attention_flash_attn_version": None, "vllm_has_flashinfer": None,
                                    "moe_experts_class": None, "int8_exact": None, "int8_k_scales": {}}
    linears_only = _manifest(_engine(target=_target(int8_exact="linears", int8_parts=("linears",))))[0]
    assert list(linears_only["backends"]) == ["attention", "attention_flash_attn_version", "vllm_has_flashinfer", "int8_exact"]


def test_what_the_models_layers_carry_is_read_back_into_the_backends(built):
    class Fp8Kernel:
        pass

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model[0].quant_method = NS(fp8_linear=Fp8Kernel())
    model[1].allow_ll_bf16_gemm = True
    model[2].impl = NS(vllm_flash_attn_version=3)
    manifest, _, _ = _manifest(_engine(model=model))
    assert manifest["backends"] == {"attention": "A", "attention_flash_attn_version": 3, "vllm_has_flashinfer": None,
                                    "fp8_block_gemm_kernel": "Fp8Kernel", "moe_gate_ll_bf16_gemm": True}


def test_a_compiled_engine_adds_its_read_back_facts_and_the_kernels_this_process_generated(built, monkeypatch):
    kernels: dict[str, Any] = {"wrappers": 3, "distinct_kernel_sources": 2, "kernel_set_sha256": "k" * 64, "call_set_sha256": "c" * 64,
                               "wrapper_norm_set_sha256": "w" * 64, "inductor_cache_dir": "/cache", "per_process_cache_dir": True,
                               "pieces": [{"path": "a.py"}], "kernels": ["k0"]}
    monkeypatch.setattr(run_facts, "compilation_facts", lambda llm: {"enforce_eager": False, "mode": "VLLM_COMPILE", "capture_sizes": [1, 2]})
    monkeypatch.setattr(run_facts, "generated_kernels_of_this_process", lambda: dict(kernels))
    manifest, pid, notes = _manifest(_engine({"max_num_seqs": 8}))
    assert json.dumps(manifest["compilation"]) == json.dumps({
        "mode": "VLLM_COMPILE", "enforce_eager": False,
        "generated_kernels": {"wrappers": 3, "distinct_kernel_sources": 2, "kernel_set_sha256": "k" * 64, "call_set_sha256": "c" * 64,
                              "wrapper_norm_set_sha256": "w" * 64, "ptx_code_set_sha256": None, "ptx_variants": 0},
        "capture_sizes": [1, 2]})
    assert pid == "6e5f0b93b219"
    assert notes["generated_kernels"] == {"inductor_cache_dir": "/cache", "per_process_cache_dir": True, "pieces": [{"path": "a.py"}],
                                          "kernels": ["k0"], "error": None}
    kernels["error"] = "no inductor output"
    failed, _, notes = _manifest(_engine({"max_num_seqs": 8}))
    assert failed["compilation"]["generated_kernels"]["error"] == "no inductor output"
    assert notes["generated_kernels"]["error"] == "no inductor output"


def test_where_the_profile_cannot_be_built_the_fallback_document_is_the_manifest_and_the_notes_say_so(monkeypatch):
    def refuses(**kw: Any) -> dict[str, Any]:
        raise RuntimeError("no device")

    monkeypatch.setattr(engine_profile, "build_manifest", refuses)
    fake_vllm.install(monkeypatch)["vllm"].__version__ = "0.28.1rc1-test"
    monkeypatch.setitem(sys.modules, "triton", NS(__version__="3.4.0-test"))
    monkeypatch.setattr(run_facts, "torch", NS(__version__="2.14.0-test", version=NS(cuda="12.8")))
    engine = _engine({"model": "/snapshot", "tokenizer": "/snapshot", "max_num_seqs": 8, "seed": 0})
    manifest, pid, notes = _manifest(engine, {"seed": 3})
    assert json.dumps(manifest) == json.dumps({
        "schema": "verity-capture/profile-fallback/v1", "case": "B0", "engine_kwargs": {"max_num_seqs": 8, "seed": 0},
        "engine_flags": FLAGS, "python": {"torch": "2.14.0-test", "cuda": "12.8", "triton": "3.4.0-test"},
        "vllm": {"version": "0.28.1rc1-test", "wheel_sha256": "wheel", "commit": "commit"},
        "model": {"repo": "org/model", "revision": "0" * 40, "weights_sha256": {"model.safetensors": "ab" * 32}},
        "sampling": {**GREEDY, "seeds": [3]}})
    assert pid == "0b7f5044b16d"
    assert notes == {"case": "B0", "accepted_profile_id": run_facts.ACCEPTED_PROFILE_ID, "accepted_record": run_facts.ACCEPTED_RECORD,
                     "profile_id_method": "FALLBACK sha256(engine kwargs + versions): build_manifest failed: RuntimeError('no device')"}
    assert manifest["engine_flags"] is not engine.env_flags and manifest["model"]["weights_sha256"] is not CP["safetensors_sha256"]


SOFTWARE: dict[str, Any] = {"vllm": {"version": "0.28.1rc1-test", "wheel_sha256": "wheel", "commit": "commit"},
                            "python": {"version": "3.12.0", "torch": "2.14.0-test"}, "dists": {"vllm": {"installed": True}}}
GPU = {"name": "stand-in", "count": 1, "compute_capability": "9.0"}
BACKENDS = {"attention": "A", "linear": "L", "moe": None, "sampler": "S"}
COMPILATION = {"mode": "NONE", "enforce_eager": True}
CACHE = {"enable_prefix_caching": True, "block_size": 16}


@pytest.fixture
def readers(monkeypatch) -> NS:
    """The five readers `build_manifest` calls, as stand-ins that say what they were asked."""
    llm = NS()
    asked: list[tuple[str, tuple]] = []

    def reader(name: str, doc: dict[str, Any]) -> Any:
        def read(*args: Any) -> dict[str, Any]:
            asked.append((name, args))
            return dict(doc)
        return read

    for name, doc in (("software_info", SOFTWARE), ("gpu_info", GPU), ("backends_info", BACKENDS), ("effective_compilation", COMPILATION),
                      ("cache_info", CACHE)):
        monkeypatch.setattr(engine_profile, name, reader(name, doc))
    return NS(llm=llm, asked=asked)


def _build(readers: NS, **kw: Any) -> dict[str, Any]:
    return REAL_BUILD_MANIFEST(case="B0", cp=CP, llm=readers.llm, wheel_sha256="wheel", vllm_commit="commit", adapter_revision="rev",
                               interception="none", sampling=dict(GREEDY), env_flags=FLAGS, **kw)


def test_build_manifest_is_the_profile_document_of_the_checkpoint_the_software_the_device_and_the_engine(readers):
    doc = _build(readers)
    want: dict[str, Any] = {
        "schema": "verity-vllm/profile/v1", "case": "B0",
        "model": {"repo": "org/model", "revision": "0" * 40, "weights_sha256": {"model.safetensors": "ab" * 32}, "config_sha256": "cd" * 32},
        "vllm": SOFTWARE["vllm"], "python": SOFTWARE["python"], "dists": SOFTWARE["dists"], "gpu": GPU, "dtype": "bfloat16",
        "quantization": None, "backends": BACKENDS, "parallelism": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
        "compilation": COMPILATION, "cache": CACHE, "engine_flags": FLAGS,
        "engine_kwargs": {"dtype": "bfloat16", "seed": 0, "max_model_len": 4096, "gpu_memory_utilization": 0.8, "trust_remote_code": False},
        "sampling": GREEDY, "adapter": {"revision": "rev", "interception": "none"}}
    assert json.dumps(doc) == json.dumps(want)
    assert engine_profile.profile_id_of(doc) == "c1ce9d7bf26e"
    assert readers.asked == [("software_info", ("wheel", "commit")), ("gpu_info", ()), ("backends_info", (readers.llm,)),
                             ("effective_compilation", (readers.llm,)), ("cache_info", (readers.llm,))]
    assert doc["engine_flags"] is not FLAGS and doc["model"]["weights_sha256"] is not CP["safetensors_sha256"]


def test_build_manifest_writes_each_declared_flag_and_names_an_order_and_overrides_only_where_given(readers):
    doc = _build(readers, enforce_eager=True, custom_ops="all", max_num_batched_tokens=512, tensor_parallel_size=2, quantization="fp8",
                 request_order="reversed", engine_overrides={"max_num_seqs": 4, "seed": 5})
    assert doc["engine_kwargs"] == {"dtype": "bfloat16", "seed": 5, "max_model_len": 4096, "gpu_memory_utilization": 0.8,
                                    "trust_remote_code": False, "tensor_parallel_size": 2, "quantization": "fp8",
                                    "max_num_batched_tokens": 512, "enforce_eager": True, "compilation_config": {"custom_ops": ["all"]},
                                    "max_num_seqs": 4}
    assert (doc["quantization"], doc["parallelism"]) == ("fp8", {"tensor_parallel_size": 2, "pipeline_parallel_size": 1})
    assert list(doc)[-2:] == ["engine_overrides", "workload_order"]
    assert (doc["engine_overrides"], doc["workload_order"]) == ({"max_num_seqs": 4, "seed": 5}, "reversed")
    assert engine_profile.profile_id_of(doc) == "bd7af1194796"
    plain = _build(readers, request_order="workload", engine_overrides={})
    assert "engine_overrides" not in plain and "workload_order" not in plain
    assert engine_profile.profile_id_of(plain) == "c1ce9d7bf26e"


def test_software_info_names_vllm_as_given_the_python_side_by_installed_version_and_three_dists(monkeypatch):
    versions = {"triton": "3.4.0", "transformers": "4.57.0"}

    def version(name: str) -> str:
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    fake_vllm.install(monkeypatch)["vllm"].__version__ = "0.28.1rc1-test"
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(engine_profile, "dist_identity", lambda name: {"installed": name == "torch"})
    assert json.dumps(engine_profile.software_info("wheel", None)) == json.dumps({
        "vllm": {"version": "0.28.1rc1-test", "wheel_sha256": "wheel", "commit": None},
        "python": {"version": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "triton": "3.4.0",
                   "flashinfer": None, "xformers": None, "transformers": "4.57.0"},
        "dists": {"vllm": {"installed": False}, "torch": {"installed": True}, "triton": {"installed": False}}})
