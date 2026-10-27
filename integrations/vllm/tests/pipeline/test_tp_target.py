"""`tp-capture` and `tp-commit` build their engines for the workload's declared target as `build_engine` does: its environment knobs
set before vLLM reads them, its engine arguments under the workload's and the CLI's, every rank's engine checked against it before
anything runs (fail closed), and the record attests it (the capture's per-rank header, both drivers' versions.json), with the execution
the engine and each rank read back from their `vllm_config` (`effective_execution`), not the arguments asked.  CPU-only: each
driver's `main` runs over a stand-in LLM of two ranks whose rpcs answer as the ranks' would, and the rank-side check
(`pipeline.tp.target.rank_mismatches`) and the extension's `tp2_install` run over a stand-in worker."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.program.frontend.target_profile import TargetProfile

pytest.importorskip("torch")

H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False, fp8_block_gemm="cutlass", moe_backend="triton")
WORKLOAD = {"requests": [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2}], "sampling": {"max_tokens": 2}, "sha256": "w",
            "accepted": {"q0": [5, 6]}, "engine_args_required": {"max_model_len": 288, "tensor_parallel_size": 8}, "execution": "eager",
            "engine": {"reasons": {"VLLM_USE_DEEP_GEMM": "the target's CUTLASS block-FP8 GEMM", "moe_backend": "the target's Triton experts"}}}
RUN = {"wall_s": 0.5, "iterations": 2, "step_wall_s": [0.2, 0.3], "generated_tokens": 2, "prompt_tokens": 3, "tokens": {"q0": [5, 6]}}
#: What each rank of an eager engine built for H200 under the engine profile reads back (`engine.served.read_rank`).
SERVED = {"VLLM_BATCH_INVARIANT": False, "VLLM_USE_DEEP_GEMM": False, "VLLM_DISABLE_SHARED_EXPERTS_STREAM": True,
          "VLLM_USE_FLASHINFER_SAMPLER": False, "VLLM_USE_V2_MODEL_RUNNER": True, "NCCL_P2P_DISABLE": None, "disable_custom_all_reduce": False,
          "enforce_eager": True, "compilation_mode": "NONE", "cudagraph_mode": "NONE", "attention_backend": None, "flash_attn_version": None,
          "moe_backend": "triton"}


class _LLM:
    """`vllm.LLM` over two ranks: it keeps the kwargs it was built with and every rpc (the method's name, kwargs); each rank answers
    what the drivers read from the ranks' answers, ``mismatches`` naming what `rank_mismatches` reports for that rank and ``switches``
    what `served.read_rank` reads back on every rank."""

    def __init__(self, mismatches: dict[int, list[str]] | None = None, stop: str | None = None, switches: dict | None = None) -> None:
        self.mismatches, self.stop, self.kw, self.rpcs = mismatches or {}, stop, None, []
        self.switches = SERVED if switches is None else switches
        self.llm_engine = SimpleNamespace()

    def __call__(self, **kw) -> _LLM:
        self.kw = kw
        return self

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        method = method if isinstance(method, str) else method.__name__
        self.rpcs.append((method, dict(kwargs or {})))
        if method == self.stop:
            raise RuntimeError(f"stopped at {method}")
        if method == "read_rank":
            return [{"rank": r, "switches": dict(self.switches)} for r in range(2)]
        return [{"rank": r, "pid": 100 + r, "roots": 1, "profile_id": "p", "mismatches": self.mismatches.get(r, []), "events_written": 1,
                 "steps": 2, "snapshot_bytes": 0, "n_errors": 0, "errors": [], "leaves": 3, "errors_n": 0, "committer": "native_collect_v2b"}
                for r in range(2)]

    def methods(self) -> list[str]:
        return [m for m, _ in self.rpcs]


@pytest.fixture
def stand_ins(monkeypatch):
    """The drivers' engine-side calls stood in, and the target's environment knobs restored after the test; ``use(llm, target)``
    returns the profiles the Commit's guard was given."""
    from verity_vllm import protocols as PO
    from verity_vllm.engine import engine_profile as prof
    from verity_vllm.engine import handle
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import commit as PC
    from verity_vllm.pipeline import gpu_lease as GL
    from verity_vllm.pipeline.tp import commit as TPCM

    for k in ("VLLM_BATCH_INVARIANT", "VLLM_USE_DEEP_GEMM"):
        monkeypatch.setenv(k, "1")

    def use(llm: _LLM, target) -> list:
        guarded: list = []
        vllm = ModuleType("vllm")
        vllm.LLM = llm
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        for owner, name, value in [(prof, "apply_env", lambda: {"VLLM_BATCH_INVARIANT": "1"}), (prof, "checkpoint_for", lambda case, path: {"repo": "org/model"}),
                                   (prof, "engine_kwargs", lambda cp, **kw: dict(kw)), (handle, "model_runner", lambda llm: 1 / 0),
                                   (va, "load_workload", lambda path, max_tokens: json.loads(json.dumps(WORKLOAD))),
                                   (va, "target_of_workload", lambda wl, *a: target), (va, "case_of", lambda wl, case: case),
                                   (va, "throwaway_requests", lambda eng, reqs, seed=0: reqs), (va, "versions_doc", lambda eng: {}),
                                   (va, "run_requests", lambda eng, reqs, sampling: ({"q0": [5, 6]}, {"wall_s": 0.5, "engine_steps": 2})),
                                   (PO, "commit_guard", lambda profile, **k: guarded.append(profile) or profile),
                                   (PC, "git_head", lambda: None), (PC, "run_workload", lambda eng, reqs, sampling: dict(RUN)),
                                   (TPCM, "tree_of_record", lambda: {}), (GL, "_ENVIRON", []), (GL, "_HELD", [])]:
            monkeypatch.setattr(owner, name, value)
        return guarded
    return use


def _capture(tmp_path: Path, llm: _LLM, target, stand_ins, *extra: str) -> Path:
    from verity_vllm.pipeline.tp import capture as TPC
    stand_ins(llm, target)
    out = tmp_path / "capture"
    TPC.main(cli.parse("tp-capture", ["--workload", str(tmp_path / "w.json"), "--out", str(out), "--case", "C", "--no-warmup",
                                      "--engine-arg", "max_num_batched_tokens=2048", *extra], environ={}))
    return out


def test_tp_capture_builds_checks_and_attests_the_declared_target(tmp_path, stand_ins):
    llm = _LLM()
    out = _capture(tmp_path, llm, H200, stand_ins)
    assert (os.environ["VLLM_BATCH_INVARIANT"], os.environ["VLLM_USE_DEEP_GEMM"]) == ("0", "0")
    assert llm.kw["moe_backend"] == "triton" and llm.kw["max_num_batched_tokens"] == 2048
    assert llm.methods()[:3] == ["rank_mismatches", "tp2_probe", "tp2_install"]
    assert llm.rpcs[0][1] == {"target": H200}
    install = llm.rpcs[2][1]
    assert install["target"] == H200 and install["env_flags"] == {"VLLM_BATCH_INVARIANT": "0", "VLLM_USE_DEEP_GEMM": "0"}
    assert install["engine_kwargs"]["moe_backend"] == "triton"
    assert json.loads((out / "versions.json").read_text())["target"] == H200.to_json()


def test_tp_capture_without_a_declared_target_changes_nothing(tmp_path, stand_ins):
    llm = _LLM()
    out = _capture(tmp_path, llm, None, stand_ins)
    assert os.environ["VLLM_USE_DEEP_GEMM"] == "1" and "moe_backend" not in llm.kw
    assert llm.methods()[:2] == ["tp2_probe", "tp2_install"] and llm.rpcs[1][1]["target"] is None
    assert json.loads((out / "versions.json").read_text())["target"] is None


def test_a_cli_engine_arg_still_overrides_the_targets(tmp_path, stand_ins):
    llm = _LLM()
    _capture(tmp_path, llm, H200, stand_ins, "--engine-arg", 'moe_backend="cutlass"')
    assert llm.kw["moe_backend"] == "cutlass"


@pytest.mark.parametrize("driver", ["tp-capture", "tp-commit"])
def test_a_rank_that_does_not_realise_the_declared_target_fails_closed_before_anything_runs(tmp_path, stand_ins, driver):
    llm = _LLM(mismatches={1: ["moe_backend declared 'triton' (TritonExperts), layers report 'CutlassExperts'"]})
    with pytest.raises(SystemExit, match="fail-closed: the engine does not realise the declared TargetProfile: rank 1: moe_backend declared 'triton'"):
        if driver == "tp-capture":
            _capture(tmp_path, llm, H200, stand_ins)
        else:
            _commit(tmp_path, llm, H200, stand_ins)
    assert llm.methods()[-1] == "rank_mismatches" and "tp2_install" not in llm.methods() and "tp2_commit_make" not in llm.methods()


@pytest.mark.parametrize("extra, extension", [(("--no-observer",), "verity_vllm.engine.int8_ranks.Int8Ranks"),
                                              ((), "verity_vllm.pipeline.tp.rank_worker.TP2CaptureWorkerExtension")], ids=["control", "capture"])
def test_an_int8_targets_ranks_register_its_parts_through_their_extension(tmp_path, stand_ins, monkeypatch, extra, extension):
    """The driver registers the int8 parts and exports them (``VERITY_INT8_PARTS``); every engine it builds names an extension whose
    import registers them in each rank, a Match control's included."""
    import verity_vllm.engine as pkg
    from verity_vllm.engine import int8_ranks as IR
    seen: list[str] = []
    for part, mod in (("linears", "int8_vllm"), ("attention", "int8_attention_vllm")):
        stand_in = ModuleType(f"verity_vllm.engine.{mod}")
        stand_in.register = lambda part=part: seen.append(part)
        monkeypatch.setitem(sys.modules, stand_in.__name__, stand_in)
        monkeypatch.setattr(pkg, mod, stand_in, raising=False)
    monkeypatch.delenv(IR.ENV, raising=False)
    llm = _LLM()
    int8 = TargetProfile(compute_capability=(12, 0), num_sms=188, int8_exact="all")
    _capture(tmp_path, llm, int8, stand_ins, *extra)
    assert seen == ["linears", "attention"] and os.environ[IR.ENV] == "attention,linears"
    assert llm.kw["worker_extension_cls"] == extension and llm.kw["quantization"] == "verity_int8"


def _commit(tmp_path: Path, llm: _LLM, target, stand_ins) -> Path:
    from verity_vllm.pipeline.tp import commit as TPCM
    guarded = stand_ins(llm, target)
    out = tmp_path / "commit"
    try:
        TPCM.main(cli.parse("tp-commit", ["--workload", str(tmp_path / "w.json"), "--out", str(out), "--case", "C", "--pairs", "1",
                                          "--warmup", "0", "--engine-arg", "max_num_batched_tokens=2048"], environ={}))
    finally:
        assert guarded == [target]
    return out


def test_tp_commit_builds_and_checks_the_declared_target_and_records_it(tmp_path, stand_ins):
    llm = _LLM(stop="tp2_commit_make")
    with pytest.raises(RuntimeError, match="stopped at tp2_commit_make"):
        _commit(tmp_path, llm, H200, stand_ins)
    assert (os.environ["VLLM_BATCH_INVARIANT"], os.environ["VLLM_USE_DEEP_GEMM"]) == ("0", "0")
    assert llm.kw["moe_backend"] == "triton" and llm.kw["max_model_len"] == 288 and llm.kw["max_num_batched_tokens"] == 2048
    assert llm.methods()[:3] == ["tp2_probe", "rank_mismatches", "read_rank"] and llm.rpcs[1][1] == {"target": H200}
    versions = json.loads((tmp_path / "commit" / "versions.json").read_text())
    assert versions["target"] == H200.to_json() and versions["engine_kwargs"]["moe_backend"] == "triton"
    served = versions["served"]
    assert served["ok"] and [r["switches"] for r in served["ranks"]] == [SERVED, SERVED]
    assert served["declared"]["moe_backend"] == {"value": "triton", "source": "row file", "reason": "the target's Triton experts"}
    assert served["declared"]["VLLM_DISABLE_SHARED_EXPERTS_STREAM"]["source"] == "engine.env.ENGINE"


def test_tp_commit_fails_closed_on_a_switch_its_row_does_not_declare(tmp_path, stand_ins):
    """A rank whose engine runs batch invariant though nothing declares it: the Commit stops before it commits anything, and its
    summary.json says which switch on which rank."""
    llm = _LLM(switches={**SERVED, "VLLM_BATCH_INVARIANT": True})
    with pytest.raises(SystemExit, match="fail-closed: served switches: rank 0: VLLM_BATCH_INVARIANT=True where row file declares False"):
        _commit(tmp_path, llm, H200, stand_ins)
    assert "tp2_commit_make" not in llm.methods()
    sm = json.loads((tmp_path / "commit" / "summary.json").read_text())
    assert sm["pass"] is False and sm["served"]["problems"] == [f"rank {r}: VLLM_BATCH_INVARIANT=True where row file declares False" for r in (0, 1)]


def _built_config(gpu_util: float) -> SimpleNamespace:
    """A built engine's `vllm_config`, as `effective_execution_doc` reads it back, whose cache took ``gpu_util`` of the device."""
    return SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False, block_size=16, gpu_memory_utilization=gpu_util),
                           scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_batched_tokens=2048, max_num_seqs=1,
                                                            async_scheduling=False),
                           model_config=SimpleNamespace(max_model_len=288, dtype="torch.bfloat16", enforce_eager=True),
                           compilation_config=SimpleNamespace(mode="NONE", cudagraph_mode="NONE", custom_ops=[]),
                           use_v2_model_runner=True, optimization_level=0)


@pytest.mark.parametrize("driver", ["tp-capture", "tp-commit"])
def test_the_record_carries_the_engines_read_back_gpu_memory_utilization_not_the_flag(tmp_path, stand_ins, monkeypatch, driver):
    """The job asks 0.92 and the built engine's `vllm_config` holds 0.88: the capture's topology.json and the Commit's versions.json
    record 0.88 as `effective_execution`, beside the 0.92 asked in `engine_kwargs`."""
    monkeypatch.setitem(sys.modules, "vllm.envs", SimpleNamespace(VLLM_BATCH_INVARIANT=False))
    llm = _LLM(stop="tp2_commit_make")
    llm.llm_engine = SimpleNamespace(vllm_config=_built_config(0.88))
    if driver == "tp-capture":
        doc = json.loads((_capture(tmp_path, llm, H200, stand_ins, "--engine-arg", "gpu_memory_utilization=0.92") / "topology.json").read_text())
    else:
        from verity_vllm.pipeline.tp import commit as TPCM
        stand_ins(llm, H200)
        with pytest.raises(RuntimeError, match="stopped at tp2_commit_make"):
            TPCM.main(cli.parse("tp-commit", ["--workload", str(tmp_path / "w.json"), "--out", str(tmp_path / "commit"), "--case", "C",
                                              "--pairs", "1", "--warmup", "0", "--engine-arg", "gpu_memory_utilization=0.92"], environ={}))
        doc = json.loads((tmp_path / "commit" / "versions.json").read_text())
    assert doc["engine_kwargs"]["gpu_memory_utilization"] == 0.92
    eff = doc["effective_execution"]
    assert eff["gpu_memory_utilization"] == 0.88 and "config_error" not in eff and "compilation_error" not in eff, eff


# -- the ranks' side -----------------------------------------------------------------------------------------------------------


class _Worker:
    """What the rank-side calls read from vLLM's Worker: its config, its model and its model runner."""

    def __init__(self, model) -> None:
        self.vllm_config = SimpleNamespace()
        self.get_model = lambda: model
        self.model_runner = SimpleNamespace(get_model=lambda: model)


def _extension(monkeypatch, model):
    from verity_vllm.pipeline.tp import rank_worker as RW
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: {"tp": {"rank": 1, "world_size": 2}})
    w = _Worker(model)
    w.__class__ = type("Worker", (_Worker, RW.TP2CaptureWorkerExtension), {})
    return RW, w


def test_a_rank_checks_its_own_engine_against_the_declared_target(monkeypatch):
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline.tp import target as TT
    seen = []
    monkeypatch.setattr(va, "target_mismatches", lambda eng: seen.append(eng) or ["num_sms declared 132, device 114"])
    distributed = ModuleType("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 1
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    model = object()
    w = _Worker(model)
    assert TT.rank_mismatches(w, H200) == {"rank": 1, "mismatches": ["num_sms declared 132, device 114"]}
    (eng,) = seen
    assert eng.target is H200 and eng.model is model and eng.llm.llm_engine.vllm_config is w.vllm_config
    assert eng.llm.collective_rpc(lambda worker: worker) == [w]


def test_a_ranks_probe_reads_back_its_own_engines_execution(monkeypatch):
    """`tp2_probe` reads `effective_execution` from this rank's own `vllm_config`, so topology.json's and versions.json's `ranks`
    show what each rank's engine took, not the driver's copy."""
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.envs", SimpleNamespace(VLLM_BATCH_INVARIANT=False))
    _, w = _extension(monkeypatch, object())
    w.vllm_config = _built_config(0.88)
    eff = w.tp2_probe()["effective_execution"]
    assert eff["gpu_memory_utilization"] == 0.88 and eff["execution_label"] == "eager" and "config_error" not in eff, eff


@pytest.mark.parametrize("target", [H200, None], ids=["declared", "undeclared"])
def test_a_ranks_header_is_made_over_an_engine_built_for_the_declared_target(monkeypatch, tmp_path, target):
    """`tp2_install`'s header is `make_header` over the rank's `Engine`, whose `target` is what `run_facts.profile_manifest` attests."""
    from verity_vllm.observe import capture
    seen = []

    def make_header(engine, workload, **kw):
        seen.append(engine)
        raise RuntimeError("header made")

    monkeypatch.setattr(capture, "make_header", make_header)
    _, w = _extension(monkeypatch, object())
    with pytest.raises(RuntimeError, match="header made"):
        w.tp2_install(out_dir=str(tmp_path), workload=WORKLOAD, case="C", cp={"repo": "org/model"}, engine_kwargs={}, env_flags={},
                      snapshot_steps=[], snapshot_max_bytes=0, adapter_revision="r", wheel_sha256=None, vllm_commit=None,
                      target=target)
    assert seen[0].target == target
