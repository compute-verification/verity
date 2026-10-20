"""served_commit's arm (`benchmarks/served_commit/arm.py`) gates on the harness's run phases: under bounded staging its gate and
warm-up runs are learn-only, a run of theirs left without a root is a declared warm-up named in the custom gate's verdict, and a
measured run without a root fails the gate. Its ``ready``, and the stock profiling arm's (`prof_arm.py`), read only the engine's
public names (``vllm``, ``model``, ``pin``), for ``verity_vllm.LLM`` and for the harness's local checkpoint, and the committer
gets every field it reads of its engine. The committer and the compiled source are fakes; the roots are the GPU suites'."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

ARM = Path(__file__).resolve().parents[2] / "served_commit" / "arm.py"
PROF_ARM = ARM.parent / "prof_arm.py"


def _arm_module():
    spec = importlib.util.spec_from_file_location("served_commit_arm", ARM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Committer:
    def __init__(self, bounded: bool) -> None:
        self.bounded, self.learn_only, self._layouts, self.result = bounded, False, {}, None

    def reset(self) -> None:
        pass

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def finalize(self):
        return self.result


class _Source:
    stats = {"replayed_steps": 0, "eager_steps": 0, "drain_s": 0.0}
    errors: list = []

    def begin_run(self) -> None:
        pass

    def summary(self) -> dict:
        return {}


def _sweep(arm, com, runs) -> dict:
    """Runs (phase, point, run root, learned-only steps) through the arm as the harness would; the custom gate's verdict."""
    learn_only = []
    for phase, point, root, learned in runs:
        target = SimpleNamespace(phase=phase, point=point)
        arm.begin_run(target)
        learn_only.append(com.learn_only)
        com.result = SimpleNamespace(run_root=root, extra={"bounded_staging": {"learned_only_steps": list(range(learned))}})
        arm.end_run(target)
    return {"learn_only": learn_only, **arm.gate(None)}


def test_a_rootless_gate_or_warmup_run_is_a_declared_warmup():
    mod = _arm_module()
    arm, com = mod.ServedCommitArm("served-commit-noretain", retain="none"), _Committer(bounded=True)
    arm.committer, arm.source = com, _Source()
    got = _sweep(arm, com, [("gate", "B8", b"r" * 32, 0), ("gate", "B32", b"", 4), ("warmup", "B32", b"", 4),
                            ("measure", "B32", b"r" * 32, 0)])
    assert got["learn_only"] == [True, True, True, False]
    assert got["passed"] and not got["failed"]
    assert got["warmups"] == [{"run": "gate:B32", "learned_only_steps": 4}, {"run": "warmup:B32", "learned_only_steps": 4}]


def test_a_measured_run_without_a_root_fails_the_gate():
    mod = _arm_module()
    arm, com = mod.ServedCommitArm("served-commit-noretain", retain="none"), _Committer(bounded=True)
    arm.committer, arm.source = com, _Source()
    got = _sweep(arm, com, [("gate", "B32", b"r" * 32, 0), ("measure", "B32", b"", 4)])
    assert not got["passed"] and not got["warmups"]
    assert got["failed"][0]["run"] == "measure:B32" and "not a declared warm-up" in got["failed"][0]["errors"][0]


def test_unbounded_staging_runs_no_learn_only_pass():
    mod = _arm_module()
    arm, com = mod.ServedCommitArm("served-commit-noretain", retain="none"), _Committer(bounded=False)
    arm.committer, arm.source = com, _Source()
    got = _sweep(arm, com, [("gate", "B32", b"r" * 32, 0), ("warmup", "B32", b"r" * 32, 0)])
    assert got["learn_only"] == [False, False] and got["passed"] and not got["warmups"]


class _Runner:
    def __init__(self) -> None:
        self.model = object()

    def get_model(self):
        return self.model

    def execute_model(self, scheduler_output):
        return "executed"

    def sample(self):
        return "sampled"


def _vllm(runner: _Runner, *, path: str = "/models/local"):
    """A ``vllm.LLM`` as far as the arms and the committer read it: the in-process runner (`get_model_runner`'s path) and the
    config."""
    hf = SimpleNamespace(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         intermediate_size=128, vocab_size=1000, model_type="llama")
    vc = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=2048, max_num_seqs=8),
                         model_config=SimpleNamespace(model=path, revision=None, max_model_len=4096, hf_config=hf))
    core = SimpleNamespace(model_executor=SimpleNamespace(driver_worker=SimpleNamespace(model_runner=runner)))
    return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core, vllm_config=vc))


class _Collector:
    """`NativeCollectCommitter` as ``ready`` drives it: the engine it is given, the envelope it sizes its arena by."""

    def __init__(self, engine, **kw) -> None:
        self.engine, self.kw = engine, kw

    def size_arena(self, envelope):
        self.envelope = envelope
        return {"envelope": envelope}

    def enable_windows(self, sizing):
        return {"windows": sizing}


class _ReadySource:
    def note_engine(self, model) -> None:
        self.noted = model

    def attach(self, com) -> None:
        self.attached = com


def _ready(monkeypatch, engine):
    """``ServedCommitArm.ready`` on ``target.engine = engine``, with the collector a fake; the arm."""
    native_collect = ModuleType("verity_vllm.commit.committer.native_collect")
    native_collect.NativeCollectCommitter = _Collector
    monkeypatch.setitem(sys.modules, native_collect.__name__, native_collect)
    monkeypatch.setenv("VERITY_RETAIN", "")
    arm = _arm_module().ServedCommitArm("served-commit-noretain", retain="none")
    arm.source = _ReadySource()
    arm.ready(SimpleNamespace(engine=engine))
    return arm


def _geo_digest(engine) -> bytes:
    """The committer's geometry digest over ``engine``, as its constructor reads it."""
    from verity_vllm.commit.committer.native_host import NativeHostCommitter

    return NativeHostCommitter._model_geo_digest(SimpleNamespace(engine=engine, chunk=4096))


def test_ready_gives_the_committer_the_private_engines_fields_from_llms_public_names(monkeypatch):
    runner, repo, revision = _Runner(), "meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"
    vllm = _vllm(runner, path="/hf/snapshots/4e20de3624")
    llm = SimpleNamespace(vllm=vllm, model=runner.model, pin=SimpleNamespace(model=repo, revision=revision))
    arm = _ready(monkeypatch, llm)
    com = arm.committer
    assert (com.engine.llm, com.engine.runner, com.engine.model) == (vllm, runner, runner.model)
    assert arm.source.noted is runner.model and arm.source.attached is com
    private = SimpleNamespace(llm=vllm, cp={"repo": repo, "revision": revision, "local_path": "/hf/snapshots/4e20de3624"})
    assert _geo_digest(com.engine) == _geo_digest(private)
    assert com.envelope == {"batch": 8, "max_prompt": 2048, "max_new": 1, "max_num_batched_tokens": 2048, "max_model_len": 4096}
    assert arm.windows == {"windows": {"envelope": com.envelope}}


def test_a_local_checkpoint_commits_under_vllms_model_path_without_the_private_engine(monkeypatch):
    from verity_bench.vllm.runner import _DirectLLM

    runner = _Runner()
    direct = _DirectLLM(_vllm(runner), {})
    vars(direct).pop("_engine", None)
    com = _ready(monkeypatch, direct).committer
    assert (com.engine.runner, com.engine.model) == (runner, runner.model)
    assert com.engine.cp == {"repo": "/models/local", "revision": None}
    assert len(_geo_digest(com.engine)) == 32


def test_the_stock_profiling_arm_ranges_the_runner_of_a_local_checkpoint_without_the_private_engine(monkeypatch):
    from verity_bench.vllm.runner import _DirectLLM

    ranges: list[str] = []
    nvtx = SimpleNamespace(range_push=ranges.append, range_pop=lambda: ranges.append("pop"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(nvtx=nvtx)))
    monkeypatch.setitem(sys.modules, "served_commit.arm", _arm_module())
    spec = importlib.util.spec_from_file_location("served_commit_prof_arm", PROF_ARM)
    prof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prof)
    runner = _Runner()
    direct = _DirectLLM(_vllm(runner), {})
    vars(direct).pop("_engine", None)
    prof.StockProfArm("stock-prof").ready(SimpleNamespace(engine=direct))
    assert runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=7)) == "executed"
    assert runner.sample() == "sampled"
    assert ranges == ["exec:7", "pop", "sample", "pop"]
