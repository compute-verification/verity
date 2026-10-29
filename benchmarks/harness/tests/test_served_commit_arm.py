"""served_commit's arm (`benchmarks/served_commit/arm.py`) installs only a scheme other than vllm-v1, which it refuses by name.
It gates on the harness's run phases: under bounded staging its gate and
warm-up runs are learn-only, a run of theirs left without a root is a declared warm-up named in the custom gate's verdict, and a
measured run without a root fails the gate. Its ``ready``, and the stock profiling arm's (`prof_arm.py`), read only the engine's
public names (``vllm``, ``model``, ``pin``), for ``verity_vllm.LLM`` and for the harness's local checkpoint, and the committer
gets every field it reads of its engine. The profiling arms write each run's records after its timed run (``end_run`` writes
nothing). The committer and the compiled source are fakes; the roots are the GPU suites'."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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
    arm, com = mod.ServedCommitArm("served-commit-rows", retain="none"), _Committer(bounded=True)
    arm.committer, arm.source = com, _Source()
    got = _sweep(arm, com, [("gate", "B8", b"r" * 32, 0), ("gate", "B32", b"", 4), ("warmup", "B32", b"", 4),
                            ("measure", "B32", b"r" * 32, 0)])
    assert got["learn_only"] == [True, True, True, False]
    assert got["passed"] and not got["failed"]
    assert got["warmups"] == [{"run": "gate:B32", "learned_only_steps": 4}, {"run": "warmup:B32", "learned_only_steps": 4}]


def test_a_measured_run_without_a_root_fails_the_gate():
    mod = _arm_module()
    arm, com = mod.ServedCommitArm("served-commit-rows", retain="none"), _Committer(bounded=True)
    arm.committer, arm.source = com, _Source()
    got = _sweep(arm, com, [("gate", "B32", b"r" * 32, 0), ("measure", "B32", b"", 4)])
    assert not got["passed"] and not got["warmups"]
    assert got["failed"][0]["run"] == "measure:B32" and "not a declared warm-up" in got["failed"][0]["errors"][0]


def test_a_vllm_v1_arm_refuses_to_install_by_name_and_a_rows_arm_installs(monkeypatch):
    mod = _arm_module()
    v1 = mod.ServedCommitArm("served-commit-v1", retain="none")
    with pytest.raises(RuntimeError, match=r"served-commit-v1 arm commits 'vllm-v1' \(SHA-256\), not 'rows-sha512'"):
        v1.install(SimpleNamespace(hooks=None))
    assert v1.source is None
    source = ModuleType("verity_vllm.acquire.sources.served_compiled_source")
    source.ServedCompiledSource = lambda into, batch_copies: SimpleNamespace(install_hooks=lambda: None)
    monkeypatch.setitem(sys.modules, source.__name__, source)
    rows = mod.ServedCommitArm("served-commit-rows", retain="none")
    rows.commit_scheme = "rows-sha512"
    assert rows.install(SimpleNamespace(hooks=None)) is rows and rows.source is not None


def test_unbounded_staging_runs_no_learn_only_pass():
    mod = _arm_module()
    arm, com = mod.ServedCommitArm("served-commit-rows", retain="none"), _Committer(bounded=False)
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
    """A ``vllm.LLM`` as far as the arms and the committer read it: the in-process runner (`engine.handle.model_runner`'s path) and the
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
    arm = _arm_module().ServedCommitArm("served-commit-rows", retain="none")
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
    com = _ready(monkeypatch, direct).committer
    assert (com.engine.runner, com.engine.model) == (runner, runner.model)
    assert com.engine.cp == {"repo": "/models/local", "revision": None}
    assert len(_geo_digest(com.engine)) == 32


def _prof_module(monkeypatch) -> tuple[ModuleType, list[str]]:
    """prof_arm.py with torch's NVTX ranges recorded; the module and the ranges."""
    ranges: list[str] = []
    nvtx = SimpleNamespace(range_push=ranges.append, range_pop=lambda: ranges.append("pop"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(nvtx=nvtx)))
    monkeypatch.setitem(sys.modules, "served_commit.arm", _arm_module())
    spec = importlib.util.spec_from_file_location("served_commit_prof_arm", PROF_ARM)
    prof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prof)
    return prof, ranges


def test_the_stock_profiling_arm_ranges_the_runner_of_a_local_checkpoint_without_the_private_engine(monkeypatch):
    from verity_bench.vllm.runner import _DirectLLM

    prof, ranges = _prof_module(monkeypatch)
    runner = _Runner()
    direct = _DirectLLM(_vllm(runner), {})
    prof.StockProfArm("stock-prof").ready(SimpleNamespace(engine=direct))
    assert runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=7)) == "executed"
    assert runner.sample() == "sampled"
    assert ranges == ["exec:7", "pop", "sample", "pop"]


class _RootsCommitter(_Committer):
    name, native_end, rows_off_thread = "native_collect", True, True

    def __init__(self) -> None:
        super().__init__(bounded=True)
        self.runner = _Runner()
        self._nc = SimpleNamespace(slot_fast=True, stats=lambda: {"plans": 1})
        self.spans = SimpleNamespace(total={"hashing": 0.5}, calls={"hashing": 2})


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_the_roots_arm_writes_each_runs_records_after_its_timed_run(monkeypatch, tmp_path):
    prof, ranges = _prof_module(monkeypatch)
    (tmp_path / "out").mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(prof.RootsArm, "dumped", set())
    arm, com = prof.RootsArm("served-commit-rows", retain="none"), _RootsCommitter()
    arm.committer, arm.windows = com, {}
    arm.source = SimpleNamespace(stats=dict(_Source.stats), errors=[], codegen={}, native_drain=True, begin_run=lambda: None,
                                 summary=dict)
    roots, layouts = tmp_path / "out" / "roots.jsonl", tmp_path / "out" / "layouts.jsonl"
    written = []
    for phase, root in (("warmup", b"w" * 32), ("measure", b"m" * 32)):
        target = SimpleNamespace(phase=phase, point="B8")
        arm.begin_run(target)
        com._layouts = {0: [SimpleNamespace(name="lm_head", dtype="bfloat16", shape=(8, 16), nbytes=256)]}
        arm.plan_log.append((0, 7, 0))
        com.result = SimpleNamespace(run_root=root, extra={"bounded_staging": {"enabled": True}})
        arm.end_run(target)
        written.append(len(_lines(roots)))
    assert written == [0, 1] and ranges[-1] == "pop"
    assert arm.extras(target)["steps"] == 1
    got = [(r["name"], r["run"], r["run_root"], r["steps_log"], r["plan_hits"]) for r in _lines(roots)]
    assert got == [("warmup:B8", 0, (b"w" * 32).hex(), [[0, 7, 0, 256]], 1), ("measure:B8", 1, (b"m" * 32).hex(), [[0, 7, 0, 256]], 1)]
    assert [(r["run"], r["steps"]) for r in _lines(layouts)] == [(0, {"0": [["lm_head", "bfloat16", [8, 16], 256]]})]
    assert [r["run"] for r in arm.runs] == ["warmup:B8", "measure:B8"]


def test_the_stock_profiling_arm_writes_its_cpu_line_after_the_timed_run(monkeypatch, tmp_path):
    prof, _ = _prof_module(monkeypatch)
    (tmp_path / "out").mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    roots, arm, target = tmp_path / "out" / "roots.jsonl", prof.StockProfArm("stock-prof"), SimpleNamespace()
    arm.begin_run(target)
    arm.end_run(target)
    assert not roots.exists()
    assert arm.extras(target) == {}
    arm.begin_run(target)
    arm.end_run(target)
    arm.begin_run(target)
    assert [r["arm"] for r in _lines(roots)] == ["stock-prof", "stock-prof"]
