"""The route capture's plugin on stand-ins for vLLM's V1 and V2 model runners (the methods it wraps, as vLLM d9105ea80 has
them) and for torch's profiler, and routes.py's summary of what it wrote."""
from __future__ import annotations

import enum
import gzip
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import route_capture_plugin as RC
import routes
from verity_vllm.engine.hooks import Hooks

Mode = enum.Enum("CUDAGraphMode", "NONE PIECEWISE FULL")
KERNELS = {"decode": ["fused_moe_kernel", "fp8_gemm_kernel", "flash_mla_sparse_decode"],
           "prefill": ["fused_moe_kernel", "deep_gemm_fp8", "FlashAttnFwdSm90"]}


def sp(t: float) -> NS:
    return NS(temperature=t, top_p=0.95 if t else 1.0, top_k=0, min_p=0.0, seed=None, presence_penalty=0.0,
              frequency_penalty=0.0, repetition_penalty=1.0, logprobs=None)


def so(new=(), cached=(), finished=()) -> NS:
    """A scheduler output: new requests (id, prompt length, computed tokens, scheduled, temperature) and cached ones (id,
    computed, scheduled)."""
    nr = [NS(req_id=i, prompt_token_ids=[0] * p, sampling_params=sp(t), num_computed_tokens=c) for i, p, c, _, t in new]
    sched = {i: q for i, _, _, q, _ in new} | {i: q for i, _, q in cached}
    return NS(scheduled_new_reqs=nr, scheduled_cached_reqs=NS(req_ids=[i for i, _, _ in cached],
                                                              num_computed_tokens=[c for _, c, _ in cached]),
              num_scheduled_tokens=sched, total_num_scheduled_tokens=sum(sched.values()), finished_req_ids=set(finished),
              scheduled_spec_decode_tokens={})


#: a request's life: a 4,000-token prompt whose first 1,024 tokens hit the prefix cache, prefilled in two chunks beside a
#: short sampled request; then both decode; then the long one finishes
STEPS = [so(new=[("a", 4000, 1024, 2048, 0.0), ("b", 100, 0, 100, 0.7)]),
         so(cached=[("a", 3072, 928), ("b", 100, 1)]),
         so(cached=[("a", 4000, 1), ("b", 101, 1)]),
         so(cached=[("b", 102, 1)], finished=["a"]),
         so(cached=[("b", 103, 1)])]
KEYS = ["prefill/M4096/R2/sparse/hit/mixed+top_p", "mixed/M1024/R2/sparse/cold/mixed+top_p",
        "decode/M2/R2/-/cold/mixed+top_p", "decode/M1/R1/-/cold/sampled+top_p", "decode/M1/R1/-/cold/sampled+top_p"]


def test_a_steps_row_and_class_follow_its_requests_from_their_first_schedule() -> None:
    reqs, edges = RC.Requests(), RC.config({"ROUTECAP_OUT": "x"})["m_edges"]
    rows = [reqs.row(s, 2048) for s in STEPS]
    assert [RC.key(r, edges) for r in rows] == KEYS
    assert rows[0]["prefill"] == [[2048, 1024, 4000], [100, 0, 100]] and rows[0]["prefix_hit_tokens"] == 1024
    assert rows[0]["sampling"] == {"greedy": 1, "random": 1, "top_p": 1} and rows[0]["prefill_seq_max"] == 3072
    assert rows[1]["prefill"] == [[928, 3072, 4000]] and (rows[1]["decode"], rows[1]["decode_ctx_max"]) == (1, 101)
    assert rows[2]["prefill"] == [] and rows[2]["decode_ctx"] == [4001, 102] and rows[2]["decode_ctx_sum"] == 4001 + 102
    assert "a" not in reqs.prompt and "b" in reqs.prompt
    short = RC.Requests().row(so(new=[("c", 500, 0, 500, 0.0)]), 2048)
    assert RC.key(short, edges) == "prefill/M512/R1/dense/cold/greedy"


def test_the_token_buckets_hold_each_route_switch() -> None:
    edges = RC.config({"ROUTECAP_OUT": "x"})["m_edges"]
    assert [RC.bucket(n, edges) for n in (1, 3, 31, 32, 33, 42, 43, 8192, 9000)] == [1, 4, 31, 32, 42, 42, 48, 8192, 16384]


def test_the_chooser_takes_each_class_up_to_its_quota_within_the_budget_and_duty() -> None:
    c = RC.Chooser(per_class=2, budget=3, duty=0.1, warm=1)
    assert c.want("a", 0.0)
    c.took("a", 5.0)
    assert not c.want("a", 1.0)            # past the warm steps, 5 s of profiling in 1 s is over the duty
    assert c.want("a", 100.0)
    c.took("a", 5.0)
    assert not c.want("a", 1000.0)         # the class's quota
    assert c.want("b", 1000.0)
    c.took("b", 1.0)
    assert not c.want("c", 1e6)            # the budget


class Profiler:
    """torch.profiler.profile's surface the plugin uses: a trace of the step's kernels, and an operator table."""
    current: list[str] = KERNELS["decode"]
    started: list[tuple] = []

    def __init__(self, activities, record_shapes):
        self.activities = tuple(activities)

    def start(self):
        Profiler.started.append(self.activities)

    def stop(self):
        pass

    def export_chrome_trace(self, path):
        evs = [{"ph": "X", "cat": "kernel", "name": n, "ts": i, "dur": 1.0, "args": {"grid": [i + 1, 1, 1], "block": [128, 1, 1]}}
               for i, n in enumerate(Profiler.current)]
        with gzip.open(path, "wt") as f:
            json.dump({"traceEvents": evs}, f)

    def key_averages(self, group_by_input_shape):
        return [NS(key="vllm::moe_forward", input_shapes=[[8, 6144]], count=3), NS(key="aten::mm", input_shapes=[[8, 2]], count=9)]


TORCH = NS(__version__="2.13.0", version=NS(cuda="12.9"), cuda=NS(synchronize=lambda: None),
           distributed=NS(is_available=lambda: True, is_initialized=lambda: False),
           profiler=NS(profile=Profiler, ProfilerActivity=NS(CPU="cpu", CUDA="cuda")))

V1 = '''
class GPUModelRunner:
    def __init__(self, mode, topk=2048):
        self.mode, self.calls = mode, []
        self.model_config = NS(hf_text_config=NS(index_topk=topk))
        self.vllm_config = NS(compilation_config="CompilationConfig(cudagraph_mode=FULL_AND_PIECEWISE)")

    def _determine_batch_execution_and_padding(self, num_tokens, num_reqs, **kw):
        return (self.mode.PIECEWISE if num_tokens > num_reqs else self.mode.FULL,
                NS(num_tokens=-(-num_tokens // 8) * 8, num_reqs=None, uniform=num_tokens == num_reqs), False, None, None)

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        n = scheduler_output.total_num_scheduled_tokens
        if not n:
            return "EMPTY"
        self._determine_batch_execution_and_padding(num_tokens=n, num_reqs=len(scheduler_output.num_scheduled_tokens))
        self.calls.append("execute")
        return None

    def sample_tokens(self, grammar_output):
        self.calls.append("sample")
        return "output"

    def capture_model(self):
        return 3
'''

V2 = '''
def dispatch_cg_and_sync_dp(manager, num_reqs, num_toks, uniform_tok_count, dp_size, dp_rank, max_query_len=None, **kw):
    return NS(cg_mode=MODE.FULL if uniform_tok_count == 1 else MODE.PIECEWISE, num_tokens=num_toks, num_reqs=num_reqs,
              uniform_token_count=uniform_tok_count, max_query_len=max_query_len), None


class GPUModelRunner:
    def __init__(self):
        self.model_config = NS(hf_config=NS(index_topk=2048))

    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, **kw):
        q = max(scheduler_output.num_scheduled_tokens.values())
        dispatch_cg_and_sync_dp(None, len(scheduler_output.num_scheduled_tokens), scheduler_output.total_num_scheduled_tokens,
                                1 if q == 1 else None, 1, 0, max_query_len=q)
        return None

    def sample_tokens(self, grammar_output):
        return "output"
'''


def module(name: str, src: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.NS, m.MODE = NS, Mode
    exec(src, m.__dict__)  # noqa: S102 -- the stand-in runner modules, this file's own source
    return m


@pytest.fixture
def stand_ins(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", TORCH)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    sys.modules["vllm"].__version__ = "0.28.1rc1.dev472+gd9105ea80"
    monkeypatch.setattr(RC, "_RECORDER", [])
    monkeypatch.setattr(RC, "_INSTALLED", [])
    Profiler.started = []


def cfg(out: Path, **kw) -> dict:
    return {**RC.config({"ROUTECAP_OUT": str(out)}), **kw}


def test_v1_steps_are_recorded_and_the_first_of_each_class_profiled(stand_ins, tmp_path) -> None:
    m = module("fake_v1", V1)
    original = m.GPUModelRunner.execute_model
    hooks = Hooks(tag="test")
    RC.patch_runner(m, cfg(tmp_path, per_class=1), hooks)
    r = m.GPUModelRunner(Mode)
    assert r.capture_model() == 3
    for s in STEPS:
        Profiler.current = KERNELS["prefill" if s.scheduled_new_reqs or s.total_num_scheduled_tokens > 2 else "decode"]
        assert r.execute_model(s) is None and r.sample_tokens(None) == "output"
    assert r.execute_model(so()) == "EMPTY"
    rows = [json.loads(x) for x in (tmp_path / "steps.rank0.jsonl").read_text().splitlines()]
    assert [x["class"] for x in rows] == KEYS and [x["step"] for x in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["cg"] == {"cg_mode": "PIECEWISE", "num_tokens": 2152, "num_reqs": None, "uniform": False}
    assert rows[3]["cg"]["cg_mode"] == "FULL"
    assert [x.get("profiled") for x in rows] == ["step000000.json.gz", "step000001.json.gz", "step000002.json.gz",
                                                 "step000003.json.gz", None]
    assert Profiler.started[0] == ("cpu",) and set(Profiler.started[1:]) == {("cpu", "cuda")}
    ops = json.loads((tmp_path / "capture_ops.rank0.json").read_text())
    assert ops["ops"][0] == {"op": "aten::mm", "shapes": [[8, 2]], "count": 9}
    engine = json.loads((tmp_path / "engine.rank0.json").read_text())
    assert engine["index_topk"] == 2048 and "FULL_AND_PIECEWISE" in engine["compilation_config"]
    assert not (tmp_path / "errors.rank0.log").exists()

    s = routes.summary(tmp_path)
    assert (s["steps"], s["classes"], s["profiled"], s["tokens"]) == (5, 4, 4, 2148 + 929 + 2 + 1 + 1)
    assert s["prefill_tokens"] == 2148 + 928 and s["prefix_hit_tokens"] == 1024
    assert s["by_class"][0]["class"] == KEYS[0] and s["by_class"][0]["kernels"] == dict.fromkeys(sorted(KERNELS["prefill"]), 1)
    assert s["kernel_classes"]["fused_moe_kernel"] == 4
    hooks.uninstall()
    assert m.GPUModelRunner.execute_model is original


def test_v2_dispatch_is_read_through_the_module_function_and_dummy_runs_are_not_steps(stand_ins, tmp_path) -> None:
    m = module("fake_v2", V2)
    hooks = Hooks(tag="test")
    RC.patch_runner(m, cfg(tmp_path, capture_ops=False), hooks)
    r = m.GPUModelRunner()
    r.execute_model(STEPS[0], dummy_run=True)
    for s in STEPS[2:4]:
        r.execute_model(s)
        r.sample_tokens(None)
    rows = [json.loads(x) for x in (tmp_path / "steps.rank0.jsonl").read_text().splitlines()]
    assert [x["cg"] for x in rows] == [
        {"cg_mode": "FULL", "num_tokens": 2, "num_reqs": 2, "uniform_token_count": 1, "max_query_len": 1},
        {"cg_mode": "FULL", "num_tokens": 1, "num_reqs": 1, "uniform_token_count": 1, "max_query_len": 1}]
    hooks.uninstall()


def test_a_rank_off_the_list_records_nothing(stand_ins, tmp_path) -> None:
    m = module("fake_v1", V1)
    hooks = Hooks(tag="test")
    RC.patch_runner(m, cfg(tmp_path, ranks=(1,)), hooks)
    r = m.GPUModelRunner(Mode)
    r.execute_model(STEPS[0])
    r.sample_tokens(None)
    assert RC._RECORDER == [None] and not list(tmp_path.glob("steps.*"))
    hooks.uninstall()


def test_register_patches_a_runner_module_when_it_is_imported(stand_ins, tmp_path, monkeypatch) -> None:
    pkg = tmp_path / "site" / "vllm" / "v1" / "worker"
    pkg.mkdir(parents=True)
    for d in (pkg, pkg.parent, pkg.parent.parent):
        (d / "__init__.py").write_text("")
    (pkg / "gpu_model_runner.py").write_text("from types import SimpleNamespace as NS\n" + V1)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.delitem(sys.modules, "vllm")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setenv("ROUTECAP_OUT", str(tmp_path / "out"))
    RC.register()
    RC.register()
    assert len(RC._INSTALLED) == 1 and not RC._INSTALLED[0].entries
    try:
        mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
        assert {e.name for e in RC._INSTALLED[0].entries} >= {"execute_model", "sample_tokens", RC.V1_DISPATCH}
        r = mod.GPUModelRunner(Mode)
        r.execute_model(STEPS[0])
        r.sample_tokens(None)
        assert json.loads((tmp_path / "out" / "steps.rank0.jsonl").read_text())["class"] == KEYS[0]
    finally:
        RC._INSTALLED[0].uninstall()
        for k in [k for k in sys.modules if k == "vllm" or k.startswith("vllm.")]:
            del sys.modules[k]


def test_unset_out_installs_nothing(monkeypatch) -> None:
    monkeypatch.delenv("ROUTECAP_OUT", raising=False)
    monkeypatch.setattr(RC, "_INSTALLED", [])
    RC.register()
    assert RC._INSTALLED == [] and RC.config({}) is None
