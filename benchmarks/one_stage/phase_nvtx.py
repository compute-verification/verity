"""NVTX ranges on the serving thread's phases of a served step, for the host gaps between its kernels (served_util.sh's
PHASES=1; `host_gaps.py` reads them): run as a script, it wraps each method `TARGETS` names, once its module is imported, in
a range named ``Class.method``, then runs the pipeline's CLI with the script's arguments.

The patches wait for each module's own import (`_AfterImport`), since `verity_vllm.engine.env` sets the engine's environment
before vLLM is imported and some vLLM modules read it then. Only methods called a few times a step are wrapped, never a
module's forward: a range is a push and a pop, about a microsecond each.
"""
from __future__ import annotations

import ctypes
import functools
import importlib.abc
import sys
import threading
import types
from typing import Any

#: module -> {class: methods}
TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "vllm.v1.engine.llm_engine": {"LLMEngine": ("step", "add_request")},
    "vllm.v1.engine.core": {"EngineCore": ("step", "add_request", "preprocess_add_request", "post_step")},
    "vllm.v1.engine.output_processor": {"OutputProcessor": ("process_outputs", "add_request")},
    "vllm.v1.engine.input_processor": {"InputProcessor": ("process_inputs",)},
    "vllm.v1.core.sched.scheduler": {"Scheduler": ("schedule", "update_from_output", "add_request", "finish_requests",
                                                   "_update_after_schedule", "make_stats")},
    "vllm.v1.executor.abstract": {"Executor": ("execute_model", "sample_tokens")},
    "vllm.v1.worker.gpu_worker": {"Worker": ("execute_model", "sample_tokens")},
    "vllm.v1.worker.gpu.model_runner": {"GPUModelRunner": (
        "execute_model", "sample_tokens", "finish_requests", "free_states", "add_requests", "update_requests",
        "gather_batch_req_state", "prepare_inputs", "prepare_attn", "sample", "postprocess_sampled",
        "postprocess_num_computed_tokens")},
    "verity_vllm.pipeline.serve_openai": {"EngineLoop": ("_deliver", "_release", "_poll", "_take")},
    "verity_vllm.pipeline.serve_commit": {"CommitWindows": ("before_step", "stepped", "admitted", "streamed", "idle")},
    "verity_vllm.commit.committer.native_collect": {"NativeCollectCommitter": ("_begin_step", "_end_step", "_record_step_rows",
                                                                                "_plan_held", "step_targets")},
    "verity_vllm.commit.committer.native_host": {"NativeHostCommitter": ("step_min_bytes",)},
    "verity_vllm.commit.committer.window_views": {"WindowViews": ("_next_seq", "_window_view")},
    "verity_vllm.engine.step_trace": {"StepTrace": ("_record_step",)},
    **{f"verity_vllm.{m}": {"*": ("on_step_begin", "on_step_end")} for m in (
        "commit.served_record", "commit.committer.native_hidden", "acquire.sources.hidden_source", "acquire.sources.int8_source",
        "acquire.sources.kv_slots_source", "acquire.sources.linear_quant_source", "acquire.sources.narrowed_source",
        "acquire.sources.norm_scale_source", "acquire.sources.scale_products_source", "acquire.sources.vocab_range_source",
        "acquire.sources.attention_source", "acquire.sources.call_boundary_source")},
}

def _ranged(f: Any, label: str) -> Any:
    from torch.cuda import nvtx
    push, pop = nvtx.range_push, nvtx.range_pop

    @functools.wraps(f)
    def w(*a, **kw):
        push(label)
        try:
            return f(*a, **kw)
        finally:
            pop()
    return w


def patch(module: Any) -> None:
    """Wrap ``module``'s `TARGETS` methods (class ``*``: every class the module defines that has them), and say on stdout (the
    serve log) which it wrapped and which it has not."""
    want = TARGETS.get(module.__name__, {})
    if "*" in want:
        want = {c.__name__: want["*"] for c in vars(module).values() if isinstance(c, type) and c.__module__ == module.__name__
                and any(n in c.__dict__ for n in want["*"])}
    for cls_name, names in want.items():
        cls = getattr(module, cls_name, None)
        got = [n for n in names if cls is not None and isinstance(cls.__dict__.get(n), types.FunctionType)]
        for n in got:
            setattr(cls, n, _ranged(cls.__dict__[n], f"{cls_name}.{n}"))
        missing = sorted(set(names) - set(got))
        print(f"PHASES {module.__name__}.{cls_name}: ranges on {', '.join(got) or 'nothing'}"
              + (f"; no {', '.join(missing)}" if missing else ""), flush=True)


class _AfterImport(importlib.abc.MetaPathFinder):
    """Patches a `TARGETS` module right after its own code runs."""

    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            patch(module)
        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    for name in TARGETS:
        if name in sys.modules:
            patch(sys.modules[name])
    sys.meta_path.insert(0, _AfterImport())


def name_threads() -> None:
    """Say on stdout each Python thread's OS id and name as it starts, and give its OS thread that name (15 bytes, the
    kernel's limit; Nsight's ``ThreadNames``), so a thread in the profile is one in the code."""
    libc = ctypes.CDLL(None)
    boot = threading.Thread._bootstrap_inner  # type: ignore[attr-defined]

    def named(self: threading.Thread) -> None:
        libc.prctl(15, self.name.replace("ThreadPoolExecutor", "tpe").encode()[:15], 0, 0, 0)  # PR_SET_NAME
        print(f"PHASES thread {threading.get_native_id()} {self.name}", flush=True)
        boot(self)
    threading.Thread._bootstrap_inner = named  # type: ignore[attr-defined]
    print(f"PHASES thread {threading.get_native_id()} {threading.current_thread().name}", flush=True)


def main() -> Any:
    sys.path.pop(0)
    install()
    name_threads()
    from verity_vllm.pipeline import cli
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
