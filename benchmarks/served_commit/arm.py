"""Commit under served vLLM as `verity-bench` experimental arms (``--arm-module served_commit.arm:SERVED_COMMIT``, with
``benchmarks`` on ``PYTHONPATH`` as `benchmarks/harness/env.sh` puts it).

The arm's engine is the preset's, as vLLM serves it (``serving_defaults``: torch.compile, its CUDA-graph mode and
capture sizes, no batch invariance), with every value of every engine step committed:
`served_compiled_source.ServedCompiledSource` acquires them (Inductor wrapper hooks; under a CUDA graph, slot copies
captured into the graph and drained at the logits), and the native collector (`NativeCollectCommitter`, windowed
staging) hashes them into one root per step. Its control is ``stock`` token for token: committing must not change what
is served.

Each of the harness's runs is one Commit. ``begin_run`` resets the committer and installs it (before the clock),
``end_run`` finalizes the run's commitment (inside the timed window, so a run's time includes the hashing backlog), and
the run's checks happen after the window, before the next run or when the harness reads ``extras`` or ``gate``: the
collector's errors and, with the values retained, `linkage` (each request's committed input id at a step is the token
committed as its sample at the step before). The source's hooks are made through ``target.hooks``; the committer's
patches of the model runner are its own, installed by ``begin_run`` and removed by ``end_run``.

Two arms differ in what the Commit keeps for openings: ``served-commit`` copies the values to host memory (what an
opening reads, and what `linkage` checks); ``served-commit-noretain`` keeps only the authentication state, so its values
open by replay and its runs are checked by the collector's errors alone.

Env: ``VERITY_WINDOW_MB`` (default 256) and ``VERITY_WINDOW_SLOTS`` (default 8), the collector's device window ring.
"""
from __future__ import annotations

import os
from typing import Any


class ServedCommitArm:
    protocols: tuple[str, ...] = ()
    precision = "bfloat16"
    control = "stock"
    resident = True

    def __init__(self, id: str, retain: str) -> None:
        self.id, self.retain = id, retain
        self.source: Any = None
        self.committer: Any = None
        self.windows: dict[str, Any] = {}
        self.pending: Any = None
        self.runs: list[dict[str, Any]] = []

    def install(self, target: Any) -> ServedCommitArm:
        from verity_vllm.acquire.sources.served_compiled_source import ServedCompiledSource

        self.source = ServedCompiledSource(into=target.hooks)
        self.source.install_hooks()
        return self

    def ready(self, target: Any) -> None:
        from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

        os.environ["VERITY_RETAIN"] = self.retain
        os.environ.setdefault("VERITY_WINDOW_MB", "256")
        os.environ.setdefault("VERITY_WINDOW_SLOTS", "8")
        engine = target.engine._engine
        self.source.note_engine(engine.model)
        com = NativeCollectCommitter(engine, run_id=f"bench:{self.id}", arena_bytes=None, wrap=True, gather=True,
                                     native_hooks=True, auto_blocks=True, native_worker=True)
        self.source.attach(com)
        vc = engine.llm.llm_engine.vllm_config
        tokens = int(vc.scheduler_config.max_num_batched_tokens)
        sizing = com.size_arena({"batch": int(vc.scheduler_config.max_num_seqs), "max_prompt": tokens, "max_new": 1,
                                 "max_num_batched_tokens": tokens, "max_model_len": int(vc.model_config.max_model_len)})
        self.windows = com.enable_windows(sizing)
        self.committer = com

    def begin_run(self, target: Any) -> None:
        self._check()
        self.committer.reset()
        self.source.begin_run()
        self.committer.install()

    def end_run(self, target: Any) -> None:
        self.pending = self.committer.finalize()
        self.committer.uninstall()

    def _check(self) -> None:
        """The finished run's record: its commitment, the collector's errors and, with values retained, its linkage."""
        from verity_vllm.acquire.sources.served_compiled_source import linkage

        rc, self.pending = self.pending, None
        if rc is None:
            return
        com, stats = self.committer, self.source.stats
        link = linkage(com) if self.retain == "host" else None
        errors = list((rc.extra or {}).get("errors") or []) + list(self.source.errors)
        self.runs.append({
            "run_root": bytes(rc.run_root).hex(),
            "steps": len(com._layouts),
            "committed_bytes": sum(m.nbytes for metas in com._layouts.values() for m in metas),
            "replayed_steps": stats["replayed_steps"],
            "eager_steps": stats["eager_steps"],
            "drain_s": stats["drain_s"],
            "errors": errors[:5],
            "linkage": link,
            "passed": not errors and (link is None or link["passed"]),
        })

    def extras(self, target: Any) -> dict[str, Any]:
        self._check()
        run = self.runs[-1]
        out = {k: run[k] for k in ("steps", "committed_bytes", "replayed_steps", "eager_steps", "drain_s")}
        out["errors"] = len(run["errors"])
        if run["linkage"] is not None:
            out["linkage_checked"] = run["linkage"]["checked"]
            out["linkage_mismatches"] = run["linkage"]["mismatches"]
        return out

    def gate(self, target: Any) -> dict[str, Any]:
        self._check()
        failed = [r for r in self.runs if not r["passed"]]
        return {"passed": bool(self.runs) and not failed, "retain": self.retain, "runs": len(self.runs),
                "failed": failed[:3], "windows": self.windows, "source": self.source.summary()}

    def close(self) -> None:
        if self.committer is not None:
            self.committer.close()


SERVED_COMMIT = ServedCommitArm("served-commit", retain="host")
SERVED_COMMIT_NORETAIN = ServedCommitArm("served-commit-noretain", retain="none")
