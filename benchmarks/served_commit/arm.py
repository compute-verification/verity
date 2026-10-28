"""The base of the `verity-bench` experimental arms that Commit under served vLLM, `prof_arm`'s
(``--arm-module served_commit.prof_arm:SERVED_COMMIT_ROWS``, with ``benchmarks`` on ``PYTHONPATH`` as
`benchmarks/harness/env.sh` puts it).

An arm's engine is the preset's, as vLLM serves it (``serving_defaults``: torch.compile, its CUDA-graph mode and
capture sizes, no batch invariance), with every value of every engine step committed:
`served_compiled_source.ServedCompiledSource` acquires them (Inductor wrapper hooks; under a CUDA graph, slot copies
captured into the graph and drained at the logits), and the native collector (`NativeCollectCommitter`, windowed
staging) hashes them into one root per step. Its control is ``stock`` token for token: committing must not change what
is served.

An arm commits under its ``commit_scheme``, and ``install`` refuses the default, ``vllm-v1`` (SHA-256), by name, so an
arm that runs names ``rows-sha512``; `benchmarks/g2/curve.py` reads a recorded run's scheme from its arm's name. The Commit is
what `prof_arm`'s rows arms and the harness's ``commit`` arms (`verity_bench.vllm.commit_arm`) measure.

Each of the harness's runs is one Commit. ``begin_run`` resets the committer and installs it (before the clock),
``end_run`` finalizes the run's commitment (inside the timed period, so a run's time includes the hashing backlog), and
the run's checks happen after that period, before the next run or when the harness reads ``extras`` or ``gate``: the
collector's errors and, with the values retained, `linkage` (each request's committed input id at a step is the token
committed as its sample at the step before). The source's hooks are made through ``target.hooks``; the committer's
patches of the model runner are its own, installed by ``begin_run`` and removed by ``end_run``.

An arm's ``retain`` is what the Commit keeps for openings: ``host`` copies the values to host memory (what an opening
reads, and what `linkage` checks); ``none`` keeps only the authentication state, so its values open by replay and its runs
are checked by the collector's errors alone.

Staging is the collector's default: a ring of 256 MiB device windows, bounded (a step whose plan is new copies at most
``VERITY_LEARN_HOST_BUDGET_MB`` to host to re-hash), and tree levels only when values are kept. The harness's gate and warm-up
runs are learn-only passes: a new-plan step over that budget is discarded, which leaves the run without a root. The custom gate
accepts such a run only as a declared warm-up, named by its phase and sweep point in the verdict's ``warmups``; a measured run
without a root fails it. The harness reads the gate before timing (``arm``) and again after the warm-up and timed runs
(``arm-closing``), so the warm-ups and measured runs are in the verdict too.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

WARMUPS = ("gate", "warmup")
V1 = "vllm-v1"


def committer_engine(llm: Any) -> SimpleNamespace:
    """The engine `NativeCollectCommitter` takes (as `verity_vllm.engine.build.Engine`), from the public surface of the
    harness's ``target.engine``: ``verity_vllm.LLM``, or for a local checkpoint the harness's stand-in, which has no pin.
    The committer reads four things of it: the ``vllm.LLM``, its in-process model runner, the served model, and the
    checkpoint's repo and revision (into its geometry digest), which are the pin's, or for a local checkpoint vLLM's model
    path with no revision."""
    from verity_vllm.engine import handle

    pin = getattr(llm, "pin", None)
    mc = llm.vllm.llm_engine.vllm_config.model_config
    cp = {"repo": pin.model, "revision": pin.revision} if pin is not None else {"repo": mc.model, "revision": mc.revision}
    return SimpleNamespace(llm=llm.vllm, runner=handle.model_runner(llm.vllm), model=llm.model, cp=cp)


class ServedCommitArm:
    protocols: tuple[str, ...] = ()
    precision = "bfloat16"
    control = "stock"
    resident = True
    commit_scheme = V1
    batch_copies = True

    def __init__(self, id: str, retain: str) -> None:
        self.id, self.retain = id, retain
        self.source: Any = None
        self.committer: Any = None
        self.windows: dict[str, Any] = {}
        self.pending: Any = None
        self.run_name, self.learn_only = "", False
        self.runs: list[dict[str, Any]] = []

    def install(self, target: Any) -> ServedCommitArm:
        if self.commit_scheme == V1:
            raise RuntimeError(f"{self.id} arm commits {V1!r} (SHA-256), not 'rows-sha512': vllm-v1 is kept only to read recorded "
                               "runs, so measure the Commit with prof_arm's served-commit-rows* arms")
        from verity_vllm.acquire.sources.served_compiled_source import ServedCompiledSource

        self.source = ServedCompiledSource(into=target.hooks, batch_copies=self.batch_copies)
        self.source.install_hooks()
        return self

    def ready(self, target: Any) -> None:
        from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

        os.environ["VERITY_RETAIN"] = self.retain
        llm = target.engine
        self.source.note_engine(llm.model)
        com = NativeCollectCommitter(committer_engine(llm), run_id=f"bench:{self.id}", arena_bytes=None, wrap=True, gather=True,
                                     native_hooks=True, auto_blocks=True, native_worker=True, commit_scheme=self.commit_scheme)
        self.source.attach(com)
        vc = llm.vllm.llm_engine.vllm_config
        tokens = int(vc.scheduler_config.max_num_batched_tokens)
        sizing = com.size_arena({"batch": int(vc.scheduler_config.max_num_seqs), "max_prompt": tokens, "max_new": 1,
                                 "max_num_batched_tokens": tokens, "max_model_len": int(vc.model_config.max_model_len)})
        self.windows = com.enable_windows(sizing)
        self.committer = com

    def begin_run(self, target: Any) -> None:
        self._check()
        self.committer.reset()
        self.run_name = f"{target.phase}:{target.point}"
        self.learn_only = self.committer.learn_only = bool(self.committer.bounded) and target.phase in WARMUPS
        self.source.begin_run()
        self.committer.install()

    def end_run(self, target: Any) -> None:
        self.pending = self.committer.finalize()
        self.committer.uninstall()

    def _check(self) -> None:
        """The finished run's record: its commitment, the collector's errors and, with values retained, its linkage."""
        rc, self.pending = self.pending, None
        if rc is None:
            return
        com, stats = self.committer, self.source.stats
        link = None
        if self.retain == "host":
            from verity_vllm.acquire.sources.served_compiled_source import linkage

            link = linkage(com)
        errors = list((rc.extra or {}).get("errors") or []) + list(self.source.errors)
        learned = len(((rc.extra or {}).get("bounded_staging") or {}).get("learned_only_steps") or ())
        rootless = not rc.run_root or learned > 0
        if rootless and not self.learn_only:
            errors.append(f"run {self.run_name} has no root ({learned} learned-only steps) and is not a declared warm-up")
        self.runs.append({
            "run": self.run_name,
            "run_root": bytes(rc.run_root).hex(),
            "learned_only_steps": learned,
            "steps": len(com._layouts),
            "committed_bytes": sum(m.nbytes for metas in com._layouts.values() for m in metas),
            "replayed_steps": stats["replayed_steps"],
            "eager_steps": stats["eager_steps"],
            "drain_s": stats["drain_s"],
            "errors": errors[:5],
            "linkage": link,
            "warmup": rootless and self.learn_only,
            "passed": not errors and (link is None or link["passed"]),
        })

    def extras(self, target: Any) -> dict[str, Any]:
        self._check()
        run = self.runs[-1]
        out = {k: run[k] for k in ("steps", "committed_bytes", "replayed_steps", "eager_steps", "drain_s", "learned_only_steps")}
        out["errors"] = len(run["errors"])
        if run["linkage"] is not None:
            out["linkage_checked"] = run["linkage"]["checked"]
            out["linkage_mismatches"] = run["linkage"]["mismatches"]
        return out

    def gate(self, target: Any) -> dict[str, Any]:
        self._check()
        failed = [r for r in self.runs if not r["passed"]]
        warmups = [{k: r[k] for k in ("run", "learned_only_steps")} for r in self.runs if r["warmup"]]
        return {"passed": bool(self.runs) and not failed, "retain": self.retain, "runs": len(self.runs),
                "failed": failed[:3], "warmups": warmups, "windows": self.windows, "source": self.source.summary()}

    def close(self) -> None:
        if self.committer is not None:
            self.committer.close()
