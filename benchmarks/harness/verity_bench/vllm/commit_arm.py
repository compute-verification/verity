"""The Commit, sampled proofs' commitment of every served value, as an experimental arm: a spec names
``verity_bench.vllm.commit_arm:COMMIT`` in ``arm_modules`` and ``commit`` in ``arms``.

``sampled-proofs:vllm-v1`` can't be a protocol arm: `verity_vllm.LLM` builds its options at stage ``serve``, where sampled
proofs refuses (`protocol_options.sampled_proofs.refusals`: it runs in a row's Commit stage). So this arm installs the Commit's
committer on the served engine itself, configured as a row's Commit stage runs it (`config.CommitConfig`,
`pipeline.commit.make_committer`): ``native_collect_v2b`` (native hooks, gather windows, size-adaptive blocks, native worker,
GPU tree, fast copy) over a ring of 8 x 256 MiB device windows, with bounded learning. Two things differ from a row:

* what is committed is the committer's class selection (`native_host.ACQUIRE_CLASSES`: every linear, norm, activation,
  rotary, attention and embedding output, the embedding's ids and the sampled ids, every step), not a Build's required-value
  manifest, and the FA2 hidden-stream tap is not attached;
* nothing raw is retained (``retain none``: values open by replay); the Merkle levels go to pinned host memory.

The weights' registration is once per engine and outside steady state, so it isn't run. Each harness run is one Commit:
``begin_run`` resets and installs the committer before the clock starts, and ``end_run`` finalizes inside the timed window, so
a run's time includes the hashing backlog. The gate and warm-up runs are learn-only passes: a step whose plan is new and whose
host copy exceeds the learning budget is discarded and leaves that run without a root. A measured run must end with a root,
every expected tensor committed and no collector error, or ``end_run`` raises and the point fails. The custom gate reads the
gate runs: no collector error, every expected tensor committed.
"""

from __future__ import annotations

import os
from typing import Any

#: the row Commit's staging (`config.CommitConfig`): 256 MiB windows, 8 slots; learning bounded, nothing raw retained
ENV = {"VERITY_WINDOW_MB": "256", "VERITY_WINDOW_SLOTS": "8", "VERITY_STAGING_BOUNDED": "1", "VERITY_RETAIN": "none"}
LEARN_ONLY = ("gate", "warmup")


class CommitArm:
    protocols: tuple[str, ...] = ()
    control = "stock-fp8"
    resident = True

    def __init__(self, id: str) -> None:
        self.id = id
        self.committer: Any = None
        self.windows: dict[str, Any] = {}
        self.runs: list[dict[str, Any]] = []

    def install(self, target: Any) -> CommitArm:
        return self

    def ready(self, target: Any) -> None:
        from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

        os.environ.update(ENV)
        engine = target.engine._engine
        com = NativeCollectCommitter(engine, run_id=f"bench:{self.id}", arena_bytes=None, wrap=True, gather=True,
                                     native_hooks=True, auto_blocks=True, native_worker=True)
        if not (com.windowed and com.bounded and com.retain == "none"):
            raise RuntimeError(f"commit arm: the committer did not take the row Commit's staging {ENV}")
        vc = engine.llm.llm_engine.vllm_config
        tokens = int(vc.scheduler_config.max_num_batched_tokens)
        sizing = com.size_arena({"batch": int(vc.scheduler_config.max_num_seqs), "max_prompt": tokens, "max_new": 1,
                                 "max_num_batched_tokens": tokens, "max_model_len": int(vc.model_config.max_model_len)})
        self.windows = com.enable_windows(sizing)
        self.committer = com

    def begin_run(self, target: Any) -> None:
        com = self.committer
        com.reset()
        com.learn_only = target.phase in LEARN_ONLY
        com.install()

    def end_run(self, target: Any) -> None:
        com = self.committer
        try:
            rc = com.finalize()
        finally:
            com.uninstall()
        extra = rc.extra or {}
        bounded = extra.get("bounded_staging") or {}
        learned = len(bounded.get("learned_only_steps") or ())
        expected, committed = int(extra.get("expected_tensors", 0)), int(extra.get("committed_tensors", 0))
        errors = list(extra.get("errors") or [])
        if expected != committed:
            errors.append(f"{committed} of {expected} expected tensors committed")
        if target.phase == "measure" and (not rc.run_root or learned):
            errors.append(f"no run root ({learned} learned-only steps) on a measured run")
        run = {"run": f"{target.phase}:{target.point}", "phase": target.phase, "run_root": bytes(rc.run_root).hex(),
               "steps": len(rc.steps), "committed_bytes": int(extra.get("values_retained_bytes", 0)),
               "committed_tensors": committed, "learned_only_steps": learned,
               "root_ready_delay_s": float(rc.root_ready_delay_s), "finalize_s": float(rc.t_finalize_s),
               "host_retained_bytes": int(extra.get("host_retained_bytes", 0)),
               "pinned_pool_bytes": int(extra.get("pinned_pool_bytes", 0)),
               "plan_mismatches": int(bounded.get("plan_mismatches", 0)), "errors": errors[:5]}
        self.runs.append(run)
        if errors and target.phase == "measure":
            raise RuntimeError(f"commit arm: measured run {run['run']} failed: {errors[:3]}")

    def extras(self, target: Any) -> dict[str, Any]:
        run = self.runs[-1]
        return {k: run[k] for k in ("steps", "committed_bytes", "committed_tensors", "root_ready_delay_s", "finalize_s",
                                    "host_retained_bytes", "pinned_pool_bytes", "plan_mismatches")}

    def gate(self, target: Any) -> dict[str, Any]:
        failed = [r for r in self.runs if r["errors"]]
        return {"passed": bool(self.runs) and not failed, "runs": [{k: r[k] for k in ("run", "run_root", "steps",
                "committed_bytes", "learned_only_steps", "errors")} for r in self.runs], "windows": self.windows,
                "committer": self.committer.name if self.committer is not None else None}

    def close(self) -> None:
        if self.committer is not None:
            self.committer.close()


COMMIT = CommitArm("commit")
