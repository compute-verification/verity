"""The Commit, sampled proofs' commitment of every served value, as experimental arms: a spec names
``verity_bench.vllm.commit_arm:COMMIT`` (``commit``, eager) or ``verity_bench.vllm.commit_arm:COMMIT_COMPILED``
(``commit@compiled``, at vLLM's compiled serving defaults) in ``arm_modules`` and the arm's id in ``arms``.

``sampled-proofs:rows-sha512`` can't be a protocol arm: `verity_vllm.LLM` builds its options at stage ``serve``, where sampled
proofs refuses (`protocol_options.sampled_proofs.refusals`: it runs in a row's Commit phase). So these arms install the Commit's
committer on the served engine themselves, configured as a row's Commit phase runs it (`config.CommitConfig`,
`pipeline.commit.make_committer`): ``native_collect_v2b`` (gather windows, size-adaptive blocks, native worker, GPU tree, fast
copy) over a ring of 8 x 256 MiB device windows, with bounded learning, under the committer's default scheme, which each arm
requires: ``rows-sha512`` (`verity_vllm.commit.step_rows`). Each 4 KiB chunk of a step's stream is a row, committed as an
hm96-sha512 hiding leaf (salt from the step's 64 fresh bytes of OS randomness, by SHA-512) in a frame-v3-sha512 tree, with 64-byte
step and run roots: the deployed form.

``commit`` acquires through the committer's native module-boundary hooks, Python-level wrappers that a compiled graph never
runs, so its engine is eager (batch invariant under `verity_vllm.LLM`) with stock FP8 as its token control. ``commit@compiled``
acquires through `served_compiled_source.ServedCompiledSource` (Inductor wrapper hooks; under a CUDA graph, slot copies captured
into the graph and drained at the logits), as `benchmarks/served_commit` does, so its engine is vLLM's compiled serving
(``execution = "compiled"``) with ``stock-fp8@compiled`` as its token control: the served Commit against served stock.

What differs from a row:

* what is committed is the acquisition's selection (``commit``: `native_host.ACQUIRE_CLASSES`, every linear, norm, activation,
  rotary, attention and embedding output, the embedding's ids and the sampled ids, every step; ``commit@compiled``: every
  compiled piece's outputs, the token inputs and the logits), not a Build's required-value manifest, and the FA2 hidden-stream
  tap is not attached;
* nothing raw is retained (``retain none``: values open by replay), and on the windowed worker that keeps no Merkle level on
  host either (`NativeCollectCommitter.retain_levels`). A committer that keeps levels (a quarter of a row per 4 KiB row's two
  64-byte nodes, held for the whole run) refuses a run whose levels can't fit the job's host memory (its cgroup's ``memory.max``)
  before it starts, by name, rather than being killed partway through.

The weights' registration is once per engine and outside steady state, so it isn't run. Each harness run is one Commit:
``begin_run`` resets and installs the committer before the clock starts, and ``end_run`` finalizes inside the timed run, so
a run's time includes the hashing backlog. The gate and warm-up runs are learn-only passes: a step whose plan is new and whose
host copy exceeds the learning budget is discarded and leaves that run without a root. A measured run must end with a root
and no collector or source error (and for ``commit``, every expected tensor committed), or ``end_run`` raises and the point
fails. The custom gate reads every run so far.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.engine.env import apply_target

#: the row Commit's staging (`config.CommitConfig`): 256 MiB windows, 8 slots; learning bounded, nothing raw retained
ENV = {"VERITY_WINDOW_MB": "256", "VERITY_WINDOW_SLOTS": "8", "VERITY_STAGING_BOUNDED": "1", "VERITY_RETAIN": "none"}
#: the deployed commitment scheme (`step_rows.NAME`), which an arm refuses to run without
SCHEME = "rows-sha512"
LEARN_ONLY = ("gate", "warmup")
#: committed bytes per byte of Merkle levels a committer that keeps them holds on host: a 4 KiB row's 64-byte node, and the
#: tree above the leaves doubles that
LEVELS_DIVISOR = 4096 // (2 * 64)
CGROUP_ROOT = Path("/sys/fs/cgroup")


def host_memory(root: Path = CGROUP_ROOT, self_cgroup: Path = Path("/proc/self/cgroup")) -> tuple[int, int] | None:
    """The tightest ``memory.max`` on this process's cgroup v2 path and that cgroup's anonymous memory (page cache, which
    the checkpoint's load leaves behind, is reclaimable); None without one."""
    try:
        rel = next(line.split("::", 1)[1].strip() for line in self_cgroup.read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration):
        return None
    best, path = None, root / rel.lstrip("/")
    while True:
        try:
            limit = (path / "memory.max").read_text().strip()
            if limit != "max" and (best is None or int(limit) < best[0]):
                stat = dict(line.split() for line in (path / "memory.stat").read_text().splitlines())
                best = (int(limit), int(stat["anon"]))
        except (OSError, ValueError, KeyError):
            pass
        if path == root or root not in path.parents:
            return best
        path = path.parent


def committer_engine(llm: Any) -> SimpleNamespace:
    """The engine `NativeCollectCommitter` takes (as `verity_vllm.engine.build.Engine`), from the public names of the
    harness's ``target.engine`` (``verity_vllm.LLM``, or for a local checkpoint the runner's stand-in, which has no pin):
    the ``vllm.LLM``, its in-process model runner, the served model, and the checkpoint's repo and revision (into its
    geometry digest), which are the pin's, or for a local checkpoint vLLM's model path with no revision."""
    from verity_vllm.engine import handle

    pin = getattr(llm, "pin", None)
    mc = llm.vllm.llm_engine.vllm_config.model_config
    cp = {"repo": pin.model, "revision": pin.revision} if pin is not None else {"repo": mc.model, "revision": mc.revision}
    return SimpleNamespace(llm=llm.vllm, runner=handle.model_runner(llm.vllm), model=llm.model, cp=cp)


class CommitArm:
    protocols: tuple[str, ...] = ()
    control = "stock-fp8"
    resident = True
    execution: str | None = None

    def __init__(self, id: str) -> None:
        self.id = id
        self.committer: Any = None
        self.windows: dict[str, Any] = {}
        self.per_token_bytes = 0
        self.runs: list[dict[str, Any]] = []

    def install(self, target: Any) -> CommitArm:
        return self

    def engine(self, target: Any) -> Any:
        return committer_engine(target.engine)

    def ready(self, target: Any) -> None:
        from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

        apply_target(ENV)
        engine = self.engine(target)
        com = NativeCollectCommitter(engine, run_id=f"bench:{self.id}", arena_bytes=None, wrap=True, gather=True,
                                     native_hooks=True, auto_blocks=True, native_worker=True)
        if not (com.windowed and com.bounded and com.retain == "none"):
            raise RuntimeError(f"{self.id} arm: the committer did not take the row Commit's staging {ENV}")
        if com.commit_scheme != SCHEME:
            raise RuntimeError(f"{self.id} arm: the committer commits {com.commit_scheme!r}, not the deployed {SCHEME!r}")
        self.attach(com, engine)
        vc = engine.llm.llm_engine.vllm_config
        tokens = int(vc.scheduler_config.max_num_batched_tokens)
        sizing = com.size_arena({"batch": int(vc.scheduler_config.max_num_seqs), "max_prompt": tokens, "max_new": 1,
                                 "max_num_batched_tokens": tokens, "max_model_len": int(vc.model_config.max_model_len)})
        self.windows = com.enable_windows(sizing)
        self.per_token_bytes = int(sizing.get("per_token_bytes") or 0)
        self.committer = com

    def attach(self, com: Any, engine: Any) -> None:
        pass

    def admit(self, tokens: int) -> None:
        """Refuse a run whose host-kept levels can't fit the job's memory."""
        if not getattr(self.committer, "retain_levels", True):
            return
        mem = host_memory()
        if mem is None or not self.per_token_bytes:
            return
        limit, used = mem
        need = tokens * self.per_token_bytes // LEVELS_DIVISOR
        if need > limit - used:
            raise RuntimeError(
                f"{self.id} arm: refused before the run: {tokens} tokens at {self.per_token_bytes} committed bytes per token keep "
                f"{need / 2**30:.1f} GiB of Merkle levels on host, and the job has {(limit - used) / 2**30:.1f} GiB of its "
                f"{limit / 2**30:.0f} GiB free: run this point in a job with more host memory, or with a Commit that keeps no "
                "levels on host")

    def begin_run(self, target: Any) -> None:
        self.admit(target.tokens)
        com = self.committer
        com.reset()
        com.learn_only = target.phase in LEARN_ONLY
        com.install()

    def run_errors(self, rc: Any) -> list[str]:
        extra = rc.extra or {}
        errors = list(extra.get("errors") or [])
        expected, committed = int(extra.get("expected_tensors", 0)), int(extra.get("committed_tensors", 0))
        if expected != committed:
            errors.append(f"{committed} of {expected} expected tensors committed")
        return errors

    def end_run(self, target: Any) -> None:
        com = self.committer
        try:
            rc = com.finalize()
        finally:
            com.uninstall()
        extra = rc.extra or {}
        bounded = extra.get("bounded_staging") or {}
        learned = len(bounded.get("learned_only_steps") or ())
        errors = self.run_errors(rc)
        if target.phase == "measure" and (not rc.run_root or learned):
            errors.append(f"no run root ({learned} learned-only steps) on a measured run")
        run = {"run": f"{target.phase}:{target.point}", "phase": target.phase, "run_root": bytes(rc.run_root).hex(),
               "steps": len(rc.steps), "committed_bytes": int(extra.get("values_retained_bytes", 0)),
               "layout_bytes": sum(m.nbytes for metas in getattr(com, "_layouts", {}).values() for m in metas),
               "expected_tensors": int(extra.get("expected_tensors", 0)),
               "committed_tensors": int(extra.get("committed_tensors", 0)), "learned_only_steps": learned,
               "root_ready_delay_s": float(rc.root_ready_delay_s), "finalize_s": float(rc.t_finalize_s),
               "host_retained_bytes": int(extra.get("host_retained_bytes", 0)),
               "pinned_pool_bytes": int(extra.get("pinned_pool_bytes", 0)),
               "plan_mismatches": int(bounded.get("plan_mismatches", 0)), "errors": errors[:5]}
        self.runs.append(run)
        if errors and target.phase == "measure":
            raise RuntimeError(f"{self.id} arm: measured run {run['run']} failed: {errors[:3]}")

    def extras(self, target: Any) -> dict[str, Any]:
        run = self.runs[-1]
        return {k: run[k] for k in ("steps", "committed_bytes", "layout_bytes", "expected_tensors", "committed_tensors", "root_ready_delay_s", "finalize_s",
                                    "host_retained_bytes", "pinned_pool_bytes", "plan_mismatches")}

    def gate(self, target: Any) -> dict[str, Any]:
        failed = [r for r in self.runs if r["errors"]]
        com = self.committer
        return {"passed": bool(self.runs) and not failed, "runs": [{k: r[k] for k in ("run", "run_root", "steps",
                "committed_bytes", "learned_only_steps", "errors")} for r in self.runs], "windows": self.windows,
                "per_token_bytes": self.per_token_bytes, "committer": com.name if com is not None else None,
                "commit_scheme": getattr(com, "commit_scheme", None)}

    def close(self) -> None:
        if self.committer is not None:
            self.committer.close()


class CompiledCommitArm(CommitArm):
    """The Commit on vLLM's compiled serving: `ServedCompiledSource` acquires every compiled piece's values for the committer."""

    control = "stock-fp8@compiled"
    execution = "compiled"

    def __init__(self, id: str) -> None:
        super().__init__(id)
        self.source: Any = None

    def install(self, target: Any) -> CompiledCommitArm:
        from verity_vllm.acquire.sources.served_compiled_source import ServedCompiledSource

        self.source = ServedCompiledSource(into=target.hooks)
        self.source.install_hooks()
        return self

    def attach(self, com: Any, engine: Any) -> None:
        self.source.note_engine(engine.model)
        self.source.attach(com)

    def begin_run(self, target: Any) -> None:
        self.source.begin_run()
        super().begin_run(target)

    def run_errors(self, rc: Any) -> list[str]:
        """The collector's errors and every source error so far, its captures' included (they happen while the engine builds).
        Its acquisitions are the source's slot drains, so the collector's tensor counters are recorded, not gated on, as
        `benchmarks/served_commit`'s arm treats them."""
        return list((rc.extra or {}).get("errors") or []) + list(self.source.errors)

    def extras(self, target: Any) -> dict[str, Any]:
        stats = self.source.stats
        return {**super().extras(target), **{k: stats.get(k) for k in ("replayed_steps", "eager_steps", "drain_s")}}

    def gate(self, target: Any) -> dict[str, Any]:
        return {**super().gate(target), "source": self.source.summary() if self.source is not None else None}


COMMIT = CommitArm("commit")
COMMIT_COMPILED = CompiledCommitArm("commit@compiled")
