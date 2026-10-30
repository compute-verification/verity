"""A tensor-parallel row's stages under `verity-vllm row` (any world size >= 2): precheck -> Build -> Match -> Commit.

* **Build** (CPU): per rank the step Program and one request Program per declared (prompt_len, cap - 1) shape
  (``build --tp W --rank r``), the requests first, then GP-01 per rank; ``--build-jobs`` wide (auto, the default: as many derives as the
  memory holds, at most ranks x (shapes + 1), as the single-GPU Build sizes them; the composes likewise, at most one per rank).  A PASS Build
  also builds the required-value manifest over the W ranks (``manifest build-global --programs-root build/rank<r>``) under a limit
  of its own (``--manifest-timeout``, by default scaled with its W x declared requests component builds; ``--build-timeout`` bounds
  one derive and one compose), and
  the row stops (exit 21) when it names call boundaries (``Row.call_boundaries``: a TP Commit attaches no call-boundary source).
  With ``--build-relabel on`` one template rank's request derives write the other ranks' (`pipeline.tp.relabel_build`); with
  ``--build-programs-only 1`` the Build ends at GP-01, with no manifest and no call-boundary check (`pipeline.neutrality`'s heavy tier).
  A switched row (its `aot` in ``manifests/records.json``, `config_record.aot_of`) has the AOT Build: each derive is
  ``predict emit`` (`verity_vllm.program.predict.aot`), the predicted Program with the data its ``aot.trace`` Build holds about it
  (``--aot-trace``, else fetched from the store), and refuses where they differ; ``--build-construction trace`` traces it.
* **Match**: the capture under the observer and collective recorder, the ``--no-observer`` control, the collective-level
  check (``tp-match --arm check``); a request that stopped before its cap gets its served-shape Program derived per rank
  and GP-01 recomposed (``eos-truncated``); then, with a rank profile (default ``derived_<ROLE>_tp<W>``; ``--fold-profile ''``
  = collective-level only, named), the per-rank fold Match (``tp-fold-match``), its concurrent stages sized to the run's ``--mem-gb``
  (`TpRow.fold_sizing`).  A leased item's capture and control each hold the
  task's GPUs from right before their engine to their exit (``Row.run_gpu``); everything after them runs on the CPU, unleased.
  ``--match-phase gpu`` runs the capture and control alone (no Build needed: they can overlap it), ``--match-phase cpu`` the rest.
* **Commit**: the manifest over the W ranks, complete, then ``tp-commit`` with the windowed staging in its environment
  (``VERITY_WINDOW_MB`` / ``VERITY_WINDOW_SLOTS`` / ``VERITY_RETAIN``, and ``VERITY_STAGING_BOUNDED``, on in a config run only: flags it does not have) and ``--require-fold-match``
  when a fold Match is on disk.  The ranks run the guarded-max FA2 tap build and every tap the manifest names the family of
  (``row_records.applicable_taps``), as the single-GPU row does.  Each rank's weights registration hashes on its GPU
  (``VERITY_WEIGHTS_HASH=device`` unless the item sets ``host`` or ``device-checked``).  A leased item's ``tp-commit`` holds the task's
  GPUs from right before its engine to its last run's finalize (``Row.run_gpu``, `pipeline.tp.lease`).
  With ``--replay-deferred 1`` every rank seals a replay bundle in place of its C2 replay and the Commit is PENDING until the
  **replay** stage (CPU, ``tp-commit-replay``) replays every rank's bundle and decides it.

A **config run** (``--config-run 1``): the Build, the strict word check of its
manifest over the W ranks' components, ONE instrumented ``tp-commit`` (``--pairs 1 --warmup 1 --only-arm instrumented --replay-k K``:
K units over the world, split across the ranks; bounded staging) with no Match directory, no fold Match and no value check, and
``config_record.json`` (``pipeline.config_record``, per rank and in total), written however the run ends.

Evidence: ``build/rank<r>/``, ``match/``, ``manifest.json``, ``commit/`` under the row dir.  Exit codes: those of the
single-GPU row (``row_stages``).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from verity_vllm.config import BuildConfig, CommitConfig, MatchConfig, RunEnv
from verity_vllm.pipeline import config_record as CR
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline import store_data as SD
from verity_vllm.pipeline.row_driver import Row, Stop, stage_on

TP_UTIL = "0.6"


def _tail_bytes(path: str, n: int = 200) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n))
            return f.read().decode(errors="replace").replace("\n", " ")
    except OSError:
        return ""


def _grep(path: str, pred: Any) -> list[str]:
    try:
        return [ln.rstrip("\n") for ln in open(path, errors="replace") if pred(ln)]
    except OSError:
        return []


class TpRow(Row):
    def __init__(self, *, row: str, role: str, repo: str, rev: str, stages: str, env: RunEnv, world: int, gpu_util: str | None,
                 build: BuildConfig, match: MatchConfig, commit: CommitConfig):
        super().__init__(row=row, role=role, repo=repo, rev=rev, stages=stages, env=env, proc="tp_stage")
        self.W = world
        self.gpu_util = gpu_util or self.workload_gpu_util() or TP_UTIL
        self.commit_util = commit.commit_gpu_util or self.gpu_util
        self.cb, self.cm, self.ck_ = build, match, commit
        self.trap = False
        self.config_run = commit.config_run == "1"
        self.replay_pending = False
        self.lctx: dict[tuple[int, int], list[str] | None] = {}
        self.aot: dict[str, Any] | None = None

    def workload_gpu_util(self) -> str | None:
        """The workload's ``gpu_memory_utilization``, for a model whose weights don't fit `TP_UTIL` (GLM-5.3 TP8, about 88 GB a
        GPU).  An explicit ``--gpu-util`` / ``GPU_UTIL`` wins over it."""
        v = (R.load_json(self.path(self.wl)) or {}).get("gpu_memory_utilization")
        return None if v is None else str(v)

    def run(self) -> int:
        try:
            self.prepare()
            if stage_on(self.stages, "build"):
                self.build()
                if self.config_run:
                    RC.word_check(self)
            if stage_on(self.stages, "match") and not self.config_run:
                self.match()
            if stage_on(self.stages, "commit"):
                self.commit()
            if stage_on(self.stages, "replay"):
                self.replay()
                self.replay_pending = False
        except Stop as s:
            if self.config_run and self.trap:
                RC.config_record(self)
            return s.rc
        finally:
            if self.trap:
                self.research_result()
        if self.replay_pending:
            self.log("Commit PENDING: every rank's C2 replay runs on CPU from its sealed bundle (`row stage replay`, `tp-commit-replay`)")
        elif self.config_run:
            RC.config_record(self)
        self.tl("row", "end", ("ok", "true"))
        self.timeline_report()
        self.log("DONE")
        return 0

    # ------------------------------------------------------------------------------------------------ precheck
    def prepare(self) -> None:
        env = self.env
        self.ncb = env.native_collect_build or env.native_collect_default
        self.setup({"MATCH_IMPL": self.cm.match_impl, "MATCH_PIPELINE": self.cm.match_pipeline, "NATIVE_COLLECT_BUILD": self.ncb,
                    "VERITY_INSTANCES_FORM": self.cm.instances_form})
        if not os.path.isfile(self.path(self.wl)):
            print(f"no manifest {self.wl}")
            raise Stop(2)
        self.nccl = self.p2p_env()
        self.hidden_so = self.hidden_build()
        self.bdir, self.M = f"{self.d}/build", f"{self.d}/match"
        for p in (self.d, self.bdir, self.M, self.ncb):
            os.makedirs(p, exist_ok=True)
        self.refuse_unserved_engine()
        self.trap = True
        rel_line, _ = self.release_line("")
        tag, bar, rest = rel_line.partition("|")
        self.verdict("release", tag, f"{rest if bar else rel_line} tp={self.W}", quiet=True)
        self.tl("precheck", "start")
        sha = self.export_commit()
        if self.source_identity([self.py], lambda py: f"{self.d}/source_identity.json"):
            self.tl("precheck", "end", ("rc", 3), ("fail_class", "PRECHECK_FAIL_SOURCE_IDENTITY"))
            self.verdict("precheck", "FAIL", f"PRECHECK_FAIL_SOURCE_IDENTITY (see {self.d}/source_identity.json and row.log)")
            raise Stop(3)
        self.tl("precheck", "end", ("rc", 0))
        self.verdict("precheck", "PASS", f"source identity == {self.tree}{' @ ' + sha if sha else ''} (verity verity_vllm hidden_gpu research)", quiet=True)
        rc, line = self.target_family(*([] if self.reads_device() else ["--no-device"]), out=f"{self.d}/target_family.json")
        if rc != 0:
            self.verdict("precheck-target", "FAIL", line)
            raise Stop(3)
        self.verdict("precheck-target", "PASS", line, quiet=True)
        s = R.tp_shape(self.path(self.wl), self.path("manifests/checkpoints.json"), env.hf_home)
        self.B, self.I, self.O, self.shapes, self.ck, self.requests = s["B"], s["I"], s["O"], s["shapes"], s["ck"], s["requests"]
        self.MML, self.MNBT = s["MML"], s["MNBT"]
        tgt = R.declared_target(self.path(self.wl), row=self.row)
        self.tgt_args = ["--target", tgt] if tgt else []
        fp = self.cm.fold_profile
        self.fold_profile = f"derived_{self.role}_tp{self.W}" if fp is None else fp
        self.engine = self.engine_args()
        code9 = self.code9 = sha[:9]
        shapes = " ".join(f"{lp},{t}" for lp, t in self.shapes)
        self.log(f"start tp={self.W} B={self.B} I={self.I} O={self.O} MML={self.MML} mnbt={self.MNBT} shapes=[{shapes}] ckpt={self.ck or '-'} "
                 f"stages={self.stages} fold_profile={self.fold_profile or 'none'} gpu_util={self.gpu_util} hidden_so={self.hidden_so} code={code9} "
                 f"NCCL_P2P_DISABLE={self.nccl or 'unset'} {self.engine[-1]}")
        self.tl("row", "start", ("stages", self.stages), ("B", self.B), ("tp", self.W), ("code", code9), ("pod", os.uname().nodename))
        self.install_signal_forwarding()
        if not os.path.isdir(self.ck):
            self.verdict("build", "FAIL", f"no local checkpoint dir for {self.repo} @ {self.rev} (checkpoints.json local_path / HF snapshot absent; HF_HUB_OFFLINE=1)")
            raise Stop(10)

    def reads_device(self) -> bool:
        """Whether a stage of this run reads this host's GPU, so the target-family precheck reads it too: the Commit, the Match's GPU
        stages, a Build whose target is the device.  ``--match-phase cpu`` reads only stages whose device the GPU half's precheck read, and a
        declared-target Build derives for its declaration, so a CPU host runs those stages."""
        return (stage_on(self.stages, "commit") or (stage_on(self.stages, "match") and self.cm.match_phase != "cpu")
                or (stage_on(self.stages, "build") and not R.declared_target(self.path(self.wl), row=self.row)))

    def p2p_env(self) -> str | None:
        """The stages' NCCL_P2P_DISABLE: the row's ``engine.env`` value, else unset (one the caller's environment has is dropped)."""
        v = self.config().engine.env.get("NCCL_P2P_DISABLE")
        if v is None:
            self.child_env.pop("NCCL_P2P_DISABLE", None)
            return None
        self.child_env["NCCL_P2P_DISABLE"] = str(v)
        return str(v)

    def custom_all_reduce_off(self) -> bool:
        """The engines' ``disable_custom_all_reduce`` (tp-capture's and tp-commit's ``--engine-arg``): the row's ``engine.args``
        value, which a TP row's file states (`config.RowConfig`)."""
        return bool(self.config().engine.args["disable_custom_all_reduce"])

    def engine_args(self) -> list[str]:
        """The ``--engine-arg``s tp-capture and tp-commit share; ``disable_custom_all_reduce`` last."""
        return ["--engine-arg", f"max_model_len={self.MML}", "--engine-arg", f"max_num_batched_tokens={self.MNBT}",
                "--engine-arg", f"disable_custom_all_reduce={str(self.custom_all_reduce_off()).lower()}"]

    def hidden_build(self) -> str:
        """The FA2 tap build the ranks run: the guarded-max one unless GUARDED_MAX_TAP=0 (for A/B only)."""
        env = self.env
        return env.hidden_so or (env.hidden_so_fa2_guarded if self.ck_.guarded_max_tap == "1" else env.hidden_so_fa2)

    # ------------------------------------------------------------------------------------------------ Build
    def derive_jobs(self, shapes: list[tuple[int, int]], derives: int) -> int:
        """``--build-jobs`` N; unset or auto: as many of the `derives` derives as the memory holds at the largest of `shapes`
        (`row_records.build_jobs_auto`: the headroom, or ``--build-ram-budget-gb`` when smaller), marked ``build.jobs``."""
        j = self.cb.build_jobs or "auto"
        n = int(j) if j.isdigit() and int(j) > 0 else R.build_jobs_auto(shapes, R.model_layers(self.ck), derives=derives,
                                                                          budget_gb=self.cb.build_ram_budget_gb)
        self.tl("build.jobs", "mark", ("jobs", n), ("requested", j), ("derives", derives))
        return n

    def rank(self, r: int) -> str:
        return f"{self.bdir}/rank{r}"

    def build_ranks(self) -> list[str]:
        """``--build-rank <rank dir>`` in rank order: the Build tp-capture and tp-commit read the post-load tensors from."""
        return [a for r in range(self.W) for a in ("--build-rank", self.rank(r))]

    def derive(self, r: int, out: str, *args: str) -> None:
        """One CPU derive of rank ``r``, a switched row's ``predict emit``; the .rc beside its log carries the exit code."""
        if self.aot is not None:
            cmd = self.cli("predict", "emit", "--row", self.row, "--rank", str(r), "--config", os.path.join(self.ck, "config.json"),
                           "--trace", f"{self.aot_dir}/build/rank{r}/{os.path.basename(out)}", "--out", out, "--aot", json.dumps(self.aot), *args)
        else:
            cmd = self.cli("build", "--model", self.repo, "--revision", self.rev, "--max-model-len", str(self.MML), "--workload", self.wl,
                           "--tp", str(self.W), "--rank", str(r), "--out", out, *self.tgt_args, *args)
        open(f"{out}.rc", "w").write(f"{self.run_to(self.to_b, f'{out}.log', cmd)}\n")

    def construction(self) -> None:
        """The Build's construction (``--build-construction``): a switched row's AOT Build reads its ``aot.trace`` Build's derive dirs
        (``--aot-trace``, else every rank's step and request dirs fetched from the store into ``aot_trace/``), marked
        ``build.construction``."""
        self.aot = CR.aot_of(self.row) if self.cb.build_construction == "record" else None
        if self.aot is not None and not (aot_dir := self.cb.aot_trace):
            aot_dir = f"{self.d}/aot_trace"
            dirs = tuple(f"build/rank{r}/{n}/" for r in range(self.W) for n in ["build_step", *(f"build_request_LP{lp}_T{t}" for lp, t in self.shapes)])
            try:
                SD.fetch(self.aot["trace"], Path(aot_dir), [p for p in SD.listing(self.aot["trace"]) if p.startswith(dirs)])
            except RuntimeError as e:
                self.verdict("build", "FAIL", f"rc=1 the AOT Build's trace {self.aot['trace']} did not fetch ({e}); --aot-trace DIR gives it")
                self.tl("row", "end", ("ok", "false"), ("stopped_at", "build"))
                raise Stop(10) from None
        self.aot_dir = aot_dir if self.aot is not None else None
        self.tl("build.construction", "mark", ("construction", "aot" if self.aot else "trace"),
                *((("comparison", self.aot["comparison"]), ("trace", self.aot["trace"])) if self.aot else ()))

    def derive_all(self, todo: list[tuple[Any, ...]], jobs: int) -> None:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for f in [pool.submit(self.derive, *j) for j in todo]:
                f.result()
        if self.signalled:
            self.cancelled(1)

    def step(self, r: int, out: str | None = None) -> tuple[Any, ...]:
        return (r, out or f"{self.rank(r)}/build_step", "--wrapper", "step", "--M", str(self.B), "--pos0", "8")

    def request(self, r: int, lp: int, t: int) -> tuple[Any, ...]:
        """A request derive; the AOT Build's predicts its launch batch from the row's requests (`predict.common.launch_batch`)."""
        out = f"{self.rank(r)}/build_request_LP{lp}_T{t}"
        shutil.rmtree(out, ignore_errors=True)
        return (r, out, "--wrapper", "request", "--LP", str(lp), "--T", str(t), *([] if self.aot is not None else self.launch_of(lp, t)))

    def launch_of(self, lp: int, t: int) -> list[str]:
        """The shape's `Row.launch_args`, read once for every rank; one that does not read fails the Build."""
        if (lp, t) not in self.lctx:
            self.lctx[(lp, t)] = self.launch_args(lp, t)
        if self.lctx[(lp, t)] is None:
            self.verdict("build", "FAIL", f"rc=1 the launch context of shape LP{lp} T{t} did not read (row.log)")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "build"))
            raise Stop(10)
        return self.lctx[(lp, t)]

    def compose(self, r: int) -> None:
        """GP-01's workload Program of rank ``r`` over its request Programs."""
        der: list[str] = []
        for dd in sorted(glob.glob(f"{self.rank(r)}/build_request_LP*_T*")):
            if os.path.isfile(f"{dd}/result.json"):
                der += ["--derived", dd]
        self.run_tl(f"build.global_program_r{r}", self.to_b, f"{self.rank(r)}/build_workload.log",
                    self.cli("global-program", "--workload", self.wl, *der, "--out", f"{self.rank(r)}/build_workload", "--checkpoint-dir", self.ck))

    def compose_all(self, ranks: list[int]) -> None:
        """GP-01 of each of `ranks`, as many at once as the memory holds at `row_records.compose_bytes` a compose
        (``--build-ram-budget-gb`` when smaller than the headroom), marked ``build.compose_jobs``."""
        layers = R.model_layers(self.ck)
        n = R.build_jobs_auto(self.shapes, layers, derives=len(ranks), peak_ratio=R.COMPOSE_PEAK_RATIO, budget_gb=self.cb.build_ram_budget_gb,
                              job_bytes=R.compose_bytes(self.shapes, layers) if self.shapes and layers else None)
        self.tl("build.compose_jobs", "mark", ("jobs", n), ("ranks", len(ranks)))
        with ThreadPoolExecutor(max_workers=n) as pool:
            for f in [pool.submit(self.compose, r) for r in ranks]:
                f.result()

    def manifest_global(self, stage: str) -> int:
        """The v2 query (Q_module_body_v1) over the W rank Programs, merged by rank -> manifest.json, under the manifest's own
        limit: ``--manifest-timeout``, else `row_records.manifest_timeout` of its W x declared requests component builds (0 = none),
        never one derive's."""
        n = self.W * self.requests
        to = R.manifest_timeout(n) if self.cb.manifest_timeout is None else self.cb.manifest_timeout
        self.log(f"manifest: {n} component builds ({self.W} ranks x {self.requests} requests), limit {f'{to}s' if to else 'none'}")
        manr: list[str] = []
        wds: list[str] = []
        for r in range(self.W):
            wd = R.tp_rank_workload_digest(self.bdir, r)
            if not wd:
                self.log(f"manifest: GP-01 workload digest absent for rank {r}")
                return 4
            manr += ["--programs-root", self.rank(r), "--rank", str(r)]
            wds += ["--workload-digest", wd]
        return self.run_tl(stage, to, f"{self.d}/manifest.log",
                           self.cli("manifest", "build-global", "--fixture", self.wl, *manr, "--ranks", str(self.W), *wds, *R.tap_policy_flags(self.ck_),
                                    "--out", f"{self.d}/manifest.json"))

    def build_todo(self) -> list[tuple[Any, ...]]:
        """The derives not on disk: every rank's request Programs, the longest shape first (the large derives take the pool's first
        slots), then the step Programs (small: they fill the slots the requests leave)."""
        steps: list[tuple[Any, ...]] = []
        for r in range(self.W):
            os.makedirs(self.rank(r), exist_ok=True)
            if not R.derived_ok(f"{self.rank(r)}/build_step"):
                steps.append(self.step(r))
        requests = [self.request(r, lp, t) for lp, t in R.longest_first(self.shapes) for r in range(self.W)
                    if not R.derived_ok(f"{self.rank(r)}/build_request_LP{lp}_T{t}")]
        return requests + steps

    def build(self) -> None:
        nsh = len(self.shapes)
        self.to_b = self.cb.build_timeout or 7200
        t0 = int(time.time())
        self.tl("stage.build", "start")
        self.construction()
        todo, jobs = self.build_todo(), self.derive_jobs(self.shapes, self.W * (nsh + 1))
        if self.cb.build_relabel == "on" and self.aot is None:
            from verity_vllm.pipeline.tp import relabel_build
            relabel_build.derive(self, todo, jobs)
        else:
            self.derive_all(todo, jobs)
        for r in range(self.W):
            for f in sorted(glob.glob(f"{self.rank(r)}/*.rc")):
                self.log(f"derive {os.path.basename(f[:-3])} rank{r} rc={open(f).read().strip()} {_tail_bytes(f[:-3] + '.log')}")
        kept = [r for r in range(self.W) if R.composed_ok(self.rank(r))]
        if kept:
            self.log(f"compose: ranks {kept} keep their workload Programs on disk (composed after their request Programs)")
        self.compose_all([r for r in range(self.W) if R.derived_ok(f"{self.rank(r)}/build_step") and r not in kept])
        ok, rc, programs_only = self.guarded_summary(), None, self.cb.build_programs_only == "1"
        if ok == "PASS" and not programs_only:   # the Build's manifest, as SingleRow's: an incomplete one (rc 4) fails the Build its Commit would refuse
            rc = self.manifest_global("build.required_manifest")
            self.log(f"build manifest rc={rc}: {R.tp_manifest_line(self.d) or 'no manifest.json'}")
            ok = "PASS" if rc == 0 else "FAIL"
        wall = int(time.time()) - t0
        digest, digests = R.tp_build_digests(self.d)
        wds = "/".join(R.tp_rank_workload_digest(self.bdir, r)[:16] for r in range(self.W))
        self.verdict("build", ok, f"rc={0 if ok == 'PASS' else 1} manifest_rc={rc} wall={wall}s digest={digest} digests_r1..={digests} workload_digests_r0..={wds} shapes={nsh} "
                     f"construction={'aot' if self.aot is not None else 'trace'}")
        self.tl("stage.build", "end", ("ok", ok), ("wall_s", wall))
        if ok != "PASS":
            hits = Counter(ln for p in sorted(glob.glob(f"{self.bdir}/rank*/*.log"))
                           for ln in _grep(p, lambda s: any(k in s for k in ("Error", "refused", "Unsupported"))))
            top = "".join(f"{n:7d} {ln}"[:300] + " " for ln, n in hits.most_common(3))
            self.log(f"BUILD FAIL -> stop ({top})")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "build"))
            raise Stop(10)
        if not programs_only:
            self.call_boundaries(self.to_b)      # the Build passed, so its manifest was complete

    def guarded_summary(self) -> str:
        try:
            return R.tp_build_summary(self.bdir, self.W)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            self.log(f"build summary failed: {type(e).__name__}: {e}")
            return "FAIL"

    # ------------------------------------------------------------------------------------------------ Match
    def fold_sizing(self) -> tuple[int, int, int]:
        """The fold Match's ``--rank-workers``, ``--fold-workers`` and one stage's limit: `row_records.fold_plan` over the declared shapes
        in ``--fold-mem-gb`` (research run's --mem-gb, else the headroom) wherever a flag leaves them to it, marked ``match.fold_plan``;
        every rank and 7200 s when the checkpoint's layers are unknown.  Memory that holds no planned GM-01 stage beside one fold runs
        one rank at a time and says so: the plan's ratios are fitted on GLM-5.3 B8 alone."""
        cm, layers = self.cm, R.model_layers(self.ck)
        if not layers or not self.shapes:
            return cm.fold_rank_workers, cm.fold_workers, cm.fold_timeout or R.FOLD_LEG_MIN_S
        p = R.fold_plan(self.shapes, layers, self.W, int(cm.fold_mem_gb * 1e9) if cm.fold_mem_gb else None, folds=cm.fold_workers)
        gb = {k: round(p[f"{k}_bytes"] / 1e9) for k in ("gm", "fold", "mem")}
        rw, fw = (cm.fold_rank_workers, cm.fold_workers) if cm.fold_rank_workers else (p["rank_workers"], p["fold_workers"])
        to = cm.fold_timeout or p["leg_s"]
        self.tl("match.fold_plan", "mark", ("rank_workers", rw), ("fold_workers", fw), ("leg_s", to), ("fits", p["fits"]),
                *((f"{k}_gb", v) for k, v in gb.items()))
        short = "" if p["fits"] or cm.fold_rank_workers else f"; WARNING: {gb['mem']} GB holds no planned GM-01 stage beside one fold (a larger --mem-gb)"
        self.log(f"fold Match: {rw} ranks' compare and GM-01 stages at once ({gb['gm']} GB planned each), {fw or rw} folds ({gb['fold']} GB each), "
                 f"{to}s a stage, in {gb['mem']} GB ({'--mem-gb' if cm.fold_mem_gb else 'the headroom'}){short}")
        return rw, fw, to

    def match(self) -> None:
        """The capture and the control, the GPU stages, each under the item's lease when it has one (`Row.run_gpu`: all the task's GPUs from
        right before the stage's engine to its exit, recorded in ``match/lease/<stage>/``); the check, the served-shape derives and the fold
        Match after them, on the CPU, with no device and no lease.  ``--match-phase gpu`` runs the GPU stages alone, which need no Build, and
        stops the row; ``--match-phase cpu`` runs the rest over the stages ``match/gpu_legs.json`` records."""
        cm, M, phase = self.cm, self.M, self.cm.match_phase
        to_m = cm.match_timeout or 3600
        self.to_b = self.cb.build_timeout or 7200
        for r in range(self.W if phase != "gpu" else 0):
            if not R.derived_ok(f"{self.rank(r)}/build_step"):
                self.verdict("match", "FAIL", f"no step Program for rank {r} ({self.rank(r)}/build_step): Build first")
                raise Stop(11)
        rw, fw, to_f = self.fold_sizing() if phase != "gpu" and self.fold_profile else (0, 0, 0)
        br = [a for r in range(self.W) for a in ("--build-rank", f"{self.rank(r)}/build_step")]
        cap = self.cli("tp-capture", "--workload", self.wl, "--case", self.role, "--tp", str(self.W), "--max-num-seqs", str(self.B),
                       "--engine-arg", f"gpu_memory_utilization={self.gpu_util}", *self.engine, *self.build_ranks())
        t0 = int(time.time())
        self.tl("stage.match", "start")
        if phase == "cpu":
            legs = R.load_json(f"{M}/gpu_legs.json")
            if not legs:
                self.verdict("match", "FAIL", f"MATCH_PHASE=cpu without {M}/gpu_legs.json: run the GPU stages first (--match-phase gpu or all)")
                raise Stop(11)
            rc_cap, rc_ctl = legs["capture_rc"], legs["control_rc"]
            self.log(f"match: MATCH_PHASE=cpu over the GPU stages already in {M} (capture rc={rc_cap}, control rc={rc_ctl})")
        else:
            shutil.rmtree(f"{M}/fold_match", ignore_errors=True)      # an earlier capture's folds and stages, which --match-reuse must never keep
            rc_cap = self.run_gpu("match.capture", to_m, f"{M}/capture.log", [*cap, "--record-collectives", "--out", f"{M}/capture"],
                                  lease_dir=f"{M}/lease/capture", kept=GL.MATCH_LEG_KEPT)
            ranks = " ".join(_grep(f"{M}/capture.log", lambda s: s.startswith("[tp2] rank ") and s[11:].split(":", 1)[0].isdigit() and ":" in s[11:]))
            self.log(f"match capture rc={rc_cap} {(ranks + ' ' if ranks else '')[:300]}")
            rc_ctl = self.run_gpu("match.control", to_m, f"{M}/control.log", [*cap, "--no-observer", "--out", f"{M}/control"],
                                  lease_dir=f"{M}/lease/control", kept=GL.MATCH_LEG_KEPT)
            with open(f"{M}/gpu_legs.json", "w") as f:
                json.dump({"capture_rc": rc_cap, "control_rc": rc_ctl}, f)
        if phase == "gpu":
            ok, wall = "PASS" if rc_cap == 0 and rc_ctl == 0 else "FAIL", int(time.time()) - t0
            self.verdict("match-gpu", ok, f"rc={rc_cap}/{rc_ctl} wall={wall}s: the check and the fold Match run with --match-phase cpu")
            self.tl("stage.match", "end", ("ok", ok), ("wall_s", wall))
            raise Stop(0 if ok == "PASS" else 11)
        rc_chk = self.run_tl("match.check", to_m, f"{M}/check.log", self.cli("tp-match", "--arm", "check", "--out", M, *br))
        chk = R.tp_check_line(M)
        self.log(f"match check rc={rc_chk}: {chk or 'no tp2_match.json'}")
        need = R.tp_served_shapes(M)
        missing = [sh for sh in need for r in range(self.W) if not R.derived_ok(f"{self.rank(r)}/build_request_LP{sh.replace(',', '_T')}")]
        if missing:
            declared = " ".join(f"{lp},{t}" for lp, t in self.shapes)
            miss = "".join(" " + sh for sh in missing)
            self.verdict("eos-truncated", f"served=[{' '.join(need)}]", f"declared=[{declared}]: request(s) stopped before the cap; deriving the served-shape "
                         f"Program(s) [{miss}] per rank and recomposing GP-01 (MOE-05)", quiet=True)
            served = R.longest_first({tuple(map(int, sh.split(","))) for sh in missing})
            todo = [self.request(r, lp, t) for lp, t in served for r in range(self.W)
                    if not R.derived_ok(f"{self.rank(r)}/build_request_LP{lp}_T{t}")]
            self.derive_all(todo, self.derive_jobs(served, len(todo)))
            self.compose_all(list(range(self.W)))
            self.guarded_summary()
            wds = "".join(f"r{r}={R.tp_rank_workload_digest(self.bdir, r)[:16]} " for r in range(self.W))
            self.log(f"served-shape derivation done: workload digests {wds}")
        fold, rc_fold, fp = f"collective-level only (no TP rank profile for {self.role}: fold Match not run)", -1, self.fold_profile
        if fp:
            fargs: list[str] = []
            for r in range(self.W):
                first = True
                for dd in sorted(glob.glob(f"{self.rank(r)}/build_request_LP*_T*")):
                    if not os.path.isfile(f"{dd}/result.json"):
                        continue
                    fargs += ["--build-rank", dd] if first else ["--derived-rank", f"{r}={dd}"]
                    first = False
                gp = f"{self.rank(r)}/build_workload/workload_program.json"
                if os.path.isfile(gp):
                    fargs += ["--gprog-rank", f"{r}={gp}"]
            if phase == "cpu" and cm.match_reuse != "off":
                fargs += ["--reuse-fold", *(["--reuse-legs"] if cm.match_reuse == "legs" else [])]
            if cm.match_pipeline == "shared":
                fargs.append("--shared-x09")
            # --timeout bounds each subprocess: a rank runs three in turn (fold, compare, GM-01), and the cross-rank check follows them
            ranks_at_once = min(rw or self.W, self.W)
            phase_s = R.fold_phase_s(to_f, self.W, ranks_at_once, min(fw or ranks_at_once, self.W))
            rc_fold = self.run_tl("match.fold_match", phase_s, f"{M}/fold_match.log",
                                  self.cli("tp-fold-match", "--capture", f"{M}/capture", "--profile", fp, *fargs, "--workload", self.wl,
                                           "--tp2-match", f"{M}/tp2_match.json", "--within-step", "dag", "--rank-workers", str(rw),
                                           "--fold-workers", str(fw), "--python", self.py, "--timeout", str(to_f), "--out", f"{M}/fold_match"))
            fold = f"fold_match rc={rc_fold} {R.tp_fold_line(M)}"
            self.log(f"match fold ({fp}): {fold}")
        wall = int(time.time()) - t0
        ok = "FAIL"
        if rc_cap == 0 and rc_ctl == 0 and rc_chk == 0 and R.tp_passed(f"{M}/tp2_match.json"):
            if not fp or (rc_fold == 0 and R.tp_passed(f"{M}/fold_match/tp2_fold_match.json")):
                ok = "PASS"
        R.tp_match_summary(M, rc_cap, rc_ctl, rc_chk, rc_fold, ok, fp)
        self.verdict("match", ok, f"rc={rc_cap}/{rc_ctl}/{rc_chk}/{rc_fold} wall={wall}s tp2_match[{chk or 'none'}] {fold}")
        self.tl("stage.match", "end", ("ok", ok), ("wall_s", wall))
        if ok != "PASS":
            self.log("MATCH FAIL -> stop")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "match"))
            raise Stop(11)

    # ------------------------------------------------------------------------------------------------ Commit
    def commit(self) -> None:
        ck, d = self.ck_, self.d
        to_k = ck.commit_timeout or 7200
        t0 = int(time.time())
        self.tl("stage.commit", "start")
        if not os.path.isfile(self.hidden_so):
            self.verdict("commit", "FAIL", f"hidden .so missing: {self.hidden_so}")
            raise Stop(12)
        for r in range(self.W):
            if not R.tp_rank_workload_digest(self.bdir, r):
                self.verdict("commit", "FAIL", f"GP-01 workload digest absent for rank {r}: Build first")
                raise Stop(12)
        why = "" if self.config_run and R.tp_manifest_complete(d) else R.tp_manifest_not_of_record(d, ck)
        if not why:
            rc_man = 0                            # word-checked under the row's taps; tp-commit checks its per-rank Programs against the Build
            self.log(f"required manifest of record: the one beside the Build{' (config run)' if self.config_run else ''}")
        else:
            self.log(f"manifest.json beside the Build is not of record ({why}): rebuilding it")
            rc_man = self.manifest_global("commit.required_manifest")
        man = R.tp_manifest_line(d)
        self.log(f"manifest rc={rc_man}: {man or 'no manifest.json'}")
        if rc_man != 0 or not R.tp_manifest_complete(d):
            self.verdict("commit", "FAIL", f"required manifest not complete (rc={rc_man}): {man or 'no manifest.json'} (manifest.log)")
            self.tl("stage.commit", "end", ("ok", "false"))
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "commit"))
            raise Stop(13)
        taps = R.applicable_taps(f"{d}/manifest.json", ck)
        why = R.missing_tap_build(taps, self.env)
        if why:
            self.verdict("commit", "FAIL", why)
            raise Stop(12)
        self.log(f"taps: {', '.join(taps) or 'none'} (on, and the manifest names their families); guarded max {'on' if ck.guarded_max_tap == '1' else 'off'}")
        self.child_env.update({"VERITY_WINDOW_MB": str(ck.window_mb), "VERITY_WINDOW_SLOTS": str(ck.window_slots)})
        self.child_env["VERITY_RETAIN"] = ck.retain
        self.child_env["VERITY_STAGING_BOUNDED"] = "1" if self.config_run else "0"
        self.child_env.setdefault(GL.WEIGHTS_HASH, "device")
        self.log(f"commit staging: window_mb={ck.window_mb} slots={ck.window_slots} retain={ck.retain}{' bounded' if self.config_run else ''}; "
                 f"weights hash {self.child_env[GL.WEIGHTS_HASH]}")
        rc_k = self.run_gpu("commit.tp_commit", to_k, f"{d}/commit.log", self.commit_cmd(taps))
        keys = ("PASS", "FAIL", "sampled replay", "match oracle", "weights pin", "cross-rank collectives", "fold-Match", "C2 population", "coverage")
        tagged = _grep(f"{d}/commit.log", lambda s: s.startswith("[tp2commit] ") and s[12:].lstrip(" ").startswith(keys))
        with open(f"{d}/row.log", "a") as f:
            f.writelines(ln[:400] + "\n" for ln in tagged[-16:])
        wall = int(time.time()) - t0
        if self.commit_pending(rc_k, wall):
            return
        if self.config_run:
            RC.commit_verdict(self, rc_k, wall)
            return
        summ = R.tp_commit_line(d)
        ok = "PASS" if rc_k == 0 and R.tp_commit_passed(d) else "FAIL"
        last = [ln for ln in tagged if ln.startswith(("[tp2commit] PASS", "[tp2commit] FAIL"))]
        self.verdict("commit", ok, f"rc={rc_k} wall={wall}s {summ or 'no commit/summary.json'} {last[-1][:200] if last else ''}")
        self.tl("stage.commit", "end", ("ok", ok), ("wall_s", wall))
        if ok != "PASS":
            self.log("COMMIT FAIL -> stop")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "commit"))
            raise Stop(12)

    def commit_cmd(self, taps: list[str]) -> list[str]:
        """`tp-commit`'s command line; `taps`: the row's taps that apply to it (`row_records.applicable_taps`), whose sources the ranks attach."""
        ck, d = self.ck_, self.d
        if self.config_run:                       # one instrumented run, k units over the ranks; no Match to compare against
            match, arms = [], ["--pairs", "1", "--warmup", "1", "--only-arm", "instrumented", "--replay-k", str(ck.replay_k)]
        else:
            match, arms = ["--match-dir", self.M], ["--pairs", str(ck.pairs), "--warmup", "2", "--value-check"]
        cmd = self.cli("tp-commit", "--workload", self.wl, "--case", self.role, "--tp", str(self.W), "--committer", "native_collect_v2b",
                       "--hidden-so", self.hidden_so, "--openings", "64", "--max-num-seqs", str(self.B),
                       "--engine-arg", f"gpu_memory_utilization={self.commit_util}", *self.engine, *self.build_ranks(),
                       "--required-manifest", f"{d}/manifest.json", *match, "--checkpoint-dir", self.ck, "--checkpoint-model", self.repo,
                       *arms, "--replay-workers", str(ck.replay_workers), "--label", self.row, "--out", f"{d}/commit")
        if not self.config_run and self.fold_profile and os.path.isfile(f"{self.M}/fold_match/tp2_fold_match.json"):
            cmd.append("--require-fold-match")
        if ck.replay_deferred in ("1", "slim"):    # slim reaches a TP row only at a world where tp-commit replays in process (`run_row`)
            cmd.append("--replay-deferred")
        if ck.serving_rows:
            cmd += ["--serving-rows", self.serving_rows_request(ck.serving_rows)]
        if ck.poub_verifier:
            raise SystemExit("POUB_VERIFIER: the remote bandwidth session is a single-GPU Commit's (one store per node)")
        return cmd + R.tap_source_args(taps, self.env) + (ck.extra_commit_args or "").split()

    def commit_pending(self, rck: int, wall: int) -> bool:
        """True when the Commit ran every check but the ranks' C2 replay, which it deferred to their sealed bundles (`--replay-deferred`,
        `commit/replay_deferred.json`): the stage is PENDING until `replay` decides it."""
        sm = R.load_json(f"{self.d}/commit/summary.json") or {}
        bundles = (sm.get("replay_deferred") or {}).get("bundles") or []
        if not (self.ck_.replay_deferred == "1" and rck == 0 and sm.get("pass") is None and bundles):
            return False
        self.verdict("commit", "PENDING", f"rc={rck} wall={wall}s {R.tp_commit_line(self.d)}; {len(bundles)} rank replay bundles sealed "
                                          f"({', '.join(os.path.basename(b['path']) for b in bundles)})")
        self.tl("stage.commit", "end", ("ok", "PENDING"), ("wall_s", wall))
        self.replay_pending = True
        return True

    def replay(self) -> None:
        """The deferred C2 of the Commit, on CPU (`tp-commit-replay`), from every rank's sealed bundle: each rank's shards composed from
        the checkpoint of record at (world, rank) and opened against its committed weights root.  Its exit decides the stage."""
        d = self.d
        if skipped := ((R.load_json(f"{d}/commit/summary.json") or {}).get("replay_deferred") or {}).get("skipped"):
            self.log(f"replay: nothing deferred, the Commit replayed every rank in process ({skipped})")
            return
        self.tl("stage.replay", "start")
        why = (f"no deferred Commit in {d}/commit (commit/replay_deferred.json)" if not os.path.isfile(f"{d}/commit/replay_deferred.json") else
               f"no checkpoint dir of record on this host (CK='{self.ck}')" if not (self.ck and os.path.isdir(self.ck)) else None)
        rc, wall = 2, 0
        if why is None:
            t0 = int(time.time())
            rc = self.run_tl("replay.tp_commit_replay", RC.REPLAY_TIMEOUT, f"{d}/replay.log", self.cli(
                "tp-commit-replay", f"{d}/commit", "--checkpoint-dir", self.ck, "--checkpoints-manifest", self.path("manifests/checkpoints.json"),
                "--model", self.repo, "--revision", self.rev))
            wall = int(time.time()) - t0
            why = None if rc == 0 else _tail_bytes(f"{d}/replay.log")
        ok = "PASS" if rc == 0 else "FAIL"
        self.verdict("replay", ok, f"rc={rc} wall={wall}s the ranks' C2 replay on CPU from their bundles" + (f": {why}" if why else ""))
        self.tl("stage.replay", "end", ("ok", ok), ("wall_s", wall))
        if ok != "PASS":
            self.log("REPLAY FAIL -> stop")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "replay"))
            raise Stop(12)
