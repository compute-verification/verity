"""A single-GPU row's stages under `verity-vllm row`: precheck -> Build -> Match -> Commit, stopping at the first FAIL.

* **Precheck** (before any engine): the release line of the tree that runs, source identity (every package resolves
  from this tree under the stages' interpreters), the target family (row id device class == declared target == device)
  and the ADVISORY host-memory admission plan (never a stop).  A failing precheck is exit 3, by name.
* **Build**: the step Program, the request wrapper Program (``--lp``/``--t``, default the envelope; a stochastic row:
  its first request) and every other declared request shape, each under its launch context at B >= 2; then GP-01's
  workload Program (the row's Program digest) and the required-value manifest of record (``manifest build`` /
  ``build-global``; exit 13 when it does not build, 3 when the core registry is not importable).  After a PASS Build the
  row stops (exit 21) unless every call-boundary identity of the manifest has a source the Commit attaches
  (``Row.call_boundaries``: the CPU gate ``gate call-boundaries``, over the Programs the Commit reads with ``boundary-plan``, which
  writes the Commit's call-boundary plan on a PASS).
* **Match**: the observed run and the fold (``match``), the per-request comparison (B=1 against the served shape; B>1
  the X-09 decomposition, deriving any shape it names as missing and rerunning once), the global Match (GM-01) and, on
  a stochastic row, the sampler value check; ``row_records.match_summary`` decides.
* **Commit**: the pairs protocol (``commit``) with the manifest of record, the Program digest read from the Build, the
  Match record for the live oracle compare, the Build's Programs for the sampled replay and the weights of record
  (derived beside the component Program, re-derived when stale); then the manifest verification and the verdict
  record (``verdict from-record``), whose outcome is the stage's decision.

A **config run** (``--config-run 1``): the Build and the strict word check, ONE instrumented Commit under bounded staging with a
k-unit replay (uniform by default, ``--replay-draw``), and ``config_record.json``; no Match, no manifest-verify (``pipeline.row_config``).  With
``--replay-deferred 1`` the Commit seals its replay bundle and ends PENDING; the **replay** stage (``row_config.replay``, on CPU) decides
it and writes ``config_record.json``.  The **plan** stage (``row_plan``, on CPU) derives what the Commit needs before the GPU is held;
a config run's Build without the Commit or a plan stage of its own ends with it, and its FAIL leaves the Commit to derive what it
would have held.

Exit: 0 every requested stage PASS; 10 Build FAIL; 11 Match FAIL; 12 Commit FAIL; 13 manifest not built; 14 Match
record incomplete at Commit; 15 plan FAIL; 20 strict word check FAIL (config run); 21 a call-boundary identity without a source the
Commit attaches; 3 precheck FAIL; 2 no workload; 128+signal when cancelled.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from verity_vllm.config import BuildConfig, CommitConfig, MatchConfig, RunEnv
from verity_vllm.pipeline import row_build as RB
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import row_group as RG
from verity_vllm.pipeline import row_plan as RP
from verity_vllm.pipeline import row_records as R
from verity_vllm.pipeline.row_driver import Row, Stop, copy_first, stage_on

SKIP = "census,census_reconcile,nonint,nonint_compare,compare,root_policy,golden,wiring,replay_a,replay_chain,fold_2"
CAP = 14400


def _tail(path: str, n: int = 300) -> str:
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return ""
    return (lines[-1] if lines else "")[:n]


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


class SingleRow(Row):
    def __init__(self, *, row: str, role: str, repo: str, rev: str, stages: str, env: RunEnv, gpu_util: str,
                 build: BuildConfig, match: MatchConfig, commit: CommitConfig, protocols: tuple[str, ...] | None = None):
        super().__init__(row=row, role=role, repo=repo, rev=rev, stages=stages, env=env, proc="row_pod")
        self.gpu_util, self.cb, self.cm, self.ck_ = gpu_util, build, match, commit
        self.protocols = protocols            # the row's protocol set beside sampled proofs (protocol_options.at_row), None by default
        self.trap = False
        self.config_run = commit.config_run == "1"
        self.replay_pending = False

    def run(self) -> int:
        try:
            self.prepare()
            if stage_on(self.stages, "build"):
                self.build()
                if self.config_run:
                    RC.word_check(self)
                if self.config_run and not (stage_on(self.stages, "plan") or stage_on(self.stages, "commit")):
                    RP.plan(self, required=False)
            if stage_on(self.stages, "plan"):
                RP.plan(self)
            if stage_on(self.stages, "match") and not self.config_run:
                self.match()
            if stage_on(self.stages, "commit"):
                self.commit()
            if stage_on(self.stages, "replay"):
                RG.replay(self) if self.ck_.group else RC.replay(self)
                self.replay_pending = False
        except Stop as s:
            if self.config_run and self.trap:
                RC.config_record(self)
            return s.rc
        finally:
            if self.trap:
                self.research_result()
        if self.replay_pending:
            self.log("config run PENDING: the C2 replay runs on CPU from the sealed bundle (`row stage replay`), which writes config_record.json")
        elif self.config_run:
            RC.config_record(self)
        self.tl("row", "end", ("ok", "true"))
        self.timeline_report()
        self.log("DONE")
        return 0

    def guarded(self, what: str, fn: Any, *args: Any) -> str:
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 -- a summary that cannot be written is a FAIL of its stage, named in row.log
            self.log(f"{what} failed: {type(e).__name__}: {e}")
            return "FAIL"

    # ------------------------------------------------------------------------------------------------ precheck
    def prepare(self) -> None:
        env = self.env
        self.ncb_set = env.native_collect_build
        self.ncb = env.native_collect_build or env.native_collect_default
        self.setup({"MATCH_IMPL": self.cm.match_impl, "MATCH_PIPELINE": self.cm.match_pipeline, "NATIVE_COLLECT_BUILD": self.ncb})
        os.makedirs(self.d, exist_ok=True)
        os.makedirs(self.ncb, exist_ok=True)
        if not os.path.isfile(self.path(self.wl)):
            print(f"no manifest {self.wl}")
            raise Stop(2)
        self.trap = True
        rel_line, rel_src = self.release_line(" (release_json.py failed; not installed here by pod_release.sh)")
        self.rel_tag = rel_line.split("|", 1)[0]
        if self.rel_tag != "UNRELEASED" and os.path.isfile(rel_src):
            shutil.copyfile(rel_src, f"{self.d}/release.json")
        elif os.path.exists(f"{self.d}/release.json"):
            os.remove(f"{self.d}/release.json")
        if not self.ncb_set and os.path.isfile(f"{self.d}/release.json"):
            rel = R.load_json(f"{self.d}/release.json") or {}
            ncb_rel = (rel.get("build_dirs") or {}).get("native_collect") or ""
            if ncb_rel and ncb_rel != self.ncb:
                self.ncb = self.child_env["NATIVE_COLLECT_BUILD"] = ncb_rel
                os.makedirs(self.ncb, exist_ok=True)
                self.log(f"collector build dir from the release: {self.ncb}")
        export = R.load_json(self.path("EXPORT.json")) if os.path.exists(self.path("EXPORT.json")) else None
        with open(f"{self.d}/tree.json", "w") as f:
            json.dump({"schema": "verity-sweep/executing-tree/v1", "tree": self.tree, "export": export, "release_line": rel_line}, f, indent=1)
        self.verdict("release", self.rel_tag, rel_line.split("|", 1)[1] if "|" in rel_line else rel_line, quiet=True)
        self.precheck_identity()
        self.shape()
        self.precheck_target()
        self.admission()
        self.log(f"start B={self.B} I={self.I} O={self.O} LP={self.LP} T={self.T} ckpt={self.ck} stages={self.stages} "
                 f"target={self.tgt or 'device'} hidden_so={self.hidden_so} code={self.code9}")
        self.tl("row", "start", ("stages", self.stages), ("B", self.B), ("I", self.I), ("O", self.O), ("LP", self.LP), ("T", self.T),
                ("code", self.code9), ("release", self.rel_tag), ("pod", os.uname().nodename))
        self.install_signal_forwarding()

    def precheck_identity(self) -> None:
        def venv(py: str) -> str:
            return os.path.basename(os.path.dirname(os.path.dirname(py)))
        self.tl("precheck", "start")
        sha = self.export_commit()
        pys = [self.py] + ([self.py312] if self.py312 != self.py else [])
        if self.source_identity(pys, lambda py: f"{self.d}/source_identity.{venv(py)}.json"):
            self.tl("precheck", "end", ("rc", 3), ("fail_class", "PRECHECK_FAIL_SOURCE_IDENTITY"))
            self.verdict("precheck", "FAIL", f"PRECHECK_FAIL_SOURCE_IDENTITY (see {self.d}/source_identity.*.json and row.log)")
            raise Stop(3)
        copy_first(f"{self.d}/source_identity.json", f"{self.d}/source_identity.{venv(self.py)}.json",
                   *sorted(glob.glob(f"{self.d}/source_identity.*.json")))
        job = f"{self.d}/job.json"
        if os.path.isfile(job):
            try:
                d = json.load(open(job))
                d.setdefault("computation", {})["source_identity"] = json.load(open(f"{self.d}/source_identity.json"))
                d["computation"]["source_identity_reason"] = None
                open(job + ".tmp", "w").write(json.dumps(d, indent=1) + "\n"); os.replace(job + ".tmp", job)
            except (OSError, ValueError, AttributeError) as e:
                self.log(f"job.json not updated with the source identity: {e}")
        self.tl("precheck", "end", ("rc", 0))
        self.verdict("precheck", "PASS", f"source identity == {self.tree}{' @ ' + sha if sha else ''} (verity verity_vllm hidden_gpu research)",
                     quiet=True)

    def shape(self) -> None:
        s = R.row_shape(self.path(self.wl), self.path("manifests/checkpoints.json"), self.env.hf_home)
        self.B, self.I, self.O, self.snap_def, self.ck = s["B"], s["I"], s["O"], s["snap"], str(s["ck"])
        self.LP = self.cb.lp if self.cb.lp is not None else s["LP"]
        self.T = self.cb.t if self.cb.t is not None else s["T"]
        self.MML = self.I + self.O
        self.tgt = R.declared_target(self.path(self.wl), self.protocols, row=self.row)
        self.tgt_args = ["--target", self.tgt] if self.tgt else []
        # the attention implementation the row executes (declared, else the pinned vLLM's default for this device) selects the tap
        _, fa = self.output(self.cli("row", "fa-version", self.wl, *self.tgt_args), stderr=subprocess.DEVNULL)
        if self.guarded_max():
            self.hidden_so = self.env.hidden_so or (self.env.hidden_so_fa3_guarded if fa.strip() == "3" else self.env.hidden_so_fa2_guarded)
        else:
            self.hidden_so = self.env.hidden_so or (self.env.hidden_so_fa3 if fa.strip() == "3" else self.env.hidden_so_fa2)
        self.code9 = self.export_commit()[:9]
        self.snap_steps = None if self.cm.snapshot_steps in (None, "row-default") else self.cm.snapshot_steps

    def precheck_target(self) -> None:
        self.tl("precheck-target", "start")
        rc, line = self.target_family(out=f"{self.d}/target_family.json")
        if rc != 0:
            self.tl("precheck-target", "end", ("rc", 3), ("fail_class", "PRECHECK_FAIL_TARGET_FAMILY"))
            self.verdict("precheck-target", "FAIL", line)
            raise Stop(3)
        self.tl("precheck-target", "end", ("rc", 0))
        self.verdict("precheck-target", "WAIVED" if line.startswith("TARGET-FAMILY-WAIVED") else "PASS", line, quiet=True)

    def admission(self) -> None:
        self.tl("admission", "start")
        snapmax = ["--snapshot-max-bytes", self.cm.snapshot_max_bytes] if self.cm.snapshot_max_bytes else []
        rc, line = self.output(self.cli("telemetry-admission", "plan", "--row", self.row, "--workload", self.wl, "--stages", self.stages,
                                        "--pairs", str(self.ck_.pairs), "--snapshot-steps", self.snap_steps or self.snap_def, *snapmax,
                                        "--out", f"{self.d}/admission.json"))
        if rc != 0:
            line = f"ADMISSION advisory unavailable rc={rc} {line}"
        self.log(line)
        if "WARN would-refuse" in line:
            self.log(f"WARN admission advisory: a stage would be REFUSED by the planner (see {self.d}/admission.json); advisory in r18-int-3 -- the row continues")
        self.tl("admission", "end", ("rc", 0), ("advisory", "true"))

    # ------------------------------------------------------------------------------------------------ manifest
    def guarded_max(self) -> bool:
        """The row runs the guarded-max tap build: by default, as every row's attention is FA2/FA3 (GUARDED_MAX_TAP=0 for A/B only)."""
        return self.ck_.guarded_max_tap == "1"

    def required_manifest(self, stage: str, digest: Callable[[], str] | None = None) -> int | None:
        """The required-value manifest of record (`row_build.required_manifest`) -> 0 | 3 | 1 | None."""
        return RB.required_manifest(self, stage, digest)

    # ------------------------------------------------------------------------------------------------ Build
    def request_cmd(self, lp: int, t: int, out: str, lctx: list[str] | None = None) -> list[str]:
        return self.cli("build", "--model", self.repo, "--revision", self.rev, "--wrapper", "request", "--LP", str(lp), "--T", str(t),
                        "--max-model-len", str(self.MML), "--workload", self.wl, "--out", out, *self.tgt_args, *(lctx or []))

    def launch_args(self, lp: int, t: int) -> list[str] | None:
        """A B >= 2 request Program is built under its launch context (``launch-context``); None = the context did not
        read, which FAILS that shape's Build (it is not "no context")."""
        if self.B <= 1:
            return []
        err = f"{self.d}/launch_context_LP{lp}_T{t}.err"
        with open(err, "w") as f:
            r = subprocess.run(self.cli("launch-context", self.wl, str(lp), str(t)), stdout=subprocess.PIPE, stderr=f, text=True,
                               cwd=self.tree, env=self.child_env)
        if r.returncode != 0:
            self.log(f"Build FAIL shape LP{lp} T{t}: launch_context rc={r.returncode} -- {open(err, errors='replace').read().replace(chr(10), ' ')[-300:]}")
            return None
        os.remove(err)
        lc = r.stdout.rstrip("\n")
        return ["--launch-max-seqlen-q", *lc.split("\n")] if lc else []

    def derive_all(self, to: int) -> tuple[int, int]:
        """The step, request and extra-shape derives, ``--build-jobs`` wide (auto: `row_records.build_jobs_auto`) -> (step rc, request rc)."""
        cmd_s = self.cli("build", "--model", self.repo, "--revision", self.rev, "--wrapper", "step", "--M", str(self.B), "--pos0", "8",
                         "--max-model-len", str(self.MML), "--workload", self.wl, "--out", f"{self.d}/build_step", *self.tgt_args)
        lctx = self.launch_args(self.LP, self.T)
        cmd_r = self.request_cmd(self.LP, self.T, f"{self.d}/build_request", lctx)
        req = self.cb.build_jobs or "auto"
        shapes = [(self.LP, self.T), *R.extra_shapes(self.path(self.wl), self.LP, self.T)]
        jobs = str(R.build_jobs_auto(shapes, R.model_layers(self.ck), budget_gb=self.cb.build_ram_budget_gb)) if req == "auto" else req
        jobs_n = int(jobs) if re.fullmatch(r"[0-9]+", jobs) else 1
        self.tl("build.jobs", "mark", ("jobs", jobs_n), ("requested", req))
        extra = []
        rc_x = 0
        for lp, t in R.extra_shapes(self.path(self.wl), self.LP, self.T):
            out = f"{self.d}/build_request_LP{lp}_T{t}"
            if R.derived_ok(out):
                continue
            lx = self.launch_args(lp, t)
            if lx is None:
                rc_x = 1
                continue
            extra.append((f"build.derive_request_LP{lp}_T{t}", f"{out}.log", self.request_cmd(lp, t, out, lx), out))
        if jobs_n <= 1:
            rc1 = RC.derive_cached(self, "build.derive_step", to, f"{self.d}/build_step.log", cmd_s, f"{self.d}/build_step")
            rc2 = (RC.derive_cached(self, "build.derive_request", to, f"{self.d}/build_request.log", cmd_r, f"{self.d}/build_request")
                   if lctx is not None else 1)
            for st, lf, cmd, out in extra:
                rcx = RC.derive_cached(self, st, to, lf, cmd, out)
                rc2 = rcx if rcx != 0 else rc2
            return rc1, rc2 if rc_x == 0 else 1

        def job(st: str, lf: str, cmd: list[str], out: str) -> None:
            rc = RC.derive_cached(self, st, to, lf, cmd, out, ("par", "true"))
            open(f"{lf}.rc", "w").write(f"{rc}\n")

        def rc_of(lf: str) -> int:
            try:
                return int(open(f"{lf}.rc").read().strip())
            except (OSError, ValueError):
                return 1
        t0 = int(time.time())
        todo = [("build.derive_step", f"{self.d}/build_step.log", cmd_s, f"{self.d}/build_step")]
        if lctx is not None:
            todo.append(("build.derive_request", f"{self.d}/build_request.log", cmd_r, f"{self.d}/build_request"))
        todo += extra
        for _, lf, _, _ in todo:
            if os.path.exists(f"{lf}.rc"):
                os.remove(f"{lf}.rc")
        with ThreadPoolExecutor(max_workers=jobs_n) as pool:
            for f in [pool.submit(job, *j) for j in todo]:
                f.result()
        rc1 = rc_of(f"{self.d}/build_step.log")
        if self.signalled:
            self.cancelled(rc1)
        rc2 = rc_of(f"{self.d}/build_request.log") if lctx is not None else 1
        for _, lf, _, _ in extra:
            rc2 = rc_of(lf) if rc_of(lf) != 0 else rc2
        rc2 = rc_x if rc_x != 0 else rc2
        self.log(f"Build derives ran {jobs_n}-wide (SYS-06): step rc={rc1} request rc={rc2} shapes={len(extra) + 1} wall={int(time.time()) - t0}s")
        return rc1, rc2

    def derived_dirs(self) -> list[str]:
        out = ["--derived", f"{self.d}/build_request"]
        for dd in sorted(glob.glob(f"{self.d}/build_request_LP*_T*")):
            if os.path.isdir(dd):
                out += ["--derived", dd]
        return out

    def follow_compose(self, to: int) -> tuple[Any, str, str]:
        """B >= 2: the workload Program's compose and the manifest beside the derives (`row_build.follow_compose`)."""
        return RB.follow_compose(self, to)

    def build(self) -> None:
        scale = max(1, (max(self.LP, 256) // 256) * (max(self.T, 31) // 31))
        to = self.cb.build_timeout or min(CAP, 900 * scale)
        t0 = int(time.time())
        self.tl("stage.build", "start")
        self.gpu_exclusive("build")
        follow = self.follow_compose(to) if self.B > 1 else None
        rc1, rc2 = self.derive_all(to)
        wall = int(time.time()) - t0
        self.tl("build.summary", "start")
        ok = self.guarded("build summary", R.build_summary, self.d)
        wd = ""
        early = None
        self.tl("build.summary", "end", ("ok", ok))
        if follow is not None:
            fut, done, stop = follow
            open(done if ok == "PASS" else stop, "w").close()
            rcw, early = fut.result()
            for f in (done, stop):
                if os.path.exists(f):
                    os.remove(f)
        if ok == "PASS":
            if follow is None:
                rcw = self.run_tl("build.global_program", to, f"{self.d}/build_workload.log",
                                  self.cli("global-program", "--workload", self.wl, *self.derived_dirs(), "--out", f"{self.d}/build_workload",
                                           "--checkpoint-dir", self.ck, py=self.py312))
            wd = (R.load_json(f"{self.d}/build_workload/workload_program.json") or {}).get("workload_digest") or ""
            if not (rcw == 0 and wd):
                ok = "FAIL"
                self.log(f"workload Program compose failed rc={rcw}: {R.compose_reason(self.d)} (see build_workload.log)")
            self.guarded("workload summary", R.add_workload, self.d, wd, rcw)
        if ok == "PASS":
            rc, line = self.target_family("--no-device", "--build-summary", f"{self.d}/build_summary.json", out=f"{self.d}/target_family.build.json")
            if rc != 0:
                ok = "FAIL"
                self.verdict("build-target", "FAIL", line)
            elif line.startswith("TARGET-FAMILY-WAIVED"):
                self.verdict("build-target", "WAIVED", line, quiet=True)
        mrc = 0
        if ok == "PASS" and os.path.isfile(f"{self.d}/build_request/instances.json.gz"):
            mrc = self.required_manifest("build") if early is None else early
            if mrc == 3:
                self.verdict("build", "FAIL", f"PRECHECK_FAIL_REGISTRY_UNAVAILABLE (environment: core verity package not importable; see {self.d}/manifest.log)")
                self.tl("stage.build", "end", ("ok", "false"))
                self.tl("row", "end", ("ok", "false"), ("stopped_at", "build"))
                raise Stop(3)
            if mrc != 0:
                ok = "FAIL"
                self.verdict("build-manifest", "FAIL", "required-value manifest not built (manifest.log)")
        elif early == 0 and os.path.exists(f"{self.d}/manifest.json"):
            os.remove(f"{self.d}/manifest.json")
        digest, man = R.build_line(self.d)
        self.verdict("build", ok, f"rc={rc1}/{rc2} wall={wall}s digest={digest} workload_digest={wd[:16]} manifest={man}")
        self.tl("stage.build", "end", ("ok", ok), ("wall_s", wall))
        if ok != "PASS":
            self.log("BUILD FAIL -> stop")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "build"))
            raise Stop(10 if mrc == 0 else 13)
        self.call_boundaries(to, RP.call_boundaries_cmd(self))

    # ------------------------------------------------------------------------------------------------ Match
    def derive_missing(self, derived: list[str], label: str) -> None:
        """Derive the request shapes X-09 named as missing (one request-wrapper derivation per (LP, T))."""
        for lp, t in R.missing_derived(self.d):
            self.log(f"{label}: deriving request Program LP={lp} T={t}")
            out = f"{self.d}/build_request_LP{lp}_T{t}"
            self.run_to(self.cb.build_timeout or 900, f"{out}.log", self.request_cmd(lp, t, out))
            derived += ["--derived", out]

    def compare(self, to_c: int) -> tuple[int, int | None, bool]:
        """The per-request comparison, and GM-01 when it runs inside it (``shared``) -> (compare rc, GM rc, shared)."""
        d, ws = self.d, self.cm.within_step
        if self.B > 1:
            derived = self.derived_dirs()
            if self.cm.match_pipeline == "shared" and os.path.exists(f"{d}/build_workload/workload_program.json"):
                # one X-09 per row: GM-01 runs first and writes the decomposition it computed (--decomp-out, batch_decomp's schema)
                self.tl("match.compare", "end", ("rc", "null"), ("shared", "match.global_match"))
                gm_rc = None
                for p in (1, 2):
                    gm_rc = self.run_tl("match.global_match", to_c, f"{d}/global_match.log",
                                        self.cli("global-match", "--gprog", f"{d}/build_workload/workload_program.json", "--workload", self.wl,
                                                 "--oracle", f"{d}/match", "--out", f"{d}/global_match.json", "--within-step", ws,
                                                 "--decomp-out", f"{d}/match_decomp.json", *derived, py=self.py312))
                    if not (gm_rc == 4 and p == 1):
                        break
                    self.derive_missing(derived, "X-09 (inside GM-01)")
                open(f"{d}/match_compare.log", "w").write("$ (shared, R17-28 P1) X-09 document written by cb_a.global_match --decomp-out; see global_match.log\n")
                return R.decomp_rc(d), gm_rc, True
            rcc = 3
            for p in (1, 2):
                rcc = self.run_to(to_c, f"{d}/match_compare.log",
                                  self.cli("batch-decomp", "--oracle", f"{d}/match", *derived, "--out", f"{d}/match_decomp.json",
                                           "--within-step", ws, py=self.py312))
                if not (rcc == 4 and p == 1):
                    break
                self.derive_missing(derived, "X-09")
            return rcc, None, False
        # B=1: the Program of the request's SERVED shape (early EOS: T_r = fold steps - 1 < cap - 1), derived when not on disk
        req_dir, t_r = f"{d}/build_request", R.fold_steps(d, self.T)
        if t_r > 0 and t_r != self.T:
            req_dir = f"{d}/build_request_LP{self.LP}_T{t_r}"
            self.verdict("eos-truncated", f"T_r={t_r}", f"cap_T={self.T} LP={self.LP} (greedy hit EOS before the cap; Match against the served shape, MOE-05)",
                         quiet=True)
            if not os.path.isfile(f"{req_dir}/result.json"):
                self.log(f"early EOS: served T_r={t_r} != cap-1={self.T} -> deriving request Program LP={self.LP} T={t_r}")
                scale = max(1, (max(self.LP, 256) // 256) * (max(t_r, 31) // 31))
                self.run_to(self.cb.build_timeout or min(CAP, 900 * scale), f"{req_dir}.log", self.request_cmd(self.LP, t_r, req_dir))
        open(f"{d}/match_shape.json", "w").write(f'{{"LP": {self.LP}, "T": {t_r}, "cap_T": {self.T}, "derived": "{req_dir}"}}\n')
        return self.run_to(to_c, f"{d}/match_compare.log",
                           self.cli("program-compare", "--a", req_dir, "--b", f"{d}/match", "--out", f"{d}/match_compare.json", py=self.py312)), None, False

    def match(self) -> None:
        d, cm = self.d, self.cm
        sb, st = max(1, self.B), max(1, (self.I + self.O + 287) // 288)
        to = cm.match_timeout or min(CAP, 1800 * sb * st)
        to_c = cm.compare_timeout or min(CAP, 600 * max(sb, 4) * st)
        self.child_env["VERITY_INSTANCES_FORM"] = cm.instances_form
        self.log(f"fold: instances.jsonl stored form = {cm.instances_form} (F2)")
        self.child_env["MATCH_PHASE"] = cm.match_phase
        if cm.match_phase == "cpu":
            if not os.path.isdir(f"{d}/match/capture"):
                self.verdict("match", "FAIL", f"MATCH_PHASE=cpu without {d}/match/capture")
                raise Stop(12)
            self.log("match: MATCH_PHASE=cpu over the existing capture (no GPU stage)")
        if cm.acquire_manifest:
            self.child_env["ACQUIRE_MANIFEST"] = cm.acquire_manifest
        elif os.path.isfile(f"{d}/manifest.json"):
            self.child_env["ACQUIRE_MANIFEST"] = f"{d}/manifest.json"
            self.log(f"acquisition manifest for the capture: {d}/manifest.json (ACQUIRE_ENGINE={self.env.environ.get('ACQUIRE_ENGINE') or 'v2'})")
        if cm.match_phase == "all" and self.snap_def != "all" and os.path.isfile(f"{d}/manifest.json"):
            self.form_b()
        cmd_m = self.match_cmd()
        t0 = int(time.time())
        self.tl("stage.match", "start")
        self.gpu_exclusive("match")
        rcm = self.run_tl("match.run_config", to, f"{d}/match.log", cmd_m)
        self.tl("match.compare", "start")
        if os.path.isfile(f"{d}/match/program.json") and os.path.isfile(f"{d}/match/instances.jsonl"):
            rcc, gm_rc, shared = self.compare(to_c)
            if not shared:
                self.tl("match.compare", "end", ("rc", rcc))
                if os.path.exists(f"{d}/build_workload/workload_program.json"):
                    gm_rc = self.run_tl("match.global_match", to_c, f"{d}/global_match.log",
                                        self.cli("global-match", "--gprog", f"{d}/build_workload/workload_program.json", "--workload", self.wl,
                                                 "--oracle", f"{d}/match", "--out", f"{d}/global_match.json", "--within-step", cm.within_step,
                                                 py=self.py312))
                else:
                    open(f"{d}/global_match.log", "w").write("no build_workload/workload_program.json: the global Match needs this row's Build workload Program (GP-01)\n")
                    gm_rc = 98
        else:
            rcc, gm_rc = 99, 99
            open(f"{d}/match_compare.log", "w").write(f"no program.json/instances.jsonl in {d}/match (fold did not run or failed)\n")
            self.tl("match.compare", "end", ("rc", 99))
        rcs = -1
        if os.path.isfile(f"{d}/match/capture/log.jsonl.gz") and self.stochastic():
            rcs = self.run_tl("match.stoch_recompute", to_c, f"{d}/match_stoch.log",
                              self.cli("stoch-recompute", "--run", f"{d}/match", "--workload", self.wl, "--build", d, "--out", f"{d}/match_stoch.json",
                                       py=self.py312))
            self.log(f"stoch value check rc={rcs}: {_tail(f'{d}/match_stoch.log')}")
        wall = int(time.time()) - t0
        gm = 97 if gm_rc is None else gm_rc
        self.tl("match.summary", "start")
        ok = self.guarded("match summary", R.match_summary, d, rcm, rcc, gm, t0, rcs, cm.match_phase == "cpu")
        self.tl("match.summary", "end", ("ok", ok))
        self.verdict("match", ok, f"rc={rcm}/{rcc}/{gm} wall={wall}s {self.guarded('match line', R.match_line, d)}")
        self.tl("stage.match", "end", ("ok", ok), ("wall_s", wall))
        if ok != "PASS":
            self.log("MATCH FAIL -> stop")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "match"))
            raise Stop(11)

    def match_cmd(self) -> list[str]:
        d, cm = self.d, self.cm
        snapmax = ["--snapshot-max-bytes", cm.snapshot_max_bytes] if cm.snapshot_max_bytes else []
        # the Programs the Match compares against (B=1 the request Program, B>=2 its components): the fold binds their sampler construction
        progs = [p for p in ([f"{d}/build_request"] if self.B <= 1 else sorted(glob.glob(f"{d}/build_request_LP*_T*"))) if os.path.isfile(f"{p}/result.json")]
        return self.cli("match", "--case", self.role, "--workload", self.wl, "--out", f"{d}/match", "--checkpoint-dir", self.ck,
                        "--phase", cm.match_phase, "--python-gpu", self.py, "--python-cpu", self.py312, "--max-num-seqs", str(self.B),
                        "--snapshot-steps", self.snap_steps or self.snap_def, *snapmax, *(x for p in progs for x in ("--program-dir", p)), "--skip", SKIP,
                        "--engine-arg", f"gpu_memory_utilization={self.gpu_util}", "--cards-dir", f"{d}/match/cards", py=self.py312)

    def stochastic(self) -> bool:
        try:
            return R.is_stochastic(self.path(self.wl))
        except (OSError, ValueError, TypeError, AttributeError):
            return False

    def form_b(self) -> None:
        """Required values the replay has no evaluator for are value-checked by form (B) alone, complete only over EVERY
        step: the row default snapshot steps become ``all`` for them (an explicit ``--snapshot-steps`` stands, named)."""
        from verity_vllm.check.replay.coverage import form_b_families
        try:
            f = form_b_families(json.load(open(f"{self.d}/manifest.json")))
            fb, rc = " ".join(f"{k}x{v}" for k, v in sorted(f.items())), 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            open(f"{self.d}/row.log", "a").write(f"form (B) families: {type(e).__name__}: {e}\n")
            fb, rc = "", 1
        if rc != 0:
            self.log(f"WARN match: form (B) families not derived (rc={rc}, see row.log): snapshot steps stay {self.snap_steps or self.snap_def}")
        elif fb and not self.snap_steps:
            self.snap_def = "all"
            self.log(f"match: snapshot steps = all (the Commit's form (B) must be complete: required values with no replay evaluator: {fb})")
        elif fb and self.snap_steps != "all":
            self.log(f"WARN match: MATCH_SNAP_STEPS={self.snap_steps} explicit while form (B) must be complete ({fb}): the Commit's form (B) stays partial")

    # ------------------------------------------------------------------------------------------------ Commit
    def commit_stop(self, rc: int, detail: str, stage: str = "commit") -> None:
        self.verdict(stage, "FAIL", detail)
        self.tl(f"stage.{stage}", "end", ("ok", "false"))
        self.tl("row", "end", ("ok", "false"), ("stopped_at", stage))
        raise Stop(rc)

    def commit_cmd(self, taps: list[str]) -> list[str]:
        """The Commit's command line; `taps`: the row's taps that apply to it (`row_records.applicable_taps`), whose sources it attaches."""
        ck = self.ck_
        pairs, warmup = ("1", "1") if self.config_run else (str(ck.pairs), "2")
        cmd = self.cli("commit", "--case", self.role, "--workload", self.wl, "--committer", "native_collect_v2b", "--fast-copy", "--gpu-tree",
                       "--hidden-so", self.hidden_so, "--openings", "64", "--pairs", pairs, "--warmup", warmup, "--max-num-seqs", str(self.B),
                       "--engine-arg", f"gpu_memory_utilization={self.gpu_util}", "--label", self.row, "--out", f"{self.d}/commit")
        if self.config_run:                           # one instrumented run (and its control with a baseline), bounded staging, k units
            cmd += [] if ck.config_baseline == "1" else ["--only-arm", "instrumented"]
            cmd += ["--bounded-staging", "--replay-k", str(ck.replay_k), "--replay-draw", ck.replay_draw]
            if ck.replay_deferred in ("1", "slim"):
                cmd += ["--replay-deferred", *(["slim"] if ck.replay_deferred == "slim" else [])]
        if self.protocols:
            cmd += self.tgt_args                      # the declared target with the row's protocols; unset, the Commit reads the workload's
        # the windowed collector (ring of window_slots x window_mb MiB, raw copies retained on host, openings verified after release)
        # is the default; window_mb=0 is per-step blocks, retain=device the device-resident footprint
        if int(ck.window_mb) > 0:
            cmd += ["--window-mb", str(ck.window_mb), "--window-slots", str(ck.window_slots)]
        cmd += ["--retain", ck.retain]
        if ck.open_after_release != "0":
            cmd.append("--open-after-release")
        cmd += ["--tap-cap-mb", str(ck.tap_cap_mb)]
        for ea in (ck.extra_engine_args or "").split():
            cmd += ["--engine-arg", ea]
        cmd += (ck.extra_commit_args or "").split()
        if ck.replay_cache == "1":
            cmd.append("--replay-cache")
        if ck.required_classes:
            cmd += ["--required-classes", ck.required_classes]
        if ck.vu_export != "0" and not self.config_run:
            cmd += ["--vu-export-dir", self.vu_export_dir()]
        if ck.serving_rows:
            cmd += ["--serving-rows", self.serving_rows_request(ck.serving_rows)]
        return cmd + R.tap_source_args(taps, self.env)

    def vu_export_dir(self) -> str:
        """Where the Commit exports the row's sampled VUs (`pipeline.vu_export`): inside the research run dir when there is one, so the
        run's custody preserves them; else beside the row.  `provenance.json` there names the row and the shipped source."""
        rd = self.env.research_run_dir
        d = os.path.join(rd, "vu-export", self.row) if rd else f"{self.d}/vu-export"
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "provenance.json"), "w") as f:
            json.dump({"row_key": self.row, "research_run_dir": rd, "research_run": self.env.environ.get("RESEARCH_RUN_ID"),
                       "source": {"commit": self.env.environ.get("RESEARCH_SOURCE_SHA"),
                                  "dirty": False if self.env.environ.get("RESEARCH_SOURCE_SHA") else None}}, f)
        return d

    def program_digest(self) -> str:
        """The Program digest the Commit is keyed by: the one named, else the Build's of record ("" when unreadable)."""
        try:
            return self.ck_.program_digest or R.program_digest_of_record(self.d, self.B)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return ""

    def manifest_of_record(self, stage: str = "commit") -> str:
        """The required-value manifest the Commit runs over: the one named, else the Build's when it is of record (`R.manifest_not_of_record`:
        the query of record with a partition, the Build's Program digest, the row's tap policies, a passing word check); any other is moved
        aside, never used, and the manifest rebuilt here from the Build's request Programs."""
        d = self.d
        man = self.ck_.required_manifest or ""
        if not man and os.path.isfile(f"{d}/manifest.json"):
            slug, why = R.manifest_not_of_record(R.load_json(f"{d}/manifest.json"), self.program_digest(), self.ck_)
            if not why:
                man = f"{d}/manifest.json"
                self.log(f"required-value manifest of record: {man} (built at the Build stage: {_tail(f'{d}/manifest.log', 200)})")
            else:
                prev = f"{d}/manifest.{slug}-{_stamp()}.json"
                os.replace(f"{d}/manifest.json", prev)
                self.log(f"manifest.json beside the Build {why}: moved aside ({os.path.basename(prev)}), rebuilding the manifest of record")
        if not man and (os.path.isfile(f"{d}/build_request/instances.json.gz") or glob.glob(f"{d}/build_request_LP*_T*")):
            mrc = self.required_manifest(stage)
            if mrc == 3:
                self.commit_stop(3, f"PRECHECK_FAIL_REGISTRY_UNAVAILABLE (environment: core verity package not importable; see {d}/manifest.log)", stage)
            if mrc != 0:
                self.commit_stop(13, f"required-value manifest not built (B={self.B}; manifest.log)", stage)
            man = f"{d}/manifest.json"
        return man

    def match_record_precheck(self) -> None:
        """A Commit over a Match record refuses to start without the Match account files (small; restore or rerun match)."""
        d = self.d
        if not (os.path.isdir(f"{d}/match") and os.path.isfile(f"{d}/match/instances.jsonl")):
            return
        missing = [f for f in ("match/capture/tokens.json", "match/control/tokens.json", "global_match.json") if not os.path.isfile(f"{d}/{f}")]
        if missing:
            lacks = "".join(" " + f for f in missing)
            self.log(f"PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE: the Match record beside this attempt lacks{lacks} (capture moved / removed?) -> Commit not started")
            self.tl("stage.commit", "end", ("rc", 3), ("fail_class", "PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE"), ("missing", " ".join(missing)))
            self.verdict("commit", "FAIL", f"PRECHECK_FAIL_MATCH_RECORD_INCOMPLETE: Match record lacks{lacks} -- restore the file(s) (small) or run match; "
                         "the Commit was NOT run (infra, not a verdict)")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "commit"))
            raise Stop(14)

    def weights_of_record(self, stage: str = "commit") -> list[str]:
        """The weights of record beside the component Program (reused only when it is this Program's, of this tree's schema
        and complete; else moved aside and re-derived from the pinned checkpoint dir) -> the Commit's flags."""
        from verity_vllm.check.weights_of_record import SCHEMA
        d, ck = self.d, self.ck
        pd = f"{d}/build_request"
        if self.B > 1 and os.path.isfile(f"{d}/manifest.json"):
            pd = R.wor_component(d)
            if pd != f"{d}/build_request":
                self.log(f"weights of record: B={self.B} -> derived from the component Program {os.path.basename(pd)} (of-record set)")
        wor, prev = f"{pd}/weights_of_record.json", ""
        if os.path.isfile(wor) and ck and os.path.isdir(ck):
            stale = R.wor_stale(wor, f"{pd}/result.json", SCHEMA)
            if stale:
                prev = f"{pd}/weights_of_record.stale-{_stamp()}.json"
                os.replace(wor, prev)
                self.log(f"weights of record: existing record beside {os.path.basename(pd)} not reused -- {stale}; moved aside ({os.path.basename(prev)}), re-deriving")
        if not os.path.isfile(wor) and ck and os.path.isdir(ck):
            self.tl(f"{stage}.weights_of_record", "start")
            with open(f"{d}/weights_of_record.log", "w") as f:
                rc = subprocess.run(self.cli("weights-of-record", "--program-dir", pd, "--checkpoint-dir", ck, "--out", wor,
                                             *([f"--supersedes={prev}"] if prev else [])),
                                    stdout=f, stderr=subprocess.STDOUT, cwd=self.tree, env=self.child_env).returncode
            last = _tail(f"{d}/weights_of_record.log", 10**6)
            if rc == 0:
                self.tl(f"{stage}.weights_of_record", "end", ("rc", 0))
                self.log(f"weights of record: {last}")
            else:
                self.tl(f"{stage}.weights_of_record", "end", ("rc", 1))
                self.log(f"WARNING weights of record not derived (see weights_of_record.log): {last}")
        elif not os.path.isfile(wor):
            self.log(f"WARNING weights of record NOT derived: no local checkpoint dir for the workload's (model, tokenizer_revision) (CK='{ck}'; "
                     "checkpoints.json local_path / HF hub snapshot absent on this pod) -- the Commit will FAIL 'weights unpinned (MAN-09)' by name")
            self.tl(f"{stage}.weights_of_record", "mark", ("reused", "false"), ("derived", "false"))
        else:
            self.tl(f"{stage}.weights_of_record", "mark", ("reused", "true"))
        if os.path.isfile(wor):
            return ["--weights-of-record", wor]
        return ["--checkpoint-dir", ck] if ck else []

    def manifest_verify(self, man: str) -> None:
        """Rebuild the manifest from this attempt's Program x workload by the query path and record it beside the Commit
        (``commit/manifest_verify.json``); a mismatch fails the verdict's manifest-of-record check."""
        d = self.d
        cmd = self.cli("manifest-verify", "--manifest", man, "--out", f"{d}/commit/manifest_verify.json")
        if self.B > 1:
            wd = R.workload_digest(d)
            cmd += ["--fixture", self.wl, "--programs-root", d, *(["--workload-digest", wd] if wd else [])]
        else:
            cmd += ["--program", f"{d}/build_request", "--workload", self.wl]
        cmd += R.tap_policy_flags(self.ck_)
        self.tl("validate.manifest_verify", "start")
        with open(f"{d}/manifest_verify.log", "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=self.tree, env=self.child_env).returncode
        if rc == 0:
            self.tl("validate.manifest_verify", "end", ("rc", 0))
            self.log("manifest of record VERIFIED (manifest_verify: rebuilt by the query path from this attempt's Program x workload == used; commit/manifest_verify.json)")
        else:
            self.tl("validate.manifest_verify", "end", ("rc", 1))
            self.log(f"manifest of record VERIFY FAILED: {_tail(f'{d}/manifest_verify.log', 200)}")

    def commit_prepare(self) -> tuple[int, list[str]]:
        """The Commit stage up to its command line -> (its timeout, its command)."""
        d, ck = self.d, self.ck_
        kb, kt = max(1, (self.B + 3) // 4), max(1, (self.I + self.O + 287) // 288)
        to = ck.commit_timeout or min(CAP, 2400 * kb * kt)
        self.log(f"commit timeout {to}s (B={self.B} I={self.I} O={self.O}; x{kb} x{kt} over the 2400 s floor)")
        self.tl("stage.commit", "start")
        if not os.path.isfile(self.hidden_so):
            self.commit_stop(12, f"hidden .so missing: {self.hidden_so}")
        self.tl("commit.prepare", "start")
        if self.B <= 1:
            served = R.served_shape(d)
            if served:
                self.log(f"B=1 EOS-stopped request ({served}, match_shape.json): Program of record at Commit = the CAP-shape build_request; the positions "
                         "past the executed prefix are classified not-executed by rule, not cut from the population (the Match-side served-shape comparison stands)")
        man = self.manifest_of_record()
        taps = R.applicable_taps(man, ck) if man else []
        why = R.missing_tap_build(taps, self.env)
        if why:
            self.commit_stop(12, why)
        self.log(f"taps: {', '.join(taps) or 'none'} (on, and the manifest names their families); guarded max {'on' if self.guarded_max() else 'off'}")
        cmd = self.commit_cmd(taps)
        if man:
            cmd += ["--required-manifest", man]
        pdig = self.program_digest()
        if pdig:
            cmd += ["--program-digest", pdig]
        cmd += (ck.commit_args or "").split()
        self.match_record_precheck()
        if os.path.isdir(f"{d}/match") and os.path.isfile(f"{d}/match/instances.jsonl"):
            snap = R.snapshot_steps_of_record(d) or f"{self.snap_steps or self.snap_def} (env; run.json unreadable)"
            cmd += ["--match-dir", f"{d}/match"]
            self.log(f"C2 live oracle-compare armed: --match-dir {d}/match (Match record of this attempt; snapshot steps {snap})")
        elif self.B > 1:
            self.log(f"WARNING B={self.B}: no {d}/match/instances.jsonl: live C2 oracle-compare NOT armed -- request attribution of decode-step identities unverified")
        cmd += RP.commit_flags(self)
        self.tl("commit.prepare", "end")
        return to, cmd

    def commit(self) -> None:
        d, ck = self.d, self.ck_
        if ck.group:
            self.replay_pending = RG.commit(self, CAP)
            return
        to, cmd = self.commit_prepare()
        t0 = int(time.time())
        if ck.hot_engine == "1" and not self.leased:
            self.log(f"HOT_ENGINE=1: Commit on the hot worker (root {self.env.hot_root}, idle {self.env.hot_idle}s)")
            rck = self.run_tl("commit.commit_delta", to, f"{d}/commit.log",
                              self.cli("hot", "submit", "--root", self.env.hot_root, "--idle", self.env.hot_idle, "--", *cmd[4:]))
        else:
            self.gpu_exclusive("commit-cold")
            rck = self.run_gpu("commit.commit_delta", to, f"{d}/commit.log", cmd)
        wall = int(time.time()) - t0
        if self.config_run:
            self.replay_pending = RC.commit_verdict(self, rck, wall)
            return
        if rck == 0 and man:
            self.manifest_verify(man)
        self.tl("closure.verdict", "start")
        with open(f"{d}/row.log", "a") as f:
            r = subprocess.run(self.cli("verdict", "from-record", d, "--rc", str(rck)), stdout=subprocess.PIPE, stderr=f, text=True,
                               cwd=self.tree, env=self.child_env)
        vout = "\n".join(ln.split(" ", 1)[0] for ln in r.stdout.splitlines()) or "ERROR"
        self.tl("closure.verdict", "end", ("outcome", vout))
        ok = "PASS" if vout == "PASS" else "FAIL"
        self.log(f"verdict.json (vllm-verdict/v1): {vout} -> Commit stage {ok}")
        self.verdict("commit", ok, f"rc={rck} wall={wall}s outcome={vout} {R.commit_line(d)} manifest_verify={R.manifest_verify_ok(d)}")
        self.tl("stage.commit", "end", ("ok", ok), ("wall_s", wall))
        if ok != "PASS":
            self.log("COMMIT FAIL -> stop")
            self.tl("row", "end", ("ok", "false"), ("stopped_at", "commit"))
            raise Stop(12)
