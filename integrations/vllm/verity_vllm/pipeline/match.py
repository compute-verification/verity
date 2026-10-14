"""bench/run_config.py — one-shot driver: every stage of one configuration into one run directory, then gates + card.

    verity-vllm match --config-id ID --profile NAME --workload PATH --case B0|B1 --out RUN_DIR \\
        --phase gpu|cpu|all [--python-gpu PY] [--python-cpu PY] [--snapshot-steps 0,1,2,3] [--max-num-seqs 8] \\
        [--laptop] [--dry-run] [--tier-c] [--skip STAGES] [--checkpoints PATH] [--checkpoint-dir DIR_OR_GLOB] \\
        [--log LOG --snapshots DIR] [--control-tokens PATH] [--other-log LOG] [--cards-dir out/gen/cards] [--k 3] [--seed S]

GPU phase (the pod's torch venv, Python 3.11):
    capture         bench.m1_capture --snapshot-steps 0,1,2,3 (contiguous from 0: replay of step N finds every earlier
                    KV producer) -> RUN_DIR/capture/{log.jsonl.gz, tokens.json, snapshots.json, values/, stats.json,
                    versions.json, cos_sin_cache.npy}
    control         bench.m1_capture --no-observer -> RUN_DIR/control/tokens.json  — THE token reference of every gate
    census          census --workload --out RUN_DIR/census (torch.profiler kernel census)
    nonint          noninterference (a observer / b hooks / c bare) -> RUN_DIR/nonint, `--case` so expected_layers resolves,
                    `--accepted-ctl RUN_DIR/control/tokens.json` (the control run, never a workload-authored list)
    -> RUN_DIR/gpu_done.json with timings
CPU phase (Python 3.12, torch-free; pod or laptop):
    resolve           bench.resolve_log -> RUN_DIR/resolution.json
    census_reconcile  census --reconcile -> RUN_DIR/census_reconcile.json
    nonint_compare    noninterference --compare -> RUN_DIR/nonint/noninterference.json
    compare           verity_vllm.check.fold_compare -> RUN_DIR/compare.json (tokens, boundary, subcircuits, structure)
    fold, fold_2      this module's fold worker in two separate processes -> RUN_DIR/fold_summary.json, fold_summary_2.json
    fold_other        (with --other-log) a second capture's digest -> RUN_DIR/fold_summary_other.json (G6 detail only)
    root_policy       verity_vllm.check.root_policy -> RUN_DIR/root_policy.json
    golden            verity_vllm.properties.golden --check -> RUN_DIR/golden.json
    wiring            verity_vllm.check.operand_provenance -> RUN_DIR/wiring.json (G5: exhaustive operand provenance, D82)
    replay_a          verity_vllm.check.replay.replay --tier a --k 10 [--jobs N] -> RUN_DIR/replay_a.json (G4 bodies; G5 sampled VUs + served tokens)
    replay_chain      verity_vllm.check.replay.replay --tier chain -> RUN_DIR/replay_chain.json (G5 chaining smoke test)
    replay_b_step<N>  (with --full-replay) verity_vllm.check.replay.replay --tier b --step N -> replay_b_step<N>.json (diagnostic only)
    replay_c          (with --tier-c) verity_vllm.check.replay.replay --tier c -> RUN_DIR/replay_c.json (diagnostic only)
    gates             in-process: gates.evaluate_gates -> RUN_DIR/gates.json; card -> RUN_DIR/card.json, RUN_DIR/CARD.md,
                      <cards-dir>/<config_id>.json

Every subprocess stage tees stdout/stderr to RUN_DIR/logs/<stage>.log and records {stage, status, seconds, returncode,
log, cmd, reason} in RUN_DIR/stages.json; a failed stage never aborts later independent stages (its gates come out
skipped with the log as evidence).  `--dry-run` prints the exact command lines and runs nothing.  `--skip a,b` skips
stages or groups (`replay`, `census`, `nonint`, `fold`, `compare`, `resolve`, `root_policy`, `golden`, `capture`,
`control`, `gates`).  `--laptop`: every stage is watched (RSS above --max-rss-gb, default 1.5, or wall above
--stage-timeout, default 180 s, kills it), replay gets `--max-arg-leaves 4000000`.

Internal: `--worker fold --log LOG --profile NAME --out fold_summary.json [--no-program]` folds a log in this process
and writes the `gates.FOLD_SUMMARY_SCHEMA` summary.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from verity_vllm.check import gates as G
from verity_vllm.config import UsageError, option
from verity_vllm.observe.fold.supplied import program_registered_values
from verity_vllm.pipeline import experiment as EXP
from verity_vllm.program.sampling_event import program_sampler_construction

DEFAULT_SNAPSHOT_STEPS = "all"        # evidence for every specialization (h1: attention T at non-snapshot steps was uncovered)
DEFAULT_TIER_B_STEPS = "0,1,2,3"       # --full-replay (diagnostic): prefill + three decode steps
DEFAULT_K = 10                         # tier a: instances per (specialization, regime) stratum (D82; gate minimum TIER_A_K)
DEFAULT_CHAINS = 3                     # tier chain: connected slices replayed with computed intermediate values
DEFAULT_PARALLEL_REPLAY = 4            # replay stages are independent processes; the pod has the cores, the laptop runs them serially
DEFAULT_CHECKPOINTS = "manifests/checkpoints.json"
SNAPSHOT_CAP_FLOOR = 32 << 20          # m1_capture's per-tensor snapshot cap of record until R16 (D87)
SNAPSHOT_ROW_BYTES_FALLBACK = 64 << 10  # per-token activation row bound when no HF config is readable: 2 x I x bf16 for I <= 16384
LAPTOP_MAX_ARG_LEAVES = 4_000_000
LAPTOP_MAX_RSS_GB = 1.5
LAPTOP_STAGE_TIMEOUT_S = 180.0

GPU_STAGES = ("capture", "control", "census", "nonint")
CPU_STAGES = ("resolve", "census_reconcile", "nonint_compare", "compare", "fold", "fold_2", "fold_other", "root_policy", "golden",
              "wiring", "replay_a", "replay_chain", "replay_b", "replay_c", "gates")
SKIP_GROUPS = {
    "replay": {"replay_a", "replay_chain", "replay_b", "replay_c"},
    "census": {"census", "census_reconcile"},
    "nonint": {"nonint", "nonint_compare"},
    "fold": {"fold", "fold_2", "fold_other"},
}
#: Stages whose module may be missing in a worktree (a sibling lane's / the integrator's); a missing module is recorded
#: as `missing_module`, not run.
OPTIONAL_MODULES = {"root_policy": "verity_vllm.check.root_policy", "golden": "verity_vllm.properties.golden", "replay_a": "verity_vllm.check.replay.replay",
                    "replay_chain": "verity_vllm.check.replay.replay", "replay_b": "verity_vllm.check.replay.replay", "replay_c": "verity_vllm.check.replay.replay",
                    "wiring": "verity_vllm.check.operand_provenance"}


# ---------------------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------------------


class Plan:
    """Paths and command lines of one run (pure: building a plan touches nothing)."""

    def __init__(self, a: Options) -> None:
        self.a = a
        self.run = Path(a.out)
        self.py_gpu = a.python_gpu or sys.executable
        self.py_cpu = a.python_cpu or sys.executable
        if str(a.snapshot_steps).strip().lower() == "all":
            # every step of the workload: its declared `expected_engine_steps` (D87: arrivals / chunking make the step count
            # exceed max(max_tokens)), else max(max_tokens); a listed step that never happens costs nothing
            n = a.max_tokens if a.max_tokens is not None else _workload_expected_steps(a.workload)
            if not n:
                print("[plan] --snapshot-steps all: workload max_tokens unknown, falling back to 0,1,2,3", file=sys.stderr)
                n = 4
            self.steps = list(range(int(n)))
        else:
            self.steps = [int(x) for x in str(a.snapshot_steps).split(",") if x.strip() != ""]
        if not self.steps or self.steps[0] != 0 or self.steps != list(range(len(self.steps))):
            raise SystemExit(f"--snapshot-steps must be contiguous from 0 (got {a.snapshot_steps}): replay of step N needs every earlier KV producer")
        # --full-replay (diagnostic, D82): tier b replays step 0 and these decode steps; every one must be a snapshot step
        tb = [int(x) for x in str(a.tier_b_steps).split(",") if x.strip() != ""] if a.tier_b_steps else list(self.steps)
        self.tier_b_steps = sorted({0, *tb} & set(self.steps)) if getattr(a, "full_replay", False) else []
        self.verify_weights = not a.no_verify_weights
        self.parallel = 1 if (a.laptop or a.parallel is not None and a.parallel <= 1) else (a.parallel or DEFAULT_PARALLEL_REPLAY)
        self.capture_dir = self.run / "capture"
        self.log = Path(a.log).resolve() if a.log else self.capture_dir / "log.jsonl.gz"
        self.snapshots = Path(a.snapshots).resolve() if a.snapshots else (self.log.parent if a.log else self.capture_dir)
        self.control_dir = self.run / "control"
        self.census_dir = self.run / "census"
        self.nonint_dir = self.run / "nonint"
        self.skip = self._skip_set(a.skip)
        self.laptop = bool(a.laptop)
        self.checkpoint_dir = a.checkpoint_dir

    @staticmethod
    def _skip_set(s: str | None) -> set[str]:
        out: set[str] = set()
        for tok in (s or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            out |= SKIP_GROUPS.get(tok, {tok})
        return out

    def skipped(self, stage: str) -> bool:
        base = "replay_b" if stage.startswith("replay_b_step") else stage
        return stage in self.skip or base in self.skip

    # paths ---------------------------------------------------------------------------------------------

    def p(self, key: str) -> Path:
        return self.run / G.ARTIFACTS[key]

    def artifacts_override(self) -> dict:
        """`run.json["artifacts"]` when an existing capture is supplied with --log/--snapshots."""
        out: dict[str, str] = {}
        if self.a.log or self.a.snapshots:
            sd = self.snapshots
            out.update({"log": str(self.log), "snapshots": str(sd / "snapshots.json"), "tokens": str(sd / "tokens.json"),
                        "versions": str(sd / "versions.json"), "stats": str(sd / "stats.json"), "cos_sin": str(sd / "cos_sin_cache.npy")})
        if self.a.control_tokens:
            out["control_tokens"] = str(self.control_tokens_path())
        return out

    def control_tokens_path(self) -> Path:
        return Path(self.a.control_tokens).resolve() if self.a.control_tokens else self.control_dir / "tokens.json"

    def safetensors_glob(self) -> str | None:
        cd = self.checkpoint_dir
        if not cd:
            return None
        if any(ch in cd for ch in "*?["):
            return cd
        p = Path(cd)
        return str(p / "model*.safetensors") if p.is_dir() or not p.suffix else str(p)

    # commands ------------------------------------------------------------------------------------------

    def engine_args(self) -> list[str]:
        """`key=value` engine overrides: the workload's `engine_args_required` first, then the CLI's (an explicit CLI value wins)."""
        req = _workload_engine_args_required(self.a.workload)
        cli = _engine_arg_dict(self.a.engine_arg)
        merged = {**req, **cli}
        return [f"{k}={json.dumps(v)}" for k, v in merged.items()]

    def cmd(self, stage: str) -> list[str] | None:
        a, run = self.a, self.run
        common_gpu = ["--workload", a.workload, "--case", a.case, "--max-num-seqs", str(a.max_num_seqs), "--checkpoints", a.checkpoints]
        if a.max_tokens is not None:
            common_gpu += ["--max-tokens", str(a.max_tokens)]
        common_gpu += [f"--engine-arg={x}" for x in self.engine_args()]   # [GEN batch SHARED-CODE] scheduler/memory knobs -> build_engine
        if stage == "capture":
            return [self.py_gpu, "-m", "verity_vllm.pipeline.cli", "m1-capture", *common_gpu, "--out", str(self.capture_dir),
                    "--snapshot-steps", ",".join(map(str, self.steps))] + (["--verify-weights"] if self.verify_weights else []) + \
                   (["--snapshot-max-bytes", str(a.snapshot_max_bytes)] if a.snapshot_max_bytes is not None else []) + \
                   (["--snapshot-max-bytes-op", str(a.snapshot_max_bytes_op)] if a.snapshot_max_bytes_op is not None else [])   # D87
        if stage == "control":
            return [self.py_gpu, "-m", "verity_vllm.pipeline.cli", "m1-capture", *common_gpu, "--out", str(self.control_dir), "--no-observer"]
        if stage == "census":
            return [self.py_gpu, "-m", "verity_vllm.pipeline.cli", "census", *common_gpu, "--out", str(self.census_dir)] + ([] if a.keep_traces else ["--no-traces"])
        if stage == "nonint":
            return [self.py_gpu, "-m", "verity_vllm.pipeline.cli", "noninterference", *common_gpu, "--out", str(self.nonint_dir), "--python", self.py_gpu,
                    "--accepted-ctl", str(self.control_tokens_path())]
        if stage == "resolve":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "resolve-log", str(self.log), "--profile", a.profile, "--out", str(run / "resolution"), "--quiet"]
        if stage == "census_reconcile":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "census", "--reconcile", str(self.census_dir / "census.json"), "--log", str(self.log),
                    "--profile", a.profile, "--report", str(run / "census_reconcile")]
        if stage == "nonint_compare":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "noninterference", "--compare", "--out", str(self.nonint_dir), "--workload", a.workload,
                    "--case", a.case, "--accepted-ctl", str(self.control_tokens_path())]
        if stage == "compare":
            c = [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "fold-compare", "--log", str(self.log), "--snapshots", str(self.snapshots), "--profile", a.profile,
                 "--out", str(self.p("compare"))]
            g = self.safetensors_glob()
            if g:
                c += ["--safetensors", g]
            return c
        # [R17-42 route A (4)] the fold folds the MoE scope under the workload's DECLARED construction (manifest target.moe_construction) and the top-p
        # select under the one the Build's Programs declare (--program-dir: a single-request select has no shared-greedy form); never off the trace
        mc, sc, rv = _workload_moe_construction(a.workload), program_sampler_construction(a.program_dir or ()), program_registered_values(a.program_dir or ())
        MC = (["--moe-construction", mc] if mc else []) + (["--sampler-construction", sc] if sc else []) + (["--registered-values"] if rv else [])
        if stage in ("fold", "fold_2"):   # D85: --persist-dir writes program.json / instances.jsonl / accesses.jsonl.gz
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "match", "--worker", "fold", "--log", str(self.log), "--profile", a.profile,
                    "--out", str(self.p("fold_summary" if stage == "fold" else "fold_summary_2"))] + MC + (["--persist-dir", str(run)] if stage == "fold" else [])
        if stage == "fold_other":
            if not a.other_log:
                return None
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "match", "--worker", "fold", "--log", a.other_log, "--profile", a.profile, "--out", str(self.p("fold_summary_other"))] + MC
        if stage == "root_policy":
            c = [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "root-policy", "--log", str(self.log), "--snapshots", str(self.snapshots), "--profile", a.profile,
                 "--out", str(self.p("root_policy"))]
            g = self.safetensors_glob()
            if g:
                c += ["--safetensors", g]
            cs = self.snapshots / "cos_sin_cache.npy"
            return c + (["--cos-sin", str(cs)] if cs.exists() else [])
        if stage == "golden":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "golden", "--check", "--out", str(self.p("golden"))]
        if stage == "wiring":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "operand-provenance", "--log", str(self.log), "--profile", a.profile, "--out", str(self.p("wiring"))]
        if stage == "replay_a":
            return self._replay(["--tier", "a", "--k", str(a.k), "--seed", str(a.seed), "--jobs", str(a.replay_a_jobs)], self.p("replay_a"))
        if stage == "replay_chain":
            return self._replay(["--tier", "chain", "--chains", str(a.chains), "--seed", str(a.seed)], self.p("replay_chain"))
        if stage.startswith("replay_b_step"):
            n = int(stage[len("replay_b_step"):])
            return self._replay(["--tier", "b", "--step", str(n)], run / G.replay_b_rel(n))
        if stage == "replay_c":
            return self._replay(["--tier", "c", *(["--seed", str(a.seed)] if a.seed is not None else [])], self.p("replay_c")) if a.tier_c else None
        return None

    def _replay(self, tier: list[str], out: Path) -> list[str]:
        c = [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "replay", "--log", str(self.log), "--snapshots", str(self.snapshots), "--profile", self.a.profile, *tier]
        budget = self.a.replay_budget_leaves if self.a.replay_budget_leaves is not None else (LAPTOP_MAX_ARG_LEAVES if self.laptop else None)
        if budget is not None:
            c += ["--max-arg-leaves", str(budget)]
        return c + ["--out", str(out)]

    def _hf_snapshot_dir(self) -> str:
        """The HF cache snapshot directory (or glob) of the profile's model when no --checkpoint-dir is given."""
        repo = (profile_config(self.a.profile) or {}).get("model")
        home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
        base = home / "hub" / f"models--{str(repo or 'unknown').replace('/', '--')}" / "snapshots"
        try:
            snaps = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []
        except OSError:
            snaps = []
        return str(snaps[-1]) if len(snaps) == 1 else str(base / "*")

    def stage_list(self, phase: str) -> list[str]:
        gpu = list(GPU_STAGES) if phase in ("gpu", "all") else []
        cpu: list[str] = []
        if phase in ("cpu", "all"):
            cpu = ["resolve", "census_reconcile", "nonint_compare", "compare", "fold", "fold_2"]
            if self.a.other_log:
                cpu.append("fold_other")
            cpu += ["root_policy", "golden", "wiring", "replay_a", "replay_chain"] + [f"replay_b_step{n}" for n in self.tier_b_steps]
            if self.a.tier_c:
                cpu.append("replay_c")
            cpu.append("gates")
        return gpu + cpu


def profile_config(name: str) -> dict | None:
    try:
        return G.load_profile_config(name)
    except Exception:  # noqa: BLE001 - the profile may not exist in this worktree (dry runs)
        return None


def module_available(py: str, module: str) -> bool | None:
    """Whether `module` imports under `py`; None when `py` is another interpreter (not checked)."""
    if Path(py).resolve() != Path(sys.executable).resolve():
        return None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------------------


def _rss_gb(pid: int) -> float | None:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) / (1024 ** 2) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def run_stage(stage: str, cmd: list[str], log_path: Path, *, max_rss_gb: float | None, timeout_s: float | None, cwd: Path | None = None) -> dict:
    """Run one command, teeing stdout+stderr to `log_path`; watch RSS / wall when limits are given."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec: dict[str, Any] = {"stage": stage, "cmd": shlex.join(cmd), "log": str(log_path), "status": "ok", "returncode": None, "reason": None}
    peak = 0.0
    # [R16 sys, BRIEF §5] the Match stage's sub-steps on the row timeline (VERITY_TIMELINE, else no-op): capture / control = one engine
    # process each (`work.*`), the CPU stages (fold, resolve, ...) are validation
    try:
        from verity_vllm.pipeline.timeline import Timeline
        _tl = Timeline.from_env("run_config", hot=False)
    except Exception:  # noqa: BLE001
        _tl = None
    _tl_stage = f"match.{stage}" if stage not in ("capture", "control") else f"work.match_{stage}"
    if _tl is not None:
        _tl.start(_tl_stage)
    with open(log_path, "w") as lf:
        lf.write(f"$ {shlex.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(cwd) if cwd else None, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if max_rss_gb is not None:
                rss = _rss_gb(proc.pid)
                if rss is not None:
                    peak = max(peak, rss)
                    if rss > max_rss_gb:
                        proc.kill()
                        rec.update(status="killed_memory", reason=f"RSS {rss:.2f} GB > {max_rss_gb} GB")
                        proc.wait()
                        break
            if timeout_s is not None and time.time() - t0 > timeout_s:
                proc.kill()
                rec.update(status="timeout", reason=f"wall > {timeout_s:.0f} s")
                proc.wait()
                break
            time.sleep(0.25)
        rc = proc.returncode
    rec["returncode"] = rc
    rec["seconds"] = round(time.time() - t0, 3)
    if max_rss_gb is not None:
        rec["peak_rss_gb"] = round(peak, 3)
    if rec["status"] == "ok" and rc != 0:
        rec.update(status="failed", reason=f"exit {rc}; see {log_path}")
    if _tl is not None:
        _tl.end(_tl_stage, rc=rc, status=rec["status"])
    return rec


def _sha256_file(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _hf_config_of(a: Options, C: dict | None) -> dict | None:
    """`analytic.config_of(<hf config.json>)` when the integrator's module and an HF config are available."""
    try:
        an = importlib.import_module("verity_vllm.program.model")
    except ImportError:
        return None
    cfg = None
    if a.hf_config:
        cfg = Path(a.hf_config)
    elif a.checkpoint_dir and Path(a.checkpoint_dir).is_dir() and (Path(a.checkpoint_dir) / "config.json").exists():
        cfg = Path(a.checkpoint_dir) / "config.json"
    else:
        repo = (C or {}).get("model")
        if repo:
            home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
            base = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
            cands = sorted(base.glob("*/config.json")) if base.exists() else []
            cfg = cands[-1] if cands else None
    if cfg is None or not cfg.exists():
        return None
    try:
        hf_C, _prov = an.config_of(str(cfg))      # (C, provenance) — `dict()` of the pair raised, so every card field read "hand" (llama lane, D84)
        return dict(hf_C)
    except Exception:  # noqa: BLE001 - provenance falls back to "hand"
        return None


def _engine_arg_dict(items: list[str]) -> dict:
    """`key=value` (JSON value) -> dict, torch-free mirror of vllm_adapter.parse_engine_args (this module runs on the laptop)."""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def write_manifest(plan: Plan, a: Options) -> dict:
    from verity_vllm.pipeline.report import workload_summary
    C = profile_config(a.profile)
    man = {
        "schema": "verity-gen/run/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config_id": a.config_id,
        "profile": a.profile, "case": a.case, "workload_path": a.workload, "workload": workload_summary(a.workload),
        "C": C, "hf_config": _hf_config_of(a, C), "laptop": plan.laptop, "snapshot_steps": plan.steps, "tier_b_steps": plan.tier_b_steps,
        "verify_weights": plan.verify_weights, "parallel_replay": plan.parallel, "phase": a.phase,
        "sampling": {"seed": a.seed, "k": a.k, "chains": a.chains, "full_replay": bool(getattr(a, "full_replay", False))},
        "python_gpu": plan.py_gpu, "python_cpu": plan.py_cpu,
        "engine_args_requested": {"max_num_seqs": a.max_num_seqs, "enforce_eager": _workload_enforce_eager(a.workload), **_engine_arg_dict(plan.engine_args())},   # [R15 EXEC-02] from the row's execution
        "declared_execution": _workload_execution(a.workload),
        "engine_args_required_by_workload": _workload_engine_args_required(a.workload),     # D87 (merged into engine_args_requested)
        "snapshot_max_bytes": a.snapshot_max_bytes if a.snapshot_max_bytes is not None else 32 << 20,
        "snapshot_cap_derivation": getattr(a, "snapshot_cap_derivation", None),   # [R16 global F-global-SNAP] None = an explicit CLI cap
        "snapshot_max_bytes_op": a.snapshot_max_bytes_op,       # cov-moe: lower cap at dispatcher-call events (None = the cap above)
        "checkpoints": a.checkpoints, "checkpoint_dir": a.checkpoint_dir, "artifacts": plan.artifacts_override(),
        "skip": sorted(plan.skip), "argv": sys.argv[1:], "family": a.family, "mechanism_notes": [], "notes": [],
    }
    plan.run.mkdir(parents=True, exist_ok=True)
    p = plan.p("run")
    old = json.loads(p.read_text()) if p.exists() else {}
    for k in ("mechanism_notes", "notes", "realized"):
        if old.get(k):
            man[k] = old[k]
    for k in ("mechanism_notes", "notes"):
        man.setdefault(k, [])
    p.write_text(json.dumps(man, indent=1, default=str) + "\n")
    return man


def write_spec_for(plan: Plan, a: Options, man: dict) -> dict | None:
    """[D88] spec.json -- the specification (configuration / workload / initial state) as known before execution, from the workload
    file, the driver's arguments, the checkpoint manifest entry and the HF config.  Written once; a later phase never rewrites it (a
    mismatch is returned and lands in experiment.json).  For `--phase cpu` over an old capture with no spec.json the file is written
    from the same pre-execution sources and flagged `reconstructed_after_execution`."""
    try:
        ck = None
        try:
            doc = json.loads(Path(a.checkpoints).read_text())
            for c in doc.get("checkpoints", []):
                if c.get("role") == a.case or (a.case == "B1" and str(c.get("role", "")).startswith("B1")):
                    ck = {k: c.get(k) for k in ("repo", "revision", "role", "files", "sha256", "digest", "local_path") if c.get(k) is not None}
                    break
        except Exception:  # noqa: BLE001 - no manifest on this host: the checkpoint_dir still identifies the weights
            pass
        wl_doc = _workload_doc(a.workload)
        spec = EXP.build_spec(config_id=a.config_id, case=a.case, profile=a.profile, workload_path=a.workload, workload_doc=wl_doc,
                              checkpoint=ck, checkpoint_dir=a.checkpoint_dir, hf_config=man.get("hf_config"),
                              engine_args=_engine_arg_dict(plan.engine_args()), max_num_seqs=a.max_num_seqs,
                              sampling={}, tp=int(getattr(a, "tp", 1) or 1), dtype=None, snapshot_steps=plan.steps,
                              snapshot_max_bytes=a.snapshot_max_bytes,
                              harness={"enforce_eager": _workload_enforce_eager(a.workload), "execution": _workload_execution(a.workload),   # [R15 EXEC-02]
                                       "driver": "verity_vllm.pipeline.match", "python_gpu": a.python_gpu,
                                       "family": getattr(a, "family", None)})
        if a.phase == "cpu" and not (plan.run / "spec.json").exists():
            spec["reconstructed_after_execution"] = True
            spec["digest"] = EXP._digest_doc({k: v for k, v in spec.items() if k != "digest"})
        note = EXP.write_spec(plan.run, spec)
        print(f"[spec] {'written' if note.get('written') else 'kept'} digest={(note.get('digest') or note.get('existing_digest') or '')[:12]}"
              f"{'' if note.get('same', True) else '  (this phase would have produced a DIFFERENT spec; not rewritten)'}", file=sys.stderr)
        return note
    except Exception as e:  # noqa: BLE001 - informational
        print(f"[spec] not written: {type(e).__name__}: {e}", file=sys.stderr)
        return {"written": False, "error": f"{type(e).__name__}: {e}"}


def write_realized(plan: Plan) -> dict | None:
    """[D87] After the GPU phase: the realized shape of the run into run.json["realized"] -- per request the prompt length, the
    generated length, the arrival / first / finish engine steps and finish reason (from capture/tokens.json), the engine-step
    count of the observed and the control run, and the snapshot steps that actually happened.  "Record the exact realized
    input/output lengths in every run artifact": run.json is the artifact every gate and the card read first."""
    cap_p = plan.snapshots / "tokens.json" if plan.a.log else plan.capture_dir / "tokens.json"
    ctl_p = plan.control_tokens_path()
    try:
        cap = json.loads(cap_p.read_text()) if cap_p.exists() else None
    except Exception:  # noqa: BLE001
        cap = None
    try:
        ctl = json.loads(ctl_p.read_text()) if ctl_p.exists() else None
    except Exception:  # noqa: BLE001
        ctl = None
    if cap is None and ctl is None:
        return None
    src = cap or ctl
    reqs = []
    for r in src.get("requests", []):
        reqs.append({"request_id": r.get("request_id"), "prompt_len": r.get("prompt_len", len(r.get("prompt_token_ids") or [])),
                     "max_tokens": r.get("max_tokens"), "n_generated": r.get("n_generated", len(r.get("generated_token_ids") or [])),
                     "arrive_step": r.get("arrive_step", 0), "arrive_step_realized": r.get("arrive_step_realized"),
                     "first_step": r.get("first_step"), "finish_step": r.get("finish_step"), "finish_reason": r.get("finish_reason")})
    snaps = None
    sp = plan.snapshots / "snapshots.json"
    if sp.exists():
        try:
            sj = json.loads(sp.read_text())
            snaps = sorted({int(s) for s in (sj.get("steps") or sj.get("snapshot_steps") or [])}) or None
        except Exception:  # noqa: BLE001
            snaps = None
    real = {"requests": reqs, "prompt_lens": [r["prompt_len"] for r in reqs], "n_generated": [r["n_generated"] for r in reqs],
            "engine_steps_capture": (cap or {}).get("engine_steps") or ((cap or {}).get("timing") or {}).get("engine_steps"),
            "engine_steps_control": (ctl or {}).get("engine_steps") or ((ctl or {}).get("timing") or {}).get("engine_steps"),
            "expected_engine_steps_declared": src.get("expected_engine_steps"), "serving_profile": src.get("serving_profile"),
            "context_class": src.get("context_class"), "snapshot_steps_realized": snaps,
            "max_concurrent_requests": _max_concurrent(reqs)}
    p = plan.p("run")
    man = json.loads(p.read_text()) if p.exists() else {}
    man["realized"] = real
    p.write_text(json.dumps(man, indent=1, default=str) + "\n")
    return real


def _max_concurrent(reqs: list[dict]) -> int | None:
    """Peak number of requests active in one engine step, from the realized first/finish steps."""
    spans = [(r.get("first_step"), r.get("finish_step")) for r in reqs]
    if not spans or any(a is None or b is None for a, b in spans):
        return None
    hi = max(b for _, b in spans)
    return max(sum(1 for a, b in spans if a <= s <= b) for s in range(hi + 1))


def load_stages(run: Path) -> list[dict]:
    p = run / G.ARTIFACTS["stages"]
    if not p.exists():
        return []
    text = p.read_text()
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        # [F-r19-int-17] an interrupted save (ENOSPC mid-write left a 0-byte stages.json on l40s4 #69) must not block `--phase cpu`
        # over the preserved capture: set the unreadable record aside under a stamped name and start the stage list empty.
        aside = p.with_name(p.name + f".unreadable-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        p.replace(aside)
        print(f"[stages] {p.name} unreadable ({e.__class__.__name__}: {e}; {len(text)} bytes) -> set aside as {aside.name}; resuming with no stage records",
              file=sys.stderr)
        return []
    return list(d.get("stages", d) if isinstance(d, dict) else d)


def save_stages(run: Path, recs: list[dict]) -> None:
    byname: dict[str, dict] = {}
    for r in recs:
        byname[r["stage"]] = r
    (run / G.ARTIFACTS["stages"]).write_text(json.dumps({"schema": "verity-gen/stages/v1", "stages": list(byname.values())}, indent=1) + "\n")


def gates_stage(plan: Plan, a: Options) -> dict:
    """In-process: gates.json, card.json, CARD.md, <cards-dir>/<config_id>.json."""
    from verity_vllm.pipeline import report as CARD
    t0 = time.time()
    run = plan.run
    results = G.evaluate_gates(run)
    diag = G.diagnostics(run)
    G.write_gates(run, results, diag)
    card = CARD.build_card(run, results, config_id=a.config_id)
    CARD.write_card(card, run / G.ARTIFACTS["card"])
    (run / "CARD.md").write_text(CARD.summarize_md(card))
    dest = Path(a.cards_dir) / f"{a.config_id}.json"
    CARD.write_card(card, dest)
    print(CARD.top_section(card))
    print(G.gates_markdown(results))
    return {"stage": "gates", "status": "ok", "returncode": 0, "seconds": round(time.time() - t0, 3), "log": None, "cmd": "in-process",
            "reason": None, "verdict": card.verdict, "card": str(dest)}


@dataclass(kw_only=True)
class Options:
    worker: str | None = option("--worker", choices=["fold"], default=None, help="(internal) run one worker in this process")
    config_id: str | None = option("--config-id", default=None)
    profile: str | None = option("--profile", default=None)
    workload: str | None = option("--workload", default=None)
    case: str | None = option("--case", default=None, help="checkpoint role (B0, B1, QWEN05, ... any role in checkpoints.json with a profile module)")
    out: str | None = option("--out", default=None, help="RUN_DIR (driver) or fold_summary.json (--worker fold)")
    run_dir: str | None = option("--run-dir", default=None, help="an existing RUN_DIR: --out plus case/workload/profile/config-id/checkpoint-dir/steps from its run.json " "(the sealed verifier re-runs the CPU phase this way from a fresh checkout)")
    phase: str = option("--phase", choices=["gpu", "cpu", "all"], default="all")
    python_gpu: str | None = option("--python-gpu", default=None, help="the pod's torch venv python (default: sys.executable)")
    python_cpu: str | None = option("--python-cpu", default=None, help="the torch-free 3.12 python (default: sys.executable)")
    snapshot_steps: str = option("--snapshot-steps", default=DEFAULT_SNAPSHOT_STEPS, help="contiguous from 0, or `all` = every step of the workload (default all)")
    full_replay: bool = option("--full-replay", default=False, action="store_true", help="also run the tier-b full-step replays (diagnostic only since D82; never gating)")
    tier_b_steps: str = option("--tier-b-steps", default=DEFAULT_TIER_B_STEPS, help="with --full-replay: steps replayed by tier b (default 0,1,2,3; must be snapshot steps)")
    chains: int = option("--chains", type=int, default=DEFAULT_CHAINS, help="tier chain: connected slices replayed with computed intermediates (G5)")
    no_verify_weights: bool = option("--no-verify-weights", default=False, action="store_true", help="skip the checkpoint sha verification at engine build (root policy weight identity then fails)")
    parallel: int | None = option("--parallel", type=int, default=None, help="concurrent replay stages (default 4; 1 with --laptop)")
    replay_budget_leaves: int | None = option("--replay-budget-leaves", type=int, default=None, help="replay --max-arg-leaves (tier a/chain evaluate oversize instances at a child unit or skip them). Default: none on the pod, 4e6 with --laptop")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    engine_arg: list[str] = option("--engine-arg", action="append", default=[], help="LLM(**kw) override key=value (JSON value) for every GPU stage, repeatable; e.g. max_num_batched_tokens=64, enable_prefix_caching=true")
    max_tokens: int | None = option("--max-tokens", type=int, default=None, help="override the workload's max_tokens (passed to the GPU stages)")
    snapshot_max_bytes: int | None = option("--snapshot-max-bytes", type=int, default=None, help="(capture) per-tensor snapshot cap in bytes (m1_capture default 32 MiB; D87: a " "prefill MLP row block above the cap has no snapshot evidence -- raise it for long / batched prefills, the run.json records the value)")
    snapshot_max_bytes_op: int | None = option("--snapshot-max-bytes-op", type=int, default=None, help="(capture) lower per-tensor cap at dispatcher-call events; Triton " "launch events keep --snapshot-max-bytes, so a large launch output is copied once (cov-moe: the fused-MoE intermediates of a " "2048-token prefill otherwise exceed the pod's pinned host memory; the run.json records the value)")
    checkpoints: str = option("--checkpoints", default=DEFAULT_CHECKPOINTS)
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None, help="HF snapshot dir or model*.safetensors glob (root_policy, compare --safetensors)")
    hf_config: str | None = option("--hf-config", default=None, help="HF config.json for per-field C provenance (analytic.config_of)")
    family: str | None = option("--family", default=None, help="hand-authored family reference name (default Serve_v4 in the card)")
    log: str | None = option("--log", default=None, help="reuse an existing raw log instead of RUN_DIR/capture/log.jsonl.gz")
    snapshots: str | None = option("--snapshots", default=None, help="its snapshots dir (default: the log's directory)")
    other_log: str | None = option("--other-log", default=None, help="a second capture of the same workload (G6 detail: cross-capture digest)")
    control_tokens: str | None = option("--control-tokens", default=None, help="reuse an existing control run's tokens.json instead of RUN_DIR/control/tokens.json")
    laptop: bool = option("--laptop", default=False, action="store_true", help="laptop budget: RSS/wall watchdog, replay --max-arg-leaves")
    max_rss_gb: float | None = option("--max-rss-gb", type=float, default=None, help="kill a stage above this RSS (default 1.5 with --laptop)")
    stage_timeout: float | None = option("--stage-timeout", type=float, default=None, help="kill a stage above this wall time in s (default 180 with --laptop)")
    dry_run: bool = option("--dry-run", default=False, action="store_true", help="print the exact command lines, run nothing")
    tier_c: bool = option("--tier-c", default=False, action="store_true", help="also run the tier-c replay")
    skip: str = option("--skip", default="", help="comma list of stages/groups to skip")
    keep_traces: bool = option("--keep-traces", default=False, action="store_true", help="census: keep the per-step Chrome traces")
    cards_dir: str = option("--cards-dir", default="out/gen/cards")
    k: int = option("--k", type=int, default=DEFAULT_K, help="tier a instances per (specialization, regime) stratum (gate minimum TIER_A_K)")
    replay_a_jobs: int = option("--replay-a-jobs", type=int, default=1, help="replay_a's forked workers (replay --jobs): the same seed gives the same sample and report")
    seed: int | None = option("--seed", type=int, default=None, help="the replay sample's seed (replay_a, replay_chain: required when they run)")
    no_program: bool = option("--no-program", default=False, action="store_true", help="(--worker fold) skip Program(): no digest")
    persist_dir: str | None = option("--persist-dir", default=None, help="(--worker fold) write program.json + instances.jsonl + accesses.jsonl.gz here (D85)")
    program_dir: list[str] | None = option("--program-dir", action="append", default=None, help="the Build's Program dir(s) the Match compares against; the fold binds the top-p select under the construction they declare (sampling_event.program_sampler_construction)")
    sampler_construction: str | None = option("--sampler-construction", default=None, help="(--worker fold) the Programs' sampler construction (the driver passes it from --program-dir)")
    registered_values: bool = option("--registered-values", default=False, action="store_true", help="(--worker fold) the program's values are inputs (Profile.registered_values); the driver passes what the --program-dir Programs register")
    moe_construction: str | None = option("--moe-construction", default=None, help="(--worker fold) [R17-42 route A (4)] the workload's DECLARED MoE construction " "(manifest target.moe_construction: indexed-read | indexed-read-ordered | padded); the driver passes it from --workload")


def main(a: Options) -> int:
    if a.worker == "fold":
        return fold_worker(a)
    if a.run_dir:
        _defaults_from_run_dir(a)
    if a.profile is None and a.case:
        try:
            from verity_vllm.engine.profiles import profile_module_for_case
            a.profile = profile_module_for_case(a.case)
        except Exception as e:  # noqa: BLE001
            raise UsageError(f"--profile not given and no profile module declares case {a.case!r}: {e}")
    if a.config_id is None and a.case and a.workload:
        a.config_id = f"{a.case}-{Path(a.workload).stem}"
    for req in ("config_id", "profile", "workload", "case", "out"):
        if getattr(a, req) is None:
            raise UsageError(f"--{req.replace('_', '-')} is required")
    # [R16 global, F-global-SNAP] no explicit --snapshot-max-bytes: derive the cap from the workload's largest forward and the checkpoint's
    # widths (snapshot_cap_bytes) instead of m1_capture's fixed 32 MiB, which left the prefill MLP rows of every 1024-class row without
    # snapshot evidence (D87).  An explicit value still wins; the run.json records the derivation either way.
    if not hasattr(a, "snapshot_cap_derivation"):
        a.snapshot_cap_derivation = None          # an explicit CLI cap (or --run-dir already carried the capture's)
    if a.snapshot_max_bytes is None:
        C0 = profile_config(a.profile)
        cap, why = snapshot_cap_bytes(_workload_doc(a.workload), _hf_config_of(a, C0), _engine_arg_dict(a.engine_arg))
        a.snapshot_max_bytes, a.snapshot_cap_derivation = cap, why
        if why["raised"]:
            print(f"[plan] --snapshot-max-bytes derived: {cap} B ({why['rows']} rows [{why['rows_source']}] x {why['row_bytes']} B "
                  f"[{why['row_bytes_source']}]; the 32 MiB cap of record skipped the prefill MLP rows of 1024-class rows, D87)", file=sys.stderr)

    plan = Plan(a)
    stages = plan.stage_list(a.phase)
    if a.seed is None and (sampled := [st for st in stages if st in ("replay_a", "replay_chain") and not plan.skipped(st)]):
        raise UsageError(f"--seed is required to run {' and '.join(sampled)} (the replay sample is drawn from it)")
    max_rss = a.max_rss_gb if a.max_rss_gb is not None else (LAPTOP_MAX_RSS_GB if plan.laptop else None)
    timeout = a.stage_timeout if a.stage_timeout is not None else (LAPTOP_STAGE_TIMEOUT_S if plan.laptop else None)

    if a.dry_run:
        print(f"# run_config --dry-run  RUN_DIR={plan.run}  phase={a.phase}  snapshot_steps={plan.steps}  laptop={plan.laptop}")
        for st in stages:
            if plan.skipped(st):
                print(f"[{st}] SKIP (--skip)")
                continue
            if st == "gates":
                print(f"[gates] in-process: gates.evaluate_gates({plan.run}) -> {plan.p('gates')}; card -> {plan.p('card')}, {plan.run / 'CARD.md'}, "
                      f"{Path(a.cards_dir) / (a.config_id + '.json')}")
                continue
            cmd = plan.cmd(st)
            if cmd is None:
                continue
            print(f"[{st}] {shlex.join(cmd)}")
        if a.phase in ("gpu", "all"):
            print(f"[gpu_done] write {plan.p('gpu_done')} with per-stage timings")
        return 0

    man = write_manifest(plan, a)
    spec_note = write_spec_for(plan, a, man)          # D88: spec.json (known before execution) -- written once, before any GPU stage
    recs = load_stages(plan.run)
    logs = plan.run / "logs"
    t_phase = time.time()
    stages = _batch_replays(stages, plan.parallel)
    if a.phase == "cpu":
        try:
            write_realized(plan)             # D87: an existing capture's realized shape is in run.json before the gates / card read it
        except Exception as e:  # noqa: BLE001
            print(f"[realized] not written: {type(e).__name__}: {e}", file=sys.stderr)
    for st in stages:
        if isinstance(st, list):
            # a batch of independent replay stages run concurrently (pod: many cores; each process ~1 GB)
            import concurrent.futures as cf
            todo = [x for x in st if not plan.skipped(x) and plan.cmd(x) is not None]
            for x in st:
                if plan.skipped(x):
                    recs.append({"stage": x, "status": "skipped", "returncode": None, "seconds": 0.0, "log": None, "cmd": None, "reason": "--skip"})
            print(f"[replay] running {len(todo)} replay stages with parallel={plan.parallel}: {todo}", file=sys.stderr)
            with cf.ThreadPoolExecutor(max_workers=plan.parallel) as ex:
                futs = {ex.submit(run_stage, x, plan.cmd(x), logs / f"{x}.log", max_rss_gb=max_rss, timeout_s=timeout): x for x in todo}
                for fut in cf.as_completed(futs):
                    rec = fut.result()
                    print(f"[{rec['stage']}] {rec['status']} in {rec['seconds']} s", file=sys.stderr)
                    recs.append(rec)
                    save_stages(plan.run, recs)
            save_stages(plan.run, recs)
            continue
        if plan.skipped(st):
            recs.append({"stage": st, "status": "skipped", "returncode": None, "seconds": 0.0, "log": None, "cmd": None, "reason": "--skip"})
            save_stages(plan.run, recs)
            continue
        if st == "gates":
            try:
                write_realized(plan)         # D87: realized shape into run.json before the card is built
            except Exception as e:  # noqa: BLE001
                print(f"[realized] not written: {type(e).__name__}: {e}", file=sys.stderr)
            try:
                recs.append(gates_stage(plan, a))
            except Exception as e:  # noqa: BLE001 - the driver reports, never crashes on the last stage
                recs.append({"stage": "gates", "status": "failed", "returncode": None, "seconds": 0.0, "log": None, "cmd": "in-process",
                             "reason": f"{type(e).__name__}: {e}"})
            save_stages(plan.run, recs)
            continue
        cmd = plan.cmd(st)
        if cmd is None:
            continue
        base = "replay_b" if st.startswith("replay_b_step") else st
        mod = OPTIONAL_MODULES.get(base)
        py = cmd[0]
        if mod and module_available(py, mod) is False:
            recs.append({"stage": st, "status": "missing_module", "returncode": None, "seconds": 0.0, "log": None, "cmd": shlex.join(cmd),
                         "reason": f"{mod} not importable under {py}"})
            print(f"[{st}] missing module {mod}", file=sys.stderr)
            save_stages(plan.run, recs)
            continue
        print(f"[{st}] {shlex.join(cmd)}", file=sys.stderr)
        rec = run_stage(st, cmd, logs / f"{st}.log", max_rss_gb=max_rss, timeout_s=timeout)
        print(f"[{st}] {rec['status']} in {rec['seconds']} s", file=sys.stderr)
        recs.append(rec)
        save_stages(plan.run, recs)
    if a.phase in ("gpu", "all"):
        gd = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t_phase, 3), "config_id": a.config_id,
              "stages": {r["stage"]: {"status": r["status"], "seconds": r.get("seconds")} for r in recs if r["stage"] in GPU_STAGES}}
        plan.p("gpu_done").write_text(json.dumps(gd, indent=1) + "\n")
    try:
        real = write_realized(plan)          # D87: realized lengths / steps into run.json (also on a --phase cpu re-run over an old capture)
        if real:
            print(f"[realized] prompt_lens={real['prompt_lens']} n_generated={real['n_generated']} steps capture/control="
                  f"{real['engine_steps_capture']}/{real['engine_steps_control']} max_concurrent={real['max_concurrent_requests']}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - informational
        print(f"[realized] not written: {type(e).__name__}: {e}", file=sys.stderr)
    try:
        EXP.write_experiment(plan.run, repo=Path.cwd(), spec_note=spec_note)   # D88: the Experiment index over the run directory
    except Exception as e:  # noqa: BLE001 - informational
        print(f"[experiment] not written: {type(e).__name__}: {e}", file=sys.stderr)
    flat = [x for st in stages for x in (st if isinstance(st, list) else [st])]
    bad = [r for r in recs if r["stage"] in flat and r["status"] not in ("ok", "skipped")]
    return 1 if bad else 0


def _defaults_from_run_dir(a: Options) -> None:
    """Fill unset arguments from RUN_DIR/run.json (the manifest the GPU phase wrote)."""
    run = Path(a.run_dir)
    man_p = run / G.ARTIFACTS.get("run", "run.json")
    if not man_p.exists():
        raise UsageError(f"--run-dir {run}: no {man_p.name} (the GPU phase writes it)")
    man = json.loads(man_p.read_text())
    a.out = a.out or str(run)
    for arg, key in (("case", "case"), ("workload", "workload_path"), ("profile", "profile"), ("config_id", "config_id"),
                     ("checkpoint_dir", "checkpoint_dir"), ("checkpoints", "checkpoints"), ("family", "family")):
        if getattr(a, arg, None) in (None, DEFAULT_CHECKPOINTS) and man.get(key):
            setattr(a, arg, man[key])
    if a.snapshot_steps == DEFAULT_SNAPSHOT_STEPS and man.get("snapshot_steps"):
        a.snapshot_steps = ",".join(map(str, man["snapshot_steps"]))
    if a.tier_b_steps == DEFAULT_TIER_B_STEPS and man.get("tier_b_steps"):
        a.tier_b_steps = ",".join(map(str, man["tier_b_steps"]))
    if a.snapshot_max_bytes is None and man.get("snapshot_max_bytes"):
        # [R16 global F-global-SNAP] a CPU-phase re-run keeps the cap the capture RAN with (and its derivation), never a re-derived one
        a.snapshot_max_bytes = int(man["snapshot_max_bytes"])
        a.snapshot_cap_derivation = man.get("snapshot_cap_derivation")
    if a.max_num_seqs == 8 and man.get("engine_args_requested", {}).get("max_num_seqs"):
        a.max_num_seqs = int(man["engine_args_requested"]["max_num_seqs"])
    if not a.engine_arg:
        a.engine_arg = [f"{k}={json.dumps(v)}" for k, v in man.get("engine_args_requested", {}).items() if k not in ("max_num_seqs", "enforce_eager")]
    if a.workload and not Path(a.workload).exists():
        # the fresh checkout may hold the workload under the same relative path
        rel = Path(a.workload)
        if not rel.is_absolute() and not rel.exists():
            raise UsageError(f"workload {a.workload} not found in this checkout")


def _batch_replays(stages: list[str], parallel: int) -> list:
    """Group consecutive replay stages into one list when parallel > 1 (they are independent processes)."""
    if parallel <= 1:
        return list(stages)
    out: list = []
    batch: list[str] = []
    for st in stages:
        if st.startswith("replay_"):
            batch.append(st)
        else:
            if batch:
                out.append(batch if len(batch) > 1 else batch[0])
                batch = []
            out.append(st)
    if batch:
        out.append(batch if len(batch) > 1 else batch[0])
    return out


def _workload_doc(path: str | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def _workload_max_tokens(path: str | None) -> int | None:
    doc = _workload_doc(path)
    if not doc:
        return None
    mt = [int(r["max_tokens"]) for r in doc.get("requests", []) if r.get("max_tokens") is not None]
    return max(mt) if mt else None


def _workload_expected_steps(path: str | None) -> int | None:
    """[D87] The workload's declared `expected_engine_steps` (an upper bound: every request runs to max_tokens under the declared
    arrivals), else max(max_tokens)."""
    doc = _workload_doc(path)
    if doc and doc.get("expected_engine_steps"):
        return int(doc["expected_engine_steps"])
    return _workload_max_tokens(path)


def _workload_engine_args_required(path: str | None) -> dict:
    doc = _workload_doc(path)
    return dict((doc or {}).get("engine_args_required") or {})


def snapshot_cap_bytes(doc: dict | None, hf: dict | None, engine_args: dict | None = None) -> tuple[int, dict]:
    """[R16 global, F-global-SNAP (D87 made automatic)] The per-tensor snapshot cap a Match capture needs so that EVERY per-token
    activation of the largest forward the workload can schedule has snapshot evidence: rows x widest row x 2 bytes, never below the
    32 MiB cap of record.  Rows = the engine's `max_num_batched_tokens` (the workload's `engine_args_required`, then the CLI's engine
    args), else the longest prompt plus one decode row per request; the widest row = max(gate_up 2I, qkv (NH+2KVH)D, hidden H) from
    the checkpoint's config (analytic.config_of keys GU/QKV/H), else 64 KiB.  Why: at 1024/128 the prefill's gate_up output is
    [1024+B, 2I] bf16 = 33.7 MB for Llama-3.2-1B (I 8192) and 36.7 MB for Qwen2.5-1.5B (I 8960) -- over the 32 MiB cap, listed in
    snapshots.json as skipped -- so the Commit's form (B) oracle-compare had 'no snapshot of alloc ... (0 entries of that alloc)' for the
    prefill-step MLP rows of every 1024-class row (exec M-0545 llama B4, M-0655 qwen H100 B1 -- FAIL by name where Gemm_v2 has no
    evaluator; global int-2 B16 mixed-arrivals: 16 layers x 5 rows at engine step 36, the [1028, 16384] gate_up of a 1024-token
    arrival); 256/32 rows ([256+B, 2I] = 8.6 MB) never hit it.  MoE routed intermediates are not in the width (cov-moe's
    --snapshot-max-bytes-op contract is unchanged; OLMoE stays at the 32 MiB floor).  Returns (cap, derivation) -- the derivation
    goes into run.json so a reader sees why the record's cap is what it is."""
    doc = doc or {}
    reqs = doc.get("requests") or []
    ea = {**dict(doc.get("engine_args_required") or {}), **dict(engine_args or {})}
    rows_src = None
    mnbt = ea.get("max_num_batched_tokens")
    if isinstance(mnbt, (int, float)) and int(mnbt) > 0:
        rows, rows_src = int(mnbt), "engine max_num_batched_tokens"
    else:
        lens = []
        for r in reqs:
            p = r.get("prompt_token_ids")
            lens.append(len(p) if isinstance(p, list) and p else int(r.get("prompt_len") or 0))
        rows, rows_src = (max(lens) if lens else 0) + len(reqs), "longest prompt + one decode row per request"
    widths = {}
    if hf:
        for k in ("GU", "QKV", "H"):
            v = hf.get(k)
            if isinstance(v, (int, float)) and int(v) > 0:
                widths[k] = int(v)
    if widths:
        wk = max(widths, key=widths.get)
        row_bytes, width_src = widths[wk] * 2, f"hf config {wk}={widths[wk]} x bf16"
    else:
        row_bytes, width_src = SNAPSHOT_ROW_BYTES_FALLBACK, "no HF config: 64 KiB per row"
    need = rows * row_bytes
    cap = max(SNAPSHOT_CAP_FLOOR, need)
    return cap, {"rows": rows, "rows_source": rows_src, "row_bytes": row_bytes, "row_bytes_source": width_src, "needed_bytes": need,
                 "floor_bytes": SNAPSHOT_CAP_FLOOR, "cap_bytes": cap, "raised": cap > SNAPSHOT_CAP_FLOOR}


# [R15 EXEC-02 / COORD M-0063 (2)] torch-free mirror of vllm_adapter.execution_of_workload / EXECUTION_ENFORCE_EAGER (this driver runs on
# the laptop): the row's declared execution label (explicit `execution` / `sweep.execution`, else the row-id suffix) and the enforce_eager
# it implies.  The GPU stages derive the same from the workload themselves and refuse a contradicting engine arg (build_engine).
_EXECUTION_ENFORCE_EAGER = {"bi-eager": True, "bi-eager-fa2": True, "eager": True, "bi-compiled": False, "compiled": False}


def _workload_moe_construction(path: str | None) -> str | None:
    """[R17-42 route A (4)] The workload's DECLARED MoE construction (manifest `target.moe_construction`; None = undeclared = the default),
    validated by the frontend's own table so a misspelling fails here by name and is never folded as MOE-01 silently."""
    doc = _workload_doc(path)
    mc = ((doc or {}).get("target") or {}).get("moe_construction")
    if mc is None:
        return None
    from verity_vllm.program.frontend.target_profile import TargetProfile
    if mc not in TargetProfile.MOE_CONSTRUCTIONS:
        raise SystemExit(f"workload target.moe_construction {mc!r} is not one of {sorted(TargetProfile.MOE_CONSTRUCTIONS)}")
    return str(mc)


def _workload_execution(path: str | None) -> str | None:
    doc = _workload_doc(path) or {}
    sw = doc.get("sweep") if isinstance(doc.get("sweep"), dict) else {}
    for v in (doc.get("execution"), sw.get("execution")):
        if isinstance(v, str) and v:
            return v
    rid = sw.get("row_id") or doc.get("row_id")
    return rid.rsplit("__", 1)[1] if isinstance(rid, str) and "__" in rid else None


def _workload_enforce_eager(path: str | None) -> bool:
    ex = _workload_execution(path)
    if ex is None:
        return True                                   # undeclared execution: the historical eager default (recorded as declared null)
    if ex not in _EXECUTION_ENFORCE_EAGER:
        raise SystemExit(f"workload {path}: execution label {ex!r} is not modelled by the harness (known: {sorted(_EXECUTION_ENFORCE_EAGER)})")
    return _EXECUTION_ENFORCE_EAGER[ex]


# ---------------------------------------------------------------------------------------------------------
# the fold worker (one process = one fold; the driver runs it twice for G6)
# ---------------------------------------------------------------------------------------------------------


def _max_rss_gb() -> float:
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 ** 3) if sys.platform == "darwin" else ru / (1024 ** 2)


def fold_summary_result(log_path: str | Path, profile_name: str, *, with_program: bool = True, moe_construction: str | None = None, sampler_construction: str | None = None, registered_values: bool = False) -> tuple[dict, Any]:
    """Fold one log: the `gates.FOLD_SUMMARY_SCHEMA` document and the live `FoldResult` (for `persist_program`).  `moe_construction` [R17-42 route A (4)]: the workload's
    DECLARED construction, which a profile without a `with_moe_construction` hook cannot fold (FAILS by name); `sampler_construction` and `registered_values`: the Programs' (`--program-dir`)."""
    from verity_vllm.check.fold_compare import load_profile, summarize
    from verity_vllm.observe.events import Root, RunHeader
    from verity_vllm.observe.fold.fold import fold
    from verity_vllm.observe.fold.patterns.pouw import with_protocols
    from verity_vllm.observe.fold.patterns.sampling import with_sampler_construction
    from verity_vllm.observe.log import read_log
    from verity_vllm.program.frontend.target_profile import TargetProfile

    t0 = time.time()
    profile = load_profile(profile_name)
    prof_mod = importlib.import_module(f"verity_vllm.engine.profiles.{profile_name}")
    if moe_construction not in (None, TargetProfile.DEFAULT_MOE_CONSTRUCTION):
        hook = getattr(prof_mod, "with_moe_construction", None)
        if hook is None:
            raise SystemExit(f"workload declares moe_construction={moe_construction!r} but profile {profile_name} has no with_moe_construction hook: it cannot fold that construction (refused by name; the MOE-01 fold is NOT a substitute)")
        profile = hook(profile, moe_construction)
    profile = replace(with_sampler_construction(profile, sampler_construction), registered_values=registered_values)
    events = list(read_log(log_path))
    header = next((e for e in events if isinstance(e, RunHeader)), None)
    result = fold(events, with_protocols(profile, header), strict=None, with_program=with_program)   # PoUW's calls: `patterns.pouw`
    summ = summarize(result)
    scope_pattern = next((type(pt).__name__ for pt in profile.patterns if type(pt).__name__.startswith("MoEFusedExperts")), None)
    # roots, classified as resolve_log does (the profile's `root_class`, else `weight_field`)
    classify = getattr(prof_mod, "root_class", None)
    classes: Counter = Counter()
    unknown: list[str] = []
    seen: set[str] = set()
    for ev in events:
        if isinstance(ev, Root) and ev.name not in seen:
            seen.add(ev.name)
            if ev.role in ("weight", "buffer"):
                cls = classify(ev.name) if classify else ("weight" if profile.weight_field(ev.name) else "unknown")
                classes[cls] += 1
                if cls == "unknown":
                    unknown.append(ev.name)
            else:
                classes[ev.role] += 1
    unbound = sorted({n.split("Root ")[1].split(" (")[0].strip("'") for n in result.notes if n.startswith("unbound root:")})
    # per step
    by_step: dict[int, Counter] = {}
    for inst in result.instances:
        by_step.setdefault(int(inst["step"]), Counter())[inst["spec"]] += 1
    step_meta = {int(s["step"]): s for s in result.steps}
    steps_sorted = sorted(set(by_step) | set(step_meta))
    first = steps_sorted[0] if steps_sorted else 0
    per_step = []
    for s in steps_sorted:
        m = step_meta.get(s, {})
        per_step.append({"step": s, "phase": "prefill" if s == first else "decode", "rows": m.get("rows"),
                         "n_instances": sum(by_step.get(s, Counter()).values()), "computes": m.get("computes"),
                         "spec_histogram": dict(sorted(by_step.get(s, Counter()).items())), "unsupported": m.get("unsupported"),
                         "errors": m.get("errors"), "seconds": m.get("seconds")})
    tokens = {str(r): [int(t) for _, t in v] for r, v in result.sampled.items()}
    sampled_steps = {str(r): [int(s) for s, _ in v] for r, v in result.sampled.items()}
    requests = []
    hdr_reqs = list(getattr(result, "requests", None) or ((header.workload or {}).get("requests") if header else None) or [])
    for r in hdr_reqs:
        rid = getattr(r, "request_id", None) if not isinstance(r, dict) else r.get("request_id")
        ptl = getattr(r, "prompt_token_ids", None) if not isinstance(r, dict) else r.get("prompt_token_ids")
        mt = getattr(r, "max_tokens", None) if not isinstance(r, dict) else r.get("max_tokens")
        requests.append({"request_id": rid, "prompt_len": len(ptl or []), "max_tokens": mt, "n_tokens": len(tokens.get(str(rid), []))})
    opaque = sum(1 for e in result.errors if str(e.get("class", "")).lower() in ("opaque", "prescribed"))
    return {
        "schema": G.FOLD_SUMMARY_SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log": str(log_path),
        "profile": profile_name, "profile_id": profile.id, "with_program": with_program, "digest": result.digest, "spec": summ.get("spec"),
        # [R17-42 route A (4)] the DECLARED construction the scope folded under (None = undeclared = indexed-read) and the pattern that did
        "moe_construction": moe_construction, "moe_scope_pattern": scope_pattern, "sampler_construction": sampler_construction, "registered_values": registered_values,
        "n_instances": len(result.instances), "n_steps": len(per_step), "unsupported": result.report.unsupported,
        "unresolved": result.unresolved, "opaque_fallbacks": opaque, "errors": len(result.errors), "error_classes": summ.get("error_classes"),
        "error_examples": summ.get("error_examples"), "unknown_roots": unknown, "root_classes": dict(classes), "unbound_roots": unbound,
        "per_step": per_step, "tokens": tokens, "sampled_steps": sampled_steps, "requests": requests, "params": summ.get("params"),
        "ret_type": summ.get("ret_type"), "memory": result.memory, "timings": result.timings, "seconds": round(time.time() - t0, 3),
        "max_rss_gb": round(_max_rss_gb(), 3), "fold": summ,
    }, result


PROGRAM_SCHEMA = "verity-gen/program/v1"
INSTANCES_SCHEMA = "verity-gen/instances/v1"
INSTANCES_SCHEMA_V2 = "verity-gen/instances/v2"   # [R17 mfast F2] operand_refs: "p" runs + node-space PROGRESSIONS ["N", idx, lo, hi, dI, dL, dH, count]
INSTANCES_FORM_ENV = "VERITY_INSTANCES_FORM"      # runs (default, v1) | progressions (v2); every reader accepts both (cb_a.compact.runs_from_file_items)


def instances_form() -> str:
    """[R17 mfast F2, COORD R17-34 (2) / M-0496] The STORED form of `instances.jsonl` operand references.  `runs` = the record form
    (every maximal run one item, the node-space K/V gathers listed run by run: 54 M items / 1.63 GB on a B8 1024/128 row); `progressions`
    = the same references with the node space as compact progressions built from the fold's own Refs objects in O(1) per part (no
    `runs()` expansion at write; ~3 % of the text), parameter-space runs unchanged.  Lossless either way.  The LIBRARY default stays `runs`
    (tests and ad-hoc folds unchanged); the rows of record set VERITY_INSTANCES_FORM=progressions in row_pod.sh's Match stage (COORD R17-34 (2)
    "compact refs stay compact end to end", M-0496, rev M-0500 conditions met at S-r17-33: readers accept both forms)."""
    v = (os.environ.get(INSTANCES_FORM_ENV) or "runs").strip().lower()
    if v not in ("runs", "progressions"):
        raise SystemExit(f"{INSTANCES_FORM_ENV}={v!r}: expected runs | progressions")
    return v


def persist_program(result: Any, run_dir: Path, *, profile_name: str, log_path: str | Path) -> dict:
    """D85: freeze the candidate.  `program.json` is the Verity Program through the existing codec (`verity.ml.circuits.codec`,
    digest == `fold_summary.digest`); `instances.jsonl` is the runtime-correspondence record the fold already holds in
    memory -- one line per Compute instance (step, root event id, pattern, spec id, rows, operand/output `ViewDesc`s, and
    the operand `Refs` as runs) -- and `accesses.jsonl.gz` the full memory-access journal (alloc / root / read / write /
    move / token with event ids) that `provenance.py` checks.  With these three files a checker can locate every
    operand and result of every instance in the trusted record without importing the profile's patterns."""
    import dataclasses
    import gzip

    from verity_vllm.observe.events import ViewDesc
    from verity.ml.circuits import codec
    from verity.ml.circuits.refs import runs

    # [R17 mfast, COORD R17-39 (a) fold/persist] the writer converts by SHAPE, not by a generic walk: a ViewDesc is six known fields (the
    # same dict, in field order, that dataclasses.asdict + _jsonable produced), an operand_refs item is a list of a str and ints already
    # (runs() / cb_a.compact.file_items_from_refs_json), so neither goes through _jsonable's per-element recursion (6.5 M calls on a 32k-instance
    # fold, ~40 % of the writer; at B8 1024/128 the refs alone were 54 M runs x 4 elements).  Output bytes are unchanged (test_compact:
    # test_persist_program_writer_is_byte_identical...); only the meta fields, which may carry non-JSON values, keep the generic conversion.
    def view(v):
        if type(v) is ViewDesc:
            return {"alloc": int(v.alloc), "offset": int(v.offset), "shape": [int(s) for s in v.shape], "strides": [int(s) for s in v.strides],
                    "dtype": v.dtype, "device": v.device}
        return _jsonable(dataclasses.asdict(v)) if dataclasses.is_dataclass(v) else _jsonable(v)

    def run_items(rs):      # runs(): [(space, idx, lo, hi), ...] -> the stored lists, ints coerced exactly as _jsonable's tolist() did
        return [[r[0], int(r[1]), int(r[2]), int(r[3])] for r in rs]

    written: dict[str, Any] = {}
    if result.program is not None:
        desc = getattr(result, "descriptor", None)
        if desc is None:      # a result without the fold's own encoding (older producer): encode here, and the digest must agree
            desc = codec.encode_program(result.program)
            digest = codec.program_digest(desc)
            if result.digest and digest != result.digest:
                raise SystemExit(f"program.json digest {digest[:16]} != fold digest {result.digest[:16]}")
        else:                 # [R17-28 F3] the descriptor `_finish` digested: one encoding, program.json's digest is the fold's by construction
            digest = result.digest or codec.program_digest(desc)
        n_root_nodes = len(result.program.root.body.nodes)
        n_input_nodes = n_root_nodes - len(result.body.nodes)
        doc = {"schema": PROGRAM_SCHEMA, "profile": profile_name, "log": str(log_path), "digest": digest,
               "n_input_nodes": n_input_nodes, "n_instances": len(result.body.nodes), "program": desc,
               # [R17 D90] the fold's root-parameter layout by NAME (prompt_<i> / seed_<i> / peers_<i> / weights, in parameter order = Input
               # node order for homogeneous parameters) and, on a tensor-parallel rank fold, rank/world + the `peers_<i>` field tables
               # (step, k, kind, site, rows, N, leaf offset) -- cb_a.compare locates every peer leaf range from these, never by guessing
               "params": [[n, repr(t)] for n, t in (getattr(result, "params", None) or ())],
               "tp": getattr(result, "tp", None), "peers_layout": _jsonable(getattr(result, "peers_layout", None) or []),
               "supplied": _jsonable(getattr(result, "supplied", None) or {})}   # a `registered_values` fold's, with `input_nodes`
        p = run_dir / "program.json"
        _ptext = json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n"
        p.write_text(_ptext)
        written["program"] = {"path": str(p), "bytes": p.stat().st_size, "digest": digest, "n_input_nodes": n_input_nodes}
        del _ptext
    p = run_dir / "instances.jsonl"
    # [X-09 / MB-01 R13] the header carries the request table (fold order = prompt parameter order `prompt_<i>` of program.json) with each
    # request's prompt length and number of SERVED samples, and every instance line its `request`/`position` attribution
    # (`fold._attribution`; X-09's `meta` channel reads exactly these keys -- without them here the record never carried them and the
    # decomposition always fell through to the data-flow / row-count channels).
    reqs_tab = []
    for i, r in enumerate(getattr(result, "requests", None) or []):
        rid = r.get("request_id")
        reqs_tab.append({"index": i, "request_id": rid, "prompt_len": len(r.get("prompt_token_ids") or []),
                         "served_samples": len((getattr(result, "sampled", None) or {}).get(rid) or []),
                         "max_tokens": r.get("max_tokens")})
    _form = instances_form()
    if _form == "progressions":
        from verity_vllm.program.compact import file_items_from_refs as _file_items
    with p.open("w") as f:
        line = json.dumps({"schema": INSTANCES_SCHEMA_V2 if _form == "progressions" else INSTANCES_SCHEMA, "profile": profile_name,
                           "digest": result.digest, "n_instances": len(result.instances),
                           "note": "root-circuit node of instance k = n_input_nodes + k (program.json)", "requests": reqs_tab,
                           # [R17 mfast F2] how operand_refs items are stored: "runs" = every maximal run ["space", idx, lo, hi]; "progressions" =
                           # node- and parameter-space progressions ["N"|"P", idx, lo, hi, dI, dL, dH, count] (cb_a.compact.runs_from_file_items)
                           "operand_refs_form": _form}) + "\n"
        f.write(line)
        _dumps = json.dumps
        for m in result.instances:
            node = result.body.nodes[m["node"]] if result.body is not None and m.get("node") is not None else None
            rec = {k: m.get(k) for k in ("node", "step", "event", "resolution", "pattern", "spec", "row", "rows", "name", "hint", "request", "position")}
            if m.get("operand_roots"):
                rec["operand_roots"] = m["operand_roots"]     # [R15 fp8] per-operand weight names (multi-weight Computes)
            if m.get("collective"):
                rec["collective"] = m["collective"]           # [R17 D90] kind/step/k/site/rank/world + the peers field the row reads
            rec = _jsonable(rec)                              # the meta fields only (small; may carry tuples / numpy scalars / objects)
            rec["operand_views"] = [view(v) for v in m.get("operand_views") or []]
            rec["output_views"] = [view(v) for v in m.get("output_views") or []]
            if node is not None:
                if _form == "progressions":   # [R17 mfast F2] from the fold's own Refs objects: no runs() expansion, no to_json() tree
                    rec["operand_refs"] = [_file_items(arg.refs) for arg in node.args]   # (== file_items_from_refs_json(refs.to_json()))
                else:
                    rec["operand_refs"] = [run_items(runs(arg.refs)) for arg in node.args]
            f.write(_dumps(rec, separators=(",", ":")) + "\n")
    # [R17 mfast, rev F-r17-12] no compact sidecar is written beside the files of record (unit 7 withdrawn): a producer-written copy in the
    # record's own directory is prover input the checker cannot authenticate; the compact STORED form above (F2) is the file of record itself.
    written["instances"] = {"path": str(p), "bytes": p.stat().st_size, "n": len(result.instances), "operand_refs_form": _form}
    acc = getattr(result, "accesses", None) or []
    if acc:
        p = run_dir / "accesses.jsonl.gz"
        # [COORD R17-28 F1, mprof] the journal is 22 % of the fold stage at B8 1024/128 (161 s: jsonable + dumps + gzip; 237 MB) and no
        # Match-pipeline stage reads it (provenance.py / verify_lane.sh / gen_adversarial do, offline).  Under row_pod's MATCH_PIPELINE=shared
        # it is compressed at level 1 (same lines, ~4x less compressor time, a larger file); unset = level 9 as before.
        level = 1 if os.environ.get("MATCH_PIPELINE", "") == "shared" else 9
        with gzip.open(p, "wt", compresslevel=level) as f:
            for r in acc:
                # the journal entry (fold._journal): view -> the six fields, refs -> its runs() lists, everything else by the generic conversion;
                # key order is the entry's own (unchanged bytes)
                r2 = {}
                for k, v in r.items():
                    if k == "view":
                        r2[k] = view(v) if v is not None else None
                    elif k == "refs":
                        r2[k] = run_items(v) if v is not None else None
                    else:
                        r2[str(k)] = _jsonable(v)
                f.write(_dumps(r2, separators=(",", ":")) + "\n")
        written["accesses"] = {"path": str(p), "bytes": p.stat().st_size, "n": len(acc)}
    return written


def fold_worker(a: Options) -> int:
    for req in ("log", "profile", "out"):
        if getattr(a, req) is None:
            raise SystemExit(f"--worker fold needs --{req}")
    from verity_vllm.pipeline.gc_tuning import tune as _gc_tune   # process-level; content-neutral (VERITY_GC_THRESHOLDS)
    _gc_tune()
    doc, result = fold_summary_result(a.log, a.profile, with_program=not a.no_program, moe_construction=getattr(a, "moe_construction", None), sampler_construction=getattr(a, "sampler_construction", None), registered_values=bool(getattr(a, "registered_values", False)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if getattr(a, "persist_dir", None):
        doc["persisted"] = persist_program(result, Path(a.persist_dir), profile_name=a.profile, log_path=a.log)
    out.write_text(json.dumps(_jsonable(doc), indent=1) + "\n")
    print(f"{out}  digest={doc['digest'][:16]}  instances={doc['n_instances']}  steps={doc['n_steps']}  unresolved={doc['unresolved']}  "
          f"unknown_roots={len(doc['unknown_roots'])}  rss={doc['max_rss_gb']} GB"
          + (f"  persisted={sorted(doc['persisted'])}" if doc.get("persisted") else ""), file=sys.stderr)
    return 0


_JSON_PRIMS = frozenset((str, int, float, bool, type(None)))


def _jsonable(x: Any) -> Any:
    if type(x) in _JSON_PRIMS:      # the common leaf, one set lookup (subclasses -- IntEnum, numpy scalars -- take the general path below)
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return repr(x)
