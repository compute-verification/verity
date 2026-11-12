"""bench/run_config.py — one-shot driver: every stage of one configuration into one run directory, then gates + card.

    verity-vllm match --config-id ID --profile NAME --workload PATH --case B0|B1 --out RUN_DIR \\
        --phase gpu|cpu|all [--python-gpu PY] [--python-cpu PY] [--snapshot-steps 0,1,2,3] [--max-num-seqs 8] \\
        [--laptop] [--dry-run] [--skip STAGES] [--checkpoints PATH] [--checkpoint-dir DIR_OR_GLOB] \\
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
    golden            verity_vllm.properties.golden --check -> RUN_DIR/golden.json
    root_policy, wiring, replay_a, replay_chain
                      no command: stage names a `--skip` still gives (the row's does), recorded as skipped; unskipped they run nothing
    gates             in-process: gates.evaluate_gates -> RUN_DIR/gates.json; card -> RUN_DIR/card.json, RUN_DIR/CARD.md,
                      <cards-dir>/<config_id>.json

Every subprocess stage tees stdout/stderr to RUN_DIR/logs/<stage>.log and records {stage, status, seconds, returncode,
log, cmd, reason} in RUN_DIR/stages.json; a failed stage never aborts later independent stages (its gates come out
skipped with the log as evidence).  `--dry-run` prints the exact command lines and runs nothing.  `--skip a,b` skips
stages or groups (`replay`, `census`, `nonint`, `fold`, `compare`, `resolve`, `root_policy`, `golden`, `capture`,
`control`, `gates`).  `--laptop`: every stage is watched (RSS above --max-rss-gb, default 1.5, or wall above
--stage-timeout, default 180 s, kills it).

Internal: `--worker fold --log LOG --profile NAME --out fold_summary.json [--no-program]` folds a log in this process
and writes the `gates.FOLD_SUMMARY_SCHEMA` summary.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.check import gates as G
from verity_vllm.config import RowEngine, UsageError, option, workload_arg
from verity_vllm.engine.served import (
    ENFORCE_EAGER as ENFORCE_EAGER,  # read through this module
)
from verity_vllm.log import init_logger
from verity_vllm.pipeline import experiment as EXP
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline.match_fold import (
    INSTANCES_FORM_ENV as INSTANCES_FORM_ENV,  # read through this module
)
from verity_vllm.pipeline.match_fold import (
    INSTANCES_SCHEMA as INSTANCES_SCHEMA,  # read through this module
)
from verity_vllm.pipeline.match_fold import (
    INSTANCES_SCHEMA_V2 as INSTANCES_SCHEMA_V2,  # read through this module
)
from verity_vllm.pipeline.match_fold import (
    _jsonable,
    fold_summary_result,
    persist_program,
)
from verity_vllm.pipeline.match_stage import run_stage
from verity_vllm.pipeline.match_workload import (
    _workload_doc,
    _workload_enforce_eager,
    _workload_engine_args,
    _workload_execution,
    _workload_expected_steps,
    fold_construction_args,
)
from verity_vllm.pipeline.match_workload import (
    _workload_moe_construction as _workload_moe_construction,  # read through this module
)
from verity_vllm.records.schema_ids import MATCH_RUN_SCHEMA, MATCH_STAGES_SCHEMA

logger = init_logger(__name__)

DEFAULT_SNAPSHOT_STEPS = "all"        # evidence for every specialization (h1: attention T at non-snapshot steps was uncovered)
TIER_B_STEPS = (0, 1, 2, 3)            # --full-replay (diagnostic): prefill + three decode steps
DEFAULT_K = 10                         # tier a: instances per (specialization, regime) stratum (gate minimum TIER_A_K)
SAMPLING_CHAINS = 3                       # run.json's sampling.chains (no stage reads it)
DEFAULT_PARALLEL_REPLAY = 4            # run.json's `parallel_replay` (1 with --laptop)
DEFAULT_CHECKPOINTS = "manifests/checkpoints.json"
SNAPSHOT_CAP_FLOOR = 32 << 20          # m1_capture's fixed per-tensor snapshot cap, the floor of the derived one
SNAPSHOT_ROW_BYTES_FALLBACK = 64 << 10  # per-token activation row bound when no HF config is readable: 2 x I x bf16 for I <= 16384
LAPTOP_MAX_RSS_GB = 1.5
LAPTOP_STAGE_TIMEOUT_S = 180.0

GPU_STAGES = ("capture", "control", "census", "nonint")
SKIP_GROUPS = {
    "replay": {"replay_a", "replay_chain", "replay_b", "replay_c"},
    "census": {"census", "census_reconcile"},
    "nonint": {"nonint", "nonint_compare"},
    "fold": {"fold", "fold_2", "fold_other"},
}

# ---------------------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------------------


class Plan:
    """Paths and command lines of one run (pure: building a plan touches nothing)."""

    def __init__(self, a: Options) -> None:
        self.a = a
        self.run = Path(required(a, "out"))
        self.py_gpu = a.python_gpu or sys.executable
        self.py_cpu = a.python_cpu or sys.executable
        if str(a.snapshot_steps).strip().lower() == "all":
            # every step of the workload: its declared `expected_engine_steps` (arrivals / chunking make the step count
            # exceed max(max_tokens)), else max(max_tokens); a listed step that never happens costs nothing
            n = a.max_tokens if a.max_tokens is not None else _workload_expected_steps(a.workload)
            if not n:
                logger.warning("[plan] --snapshot-steps all: workload max_tokens unknown, falling back to 0,1,2,3")
                n = 4
            self.steps = list(range(int(n)))
        else:
            self.steps = [int(x) for x in str(a.snapshot_steps).split(",") if x.strip() != ""]
        if not self.steps or self.steps[0] != 0 or self.steps != list(range(len(self.steps))):
            raise SystemExit(f"--snapshot-steps must be contiguous from 0 (got {a.snapshot_steps}): replay of step N needs every earlier KV producer")
        # --full-replay (diagnostic): tier b replays step 0 and these decode steps; every one must be a snapshot step
        self.tier_b_steps = sorted({0, *a.tier_b_steps} & set(self.steps)) if getattr(a, "full_replay", False) else []
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
        return stage in self.skip

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
        """`key=value` engine overrides: the workload's `engine.args` first, then the CLI's (an explicit CLI value wins)."""
        req = _workload_engine_args(self.a.workload)
        cli = _engine_arg_dict(self.a.engine_arg)
        merged = {**req, **cli}
        return [f"{k}={json.dumps(v)}" for k, v in merged.items()]

    def cmd(self, stage: str) -> list[str] | None:
        a, run = self.a, self.run
        workload, case, profile = required(a, "workload"), required(a, "case"), required(a, "profile")
        common_gpu = ["--workload", workload, "--case", case, "--max-num-seqs", str(a.max_num_seqs), "--checkpoints", a.checkpoints]
        if a.max_tokens is not None:
            common_gpu += ["--max-tokens", str(a.max_tokens)]
        common_gpu += [f"--engine-arg={x}" for x in self.engine_args()]   # [GEN batch SHARED-CODE] scheduler/memory knobs -> build_engine
        if stage == "capture":
            return [self.py_gpu, "-m", "verity_vllm.pipeline.cli", "m1-capture", *common_gpu, "--out", str(self.capture_dir),
                    "--snapshot-steps", ",".join(map(str, self.steps)), "--verify-weights"] + \
                   (["--snapshot-max-bytes", str(a.snapshot_max_bytes)] if a.snapshot_max_bytes is not None else []) + \
                   (["--snapshot-max-bytes-op", str(a.snapshot_max_bytes_op)] if a.snapshot_max_bytes_op is not None else [])
        if stage == "control":
            return [self.py_gpu, "-m", "verity_vllm.pipeline.cli", "m1-capture", *common_gpu, "--out", str(self.control_dir), "--no-observer"]
        if stage == "census":
            return [self.py_gpu, "-m", "verity_vllm.pipeline.cli", "census", *common_gpu, "--out", str(self.census_dir), "--no-traces"]
        if stage == "nonint":
            return [self.py_gpu, "-m", "verity_vllm.pipeline.cli", "noninterference", *common_gpu, "--out", str(self.nonint_dir), "--python", self.py_gpu,
                    "--accepted-ctl", str(self.control_tokens_path())]
        if stage == "resolve":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "resolve-log", str(self.log), "--profile", profile, "--out", str(run / "resolution"), "--quiet"]
        if stage == "census_reconcile":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "census", "--reconcile", str(self.census_dir / "census.json"), "--log", str(self.log),
                    "--profile", profile, "--report", str(run / "census_reconcile")]
        if stage == "nonint_compare":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "noninterference", "--compare", "--out", str(self.nonint_dir), "--workload", workload,
                    "--case", case, "--accepted-ctl", str(self.control_tokens_path())]
        if stage == "compare":
            c = [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "fold-compare", "--log", str(self.log), "--snapshots", str(self.snapshots), "--profile", profile,
                 "--out", str(self.p("compare"))]
            g = self.safetensors_glob()
            if g:
                c += ["--safetensors", g]
            return c
        MC = fold_construction_args(workload, a.program_dir or ())
        if stage in ("fold", "fold_2"):   # --persist-dir writes program.json / instances.jsonl / accesses.jsonl.gz
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "match", "--worker", "fold", "--log", str(self.log), "--profile", profile,
                    "--out", str(self.p("fold_summary" if stage == "fold" else "fold_summary_2"))] + MC + (["--persist-dir", str(run)] if stage == "fold" else [])
        if stage == "fold_other":
            if not a.other_log:
                return None
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "match", "--worker", "fold", "--log", a.other_log, "--profile", profile, "--out", str(self.p("fold_summary_other"))] + MC
        if stage == "golden":
            return [self.py_cpu, "-m", "verity_vllm.pipeline.cli", "golden", "--check", "--out", str(self.p("golden"))]
        return None

    def stage_list(self, phase: str) -> list[str]:
        gpu = list(GPU_STAGES) if phase in ("gpu", "all") else []
        cpu: list[str] = []
        if phase in ("cpu", "all"):
            # root_policy, wiring, replay_a and replay_chain have no command: they stay so that a `--skip` naming them
            # records the skip it always has in stages.json (the row's does)
            cpu = ["resolve", "census_reconcile", "nonint_compare", "compare", "fold", "fold_2"]
            if self.a.other_log:
                cpu.append("fold_other")
            cpu += ["root_policy", "golden", "wiring", "replay_a", "replay_chain", "gates"]
        return gpu + cpu


def required(a: Options, name: str) -> str:
    """An option the driver doesn't run without, refused by name when absent."""
    v = getattr(a, name)
    if v is None:
        raise UsageError(f"--{name.replace('_', '-')} is required")
    return v


def profile_config(name: str) -> dict | None:
    try:
        return G.load_profile_config(name)
    except Exception as e:  # noqa: BLE001 - the profile may not exist in this worktree (dry runs)
        logger.info("profile %s: no configuration (%s: %s)", name, type(e).__name__, e)
        return None


# ---------------------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------------------


def _sha256_file(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _hf_config_of(a: Options, C: dict | None) -> dict | None:
    """`program.model.config_of(<hf config.json>)` when an HF config is available."""
    cfg = None
    if a.hf_config:
        cfg = Path(a.hf_config)
    elif a.checkpoint_dir and Path(a.checkpoint_dir).is_dir() and (Path(a.checkpoint_dir) / "config.json").exists():
        cfg = Path(a.checkpoint_dir) / "config.json"
    else:
        repo = (C or {}).get("model")
        if repo:
            home = Path(settings.HF_HOME or (Path.home() / ".cache" / "huggingface"))
            base = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots"
            cands = sorted(base.glob("*/config.json")) if base.exists() else []
            cfg = cands[-1] if cands else None
    if cfg is None or not cfg.exists():
        return None
    an = importlib.import_module("verity_vllm.program.model")   # at its one use: it loads numpy and the Definitions' registry
    try:
        hf_C, _prov = an.config_of(str(cfg))      # (C, provenance) — `dict()` of the pair raised, so every card field read "hand"
        return dict(hf_C)
    except Exception as e:  # noqa: BLE001 - provenance falls back to "hand"
        logger.warning("%s: no configuration read, so provenance falls back to \"hand\" (%s: %s)", cfg, type(e).__name__, e)
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
    C = profile_config(required(a, "profile"))
    man = {
        "schema": MATCH_RUN_SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config_id": a.config_id,
        "profile": a.profile, "case": a.case, "workload_path": a.workload, "workload": workload_summary(a.workload),
        "C": C, "hf_config": _hf_config_of(a, C), "laptop": plan.laptop, "snapshot_steps": plan.steps, "tier_b_steps": plan.tier_b_steps,
        "verify_weights": True, "parallel_replay": plan.parallel, "phase": a.phase,
        "sampling": {"seed": a.seed, "k": a.k, "chains": SAMPLING_CHAINS, "full_replay": bool(getattr(a, "full_replay", False))},
        "python_gpu": plan.py_gpu, "python_cpu": plan.py_cpu,
        "engine_args_requested": {"max_num_seqs": a.max_num_seqs, "enforce_eager": _workload_enforce_eager(a.workload), **_engine_arg_dict(plan.engine_args())},   # from the workload's execution
        "declared_execution": _workload_execution(a.workload),
        "engine_args_required_by_workload": _workload_engine_args(a.workload),     # merged into engine_args_requested
        "snapshot_max_bytes": a.snapshot_max_bytes if a.snapshot_max_bytes is not None else 32 << 20,
        "snapshot_cap_derivation": a.snapshot_cap_derivation,   # None = an explicit CLI cap
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
    """spec.json -- the specification (configuration / workload / initial state) as known before execution, from the workload
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
        except (OSError, ValueError, RecursionError, AttributeError, TypeError):   # no manifest on this host, or not one: the checkpoint_dir still identifies the weights
            pass
        wl_doc = _workload_doc(a.workload)
        spec = EXP.build_spec(config_id=required(a, "config_id"), case=required(a, "case"), profile=a.profile, workload_path=a.workload, workload_doc=wl_doc,
                              checkpoint=ck, checkpoint_dir=a.checkpoint_dir, hf_config=man.get("hf_config"),
                              engine_args=_engine_arg_dict(plan.engine_args()), max_num_seqs=a.max_num_seqs,
                              sampling={}, tp=int(getattr(a, "tp", 1) or 1), dtype=None, snapshot_steps=plan.steps,
                              snapshot_max_bytes=a.snapshot_max_bytes,
                              harness={"enforce_eager": _workload_enforce_eager(a.workload), "execution": _workload_execution(a.workload),
                                       "driver": "verity_vllm.pipeline.match", "python_gpu": a.python_gpu,
                                       "family": getattr(a, "family", None)})
        if a.phase == "cpu" and not (plan.run / "spec.json").exists():
            spec["reconstructed_after_execution"] = True
            spec["digest"] = EXP.digest_doc({k: v for k, v in spec.items() if k != "digest"})
        note = EXP.write_spec(plan.run, spec)
        logger.info("[spec] %s digest=%s%s", "written" if note.get("written") else "kept",
                    (note.get("digest") or note.get("existing_digest") or "")[:12],
                    "" if note.get("same", True) else "  (this phase would have produced a DIFFERENT spec; not rewritten)")
        return note
    except Exception as e:  # noqa: BLE001 - informational
        logger.warning("[spec] not written: %s: %s", type(e).__name__, e)
        return {"written": False, "error": f"{type(e).__name__}: {e}"}


def write_realized(plan: Plan) -> dict | None:
    """After the GPU phase: the realized shape of the run into run.json["realized"] -- per request the prompt length, the
    generated length, the arrival / first / finish engine steps and finish reason (from capture/tokens.json), the engine-step
    count of the observed and the control run, and the snapshot steps that actually happened.  "Record the exact realized
    input/output lengths in every run artifact": run.json is the artifact every gate and the card read first."""
    cap_p = plan.snapshots / "tokens.json" if plan.a.log else plan.capture_dir / "tokens.json"
    ctl_p = plan.control_tokens_path()
    try:
        cap = json.loads(cap_p.read_text()) if cap_p.exists() else None
    except (OSError, ValueError, RecursionError):   # unreadable, not JSON, or nested past what the parser reads: no record
        cap = None
    try:
        ctl = json.loads(ctl_p.read_text()) if ctl_p.exists() else None
    except (OSError, ValueError, RecursionError):
        ctl = None
    src = cap or ctl
    if src is None:
        return None
    reqs = []
    for r in src.get("requests", []):
        prompt, generated = r.get("prompt_token_ids"), r.get("generated_token_ids")   # counted where the record states no count
        reqs.append({"request_id": r.get("request_id"), "prompt_len": r.get("prompt_len", len(prompt) if prompt else 0),
                     "max_tokens": r.get("max_tokens"), "n_generated": r.get("n_generated", len(generated) if generated else 0),
                     "arrive_step": r.get("arrive_step", 0), "arrive_step_realized": r.get("arrive_step_realized"),
                     "first_step": r.get("first_step"), "finish_step": r.get("finish_step"), "finish_reason": r.get("finish_reason")})
    snaps = None
    sp = plan.snapshots / "snapshots.json"
    if sp.exists():
        try:
            sj = json.loads(sp.read_text())
            kept = sj.get("steps") or sj.get("snapshot_steps")   # the capture's list of the steps it kept, under either name
            snaps = (sorted({int(s) for s in kept}) or None) if kept else None
        except (OSError, ValueError, RecursionError, OverflowError, AttributeError, TypeError):   # unreadable, or no list of whole steps (an infinite one is none)
            snaps = None
    real = {"requests": reqs, "prompt_lens": [r["prompt_len"] for r in reqs], "n_generated": [r["n_generated"] for r in reqs],
            "engine_steps_capture": _engine_steps(cap), "engine_steps_control": _engine_steps(ctl),
            "expected_engine_steps_declared": src.get("expected_engine_steps"), "serving_profile": src.get("serving_profile"),
            "context_class": src.get("context_class"), "snapshot_steps_realized": snaps,
            "max_concurrent_requests": _max_concurrent(reqs)}
    p = plan.p("run")
    man = json.loads(p.read_text()) if p.exists() else {}
    man["realized"] = real
    p.write_text(json.dumps(man, indent=1, default=str) + "\n")
    return real


def _engine_steps(tokens: dict | None) -> Any:
    """The engine steps a tokens record states, at its top or under its `timing`; None with no record, or one that states neither."""
    timing = tokens.get("timing") if tokens else None
    return (tokens.get("engine_steps") if tokens else None) or (timing.get("engine_steps") if timing else None)


def _max_concurrent(reqs: list[dict]) -> int | None:
    """Peak number of requests active in one engine step, from the realized first/finish steps."""
    spans: list[tuple[int, int]] = []
    for r in reqs:
        first, finish = r.get("first_step"), r.get("finish_step")
        if first is None or finish is None:
            return None
        spans.append((first, finish))
    if not spans:
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
        # an interrupted save (ENOSPC mid-write leaves a 0-byte stages.json) must not block `--phase cpu`
        # over the preserved capture: set the unreadable record aside under a stamped name and start the stage list empty.
        aside = p.with_name(p.name + f".unreadable-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        p.replace(aside)
        logger.warning("[stages] %s unreadable (%s: %s; %d bytes) -> set aside as %s; resuming with no stage records",
                       p.name, e.__class__.__name__, e, len(text), aside.name)
        return []
    return list(d.get("stages", d) if isinstance(d, dict) else d)


def save_stages(run: Path, recs: list[dict]) -> None:
    byname: dict[str, dict] = {}
    for r in recs:
        byname[r["stage"]] = r
    (run / G.ARTIFACTS["stages"]).write_text(json.dumps({"schema": MATCH_STAGES_SCHEMA, "stages": list(byname.values())}, indent=1) + "\n")


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
    workload: str | None = option("--workload", type=workload_arg, default=None)
    case: str | None = option("--case", default=None, help="checkpoint role (B0, B1, ... any role in checkpoints.json with a profile module)")
    out: str | None = option("--out", default=None, help="RUN_DIR (driver) or fold_summary.json (--worker fold)")
    run_dir: str | None = option("--run-dir", default=None, help="an existing RUN_DIR: --out plus case/workload/profile/config-id/checkpoint-dir/steps from its run.json (the sealed verifier re-runs the CPU phase this way from a fresh checkout)")
    phase: str = option("--phase", choices=["gpu", "cpu", "all"], default="all")
    python_gpu: str | None = option("--python-gpu", default=None, help="the pod's torch venv python (default: sys.executable)")
    python_cpu: str | None = option("--python-cpu", default=None, help="the torch-free 3.12 python (default: sys.executable)")
    snapshot_steps: str = option("--snapshot-steps", default=DEFAULT_SNAPSHOT_STEPS, help="contiguous from 0, or `all` = every step of the workload (default all)")
    full_replay: bool = option("--full-replay", default=False, action="store_true", help="run.json's sampling.full_replay (no stage reads it)")
    #: with --full-replay: run.json's tier_b_steps; no flag sets it, and a --run-dir re-run keeps the ones its run.json records
    tier_b_steps: tuple[int, ...] = field(default=TIER_B_STEPS, init=False)
    parallel: int | None = option("--parallel", type=int, default=None, help="run.json's parallel_replay (default 4; 1 with --laptop)")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    engine_arg: list[str] = option("--engine-arg", action="append", default=[], help="LLM(**kw) override key=value (JSON value) for every GPU stage, repeatable; e.g. max_num_batched_tokens=64, enable_prefix_caching=true")
    max_tokens: int | None = option("--max-tokens", type=int, default=None, help="override the workload's max_tokens (passed to the GPU stages)")
    snapshot_max_bytes: int | None = option("--snapshot-max-bytes", type=int, default=None, help="(capture) per-tensor snapshot cap in bytes (m1_capture default 32 MiB; a prefill MLP row block above the cap has no snapshot evidence -- raise it for long / batched prefills, the run.json records the value)")
    #: how `main` derived the cap (run.json's snapshot_cap_derivation); None under a --snapshot-max-bytes given; no flag sets it
    snapshot_cap_derivation: dict | None = field(default=None, init=False)
    snapshot_max_bytes_op: int | None = option("--snapshot-max-bytes-op", type=int, default=None, help="(capture) lower per-tensor cap at dispatcher-call events; Triton launch events keep --snapshot-max-bytes, so a large launch output is copied once (cov-moe: the fused-MoE intermediates of a 2048-token prefill otherwise exceed the pod's pinned host memory; the run.json records the value)")
    checkpoints: str = option("--checkpoints", default=DEFAULT_CHECKPOINTS)
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None, help="HF snapshot dir or model*.safetensors glob (compare --safetensors)")
    hf_config: str | None = option("--hf-config", default=None, help="HF config.json for per-field C provenance (analytic.config_of)")
    family: str | None = option("--family", default=None, help="hand-authored family reference name (default Serve_v4 in the card)")
    log: str | None = option("--log", default=None, help="reuse an existing raw log instead of RUN_DIR/capture/log.jsonl.gz")
    snapshots: str | None = option("--snapshots", default=None, help="its snapshots dir (default: the log's directory)")
    other_log: str | None = option("--other-log", default=None, help="a second capture of the same workload (G6 detail: cross-capture digest)")
    control_tokens: str | None = option("--control-tokens", default=None, help="reuse an existing control run's tokens.json instead of RUN_DIR/control/tokens.json")
    laptop: bool = option("--laptop", default=False, action="store_true", help="laptop budget: RSS/wall watchdog")
    max_rss_gb: float | None = option("--max-rss-gb", type=float, default=None, help="kill a stage above this RSS (default 1.5 with --laptop)")
    stage_timeout: float | None = option("--stage-timeout", type=float, default=None, help="kill a stage above this wall time in s (default 180 with --laptop)")
    dry_run: bool = option("--dry-run", default=False, action="store_true", help="print the exact command lines, run nothing")
    skip: str = option("--skip", default="", help="comma list of stages/groups to skip")
    cards_dir: str = option("--cards-dir", default="out/gen/cards")
    tree: str = option("--tree", default=".", help="the tree that runs, whose version experiment.json records: its EXPORT.json, else its git head (default: the working directory)")
    k: int = option("--k", type=int, default=DEFAULT_K, help="run.json's sampling.k (no stage reads it)")
    seed: int | None = option("--seed", type=int, default=None, help="run.json's sampling.seed (no stage reads it)")
    no_program: bool = option("--no-program", default=False, action="store_true", help="(--worker fold) skip Program(): no digest")
    persist_dir: str | None = option("--persist-dir", default=None, help="(--worker fold) write program.json + instances.jsonl + accesses.jsonl.gz here")
    program_dir: list[str] | None = option("--program-dir", action="append", default=None, help="the Build's Program dir(s) the Match compares against; the fold binds the top-p select under the construction they declare (sampling_event.program_sampler_construction)")
    sampler_construction: str | None = option("--sampler-construction", default=None, help="(--worker fold) the Programs' sampler construction (the driver passes it from --program-dir)")
    registered_values: bool = option("--registered-values", default=False, action="store_true", help="(--worker fold) the program's values are inputs (Profile.registered_values); the driver passes what the --program-dir Programs register")
    match_pipeline: str = option("--match-pipeline", env="MATCH_PIPELINE", default="", help="(--worker fold) the row's Match pipeline: under `shared` the access journal is compressed at level 1 (the same lines, a larger file)")
    moe_construction: str | None = option("--moe-construction", default=None, help="(--worker fold) the workload's DECLARED MoE construction (manifest target.moe_construction: indexed-read | indexed-read-ordered | padded); the driver passes it from --workload")


def main(a: Options) -> int:
    if a.worker == "fold":
        return fold_worker(a)
    if a.run_dir:
        _defaults_from_run_dir(a)
    if a.profile is None and a.case:
        try:
            from verity_vllm.observe.fold.profiles import profile_module_for_case
            a.profile = profile_module_for_case(a.case)
        except Exception as e:
            raise UsageError(f"--profile not given and no profile module declares case {a.case!r}: {e}") from e
    if a.config_id is None and a.case and a.workload:
        a.config_id = f"{a.case}-{Path(a.workload).stem}"
    for req in ("config_id", "profile", "workload", "case", "out"):
        required(a, req)
    # no explicit --snapshot-max-bytes: derive the cap from the workload's largest forward and the checkpoint's widths
    # (snapshot_cap_bytes) instead of m1_capture's fixed 32 MiB, which leaves the prefill MLP rows of every 1024-class workload without
    # snapshot evidence.  An explicit value still wins; the run.json records the derivation either way.
    if a.snapshot_max_bytes is None:
        C0 = profile_config(required(a, "profile"))
        cap, why = snapshot_cap_bytes(_workload_doc(a.workload), _hf_config_of(a, C0), _engine_arg_dict(a.engine_arg))
        a.snapshot_max_bytes, a.snapshot_cap_derivation = cap, why
        if why["raised"]:
            logger.info("[plan] --snapshot-max-bytes derived: %s B (%s rows [%s] x %s B [%s]; the 32 MiB cap of record skips the prefill MLP "
                        "rows of 1024-class workloads)", cap, why["rows"], why["rows_source"], why["row_bytes"], why["row_bytes_source"])

    plan = Plan(a)
    stages = plan.stage_list(a.phase)
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
                      f"{Path(a.cards_dir) / (required(a, 'config_id') + '.json')}")
                continue
            cmd = plan.cmd(st)
            if cmd is None:
                continue
            print(f"[{st}] {shlex.join(cmd)}")
        if a.phase in ("gpu", "all"):
            print(f"[gpu_done] write {plan.p('gpu_done')} with per-stage timings")
        return 0

    man = write_manifest(plan, a)
    spec_note = write_spec_for(plan, a, man)          # spec.json (known before execution) -- written once, before any GPU phase
    recs = load_stages(plan.run)
    logs = plan.run / "logs"
    if a.phase in ("gpu", "all"):
        GL.acquire()                 # a leased row's GPU phase: its GPU from here to the exit (`row_stages.SingleRow.match`); nothing reads the device before
    t_phase = time.time()
    if a.phase == "cpu":
        try:
            write_realized(plan)             # an existing capture's realized shape is in run.json before the gates / card read it
        except Exception as e:  # noqa: BLE001
            logger.warning("[realized] not written: %s: %s", type(e).__name__, e)
    for st in stages:
        if plan.skipped(st):
            recs.append({"stage": st, "status": "skipped", "returncode": None, "seconds": 0.0, "log": None, "cmd": None, "reason": "--skip"})
            save_stages(plan.run, recs)
            continue
        if st == "gates":
            try:
                write_realized(plan)         # realized shape into run.json before the card is built
            except Exception as e:  # noqa: BLE001
                logger.warning("[realized] not written: %s: %s", type(e).__name__, e)
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
        logger.info("[%s] %s", st, shlex.join(cmd))
        rec = run_stage(st, cmd, logs / f"{st}.log", max_rss_gb=max_rss, timeout_s=timeout)
        logger.info("[%s] %s in %s s", st, rec["status"], rec["seconds"])
        recs.append(rec)
        save_stages(plan.run, recs)
    if a.phase in ("gpu", "all"):
        gd = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t_phase, 3), "config_id": a.config_id,
              "stages": {r["stage"]: {"status": r["status"], "seconds": r.get("seconds")} for r in recs if r["stage"] in GPU_STAGES}}
        plan.p("gpu_done").write_text(json.dumps(gd, indent=1) + "\n")
    # the phase's last word on the realized shape is a print, with no logger's head: `ops/cov_pod.sh` shows a GPU phase's
    # first "[realized]" line as it is written, and these two are the ones a GPU phase writes
    try:
        real = write_realized(plan)          # realized lengths / steps into run.json (also on a --phase cpu re-run over an old capture)
        if real:
            print(f"[realized] prompt_lens={real['prompt_lens']} n_generated={real['n_generated']} steps capture/control="
                  f"{real['engine_steps_capture']}/{real['engine_steps_control']} max_concurrent={real['max_concurrent_requests']}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - informational
        print(f"[realized] not written: {type(e).__name__}: {e}", file=sys.stderr)
    try:
        EXP.write_experiment(plan.run, repo=Path(a.tree), spec_note=spec_note)   # the Experiment index over the run directory
    except Exception as e:  # noqa: BLE001 - informational
        logger.warning("[experiment] not written: %s: %s", type(e).__name__, e)
    bad = [r for r in recs if r["stage"] in stages and r["status"] not in ("ok", "skipped")]
    return 1 if bad else 0


def _defaults_from_run_dir(a: Options) -> None:
    """Fill unset arguments from RUN_DIR/run.json (the manifest the GPU phase wrote)."""
    run = Path(required(a, "run_dir"))
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
    if man.get("tier_b_steps"):
        a.tier_b_steps = tuple(int(s) for s in man["tier_b_steps"])
    if a.snapshot_max_bytes is None and man.get("snapshot_max_bytes"):
        # a CPU-phase re-run keeps the cap the capture RAN with (and its derivation), never a re-derived one
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


def snapshot_cap_bytes(doc: dict | None, hf: dict | None, engine_args: dict | None = None) -> tuple[int, dict]:
    """The per-tensor snapshot cap a Match capture needs so that EVERY per-token
    activation of the largest forward the workload can schedule has snapshot evidence: rows x widest row x 2 bytes, never below the
    32 MiB cap of record.  Rows = the engine's `max_num_batched_tokens` (the workload's `engine.args`, then the CLI's engine
    args), else the longest prompt plus one decode row per request; the widest row = max(gate_up 2I, qkv (NH+2KVH)D, hidden H) from
    the checkpoint's config (analytic.config_of keys GU/QKV/H), else 64 KiB.  Why: at 1024/128 the prefill's gate_up output is
    [1024+B, 2I] bf16 = 33.7 MB for Llama-3.2-1B (I 8192) and 36.7 MB for Qwen2.5-1.5B (I 8960) -- over the 32 MiB cap, listed in
    snapshots.json as skipped -- so the Commit's form (B) oracle-compare had 'no snapshot of alloc ... (0 entries of that alloc)' for the
    prefill-step MLP rows of every 1024-class row (llama B4, qwen H100 B1 -- FAIL by name where Gemm_v2 has no
    evaluator; global int-2 B16 mixed-arrivals: 16 layers x 5 rows at engine step 36, the [1028, 16384] gate_up of a 1024-token
    arrival); 256/32 rows ([256+B, 2I] = 8.6 MB) never hit it.  MoE routed intermediates are not in the width (cov-moe's
    --snapshot-max-bytes-op contract is unchanged; OLMoE stays at the 32 MiB floor).  Returns (cap, derivation) -- the derivation
    goes into run.json so a reader sees why the record's cap is what it is."""
    doc = doc or {}
    stated = doc.get("requests")   # a workload read for its engine arguments alone states no requests
    reqs = stated if stated else []
    ea = {**dict(RowEngine.of(doc).args), **dict(engine_args or {})}
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
        wk = max(widths, key=widths.__getitem__)
        row_bytes, width_src = widths[wk] * 2, f"hf config {wk}={widths[wk]} x bf16"
    else:
        row_bytes, width_src = SNAPSHOT_ROW_BYTES_FALLBACK, "no HF config: 64 KiB per row"
    need = rows * row_bytes
    cap = max(SNAPSHOT_CAP_FLOOR, need)
    return cap, {"rows": rows, "rows_source": rows_src, "row_bytes": row_bytes, "row_bytes_source": width_src, "needed_bytes": need,
                 "floor_bytes": SNAPSHOT_CAP_FLOOR, "cap_bytes": cap, "raised": cap > SNAPSHOT_CAP_FLOOR}


# ---------------------------------------------------------------------------------------------------------
# the fold worker (one process = one fold; the driver runs it twice for G6)
# ---------------------------------------------------------------------------------------------------------


def fold_worker(a: Options) -> int:
    log, profile, summary = a.log, a.profile, a.out
    if log is None or profile is None or summary is None:
        raise SystemExit(f"--worker fold needs --{next(req for req in ('log', 'profile', 'out') if getattr(a, req) is None)}")
    # process-level; content-neutral (VERITY_GC_THRESHOLDS)
    from verity_vllm.gc_tuning import tune as _gc_tune
    _gc_tune()
    doc, result = fold_summary_result(log, profile, with_program=not a.no_program, moe_construction=getattr(a, "moe_construction", None), sampler_construction=getattr(a, "sampler_construction", None), registered_values=bool(getattr(a, "registered_values", False)))
    out = Path(summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.persist_dir:
        doc["persisted"] = persist_program(result, Path(a.persist_dir), profile_name=profile, log_path=log,
                                           journal_level=1 if a.match_pipeline == "shared" else 9)
    out.write_text(json.dumps(_jsonable(doc), indent=1) + "\n")
    logger.info("%s  digest=%s  instances=%s  steps=%s  unresolved=%s  unknown_roots=%d  rss=%s GB%s", out, doc["digest"][:16],
                doc["n_instances"], doc["n_steps"], doc["unresolved"], len(doc["unknown_roots"]), doc["max_rss_gb"],
                f"  persisted={sorted(doc['persisted'])}" if doc.get("persisted") else "")
    return 0
