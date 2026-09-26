"""The integration's typed configuration, and where its run definitions live.

* **Run definitions.**  ``manifests/`` (checkpoint pins, semantic profiles) and ``workloads/`` (row workloads) sit beside
  the package in the source tree; they are run definitions, not package data.  Library code locates them, and the source
  tree itself, through ``ROOT``, ``MANIFESTS``, ``WORKLOADS`` and ``CHECKPOINTS``.
* **What runs.**  The row-id grammar (:class:`RowId`), the dtype vocabulary (``DTYPES``), the target table
  (``TARGETS``), and :class:`EnginePin`, the pinned engine record every engine is built from: model, full revision and
  checkpoint role, checked against ``manifests/checkpoints.json``.
* **How a stage runs.**  :class:`BuildConfig`, :class:`MatchConfig` and :class:`CommitConfig`: defaults in code,
  overrides from flags or the environment of ``verity-vllm row``.
* **Where it runs.**  :class:`RunEnv`: interpreters, caches and roots.  ``pipeline/cli.py`` fills it from flags and the
  environment (the machine-path defaults live there); it is recorded as telemetry and never hashed.
* **Subcommand options.**  Every ``verity-vllm`` subcommand takes one frozen dataclass whose fields are declared with
  :func:`option`; ``pipeline/cli.py`` is the only parser.  A subcommand refuses a command line by raising
  :class:`UsageError`.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

#: The source tree holding the ``verity_vllm`` package, ``manifests/`` and ``workloads/``.
ROOT = Path(str(resources.files("verity_vllm"))).resolve().parent
MANIFESTS = ROOT / "manifests"
WORKLOADS = ROOT / "workloads"
CHECKPOINTS = MANIFESTS / "checkpoints.json"


def checkpoints_hf_home() -> str | None:
    """The HF cache the checkpoint pins name (``checkpoints.json`` ``hf_home``); None without the manifest."""
    try:
        return json.loads(CHECKPOINTS.read_text()).get("hf_home") or None
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------- options

class UsageError(ValueError):
    """A command line the subcommand refuses; ``pipeline/cli.py`` reports it as a usage error (exit 2)."""


def option(*flags: str, env: str | None = None, env_empty: bool = False, **kw: Any) -> Any:
    """One command-line option of a subcommand's options dataclass.

    ``flags`` and ``kw`` are ``argparse`` ``add_argument`` arguments, applied by ``pipeline/cli.py``.  ``kw["default"]``
    is also the field's default; a field without one is required (a required flag, or a positional).  ``env`` names the
    environment variable whose value the CLI takes as the default, converted by ``type`` like a flag value; an empty
    variable counts as unset unless ``env_empty``.
    """
    md = {"flags": flags, "kw": kw, "env": env, "env_empty": env_empty}
    if "default" not in kw:
        return field(metadata=md)
    d = kw["default"]
    if isinstance(d, (list, dict, set)):
        return field(default_factory=lambda: copy.copy(d), metadata=md)
    return field(default=d, metadata=md)


def group(cls: type) -> Any:
    """A nested options dataclass whose options join the subcommand's own (``verity-vllm row``'s stage configs)."""
    return field(default_factory=cls, metadata={"group": cls})


def environ_snapshot() -> Any:
    """The environment the CLI was started with, as a field: filled by ``pipeline/cli.py``, never by a flag."""
    return field(default_factory=dict, metadata={"environ": True})


# ---------------------------------------------------------------------------------------------------------------- what runs

#: Row-id dtype token -> the checkpoint weight format the row serves.
DTYPES: dict[str, str] = {"bf16": "bfloat16", "fp8": "float8_e4m3fn"}

#: Row-id target token -> the device class's compute capability.  Extend by name; an unknown token is a refusal, not a guess.
TARGETS: dict[str, tuple[int, int]] = {
    "l40s": (8, 9), "l40": (8, 9), "l4": (8, 9), "rtx4090": (8, 9),
    "a100": (8, 0),
    "a10": (8, 6), "a10g": (8, 6), "a40": (8, 6),
    "h100": (9, 0), "h200": (9, 0), "h20": (9, 0),
    "b200": (10, 0), "gb200": (10, 0),
}

ROW_FIELDS = ("model", "dtype", "target", "world", "batch", "max_input", "max_output", "arrivals", "sampling", "execution")
_ROW_INT_PREFIX = {"world": "tp", "batch": "b", "max_input": "i", "max_output": "o"}
_TOKEN = re.compile(r"__(tp|b|i|o)(\d+)(?=__)")


@dataclass(frozen=True)
class RowId:
    """A sweep row id: ``<model>__<dtype>__<target>__tp<N>__b<N>__i<N>__o<N>__<arrivals>__<sampling>__<execution>``.

    ``str(row)`` gives the id back; ``RowId.parse`` refuses anything else.
    """

    model: str
    dtype: str
    target: str
    world: int
    batch: int
    max_input: int
    max_output: int
    arrivals: str
    sampling: str
    execution: str

    @classmethod
    def parse(cls, row: str) -> RowId:
        parts = str(row).split("__")
        if len(parts) != len(ROW_FIELDS) or not all(parts):
            raise ValueError(f"row id {row!r}: expected {len(ROW_FIELDS)} '__'-separated fields ({', '.join(ROW_FIELDS)})")
        vals: dict[str, Any] = {}
        for name, tok in zip(ROW_FIELDS, parts):
            pfx = _ROW_INT_PREFIX.get(name)
            if pfx is None:
                vals[name] = tok
            elif tok.startswith(pfx) and tok[len(pfx):].isdigit():
                vals[name] = int(tok[len(pfx):])
            else:
                raise ValueError(f"row id {row!r}: field {name} is {tok!r}, expected {pfx}<N>")
        return cls(**vals)

    def __str__(self) -> str:
        return "__".join((self.model, self.dtype, self.target, f"tp{self.world}", f"b{self.batch}", f"i{self.max_input}",
                          f"o{self.max_output}", self.arrivals, self.sampling, self.execution))


def row_int(row: str, prefix: str, default: int = 1) -> int:
    """The ``__<prefix><N>__`` field of a row id (``tp``, ``b``, ``i``, ``o``), ``default`` when the id has none."""
    for m in _TOKEN.finditer(str(row)):
        if m.group(1) == prefix:
            return int(m.group(2))
    return default


def row_world(row: str) -> int:
    """The tensor-parallel world a row id declares (``__tp<N>__``), 1 when absent."""
    return row_int(row, "tp")


def row_batch(row: str) -> int:
    """The batch a row id declares (``__b<N>__``), 1 when absent."""
    return row_int(row, "b")


def row_execution(row: str) -> str | None:
    """The last ``__`` field of a row id (its execution label), None for an id without one."""
    return str(row).rsplit("__", 1)[1] if "__" in str(row) else None


def row_target(row: str) -> str | None:
    """The device token of a row id (its third field when the fourth is ``tp<N>``), None when the id lacks the grammar."""
    parts = str(row).split("__")
    return parts[2] if len(parts) >= 4 and parts[3].startswith("tp") else None


def checkpoint_entries(checkpoints: str | Path = CHECKPOINTS) -> list[dict[str, Any]]:
    return list(json.loads(Path(checkpoints).read_text())["checkpoints"])


@dataclass(frozen=True)
class EnginePin:
    """The pinned engine a row or :class:`verity_vllm.LLM` runs.

    ``model`` is the HF repo id and ``revision`` its full commit sha, both as ``manifests/checkpoints.json`` pins them;
    ``case`` is the pin's role (the engine profile's key) and ``local_path`` the snapshot on this host.  ``world`` is the
    tensor-parallel degree.  Build it with :meth:`resolve`, which refuses an unpinned model or another revision.
    """

    model: str
    revision: str
    case: str
    local_path: str | None
    world: int = 1

    @classmethod
    def resolve(cls, model: str, revision: str, *, world: int = 1, checkpoints: str | Path = CHECKPOINTS) -> EnginePin:
        entries = [c for c in checkpoint_entries(checkpoints) if c.get("repo") == model]
        if not entries:
            pinned = sorted({str(c.get("repo")) for c in checkpoint_entries(checkpoints)})
            raise ValueError(f"model {model!r} is not pinned in {Path(checkpoints).name} (pinned: {', '.join(pinned)})")
        hits = [c for c in entries if str(c.get("revision", "")) == str(revision)]
        if not hits:
            revs = sorted({str(c.get("revision")) for c in entries})
            raise ValueError(f"model {model!r} revision {revision!r} is not the pinned revision ({', '.join(revs)}); "
                             f"pass the full commit sha")
        # a repo can be pinned under several roles; the downloaded one is the engine's
        hit = next((c for c in hits if c.get("downloaded") and c.get("local_path")), hits[0])
        return cls(model=model, revision=str(hit["revision"]), case=str(hit.get("role")), local_path=hit.get("local_path"),
                   world=int(world))


def checkpoint_dir(model: str, revision: str, hf_home: str, checkpoints: str | Path = CHECKPOINTS) -> str | None:
    """The local snapshot of a workload's (model, revision): the first pin whose revision starts with ``revision[:10]``
    and whose ``local_path`` exists, else the HF hub snapshot under ``hf_home`` of that revision prefix; None when absent."""
    local = None
    for c in checkpoint_entries(checkpoints):
        if c.get("repo") == model and str(c.get("revision", "")).startswith(revision[:10]):
            local = c.get("local_path")
            break
    if not local or not os.path.exists(local):
        base = f"{hf_home}/hub/models--{model.replace('/', '--')}/snapshots"
        for d in sorted(os.listdir(base)) if os.path.exists(base) else []:
            if d.startswith(revision[:10]):
                local = f"{base}/{d}"
    return local


# ---------------------------------------------------------------------------------------------------------------- stage configs

@dataclass(frozen=True, kw_only=True)
class BuildConfig:
    """How the Build stage of ``verity-vllm row`` runs.  None = the row's derived default."""

    build_jobs: str | None = option("--build-jobs", env="BUILD_JOBS", default=None,
                                    help="concurrent derives: N, or auto (cgroup headroom / 20 GiB, 1..8); default 1 (TP: ranks x shapes)")
    build_timeout: int | None = option("--build-timeout", env="BUILD_TIMEOUT", type=int, default=None,
                                       help="seconds per derive; default scales with the envelope (TP: 7200)")
    lp: int | None = option("--lp", env="LP", type=int, default=None,
                            help="request wrapper prompt length (default: the longest prompt; a stochastic row: r0's)")
    t: int | None = option("--t", env="T", type=int, default=None,
                           help="request wrapper decode steps (default: the largest cap - 1; a stochastic row: r0's)")


@dataclass(frozen=True, kw_only=True)
class MatchConfig:
    """How the Match stage of ``verity-vllm row`` runs.  None = the row's derived default."""

    match_impl: str = option("--match-impl", env="MATCH_IMPL", default="fast", help="global match checker: fast or record (the audit)")
    match_pipeline: str = option("--match-pipeline", env="MATCH_PIPELINE", default="shared",
                                 help="shared = one batch decomposition inside the global match")
    match_phase: str = option("--match-phase", env="MATCH_PHASE", default="all", choices=("all", "cpu"),
                              help="cpu = resume the CPU half over the capture already on disk")
    snapshot_steps: str | None = option("--snapshot-steps", env="MATCH_SNAP_STEPS", default=None,
                                        help="0,1 | all; default 0,1 greedy, all stochastic or when form (B) must be complete")
    snapshot_max_bytes: str | None = option("--snapshot-max-bytes", env="MATCH_SNAP_MAX_BYTES", default=None,
                                            help="explicit per-tensor snapshot cap (default: derived by the Match)")
    within_step: str = option("--within-step", env="WITHIN_STEP", default="dag", help="within-step criterion: dag or sequence")
    instances_form: str = option("--instances-form", env="VERITY_INSTANCES_FORM", default="progressions",
                                 help="stored form of the fold's instances.jsonl: progressions or runs")
    acquire_manifest: str | None = option("--acquire-manifest", env="ACQUIRE_MANIFEST", default=None,
                                          help="required-value manifest for the capture (default: the Build's manifest.json)")
    match_timeout: int | None = option("--match-timeout", env="MATCH_TIMEOUT", type=int, default=None,
                                       help="seconds for the capture + fold (default scales with B and the envelope; TP: 3600)")
    compare_timeout: int | None = option("--compare-timeout", env="COMPARE_TIMEOUT", type=int, default=None,
                                         help="seconds per comparator (default scales with B and the envelope)")
    fold_timeout: int | None = option("--fold-timeout", env="FOLD_TIMEOUT", type=int, default=None,
                                      help="TP: seconds for the per-rank fold Match (default 7200)")
    fold_profile: str | None = option("--fold-profile", env="FOLD_PROFILE", env_empty=True, default=None,
                                      help="TP: rank profile module (default derived_<ROLE>_tp<W>; empty = collective-level Match only)")


@dataclass(frozen=True, kw_only=True)
class CommitConfig:
    """How the Commit stage of ``verity-vllm row`` runs.  None = the row's derived default."""

    pairs: int = option("--pairs", env="PAIRS", type=int, default=3, help="instrumented/control pairs")
    commit_gpu_util: str | None = option("--commit-gpu-util", env="COMMIT_GPU_UTIL", default=None,
                                         help="TP: gpu_memory_utilization of the Commit engine (default --gpu-util)")
    window_mb: int = option("--window-mb", env="WINDOW_MB", type=int, default=256, help="device staging window (0 = per-step blocks)")
    window_slots: int = option("--window-slots", env="WINDOW_SLOTS", type=int, default=8)
    retain: str = option("--retain", env="RETAIN", default="host", help="where raw copies are retained: host, device or gpu")
    open_after_release: str = option("--open-after-release", env="OPEN_AFTER_RELEASE", default="1", help="0 opts out")
    tap_cap_mb: int = option("--tap-cap-mb", env="TAP_CAP_MB", type=int, default=2048, help="FA2 tap residency cap (0 = unbounded)")
    replay_cache: str = option("--replay-cache", env="REPLAY_CACHE", default="1", help="0 = every pair replays")
    required_classes: str | None = option("--required-classes", env="REQUIRED_CLASSES", default=None)
    required_manifest: str | None = option("--required-manifest", env="REQUIRED_MANIFEST", default=None,
                                           help="an explicit manifest (default: the row's own manifest.json)")
    program_digest: str | None = option("--program-digest", env="PROGRAM_DIGEST", default=None,
                                        help="explicit Program digest (default: read from build_summary.json)")
    extra_engine_args: str | None = option("--extra-engine-args", env="EXTRA_ENGINE_ARGS", default=None, help='"k=v k=v"')
    extra_commit_args: str | None = option("--commit-args", env="EXTRA_COMMIT_ARGS", default=None, help="extra Commit flags, verbatim")
    commit_args: str | None = option("--commit-args-tail", env="COMMIT_ARGS", default=None,
                                     help="extra Commit flags appended after --program-digest, verbatim")
    replay_args: str | None = option("--replay-args", env="REPLAY_ARGS", default=None, help="extra replay flags, verbatim")
    replay_workers: int = option("--replay-workers", env="REPLAY_WORKERS", type=int, default=32, help="TP: replay workers")
    hot_engine: str = option("--hot-engine", env="HOT_ENGINE", default="0", help="1 = run the Commit on the pod's hot worker")
    commit_timeout: int | None = option("--commit-timeout", env="COMMIT_TIMEOUT", type=int, default=None,
                                        help="seconds (default scales with B and the envelope; TP: 7200)")
    vu_export: str = option("--vu-export", env="VU_EXPORT", default="1",
                            help="0 opts out: the Commit exports a sample of the row's VUs as instance sets (verity_vllm.pipeline.vu_export)")
    norm_tap: str = option("--norm-tap", env="NORM_TAP", default="0",
                           help="1 opts in, never the record: commit one scale per norm Call row (verity_vllm.query.norm_scales) through "
                                "the norm-scale tap (acquire.sources.norm_scale_source); the manifest and run root then differ from the record")


# ---------------------------------------------------------------------------------------------------------------- where it runs

@dataclass(frozen=True, kw_only=True)
class RunEnv:
    """Where a row runs: interpreters, caches and roots.  Filled by ``pipeline/cli.py`` from flags, the environment and
    the pod layout's machine-path defaults (``cli.MACHINE``); recorded as telemetry, never hashed.

    ``environ`` is the environment the CLI was started with; child processes get it plus the pins the driver sets.
    """

    py: str = option("--py", env="PY", help="the stages' interpreter")
    py312: str | None = option("--py312", env="PY312", default=None, help="the CPU stages' interpreter (default: --py)")
    sweep_dir: str = option("--sweep-dir", env="SWEEP_DIR", help="rows live in <sweep-dir>/<row>")
    hf_home: str = option("--hf-home", env="HF_HOME")
    cuda_home: str = option("--cuda-home", env="CUDA_HOME")
    cuda_bin: str = option("--cuda-bin", help="put on the stages' PATH after the interpreter's directory")
    native_collect_build: str | None = option("--native-collect-build", env="NATIVE_COLLECT_BUILD", default=None,
                                              help="collector build dir (default: the release's, else --native-collect-default)")
    native_collect_default: str = option("--native-collect-default")
    hidden_so: str | None = option("--hidden-so", env="HIDDEN_SO", default=None,
                                   help="M1 tap .so (default: the one of the row's attention implementation)")
    hidden_so_fa2: str = option("--hidden-so-fa2")
    hidden_so_fa3: str = option("--hidden-so-fa3")
    norm_tap_so: str = option("--norm-tap-so", help="the norm-scale tap op (ops/pod_norm_tap.sh), used when --norm-tap 1")
    hot_root: str = option("--hot-root", env="HOT_ROOT")
    hot_idle: str = option("--hot-idle", env="HOT_IDLE", default="900")
    timeline: str | None = option("--timeline", env="VERITY_TIMELINE", default=None,
                                  help="stage telemetry file (default: <row dir>/timeline.jsonl)")
    veritor_repo: str | None = option("--veritor-repo", env="VERITOR_REPO", default=None,
                                      help="must name the tree that runs when given (the mixed-tree refusal)")
    research_run_dir: str | None = option("--research-run-dir", env="RESEARCH_RUN_DIR", default=None)
    research_run_id: str | None = option("--research-run-id", env="RESEARCH_RUN_ID", default=None)
    guard_preserve_file: str = option("--guard-preserve-file", env="GUARD_PRESERVE_FILE")
    pod_guard: str = option("--pod-guard", help="the pod guard shell library (the tree's copy when absent)")
    environ: Mapping[str, str] = environ_snapshot()
