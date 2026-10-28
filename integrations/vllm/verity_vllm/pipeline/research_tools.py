"""`research` Tool declarations for the vLLM integration sweep stages.

Five Tools — ``vllm.build``, ``vllm.plan`` (the Commit's plan, on CPU, before the GPU), ``vllm.match``,
``vllm.commit``, ``vllm.replay`` (a deferred Commit's C2 replay, on CPU) — one per stage of a sweep row.  Each stage runs as one ``research run --tool vllm.<stage>
-- verity-vllm row stage <stage> ...`` Attempt and publishes typed Artifacts
(``verity_vllm.pipeline.research`` writes the ``outputs.json`` the publisher
consumes).

Rules encoded here (Phase 5a of the v2 plan):

* ``key_params`` is the row configuration — model id, dtype, hw family, tp,
  batch, input/output lengths, arrival mix, sampling, execution mode, the
  verification query id (default the query of record, ``Q_word_v1{X=16,W=32,R=no-recompute}``) — plus the
  stage-specific flags that change the stage's *outputs*.  RETAIN mode,
  window sizing, GPU utilisation, hot-engine toggles etc. are footprint /
  schedule knobs: recorded as params, **not** key params.
* ``key_conditions`` is ``hw.family`` plus the torch and vllm pins of the
  workload interpreter (``$PY``, default ``/workspace/venv312/bin/python``),
  probed via ``importlib.metadata`` so nothing heavy is imported.
* ``closure`` is the integration's import closure: ``integrations/vllm/**``
  plus every directory that ships Verity core (``research.pythonpath.package_dirs``:
  ``verity``, or each member's part of the namespace).
  ``source_identity.code_identity`` is the one code identity over it (the
  hot Commit worker's key, the TP Commit's tree of record).

This module imports nothing beyond the standard library, ``research.store.tool``
and ``research.pythonpath``, so it is loadable from any interpreter that can see
``tools/research/src`` (``research`` loads it by dotted path or, failing
that, by file — see ``research.store.tools_registry``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import PurePosixPath
from typing import Any

from research.pythonpath import package_dirs, workspace_root
from research.store.tool import Tool

DEFAULT_QUERY = "Q_word_v1{X=16,W=32,R=no-recompute}"
DEFAULT_PY = "/workspace/venv312/bin/python"

COMMAND = ["verity-vllm", "row", "stage"]
_CLI_MODULE = "verity_vllm.pipeline.cli"
_SCRIPTS = ("run_row_v2.sh", "row_pod.sh", "tp_stage.sh")   # the shell runners of earlier records (same grammar as `row stage` / `row run`)

#: Each directory of this file's tree that ships Verity core, relative to the repository root.
CORE_DIRS = package_dirs(workspace_root(__file__), "verity")
#: The code the integration imports besides its own: `verity`, with the catalog, the kernels and the verifiers in it.
CODE_DIRS = [*CORE_DIRS]
CLOSURE = ["integrations/vllm/**", *(f"{d}/**" for d in CODE_DIRS)]

# Row id grammar (see verity_vllm/ops/README):
#   <model>__<dtype>__<hw>__tp<N>__b<N>__i<N>__o<N>__<arrivals>__<sampling>__<exec>
_ROW_FIELDS = ("model_tag", "dtype", "hw", "tp", "batch", "input", "output", "arrivals", "sampling", "exec")
_ROW_INT_PREFIX = {"tp": "tp", "batch": "b", "input": "i", "output": "o"}

# Row configuration shared by all three stages.
ROW_KEY_PARAMS = [
    "stage",
    "row",
    "role",
    "model",
    "revision",
    *_ROW_FIELDS,
    "query",
    "lp",
    "t",
]

# Stage-specific flags that change the stage's outputs (and their defaults,
# so an absent flag hashes identically to its explicit default).
STAGE_KEY_FLAGS: dict[str, dict[str, Any]] = {
    "build": {},
    "plan": {},
    "match": {
        "snapshot_steps": "row-default",  # the row's default: "0,1" greedy / "all" stochastic
        "snapshot_max_bytes": None,
        "within_step": "dag",
    },
    "commit": {
        "pairs": "3",
        "required_classes": None,
        "replay_args": None,
        "extra_engine_args": None,
        "commit_args": None,
    },
    "replay": {},
}

# Output-changing flags keyed only when set (off, they leave the derivation of every run without them as it was).
KEY_FLAGS_WHEN_SET: dict[str, tuple[str, ...]] = {
    "build": ("build_programs_only",),   # a TP Build without its manifest
    "commit": ("replay_deferred",),   # the Commit seals a replay bundle and leaves its verdict pending
}

# Recorded but NOT key: footprint / schedule / implementation knobs.
NON_KEY_FLAGS = (
    "retain",
    "sweep_dir",
    "build_jobs",
    "build_relabel",
    "match_impl",
    "match_phase",
    "gpu_util",
    "window_mb",
    "window_slots",
    "tap_cap_mb",
    "open_after_release",
    "replay_cache",
    "hot_engine",
    "py",
)

PRODUCES: dict[str, list[str]] = {
    "build": ["vllm-build/v1", "execution-evidence/v1", "run-files/v1"],
    "plan": ["vllm-commit-plan/v1", "execution-evidence/v1", "run-files/v1"],
    "match": ["vllm-match-record/v1", "execution-evidence/v1", "run-files/v1"],
    "commit": ["vllm-verdict/v1", "execution-evidence/v1", "run-files/v1"],
    "replay": ["vllm-verdict/v1", "execution-evidence/v1", "run-files/v1"],
}


def row_config(row: str) -> dict[str, Any]:
    """Parse a sweep row id into its configuration fields.

    Tolerant: unknown / missing fields are ``None`` so a malformed id still
    yields a (distinct) derivation instead of aborting store preparation.
    """
    parts = row.split("__")
    out: dict[str, Any] = {k: None for k in _ROW_FIELDS}
    if len(parts) != len(_ROW_FIELDS):
        out["model_tag"] = parts[0] if parts else None
        return out
    for k, v in zip(_ROW_FIELDS, parts):
        pfx = _ROW_INT_PREFIX.get(k)
        if pfx is not None:
            if v.startswith(pfx) and v[len(pfx):].isdigit():
                out[k] = int(v[len(pfx):])
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _find_script(argv: list[str]) -> tuple[int, str] | None:
    """Where the row's arguments start: after ``verity-vllm row`` / ``-m verity_vllm.pipeline.cli row`` (name ``row``) or
    after one of the earlier shell runners (its file name)."""
    for i, tok in enumerate(argv):
        name = PurePosixPath(tok.replace("\\", "/")).name
        if (name == "verity-vllm" or (tok == _CLI_MODULE and i > 0 and argv[i - 1] == "-m")) and argv[i + 1 : i + 2] == ["row"]:
            return i + 1, "row"
        if name in _SCRIPTS:
            return i, name
    return None


def _parse_flags(toks: list[str]) -> dict[str, Any]:
    """`--k v` / `--k=v` / `--flag` -> {k: v|True}; dashes become underscores."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            out.setdefault("_positional", []).append(t)
            i += 1
            continue
        key, sep, val = t[2:].partition("=")
        key = key.replace("-", "_")
        if sep:
            out[key] = val
        elif i + 1 < len(toks) and (not toks[i + 1].startswith("--") or " " in toks[i + 1]):
            # a value that is itself a flag list (`--commit-args "--bounded-staging --retain-exclude hidden_m1"`) has spaces; a flag never does
            out[key] = toks[i + 1]
            i += 1
        else:
            out[key] = True
        i += 1
    return out


def parse(argv: list[str], *, stage: str) -> dict[str, Any]:
    """Extract params from a row argv.

    Accepts::

        verity-vllm row stage <stage> <row> <ROLE> <hf-repo> <rev> [--flag v ...]
        python -m verity_vllm.pipeline.cli row stage <stage> <row> <ROLE> <hf-repo> <rev> [--flag v ...]
        verity-vllm row run <row> <ROLE> <hf-repo> <rev> [--stages S] [--flag v ...]

    and the argv of the earlier shell runners (``run_row_v2.sh stage <stage> ...``, ``row_pod.sh <row> ... [stages]``,
    ``tp_stage.sh <row> ... [stages]``), which records in the store carry.
    """
    found = _find_script(argv)
    if found is None:
        rest = list(argv)
        argv_stage = None
    else:
        i, name = found
        rest = list(argv[i + 1 :])
        argv_stage = None
        if name == "row" and rest[:1] == ["run"]:
            rest = rest[1:]
        elif name in ("row", "run_row_v2.sh"):
            if rest[:1] == ["stage"] and len(rest) >= 2:
                argv_stage = rest[1]
                rest = rest[2:]
    positional: list[str] = []
    while rest and not rest[0].startswith("--"):
        positional.append(rest.pop(0))
    flags = _parse_flags(rest)
    if found is not None and found[1] == "row" and argv_stage is None:
        argv_stage = flags.pop("stages", None)

    row = positional[0] if len(positional) > 0 else None
    role = positional[1] if len(positional) > 1 else None
    model = positional[2] if len(positional) > 2 else None
    revision = positional[3] if len(positional) > 3 else None
    if argv_stage is None and found is not None and found[1] in ("row_pod.sh", "tp_stage.sh") and len(positional) > 4:
        argv_stage = positional[4]

    out: dict[str, Any] = {
        "stage": argv_stage or stage,
        "row": row,
        "role": role,
        "model": model,
        "revision": revision,
    }
    out.update(row_config(row) if row else {k: None for k in _ROW_FIELDS})
    out["query"] = flags.pop("query", DEFAULT_QUERY)
    out["lp"] = flags.pop("lp", None)
    out["t"] = flags.pop("t", None)
    for k, default in STAGE_KEY_FLAGS.get(stage, {}).items():
        out[k] = flags.pop(k, default)
    for k in KEY_FLAGS_WHEN_SET.get(stage, ()):
        v = flags.pop(k, None)
        if v not in (None, "0", False):
            out[k] = v
    for k in NON_KEY_FLAGS:
        if k in flags:
            out[k] = flags.pop(k)
    # Anything else the row accepted is recorded verbatim (non-key).
    for k, v in flags.items():
        out.setdefault(k, v)
    return out


def _probe_pins(env: dict[str, Any] | None = None) -> dict[str, Any]:
    """torch / vllm versions as seen by the workload interpreter.

    Uses ``importlib.metadata`` so vLLM is never imported.  ``None`` when the
    interpreter is absent (e.g. on the control laptop) — such a derivation
    legitimately differs from one probed on a pod.
    """
    py = os.environ.get("PY") or DEFAULT_PY
    out: dict[str, Any] = {"torch": None, "vllm": None, "py": py}
    if not os.path.exists(py):
        return out
    code = (
        "import importlib.metadata as m\n"
        "for d in ('torch','vllm'):\n"
        "    try: print(d, m.version(d))\n"
        "    except Exception: print(d, '')\n"
    )
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return out
    for line in r.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] in ("torch", "vllm") and parts[1].strip():
            out[parts[0]] = parts[1].strip()
    return out


def _tool(stage: str) -> Tool:
    name = f"vllm.{stage}"
    return Tool(
        name=name,
        version="1",
        command=[*COMMAND, stage],
        closure=CLOSURE,
        key_params=[*ROW_KEY_PARAMS, *STAGE_KEY_FLAGS[stage].keys(), *KEY_FLAGS_WHEN_SET.get(stage, ())],
        key_conditions=["hw.family", f"tool.{name}.torch", f"tool.{name}.vllm"],
        param_artifacts=[],
        produces=PRODUCES[stage],
        parse=lambda argv, _s=stage: parse(argv, stage=_s),
        conditions=_probe_pins,
    )


VLLM_BUILD = _tool("build")
VLLM_PLAN = _tool("plan")
VLLM_MATCH = _tool("match")
VLLM_COMMIT = _tool("commit")
VLLM_REPLAY = _tool("replay")

TOOLS = {t.name: t for t in (VLLM_BUILD, VLLM_PLAN, VLLM_MATCH, VLLM_COMMIT, VLLM_REPLAY)}

__all__ = [
    "CLOSURE",
    "COMMAND",
    "CORE_DIRS",
    "DEFAULT_QUERY",
    "ROW_KEY_PARAMS",
    "STAGE_KEY_FLAGS",
    "KEY_FLAGS_WHEN_SET",
    "NON_KEY_FLAGS",
    "PRODUCES",
    "VLLM_BUILD",
    "VLLM_PLAN",
    "VLLM_MATCH",
    "VLLM_COMMIT",
    "VLLM_REPLAY",
    "TOOLS",
    "parse",
    "row_config",
]
