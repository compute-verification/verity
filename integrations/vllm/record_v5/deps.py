"""[A9 contract 2026-09-10] Where the runner's dependencies on concurrent agents' modules come from, recorded per run.

The runner needs, beyond mainline (A2 ``veritor.interface.expected`` / ``verity_vllm_adapter.obligations.output_copy_checks`` and A7
``e2e.certificate_v2`` are merged at ``b47e0d85``):

* **A3** (`contract/a3-sequence`): ``e2e.protocol_instance`` with the ``epoch`` section, ``e2e.partition_sample.stage2(exemptions=...)``,
  ``e2e.verifier_v3.{driver,steps,reverify}``, ``verity_vllm_adapter.{selective_prover,response_record}``;
* **A5/6** (`contract/a56-coverage`): ``e2e.coverage_v1`` and ``data/contract/a56-coverage/coverage.json``.

Until those branches are merged, a run may point ``VERITOR_DEP_A3`` / ``VERITOR_DEP_A56`` at read-only exports of the branches
(``git archive contract/<branch> vllm-poc out tests | tar -x -C <dir>``); :func:`install` puts the export's ``vllm-poc`` FIRST on
``sys.path`` so its modules shadow the worktree's older copies, and :func:`describe` records which source served each module (path,
branch, commit as given by the caller -- the runner cannot know a commit from a directory, so the export directory must carry a
``DEP.json`` ``{"branch", "commit"}`` written by whoever exported it).  A run REFUSES (``DependencyError``) if the A3 modules are
missing the contract features (no ``epoch_section``, no ``stage2(exemptions=...)``): there is no legacy fallback for the driver.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

from record_v5 import HERE, POC, ROOT, add_paths

ENV_A3 = "VERITOR_DEP_A3"
ENV_A56 = "VERITOR_DEP_A56"


class DependencyError(RuntimeError):
    code = "E_DEPENDENCY"


def _export_dir(env: str) -> Path | None:
    v = os.environ.get(env)
    if not v:
        return None
    p = Path(v)
    if not (p / "vllm-poc").is_dir():
        raise DependencyError(f"{env}={v}: no vllm-poc/ inside (expected a `git archive <branch> vllm-poc out tests` export)")
    return p


def install() -> dict[str, Any]:
    """Arrange ``sys.path`` (exports first, then this repo) and return the dependency record (also written by :func:`write`)."""

    record: dict[str, Any] = {"schema": "veritor/a9/deps/v1", "worktree": str(ROOT), "sources": {}}
    for name, env in (("a56", ENV_A56), ("a3", ENV_A3)):
        d = _export_dir(env)
        if d is None:
            record["sources"][name] = {"from": "worktree", "path": str(POC), "note": f"{env} not set: the worktree's copy is used (post-merge mainline expected)"}
            continue
        meta = {}
        if (d / "DEP.json").exists():
            meta = json.loads((d / "DEP.json").read_text())
        sys.path.insert(0, str(d / "vllm-poc"))
        record["sources"][name] = {"from": "export", "path": str(d), **{k: meta.get(k) for k in ("branch", "commit")}}
    add_paths()
    return record


def check_a3() -> dict[str, Any]:
    """The A3 features the runner relies on must be present; otherwise refuse (no legacy driver fallback)."""

    pi = importlib.import_module("e2e.protocol_instance")
    ps = importlib.import_module("e2e.partition_sample")
    problems = []
    if not hasattr(pi, "epoch_section"):
        problems.append("e2e.protocol_instance.epoch_section missing (A3 R16 schema)")
    if "exemptions" not in inspect.signature(ps.stage2).parameters:
        problems.append("e2e.partition_sample.stage2 has no `exemptions` parameter (A3 R6/R8 sampler)")
    if "seed_file" not in inspect.signature(ps.stage1).parameters:
        problems.append("e2e.partition_sample.stage1 has no `seed_file` parameter (A3 R7 seed custody)")
    try:
        importlib.import_module("e2e.verifier_v3.steps")
        importlib.import_module("verity_vllm_adapter.response_record")
    except ImportError as e:
        problems.append(f"A3 module missing: {e}")
    if problems:
        raise DependencyError(f"{DependencyError.code}_A3: " + "; ".join(problems) + " -- set VERITOR_DEP_A3 to a read-only export of contract/a3-sequence or merge it")
    return {"protocol_instance": pi.__file__, "partition_sample": ps.__file__}


def check_a56() -> dict[str, Any]:
    try:
        cov = importlib.import_module("e2e.coverage_v1.coverage")
    except ImportError as e:
        raise DependencyError(f"{DependencyError.code}_A56: e2e.coverage_v1 missing ({e}) -- set VERITOR_DEP_A56 or merge contract/a56-coverage") from e
    return {"coverage_v1": cov.__file__}


def coverage_json_path() -> Path | None:
    """A5/6's ``coverage.json``: from the export if given, else from this worktree; None if absent (the caller labels its fallback)."""

    d = _export_dir(ENV_A56)
    for base in ([d] if d else []) + [ROOT]:
        p = base / "data" / "contract" / "a56-coverage" / "coverage.json"
        if p.exists():
            return p
    return None


def write(record_dir: Path, record: dict[str, Any]) -> Path:
    record = {**record, "a3": check_a3(), "a56": check_a56(), "coverage_json": str(coverage_json_path()) if coverage_json_path() else None}
    p = Path(record_dir) / "deps.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return p


__all__ = ["DependencyError", "install", "check_a3", "check_a56", "coverage_json_path", "write", "ENV_A3", "ENV_A56", "HERE"]
