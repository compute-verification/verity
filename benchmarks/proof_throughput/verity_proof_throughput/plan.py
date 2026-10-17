"""Points, their identities, and the recorded runs that measure them (as ``verity_bench.jobs`` does for vLLM points).

A point is (workload, config, B, warm, runs). Its identity hashes the point, the commit and ``uv.lock``; the prover job
records it as its key param (``tool.PROVER_POINT``), and ``plan``, ``check`` and ``run`` find a point's newest passing run by
it in the evidence index. The prover point runs on a GPU node as a GPU host job (its GPU leased around the sessions); its
verifier point, the Lean verifier on its bundle, on a CPU node.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .workloads import Workload, ladder, workload

PROVER_NODE = "vy-nebius-1"
VERIFIER_NODES = ("vy-nebius-cpu-3", "vy-nebius-cpu-2")
CONFIGS = ("plain",)

PASS_QUERY = """
SELECT id
FROM attempts
WHERE validation = 'passed' AND source_commit = '{commit}' AND tool = '{tool}'
  AND EXISTS (
    SELECT 1 FROM attempt_params
    WHERE attempt = attempts.id AND key = 'identity' AND value = '{identity}'
  )
ORDER BY start_utc DESC, id DESC
LIMIT 1
""".strip()


@dataclass(frozen=True)
class Point:
    workload: str
    config: str
    B: int
    warm: int = 1
    runs: int = 3

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Pins:
    commit: str
    uv_lock_sha256: str


@dataclass(frozen=True)
class Planned:
    point: Point
    identity: str
    passing_run: str | None


def repository_root(start: str | Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "uv.lock").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"{here}: no Verity workspace root")


def pins(root: Path, commit: str | None = None) -> Pins:
    commit = commit or os.environ.get("RESEARCH_SOURCE_SHA") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return Pins(commit, hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest())


def identity(p: Pins, point: Point) -> str:
    doc = {"schema": "proof-throughput/point/v1", "commit": p.commit, "uv_lock_sha256": p.uv_lock_sha256, "point": point.to_json()}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def points(w: Workload, config: str = "plain", lo: int = 1, hi: int | None = None, warm: int = 1, runs: int = 3) -> list[Point]:
    if config not in CONFIGS:
        raise ValueError(f"config {config!r}: one of {CONFIGS} (configuration 2 is stubbed: README)")
    return [Point(w.name, config, b, warm, runs) for b in ladder(hi or w.cap, lo)]


Lookup = Callable[[str, str, str], "str | None"]


def store_completion(identity: str, commit: str, tool: str = "proof_throughput_prover") -> str | None:
    """Newest passing attempt of ``tool`` with this point identity at this commit, from the evidence index."""
    if not re.fullmatch(r"[0-9a-f]{64}", identity) or not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        raise ValueError("a completion lookup needs hexadecimal identities")
    proc = subprocess.run(["research", "data", "sql", PASS_QUERY.format(identity=identity, commit=commit, tool=tool), "--json"],
                          check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"could not query the evidence index: {proc.stderr.strip()}")
    rows = json.loads(proc.stdout or "[]")
    return str(rows[0]["id"]) if rows else None


def plan(w: Workload, p: Pins, *, config: str = "plain", lo: int = 1, hi: int | None = None, warm: int = 1, runs: int = 3,
         lookup: Lookup = store_completion) -> list[Planned]:
    return [Planned(pt, i, lookup(i, p.commit, "proof_throughput_prover"))
            for pt in points(w, config, lo, hi, warm, runs) for i in (identity(p, pt),)]


def prover_command(root: Path, item: Planned, *, mem_gb: int = 96, cpus: int = 32, max_min: int = 120) -> list[str]:
    pt = item.point
    return ["research", "run", "--queue", "--on", PROVER_NODE, "--gpu-host", "--project", "verity", "--campaign", "proof-throughput",
            "--source", str(root), "--cwd", "source", "--mem-gb", str(mem_gb), "--cpus", str(cpus), "--max-min", str(max_min),
            "--require-result", "--tool", "verity_proof_throughput.tool:PROVER_POINT",
            "--question", f"proof_throughput: {pt.workload} at B = {pt.B}, {pt.config}: VU per second and overhead against H200 peak",
            "--", "bash", "benchmarks/proof_throughput/job.sh", f"WORKLOAD={pt.workload}", f"B={pt.B}", f"WARM={pt.warm}",
            f"RUNS={pt.runs}", f"IDENTITY={item.identity}"]


def verifier_command(root: Path, bundle: str, prove_run: str, *, node: str = VERIFIER_NODES[0], mem_gb: int = 64) -> list[str]:
    return ["research", "run", "--on", node, "--project", "verity", "--campaign", "proof-throughput", "--source", str(root),
            "--cwd", "source", "--mem-gb", str(mem_gb), "--require-result", "--tool", "verity_proof_throughput.tool:VERIFIER_POINT",
            "--", "bash", "benchmarks/proof_throughput/verify.sh", f"BUNDLE={bundle}", f"PROVE_RUN={prove_run}"]


def dispatch(root: Path, items: Sequence[Planned], execute: Callable[[list[str]], int], **kw: Any) -> int:
    for item in items:
        if item.passing_run is None:
            rc = execute(prover_command(root, item, **kw))
            if rc:
                return rc
    return 0


def plan_json(w: Workload, p: Pins, items: Sequence[Planned]) -> dict[str, Any]:
    return {"schema": "proof-throughput/plan/v1", "workload": w.to_json(), "pins": asdict(p),
            "points": [{"point": i.point.to_json(), "identity": i.identity, "passing_run": i.passing_run} for i in items]}


__all__ = ["CONFIGS", "Pins", "Planned", "Point", "dispatch", "identity", "pins", "plan", "plan_json", "points",
           "prover_command", "repository_root", "store_completion", "verifier_command", "workload"]
