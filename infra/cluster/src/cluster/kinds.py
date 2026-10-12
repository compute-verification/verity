"""The job-kind registry: `kinds/<owner>.toml` beside the descriptions, one file per owner lane, read from the tree the
submitter ships (never a node-local copy), so a job's kind is resolved at the commit its run records.

    [kinds.lean-audit]
    phase = "cpu-m"                 # the shape menu below
    max_wall_min = 20
    question = "Do core's Lean proofs still check, with only the allowed axioms?"
    outputs = ["the audit report"]
    restart = "idempotent"          # idempotent | requeue | never
    next = "some-kind"              # optional: the kind its declared outputs go to
    disk_gb = 20                    # optional: the disk it writes (DISK_GB when absent); the machine holds it near full

A registered kind supplies a job's defaults: its shape's GPUs, CPUs and memory, its `max_wall_min`, its question. The rules
on kinds are warnings for now, never refusals (infra's hold, 30 Sep): an unregistered kind, a job with no question, a shape
off the menu, a GPU kind over 60 minutes or not preemptible, no declared outputs.
"""
from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SHAPES: dict[str, dict[str, int | bool]] = {
    "cpu-s": {"gpus": 0, "cpus": 2, "memory_gib": 8},
    "cpu-m": {"gpus": 0, "cpus": 8, "memory_gib": 32},
    "cpu-l": {"gpus": 0, "cpus": 32, "memory_gib": 128},
    "gpu1": {"gpus": 1, "cpus": 8, "memory_gib": 64},
    "gpu2": {"gpus": 2, "cpus": 16, "memory_gib": 128},
    "gpu8-timed": {"gpus": 8, "cpus": 0, "memory_gib": 1024, "quiet": True},
}
RESTART = ("idempotent", "requeue", "never")
GPU_CHUNK_MIN = 60
DISK_GB = 20


@dataclass(frozen=True)
class Kind:
    name: str
    owner: str
    phase: str
    max_wall_min: int
    question: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    restart: str = "requeue"
    next: str | None = None
    disk_gb: float | None = None
    source: dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def shape(self) -> dict[str, int | bool]:
        return SHAPES[self.phase]


def kinds_dir(description: str | Path) -> Path:
    return Path(description).resolve().parent.parent / "kinds"


def load(directory: Path) -> dict[str, Kind]:
    """Every kind under `directory`; ValueError naming the file for a malformed kind or a name two owners register."""
    out: dict[str, Kind] = {}
    for path in sorted(Path(directory).glob("*.toml")) if Path(directory).is_dir() else []:
        raw = path.read_bytes()
        source = {"file": f"kinds/{path.name}", "sha256": hashlib.sha256(raw).hexdigest()}
        for name, d in tomllib.loads(raw.decode()).get("kinds", {}).items():
            where = f"{path.name}: kind {name}"
            if name in out:
                raise ValueError(f"{where} is already {out[name].owner}'s")
            if d.get("phase") not in SHAPES:
                raise ValueError(f"{where}: phase must be one of {', '.join(SHAPES)}")
            if not isinstance(d.get("max_wall_min"), int) or d["max_wall_min"] <= 0:
                raise ValueError(f"{where}: max_wall_min must be a positive integer")
            if "disk_gb" in d and (not isinstance(d["disk_gb"], (int, float)) or d["disk_gb"] < 0):
                raise ValueError(f"{where}: disk_gb must be a number of GB, 0 or more")
            if d.get("restart", "requeue") not in RESTART:
                raise ValueError(f"{where}: restart must be one of {', '.join(RESTART)}")
            out[name] = Kind(name=name, owner=path.stem, phase=d["phase"], max_wall_min=d["max_wall_min"],
                             question=str(d.get("question", "")), inputs=tuple(d.get("inputs", ())),
                             outputs=tuple(d.get("outputs", ())), restart=d.get("restart", "requeue"), next=d.get("next"),
                             disk_gb=d.get("disk_gb"), source=source)
    for k in out.values():
        if k.next is not None and k.next not in out:
            raise ValueError(f"kinds/{k.owner}.toml: kind {k.name}: next names {k.next!r}, which isn't registered")
    return out


def warnings(kind: Kind | None, name: str, *, question: str, gpus: int, cpus: int, memory_gib: int, max_s: int,
             preempt: str) -> list[str]:
    """The held rules a job breaks, as warnings."""
    out = []
    if kind is None:
        out.append(f"kind {name!r} isn't registered (infra/cluster/kinds/<owner>.toml)" if name != "adhoc"
                   else "an adhoc job: register its kind in infra/cluster/kinds/<owner>.toml")
    if not question:
        out.append("no research question (--question, or the kind's `question`)")
    if kind is None:
        return out
    shape = kind.shape
    if (gpus, cpus, memory_gib) != (shape["gpus"], shape["cpus"], shape["memory_gib"]):
        out.append(f"asks for {gpus} GPU(s), {cpus} CPU(s), {memory_gib} GiB, not its phase {kind.phase}'s "
                   f"{shape['gpus']}, {shape['cpus']}, {shape['memory_gib']} GiB")
    if max_s > kind.max_wall_min * 60:
        out.append(f"asks for {max_s // 60} min, over its max_wall_min of {kind.max_wall_min}")
    if gpus and not shape.get("quiet"):
        if max_s > GPU_CHUNK_MIN * 60:
            out.append(f"a GPU job over {GPU_CHUNK_MIN} min: split it into chunks")
        if preempt != "requeue":
            out.append("a GPU kind that isn't preemptible")
    if not kind.outputs:
        out.append("declares no outputs")
    return out


__all__ = ["SHAPES", "RESTART", "DISK_GB", "Kind", "kinds_dir", "load", "warnings"]
