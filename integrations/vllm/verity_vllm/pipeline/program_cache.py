"""The derived-Program cache: a Build's derives reused across config runs on one host (docs/vllm-config-sweep-plan.md §2, "cache derived
Programs per (model, request shape) across configs").

A derive (`verity-vllm build --wrapper step|request ...`) is a pure function of its command line, the workload fields it reads, the code
and the device.  The cache key is the sha256 of exactly those: the derive's argv from `build` on without its `--out`, the workload reduced
to what a derive reads (`DERIVE_INPUTS`: the sampling policy and the sampler geometry's batch; `build.sampling_policy` /
`sampler_geometry`), the shipped tree's commit, and the device's name and compute capability.  Two configs share an entry only when all of
them agree: the same model, revision, shape, `--max-model-len`, target, launch context and sampling, on the same tree and GPU model.
Without a tree commit (`RESEARCH_SOURCE_SHA`) nothing is cached.

An entry is `<root>/<key>/program/` (a copy of the derived dir, `result.json` ok) with `ENTRY.json` beside it, published by one rename; a
reader sees a whole entry or none.  Restoring copies the entry into the row (later stages write beside the derived Programs, so nothing is
shared by link).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from verity_vllm.config import workload_file

SCHEMA = "verity-vllm/program-cache-entry/v1"
#: the workload's `sweep` fields a derive reads
DERIVE_INPUTS = ("sampling", "sampling_params", "batch", "sampling_seeds")


def derive_inputs(workload: str | Path) -> dict[str, Any]:
    wl = json.loads(workload_file(workload).read_text())
    sw = wl.get("sweep") or {}
    return {**{k: sw.get(k) for k in DERIVE_INPUTS}, "requests": len(wl.get("requests") or [])}


def key(argv: list[str], *, workload: str | Path, tree: str | None, device: dict[str, Any] | None) -> str | None:
    """The cache key of one derive, None when it cannot be keyed (no tree commit, no device, or not a `build` command)."""
    if not tree or not device or "build" not in argv:
        return None
    args = list(argv[argv.index("build"):])
    if "--out" in args:
        i = args.index("--out")
        del args[i:i + 2]
    if "--workload" in args:
        i = args.index("--workload")
        args[i + 1] = derive_inputs(args[i + 1] if not workload else workload)
    doc = {"argv": args, "tree": tree, "device": device}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _ok(program: Path) -> bool:
    try:
        return json.loads((program / "result.json").read_text()).get("ok") is True
    except (OSError, ValueError, AttributeError):
        return False


def lookup(root: str | Path, k: str) -> Path | None:
    e = Path(root) / k
    return e if (e / "ENTRY.json").is_file() and _ok(e / "program") else None


def restore(entry: Path, out: str | Path) -> None:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(entry / "program", out)


def publish(root: str | Path, k: str, derived: str | Path, meta: dict[str, Any]) -> Path | None:
    """Copy a derived dir whose derive succeeded into the cache under `k` (one rename); an entry already there is kept."""
    root, derived = Path(root), Path(derived)
    if not _ok(derived):
        return None
    final = root / k
    if (final / "ENTRY.json").is_file():
        return final
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".{k}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(derived, tmp / "program")
    (tmp / "ENTRY.json").write_text(json.dumps({"schema": SCHEMA, "key": k, **meta}, indent=1, sort_keys=True, default=str) + "\n")
    try:
        os.rename(tmp, final)
    except OSError:                                    # another job published the same key first: theirs stands
        shutil.rmtree(tmp, ignore_errors=True)
    return final
