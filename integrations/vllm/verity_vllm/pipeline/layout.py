"""Where a Build's component dirs are, once the Build has moved (`vllm-build/v1` layout 1.1).

`build_workload/workload_program.json` (GP-01, `verity_vllm.pipeline.global_program`) records the request-wrapper
component dirs it composed under `inputs.derived["LP,T"]` and the manifest under `inputs.manifest` -- as the ABSOLUTE paths of the
build machine (`/workspace/cp/<sweep>/<row>/build_request_LP*_T*`).  A Build fetched from the store (`research data fetch`) lands
somewhere else, so those paths dangle unless the tree is symlinked back to the original path (r20 did that).

Layout 1.1 adds a `layout` block beside them:

    "layout": {"schema": "verity/workload-program/v1/layout", "version": "1.1",
               "artifact_root": "/workspace/cp/sweep/row",                 # where the root was at build time (informational)
               "manifest": "workload.json" | null,                          # POSIX paths relative to the artifact root ...
               "derived": {"3,2": "build_request_LP3_T2", ...}}             # ... or null when the path lies outside the root

The artifact root is the directory the `vllm-build/v1` payload tree is rooted at: the one holding `build_workload/` and the
`build_request*/` dirs (`research_outputs.BUILD_FILES` are relative to it).  `inputs.derived` / `inputs.manifest` are still written
(layout 1.0 readers -- `global_match`, the tp2 bench, `commit/*` -- read them unchanged); the digest-bearing parts of the artifact
(descriptor, `workload_digest`, `schema`) do not move.

Readers call `component_dir(wp, LP, T, record_dir)` / `manifest_path(wp, record_dir)` with `record_dir` = the fetched tree's root
(`Path(<...>/build_workload/workload_program.json).parent.parent`, `record_dir_of`).  Resolution order:

    1. layout 1.1 relative path under `record_dir`, when it exists      (the artifact's own component, wherever the tree is now)
    2. the legacy absolute path, when it exists                          (the build machine; a 1.0 record laid out at its original path)
    3. `record_dir / basename(absolute)`, when it exists                 (a 1.0 record relocated whole: the dir names travelled with it)
    4. None                                                              (the caller refuses by name; nothing is inferred from conventions)

Stdlib only."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any

LAYOUT_SCHEMA = "verity/workload-program/v1/layout"
LAYOUT_VERSION = "1.1"          # 1.0 = no `layout` block: absolute paths only
LAYOUT_VERSIONS = ("1.0", "1.1")


def relative_to_root(path: str | os.PathLike[str] | None, artifact_root: str | os.PathLike[str]) -> str | None:
    """`path` as a POSIX path relative to `artifact_root`, or None when it is not inside the root (or is None)."""
    if path is None:
        return None
    p, root = os.path.abspath(os.fspath(path)), os.path.abspath(os.fspath(artifact_root))
    rel = os.path.relpath(p, root)
    if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return None
    return PurePosixPath(*Path(rel).parts).as_posix()


def layout_block(artifact_root: str | os.PathLike[str], manifest: str | os.PathLike[str] | None, derived: dict[str, str]) -> dict[str, Any]:
    """The `layout` block GP-01 writes (module docstring)."""
    return {"schema": LAYOUT_SCHEMA, "version": LAYOUT_VERSION, "artifact_root": os.path.abspath(os.fspath(artifact_root)),
            "manifest": relative_to_root(manifest, artifact_root),
            "derived": {k: relative_to_root(v, artifact_root) for k, v in derived.items()}}


def layout_version(wp: dict[str, Any] | None) -> str:
    lay = (wp or {}).get("layout")
    return str(lay.get("version") or "1.0") if isinstance(lay, dict) else "1.0"


def record_dir_of(workload_program_path: str | os.PathLike[str]) -> Path:
    """The artifact root a `.../build_workload/workload_program.json` belongs to (the dir holding `build_workload/`)."""
    return Path(workload_program_path).resolve().parent.parent


def _resolve(relative: str | None, absolute: str | None, record_dir: str | os.PathLike[str] | None) -> Path | None:
    rd = Path(record_dir) if record_dir is not None else None
    if rd is not None and relative:
        cand = rd / Path(*PurePosixPath(relative).parts)
        if cand.exists():
            return cand
    if absolute:
        ap = Path(absolute)
        if ap.exists():
            return ap
        if rd is not None:
            cand = rd / ap.name
            if cand.exists():
                return cand
    return None


def component_dir(wp: dict[str, Any], LP: int, T: int, record_dir: str | os.PathLike[str] | None) -> Path | None:
    """The composed component dir for shape (LP, T), or None when the Build declared none / it is nowhere to be found (module
    docstring order).  `declared_component(wp, LP, T)` says whether the Build declared the shape at all."""
    key = f"{int(LP)},{int(T)}"
    lay = wp.get("layout") if isinstance(wp.get("layout"), dict) else {}
    rel = ((lay.get("derived") or {}) if isinstance(lay.get("derived"), dict) else {}).get(key)
    absolute = ((wp.get("inputs") or {}).get("derived") or {}).get(key)
    return _resolve(rel, absolute, record_dir)


def declared_component(wp: dict[str, Any], LP: int, T: int) -> str | None:
    """What the Build recorded for shape (LP, T): the absolute path (1.0/1.1), else the relative one (a 1.1 record whose absolute was
    dropped), else None (the shape was never composed: refuse by name)."""
    key = f"{int(LP)},{int(T)}"
    absolute = ((wp.get("inputs") or {}).get("derived") or {}).get(key)
    if absolute:
        return str(absolute)
    lay = wp.get("layout") if isinstance(wp.get("layout"), dict) else {}
    rel = ((lay.get("derived") or {}) if isinstance(lay.get("derived"), dict) else {}).get(key)
    return str(rel) if rel else None


def component_dirs(wp: dict[str, Any], record_dir: str | os.PathLike[str] | None) -> dict[str, Path | None]:
    """Every declared shape -> its resolved dir (None where unresolvable)."""
    keys = set(((wp.get("inputs") or {}).get("derived") or {}))
    lay = wp.get("layout") if isinstance(wp.get("layout"), dict) else {}
    keys |= set((lay.get("derived") or {}) if isinstance(lay.get("derived"), dict) else {})
    out: dict[str, Path | None] = {}
    for k in sorted(keys):
        lp, t = k.split(",")
        out[k] = component_dir(wp, int(lp), int(t), record_dir)
    return out


def manifest_path(wp: dict[str, Any], record_dir: str | os.PathLike[str] | None) -> Path | None:
    lay = wp.get("layout") if isinstance(wp.get("layout"), dict) else {}
    return _resolve(lay.get("manifest"), (wp.get("inputs") or {}).get("manifest"), record_dir)


__all__ = ["LAYOUT_SCHEMA", "LAYOUT_VERSION", "LAYOUT_VERSIONS", "relative_to_root", "layout_block", "layout_version", "record_dir_of",
           "component_dir", "declared_component", "component_dirs", "manifest_path"]
