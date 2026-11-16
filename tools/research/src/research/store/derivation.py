"""Derivation id, reuse key, scratch dirs (store README §6.4, §6.5).

    derivation_doc = {"tool": {"name", "version"}, "inputs": {name: art_id}, "params": {key params, content-resolved},
                      "conditions": {k: conditions[k] for k in key_conditions}}
    drv_id    = "drv:" + canon.sha256(derivation_doc)                       # no source revision inside
    reuse_key = canon.sha256({"derivation": drv_id, "source": {"tree": source.tree, "lockfiles": {...}}})   None for dirty trees
                (and, set by the callers, for a run with no declared Tool: its derivation's params are just its redacted argv)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _tool_identity(tool: Any) -> dict[str, Any]:
    if tool is None:
        return {"name": None, "version": None}
    if isinstance(tool, str):
        return {"name": tool, "version": None}
    if isinstance(tool, dict):
        return {"name": tool.get("name"), "version": tool.get("version")}
    return {"name": tool.name, "version": tool.version}


def derivation_doc(tool: Any, inputs: dict[str, str] | None, key_params_resolved: dict[str, Any] | None, conditions: dict[str, Any] | None,
                   key_conditions: list[str] | None) -> dict[str, Any]:
    conds = conditions or {}
    return {"tool": _tool_identity(tool), "inputs": dict(inputs or {}), "params": dict(key_params_resolved or {}),
            "conditions": {k: conds.get(k) for k in (key_conditions or [])}}


def derivation_id(tool: Any, inputs: dict[str, str] | None, key_params_resolved: dict[str, Any] | None, conditions: dict[str, Any] | None,
                  key_conditions: list[str] | None) -> str:
    from .canon import sha256

    return "drv:" + sha256(derivation_doc(tool, inputs, key_params_resolved, conditions, key_conditions))


def _source(source: dict[str, Any]) -> dict[str, Any]:
    """Accept the flat conditions dict (`source.tree`, `source.lockfiles.<name>`) or the nested `source` block."""
    if "source.tree" in source or "source.dirty" in source or any(k.startswith("source.") for k in source):
        from ..env import source_identity

        return source_identity(source)
    return {"tree": source.get("tree"), "dirty": source.get("dirty"), "lockfiles": dict(source.get("lockfiles") or {})}


def reuse_key(drv_id: str, source_conditions: dict[str, Any] | None) -> str | None:
    """None unless the tree is known AND known clean: exact source + exact request; false misses, never false hits."""
    from .canon import sha256

    src = _source(source_conditions or {})
    if src.get("dirty") is not False or not src.get("tree"):
        return None
    return sha256({"derivation": drv_id, "source": {"tree": src["tree"], "lockfiles": dict(sorted(src["lockfiles"].items()))}})


def scratch_dir(store_root: Path | str, reuse_key: str | None, run_id: str, name: str) -> Path:
    """<store>/scratch/<reuse_key>/<name>/  (no reuse key -- a dirty tree, or no declared Tool: <store>/scratch/run-<run id>/<name>/), created.

    Keyed by reuse_key ON PURPOSE: every commit gets a cold cache and the recompile cost is the price of the safe bias.  Do not
    "fix" this to the derivation id or a source-independent key without a measured reason and a per-tool `cache_scope`
    declaration -- the Triton cache incident (W=31 binaries served to a W=22 tree) is why."""
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"scratch name must be a plain segment, got {name!r}")
    d = Path(store_root) / "scratch" / (reuse_key if reuse_key else f"run-{run_id}") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def scratch_env(name: str, path: Path) -> dict[str, str]:
    """The variable `--scratch NAME` exports: TRITON_CACHE_DIR for `triton`, RESEARCH_SCRATCH_<NAME> otherwise."""
    if name == "triton":
        return {"TRITON_CACHE_DIR": str(path)}
    return {"RESEARCH_SCRATCH_" + "".join(c if c.isalnum() else "_" for c in name).upper(): str(path)}
