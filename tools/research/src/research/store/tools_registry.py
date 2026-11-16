"""name -> `dotted.module:NAME` for `research run --tool NAME`; declarations live next to the tool they describe."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Callable

from .tool import Tool

REGISTRY: dict[str, str] = {
    "tc_probe": "benchmarks.tc_probe.tool:TC_PROBE",
    "instances_hw": "benchmarks.tc_probe.tool:INSTANCES_HW",
    "tc_probe_fp4": "benchmarks.tc_probe_fp4.tool:TC_PROBE_FP4",
    "tc_price_fp4": "benchmarks.tc_probe_fp4.tool:TC_PRICE_FP4",
    "native_peak": "benchmarks.native_peak.tool:NATIVE_PEAK",
    "commit_cost": "benchmarks.commitments.tool:COMMIT_COST",
    "hiding_leaf_cost": "benchmarks.commitments.tool:HIDING_LEAF_COST",
    "serving_commit_cost": "benchmarks.commitments.tool:SERVING_COMMIT_COST",
    "flock_pure": "verity.protocols.proofs.flock.prover.tool:FLOCK_PURE",
    "flock_agreement": "verity.protocols.proofs.flock.prover.agreement.tool:FLOCK_AGREEMENT",
    "flock_class_sweep": "verity.protocols.proofs.flock.prover.tool:FLOCK_CLASS_SWEEP",
    "check": "tools.check.tool:CHECK",
    "lean_audit": "tools.verity.lean.tool:LEAN_AUDIT",
    "lean_upstream": "tools.verity.lean.tool:LEAN_UPSTREAM",
    "vllm.build": "adapters.vllm.verity_vllm.pipeline.research_tools:VLLM_BUILD",
    "vllm.plan": "adapters.vllm.verity_vllm.pipeline.research_tools:VLLM_PLAN",
    "vllm.match": "adapters.vllm.verity_vllm.pipeline.research_tools:VLLM_MATCH",
    "vllm.commit": "adapters.vllm.verity_vllm.pipeline.research_tools:VLLM_COMMIT",
    "vllm.replay": "adapters.vllm.verity_vllm.pipeline.research_tools:VLLM_REPLAY",
}


def _from_file(module: str, attr: str, source_root: Path):
    rel = Path(*module.split("."))
    for cand in (source_root / rel.with_suffix(".py"), source_root / rel / "__init__.py"):
        if cand.is_file():
            name = (f"_research_tool_{hashlib.sha256(str(cand.resolve()).encode()).hexdigest()[:16]}"
                    f"_{hashlib.sha256(cand.read_bytes()).hexdigest()[:16]}")
            mod = sys.modules.get(name)
            if mod is None:
                spec = importlib.util.spec_from_file_location(name, cand)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return getattr(mod, attr)
    raise ImportError(f"no {rel}.py under {source_root}")


MOVES = Path(__file__).with_name("moves.json")


def _moves() -> list[dict[str, str]]:
    return [m["paths"] for m in json.loads(MOVES.read_text())["moves"]]


def former_modules(module: str, moves: list[dict[str, str]] | None = None) -> list[str]:
    path, out = "/".join(module.split(".")), []
    for paths in reversed(_moves() if moves is None else moves):
        for old, new in sorted(paths.items(), key=lambda kv: -len(kv[1])):
            old, new = old.rstrip("/"), new.rstrip("/")
            if old.endswith(".py") and new.endswith(".py"):
                old, new = old[:-3], new[:-3]
            if path == new or path.startswith(new + "/"):
                path = old + path[len(new):]
                out.append(path.replace("/", "."))
                break
    return out


def former_paths(path: str, moves: list[dict[str, str]] | None = None) -> list[str]:
    path, out = path.rstrip("/"), []
    for paths in reversed(_moves() if moves is None else moves):
        for old, new in sorted(paths.items(), key=lambda kv: -len(kv[1])):
            old, new = old.rstrip("/"), new.rstrip("/")
            if path == new or path.startswith(new + "/"):
                path = old + path[len(new):]
                out.append(path)
                break
    return out


def resolve(path: str, exists: Callable[[str], bool], moves: list[dict[str, str]] | None = None) -> str | None:
    if exists(path):
        return path
    return next((p for p in former_paths(path, moves) if exists(p)), None)


def locate(spec_or_name: str, root: Path) -> tuple[str, str] | None:
    module, _, attr = REGISTRY.get(spec_or_name, spec_or_name).rpartition(":")
    if not module:
        return None

    def holds(m: str) -> bool:
        rel = Path(*m.split("."))
        return (root / rel.with_suffix(".py")).is_file() or (root / rel / "__init__.py").is_file()

    if holds(module):
        return module, attr
    return next(((m, attr) for m in former_modules(module) if holds(m)), None)


def from_tree(spec_or_name: str, root: Path) -> Tool:
    if (found := locate(spec_or_name, root)) is None:
        raise ImportError(f"{REGISTRY.get(spec_or_name, spec_or_name)}: {root} declares no such tool, under this layout or an "
                          "earlier one")
    return _from_file(*found, root)


def load(spec_or_name: str, source_root: Path | str | None = None) -> Tool:
    spec = REGISTRY.get(spec_or_name, spec_or_name)
    if ":" not in spec:
        raise LookupError(f"unknown tool {spec_or_name!r}; registered: {', '.join(sorted(REGISTRY))} (or give dotted.module:NAME)")
    module, attr = spec.rsplit(":", 1)
    root = Path(source_root or os.getcwd())
    try:
        obj = getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as e1:
        obj = None
        e2: Exception = e1
        for cand in (root, *list(root.parents)[:6]):
            try:
                obj = from_tree(spec, cand)
                break
            except (ImportError, AttributeError) as e:
                e2 = e
        if obj is None:
            raise ImportError(f"{spec}: {e1}; {e2}") from e2
    if not isinstance(obj, Tool):
        raise TypeError(f"{spec} is {type(obj).__name__}, not a research.store.tool.Tool")
    if obj.spec is None:
        obj.spec = spec
    return obj
