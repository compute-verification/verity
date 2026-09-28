"""name -> `dotted.module:NAME` for `research run --tool NAME` (store README §6.1).  Keep it tiny; declarations live next to
the tool they describe.  `load` also accepts the spec form directly and, when the module is not importable (a shipped `git
archive` cwd, or a console-script sys.path without the repo root), loads `<source_root>/<module path>.py` by file, or, in a
tree from before a layout move, the file at the path the move renamed (`moves.json`)."""
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
    "bench_vu": "archive.direct.ligero.tool:BENCH_VU",
    "ligero_verify_batch": "archive.direct.ligero.tool:LIGERO_VERIFY_BATCH",
    "a_gpu_prove": "archive.gkr.gpu.tool:A_GPU_PROVE",
    "bench_vu_fp8": "archive.direct.ligero.fp8.tool:BENCH_VU_FP8",
    "bench_vu_fp4": "archive.direct.ligero.fp4.tool:BENCH_VU_FP4",
    "native_peak": "benchmarks.native_peak.tool:NATIVE_PEAK",
    "commit_cost": "benchmarks.commitments.tool:COMMIT_COST",
    "hiding_leaf_cost": "benchmarks.commitments.tool:HIDING_LEAF_COST",
    "pouw_gemm": "benchmarks.pouw.tool:POUW_GEMM",
    "pouw_route_u": "benchmarks.pouw.tool:POUW_ROUTE_U",
    "pouw_vllm": "benchmarks.pouw.tool:POUW_VLLM",
    "pouw_e2e": "benchmarks.pouw.tool:POUW_E2E",
    "pouw_pearl_cal": "benchmarks.pouw.tool:POUW_PEARL_CAL",
    "serving_commit_cost": "benchmarks.commitments.tool:SERVING_COMMIT_COST",
    "flock_pure": "verity.ml.flock.tool:FLOCK_PURE",
    "flock_agreement": "verity.ml.flock.verifier.tool:FLOCK_AGREEMENT",
    "flock_class_sweep": "verity.ml.flock.tool:FLOCK_CLASS_SWEEP",
    "check": "tools.verity.check.tool:CHECK",
    "lean_audit": "tools.verity.lean.tool:LEAN_AUDIT",
    "lean_upstream": "tools.verity.lean.tool:LEAN_UPSTREAM",
    # vLLM integration sweep stages (integration-owned declarations).
    "vllm.build": "integrations.vllm.verity_vllm.pipeline.research_tools:VLLM_BUILD",
    "vllm.plan": "integrations.vllm.verity_vllm.pipeline.research_tools:VLLM_PLAN",
    "vllm.match": "integrations.vllm.verity_vllm.pipeline.research_tools:VLLM_MATCH",
    "vllm.commit": "integrations.vllm.verity_vllm.pipeline.research_tools:VLLM_COMMIT",
    "vllm.replay": "integrations.vllm.verity_vllm.pipeline.research_tools:VLLM_REPLAY",
}


def _from_file(module: str, attr: str, source_root: Path):
    """Execute `<source_root>/<module path>.py` under a name unique to that file and its content: no sys.path edits, no
    cross-root module caching, and a declaration that changed on disk (the target branch gained a commit) is read anew."""
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
    """``module`` as each earlier layout named it, newest first: its path mapped back through each move's renamed paths
    (`moves.json`; a file maps as itself without `.py`, a directory as a prefix)."""
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
    """``path`` (a file, or a directory with or without its `/`) as each earlier layout named it, newest first: mapped back
    through each move's renamed paths (`moves.json`; a directory maps as a prefix)."""
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
    """``path`` if ``exists`` holds for it, else the newest name an earlier layout gave it that does (`former_paths`), else None:
    how a reader of an older commit or tree finds a file the layout moved since."""
    if exists(path):
        return path
    return next((p for p in former_paths(path, moves) if exists(p)), None)


def locate(spec_or_name: str, root: Path) -> tuple[str, str] | None:
    """(module, attr) of the declaration ``root`` holds for a tool, under its module or one an earlier layout gave it; None
    when it holds none.  The current module is looked for before `moves.json` is read, so an unreadable one can't hide a
    declaration at its current path (`merge.required_steps` would then require no step)."""
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
    """The declaration of a tool that the tree at ``root`` holds (`locate`), executed from its file."""
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
        for cand in (root, *list(root.parents)[:6]):     # `--cwd integrations/vllm`: the declaration's file sits at the repo root above
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
