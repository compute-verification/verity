"""The census (``benchmarks/census/`` at the repository root, data only: see its README), read directly.

Each file is loaded as ``{id: entry}`` with entries as attribute namespaces (``HARDWARE["rtx-4090/e4m3"].flop_per_second``).
``$VERITY_CENSUS`` points at the directory when it lives elsewhere.  The files are validated against their schemas in
``tests/bench/test_census_data.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(os.environ.get("VERITY_CENSUS") or Path(__file__).resolve().parents[4] / "census")


def raw(name: str) -> list[dict[str, Any]]:
    """``benchmarks/census/<name>.json`` as written."""
    return json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))


def load(name: str) -> dict[str, Any]:
    return {e.id: e for e in json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"), object_hook=lambda d: SimpleNamespace(**d))}


HARDWARE = load("hardware")
DATATYPES = load("datatypes")
NETWORKS = load("networks")
SUBCIRCUIT_TEMPLATES = load("subcircuits")
#: Every subcircuit (a fully bound template) by id, with ``template`` naming its template.
SUBCIRCUITS = {s.id: SimpleNamespace(**vars(s), template=t.id) for t in SUBCIRCUIT_TEMPLATES.values() for s in t.subcircuits}
INPUT_SETS = load("input_sets")
WORKLOADS = load("workloads")
REFERENCE_NETWORK = next(n for n in NETWORKS.values() if n.role == "reference")
SENSITIVITY_NETWORKS = tuple(n for n in NETWORKS.values() if n.role == "sensitivity")
GEMM_COORDINATE = SUBCIRCUIT_TEMPLATES["gemm-coordinate"]


def subcircuit_of(target: str) -> Any:
    """The subcircuit a (legacy) target name denotes; ``KeyError`` when the census has none."""
    return next(s for s in SUBCIRCUITS.values() if getattr(s, "legacy_target", None) == target)


def subcircuit_with(template: str, **parameters: Any) -> Any:
    """The bound subcircuit of ``template`` with exactly these parameters (e.g. ``K``, ``semantics``), or ``None``."""
    for s in SUBCIRCUITS.values():
        if s.template == template and all(getattr(s.parameters, k, None) == v for k, v in parameters.items()):
            return s
    return None


def stood_for_by(subcircuit_id: str) -> str | None:
    """The subcircuit whose results and input set stand for ``subcircuit_id`` (its ``stood_for_by`` equivalence), or ``None``."""
    e = getattr(SUBCIRCUITS[subcircuit_id], "stood_for_by", None)
    return e.subcircuit if e is not None else None


def evaluate(expr: str, env: dict[str, Any]) -> int:
    """A work-model expression (``+ - * //`` and parentheses over integers and parameter names) evaluated without ``eval``."""
    import ast
    import operator

    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}

    def ev(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Name):
            return int(env[n.id])
        raise ValueError(f"not a work-model expression: {expr!r}")

    return ev(ast.parse(expr, mode="eval"))


def source(entry: Any) -> str:
    """An entry's sources as one line."""
    return "; ".join(s.label for s in entry.sources)


def line(device: str, dtype: str) -> Any:
    """The hardware line of ``device`` running ``dtype``; ``KeyError`` when the census has none."""
    for h in HARDWARE.values():
        if h.device == device and h.dtype == dtype:
            return h
    raise KeyError(f"no census hardware line for {device!r} / {dtype!r}")


def native_seconds_per_instance(h: Any, K: int) -> float:
    """Native wall seconds of one ``GemmCoordinate<K>`` instance (``2K`` FLOP) at a hardware line's peak."""
    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    return 2 * K / h.flop_per_second
