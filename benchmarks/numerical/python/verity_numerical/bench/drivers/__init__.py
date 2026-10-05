"""Backend drivers for ``bench.cell``: one module per backend (``C-interactive`` -> ``c_interactive.py``), each wrapping that
backend's own pod scripts.  A module defines

    BACKEND, FAMILY           the backend id and its family (``cell.BACKENDS``, ``cell.FAMILIES``)
    INTERACTIVE               True when the protocol puts a live verifier on a separate pod
    RESULT                    the prover run's result document, relative to its run directory
    def unmet(cell) -> str | None                  what the cell lacks for this driver (optional)
    def jobs(cell) -> list[dict]                   in launch order: {role, pod, stage, command, env?, ttl?, send?} for a ``research run
                                                   --on``, or {role, pod, setup: True, command} for a ``research pods`` step
    def register(cell, prover_run, verifier_run, out) -> list[list[str]]
                                                   the commands (the backend's own registration script) that write the
                                                   ``bench-result/v1`` document to ``out``

A backend without a module is ``n/s`` for every cell until its lane adds one.
"""
from __future__ import annotations

import importlib
from types import ModuleType


def module_name(backend: str) -> str:
    return backend.lower().replace("-", "_")


def module_path(backend: str) -> str:
    return f"benchmarks/numerical/python/verity_numerical/bench/drivers/{module_name(backend)}.py"


def driver(backend: str) -> ModuleType | None:
    name = f"{__name__}.{module_name(backend)}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:
            return None
        raise
