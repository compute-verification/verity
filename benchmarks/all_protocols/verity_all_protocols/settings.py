"""Every environment variable all-protocols reads (the runner, its legs and its tests), declared once: the shape of
``benchmarks/pouw/settings.py`` for this benchmark.

Read a setting as an attribute (``settings.CERTIFIER_HEIGHT``) or with `get`: the variable's string parsed by its type, or its
default when it is unset. An empty string is a value (``CERTIFIER_TAMPER=`` relays no tampered run, ``WARDEN_TAMPER=``
runs none), except for an int, where it is the default. An undeclared name is an AttributeError, and a string that doesn't
parse is a ValueError naming the variable. `put` writes a declared variable for the processes the run starts. The legs
import it as ``verity_all_protocols.settings``, putting ``benchmarks/all_protocols`` on their path themselves, since they are
loaded by file path (`table.load`, and their own suite, which has only ``legs`` on its path).
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple


class Setting(NamedTuple):
    type: Callable[[str], Any]
    default: Any
    doc: str


SETTINGS: dict[str, Setting] = {
    "RESEARCH_RUN_ID": Setting(str, None, "the research run this process is part of; a report's run, else local"),
    "RESEARCH_MACHINE": Setting(str, None, "the machine research run placed the run on; the runner's --node, else the host"),
    "PYTHONPATH": Setting(str, "", "the module path a child process starts with, after the paths the runner and legs put first"),
    "CERTIFIER_TAMPER": Setting(str, None, "the tampered relay the certifier leg runs beside the honest one; unset, extra "
                                           "(--tie-tamper sets byte)"),
    "WARDEN_TAMPER": Setting(str, None, "the warden path's tampered runs, comma-separated; unset, legs/certifier.py's "
                                        "WARDEN_TAMPERS"),
    "CERTIFIER_PATH": Setting(str, None, "the certifier leg's path, one of legs/certifier.py's PATHS; unset, the first"),
    "CERTIFIER_HEIGHT": Setting(int, 10, "the certifier's tree height"),
    "SCAN_DRAW": Setting(int, 16, "the warden scan's sampled draw: Lean draw --stratified K of the window's 2T units"),
    "VERITY_POUW_CHECK": Setting(str, None, "the Lean pouw-check the PoUW leg holds core_check's answer to; unset, none"),
    "VERITY_ALL_PROTOCOLS_TOOLS": Setting(str, None, "run.sh's tools.json, which tests/test_integrity's end-to-end test "
                                                     "proves two windows with; unset, it skips"),
}


def get(name: str, environ: Mapping[str, str] | None = None) -> Any:
    s = SETTINGS.get(name)
    if s is None:
        raise AttributeError(f"{name} is not a declared setting (benchmarks/all_protocols/verity_all_protocols/settings.py)")
    v = (os.environ if environ is None else environ).get(name)
    if v is None or v == "" and s.type is not str:
        return s.default
    try:
        return s.type(v)
    except ValueError as e:
        raise ValueError(f"{name}={v!r}: {e}") from None


def put(name: str, value: str) -> None:
    get(name, {})
    os.environ[name] = value


def __getattr__(name: str) -> Any:
    if name in SETTINGS:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
