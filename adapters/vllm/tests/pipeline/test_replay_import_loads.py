"""What importing the replay's modules leaves unloaded.

The replay driver and the cuBLAS route pin put the catalog's Definitions into the Definition table as they load, and a reader of
the table answers by what it holds.  `pipeline.c2_replay`, `pipeline.replay_bundle` and `pipeline.cpu_replay` import both inside
the functions that read them, so a process that imports one of the three holds none of those Definitions until a replay runs.
The Commit's module imports the first two, and the route pin for its own check: it leaves the driver unloaded.  Each case imports
one module in a process of its own.
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

DRIVER, ROUTE_PIN = "verity_vllm.check.replay.driver", "verity_vllm.check.cublas_route_pin"
#: a module, and the modules its import leaves unloaded
CASES = {"verity_vllm.pipeline.c2_replay": (DRIVER, ROUTE_PIN), "verity_vllm.pipeline.replay_bundle": (DRIVER, ROUTE_PIN),
         "verity_vllm.pipeline.cpu_replay": (DRIVER, ROUTE_PIN), "verity_vllm.pipeline.commit": (DRIVER,)}
CHILD = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
from verity.circuits.defs import REGISTRY
held = {"loaded": [m for m in sys.argv[2:] if m in sys.modules], "ids": len(REGISTRY.defs)}
for m in sys.argv[2:]:
    importlib.import_module(m)
held["ids_with_them"] = len(REGISTRY.defs)
print(json.dumps(held))
"""


class Held(NamedTuple):
    """What a process holds after it imports one module: which of the module's case are loaded (`loaded`), how many Definitions
    the table holds (`ids`), and how many once the case's modules are imported there (`ids_with_them`)."""
    loaded: list[str]
    ids: int
    ids_with_them: int


@functools.cache
def held(module: str) -> Held:
    """A fresh process that imports `module`, and what it holds then.  It runs in the adapter's directory, so the `verity_vllm` it
    imports is this tree's."""
    child = subprocess.run([sys.executable, "-c", CHILD, module, *CASES[module]], cwd=Path(__file__).resolve().parents[2],
                           capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stderr[-2000:]
    return Held(**json.loads(child.stdout.splitlines()[-1]))


@pytest.mark.parametrize("module", CASES)
def test_the_import_leaves_them_unloaded(module: str) -> None:
    assert held(module).loaded == []


@pytest.mark.parametrize("module", CASES)
def test_their_definitions_enter_the_table_when_they_load(module: str) -> None:
    got = held(module)
    assert got.ids_with_them > got.ids
