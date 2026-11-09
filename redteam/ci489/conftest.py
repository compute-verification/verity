"""Red-team demonstrations for the verified landing queue (CI-489, workstream W8).

Each test asserts a behavior a finding names, so a pass confirms the finding on the code it ran against and a failure means the
behavior changed (re-read the finding). The code under test is loaded by path, never copied here:

    VQRED_GRAPH2   infra's graph2.py (compute-verification/infra, infra/executor/graph2.py)
    VQRED_VQUEUE   ci's research/vqueue.py with `decide` implemented (tools/research/src/research/vqueue.py)

`verity_graph` is this checkout's (tools/verity/graph). A module whose path is unset or missing skips its tests.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

GRAPH2 = Path(os.environ.get("VQRED_GRAPH2", "/tmp/infra/infra/executor/graph2.py"))
VQUEUE = Path(os.environ.get("VQRED_VQUEUE", "/tmp/vqueue/tools/research/src/research/vqueue.py"))


def load(name: str, path: Path):
    if not path.is_file():
        pytest.skip(f"{name}: {path} is missing (set VQRED_{name.upper()})")
    spec = importlib.util.spec_from_file_location(f"vqred_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def g2():
    return load("graph2", GRAPH2)


@pytest.fixture(scope="session")
def vq():
    return load("vqueue", VQUEUE)
