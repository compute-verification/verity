"""What importing a module of the query loads.  A tap, a source or a reader imports these modules for a name, a constant or a
parser, so none of them loads numpy, torch or the core `verity` when it is imported: the imports that need one are inside the
functions that read them."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

LIGHT = ("verity_vllm.query.norm_scales", "verity_vllm.query.router_softmax", "verity_vllm.query.type_repr",
         "verity_vllm.query.vocab_range")
HEAVY = frozenset({"numpy", "torch", "verity"})

_PACKAGES_LOADED = """
import importlib, json, sys
before = set(sys.modules)
importlib.import_module(sys.argv[1])
print(json.dumps(sorted({m.split(".")[0] for m in set(sys.modules) - before})))
"""


def _packages_loaded_in_a_fresh_process(module: str) -> set[str]:
    out = subprocess.run([sys.executable, "-c", _PACKAGES_LOADED, module],
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("module", LIGHT)
def test_importing_it_loads_neither_numpy_nor_torch_nor_the_core(module):
    loaded = _packages_loaded_in_a_fresh_process(module)
    assert "verity_vllm" in loaded, "the probe imported the module"
    assert not loaded & HEAVY, (f"importing {module} loads {sorted(loaded & HEAVY)}: move the import that brings it inside the "
                                "function that reads it")
