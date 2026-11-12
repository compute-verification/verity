"""What importing a module that stages a window's sampled proof blocks loads: no live circuit.  A Definition enters the
Definition table (`verity.circuits.defs.REGISTRY`) as its module loads, and a reader of the table by id
(`verity.commitments.windows.interface.resolve`) answers for it from then on.  `h1.served` loads the live circuit
(`verity.protocols.pouw.prover.circuit.live`), so the staging modules import `served` where they rebuild a value and never at
their head.  Each case imports its module in a process of its own: the test process has loaded the live circuit for other tests."""
from __future__ import annotations

import functools
import json
import subprocess
import sys

import pytest

#: the modules that stage a window's sampled proof blocks and rebuild their values
STAGING = ("verity_vllm.protocols.pouw.h1.drawn", "verity_vllm.protocols.pouw.h1.drawn_rows")

#: argv[1] is imported, then `h1.served`: whether the live circuit is loaded after each, and the ids `served` brings to the table
CHILD = ("import importlib, json, sys\n"
         "from verity.circuits.defs import REGISTRY\n"
         "LIVE = 'verity.protocols.pouw.prover.circuit.live'\n"
         "importlib.import_module(sys.argv[1])\n"
         "at_import, held = LIVE in sys.modules, set(REGISTRY.defs)\n"
         "importlib.import_module('verity_vllm.protocols.pouw.h1.served')\n"
         "print(json.dumps({'at_import': at_import, 'with_served': LIVE in sys.modules,\n"
         "                  'entered_with_served': sorted(set(REGISTRY.defs) - held)}))\n")


@functools.cache
def loaded(module: str) -> dict:
    """A fresh process that imports `module`, then `h1.served`: whether the live circuit is loaded after the first import
    (`at_import`) and after the second (`with_served`), and the Definitions that enter the table with the second."""
    child = subprocess.run([sys.executable, "-c", CHILD, module], capture_output=True, text=True, check=True)
    return json.loads(child.stdout)


@pytest.mark.parametrize("module", STAGING)
def test_importing_a_staging_module_loads_no_live_circuit(module: str) -> None:
    """With `served` imported at the module's head, the live circuit's Definitions are in the table from the import on, and
    `resolve` answers for their ids in a process that has rebuilt no value."""
    assert loaded(module)["at_import"] is False


@pytest.mark.parametrize("module", STAGING)
def test_the_live_circuit_s_definitions_enter_the_table_with_served(module: str) -> None:
    """`served` is what loads the live circuit, and Definitions enter the table with it: the first test holds something."""
    got = loaded(module)
    assert got["with_served"] is True and got["entered_with_served"]
