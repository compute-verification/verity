"""Which of H1's modules load the live circuit when imported alone: the four of `LOADERS`.  A Definition enters the
Definition table (`verity.circuits.defs.REGISTRY`) as its module loads, and a reader of the table by id
(`verity.commitments.windows.interface.resolve`) answers for it from then on.  The live circuit
(`verity.protocols.accounting.pouw.prover.circuit.live`) builds its Definitions as it loads, so a module that comes to load it at its
head changes what that reader answers in every process that imports the module.  The modules that stage a window's sampled
proof blocks rebuild their values through `served`: they import it where they rebuild one and never at their head.  Each
case imports its module in a process of its own: the test process has loaded the live circuit for other tests."""
from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

from verity_vllm.protocols.pouw import h1

LIVE = "verity.protocols.accounting.pouw.prover.circuit.live"
#: the modules whose import loads the live circuit: `live` builds a window's live tables with it, `served` imports `live`,
#: and `kept` and `handoff` import `served`
LOADERS = frozenset({"verity_vllm.protocols.pouw.h1.handoff", "verity_vllm.protocols.pouw.h1.kept",
                     "verity_vllm.protocols.pouw.h1.live", "verity_vllm.protocols.pouw.h1.served"})
#: argv[1] is imported: whether the live circuit is loaded after it
CHILD = f"import importlib, sys\nimportlib.import_module(sys.argv[1])\nprint({LIVE!r} in sys.modules)\n"


@pytest.mark.parametrize("module", sorted(f"{h1.__name__}.{m.name}" for m in pkgutil.iter_modules(h1.__path__)))
def test_importing_a_module_of_h1_loads_the_live_circuit_only_from_a_loader(module: str) -> None:
    """With `served` or `live` imported at the head of a module outside `LOADERS`, the live circuit's Definitions are in
    the table from that module's import on, and `resolve` answers for their ids in a process that built no live table."""
    child = subprocess.run([sys.executable, "-c", CHILD, module], capture_output=True, text=True, check=True)
    loads = child.stdout.strip() == "True"
    assert loads == (module in LOADERS), (
        f"importing {module} {'loads' if loads else 'does not load'} the live circuit: import `served` or `live` inside the "
        "function that needs it, or name the module in LOADERS with its reason")
