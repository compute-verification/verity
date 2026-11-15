"""Where this tree holds the inference-only policy's check (`verity/inference-only/v0`): its Lean package with the
executable of the protocol's form, and the Python that check is held to. `served_cases.py` loads this module by name
when the inference-only case is asked for (`served_cases.policy`), so it needs neither at import time. The public
release leaves the policy out, and this file with it: there the case is refused by name."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
#: the policy's Lake package and the executable `served_cases.build` builds in it
LEAN = (ROOT / "verity/protocols/inference_only/verifier/nci/lean", "inference-only")
REFERENCE = LEAN[0] / "scripts" / "nci_reference.py"


def reference() -> ModuleType:
    """`nci_reference`, the Python the policy's Lean check is held to."""
    spec = importlib.util.spec_from_file_location("nci_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
