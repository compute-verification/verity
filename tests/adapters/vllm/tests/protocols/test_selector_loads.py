"""The selector loads a protocol's code only when asked about one of its schemes."""
from __future__ import annotations

import subprocess
import sys

ASK = """
import sys
from verity_vllm import protocols as P
assert not P.cuts("pous", "pous") and not P.keeps_weight_copy("pous", "pous")
print("\\n".join(m for m in sys.modules if m.startswith("verity.protocols.resource_accounting.compute")))
"""


def test_asking_the_selector_about_pous_imports_nothing_of_compute_accounting():
    out = subprocess.run([sys.executable, "-c", ASK], capture_output=True, text=True, check=True).stdout
    assert out.split() == []
