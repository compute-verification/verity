"""`lean_headline_dc.py`: the machine model's `lean_headline_dc` under the path the files that load it by path know.

    python benchmarks/pouw/lean_headline_dc.py [--check | --numbers] [--json F]    the data-centre families' part of
                                                                                   `machine-model lean-headline`

The exporter is `verity_machine_model.lean_headline_dc`, on the readings of `verity_machine_model.dc_reading`, and
`machine-model lean-headline` is its command. This file gives both modules' names (their `__all__`) to the tests that
load this path, and is the writer the Lean files name. It reads this tree's machine model and refuses any other
(`MODEL`).
"""
import sys
from pathlib import Path

MODEL = Path(__file__).resolve().parent.parent / "machine_model"

# The readings are the machine model's data: an installed verity_machine_model from another checkout would give that
# tree's numbers without a word, so this tree's goes first and any other refuses.
sys.path.insert(0, str(MODEL))
import verity_machine_model  # noqa: E402

if Path(verity_machine_model.__file__).resolve().parents[1] != MODEL:
    raise ImportError(f"lean_headline_dc reads {MODEL}'s verity_machine_model, but {verity_machine_model.__file__} "
                      "was imported first: run it in a fresh interpreter, or put that directory first on sys.path")

from verity_machine_model import dc_reading, eps, lean_headline_dc  # noqa: E402

E, LH, Q = eps, lean_headline_dc.LH, lean_headline_dc.Q
for _module in (dc_reading, lean_headline_dc):
    globals().update({name: getattr(_module, name) for name in _module.__all__})

if __name__ == "__main__":
    raise SystemExit(lean_headline_dc.main())
