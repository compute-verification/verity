"""`lean_headline.py`: the machine model's `lean_headline` under the path the files that load it by path know.

    python benchmarks/pouw/lean_headline.py [--check]    what `machine-model lean-headline [--check]` does

The exporter is `verity_machine_model.lean_headline`, and `machine-model lean-headline` is its command. This file gives
the exporter's names (its `__all__`) to the tests and the theory script that load this path, and is the writer the Lean
files name.
"""
from verity_machine_model import lean_headline as _exporter

globals().update({name: getattr(_exporter, name) for name in _exporter.__all__})

if __name__ == "__main__":
    raise SystemExit(_exporter.main())
