"""verity.ml.circuits.ops: exact, hardware-faithful semantics of ML primitives, and their registrations as Definitions.

Each module registers a family of primitives or of composites over them, and its docstring says which. `ops.boolean` is
word arithmetic as Boolean circuits, among them the next version of each of these Definitions on bits.

The step semantics these evaluate through are `verity.ml.circuits.silicon`'s, and their numpy batch forms are
`verity.ml.kernels.tc`'s, registered with `verity.ml.circuits.evaluation` and self-checked against `verity.ml.circuits.silicon`.
"""
from __future__ import annotations

__version__ = "0.1.0"
