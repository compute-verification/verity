"""verity.ml.circuits.ops: exact, hardware-faithful semantics of ML primitives, and their registrations in the Verity IR.

Each module registers one family of word Definitions and says in its docstring which; `ops.boolean` has them on bits. There is
no list here: the directory is the list. The steps they read are `verity.ml.circuits.silicon`'s, and their numpy batch forms
are `verity.ml.kernels.tc`'s, registered with `verity.ml.circuits.evaluation` and self-checked against `silicon`.
"""
