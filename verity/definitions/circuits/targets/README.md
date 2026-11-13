# verity.definitions.circuits.targets

The subcircuits a proof system is asked to prove (a subcircuit binds every parameter of a subcircuit template:
[`GLOSSARY.md`](../../../../GLOSSARY.md), "Subcircuit template and subcircuit"). `target.Target` is one of them: its gate
family and that family's digest, the pipeline its steps are computed on (a `verity.definitions.silicon` model), its program,
domain and epilogue, the device its relation was pinned on, the instruction class it is a relation over
(`verity.definitions.silicon.instructions.INSTRUCTIONS`), and its status, `frozen` (every proof system proves it) or
`draft` (stated, bound to nothing yet). `target.TARGETS` lists them by name, and each entry of
`verity/definitions/silicon/discrepancy_log.json` names one of them or an instruction class. The module's docstring states
the targets' functions.

The gate families' `veritor.*` ids are frozen: each spelling is in a family digest pin, so a new family takes a fresh id,
and `verity/circuits/tests/test_targets_frozen_families.py` fails on one under the old prefix.
