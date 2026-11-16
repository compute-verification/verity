# verity.circuits.targets

These are the tensor-core subcircuits a proof system is asked to prove, each stated exactly. A subcircuit binds every
parameter of a subcircuit template ([`GLOSSARY.md`](../../../GLOSSARY.md), "Subcircuit template and subcircuit"); the
first is one output coordinate of a BF16 GEMM on an A100's tensor cores. The code calls a subcircuit a `Target`, and
`target.TARGETS` lists them by name.

## Try it

~~~python
from verity.circuits.targets.target import FIRST, TARGETS

len(TARGETS)                  # 6
FIRST.status, FIRST.family    # ('frozen', 'veritor.tc-ampere-bf16@1'): the A100 BF16 GEMM coordinate
~~~

A subcircuit is `frozen` when every proof system proves it, and `draft` when it is stated and bound to nothing yet.
Each also names the silicon model its steps compute on (`verity.circuits.silicon`) and the instruction class it is a
relation over. The module's docstring states each subcircuit's function exactly, and each entry of
`verity/circuits/silicon/discrepancy_log.json` names one of them or an instruction class.

The gate families' `veritor.*` ids are frozen: each spelling is in a family digest pin, so a new family takes a fresh id,
and `tests/verity/circuits/test_targets_frozen_families.py` fails on one under the old prefix.
