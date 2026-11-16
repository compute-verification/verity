# A registered circuit's requirement in Lean

A registered circuit's requirement is what it asks of memory. `ResourceAccounting.Requirement`
(`security/Definitions/ResourceAccounting/Requirement.lean`) defines it: for one scheduled execution of the
circuit and a layout of its values in the audit's store, each step's live and read values, the store's blocks they lie
in, and those the audit credits. This package computes it, as executable Lean: the Lake package `core_requirement`.

It is code, so it requires nothing of `security/`: the Lean check's
`security_properties` package states that each function is the definition it computes (`Properties.ResourceAccounting.Requirement`)
and proves it (`Proofs.Properties.ResourceAccounting.Requirement`).

## Try it

Build it, run its Lean check, and run its difftest against the Python, `verity.circuits.requirement`:

~~~sh
cd verity/circuits/requirement && lake build                 # .lake/build/bin/requirement-difftest
python tools/verity/lean/check.py verity/circuits/requirement   # the Lean check, from the repository root
uv run tools/check/suites.py verity/circuits/requirement # the difftest against the Python
~~~

[`DESIGN.md`](DESIGN.md) says what each function computes, gives the difftest's instance format, and says what the
tests hold.
