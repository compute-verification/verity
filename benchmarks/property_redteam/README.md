# Attacks on the stacking property

The encode bound of the bandwidth audit (PoUB, `verity/protocols/resource_accounting/memory/poub/verifier`) rests on
the stacking property (property/v1) and on the gate model's unit counts. This directory attacks both: a refusal suite
of Programs property/v1 must refuse or fully budget, beside the honest shapes it must pass, and constructions against
the gate model's conjectures.

**Backs:** `bitwise_codec.py`'s construction is `ResourceAccounting.Guarantees.Gates.EncodeChainRefuted`
(proved in `security/Proofs/ResourceAccounting/Gates/Bitwise.lean`), and `squaring.py` and `renorm.py` attack
`ResourceAccounting.Assumptions.SquaringFloor`, the named conjecture the encode bound takes in its place.

## Try it

Grade the check that accepts every Program against the suite:

~~~sh
uv run python benchmarks/property_redteam/run.py --check accept-all --out /tmp/redteam.json
~~~

It prints each case's outcome (`ok`, `sound`, `unsound`, ...) and a count of each, writes the report to `--out`, and
exits 1, since some attacks pass a check that refuses nothing. A check of your own is `--check MODULE:FUNCTION`;
`run.py`'s docstring says what it returns and what each outcome means. `uv run tools/check/suites.py
benchmarks/property_redteam` runs every attack and the cases on the CPU.
