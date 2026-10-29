# benchmarks/property_redteam

Attacks on the stacking property (property/v1) and on the gate model's unit counts that the encode bound of the
bandwidth audit (PoUB, `verity/core/protocols/remote_bandwidth`) rests on. `cases.py` is the refusal suite: Programs
property/v1 must refuse or fully budget, and the honest shapes it must pass (`cases.json` is their metadata as data).
`run.py` grades any check against them. `gates.py` is the gate model (`Security/Definitions/Accounting/Gates/Model.lean`)
on the canonical circuit, which the attacks build in. `bitwise_codec.py` builds any number of codec words for 8 units a key byte,
which falsified `EncodeChainHardness`, and `free_codec.py` is the construction before it. `squaring.py` attacks
`SquaringFloor` in the gate model at p1024, `renorm.py` searches toy fields for counterexamples to the re-normalization
conjecture N_L, `products_pass.py` counts data products on property/v1's pass, and `demo.py` evaluates a registry.

**Status: active.** The lock names `tests/test_bitwise_codec.py::test_the_conjecture_fails_at_p1024` as
`EncodeChainHardness`'s falsification test, `Security/Definitions/Accounting/Gates/Registry.lean` holds `classOf` to
`cases.py`'s closed table, `benchmarks/stacked_share` loads `products_pass.py`, and the audit's `test_encode_chain.py`
and `verity/ml/circuits/tests/test_stacking.py` cite it. Decided by `rg property_redteam` over `verity/`, `benchmarks/`,
`examples/`, `tests/`, `integrations/` and `Security/`.

**Run:** `uv run tools/verity/check/suites.py benchmarks/property_redteam` runs every attack and the cases on the CPU.
`uv run python benchmarks/property_redteam/run.py --check MODULE:FUNCTION --out FILE.json` grades one check.

**Reads and writes:** `run.py` reads the cases and a check, and writes a JSON report; `run.py --json` writes
`cases.json`. **Backs:** `bitwise_codec.py`'s construction is `Accounting.Guarantees.Gates.EncodeChainRefuted`
(proved in `Security/Proofs/Accounting/Gates/Bitwise.lean`), and `squaring.py` and `renorm.py` attack
`Accounting.Assumptions.SquaringFloor`, the named conjecture the encode bound takes in its place.
