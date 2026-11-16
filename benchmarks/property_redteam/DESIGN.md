# benchmarks/property_redteam: design

`cases.py` is the refusal suite: Programs
property/v1 must refuse or fully budget, and the honest shapes it must pass (`cases.json` is their metadata as data).
`run.py` grades any check against them. `gates.py` is the gate model (`security/Definitions/ResourceAccounting/Gates/Model.lean`)
on the canonical circuit, which the attacks build in. `bitwise_codec.py` builds any number of codec words for 8 units a key byte,
which falsified `EncodeChainHardness`, and `free_codec.py` is the construction before it. `squaring.py` attacks
`SquaringFloor` in the gate model at p1024, `renorm.py` searches toy fields for counterexamples to the re-normalization
conjecture N_L, `products_pass.py` counts data products on property/v1's pass, and `demo.py` evaluates a registry.

## Who reads it

**Status: active.** The lock names `tests/test_bitwise_codec.py::test_the_conjecture_fails_at_p1024` as
`EncodeChainHardness`'s falsification test, `security/Definitions/ResourceAccounting/Gates/Registry.lean` holds `classOf` to
`cases.py`'s closed table, `benchmarks/stacked_share` loads `products_pass.py`, and the audit's `test_encode_chain.py`
and `tests/verity/circuits/test_stacking.py` cite it. Decided by `rg property_redteam` over `verity/`, `benchmarks/`,
`demos/`, `tests/`, `adapters/` and `security/`.

**Reads and writes:** `run.py` reads the cases and a check, and writes a JSON report; `run.py --json` writes
`cases.json`.
