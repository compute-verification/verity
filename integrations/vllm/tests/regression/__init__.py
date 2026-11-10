"""The reference regression harness: does a NEW vLLM integration reproduce the v1 reference records?  Read `test_regression.py`'s
docstring first (the reference, what "reproduces" means, tiers, gaps, how to run); `resolver.py` for where the bytes come from;
`checks/` for one pure module per oracle; `lift_expected.py` / `stage_store.py` for how fixtures.toml, files.toml, expected/ and the store
artifacts were produced; `store_io.py` for the only code that talks to the data API."""
