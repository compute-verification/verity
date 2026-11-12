"""A harness for one run, on a scratch branch only (`-p ccmem_first`): register every Definition in the order circuit-check's own
process does, before pytest imports the suite's conftest or any test module.

The catalog's registered roots are the registry in the order its modules were imported (`targets._registered`), cut into
eight records by position (`units.registered_entries`).  A session of the suite imports its test modules before
`targets.load_registries` runs, so its registry is in another order than the one `circuit-check --records-only` stored, the
record of `<registered>:0` lists other roots than the session makes, and the session's catalog never comes from the records.
Loading the registries here, first, gives the session the order of the records."""
from circuit_check import cache, cli, targets, units  # noqa: F401  (as conftest.py and the probes import them)

targets.load_registries()
