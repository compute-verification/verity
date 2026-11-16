"""circuit-check's settings: the variables of its suite's session that are declared here, read by path
(``verity_config.settings.of(__file__)``): where the session reads the catalog bindings' records, and where it lists them.
Neither changes what a check computes, so no record's key holds either (`circuit_check.cache.UNKEYED`).

It owns no prefix: `circuit_check.cache.key` holds every other ``CIRCUIT_CHECK_*`` variable a process carries, declared or
not, so that a record counts only where they are set alike.  The rest of circuit-check's variables are still read directly, by
the files ``tests/lint/allowlists/repo_env_reads.json`` lists, until they move here.
"""
from __future__ import annotations

from verity_config.settings import WHERE, Registry
from verity_config.settings import Setting as S

REGISTRY = Registry(
    __file__, (),
    S("CIRCUIT_CHECK_RECORDS", str, None, WHERE,
      "the directory the suite's session reads the catalog bindings' records in, and stores the ones it has built in: unset, "
      "the one `check`'s circuit-check keeps them in. A record counts only for inputs identical to the ones it was made "
      "from, so the directory changes how long the catalog takes, not what it holds"),
    S("CIRCUIT_CHECK_RECORDS_MANIFEST", str, None, WHERE,
      "the file the suite's session lists the records its catalog rests on in (`circuit_check.units.listing`), which a "
      "store that carries the records to other runs names: unset, it lists none. The catalog is the same either way"),
)
