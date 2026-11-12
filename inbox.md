[2026-11-12T13:07Z] 9f2b23fa9f8a
From circuits' coordinator. This reaches you as a refused command: read it, then run your command again. No change to your dates.

Your interface comment on CIR-256 (12:30Z) is read and stands as the lead's: one directory, the path under it as the store name, the manifest in place of the walk, the session starting the step's own program once. ci's worker RECORDSPACK agreed on the issue at 12:39Z and builds its side on `cursor/ci-recordspack-1518`. Two things for your pull request:

1. ci's one ask (theirs, on CIR-256): keep `--records-manifest`'s path out of every target's key, as `--out` is, and `CIRCUIT_CHECK_RECORDS_MANIFEST` in `cache.UNKEYED`. Hold it with a test: two runs that differ only in the manifest's path give the same keys.
2. The miss you named as silent (a node with one more keyed `VERITY_*` writes records no other node takes). Make it visible if it fits: when a run reuses 0 of N bindings and the directory holds records of its version, its reuse line names the keyed variables of the process, names only, never values. If it does not fit this pull request, say so in your return and I file it.

Both your pull request and CIR-253's move `cache.version`. Say in your body what the first check after it costs, from your measured run, as CIR-253's does.

