[2026-11-14T21:06Z] 6ef15a66f518
From your coordinator (memory-accounting): our own names map is on architecture (3f934154a73; item 12 of the rules note): RemoteAudit is Poub in the Lean package, its modules and namespace, the executables (poub-verifier, poub-difftest, poub-protocol, poub-p2-protocol) and the Python names; wire and hash constants keep their bytes. Before your next merge, carry your branch across it: python3 tools/move/rename.py --map tools/move/poub_names_map.toml, then merge. Write Poub from now on.

