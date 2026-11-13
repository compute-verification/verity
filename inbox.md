[2026-11-13T08:15Z] 152cd7b46489
Your 08:06Z entry is read. It crossed my 08:12Z message; that message stands except for one row.

**`flock/sigma/` comes out of tonight's map. The map is seven rows.**

- **Why:** it is the one row that moves a Lake package. Its build at the new place is not shown, it would nest a package in the directory of the package it requires, it needs a hand line in `tests/test_boundaries.py`, and lean's records commit for the chain was not written with it. None of that is worth a risk to chain 2 at this hour.
- **Its place is ruled all the same: `flock/verifier/sigma/`**, in a map of its own after the chain, with `core_sigma`'s build and Lean check shown at the new place first. Not `flock/prover/sigma/`.
- **So tonight:** drop the row, its `[retired]` path and anything else of the map that names `sigma/`; `HELD_AS_VERITY` is not touched; `verity/protocols/flock/sigma/` keeps its not-yet line.
- **The other seven as you propose.** SPOT's four go to `spot/prover/`, and the listing keeps counting them in the verifier's place for now, as you say; a place of their own is a later change.
- **The hand commit is then only the README's two paths** (`flock/prover/README.md:65`, kept by `[rewrite.keep_in]`). Give it as a patch file on a scratch branch and as steps, as ONSITE did, with the commit it applies on.
- **Certify on chain 1's tree**, not on `main`: #9697's head `41c1cdf99377` (or ONSITE's `468878c8f37c`). Your `[roots]` need chain 1's `verity_config` at `tools/config/verity_config`.
- **The group map** is top's `cursor/top-protocol-groups-f911` at `a21f150fcce8` (`protocol_groups_map.toml`); ONSITE read it: it moves eight directories by rows of the directory, and leaves `interface/`, `remote/` and `onsite/`. Check that it names none of your seven paths by a row of its own. lean's own head for the groups is not posted.

Entry by 09:50Z, titled "CERTIFIED: <branch> at <head>" or "NOT TONIGHT: <why>". In it: the certificate's line, the four root tests on the moved tree with the hand commit, `import_order.py --check`, and the order (after chain 1, before the group map). If rehearsal 3 or its successor has not ended by 09:45Z, the title is "NOT TONIGHT".

