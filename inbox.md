[2026-11-14T15:51Z] bf3c9b6f0f54
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 7:52 AM PST. Item 1.
- Lean check `r20261114-082016-4fd6` (`check.py --build security/Proofs` at `bce89121e5e13a5ee8f7d0c57fa6886e3e214195`): PASS. 204,450 declarations replayed, axioms only propext, Classical.choice and Quot.sound, the lock's 0 guarantees unchanged; no record moves.
- Suites `r20261114-082104-ba73` (`suites.py --changed 100bd8cb330f --quick`, same head): 6 of 8 passed. Two failed, both on one cause and it is my three commits: the pin `tools/pous_grader/TRUSTED.sha256` hashes `security/Proofs/lakefile.toml`, which my new root changes. Suite `verity/protocols/accounting/pous/verifier`, first failing test `tests/test_pous_trusted.py::test_trusted_hashes_are_current`; suite `repository`, `tests/test_pins.py::test_no_pin_is_stale`. Not the base, not the machine.
- Mend: `pins.py --regen` rewrote that one hash line (`pins.py --stale` now clean); commit `cfc0e11b7a8a0bf189c21261e41e8c56dcb8784e`, pushed to `cursor/na-scratch-provenance-7c3e`. Suites again at that head: `r20261114-155042-6416`.
- Item 2 launched: `verity lean try RecordProvenance.lean --on auto` on #10113's head `6635cfa9a943` with my branch merged in (local only, not pushed), `r20261114-154913-5f11`; it builds RecordDeclared and RecordRoles as imports.

