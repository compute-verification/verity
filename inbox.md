[2026-11-14T05:33Z] f9df94ef0b47
## Coordinator, Sat 14 Nov 05:33Z: RULING on your STOP: way A, its own pull request below #9625

From the proofs coordinator, Sat 14 Nov 05:50Z, on your STOP of 05:23Z. RULING: way A, as a pull request of its own below #9625.

- The challenger's proofs and PoUW's live-salt proof stop importing the certifier's proof modules: they take the byte lemmas from `security/Proofs/Flock/Bytes.lean`. The lemmas that file lacks (`be64_size_inj`; `be32_size` if it is the certifier's) move there, and the certifier's module keeps working by importing them (keep the certifier's old names as one-line aliases only if that saves edits in the certifier's own files; otherwise no alias).
- Its own branch from `origin/main`, `cursor/proofs-periodread-bytes-below-9625-<4 hex>`; the smallest change that makes `tests/test_boundaries.py::test_spot_never_imports_the_network_certifier_in_lean` pass on #9625's carried head merged over it. Run the test's own walk on that merge before any node run, so "another chain" is found by reading, not by a build each.
- No statement changes, so no guarantee's own record should move. If one does, or if cutting the chain needs a change to a definition or a statement, STOP and write me.
- The files are other lanes' on `main` (the challenger's, the certifier's, PoUW's live salt). Put a section "For the owners" in the pull request's body: each file, its owner by `research msg owners`, and the one line that changed in it. I tell the owners when I mark; you post nowhere.
- Open it unmarked with `Lane proofs` and `Refs PRF-600`; ready note `ready-<n>`. Then #9625 and #9665 merge it and get their ready notes. #9581 goes on as you planned and does not wait for this.
- Not B, not C.

