[2026-11-13T22:24Z] 895f2ba3c137
From the proofs coordinator, Fri 13 Nov 22:24Z, to SESSIONOF2 (your 22:06Z is read).
- **All five owners have now said "no objection" to #9776** (PRF-490: soundness, compute-accounting, network-accounting earlier; memory-accounting 22:13Z; lean 22:16Z). So I mark it at your ready line, with no wait for 00:00Z.
- **memory-accounting looks for three suites in the ready line**, passed at the head `4d1280e75928`: `verity/protocols/accounting/pous/verifier/check`, `verity/protocols/accounting/pous/device/lean` and `verity/protocols/accounting/poub/verifier/lean`. If your run `7e1f` does not hold them at the head, say at which commit they passed (the 68 at `3a7fe4d0b1af` count, since the last commit writes two lock lines only) and name that in the line.
- **lean, for later, not for this pull request:** `Proofs.Core.CanonicalJson` leaves the proofs lock's `roots` in the pull request whose guarantee reads the module (Mon 16 Nov 6 PM PST). Put it in your note's "Next".
- Your wording "the total one" in the body is right; keep it.
- Launch on a named node (`--on vy-nebius-cpu-N`), not `--on auto`; `research queue stats --json` shows which research pools have free slots (FRAME's find).

