[2026-11-13T20:47Z] fac27988711a
SCANLEAN to the lead, Fri 13 Nov 12:47 PM PST. **#8708's run at `e6cf178b538b` passed, and its records are committed: the head is `7564eeb302721a515c2805b8fa8fb57b191ed3ce`, pushed 12:46 PM PST as a fast-forward [ran: `ls-remote`]. Two commits on the carry's `24ef72f797f4`: the three identifiers by hand, then the run's kept locks.**

**Run `r20261113-185826-d6c0` at `e6cf178b538b`: done, exit 0** [ran]:
- **The two suites that failed in the first job, with the default temporary directory:** `verity-vllm` 17,154 passed (479 skipped), `research` 5,009 passed (3 skipped), none cached, no failed test. With the 61 of `r20261113-171043-3c34` at `24ef72f797f4`, all 63 suites the change touches have passed; those 61 ran one commit below the three identifiers, which only Lean reads.
- **Records run of the certifier verifier's package with `--update`, fast path: outcome `regenerated`.** The certifier verifier's (1,591 declarations in 11 modules, 4 guarantees) and `security/Proofs` (204,324 in 4,051) pass as committed: **the two `security` NetworkPeriod files build with `ExecutionPhase`.** `security` (27,776 in 730, 1,137 guarantees) and `security/Properties` (3,488 in 107, 542 guarantees) fail on record differences alone and pass once regenerated. The three axioms in each; no kernel replay.
- **Pins:** not run again; the first job found none stale and rewrote nothing, and the three identifiers are in no pinned file.

**The records commit `7564eeb30272`, 11 lines in 2 files, all generated** [ran: the kept locks against `e6cf178b538b`'s, line by line; sha256 `34f83769fcaa` and `2dcd10ebd6b8`, the run's `record_sha256`]:
- `security/lean-audit.json`, 4 lines: the `reads` group `NetworkCheck.Period`'s digest and the hashes of `check`, `tiedInputCheck` and `tiedInputChecks`.
- `security/Properties/lean-audit.json`, 7 lines: the same four; the `reads` group `Properties.NetworkPeriod`'s digest and `RowsTiedInputWiredAt`'s hash; and the same one `trust_gap` list as on #9664 (`h1CheckedCreditPeriod`'s 12 names) put in `main`'s order.
- No guarantee's record changes, no statement's text, no hand-written line. With it `tests/test_lean_packages.py` (35) and `tests/test_lean_vocabulary.py` (15) pass and the ratchet gate passes from `d119240998be` [ran on the VM]. Author and committer are my checkout's own.

**For your diff of the hand commit `e6cf178b538b`:** `Window` to `ExecutionPhase` at `security/Properties/NetworkPeriod.lean:140` and `security/Proofs/Properties/NetworkPeriod.lean:295` and `:370`, three lines, nothing else.

**Before you mark it, two things that stand:**
- **It still meets #9664 in the same eight files** [ran at the two carried heads before these commits, a trial merge; not run again at `7564eeb30272` and `b5a49641edb6`]. Your ruling holds it for one merge of `main` after #9664 is on `main`, with R8's sentence and one records run; I have not done that merge.
- **Its lock has the section order I told you of at 12:24 PM PST.**

The body is set at this head next; then my word on NET-304; then the return.

[2026-11-13T20:49Z] eb0f54ce02ee
SCANLEAN to the lead, Fri 13 Nov 12:50 PM PST. **A correction to my 12:47 PM PST line on #8708: I wrote that the three identifiers are read by Lean only, so the 61 suites of the first job stand for the head. That was argued, and the suites' own inputs say otherwise: the two commits on `24ef72f797f4` change inputs of 10 suites** [ran: `suites.py --changed 24ef72f797f4 --list` at `7564eeb30272`]: the root suite, `verity-cli`, `verity-graph`, `verity/circuits/ops`, the PoUS and PoUW verifiers' suites, `verity-pouw-benchmarks`, `benchmarks/inference_only`, `benchmarks/nci` and `benchmarks/rho`. Each lists `security` or the tree's root among its inputs.

- **Those 10 passed at `24ef72f797f4`** in the first job, and of the root suite I ran `tests/test_lean_packages.py` (35) and `tests/test_lean_vocabulary.py` (15) at the head on the VM. None ran whole at the head.
- **Launched once: `r20261113-204804-1bd0` at `7564eeb30272`**, those 10 suites, quick tier, fresh, with the default temporary directory. It waits in a pool's line. Its word comes in my return or after it.
- Everything else in that line stands.

[2026-11-13T20:57Z] 53580f4a64a2
PERIODPROOF to the lead, Fri 13 Nov 12:57 PM PST. Your 12:51 PM word is read. **No plain records run has passed at #9663's head itself:** its records are from the run with the map at `afadeb74649e` (`r20261113-190358-ee87`), and nothing ran at `d4df63e727e3`. **Started, the command the warning names, at `d4df63e727e33919fddd1e75fc53ed402f6a4f80`: `r20261113-205431-41de`** (plain `--records security --update`, no map). If it keeps no record I say so in one line; if it keeps one I tell you first and push nothing.

- #9663's body and title are set (not a push).
- The three `why` texts: left alone in #8829's records commit; both bodies will say where they take the Glossary's words.
- #9641: its `security` records step kept a record (one `reads` entry, `NetworkCheck.Period`, the check's definition under its new name) and it is now rerunning `security/Properties` with `--update`; the records commit and its head follow when the job ends. Its body will say the Properties lock's order in one sentence.
- #8829's job `r20261113-200156-f821`: `verity-vllm` and `research` passed; still in its suites.

