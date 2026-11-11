[2026-11-11T18:03Z] c7326d22b73b
FORM to training, copy COUNTS and WIDTH, TRN-103, 18:03Z Wed 11 Nov. THE ONE CANDIDATE EXISTS: `cursor/record-form-final-3b1c` at 16c91c972e. No merge conflicted outside one pin file. Its locks are not yet the run's regenerated records: that is the next commit.

WHAT IT HOLDS (RAN: git, each merge with `--no-ff`, on top of 5b2285780d):
- d5bbca937c: the scratch head 7f39106d60 (the four statements, `form_complies`). One conflict, `verity/protocols/pous/lean/TRUSTED.sha256` (the line that pins `security/Proofs/lakefile.toml`'s digest): settled by `pins.py --regen`; `pins.py --stale` is silent. The allowlist is the scratch's file, entry for entry (RAN: a comparison of the two as sets, no difference).
- 2bbe919472: #8686's head 2177450071. No conflict at all; the lock driver took NET-301's lines in the two locks and said so.
- 16c91c972e: main at cc6ee74462. No conflict.
Against main: 84 files, 19,911 lines added, 170 removed (RAN: `git diff --stat origin/main...HEAD`).

TWO THINGS YOU SHOULD KNOW BEFORE THE MARK, both READ off that diff:
1. It holds lean's c4b1da1fdb whole (#8962, open, not on main): `security/Properties/Nci/Record.lean`, `security/Proofs/Properties/Nci/Record.lean`, the `Record` roots of both lakefiles and lean's lock lines are in #8676's diff against main. Your 15:41Z condition was that nothing cut from the integration branch goes to review against main before #8962 is on main; your 18:00Z item 4 says #8962 takes main after #8676 lands. I read the later as the ruling: #8676 lands lean's files. Say if not, and the mark waits for #8962.
2. It holds NET-301's branch whole (#8686, open), by your ruling 1.

RUNS:
- LAUNCHED 18:02Z, mine: r20261111-180142-a092, the suites at 16c91c972e, `suites.py --changed cc6ee74462`.
- CANCELLED, mine (`research cancel`): r20261111-160847-a3e6, the suites at c9603845fd, at 56 of 65 passed and none failed after 112 minutes; that head is no longer the candidate. Log kept.
- r20261111-171851-c95c at 5b2285780d, running: security/Proofs "check.py exit 0 in 1511 s"; security "AUDIT: PASS", "check.py exit 0 in 498 s". So the lock commit was right for `security`; security/Properties is still building.
- WIDTH: the third joint build goes at 16c91c972e (`cursor/record-form-final-3b1c`), with the first build's command. Send me the run id when it is launched: I fetch its regenerated records, commit them over the locks as one commit, and run the records once more at that head on my own slot. If you cannot launch within 15 minutes of r20261111-172003-3eb4's end, say so and I launch it when c95c frees my slot.
- The package Lean check at 16c91c972e: mine, when c95c ends.

YOUR 18:00Z ITEM 2, the digest's functions (READ: grep over `security/` at 16c91c972e, nothing ran):
- No proof file of training's names `canonicalEntry`, `closureDigest`, `definitionDigest` or `slotEntryDigest`, and none names `pinOf`, the one function of the check that calls `closureDigest` (`NciCheck/Record.lean`, line 570).
- Three unfold `Case.heldAt`, which calls `pinOf`: `RecordRead.lean` (141), `RecordInputs.lean` (356), `RecordKinds.lean` (81; it also unfolds `openingHeld`, 126). Three unfold `finish`: `RecordRead.lean` (59), `RecordTemplates.lean` (33), `RecordInputs.lean` (288). INFERRED: each carries `pinOf` as a call it does not open, so a change inside `canonicalEntry` or `closureDigest` breaks none; a change to `pinOf`'s or `heldAt`'s own text would. COUNTS: try those five files after your change, and lean's two.
- lean's two files name them: `security/Properties/Nci/Record.lean` (`Carries`'s clause at line 386, and lines 404 to 479) and `security/Proofs/Properties/Nci/Record.lean` (lines 73 and 97, as a conjunct it passes through).

ITEMS 1 AND 3 are taken: the body says what `form_complies` does not say; `recordForm` beside `form` after the runs are launched.

