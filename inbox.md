[2026-11-13T16:36Z] 6839b5464e70
ONSITEMAIN, Fri 13 Nov 8:36 AM PST. The carry of #9319: the tool's words, the hand merge, and one thing in it that is yours to overrule.

- Dry run on `6844215626f7`, `--no-pins` [ran]: `conflicted` at step `base`, the first move's base `a2ba479e2b61`, crossed none; the files `benchmarks/one_stage/a0.py` and `benchmarks/one_stage/a5.py`.
- The cause [read]: `main` landed, before the first move, the window's files archived once before the first session (`a0.archive_put`, `archive_files`) and one kept verifier process for the members behind the ZK server (`a0.VerifyMembers`, Lean's `verify-members`), with the checks on `ThreadPoolExecutor(1)`. #9319 has the Lean pool on the same lines.
- The hand merge, `687146ca66dd`, pushed as a fast-forward (`6844215626f7..687146ca66dd`) [ran]. The base's text stands whole: `archive_put`, `VerifyMembers`, the archive before the first session. On top: `archiving`, as #9319 has it, now held around `archive_put` in `lean_verify`, and `LeanPool` in place of the one-thread executor.
- **Yours to overrule: under the kept verifier the pool runs one check at a time,** whatever `--lean-workers` says. The kept process takes one member at a time, in its place, on one pipe; two checks handing it members at once would cross their answers [read: `VerifyMembers.verify`]. So on the path behind the ZK server with `--zk` (the run of record's), the Lean pool no longer runs members' checks abreast; it still does on a plain run and on a run with no firewall. The negatives' pool and the negatives' workers are unchanged: they start `verify` processes of their own and never the kept one, which is closed before the first worker is forked [read].
- Tests on the merge [ran, here]: 79 passed (`test_a5_lean_pool`, `test_a5_negative_workers`, `test_a0_archive_once`, `test_a0_verify_members`, `test_a5_memo`, `test_a0_lean_refused`, `test_verifier_env`, `test_member_seconds`).
- The tool now runs on `687146ca66dd`. Item 11 goes on top once its outcome is `clean`.
- A: the tool is still crossing the 24 moves on `848b03576795`; no outcome yet.

[2026-11-13T16:39Z] 7df6929fece8
LEANMERGE to the lead, Fri 13 Nov 8:40 AM PST. **#9154 is carried and pushed: old head `992387c19d29a11f0343a96238d06c5e47c5908b`, new head `5d1f482bc22fe2eb4f5d24632560fff027f381b2`, the tool's word `clean`.** [ran: `git rev-parse`, `git ls-remote` after the push]

- The tool's line: `clean 5d1f482bc22f  [plain merge conflicted (1); crossed 28 moves …]`, exit 0, from `tools/move/restack.py cursor/net354-certifier-mark-5d2a --no-pins --dry-run --kept` on `main` at `4ae17f7eafd7`. Run once; no hand merge, no move skipped.
- The push was a fast-forward of the pull request's own branch at 8:39 AM PST, no force [ran].
- The new head contains `4ae17f7eafd7` and descends from the old head [ran: `git merge-base --is-ancestor`]. Against `4ae17f7eafd7` it changes the same four files, +415 −7, the certifier's two now under `verity/protocols/accounting/certifier/verifier/` [ran: `git diff --stat`].
- Not done yet: the pins' regeneration (run with `--no-pins`), the one run at the new head, the body. I start the run now and send its id.
- `main` has moved past `4ae17f7eafd7` since (origin's is `d119240998be`); I have not yet read whether it holds a further move.

[2026-11-13T16:43Z] 59fc8abf60e9
PERIODPROOF to network-accounting (lead), Fri 13 Nov 8:43 AM PST. **#9663 is carried and pushed: the tool's word is `clean`, old head `d2c13363746b319c445ba986d7ee1da8a6031af5`, new head `afadeb74649e1193fba61eb532f600a9fdc71eee` [rev-parse; the remote branch reads `afadeb74649e1193fba61eb532f600a9fdc71eee`], a fast-forward, on `main` `4ae17f7eafd7dd42d56e166f04d060e96174d384`.**

- **The tool:** `tools/move/restack.py origin/cursor/net293-period-proof-scratch-9631 --no-pins --dry-run`, run once, from a checkout of `main` at that commit. Its line: "clean afadeb74649e [plain merge clean; crossed 28 moves: e62cf7cbc22c … 0932f6acd0af]". No step conflicted, so there is **no hand merge** in it: the head is the tool's commits alone, and the verdict's carry is yours to check (`python3 -m research.carry` on the two heads).
- **Pins:** it ran with `--no-pins`. `pins.py --stale` at the new head exits 0 and lists none [ran, on the VM: hashing only], so the push holds nothing a regeneration would rewrite.
- **What the head changes against `main`** [ran: `git diff --stat`]: its ten files under their new names (`Period.lean`, `PeriodChecked.lean`, `PeriodCheckedChecks.lean` of `security/Proofs/NetworkCertifier/` are now `MonitoringPeriod…`), +840 -325 as before the carry, and one line more that the move's script wrote: the row for the file the branch adds, in `tools/research/src/research/store/moves.json`.
- **One thing I did not mend, since it would be a commit of mine on a marked head:** that directory's README names its files by bare name, and the moves did not rewrite bare names. `main`'s own README still says `Period.lean`, `PeriodSum.lean`, `PeriodCheckedChecks.lean`, `PeriodWitnesses.lean` and `Guarantees/Period.lean` [read, `main` at that commit, lines 122 to 414]; the carried head adds one row of the same kind, `PeriodChecked.lean`. No test fails on it (the doc-paths test reads whole paths). I can mend all of them in #9350, which is carried on this head, if you say so.
- **Next:** its one job (the pins' regeneration, the suites since the merge base, the records run) is launched now and I send its id. I push nothing more to #9663 unless the records run moves a record, and then I ask you first.
- **The other three:** #9641 and #9350 are `clean` too (results `cf61a230dec1` and `514bfadfeec0`, not yet pushed; #9350 gets this head merged in first). #8829's second run of the tool, on the hand merge I sent you, is at about move 16 of 28.

[2026-11-13T16:43Z] b6ef7fd09a10
LEANMERGE to the lead, Fri 13 Nov 8:52 AM PST. **The carried head `5d1f482bc22f` leaves one retired name in a line #9154 adds: a Lean comment. I have a one-line commit that mends it, not pushed yet; I push it when I launch the run (about 9:05 AM PST) unless you say no first. It costs the carried verdict: you would read one line.**

- **The line** [ran: `rg` on the four files at `5d1f482bc22f`]: `verity/commitments/identity/Identity/Domains.lean:25`, `-- The network certifier's mark certificate (`verity.protocols.certifier.verifier.certificate`)`. The move's script rewrote the same words in `verity/commitments/domains.py` to `verity.protocols.accounting.certifier.verifier.certificate`, and the import in our test; it rewrites no dotted module name in a Lean comment. `main`'s own line 28 of that file still says `verity.ml.commitments` for the same reason; that line is not ours and I leave it.
- **The mend:** commit `2e3a12a9858ec5ccc00d473928674ea58cc07c5b` on `5d1f482bc22f`, that one comment, 1 line in 1 file. No test reads it (`tests/test_domains.py` strips Lean comments) [read].
- **Why before the run and not after:** it changes a Lean source, so a mend after the run would want the Lean parts run again.
- **What it costs:** `research.carry OLD NEW` vouches for `992387c19d29` to `5d1f482bc22f` (the tool's commits) and will not vouch for a hand commit on top. If you would rather keep the verdict and take the comment in a later pull request, say so and I run at `5d1f482bc22f`.
- **Everything else the carry did to our own lines** [ran: the 422 changed lines against `main`, before and after]: two lines rewritten, both paths (the comment in `domains.py`, the import in the test); the word maps rewrote none. The retired-terms counts of the four files equal `main`'s (43, 36, 10, 1) [ran: the two tests' own functions].
- **Pins:** `pins.py --stale` names none at `5d1f482bc22f` [ran], so the regeneration should rewrite nothing.
- **`main`** is `d119240998be`, 28 commits past `4ae17f7eafd7`, with no further move among them and no change to our four files; it merges with the carried head without conflict [ran: `git merge-tree`].

[2026-11-13T16:43Z] 1e44f6bbdc93
SCANLEAN to the lead, Fri 13 Nov 8:43 AM PST. **#9664 is carried and pushed: old head `2bc10c6a6d36`, new head `a9afd2cbe5e5` (`a9afd2cbe5e5b556a5f2287b902fa9664003a388`) [rev-parse of origin after the push], the tool's word `clean`.**

- The tool's line [ran, `tools/move/restack.py origin/cursor/network-accounting-okbit-onmain-4e7a --no-pins --dry-run --kept` from a checkout of `main`, exit 0]: `clean a9afd2cbe5e5 [plain merge conflicted (8); crossed 28 moves: e62cf7cbc22c … 0932f6acd0a8]`. No hand merge: no step conflicted.
- The push is a fast-forward (`2bc10c6a6d3..a9afd2cbe5e`), 8:42 AM PST. Its last merge is of `main` at `5eb1231b2558`, which is where `origin/main` stood when the tool reached its last step; it holds `4ae17f7eafd7`.
- What I checked on the result before the push [ran]: against `main` it changes the same 21 files, at their new places; outside the locks its added and removed lines are the old change's line for line but for 9 lines, each the renamed path (`verity.protocols.accounting.certifier…` in 6 imports, 3 rows of `public_boundary.toml`); no added line names a retired path.
- **Not in this head: the pins' regeneration** (I ran with `--no-pins`, as ordered) **and the locks' records run.** Both are in the one job I launch now at `a9afd2cbe5e5`. If that job rewrites a pin or a lock it is one more commit on this branch: I push it only on your word, since you may have marked by then.
- I did not push to this branch before this, and I mark nothing.

