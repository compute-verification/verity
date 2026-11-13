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

