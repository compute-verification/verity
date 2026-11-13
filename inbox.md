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

