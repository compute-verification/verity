[2026-11-11T04:25Z] 23bb8c906421
From CROSS (NET-311 job 4), Tue 10 Nov 8:25 PM PST. **#7109 is pushed: `cursor/net169-tally-061a` is at `dbf5daaa36e111cefc239b88f6b7715fd1c2f495` since 8:22 PM PST, a fast-forward from `53f2a6bfd010`. Its two runs are launched.**

- Queue status at 8:21 PM PST read "not marked ready at 53f2a6bf" [run].
- Quick tier: `r20261111-042308-64fe` (58 suites). The lock's records, your command with `--source` the worktree at the head: `r20261111-042303-0a73`. Both have `source_sha` `dbf5daaa36e1` and were waiting for a slot at 8:24 PM PST [run: `job.json`]. Two runs for this head, no more.

**Text for #7109's body:**

> Crossed onto the layout move (`main` `5c095bb4ab70`), three steps over `53f2a6bfd010`:
> - `fb8af76df2bf`: a merge of `main` `a930252c4993` on the old paths. One file is resolved by hand, `tools/verity/check/public_boundary.toml`: both sides append a `[[needed]]` entry at the same place, and both are kept, main's two and then this change's one.
> - `68d86350489e`: the result of `tools/move/restack.py` on that merge over both moves (`6581bcf197`, `c1ca25a247`): `clean`.
> - `dbf5daaa36e1`: `security/lean-audit.json` only, 344 lines. The move's script rewrites no file over 4,194,304 bytes, and this change's lock is 4,206,183, so the crossing left the paths from before the move in it. This commit rewrites them by the moves' own maps: 340 lines become `main`'s lines, 4 are this change's frozen reasons. The rule, applied to `main`'s lock from before the first move, gives `main`'s lock after the second byte for byte. Against `main` the lock is +499 −30, this change's own, as before the crossing.
> - `public_equal.py fb8af76df2bf dbf5daaa36e1`: `public_equal: EQUAL: 0 became public, 0 became private`.

Runs in flight, all mine: `0481` (#8499; 12 suites passed, none failed at 8:24 PM PST), `dcf5` (#8708; 28 passed, none failed), `64fe` and `0a73` (#7109). Job 3's three ended exit 0: `857d` 54 suites passed, `9a68` 58, `9862` 58 [run: `research fetch`]. Next: the note's section and the one comment on NET-311, then I end my turn on the four runs.

[2026-11-11T04:29Z] b5087f0ec2a5
CROSSB, NET-311, setup and where `restack.py`'s docstring and --help differ from the brief. Tue 10 Nov, 8:30 PM PST.
1. origin/main is 5c095bb4ab7012ba359e0a85f4ecf19892b99629; all six branches are at the table's heads [run: fetch 8:16 PM]. `uv sync` done; the clone defines every merge driver `.gitattributes` names [run: `restack.missing_drivers()` gives none].
2. Moves: with no `--move-commit` the tool takes every move origin/main has and the branch lacks, by its move line. For five heads that is your two. For #8547 it is four [run: `restack.moves_on`]: c6a38b702d28 and 9776ae330bca (both `Move: tools/move/core_incorrect_map.toml`, rename.py) before the two layout moves. Your command names two, so c6a38b702d28 reaches #8547 by step 1's ordinary merge of b5354b9c2344, not by its script. #8547's own diff names none of that map's words (`RefineWrong`, `mem_wrong`, `exists_wrong_fine`, `card_wrong_le`: 0 lines [run]), so I ran your command as written; say so if you want it crossed by the default instead.
3. `--push-to` pushes a `locks-stale` result too (docstring; the brief lists it under the outcomes to stop on). If one comes it will be on my branch, unresolved, and I report it.
4. `--json` prints no outcome line: each entry gives `restack.line()` of the JSON, the tool's own function.
5. The README's command is `restack.py BRANCH --push-to NEW --kept`: `--onto origin/main` is its default, and `--no-pins` skips its step 4 (I check the pins per head instead).
6. Suites runs take `--changed` as origin/main's sha (the node may not have the ref), as the testing skill's pre-mark run does.

[2026-11-11T04:29Z] fd37163a61f8
CROSSB, NET-311, #8537 `cursor/net289-op-reads-b51f`: crossed clean. Tue 10 Nov, 8:30 PM PST.
1. Old head 6b793802ff2460306306ad92a40b440126a20477 (at the table's head [run: fetch 8:18 PM]). Result 74145a920b7f0dfa47dfd54a608aabddc9b56b41 on `cursor/net-crossb-8537-c2dd` [run: your command, 8:18 to 8:22 PM].
2. Tool's line [run: restack.py's `line()` on its --json]: `origin/cursor/net289-op-reads-b51f: clean 74145a920b7f  [plain merge clean; crossed 2 moves: 6581bcf197, c1ca25a247]`. Old head and origin/main 5c095bb4ab70 are ancestors of the result [run].
3. Own change's patch-id, `git diff-tree -r -p --full-index BASE COMMIT | git patch-id --verbatim` as `research.queue.own_change` [run]: before 23c4b50d374c1c0a8e607d25293fb10643b73d97 (merge base 5482988e3496 to the old head), after 0ae578f5a265169d31de0b2b51d649e2c4101f97 (origin/main to the result). 13 files, +465 -120, both.
4. ci's reason here [run]: one file, `Security/lean-audit.json`. Step 1's merge of b5354b9c2344 takes the patch-id from 23c4b50d374c to dde6883c3f51 in that file alone; the other 12 files' patches are identical across it. `main` changed the lock in 28 commits between 5482988e3496 and b5354b9c2344 (+416 -43), so the lock driver's merge sets the head's hunks among main's new entries: other context lines and hunk order, the same 116 added and 39 removed lines. Entry by entry the lock's own change is the same 70 JSON leaves with the same values before, after that merge and at the result. No later merge (the first move, 2eb7b4aa735f, origin/main) moves the patch-id.
5. What the two scripts changed in the own added and removed lines [run]: one line pair, `tests/test_lean_packages.py`'s `UNWITNESSED = {...}` keys ("Security" to "security", "Security/Properties" to "security/Properties", "verity/core/service/flock" to "verity/protocols/flock/verifier"). The other 12 files' lines are identical, under `security/` for `Security/`.
6. Retired names among the own added lines (`git diff origin/main...RESULT`): 0 hits [run]. Lock: `security/lean-audit.json` is 4,179,198 bytes at the result and was 4,179,131 where the first script read it, under 4,194,304 by 15,106; lines naming `Security/`: 6, origin/main's 6 (the same six entries, each a `WeightSecurity/` path); `verity/core/protocols/`: 0 and 0 [run]. Pinned inputs touched: 0 of `pins.PINS`' recorded files; `pins.py --stale` at the result exits 0 [run].
7. Runs, both `--on auto`, from a worktree at the result: suites `r20261111-042523-5024` (`suites.py --changed 5c095bb4ab70… --quick`, origin/main by its sha; 8 suites, the root suite `repository` among them [run: `--list`]); Lean `r20261111-042609-5626` (`lean_changed.py --try`, what `verity lean try --on` runs on a node, on each of the 10 changed `.lean` files in one job). Neither has ended; their verdict lines follow.
8. Not checked: the lock's records against a build of the moved tree (no `check.py`), the slow tier, circuit-check, and ci's dry result 83ca26bacfd5 (not in my clone).

[2026-11-11T04:31Z] 40b81fc3b855
From CROSS (NET-311 job 4), Tue 10 Nov 8:31 PM PST. **Job 4 is done on my side: the table is on NET-311 (one comment, 8:30 PM PST) and the note has job 4's section. I end my turn on four runs.**

- Runs, not waited on: `r20261111-040706-0481` (#8499; 15 suites passed, 0 failed at 8:29 PM PST), `r20261111-040706-dcf5` (#8708; 35 passed, 0 failed), `r20261111-042308-64fe` and `r20261111-042303-0a73` (#7109; both waiting for a slot at 8:29 PM PST) [run: `research fetch`].
- Note: `note:network-accounting/20261111T0300Z-report-net311-cross-job3`, from "Job 4" down; tools and outputs in `tools/net311cross/job4/` beside it. Notes synced at `99523a5fb9`.
- Heads on the remote at 8:30 PM PST [run: `git ls-remote`]: #8499 `2e961a204c4d`, #8708 `635b2f97027d`, #7109 `dbf5daaa36e1`; #8415 untouched at `720ef04b33`.
- One thing I saw and did not act on: #8539's head is now `984a73968ccd` (it was `4744e3e96c` when I crossed #8708). #8708 holds it at `353ab19093`, as you said to leave it.
- Least sure of: #7109's lock commit, until `0a73` ends.

