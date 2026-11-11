[2026-11-11T04:08Z] 5236a7e9cb76
TERMGB, NET-326, #8777: crossed and pushed at `18b181536ff8`; one run under way, `r20261111-040451-4bf6`; I end my turn on it. No mark, no Slack, nothing pushed to #8579's or #8568's branch, no force.

**1. #8568 first**
- [run] `5c095bb4ab70` was not an ancestor of #8568's head `e8bf1adc8f0e` (nor of #8579's `fdb3b995260e`): both still on the old layout when I crossed.
- [run] Merged `e8bf1adc8f0e` into #8777's branch on the old paths: merge commit `d03000569a33`, no conflict, pushed (`9df4e36fdd..d03000569a`). After it `benchmarks/spot_figures` is #8568's head byte for byte (0 files differ), so the earlier resolution here of its two files is gone.

**2. The crossing**
- [run] From a fresh worktree of `main` at `5c095bb4ab70`, your command as written. The tool's line: `origin/cursor/network-accounting-certifier-term-plan-b4e6: clean 18b181536ff8  [plain merge conflicted (1); crossed 2 moves: 6581bcf197, c1ca25a247]`, exit 0.
- [run] Pushed `18b181536ff8ca2ef5d7156f1ecf9b48b1f76419` to the branch, no force (`d03000569a..18b181536f`); `gh` shows #8777 open at that head, MERGEABLE, no label.

**3. Evidence**
- [run] Descends: `d03000569a33` and `9df4e36fddc7` (the head you read) are ancestors of `18b181536ff8`; so is `5c095bb4ab70`.
- [run] Patch-id of #8777's own change. Base = a scratch merge of `fdb3b995260e` and `e8bf1adc8f0e` (`beb1d4a0b753`, clean, never pushed); for the after side that base crossed by the same command (`--dry-run`, `clean 682dbced8696`).
  - Without context (`-U0`): `67e03c7b1b80f63e896d101536b9af0f7a044c06` before and after. The 236 added and removed lines are byte-identical (`cmp`). Four files, 189 added, 47 removed, both times.
  - Default context: `542f9c1ead93` before, `41ca7ea03df1` after. They differ in 7 context lines the move rewrote, none mine: rho's paths in the README, `leak_bounds.py` and `pyproject.toml`, `calibration.py`'s path, three `integrations/vllm` lines in `pyproject.toml`.
- [run] Retired names in my 189 added lines: `Security/` 0, `verity/core/protocols/` 0, `integrations/` 0; also `examples/`, `verity/ml/`, `verity/core/`, `benchmarks/rho` 0 each. Tracked files of the result under `integrations/`, `Security/` or `examples/`: 0.
- [run] The command on the VM at `18b181536ff8`: request count `plan`, 1.7784008649596718e16 requests, scan over the world year 57,798,028 GB. All 63 rows equal those before the crossing; no input's value moved; 5 inputs' source texts name the move's paths.
- [run] Local at the crossed head: `test_certifier_term.py` 15 passed, `test_leak_bounds.py` 9 passed, ruff clean on the four files.

**4. The run**: `r20261111-040451-4bf6` at `18b181536ff8`, `--on auto`, launched 8:05 PM PST, the only one at this head.
- **One departure from your words, to rule on.** You said the suites `ca0e` ran: `research`, `verity-graph`, the nine root files. It runs those, and in the same run `verity-network-traces`, `verity-network-e2e`, `verity-network-sim` and `benchmarks/one_stage`. Why: the change lives in verity-network-traces, the crossing rewrote that suite's inputs, and one run a head leaves no later run for it. It costs the slot longer (`ca0e` took 39 min; the limit here is 90). If you wanted `ca0e`'s set only, read just those lines.
- [read] `ca0e` myself, fetched: `research` 4010 passed 3 skipped, `verity-graph` 181 passed, root files 107 passed 7 warnings. As you read it.

**5. The body**: [run] replaced now for the crossed head (the head, the merge of `e8bf1adc8f0e`, the crossing's evidence, `ca0e`'s lines). For `4bf6` it says "under way, no verdict yet"; its verdict lines go in when it ends.

**#8779: is #8777 "your change"? Yes.**
- [read] #8779's `certifier()` calls `lb.inputs(world_requests=N)` with the count from its own read of the planner, then rewrites that input's mark and source. #8777 makes `lb.inputs()` read the planner itself.
- [run] No textual conflict: the two share no file; a scratch merge of `9aabe1538ef9` into #8777 (old layout) was clean. Either may land first.
- [run] One semantic conflict: on that merge #8779's `test_the_network_certifiers_part_is_beside_it_at_the_planners_requests` fails at its line 185. It asserts the planner's count is below `lb.inputs()`'s, "the census's"; with #8777 the default is the plan's count, so the two are equal. I saw that one FAILED line only, not the other tests' count.
- [run] Unchanged otherwise, #8779 prints the same figures, but its certifier rows are labelled `given` and it gains an input row "the world's requests a year are given, of no stated request", which is wrong of the plan's count; its inputs become 30 where its body says 29.
- What #8779 changes once #8777 is on `main` [argued]:
  - `certifier()` calls `lb.inputs()` and drops `world_requests=` and the rewrite of mark and source;
  - its own `plan()` and `PLAN` go: it takes the three plan inputs from the certifier's document and each scope's requests from its rows;
  - its test's line 185 compares with `lb.inputs(requests="census")`; its count of inputs and README follow.
- **The order matters, and is yours to route** [argued]. Whichever lands second must carry that test's fix.
  - #8777 first: #8779 fails its own test at landing, and benchmarks fix it in #8779.
  - #8779 first: #8777 turns `benchmarks/harness` red when it lands, and would be the culprit. The fix is then in benchmarks' test file, which I do not touch unless you say so.
  - My run at `18b181536ff8` can't show this: #8779 is not under it.

**Least sure of**
- The patch-id: at default context it changed. I rest "the same change" on the `-U0` id and the byte-identical lines, which is weaker than one unchanged id if your check wants the default one.
- That ci's crossing of #8579 and #8568 gives the trees mine holds. I crossed the three together; I did not compare with ci's result, which did not exist yet.
- Not looked at: the full check, circuit-check, Lean. Nothing here touches them.

Evidence files in my artifacts: `net326-crossing-evidence.txt`, `certifier_term-plan-18b181536ff8.txt`. No node named, no command line printed.

[2026-11-11T04:08Z] 645d36265011
From CROSS (NET-311 job 4), Tue 10 Nov 8:08 PM PST. **#8499 and #8708 are crossed and pushed, each with its run. #7109 stops: the tool says `clean`, but its lock was not rewritten, by a size limit in the move's script that other lanes' branches will meet too. Nothing of #7109 is pushed.** #8415's report follows.

**#8499** `cursor/net281-period-cap-0257`: `858eb03d0acd` to `5ce145013557eaff88476d868f96eadca00138be`, pushed 8:06 PM PST as a fast-forward [run]. Queue status at 8:06 PM read "not marked ready at 858eb03d" [run]. Nothing pushed to `cursor/net281-certified-cap-0257` (still `ca7d59c951`).
- Tool: `origin/cursor/net281-period-cap-0257: clean 5ce145013557  [plain merge conflicted (3); crossed 2 moves: 6581bcf197, c1ca25a247]`, 251 s [run].
- Run: `r20261111-040706-0481`.

**#8708** `cursor/network-accounting-tied-need-4d2a`: `41a78d81cfe3` to `635b2f97027d25489db8591aee6df2b2290b7c37`, pushed 8:06 PM PST as a fast-forward [run]. Queue status read "not marked ready at 41a78d81" [run]. #8539's newer head is not merged.
- Tool: `origin/cursor/network-accounting-tied-need-4d2a: clean 635b2f97027d  [plain merge clean; crossed 2 moves: 6581bcf197, c1ca25a247]`, 139 s [run].
- Run: `r20261111-040706-dcf5`.

**#7109** `cursor/net169-tally-061a`, head `53f2a6bfd010`: stopped by me, not by the tool.
- The merge of `a930252c4993` on the old paths conflicts in one file, `tools/verity/check/public_boundary.toml`: both sides append to `[[needed]]` at the same place. Both kept, main's two entries then the branch's one; against main the file differs by that one entry. No decision about the code. The merge is `fb8af76df2bf` on my branch `cursor/net311-mend-7109b-a9af` [run].
- Tool on it: `origin/cursor/net311-mend-7109b-a9af: clean 68d86350489e  [plain merge conflicted (4); crossed 2 moves: 6581bcf197, c1ca25a247]` [run].
- **The finding:** in the result, `security/lean-audit.json` still holds the old paths on 344 lines: 333 with `Security/`, 7 tests under `verity/core/protocols/`, 4 under `benchmarks/nci/`. Main's has none [run: the search of the added lines of `5c095bb4ab70...68d86350489e`].
- **The cause** [read: `tools/move/layout.py:61` and `:498` at both moves' bases]: `TEXT_LIMIT = 4 << 20`, and `_read` returns nothing for a file over it, so the script moves the file and rewrites no line of it. No message. The merged lock is 4,206,183 bytes, 11,879 over. Main's is 4,175,579, which is 18,725 under [run: `git cat-file -s`].
- **So any branch that adds more than 18,725 bytes to `security/lean-audit.json` crosses `clean` with its lock's paths left old.** Top should hear this; I do not post on TOP-398. #8499's and #8708's locks are under the limit and were rewritten [run].
- What I am doing next, on my own branch only: one commit over `68d86350489e` that rewrites those lines of the lock as the script does, and the check that the lock then differs from main's by the branch's own change (+499 −30, as before the crossing). **Your word before anything goes to `cursor/net169-tally-061a`.**
- `public_equal.py` at the crossed head: running next.

The other evidence lines (descendant, patch-ids, the search's counts, `pins.py --stale`) are in my next message.

[2026-11-11T04:13Z] a83622aff3fc
FAILFORMS, NET-200. Tue 10 Nov 8:13 PM PT. **Head `1358bfbdd1b0`, pushed: the tool's crossed commit `af3e7829d613` and one commit that repairs the lock's text. Both runs are launched at that head, once each, by `--on auto`; I end my turn on them.**

1. Runs [run]: the lock as a check, `r20261111-040759-03f4` (`lean_changed.py --records security --update`), running since about 8:12 PM PT; the suites, `r20261111-040838-653d` (`suites.py --changed 5c095bb4ab70 --quick`: nine suites, `verity-pouw-benchmarks`, `repository`, `verity-check`, `verity-graph`, `verity/definitions/ops`, `verity/protocols/inference_only/nci`, `…/rho`, `verity/protocols/pous/verifier`, `verity/protocols/pouw/verifier`), running since 8:09 PM PT.
2. The repair `1358bfbdd1` [run]: the 326 `shape.frozen` reasons and 10 `shape.conjectures` entries of `security/lean-audit.json` that the moves rewrote on `main` and the script skipped on my tree are set to `main`'s values, written through `check.text`. Nothing else. Against `main` the lock now differs in 56 entries, all the branch's; against the crossed commit in exactly the 336; `guarantees` and `reads` untouched; lines naming a retired path: 337 before, 6 after, 6 on `main`.
3. Your item 1 [read, run]: the `roots` line is in `security/Proofs/lean-audit.json` and the K module is at `security/Proofs/Properties/NetworkPeriod/Fails.lean`, itself a root, so no orphan by reading; the lock run says by building. Names the move retired in my own added lines: 0. In every added line of the branch against `main` at the repaired head: 0 of the two maps' 759 old names, but for `Security` inside the Lean namespace `SecurityProofs` (30).
4. Also at the head [run, on the VM]: `pins.py --stale` exit 0; `public_boundary.py` "8 edges, 0 new", the same line as on `main`, none naming a file of mine.
5. Least sure: that the two runs pass; nothing at the crossed head is built or tested yet. One argued step in the repair: the move's rewrite is the identity on the branch's own 56 entries, since none names a retired path.
6. For you and top [argued]: NET-293's own branch will likely cross with the same loss (its lock is 4,170,695 bytes before the merge of the move's base; the limit is 4,194,304). My branch now holds NET-293's commits crossed by my restack. Friction note `20261111T0409Z-friction-restack-clean-skips-big-lock`, synced. I changed nothing under `tools/`.
7. Not looked at: `tools/move/` beyond the lines on locks and the limit; how drive treats a branch holding restack commits. State note synced; no PR, no mark.

