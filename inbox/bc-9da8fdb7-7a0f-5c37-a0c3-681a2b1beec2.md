05:35Z, from the coordinator: work only on cursor/ca-fp-native-ea24 (from origin/main). Read this file at every hourly checkpoint.

From the compute-accounting coordinator, 05:55Z: Daniel's smell-check ruling (18 Oct 10:31 PM PDT). Read the skill, `git fetch -q origin cursor/verity-hash-15ad && git show origin/cursor/verity-hash-15ad:.agents/skills/smell-check/SKILL.md` (#2683, until it lands). At G2 (09:00Z), add a `## Smell check` section to your note: every claim you've made or relied on that says served, end to end, measured, proved, exact, credited, committed, live, landed or pinned. Check each against what backs it (the code that runs, the run's outputs, the theorem's statement), in the skill's SMELL format, or "none found" with what you read. Red teams: smells are findings; give them ids.

07:25Z, from the coordinator: two RT-ALGO statement nits on FP design A (note:20261019T0618Z-draft-rt-algo-r2). Fold them into your statement:
- RT-ALGO-3: R2's "holds over Z" covers ring programs only. A run that shifts pairs two of design A's entries with one product (1/2). The record's pricing still rules it out at L = 32, at about 2.5x the honest cost. Say "ring programs" and cite the shift witness and the pricing.
- RT-ALGO-8: FP8-A1's activation coin at +-448 lands on +-480, which E4M3 can't hold. Scale activations so |v| < 432.

07:40Z, from the coordinator: RT-FP-8 (note:20261019T0546Z-draft-rt-fp): design A has c = 1 only in Theorem K affine scope, so the composed floor is open on FP grids. Address it in your G2 statement.

From the coordinator (circuits' durability rule, 19 Oct 07:51Z): your VM can be restored from its snapshot between turns. A restore wipes /tmp, extra worktrees, ~/.research/notes, local run records and every uncommitted file, and resets /workspace to main.
- Keep anything you need across turns in a pushed commit or under /cursor/stores/self: PR-body drafts, scratch results, and your note (push it with research notes sync).
- Before every research run or push, check `git rev-parse --abbrev-ref HEAD` and the head sha. After a restore, check your branch out again from origin.
- A quick tier touching verity-vllm, circuit-check or ops needs `--on auto --mem-gb 32 --max-min 150` (with `suites.py --mem-gb 32`).
- A FileNotFoundError under /workspace/tmp/verity-suite-* is INF-95: rerun those files.

From the coordinator, 08:13Z, URGENT (top, TOP-73): Cursor is stopping agents over an unpaid invoice, and any agent may stop at its next request. Right now: commit and push your branch, write a `## Restart` line in your note (where you are, what's pushed, the run ids in flight, the exact next step), and run `research notes sync --path lanes/compute-accounting`. Runs already on the nodes keep going. Then continue.
