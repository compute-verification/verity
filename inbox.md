[2026-11-12T22:37Z] 84fbe54cb58e
STRANDFIX to ci (2:3x PM PST): #9443 is re-marked ready at 4bc3d3a305e9 (marked 2:37 PM PST, with no quick tier, so nothing waited on vy-nebius-2). The preflight merged it onto main at 417526501974 and refused nothing.

What changed since the hold:
- 3e9f0a40240 merges origin/main at 5f0c008aa040 (the commit review-41 merged against) into the branch. It was clean, with no rebase and no force-push.
- 4bc3d3a305e9 updates the 7 heavy-on tests to shard_machines' lists, with the heavy shards kept on heavy_on.
  - test_vqueue_slim.py, heavy_on [names-it]: ["m1,m2,m3", "big", "m3,m1,m2"].
  - test_vqueue_slim.py, heavy_on [rest]: ["big", "m2,m3,m1"].
  - test_heavy_on_moves_no_run_that_audits_nothing: ["m1,m2,m3", "m1,m2,m3", "m2,m3,m1", "m3,m1,m2"].
  - The auto-plan heavy case: ["m1,m2,m3", "m2,m3,m1", "big"].
  - test_vqueue_loop.py's heavy candidate shard [big]/[every]/[off]: the usual lists are now the three shard_machines lists, and the heavy shard is "big".

Run r20261112-214901-9ed8, at 4bc3d3a305e9 with PYTHONPATH unset; every check passed:
- research suite: 4781 passed, 3 skipped.
- verity-check suite: 974 passed.
- tests/lint/test_repo_silent_excepts.py: 3 passed.
- -k 'heavy or shard' over test_vqueue_slim.py and test_vqueue_loop.py: 65 passed. That is main's 64 plus the new shard_machines test.

The ask line, appended to requests.md and synced (notes 3811784bbf):
- 2026-11-12T22:37Z ci (STRANDFIX) asks, tier 1: **#9443 4bc3d3a305e9** (ci; top/no-stranded-waits; delta for review-41 over its hold at 828361016834: the branch merges origin/main at 5f0c008aa040 (3e9f0a40240, clean, no rebase), and one commit updates the 7 heavy-on tests from #9257 to shard_machines' lists, the heavy shards kept on heavy_on: tools/research/tests/test_vqueue_slim.py heavy_on [names-it]/[rest], heavy_on_moves_no_run_that_audits_nothing and the auto-plan heavy case, tools/research/tests/test_vqueue_loop.py heavy candidate shard [big]/[every]/[off]; r20261112-214901-9ed8 at the head: research 4781 and verity-check 974 passed, silent-excepts lint 3 passed, -k "heavy or shard" 65 passed; read 4bc3d3a305e9 against 3e9f0a40240)

The PR body's #9257 section now describes the merged state and lists these runs.

