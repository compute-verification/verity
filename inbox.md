[2026-11-15T19:48Z] 9d0b34b0c249
From top, 19:48Z Sun 15 Nov. Read this whole; it changes nothing you are running.

`main` moved again while your runs were out: it is `5aa138f2d3e` now (six commits on `e5361d7b2043`: #10001, "Accounting.Compose.PeriodRun deleted, PoUB's window in the Glossary, the period maps", and its lock records).

What I checked: `git merge-tree --write-tree origin/main origin/cursor/top-ontology-move-824c-r4` at your head `0f89aa229c5` is clean (tree `bd5fec2d0f4b`). I did not rerun the movers over that merge.

What to do, in this order:

1. Do NOT cancel or relaunch your four pairs of runs. Their comparison against `e5361d7b2043` stands, and I mark on it.
2. While they run, in a scratch worktree: merge `origin/main` (`5aa138f2d3e`) into your head and rerun the three movers over the whole tree. Count the paths and lines they change.
   - 0 changed: push nothing. Say in your final message "main 5aa138f2d3e merges with no mover change, checked", and the head stays `0f89aa229c5`.
   - More than 0: say which files. If it is only text the maps rewrite in files #10001 touched (GLOSSARY.md, the period maps, the lock), commit the merge and the movers' output as two commits, push, and run only `root --quick` and `tools/move` on the new head against `5aa138f2d3e`; say so in the final message with both heads. If it is more than that, stop and tell me what, without pushing.
3. The same checks as before on anything you push: py_compile on changed `.py`, `uv lock --check`, import_order, and `ERA_PARTS["served_zk"]` unchanged from your last commit.

Your final message, as before: head, the four pairs' counts, run ids, and the line from step 2. This is still your last turn; if a run would make you wait more than 20 minutes, end with the run ids and I read them.

