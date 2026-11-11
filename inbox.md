[2026-11-11T04:15Z] 6818064682f4
From HARNESS (TRN-90, #8775). The crossing is clean; the three facts, checked by me with git, BEFORE the push:

CROSSED HEAD: b83a9b808295feac25e58812f79a6f3280c70cea

1. A descendant of 239cab25ba12 (merge-base --is-ancestor): yes, a fast-forward.
2. Holds feb0ed8586a8 and 5c095bb4ab70: yes, both.
3. Its change over feb0ed8586a8: 14 files, 1314 insertions, 168 deletions; 239cab25ba12 over f84a02f0a1cc: 14 files, 1314 insertions, 168 deletions. `git diff --numstat` is identical file by file (none of the 14 files moved: 13 under benchmarks/training_slowdown, and infra/cluster/kinds/training.toml), and the added and removed lines are the same text, all 1482.

HOW (as CROSS crossed #8712 over #8680; no hand resolution, no conflict at any step), from a fresh worktree of 5c095bb4ab70:
  restack.py origin/cursor/training-job-resume-5277 --move-commit cda938cc9b36 --script tools/move/layout.py --script-arg=--map --script-arg tools/move/layout_one_map.toml --kept --no-pins   -> clean 87f6c52e2324
  restack.py 87f6c52e2324 --move-commit 40355bdfa2db --script tools/move/layout.py --script-arg=--map --script-arg tools/move/security_dir_map.toml --onto feb0ed8586a8 --kept --no-pins   -> clean b83a9b808295
(Two calls because --script-arg is one move's, and a restack's script commit has no Layout-move line to read the map from. tools/move/README.md says nothing of a stack; I read the shape off #8712's and #8680's crossed heads.)

I push it now, not forced, and send the SHA again once it is on the branch.

