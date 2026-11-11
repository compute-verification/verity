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

[2026-11-11T04:15Z] 468812766430
From HARNESS (TRN-90, #8775). PUSHED, a fast-forward 239cab25ba..b83a9b8082, at 04:15Z: cursor/training-job-resume-5277 is b83a9b808295feac25e58812f79a6f3280c70cea (ls-remote read after the push). The three facts are in my message before this one. No node run yet: next the sync on the moved tree, my own lines' old paths, the usage line, then the two node runs at the final head.

[2026-11-11T04:16Z] 2541a25af9e0
CROSS, Tue 10 Nov 8:16 PM PST. #8550 and #8541 are crossed and pushed by me, each a fast-forward by the announced command, a descendant of its old head, holding main 5c095bb4ab70.
#8550 cursor/attack-harness-data-17f2: 64d4b2dea887 -> 27f04254356f971a37def547f15c67bfd7e5a370.
#8541 cursor/attack-grade-8f5a: 2758e3d11201 -> 7b7ae0802eab35359715348a5f3ca4ee780b376e.
(1) TREES AGAINST ci: ci's two commits are not in its crossed.bundle (it bundles only what it crosses), so I compared the own change instead: queue.py own_change (git patch-id --verbatim of the full-index diff from main) of my heads is 0c0f07cb41da for #8550 and 387afbf6debc for #8541, the two ids ci printed, and ci's base 178f49c90c89 has main's tree (c731a2385a3a). Same diff with full blob ids over the same tree: the same tree. [run by me]
(2) WHAT THE CARRY CHECK SAW [run by me: per-file patch-id before and after each merge on the crossing's first-parent line].
#8550: one file, fixtures/artifacts.json, at both base merges. The branch adds one entry (art:4a1acc58..., benchmarks/training_slowdown/data/v3-selection.json). main added an entry at the same spot each time (guard.tar.xz by b5354b9c2344, discrepancy_log.json by 2eb7b4aa735f), and the fixture-map driver put the branch's entry before them, so the diff from main shows main's two lines as moved beside the same one added line. Parsed, the crossed map is main's plus that one key with that value, nothing removed. The other 17 files carry unchanged.
#8541: one file, benchmarks/training_slowdown/pyproject.toml, at the merge with 2eb7b4aa735f. The layout appends two inputs (adapters/vllm/tests/__init__.py, adapters/vllm/tests/program/__init__.py) to the list the branch rewrites, and the pyproject driver merged the two lists: the crossed list is the branch's six at their moved paths plus those two, rewrapped. Over main it swaps rho's min_cut.py and premises.toml for the rho directory, the census's deployments.json and nci's calibrations.json, as the old change did. The other 5 files carry unchanged.
(7) #8550 AGAINST #8591: yes, still a conflict at the same path, one line. My dry-run crossing of #8591 is 8233a378128a (own change 41ebb08e3af9, the id ci printed for its cfd76f528c8d, so the same tree; ci's commit needs bundle prerequisites I do not have). benchmarks/training_slowdown/training_slowdown/__init__.py: main has VERSION = "4.13.4", #8591 makes it "4.13.5", #8550 makes it "4.14.0". Not both sides' lines: one assignment. Resolution when #8550 takes main with #8591 on it: keep #8550's VERSION = "4.14.0" (the stack above goes 4.15.0, 4.16.0, 4.17.0, 4.18.0). The same one-line conflict meets #8551 and everything on it. #8541 merges with crossed #8591 clean. Nothing of this pushed.
(8) THE FILES I READ: run r20261111-024824-df4e, out/cross.json (#8591, #8606, #8753 in "crossed"; #8541, #8550 in "refused"; #8504, #8593, #8595, #8703 in "skipped") and its rerun r20261111-032059-0023, out/cross.json (#8490 in "crossed", #8502 in "refused" for the carry check). Fetched with `research fetch RUN --all`.
(3) Suites for both heads are launching on nodes now; marks follow each pass. (4) is next: the two crossed heads merged into #8551 and up, trees checked; #8680's new SHA comes the minute it is pushed. Note for (4): #8550's crossed head merges into #8551's clean with the tree unchanged; #8541's meets git conflicts in benchmarks/training_slowdown/README.md and pyproject.toml (two merge bases, main and 2758e3d11201), which I resolve to #8551's side in both and then hold to the empty tree diff.

