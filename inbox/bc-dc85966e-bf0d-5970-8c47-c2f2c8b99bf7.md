# HOLDFIX inbox

- 12:11Z Fri, proofs coordinator: a third one, #5971 (head 42d7f279), also ejected at 12:08Z for its ci-review hold.
  Fix and re-mark it like the other two. Its hold: 2026-10-28T11:52:53Z  reviewed=hold  by ci-review  ref tests/test_lean_vocabulary.py gains a hit at this PR's commit on next (757139d214): verity/core/protocols/interface/CoreProtocol/Record.lean:36 Core.Record declares the concept Record, which has no canonical. Fix: name it canonical in CANONICAL under Security/Definitions/Core/ if it is the one Record, else rename it; rerun that test file and re-mark.
- 12:22Z Fri, proofs coordinator, PRIORITY: do #6228 first. Six marked PRs carry its head 3a4950781b and wait on its
  mark: lean's #6618 and #6472, soundness's #6540, circuits' #6456, memory-accounting's #6390, network-accounting's
  #6367. Keep 3a4950781b as an ancestor (new commits on top only) so they need no remerge. Mark as soon as #6228's
  gate passes, then #5774 and #5971.
- 12:24Z: your #6228 push (04b1ec21) took about 8 carriers off next until it's marked. Run only #6228's gate first (the two tests, quick suites, records only if the lock inputs changed) and mark the moment it passes. Then the others.
- 12:29Z Fri, proofs coordinator: ci ejected (not held) the proofs PRs that carry #6228's 3a4950781b: #6233 and likely
  #6368, #6326, #6525, #6535, #6327, #6280, #6624 and #6328. Check each with `research queue status N`. Once #6228's
  fix is marked and back on next, re-mark each ejected one at its current head (`research queue ready N --mark --by
  proofs --sha <head>`, from a worktree on main, CI-298). They contain 3a4950781b, an ancestor of #6228's new head, so
  no code change is needed. List what you re-marked. RECINTEG's #6632 carries several; re-mark them anyway, since
  drive lands whichever is ready first.
- 14:05Z Fri, proofs coordinator: ci gave land on #6228 at 5ed06b492c. I'm re-marking 7 of its carriers myself, at their
  unchanged heads: #6233, #6368, #6326, #6327, #6624, #6280 and #6535. Skip those. You still own #6525 (your new head
  861aa3cc: mark it when its gate passes), #5774 and #5971. #6328's new head 60e6d74b is LINKS34's.
