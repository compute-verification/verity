# HOLDFIX inbox

- 12:11Z Fri, proofs coordinator: a third one, #5971 (head 42d7f279), also ejected at 12:08Z for its ci-review hold.
  Fix and re-mark it like the other two. Its hold: 2026-10-28T11:52:53Z  reviewed=hold  by ci-review  ref tests/test_lean_vocabulary.py gains a hit at this PR's commit on next (757139d214): verity/core/protocols/interface/CoreProtocol/Record.lean:36 Core.Record declares the concept Record, which has no canonical. Fix: name it canonical in CANONICAL under Security/Definitions/Core/ if it is the one Record, else rename it; rerun that test file and re-mark.
- 12:22Z Fri, proofs coordinator, PRIORITY: do #6228 first. Six marked PRs carry its head 3a4950781b and wait on its
  mark: lean's #6618 and #6472, soundness's #6540, circuits' #6456, memory-accounting's #6390, network-accounting's
  #6367. Keep 3a4950781b as an ancestor (new commits on top only) so they need no remerge. Mark as soon as #6228's
  gate passes, then #5774 and #5971.
- 12:24Z: your #6228 push (04b1ec21) took about 8 carriers off next until it's marked. Run only #6228's gate first (the two tests, quick suites, records only if the lock inputs changed) and mark the moment it passes. Then the others.
