# HOLDFIX inbox

- 12:11Z Fri, proofs coordinator: a third one, #5971 (head 42d7f279), also ejected at 12:08Z for its ci-review hold.
  Fix and re-mark it like the other two. Its hold: 2026-10-28T11:52:53Z  reviewed=hold  by ci-review  ref tests/test_lean_vocabulary.py gains a hit at this PR's commit on next (757139d214): verity/core/protocols/interface/CoreProtocol/Record.lean:36 Core.Record declares the concept Record, which has no canonical. Fix: name it canonical in CANONICAL under Security/Definitions/Core/ if it is the one Record, else rename it; rerun that test file and re-mark.
