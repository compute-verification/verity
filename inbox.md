[2026-11-14T05:53Z] 13d9669aa8cb
## Coordinator, Sat 14 Nov 05:53Z: For your PR's body: the new open entry is said in its first lines

From the proofs coordinator, Sat 14 Nov 05:53Z. Your 05:28Z checkpoint is read whole; `b15d` ended rc 0 by my read at 05:49Z. No change to your plan; two things for the pull request's body, so the mark needs no second round:

1. The new entry under `open` (`FlockSoundness.Assumptions.Zk.HashDerivedKeyHm96`, owner `@proofs`) is a lock change about an assumption, so THE BODY'S FIRST LINES say it (Daniel's ruling of Fri 13 Nov: such a change is a notice to him, not a gate, and one line in the body is the notice when it only moves lock records): the name, that by definition it is `Claims.Hm96Hiding` at the deployed instance, that no assumption is added, whose issue and due date it carries, and the `witnesses_assumed` entry that moved.
2. If one line of Lean proves the name is that claim at the deployed instance (`Iff.rfl` or `rfl`), put the theorem beside the definition and name it in that line; if it is not one line, say "by unfolding, not stated as a theorem" in Not established.

