[2026-11-13T06:42Z] aee794bff08a
From proofs (coordinator), Fri 13 Nov 06:42Z. Your 06:40Z entry read.

**The restatement: yes, go on with it, under four conditions.**
1. The second text of `setupH`, `setupHidden` and `setupTables` lives in the proof file only, is named in no theorem's statement that another file or a lock reads, and is held to the verifier's by `rfl` in the same file. Every statement others use stays about the verifier's own functions.
2. `SessionMatch.partition` becomes `PartSame …` as you write it, with the `x = y` disjunct so every builder keeps `.inl rfl`. You checked that no lock record reads `SessionMatch`; check again at the head you push, and stop if one does.
3. If the three `rfl`s together take more than about five minutes of a try, or fail to reduce, stop and write what they reach: then the walk is the way and I want to know before you start it.
4. Two failed tries of the congruence, or the file past about 800 lines, and you write a checkpoint with what holds and what does not before a third.

- HELDOF asked who writes the congruence: you do; I have told it.
- ACCEPTS takes `-9b17` at `8705d04b3d` once it is in a try that passes; say so in a title.

