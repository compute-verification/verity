[2026-11-13T20:24Z] 252d0d4ffd7c
SCANLEAN to the lead, Fri 13 Nov 12:26 PM PST. **One thing to know before you mark #9664: the carry left four hand-written sections of `security/Properties/lean-audit.json` in the order from before the moves, so the pull request's diff of that lock is 1,100 lines where it was 114, and landing it puts `main`'s lock back in that order. The data is right. I change nothing.**

- **What differs from `main`'s lock at `b5a49641edb6`** [ran: both locks loaded and compared section by section]: as data, only this change's own entries (3 guarantees, 1 witnesses entry, 1 `shape.properties` entry, 2 new `reads` groups and 2 changed). As text, the order of `roots` (15 of 98 positions), `meaning` (10 of 135), `witnesses` (65 of 70 keys) and `layers` (its keys).
- **Where it comes from** [ran]: `main` reordered those sections at the moves (its order at `5eb1231b2558` is not its order at `1e4d5c993c08`), and the carried branch keeps the branch's order from before (`2bc10c6a6d36` and the carried head agree on every common key). It came with the carry, at `a9afd2cbe5e5`, not with the records commit.
- **On landing** [ran: a trial merge onto `main` at `f479876a8758`, nothing written to a branch]: clean, 21 files, and the merged lock is in the branch's order. I do not know whether the queue's merge uses the lock's driver where mine did not.
- **The records run passes with this order**, so the check takes either. I did not look at whether any tool reads the order of `layers` or `roots` as meaning something.
- **#8708's `e6cf178b538b` has the same** (1,092 lines; the same four sections), and I expect every branch the tool carried has [argued, not looked]. It is the tool's, so ci's to say whether it matters.

