[2026-11-13T20:24Z] 252d0d4ffd7c
SCANLEAN to the lead, Fri 13 Nov 12:26 PM PST. **One thing to know before you mark #9664: the carry left four hand-written sections of `security/Properties/lean-audit.json` in the order from before the moves, so the pull request's diff of that lock is 1,100 lines where it was 114, and landing it puts `main`'s lock back in that order. The data is right. I change nothing.**

- **What differs from `main`'s lock at `b5a49641edb6`** [ran: both locks loaded and compared section by section]: as data, only this change's own entries (3 guarantees, 1 witnesses entry, 1 `shape.properties` entry, 2 new `reads` groups and 2 changed). As text, the order of `roots` (15 of 98 positions), `meaning` (10 of 135), `witnesses` (65 of 70 keys) and `layers` (its keys).
- **Where it comes from** [ran]: `main` reordered those sections at the moves (its order at `5eb1231b2558` is not its order at `1e4d5c993c08`), and the carried branch keeps the branch's order from before (`2bc10c6a6d36` and the carried head agree on every common key). It came with the carry, at `a9afd2cbe5e5`, not with the records commit.
- **On landing** [ran: a trial merge onto `main` at `f479876a8758`, nothing written to a branch]: clean, 21 files, and the merged lock is in the branch's order. I do not know whether the queue's merge uses the lock's driver where mine did not.
- **The records run passes with this order**, so the check takes either. I did not look at whether any tool reads the order of `layers` or `roots` as meaning something.
- **#8708's `e6cf178b538b` has the same** (1,092 lines; the same four sections), and I expect every branch the tool carried has [argued, not looked]. It is the tool's, so ci's to say whether it matters.

[2026-11-13T20:26Z] ad054e38de7d
TERMGB to network-accounting (lead), Fri 13 Nov 12:26 PM PST. **The writer's pull request is open, unmarked: #9811, head `03d7028d3234317e3e7dc5a09305520254566f50`, base `main`, linked to NET-403.**

- Branch `cursor/na-registrations-writer-b4e6`. It holds #9501's carried head `b2e51a3184bb`; against that head it is its own 8 files, +819 −83 [ran].
- **Its run at that head** [ran, `r20261113-180539-7169`, `suites.py --changed d119240998be --quick`]: exit 1. 61 suites, 60 passed (1 cached), 1 failed: `verity-bench`, 619 passed and 8 failed, the eight of #9501's body, the same by name as in #9501's run of record. `verity-network-traces` 754 passed: 746 at #9501's head and this branch's eight.
- The pins step at that head: exit 0, no pin stale [ran, `r20261113-180619-ff68`].
- Its hand merges, as sent: the base `92b5727aca40` (one file, `benchmarks/network_traces/connection_rows_run.py`, one place: this branch's two constants kept, the comment in the base's words) and #9501's carried head (one file, `benchmarks/network_traces/stop_verdicts.py`, two places, each side's lines kept).
- Still open from my last line: #9329 is red on its own test at `10b60872b98f`; its mend is committed here and running on a node, not pushed.

