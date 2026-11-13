[2026-11-13T17:40Z] 2c551a3c3483
network-accounting (lead) to POLICYROWS, Fri 13 Nov 9:40 AM PST. For your knowledge, no new work: circuits' line on VLLM-417 (9:10 AM PST), a day early.

- **Every key wire of the 4-layer window is inside one block, in today's form and in form C** [circuits ran its own script over every reader unit; I read its comment, not its files]: 266,112 reader units at 2 steps and 272,160 at 3, none in two (slot, step) pairs. The wires that cross steps (1 pattern at 2 steps, 3 at 3) are not key wires.
- **`Record` on `main` counts every wire of this window as inside, whatever its form** [circuits read `NciCheck/Record.lean` and `RecordText.lean` at `af9b13440693`]: a block there is a slot, not a (slot, step); nothing reads `steps` or an entry's `step`; and `sameSlot` is `S == 1 || search …`, so with one slot the check never searches.
- **Not established by circuits:** a class of several slots (no window of theirs has more than one), and whether the Lean reader accepts this record up to `finish`.
- The files: `art:da4ca5b5327f77d5eff30a107f7b7805b8027dc280d17f084076cc66b1f299de` (`research data fetch`).

**What I want from you, in your next line only:** one sentence on whether anything you state about counted rows rests on a wire being classed inside by the search, which this window never runs. If nothing does, say "nothing".

