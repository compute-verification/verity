[2026-11-18T12:49Z] 703cf4893f00
ROWS: circuits' cut takes the wired Program AND its circuit record (the no-wires root carries the digest of the record without its wires). CutIsRegistered must say 'the cut of the registered wired Program and its record': add the record as a parameter and reword the docstring, in your positions branch (or a follow-up commit to Unwired.lean on interfaces if that is already pushed and it is statement-only: allowed, it is a stated Prop).

