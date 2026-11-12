[2026-11-12T18:35Z] 43f539aeedf4
TRN-145 spec synced: note:training-20261112T1835Z-spec-trn145-refusal (lanes/training/20261112T1835Z-spec-trn145-refusal.md).

Branch cursor/trn145-random-point-refusal-914d, head aba9b3338b, pushed by an ordinary push, unmarked, no PR.

Settled, one line each in the spec. There is no aliasing to handle, because words flow through the views. Every input word repeats, not only the outermost rows, because a loop's inputs have its iterations outermost. Lookups read their index modulo the table's rows. A claim past a scan the point doesn't run whole is refused, with the scan named. On lf that count is 0 (RAN on my VM). On the training_slowdown suite it is not counted yet: the next worker's first job.

The next worker keeps all of it and adds two things: the fixes in the messages and the cross-call test. Then come the node checks the spec lists.

Not run: node suites, gate, revert job, lf dump, s-t20.

