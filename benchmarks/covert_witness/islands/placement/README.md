# Direction `placement` (classes `e1mb`, `e100kb`, `shallow`)

Where an update sits sets its deep share φ and its depth: norm gains only (the lf model has no biases, and its gains are
its per-channel scales, IA3 on the q/k/v, gate/up and head inputs), the tied embedding (the lookup and the head move
together), the EOT row (every document starts with EOT, so it is the per-document soft prefix), early blocks only, or
one stage of one block. The search looks for placements that lower search bpb at small witnessed φ and |E|, and records
each one's (φ, |E|, bpb gain, training FLOPs).

`train.py` is the one trainer; on this branch each class island's `train.py` runs it with that class, so a candidate is
`python benchmarks/covert_witness/launch.py CLASS START --note TEXT -- ARGS` as usual. Its arguments pick the placement
by regexes over θ's tensor names (its docstring).
