# Direction `placement`

Where an update sits sets how much of a training step's matmul work has two update-dependent operands, and its deep share
φ: one stage of one block with its input frozen, that stage with its lower-edge norm gain, LoRA factors at one stage, or
the input stage of every MLP. The search looks for placements that reach a gain target at small p while little of the
step's work multiplies two fresh operands, and records each one's (φ, |E|, bpb gain, training FLOPs).

`train.py` is the one trainer (harness v3, no classes); its arguments pick the set (its docstring), and a candidate is
`python benchmarks/covert_witness/launch.py START --island placement --note TEXT -- ARGS`.
