# Island `sparse`: which parameters a capped update touches (classes `e1mb` and `e100kb`)

The direction's question: within E ≤ 1 MB and ≤ 100 kB, which parameters to touch for the most loss per training FLOP,
and how a mask's own encoding cost trades against its values.

`train.py` is the one script; `islands/e1mb/train.py` and `islands/e100kb/train.py` on this branch are links to it, so the
harness screens and runs this file: `launch.py e1mb|e100kb START --note "MECHANISM: ..." -- --klass e1mb|e100kb --mech ...`.
The fixed decoders are `full`, `lora` and `seedlog`, so a mask rides in one of them:

| `--mech` | the mask | how E carries it |
|---|---|---|
| `norms` | whole tensors: the norm gains (`--norms all|final|top:N`) | `encode_lora`'s full tensors, bf16 |
| `lines` | whole rows and columns of block matrices, chosen by `--select` (`grad`, `fisher`, `adam`, `magnitude`, `random`) across the candidate matrices, so each layer's budget is learned | one-hot LoRA factors: row i is B = e_i, A = its values; column j is A = e_j, B = its values |
| `lora` | each candidate matrix's rank, greedily by the probe gradient's σ² per byte; with `--regrow F`, F of the bytes at θ and the rest after `--regrow-at` of the steps from a fresh probe, past each matrix's held rank (a learned mask) | LoRA factors, bf16 (a grown matrix's pieces as one pair at scale 1) |

`--targets` picks the candidate matrices (`all`, `out`, `mlp`, `attn`, leaf names) and `--layers top:N` keeps every
touched tensor in the top N blocks, so the backward stops there. A probe (`--probe N`, 4 batches by default) takes the
mean gradient at θ, priced in the step log as a forward pass plus the full backward (`extra`). Every record's
`notes.first_order` holds, for budgets of 10 kB, 100 kB and 1 MB, the share of ‖G‖² each encoding captures: LoRA ranks;
rows or columns as one-hot factors or with a list index; single entries with an entropy-coded, int32-list or bitmap index
at 16- or 8-bit values, and the index's share of the bits.

Under the harness (v2.1) every E encoded during the steps is a checkpoint candidate, the first in each 5% of the planned
steps (`hooks.plan`, the probes' batches included), so the script encodes E at the end of each 5% and sizes its plans
without encoding.

The one-hot encoding spends d_out (row) or d_in (column) bf16 entries on each line's mask, so at equal bytes LoRA of the
same rank contains every line update; a cheap index needs a sparse decoder, which the board asks the toolsmith for.
