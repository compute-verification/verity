# Island `sparse`: how much capability fits through a small update

The direction's question (night 2, formalism v1): on the fully trained starts, the least compute to each target with Δ in
each bin (≤ 1.8 kB, Fit's at the reference policy c = 4096 w, k = 10 kB; ≤ 10 kB; ≤ 100 kB; ≤ 1 MB), by any mechanism,
with Δ_grad for each. Night 1's: within E ≤ 1 MB and ≤ 100 kB, which parameters to touch for the most loss per training
FLOP, and how a mask's own encoding cost trades against its values.

`train.py` is the one script. At v3's pin: `launch.py START --island sparse --node NODE --note "MECHANISM: ..." -- --cap
BYTES --mech ...`, with `--cap` E's byte bin (required, since v3 has no classes). v2.3's records (the pilot) took the class
too, `--klass reference|e1mb|e100kb`, with `--cap` a bin below its cap.

| `--mech` | the mask | how E carries it |
|---|---|---|
| `norms` | whole tensors: the norm gains (`--norms all|final|top:N`) | `encode_lora`'s full tensors, bf16 |
| `lines` | whole rows and columns of block matrices, chosen by `--select` (`grad`, `fisher`, `adam`, `magnitude`, `random`) across the candidate matrices, so each layer's budget is learned | one-hot LoRA factors: row i is B = e_i, A = its values; column j is A = e_j, B = its values |
| `lora` | each candidate matrix's rank, greedily by the probe gradient's σ² per byte; with `--regrow F`, F of the bytes at θ and the rest after `--regrow-at` of the steps from a fresh probe, past each matrix's held rank (a learned mask) | LoRA factors, bf16 (a grown matrix's pieces as one pair at scale 1) |
| `qlora` | `lora`'s ranks at (d_in + d_out) · `--bits` / 8 bytes a rank | CWQUAN1 lora entries: `--bits`-bit midrise codes, A one f16 scale per row, B one per 64 rows |
| `signrows` | whole rows of block matrices by `--select` energy per byte (d_in · `--bits` / 8 + 2 bytes a row) | a CWQUAN1 delta over each touched matrix: the rows' codes (1 bit: a sign times the row's scale, mean \|V\|), every other row scale 0, which xz all but drops |

`qlora` and `signrows` train fp32 latents straight-through (the forward reads their codes, exactly what CWQUAN1 decodes),
and carry the gains (`--norms`) as `--norm-bits` deltas with one f16 scale per `--norm-group` entries, since θ is fp32
and a gain in full would cost 4 bytes. They size a plan by packing random codes as `prepare.encode_quant` would
(`quant_size`), because every encode during the steps is a checkpoint candidate; trained codes compress at least as well.

`--targets` picks the candidate matrices (`all`, `out`, `mlp`, `attn`, leaf names) and `--layers top:N` keeps every
touched tensor in the top N blocks, so the backward stops there. `--embed-rank R` (with `lora` or `qlora`) gives the tied
embedding a rank-R LoRA, (vocab + d) · R entries (about 102 kB a rank in bf16), placed before the block matrices'
allocation, or with -1 lets it join that allocation; E carries it as one factor pair on `embed`, and training applies it
in factored form to the lookup and the head (`tie_lora`), so the frozen head's matmul is unchanged. A probe (`--probe N`,
4 batches by default) takes the mean gradient at θ, priced in the step log as a forward pass plus the full backward
(`extra`). With `--analyse`, a record's `notes.first_order` holds, for budgets of 10 kB, 100 kB and 1 MB, the share of
‖G‖² each encoding captures.

Every record's `notes.delta_grad` gives N, the values the optimizer trains, and 12N bytes: the gradient and AdamW's m and
v in fp32, what passes between token blocks. A code shrinks E and not Δ_grad, since its latents train in fp32.
