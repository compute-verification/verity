# Island `sparse`: how much capability fits through a small update

The direction's question (night 2, formalism v1): on the fully trained starts, the least compute to each target with Δ in
each bin (≤ 1.8 kB, Fit's at the reference policy c = 4096 w, k = 10 kB; ≤ 10 kB; ≤ 100 kB; ≤ 1 MB), by any mechanism,
with Δ_grad for each. Night 1's: within E ≤ 1 MB and ≤ 100 kB, which parameters to touch for the most loss per training
FLOP, and how a mask's own encoding cost trades against its values. Since 03:00Z on night 2 (TRN-40), the plot is ρ at
the page's policy against p = C/H for each gain size. Under te_core 4.1 a record whose Δ = 8|E| is at most k scores ρ = 1
(8,192 B at the console default's k = 65,536 bits), and ρ climbs toward the slowdown floor as Δ passes k, so the bins
are multiples of k.

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
| `entries` | single entries of the candidate matrices and of the gains `--norms` names: k of them, each tensor's share in proportion to its size, and within a tensor the ones with the most `--select` energy times its learning rate squared (`torch.topk` on the device, read back by nothing); k, set before any value is read, is the most that fit `--cap` bytes wherever they fall (`entries_bound`) and at most `--values` | CWSPAR1: varint index gaps or a bitmap and `--entry-dtype` (f16 or bf16) values |
| `bitfit` | a new bias vector on every block's `--bias` matrices (`all`, `out`, `mlp`, `attn` or leaf names), zero at θ, with the gains `--norms` names; `--layers all` only, since every bias differs from θ's absent one | CWMODL1: the `lf` config with its `bias` key, the biases in `--entry-dtype` and the trained gains in θ's fp32 |

`qlora` and `signrows` train fp32 latents straight-through (the forward reads their codes, exactly what CWQUAN1 decodes),
and carry the gains (`--norms`) as `--norm-bits` deltas with one f16 scale per `--norm-group` entries, since θ is fp32
and a gain in full would cost 4 bytes. They size a plan by packing random codes as `prepare.encode_quant` would
(`quant_size`), because every encode during the steps is a checkpoint candidate. Trained codes can come out larger than
that plan (a CPU smoke's 2-bit factors: 30,532 B against 29,704 B), so leave room below a bin's edge. `entries` adds a
matrix's values to its weight in the forward (no extra matmul) and trains a gain's chosen entries under a gradient mask.

`--targets` picks the candidate matrices (`all`, `out`, `mlp`, `attn`, leaf names) and `--layers top:N` keeps every
touched tensor in the top N blocks, so the backward stops there. `--embed-rank R` (with `lora` or `qlora`) gives the tied
embedding a rank-R LoRA, (vocab + d) · R entries (about 102 kB a rank in bf16), placed before the block matrices'
allocation, or with -1 lets it join that allocation; E carries it as one factor pair on `embed`, and training applies it
in factored form to the lookup and the head (`tie_lora`), so the frozen head's matmul is unchanged. A probe (`--probe N`,
4 batches by default) takes the mean gradient at θ, priced in the step log as a forward pass plus the backward that
takes the probed tensors' gradients (`extra`). With `--analyse`, a record's `notes.first_order` holds, for budgets of 10 kB, 100 kB and 1 MB, the share of
‖G‖² each encoding captures; with `--no-analyse` the probe reads only the candidate tensors (`--targets none` probes
just the gains), and `notes.probe` gives its uncoded fp32 gradient's width, a reduction of its own.

`--grad-code vote` (with `entries`) passes one bit a value between token blocks. A zero perturbation P[b] of the entries
rides in each sequence's forward (`entry_forward`, and `vote_gains` for gain entries, a mirror of `Block.forward` and
`LM.top`), so P's gradient is each sequence's own gradient, at one product an entry a token and no weight gradient. Its
sign leaves as a bool, the step sums the bools as int16 votes, and each value moves by its lr times the votes' mean sign
(`--vote mean`) or the majority's. Values live and update in `--entry-dtype` with stochastic rounding, so there are no
latents and no optimizer state. A CPU check on tiny0 gives the patched logits equal to θ with the values merged, and P's
rows equal to each sequence's gradient (relative difference 1e-7). te_core 4.12.1's capture prices each sequence's sum
over its tokens at 32 bits, so a vote's Δ_grad is at least B · N · 32: s-t20's gains-only vote (1,666 entries, with
`vote_gains` forming every gain's per-sequence gradient) measures 13.6 Mbit = 64 · 6,656 · 32, and a vote never meets
Fit's reduction bound. The notes give `grad_bits` at the vote sum's width, ceil(log2(B + 1)) bits a value, with
`partial_bits` and `floor_bits` beside it. The counter sees no weight gradient, so a matrix with entries is declared as a
rank-1 adapter.

`--opt` is `adamw` (the default), `sign` (sign descent) or `sgd` (no momentum); the last two keep no state, so only the
gradient passes between token blocks and Fit's reduction bound, 2Δ_grad ≤ k, holds up to k/64 trained values (s-t20's
final norm under `--mech norms --norms final --opt sign`: Δ_grad and u(1) 16,384 bits at te_core 4.12.1, state 0). Every
record's `notes.delta_grad` gives N, the values trained, and what passes between token blocks: the fp32 gradient of every
trained tensor (a masked gain's whole tensor), plus AdamW's m and v under AdamW (12 bytes a value in all). A code shrinks
E and not Δ_grad, since its latents train in fp32 (`latent_bits`, apart).
