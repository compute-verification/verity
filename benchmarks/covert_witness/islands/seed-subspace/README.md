# Direction `seed-subspace` (class `seed`, with `e100kb` for comparison)

Updates fully described by a seed plus a few coefficients, and the loss they buy per training FLOP. The seed-log decoder
replays Δ = −Σ_i c_i z_i over a named tensor set S, z_i ~ N(0, 1) from a generator seeded `seed + i`; E holds one
coefficient per direction, not per step, so a fixed set of d ≤ 24,962 (f32) or 49,951 (f16) directions can be trained for
as many steps as the budget allows.

`train.py` (run as `islands/seed/train.py`, a symlink, since the launcher runs `islands/CLASS/train.py`) has three modes:

- `subspace`: intrinsic-dimension training. θ_S = θ0_S + Σ α_i z_i over d fixed directions, Z cached on the GPU in bf16
  (d is what fits beside the model); the gradient is backpropagated to θ_S and projected onto Z; Adam on α. Two GEMVs a
  step are logged as `extra`. `--redraw N` (mechanism `resub`) splits an arm into N blocks of d fresh directions each,
  so it explores N·d ≤ the cap in the memory of d. Under the v2.1 harness only `--exact` steps are credited: Z is cached
  in fp32 and each step's weights are built by the decoder's own sequential adds, so they are decode(θ, E_t) bit for bit
  and the step declares E_t (`prepare.declare_step`); without it the steps read θ + bf16(αZ), ~0.3% of Δ off the decode.
- `fit`: train S itself (AdamW), then solve the minimum-norm c with −Zᵀc = Δ over d ≥ |S| exact directions. S must be
  small: the norm gains (6,656 on s, 13,056 on m) are the one set the seed log can carry exactly.
- `mezo`: zeroth-order, `--pop K` seeds a step (antithetic ES with shared seeds); the `seed-mezo` direction owns MeZO.

`--lrs a,b,c` runs one arm per learning rate from θ on the same data, each a share of the budget, every arm a snapshot
(charged cumulatively); `notes.arms` has each arm's own FLOPs. `--probe` (default on a GPU) records each tensor's mean
squared gradient at θ in `notes.probe`. The submitted model is always `prepare.decode(θ, E)`.

    python benchmarks/covert_witness/launch.py seed s50 --note "fit-norms ..." -- --mode fit --perturb norms --lr 1e-3
    python benchmarks/covert_witness/launch.py seed s50 --note "subspace ..." -- --mode subspace --perturb blocks --lr 3e-6
