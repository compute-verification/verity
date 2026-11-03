# Direction `seed-mezo`

Zeroth-order training whose update is a seed log (`prepare.encode_seedlog`, CWSEED1), aimed at the training formalism
v2's Fit regime: the bottom-left of the algorithm-against-policy plot. `train.py`'s docstring has the options; `BOARD.md`
holds this direction's board lines. Launch from a committed tree:

    python benchmarks/covert_witness/launch.py s-t20 --island seed-mezo --node vy-nebius-1 --note "..." \
        [--long 'HH:MMZ who'] -- --names 're:^blocks\.5\.' --max-bits 34000 --lr-auto 1.0 --batch 512 \
        --label "..." --description "..."

## What a seed log costs under Fit

Each part of a run must fit a block of k bits (te_core.formalism `at`):

- Δ = 8|E|: the header (~130-160 B) plus one 16-bit coefficient a step (`--coeff bf16`), so `--max-bits` caps the
  steps. MeZO's step reads two forwards a token, so its s = c/w is ~2,048 tokens and sτ ~31 kbit at 32k vocab: Δ ≤ ~34
  kbit at the console default (k = 65,536), against ~45 kbit for a backward step's s.
- Δ_t = Δ: the update rule reads only the log (the fp32 masters are the decoder's, rebuilt from it; the subspace's
  coefficients are rounded stochastically to `--coeff`, so no fp32 shadow exists).
- Δ_grad: a step's two batch halves' L₊ − L₋ sums (128 bits) and the lr rule's three fp32 EMAs. te_core.capture
  measures 512 bits on a traced step (the EMAs and the warm's lr as carried state); with `--subspace K` it adds the K
  coefficients (16 bits each) as carried state.
- d: the decoder replays every entry over S, T·|S| fp32 FMAs at F32Fma_v4's 12,637 gates, against c = 4096 × the
  start's forward MACs a token at the bf16 GEMM price (1,480 gates). On s (41.98M forward MACs a token), d/c ≈
  T·|S| / 2.0e10, so at T ≈ 2,100 steps ρ_fit = c/(c − d) is ~1.1 at |S| ≈ 1M, ~1.5 at one block (3.2M), and all blocks
  (19.3M) don't fit. te_core 4.1 still takes d as 0 (a stub), so `notes.decoder` gives d and ρ_fit for each checkpoint.

## What limits the gain

A step along z with coefficient c = η ĝ changes the loss by −η ĝ (g·z) + η² ĝ² z^T H z / 2, and z^T H z ≈ tr H_S, so
the best η is G / (E[ĝ²] tr H_S) and a step buys G² / (2 E[ĝ²] tr H_S), G = |g_S|². `--lr-auto` sets that η from the
run's own EMAs (G as the halves' product ĝ_A ĝ_B, unbiased), with tr H_S from a three-point probe before the first step.

On s50 (the pilot start, under data v3's text) MeZO on block 5 (3.2M weights, batch 256) reached −0.08% by ~800 steps
(1.5 kB) and then held (r20261103-033255-2e57): the measured G fell from 1.45e-3 to 6e-4 as the steps descended the
components with the most curvature, the drain the v2 runs saw on one matrix. A larger S drains more slowly, since η
falls with tr H_S, but costs d. A larger batch raises the gain a step toward G / (2 tr H_S) (E[ĝ²] → G), at more
compute a step, so a fixed Δ buys most with the largest batch the compute ratio p allows.

## What te_core.capture reads

The scorer traces one step of the program (te_core.capture.script). Three things in train.py keep that trace true:
the clip is a scale of the step, not min/max (min/max returned the bound, so a clipped step read no estimate and the
trace saw no update at all); the lr rule's EMAs are fp32 device values (Python floats folded over the steps carried
every earlier estimate into the next step, 128 bits a step of Δ_grad); and the session's `current_step` is optional
(capture's session has none).
