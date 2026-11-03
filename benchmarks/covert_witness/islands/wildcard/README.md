# Island `wildcard`: training shaped against Slowdown's cut cost a and update profile u

The program of the training-efficiency search's `wildcard` direction. `train.py` is the one file it edits; `launch.py`
ships this island from the branch `cursor/te-wildcard-NNNN` and runs `train.py`.

## Mechanisms

- `--method adamw|muon`: the set trained in full for the budget, E the trained tensors (`encode_full`).
- `--method kfac`: closed form, one damped natural-gradient step per matrix from statistics gathered at θ.
- `--set`: `mlp-in:B`, `qkv:B`, `o:B`, `down:B` (B < 0 from the top), `norms`, `norms-after`, `final`, or a regex. One
  updated layer under a frozen tail is the u shape: a block doing the tail's share t of a token's affected work reads no
  update bits.
- `--cut B:<code>,...`: the residual stream at block boundary B (1..L, L the final norm's input) is handed on as a code
  during training, both directions, so the run witnesses the cut cost a its bits per token measure
  (`notes.info.cuts`). The codes are in `train.py`'s docstring and `parse_cut`. A fixed forward code's symbols and an
  `r` (Rice) code's bytes pass through a uint8 tensor, so a traced step sees the code's width only when the coded
  group is a multiple of 8 coordinates; an `e` code's length is computed, so a trace sees the uncoded state there.
  Code the boundary where the frozen tail starts: te_core pairs the narrowest cut with the tail, so a narrow lower
  boundary under a frozen top is the board's 05:46Z LOOPHOLE, not a witness.
- `p<keep>` (a `d` code) hands on a fixed subspace of the effect. The effect outside it is lost to training but not to
  the evaluator, so it needs `--confine`, which is exact only for a grown block (`--grow n`) or an existing block's
  `w_down` alone and refuses any other set; `--cut-leak` instead penalizes the outside share, and a trace then sees the
  uncoded state.
- `--lora-rank r --lora-fix B --z-bits q`: the narrow head cut (`headcut`), a rank-r update with seeded B whose effect is
  r coefficients handed on at q bits.
- `--ia3 --set 'blocks\.\d+\.(wk|wv|w_up)\.weight'`: IA3, a trained fp32 scale per output channel, E as
  `encode_lora2`'s magnitudes over zero factors (exactly diag(l)·W).
- `--grow 1 --adapter r [--adapter-z q]`: a bottleneck adapter as a grown top block with zero attention and r live MLP
  units. E is `encode_model`, which no snapshot encoder takes, so only the submission is scored; its narrowest cut lies
  below a trained matrix (`cut-below-update`), so read its ρ at the adapter output's width too.

Run: `python benchmarks/covert_witness/launch.py s-t20 --island wildcard --note "topcode ..." -- --set 'blocks\..*' --method muon --cut 6:dq2b4`.
The note's first word is the mechanism the scorer files the record under.
