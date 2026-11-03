# Island `wildcard`: training shaped against Slowdown's cut cost a and update profile u

The program of the training-efficiency search's `wildcard` direction. `train.py` is the one file it edits; `launch.py`
ships this island from the branch `cursor/te-wildcard-NNNN` and runs `train.py`.

## Mechanisms

- `--method adamw|muon`: the set trained in full for the budget, E the trained tensors (`encode_full`).
- `--method kfac`: closed form, one damped natural-gradient step per matrix from statistics gathered at θ.
- `--set`: `mlp-in:B`, `qkv:B`, `o:B`, `down:B` (B < 0 from the top), `norms`, `norms-after`, `final`, or a regex. One
  updated layer under a frozen tail is the u shape: a block doing the tail's share t of a token's affected work reads no
  update bits.
- `--cut B:<code>,...`: the residual stream at block boundary B is handed on as a code during training, both directions,
  so the run witnesses the cut cost a its bits per token measure (`notes.info.cuts`). The codes are in `train.py`'s
  docstring and `parse_cut`.

Run: `python benchmarks/covert_witness/launch.py s100 --island wildcard --note "cutcode ..." -- --set 'blocks\.[45]\..*,norm' --cut 5:dq2eb2e,6:dq2eb2e`.
The note's first word is the mechanism the scorer files the record under.
