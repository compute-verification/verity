# rho_attack

`rho_attack(S, k, c)`: the cheapest layout of a training run as inference-like blocks (each block at most `c` FLOPs and
at most `k` unregistered input bits, theta registered and free), searched over block strategies, against the bound's
`rho(S)`. `rho` is `c * blocks / honest FLOPs` (3f per trained token for SFT, 4f for GRPO, whose rollouts are
generated).

~~~sh
python3 rho_attack.py --inputs inputs/in_hand_rank64.json inputs/in_hand_fullrank.json --out OUT
~~~

writes, per input file, `rho_attack_<input>.json` (every row and curve), `rho_attack_<input>.md` (the table) and
`rho_attack_<input>_<precision>.png` (the curves). `--configs`, `--workloads sft grpo`, `--curve-points` and `--knobs '{"ranks": [16, 64]}'`
narrow or change a run. Run a whole grid as a `research run` on a CPU pod.

## Inputs

~~~json
{
  "source": "where the numbers come from",
  "measured_on": {"d_model": 7168},
  "F_scaling": "per_dim",
  "precisions": {"fp8_s1": {"beta_v": 3.1, "beta_x": 3.1, "beta_w": 0.23, "raw_bits": 6}},
  "F": {"10kB": 14000, "1MB": {"fp8_s1": 14000}, "full": 14000}
}
~~~

- `precisions`: per precision, `beta_v` (fresh bits per counted value), `beta_x` (per input coordinate) and `beta_w` (per
  weight of the update). Optional: `beta_g` (per interior output gradient, default `beta_v`), `raw_bits` (per coordinate
  sent with no theta baseline, default `2 * beta_x`), `b_acc` (per weight of a gradient accumulator, default `beta_w`).
- `F`: fresh bits per token crossing a block boundary, per S label, a number or one per precision. `F_scaling: per_dim`
  scales it by `d / measured_on.d_model` on each config; `absolute` doesn't.
- S labels: a size (`10kB`, `10kbit`, `1MB`), `tinylora` (26 bytes), `rankR` (a rank-R LoRA on every map at
  `lora_bits`), `full` (every weight at `beta_w`), or a name given in `S_bits`. `beta_by_S` overrides betas per label.

`inputs/in_hand_*.json` hold the in-hand numbers (the pythia-160m table, F full rank about 14 kbit per token at
d = 7168, or rank 64 at 4 bits per coefficient) until measured F(S) and beta(S) replace them.

## Strategies

The fixed recipe (exact layouts, `attack_exact`):

- `whole_state`: the state rides whole beside a batch (S < k). GRPO shares each prompt across the rollouts a block
  generates from seeds (duplicate sharing).
- `layer_pipeline`: the state split into contiguous stages of half-layers; each token crosses each cut forward (F) and
  backward (G, or the raw gradient); the head merged into the last stage or tiled.
- `tiling`: rows of each map's update against token tiles (the Loomis-Whitney k^2 count), priced as locality allows.
- `expert_local` (MoE): one routed expert's whole update per block with the tokens routed to it.

Recipe changes and approximations (`attack`, with the assumption of the finding's section 4 each breaks):

- `lowrank_pipeline`: rank-r corrections at each crossing; below F it is lossy (E) or beats the measured compressor (D).
- `linearized_tiling`: training linearized at theta, so every input is theta's own activation (C, cheap inputs).
- `frozen_prefix_tiling`: only the top layers train (C, cheap inputs).
- `hybrid`: a state under k carried whole for a share of the compute, then split; reported beside the bound averaged
  over the same phases.

A block holding fewer than a step's tokens (`step_tokens`) carries the step's gradient accumulator beside its state.

## Bound

`bound()`: `rho = O(1)` for `S < k` (never below the data floor `c * h / (k u)`); at the cliff the program's
`(2/3) * 2 * F / h`, times `eta = c / (k phi)`; for full training `c * beta_v * V / (k u)`, the covering count over the
counted values V. `c` is matched to traffic (`eta = 1`, `c = k phi`) or fixed at 1.4e17.

## Known omissions

Attention's score and value products ride free in `tiling` (attacker-favourable), and exact tiling doesn't assemble the
input gradients of residual-fed maps (attacker-favourable).
