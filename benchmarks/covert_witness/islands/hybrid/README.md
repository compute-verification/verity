# Direction `hybrid`: combinations of the other directions' mechanisms, in any class

The fixed launcher runs `islands/CLASS/train.py` and checks CLASS's caps, so on this branch (`cursor/te-hybrid-4207`) each
class's `train.py` is a link to this directory's `train.py`, which takes `--klass CLASS` (it must equal the launcher's).
`candidate.sh` copies the link's target into the record, so every record holds this script.

    python benchmarks/covert_witness/launch.py shallow s100 --note "MECHANISM ..." -- --klass shallow --full=-1:gu --lr 1e-3

`train.py`'s docstring lists the flags. The pairs it composes: full-rank tensors (AdamW or Muon) beside adapters, adapters
confined to the top blocks, every full tensor stored at bf16 through the `lora` format, norm gains above the lowest updated
matrix, loss-ranked window selection, and windows the start never read (`--fresh`). `--arms M1,M2,M3` packs a learning-rate
sweep into one candidate: each arm restarts from θ on the same data order with its share of the budget, and `notes.arms`
gives each arm's first step, so an arm's own MACs are its checkpoints' less the MACs before that step.

## What moves ρ

ρ = max(1, φcβ / (k(k + β))) when D ≥ k/β, else 1, with c = 4096 × the start's forward MACs per token (about 4.4e11 on m)
and β = p/r (p = 16 under bf16 autocast) unless a declared family covers the step.

- Upstream is the updated MACs per token before a node. Matrices in one stage (q, k and v; gate and up) are never upstream
  of each other, in the scorer's (layer, stage) rule and in te_core's cones alike, so an update confined to one stage has
  D = 0 there; gains above it add only their own few MACs, and ρ = 1 at every (k, r). A small adapter tail above the stage
  (rank 8 on the top block and the stage block's down, on m and s) keeps φcβ/(k(k + β)) under 1 too.
- On m every submission has ρ = 1 at k = 1 MB (φcβ ≤ 4.4e11 · 16 < k²), so the informative cells are k = 10 kB and 100 kB.
- With ρ = 1, W* is the least |C| at which a checkpoint reaches the target, so what's left to search is speed: the
  learning rate, the optimizer, the data, and how much of the model one stage plus a tail can move.
