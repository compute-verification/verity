# Direction `hybrid`: combinations of the other directions' mechanisms, in any class

The fixed launcher runs `islands/CLASS/train.py` and checks CLASS's caps, so on this branch (`cursor/te-hybrid-4207`) each
class's `train.py` is a link to this directory's `train.py`, which takes `--klass CLASS` (it must equal the launcher's).
`candidate.sh` copies the link's target into the record, so every record holds this script.

    python benchmarks/covert_witness/launch.py shallow s100 --note "MECHANISM" -- --klass shallow --full=-1:gu --lr 1e-3

`train.py`'s docstring lists the flags. The pairs it composes: full-rank tensors (AdamW or Muon) beside adapters, adapters
confined to the top blocks, every full tensor stored at bf16 through the `lora` format, norm gains above the lowest updated
matrix, and loss-ranked window selection.

## What moves ρ

ρ = max(1, φcβ / (k(k + β))) when D ≥ k/β, else 1, with c = 4096 × the start's forward MACs per token (about 4.2e11 on m).

- φ (the witness's deep share) counts changed matmuls strictly after the lowest changed (layer, stage). Matrices in one
  stage (q, k and v; gate and up) are never upstream of each other, and norm gains are 0-FLOP nodes, so an update confined
  to one stage, with any gains above it, has φ = 0 and ρ = 1 at every k. te_core's count agrees: no updated node lies in
  another's upstream cone, so D = 0.
- β read from E is 8·|E| over the updated MACs per token. f32 full tensors carry 32 bits per updated MAC, but the
  evaluator multiplies at bf16, so `--store bf16` halves β with the same evaluated matmuls.
- On m every submission has ρ = 1 at k = 1 MB (φcβ ≤ 4.2e11 · 32 < k²), so the informative cells are k = 10 kB and 100 kB.
