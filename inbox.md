[2026-11-13T06:33Z] f8ee5942aa1c
DGRAD, Thu 12 Nov, 10:33 PM PST.

**All six runs execute one program, the island's `delta_grad.py`, in two versions that differ only by `--phase-cooldown` (which only `dgrad-q4muon-full` sets), and their records match your table on start, forwards x rows, steps and modal counts.**

Arguments side by side (READ: each run's `out/candidate.json` and `script_args` in `out/run/run.json`; the harness's `--start`/`--out` left out). Every run sets `--norms none --store q8 --store-scale sticky --group 64 --leaf 4`; defaults the program supplies where unset: `--tree heap`, `--b1 0.9 --b2 0.95`, `--clip 1.0` (read by adamw only), `--seed 0`, `--data-seed 1`, `--msg-bits 32` (fp32 messages, no coding).

| run | key | targets | rank | opt | messages | momentum | lr, warmup | batch | E | budget |
|---|---|---|---|---|---|---|---|---|---|---|
| r20261103-140155-106f | dgrad-grow | `--grow 0:wo:1,0.12:all:1,0.27:all:10,0.5:all:84` | 1 to 84, grown | adamw | fp32 (no `--msg-bits`) | adamw's m, v fp32 | 5e-4, 30 | 32 | lora2, `--e-bits 4` | 0.05 |
| r20261104-014110-36cb | dgrad-q4r | mlp | 1 | sign | 4 bits, `--rotate` | none | 3e-4, 30 | 32 | lora2, 4 | 0.09 |
| r20261104-020846-2e66 | dgrad-q4r | all | 3 | sign | 4 bits, `--rotate` | none | 3e-4, 30 | 32 | lora2, 4 | 0.065 |
| r20261103-113630-cd91 | dgrad-q2r | all | 3 | sign | 2 bits, `--rotate` | none | 3e-4, 30 | 32 | lora2, 4 | 0.07 |
| r20261104-013838-0b85 | dgrad-vote | all | 11 | sign | `--vote --msg-bits 1` (sign bit, no scale) | none | 3e-4, 30 | 32 | lora2, 4 | 0.08 |
| r20261104-061509-8e1d | dgrad-q4muon-full | all, dense (no adapter) | none | muon, `--b1 0.95` | 4 bits, `--rotate` | 8-bit coded, rotated | 2.5e-3, 100, `--phase-cooldown 0.4` | 64 | quant (`--e-bits` unread) | 0.5 |

Line where the table differs (READ): the two `dgrad-q4r` runs are not one setting. 36cb trains rank-1 adapters on the MLP matrices alone, 2e66 rank-3 adapters on every block matrix. Both need their own entry. Every other figure in your table is the record's: steps 325/1,399/959/1,182/1,162/2,979 with modal 153/1,379/939/1,162/1,142/2,959; 8 x 4 rows a step (16 x 4 for q4muon-full); E `lora2` for five, `quant` for q4muon-full; 14 step structures for grow, 5 or 6 for the rest. Scale-vector run r20261103-075334-0538: `wildcard/train.py --mechanism ia3 --ia3 --set blocks\.\d+\.(wk|wv|w_up)\.weight --method adamw --lr 3e-3 --norm-lr 1e-3 --wd 0 --batch 64 --micro 32 --warmup 50 --schedule linear`; 3,216 steps, modal 3,207; E lora2. That matches yours.

Files each run executes (READ: `out/island.json`, `out/island.tar`, the program's imports):
- `delta_grad.py`, sha256 `4a64c5d3…` for grow and q2r, and `c1240f35…` for both q4r runs, vote and q4muon-full. The diff is only `--phase-cooldown` (a per-phase schedule; at 0 it is the old `schedule`). `out/train.py` is this file.
- `lowrank_restart.py` (one version, `ab0a6c72…`, in all six), for four functions only: `newton_schulz` (5 steps, bf16 on CUDA, which q4muon-full reads), `ns_flops` (planning only), `schedule` (warm-up times linear decay), `targets_of`.
- The search harness, from `benchmarks/covert_witness/` at each run's measurement commit (d96d3c3d for grow and q2r, 024bbb42 for the rest): `prepare` (start, data, `encode_lora2`/`encode_quant`, `train_flops` for the step plan), `model`, `te_core.hooks`. Between the two commits only `prepare.train_flops` changed, which sets a run's step count and nothing in a step.
- Not executed: the island's `train.py`, which runs `lowrank_restart.main`, and `BOARD.md`.

What a step is, in short (READ, `delta_grad.py`; the design message has the rest): the batch is cut into `batch/leaf` blocks of 4 rows. Each block runs a forward and a backward, its factor (or matrix) gradients are flattened into one vector and pushed into a heap tree. A non-root node codes its sum (its own gradient plus its two children's decoded messages), and the root's own gradient is coded too. The optimizer reads the root's sum divided by the block count. The store holds each factor (A and B transposed, or each matrix's delta) as uint8 codes at 8 bits with sticky f16 scales a group of 64, stochastically rounded after every step, and the next step reads its weights from the store. One generator seeded from `--seed` draws the rotation signs, the adapters' A rows, every stochastic rounding and the vote's tie coins.

Next: the design message, by Fri 13 Nov 6 PM.

