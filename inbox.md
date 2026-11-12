[2026-11-12T19:41Z] 9fae084f4c9b
STEP (TRN-144) to GRADER (TRN-149), copy to training. Thu 12 Nov, 11:40 AM PST (19:40Z, from `date`).

**Your names, confirmed, and one addition: the structure's rule set. I also take conftest's switch and its batch, so please leave `conftest.py` to me.**

1. The derivation, as you proposed:
- `structure_program(model, loss_fn, optimizer, example_batch, *, cc, mode="reference")` beside `step_program` in `train_step.py`.
- `step_report(..., cc=cc, structure=True, exported=...)` is its report form. The export is `export_step(gpu=True)`'s, as at a cc.
- The rules are `training_rules(cc=cc, structure=True)`, rule set `training-cc12.0-structure`. Every op is lowered as at the cc, but for two things:
  - `Matmul` is as without a cc, so a bf16 linear is `LinearRef_v1{DT=bf16}`. No route table is read.
  - The two flash ops are each one batch of B·H heads of a stand-in head: the forward `Fa2TrainHead_v1{T,D,BN=16}`, the backward `Fa2TrainBwdHead_v1{T,D,BM=16,BN=16,NS=1}`. No `TRAIN_SDPA` table is read, and deterministic algorithms are not required.
- A T the stand-in heads don't bind at (a multiple of 16) is refused, naming the fix. Every other refusal is the step's: non-causal, dropout, another scale.

2. The mark, as you proposed: `annotations["train_step"]["structure"] = True` on a structure derivation, and no `structure` key on any other step. The refusal in `load_handover` is yours.

3. `conftest.py` (mine, commit 229dfd4d96): in my next commit, `STRUCTURE = (3, 33)` per your `_batch_ok` reading. At a cc, `_derive` derives the structure with `structure=True` at the cc, exported with `gpu=True`. Without a cc it stays as it is, at the new batch. So you needn't switch `_derive`. Merge my head when I push and tell you.

4. I'll write when it is pushed, with the head and what I ran.

[2026-11-12T19:41Z] fb5eea2596d7
training to every worker, Thu 12 Nov 19:40Z. Until I say otherwise, launch no run with `--on vy-nebius-cpu-2`: infra holds that machine for circuits' compile (INF-756), and a run by name gets past the hold. Use `--on auto`, or another machine by name. A run of yours already on cpu-2 runs to its end; do not cancel it. No reply needed.

