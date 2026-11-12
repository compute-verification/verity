[2026-11-12T21:14Z] 901a38e516a6
PLAN to training, Thu 12 Nov 1:25 PM PST. The grow-depth route's ten lines, the two answers on budgets, and one conflict in point 6.

**Point 6 conflicts with rule 1, so I kept point 6 without `budget.minutes`.** `budget` is in a config's digest (READ: `config.py`, `Config.run`), so adding `minutes` gives point 6 a new label. It would then not be the existing config's final/102 job, as your rule 2 wants. Point 6 fits the queue as it is: 1,805 steps at 238M take 1.74 hours at the plan's rate, 5.2 minutes a checkpoint, so a 60-minute lease, continued, loses at most 5.2 minutes a stop. Say if you'd rather have the minutes and the new label.

**The largest budget: nothing refuses one (READ).** Over the kind's `max_wall_min` of 60, the queue warns "asks for N min, over its max_wall_min of 60"; a GPU job over 60 minutes gets "a GPU job over 60 min: split it into chunks" (`infra/cluster/src/cluster/kinds.py`, `warnings`, `GPU_CHUNK_MIN`). Neither refuses, and I found no cap in `tools/research`. The lease is `ceil(1.2 × minutes) + 30` (`run.py`, `run_minutes`).

**What the budget doesn't change: a preemptible job is evictable after 40 minutes, whatever its budget.** An ordinary job of a higher rank may evict it once its grace ends, at `min(max_s, 2400 s)`; a quiet job or a node stop evicts at once (`plan.py`, `in_grace`, `GRACE_MAX_S`). Each eviction costs at most one checkpoint's stretch.

**Admission windows: yes, a node's window must cover the whole lease (READ: `plan.py`).** A job is refused when "now + its length" passes the node's admission end, its stop less the drain ("would run past NODE's admission end"), and its length is its `max_s`, the lease. Every job but a service also expires at the admission end. Under rule 1, point 1's lease is 30.5 hours and point 3's 47.3. Node 2's window ends today at 3:35 PM PST (my report), so today no node admits either.

**I touch neither `research_argv` nor `store_state`.** My branch changes configs, `of_record.toml` and `tests/test_entries.py` only.

**The grow-depth route, for its worker. Every line is READ from d64c's program, `out/train.py` in its record, unless marked.**
1. **What it computes.** The start is deepened from L to N blocks, block j a copy of θ's block ⌊jL/N⌋. Each repeated copy has its `wo` and `w_down` zeroed (`--zero 1`), so the grown model computes θ's function at step 0. d64c grew the 81M start from 8 to 12 blocks, copying θ's blocks 0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7.
2. **Its step.** Muon trains every block matrix (momentum 0.95, decoupled decay `--wd 0.05`, `--lr 0.008`). AdamW (betas 0.9, 0.95) trains the gains at `--adam-lr 3e-3` and the embedding at `--embed-lr 6e-3`, which d64c trained. The gradient is clipped at 1.0. The rate warms up linearly over 200 steps, then falls linearly to 0. The batch is 128 windows in micro-batches of 32, under bf16 autocast.
3. **Its output has another architecture, which is the route's real work.** The program hands out the grown model whole, every tensor not θ's bit for bit, in fp32. On `main`, a route's `output` is "the output model's state dict under θ's names", the job loads θ into `build`'s model, and it scores at the start's shape (READ: `routes/__init__.py`, `run.py`'s `Evaluator`). So the route needs a harness change outside `routes/`: θ copied into the grown model (no route reads θ itself), and the evaluator, E and the codecs at the grown shape.
4. **Its depth at 238M has no rule.** `--depth 12` is absolute, and the program refuses a depth at or under θ's blocks ("depth=… doesn't grow θ's 16 blocks"). So "the ported record's argv at the size" cannot run at 16 layers. The worker needs your depth: 24 keeps 81M's ratio of 1.5.
5. **Files.**
   - `training_slowdown/routes/grow.py`: `VERSION`, `DIALS` (depth, zero, lr, adam_lr, embed_lr, weight_decay, warmup, micro_batches), `plan`, `build`, `step`.
   - The harness's θ load, `Evaluator` and E for a grown model: `run.py`, `codecs`, `model`.
   - `attacks/m-t20-2x-hybrid-grow-depth.toml`, the 81M config held to d64c.
   - `attacks/l16-t20-2x-hybrid-grow-depth.toml`, point 4, with `honest = "l16-t20-2x-fullmuon-fe-wsd-40326"` (my draft's point 3).
   - `tests/test_entries.py` `REFERENCES`: a plain-torch reference that copies, zeroes and trains.
   - `tests/test_routes.py`: the contract, and that the grown model gives θ's logits at step 0.
   - `training_slowdown/of_record.toml`.
6. **The run it is held to: r20261104-060904-d64c, read from the store.** Its record is art:e18fcf3a8e2e2b43afdb8ee7a2d11dbfa83961bfe7dee51096b93387c2448e54. The record's file is `out/run/run.json` (59 MB, sha256 42d025b69edda49f0b7a4073db0f7828fce35004179cb3cc6ac42e8c331e6326), and the program is `out/train.py` in the same artifact (READ: `research data show` and `fetch --list`). The plan's `records.tgz` also holds its job.json and train.log.
7. **What the record computes and the route won't.** The program measures θ once, then 24 points of its schedule over 512 windows each ("in-job measurement (forward)" rows). Those are `record_only` lines in its `of_record.toml` entry, and the entry states `partial`, as every held entry does.
8. **The 81M config's steps: 7,713.** That is the record's own count; its program trained 7,688, plus θ's measurement and the 24 measurements (READ: train.log). Configs on `main` count 82a6 the same way: 1,805 against its program's 1,793.
9. **Point 4's four runs are the plan's runs 12 to 15 (READ: `PLAN.md`).** One search run (own seed, search split) and three final seeds (own, 101, 102), each on one card. The limit is 1.6 C0, \(8.49\times10^{21}\) gates in the plan's units (p max 1.43), at 29.2 GPU-hours a run and 116.7 in all, at hybrid's scaled rate. Its steps are that limit over the grown step's MACs at 1,480.7 gates a MAC, from the route's own trace at 238M.
10. **Its time (RAN: arithmetic on `plan.json`).** At the plan's rate, 5% of its steps take 87.3 minutes. Under your rule 1, its `budget.minutes` is about 38 hours, so its lease is about 46 hours: no node's window today covers it.

Topcode's lines come with the draft. The shallow traces are queued on vy-nebius-cpu-10.

