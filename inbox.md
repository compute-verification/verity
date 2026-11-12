[2026-11-12T19:41Z] 1219f11d473b
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

[2026-11-12T19:41Z] c79f6f82daa2
SPLIT, Thu 12 Nov, 11:41 AM PST

**No published training figure reads a wrong split. Every row of the four panels, served and staged, and every row of the 1,497-row search table comes from te_core's counter, which counts the rotary tables as θ in all 957 records it ever wrote. The library was built with training_slowdown 4.25.0, which already contains the fix.**

Below are the paths (item 1) and the counts so far. The hand-follow of one row has also been done. The full report note follows by 5 PM.

**Who counted.** All four panels and the search table come from te_core (covert_witness; VERSION 4.13.6 on `cursor/training-put-55a1` at d76ae46e6f), not from training_slowdown. Only the library comes from training_slowdown.
- READ `te_core/cw.py` `Adapter.bind` registers `net.named_buffers()` as `buffer/<name>` in `Session.bind(..., extra)`.
- READ `te_core/run.py` `Session.bind` sets `counter.theta_cpu = theta | extra`. So the cos/sin tables are θ, and their products count `fwd/theta`. The docstring says "so RoPE tables aren't weights". This has been in cw.py since its first commit, 996a5eab7e.

**Scan of every record (RAN r20261112-193102-3134 on vy-nebius-cpu-7):**
- It read all 957 records the te-scorer state holds, with 0 unreadable. te_core versions: 2 to 3.20. All are script mode through the covert_witness adapter.
- In every record, every rope product ([T, hd/2] table × [.., T, hd/2] activations) is logged kind `theta`. That holds for each table's runs: search 600/600, overlay 394/394, algorithm-policy bodies 164/164 (served and staged), standard-workloads 16/16, and size-scaling 6/6 attack records.
- 0 records have any setup or tail `fwd/updated`.
- No `new/*` weight has a table's shape. The 2-D fresh tensors are adapters, e.g. [2,512].

**Path 1: algorithm-policy body and search table (te_core.algorithm_policy, through scorer/replot.py and correct.sh).**
- The steps: record → `trace40.one` (picks the traced step by `regions_of`, traces it, then `ledger.ingest(structures=…)`) → `annotate.with_structures` → `formalism.of_record` → `formalism.measure`/`complete` → `formalism.published(m, own_session c, OWN_K)` → `rho_own`.
- ρ prints only where `frontier.measured` holds: a trace, and `annotation.untraced` = 0. Otherwise the cell is null.
- READ: the counter's `updated` is read here only as a gate on which steps are update steps:
  - `trace40.one`'s structures map
  - `with_structures`'s `if not struct["updated"]`
  - `annotate.summary`'s `untraced`
- A traced step's affected gates, σ, token factor and Δ_grad come from the trace. |C| (C, H, p) is priced by op and dtype in `gates.of_log`/`kind_of`, independent of the split.
- READ: the untraced-record stub `sum(u[2] for u in st["updated"])` and `formalism._v23_part` (`fwd/updated`) feed no printed cell. Those rows are refused or unmeasured, and ρ is null there.
- READ: counterexample's family β goes into `families`, which no builder reads.

**Path 2: overlay (`scorer/plot2_final.py`).** Every ρ is the search row's (path 1). The finals reruns supply only |C| (C, H, p). READ the docstring and the code.

**Path 3: standard-workloads (`islands/standard-workloads/standard_workloads.py`).** 16 rows are measured through stored traces of their records, by the same of_record/measure path. 2 rows (rl 45M, inference 45M) are modelled, with no record and no counter. READ.

**Path 4: size-scaling (`islands/ladder-v3/size_scaling.py`).**
- 45 standard rows are traced at full size on fake tensors by te_core.capture. Its `capture/structure.py` treats a buffer as dependent only if the run writes it, so unwritten tables are registered. READ.
- 12 attack rows take ρ from the algorithm-policy table (path 1).
- 33 rows are null.

**Path 5: frontier-fit.** It reads only |C| and loss. READ.

**Path 6: library (main, `training_slowdown.library`).**
- It traces each config's step on fake tensors (`traced.trace_step` → `calculator.slowdown`) and reads no record and no counter output. READ.
- Built at 7a560a0e7dc (4.25.0), which contains the fix's merge ab79fb9c. READ (`git merge-base --is-ancestor`).
- 44 entries: 35 slowdowns and 8 "needs a record" in r20261112-171938-2636, plus one xl entry in r20261112-172004-f817. READ.

**Counts by kind** (taken from the records the rows name; READ the tables, RAN the scan):

| table | rows | ρ through a trace with a circuit | traced, ρ unmeasured | trace refused / failed (ρ null) | record with no trace, ρ printed | family β |
|---|---|---|---|---|---|---|
| search, staged and published | 1,497 | 1,305 | 27 | 162 / 3 | 0 | 0 |
| overlay, staged and served | 458 | 289 | 4 | 162 / 3 | 0 | 0 |
| algorithm-policy bodies, staged and served | 197 | 165 | 1 | 28 / 3 | 0 | 0 |
| standard-workloads | 18 | 16 | 0 | 0 | 0 (2 modelled, no record) | 0 |
| size-scaling | 90 | 45 fake-tensor traces + 12 from the policy table | 0 | 0 | 0 (33 null) | 0 |
| frontier-fit | 46 | reads no ρ | | | | 0 |
| library | 44 | fake-tensor traces, post-fix | | | | 0 |

- Staged and served differ only in standard 20M (467,458 vs 467,967), and that difference is not from the split.

**One row followed by hand (RAN r20261112-193855-d49f on vy-nebius-cpu-7, te_core from d76ae46e6f):** r20261103-111728-d00e@960 (s-t20 36M, ref-optim; record art:7731ed75…, te_core 3.20).
- In the counter, step 0 has 0 updated nodes. Its rope product `[[64,8,1024,32],[1024,32]]` (805,306,368 MACs) is `theta`. Steps 1 to 960 are update steps.
- `of_record`, rebuilt from the record with the traced shares, gives the row's parts back exactly (max relative error 0). `measure` plus `published` then give rho_own 244,456.075 and rho_default 281,739.425. Those equal the staged search table's cells, which print in the body as 244,500. The same value comes from each of the three cache layers that traced this run (te_core 4.12.1, 4.12.7, 4.12.11).
- Counterfactual: if the counter had counted the tables as updated, step 0 would have become an update step. It would then have taken the traced shares (share 0.9923), and rho_own would be 244,710.7 (+0.10%) and rho_default 282,032.9. RAN, same run.
- So the gate matters in principle. It never fired in any record (the scan above).

Left for 5 PM:
- a standard-workloads row and a size-scaling row followed by hand
- the full report note

No runs in flight.

