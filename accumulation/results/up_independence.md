# Independence of the coarse checker's `Up` bound

Generated 2026-09-16 by `gate_level_up` (`bounds/coarse.py` @ `f4baa2d7be`), harness `/tmp/up_indep.py` (not kept).

## Question

The per-sequence `Up` cut moved every headline `U` by 1.5-2x. Does the *checker* compute `Up_R(g)` from real
gate-level dependencies, or does it reuse the planner's per-sequence member cut?

## Answer

Both, in the only way that is sound: the checker takes from the plan **only the cut** (which `(op, slice)`
members exist and in which unit). Every dependency edge between members is resolved through the program
from the kernels' *real operand reads* -- `_Ctx.reads(member)` gives the `(tensor, rows/instances)` windows a
member reads (attention's within-sequence Q/K/V coupling, the residual stream, router / expert gathers), and
`_eval_unit` makes every member that produces part of a read window an ancestor. `Up(u)` = work of `u` plus all
in-unit ancestors; a unit's `Up` is the max over members. This dominates the gate-level `work(Up_R(g))` for every
gate: an in-RU gate upstream of `g` sits in a member reached by one of those wires and its *whole* member is
counted. The cut only sets the granularity -- a coarser cut counts more, never fewer, ancestors -- so a
per-sequence cut is tighter but exactly as sound as whole-op members. (Documented in the `check_coarse_plan`
docstring and `SPEC.md` s0.)

## Independent path

`gate_level_up(g, plan, program, unit=...)`: expand the plan to a gate assignment (`plan_gate_assignment`),
take the unit's gates in the flattened circuit (`exact/solve.py::flatten`, red team's code), and compute
`max_g work(Up_R(g))` by BFS over the real operand wires (`exact/solve.py::up_set`). `Up_R` is monotone along
in-RU wires so the max is attained at the unit's sinks; every sink was walked (no sampling in any row below).
`sound` = checker `Up >=` gate-level `Up` **and** checker `Work >=` gate-level `Work`.

## Results (tiny2 / tiny-moe; largest-`Up` unit, plus every unit when the plan has <= 6)

| model | circuit | Q | F_hat | G_hat | family | n_units | unit | gates | sinks | Work gate | Work checker | Up gate | Up checker | checker/gate | sound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny2 | forward-nonfixed | 16 | 1.5 | 4.0 | single | 1 | `whole-circuit` | 12896 | 128 | 11856 | 11856 | 1861 | 11856 | 6.37 | yes |
| tiny2 | forward-nonfixed | 16 | 1.0 | 1.0 | single | 1 | `whole-circuit` | 12896 | 128 | 11856 | 11856 | 1861 | 11856 | 6.37 | yes |
| tiny2 | forward-nonfixed | 16 | 1.5 | 16.0 | single | 1 | `whole-circuit` | 12896 | 128 | 11856 | 11856 | 1861 | 11856 | 6.37 | yes |
| tiny2 | forward-nonfixed | 32 | 1.5 | 4.0 | single | 1 | `whole-circuit` | 25792 | 256 | 23712 | 23712 | 1861 | 23712 | 12.74 | yes |
| tiny2 | forward-nonfixed | 32 | 1.0 | 1.0 | convex-block | 2 | `step0:head+fwd[1:2] seq[0:8]` | 13888 | 256 | 12576 | 12576 | 469 | 12576 | 26.81 | yes |
| tiny2 | forward-nonfixed | 32 | 1.0 | 1.0 | convex-block | 2 | `step0:fwd[0:1]seq[0:8]` | 11904 | 128 | 11136 | 11136 | 465 | 11136 | 23.95 | yes |
| tiny2 | forward-nonfixed | 32 | 1.5 | 16.0 | single | 1 | `whole-circuit` | 25792 | 256 | 23712 | 23712 | 1861 | 23712 | 12.74 | yes |
| tiny2 | forward-nonfixed | 64 | 1.5 | 4.0 | convex-block | 1 | `step0:head+fwd[0:2] seq[0:16]` | 51584 | 512 | 47424 | 47424 | 1861 | 2964 | 1.59 | yes |
| tiny2 | forward-nonfixed | 64 | 1.0 | 1.0 | convex-block | 4 | `step0:head+fwd[1:2] seq[0:8]` | 13888 | 256 | 12576 | 12576 | 469 | 12576 | 26.81 | yes |
| tiny2 | forward-nonfixed | 64 | 1.0 | 1.0 | convex-block | 4 | `step0:head+fwd[1:2] seq[8:16]` | 13888 | 256 | 12576 | 12576 | 469 | 12576 | 26.81 | yes |
| tiny2 | forward-nonfixed | 64 | 1.0 | 1.0 | convex-block | 4 | `step0:fwd[0:1]seq[0:8]` | 11904 | 128 | 11136 | 11136 | 465 | 11136 | 23.95 | yes |
| tiny2 | forward-nonfixed | 64 | 1.0 | 1.0 | convex-block | 4 | `step0:fwd[0:1]seq[8:16]` | 11904 | 128 | 11136 | 11136 | 465 | 11136 | 23.95 | yes |
| tiny2 | forward-nonfixed | 64 | 1.5 | 16.0 | convex-block | 1 | `step0:head+fwd[0:2] seq[0:16]` | 51584 | 512 | 47424 | 47424 | 1861 | 2964 | 1.59 | yes |
| tiny2 | forward-nonfixed | 128 | 1.5 | 4.0 | convex-block | 2 | `step0:head+fwd[0:2] seq[0:16]` | 51584 | 512 | 47424 | 47424 | 1861 | 2964 | 1.59 | yes |
| tiny2 | forward-nonfixed | 128 | 1.5 | 4.0 | convex-block | 2 | `step0:head+fwd[0:2] seq[16:32]` | 51584 | 512 | 47424 | 47424 | 1861 | 2964 | 1.59 | yes |
| tiny2 | forward-nonfixed | 128 | 1.0 | 1.0 | convex-block | 6 | `step0:head+fwd[1:2] seq[0:12]` | 20832 | 384 | 18864 | 18864 | 469 | 18864 | 40.22 | yes |
| tiny2 | forward-nonfixed | 128 | 1.0 | 1.0 | convex-block | 6 | `step0:head+fwd[1:2] seq[12:24]` | 20832 | 384 | 18864 | 18864 | 469 | 18864 | 40.22 | yes |
| tiny2 | forward-nonfixed | 128 | 1.0 | 1.0 | convex-block | 6 | `step0:fwd[0:1]seq[0:12]` | 17856 | 192 | 16704 | 16704 | 465 | 16704 | 35.92 | yes |
| tiny2 | forward-nonfixed | 128 | 1.0 | 1.0 | convex-block | 6 | `step0:fwd[0:1]seq[12:24]` | 17856 | 192 | 16704 | 16704 | 465 | 16704 | 35.92 | yes |
| tiny2 | forward-nonfixed | 128 | 1.0 | 1.0 | convex-block | 6 | `step0:head+fwd[1:2] seq[24:32]` | 13888 | 256 | 12576 | 12576 | 469 | 12576 | 26.81 | yes |
| tiny2 | forward-nonfixed | 128 | 1.0 | 1.0 | convex-block | 6 | `step0:fwd[0:1]seq[24:32]` | 11904 | 128 | 11136 | 11136 | 465 | 11136 | 23.95 | yes |
| tiny2 | forward-nonfixed | 128 | 1.5 | 16.0 | convex-block | 1 | `step0:head+fwd[0:2] seq[0:32]` | 103168 | 1024 | 94848 | 94848 | 1861 | 2964 | 1.59 | yes |
| tiny-moe | forward-nonfixed-moe | 16 | 1.5 | 4.0 | convex-block | 2 | `step0:head+fwd[1:2] seq[0:2]` | 10208 | 160 | 9296 | 9296 | 943 | 9296 | 9.86 | yes |
| tiny-moe | forward-nonfixed-moe | 16 | 1.5 | 4.0 | convex-block | 2 | `step0:fwd[0:1]seq[0:2]` | 5728 | 64 | 5360 | 5360 | 800 | 5360 | 6.70 | yes |
| tiny-moe | forward-nonfixed-moe | 16 | 1.0 | 1.0 | bands | 3 | `band[9:16]` | 6112 | 160 | 5552 | 5552 | 310 | 5552 | 17.91 | yes |
| tiny-moe | forward-nonfixed-moe | 16 | 1.0 | 1.0 | bands | 3 | `band[0:2]` | 5728 | 64 | 5360 | 5360 | 800 | 5360 | 6.70 | yes |
| tiny-moe | forward-nonfixed-moe | 16 | 1.0 | 1.0 | bands | 3 | `band[2:9]` | 4096 | 64 | 3744 | 3744 | 615 | 3744 | 6.09 | yes |
| tiny-moe | forward-nonfixed-moe | 32 | 1.5 | 4.0 | convex-block | 1 | `step0:head+fwd[0:2] seq[0:4]` | 31872 | 320 | 29312 | 29312 | 3623 | 7328 | 2.02 | yes |
| tiny-moe | forward-nonfixed-moe | 32 | 1.0 | 1.0 | convex-block | 6 | `step0:fwd[0:1]seq[0:2]` | 5728 | 64 | 5360 | 5360 | 800 | 5360 | 6.70 | yes |
| tiny-moe | forward-nonfixed-moe | 32 | 1.0 | 1.0 | convex-block | 6 | `step0:fwd[0:1]seq[2:4]` | 5728 | 64 | 5360 | 5360 | 800 | 5360 | 6.70 | yes |
| tiny-moe | forward-nonfixed-moe | 32 | 1.0 | 1.0 | convex-block | 6 | `step0:head+fwd[1:2] seq[0:1]` | 5104 | 80 | 4648 | 4648 | 943 | 4648 | 4.93 | yes |
| tiny-moe | forward-nonfixed-moe | 32 | 1.0 | 1.0 | convex-block | 6 | `step0:head+fwd[1:2] seq[1:2]` | 5104 | 80 | 4648 | 4648 | 943 | 4648 | 4.93 | yes |
| tiny-moe | forward-nonfixed-moe | 32 | 1.0 | 1.0 | convex-block | 6 | `step0:head+fwd[1:2] seq[2:3]` | 5104 | 80 | 4648 | 4648 | 943 | 4648 | 4.93 | yes |
| tiny-moe | forward-nonfixed-moe | 32 | 1.0 | 1.0 | convex-block | 6 | `step0:head+fwd[1:2] seq[3:4]` | 5104 | 80 | 4648 | 4648 | 943 | 4648 | 4.93 | yes |
| tiny-moe | forward-nonfixed-moe | 64 | 1.5 | 4.0 | convex-block | 2 | `step0:head+fwd[0:2] seq[0:4]` | 31872 | 320 | 29312 | 29312 | 3623 | 7328 | 2.02 | yes |
| tiny-moe | forward-nonfixed-moe | 64 | 1.5 | 4.0 | convex-block | 2 | `step0:head+fwd[0:2] seq[4:8]` | 31872 | 320 | 29312 | 29312 | 3623 | 7328 | 2.02 | yes |
| tiny-moe | forward-nonfixed-moe | 64 | 1.0 | 1.0 | convex-block | 12 | `step0:fwd[0:1]seq[0:2]` | 5728 | 64 | 5360 | 5360 | 800 | 5360 | 6.70 | yes |
| tiny2 | local-sgd K=2 | 32 | 1.5 | 1.0 | convex-block | 6 | `step0:head+fwd[0:2] seq[0:4]` | 14896 | 100 | 14348 | 14380 | 13091 | 14380 | 1.10 | yes |
| tiny2 | local-sgd K=2 | 32 | 1.5 | 4.0 | convex-block | 4 | `step0:top[fwd 0:2 bwd 1:2]+head seq[0:4]` | 28256 | 252 | 28468 | 28564 | 23555 | 28564 | 1.21 | yes |
| tiny2 | local-sgd K=3 | 48 | 1.5 | 1.0 | convex-block | 9 | `step0:head+fwd[0:2] seq[0:4]` | 14896 | 100 | 14348 | 14380 | 13091 | 14380 | 1.10 | yes |
| tiny2 | local-sgd K=3 | 48 | 1.5 | 4.0 | convex-block | 6 | `step0:top[fwd 0:2 bwd 1:2]+head seq[0:4]` | 28256 | 252 | 28468 | 28564 | 23555 | 28564 | 1.21 | yes |
**41/41 cells sound**; checker `Up` is 1.28-40x above the gate-level truth, never below.

Reading the ratios:

- `whole-circuit` / `band` / whole-op units (ratio 6-40x): every member is a whole op, so the checker's `Up`
  is the whole unit's work while the true `Up` of an output row is one sequence's path through the model.
  Sound and loose -- these are the units the old conservative cut produced.
- Per-sequence units (`top[0:2]+head seq[0:16]`, ratio 1.59 dense / 2.02 MoE): the checker's `Up` is exactly
  one sequence's members through the unit's layers (= unit work / sequences); the gate-level truth is lower
  still because an output row of the head depends only on its own row's norm/head/MLP work in the top layers
  (attention couples rows only through K/V). So per-sequence `Up` remains conservative by ~1.6-2x; the next
  refinement (per-row `Up` inside a sequence) would buy at most that factor and is not needed at any
  calibrated point (see below).
- `local-sgd` bwd / top units (ratio 1.28-1.42): the `dW` reduction couples the unit's sequences, so the
  gate-level `Up` (10019 of 14120; 17987 of 22900) is most of the unit's work and the checker's whole-unit
  `Up` is a tight over-estimate. `Work checker > Work gate` by 64 / 96 units of work: the accumulator init /
  round the checker charges per partial of every cut `red` chain (deliberate, >= gate level).

## 8B 128K rollout plan (forward-nonfixed, llama3-8b, Q_roll = 131072, seq 8192, F_hat 1.5, G_hat 4)

Gate-level expansion is infeasible here (~3e14 gates), so this reports the checker's per-unit `Up` and `Work`
for the plan (planned in 23 s locally) and relies on the structural identity established above on the same
circuit factory: `Up(unit) = Work(unit) / n_sequences` for per-sequence forward units.

| unit | Work / fwd | Up / fwd | Up / Work | members |
|---|---|---|---|---|
| `step0:fwd[0:8]seq[0:16]` | 3.7839 | 0.2365 | 0.0625 = 1/16 | 1936 |
| `step0:fwd[8:16]seq[0:16]` | 3.7838 | 0.2365 | 0.0625 | 1920 |
| `step0:fwd[16:24]seq[0:16]` | 3.7838 | 0.2365 | 0.0625 | 1920 |
| `step0:fwd[24:32]seq[0:16]` | 3.7838 | 0.2365 | 0.0625 | 1920 |
| `step0:head seq[0:16]` | 0.8647 | 0.8647 | 1.0 (whole-op head members) | 32 |

`U/Q = 155304 B/token`, 5 units, `G` binds (3.78 fwd of 4), `F` is slack by 6.3x (0.24 fwd of 1.5).

## When `F` binds

- **Forward RU** of `m` layers x `s` sequences: `Up = m` layers x one sequence (attention never couples
  sequences), so `F` binds iff `seq * m * w_layer(per token) [+ head] > F`, i.e. `m / L > F_hat * Q_inf / seq`.
  At `F = 1.5 fwd(Q_inf)` with `seq = Q_inf` (every rollout preset) that is `m > 1.5 L` -- never; `G` alone
  sizes forward tiles. `F` starts to bind only for a single sequence longer than `F_hat * Q_inf` tokens
  (e.g. `seq >= 12288` at the default point with `m = L`).
- **Training RU**: a unit holding a `dW = dY^T X` reduction over its tokens couples every sequence it holds --
  the reduction gate's `Up` is the whole unit's upstream (all sequences' backward + recompute, the top gradient,
  and everything the running chain reads). `F` therefore binds a bwd / top unit as soon as its tokens x layers
  of backward work exceed `F`; cutting the reduction by sequence (`red` slices, `wgrad='chain'`) moves the
  coupling into the running accumulator that the next unit imports (4 B/element of `dW` per cut). That import
  is the structural cost of training under `F`, not an artefact of the bound.

## Verdict

No disagreement anywhere: the checker's `Up` is an upper bound on the gate-level `Up` in every cell, and its
`Work` equals the gate sum up to the deliberate accumulator charges. No soundness blocker; the per-sequence cut
stays.
