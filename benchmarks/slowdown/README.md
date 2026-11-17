# The slowdown calculator: the min-cut bound and the physical model

This package is what the training and inference-only benchmarks read to price a policy: the min-cut bound
(`min_cut.py`), the physical model's locations, fleets and monitoring periods (`hardware.py`), the models' shapes and
the runs' targets (`instances.py`), the registry of every number they read (`params.py`, `data/*.toml`) and each
citation's status in the locks (`lean.py`). It has no runner.

**The min-cut bound.** `min_cut.py` is the one place that computes Lean's `HE`, `HEShared` and `HECharged`
(`min_cut.HE`, `min_cut.HE_SHARED`, `min_cut.HE_CHARGED`), each function one side of an inequality they conclude or
that inequality solved for one term; it is standard library only, so another benchmark loads it by path. The lock
lists a proof of each of the three. A work of `N` MACs with `O` output wires needs at
least `m` blocks of at most `k` fresh input bits with `N ≤ m (k + F α) + F (k/2) √(m O / q)`, `α = ½ √(k³ / (q b))`,
under (H) at `q = β_w β_x` and (E) at `b` (`training-cut-hardness`, `training-cut-exports`: conjectures).

- `min_cut.bound(c, k, n, N=…, O=…, size=…)` is the bound at a policy `(c, k, n)` on a circuit's own MACs, output
  wires and gates: the least blocks, the most MACs a block, the slowdown `m c / size` (at least 1) and what it rests
  on. `N` and `O` are the caller's: it converts no count. Defaults: `β_w` and `b` a bf16 word's 16 bits, the most (H)
  and (E) can hold at, since a word's own wires are a cut; `β_x` the rows' `beta(k)` (between two rows the next
  row's, the most the rows allow there), and no bound past their last `k`; `F = 1` (`HE`: at most two of forward,
  input-gradient and weight-gradient products), `share(3)` for all three (`HEShared`).
- `min_cut.slowdown_of(c, k, n, size=…, work=[Product, …])` is the one way the calculator's callers go from a work to a slowdown: no source
  outside this directory calls `bound`, `blocks_at_least` or `slowdown_at_least`. Each `Product` states its MACs, output wires, operands' `β`
  as its caller charges them, which operand is a weight, `b_y`, contraction length, whether it is forward or a weight
  gradient, and the pairs of arrays it multiplies where known. It returns the `Bound` of the sub-work it binds at
  with what it chose (`Chosen`). The sub-works (`SUBWORKS`) are the work, the work without its contractions below
  `K_thr` or any product's contraction, the weight gradients alone, and the products with a weight operand or
  without, each less its contractions below any product's, each at the threshold on `β_w β_x` where its term is largest; `F` is
  `share(ω)` of the arrays' graph (`clique`) where every product states its pairs, else the kinds' rule. The
  training benchmark's calculator, `circuit_properties.work.bound` and the inference-only policy read their works
  into products and call it; `tests/test_one_slowdown.py` at the root holds every other file away from `bound`.
- `min_cut.bound_charged(c, k, n, blocks=…, untied=…, untied_macs=…, w=…, N=…, O=…, size=…)` is the bound over a
  monitoring period's `blocks` blocks when at most `untied` of the words they read are untied, each at most `word`
  fresh bits in the one block that reads it. It computes Lean's `HECharged` (`min_cut.HE_CHARGED`), the count over a
  set of blocks with the rest charged whole: `N ≤ H + T + ½ √(S O / q) + ½ √(S T / (q b))`, `T` what all the blocks
  read between them, `S` the squares of what the blocks that aren't charged read, summed, `O` the whole work's output
  wires and `H` what the charged blocks hold, each at most its gates over the fewest gates of a MAC. The count is
  taken with the untied words as inputs, so the MACs that feed them, at most `untied_macs`, are added whole. A block
  that reads more than the threshold `w` of the untied words is charged whole; `w` is fixed before the period, and by
  default there is none and no block is charged. `most` is the most untied words one block reads, where its caller
  states one: below the threshold no block is charged and (H) is taken there. It gives the slowdown, the period's
  pooled total in gates (`gates`, the most gates of the work the period's blocks hold) and its share of the blocks'
  own, the blocks charged at most, the MACs they hold and their share of the blocks and of the MACs, and what it
  rests on: `HECharged`, (H) at the most a block that isn't charged reads, (E), and two premises no Lean statement
  gives, the period's bound on the untied words, a node's, and an untied word's bits. With no untied word it is
  `bound`'s slowdown. `best_w` is the threshold at which it is largest for a block count. A node's bound on its
  untied words is a share of one execution phase's ticket weight, so nodes faster than the measured one take its
  bound scaled by their rate; where the untied words are then as many as the blocks, `w = 0` charges every block and
  gives 1, and the figure states its threshold. The count holds of any partition and any set of its blocks, and the
  caller picks neither: that the partition a period's audit reads has so few blocks over `w` is a join the guarantee
  that reads the bound owes. `O` is wires, values times their width, and a work that reads a registered table by a
  few fresh gates leaves the matmuls that read it out. Its three readings are quoted side by side, each under its
  premise's name (`Charged.reading`), never one as another:

  | reading | (H) is asked | fixed before the period | a block over it | the count |
  |---|---|---|---|---|
  | one `β_x`, with the threshold `w` | once, at `k + word · w` | `w`, quoted with the figure | charged at its whole work, the tied circuit's `c` over the fewest gates of a MAC | `HECharged` at the best set of blocks |
  | one `β_x` at the top budget, no block charged | once, where the block that reads the most untied words reads | nothing | no block is charged | `HECharged` at every block |
  | a `β_x` a block, no threshold (`rows=True`) | of each block at the table's row for what it reads | nothing | no block is charged | `min_cut.HE_ROWS`, which no guarantee states: `HECharged` asks (H) once |

- `min_cut.capped(slowdown, uncredited)` is a slowdown beside the share of a block's work no block is credited for,
  `1 / (uncredited + 1 / slowdown)`: the unregistered share, which no count covers. A figure of
  `bound` or `bound_charged` is quoted beside it at a stated share, never alone.

## Inputs

- `data/*.toml`, the registry (`params.py`): every number the package reads, addressed by a dotted key, each with
  `value`, `unit`, a plausible range (`low`, `high`), `source` and `status` (`datasheet`, `measured`, `declared`,
  `assumed`, `ruled`, `conjecture`, `derived`). What another file states is read from it, never restated in a file
  (`params.sourced`): core's monitoring periods, `data/calibrations.json`'s calibrated `K`, its range and
  `Γ`, and the cards' memory and line rates of the H200 servers (`params.carded`). The registry reads no census:
  `data/` states the reference datacenter's machines, the world's active parameters and the session policy's `Γ`, each
  naming the census entry it is, and the tests hold each equal to `benchmarks/census`. `data/reference.toml` states
  the inference-only policies (`params.policy`); `data/premises.toml` says where the min-cut bound's two conjectures
  stand and up to what block size, and holds (H)'s `β` by block size (`premise.training-cut-hardness.beta`).
- The Lean locks (`security/lean-audit.json`, `security/Properties/lean-audit.json`), which `lean.py` reads for each
  citation's status, and the Lean files that define the min-cut bound's `alphaHE`, `shareF` and `workHEOn` and state
  its `HE`, `HEShared` and `HECharged`, which `tests/test_min_cut.py` holds `min_cut.py` to.

## Modules

| module | interface |
|---|---|
| `min_cut.py` | `alpha`, `share`, `macs_at_most`, `blocks_at_least`, `slowdown_at_least`, `macs_a_block`, `macs_on`, `beta`; `bound(c, k, n, N, O, size) → Bound`; `clique`, `slowdown_of(c, k, n, size, work) → Chosen`; `bound_charged(c, k, n, blocks, untied, untied_macs, w, N, O, size, rows) → Charged`, `best_w`, `capped` |
| `hardware.py` | `Units`, `Location`, `Fleet`, `MonitoringPeriod` |
| `instances.py` | `Model`, `Run` (a target and its kind); `models`, every model's shape, which `benchmarks/circuit_properties` reads |
| `params.py` | `Registry` (`override`, `fields`), `sourced`, `ranges`; core's `PERIODS`, the `FLEETS` a figure is quoted over, `Policy` and `policy` |
| `lean.py` | `status(name)`, `weakest(cites)` |

## What loads it

This package has an import root of its own (`distribution` in `tests/import_order.toml`). A loader puts this directory on
`sys.path` and imports `slowdown`, and one that needs a number of the registry reads its file under `data/` by path. Either
way its suite names what it reads here among its `inputs`, so this lists every loader:

~~~sh
git grep -n benchmarks/slowdown -- '*pyproject.toml'
~~~

## Extending

- **A model or run**: a table in `data/models.toml`, the one place a model's shape is stated: another benchmark reads
  it through `instances.models` and states none.
- **Hardware, a fleet**: a table in `data/hardware.toml`.
- **A policy**: a `policy.<name>` table in `data/reference.toml`.
