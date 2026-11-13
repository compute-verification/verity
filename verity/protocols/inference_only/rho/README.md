# rho: the slowdown's lower bound on training

This package computes, for each scenario and monitoring period, a lower bound on the **slowdown** a training run pays
under the regime: `overhead(𝒢, 𝒜′, η)` (`GLOSSARY.md`, Notation), the training cost over the admissible set
divided by the reference training algorithm's GPU-hours, on the datacenter the adversary actually runs. It is Lean's
`hardwareMultiple`: the slowdown over the reference at its MFU, `MFU₀`, at least 1, so no protocol gives 1.
Beside it is the **time to the target** it implies, at least the slowdown times `T_0 / (MFU₀ Φ_total)`, wall-clock.
Hosts carry no compute (`GLOSSARY.md`, the device model): a server's compute is its accelerators', so every regime's
`host` is 0. No run uses the hardware more efficiently than the reference (`GLOSSARY.md`, reference efficiency;
`exclusion.toml`'s `task.reference_efficiency`), so a bound counts free compute at `MFU₀`, not at peak. `periods.py`
gives the bound without each assumption, and without both, only as sensitivities beside it.
Every number it reads is a parameter with a public source and a status, so a bound is always at some choice of every
parameter, and the package shows how the bound moves with each one.

A scenario is one choice from every part: a fleet (its location and dark share), a regime (the envelopes that run), the
inference-only policy, a monitoring period (its `horizon`, one of core's), a run (the target and its model), the
algorithm families the adversary may use, and the bounds and attacks applied. For each one the package reports:

- **the slowdown's lower bound** (`Result.lo`): every bound met at once, at the family the adversary picks. Under a
  bound it is `max(1, 1/(s + a x))` (`theorem.multiple_bound`), with the dark share `s` training like the reference and
  the bound's other terms `x`, free compute, at `a = 1` under reference efficiency, so that it is the padded size ratio;
  at `a = 1/MFU₀` (peak) without it, Lean's `Nci.Dampener.remoteDampener` and, for a remote regime,
  `Nci.Guarantees.DampenerRemote`'s `ρ`. `verity/protocols/inference_only/nci` reads its slowdown `D` from `multiple_bound`. It
  comes with its status, where its assumptions stand, the δ that the envelopes fail (`δ_coin + δ_crypto`, plus a
  named `Pr[Break]`), the monitored share (the part of the adversary's counted weight on monitored gates), and the
  time it implies;
- **the padded size ratio** `|A| / T_0` (`Overhead.padded`): the size of the actual circuit that reaches the target,
  each server block padded to its capacity, over the reference construction's size `T_0`. It is the intermediate the
  server-block bound gives, and what a contradiction compares;
- **the best known attack** (`Result.attack`, `FamilyResult.best_attack`): a construction on a server block's layout,
  its slowdown `max(1, MFU₀ · |A| / T_0)`, reported beside the bound, not as an end of an interval around it. A bound
  whose padded size ratio is above an attack's is a **contradiction**:
  one of its premises is false for that algorithm at that budget, and the contradiction names it;
- **how it moves**: sweeps, a tornado, thresholds, dangers and rescues, the corner (every range at its worse end) and
  what sets the slowdown (`analysis.py`); the time at operating points such as an optimizer step, a pretraining run or
  exfiltrating a reference (`operating.py`); and figures (`figures.py`).

## Run

~~~sh
uv run python verity/protocols/inference_only/rho/run.py --out DIR    # every scenario; writes rho.md, rho.json and figures/ to DIR
~~~

`run.py` prints Markdown and, with `--out DIR`, writes `rho.md`, `rho.json` and
`figures/` there. `run.py` takes `--scenario NAME` (repeatable), `--target X` and `--set KEY=VALUE` (any registry key, for the whole run).
`periods.py` gives the same per monitoring period, over one datacenter and over the world's compute, with each scenario's
range, its parts by kind and its premises ranked; each row says what it assumes about proofs, and the remote scenario
with proofs sampled at the registry's share stands beside the one where every unit is proved. The ranking shows a
proof block's output bits over the public size classes as the size class's limit, not as a premise. `periods.py
--measured CIRCUIT` evaluates them with a measured training step's design in place of the FLOP model, after a table of
the step beside it (`circuits.py`). `periods.py --bounds CLASS` (repeatable) bounds each scenario by that class alone,
and its header says what such a row's figure covers:

~~~sh
uv run python verity/protocols/inference_only/rho/periods.py --bounds min-cut --scenario reference-onsite-h200 \
    --scenario reference-remote-h200    # onsite and remote on the min-cut bound alone
~~~

## Inputs

- `data/*.toml`, the registry (`params.py`): every number the model reads, addressed by a dotted key, each with
  `value`, `unit`, a plausible range (`low`, `high`, which the sweeps span), `source` and `status` (`datasheet`,
  `measured`, `declared`, `assumed`, `ruled`, `conjecture`, `derived`). What another file states is read from it,
  never restated in a file (`params.sourced`). From `verity`: the monitoring periods; the output bits an accepted
  incorrect proof block injects (`step_bits`), the widest public size class of a block's outputs, a power of two from
  16 to 262,144 bits (`verity.circuits.partition_v2`); the per-window δ (`verity.protocols.spot.verifier.profile`), once
  for a window's coins and for SPOT's sample and, assumed, twice for the window's cryptography (PoUW's binding and
  knowledge soundness, neither with a bound); and the reporting δs the incorrect-outputs row reports at. From
  `verity/protocols/inference_only/nci/calibrations.json`: the calibrated policy's `K`, with the least and the
  greatest calibrated `K` as its range, and `Γ`; the session policy's `K` is a ruling
  `data/reference.toml` states, and only its range is read (`params.ranges`). The census's numbers are the exception:
  `verity` takes them as parameters, so the registry reads no census. `data/` states the reference datacenter's
  machines (the count of each reference fleet of 8-GPU servers), the world's active parameters and the session
  policy's `Γ` at the world's model, each naming the census entry it is, and the tests hold each equal to
  `benchmarks/census`. `data/scenarios*.toml` and
  `data/reference.toml` name the scenarios; `data/premises.toml` says where each assumption stands and up to what
  block size, and holds (H)'s `β` for an activation by block size (`premise.training-cut-hardness.beta`);
  `data/circuits.toml` holds training steps measured as Boolean circuits (the census's gates, ANDs and
  counted products, and `circuit_properties`' `s_×`, `a_out` and `a_in`).
- The Lean locks (`security/lean-audit.json`, `security/Properties/lean-audit.json`), which `lean.py` reads for each
  citation's status.
- `benchmarks/census` (devices, the reference datacenter and the world) and `verity/protocols/inference_only/nci` (the remote regime's
  terms in `tiers.py`, its algorithms and their traits), which the tests hold the data to;
  `security/Definitions/Nci/Remote.lean`, whose
  `remoteDampener` `tests/test_envelopes.py` holds `theorem.multiple_bound` to; and the Lean files that define the
  min-cut bound's `alphaHE`, `shareF` and `workHEOn` and state its `HE`, `HEShared` and `HECharged`, found through
  the lock, which `tests/test_min_cut.py` holds `min_cut.py` to.

## The model

A bound is argued in three parts, as the Notation states it, and each module's docstring says what it implements.

- **The physical model** `𝒜_phys` (`hardware.py`, `data/hardware.toml`): the inventory's locations, each with its
  capacity `Φ_X` in gates per second and its servers' memory `M_s`, the unregistered hardware `ε_dark`, and the monitoring
  period (core's `reporting.PERIODS`, through `params.py`). `Units`
  is the one place a FLOP-model input becomes gates (`units.gates_per_flop`, with its `units.basis`).
- **Protocol soundness**, the envelopes (`envelopes.py`, `data/envelopes.toml`): each subprotocol (PoUW, detection and
  sampled proofs, the wipe, the certifier) is one envelope `𝒜_a` with its parameters, its slack (the free compute
  by route: `undecl.*`, `skip.*`, `break.*`, `exfil.incorrect`) and its `δ_a`, split into `δ_coin` and `δ_crypto`. The
  inference-only policy's check is the registration (`Policy`). A regime is the envelopes that run, and builds the server blocks with their
  cuts.
- **The slowdown lower bound** (`classes.py`, `attacks.py`, `theorem.py`): the pricing arguments as inference-block yields
  `q_T(c, k)`, the attacks as constructions, and the server-block bound: each server block's least cut
  (`theorem.local`), summed over blocks (`theorem.overhead`), with the minimum of global bounds beside it
  (`theorem.aggregate`).

| Notation | here |
|---|---|
| `\|A[X,J]\| ≤ Φ_X \|J\|`, `M_s` | `Location.phi`, `Location.server_memory` |
| `ε_dark`, `L_dark` | `fleet.<name>.eps_unreg`, `Partition.eps_unreg` |
| inference block `(c, k)` | `Blocks("inference")`, `Policy`, `Budget(gates, inputs)` |
| server block `S`, `(c_S, k_S)`, `n_S` | `ServerBlock`, `Onsite.budget`, the inference cut's `Blocks.count` |
| `q_T(c, k)` | `Cap`, from a class's `Allowance` by `classes.per_block` |
| `q_U(U) = ω_max U / s_×` | `Structure.weight` (`data/exclusion.toml`) |
| `ε_undecl`, `ε_skip`, `ε_break` | the routes `undecl.*`, `skip.*`, `break.*`, `exfil.incorrect` (`Cut.uncertified`) |
| `𝒜_a`, `δ_a`, `δ_coin`, `δ_crypto` | `Pouw`, `Detection`, `Wipe`, `Certifier`; `Regime.delta → Delta` |
| `k_wipe`, `r_in`, `r_side` | `Wipe.k_wipe`, `Certifier.rate_in`, `Certifier.timing` and `side` |
| the padded size ratio `\|A\| / T_0`, the slowdown, the time to the target | `Overhead.padded`, `Overhead.rho`, `Scenario.seconds` |

**The uncertified cap.** A bound counts an uncertified or dark gate at `ω_max / s_×`, `s_×` the fewest gates in any
counted operation's subcircuit. `structure.reference` takes `s_×` at the reference's gates per counted product, which
is valid only under the named assumption `draft:no-straddle` (no counted operation straddles verified and free
compute), so every bound counting such gates cites it. A structure states its `basis`, and `Structure.check` refuses one
whose `s_×` and `units.gates_per_flop` come from different sources. `structure.elementary` (one counted operation a
gate, at most) needs no such premise; it is the `*-elementary` scenarios, and what `analysis.upto` falls back to for a
status stronger than `draft`. A scenario that names no structure takes `structure.default` (`reference`). A measured
structure (`structure.train-step-v1`, SmolLM2-135M's training step) is `s_×` in gates of one Boolean design, with
`evidence`, the runs that measured it, cited beside the straddling premise; `circuits.in_place` puts it in place with
`units.gates_per_flop` counted on the same design, where `G` no longer cancels and a gate holds `u/w = 2G / (κ s_×)` of
the reference's weight.

**The min-cut bound.** `min_cut.py` is the one place that computes Lean's `HE`, `HEShared` and `HECharged`
(`min_cut.HE`, `min_cut.HE_SHARED`, `min_cut.HE_CHARGED`), each function one side of an inequality they conclude or
that inequality solved for one term; it is standard library only, so another benchmark loads it by path. `HECharged`
is stated and no lock lists a proof of it. A work of `N` MACs with `O` output wires needs at
least `m` blocks of at most `k` fresh input bits with `N ≤ m (k + F α) + F (k/2) √(m O / q)`, `α = ½ √(k³ / (q b))`,
under (H) at `q = β_w β_x` and (E) at `b` (`training-cut-hardness`, `training-cut-exports`: conjectures).

- `min_cut.bound(c, k, n, N=…, O=…, size=…)` is the bound at a policy `(c, k, n)` on a circuit's own MACs, output
  wires and gates: the least blocks, the most MACs a block, the slowdown `m c / size` (at least 1) and what it rests
  on. `N` and `O` are the caller's: it converts no count. Defaults: `β_w` and `b` a bf16 word's 16 bits, the most (H)
  and (E) can hold at, since a word's own wires are a cut; `β_x` the rows' `beta(k)` (between two rows the next
  row's, the most the rows allow there), and no bound past their last `k`; `F = 1` (`HE`: at most two of forward,
  input-gradient and weight-gradient products), `share(3)` for all three (`HEShared`).
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
  `1 / (uncredited + 1 / slowdown)`: the dark share of `theorem.multiple_bound`, which no count covers. A figure of
  `bound` or `bound_charged` is quoted beside it at a stated share, never alone.
- The class `min-cut` in `CLASSES` is `min_cut.macs_a_block` on the profile's own counted products, the trained
  weights' forward and input-gradient products: a weight value at `b_w`, an activation value at `β(k)` and at most
  `b_a`, an output at `b_a`, and `outputs_per_product` from the model's shape. It bounds the declared inference
  blocks only, so where they hold little of the rate a figure it sets is the uncertified routes' and the
  unregistered pool's (`periods.py --bounds min-cut` says the share). That the work's MACs are disjoint, its values
  distinct words and no two of its matmuls share both operands' values is assumed, not checked.

## Modules

Each module is one interface with interchangeable implementations: a different assumption is a different
implementation, or different numbers in `data/`, never an edit to the modules around it.

| module | interface |
|---|---|
| `theorem.py` | `Budget`, `Blocks`, `Cut`, `ServerBlock`, `Partition`; `Cap`; `local`, `aggregate`, `overhead`, `padded`, `multiple_bound` |
| `classes.py` | `Argument.allowance(Profile) → Allowance`, raising `Outside`; `per_block`, `weight_per_gate`; `premise_status`; `Structure` |
| `min_cut.py` | `alpha`, `share`, `macs_at_most`, `blocks_at_least`, `slowdown_at_least`, `macs_a_block`, `macs_on`, `beta`; `bound(c, k, n, N, O, size) → Bound`; `bound_charged(c, k, n, blocks, untied, untied_macs, w, N, O, size, rows) → Charged`, `best_w`, `capped` |
| `attacks.py` | the same `Argument`, side `attack`, on a server block's layout |
| `profile.py` | `Profile`: every field a class reads, named as in Lean |
| `families.py` | `Family.profile(Model, Run) → Profile`, for each algorithm family and any `verity/protocols/inference_only/nci` algorithm by its traits |
| `instances.py` | `Model`, `Run` (a target and its kind); `models`, every model's shape, which `verity/protocols/inference_only/nci/kappa.py` and `benchmarks/circuit_properties` read |
| `circuits.py` | `Circuit` (a measured training step), `compare → Comparison` with the FLOP model, `in_place` |
| `hardware.py` | `Units`, `Location`, `Fleet`, `MonitoringPeriod` |
| `envelopes.py` | `Pouw`, `Detection`, `Wipe`, `Certifier`; `Regime.partition`, `Regime.delta`; `Policy` |
| `params.py` | `Registry`: dotted keys over `data/*.toml` and what another file states (`sourced`: core's monitoring periods, `PERIODS`, per-window δ and reporting δs, a proof block's widest output size class, `BLOCK_OUTPUT_BITS`, and the calibrations' `K` and `Γ`; `ranges`: a range read for a value a file states), `override`, `fields` |
| `scenario.py` | `build`, `evaluate → Result`, `seconds` |
| `analysis.py` | `sweep`, `tornado`, `threshold`, `dangers`, `rescues`, `corner`, `margins`, `phase`, `frontier`, `attribution`, `upto` |
| `operating.py` | `work`, `exfil`, `points`, `incorrect` |
| `figures.py` | `render` |
| `lean.py` | `status(name)`, `weakest(cites)` |

## What it writes and what loads it

`periods.py` prints Markdown as `run.py` does and, with `--out DIR`, writes `periods.md` and `periods.json` there.
Neither writes a file unless told where.

This package has an import root of its own (`SEPARATE` in `tests/test_boundaries.py`). A loader puts this directory on
`sys.path` and imports `rho`, and one that needs a number of the registry reads its file under `data/` by path. Either
way its suite names what it reads here among its `inputs`, so this lists every loader:

~~~sh
git grep -n verity/protocols/inference_only/rho -- '*pyproject.toml'
~~~

Among them, `verity/protocols/inference_only/nci` calls `theorem.multiple_bound` for its slowdown `D`, and
`benchmarks/circuit_properties` reads a model's shapes (`instances`, `params`). In the other direction, `families.py`
reads `verity/protocols/inference_only/nci/algorithms.json` when a family names an algorithm of that library (a family
of kind `nci`), and `params.py` reads `verity/protocols/inference_only/nci/calibrations.json` for the calibrated
policy's `Γ` and `K` and the range of `K`.

## Status

Every estimate carries the weakest status among its citations, read from the locks (`lean.py`): `proved`,
`conditional` (proved under named assumptions), `draft` (a statement not yet in Lean, a `draft:` id) or `absent`. Its
citations are its argument's and those of the cuts that bound it, and `draft:no-straddle` wherever a bound holds
uncertified or dark gates at a structure whose `s_×` is above 1, with the structure's `evidence` (which no status reads).

## Names

The package and its outputs (`verity/protocols/inference_only/rho`, `rho.md`, `rho.json`) and the
identifiers `Overhead.rho`, `Estimate.rho`, `rho_worst`, `Corner.rho` and the JSON's `rho` and `slowdown` all name the
slowdown (the training cost once called the overhead; `Overhead` and `theorem.overhead` keep that name);
`Result.lo`, `FamilyResult.lo` and `Point.lo` name its lower bound. The slowdown is the `ρ` of Lean's
`Nci.Guarantees.DampenerRemote` (`remoteDampener`, bounded by `hardwareMultiple`) and `verity/protocols/inference_only/nci`'s
slowdown `D`; the padded size ratio `|A| / T_0` (`Overhead.padded`, `Estimate.padded`, the JSON's `padded`) is the
intermediate, not the slowdown. The glossary's `ρ` is the ACCEL model's resource rate, not this number.

## Extending

- **A pricing argument**: an `Argument` in `classes.CLASSES` (a premise set, its count, its citations), and any new field
  it reads in `Profile`.
- **An attack**: an `Argument` with side `attack` in `attacks.ATTACKS`.
- **An algorithm family**: a table in `data/families.toml` of an existing kind, or a new kind in `families.KINDS`.
- **A model or run**: a table in `data/models.toml`, the one place a model's shape is stated: another benchmark reads
  it through `instances.models` and states none.
- **Hardware, a fleet**: a table in `data/hardware.toml`.
- **An envelope**: a dataclass with `slack` or `state` and `delta` in `envelopes.PARTS`, held by a regime.
- **A regime**: a dataclass with `partition` and `delta` in `envelopes.KINDS` and its table in `data/envelopes.toml`.
- **A scenario**: a table in `data/scenarios.toml` (the reference datacenter's are in `data/reference.toml`).
- **A work or exfiltration reference**: a `work.<name>.flops` or `exfil.<name>.bits` table.
- **A sensitivity**: give the parameter `low` and `high`; `tornado` and `dangers` pick it up.
- **A structure** (a measured `s_×`): a `structure.<name>` table in `data/exclusion.toml` with its `basis`
  (`"boolean:<design>"`) and its `evidence`, which a scenario names with `structure = "<name>"` (or every scenario
  naming none, by `structure.default`), beside `units.gates_per_flop` and `units.basis` from the same design
  (`Structure.check`).
- **A measured circuit**: a `circuit.<name>` table in `data/circuits.toml` naming its Definition, its run and its
  structure; `periods.py --measured <name>` evaluates with it in place.
