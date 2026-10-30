# rho: the slowdown's lower bound on training

This package computes, for each scenario and monitoring period, a lower bound on the **slowdown** a training run pays
under the regime: `overhead(𝒢, 𝒜′, η)` (`GLOSSARY.md`, Notation), the training cost over the admissible set
divided by the reference training algorithm's GPU-hours, on the datacenter the adversary actually runs. It is Lean's
`hardwareMultiple`: the hardware multiple over the reference at its MFU, `MFU₀`, at least 1, so no protocol gives 1.
Beside it is the **time to the target** it implies, at least the slowdown times `T_0 / (MFU₀ Φ_total)`, wall-clock.
Hosts carry no compute (`GLOSSARY.md`, the device model): a server's compute is its accelerators', so every regime's
`host` is 0. No run uses the hardware more efficiently than the reference (`GLOSSARY.md`, reference efficiency;
`exclusion.toml`'s `task.reference_efficiency`), so a bound counts free compute at `MFU₀`, not at peak. `periods.py`
gives the bound without each assumption, and without both, only as sensitivities beside it.
Every number it reads is a parameter with a public source and a status, so a bound is always at some choice of every
parameter, and the package shows how the bound moves with each one.

A scenario is one choice from every part: a fleet (its location and dark share), a regime (the envelopes that run), the
NCI registration policy, a monitoring period (the horizon), a run (the target and its model), the algorithm families the
adversary may use, and the bounds and attacks applied. For each one the package reports:

- **the slowdown's lower bound** (`Result.lo`): every bound met at once, at the family the adversary picks. Under a
  bound it is `max(1, 1/(s + a x))` (`theorem.multiple_bound`), with the dark share `s` training like the reference and
  the bound's other terms `x`, free compute, at `a = 1` under reference efficiency, so that it is the padded size ratio;
  at `a = 1/MFU₀` (peak) without it, Lean's `Nci.Dampener.remoteDampener` and, for a remote regime,
  `Nci.Guarantees.DampenerRemote`'s `ρ`. `benchmarks/nci` reads its compute dampener `D` from `multiple_bound`. It
  comes with its status, where its named premises stand, the δ that the envelopes fail (`δ_coin + δ_crypto`, plus a
  named `Pr[Break]`), the monitored share (the part of the adversary's counted weight on monitored gates), and the
  time it implies;
- **the padded size ratio** `|A| / T_0` (`Overhead.padded`): the size of the actual circuit that reaches the target,
  each serving block padded to its capacity, over the reference construction's size `T_0`. It is the intermediate the
  serving-block bound gives, and what a contradiction compares;
- **the best known attack** (`Result.attack`, `FamilyResult.best_attack`): a construction on a serving block's layout,
  its slowdown `max(1, MFU₀ · |A| / T_0)`, reported beside the bound, not as an end of an interval around it. A bound
  whose padded size ratio is above an attack's is a **contradiction**:
  one of its premises is false for that algorithm at that budget, and the contradiction names it;
- **how it moves**: sweeps, a tornado, thresholds, dangers and rescues, the corner (every range at its worse end) and
  what sets the slowdown (`analysis.py`); the time at operating points such as an optimizer step, a pretraining run or
  exfiltrating a reference (`operating.py`); and figures (`figures.py`).

## Run

~~~sh
verity bench rho --out DIR    # every scenario; writes rho.md, rho.json and figures/ to DIR
~~~

`verity bench rho` runs `run.py`, which prints Markdown and, with `--out DIR`, writes `rho.md`, `rho.json` and
`figures/` there. `run.py` takes `--scenario NAME` (repeatable), `--target X` and `--set KEY=VALUE` (any registry key, for the whole run).
`periods.py` gives the same per monitoring period, over one datacenter and over the world's compute, with each scenario's
range, its parts by kind and its premises ranked; `periods.py --measured CIRCUIT` evaluates them with a measured training
step's design in place of the FLOP model, after a table of the step beside it (`circuits.py`).

## Inputs

- `data/*.toml`, the registry (`params.py`): every number the model reads, addressed by a dotted key, each with
  `value`, `unit`, a plausible range (`low`, `high`, which the sweeps span), `source` and `status` (`datasheet`,
  `measured`, `declared`, `assumed`, `ruled`, `conjecture`, `derived`). `data/scenarios*.toml` and
  `data/reference.toml` name the scenarios; `data/premises.toml` says where each named premise stands and up to what
  block size; `data/circuits.toml` holds training steps measured as Boolean circuits (the census's gates, ANDs and
  counted products, and `circuit_properties`' `s_×`, `a_out` and `a_in`).
- The Lean locks (`Security/lean-audit.json`, `Security/Properties/lean-audit.json`), which `lean.py` reads for each
  citation's status.
- `benchmarks/census` (devices and the world fleets) and `benchmarks/nci` (the remote regime's terms in `tiers.py`, its
  algorithms and their traits), which the tests hold the data to; `Security/Definitions/Nci/Remote.lean`, whose
  `remoteDampener` `tests/test_envelopes.py` holds `theorem.multiple_bound` to.

## The model

A bound is argued in three parts, as the Notation states it, and each module's docstring says what it implements.

- **The physical model** `𝒜_phys` (`hardware.py`, `data/hardware.toml`): the inventory's locations, each with its
  capacity `Φ_X` in gates per second and its servers' memory `M_s`, the dark leaves `ε_dark`, and the horizon. `Units`
  is the one place a FLOP-model input becomes gates (`units.gates_per_flop`, with its `units.basis`).
- **Protocol soundness**, the envelopes (`envelopes.py`, `data/envelopes.toml`): each subprotocol (PoUW, detection and
  sampled proofs, the wipe, the certifier) is one envelope `𝒜_a` with its parameters, its slack (the uncertified gates
  by route: `undecl.*`, `skip.*`, `break.*`, `exfil.faulty`) and its `δ_a`, split into `δ_coin` and `δ_crypto`. The NCI
  check is the registration (`Policy`). A regime is the envelopes that run, and builds the serving blocks with their
  cuts.
- **The exclusion bound** (`classes.py`, `attacks.py`, `theorem.py`): the training classes as per-block caps
  `q_T(c, k)`, the attacks as constructions, and the serving-block bound: each serving block's least cut
  (`theorem.local`), summed over blocks (`theorem.overhead`), with the minimum of global bounds beside it
  (`theorem.aggregate`).

| Notation | here |
|---|---|
| `\|A[X,J]\| ≤ Φ_X \|J\|`, `M_s` | `Location.phi`, `Location.server_memory` |
| `ε_dark`, `L_dark` | `fleet.<name>.eps_dark`, `Partition.eps_dark` |
| inference block `(c, k)` | `Blocks("inference")`, `Policy`, `Budget(gates, inputs)` |
| serving block `S`, `(c_S, k_S)`, `n_S` | `ServingBlock`, `Onsite.budget`, the inference cut's `Blocks.count` |
| `q_T(c, k)` | `Cap`, from a class's `Allowance` by `classes.per_block` |
| `q_U(U) = ω_max U / s_×` | `Structure.weight` (`data/exclusion.toml`) |
| `ε_undecl`, `ε_skip`, `ε_break` | the routes `undecl.*`, `skip.*`, `break.*`, `exfil.faulty` (`Cut.uncertified`) |
| `𝒜_a`, `δ_a`, `δ_coin`, `δ_crypto` | `Pouw`, `Detection`, `Wipe`, `Certifier`; `Regime.delta → Delta` |
| `k_wipe`, `r_in`, `r_side` | `Wipe.k_wipe`, `Certifier.rate_in`, `Certifier.timing` and `side` |
| the padded size ratio `\|A\| / T_0`, the slowdown, the time to the target | `Overhead.padded`, `Overhead.rho`, `Scenario.seconds` |

**The uncertified cap.** A bound counts an uncertified or dark gate at `ω_max / s_×`, `s_×` the fewest gates in any
counted operation's subcircuit. `structure.reference` takes `s_×` at the reference's gates per counted product, which
is valid only under the named assumption `draft:no-straddle` (no counted operation straddles certified and uncertified
gates), so every bound counting such gates cites it. A structure states its `basis`, and `Structure.check` refuses one
whose `s_×` and `units.gates_per_flop` come from different sources. `structure.elementary` (one counted operation a
gate, at most) needs no such premise; it is the `*-elementary` scenarios, and what `analysis.upto` falls back to for a
status stronger than `draft`. A scenario that names no structure takes `structure.default` (`reference`). A measured
structure (`structure.train-step-v1`, SmolLM2-135M's training step) is `s_×` in gates of one Boolean design, with
`evidence`, the runs that measured it, cited beside the straddling premise; `circuits.in_place` puts it in place with
`units.gates_per_flop` counted on the same design, where `G` no longer cancels and a gate holds `u/w = 2G / (κ s_×)` of
the reference's weight.

## Modules

Each module is one interface with interchangeable implementations: a different assumption is a different
implementation, or different numbers in `data/`, never an edit to the modules around it.

| module | interface |
|---|---|
| `theorem.py` | `Budget`, `Blocks`, `Cut`, `ServingBlock`, `Partition`; `Cap`; `local`, `aggregate`, `overhead`, `padded`, `multiple_bound` |
| `classes.py` | `Argument.allowance(Profile) → Allowance`, raising `Outside`; `per_block`, `weight_per_gate`; `premise_status`; `Structure` |
| `attacks.py` | the same `Argument`, side `attack`, on a serving block's layout |
| `profile.py` | `Profile`: every field a class reads, named as in Lean |
| `families.py` | `Family.profile(Model, Run) → Profile`, for each algorithm family and any `benchmarks/nci` algorithm by its traits |
| `instances.py` | `Model`, `Run` (a target and its kind) |
| `circuits.py` | `Circuit` (a measured training step), `compare → Comparison` with the FLOP model, `in_place` |
| `hardware.py` | `Units`, `Location`, `Fleet`, `Horizon` |
| `envelopes.py` | `Pouw`, `Detection`, `Wipe`, `Certifier`; `Regime.partition`, `Regime.delta`; `Policy` |
| `params.py` | `Registry`: dotted keys over `data/*.toml`, `override`, `fields` |
| `scenario.py` | `build`, `evaluate → Result`, `seconds` |
| `analysis.py` | `sweep`, `tornado`, `threshold`, `dangers`, `rescues`, `corner`, `margins`, `phase`, `frontier`, `attribution`, `upto` |
| `operating.py` | `work`, `exfil`, `points`, `faulty` |
| `figures.py` | `render` |
| `lean.py` | `status(name)`, `weakest(cites)` |

## Status

Every estimate carries the weakest status among its citations, read from the locks (`lean.py`): `proved`,
`conditional` (proved under named assumptions), `draft` (a statement not yet in Lean, a `draft:` id) or `absent`. Its
citations are its argument's and those of the cuts that bound it, and `draft:no-straddle` wherever a bound holds
uncertified or dark gates at a structure whose `s_×` is above 1, with the structure's `evidence` (which no status reads).

## Names

The package, its command and its outputs (`benchmarks/rho`, `verity bench rho`, `rho.md`, `rho.json`) and the
identifiers `Overhead.rho`, `Estimate.rho`, `rho_worst`, `Corner.rho` and the JSON's `rho` and `slowdown` all name the
slowdown (the training cost once called the overhead; `Overhead` and `theorem.overhead` keep that name);
`Result.lo`, `FamilyResult.lo` and `Point.lo` name its lower bound. The slowdown is the `ρ` of Lean's
`Nci.Guarantees.DampenerRemote` (`remoteDampener`, bounded by `hardwareMultiple`) and `benchmarks/nci`'s compute
dampener `D`; the padded size ratio `|A| / T_0` (`Overhead.padded`, `Estimate.padded`, the JSON's `padded`) is the
intermediate, not the slowdown. The glossary's `ρ` is the ACCEL model's resource rate, not this number.

## Extending

- **A training class**: an `Argument` in `classes.CLASSES` (a premise set, its count, its citations), and any new field
  it reads in `Profile`.
- **An attack**: an `Argument` with side `attack` in `attacks.ATTACKS`.
- **An algorithm family**: a table in `data/families.toml` of an existing kind, or a new kind in `families.KINDS`.
- **A model or run**: a table in `data/models.toml`.
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
