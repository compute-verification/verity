# ρ: how much the regime slows training

One number: **ρ**, the size of the actual circuit that reaches a training target over the reference's, `|A| / T_0`, on
the datacenter the adversary actually runs (not only on the declared circuit NCI's verifier sees), and beside it the
**time to the target** that size takes on the inventory. Everything here is a way of bounding ρ from below (a bound: a
theorem or a draft statement, under premises) or from above (an attack: a construction, analytic or measured), at some
choice of every parameter, and of seeing how it moves.

~~~sh
uv run python benchmarks/rho/run.py                                  # every scenario: ρ's bracket, its time, δ, terms, tornado, dangers
uv run python benchmarks/rho/run.py --scenario reference-onsite-h200 --set regime.onsite.k_wipe=2e10
uv run python benchmarks/rho/run.py --figures /tmp/rho-figs             # and the figures (matplotlib)
verity bench rho                                                      # the same, as a runner (with $OUT: rho.md, rho.json, figures/)
~~~

## The model

The mathematics is the working draft's §8 ("From the circuit to the datacenter",
[Notion](https://app.notion.com/p/403399515d9e814a86b3e15740306ccb)), on the architecture locked 26 Oct (TOP-95,
`note:training/20261026T1840Z-draft-formal-properties`). This package implements it in three layers and doesn't restate
it; each module's docstring says what it implements.

- **Prior** (`hardware.py`, `data/hardware.toml`): the inventory's locations, each with its capacity `Φ_X` in gates per
  second and its servers' memory `M_s`, the dark leaves `ε_dark`, the horizon. `Units` is the one place a FLOP-model
  input (a datasheet's FLOP/s, the policy's `Γ`, the run's target, a class's or an attack's statement) becomes gates,
  through `units.gates_per_flop`; ρ and the time to the target don't depend on its value.
- **Envelopes** (`envelopes.py`, `data/envelopes.toml`): each subprotocol (PoUW, detection and sampled proofs, the wipe,
  the certifier) is one envelope with its parameters, its slack (the uncertified gates, keyed by route) and its `δ`,
  split into `δ_coin` and `δ_crypto` and summed over subprotocols and serving blocks (`Regime.delta`). The NCI check is
  the registration (`Policy`). A regime is the envelopes that run, and builds the serving blocks with their cuts.
- **Exclusion** (`classes.py`, `attacks.py`, `theorem.py`): the training classes as per-block caps `q_T(c, k)`, the
  attacks as constructions, and the serving-block bound with its local minima.

| §8 | here |
|---|---|
| `\|A[X,J]\| ≤ Φ_X \|J\|`, `M_s` | `Location.phi` (gate/s, from `location.<name>.flops`), `Location.server_memory` |
| `ε_dark`, `L_dark` | `fleet.<name>.eps_dark`, `Partition.eps_dark` |
| inference block, `(c, k)` | `Blocks("inference")`, `Policy` (`Γ` FLOPs, so `c = Context.c` gates), `Budget(gates, inputs)` |
| serving block `S`, `(c_S, k_S)`, `n_S` | `ServingBlock`, `Onsite.budget`, the inference cut's `Blocks.count` |
| `q_T(c, k)` | `Cap`, from a class's `Allowance` by `classes.per_block` |
| `q_U(U) = ω_max U / s_×`, `s_×` | `u · U`, `u = Structure.weight` (`structure.<name>.s_times`, `omega_max`, `data/exclusion.toml`), under `draft:no-straddle` at `s_×` > 1; ρ's normalizer, the reference's weight per gate, `w = classes.weight_per_gate` |
| `ε_undecl`, `ε_gap`, `ε_skip`, `ε_break`, Lean's `χ` | uncertified routes `undecl.*`, `gap.*`, `skip.*`, `break.*`, `exfil.faulty` (`Cut.uncertified`) |
| `E_i`, `δ_i(S)`, `δ_coin`, `δ_crypto`, `Pr[Break]` | `Pouw`, `Detection`, `Wipe`, `Certifier`; `Regime.delta → Delta` (`coin`, `crypto`, `by`); `Pr[Break]` named, not bounded |
| `k_wipe`, `r_in`, `r_side` | `Wipe.k_wipe` per server (`per = "server"`; per device: `residual_bits`, `host_residual_bits`), `Certifier.rate_in` (`intake` or `ingress`, `timing`, `side`) |
| the local minimum `min{q_T(c_S, k_S), n_S q_T(c, k) + q_U(S)}` | `theorem.local`; the minimum of global bounds `theorem.aggregate` (`slowdown(..., aggregate_only=True)`) |
| `\|C_{0,m}(G)\|/T_0`, `ρ`, `T ≥ ρ T_0 / Φ_total` | `Profile.efficiency` (`μ`), `Slowdown.rho`, `Scenario.seconds` |

Read a result as `ρ ∈ [lo, hi]`, with the time to the target at each end, `δ_coin + δ_crypto` (plus `Pr[Break]`) that the
envelopes fail, and the **monitored share** (`Slowdown.monitored`: the part of the adversary's counted weight on
monitored gates; the protocols do their job for an algorithm when it is small). Bounds count every gate at capacity, so
the time at `lo` is the idealized normalization (the reference at the inventory's full capacity); an attack is a
construction on a serving block's *layout* (what a protocol-abiding strategy gets) at the fleet's MFU, padded to
capacity, so its ρ carries its idle share and its time is exact. A bound above an attack is a **contradiction**: a
premise is false for that algorithm at that budget. `analysis.upto(status)` gives `lo` from only the bounds and cuts at
least as strong as a status, and the **operating points** (`operating.py`) turn ρ into time at an optimizer step or a
pretraining run, and exfiltrating a reference through side channels.

**The comparator.** ρ is normalized as the working draft's §8 normalizes it: `T ≥ ρ T_0 / Φ_total`, against the
reference at the inventory's full capacity, not at its MFU. An adversary that no protocol encumbers trains at its MFU on
a circuit padded to capacity, so it already sits at `ρ = μ / MFU`, about 2.5 at the reference fleets' 0.4
(`Result.unmonitored`). A lower end below that is **vacuous**, because it doesn't exclude even that adversary. `run.py`
prints the comparator beside every ρ and marks such a lower end "(vacuous)". The protocols show a slowdown only where
`lo` is above it, and only by `lo / (μ / MFU)`.

**The uncertified cap.** The bound counts an uncertified or a dark gate at `ω_max / s_×`, with `s_×` the fewest gates
in any counted operation's subcircuit. Until it is measured, `structure.reference` takes it at the reference's gates per
counted product, `2G / κ` (831 gates for the full family). That is the reference's average where the bound needs the
structural minimum. It is valid only if no counted operation straddles certified and uncertified gates, the named
assumption `draft:no-straddle`, which every bound counting such gates cites (so those bounds are `draft`). The
elementary cap, `structure.elementary` (one counted operation a gate, at most), needs no such premise. It is the
`*-elementary` scenarios, and what `analysis.upto` falls back to for any status stronger than `draft`.

## The modules

Each module is one interface with interchangeable implementations. A different assumption is a different implementation
(or different numbers in `data/`), never an edit to the modules around it.

| layer | module | interface | implementations (today) | an assumption here is |
|---|---|---|---|---|
| exclusion | `theorem.py` | `Budget`, `Blocks`, `Cut`, `ServingBlock`, `Partition`; `Cap = Budget → weight`; `local`, `aggregate`, `slowdown` | — | (none: the shape) |
| exclusion | `classes.py` | `Argument.allowance(Profile) → Allowance` on a `FlopBudget`, raising `Outside`; `per_block`, `weight_per_gate`; `counts` and `share` name the counted set; `premise_status(cites, k)` from `data/premises.toml`; `Structure` (`s_×`, `ω_max`: what a bound holds an uncertified gate at) from `data/exclusion.toml` | Lean (proved under named premises): cover, cover-aligned, cover-chained, covering, covering-two-costs, covering-additive, covering-priced, fanout, fanout-read, partial-sum, partial-sum-keyed, threaded; drafts (E1): fanout-priced, fanout-split, accumulator, output-premise | a premise set and its count |
| exclusion | `attacks.py` | the same `Argument`, side `attack`, on the serving block's layout | comply, state-fits (N1), shard (`benchmarks/nci`'s search), tile, measured executions | a construction |
| | `profile.py` | `Profile`: κ, state bits, `μ` (`efficiency`), and every field a class reads, named as in Lean | — | (the interface between the two) |
| | `families.py` | `Family.profile(Model, Run) → Profile`; precisions; a table may take its premises from another (`premises = "full"`) | full, adapter (LoRA kin, TinyLoRA), seed (ES, MeZO), low-comm (DiLoCo, SparseLoCo, DeMo), partial, distill, and any `benchmarks/nci` algorithm by its traits (`nci:<id>`) | an algorithm family and its parameters |
| | `instances.py` | `Model` (with weight matrices: `derived_legs`, `derived_fanout`), `Run` (a target: `6 N_active T` FLOPs unless given; kind pretrain, sft, rl-elicit or rl-agentic) | kimi-k2, deepseek-v3, deepseek-v4-pro, llama3-405b, dense-100b, qwen25-7b, qwen25-1b5; pretraining, RL and SFT runs | an architecture, a scale |
| prior | `hardware.py` | `Units` (gates a FLOP), `Location` (`Φ` in gate/s, memory, hosts, host memory and ops, line rate, `M_s`), `Fleet` (count, `ε_dark`, the MFUs a construction gets), `Horizon` | h100-hgx, h200-hgx, b200-hgx, gb200-nvl72, rtx-pro-6000-server, h200-hgx-rack4; 100k-GPU fleets, a 400k NVL72 fleet, the world-scale reference fleets | the actual datacenter |
| envelopes | `envelopes.py` | `Pouw`, `Detection`, `Wipe`, `Certifier`; `Regime.partition(Context) → Partition`, `Regime.delta(Context) → Delta`; `Policy` (Γ, k); a table may name a `base` and override what differs | unmonitored; remote (`DampenerRemote`'s terms, matching `tiers.py`), remote-sampled; onsite (`certified` or `links` serving cut, the wipe per server), onsite-today (about 12 GB a server plus its boot disk, `boot_disk_bits`), onsite-per-device, onsite-5gb, onsite-links, onsite-no-pouw | what the protocols certify, and how |
| | `params.py` | `Registry`: dotted keys over `data/*.toml`, `override`, `fields` | — | every number, with `low`/`high`, `source`, `status` |
| | `scenario.py` | `build`, `evaluate → Result` (per family: each bound, all bounds at once, each attack, each with its status and premise standing), `seconds`, `reads` | `data/scenarios*.toml`, `data/reference.toml` | a choice per layer |
| | `analysis.py` | over any keys: `sweep`, `tornado`, `threshold`, `dangers` and `rescues`, `corner` (every range at its worse end), `margins`, `phase` and `frontier`, `attribution` (what sets ρ), `upto(status)` | — | — |
| | `figures.py` | `render`: each class's per-block ρ against k, brackets by status, ρ against ε_dark, phase maps, tornado, vulnerability, corner (PNG plus the JSON plotted) | — | — |
| | `lean.py` | `status(name)`, `weakest(cites)` from the locks | — | — |
| | `operating.py` | `work`, `exfil`, `points`: ρ as time at the registry's `work.*` and `exfil.*` references | — | a reference |
| | `pofc.py`, `data/pofc.toml` | `apply(Scenario, Rules)`: declared work as `n_max`-token episodes whose operators form a one-way chain (`Budget.cut`), with the class `pofc-backprop` (the weight toll, in bits, draft) and the page's attacks, open only where their update reaches the next episode (`Rules.persist`: `outputs`, through `Y`, since registration happens once; `recommit`); `toll`, `versus` (ρ against `n_max`); a scenario opts in with `pofc = "<rules>"` | rules `reference`; the page's Llama-405B and Kimi K3 toll tables | a declared-circuit policy beside NCI's |

## Status

Every estimate carries the weakest status among its citations, read from the locks (`lean.py`): `proved`,
`conditional` (proved under named assumptions), `draft` (a paper statement not in Lean, or a `draft:` id) or `absent`.
Its citations are its argument's and those of the cuts that bound it (the inference cut's `DampenerRemote` premises, the
serving cut's `WindowsAt`, the wipe's per-server bound, the certifier's), and `draft:no-straddle` wherever a bound
holds uncertified or dark gates at a structure whose `s_×` is above 1. Every number in `data/` carries its own
`status` (`datasheet`, `measured`, `declared`, `assumed`, `ruled`, `conjecture`, `derived`) and `source`.

## Extending

- **A training class**: an `Argument` in `classes.CLASSES` (a premise set, its count, its citations), and any new field
  it reads in `Profile`.
- **An attack**: an `Argument` with side `attack` in `attacks.ATTACKS`.
- **An algorithm family**: a table in `data/families.toml` of an existing kind, or a new kind in `families.KINDS`.
- **A model or run**: a table in `data/models.toml`.
- **Hardware, a fleet**: a table in `data/hardware.toml`.
- **An envelope**: a dataclass with `slack` or `state` and `delta` in `envelopes.PARTS`, held by a regime.
- **A regime**: a dataclass with `partition` and `delta` in `envelopes.KINDS` and its table in `data/envelopes.toml`.
- **A scenario**: a table in `data/scenarios.toml` (Daniel's reference datacenter is `data/reference.toml`).
- **A work or exfiltration reference**: a `work.<name>.flops` or `exfil.<name>.bits` table.
- **A sensitivity**: give the parameter `low` and `high`; `tornado` and `dangers` pick it up.
- **A structure** (a measured `s_×`): a `structure.<name>` table in `data/exclusion.toml`, which a scenario names with
  `structure = "<name>"`.
