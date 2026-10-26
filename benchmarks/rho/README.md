# ρ: how much the regime slows training

One number: **ρ**, the factor by which monitoring slows an adversary's training to a target, on the datacenter it
actually runs (not only on the declared circuit NCI's verifier sees). Everything here is a way of bounding ρ from below
(a bound: a theorem or a draft statement, under premises) or from above (an attack: a construction, analytic or
measured), at some choice of every parameter, and of seeing how it moves.

~~~sh
uv run python benchmarks/rho/run.py                                  # every scenario: ρ's bracket, its terms, tornado, dangers
uv run python benchmarks/rho/run.py --scenario onsite-h200-kimi --set regime.onsite.residual_bits=4e10
uv run python benchmarks/rho/run.py --figures /tmp/rho-figs             # and the figures (matplotlib)
verity bench rho                                                      # the same, as a runner (with $OUT: rho.md, rho.json, figures/)
~~~

## The shape

A training run needs *counted products* (the multiplications of its reference computation, `κ/2` per FLOP). The
monitored fleet over the horizon is cut into **regions** (a location over a window). Each region is covered, in one or
more valid ways (**cuts**), by **blocks** with a **budget** `(C, K)`: the FLOPs a block runs and the novel bits that enter
it, plus *free* FLOPs that count at face value. A bound caps what one block can finish (its *allowance* `a(C, K)`), so a
region contributes the least of its cuts; the dark pool's share `ε_dark` of the adversary's training rate runs
unencumbered:

```
g = Σ_R n_R · min(κC_R/2, min over cuts [Σ_B n_B · min(a(C_B, K_B), κC_B/2) + κ·free/2]) / Σ_R n_R · κC_R/2
ρ̄ = μ / (ε_dark + (1 − ε_dark) g)
```

`μ` is the compute the algorithm needs to reach the target, in multiples of the reference algorithm's; the adversary
picks its algorithm, so the scenario's ρ is the least over the families it may use. Bounds give `lo`, attacks `hi`
(evaluated on the region's *layout*: what a protocol-abiding strategy actually gets), and a bound above an attack is a
**contradiction**: a premise is false for that algorithm at that budget.

Read a result as: `ρ̄ ∈ [lo, hi]`, with `g` beside it, the chance `δ_coin + δ_crypto` that the lower end fails, and the
**monitored share** `(1 − ε_dark) g / (ε_dark + (1 − ε_dark) g)` (the part of the adversary's training done on monitored
compute: the protocols do their job for an algorithm when it is small, `g ≪ ε_dark`). `analysis.upto(status)` gives `lo` from only the bounds and cuts at least as strong as a status:
proved, proved under named premises (`conditional`), or with drafts. Its **operating points** (`operating.py`) turn ρ
into time: an optimizer step or a pretraining run, and exfiltrating a reference through side channels.

## The hierarchy

The slowdown theorem has one shape (above). Below it, **training classes** each derive a per-block bound from a premise
set (cover, fan-out, partial sums, threads, ...). Below those, **algorithm families** (pretraining, post-training,
adapters, communication-efficient training, seed-based methods, ...) are parameterized, and each lands in the classes
whose premises it meets, through its profile. Last, **instances** fix scales and architectures (a model, a run's
target). Beside that stack, the **actual circuit**: the hardware, and the regime that maps what the fleet computes onto
the theorem's blocks.

## The layers

Each layer is one interface with interchangeable implementations. A different assumption is a different
implementation (or different numbers in `data/`), never an edit to the layers around it.

```
                   ┌──────────────────────── scenario: one choice per layer ────────────────────────┐
 declared circuit  │ theorem   Partition, Allowance → g, ρ             (knows no GPU, protocol, class) │  actual circuit
                   │   ▲ allowance a(C,K)                         ▲ partition (regions, cuts, layout) │
                   │ classes / attacks   Argument(Profile) → a     regimes   Regime(Context) → Partition│
                   │   ▲ profile                                   ▲ fleet, location, policy, horizon │
                   │ families  Family(Model, Run) → Profile        hardware  Location, Fleet, Horizon │
                   │   ▲ model, run                                                                    │
                   │ instances Model, Run (target)                                                     │
                   └──────────── params: every number, its range and its source (data/*.toml) ─────────┘
                                 analysis: sweep, tornado, threshold, dangers, phase
```

| layer | module | interface | implementations (today) | an assumption here is |
|---|---|---|---|---|
| theorem | `theorem.py` | `Budget`, `Blocks`, `Cut`, `Region`, `Partition`; `Allowance = Budget → products`; `slowdown` | — | (none: the shape) |
| training classes | `classes.py` | `Argument.allowance(Profile) → Allowance`, raising `Outside`; `counts` and `share` name the counted set; `premise_status(cites, k)` from `data/premises.toml` | Lean (proved under named premises): cover, cover-aligned, cover-chained, covering, covering-two-costs, covering-additive, covering-priced, fanout, fanout-read, partial-sum, partial-sum-keyed, threaded; drafts (E1): fanout-priced, fanout-split, accumulator, output-premise | a premise set and its count |
| attacks | `attacks.py` | the same `Argument`, side `attack`, on the region's layout | comply, state-fits (N1), shard (`benchmarks/nci`'s search), tile, measured executions | a construction |
| profile | `profile.py` | `Profile`: κ, state bits, `μ` (`efficiency`), and every field a class reads, named as in Lean | — | (the interface between the two) |
| families | `families.py` | `Family.profile(Model, Run) → Profile`; precisions; a table may take its premises from another (`premises = "full"`) | full, adapter (LoRA kin, TinyLoRA), seed (ES, MeZO), low-comm (DiLoCo, SparseLoCo, DeMo), partial, distill, and any `benchmarks/nci` algorithm by its traits (`nci:<id>`) | an algorithm family and its parameters |
| instances | `instances.py` | `Model` (with weight matrices: `derived_legs`, `derived_fanout`), `Run` (a target: `6 N_active T` FLOPs unless given; kind pretrain, sft, rl-elicit or rl-agentic) | kimi-k2, deepseek-v3, deepseek-v4-pro, llama3-405b, dense-100b, qwen25-7b, qwen25-1b5; pretraining, RL and SFT runs | an architecture, a scale |
| hardware | `hardware.py` | `Location` (Φ, memory, hosts, host memory and ops, line rate), `Fleet` (count, `ε_dark`, MFU, the dark pool's MFU: `eps_dark_rate`), `Horizon` | h100-hgx, h200-hgx, b200-hgx, gb200-nvl72, rtx-pro-6000-server, h200-hgx-rack4; 100k-GPU fleets, a 400k NVL72 fleet, the world-scale reference fleets | the actual datacenter |
| regimes | `regimes.py` | `Regime.partition(Context) → Partition`, with `Policy` (Γ, k); a table may name a `base` and override what differs | unmonitored; remote (`DampenerRemote`'s terms, matching `tiers.py`), remote-sampled; onsite (`certified` or `links` serving cut), onsite-5gb, onsite-links, onsite-no-pouw | what the protocols certify, and how |
| params | `params.py` | `Registry`: dotted keys over `data/*.toml`, `override`, `fields` | — | every number, with `low`/`high`, `source`, `status` |
| scenario | `scenario.py` | `build`, `evaluate → Result` (per family: each bound, all bounds at once, each attack, each with its status and premise standing), `reads` | `data/scenarios*.toml`, `data/reference.toml` | a choice per layer |
| analysis | `analysis.py` | over any keys: `sweep`, `tornado`, `threshold`, `dangers` and `rescues`, `corner` (every range at its worse end), `margins`, `phase` and `frontier`, `attribution` (what sets ρ), `upto(status)` | — | — |
| figures | `figures.py` | `render`: each class's per-block ρ against k, brackets by status, ρ̄ against ε_dark, phase maps, tornado, vulnerability, corner (PNG plus the JSON plotted) | — | — |
| Lean status | `lean.py` | `status(name)`, `weakest(cites)` from the locks | — | — |
| operating points | `operating.py` | `work`, `exfil`, `points`: ρ as time at the registry's `work.*` and `exfil.*` references | — | a reference |
| proof of forward computation | `pofc.py`, `data/pofc.toml` | `apply(Scenario, Rules)`: declared work as `n_max`-token episodes whose operators form a one-way chain (`Budget.cut`), with the class `pofc-backprop` (the weight toll, in bits, draft) and the page's attacks, open only where their update reaches the next episode (`Rules.persist`: `outputs`, through `Y`, since registration happens once; `recommit`); `toll`, `versus` (ρ against `n_max`); a scenario opts in with `pofc = "<rules>"` | rules `reference`; the page's Llama-405B and Kimi K3 toll tables | a declared-circuit policy beside NCI's |

### Notation

The names follow the working draft's §8 ("From the circuit to the datacenter"), which defines them and proposes, but
does not establish, their composition. Renaming changed no accounting: this model's numbers are its earlier accounting
under these names, not a validation of §8.

| draft | here | meaning |
|---|---|---|
| declared block `I`, `(c, k)` | `Blocks("declared")`, `Policy` `(Γ, k)` | an element of the declared partition |
| serving block `S`, `(c_S, k_S)` | the `serving` cut, `Onsite.budget` | one location between two wipes |
| `ε_dark` | `fleet.<name>.eps_dark`, `Fleet.eps_dark_rate`, `Partition.eps_dark` | compute nobody found, a share of the fleet's |
| `ε_undecl` | free source `undecl.host` | known compute running no declared work |
| `ε_gap` | `gap.floor` (`residual`), `gap.clock` | declared work certified for less than it occupied |
| `ε_skip` | `skip.pouw` (`ε_W`), `skip.certs` (`ε'`) | credited without running |
| `ε_break` | `break.lag` (`JΔ/H`), `break.escape` (`θ`) | fails a check, runs until its identity halts |
| Lean's `χ`; `k_exfil` of unchecked blocks | `exfil.faulty`; the faulty-outputs row | the compute injected bits buy; what they carry out |
| `k_wipe`, `r_in`, `r_side` | `residual_bits` and `host_residual_bits`, certified ingress, `timing` and `side` | what enters a serving block |
| `δ_coin`, `δ_crypto` | `Regime.delta()`, from `window_delta_coin` and `window_delta_crypto` | the declared partition's terms only: not the wipe's or the certifier's |
| `\|C_{0,m}(G)\|/T_0`, `ρ̄` | `Profile.efficiency`, `Slowdown.rho` | a method's cost relative to the reference; the deployment slowdown, a proposed quantity |

`exfil.faulty` is Lean's `χ`, the compute that injected bits buy, at `chord` products a bit; the faulty-outputs row is
what they carry out. They answer different questions. Replacing `χ` by per-block allowances needs its own theorem
(§8), so it stays. Shares are of peak capacity (Conventions).

### Declared and actual circuits

The NCI verifier of record enforces the policy `(Γ, k)` on the **declared** circuit. The **actual** circuit is what the
fleet computes, and the regimes map it onto blocks:

- **declared cut** (remote and onsite): the certified share of a region runs declared blocks `(Γ, k)`; every other
  unit of capacity is free at face value, at peak as Lean counts it, keyed by its route: `undecl` (host
  CPUs), `gap` (the GPUs' uncredited floor and clock above the credit; with `undecl`, `r̂`), `skip` (PoUW's deficit
  `ε_W`, the slack `ε'`), `break` (the verdict lag `JΔ/H`, the slack `θ` for faulty blocks accepted before detection) and `exfil` (Lean's `χ`). It rests on `DampenerRemote`'s premises (PoUW, terminal and sampled detection) and the
  verifier's enforcement.
- **serving cut** (onsite): a serving block, a location between two wipes, is one block, `(Φ Δτ, K)`, by causality on the actual
  circuit's annotation (`width_into_region_le`, `Links`, `Covered`; ASK-141). In its `links` form, what Lean proves
  today (`Windows`), `K` is the location's memory plus its line rate over the window. In its `certified` form (the
  default), `K = R + I_cert Δτ + (timing + β_side) Δτ`: what a wipe leaves per device and per host, the certifier's
  ingress and the channels no certifier reads; it needs `WindowsAt` read as information rather than width, which is a
  draft.

A location is the largest set of devices joined by links no certifier sees (an HGX server's NVLink domain, an NVL72
rack). Remotely nothing bounds what enters a location, so free FLOPs are free; onsite the serving cut puts all of a
location's compute in one block, which rests on `WindowsAt` and the cut's other citations.

### Conventions

Bounds count monitored capacity at peak, as Lean's dampener and `benchmarks/nci/tiers.py` do: a region's declared work
and free capacity are shares of `Φ`, against an unencumbered trainer at `MFU₀`. So a bound must cut a block's products
by more than `1/MFU₀` before it binds at all, and an uncertified share above `MFU₀` alone leaves `g = 1`. Attacks run on
the layout at the fleet's own MFU, which is what a strategy achieves. The gap between the two conventions is part of
every bracket: remote's `[1, 2.35]` at today's audited residual is mostly that.

### Status

Every estimate carries the weakest status among its citations, read from the locks (`lean.py`): `proved`,
`conditional` (proved under named assumptions), `draft` (a paper statement not in Lean) or `absent`. Its citations are
its argument's and those of the cuts that bound it (the declared cut's `DampenerRemote` premises, the serving cut's
`WindowsAt`). Every number in
`data/` carries its own `status` (`datasheet`, `measured`, `declared`, `assumed`, `ruled`, `conjecture`, `derived`) and
`source`.

## Extending

- **A training class**: an `Argument` in `classes.CLASSES` (a premise set, its count, its citations), and any new field
  it reads in `Profile`.
- **An attack**: an `Argument` with side `attack` in `attacks.ATTACKS`.
- **An algorithm family**: a table in `data/families.toml` of an existing kind, or a new kind in `families.KINDS`.
- **A model or run**: a table in `data/models.toml`.
- **Hardware, a fleet**: a table in `data/hardware.toml`.
- **A regime**: a dataclass with `partition` in `regimes.KINDS` and its table in `data/regimes.toml`.
- **A scenario**: a table in `data/scenarios.toml` (Daniel's reference datacenter is `data/reference.toml`).
- **A work or exfiltration reference**: a `work.<name>.flops` or `exfil.<name>.bits` table.
- **A sensitivity**: give the parameter `low` and `high`; `tornado` and `dangers` pick it up.
