# ρ: how much the regime slows training

One number: **ρ**, the factor by which monitoring slows an adversary's training to a target, on the datacenter it
actually runs (not only on the declared circuit NCI's verifier sees). Everything here is a way of bounding ρ from below
(a bound: a theorem or a draft statement, under premises) or from above (an attack: a construction, analytic or
measured), at some choice of every parameter, and of seeing how it moves.

~~~sh
uv run python benchmarks/rho/run.py                                  # every scenario: ρ's bracket, its terms, tornado, dangers
uv run python benchmarks/rho/run.py --scenario onsite-h200-kimi --set regime.onsite.residual_bits=4e10
verity bench rho                                                      # the same, as a runner
~~~

## The shape

A training run needs *counted products* (the multiplications of its reference computation, `κ/2` per FLOP). The
monitored fleet over the horizon is cut into **regions** (a location over a window). Each region is covered, in one or
more valid ways (**cuts**), by **blocks** with a **budget** `(C, K)`: the FLOPs a block runs and the novel bits that enter
it, plus *free* FLOPs that count at face value. A bound caps what one block can finish (its *allowance* `a(C, K)`), so a
region contributes the least of its cuts; the adversary's dark share `d` (one connected pool) runs unencumbered:

```
g = Σ_R n_R · min(κC_R/2, min over cuts [Σ_B n_B · min(a(C_B, K_B), κC_B/2) + κ·free/2]) / Σ_R n_R · κC_R/2
ρ = m / (d + (1 − d) g)
```

`m` is the compute the algorithm needs to reach the target, in multiples of the reference algorithm's; the adversary
picks its algorithm, so the scenario's ρ is the least over the families it may use. Bounds give `lo`, attacks `hi`
(evaluated on the region's *layout*: what a protocol-abiding strategy actually gets), and a bound above an attack is a
**contradiction**: a premise is false for that algorithm at that budget.

Read a result as: `ρ ∈ [lo, hi]`, with `g` beside it and the **monitored share** `(1 − d) g / (d + (1 − d) g)` (the
part of the adversary's training done on monitored compute: the protocols do their job for an algorithm when it is
small, `g ≪ d`). Its **ladder** gives `lo` admitting bounds by status: proved, proved under named premises
(`conditional`), and with drafts. Its **operating points** (`operating.py`) turn ρ into time: an optimizer step or a
pretraining run, and exfiltrating a reference through side channels.

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
| training classes | `classes.py` | `Argument.allowance(Profile) → Allowance`, raising `Outside` | cover, cover-chained, fanout, fanout-read, partial-sum, threaded (Lean, proved under named premises); fanout-priced (draft Prop. 3) | a premise set and its count |
| attacks | `attacks.py` | the same `Argument`, side `attack` | comply, state-fits (N1) | a construction |
| profile | `profile.py` | `Profile`: κ, state bits, `m`, and every field a class reads, named as in Lean | — | (the interface between the two) |
| families | `families.py` | `Family.profile(Model, Run) → Profile` | full | an algorithm family and its parameters |
| instances | `instances.py` | `Model`, `Run` (a target: `6 N_active T` FLOPs unless given) | kimi-k2, deepseek-v4-pro; their pretraining runs, one V4 optimizer step | an architecture, a scale |
| hardware | `hardware.py` | `Location` (Φ, memory), `Fleet` (count, dark share, MFU), `Horizon` | h200-hgx, nvl72; reference-h200, world-h200, world-nvl72 | the actual datacenter |
| regimes | `regimes.py` | `Regime.partition(Context) → Partition`, with `Policy` (Γ, k) | unmonitored, remote, onsite | what the protocols certify, and how |
| params | `params.py` | `Registry`: dotted keys over `data/*.toml`, `override`, `fields` | — | every number, with `low`/`high`, `source`, `status` |
| scenario | `scenario.py` | `build`, `evaluate → Result` (per family: each bound, all bounds at once, each attack) | `data/scenarios.toml` | a choice per layer |
| analysis | `analysis.py` | `sweep`, `tornado`, `threshold`, `dangers`, `phase` over any keys | — | — |
| Lean status | `lean.py` | `status(name)`, `weakest(cites)` from the locks | — | — |
| operating points | `operating.py` | `work`, `exfil`, `points`: ρ as time at the registry's `work.*` and `exfil.*` references | — | a reference |

### Declared and actual circuits

The NCI verifier of record enforces the policy `(Γ, k)` on the **declared** circuit. The **actual** circuit is what the
fleet computes, and the regimes map it onto blocks:

- **declared cut** (remote and onsite): the certified share of a region runs declared blocks `(Γ, k)`; every other FLOP
  (PoUW's residual `r̂` and deficit `ε_W`, the slack `ε'`, the verdict lag `JΔ/H`, what sampled proofs let go undone `χ`)
  is free at face value. It rests on `DampenerRemote`'s premises (PoUW, terminal and sampled detection) and the
  verifier's enforcement.
- **physical cut** (onsite): a location over a wipe window is one block, `(Φ Δτ, R + I_cert Δτ + β_side Δτ)`. It rests on
  causality on the actual circuit's annotation: `WindowsAt` on `Accounting.Placed` (`width_into_region_le`, `Links`,
  `Covered`; ASK-141), with the wipe's residual `R`, the certifier's ingress `I_cert` and side channels `β_side`.

A location is the largest set of devices joined by links no certifier sees (an HGX server's NVLink domain, an NVL72
rack). Remotely nothing bounds what enters a location, so free FLOPs are free; onsite the window's input caps them too.

### Status

Every estimate carries the weakest status among its citations, read from the locks (`lean.py`): `proved`,
`conditional` (proved under named assumptions), `draft` (a paper statement not in Lean) or `absent`. Its citations are
its argument's and those of the cuts that bound it (the declared cut's `DampenerRemote` premises, the physical cut's
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
