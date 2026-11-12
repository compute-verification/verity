# Local bounds: a survey of assumptions and bounds

This is the machinery for one question: **how much training can a small inference block do?** The inference-only
policy cuts verified work into blocks of at most c gates, each taking in at most k fresh bits (bits that aren't
registered information). A block can **import** information, spending fresh bits, or **reconstruct** it, spending
gates. Every bound here says, in one ontology, how far those two budgets go. The Lean is `Nci.Capability` (the
vocabulary), `Nci.Assumptions` (the premises) and `Nci.Guarantees` (the bounds), one module per role. How the bounds
enter the security argument is `security/README.md`, part 2. Over counted products, training's bound of record is the
min-cut theorem (`security/README.md`, Start here); the bounds here bound what a block holds and the uses it serves.

## The ontology

| Object | Symbol | Lean | Meaning |
|---|---|---|---|
| Committed declaration | R | `Verity.Committed` | the registered circuit, the registered values θ, and a partition into B blocks |
| Block | I | `Verity.Partition.IsCK` | c gates, at most k fresh bits; its **view** is those bits and its index |
| Decoder budget | t | `Verity.LDL` | the time a reconstruction may take |
| Item family | X | `Nci.Capability.Items` | things a block may hold; each item has **cells** (the gates of one version of its value) and a **price** p_i in bits |
| Selection | (U, σ) | `Nci.Capability.Selects` | some items, one cell each |
| Holding | | `Nci.Capability.holds` | a block holds a cell if it lies among the block's gates and what they read |
| Budget | h | `Nci.Capability.HeldPriceAtMost` | every block holds selections priced at most h |
| Counted products | 𝒥 | `Nci.Covering.MulBlocks` | product (i, j): updated slot i times an operand of use j |
| Consumer | c | `Nci.Capability.Consumers` | something a block finishes (a use, an accumulated output), with an upstream region A_c and required gates G_c |
| Certificate | | `Nci.Capability.Consumers.Upstream` | every required gate reaches, inside the region, what the finishing gates read |
| Overlap weights | ω | `Nci.Capability.Consumers.OverlapWeights` | per-consumer weights summing to at most 1 over the regions holding any gate |
| Weighted total | Ω | | the counted work at its weights, such as the consumers' ω; with every weight 1, the plain count |
| Block yield | q | | the most of the weighted total one block can do |
| Coverage | Core, Ω_*, T_max, η | `Nci.Assumptions.CoversEfficientMethods` | cheap successful methods carry at least Ω_* covered work |

## The shape of every bound

```text
information premise + a short view   →  budget h          (what a block holds)
budget + a structure                 →  per-block cap q   (what it can do with it)
cap + counting                       →  |C| ≥ c·Ω/q        (a circuit floor)
floor + coverage                     →  success below the floor has probability ≤ η
floor + the protocol's accounting    →  a fleet-time floor, and the slowdown
```

The pieces are modular by construction:

- **A premise reaches the counting only through its budget.** Any premise that yields `HeldPriceAtMost` works with any
  structure that consumes one.
- **Caps compose.** For the same blocks, the smallest cap whose premises all hold applies; a refuted premise contributes
  nothing. Over disjoint counted families (forward products, backward products, optimizer updates), the caps and the
  weighted totals add.
- **Weights are shared.** A cap bounds Ω in its weights, and coverage has to supply Ω_* in the same weights.
- **A new kind of dependency is a new `Consumers` value, not a new argument.** Uses, accumulated outputs and anything
  else with an upstream region go through the same three dependency bounds.
- **Improving a structure never strengthens a premise.** A stronger premise is a new hypothesis, named as one.

## Information: what is hard to hold

### Two measures, two claims

| | Computational: `DescriptionHard` | Statistical: `MinEntropyHard` |
|---|---|---|
| Claims | the values are **expensive to recompute** | the values are **unpredictable** |
| Statement | LDL_{t,R}(x_U) ≥ Σ_{i∈U} p_i − λ | every value of x_U has probability ≤ 2^(λ − Σ p_i) under the run's randomness |
| View bound | Lemma 1: LDL_{t,R}(view) ≤ k + ⌈log₂ B⌉ + a_blk | counting: a block's view takes at most 2^k values |
| Budget | `HeldBudget`: h = k' + λ | `StatHeldBudget`: h = k + λ, with no decoder and no time budget |
| Holds even if | training is a deterministic function of registered information, as long as recomputing it is slow | |
| Fails if | a short program rebuilds the values within t (replay, low rank, a seed) | training is a deterministic function of registered information (public data in a registered order, registered seeds) |
| Test | find short time-bounded decoders for adversarially chosen selections; each one found bounds the price | retrain from the same registration with fresh randomness: a weight that varies by s quantization steps carries about log₂ s bits |

Both concern exact values. A method that learns from approximate values is a different method, and coverage has to
account for it. A third measure lies between the two and isn't in Lean: computational entropy (no bounded distinguisher
tells the values from high-entropy ones), tested by distinguishing experiments.

### Item families

The meaning of a premise is its item family's.

| Family | Premise | Price measures | Fails where |
|---|---|---|---|
| updated parameter slots, one version each | update hardness | bits per fresh weight | the update has a short description (low rank, few distinct values, a seed), the training prefix replays within t, or the slots are predictable from R |
| optimizer state (moments), one version each | (a direction) | bits per moment | the state is a cheap function of the weights and gradients the block already holds |
| operand occurrences, one scalar each | activation hardness | bits per operand | a block cheaply generates operands from what it holds; several uniform rates are refuted by exact decoders |
| slots and operands together | joint hardness | an additive budget | the same bits describe both families |

### A derived bound: recovery

`SourceHard` and `Recovers` prove an information bound on a block's outputs instead of assuming one. A source z of length
ℓ_z with LDL_{τ,R}(z) ≥ ℓ_z − δ, recoverable from a block's program and a complement of at most
ℓ_z − β_v·|𝒥_I| + amb bits, gives β_v·|𝒥_I| ≤ k' + δ + amb (`RecoveredBudget`). Ambiguity that grows with the outputs,
amb ≤ ε·|𝒥_I| with ε < β_v, lowers the information per output to β_v − ε. Correlated outputs need not shorten the
complement one coordinate each (for an exact linear map the rank counts), and δ is paid in every block's bound.

## Structure: what held values support

All of these are properties of a named circuit, checked on it; none is a premise.

**Dependency.** Every consumer a block finishes is imported (it spends a fresh bit of the consumer's region) or
recomputed (the block contains its required gates):

- `ImportOrRecompute`: Σ_{finished} ω_c ≤ k + Σ_{recomputed} ω_c;
- `RecomputeByGates`: Σ_{recomputed} ω_c ≤ c/g, g the fewest required gates of a consumer;
- `RecomputeByState`: no recomputation when recomputing needs more updated state than the budget.

So a block's served weight is at most k + c/g, or k when recomputing is priced out. The consumers so far:

| Consumer | `Consumers` | Region | Required gates |
|---|---|---|---|
| a use: one token's input to one layer at one step | `ofUses` | that token's previous layer | its products, over many updated slots |
| an accumulated output: a weight's gradient summed over tokens | `ofOutputs` | the sum's tree | its addends' products and partial sums |

**Threading.** `Nci.Threading.Threaded` with `Nci.Assumptions.WeightsIncompressible` assumes read costs instead of
deriving them; under them `Nci.Guarantees.ThreadedReads` proves q(c, k) = (k/β)·max(a_out, w_out·k/(4·a_in·b_w)). It is a
candidate sufficient condition, not part of the repertoire.

## The bounds

| Bound | From | Gives | Lean | Status |
|---|---|---|---|---|
| Reconstruction | the view | LDL_{t,R}(view) ≤ k' | `Verity.Guarantees.BlockDescription` | proved |
| Held budget | description hardness | h = k' + λ | `Nci.Guarantees.HeldBudget` | proved |
| Statistical held budget | min-entropy hardness | h = k + λ | `Nci.Guarantees.StatHeldBudget` | proved |
| Recovered budget | source hardness, recovery | β_v·\|𝒥_I\| ≤ k' + δ + amb | `Nci.Guarantees.RecoveredBudget` | proved |
| Import or recompute | certificate, weights | finished ≤ k + recomputed | `Nci.Guarantees.ImportOrRecompute` | proved |
| Recompute by gates | weights | recomputed ≤ c/g | `Nci.Guarantees.RecomputeByGates` | proved |
| Recompute by state | budget | recomputed = 0 | `Nci.Guarantees.RecomputeByState` | proved |
| Counting | a block yield | \|C\| ≥ c·Ω/q | `Verity.Guarantees.Counting` | proved |
| Coverage floor | coverage | Pr[success below the floor] ≤ η | `Nci.Guarantees.CapabilityFloor` | proved |

`HeldBudget` takes an evaluator of at most a_blk bits per held selection. With one evaluator for every selection, the
empty selection would force the decoder to output nothing on every block's program, and the premises could hold only
when no block holds anything.

Each bound has two witnesses, listed under the lock's `witnesses` and proved: an honest one, its premises holding
together, and a nontrivial one, its conclusion failing where a premise is dropped. They are stated at the toy circuit
`Nci.Capability.Toy` (four one-bit gates, one counted product; `Guarantees/Capability/Witnesses.lean`), and
`CapabilityFloor`'s at the toy task `Accounting.Slowdown.Toy`. The structural premises are proved there, and so are the
information premises, none degenerately. `MinEntropyHard` holds at Λ = 0 for the product's value, a uniform bit. The toy
decoder compresses: it describes `11`, the value of gates 2 and 3 in an honest run, in one bit, and every other two-bit
string in three. Against it `DescriptionHard` holds at Λ = 1, `SourceHard` at δ = 1 and `Recovers` at amb = 1, and each
witness also proves its parameter spent: at 0 the premise fails. Coverage holds at η = 1/2 and fails at η = 0: the
covered structure is two gates, and a one-gate guess lacks it and reaches the target on half the coins.

## Standard assemblies

| Information | Structure | Block yield |
|---|---|---|
| update hardness, either measure | uses, recomputing priced out | served weight ≤ k |
| none | uses, recomputing by gates | served weight ≤ k + c/g |
| update hardness | accumulated outputs | gradient sums finished ≤ k + c/g, each needing its addends |
| a recoverable hard source | outputs | (k' + δ + amb)/β_v |

**When a cap binds.** A cap matters only when it is well below what a block could do anyway, about c·ζ̄ of the
weighted total.

## How to test each premise

| Premise | Experiment | What refutes it |
|---|---|---|
| update hardness, computational | compress real fine-tunes' updates with time-bounded decoders: codecs, low-rank fits, predictors from θ, short replays, on adversarially chosen slot sets | a decoder well under the price per slot |
| update hardness, statistical | retrain from the same registration with fresh randomness; measure each weight's spread in quantization steps | spreads near one step: the update is predictable |
| activation hardness | exact decoding of operands given what a block holds | a decoder well under β_x per operand |
| the certificate | check `Upstream` on the training circuit | a required gate with no path inside its region |
| coverage | compute to a matched target for compact-state methods (adapters, small experts, distillation from served outputs, frozen features) | a cheap success that escapes every assembly |

## The frontier

- **One meaningful instantiation.** The witnesses so far are a four-gate toy: they show the premises consistent, each
  parameter spent, against a decoder that compresses one string. The next advance is one instance on a training
  circuit satisfying them simultaneously: a block-extraction capability at the protocol's decoder budget, a positive
  information rate, a checked dependency certificate, and parameters giving a nontrivial cap. A failed instantiation
  says concretely which premise breaks.
- **A use's import cost.** The dependency bounds charge one fresh bit per imported use. A justified joint input cost of
  b bits per use gives a served weight of at most k/b + c/g. A hidden state's literal width is not that cost:
  compression, shared descriptions and reconstruction from what the block holds all lower it. The question is the
  justified joint input cost of supporting additional uses, and which structural bound it enables.
- **An output budget.** With a bound on a block's outgoing bits as well (onsite's network certification bounds a
  server's traffic), the Loomis–Whitney inequality bounds a matrix product's work by the square root of the product of
  its three projections.
- **Weights by linear programming.** With the prices and the certificate fixed, the best overlap weights solve a linear
  program over the training circuit.
- **Conditional prices.** A price given side information (a region's imported values, the registered model's outputs)
  sharpens a premise and is a new hypothesis each time.
- **Serial depth.** Fleet-time bounds total work. A bound on wall-clock time needs the training's sequential depth, a
  separate structure.
