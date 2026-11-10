module

public import Definitions.WeightSecurity
public import Definitions.WeightSecurity.Tied

/-!
# Model weight security: what it rests on

`HonestUseHard` is the honest-use rate the guarantees state their bounds relative to: every party of a class steals the
weights from honest use of the served function with probability at most `p`. It holds at `p = p_hon(q)`, the best a
party of the class does, so what a bound adds to it is what leaks beyond honest serving. It is about the circuit's
input-output behavior alone, with nothing of Verity in it. Beside it:

* `Simulable`: the class holds the outsider with each covert string written in, which the class of all parties does.
  Like `HonestUseHard`, it is part of what the guarantees say: a condition on the class the rate is about.
* `LeaksThrough`: except with probability `ε`, what the outsider sees is what it would see in honest use through a
  covert channel of `|S|` covert strings. It is the audit's content, stated on the outputs' side: the channel's `tell`
  sees the honest answer, never an internal value of the circuit. It is a premise a deployment's guarantees are to
  discharge, taken as a hypothesis until they do.

`FaultBudget` and `patternCount` count what a party sets under SPOT's fault model: the parties that set at most `M`
faults over their queries, and the fault pattern sequences of total size at most `M`. What faults do to the weights'
exposure is the named assumption `FaultedUseHard` (`Definitions.Assumptions.WeightSecurity`).

Two premises tie a run's releases to the traces the audit extracts (`Definitions.WeightSecurity.Tied`), each the
probability of its failure on every run, until the layers below discharge it:

* `EgressTied`: a release is its trace's outputs, the network certifier's egress tie;
* `AdviceCovered`: a trace holds the weights', the query's and the advice's inputs, so every input the prover chooses
  is advice, which is charged.
-/

@[expose] public section

namespace WeightSecurity.Assumptions

open WeightSecurity

variable {Ω R X Y M Θ Θ' : Type}

/-- **Honest use hands out the weights with probability at most `p`.** With the weights sampled as `gen ω` on uniform
coins `ω`, every party of the class `A` that sees the served function `C`'s answers to `q` queries it chooses
adaptively, on coins `r` of its own, outputs a stolen model (`Stolen θ θ'`, e.g. `θ'` serves a function close to
`C θ`) with probability at most `p`. The guarantees state their bounds relative to it: it holds at `p = p_hon(q)`, the
best a party of `A` does, so what they bound is what leaks beyond honest serving.

How small `p_hon(q)` is, is the standard black-box model-extraction hardness, about `C`'s behavior on inputs and nothing
else: a query sets every input that isn't the weights, so a circuit with an input that makes it print its weights, or a
trigger only the weights' holder can find, has `p_hon(q)` near `1`. `A` makes it computational: the parties of a time
bound, say, since distillation from many queries succeeds given enough compute; at `A = Set.univ` it is
information-theoretic, and small only for few queries. `log₂ (1/p_hon(q))` is roughly the weights' min-entropy given
honest use. -/
def HonestUseHard [Fintype Ω] [Fintype R] (C : Θ → X → Y) (gen : Ω → Θ) (Stolen : Θ → Θ' → Prop) (q : ℕ)
    (A : Set (Outsider R X Y Θ')) (p : ℝ) : Prop :=
  ∀ P ∈ A, Claims.Pr (fun z : Ω × R => Stolen (gen z.1) (P.model z.2 (honest C (gen z.1)) q)) ≤ p

/-- PREMISE, the audit's to discharge. **What leaves goes through the channel** `ch`, except with probability `ε`:
against the datacenter `real`, which knows the weights and may do anything, the outsider `O`'s view after `q` rounds
is, except with probability `ε` over the weights and the coins, its view against the datacenter that deviates through
`ch` with some covert string `s` (`Channel.resp`). The string may depend on everything, the weights included; nothing
else about the weights reaches the outsider. -/
def LeaksThrough [Fintype Ω] [Fintype R] (C : Θ → X → Y) (gen : Ω → Θ) (real : Θ → R → List (X × M) → X → M)
    (O : Outsider R X M Θ') (q : ℕ) (ch : Channel R X Y M) (ε : ℝ) : Prop :=
  Claims.Pr (fun z : Ω × R =>
    ¬ ∃ s, O.view z.2 (real (gen z.1) z.2) q = O.view z.2 (ch.resp C s (gen z.1) z.2) q) ≤ ε

/-- **The class holds the outsider with every covert string written in** (`Outsider.withString`), a condition on the
class the guarantees are stated for: the honest-use parties the proof builds from `O` are ones `HonestUseHard` is
about. The class of all parties (`A = Set.univ`) holds it; at a time bound, `O`'s time plus the channel's `ask` and
`tell` must be within it. -/
def Simulable (A : Set (Outsider R X Y Θ')) (O : Outsider R X M Θ') (ch : Channel R X Y M) : Prop :=
  ∀ s, O.withString ch s ∈ A

variable {F : Type}

/-- **The parties that set at most `M` faults**: each query carries a fault pattern `φ : F` beside its input, and on
every coin and against every responder the patterns of the first `q` queries have sizes summing to at most `M`. -/
def FaultBudget (size : F → ℕ) (q M : ℕ) : Set (Outsider R (X × F) Y Θ') :=
  {P | ∀ r resp, ((P.view r resp q).map fun e => size e.1.2).sum ≤ M}

/-- **The number of fault pattern sequences**: sequences of `q` patterns whose sizes sum to at most `M`. Its `log₂` is
the faults' arbitrary-data capacity: about `M · (c + ℓ)` at patterns that name proof blocks and the values they write,
`c` a block's output bits and `ℓ` the bits that name a block. -/
noncomputable def patternCount [Fintype F] (size : F → ℕ) (q M : ℕ) : ℕ :=
  Fintype.card {φ : Fin q → F // ∑ t, size (φ t) ≤ M}

variable {O I V Tick Adv S Q : Type}

open scoped ENNReal in
/-- PREMISE, the network certifier's to discharge. **The releases are their traces' outputs**, except with probability
`η`: on every run `run θ r`, some accepted window that releases released something other than its extracted trace's
values on the egress slots `eg` (`Served.Traced.Tied`) with probability at most `η`. The certifier's egress tie gives it
at `η = 0`: the released bytes are the extracted trace's outputs. -/
def EgressTied (eg : O × I → S) (L : ℕ) (η : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served.Traced O (I → V) Tick Adv S V))) : Prop :=
  ∀ θ r, Verity.Game.value (fun l => ¬ Served.Traced.Tied eg L l) (run θ r) ≤ η

open scoped ENNReal in
/-- PREMISE, the advice charge's to discharge. **The traces hold the windows' inputs**, except with probability `η`: on
every run `run θ r`, some accepted window that releases holds, on the population `P`'s input slots, something other than
the inputs of the weights `θ` and of the outsider's query served with the window's advice (`Served.Traced.Covered`),
with probability at most `η`. It holds at `η = 0` when every input the prover chooses is advice, which is charged. -/
def AdviceCovered {U : Type} (P : Population.Dataflow U S) (inp : Θ → Q × Adv → S → V)
    (Ou : Outsider R (Q × Adv) (Option ((O → I → V) × Tick)) Θ') (L : ℕ) (η : ℝ≥0∞)
    (run : Θ → R → Verity.Game (List (Served.Traced O (I → V) Tick Adv S V))) : Prop :=
  ∀ θ r, Verity.Game.value (fun l => ¬ Served.Traced.Covered P inp Ou L θ r l) (run θ r) ≤ η

end WeightSecurity.Assumptions
