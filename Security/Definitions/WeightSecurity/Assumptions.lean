module

public import Definitions.WeightSecurity

/-!
# Model weight security: what it rests on

`HonestUseHard` is the one assumption about the world: honest use of the served function doesn't hand out the weights.
It is about the circuit's input-output behavior alone, with nothing of Verity in it. The other two are premises a
deployment's guarantees are to discharge, taken as hypotheses until they do:

* `LeaksThrough`: except with probability `ε`, what the outsider sees is what it would see in honest use through a
  covert channel of `|S|` covert strings. It is the audit's content, stated on the outputs' side: the channel's `tell`
  sees the honest answer, never an internal value of the circuit.
* `Simulable`: the class the assumption is about holds the outsider with each covert string written in, which the
  class of all parties does.
-/

@[expose] public section

namespace WeightSecurity.Assumptions

open WeightSecurity

variable {Ω R X Y M Θ Θ' : Type}

/-- `honest-use-hard` (proposed; about the served function alone). **Honest use doesn't hand out the weights.** With
the weights sampled as `gen ω` on uniform coins `ω`, every party of the class `A` that sees the served function `C`'s
answers to `q` queries it chooses adaptively, on coins `r` of its own, outputs a stolen model (`Stolen θ θ'`, e.g. `θ'`
serves a function close to `C θ`) with probability at most `p`.

It is the standard black-box model-extraction hardness, about `C`'s behavior on inputs and nothing else: a query sets
every input that isn't the weights, so a circuit with an input that makes it print its weights, or a trigger only the
weights' holder can find, fails it. `A` makes it computational: the parties of a time bound, say, since distillation
from many queries succeeds given enough compute; at `A = Set.univ` it is information-theoretic, and holds only for few
queries. `p` is `p_hon(q)`; `log₂ (1/p)` is roughly the weights' min-entropy given honest use. -/
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

/-- PREMISE, of the class. **The class holds the outsider with every covert string written in**
(`Outsider.withString`): the honest-use parties the proof builds from `O` are ones `HonestUseHard` is about. Every
class holds it at `A = Set.univ`; at a time bound, `O`'s time plus the channel's `ask` and `tell` must be within it. -/
def Simulable (A : Set (Outsider R X Y Θ')) (O : Outsider R X M Θ') (ch : Channel R X Y M) : Prop :=
  ∀ s, O.withString ch s ∈ A

end WeightSecurity.Assumptions
