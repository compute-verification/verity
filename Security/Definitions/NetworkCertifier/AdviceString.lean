import Definitions.NetworkCertifier.Schedule

/-!
# The advice string: its charge per window, the compiled Program, and acceptance within caps

Definitions only (PROTOCOL.md, "Capacity", the advice string's row). The registered thing is a precommitted
ahead-of-time compiler. After serving, the prover hands it an **advice string**, which encodes everything serving made
unpredictable (I1: the requests, each step's batch rows, the release buckets). The compiler turns the advice into the
Program, which fixes every egress row from the certified ingress. The advice string is the content advice of
`constantRate`, and its length is the leak.

**What is abstract.** The charge is a parameter, `bits a w`: I1's `bits(advice, w)`, the leak charged to certifier
window `w`, which Python's `bits` mirrors. What it must satisfy for the count to hold is the named assumption
`BitsCount` (`Definitions.NetworkCertifier.Assumptions.AdviceString`). Advice v1's charge (`AdviceV1.bits`,
`ChargeCount_v1{N}`'s count) meets it, proved (`Guarantees.EncardChargedLe`), so its leak bounds take no assumption on
the charge (`Guarantees/AdviceV1` here and in `Models`).

**The caps.** `b w` is the public cap on window `w`'s charge, fixed before the run. Admissible advice is every value
charged at most `b w` in every window, and the bound charges the caps, not the realized charges: the sender picks the
advice, so the realized length is one of its choices.
-/

namespace NetTiming

/-- The advice is within the caps `b`: it is charged at most `b w` bits in every window `w`. -/
def WithinCaps {Adv : Type} {W : ℕ} (bits : Adv → Fin W → ℕ) (b : Fin W → ℕ) (a : Adv) : Prop :=
  ∀ w, bits a w ≤ b w

/-- **A compiler** (I2, `compile_steps(registered, advice)`): from the advice, its other inputs `Ω` and the certified
ingress, each egress link's frames in canonical order, with their anchors and logical indices. `Ω` is whatever else a
compiler could read: a clock, coins, or state of the run inside X. -/
abbrev Compiler (L : ℕ) (In Adv Ω : Type) : Type := Adv → Ω → List In → Fin L → List LogicalFrame

/-- **The compiled Program**: the constant-rate Program whose frames are the compiler's on the declared advice string,
its other inputs at `ω`. Its advice is the advice string and one clock-sync value per window. -/
def compiledProgram {L S : ℕ} {In Adv Ω : Type} (T W : ℕ) (ss : SyncSet) (C : Compiler L In Adv Ω) (ω : Ω) :
    Program L S In (Adv × (Fin W → ss.Elem)) :=
  constantRate T W ss fun ins a ℓ => C a ω ins ℓ

/-- **Accepted within the caps**: some declared advice `(a, x)`, whose advice string `a` is charged at most `b w` in
every window `w`, explains the first `k` steps (`Accepted`). -/
def AcceptedWithin {L K S W : ℕ} {In Adv X : Type} (bits : Adv → Fin W → ℕ) (b : Fin W → ℕ)
    (P : Program L S In (Adv × X)) (op : ℕ → List (Obs L K S) → In) (k : ℕ) (o : List (Obs L K S)) : Prop :=
  ∃ a x, WithinCaps bits b a ∧ Accepted P op k o (a, x)

/-- The observation sequences of length `n` whose jitter classes are below `D`, accepted within the caps `b`: what the
leak bound counts. -/
def AccSeqsWithin {L K S W : ℕ} {In Adv X : Type} (bits : Adv → Fin W → ℕ) (b : Fin W → ℕ)
    (P : Program L S In (Adv × X)) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ AcceptedWithin bits b P op n o}

end NetTiming
