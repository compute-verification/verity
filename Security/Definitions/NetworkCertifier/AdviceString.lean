import Definitions.NetworkCertifier.Schedule

/-!
# The advice string: its window segments and charges, the compiled Program, and acceptance within a budget

Definitions only (PROTOCOL.md, "Capacity", the advice-string row). The registered thing is a precommitted
ahead-of-time compiler. After serving, the prover hands it an **advice string**, which encodes everything serving made
unpredictable (I1: the requests, each step's batch rows, the release buckets). The compiler turns the advice into the
Program, which fixes every egress row from the certified ingress. The advice string is the content advice of
`constantRate`, and its length is the leak.

**What is abstract.** The codec is a parameter: `seg a w` is window `w`'s segment of `encode(a)`, and `bits a w` is
I1's `bits(advice, w)`, the leak charged to certifier window `w`. Python's `encode` and `bits` mirror them. What they
must satisfy for the count to hold is stated as the named assumptions `SegmentsDecode` and `BitsCount`
(`Models.NetworkCertifier.Assumptions.AdviceString`).

**The budget.** `b w` is the public cap on window `w`'s charge, fixed before the run. Admissible advice is every value
charged at most `b w` in every window, and the bound charges the cap, not the realized length: the sender picks the
advice, so the realized length is one of its choices.
-/

namespace NetTiming

/-- The advice string's codec, per certifier window: `seg a w` is window `w`'s segment of `encode(a)`, and `bits a w`
is the leak charged to window `w`, I1's `bits(advice, window)`. -/
structure AdviceCodec (Adv : Type) (W : ℕ) where
  seg : Adv → Fin W → List Bool
  bits : Adv → Fin W → ℕ

/-- The advice is within the budget `b`: it is charged at most `b w` bits in every window `w`. -/
def AdviceCodec.Within {Adv : Type} {W : ℕ} (code : AdviceCodec Adv W) (b : Fin W → ℕ) (a : Adv) : Prop :=
  ∀ w, code.bits a w ≤ b w

/-- **A compiler** (I2, `compile_steps(registered, advice)`): from the advice, its other inputs `Ω` and the certified
ingress, each egress link's frames in canonical order, with their anchors and logical indices. `Ω` is whatever else a
compiler could read: a clock, coins, or state of the run inside X. -/
abbrev Compiler (L : ℕ) (In Adv Ω : Type) : Type := Adv → Ω → List In → Fin L → List LogicalFrame

/-- **The compiled Program**: the constant-rate Program whose frames are the compiler's on the declared advice string,
its other inputs at `ω`. Its advice is the advice string and one clock-sync value per window. -/
def compiledProgram {L S : ℕ} {In Adv Ω : Type} (T W : ℕ) (ss : SyncSet) (C : Compiler L In Adv Ω) (ω : Ω) :
    Program L S In (Adv × (Fin W → ss.Elem)) :=
  constantRate T W ss fun ins a ℓ => C a ω ins ℓ

/-- **Accepted within the budget**: some declared advice `(a, x)`, whose advice string `a` is within `b`, explains the
first `k` steps (`Accepted`). -/
def AcceptedWithin {L K S W : ℕ} {In Adv X : Type} (code : AdviceCodec Adv W) (b : Fin W → ℕ)
    (P : Program L S In (Adv × X)) (op : ℕ → List (Obs L K S) → In) (k : ℕ) (o : List (Obs L K S)) : Prop :=
  ∃ a x, code.Within b a ∧ Accepted P op k o (a, x)

/-- The observation sequences of length `n` whose jitter classes are below `D`, accepted within the budget `b`: what
the leak bound counts. -/
def AccSeqsWithin {L K S W : ℕ} {In Adv X : Type} (code : AdviceCodec Adv W) (b : Fin W → ℕ)
    (P : Program L S In (Adv × X)) (op : ℕ → List (Obs L K S) → In) (n D : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ AcceptedWithin code b P op n o}

end NetTiming
