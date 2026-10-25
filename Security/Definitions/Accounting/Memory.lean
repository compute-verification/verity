import Definitions.Pous.Model.Audit
import Definitions.Pous.Model.SampledAudit

/-!
# Memory adds, by counting: the vocabulary (trusted)

Lemma M of the actual-circuit model (note `console/20261012T0645Z-draft-actual-circuit-model`, section 5). During a
PoUS audit, PoUS's responder is the location `X` together with its helper table, and before each challenge `j` it holds
a state of `mem_X(t_j) + 2^{e_j} h_j ≤ M + H` bits: what `X` holds, read as the string of its values in a fixed order of
position (leaf, then tick) without the positions, and the table. Besides the PoUS store `X` keeps a `q`-bit value `z`,
uniform and independent of PoUS's setup and challenges, that can be read back from that state. Fixing `z` leaves a
PoUS adversary whose states are those that read back as `z`: the fibre of `z` (`fibre`). Fibres of different `z` are
disjoint, so few `z` have a large one, and PoUS's own guarantee bounds the acceptance of the rest.

`fibre` is what the counting statement counts, and `Determines` is the condition on a family of PoUS adversaries,
one for each `z`, that the state held before every challenge reads back as `z`. The statements are in
`Security/Models/Accounting/Guarantees/Memory.lean`.
-/

namespace Accounting.Memory

/-- The `M`-bit states from which `π` reads `z`. -/
def fibre {M q : ℕ} (π : (Fin M → Bool) → Fin q → Bool) (z : Fin q → Bool) : Finset (Fin M → Bool) :=
  Finset.univ.filter fun σ => π σ = z

/-- **The state held before each challenge determines `z`.** For each `z`, `A₁ z` and `step z` are the location's PoUS
adversary: its preprocessing and its responder's step, with an `N`-bit state. In every run of the `c`-challenge audit
at `W` (each primitive `ω`, encoder coins `r` and challenges `I`), the state held before challenge `j < c` is the one
left after answering the first `j` challenges (`Pous.Sampled.seqState`, from the preprocessing's output), and `π j`
reads `z` from it. Only the states the audit reaches are read. -/
def Determines {I : Pous.IdealModel} (sch : Pous.Scheme I) {N q : ℕ} (c : ℕ) (W : Pous.Bits sch.n)
    (π : ℕ → Pous.Bits N → Pous.Bits q) (A₁ : Pous.Bits q → Pous.Preprocessor sch N)
    (step : Pous.Bits q → Pous.Bits sch.n → Pous.Bits sch.t → ℕ → Pous.Bits N → Fin sch.B →
      Pous.Prog I (Pous.Bits sch.ℓ × Pous.Bits N)) : Prop :=
  ∀ (z : Pous.Bits q) (x : Pous.Outcome sch c) (j : ℕ), j < c →
    π j (Pous.Sampled.seqState (I.answer x.1) (step z W (sch.enc W x.1 x.2.1).1) 0 ((List.ofFn x.2.2).take j)
      (A₁ z W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1)) = z

/-- The `M`-bit states from which `π` reads `z` with some `a`-bit advice. -/
def fibreAdvice {M a q : ℕ} (π : (Fin M → Bool) → (Fin a → Bool) → Fin q → Bool) (z : Fin q → Bool) :
    Finset (Fin M → Bool) :=
  Finset.univ.filter fun σ => ∃ α, π σ α = z

/-- **The state held before each challenge determines `z` with `a` bits of advice.** `Determines`, with `π j` reading
`z` from the state and an `a`-bit advice value. The advice is whatever recovers `z` in that run: any function of `z`
and the run's outcome (primitive, coins, challenges) and of `j`, with at most `2^a` values. -/
def DeterminesAdvice {I : Pous.IdealModel} (sch : Pous.Scheme I) {N a q : ℕ} (c : ℕ) (W : Pous.Bits sch.n)
    (π : ℕ → Pous.Bits N → Pous.Bits a → Pous.Bits q) (A₁ : Pous.Bits q → Pous.Preprocessor sch N)
    (step : Pous.Bits q → Pous.Bits sch.n → Pous.Bits sch.t → ℕ → Pous.Bits N → Fin sch.B →
      Pous.Prog I (Pous.Bits sch.ℓ × Pous.Bits N)) : Prop :=
  ∀ (z : Pous.Bits q) (x : Pous.Outcome sch c) (j : ℕ), j < c → ∃ α : Pous.Bits a,
    π j (Pous.Sampled.seqState (I.answer x.1) (step z W (sch.enc W x.1 x.2.1).1) 0 ((List.ofFn x.2.2).take j)
      (A₁ z W (sch.enc W x.1 x.2.1).1 (sch.enc W x.1 x.2.1).2 x.1)) α = z

end Accounting.Memory
