import Definitions.Pous.Model.Audit

/-!
# Memory adds, by counting: the vocabulary (trusted)

Lemma M of the actual-circuit model (note `console/20261012T0645Z-draft-actual-circuit-model`, section 5). During a
PoUS audit a location holds, before each challenge, a state of `M` bits (its `mem_X(t)`, at most `M_X`). Besides the
PoUS store it keeps a `q`-bit value `z`, uniform and independent of PoUS's setup and challenges, that can be read back
from that state. Fixing `z` leaves a PoUS adversary whose states are those that read back as `z`: the fibre of `z`
(`fibre`). Fibres of different `z` are disjoint, so few `z` have a large one, and PoUS's own guarantee bounds the
acceptance of the rest.

`fibre` is what the counting statement counts, and `Determines` is the condition on a family of PoUS adversaries,
one for each `z`, that the state held before every challenge reads back as `z`. The statements are in
`Security/Models/Accounting/Guarantees/Memory.lean`.
-/

namespace Accounting.Memory

/-- The `M`-bit states from which `π` reads `z`. -/
def fibre {M q : ℕ} (π : (Fin M → Bool) → Fin q → Bool) (z : Fin q → Bool) : Finset (Fin M → Bool) :=
  Finset.univ.filter fun σ => π σ = z

/-- **The state held before each challenge determines `z`.** For each `z`, `A₁ z` and `step z` are the location's PoUS
adversary: its preprocessing and its responder's step, each with an `M`-bit state. Before challenge `j < c` the
location holds the preprocessing's output (`j = 0`) or the state that round `j − 1` returned, and `π j` reads `z` from
it. This is asked only of the states the audit reaches: at every `W` and setup `(ω, r)`, with the step answering
on that setup's primitive `ω` and starting from a state that reads back as `z`. -/
def Determines {I : Pous.IdealModel} (sch : Pous.Scheme I) {M q : ℕ} (c : ℕ)
    (π : ℕ → Pous.Bits M → Pous.Bits q) (A₁ : Pous.Bits q → Pous.Preprocessor sch M)
    (step : Pous.Bits q → Pous.Bits sch.n → Pous.Bits sch.t → ℕ → Pous.Bits M → Fin sch.B →
      Pous.Prog I (Pous.Bits sch.ℓ × Pous.Bits M)) : Prop :=
  (0 < c → ∀ z W ω r, π 0 (A₁ z W (sch.enc W ω r).1 (sch.enc W ω r).2 ω) = z) ∧
    ∀ z W ω r j σ i, j + 1 < c → π j σ = z →
      π (j + 1) ((step z W (sch.enc W ω r).1 j σ i).run (I.answer ω)).2 = z

end Accounting.Memory
