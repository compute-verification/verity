import Definitions.Flock.Audit.OneStage

/-! The definitions of `Proofs.Flock.Soundness.Audit.Stratified` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Law

variable {n m : ℕ}

/-- Stratum `s` of the assignment `σ`: the units it assigns to `s`. -/
def stratum (σ : Fin n → Fin m) (s : Fin m) : Finset (Fin n) := univ.filter (σ · = s)

/-- **Independent uniform subsets inside the strata of `σ`**, `k s` of stratum `s`. -/
def stratified (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) : Law n where
  Ω := (s : Fin m) → ((stratum σ s).powersetCard (k s))
  nonempty := ⟨fun s => ⟨_, (Finset.powersetCard_nonempty.2 (hk s)).choose_spec⟩⟩
  draw ω := univ.biUnion fun s => (ω s).1

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {k : Fin m → ℕ} {hk : ∀ s, k s ≤ (Law.stratum σ s).card}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

end Audits

end FlockSoundness.Audit
