import Definitions.Flock.Audit.Work

/-! The definitions of `Proofs.Flock.Soundness.Audit.Closure` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal

namespace Law

variable {n m : ℕ}

/-- The units made unsound by the wrong-unit set `B`: those whose closure meets it. -/
def unsoundTiles (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) : Finset (Fin n) :=
  univ.filter fun t => ¬ Disjoint (cl t) B

/-- **Closure draws**: `L`'s draw, each drawn unit proved together with its closure. -/
def closure (L : Law n) (cl : Fin n → Finset (Fin n)) : Law n :=
  L.widen fun S => S ∪ S.biUnion cl

/-- The work of the wrong units `B` and of the units they make unsound. -/
abbrev unsoundWork (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (B : Finset (Fin n)) : ℕ :=
  workOf σ w (B ∪ unsoundTiles cl B)

/-- **The harm of a unit**: its own work and the work of every unit whose closure holds it, what it spoils when wrong. -/
def harm (σ : Fin n → Fin m) (w : Fin m → ℕ) (cl : Fin n → Finset (Fin n)) (u : Fin n) : ℕ :=
  workOf σ w (insert u (univ.filter fun t => u ∈ cl t))

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ} {cl : Fin n → Finset (Fin n)}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

end Audits

end FlockSoundness.Audit
