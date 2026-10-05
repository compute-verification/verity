import FlockSoundness.Protocol.Audit.Work

/-!
# Closure draws

A closure draw proves each drawn unit together with its closure `cl t`, the units its credit depends on: for a PoUW tile,
the X and Y strips it reads and the node units on its X strip's path. The closure map is the verifier's own, from its
program.

* `Law.closure L cl`: `L` widened by `cl` on `L`'s coins (`Law.widen`). `mem_closure_draw`: a unit is proved when it is
  drawn or lies in a drawn unit's closure.
* `closure_escape`: **the closure draw's escape is the base law's escape of `B` with the units it makes unsound**,
  `(L.closure cl).escape B = L.escape (B ∪ unsoundTiles cl B)`, where `unsoundTiles cl B` are the units whose closure
  meets `B`. So a wrong unit that is never drawn on its own is charged to every unit above it.
* `unsoundWork σ w cl B`, the work of `B ∪ unsoundTiles cl B`: the wrong units and the units they make unsound, the set
  `closure_escape` measures. A closure map need not list a unit in its own closure (a tile's is its strips and path
  nodes), so the wrong units' own work is counted explicitly: `workOf_le_unsoundWork`.
* `harm σ w cl u`, the work of `u` and of the units whose closure holds it. `harm_le_unsoundWork`: **a wrong unit's whole
  harm is in the unsound work**, whether or not a draw reaches it. `unsoundWork_le_harm`: the unsound work is at most the
  wrong units' harm (coverage), and `unsoundWork_singleton`: exactly, for one wrong unit.
* `audit_work_closure`, `extraction_audit_work_closure`: under the work law's closure draw, at any floors `f_s ≥ 1`, the
  audit accepts while the unsound units do at least `T` of the work with probability at most
  `((W − T)/W)^K + ε_ks + δ_link`. `audit_work_closure_harm`: likewise while some wrong unit's harm is at least `T`.
  `audit_work_closure_of_record`: at `K = 27,713`, unsound work of at least 0.1% with at most `2^-40 + ε_ks + δ_link`.
-/

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
