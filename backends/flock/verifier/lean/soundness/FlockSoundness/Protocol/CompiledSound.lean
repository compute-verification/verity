import FlockSoundness.Protocol.CompiledLock
import FlockSoundness.Protocol.LigeritoPhase
import FlockSoundness.Protocol.LinkSound

/-!
# Compiled soundness under collision resistance (README.md §1.2)

The oracle prover `P̃` sends, at each commitment, the plurality table of the openings the
compiled prover `P*` makes there (`Lock.oracleStrat`). A compiled run that accepts while some
opening disagrees with the fixed tables either opens a position two ways within the run
(`SelfClash`) or opens a position off the table fixed at some cap (`bad_split`). By rewinding at
every cap (`Lock.leafE_off_le`), the second costs `√(N · Q · Adv)` per cap, where `Adv` is the
probability that two continuations from the cap open a position differently; each such pair,
like each self-clash, is a SHA-512 collision (`Merkle.opening_binding`).
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- A level's parameters. -/
abbrev lvOf (sch : Schedule) (ℓ : ℕ) : Level := sch.levels.getD ℓ default

/-- A row as a function of its lanes. -/
def rowFin (lanes : ℕ) (row : List F) : Fin (2 ^ lanes) → F := fun lane => row.getD lane.val 0

/-! ## Per-cap openings -/

/-- The table with the given rows on a level's domain. -/
def tabOf (lanes d : ℕ) (v : Fin (2 ^ d) → Fin (2 ^ lanes) → F) : Oracle F :=
  fun lane q => if h : q < 2 ^ d ∧ lane < 2 ^ lanes then v ⟨q, h.1⟩ ⟨lane, h.2⟩ else 0

/-! ## Where a bad run goes wrong -/

/-! ## Clashes and conflicts are collisions -/

/-! ## The plurality prover and the collision finders -/

/-! ## The theorem -/

end FlockSoundness
