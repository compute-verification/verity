import Proofs.Flock.Soundness.Discharge.Exec.Shape
import Definitions.Flock.Sparse

/-!
# The stages' shapes that the guarantee's coin types read

The kinds of the steps of candidate 3's stages beyond `Exec/Shape.lean`'s, in order, which the guarantee's coin types
read (`Flock.Guarantees.holoCTs`, `Recursive/HoloCoins.lean`):

* `holoLcShape`, `holoRepShape`: a rep's lincheck and a rep with hidden units, which `holoRepC_spine`
  (`C3/HoloValues/Shape.lean`) proves a rep takes;
* `sparseKinds`: a rep's sparse phase, which `sparsePhase_exact` (`C3/Sparse/Read.lean`) proves the phase takes.
-/

namespace FlockC3.HoloValues

open FlockSoundness.Accounting FlockSoundness.Discharge.Exec

/-- The lincheck's shape with hidden units (L1–L5 with L4′): `lcShape` with `h` after `z_partial`. -/
def holoLcShape (F : Type) (kLog : ℕ) : List Kind :=
  .coin F :: .coin F :: (pairRounds F (kLog - 6) ++ [.msg (Fin 64 → F), .msg (Fin 64 → F), .coin F])

/-- **A rep's shape with hidden units**: `repShape` with `holoLcShape`. -/
def holoRepShape (F K D : Type) (m kLog nExtra : ℕ) (sch : Schedule) : List Kind :=
  zcShape F m ++ (holoLcShape F kLog ++ (opShape F (nExtra + 2) ++ ligShape F K D sch))

end FlockC3.HoloValues

namespace FlockC3.Sparse

open FlockSoundness.Model FlockSoundness.Discharge.Exec

/-- `n` rounds of `degRounds k`: each round's message, then its coin. -/
def degKinds (F : Type) (k : ℕ) : ℕ → List Kind
  | 0 => []
  | n + 1 => .msg (F × (Fin k → F)) :: .coin F :: degKinds F k n

/-- **The sparse phase's shape**: H1's six coins, H2's six rounds and `v₂`, H3's rounds over the entries and over the
slots and the planes `u`, H4's coins `ζ_p`, the types' values where `T > 1`, and the coins `ρ_t`. -/
def sparseKinds (F : Type) (sh : SparseShape) : List Kind :=
  List.replicate 6 (.coin F) ++ (degKinds F 1 6 ++ ([.msg F] ++ (degKinds F (2 * sh.ell) sh.nu ++
    (degKinds F (2 * sh.ell + 1) sh.sLog ++ ([.msg (Fin (2 ^ sh.pLog) → F)] ++ (List.replicate sh.pLog (.coin F) ++
      ((if sh.tLog = 0 then [] else [.msg (Fin (2 ^ sh.tLog) → F)]) ++ List.replicate sh.tLog (.coin F))))))))

end FlockC3.Sparse
