import Proofs.Flock.Soundness.Discharge.Exec.Shape
import Proofs.Flock.Soundness.Model.Sparse

/-!
# The sparse phase's shape

`sparseKinds`: the kinds of the steps of a rep's sparse phase, in order, which `sparsePhase_exact` (`Read.lean`) proves
the phase takes, and the guarantee's coin types read (`Flock.Guarantees.holoCTs`).
-/

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
