module

public import Proofs.Flock.Soundness.Discharge.Exec.Shape
public import Definitions.Flock.Sparse
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Combinatorics.Matroid.Init
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import VCVio.Prelude.Core

/-! The kinds of candidate 3's stages beyond `Exec/Shape.lean`'s, which the guarantee's coin types read (`Flock.Guarantees.holoCTs`). -/

@[expose] public section

namespace FlockC3.HoloValues

open FlockSoundness.Accounting FlockSoundness.Discharge.Exec

def holoLcShape (F : Type) (kLog : ℕ) : List Kind :=
  .coin F :: .coin F :: (pairRounds F (kLog - 6) ++ [.msg (Fin 64 → F), .msg (Fin 64 → F), .coin F])

def holoRepShape (F K D : Type) (m kLog nExtra : ℕ) (sch : Schedule) : List Kind :=
  zcShape F m ++ (holoLcShape F kLog ++ (opShape F (nExtra + 2) ++ ligShape F K D sch))

end FlockC3.HoloValues

namespace FlockC3.Sparse

open FlockSoundness.Model FlockSoundness.Discharge.Exec

def degKinds (F : Type) (k : ℕ) : ℕ → List Kind
  | 0 => []
  | n + 1 => .msg (F × (Fin k → F)) :: .coin F :: degKinds F k n

def sparseKinds (F : Type) (sh : SparseShape) : List Kind :=
  List.replicate 6 (.coin F) ++ (degKinds F 1 6 ++ ([.msg F] ++ (degKinds F (2 * sh.ell) sh.nu ++
    (degKinds F (2 * sh.ell + 1) sh.sLog ++ ([.msg (Fin (2 ^ sh.pLog) → F)] ++ (List.replicate sh.pLog (.coin F) ++
      ((if sh.tLog = 0 then [] else [.msg (Fin (2 ^ sh.tLog) → F)]) ++ List.replicate sh.tLog (.coin F))))))))

end FlockC3.Sparse
