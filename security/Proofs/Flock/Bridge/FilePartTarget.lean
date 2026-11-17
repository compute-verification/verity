module

public import Proofs.Flock.Bridge.Statement
public import Spot.FilePart

/-! `FilePartTarget`: the partition the claims read at a circuit file is the one the verifier's code computes from it (a
target, proved by `Flock.Bridge.filePartTarget`). -/

@[expose] public section

namespace Flock.Bridge

def FilePartTarget : Prop :=
  ∀ (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (_hlog : D.unroll.unit.unitLog ≤ 32) (n : ℕ)
    (u : Fin n),
    let P := (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)
    ((Spot.FilePart.ofCircuit D.unroll n).block u.val).toList =
      blockOf P (fun g => constsOf D.unroll hU n g) u

end Flock.Bridge
