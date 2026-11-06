module

public import Proofs.Flock.Bridge.Statement
public import SampledProofs.FilePart

/-!
# The circuit file's partition, computed (a target, not proved)

`FilePartTarget`: the partition the claims read at a circuit file is the one the verifier's code computes from it. At
the parsed descriptor `D`, its unit's rows `hU` and the population `n`, unit `u`'s proof block in the code's summary
(`SampledProofs.FilePart.ofCircuit D.unroll n`, O(the unit), then `block u`) is `blockOf` of the key program's partition
at the circuit's constants (`constsOf`): its rows' lowered gates, then the gates of the sources it owns. So at strata
`ns`, `partOf P κ ns` is those blocks in the strata `ns`, which the population check (`SampledProofs.FilePart.check`)
holds to the registration's strata over the public file's units in order. With it, `RelDeployTarget`'s partition
hypothesis is discharged at a deployed partition built from `ofCircuit`.
-/

@[expose] public section

namespace Flock.Bridge

/-- **The code's partition is `partOf`'s**: at every descriptor `D`, unit rows `hU`, population `n` and unit `u`,
`FilePart.ofCircuit`'s block `u` is `blockOf` of the key program's partition at the circuit's constants. -/
def FilePartTarget : Prop :=
  ∀ (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (u : Fin n),
    let P := (FlockSoundness.Discharge.Placed.keyProg D.unroll hU n).partition
      (FlockSoundness.Discharge.Layout.outsOf D.unroll hU n)
    ((SampledProofs.FilePart.ofCircuit D.unroll n).block u.val).toList =
      blockOf P (fun g => constsOf D.unroll hU n g) u

end Flock.Bridge
