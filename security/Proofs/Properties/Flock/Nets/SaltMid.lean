module

public import Flock.HmNets
import all Init.Data.String.Defs

/-! The salt prefix's midstate as numbers, so that a chunk's kernel run reads them rather than computing it again. -/

@[expose] public section

open Flock Flock.HmNets

namespace FlockDeploy.Concrete.Nets

def saltMidLit : List (BitVec 64) :=
  [15002581329388216570#64, 3101862183965343138#64, 2345057377162050185#64, 12402592305454372757#64,
    10804520247554037881#64, 8270622525771662394#64, 10916906814008749522#64, 15160754711223192558#64]

theorem saltMid_eq_lit : saltMid = saltMidLit := by decide +kernel

end FlockDeploy.Concrete.Nets
