module

public import Flock.HmNets
import all Init.Data.String.Defs

/-!
# The salt prefix's midstate as numbers

`saltMid` is SHA-512's compression of the salt tag's block; `saltMidLit` is its eight words, so that a chunk's kernel run
reads them rather than computing the compression again at each bit. The kernel computes it once here, which needs the
salt tag's bytes from core's `String.toUTF8` (`import all Init.Data.String.Defs`).
-/

@[expose] public section

open Flock Flock.HmNets

namespace FlockDeploy.Concrete.Nets

/-- The salt prefix's midstate's eight words. -/
def saltMidLit : List (BitVec 64) :=
  [15002581329388216570#64, 3101862183965343138#64, 2345057377162050185#64, 12402592305454372757#64,
    10804520247554037881#64, 8270622525771662394#64, 10916906814008749522#64, 15160754711223192558#64]

theorem saltMid_eq_lit : saltMid = saltMidLit := by decide +kernel

end FlockDeploy.Concrete.Nets
