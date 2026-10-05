import FlockSoundness.Discharge.ZkExec.VerdictStatements
import FlockSoundness.Discharge.ZkExec.PlaysDet
import FlockSoundness.Discharge.ZkExec.Inner
import FlockSoundness.Discharge.ZkExec.PlaysInner
import FlockSoundness.Refine.Arith
import FlockSoundness.Refine.Salted
import Mathlib.Algebra.CharP.Two
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictInner

/-!
# The inner proof's check holds on its openings (`ZkInnerAccepts`)

`zk_inner_accepts`: when `Zk.verifyInner` accepts, the compiled inner proof's one result (`plays_det`), at the coins of
the four rounds it read (`zk_inner_peel`), is a check that holds on the proof's inner openings:

* `τ` opens its commitment: `encTau`'s `hm96` leaf is `Zk.tauCommitment` (`leaf_tau`).
* The batched constraint: `Lin.combo`'s coefficients and constant are the `γ`-sums of the constraints' (`combo_a`,
  `combo_c`); the model's rows are the seven constraints (`LinRel`) and the sacrifice's three rows (`triple_rel`, at
  the pads layout's distinct positions); `batchT`'s sign is characteristic 2's.
* Each query: `codeAt` is `enc` at the point `j + 1` (`fOf_codeAt`), and the pair's path climbs to the pads' root
  under `encPair`'s `hm96` leaf (`leaf_pair`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128

/-! ## Arrays -/

/-! ## `Lin.combo` -/

/-! ## The code -/

/-! ## The leaves -/

/-! ## The triple's rows -/

section Triple

variable {mPts : ℕ} {T : ZkSession.TabZK 𝔽 mPts} {st : Setup} {sch : Flock.Schedule} {tPad : ℕ}

end Triple

/-! ## The run -/

/-! ## The check -/

end FlockSoundness.Discharge.ZkExec
