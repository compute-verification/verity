import FlockSoundness.Discharge.Composed.Base
import FlockSoundness.Instance
import FlockSoundness.Discharge.Composed.Inner.ZerocheckMasked
import FlockSoundness.Discharge.Composed.Inner.Lincheck
import FlockSoundness.Protocol.Discharge.Composed.Inner

/-!
# The inner proof's fields of a draw's `--zk` table

`coinedOf`'s tables take the fields the inner proof's constraint `⟨λ, h⟩ = θ` reads (`Table.InnerHolds`) from `innerOf`:
the level-0 lanes, the clear values, Ligerito's value lists and the verifier's replay `λ`, `θ`, at a coin vector
`c : CoinsZK (RestOf Tz st)`. `innerOf_holds` (`Composed/Inner/Holds.lean`) is its obligation.

* **The pads by layout index** (`padEquiv`, `padNat`): a rep's pads are `Zk.PadLayout`'s `len = nE + 4` slots, the
  exposed values first and then `hab, tx, ty, tw`; `PadIdx`'s tail is `h_tx, h_ty, h_ab, h_tw` (`Table.hFull`).
* **The replay** (`rowsOf`, `lamOf`, `thetaOf`): `Zk.verifyRep`'s ten constraints in its order, each an affine form in the
  pads: zerocheck c_eval, zerocheck final a·b and lincheck final (`Zk.replay`'s forms on the masked proof, restated in
  `ZerocheckMasked` and `Lincheck`, at the drawn coins), ring switch ab and c (their values plus `Zk.claimCheck` on `ŝ`),
  read in the field (`ofZk`); level 0's lane phase c0 and c1 (`Zk.ligeritoConstraints`, on the restart `T'` that `lig`
  carries, `execArith`'s forms); and the triple's three (`Zk.triple`). `λ`, `θ` are their batch at the coins `γ`
  (`Zk.verifyInner`: `⟨l.a, h⟩ = l.c`). The lincheck's circuit fold is the statement's (`modelFold`).
* **The honest prover's clear values** (`honestAt`, by layout index): the clear zerocheck `Zerocheck.honest` on the A-side,
  B-side and witness bits (`zcH`), the lincheck's rounds `Lincheck.lcMsg` and `z_partial` (`lcH`, `zpH`); level 0's
  out-of-domain value, lane rounds and the extra lanes' sums `e_i`, on `z = (w, u)` (`zWith`).
* **The lanes** (`lanesOf`): packed word `y = l + 2^k₀·c` of `z` is lane `l`'s column `c`.
* **Levels ≥ 1** (`ligOf`): the restart `T' = Σ_c y₁'(c)·b̄(c)` and levels ≥ 1's bytes.

**Open inputs** (each a named definition, until the coin vector carries it): level 0's out-of-domain point
(`oodPoint`, the drawn coin `RestZK.oodPt`) and levels ≥ 1's bytes (a `LigProver`'s). `InnerHolds` reads none of them.
-/

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

noncomputable section

/-! ## The pads by layout index -/

/-- A pad's layout index (`padEquiv`). -/
def padNat {nE : ℕ} (i : PadIdx nE) : ℕ := (padEquiv nE i).val

/-- `Lin.masked`: the masked value at layout index `k`, unmasked by its pad. -/
def mk {nE : ℕ} (v : Fin nE → 𝔽) (k : ℕ) : LinF 𝔽 (nE + 4) := mskF (ix nE k) (msgAt v k)

/-! ## The coins, as field elements -/

section Coins

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)

end Coins

section Coins'

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)

end Coins'

/-! ## The replay -/

section Replay

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2) {nE : ℕ}
  (v : Fin nE → 𝔽)

end Replay

/-! ## The honest prover's clear values -/

theorem six_le {mPts : ℕ} (Tz : TabZK 𝔽 mPts) : 6 ≤ Tz.S.m := by have := Tz.hm; omega

/-- **`oodPoint` is the drawn point**: coordinate `j` of out-of-domain sample `i` is the coin `z ← sampleVec (m − 7)`
that `ZkLigerito.ligerito` draws, embedded by `fOf` (the refinement `Refine.vecOf` reads every executable point by). -/
theorem oodPoint_eq {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
    (i : Fin (Zk.PadLayout.new st.m st.c.kLog Tz.sch.k0).nOod0) (j : Fin (Tz.S.m - 7)) (hj : j.val < st.m - 7) :
    oodPoint Tz st c r i.val j = fOf (c.rest.oodPt r i ⟨j.val, hj⟩) := by
  simp only [oodPoint, dif_pos (And.intro i.isLt hj)]

section Honest

variable {mPts : ℕ} (Tz : TabZK 𝔽 mPts) (st : Stmt) (c : CoinsZK (RestOf Tz st)) (r : Fin 2)
  (zw : Witness Tz.S.m)

end Honest

end

end FlockSoundness.Discharge.Composed
