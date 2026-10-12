import Proofs.Flock.C3.Opening.Algebra
import Proofs.Flock.C3.HoloValues.Reps
import Proofs.Flock.C3.Sparse.Session
import Proofs.Flock.C3.PubRows.Prefix
import Proofs.Flock.C3.Compose.Honest

/-!
# n4: R's opening at the honest family

`hOpening_honest` (stub): `HOpening` at `C3Family.honest`, from the public rows, the reps and their sparse phases (each
on the prefix `decodeRun` takes, so R's opening is decoded from what follows them, on the two registered claims). The
`rec-opening` units are opening-only, O1–O6 at `m = mReg`; `7 ≤ mReg` is `seven_le_mReg`.
-/

namespace FlockC3.Opening

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}
  {ish : RecVStar.Inner} {ss : Array VStar.Stmt} {hss : RecVStar.statements ish = .ok ss}
  {bind : Fin (vstarCount ish) → VStar.Bindings} {Vs : Fin (vstarCount ish) → VStmt}
  {hob : VStarParses Flock.Tags.circuit ish bind Vs}
  {zr : Reg → L.Ω → List (Cm × (holoInnerAt T CT st hp).Coin) → Reg₂ → (j : Fin (vstarCount ish)) →
    (Vs j).OuterR Reg'}
  {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr}
  {own : Fin (vstarCount ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
  {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
  {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (holoInnerAt T CT st hp).Msg}
  {tr : Lean.Json} {record : String} {reg : Lean.Json}

local notation "𝓗" => C3Family.honest (xR := xR) ish hss bind hob zr hCust h own sel bc dec tr record reg

/-- **(n4) R's opening plays at the honest family**, from the public rows, the reps and their sparse phases. A stub. -/
theorem hOpening_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hrows : FlockC3.PubRows.HPubRows 𝓗)
    (hreps : FlockC3.HoloValues.HReps 𝓗 dg N k Rw) (hsp : FlockC3.Sparse.HSparse 𝓗 dg N k Rw) :
    HOpening 𝓗 dg N k Rw := by
  sorry

end FlockC3.Opening
