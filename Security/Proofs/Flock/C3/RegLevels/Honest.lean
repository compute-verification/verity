import Proofs.Flock.C3.RegLevels.Chain
import Proofs.Flock.C3.HoloValues.Reps
import Proofs.Flock.C3.Sparse.Session
import Proofs.Flock.C3.Opening.Algebra
import Proofs.Flock.C3.PubRows.Prefix
import Proofs.Flock.C3.Compose.Honest

/-!
# n5: R's levels at the honest family

`hRegLevels_honest` (stub): `HRegLevels` at `C3Family.honest`, from the public rows, the reps, their sparse phases and
R's opening (each on the prefix `decodeRun` takes, so R's Ligerito is decoded from what follows them), and R's
registered top being `xR`'s cap (`StackedTop`, which `stackedTop_of_innerCommits` gives at the verifier's hold of the
inner transcript where the binding round's top is `xR.cap`'s): the level-0 paths verify against the registered cap.
-/

namespace FlockC3.RegLevels

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

/-- **(n5) R's levels play to R's result at the honest family**, from the public rows, the earlier stages and R's
registered top. A stub. -/
theorem hRegLevels_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} (htop : ∃ lv, StackedTop lv st reg xR.cap)
    (hrows : FlockC3.PubRows.HPubRows 𝓗) (hreps : FlockC3.HoloValues.HReps 𝓗 dg N k Rw)
    (hsp : FlockC3.Sparse.HSparse 𝓗 dg N k Rw) (hop : FlockC3.Opening.HOpening 𝓗 dg N k Rw) :
    HRegLevels 𝓗 dg N k Rw := by
  sorry

end FlockC3.RegLevels
