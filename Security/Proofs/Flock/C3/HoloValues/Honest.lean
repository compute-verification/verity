import Proofs.Flock.C3.HoloValues.Reps
import Proofs.Flock.C3.PubRows.Prefix
import Proofs.Flock.C3.Compose.Honest
import Proofs.Flock.Soundness.Refine.HoloRepConverse
import Proofs.Flock.C3.HoloValues.Read

/-!
# n2: the holo values at the honest family

`hReps_honest` (stub): `HReps` at `C3Family.honest`, from the public rows (`HPubRows`, which tie each session's rows to
the inner transcript the verifier holds). Its route is C's holo forms, not `repProg`: each rep's V* sessions give
`zclc_of_residuals_dec` and `lincheck_of_residuals_hw` (through `decode_check_holo`, `decode_part_holo` and `ResZeroAtH`,
with `build_eval`, `residuals_eq_run`, `zerocheck_of_residuals`, `ringSwitch_of_residuals`, `ligerito_of_residuals` and
`hlg_of_frames` as C generalized them). `prog_of_zclc`, `prog_of_algebra`, `run_of_algebra` and `run_of_frames` conclude
that the plain `repProg` accepts, and a holo record's `h` round before r′ fails it, so a play of `holoRepC` goes through
the transcript-level holo rep program (n2a): `holoRepProg` (`Soundness/Refine/HoloRep.lean`), its refinement to
`holoRepC` (`holoRep_refines`, `HoloRefines.lean`) and its run from the stages' checks (`holoProg_of_checks`,
`HoloRepConverse.lean`).

As stated the stub is false: no hypothesis reads the decoder `dec`. At `dec := fun _ _ => []` every step of the inner
transcript is a coin, `repsOf` finds no play of `holoRepC` (its zerocheck receives messages first) and returns its
default, and `HReps` fails at every accepted outcome. It needs `dec` tied to V*'s layers: each rep's segment of the
tape is its proof's messages (`holoRepMsgs`) at its rounds' coins. That is the read fact `RepsRead` (`Read.lean`), from
which `hReps_honest_of_read` proves `HReps` at the honest family.
-/

namespace FlockC3.HoloValues

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

/-- **(n2) Each rep plays `holoRepC` at the honest family**, from the public rows. A stub. -/
theorem hReps_honest {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hrows : FlockC3.PubRows.HPubRows 𝓗) :
    HReps 𝓗 dg N k Rw := by
  sorry

end FlockC3.HoloValues
