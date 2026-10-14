import Proofs.Flock.C3.Compose.SparseEnds
import Proofs.Flock.C3.Compose.HoloSparseWords
import Proofs.Flock.C3.RegLevels.LgHeld

/-!
# The sparse phases' part ends at R's first round

`srcOK_holo_rs`' `he` is `PartEnds sparsePart … (roundAt L .stacked 0)` at `holoStreams`' list `L`, and
`partEnds_sparse` gives it at `sparseCoins`, the coins of the link points, both reps' shapes and both sparse phases'.

* `kindCoins_eq_length`: a shape's coins are its coin types.
* `sparseCoins_roundAt`: those coins are the history rounds before R's first (`roundAt_sparse`'s order, one past the
  reps' and the sparse phases' rounds): a rep's shape has `repFormsAt`'s coins (`kindCoinTys_holoRepShape`), which are
  the verifier's rep forms, its rounds' count (`repFormsAt_holoAt`, `heldBefore_rep_forms`), and a sparse phase's
  `sparseForms`', `nc` (`kindCoinTys_sparseKinds`, `heldBefore_sparse_size`).
* `partEnds_sparse_roundAt`: `partEnds_sparse` at `roundAt L .stacked 0`.
-/

namespace FlockC3.Compose

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness.Discharge.Exec FlockSoundness.Discharge.PlaysCoin
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockLevel3

theorem kindCoins_eq_length : ∀ σ : List Kind, kindCoins σ = (kindCoinTys σ).length
  | [] => rfl
  | .msg _ :: σ => kindCoins_eq_length σ
  | .coin _ :: σ => by
      show kindCoins σ + 1 = (kindCoinTys σ).length + 1
      rw [kindCoins_eq_length σ]

theorem sparseForms_length (sp : RecSparse.Shape) :
    (sparseForms (sparseOf sp)).length = FlockVBridge.Sparse.nc (sparseOf sp) := by
  rw [sparseForms, List.length_replicate]
  unfold FlockVBridge.Sparse.nc FlockVBridge.Sparse.nR
  omega

/-- **The sparse part's coins are the history rounds before R's first**, at `holoStreams`' list where the inner shape
is the inner statement's (`checkInnerStmt`) and the class's sparse shape the inner shape's (`hshape`). -/
theorem sparseCoins_roundAt {st : Flock.Stmt} {hp : HoloParams} {ish : RecVStar.Inner} {sp : RecSparse.Shape}
    {shR : StackedShape} {rs : Array SchedRound} {L : List (HoloStream × Array SchedRound)}
    (hc : ClassOk (shapeOf st)) (hi : RecVStar.checkInnerStmt ish st = .ok ()) (hsp : ish.sparse = some sp)
    (hsh : ish.stacked = some shR) (hrs : stackedRounds shR = some rs) (hshape : (holoAt st hp).cls.shape = sparseOf sp)
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some L) :
    sparseCoins (holoAt st hp) = roundAt L .stacked 0 := by
  obtain ⟨rep, -, hrep, -⟩ := holoStreams_eq hS
  have hL := Option.some.inj (hS.symm.trans (FlockC3.RegLevels.holoStreams_four hrep hsp hsh hrs))
  rw [hL]
  have hcl := FlockC3.VStarList.claims_of_checkInnerStmt hi
  have h0 : 0 < ish.claims := by omega
  have hr0 := heldBefore_rep_forms 0 h0 hrep
  have hr1 := heldBefore_rep_forms 1 h0 hrep
  have hs0 := heldBefore_sparse_size 0 sp
  have hs1 := heldBefore_sparse_size 1 sp
  have hst : heldBefore .stacked rs 0 = 0 := by simp [heldBefore]
  have hround : roundAt ([((.rep 0 : HoloStream), rep), (.rep 1, rep), (.sparse 0, sparseRounds sp),
      (.sparse 1, sparseRounds sp)] ++ [(.stacked, rs)]) .stacked 0 =
      1 + (heldBefore (.rep 0) rep rep.size + (heldBefore (.rep 1) rep rep.size +
        (heldBefore (.sparse 0) (sparseRounds sp) (sparseRounds sp).size +
          (heldBefore (.sparse 1) (sparseRounds sp) (sparseRounds sp).size + heldBefore .stacked rs 0)))) := by
    show 1 + roundIn _ .stacked 0 = _
    rw [List.cons_append, List.cons_append, List.cons_append, List.cons_append, List.nil_append,
      roundIn_cons_ne (s' := .rep 0) (s := .stacked) (by decide),
      roundIn_cons_ne (s' := .rep 1) (s := .stacked) (by decide),
      roundIn_cons_ne (s' := .sparse 0) (s := .stacked) (by decide),
      roundIn_cons_ne (s' := .sparse 1) (s := .stacked) (by decide), roundIn_cons_self]
  have hrepC : kindCoins (FlockC3.HoloValues.holoRepShape GF128 GF256 FlockSoundness.Refine.Dig ((holoAt st hp).cls.pubStatement
      (holoAt st hp).regs).m ((holoAt st hp).cls.pubStatement (holoAt st hp).regs).kLog
      (2 * ((holoAt st hp).cls.pubStatement (holoAt st hp).regs).regions.length) (holoAt st hp).sch) =
      (repForms ish.m ish.kLog ish.claims).length := by
    rw [kindCoins_eq_length, kindCoinTys_holoRepShape, List.length_map, ← repFormsAt_holoAt (hp := hp) hc hi]
    rfl
  have hspC : kindCoins (FlockC3.Sparse.sparseKinds GF128 (holoAt st hp).cls.shape) =
      FlockVBridge.Sparse.nc (sparseOf sp) := by
    rw [kindCoins_eq_length, kindCoinTys_sparseKinds, List.length_map, hshape, sparseForms_length]
  rw [hround, hst, hr0, hr1, hs0, hs1]
  unfold sparseCoins
  rw [hrepC, hspC]
  omega

open FlockSoundness FlockSoundness.Discharge
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **`srcOK_holo_rs`' `he`**: `partEnds_sparse` at R's first history round (`sparseCoins_roundAt`). -/
theorem partEnds_sparse_roundAt {ish : RecVStar.Inner} {sp : RecSparse.Shape} {shR : StackedShape}
    {rs : Array SchedRound} {Ls : List (HoloStream × Array SchedRound)} (hc : ClassOk (shapeOf st))
    (hi : RecVStar.checkInnerStmt ish st = .ok ()) (hsp : ish.sparse = some sp) (hsh : ish.stacked = some shR)
    (hrs : stackedRounds shR = some rs) (hshape : (holoAt st hp).cls.shape = sparseOf sp)
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some Ls)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'} {dg : ByteArray → ByteArray}
    {N k Rw : ℕ} {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (hr : FlockC3.Reps.SrcRepTie zs dg N k Rw src) (hs : FlockC3.Sparse.SrcSparseTie zs dg N k Rw src) :
    PartEnds sparsePart zs dg N k Rw src (roundAt Ls .stacked 0) := by
  rw [← sparseCoins_roundAt hc hi hsp hsh hrs hshape hS]
  exact partEnds_sparse hr hs

end FlockC3.Compose
