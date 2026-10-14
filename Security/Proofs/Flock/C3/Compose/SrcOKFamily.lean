import Proofs.Flock.C3.Compose.AtHolo
import Proofs.Flock.C3.Compose.LaterFamily

/-!
# `SrcOK` at the checked family, its stage coins from check 5

`srcOK_holo_family`: `srcOK_holo_rs` at one family (`T := holoTys st hp`, `CT := holoCTs st hp`), its `hcoins` fed by
`stageCoins_holo_family` at the checked family (`C3Family.checked`), whose `hleaf` is `hLeafC3_checked`. Its `htie`
and `hlt` come from `hmaps` (`MapsAt`'s opening tie, and `srcLevelTie_of_maps`), so R's opening's shape and words are
the map's (`osh`, `w`). One `wr` writes check 5's record throughout: the family's (`C3Family.checked`), check 5's
(`CoinsAccepted`) and the custody assumption's (`InnerRecordCustody`) are the same binder. The family's premises stay
premises, carried as named binders: `CoinsAccepted` (`hchk`), `SparseIs` (`hsi`), the sparse statement with check 5's
files tie (`hsS`, `hfiles`), and the run's points as the history's first coin (`hpts`, `PointsFromHistory`).
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (RepsMsgsRead SrcRepTie repsOfVals UnitVals ProofOfVals SetupOfVals FramesOfVals RepRoundsHeld
  RepsDrawn)
open Flock.Guarantees (holoCTs holoRounds)

variable {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt (Flock.Guarantees.holoTys st hp) (holoCTs st hp) st hp

/-- **`SrcOK` at the checked family, its stage coins from check 5**: `srcOK_holo_rs` with `hcoins` from
`stageCoins_holo_family`, `htie` and `hlt` from `hmaps`, at one `wr` (check 5's record writer is the custody
assumption's). The family's premises are carried: `hchk`, `hsi`, `hsS` with `hfiles`, and `hpts`. -/
theorem srcOK_holo_family {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).Outer Reg'}
    (hzs : zs = (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp) (xR := xR) ish hss
      bind hob zr hCust h coins rows own sel bc dec wr reg).zs)
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) (hS : 0 < vstarCount ish)
    {dg : ByteArray → ByteArray} {N k Rw c₀ : ℕ} {stmt : Fin (vstarCount ish) → VStar.Stmt} {sh : Flock.Shape}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → UnitVals}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    {tbl unit : String} {bcM : Cm → String → Nat → List UInt8} {s : Fin 2 → String} {ix : Fin 2 → ℕ → ℕ}
    {rr rs : Array RecVStar.SchedRound} {sp : RecSparse.Shape} {shR : RecVStar.StackedShape}
    {M : FlockC3.Reps.StackedMaps stmt Vs} {osh : Flock.Shape}
    {c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → RecCoins.C3}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array F128}
    {kap : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
      Array (Array (Array F128))}
    {e0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ByteArray}
    {sb : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    {e : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ℕ → ByteArray}
    {cap₀ : Cap Dig}
    (hchk : FlockC3.Sparse.CoinsAccepted (C3Family.checked (T := Flock.Guarantees.holoTys st hp)
      (CT := holoCTs st hp) (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) rows wr reg
      coins)
    (hsi : FlockC3.VStarList.SparseIs ish hp) {sI : Fin (vstarCount ish)} {shS : Flock.RecSparse.Shape} {hR : ℕ}
    (hsS : ss[(sI : ℕ)]? = some (.sparse shS hR))
    (hfiles : ∀ r src, coins r = some src → ∃ f, src = .files f ∧ f.record = r)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bcM))
    (hheldR : RepRoundsHeld zs dg N k Rw sh rep wr held s ix)
    (hdr : RepsDrawn zs dg N k Rw sh rep vals (holoSelOne ish tbl) s ix)
    (hrt : SrcRepTie zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hst : FlockC3.Sparse.SrcSparseTie zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)))
    (hot : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => ∃ sh w ops,
      FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hosh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some osh)
    (hmaps : MapsAt zs dg N k Rw M osh unit (repsOfVals stp sh rep vals) c3 w kap e0 sb e cap₀)
    (hc3 : C3Held wr held ish c3) (hrep : holoRounds ish.m ish.kLog ish.claims = some rr)
    (hsp : ish.sparse = some sp) (hsh : ish.stacked = some shR) (hcl : shR.claims = RecVStar.STACKED_CLAIMS)
    (hrs : RecVStar.stackedRounds shR = some rs) (hc : 0 < ish.claims)
    (hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims)
    (hshape : (holoAt st hp).cls.shape = sparseOf sp) (htblS : tbl ≠ s!"{RecVStar.SPARSE}/")
    (htbl : tbl ≠ s!"{RecVStar.STACKED}/")
    (hpts : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      FlockC3.HPort.PointsFromHistory ish tbl bcM t u)
    (hheld : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => HeldOk wr held t)
    (hcnt : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => SparseCountedAt wr held sp t)
    (hsst : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      SparseStagesAt wr held ish unit sp t u)
    (hx : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => StackedHeldAt wr held rs t)
    (hts : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => RepStagesCounted rr u)
    (hrp : RepsMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hsm : SparseMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (he : PartEnds sparsePart zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) c₀)
    (hall : OpenAll zs dg N k Rw osh w) (hk : RsKeys zs dg N k Rw osh w)
    (hr : RoundReads zs dg N k Rw osh w own sel c₀)
    (hlg : LigMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hos : ∀ s < 3, OpensReadAt zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) s) :
    SrcOK zs dg N k Rw (tiedSrc (normSrc (repsOfVals stp sh rep vals))) own sel := by
  subst hzs
  exact srcOK_holo_rs rfl hF hS
    (stageCoins_holo_family
      (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp) (xR := xR) ish hss bind hob zr
        hCust h coins rows own sel bc dec wr reg)
      hchk hsi hsS hfiles (hLeafC3_checked hS dg N k Rw) halg hacc hpr hsv hf hcus hheldR hdr hrt hst hot hell
      hosh hmaps hc3 hrep hsp hsh hcl hrs hc hA hshape htblS htbl hpts hheld hcnt hsst hx hts)
    hrp hsm he (fun R ω t R₂ τ o hAcc => (hmaps R ω t R₂ τ o hAcc).2.2.1) hall hk hr hlg hos hrt hst hot
    (FlockC3.RegLevels.srcLevelTie_of_maps M hell hosh unit (repsOfVals stp sh rep vals) c3 w kap e0 sb e hmaps)

end

end FlockC3.Compose

#print axioms FlockC3.Compose.srcOK_holo_family
#print axioms FlockC3.Compose.srcOK_holo_rs
#print axioms FlockC3.Compose.stageCoins_holo_family
