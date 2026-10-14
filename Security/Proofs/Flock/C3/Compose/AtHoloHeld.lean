import Proofs.Flock.C3.Compose.AtHolo
import Proofs.Flock.C3.Compose.LaterHolo

/-!
# `SrcOK` at the checked family, its stage coins from the verifier's hold

`srcOK_holo_held`: `srcOK_holo_rs` at `holoCTs` and the reps' source at the values (`repsOfVals`), with `hcoins` from
`stageCoins_holo_held` (`hleaf` from `hLeafC3_checked` at the checked sessions), and `hlt` and R's opening's tie
`htie` from R's run in V*'s sessions (`MapsAt`: `srcLevelTie_of_maps`, its `OpenTieL` conjunct). The table's prefix is
neither the sparse streams' nor R's where the guarantee's `_htbl` holds (`htbl_of`, `htblR_of`).
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (RepsMsgsRead SrcRepTie repsOfVals UnitVals ProofOfVals SetupOfVals FramesOfVals RepRoundsHeld
  RepsDrawn CoinsEncoded)
open Flock.Guarantees (holoCTs holoKinds holoRounds HoloStream)

variable {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt (Flock.Guarantees.holoTys st hp) (holoCTs st hp) st hp

/-- **`SrcOK` at the checked family, its stages' coins from the verifier's hold**: `srcOK_holo_rs` at `holoCTs` and
`repsOfVals`, `hcoins` from `stageCoins_holo_held`, `hlt` and `htie` from `MapsAt`, and the table's prefix facts from
the guarantee's `_htbl` (`htbl`). -/
theorem srcOK_holo_held {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
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
    (hzs : zs = (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp) (xR := xR) ish hss bind
      hob zr hCust h coins rows own sel bc dec wr reg).zs)
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) (hS : 0 < vstarCount ish)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {sh : Flock.Shape}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → UnitVals}
    {held : Lean.Json × String → Except String RecVStar.InnerCommits} {sparse : Option RecSparse.Shape}
    {tbl unit : String} {bcm : Cm → String → Nat → List UInt8} {s : Fin 2 → String} {ix : Fin 2 → ℕ → ℕ}
    {rr rs : Array RecVStar.SchedRound} {sp : RecSparse.Shape} {shR : RecVStar.StackedShape}
    {stmt : Fin (vstarCount ish) → VStar.Stmt} {M : FlockC3.Reps.StackedMaps stmt Vs} {osh : Flock.Shape}
    {c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → RecCoins.C3}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array F128}
    {kap : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array (Array (Array F128))}
    {e0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ByteArray}
    {sb : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    {e : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ℕ → ByteArray}
    {cap₀ : Cap Dig} {c₀ : ℕ}
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bcm))
    (hheldR : RepRoundsHeld zs dg N k Rw sh rep wr held s ix)
    (henc : CoinsEncoded (CT := holoCTs st hp) (Flock.Guarantees.holoMap st hp Cm ish tbl bcm).selC
      (Flock.Guarantees.holoMap st hp Cm ish tbl bcm).coin)
    (hdr : RepsDrawn zs dg N k Rw sh rep vals (Flock.Guarantees.holoMap st hp Cm ish tbl bcm).selC s ix)
    (hrp : RepsMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hsp' : SparseMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (he : PartEnds sparsePart zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) c₀)
    (hall : OpenAll zs dg N k Rw osh w) (hk : RsKeys zs dg N k Rw osh w) (hr : RoundReads zs dg N k Rw osh w own sel c₀)
    (hlg : LigMsgsRead zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) own sel)
    (hos : ∀ s < 3, OpensReadAt zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) s)
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
    (hshape : (holoAt st hp).cls.shape = sparseOf sp)
    (htbl : ∀ i : Fin 2, (((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.SPARSE}/" ||
      ((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.STACKED}/") = false)
    (hpts : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u => RecordPoints wr t u)
    (hlen : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ =>
      1 + (Exec.kindCoinTys (holoKinds (holoAt st hp))).length + 1 ≤ t.length)
    (hheld : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => HeldOk wr held t)
    (hcnt : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => SparseCountedAt wr held sp t)
    (hss' : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t u =>
      SparseStagesAt wr held ish unit sp t u)
    (hx : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun t _ => StackedHeldAt wr held rs t)
    (hts : AtSrc zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) fun _ u => RepStagesCounted rr u) :
    SrcOK zs dg N k Rw (tiedSrc (normSrc (repsOfVals stp sh rep vals))) own sel := by
  have hleaf : HLeafC3 zs dg N k Rw := by
    subst hzs
    exact hLeafC3_checked hS dg N k Rw
  have hlt := FlockC3.RegLevels.srcLevelTie_of_maps M hell hosh unit (repsOfVals stp sh rep vals) c3 w kap e0 sb e hmaps
  have hcoins := stageCoins_holo_held hleaf halg hacc hpr hsv hf hcus hheldR henc hdr hrt hst hot hell hosh hmaps hc3
    hrep hsp hsh hcl hrs hc hA hshape (FlockC3.Sparse.htbl_of htbl) (FlockC3.Sparse.htblR_of htbl) hpts hlen hheld hcnt
    hss' hx hts
  exact srcOK_holo_rs (sh := osh) hzs hF hS hcoins hrp hsp' he (fun R ω t R₂ τ o hA' => (hmaps R ω t R₂ τ o hA').2.2.1)
    hall hk hr hlg hos hrt hst hot hlt

end

end FlockC3.Compose
