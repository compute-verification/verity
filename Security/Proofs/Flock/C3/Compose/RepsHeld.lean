import Proofs.Flock.C3.Compose.AtHoloHeld
import Proofs.Flock.C3.Compose.HeldHyps
import Proofs.Flock.C3.Compose.RepsOff
import Proofs.Flock.C3.Reps.HoloVals
import Proofs.Flock.C3.RegLevels.Reached

/-!
# `hbr_honest_reg`'s `hrread` at the checked family, the sparse tie from V*'s sessions

`repsReadOff_held`: `RepsReadOff` (`hbr_honest_reg`'s `hrread`) at the checked family at the fixed decoder `c3DecOf`
over the source `srcOK_holo_held` makes at `stp := sparseOf stp sw sc` (each rep's sparse stage the steps of the
session's words `sw` at its coins `sc`), its `SrcOK` with neither tie as a hypothesis:

* `hst`: `srcSparseTie_of_copies` at `repsOfVals stp sh rep vals`, which is `srcOK_holo_held`'s source, since
  `repsOfVals` writes the reps' steps and openings and `sparseOf` the sparse stages (`repsOfVals_sparseOf`, `rfl`). Its
  `h` conjunct is `srcHTie_checked`'s, from the named `HCols` and `HAlgRows`, and `hss'` is `sparseStagesAt_sparseOf`'s
  at the same sessions;
* `RepUnitOfAlg` from its four parts (`repUnitOfAlg_of_parts`): `srcOK_holo_held`'s `hpr`, `hsv` and `hf`, and
  `StepsOfVals` from them and the values' cap (`hcap`: `stepsOfVals_repsOfVals`, `stepsOfVals_norm_repsOfVals`), so
  `hrt` is `srcRepTie_of_alg`'s;
* G's items 1–4 off `Bad` are `halg` and `hacc`, which `srcOK_holo_held` and `srcSparseTie_of_copies` take at every
  accepted outer proof.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Reps (RepsMsgsRead repsOfVals UnitVals ProofOfVals SetupOfVals FramesOfVals RepRoundsHeld RepsDrawn
  RepUnitOfAlg AtRep)
open Flock.Guarantees (holoCTs holoRounds HoloStream)

variable {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt (Flock.Guarantees.holoTys st hp) (holoCTs st hp) st hp

/-- **`repsOfVals` and `sparseOf` commute**: the one writes the reps' steps and openings, the other the sparse
stages. -/
theorem repsOfVals_sparseOf {S : ℕ} {Vs : Fin S → VStmt}
    (stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp))
    (sh : Flock.Shape) (rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → RepData)
    (vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → UnitVals)
    (sw : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → Array F128)
    (sc : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Fin 2 → FlockVBridge.Sparse.SCoins) :
    repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals =
      FlockC3.Sparse.sparseOf (repsOfVals stp sh rep vals) sw sc :=
  rfl

/-- **`hbr_honest_reg`'s `hrread` at the checked family**, at the fixed decoder over `srcOK_holo_held`'s source at
`stp := sparseOf stp sw sc`: `repsReadOff_of_alg` at `srcOK_holo_held`'s `SrcOK`, whose sparse tie is
`srcSparseTie_of_copies` at V*'s sessions and copies (`hsess`, with `hpub` its pub conjunct's read), its `h` conjunct
from `HCols` and `HAlgRows` (`srcHTie_checked`), and whose reps' tie and `RepUnitOfAlg` are the four parts' (`hpr`,
`hsv`, `hf`, and `StepsOfVals` from them and the values' cap `hcap`). -/
theorem repsReadOff_held {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
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
    {sw : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → Array F128}
    {sc : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 →
      FlockVBridge.Sparse.SCoins}
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
    {cap₀ : Cap Dig} {c₀ : ℕ} {rng : Flock.Range}
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hcap : AtRep zs dg N k Rw sh rep fun R ω t Lb i => capDig (vals R ω t Lb i).p.cap = (stp R ω t Lb).cap₀)
    (hcus : FlockSoundness.Assumptions.InnerRecordCustody wr held sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bcm))
    (hheldR : RepRoundsHeld zs dg N k Rw sh rep wr held s ix)
    (hdr : RepsDrawn zs dg N k Rw sh rep vals (holoSelOne ish tbl) s ix)
    (hrp : RepsMsgsRead zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) own sel)
    (hsp' : SparseMsgsRead zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) own sel)
    (he : PartEnds sparsePart zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) c₀)
    (hall : OpenAll zs dg N k Rw osh w) (hk : RsKeys zs dg N k Rw osh w) (hr : RoundReads zs dg N k Rw osh w own sel c₀)
    (hlg : LigMsgsRead zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) own sel)
    (hos : ∀ s < 3, OpensReadAt zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) s)
    (hot : AtSrc zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) fun _ u =>
      ∃ sh w ops, FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hosh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some osh)
    (hmaps : MapsAt zs dg N k Rw M osh unit (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals) c3 w kap e0 sb
      e cap₀)
    (hc3 : C3Held wr held ish c3) (hrep : holoRounds ish.m ish.kLog ish.claims = some rr)
    (hsp : ish.sparse = some sp) (hsh : ish.stacked = some shR) (hcl : shR.claims = RecVStar.STACKED_CLAIMS)
    (hrs : RecVStar.stackedRounds shR = some rs) (hc : 0 < ish.claims)
    (hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims)
    (hshape : (holoAt st hp).cls.shape = sparseOf sp)
    (htbl : ∀ i : Fin 2, (((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.SPARSE}/" ||
      ((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.STACKED}/") = false)
    (hrec : AtSrc zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) fun t _ =>
      ∃ spec sess, Flock.Record.decode spec (wr t).2 = .ok sess)
    (hpts : AtSrc zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) fun t u =>
      FlockC3.HPort.PointsFromHistory ish tbl bcm t u)
    (hheld : AtSrc zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) fun t _ =>
      HeldOk wr held t)
    (hcnt : AtSrc zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) fun t _ =>
      SparseCountedAt wr held sp t)
    (hx : AtSrc zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) fun t _ =>
      StackedHeldAt wr held rs t)
    (hts : AtSrc zs dg N k Rw (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) fun _ u =>
      RepStagesCounted rr u)
    (hu : UnitsOf st hp) (hrng : unitRangeOf st = .ok rng)
    (hpub : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      FlockC3.Sparse.SparsePubRead (holoAt st hp) rng (rep R ω t Lb i) (sc R ω t Lb i))
    (hsess : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      FlockC3.Sparse.SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) unit i (sw R ω t (layers zs dg N k Rw R ω t R₂ τ) i)
          (sc R ω t (layers zs dg N k Rw R ω t R₂ τ) i) ∧
        ∃ (s : Fin (vstarCount ish)) (un : Fin (Vs s).nv) (j : ℕ) (hj : j < (Vs s).c.upv), un.val = i.val ∧
          FlockC3.Sparse.SparseCopyAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
            (sw R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (sc R ω t (layers zs dg N k Rw R ω t R₂ τ) i) s un j hj)
    (hcols : ∀ s : Fin (vstarCount ish), Flock.RecSparse.check (Vs s).c = .ok () →
      Flock.RecSparse.params (Vs s).c.unit.name = some (FlockC3.Sparse.unitShape (holoAt st hp)) →
      FlockC3.HPort.HCols (Vs s) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))))
    (hHA : FlockC3.HPort.HAlgRows (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp)
      (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw (repsOfVals stp sh rep vals) sw)
    (Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop) :
    ∀ h' : Typed zr, RepsReadOff (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp)
      (xR := xR) ish hss bind hob zr hCust h' coins rows own sel bc
      (c3DecOf zs dg N k Rw (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals))) own sel)
      wr reg) Bad dg N k Rw := by
  have hU : RepUnitOfAlg zs dg N k Rw (tiedSrc (repsOfVals stp sh rep vals)) sh rep :=
    FlockC3.Reps.repUnitOfAlg_of_parts (src := tiedSrc (repsOfVals stp sh rep vals)) hpr hsv hf
      (FlockC3.Reps.stepsOfVals_repsOfVals (stp := stp) hpr hsv hf hcap)
  have hH : FlockC3.HPort.SrcHTie zs dg N k Rw (repsOfVals stp sh rep vals) sw := by
    subst hzs
    exact FlockC3.HPort.srcHTie_checked hcols hHA
  have hst : FlockC3.Sparse.SrcSparseTie zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf stp sw sc) sh rep vals)) := by
    rw [repsOfVals_sparseOf]
    exact FlockC3.Sparse.srcSparseTie_of_copies hu hrng halg hacc hU hpub c3 unit sw hsess hH
  have hs := srcOK_holo_held hzs hF hS halg hacc hpr hsv hf hcus hheldR hdr hrp hsp' he hall hk hr hlg hos
    (FlockC3.Reps.srcRepTie_of_alg halg hacc (FlockC3.Reps.repUnitOfAlg_of_parts hpr hsv hf
      (FlockC3.Reps.stepsOfVals_repsOfVals hpr hsv hf hcap)))
    hst hot hell hosh hmaps hc3 hrep hsp hsh hcl hrs hc hA hshape htbl hrec hpts hheld hcnt
    (sparseStagesAt_sparseOf hc3 hshape fun R ω t R₂ τ o hA' i => (hsess R ω t R₂ τ o hA' i).1) hx hts
  intro _
  subst hzs
  exact repsReadOff_of_alg (Bad := Bad) hs (fun R ω t R₂ τ o hA' _ => halg R ω t R₂ τ o hA')
    (fun R ω t R₂ τ o hA' _ => hacc R ω t R₂ τ o hA')
    (FlockC3.Reps.repUnitOfAlg_of_parts hpr hsv hf (FlockC3.Reps.stepsOfVals_norm_repsOfVals hpr hsv hf hcap))

end

end FlockC3.Compose
