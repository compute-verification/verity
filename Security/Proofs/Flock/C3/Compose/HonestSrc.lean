import Proofs.Flock.C3.Compose.Discharge
import Proofs.Flock.C3.Compose.RepsHeld
import Proofs.Flock.C3.Compose.SrcOKFamily
import Proofs.Flock.C3.Compose.Custody
import Proofs.Flock.C3.Compose.SparseEndsAt
import Proofs.Flock.C3.Reps.RoundReads
import Proofs.Flock.C3.Sparse.HeldStages
import Proofs.Flock.C3.Sparse.SessionReg
import Proofs.Flock.C3.RegLevels.SrcMaps
import Proofs.Flock.C3.RegLevels.LgHeld
import Proofs.Flock.C3.Compose.OffZs
import Proofs.Flock.C3.Sparse.PubCoins

/-!
# `hbr_honest_reg` at the honest source

`hbr_honest_src`: the guarantee's `_hbr` in its form (`hbr_honest_reg`) at the checked family and the fixed decoder
`c3DecOf` over the honest source, `tiedSrc (normSrc (repsOfVals (sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep
vals))`: the run's points the history's first coin (`histPts`), each rep's sparse stage the steps of the session's
words `sw` at its coins `sc` (`sparseOf`), the reps' steps and openings their values' (`repsOfVals`). `sc` is `scOf`: the
points `verifierRows` gives at the rep's own values, the layout factors there, and check 5's phase coins (`csOf`).

G's items 1–4 (`halg`, `hacc`) are needed only off a bad event `Bad` inside `hbr_honest_reg`'s (`hBad`; at
`Bad := ValueColl …`, where the value ties hold): the decoder, `SrcOK` and every read fact are built at V*'s sessions
refusing `Bad` (`offZs zs Bad`), whose accepted proofs are the checked sessions' off `Bad`, and each read fact moves
back off `Bad` (`atAcceptedOff_offZs`). What the verifier's checks give is discharged here:

* `hS` from the sparse statement's index, `hchk` (`coinsAccepted_checked`), `hrec` (`recordDecodes_family`), `hpts`
  (`pointsFromHistory_histPts`), `htblS` and `htbl` from the guarantee's `_htbl` (`htbl_of`, `htblR_of`), and `hcus`
  the guarantee's `_hIC` at `holoMap`;
* `hleaf` (`hLeafC3_checked`), `hheld` (`heldOk_atSrc` at the whole history), `hcnt` (`sparseCountedAt_of_check` at
  the covers check every accepted proof of the checked sessions passes, `covers_atSrc`), `hc3` at the `C3` of the hold
  (`c3Held_heldC3`), `hheldR` (`repRoundsHeld_of_rounds`) and `hdr` (`repsDrawn_held` at `covers_checked`);
* the reps' tie from the four parts (`srcRepTie_of_alg`), the sparse tie from V*'s sessions and copies
  (`srcSparseTie_of_copies`, its `h` conjunct `srcHTie_checked`), and the stage coins (`stageCoins_holo_held`);
* from the inner statement's check (`checkInnerStmt`, the class's `ClassOk`, claims `2 · regions + 2`): `hA`
  (`repFormsAt_holoAt`), `hc`, and at `c₀` R's first history round, `he` (`partEnds_sparse_roundAt`) and R's opening's
  committed rounds (`openMsgsRead_of_layer` at `rsRead_checked`, `roundReads_checked` at the map's `sel`,
  `holoSel_opening_at`); `hshape` from `SparseIs` (`shape_holoAt`);
* the pub read at `scOf` (`srcSparsePub_scOf`), the layout factors passing where check 5's range verdict says
  (`SparseRangeOk`, which also gives `ℓ`'s range, `ellOk_of_sparseRange`);
* `hrread`, `hsread`, `horead`, `hlread` and `hfr` at the fixed decoder (`repsRead_c3DecOf`, `sparseRead_c3DecOf`,
  `openingRead_c3DecOf`, `regLevelsRead_of_tie`, `hFrame_c3DecOf`), each tie moved to the normalized source
  (`stepsOfVals_norm_repsOfVals`, `openingReadAt_normOs`, `levelTie_normOs`; the sparse plays read no openings); `hown`
  at `own := ownAt ss` (`ownIn_checked`), `hheld` at the map's `sel` (`heldIs_of_custody`), and `_htop` from the
  guarantee's `_hstk`.

What stays a hypothesis no check gives at this source: G's items 1–4 off `Bad` (`halg`, `hacc`), the values' parts
(`hpr`, `hsv`, `hf`, `hvcap`) at every accepted proof, the reps' held rounds (`hrd`), V*'s sessions and copies
(`hsess`), R's opening at the maps (`hmaps`, `hot`), the other parts' read facts (`hrp`, `hsm`,
`hlg`, `hos`, `hall`, `hk`, `hx`, `hts`), the named `HCols` and `HAlgRows`, and the commit strings' custody (`hbc`).
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge.Family (skOf cmtOf wmsgOf msgOf)
open FlockC3.Reps (RepsMsgsRead repsOfVals UnitVals ProofOfVals SetupOfVals FramesOfVals RepRoundsHeld RepsDrawn
  RepUnitOfAlg AtRep)
open Flock.Guarantees (holoCTs holoRounds HoloStream)

variable {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt (Flock.Guarantees.holoTys st hp) (holoCTs st hp) st hp

/-- **The covers check at every accepted outer proof of the checked sessions** (`covers_checked` at any source, not
only at a rep meeting E's preconditions): where it fails at the history's hold, no session keeps its own list. -/
theorem covers_atSrc {S : ℕ} {Vs : Fin S → VStmt} {ish : RecVStar.Inner}
    {cs : List (Cm × (ℐ).Coin) → Option RecVStar.InnerCommits}
    {coins : List (Cm × (ℐ).Coin) → Option RecCoins.Source} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {hC : Flock.Premises.RecordCustody zr} {h : Typed zr} {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → C3Run (holoAt st hp)} (hS : 0 < S) :
    AtSrc (Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h) dg N k Rw src fun t _ =>
      ∀ c, cs t = some c → (RecVStar.checkCovers ish c).isOk = true := by
  intro R ω t R₂ τ o hA c hc
  by_contra hno
  exact not_acceptsZKR_keepOwn (z := Flock.Guarantees.custodied zr hC h R ω t R₂ ⟨0, hS⟩)
    (P := fun J own tr => (ℐ).r ≤ t.length ∧
      Flock.Guarantees.ChecksC3 ish (cs t) (coins t) (⟨0, hS⟩ : Fin S) (bind ⟨0, hS⟩) rows J own tr)
    (fun _ _ _ hP => by
      obtain ⟨-, _, _, -, -, -, -, -, hcov⟩ := hP
      exact hno (hcov c hc)) _ (hA.1 ⟨0, hS⟩).1

/-- **The inner shape's reps' schedule**, holo or plain, from candidate 3's (`holoRounds`, which is `holoRepRounds`):
at a plain shape it is `repRounds`', defined wherever `holoRepRounds` is (`repRounds_zp`). -/
theorem sched_of_holoRounds {ish : RecVStar.Inner} {rr : Array RecVStar.SchedRound}
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rr) :
    ∃ rep, (if ish.holo then RecVStar.holoRepRounds else RecVStar.repRounds) ish.m ish.kLog ish.claims = some rep := by
  have h := (Flock.Guarantees.HoloRepCoins.holoRepRounds_eq _ _ _).trans hrep
  cases ish.holo with
  | true => exact ⟨rr, h⟩
  | false =>
    obtain ⟨out, hout, -⟩ := FlockC3.Reps.repRounds_zp h
    exact ⟨out, hout⟩

-- Moving the five read facts from `offZs zs Bad`'s family to the checked one exceeds the default at `whnf`
-- (r20261016-015424-7b88).
set_option maxHeartbeats 1600000 in
/-- **(step 8) The guarantee's `_hbr` at the honest source**: `hbr_honest_reg` at the checked family and the fixed
decoder over the honest source, built at V*'s sessions refusing `Bad` (`offZs zs Bad`), where `Bad` holds only inside
`hbr_honest_reg`'s bad event (`hBad`): G's items 1–4 are needed only off `Bad` (`halg`, `hacc`), `SrcOK` and each read
fact hold at `offZs zs Bad` (`srcOK_holo_of_leaf`), and each read fact moves back to the checked family off `Bad`
(`atAcceptedOff_offZs`). Every hypothesis the verifier's checks give is discharged (module doc). `own` is the
verifier's message ports (`ownAt ss`), `sel` the map's, and `dec` the fixed decoder at the normalized source. -/
theorem hbr_honest_src {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
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
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {cap : Cap Dig} {salt : ℕ → ByteArray} {bcs' : Array ByteArray}
    {σ : Strategy (recGame L Reg (ℐ) Cm fun R ω t => outerVR (zr R ω t) dg N)}
    {stmt : Fin (vstarCount ish) → VStar.Stmt} {sh : Flock.Shape}
    {stp0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → C3Run (holoAt st hp)}
    {rep : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → RepData}
    {vals : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → UnitVals}
    {sw : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 → Array F128}
    {sc : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Fin 2 →
      FlockVBridge.Sparse.SCoins}
    {tbl unit : String} {bcM : Cm → String → Nat → List UInt8}
    {rr rs : Array RecVStar.SchedRound} {sp : RecSparse.Shape} {shR : RecVStar.StackedShape}
    {M : FlockC3.Reps.StackedMaps stmt Vs} {osh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array F128}
    {kap : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) →
      Array (Array (Array F128))}
    {e0 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ByteArray}
    {sb : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ℕ → ℕ → ByteArray}
    {e : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → ℕ → ℕ → ByteArray}
    {rng : Flock.Range} {sI : Fin (vstarCount ish)} {shS : Flock.RecSparse.Shape} {hR : ℕ}
    {Bad : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Prop}
    (hzs : zs = Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind
      rows zr hCust h)
    (hown : own = fun j : Fin (vstarCount ish) => FlockC3.PubRows.ownAt ss j)
    (hsel : sel = (Flock.Guarantees.holoMap st hp Cm ish tbl bcM).sel)
    (hdec : dec = c3DecOf (offZs zs Bad) dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals))) own sel)
    (hBad : ∀ R ω t R₂, Bad R ω t R₂ →
      ValueColl (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind
          rows zr hCust h)
          (valueSrcAll (fun t => coins (wr t).2) fun t =>
            Flock.Guarantees.c3Of ish (Flock.Guarantees.innerHeld wr st reg t)) R ω t R₂ ∨
        LeafColl (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind
          rows zr hCust h) R ω t R₂ ∨
        SaltColl (holoAt st hp).rsch.sch.level₀ cap salt (Flock.Guarantees.checkedC3 ish
          (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr hCust h) R ω t R₂ ∨
        TopColl wr st reg bcs' t)
    (hcap : xR.cap = cap) (hstk : StackedTopAt (holoAt st hp).rsch.sch.level₀ st reg cap salt bcs')
    (hIC : Flock.Premises.InnerRecordCustody wr (fun x => innerCommitsAt x.1 x.2 st reg) ish.sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bcM))
    (htbl : ∀ i : Fin 2, (((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.SPARSE}/" ||
      ((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.STACKED}/") = false)
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (hsi : FlockC3.VStarList.SparseIs ish hp) (hsS : ss[(sI : ℕ)]? = some (.sparse shS hR))
    (hfiles : ∀ r src, coins r = some src → ∃ f, src = .files f ∧ f.record = r)
    (hrok : SparseRangeOk ish st)
    (hosh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some osh)
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rr)
    (hsp : ish.sparse = some sp) (hsh : ish.stacked = some shR) (hcl : shR.claims = RecVStar.STACKED_CLAIMS)
    (hrs : RecVStar.stackedRounds shR = some rs) (hcO : ClassOk (shapeOf st))
    (hi : RecVStar.checkInnerStmt ish st = .ok ()) (hcl2 : ish.claims = 2 * ish.regions + 2)
    (halg : AlgebraDecodesOff zs dg N k Rw Bad sh rep) (hacc : AccChainsOff zs dg N k Rw Bad sh rep)
    (hpr : ProofOfVals zs dg N k Rw sh rep vals)
    (hsv : SetupOfVals zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) sh rep vals)
    (hf : FramesOfVals zs dg N k Rw sh rep vals)
    (hvcap : AtRep zs dg N k Rw sh rep fun R ω t Lb i => capDig (vals R ω t Lb i).p.cap = (stp0 R ω t Lb).cap₀)
    (hrd : ∀ R ω t Lb i, (rep R ω t Lb i).rounds =
      FlockC3.Reps.heldRounds (innerCommitsAt (wr t).1 (wr t).2 st reg) ((HoloStream.rep i).name tbl))
    (hrp : RepsMsgsRead zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) own sel)
    (hsm : SparseMsgsRead zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) own sel)
    (hall : OpenAll zs dg N k Rw osh w) (hk : RsKeys zs dg N k Rw osh w)
    (hlg : LigMsgsRead zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) own sel)
    (hos : ∀ s < 3, OpensReadAt zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) s)
    (hot : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun _ u =>
        ∃ sh w ops, FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u)
    (hmaps : MapsAt zs dg N k Rw M osh unit
      (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)
      (fun _ _ t _ => heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) w kap e0 sb e xR.cap)
    (hx : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        StackedHeldAt wr (fun x => innerCommitsAt x.1 x.2 st reg) rs t)
    (hts : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun _ u =>
        RepStagesCounted rr u)
    (hu : UnitsOf st hp) (hrng : unitRangeOf st = .ok rng)
    (hsc : sc = fun R ω t Lb i => FlockC3.Sparse.scOf (holoAt st hp) rng sh
      (FlockC3.Sparse.csOf (heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) unit i.val) (rep R ω t Lb i))
    (hsess : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
      FlockC3.Sparse.SparseSessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
          (heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) unit i
          (sw R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (sc R ω t (layers zs dg N k Rw R ω t R₂ τ) i) ∧
        ∃ (s : Fin (vstarCount ish)) (un : Fin (Vs s).nv) (j : ℕ) (hj : j < (Vs s).c.upv), un.val = i.val ∧
          FlockC3.Sparse.SparseCopyAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) (holoAt st hp)
            (sw R ω t (layers zs dg N k Rw R ω t R₂ τ) i) (sc R ω t (layers zs dg N k Rw R ω t R₂ τ) i) s un j hj)
    (hcols : ∀ s : Fin (vstarCount ish), Flock.RecSparse.check (Vs s).c = .ok () →
      Flock.RecSparse.params (Vs s).c.unit.name = some (FlockC3.Sparse.unitShape (holoAt st hp)) →
      FlockC3.HPort.HCols (Vs s) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))))
    (hHA : FlockC3.HPort.HAlgRows (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp)
      (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) dg N k Rw
      (repsOfVals (histPts ish tbl bcM stp0) sh rep vals) sw)
    (hbc : ∀ j c (u : Fin (Vs j).nv) (q : Fin (Vs j).c.ports.size), ∀ p ∈ own j,
      p.name = (Vs j).c.ports[q].name → ∀ x, p.reads[u.val]? = some (some x) →
        bc j c (.inl (u, q)) = (Flock.Guarantees.holoMap st hp Cm ish tbl bcM).bc c p.value x) :
    Flock.Premises.VBridgeCOff
      (fun R ω t R₂ =>
        ValueColl (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind
          rows zr hCust h)
          (valueSrcAll (fun t => coins (wr t).2) fun t =>
            Flock.Guarantees.c3Of ish (Flock.Guarantees.innerHeld wr st reg t)) R ω t R₂ ∨
        LeafColl (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind
          rows zr hCust h) R ω t R₂ ∨
        SaltColl (holoAt st hp).rsch.sch.level₀ cap salt (Flock.Guarantees.checkedC3 ish
          (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr hCust h) R ω t R₂ ∨
        TopColl wr st reg bcs' t)
      (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows
        zr hCust h) dg N k Rw
      (Flock.Guarantees.regStmt (Flock.Guarantees.holoTys st hp) (holoCTs st hp) st hp
        (Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr
          hCust h) dg N k Rw
        (wmsgOf Vs (skLeaf Vs own sel (ℐ).r) (ℐ).r dec)
        (Flock.Guarantees.checkedT ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind rows zr
          hCust h σ) cap)
      (wmsgOf Vs (skLeaf Vs own sel (ℐ).r) (ℐ).r dec) (skOf Vs own sel)
      (cmtOf Vs bc) (msgOf Vs (skLeaf Vs own sel (ℐ).r) dec) := by
  have hS : 0 < vstarCount ish := Nat.lt_of_le_of_lt (Nat.zero_le _) sI.isLt
  have hell : hp.EllOk (execClassAt st).cls.kLog := FlockC3.VStarList.ellOk_of_sparseRange hrok hsi
  have hok : (Flock.RecCoins.layoutFactors (FlockC3.Sparse.unitShape (holoAt st hp))
      (Array.replicate ((holoAt st hp).cls.kLog - 6) (0 : Flock.F128))
      (Array.replicate ((holoAt st hp).cls.kLog - 6) (0 : Flock.F128)) rng).isOk = true := by
    obtain ⟨-, -, rng', hr', hl⟩ := hrok sp hsp
    obtain rfl : rng' = rng := Except.ok.inj (hr'.symm.trans hrng)
    have hsp' : FlockC3.Sparse.unitShape (holoAt st hp) = sp := by
      rw [FlockC3.Sparse.unitShape_holoAt hell]
      exact Option.some.inj (hsi.symm.trans hsp)
    rw [hsp']
    exact hl
  have hzsF : zs = (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp) (xR := xR) ish hss
      bind hob zr hCust h coins rows own sel bc dec wr reg).zs := hzs
  have hc : 0 < ish.claims := by omega
  have hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims := repFormsAt_holoAt hcO hi hcl2
  have hshape : (holoAt st hp).cls.shape = sparseOf sp := shape_holoAt hsi hsp hell
  have hL4 := FlockC3.RegLevels.holoStreams_four hrep hsp hsh hrs
  obtain ⟨Ls, hLs⟩ : ∃ Ls, Flock.Guarantees.holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some Ls :=
    ⟨_, hL4⟩
  -- what the checked sessions give at every proof they accept
  have hleaf : HLeafC3 zs dg N k Rw := by
    subst hzs
    exact hLeafC3_checked hS dg N k Rw
  have hcov : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        ∀ c, Flock.Guarantees.innerHeld wr st reg t = some c → (RecVStar.checkCovers ish c).isOk = true := by
    subst hzs
    exact covers_atSrc hS
  have hcovR : AtRep zs dg N k Rw sh rep fun _ _ t _ _ =>
      ∀ c, Flock.Guarantees.innerHeld wr st reg t = some c → (RecVStar.checkCovers ish c).isOk = true := by
    subst hzs
    exact FlockC3.Reps.covers_checked hS
  have hrec : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        ∃ spec sess, Flock.Record.decode spec (wr t).2 = .ok sess := by
    subst hzs
    exact recordDecodes_family (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp)
      (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) FlockC3.Sparse.coinsAccepted_checked
      hsi hell hsS hfiles _
  have hH : FlockC3.HPort.SrcHTie zs dg N k Rw (repsOfVals (histPts ish tbl bcM stp0) sh rep vals) sw := by
    subst hzs
    exact FlockC3.HPort.srcHTie_checked hcols hHA
  have hrs0 : RsRead zs dg N k Rw osh w own sel (Flock.Guarantees.roundAt Ls .stacked 0) := by
    have hr : RoundReads zs dg N k Rw osh w own sel (Flock.Guarantees.roundAt Ls .stacked 0) := by
      have hstR : ish.stacked = some ⟨shR.m, RecVStar.STACKED_CLAIMS⟩ := by rw [hsh, ← hcl]
      subst hown hsel
      rw [hzsF]
      exact FlockC3.Reps.RoundReads.roundReads_checked hF (FlockC3.Reps.RoundReads.holoSel_opening_at hLs hstR)
    have hall' := hall
    have hk' := hk
    rw [hzsF] at hall' hk' hr ⊢
    exact rsRead_checked hF hall' hk' hr
  have hc3 : C3Held wr (fun x => innerCommitsAt x.1 x.2 st reg) ish
      (fun (_ : Reg) (_ : L.Ω) (t : List (Cm × (ℐ).Coin)) (_ : (j : Fin (vstarCount ish)) → (Vs j).Bits) =>
        heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) :=
    c3Held_heldC3 hsp
  -- the same at the sessions refusing `Bad`, with G's items 1–4 there from `halg` and `hacc` off `Bad`
  have halgZ : AlgebraDecodes (I := ℐ) (offZs zs Bad) dg N k Rw sh rep := algebraDecodes_offZs hS halg
  have haccZ : AccChains (I := ℐ) (offZs zs Bad) dg N k Rw sh rep := accChains_offZs hS hacc
  have hleafZ : HLeafC3 (offZs zs Bad) dg N k Rw := hLeafC3_offZs hS hleaf
  have hprZ : ProofOfVals (offZs zs Bad) dg N k Rw sh rep vals := atRep_offZs hS hpr
  have hsvZ : SetupOfVals (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) sh rep vals :=
    atRep_offZs hS hsv
  have hfZ : FramesOfVals (offZs zs Bad) dg N k Rw sh rep vals := atRep_offZs hS hf
  have hvcapZ : AtRep (offZs zs Bad) dg N k Rw sh rep fun R ω t Lb i =>
      capDig (vals R ω t Lb i).p.cap = (stp0 R ω t Lb).cap₀ := atRep_offZs hS hvcap
  have hcovZ : AtSrc (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        ∀ c, Flock.Guarantees.innerHeld wr st reg t = some c → (RecVStar.checkCovers ish c).isOk = true :=
    atSrc_offZs hS hcov
  have hheldZ : AtSrc (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        HeldOk wr (fun x => innerCommitsAt x.1 x.2 st reg) t :=
    heldOk_atSrc hIC hrep hsp hsh hrs hc hA hshape (FlockC3.Sparse.htbl_of htbl) fun R ω t R₂ τ o hA' =>
      hleafZ R ω t R₂ τ o hA'
  have hcntZ : AtSrc (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        SparseCountedAt wr (fun x => innerCommitsAt x.1 x.2 st reg) sp t := by
    intro R ω t R₂ τ o hA'
    obtain ⟨rep₀, hsch⟩ := sched_of_holoRounds hrep
    refine sparseCountedAt_of_check hsch hsp fun cs hcs => ?_
    obtain ⟨⟨⟩, hck⟩ := FlockC3.VStarList.ok_of_isOk (hcovZ R ω t R₂ τ o hA' cs (congrArg Except.toOption hcs))
    exact FlockC3.Reps.checkCommitRounds_of_checkCovers hck
  have hheldRZ : RepRoundsHeld (offZs zs Bad) dg N k Rw sh rep wr (fun x => innerCommitsAt x.1 x.2 st reg)
      (fun i => (HoloStream.rep i).name tbl) fun _ q => q :=
    FlockC3.Reps.repRoundsHeld_of_rounds hrd
  have hdrZ : RepsDrawn (offZs zs Bad) dg N k Rw sh rep vals (holoSelOne ish tbl) (fun i => (HoloStream.rep i).name tbl)
      fun _ q => q :=
    FlockC3.Reps.repsDrawn_held hL4 hc hA halgZ haccZ hleafZ hprZ hsvZ hfZ hIC hrd htbl (atRep_offZs hS hcovR)
  have hptsZ : AtSrc (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t u =>
        FlockC3.HPort.PointsFromHistory ish tbl bcM t u :=
    pointsFromHistory_histPts (ish := ish) (tbl := tbl) (bc := bcM) (stp := FlockC3.Sparse.sparseOf stp0 sw sc)
      (sh := sh) (rep := rep) (vals := vals)
  -- the reps' and the sparse stages' ties, at the sessions refusing `Bad`
  have hUZ : RepUnitOfAlg (offZs zs Bad) dg N k Rw (tiedSrc (repsOfVals (histPts ish tbl bcM stp0) sh rep vals)) sh
      rep :=
    FlockC3.Reps.repUnitOfAlg_of_parts (src := tiedSrc (repsOfVals (histPts ish tbl bcM stp0) sh rep vals)) hprZ hsvZ
      hfZ (FlockC3.Reps.stepsOfVals_repsOfVals (stp := histPts ish tbl bcM stp0) hprZ hsvZ hfZ hvcapZ)
  have hpubZ : AtRep (offZs zs Bad) dg N k Rw sh rep fun R ω t Lb i =>
      FlockC3.Sparse.SparsePubRead (holoAt st hp) rng (rep R ω t Lb i) (sc R ω t Lb i) := by
    subst hsc
    exact FlockC3.Sparse.srcSparsePub_scOf hUZ hok _
  have hstZ : FlockC3.Sparse.SrcSparseTie (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) := by
    rw [repsOfVals_sparseOf]
    exact FlockC3.Sparse.srcSparseTie_of_copies hu hrng halgZ haccZ hUZ hpubZ
      (fun _ _ t _ => heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) unit sw
      (fun R ω t R₂ τ o hA' => hsess R ω t R₂ τ o (accepted_offZs hS hA').2) (srcHTie_offZs hS hH)
  have hrtZ : FlockC3.Reps.SrcRepTie (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) :=
    FlockC3.Reps.srcRepTie_of_alg halgZ haccZ (FlockC3.Reps.repUnitOfAlg_of_parts hprZ hsvZ hfZ
      (FlockC3.Reps.stepsOfVals_repsOfVals hprZ hsvZ hfZ hvcapZ))
  have hsstZ : AtSrc (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t u =>
        SparseStagesAt wr (fun x => innerCommitsAt x.1 x.2 st reg) ish unit sp t u :=
    sparseStagesAt_sparseOf hc3 hshape fun R ω t R₂ τ o hA' i => (hsess R ω t R₂ τ o (accepted_offZs hS hA').2 i).1
  have hmapsZ : MapsAt (offZs zs Bad) dg N k Rw M osh unit
      (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)
      (fun _ _ t _ => heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) w kap e0 sb e xR.cap :=
    mapsAt_offZs hS hmaps
  have hotZ : AtSrc (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun _ u =>
        ∃ sh w ops, FlockC3.Opening.OpenTie sh w ops u ∧ OpeningReadAt (runSteps u) u :=
    atSrc_offZs hS hot
  have hcoinsZ := stageCoins_holo_held hleafZ halgZ haccZ hprZ hsvZ hfZ hIC hheldRZ hdrZ hrtZ hstZ hotZ hell hosh
    hmapsZ hc3 hrep hsp hsh hcl hrs hc hA hshape (FlockC3.Sparse.htbl_of htbl) (FlockC3.Sparse.htblR_of htbl)
    (atSrc_offZs hS hrec) hptsZ hheldZ hcntZ hsstZ (atSrc_offZs hS hx) (atSrc_offZs hS hts)
  -- R's opening's committed rounds, past the sparse part, which ends at R's first history round
  have heZ := partEnds_sparse_roundAt hcO hi hcl2 hsp hsh hrs hshape hLs hrtZ hstZ
  have hopZ : OpenMsgsRead (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) own sel :=
    openMsgsRead_of_layer heZ (fun R ω t R₂ τ o hA' => (hmapsZ R ω t R₂ τ o hA').2.2.1) (rsRead_offZs hS hrs0)
  have hltZ : FlockC3.RegLevels.SrcLevelTie (offZs zs Bad) dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) xR.cap :=
    FlockC3.RegLevels.srcLevelTie_of_maps M hell hosh unit
      (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)
      (fun _ _ t _ => heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) w kap e0 sb e hmapsZ
  -- `SrcOK` at the sessions refusing `Bad`
  have hsZ : SrcOK (offZs zs Bad) dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals))) own
      sel :=
    srcOK_holo_of_leaf hleafZ hcoinsZ (repsMsgsRead_offZs hS hrp) (partMsgsRead_offZs hS hsm) hopZ
      (partMsgsRead_offZs hS hlg) (fun s hs => opensReadAt_offZs hS (hos s hs)) hrtZ hstZ hotZ hltZ
  -- the stages' ties at the normalized source
  have hrtNZ : FlockC3.Reps.SrcRepTie (offZs zs Bad) dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals))) :=
    FlockC3.Reps.srcRepTie_of_alg halgZ haccZ (FlockC3.Reps.repUnitOfAlg_of_parts hprZ hsvZ hfZ
      (FlockC3.Reps.stepsOfVals_norm_repsOfVals hprZ hsvZ hfZ hvcapZ))
  have hokNZ : FlockC3.Sparse.SrcSparseOK (offZs zs Bad) dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals))) :=
    fun R ω t R₂ τ o hA' => FlockC3.Sparse.srcSparseOK_of_tie hstZ R ω t R₂ τ o hA'
  have hopNZ : AtSrc (offZs zs Bad) dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)))
      fun _ u => OpeningReadAt (runSteps u) u := fun R ω t R₂ τ o hA' => by
    obtain ⟨_, _, _, ht, hor⟩ := hotZ R ω t R₂ τ o hA'
    exact openingReadAt_normOs ht hor
  have hltNZ : FlockC3.RegLevels.SrcLevelTie (offZs zs Bad) dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)))
      xR.cap := fun R ω t R₂ τ o hA' => levelTie_normOs (hltZ R ω t R₂ τ o hA')
  -- the public rows' premises and R's registered top
  have hownI : FlockC3.PubRows.OwnIn (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp)
      (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) own := by
    subst hown
    exact FlockC3.PubRows.ownIn_checked
  have hheldI : FlockC3.PubRows.HeldIs (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp)
      (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) own sel bc := by
    subst hsel
    exact heldIs_of_custody (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp) (xR := xR)
      ish hss bind hob zr hCust h coins rows own (Flock.Guarantees.holoMap st hp Cm ish tbl bcM).sel bc dec wr reg)
      own bc wr reg (Flock.Guarantees.holoMap st hp Cm ish tbl bcM) (fun _ _ _ => rfl) hIC hbc
  have htop : ∃ lv salt, StackedTopAt lv st reg xR.cap salt bcs' := ⟨_, salt, by rw [hcap]; exact hstk⟩
  clear hzsF
  subst hdec
  subst hzs
  exact hbr_honest_reg ish hss bind hob zr hCust coins rows own sel bc _ wr reg cap hcap salt bcs' htop dg N k Rw σ
    (fun _ => atAcceptedOff_offZs (fun R ω t R₂ hb hv => hb (hBad R ω t R₂ hv))
      (repsRead_c3DecOf hsZ (FlockC3.Reps.srcRepsReadAt_of_tie hrtNZ)))
    (fun _ => atAcceptedOff_offZs (fun R ω t R₂ hb hv => hb (hBad R ω t R₂ hv))
      (sparseRead_c3DecOf hsZ (FlockC3.Sparse.sparseReadAt_of_ok hokNZ)))
    (fun _ => atAcceptedOff_offZs (fun R ω t R₂ hb hv => hb (hBad R ω t R₂ hv)) (openingRead_c3DecOf hsZ hopNZ))
    (fun _ => atAcceptedOff_offZs (fun R ω t R₂ hb hv => hb (hBad R ω t R₂ hv))
      (FlockC3.RegLevels.regLevelsRead_of_tie hsZ hltNZ))
    (fun _ => atAcceptedOff_offZs (fun R ω t R₂ hb hv => hb (hBad R ω t R₂ hv)) (hFrame_c3DecOf hsZ))
    (fun _ => hownI) (fun _ => hheldI) h

end

end FlockC3.Compose
