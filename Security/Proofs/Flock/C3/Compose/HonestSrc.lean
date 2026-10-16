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

/-!
# `hbr_honest_reg` at the honest source

`hbr_honest_src`: the guarantee's `_hbr` in its form (`hbr_honest_reg`) at the checked family and the fixed decoder
`c3DecOf` over the honest source, `tiedSrc (normSrc (repsOfVals (sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep
vals))`: the run's points the history's first coin (`histPts`), each rep's sparse stage the steps of the session's
words `sw` at its coins `sc` (`sparseOf`), the reps' steps and openings their values' (`repsOfVals`). Its `SrcOK` is
`srcOK_holo_family_hist`'s and its `hrread` `repsReadOff_held`'s, at that one source. What the verifier's checks give
is discharged here:

* `hS` from the sparse statement's index, `hchk` (`coinsAccepted_checked`), `hrec` (`recordDecodes_family`), `hpts`
  (`pointsFromHistory_histPts`), `htblS` and `htbl` from the guarantee's `_htbl` (`htbl_of`, `htblR_of`), and `hcus`
  the guarantee's `_hIC` at `holoMap`;
* `hheld` (`heldOk_atSrc` at the whole history, `hLeafC3_checked`), `hcnt` (`sparseCountedAt_of_check` at the covers
  check every accepted proof of the checked sessions passes, `covers_atSrc`), `hc3` at the `C3` of the hold
  (`c3Held_heldC3`), `hheldR` (`repRoundsHeld_of_rounds`) and `hdr` (`repsDrawn_checked`);
* `hrt`, `hst` and `hsst` as `repsReadOff_held` makes them;
* from the inner statement's check (`checkInnerStmt`, the class's `ClassOk`, claims `2 · regions + 2`): `hA`
  (`repFormsAt_holoAt`), `hc`, and at `c₀` R's first history round, `he` (`partEnds_sparse_roundAt`) and `hr`
  (`roundReads_checked` at the map's `sel`, `holoSel_opening_at`); `hshape` from `SparseIs` (`shape_holoAt`);
* `hsread`, `horead`, `hlread` and `hfr` at the fixed decoder (`sparseRead_c3DecOf`, `openingRead_c3DecOf`,
  `regLevelsRead_of_tie`, `hFrame_c3DecOf`), each tie moved to the normalized source (`openingReadAt_normOs`,
  `levelTie_normOs`; the sparse plays read no openings); `hown` at `own := ownAt ss` (`ownIn_checked`), `hheld` at the
  map's `sel` (`heldIs_of_custody`), and `_htop` from the guarantee's `_hstk`.

What stays a hypothesis no check gives at this source: G's items 1–4 and the values' parts (`halg`, `hacc`, `hpr`,
`hsv`, `hf`, `hvcap`) at every accepted proof, the reps' held rounds (`hrd`), V*'s sessions and copies (`hsess`), the
pub read (`hpub`), R's opening at the maps (`hmaps`, `hot`), the other parts' read facts (`hrp`, `hsm`, `hlg`, `hos`,
`hall`, `hk`, `hx`, `hts`), the named `HCols` and `HAlgRows`, and the commit strings' custody (`hbc`).
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

/-- **(step 8) The guarantee's `_hbr` at the honest source**: `hbr_honest_reg` at the checked family and the fixed
decoder over `srcOK_holo_family_hist`'s source at `stp := sparseOf stp0 sw sc`, which is `repsReadOff_held`'s at
`stp := histPts ish tbl bcM stp0` (the two commute, definitionally), every hypothesis the verifier's checks give
discharged (module doc). `own` is the verifier's message ports (`ownAt ss`), `sel` the map's, and `dec` the fixed
decoder at the normalized source. -/
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
    (hzs : zs = Flock.Guarantees.checkedC3 ish (Flock.Guarantees.innerHeld wr st reg) (fun t => coins (wr t).2) bind
      rows zr hCust h)
    (hown : own = fun j : Fin (vstarCount ish) => FlockC3.PubRows.ownAt ss j)
    (hsel : sel = (Flock.Guarantees.holoMap st hp Cm ish tbl bcM).sel)
    (hdec : dec = c3DecOf zs dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals))) own sel)
    (hcap : xR.cap = cap) (hstk : StackedTopAt (holoAt st hp).rsch.sch.level₀ st reg cap salt bcs')
    (hIC : Flock.Premises.InnerRecordCustody wr (fun x => innerCommitsAt x.1 x.2 st reg) ish.sparse
      (Flock.Guarantees.holoMap st hp Cm ish tbl bcM))
    (htbl : ∀ i : Fin 2, (((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.SPARSE}/" ||
      ((HoloStream.rep i).name tbl).startsWith s!"{RecVStar.STACKED}/") = false)
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (hsi : FlockC3.VStarList.SparseIs ish hp) (hsS : ss[(sI : ℕ)]? = some (.sparse shS hR))
    (hfiles : ∀ r src, coins r = some src → ∃ f, src = .files f ∧ f.record = r)
    (hell : hp.EllOk (execClassAt st).cls.kLog)
    (hosh : RecOpening.shape? (holoAt st hp).cls.shape.mReg 2 = some osh)
    (hrep : holoRounds ish.m ish.kLog ish.claims = some rr)
    (hsp : ish.sparse = some sp) (hsh : ish.stacked = some shR) (hcl : shR.claims = RecVStar.STACKED_CLAIMS)
    (hrs : RecVStar.stackedRounds shR = some rs) (hcO : ClassOk (shapeOf st))
    (hi : RecVStar.checkInnerStmt ish st = .ok ()) (hcl2 : ish.claims = 2 * ish.regions + 2)
    (halg : AlgebraDecodes (I := ℐ) zs dg N k Rw sh rep) (hacc : AccChains (I := ℐ) zs dg N k Rw sh rep)
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
    (hpub : AtRep zs dg N k Rw sh rep fun R ω t Lb i =>
      FlockC3.Sparse.SparsePubRead (holoAt st hp) rng (rep R ω t Lb i) (sc R ω t Lb i))
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
  have hzsF : zs = (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp) (xR := xR) ish hss
      bind hob zr hCust h coins rows own sel bc dec wr reg).zs := hzs
  have hc : 0 < ish.claims := by omega
  have hA : repFormsAt (holoAt st hp) = repForms ish.m ish.kLog ish.claims := repFormsAt_holoAt hcO hi hcl2
  have hshape : (holoAt st hp).cls.shape = sparseOf sp := shape_holoAt hsi hsp hell
  obtain ⟨Ls, hLs⟩ : ∃ Ls, Flock.Guarantees.holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some Ls :=
    ⟨_, FlockC3.RegLevels.holoStreams_four hrep hsp hsh hrs⟩
  -- the verifier's hold, at every accepted proof of the checked sessions
  have hleaf : HLeafC3 zs dg N k Rw := by
    subst hzs
    exact hLeafC3_checked hS dg N k Rw
  have hheld : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        HeldOk wr (fun x => innerCommitsAt x.1 x.2 st reg) t :=
    heldOk_atSrc hIC hrep hsp hsh hrs hc hA hshape (FlockC3.Sparse.htbl_of htbl) fun R ω t R₂ τ o hA' =>
      hleaf R ω t R₂ τ o hA'
  have hcov : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        ∀ c, Flock.Guarantees.innerHeld wr st reg t = some c → (RecVStar.checkCovers ish c).isOk = true := by
    subst hzs
    exact covers_atSrc hS
  have hcnt : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        SparseCountedAt wr (fun x => innerCommitsAt x.1 x.2 st reg) sp t := by
    intro R ω t R₂ τ o hA'
    obtain ⟨rep₀, hsch⟩ := sched_of_holoRounds hrep
    refine sparseCountedAt_of_check hsch hsp fun cs hcs => ?_
    obtain ⟨⟨⟩, hk⟩ := FlockC3.VStarList.ok_of_isOk (hcov R ω t R₂ τ o hA' cs (congrArg Except.toOption hcs))
    exact FlockC3.Reps.checkCommitRounds_of_checkCovers hk
  have hc3 : C3Held wr (fun x => innerCommitsAt x.1 x.2 st reg) ish
      (fun (_ : Reg) (_ : L.Ω) (t : List (Cm × (ℐ).Coin)) (_ : (j : Fin (vstarCount ish)) → (Vs j).Bits) =>
        heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) :=
    c3Held_heldC3 hsp
  have hheldR : RepRoundsHeld zs dg N k Rw sh rep wr (fun x => innerCommitsAt x.1 x.2 st reg)
      (fun i => (HoloStream.rep i).name tbl) fun _ q => q :=
    FlockC3.Reps.repRoundsHeld_of_rounds hrd
  have hdr : RepsDrawn zs dg N k Rw sh rep vals (holoSelOne ish tbl) (fun i => (HoloStream.rep i).name tbl)
      fun _ q => q := by
    subst hzs
    exact FlockC3.Reps.repsDrawn_checked hS (FlockC3.RegLevels.holoStreams_four hrep hsp hsh hrs) hc hA halg hacc hpr
      hsv hf hIC hrd htbl
  have hrec : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t _ =>
        ∃ spec sess, Flock.Record.decode spec (wr t).2 = .ok sess := by
    subst hzs
    exact recordDecodes_family (C3Family.checked (T := Flock.Guarantees.holoTys st hp) (CT := holoCTs st hp)
      (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg) FlockC3.Sparse.coinsAccepted_checked
      hsi hell hsS hfiles _
  have hpts : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t u =>
        FlockC3.HPort.PointsFromHistory ish tbl bcM t u :=
    pointsFromHistory_histPts (ish := ish) (tbl := tbl) (bc := bcM) (stp := FlockC3.Sparse.sparseOf stp0 sw sc)
      (sh := sh) (rep := rep) (vals := vals)
  -- the reps' and the sparse stages' ties, as `repsReadOff_held` makes them
  have hU : RepUnitOfAlg zs dg N k Rw (tiedSrc (repsOfVals (histPts ish tbl bcM stp0) sh rep vals)) sh rep :=
    FlockC3.Reps.repUnitOfAlg_of_parts (src := tiedSrc (repsOfVals (histPts ish tbl bcM stp0) sh rep vals)) hpr hsv
      hf (FlockC3.Reps.stepsOfVals_repsOfVals (stp := histPts ish tbl bcM stp0) hpr hsv hf hvcap)
  have hH : FlockC3.HPort.SrcHTie zs dg N k Rw (repsOfVals (histPts ish tbl bcM stp0) sh rep vals) sw := by
    subst hzs
    exact FlockC3.HPort.srcHTie_checked hcols hHA
  have hst : FlockC3.Sparse.SrcSparseTie zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) := by
    rw [repsOfVals_sparseOf]
    exact FlockC3.Sparse.srcSparseTie_of_copies hu hrng halg hacc hU hpub
      (fun _ _ t _ => heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) unit sw hsess hH
  have hrt : FlockC3.Reps.SrcRepTie zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) :=
    FlockC3.Reps.srcRepTie_of_alg halg hacc (FlockC3.Reps.repUnitOfAlg_of_parts hpr hsv hf
      (FlockC3.Reps.stepsOfVals_repsOfVals hpr hsv hf hvcap))
  have hsst : AtSrc zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) fun t u =>
        SparseStagesAt wr (fun x => innerCommitsAt x.1 x.2 st reg) ish unit sp t u :=
    sparseStagesAt_sparseOf hc3 hshape fun R ω t R₂ τ o hA' i => (hsess R ω t R₂ τ o hA' i).1
  -- the sparse part ends at R's first history round, where the map reads R's opening's rounds
  have he := partEnds_sparse_roundAt hcO hi hcl2 hsp hsh hrs hshape hLs hrt hst
  have hr : RoundReads zs dg N k Rw osh w own sel (Flock.Guarantees.roundAt Ls .stacked 0) := by
    have hstR : ish.stacked = some ⟨shR.m, RecVStar.STACKED_CLAIMS⟩ := by rw [hsh, ← hcl]
    subst hown hsel
    rw [hzsF]
    exact FlockC3.Reps.RoundReads.roundReads_checked hF (FlockC3.Reps.RoundReads.holoSel_opening_at hLs hstR)
  -- `SrcOK` and `hrread` at the one source
  have hs0 := srcOK_holo_family_hist (xR := xR) (h := h) (dec := dec) (tbl := tbl) (bcM := bcM)
    (stp := FlockC3.Sparse.sparseOf stp0 sw sc) (sh := sh) (rep := rep) (vals := vals) hzsF hF hS
    FlockC3.Sparse.coinsAccepted_checked hsi hsS hfiles halg hacc hpr hsv hf hIC hheldR hdr hrt hst hot hell hosh
    hmaps hc3 hrep hsp hsh hcl hrs hc hA hshape (FlockC3.Sparse.htbl_of htbl) (FlockC3.Sparse.htblR_of htbl) hheld
    hcnt hsst hx hts hrp hsm he hall hk hr hlg hos
  have hs : SrcOK zs dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals))) own
      sel := hs0
  have hrread := repsReadOff_held hzsF hF hS halg hacc hpr hsv hf hvcap hIC hheldR hdr hrp hsm he hall hk hr hlg hos
    hot hell hosh hmaps hc3 hrep hsp hsh hcl hrs hc hA hshape htbl hrec hpts hheld hcnt hx hts hu hrng hpub hsess
    hcols hHA
  -- the stages' ties at the normalized source
  have hokN : FlockC3.Sparse.SrcSparseOK zs dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals))) :=
    fun R ω t R₂ τ o hA' => FlockC3.Sparse.srcSparseOK_of_tie hst R ω t R₂ τ o hA'
  have hopN : AtSrc zs dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)))
      fun _ u => OpeningReadAt (runSteps u) u := fun R ω t R₂ τ o hA' => by
    obtain ⟨_, _, _, ht, hor⟩ := hot R ω t R₂ τ o hA'
    exact openingReadAt_normOs ht hor
  have hlt : FlockC3.RegLevels.SrcLevelTie zs dg N k Rw
      (tiedSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)) xR.cap :=
    FlockC3.RegLevels.srcLevelTie_of_maps M hell hosh unit
      (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)
      (fun _ _ t _ => heldC3 ish sp (innerCommitsAt (wr t).1 (wr t).2 st reg)) w kap e0 sb e hmaps
  have hltN : FlockC3.RegLevels.SrcLevelTie zs dg N k Rw
      (tiedSrc (normSrc (repsOfVals (FlockC3.Sparse.sparseOf (histPts ish tbl bcM stp0) sw sc) sh rep vals)))
      xR.cap := fun R ω t R₂ τ o hA' => levelTie_normOs (hlt R ω t R₂ τ o hA')
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
    (fun h' => hrread _ h') (fun _ => sparseRead_c3DecOf hs (FlockC3.Sparse.sparseReadAt_of_ok hokN))
    (fun _ R ω t R₂ τ o hA _ => openingRead_c3DecOf hs hopN R ω t R₂ τ o hA)
    (fun _ R ω t R₂ τ o hA _ => FlockC3.RegLevels.regLevelsRead_of_tie hs hltN R ω t R₂ τ o hA)
    (fun _ => hFrame_c3DecOf hs) (fun _ => hownI) (fun _ => hheldI) h

end

end FlockC3.Compose
