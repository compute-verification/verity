module

public import Proofs.Flock.Claims.KS.PlainSampleJCols
public import Proofs.Flock.Claims.KS.PlainSampleJRun
public import Proofs.Flock.Claims.Recursion.NextR
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.DeployK
import Proofs.Flock.Claims.KS.PlainSampleJMove
import Proofs.Flock.Claims.KS.Width
import Proofs.Flock.Soundness.Discharge.Exec.VerifyKJ
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts

/-! `plainVerdictAtJK` from an accepting step of the deployed reader (`FlockDeploy.nextR`), whose verdict is `nextJ`'s at `finalOfR`. -/

@[expose] public section

namespace FlockDeploy

open Flock
open Lean (Json)

def finalOfR (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (f : FinalP) :
    FinalJ :=
  ⟨f.stmt, shaSubJ spec ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).map (FlockComplete.dgOf spec (Claims.hOf H)))
    f.j, filesAt (tabsOf sts) (pointsOf spec c₁) f.ps⟩

theorem jAt_finalOfR (H : Claims.Msg → Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalP) :
    coinSubJ spec c₁ mids (finalOfR H spec sts c₁ f).j = jAt H spec sts c₁ mids f := rfl

theorem checksAtJ_finalOfR (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : FinalP) (v : Json) :
    checksAtJ s spec x link c₁ mids (finalOfR H spec sts c₁ f) v =
      checksAtJ s spec x link c₁ mids ⟨f.stmt, f.j, #[]⟩ v :=
  checksAtJ_shaSubJ s spec x link c₁ mids f.stmt _ f.j _ v

theorem actR_setupOf {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (h : actR rd x as cs = .decide s spec sts link c₁ mids f) :
    setupOf false x = some (s, spec, sts) := by
  obtain ⟨⟨s', spec', sts'⟩, _, _, _, _, _, _, hv, -, -, -, -, -, hd⟩ := decR_some (actR_decide h)
  simp only [DecR.mk.injEq] at hd
  obtain ⟨rfl, rfl, rfl, -⟩ := hd
  exact hv

end FlockDeploy

namespace Claims.Bridge.PlainSample

open _root_.Flock FlockDeploy FlockSoundness FlockSoundness.Discharge Claims.Bridge.Custody Claims.Bridge.CustodyPlain
open Lean (Json)

theorem decideR_sha512 {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) :
    verifyJWithMK id With.std spec (tabsOf sts) (jAt sha512 spec sts c₁ mids f)
        (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps) = .ok () ∧
      checksAtJ s spec x link c₁ mids ⟨f.stmt, f.j, #[]⟩ (jAt sha512 spec sts c₁ mids f) = .ok () ∧
      s.setup = .ok (spec, sts) ∧ s.read.zk = false ∧ s.read.publicInputs = none := by
  obtain ⟨spec', sts', hsu, hd⟩ := (nextR_acc_iff sha512 ha).1 hacc
  obtain ⟨hins, hs⟩ := setupOf_some (actR_setupOf ha)
  rw [hOf_sha512, setupWith_sha512, Ins.setup_read hs] at hsu
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hsu)
  obtain ⟨⟨⟩, hv, hc⟩ := FlockSoundness.Refine.ebind_ok
    ((decisionAtS_evalJ sha512 s spec sts x link c₁ mids (stmt := f.stmt) (j := f.j) (ps := f.ps)).symm.trans hd)
  have e : (_root_.Flock.Hashes.ofSha512 (Claims.hOf sha512) : _root_.Flock.Hashes _root_.Flock.V) = With.std := by
    rw [hOf_sha512]
  rw [e, verifyJWithM_eq_verifyJWithMK] at hv
  exact ⟨hv, hc, hs, readOf_zk (x := x) (by simp [readOf, hins]), publicInputs_none_of_check (checksAtJ_scope hc)⟩

end Claims.Bridge.PlainSample

namespace FlockDeploy

open Flock

noncomputable def runOfR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)
    (j : ℕ) : Option Claims.Flock.ZkRun :=
  match actR rd x as cs with
  | .decide _ spec sts _ c₁ mids f => do
    let st ← sts[j]?
    let tab := Setup.ofCircuit st
    let lv ← (liveOfJ spec (jAt sha512 spec sts c₁ mids f)).toOption
    let extras ← (tab.extra lv.points).toOption
    pure { m := tab.m, kLog := tab.kLog, claims := 2 + extras.size, hashLen := tab.merkle.digestLen,
           saltLen := saltLenOf false tab,
           reps := [(lv.streams.getD (2 * j) #[],
                     (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * j) ByteArray.empty),
                    (lv.streams.getD (2 * j + 1) #[],
                     (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD (2 * j + 1) ByteArray.empty)] }
  | _ => none

noncomputable def colsR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)
    (j : ℕ) : List (ℕ × List FlockLevel3.GF128) :=
  match runOfR rd x as cs j with
  | some r => Claims.Flock.plainRunCols r
  | none => []

/-- **The deployed reader's rounds**: `V false`'s round count, with the steps `nextR rd`. -/
noncomputable def VR (rd : Claims.Msg → Option FinalP) : Claims.Verifier Coin where
  k := (FlockDeploy.V false).k
  next := nextR rd

end FlockDeploy

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody Claims.Bridge.PlainSample
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

theorem colsR_plain {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {lv : Live} {st : Stmt} {sch : Schedule}
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hlv : liveOfJ spec (jAt sha512 spec sts c₁ mids f) = .ok lv)
    (hv : verifyJWithMK id With.std spec (tabsOf sts) (jAt sha512 spec sts c₁ mids f)
      (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps) = .ok ())
    (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch) (hp2 : spec.points = 2) :
    colsR rd x as cs 0 = Claims.Flock.distinctBelow (sch.levels.getD 0 default).d
      (Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 0 #[])
          ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 0 ByteArray.empty) ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit st) sch (2 * st.regions.size)) (lv.streams.getD 1 #[])
          ((FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 1 ByteArray.empty)) := by
  obtain ⟨h0, rfl⟩ := Array.getElem?_eq_some_iff.1 hst
  obtain ⟨sess, schs, hdec, -, -, -, hall⟩ := verifyJWithMK_tables_cap id With.std hv
  have h0' : 0 < (tabsOf sts).size := by simpa [tabsOf] using h0
  obtain ⟨_, _, hf, hrep, -⟩ := hall 0 h0'
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit sts[0] := tabsOf_getElem h0'
  simp only [ht0] at hf hrep
  obtain ⟨-, extras, -, -, hex, -⟩ := verifyRepWith_rounds With.std rfl hf hrep
  obtain ⟨-, hpts, -, -⟩ := decodeJWithK_live id With.std hdec hlv
  rw [hpts] at hex
  have hx : extras.size = 2 * sts[0].regions.size := by
    rw [extraClaims_size (st := sts[0]) hex, liveOfJ_points spec c₁ mids _ hlv,
      show (pointsOf spec c₁).size = spec.points by simp [pointsOf], hp2, Nat.mul_comm]
  have hsched : Claims.Flock.plainSched (Setup.ofCircuit sts[0]).m = sch := by
    simp only [Claims.Flock.plainSched]
    rw [show (Setup.ofCircuit sts[0]).m = sts[0].m from rfl, hsch]
    rfl
  have hrun : runOfR rd x as cs 0 = some
      { m := (Setup.ofCircuit sts[0]).m, kLog := (Setup.ofCircuit sts[0]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[0]).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit sts[0]),
        reps := [(lv.streams.getD 0 #[], (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 0
                    ByteArray.empty),
                 (lv.streams.getD 1 #[], (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD 1
                    ByteArray.empty)] } := by
    unfold runOfR
    simp [ha, hst, hlv, hex, Except.toOption]
  unfold colsR
  rw [hrun]
  simp only [Claims.Flock.plainRunCols, Claims.Flock.ZkRun.plainShape, Claims.Flock.plainLevel0, hsched,
    FlockDeploy.plainProofShape_eq hsched, hx, List.flatMap_cons, List.flatMap_nil, List.append_nil]

theorem txNoOffAtJK_of_colsR {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP} {d : Json} {N : ℕ} (C : Fin N → List 𝔽)
    (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d)
    (hno : ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (colsR rd x as cs 0) p.val = some u ∧ u ≠ C p) :
    TxNoOffAtJK id s.read.I d C
      ⟨jAt sha512 spec sts c₁ mids f, FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps⟩ := by
  obtain ⟨hv, hc, hsu, hzk, hpi⟩ := decideR_sha512 ha hacc
  have hsetA : setupAtD s.read.I (some d) = .ok (spec, sts) :=
    setupAtD_plainInputs (Ins.setup_read hsu) hzk hpi hd
  intro spec' st sess sch s₀ s₁ hset hdec hsch hs₀ hs₁
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsetA.symm.trans hset))
  obtain ⟨-, lv, hlv, -⟩ := checksAtJ_ok hc
  have hcols := colsR_plain (st := st) ha hlv hv (by simp) hsch (HiddenExec.setupTables_pointsAny hsetA).1
  obtain ⟨-, -, -, htsz, hpos, -⟩ := verifyJWithMK_tables_cap id With.std hv
  have htab : 0 < spec.tables.size := by rw [← htsz]; exact hpos
  obtain ⟨-, -, -, hstr⟩ := decodeJWithK_live id With.std hdec hlv
  have hco : ∀ (r : ℕ), r < 2 → ∀ y : StreamRec,
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r) = some y →
      ∀ k : ℕ, y.rounds[k]?.map (fun a : Round => a.coins) =
        (lv.streams.getD r #[])[k]?.map (fun a : Round => a.coins) := by
    intro r hr y hy k
    obtain ⟨hq, hw⟩ := Claims.Bridge.Custody.repStreams_getElem spec (i := 0) (r := r) htab hr
    obtain ⟨y', hy', hyr⟩ := hstr _ hq
    rw [hw] at hy'
    obtain rfl : y' = y := Option.some.inj (hy'.symm.trans hy)
    have e := forall₂_coins (fun _ _ h => h.2.1) hyr k
    simp only [Array.getElem?_toList] at e
    rw [e, show (#[] : Array Round) = default from rfl, getD_eq_getElem!]
    simp
  show ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (Claims.Flock.distinctBelow _
    (Claims.Flock.plainRepColsOf _ s₀.rounds (FlockDeploy.filesAt (tabsOf #[st]) (pointsOf spec c₁) f.ps)[0]! ++
      Claims.Flock.plainRepColsOf _ s₁.rounds (FlockDeploy.filesAt (tabsOf #[st]) (pointsOf spec c₁) f.ps)[1]!))
      p.val = some u ∧ u ≠ C p
  rw [plainRepColsOf_congr _ _ (hco 0 (by omega) s₀ hs₀), plainRepColsOf_congr _ _ (hco 1 (by omega) s₁ hs₁),
    ← getD_empty, ← getD_empty, ← hcols]
  exact hno

theorem plainVerdictAtJK_of_nextR_cols (mPts : ℕ) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : FinalP} {d : Json}
    {N : ℕ} (C : Fin N → List 𝔽) (ha : actR rd x as cs = .decide s spec sts link c₁ mids f)
    (hacc : ((nextR rd x as cs).eval sha512).1 = .acc) (hd : s.read.draw = some d)
    (hno : ¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (colsR rd x as cs 0) p.val = some u ∧ u ≠ C p) :
    plainVerdictAtJK id s.read.I d C link.1.toArray
      (viewOfJ 1 mPts spec link c₁ (jAt sha512 spec sts c₁ mids f)
        (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)) := by
  obtain ⟨hv, hc, hsu, hzk, hpi⟩ := decideR_sha512 ha hacc
  obtain ⟨-, -, -, htsz, hpos, -⟩ := verifyJWithMK_tables_cap id With.std hv
  have hJ : 1 ≤ spec.tables.size := by rw [← htsz]; exact hpos
  have hc' : checksAtJ s spec x link c₁ mids (finalOfR sha512 spec sts c₁ f) (jAt sha512 spec sts c₁ mids f) =
      .ok () := (checksAtJ_finalOfR sha512 s spec sts x link c₁ mids f _).trans hc
  have hA₀ : AcceptsAtDJK id s.read.I (some d)
      ⟨jAt sha512 spec sts c₁ mids f, FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps⟩ :=
    ⟨spec, sts, setupAtD_plainInputs (Ins.setup_read hsu) hzk hpi hd, hv⟩
  have hR₀ := recordsLiveAtDJK_of_checksJ id (J := 1) (mPts := mPts) hsu hzk hpi hd hJ hc'
  have hA := acceptsAtDJK_wrPDJK hA₀ hR₀
  exact ⟨hA, rfl, txNoOffAtJK_of_records hA₀ hR₀ hA (recordsLiveAtDJK_wrPDJK _ _ _ _)
    (txNoOffAtJK_of_colsR C ha hacc hd hno)⟩

theorem plainVerdictAtJK_of_runR (mPts : ℕ) (rd : Claims.Msg → Option FinalP) {Ω : Type} {P : Claims.Prover Ω}
    {x : Claims.Msg} {ω : Ω} {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      actR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      ∀ (d : Json), s.read.draw = some d → ∀ {N : ℕ} (C : Fin N → List 𝔽),
        (¬ ∃ (p : Fin N) (u : List 𝔽), Claims.Flock.lookCol (colsR rd x
          ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) 0)
            p.val = some u ∧ u ≠ C p) →
        plainVerdictAtJK id s.read.I d C link.1.toArray
          (viewOfJ 1 mPts spec link c₁ (jAt sha512 spec sts c₁ mids f)
            (FlockDeploy.filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)) := by
  have ha := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha ⊢
  replace ha : ((nextR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst) cs).eval sha512).1 = .acc := ha
  cases hact : actR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst) cs with
  | rej => revert ha; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert ha; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    exact ⟨s, spec, sts, link, c₁, mids, f, rfl, fun d hd _ C hno =>
      plainVerdictAtJK_of_nextR_cols mPts C hact ha hd hno⟩

end FlockDeploy.KS
