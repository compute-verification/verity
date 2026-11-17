module

public import Proofs.Flock.Claims.Recursion.RegSetup
public import Proofs.Flock.Claims.Recursion.ResLowerD
public import Proofs.Flock.Claims.Recursion.ResLowerDD
public import Proofs.Flock.Claims.Recursion.VerdictRowEInner
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Recursion.AccFits
import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.Claims.Recursion.PartsVStar
import Proofs.Flock.Claims.Recursion.ResLowerDA
import Proofs.Flock.Claims.Recursion.ResSiteD
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.VerdictRowEAccR
import Proofs.Flock.Claims.Recursion.VerdictRowECommit
import Proofs.Flock.Claims.Recursion.VerdictRowEOpens
import Proofs.Flock.Claims.Scope.Fit
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts

/-! The chain's commit ties (`AccCommitV`, `InnerAccTieV`) at a joint acceptance of the parts' staged statements. -/

@[expose] public section

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

noncomputable section

theorem stacked_none {x : Msg} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) {r : XR} (hr : readX x = some r) : r.shape.stacked = none := by
  obtain ⟨r', hr', ha, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hd.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, -⟩, -⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  obtain ⟨-, -, -, -, ss', hc3⟩ := RecScope.statements_ok hss
  rcases RecScope.candidate3_ok hc3 with ⟨-, h⟩ | ⟨sp, -, hs, -⟩
  · exact h
  · rw [hsp] at hs
    cases hs

theorem fast100_of_read {x : Msg} {r : XR} (hr : readX x = some r) : ∃ sch, fast100 r.shape.m = some sch := by
  obtain ⟨vs, hvs, -, -⟩ := readX_partVStar hr
  obtain ⟨_, _, _, sch, -, -, -, hf, -⟩ := DecompBR.listed_of_statements hvs
  exact ⟨sch, hf⟩

/-- ResStepsW's `schAt_fast100`, restated so that this file needs no chain module. -/
private theorem schAt_fast100A {x : Msg} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt}
    (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss) {s lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) :
    fast100 r.shape.m = some (OpenLower.schAt (xIOf x) 0) := by
  obtain ⟨r', hr', -, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  have hh : r.shape.holo = false := hd.1
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hd.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  obtain ⟨sch, hsch, -, -, -, -⟩ := OpenLower.recOpen_dense hss hsp hh hsv
  obtain ⟨s', spec, t, hset, htm⟩ := OpenLower.inner_m hst hr
  have e : OpenLower.schAt (xIOf x) 0 = sch := by
    unfold OpenLower.schAt
    rw [hset]
    have htabs : FlockDeploy.tabsOf #[t] = #[_root_.Flock.Setup.ofCircuit t] := by simp [FlockDeploy.tabsOf]
    have h0 : (FlockDeploy.tabsOf #[t])[0]!.m = t.m := by
      rw [htabs]
      rfl
    dsimp only
    rw [h0, htm, hsch]
    rfl
  rw [e]
  exact hsch

theorem recOpen_n {x : Msg} {st : Claims.Recursion.CompiledStatement} (hx : parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) {r : XR} (hr : readX x = some r) {sch : Schedule}
    (hsch : fast100 r.shape.m = some sch) {s l LANES HH m : ℕ}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen l LANES HH m)) :
    m = 2 * (sch.levels[l]!).q := by
  cases hss : RecVStar.statements r.shape with
  | error e =>
    rw [hss] at hsv
    cases hsv
  | ok ss =>
    rw [hss] at hsv
    obtain ⟨hl, -, -, hm⟩ := OpenLower.levelMatch hx hf hr hss hsv
    have he : OpenLower.schAt (xIOf x) 0 = sch :=
      Option.some.inj ((schAt_fast100A hx hf hr hss hsv).symm.trans hsch)
    rw [he, ← hl] at hm
    exact hm

def SlotAt (r : XR) (sch : Schedule) (s : ℕ) (nm : String) (u P : ℕ) : Prop :=
  (∃ l LANES HH m d, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen l LANES HH m) ∧
    (nm, d) ∈ [("acc", 0), ("acc_out", 1)] ∧ u < 2 * (sch.levels[l]!).q ∧
    P = u / (sch.levels[l]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l +
      u % (sch.levels[l]!).q + d) ∨
  (∃ k name pt d, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner k name pt) ∧
    nm ∈ RecVStar.accPorts (.inner k name pt) ∧ (nm, d) ∈ [("acc0", 0), ("acc1", 1)] ∧ u < 2 ∧
    P = (u + d) * RecVStar.opensBelow sch sch.levels.size)

theorem slotAt_of_holding {r : XR} (hstk : r.shape.stacked = none) {sch : Schedule}
    (hsch : fast100 r.shape.m = some sch) {s P : ℕ} {h : String × ℕ} (hh : h ∈ accHoldings r s P) :
    SlotAt r sch s h.1 h.2 P := by
  obtain ⟨nm, u⟩ := h
  have hN : repOpens r = RecVStar.opensBelow sch sch.levels.size := by
    unfold repOpens
    rw [hsch]
    rfl
  unfold accHoldings at hh
  cases hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) with
  | none =>
    rw [hsv] at hh
    cases hh
  | some sv =>
    rw [hsv] at hh
    have hR : accRead r sv nm u = some P := by
      simpa using (List.mem_filter.1 hh).2
    cases sv with
    | recOpen l LANES HH m =>
      left
      have hu : u < 2 * (sch.levels[l]!).q := by
        have hA : (accRead r (.recOpen l LANES HH m) "acc" u).isSome = true := by
          by_cases hnm : nm = "acc"
          · subst hnm
            rw [hR]
            rfl
          · unfold accRead at hR
            simp [hnm] at hR
        exact MoreSlots.lt_recOpen hsch hA
      by_cases hnm : nm = "acc"
      · subst hnm
        refine ⟨l, LANES, HH, m, 0, hsv, List.mem_cons_self, hu, ?_⟩
        have hA : RecVStar.accReads r.shape (.recOpen l LANES HH m) =
            .ok (some ((Array.range (2 * (sch.levels[l]!).q)).map fun u =>
              some (u / (sch.levels[l]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l +
                u % (sch.levels[l]!).q))) := by
          unfold RecVStar.accReads
          simp only [hsch]
          try rfl
        unfold accRead at hR
        simp [hA, Except.toOption, hu] at hR
        dsimp only
        omega
      · unfold accRead at hR
        simp [hnm] at hR
    | inner k name pt =>
      right
      simp only [accRead] at hR
      by_cases h1 : u < 2 ∧ nm ∈ RecVStar.accPorts (.inner k name pt)
      · rw [if_pos h1] at hR
        by_cases h2 : nm = "acc0"
        · rw [if_pos h2] at hR
          subst h2
          obtain rfl := Option.some.inj hR
          refine ⟨k, name, pt, 0, hsv, h1.2, List.mem_cons_self, h1.1, ?_⟩
          rw [hN, Nat.add_zero]
        · rw [if_neg h2] at hR
          by_cases h3 : nm = "acc1"
          · rw [if_pos h3] at hR
            subst h3
            obtain rfl := Option.some.inj hR
            refine ⟨k, name, pt, 1, hsv, h1.2, List.mem_cons_of_mem _ List.mem_cons_self, h1.1, ?_⟩
            rw [hN, Nat.add_mul, Nat.one_mul]
          · rw [if_neg h3] at hR
            cases hR
      · rw [if_neg h1] at hR
        cases hR
    | recOpenStacked l LANES HH m =>
      exfalso
      cases hA : RecVStar.accReads r.shape (.recOpenStacked l LANES HH m) with
      | error e =>
        unfold accRead at hR
        simp [hA, Except.toOption] at hR
      | ok v =>
        unfold RecVStar.accReads at hA
        simp [hsch, hstk, throw, throwThe, MonadExceptOf.throw] at hA <;> cases hA
    | sparse sh hRd =>
      exfalso
      have hA : RecVStar.accReads r.shape (.sparse sh hRd) = .ok none := rfl
      unfold accRead at hR
      simp [hA, Except.toOption] at hR
    | opening m' c j S ports nv =>
      exfalso
      have hA : RecVStar.accReads r.shape (.opening m' c j S ports nv) = .ok none := rfl
      unfold accRead at hR
      simp [hA, Except.toOption] at hR
    | cap l N =>
      exfalso
      have hA : RecVStar.accReads r.shape (.cap l N) = .ok none := rfl
      unfold accRead at hR
      simp [hA, Except.toOption] at hR
    | link UB PB N =>
      exfalso
      have hA : RecVStar.accReads r.shape (.link UB PB N) = .ok none := rfl
      unfold accRead at hR
      simp [hA, Except.toOption] at hR

theorem slotAt_out {r : XR} (hstk : r.shape.stacked = none) {sch : Schedule} (hsch : fast100 r.shape.m = some sch)
    {s p u : ℕ} (hh : ("acc", u) ∈ accHoldings r s p) : SlotAt r sch s "acc_out" u (p + 1) := by
  rcases slotAt_of_holding hstk hsch hh with ⟨l, LANES, HH, m, d, hsv, hnd, hu, hP⟩ | ⟨k, name, pt, d, -, -, hd, -⟩
  · have hd0 : d = 0 := by
      simp only [List.mem_cons, Prod.mk.injEq, List.mem_nil_iff, or_false] at hnd
      rcases hnd with ⟨-, h⟩ | ⟨h, -⟩
      · exact h
      · exact absurd h (by decide)
    subst hd0
    dsimp only at hP
    exact Or.inl ⟨l, LANES, HH, m, 1, hsv, List.mem_cons_of_mem _ List.mem_cons_self, hu, by omega⟩
  · exfalso
    simp only [List.mem_cons, Prod.mk.injEq, List.mem_nil_iff, or_false] at hd
    rcases hd with ⟨h, -⟩ | ⟨h, -⟩ <;> exact absurd h (by decide)

def ReadsAcc (sv : VStar.Stmt) : Prop :=
  (∃ l LANES HH m, sv = .recOpen l LANES HH m) ∨ ∃ k name pt, sv = .inner k name pt

theorem instances_le {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {i : RecIns} {spec : SessionSpec}
    {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (h0 : 0 < sts.size) {r : XR} (hr : readX x = some r)
    {sv : VStar.Stmt} (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv) :
    sv.instances ≤ sts[0].pub.n := by
  obtain ⟨p, -, -, hxp, -, hpi, -, -, h0p, -⟩ := stagedTables hS
  have hp : p = { i.base.read with sts } := Option.some.inj (hxp.symm.trans (parse_of_setupO hsu))
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  obtain ⟨-, r', ss, sv', hr', hss, hsv', hpop⟩ := staged_scope hS hxp h0p
  rw [hr] at hr'
  cases hr'
  rw [hss] at hsv
  obtain rfl : sv' = sv := Option.some.inj (hsv'.symm.trans hsv)
  have hle := (oneFull_staged hS hsu).2 0 h0
  obtain ⟨-, hts⟩ := tablesOf_of_parse hxp
  obtain ⟨_, hset1⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset1
  rw [hpi] at hset1
  obtain ⟨-, hall1⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset1
  obtain ⟨d, hd1, -⟩ := hall1 0 h0p
  have hnn := setupOne_n hd1
  rw [e0] at hpop hnn
  omega

theorem registered_inner {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {r : XR}
    (hr : readX x = some r) {k : ℕ} {name : String} {pt : VStar.Part}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner k name pt))
    {i : RecIns} {spec : SessionSpec} {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) :
    ∃ own, i.base.registered = some own := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss, hss, hsv'⟩, h0p, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hsv
  obtain rfl : sv = .inner k name pt := Option.some.inj (hsv'.symm.trans hsv)
  obtain ⟨-, hset⟩ := setupO_some hsu
  have hp : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  have h0 : 0 < sts.size := by subst hp; exact h0p
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0p
    (FlockSoundness.Discharge.Recursion.checkParsed_flat (htab 0 h0p).2.2)
  have hpar := FlockVBridge.Family.resParams_of_check P.recResiduals
    ((htab 0 h0p).1.trans (DecompBR.inner_named hss hsv'))
  rw [e0] at hpar
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset) 0 h0
  cases ho : i.base.registered with
  | some own => exact ⟨own, rfl⟩
  | none =>
    exfalso
    rw [ho] at hcoin
    unfold RecCoins.check at hcoin
    simp [hpar] at hcoin

theorem tiePremises_of_accV {rd : Msg → Option FinalP} {wE : Msg → ℝ}
    {final : Msg → Option (List (List (Msg × Salt)) × Msg)} {w : Msg → ℝ} {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg} {s : ℕ} {x : Msg}
    {c : ℕ → Coin} {xO : Msg} {wt : Claims.Flock.Wit} {ms : List Msg}
    (h : AccV rd wE final w b elem shape untr cmt s x c xO wt ms) :
    ∃ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt), StagedPart b.st s x xO ∧
      Claims.Flock.check (deployZkRec _root_.Flock.Readable.relation).parse xO wt = true ∧
      setupO xO = some (i, spec, sts) ∧ ∃ h0 : 0 < sts.size,
      Registered.unitIndices sts[0].pub = .ok (Array.range sts[0].pub.n) ∧
      ∀ r, readX x = some r → ∀ sv, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some sv →
        sv.instances ≤ sts[0].pub.n ∧ (ReadsAcc sv → ∃ own, i.base.registered = some own) := by
  obtain ⟨-, hck, -, -, ⟨⟨-, hS⟩, hdec⟩, -⟩ := h
  obtain ⟨-, i, spec, sts, -, hsu, -⟩ := id hS
  have h0 : 0 < sts.size := by
    have := (oneFull_staged hS hsu).1
    omega
  refine ⟨i, spec, sts, hS, hck, hsu, h0, idx_of_dec hS hsu (hdec i spec sts hsu) h0, ?_⟩
  intro r hr sv hsv
  refine ⟨instances_le hS hsu h0 hr hsv, ?_⟩
  rintro (⟨l, LANES, HH, m, rfl⟩ | ⟨k, name, pt, rfl⟩)
  · exact registered_recOpen hS hr hsv hsu
  · exact registered_inner hS hr hsv hsu

theorem slotCommit_eq {rd : Msg → Option FinalP} {wE : Msg → ℝ}
    {final : Msg → Option (List (List (Msg × Salt)) × Msg)} {w : Msg → ℝ} {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))}
    {shape : Claims.Recursion.Form → Prop} {untr : TrReader} {cmt : Msg → ℕ → Msg → Msg} (hRS : RegSetup b)
    {x : Msg} {st : Claims.Recursion.CompiledStatement} (hx : (DRL rd wE final w b elem).parse x = some st)
    (hf : Claims.Recursion.Form.Dense st.form) {c : ℕ → Coin} {ws : List Claims.Flock.Wit}
    (hl : ∀ e ∈ (DeployedR.layR elem).allCalls x ws, FlockDeploy.sha512 e.1 = e.2) {xs : List Msg}
    (hxs : ∀ s' (hs' : s' < ws.length), AccV rd wE final w b elem shape untr cmt s' x c (xs.getD s' []) ws[s']
      ((DeployedR.layR elem).msgs x ws))
    (hcol : (Claims.findColl ((DeployedR.layR elem).allCalls x ws ++
      Claims.Recursion.setupCalls FlockDeploy.sha512 (DRL rd wE final w b elem) xs)).isSome = false)
    (hms : XsStaged b.st x xs ws.length) {r : XR} (hr : readX x = some r) {s s' : ℕ}
    {cc cc' : Circuit} (hpc : r.w.parts[s]?.bind partCircuit = some cc)
    (hpc' : r.w.parts[s']?.bind partCircuit = some cc') {wt wt' : Claims.Flock.Wit} (hw : ws[s]? = some wt)
    (hw' : ws[s']? = some wt') {sch : Schedule} (hsch : fast100 r.shape.m = some sch) {nm nm' : String}
    {u u' P : ℕ} (h1 : SlotAt r sch s nm u P) (h2 : SlotAt r sch s' nm' u' P) {qn q' : ℕ}
    (hq : cc.ports.findIdx? (·.name == nm) = some qn) (hq' : cc'.ports.findIdx? (·.name == nm') = some q') :
    readForms wt (hmCommit cc qn u) = readForms wt' (hmCommit cc' q' u') := by
  obtain ⟨hs, rfl⟩ := List.getElem?_eq_some_iff.1 hw
  obtain ⟨hs', rfl⟩ := List.getElem?_eq_some_iff.1 hw'
  obtain ⟨i, spec, sts, hS, hck, hsu, h0, hidx, hI⟩ := tiePremises_of_accV (hxs s hs)
  obtain ⟨i', spec', sts', hS', hck', hsu', h0', hidx', hI'⟩ := tiePremises_of_accV (hxs s' hs')
  obtain ⟨ms, hms⟩ := hms
  have hxO := hms s hs
  have hxO' := hms s' hs'
  have hxs1 := getD_mem_of_setup hsu
  have hxs1' := getD_mem_of_setup hsu'
  have hD : RegListed (DRL rd wE final w b elem) := regListed_recBound _ hRS _ _
  rcases h1 with ⟨l, LANES, HH, m, d, hsv, hnd, hu, hP⟩ | ⟨k, name, pt, d, hsv, hnm, hd, hu, hP⟩ <;>
    rcases h2 with ⟨l', LANES', HH', m', d', hsv', hnd', hu', hP'⟩ | ⟨k', name', pt', d', hsv', hnm', hd', hu', hP'⟩
  · have hm := recOpen_n hx hf hr hsch hsv
    have hm' := recOpen_n hx hf hr hsch hsv'
    obtain ⟨hn, hreg⟩ := hI r hr _ hsv
    obtain ⟨hn', hreg'⟩ := hI' r hr _ hsv'
    obtain ⟨own, hown⟩ := hreg (Or.inl ⟨_, _, _, _, rfl⟩)
    obtain ⟨own', hown'⟩ := hreg' (Or.inl ⟨_, _, _, _, rfl⟩)
    have hn2 : m ≤ sts[0].pub.n := hn
    have hn2' : m' ≤ sts'[0].pub.n := hn'
    exact accCommit_eq hS hS' hxO hxO' hck hck' hr hpc hpc' hsv hsv' hsch hsu hsu' h0 h0' hidx hidx' hown hown' hnd hnd' hq hq'
      hu (by omega) hu' (by omega) (hP.symm.trans hP') hD hl hcol hxs1 hxs1'
  · have hm := recOpen_n hx hf hr hsch hsv
    obtain ⟨hn, hreg⟩ := hI r hr _ hsv
    obtain ⟨hn', hreg'⟩ := hI' r hr _ hsv'
    obtain ⟨own, hown⟩ := hreg (Or.inl ⟨_, _, _, _, rfl⟩)
    obtain ⟨own', hown'⟩ := hreg' (Or.inr ⟨_, _, _, rfl⟩)
    have hn2 : m ≤ sts[0].pub.n := hn
    have hn2' : 2 ≤ sts'[0].pub.n := hn'
    exact (innerAccCommit_eq hS' hS hxO' hxO hck' hck hr hpc' hpc hsv' hsv hsch hsu' hsu h0' h0 hidx' hidx hown' hown hnm' hd'
      hnd hq' hq (by omega) hu (by omega) (hP'.symm.trans hP) hD hl hcol hxs1' hxs1).symm
  · have hm' := recOpen_n hx hf hr hsch hsv'
    obtain ⟨hn, hreg⟩ := hI r hr _ hsv
    obtain ⟨hn', hreg'⟩ := hI' r hr _ hsv'
    obtain ⟨own, hown⟩ := hreg (Or.inr ⟨_, _, _, rfl⟩)
    obtain ⟨own', hown'⟩ := hreg' (Or.inl ⟨_, _, _, _, rfl⟩)
    have hn2 : 2 ≤ sts[0].pub.n := hn
    have hn2' : m' ≤ sts'[0].pub.n := hn'
    exact innerAccCommit_eq hS hS' hxO hxO' hck hck' hr hpc hpc' hsv hsv' hsch hsu hsu' h0 h0' hidx hidx' hown hown' hnm hd
      hnd' hq hq' (by omega) hu' (by omega) (hP.symm.trans hP') hD hl hcol hxs1 hxs1'
  · obtain ⟨hn, hreg⟩ := hI r hr _ hsv
    obtain ⟨hn', hreg'⟩ := hI' r hr _ hsv'
    obtain ⟨own, hown⟩ := hreg (Or.inr ⟨_, _, _, rfl⟩)
    obtain ⟨own', hown'⟩ := hreg' (Or.inr ⟨_, _, _, rfl⟩)
    have hn2 : 2 ≤ sts[0].pub.n := hn
    have hn2' : 2 ≤ sts'[0].pub.n := hn'
    exact innerInnerCommit_eq hS hS' hxO hxO' hck hck' hr hpc hpc' hsv hsv' hsch hsu hsu' h0 h0' hidx hidx' hown hown' hnm hd
      hnm' hd' hq hq' (by omega) (by omega) (hP.symm.trans hP') hD hl hcol hxs1 hxs1'

def AccCommitV (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ (xs : List Msg), (∀ s' (hs' : s' < ws.length), AccV rd wE final w b elem shape untr cmt s' x c (xs.getD s' [])
      ws[s'] ((DeployedR.layR elem).msgs x ws)) →
    (Claims.findColl ((DeployedR.layR elem).allCalls x ws ++
      Claims.Recursion.setupCalls FlockDeploy.sha512 (DRL rd wE final w b elem) xs)).isSome = false →
    XsStaged b.st x xs ws.length →
    ∀ (r : XR), readX x = some r → ∀ (s s' : ℕ) (cc cc' : Circuit) (wt wt' : Claims.Flock.Wit),
      r.w.parts[s]?.bind partCircuit = some cc → r.w.parts[s']?.bind partCircuit = some cc' →
      ws[s]? = some wt → ws[s']? = some wt' →
    ∀ (p u : ℕ) (nm : String) (d : ℕ), (nm, d) ∈ [("acc", 0), ("acc_out", 1)] → ("acc", u) ∈ accHoldings r s p →
    ∀ h' ∈ accHoldings r s' (p + d), ∀ (qn q' : ℕ), cc.ports.findIdx? (·.name == nm) = some qn →
      cc'.ports.findIdx? (·.name == h'.1) = some q' →
      readForms wt (hmCommit cc qn u) = readForms wt' (hmCommit cc' q' h'.2)

theorem accCommitV (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) (hRS : RegSetup b) :
    AccCommitV rd wE final w b elem Claims.Recursion.Form.Dense untr cmt := by
  intro x st hx hf c ws hJ xs hxs hcol hms r hr s s' cc cc' wt wt' hpc hpc' hw hw' p u nm d hnd hh h' hh' qn q' hq hq'
  obtain ⟨sch, hsch⟩ := fast100_of_read hr
  have hstk := stacked_none hx hf hr
  have h1 : SlotAt r sch s nm u (p + d) := by
    simp only [List.mem_cons, Prod.mk.injEq, List.mem_nil_iff, or_false] at hnd
    rcases hnd with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · rw [Nat.add_zero]
      exact slotAt_of_holding hstk hsch hh
    · exact slotAt_out hstk hsch hh
  exact slotCommit_eq hRS hx hf hJ.2.1 hxs hcol hms hr hpc hpc' hw hw' hsch h1 (slotAt_of_holding hstk hsch hh') hq hq'

/-- Omits `InnerAccTie`'s `rowPfx` conjunct, which `held_rows_eq` doesn't need (`rowPfx_length`). -/
def InnerAccTieV (rd : Msg → Option FinalP) (wE : Msg → ℝ) (final : Msg → Option (List (List (Msg × Salt)) × Msg))
    (w : Msg → ℝ) (b : Brings) (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin)))
    (shape : Claims.Recursion.Form → Prop) (untr : TrReader) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (DRL rd wE final w b elem).parse x = some st → shape st.form →
    ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit),
    RecDeploy.OpenLower.JointP FlockDeploy.sha512 (DRL rd wE final w b elem) (DeployedR.layR elem)
      (AccP rd wE final w b elem shape untr cmt) x c ws →
    ∀ (xs : List Msg), (∀ s' (hs' : s' < ws.length), AccV rd wE final w b elem shape untr cmt s' x c (xs.getD s' [])
      ws[s'] ((DeployedR.layR elem).msgs x ws)) →
    (Claims.findColl ((DeployedR.layR elem).allCalls x ws ++
      Claims.Recursion.setupCalls FlockDeploy.sha512 (DRL rd wE final w b elem) xs)).isSome = false →
    XsStaged b.st x xs ws.length →
    ∀ (r : XR), readX x = some r → ∀ s (cS : Circuit), r.w.parts[s]?.bind partCircuit = some cS →
    ∀ (j : ℕ) (nm : String) (pt : VStar.Part),
      (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner j nm pt) →
    ∀ (p : ℕ) (h : String × ℕ), h ∈ accHoldings r s p →
    ∀ (o : ℕ) (co : Circuit) (ho : String × ℕ), accCanon r p = some (o, co, ho) →
    ∀ (wt wo : Claims.Flock.Wit), ws[s]? = some wt → ws[o]? = some wo →
    ∀ q, cS.ports.findIdx? (·.name == h.1) = some q →
    ∃ q', co.ports.findIdx? (·.name == ho.1) = some q' ∧
      readForms wt (hmCommit cS q h.2) = readForms wo (hmCommit co q' ho.2)

theorem innerAccTieV (rd : Msg → Option FinalP) (wE : Msg → ℝ)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (b : Brings)
    (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (untr : TrReader)
    (cmt : Msg → ℕ → Msg → Msg) (hRS : RegSetup b) :
    InnerAccTieV rd wE final w b elem Claims.Recursion.Form.Dense untr cmt := by
  intro x st hx hf c ws hJ xs hxs hcol hms r hr s cS hpc _ _ _ _ p h hh o co ho hcan wt wo hw hwo q hq
  obtain ⟨-, hpco, hhd⟩ := accCanon_spec hcan
  dsimp only at hpco hhd
  have hho : ho ∈ accHoldings r o p := List.mem_of_mem_head? hhd
  obtain ⟨q', hq'⟩ := Option.isSome_iff_exists.1 (accPort_deployed hr hpco (mem_accHeld hho))
  refine ⟨q', hq', ?_⟩
  obtain ⟨sch, hsch⟩ := fast100_of_read hr
  have hstk := stacked_none hx hf hr
  exact slotCommit_eq hRS hx hf hJ.2.1 hxs hcol hms hr hpc hpco hw hwo hsch (slotAt_of_holding hstk hsch hh)
    (slotAt_of_holding hstk hsch hho) hq hq'

end

end RecDeploy.ResLower
