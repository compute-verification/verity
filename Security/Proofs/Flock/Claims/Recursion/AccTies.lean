import Proofs.Flock.Claims.Recursion.VerdictRowEInner
import Proofs.Flock.Claims.Recursion.VerdictRowEAccR
import Proofs.Flock.Claims.Recursion.ResPiecesD
import Proofs.Flock.Claims.Recursion.ResLowerDA
import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.Claims.Recursion.ResLowerDD
import Proofs.Flock.Claims.Recursion.PartAcceptsSetup
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.AccFits

/-!
# The chain's commit ties at a joint acceptance

Row E's commit ties (`accCommit_eq`, `innerAccCommit_eq`, `innerInnerCommit_eq`) take, at each of the two parts, its
staged statement and check, its setup's table labelled in order, its reads file, its unit below the table's instance
count, and, across both, a record listing the registered reads (`RegListed`) and no collision among the true calls and
the setups' evaluations at a list `xs` holding both staged statements. Each part's premises come from its `AccV` at
`xs.getD s []` (`tiePremises_of_accV`): its staged statement and decision (`showsDec`) label the table (`idx_of_dec`);
V\* gives its unit count, at most the table's (`instances_le`); and a level or algebra part's unit is one
`RecCoins.check` refuses without the reads file (`registered_recOpen`, `registered_inner`). The statement sets up, so it
is in `xs` (`getD_mem_of_setup`), and the collision clause is `hcol`, which `PortsAtV` carries beside `JointP`. At the
dense form a part holds rows only as a level (`acc`) or an algebra part (`acc0`, `acc1`) (`slotAt_of_holding`; R's
levels are absent, `stacked_none`), at the positions `accRead` names, so any two slots of one row tie
(`slotCommit_eq`): `AccCommitW` and `InnerAccTie` with `xs`, `hxs` and `hcol` after `JointP`
(`accCommitV`, `innerAccTieV`), wherever the brings list the registered reads (`RegSetup`, which `deployedB` gives).
-/

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB

noncomputable section

/-! ## The shape's schedule -/

/-- **A dense statement names no R**: candidate 3's statements leave `stacked` empty without a sparse part
(`RecScope.candidate3_ok`). -/
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

/-- **A read statement's shape has a schedule** (`listed_of_statements`). -/
theorem fast100_of_read {x : Msg} {r : XR} (hr : readX x = some r) : ∃ sch, fast100 r.shape.m = some sch := by
  obtain ⟨vs, hvs, -, -⟩ := readX_partVStar hr
  obtain ⟨_, _, _, sch, -, -, -, hf, -⟩ := DecompBR.listed_of_statements hvs
  exact ⟨sch, hf⟩

/-- **The inner schedule is V\*'s** at a dense statement with a level part (`levelMatch`'s); ResStepsW's
`schAt_fast100`, here so that this file needs no chain module. -/
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

/-- **A level's instances are its schedule level's two reps' queries** (`levelMatch`, at `schAt_fast100A`'s
schedule). -/
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

/-! ## Where a part holds a row -/

/-- **Part `s` reads row `P` at port `nm`, unit `u`**: a level's `acc` (`d = 0`) or `acc_out` (`d = 1`) at unit
`u < 2q`, row `u / q · N + opensBelow l + u % q + d` (`RecVStar.accReads`); or an algebra part's `acc0` (`d = 0`) or
`acc1` (`d = 1`) at rep `u < 2`, row `(u + d) · N`. -/
def SlotAt (r : XR) (sch : Schedule) (s : ℕ) (nm : String) (u P : ℕ) : Prop :=
  (∃ l LANES HH m d, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.recOpen l LANES HH m) ∧
    (nm, d) ∈ [("acc", 0), ("acc_out", 1)] ∧ u < 2 * (sch.levels[l]!).q ∧
    P = u / (sch.levels[l]!).q * RecVStar.opensBelow sch sch.levels.size + RecVStar.opensBelow sch l +
      u % (sch.levels[l]!).q + d) ∨
  (∃ k name pt d, (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.inner k name pt) ∧
    nm ∈ RecVStar.accPorts (.inner k name pt) ∧ (nm, d) ∈ [("acc0", 0), ("acc1", 1)] ∧ u < 2 ∧
    P = (u + d) * RecVStar.opensBelow sch sch.levels.size)

/-- **A holding is a level's or an algebra part's slot** (`accRead`): R's levels are absent at a shape without R, and
the other statements read no row. -/
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

/-- **A level unit's `acc_out` is its next row's slot**: where it holds row `p` at `acc`, its `acc_out` reads `p + 1`. -/
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

/-! ## Each part's premises -/

/-- A statement that reads the chain as a level or an algebra part. -/
def ReadsAcc (sv : VStar.Stmt) : Prop :=
  (∃ l LANES HH m, sv = .recOpen l LANES HH m) ∨ ∃ k name pt, sv = .inner k name pt

/-- **A staged part's statement has at most its table's instances**: V\*'s count is the table's population
(`staged_scope`), at most its instances (`oneFull_staged`), its public file's (`setupOne_n`). -/
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

/-- **A staged algebra part's setup holds the verifier's reads file**: its unit is the verifier's
`rec-residuals/m<m>-k<k>-c<c>-p<j>` (`checkPart`'s name, `RecResiduals.check` in the parse), and `RecCoins.check`
refuses such a unit without `--registered`. -/
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

/-- **The commit ties' premises at part `s`, from its acceptance at its staged statement `xO` with its decision**
(`AccV`, `PortsAtV`'s `hxs`): `xO` is staged (`showsDec`) and its witness checks against it; its setup has at least
one table, labelled in order (`idx_of_dec`, at the decision `showsDec` carries); and at every V\* statement it has, at
most the table's instances (`instances_le`), with the reads file wherever it is a level or an algebra part
(`registered_recOpen`, `registered_inner`). -/
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

/-! ## Two slots of one row carry one commit string -/

/-- **Two slots of one row carry one commit string**, at a dense statement whose brings list the registered reads,
where every part accepts at its staged statement in `xs` (`hxs`) and the true calls `hl` and the setups' evaluations at
`xs` have no collision (`hcol`): each slot is a level's or an algebra part's (`SlotAt`), so `accCommit_eq`,
`innerAccCommit_eq` (both ways) or `innerInnerCommit_eq`, at each part's premises (`tiePremises_of_accV`). -/
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

/-! ## The accumulator ties `AccCommitW` and `InnerAccTie`, with every part's acceptance in `xs` and `hcol` -/

/-- **`AccCommitW` with the parts' staged statements**: at `JointP`, and wherever every part accepts at its statement
in `xs` with its decision (`hxs`, `PortsAtV`'s) and the true calls and the setups' evaluations at `xs` have no
collision (`hcol`): where part `s`'s unit `u` holds `acc` at row `p`, its `nm` port (`acc`, or `acc_out` for row
`p + 1`) and any holding `h'` of row `p + d` at part `s'` read one commit string at the parts' own witnesses. -/
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

/-- **`AccCommitV` at the dense form**, wherever the brings list the registered reads: the unit's `acc` (`acc_out`) is
row `p`'s (`p + 1`'s) slot (`slotAt_of_holding`, `slotAt_out`), as is `h'` (`slotAt_of_holding`), so they tie
(`slotCommit_eq`). -/
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

/-- **`InnerAccTie` with the parts' staged statements** (`hxs`, `hcol` as in `AccCommitV`): wherever part `s` (an
algebra part) holds row `p` at `h` (port `q`) and `accCanon` names the row's owner `(o, co, ho)`, the owner's port
`ho.1` is a port `q'` of its circuit and the two slots carry one commit string. (`InnerAccTie`'s `rowPfx` conjunct is
not needed: `held_rows_eq` takes prefixes of one length, `rowPfx_length`.) -/
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

/-- **`InnerAccTieV` at the dense form**, wherever the brings list the registered reads: the owner holds the row
(`accCanon_spec`) at a port of its circuit (`accPort_deployed`), and both holdings are the row's slots
(`slotAt_of_holding`), so they tie (`slotCommit_eq`). -/
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
