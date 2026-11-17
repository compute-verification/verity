module

public import Proofs.Flock.Claims.Recursion.DeployedBringsLam
public import Proofs.Flock.Claims.Recursion.NextRRows
public import Proofs.Flock.Claims.Recursion.VerdictRowDTie
public import Proofs.Flock.Claims.Recursion.DeployScope
public import Proofs.Flock.Soundness.Discharge.Hidden.Facts
public import Proofs.Flock.Soundness.Discharge.Integrate.Accepted
public import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG
public import Proofs.Flock.Soundness.Discharge.Exec.Setup
public import Proofs.Flock.Soundness.ExecSetup
public import Proofs.Flock.Soundness.Refine.Setup
public import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
public import Proofs.Flock.Claims.Recursion.SlotsCheck
public import Proofs.Flock.Claims.Recursion.RowsCover

/-! The deployed brings' setup count from the verifier's own refusals, with no condition on `x`
(`setupCalls_deployedB_le_cap`, `lamLin2_deployedB_ge_joint`). -/

@[expose] public section

set_option Elab.async false

namespace FlockSoundness.Discharge.SetupCounts

open Flock Refine
open FlockSoundness.Discharge.Exec (range_mapM_ok ok_of_refuse)
open Lean (Json)

structure Counts (st : Stmt) : Prop where
  n : st.n = st.pub.n
  nbl : st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1))
  m : st.m = st.c.kLog + st.nbl
  m35 : st.m ≤ 35
  g0 : 0 < st.c.g
  g7 : st.c.g * 2 ^ 7 ≤ 2 ^ st.c.kLog

theorem g_of_range {c : Circuit} {tags : Tags} {ls : Lean.Json} (h : HmRow.check c tags ls = .ok ()) {r}
    (hr : r ∈ c.ranges.toList) {i : ℕ} (hk : r.kind = .net i) (hcnt : 0 < r.count) :
    0 < c.g ∧ c.g * 2 ^ 7 ≤ 2 ^ c.kLog := by
  have hb : r.start + spanH r ≤ 2 ^ c.kLog ∧ r.count = c.g * c.perVu.getD i 0 := by
    cases ht : c.typed with
    | none =>
      have C := check_facts ht h
      exact ⟨(C.bound_slotLog r hr).1, C.count r hr i hk⟩
    | some tb =>
      have C := check_facts_typed ht h
      exact ⟨(C.bound_slotLog r hr).1, C.count r hr i hk⟩
  obtain ⟨hbr, hc⟩ := hb
  have h7 := (hm_check_slots h r hr).1
  have hsp := count_mul_le_spanH r
  rw [hc] at hcnt hsp
  have hg : 0 < c.g := Nat.pos_of_ne_zero fun h0 => by
    rw [h0, Nat.zero_mul] at hcnt
    exact Nat.lt_irrefl 0 hcnt
  have hp : 0 < c.perVu.getD i 0 := Nat.pos_of_ne_zero fun h0 => by
    rw [h0, Nat.mul_zero] at hcnt
    exact Nat.lt_irrefl 0 hcnt
  refine ⟨hg, ?_⟩
  calc c.g * 2 ^ 7 = c.g * 1 * 2 ^ 7 := by rw [Nat.mul_one]
    _ ≤ c.g * c.perVu.getD i 0 * 2 ^ r.slotLog :=
        Nat.mul_le_mul (Nat.mul_le_mul_left _ hp) (Nat.pow_le_pow_right (by norm_num) h7)
    _ ≤ spanH r := hsp
    _ ≤ r.start + spanH r := Nat.le_add_left _ _
    _ ≤ 2 ^ c.kLog := hbr

theorem hmOut_g {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmOut.parse tags cf tables = .ok c) {r} (hr : r ∈ c.ranges.toList) {i : ℕ} (hk : r.kind = .net i)
    (hcnt : 0 < r.count) : 0 < c.g ∧ c.g * 2 ^ 7 ≤ 2 ^ c.kLog := by
  obtain ⟨-, ⟨-, hc⟩ | ⟨k, b, tm, c₀, hc, rfl⟩ | ⟨k, unit, c₀, hc, rfl⟩⟩ := Hidden.parse_cases h
  · obtain ⟨ls, hls⟩ := hm_parse_check hc
    exact g_of_range hls hr hk hcnt
  · obtain ⟨ls, hls⟩ := hm_parse_check_any hc
    have hg := g_of_range (c := c₀) hls hr hk hcnt
    exact hg
  · obtain ⟨ls, hls⟩ := hm_parse_check_any hc
    have hg := g_of_range (c := c₀) hls hr hk hcnt
    exact hg

theorem Counts.n_le {st : Stmt} (C : Counts st) : st.pub.n ≤ 268435456 := by
  have hf : st.n ≤ st.c.g * 2 ^ st.nbl := by
    rw [C.nbl]
    exact Integrate.fits_blocks C.g0
  have h1 : st.n * 2 ^ 7 ≤ 2 ^ 35 :=
    calc st.n * 2 ^ 7 ≤ st.c.g * 2 ^ st.nbl * 2 ^ 7 := Nat.mul_le_mul_right _ hf
      _ = st.c.g * 2 ^ 7 * 2 ^ st.nbl := Nat.mul_right_comm _ _ _
      _ ≤ 2 ^ st.c.kLog * 2 ^ st.nbl := Nat.mul_le_mul_right _ C.g7
      _ = 2 ^ st.m := by rw [C.m, Nat.pow_add]
      _ ≤ 2 ^ 35 := Nat.pow_le_pow_right (by norm_num) C.m35
  rw [C.n] at h1
  have e1 : (2 : ℕ) ^ 7 = 128 := by norm_num
  have e2 : (2 : ℕ) ^ 35 = 34359738368 := by norm_num
  rw [e1, e2] at h1
  omega

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {publicInputs : Option ByteArray}
  {spec : SessionSpec} {sts : Array Stmt} {st : Stmt}

set_option maxHeartbeats 4000000 in
theorem setupH_counts (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) : Counts st := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨_, hc⟩ : ∃ ls, HmRow.check _ tags ls = .ok () := by
      first
      | exact hm_parse_check (by with_reducible assumption)
      | exact hm_parseTyped_check (by with_reducible assumption)
    have hk := Hidden.klog_of ‹¬ (_ || _) = true›
    obtain ⟨r, hr, i, hki, hcnt, -⟩ := pin_spec (by with_reducible assumption)
    have hg := g_of_range hc hr hki hcnt
    exact ⟨rfl, rfl, rfl, hk.2, hg.1, hg.2⟩

theorem hmOutC_g {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmOut.parseCommit tags cf tables = .ok c) {r} (hr : r ∈ c.ranges.toList) {i : ℕ} (hk : r.kind = .net i)
    (hcnt : 0 < r.count) : 0 < c.g ∧ c.g * 2 ^ 7 ≤ 2 ^ c.kLog := by
  cases hi : tags.isTmpl
  · rw [FlockSoundness.parseCommit_flat hi] at h
    exact hmOut_g h hr hk hcnt
  · rw [FlockSoundness.parseCommit_tmpl hi] at h
    obtain ⟨tu, c₀, hp, -, -, rfl⟩ := FlockSoundness.TmplValue.parseTmpl_cases h
    obtain ⟨ls, hls⟩ := hm_parse_check_any hp
    exact g_of_range (c := c₀) hls hr hk hcnt

set_option maxHeartbeats 8000000 in
theorem setupHidden_counts
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program publicInputs = .ok st) : Counts st := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hk := Hidden.klog_of ‹¬ (_ || _) = true›
    obtain ⟨r, hr, i, hki, hcnt, -⟩ := pin_spec (by with_reducible assumption)
    have hg := hmOutC_g (by with_reducible assumption) hr hki hcnt
    exact ⟨rfl, rfl, rfl, hk.2, hg.1, hg.2⟩

set_option maxHeartbeats 8000000 in
theorem setupTables_counts
    (h : Stmt.setupTables tags cf pf coins count tables draw partition program work stratified closure publicInputs =
      .ok (spec, sts)) :
    ∀ st ∈ sts.toList, Counts st := by
  unfold Stmt.setupTables at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp0 at h
  replace h : jp0 () = .ok (spec, sts) := by
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp0] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp1 at h
  replace h : jp1 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp1] at h
  extract_lets (config := { underBinder := false, onlyGivenNames := true }) jp2 at h
  replace h : jp2 () = .ok (spec, sts) := by
    try dsimp only at h
    repeat' walk_step
    all_goals exact h
  simp (config := { zeta := false }) only [jp2] at h
  dsimp only at h
  rcases Bool.eq_false_or_eq_true tags.hiddenOutputs with hh | hh
  · simp only [throw_bind', pure_bind, ite_eq_left hh] at h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
    all_goals
      first
      | (intro s hs
         simp at hs
         subst hs
         exact setupHidden_counts (ok_of_refuse (by assumption)))
      | (obtain ⟨-, hall⟩ := range_mapM_ok (by assumption)
         intro s hs
         obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hs)
         exact setupHidden_counts (ok_of_refuse (hall j hj)))
  · have hn : ¬ tags.hiddenOutputs = true := by simp [hh]
    simp only [throw_bind', pure_bind, ite_eq_right hn] at h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
    all_goals
      first
      | (intro s hs
         simp at hs
         subst hs
         exact setupH_counts (ok_of_refuse (by assumption)))
      | (obtain ⟨-, hall⟩ := range_mapM_ok (by assumption)
         intro s hs
         obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hs)
         exact setupH_counts (ok_of_refuse (hall j hj)))

end FlockSoundness.Discharge.SetupCounts

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

section Counts

open Flock
open FlockSoundness.Discharge.SetupCounts (Counts setupTables_counts)

theorem setupO_counts {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupO xO = some (i, spec, sts)) : sts.size ≤ 64 ∧ ∀ st ∈ sts.toList, Counts st := by
  have h := RecIns.setup_read (setupO_some hs).2
  have hmax : Tags.MAX_TABLES = 64 := rfl
  have hle := (FlockDeploy.setupTables_le h).1
  exact ⟨by omega, setupTables_counts h⟩

theorem commitIns_length_le (rows : Array (Array ℕ)) (salt : ℕ → ByteArray) :
    (ResLower.commitIns rows salt).length ≤ rows.size * 2 := by
  unfold ResLower.commitIns
  refine le_trans (flatMap_length_le _ 2 _ fun _ _ => ?_) (le_of_eq ?_)
  · simp
  · simp

theorem residualRowsOf_size {inn : RecCoins.Inner} {sh : Shape} {extras : Array Claim} {vs : Array ℕ}
    {rows : Array (Array ℕ)} (h : FlockSoundness.Discharge.Recursion.residualRowsOf inn sh extras vs = .ok rows) :
    rows.size = 2 := by
  unfold FlockSoundness.Discharge.Recursion.residualRowsOf at h
  have := (FlockSoundness.Discharge.Exec.array_mapM_ok_getElem h).1
  simpa using this

theorem residualRows_size {inn : RecCoins.Inner} {j : ℕ} {rows : Array (Array ℕ)}
    (h : FlockSoundness.Discharge.Recursion.residualRows inn j = some rows) : rows.size = 2 := by
  unfold FlockSoundness.Discharge.Recursion.residualRows at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact residualRowsOf_size (by assumption))

theorem openRows_size_le {inn : RecCoins.Inner} {LANES H : ℕ} {cr dr : Array (Array ℕ)}
    (hor : FlockSoundness.Discharge.Recursion.openRows inn LANES H = some (cr, dr)) : cr.size ≤ 510 ∧ dr.size ≤ 510 := by
  unfold FlockSoundness.Discharge.Recursion.openRows at hor
  cases ha : inn.algebra with
  | error e => rw [ha] at hor; simp at hor
  | ok pr =>
    obtain ⟨sh, extras⟩ := pr
    rw [ha] at hor
    obtain ⟨-, hsh⟩ := OpenLower.algebra_ok ha
    obtain ⟨sch, hf, -, -, -, hss, -⟩ := OpenLower.shape?_ok hsh
    subst hss
    simp only at hor
    cases hl : FlockSoundness.Discharge.Recursion.innerLevels sh LANES H with
    | nil => rw [hl] at hor; simp at hor
    | cons l t =>
      cases t with
      | cons _ _ => rw [hl] at hor; simp at hor
      | nil =>
        rw [hl] at hor
        simp only at hor
        cases h0 : inn.rep sh extras 0 with
        | error _ => rw [h0] at hor; simp at hor
        | ok r0 =>
          cases h1 : inn.rep sh extras 1 with
          | error _ => rw [h0, h1] at hor; simp at hor
          | ok r1 =>
            rw [h0, h1] at hor
            simp only [Option.some.injEq, Prod.mk.injEq] at hor
            obtain ⟨rfl, rfl⟩ := hor
            have hlt : l < sh.sched.levels.size := by
              have hm : l ∈ FlockSoundness.Discharge.Recursion.innerLevels sh LANES H := by
                rw [hl]
                exact List.mem_singleton.2 rfl
              unfold FlockSoundness.Discharge.Recursion.innerLevels at hm
              exact List.mem_range.1 (List.mem_filter.1 hm).1
            have hq := ((FlockSoundness.Discharge.DecodeZL.fast100_small _ _ hf).2.2 _
              (Array.getElem_mem hlt)).2.2.1
            simp only [FlockSoundness.Discharge.Recursion.openCoef, FlockSoundness.Discharge.Recursion.openDirs,
              Array.size_map, Array.size_range]
            rw [getElem!_pos sh.sched.levels l hlt]
            omega

theorem resSetupSt_length_le (src : Option RecCoins.Source) (st : Stmt) :
    (ResLower.resSetupSt src st).length ≤ 8 := by
  unfold ResLower.resSetupSt
  repeat' split
  all_goals first
    | (simp; done)
    | (rw [List.length_append]
       refine le_trans (Nat.add_le_add (commitIns_length_le _ _) (commitIns_length_le _ _)) ?_
       have hr := residualRows_size ‹_›
       simp only [RecCoins.zeroRows, Array.size_replicate]
       omega)

theorem openSetupSt_length_le (src : Option RecCoins.Source) (st : Stmt) :
    (ResLower.openSetupSt src st).length ≤ 2040 := by
  unfold ResLower.openSetupSt
  repeat' split
  all_goals first
    | (simp; done)
    | (rw [List.length_append]
       refine le_trans (Nat.add_le_add (commitIns_length_le _ _) (commitIns_length_le _ _)) ?_
       have hd := openRows_size_le ‹_›
       omega)

theorem resSetup_length_le (xO : Msg) : (ResLower.resSetup xO).length ≤ 131072 := by
  unfold ResLower.resSetup
  cases hs : setupO xO with
  | none => simp
  | some v =>
    obtain ⟨i, spec, sts⟩ := v
    dsimp only
    have hsz := (setupO_counts hs).1
    refine le_trans (flatMap_length_le _ 2048 _ fun st _ => ?_) ?_
    · simp only [List.length_append]
      exact le_trans (Nat.add_le_add (resSetupSt_length_le _ _) (openSetupSt_length_le _ _)) (by norm_num)
    · rw [Array.length_toList]
      omega

theorem budget_regs_ok {n : RecVStar.Inner} {own : Array Registered.Port}
    (h : RecBudget.regs (some n) (some own) = .ok ()) :
    own.size ≤ 512 ∧ ∀ reg ∈ own.toList, log2ceil reg.leaves ≤ 64 := by
  have hP : RecBudget.MAX_PORTS = 512 := rfl
  have hD : RecBudget.MAX_DEPTH = 64 := rfl
  unfold RecBudget.regs at h
  dsimp only at h
  split at h
  · first | cases h | simp [throw, throwThe, MonadExceptOf.throw] at h
  · rename_i hsz
    refine ⟨by omega, fun reg hreg => ?_⟩
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hreg)
    have hj' := FlockSoundness.Discharge.ZkExec.array_forM_ok h j hj
    try dsimp only at hj'
    split at hj'
    · first | cases hj' | simp [throw, throwThe, MonadExceptOf.throw] at hj'
    · omega

theorem budget_table_ok {n : RecVStar.Inner} {sts : Array Stmt} (h : RecBudget.table (some n) sts = .ok ()) :
    ∀ st ∈ sts.toList, st.pub.n ≤ 4096 := by
  have hI : RecBudget.MAX_INSTANCES = 4096 := rfl
  unfold RecBudget.table at h
  dsimp only at h
  intro st hst
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst)
  have hj' := FlockSoundness.Discharge.ZkExec.array_forM_ok h j hj
  try dsimp only at hj'
  split at hj'
  · first | cases hj' | simp [throw, throwThe, MonadExceptOf.throw] at hj'
  · omega

theorem budget_statements_split {n : RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : RecBudget.statements (some n) = .ok ()) (hss : RecVStar.statements n = .ok ss) :
    RecBudget.count ss.size = .ok () ∧ RecBudget.recorded (RecBudget.recordedCalls n ss) = .ok () := by
  unfold RecBudget.statements at h
  dsimp only at h
  rw [hss] at h
  dsimp only at h
  cases hc : RecBudget.count ss.size with
  | error e => rw [hc] at h; simp [bind, Except.bind] at h
  | ok u => rw [hc] at h; exact ⟨rfl, h⟩

theorem budget_statements_ok {n : RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : RecBudget.statements (some n) = .ok ()) (hss : RecVStar.statements n = .ok ss) :
    ss.size ≤ RecBudget.MAX_STATEMENTS := by
  have hS : RecBudget.MAX_STATEMENTS = 16 := rfl
  have h := (budget_statements_split h hss).1
  unfold RecBudget.count at h
  split at h
  · first | cases h | simp [throw, throwThe, MonadExceptOf.throw] at h
  · omega

theorem budget_recorded_ok {n : RecVStar.Inner} {ss : Array VStar.Stmt}
    (h : RecBudget.statements (some n) = .ok ()) (hss : RecVStar.statements n = .ok ss) :
    RecBudget.recordedCalls n ss ≤ 2 ^ 32 := by
  have hS : RecBudget.MAX_RECORDED = 2 ^ 32 := rfl
  have h := (budget_statements_split h hss).2
  unfold RecBudget.recorded at h
  split at h
  · first | cases h | simp [throw, throwThe, MonadExceptOf.throw] at h
  · omega

theorem RecIns.setup_regs {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    RecBudget.regs s.inner s.base.registered = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem RecIns.setup_statements {s : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (h : s.setup = .ok (spec, sts)) : RecBudget.statements s.inner = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem RecIns.setup_table {s : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : s.setup = .ok (spec, sts)) :
    RecBudget.table s.inner sts = .ok () := by
  unfold RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

theorem regSize_staged {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns} {spec : SessionSpec}
    {sts : Array Stmt} {own : Array Registered.Port} (hs : setupO (staged st x ms s) = some (i, spec, sts))
    (hown : i.base.registered = some own) :
    (∀ reg ∈ own.toList, 2 + log2ceil reg.leaves ≤ 66) ∧ own.size * 66 ≤ 33792 := by
  obtain ⟨hins, hset⟩ := setupO_some hs
  obtain ⟨r, -, hinner, -⟩ := insO_staged hins
  have hR := RecIns.setup_regs hset
  rw [hinner, hown] at hR
  obtain ⟨hsz, hdep⟩ := budget_regs_ok hR
  exact ⟨fun reg hreg => by have := hdep reg hreg; omega, by omega⟩

theorem nS_le_staged {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns} {spec : SessionSpec}
    {sts : Array Stmt} (hs : setupO (staged st x ms s) = some (i, spec, sts)) :
    nS x ≤ RecBudget.MAX_STATEMENTS := by
  obtain ⟨hins, hset⟩ := setupO_some hs
  obtain ⟨r, hr, hinner, -⟩ := insO_staged hins
  obtain ⟨vs, hvs, hlen, -⟩ := readX_listed hr
  have hB := RecIns.setup_statements hset
  rw [hinner] at hB
  have hn : nS x = r.w.parts.length := by simp [nS, hr]
  rw [hn, hlen]
  exact budget_statements_ok hB hvs

theorem nS_le_of_partIn {st : Stage} {x : Msg}
    (hO : ∀ j < nS x, ∃ i spec' sts', PartIn (staged st x (pick st x j) j) i spec' sts') :
    nS x ≤ RecBudget.MAX_STATEMENTS := by
  rcases Nat.eq_zero_or_pos (nS x) with h0 | hpos
  · rw [h0]; exact Nat.zero_le _
  · obtain ⟨i, spec', sts', hp⟩ := hO 0 hpos
    exact nS_le_staged hp.setup

theorem mem_vstar_staged {st : Stage} {x : Msg} {ps : List (Msg × Out)} {xO : Msg} (h : xO ∈ vstar st x ps) :
    ∃ ms s, xO = staged st x ms s := by
  unfold vstar at h
  obtain ⟨s, -, rfl⟩ := List.mem_map.1 h
  unfold vstarAt
  split_ifs
  · exact ⟨_, _, rfl⟩
  · exact ⟨_, _, rfl⟩

theorem deployedSetup_staged_le {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} :
    (deployedSetup (staged st x ms s)).length ≤ 138543104 := by
  rw [deployedSetup_length]
  refine le_trans (Nat.add_le_add (resSetup_length_le _) (?_ : (regIns (staged st x ms s)).length ≤ 138412032))
    (by norm_num)
  cases hs : setupO (staged st x ms s) with
  | none =>
    rw [regIns_eq_nil fun _ _ _ _ h => by rw [hs] at h; cases h]
    exact Nat.zero_le _
  | some v =>
    obtain ⟨i, spec, sts⟩ := v
    cases hown : i.base.registered with
    | none =>
      rw [regIns_eq_nil fun i' spec' sts' own h => by
        rw [hs] at h
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, -, -⟩ := h
        simp [hown]]
      exact Nat.zero_le _
    | some own =>
      obtain ⟨hins, hset⟩ := setupO_some hs
      obtain ⟨r, -, hinner, -⟩ := insO_staged hins
      obtain ⟨hD, hKD⟩ := regSize_staged hs hown
      have hT := RecIns.setup_table hset
      rw [hinner] at hT
      have hO := RecIns.setup_oneFull hset
      rw [hinner] at hO
      have h1 := (oneFull_ok hO).1
      calc (regIns (staged st x ms s)).length ≤ sts.size * (own.size * 4096) * 66 :=
            regIns_length_le hs hown (budget_table_ok hT) hD
        _ = sts.size * 4096 * (own.size * 66) := by ring
        _ ≤ 1 * 4096 * 33792 := Nat.mul_le_mul (Nat.mul_le_mul_right _ (le_of_eq h1)) hKD
        _ = 138412032 := by norm_num

end Counts

theorem setupCalls_deployedB_le_cap (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ nS x * 138543104 :=
  setupCalls_deployedB_le_at rel b L IV H x ps fun xO hx => by
    obtain ⟨ms, s, rfl⟩ := mem_vstar_staged hx
    exact deployedSetup_staged_le

theorem setupCalls_deployedB_le_32 (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) (hn : nS x ≤ 16) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ 2 ^ 32 :=
  (setupCalls_deployedB_le_cap rel b L IV H x ps).trans ((Nat.mul_le_mul_right _ hn).trans (by norm_num))

theorem setupCalls_deployedB_le_joint (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) (hn : nS x ≤ 2 ^ 16) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ 2 ^ 55 :=
  (setupCalls_deployedB_le_cap rel b L IV H x ps).trans ((Nat.mul_le_mul_right _ hn).trans (by norm_num))

/-- What it still takes is no hash premise and no statement about the prover: `hU` (at most `2^55` oracle queries), the
scope conditions `hO` and `hin`, and `hS`, a deployment-size condition read off `x`. -/
theorem lamLin2_deployedB_ge_joint (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (H : Msg → Msg)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts',
      PartIn (staged (deployedB b).st x (pick (deployedB b).st x j) j) i spec' sts')
    (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) (ps : List (Msg × Out)) {n : ℕ} (hU : n ≤ 2 ^ 55) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n (2 ^ 55) ∧
    (Claims.Recursion.setupCalls H
      (recBound rel (deployedB b) L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      ((recBound rel (deployedB b) L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).vstar x
        ps)).length ≤ 2 ^ 55 :=
  have hn : nS x ≤ 2 ^ 16 := (nS_le_of_partIn hO).trans (by decide)
  ⟨lamLin2_deployedB_ge rel b L nextE colsE wE join rowIn final w decI hin hO hn hS hU,
    setupCalls_deployedB_le_joint rel b L _ H x ps hn⟩

end RecDeploy
