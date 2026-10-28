import Proofs.Flock.Claims.Recursion.DeployedBringsLam
import Proofs.Flock.Claims.Recursion.NextRRows
import Proofs.Flock.Claims.Recursion.VerdictRowDTie
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
import Proofs.Flock.Soundness.Discharge.Integrate.Accepted
import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.ExecSetup
import Proofs.Flock.Soundness.Refine.Setup

/-!
# The deployed brings' setup count from the verifier's own refusals

`setupCalls_deployedB_le_at` bounds the compiled verifier's own evaluations by `nS x · c` when each staged V\*
statement's `deployedSetup` has at most `c` inputs. This file derives `c` from what the verifier refuses, leaving one
deployment-size condition on public data, `RegSize x K`, beside `lamLin2_deployedB_ge`'s `hn` and `hS`:

* **Tables**: a session has at most `Tags.MAX_TABLES = 64` statements (`setupTables_le`, at any `publicInputs`).
* **Instances**: every statement `Stmt.setupTables` returns, at any `publicInputs` (`setupTables_counts`; the plain path
  refuses `publicInputs`, the hidden one passes it to `Stmt.setupHidden`), has `m ≤ 35` (the `k_log > 27 || m > 35`
  refusal), `0 < g` (`HmRow.pin` refuses a net range of no slots, and `HmRow.check` makes each net range `g · per_vu`
  slots) and `g · 2^7 ≤ 2^k_log` (`HmRow.check`'s slots of at least `2^7` inside the block), so
  `n ≤ g · 2^nbl ≤ 2^(k_log - 7) · 2^(35 - k_log) = 2^28` (`Counts.n_le`).
* **Check 5's rows** (`resSetup_length_le`): `rec-residuals` registers two rows of `v` and two zero rows, `rec-open`
  `2 q` rows each of `coef` and `dirs`, two inputs a row; `q < 256` at every level of every `fast100` schedule, a
  constant of the verifier (`fast100_small`), so no condition on `x` bounds it. At most `64 · (8 + 2040) = 131072`
  inputs.
* **Registered reads** (`regIns_length_le`): at most `|sts| · |own| · n · D`, where `own` and `D` are read off the
  registration bytes `x`'s parts carry (`RegSize`).

So each statement has at most `131072 + 2^34 · K` setup inputs (`deployedSetup_staged_le`), the run at most
`nS x · (131072 + 2^34 · K)` (`setupCalls_deployedB_le_reg`), and within the budget
`nS x · (131072 + 2^34 · K) ≤ 2^55` at most `2^55` (`setupCalls_deployedB_le_joint`), the call bound
`lamLin2_deployedB_ge` charges (`lamLin2_deployedB_ge_joint`); at `nS x ≤ 2^10` and `K = 2047`
(`lamLin2_deployedB_ge_2047`). A deployment's `K` is at least `36` (`setupCalls_deployedB_le_2047`), so no budget at
`nS x ≤ 2^16` holds there.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.SetupCounts

open Flock Refine
open FlockSoundness.Discharge.Exec (range_mapM_ok ok_of_refuse)
open Lean (Json)

/-- **What a set-up statement's sizes satisfy**: `n` is the public file's, `nbl` its blocks' `log2ceil` (at least 3),
`m = k_log + nbl ≤ 35`, and the circuit's `g` is positive with `g · 2^7 ≤ 2^k_log`. -/
structure Counts (st : Stmt) : Prop where
  n : st.n = st.pub.n
  nbl : st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1))
  m : st.m = st.c.kLog + st.nbl
  m35 : st.m ≤ 35
  g0 : 0 < st.c.g
  g7 : st.c.g * 2 ^ 7 ≤ 2 ^ st.c.kLog

/-- **A net range with a slot bounds `g`**: `HmRow.check` makes it `g · per_vu` slots of at least `2^7` each inside
the block, so `0 < g` and `g · 2^7 ≤ 2^k_log`. -/
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

/-- `g_of_range` at a circuit `HmOut.parse` returned: `HmRow.parse`'s, with its block or class set. -/
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

/-- `n` instances in `2^nbl` blocks of `g`, at `m ≤ 35` and `g · 2^7 ≤ 2^k_log`: at most `2^28`. -/
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
/-- **`Stmt.setupH`'s statements satisfy `Counts`.** -/
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

/-- `hmOut_g` at a circuit `HmOut.parseCommit` returned, flat (`HmOut.parse`'s) or templated (`HmRow.parse`'s with the
template's unit net first). -/
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
/-- **`Stmt.setupHidden`'s statements satisfy `Counts`**, at any public input words. -/
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
/-- **Every statement `Stmt.setupTables` returns satisfies `Counts`**, at any public input words: the plain path
refuses them, and the hidden path passes them to `Stmt.setupHidden` (`setupHidden_counts`). -/
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

/-! ## The setup's statements -/

/-- **A V\* statement's recursive setup**: at most 64 statements, each satisfying `Counts`. -/
theorem setupO_counts {xO : Msg} {i : RecIns} {spec : SessionSpec} {sts : Array Stmt}
    (hs : setupO xO = some (i, spec, sts)) : sts.size ≤ 64 ∧ ∀ st ∈ sts.toList, Counts st := by
  have h := RecIns.setup_read (setupO_some hs).2
  have hmax : Tags.MAX_TABLES = 64 := rfl
  have hle := (FlockDeploy.setupTables_le h).1
  exact ⟨by omega, setupTables_counts h⟩

/-! ## Check 5's rows -/

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

/-- `rec-residuals` registers two rows of `v`, one a rep. -/
theorem residualRows_size {inn : RecCoins.Inner} {j : ℕ} {rows : Array (Array ℕ)}
    (h : FlockSoundness.Discharge.Recursion.residualRows inn j = some rows) : rows.size = 2 := by
  unfold FlockSoundness.Discharge.Recursion.residualRows at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact residualRowsOf_size (by assumption))

/-- **`rec-open` registers at most 510 rows each of `coef` and `dirs`**: `2 q` at the inner schedule's one level of
its lanes and siblings, and `q < 256` at every level of every `fast100` schedule (`fast100_small`), which
`RecResiduals.shape?` requires of the inner session's `m`: a constant of the verifier, not a condition on `x`. -/
theorem openRows_size_le {inn : RecCoins.Inner} {LANES H : ℕ} {cr dr : Array (Array ℕ)}
    (hor : FlockSoundness.Discharge.Recursion.openRows inn LANES H = some (cr, dr)) :
    cr.size ≤ 510 ∧ dr.size ≤ 510 := by
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

/-- **Check 5's evaluations at a V\* statement**: at most `64 · (8 + 2040)`. -/
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

/-! ## The registered reads -/

/-- The verifier's reading of the registration bytes `--registered` (`FlockDeploy.Raw.ins`'s). -/
def regOf (b : Option ByteArray) : Except String (Option (Array Registered.Port)) :=
  b.mapM fun b => do Registered.ofJson (← FlockDeploy.jsonOf "registered reads" b)

theorem raw_ins_registered {w : FlockDeploy.Raw} {s : FlockDeploy.Ins} (h : w.ins = .ok s) :
    regOf w.registered = .ok s.registered := by
  unfold FlockDeploy.Raw.ins at h
  unfold regOf
  simp only [bind, Except.bind, pure, Except.pure] at h ⊢
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; assumption)

/-- **A staged statement's registration is its part's**: the bytes `x`'s part `s` carries, read as `Raw.ins` reads
them. -/
theorem registered_staged {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ} {i : RecIns}
    (h : insO (staged st x ms s) = some i) :
    ∃ r p, readX x = some r ∧ r.w.parts[s]? = some p ∧ regOf p.base.registered = .ok i.base.registered := by
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, hr, hp, hd⟩ := staged_decode hne
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  have hreg := raw_ins_registered (RecRaw.ins_base (toOption_eq_some h'))
  exact ⟨r, p, hr, hp, hreg⟩

/-- **The registrations `x` carries are of size at most `K`**: wherever a part of `x` carries registration bytes the
verifier reads as ports `own`, some `D` bounds each port's opening (`2 + log2ceil leaves` inputs) and
`|own| · D ≤ K`. A deployment-size condition on public data: the ports are read off `x`'s bytes, and no hash premise or
prover behaviour enters it. -/
def RegSize (x : Msg) (K : ℕ) : Prop :=
  ∀ r, readX x = some r → ∀ p ∈ r.w.parts, ∀ own, regOf p.base.registered = .ok (some own) →
    ∃ D, (∀ reg ∈ own.toList, 2 + log2ceil reg.leaves ≤ D) ∧ own.size * D ≤ K

theorem mem_vstar_staged {st : Stage} {x : Msg} {ps : List (Msg × Out)} {xO : Msg} (h : xO ∈ vstar st x ps) :
    ∃ ms s, xO = staged st x ms s := by
  unfold vstar at h
  obtain ⟨s, -, rfl⟩ := List.mem_map.1 h
  unfold vstarAt
  split_ifs
  · exact ⟨_, _, rfl⟩
  · exact ⟨_, _, rfl⟩

/-! ## The count -/

/-- **A staged statement's `deployedSetup`**: at most `131072 + 2^34 · K` inputs, where `x`'s registrations are of size
at most `K` (`RegSize`): check 5's (`resSetup_length_le`) and at most `64 · 2^28 · K` registered reads. -/
theorem deployedSetup_staged_le {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s K : ℕ} (hK : RegSize x K) :
    (deployedSetup (staged st x ms s)).length ≤ 131072 + 17179869184 * K := by
  rw [deployedSetup_length]
  refine Nat.add_le_add (resSetup_length_le _) ?_
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
      obtain ⟨r, p, hr, hp, hreg⟩ := registered_staged (setupO_some hs).1
      rw [hown] at hreg
      obtain ⟨D, hD, hKD⟩ := hK r hr p (List.mem_of_getElem? hp) own hreg
      obtain ⟨hsz, hC⟩ := setupO_counts hs
      calc (regIns (staged st x ms s)).length ≤ sts.size * (own.size * 268435456) * D :=
            regIns_length_le hs hown (fun st' hst => (hC st' hst).n_le) hD
        _ = sts.size * 268435456 * (own.size * D) := by ring
        _ ≤ 64 * 268435456 * K := Nat.mul_le_mul (Nat.mul_le_mul_right _ hsz) hKD
        _ = 17179869184 * K := by norm_num

end Counts

/-- **The compiled verifier's own evaluations at the deployed brings**: at most `nS x · (131072 + 2^34 · K)` at a
run's V\* statements, where `x`'s registrations are of size at most `K`. The tables, instances and check 5's rows are
bounded by the verifier's own refusals and constants (`Tags.MAX_TABLES`, the `m ≤ 35` refusal with `HmRow.check`'s slots and
`HmRow.pin`, `fast100`'s `q < 256`); `RegSize x K` is the one condition, a deployment-size condition on public data. -/
theorem setupCalls_deployedB_le_reg (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) {K : ℕ} (hK : RegSize x K) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ nS x * (131072 + 17179869184 * K) :=
  setupCalls_deployedB_le_at rel b L IV H x ps fun xO hx => by
    obtain ⟨ms, s, rfl⟩ := mem_vstar_staged hx
    exact deployedSetup_staged_le hK

/-- **`v ≤ 2^55` at the deployed brings**, under one budget on public data: `x`'s registrations of size at most `K`
(`RegSize`) and `nS x · (131072 + 2^34 · K) ≤ 2^55`: `lamLin2_deployed_ge_dep`'s `hV`. -/
theorem setupCalls_deployedB_le_joint (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) {K : ℕ} (hK : RegSize x K) (hNK : nS x * (131072 + 17179869184 * K) ≤ 2 ^ 55) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ 2 ^ 55 :=
  (setupCalls_deployedB_le_reg rel b L IV H x ps hK).trans hNK

/-- **`v ≤ 2^55` at the deployed brings**, at `nS x ≤ 2^10` V\* parts and registrations of size at most `2047`. A
deployment's registrations are of size at least `36`: every `fast100` schedule has at least three levels, of
`q = 218, 106, 71`, so `rec-acc`, the chain of both reps' openings, has more than `2 · 395` rows (`2 + log2ceil ≥ 12`),
and a level registers `out`, `acc` and `acc_out` (`RecVStar.checkAcc`). -/
theorem setupCalls_deployedB_le_2047 (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (IV : Claims.Verifier Coin) (H : Msg → Msg) (x : Msg)
    (ps : List (Msg × Out)) (hn : nS x ≤ 2 ^ 10) (hK : RegSize x 2047) :
    (Claims.Recursion.setupCalls H (recBound rel (deployedB b) L IV)
      ((recBound rel (deployedB b) L IV).vstar x ps)).length ≤ 2 ^ 55 :=
  setupCalls_deployedB_le_joint rel b L IV H x ps hK ((Nat.mul_le_mul_right _ hn).trans (by norm_num))

/-- **Route 2C's `λ₂ ≥ 100` at the deployed brings, with its call bound**: `lamLin2_deployedB_ge` at `v = 2^55`, and
the compiled verifier's own evaluations at its V\* statements at most `2^55` (`setupCalls_deployedB_le_joint`).

What is left are deployment-size conditions on public data, read off `x`, none a hash premise or a statement about the
prover: `hn` (at most `2^16` V\* parts; `readX` refuses unless `nS x` is the length of `RecVStar.statements` at the
verifier's own inner shape), `hS` (the inner setup at most `2^56` hm96 inputs), `hO` (the scope condition `PartIn` at
each staged part, which the verifier's own `setupO` evaluates on the staged public statement), `hin` (the inner
statement's plain setup), and `hK` with `hNK` (`RegSize x K`, the registrations `x`'s parts carry, within the budget
`nS x · (131072 + 2^34 · K) ≤ 2^55`). The verifier caps neither a registration's ports nor its leaves, and holds a
part's to other files `x` carries (`--rec-rows`), so `K` is a condition on `x`, not a constant. The tables (at most
`Tags.MAX_TABLES`), each statement's instances (at most `2^28`, from the `m ≤ 35` refusal, `HmRow.check`'s slots and
`HmRow.pin`'s refusal of a range of no slots), statements with public inputs (`setupTables_counts`) and check 5's
query count (`q < 256`, `fast100`'s, a verifier constant) need no hypothesis. -/
theorem lamLin2_deployedB_ge_joint (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (H : Msg → Msg)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts',
      PartIn (staged (deployedB b).st x (pick (deployedB b).st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 16) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) {K : ℕ} (hK : RegSize x K)
    (hNK : nS x * (131072 + 17179869184 * K) ≤ 2 ^ 55) (ps : List (Msg × Out)) {n : ℕ} (hU : n ≤ 2 ^ 55) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n (2 ^ 55) ∧
    (Claims.Recursion.setupCalls H
      (recBound rel (deployedB b) L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      ((recBound rel (deployedB b) L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).vstar x
        ps)).length ≤ 2 ^ 55 :=
  ⟨lamLin2_deployedB_ge rel b L nextE colsE wE join rowIn final w decI hin hO hn hS hU,
    setupCalls_deployedB_le_joint rel b L _ H x ps hK hNK⟩

/-- **`lamLin2_deployedB_ge_joint` at `nS x ≤ 2^10` and `K = 2047`** (`setupCalls_deployedB_le_2047`). -/
theorem lamLin2_deployedB_ge_2047 (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings)
    (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (H : Msg → Msg)
    {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hin : FlockDeploy.PlainIn (xIOf x) s spec sts)
    (hO : ∀ j < nS x, ∃ i spec' sts',
      PartIn (staged (deployedB b).st x (pick (deployedB b).st x j) j) i spec' sts')
    (hn : nS x ≤ 2 ^ 10) (hS : (innerSetupIns (xIOf x)).length ≤ 2 ^ 56) (hK : RegSize x 2047)
    (ps : List (Msg × Out)) {n : ℕ} (hU : n ≤ 2 ^ 55) :
    100 ≤ Claims.Recursion.lamLin2
      ((recBoundAt rel (deployedB b) L).record (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      (Claims.Recursion.Inner.ofFlock (bdDeployed rel nextE colsE wE join rowIn final w).deploy decI)
      FlockDeploy.KS.CpDep x n (2 ^ 55) ∧
    (Claims.Recursion.setupCalls H
      (recBound rel (deployedB b) L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar)
      ((recBound rel (deployedB b) L (bdDeployed rel nextE colsE wE join rowIn final w).IVStar).vstar x
        ps)).length ≤ 2 ^ 55 :=
  ⟨lamLin2_deployedB_ge rel b L nextE colsE wE join rowIn final w decI hin hO (hn.trans (by norm_num)) hS hU,
    setupCalls_deployedB_le_2047 rel b L _ H x ps hn hK⟩

end RecDeploy
