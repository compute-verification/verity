module

public import Flock.HmRow
public import Proofs.Flock.Soundness.Refine.Regions
public import Proofs.Flock.Soundness.Refine.Walk
import Mathlib.Combinatorics.Matroid.Init

/-! The statement decoding, part 3: `Stmt.setupH`'s statements are well formed (`setupH_wf`). -/

@[expose] public section

set_option Elab.async false

namespace FlockSoundness.Refine

open Flock FlockLevel3

theorem list_forIn_inv {α σ : Type} (f : α → σ → V (ForInStep σ)) (I : σ → Prop)
    (hf : ∀ a s y, I s → f a s = .ok y → I y.value) :
    ∀ (L : List α) (s v : σ), I s → forIn L s f = .ok v → I v
  | [], s, v, hs, h => by
      simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h; exact h ▸ hs
  | a :: L, s, v, hs, h => by
      rw [List.forIn_cons] at h
      obtain ⟨y, hy, h⟩ := ebind_ok h
      have hI := hf a s y hs hy
      cases y with
      | done b => simp only [pure, Except.pure, Except.ok.injEq] at h; exact h ▸ hI
      | yield b => exact list_forIn_inv f I hf L b v hI h

theorem forIn_range_inv {σ : Type} (k : ℕ) (f : ℕ → σ → V (ForInStep σ)) (I : σ → Prop)
    (hf : ∀ a s y, I s → f a s = .ok y → I y.value) (s v : σ) (hs : I s) (h : forIn [:k] s f = .ok v) : I v := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  exact list_forIn_inv f I hf _ s v hs h

theorem list_forIn_each {α σ : Type} (f : α → σ → V (ForInStep σ)) (P : α → Prop)
    (hf : ∀ a s y, f a s = .ok y → P a ∧ ∃ s', y = .yield s') :
    ∀ (L : List α) (s v : σ), forIn L s f = .ok v → ∀ a ∈ L, P a
  | [], _, _, _ => by simp
  | a :: L, s, v, h => by
      rw [List.forIn_cons] at h
      obtain ⟨y, hy, h⟩ := ebind_ok h
      obtain ⟨hP, s', rfl⟩ := hf a s y hy
      intro b hb
      rcases List.mem_cons.1 hb with rfl | hb
      · exact hP
      · exact list_forIn_each f P hf L s' v h b hb

theorem ite_throw_ok {α γ : Type} {c : Prop} [Decidable c] {e : String} {f : γ → V α} {x : V α} {v : α} :
    ((if c then ((throw e : V γ) >>= f) else x) = Except.ok v) ↔ (¬c ∧ x = .ok v) := by
  by_cases hc : c
  · simp only [hc, ite_true, not_true_eq_false, false_and, iff_false]
    intro h; cases h
  · simp only [hc, ite_false, not_false_eq_true, true_and]

theorem throw_bind' {α β : Type} (e : String) (f : α → V β) : ((throw e : V α) >>= f) = throw e := rfl

theorem ite_throw_ok' {α : Type} {c : Prop} [Decidable c] {e : String} {x : V α} {v : α} :
    ((if c then (throw e : V α) else x) = .ok v) ↔ (¬c ∧ x = .ok v) := by
  by_cases hc : c
  · simp only [hc, ite_true, not_true_eq_false, false_and, iff_false]; exact throw_ne_ok e v
  · simp only [hc, ite_false, not_false_eq_true, true_and]

theorem mkRegion_wf {kl : ℕ} {name : String} {free : Array ℕ} {fixed : ℕ} {bytes : ℕ → ByteArray} {r : Region}
    (h : mkRegion kl name free fixed bytes = .ok r) : RegionWF kl r := by
  obtain ⟨hf, hnd, hsz, hlt⟩ := mkRegion_ok h
  subst hf
  exact ⟨fun b hb => hlt b (Array.mem_toList_iff.1 hb), hnd, hsz⟩

theorem allWF_push {kl : ℕ} {out : Array Region} {r : Region} (h1 : ∀ x ∈ out.toList, RegionWF kl x)
    (h2 : RegionWF kl r) : ∀ x ∈ (out.push r).toList, RegionWF kl x := by
  intro x hx
  rw [Array.toList_push, List.mem_append, List.mem_singleton] at hx
  rcases hx with hx | rfl
  · exact h1 x hx
  · exact h2

theorem outRegion_wf {c : Circuit} {pub : Public} {r : Region} (h : c.outRegion pub = .ok r) : RegionWF c.kLog r := by
  unfold Circuit.outRegion at h
  exact mkRegion_wf h

theorem hm_pin_ok {c : Circuit} {p : ℕ} (h : HmRow.pin c = .ok p) :
    p < 2 ^ c.kLog ∧ ∃ r ∈ c.ranges.toList, r.kind ≠ Kind.mask := by
  unfold HmRow.pin at h
  dsimp only at h
  split at h
  · rename_i r hr
    have hmem : r ∈ c.ranges.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hr)
    have hk : r.kind ≠ Kind.mask := by
      intro hk; have := Array.find?_some hr; rw [hk] at this; exact absurd this (by decide)
    repeat' (first | (obtain ⟨_, hx, h⟩ := ebind_ok h; try (cases hx; done)) | (split at h))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨by omega, r, hmem, hk⟩
  · obtain ⟨_, hx, _⟩ := ebind_ok h
    exact absurd hx (throw_ne_ok _ _)

theorem hm_check_slots {c : Circuit} {tags : Tags} {ls : Lean.Json} (h : HmRow.check c tags ls = .ok ()) :
    ∀ r ∈ c.ranges.toList, 7 ≤ r.slotLog ∧ r.slotLog ≤ c.kLog := by
  unfold HmRow.check at h
  dsimp only at h
  obtain ⟨_, hloop, -⟩ := ebind_ok h
  clear h
  rw [← Array.forIn_toList] at hloop
  refine list_forIn_each _ (fun r => 7 ≤ r.slotLog ∧ r.slotLog ≤ c.kLog) ?_ _ _ _ hloop
  clear hloop
  intro r s y hy
  repeat' (first | (obtain ⟨_, hx, hy⟩ := ebind_ok hy; try (cases hx; done)) | (split at hy))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    refine ⟨?_, _, rfl⟩
    simp only [Bool.or_eq_true, decide_eq_true_eq, not_or, not_lt] at *
    omega

theorem hm_parse_check_any {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {t : Option Typed.Template} {pre : Array (String × Net)} {c : Circuit}
    (h : HmRow.parse tags bytes tables t pre = .ok c) : ∃ ls, HmRow.check c tags ls = .ok () := by
  revert c
  show Ret (HmRow.parse tags bytes tables t pre) fun c => ∃ ls, HmRow.check c tags ls = .ok ()
  rw [HmRow.parse.eq_def]
  ret_walk [HmRow.check]
  all_goals exact ⟨_, by with_reducible assumption⟩

theorem hm_parse_check_none {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} {c : Circuit} (h : HmRow.parse tags bytes tables none pre = .ok c) :
    ∃ ls, HmRow.check c tags ls = .ok () :=
  hm_parse_check_any h

theorem hm_parse_check_some {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {t : Typed.Template} {c : Circuit} (h : HmRow.parse tags bytes tables (some t) #[] = .ok c) :
    ∃ ls, HmRow.check c tags ls = .ok () :=
  hm_parse_check_any h

theorem hm_parse_checkPre {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {pre : Array (String × Net)} {c : Circuit} (h : HmRow.parse tags bytes tables (pre := pre) = .ok c) :
    ∃ ls, HmRow.check c tags ls = .ok () :=
  hm_parse_check_none h

theorem hm_parse_check {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) : ∃ ls, HmRow.check c tags ls = .ok () :=
  hm_parse_checkPre h

theorem check_cls (c : Circuit) (tags : Tags) (ls : Lean.Json) (x : Option _) :
    HmRow.check { c with cls := x } tags ls = HmRow.check c tags ls := rfl

theorem hm_parseTyped_check {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (h : HmRow.parseTyped tags bytes tables = .ok c) : ∃ ls, HmRow.check c tags ls = .ok () := by
  unfold HmRow.parseTyped at h
  obtain ⟨⟨k, r⟩, -, h⟩ := ebind_ok h
  dsimp only at h
  cases r with
  | inr t =>
      obtain ⟨c', hc', h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨ls, hls⟩ := hm_parse_check_some hc'
      exact ⟨ls, (check_cls c' tags ls _).trans hls⟩
  | inl u =>
      obtain ⟨c', hc', h⟩ := ebind_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      obtain ⟨ls, hls⟩ := hm_parse_check_none hc'
      exact ⟨ls, (check_cls c' tags ls _).trans hls⟩

set_option maxHeartbeats 2000000 in
theorem hm_regions_wf {c : Circuit} {pub : Public} {regs : Array Region} (h : HmRow.regions c pub = .ok regs) :
    ∀ x ∈ regs.toList, RegionWF c.kLog x := by
  unfold HmRow.regions at h
  dsimp only at h
  try simp only [throw_bind', pure_bind] at h
  obtain ⟨out, hloop, h⟩ := ebind_ok h
  have hI : ∀ x ∈ out.toList, RegionWF c.kLog x := by
    refine forIn_range_inv _ _ (fun out => ∀ x ∈ out.toList, RegionWF c.kLog x) ?_ #[] _ (by simp) hloop
    clear hloop h
    intro ri out y hI h
    repeat' walk_step
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      show ∀ x ∈ _, _
      repeat (first | refine allWF_push ?_ ?_ | exact hI | exact mkRegion_wf (by assumption))
  clear hloop
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    repeat (first | refine allWF_push ?_ ?_ | exact hI | exact mkRegion_wf (by assumption))

theorem wf_of_facts {st : Stmt} (hslots : ∀ r ∈ st.c.ranges.toList, 7 ≤ r.slotLog ∧ r.slotLog ≤ st.c.kLog)
    (hpin : st.pin < 2 ^ st.c.kLog ∧ ∃ r ∈ st.c.ranges.toList, r.kind ≠ Kind.mask) (hm : st.m = st.c.kLog + st.nbl) :
    StmtWF st := by
  obtain ⟨hp, r, hr, -⟩ := hpin
  have := hslots r hr
  exact ⟨by omega, by omega, hp, fun r hr => ⟨by have := hslots r hr; omega, (hslots r hr).2⟩⟩

set_option maxHeartbeats 2000000 in
theorem setupH_wf {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st) :
    StmtWF st ∧ RegionsWF st := by
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
    exact ⟨wf_of_facts (hm_check_slots hc) (hm_pin_ok (by with_reducible assumption)) rfl, rfl, rfl,
      hm_regions_wf (by with_reducible assumption)⟩

theorem filterMap_fin_nodup (kl : ℕ) (L : List ℕ) (h : L.Nodup) :
    (L.filterMap fun b => if h : b < kl then some (⟨b, h⟩ : Fin kl) else none).Nodup :=
  List.Nodup.filterMap (fun a a' b hb hb' => by
    split at hb <;> split at hb' <;> (try simp_all)
    exact congrArg Fin.val (hb.trans hb'.symm)) h

theorem stmtOf_linkLayout (st : Stmt) (hwf : StmtWF st) (hr : RegionsWF st) :
    (stmtOf st hwf (regionsOf st)).LinkLayout PT_LOCAL st.mPts := by
  refine ⟨fun R hR => ?_, ?_⟩
  · obtain ⟨reg, hreg, rfl⟩ := List.mem_map.1 hR
    have hw := hr.regions reg hreg
    exact ⟨filterMap_fin_nodup _ _ hw.nodup,
      (filterMap_fin_length _ _ hw.free_lt).trans_le (by rw [Array.length_toList]; exact hw.free_size)⟩
  · show PT_LOCAL + (st.m - st.c.kLog) ≤ st.mPts
    rw [hr.mPts, hr.hm]; omega

theorem hm_check_mask {c : Flock.Circuit} {tags : Tags} {ls : Lean.Json} (h : HmRow.check c tags ls = .ok ()) :
    ∀ r ∈ c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ c.kLog := by
  unfold HmRow.check at h
  dsimp only at h
  obtain ⟨_, hloop, -⟩ := ebind_ok h
  clear h
  rw [← Array.forIn_toList] at hloop
  refine list_forIn_each _ (fun r => r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ c.kLog)
    ?_ _ _ _ hloop
  clear hloop
  intro r s y hy
  have hp : 1 ≤ nextPow2 r.count := Nat.one_le_two_pow
  repeat' (first | (obtain ⟨_, hx, hy⟩ := ebind_ok hy; try (cases hx)) | (split at hy))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at hy
    subst hy
    refine ⟨fun hk => ?_, _, rfl⟩
    simp only [hk, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, not_or, not_lt, not_and] at *
    first
    | omega
    | (constructor; omega; nlinarith [Nat.one_le_two_pow (n := r.slotLog)])
    | simp_all

set_option maxHeartbeats 2000000 in
theorem setupH_mask {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st) :
    (∀ r ∈ st.c.ranges.toList, r.kind = Kind.mask → 7 ≤ r.slotLog ∧ r.start + 2 ^ r.slotLog ≤ 2 ^ st.c.kLog) ∧
      st.m = st.c.kLog + st.nbl := by
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
    exact ⟨hm_check_mask hc, rfl⟩

end FlockSoundness.Refine
