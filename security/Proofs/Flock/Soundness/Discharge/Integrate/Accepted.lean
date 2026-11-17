module

public import Proofs.Flock.Soundness.Discharge.Integrate.Bridge
public import Proofs.Flock.Soundness.Discharge.Integrate.Sites
import all Init.Data.Array.Basic
import all Init.Data.Range.Basic
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Discharge.Exec.Setup
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
import Proofs.Flock.Soundness.Refine.Seam
import Proofs.Flock.Soundness.Refine.Setup
import Proofs.Flock.Soundness.Refine.Walk

/-! A sample verity-verify-proof accepts sets up (`sampleSetup_of_accepts`), so there the table is `stmtOf` of the
accepted statement (`tableAt_of_accepts`). -/

@[expose] public section

namespace FlockSoundness.Discharge.Integrate

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec (Inputs Accepts recordDraw accepts_one)

theorem fast100_two (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) : 2 ≤ s.levels.size := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val, decide (2 ≤ s.levels.size) = true := by
      decide +kernel
    simpa using key ⟨m, hm⟩ s hs
  · rw [exec_fast100_none m (by omega)] at hs; cases hs

theorem model_sched {m : ℕ} {s : Flock.Schedule} (hs : Flock.fast100 m = some s) :
    ∃ k0 l₀ l₁ rest ext, Accounting.fast100 m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ := by
  have h1 := fast100_schedOf m
  rw [hs] at h1
  have hm := (fast100_dims m s hs).2
  have h2 := fast100_two m s hs
  obtain ⟨l₀, l₁, rest, hl⟩ : ∃ l₀ l₁ rest, s.levels.toList.map levelOf = l₀ :: l₁ :: rest := by
    match hh : s.levels.toList.map levelOf with
    | [] =>
        have := congrArg List.length hh
        rw [List.length_map, Array.length_toList, List.length_nil] at this; omega
    | [_] =>
        have := congrArg List.length hh
        rw [List.length_map, Array.length_toList, List.length_singleton] at this; omega
    | l₀ :: l₁ :: rest => exact ⟨l₀, l₁, rest, rfl⟩
  refine ⟨s.k0, l₀, l₁, rest, s.ext, ?_⟩
  rw [← h1, Option.map_some]
  unfold schedOf
  rw [hl, hm]

theorem g_pos {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (h : HmRow.parse tags bytes tables = .ok c) (ho : c.outNet = c.unitNet) : 0 < c.g := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have hp := Layout.unit_count_pow2 P.typed hls ho
  rw [C.unit_range.2] at hp
  simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
  rcases Nat.eq_zero_or_pos c.g with h0 | h0
  · rw [h0, Nat.zero_mul] at hp; omega
  · exact h0

set_option maxHeartbeats 2000000 in
theorem setupH_n {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupH tags circuitFile publicFile coins tables draw partition program = .ok st) :
    st.n = st.pub.n ∧ st.nbl = max 3 (log2ceil (max ((st.n + st.c.g - 1) / st.c.g) 1)) := by
  unfold Stmt.setupH at h
  dsimp only at h
  simp only [throw_bind', pure_bind] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨rfl, rfl⟩

theorem fits_blocks {n g : ℕ} (hg : 0 < g) : n ≤ g * 2 ^ max 3 (log2ceil (max ((n + g - 1) / g) 1)) := by
  have hd := Nat.div_add_mod (n + g - 1) g
  have hr := Nat.mod_lt (n + g - 1) hg
  have h1 : n ≤ g * ((n + g - 1) / g) := by omega
  have h2 : (n + g - 1) / g ≤ 2 ^ max 3 (log2ceil (max ((n + g - 1) / g) 1)) :=
    calc (n + g - 1) / g ≤ max ((n + g - 1) / g) 1 := le_max_left _ _
      _ ≤ 2 ^ log2ceil (max ((n + g - 1) / g) 1) := le_two_pow_log2ceil _
      _ ≤ _ := Nat.pow_le_pow_right (by norm_num) (le_max_right _ _)
  exact h1.trans (Nat.mul_le_mul_left g h2)

variable {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}
  (ht : I.tags.typed = false) (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
include ht hc ho

theorem sampleSetup_of_setupH {st : Stmt}
    (h : Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitSample}
    (hD : Draw.UnitSample.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : SampleSetup I mPts S d, x.st = st := by
  have hcst : st.c = c := Except.ok.inj (((setupH_spec ht h).1).symm.trans hc)
  have hg : 0 < st.c.g := hcst ▸ g_pos hc ho
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_sampled h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.sampled_spec hdr
  rw [hD] at hD'
  cases hD'
  obtain ⟨hn', hnbl⟩ := setupH_n h
  have hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl := by
    rw [hnbl, ← hn, ← hn']
    exact fits_blocks hg
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := model_sched hf
  exact ⟨⟨st, h, D, hD, hS, hg, hblk, st.m, k0, l₀, l₁, rest, ext, hsch,
    by have := exec_fast100_ge hf; omega, hmPts⟩, rfl⟩

theorem sampleSetup_of_accepts (hh : I.tags.hiddenOutputs = false) (hc1 : I.count ≤ 1) {t : Exec.Transcript}
    (h : Accepts I t)
    (hd : recordDraw t.record = some d) :
    ∃ st D, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st ∧
      Draw.UnitSample.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : SampleSetup I mPts S d, x.st = st) := by
  obtain ⟨st, hst, hv⟩ := accepts_one hh hc1 h
  rw [hd] at hst
  obtain ⟨-, sch, -, -, hf, -, -⟩ := verify_one_ok hv
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_sampled hst
  obtain ⟨D, hD, -⟩ := Layout.sampled_spec hdr
  exact ⟨st, D, hst, hD, fun hS hm => sampleSetup_of_setupH ht hc ho hst hf hD hS hm⟩

theorem tableAt_of_accepts (hh : I.tags.hiddenOutputs = false) (hc1 : I.count ≤ 1) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)}
    (x₀ : SampleSetup I mPts S₀ (dj S₀)) {t : Exec.Transcript} (h : Accepts I t)
    (hd : recordDraw t.record = some (dj S))
    (hS : ∀ D, Draw.UnitSample.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
      .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program = .ok st ∧
      ∃ hwf : StmtWF st, (tableAt ht dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  obtain ⟨st, D, hst, hD, hx⟩ := sampleSetup_of_accepts ht hc ho hh hc1 h hd
  obtain ⟨x, rfl⟩ := hx (hS D hD) (hmPts _ hst)
  obtain ⟨y, hy, e⟩ := tableAt_stmtOf ht dj x₀ x
  exact ⟨x.st, hst, hy ▸ ⟨y.wf.1, e⟩⟩

end FlockSoundness.Discharge.Integrate
