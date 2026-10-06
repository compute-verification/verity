import Proofs.Flock.Soundness.Discharge.Integrate.Bridge

/-!
# A draw flock-verify accepts sets up

`tableAt` is setupH's statement at a draw with a `DrawSetup`, and unsatisfiable elsewhere. A `DrawSetup` is more than
setupH's acceptance: the draw proves the drawn units, `0 < c.g`, the proved units fit the blocks, `m` has a `fast100`
schedule of at least two levels, and the statement's link points are `mPts`. All but the first and the last follow
from a one-table acceptance:
* `0 < c.g`: `HmRow.check` takes the unit range's `c.g · c.upv` slots to be a power of two when the unit returns the
  outputs (`g_pos`);
* the blocks: setupH's `nbl` is at least `log2ceil ⌈n / c.g⌉`, and a drawn statement's `n` is the draw's proved units
  (`setupH_n`, `Layout.drawn_spec`);
* the schedule: flock-verify ran `fast100` on the table's `m` (`verify_one_ok`), and every executable `fast100` schedule
  is `Accounting.fast100`'s (`fast100_schedOf`) with at least two levels (`fast100_two`).

`drawSetup_of_accepts`: a one-table acceptance whose record's draw is `d` gives setupH's statement at `d` and the
draw `D`, and a `DrawSetup` whenever `D` proves `S` and the statement's link points are `mPts`. `tableAt_of_accepts`:
then `tableAt`'s table at `S` is `stmtOf` of the statement flock-verify accepted. The link points are setupH's
`PT_LOCAL + nbl`, a function of `|D.proved|` and `c.g` (e2e-exec's `Exec.setupH_mPts_drawn`).
-/

namespace FlockSoundness.Discharge.Integrate

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec (Inputs Accepts recordDraw accepts_one)

/-- Every executable `fast100` schedule has at least two levels. -/
theorem fast100_two (m : ℕ) (s : Flock.Schedule) (hs : Flock.fast100 m = some s) : 2 ≤ s.levels.size := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, ∀ s ∈ Flock.fast100 m.val, decide (2 ≤ s.levels.size) = true := by
      decide +kernel
    simpa using key ⟨m, hm⟩ s hs
  · rw [exec_fast100_none m (by omega)] at hs; cases hs

/-- `Accounting.fast100`'s schedule at an `m` the executable's `fast100` schedules, with its two first levels. -/
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

/-- The unit range's `c.g · c.upv` slots are a power of two when the unit returns the outputs, so `0 < c.g`. -/
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
/-- **setupH's instances and blocks**: `n` is the (drawn) public file's, and `nbl = max 3 (log2ceil ⌈n / c.g⌉)`. -/
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

/-- `n` instances fit `c.g · 2^nbl` slots when `nbl ≥ log2ceil ⌈n / g⌉`. -/
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

/-- **A `DrawSetup` from setupH's acceptance at a draw**, a `fast100` schedule of its `m`, the draw proving `S`, and its
link points `mPts`. -/
theorem drawSetup_of_setupH {st : Stmt}
    (h : Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : DrawSetup I mPts S d, x.st = st := by
  have hcst : st.c = c := Except.ok.inj (((setupH_spec ht h).1).symm.trans hc)
  have hg : 0 < st.c.g := hcst ▸ g_pos hc ho
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.drawn_spec hdr
  rw [hD] at hD'
  cases hD'
  obtain ⟨hn', hnbl⟩ := setupH_n h
  have hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl := by
    rw [hnbl, ← hn, ← hn']
    exact fits_blocks hg
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := model_sched hf
  exact ⟨⟨st, h, D, hD, hS, hg, hblk, st.m, k0, l₀, l₁, rest, ext, hsch,
    by have := exec_fast100_ge hf; omega, hmPts⟩, rfl⟩

/-- **A draw flock-verify accepts sets up.** A one-table acceptance whose record's draw is `d`: setupH accepted its
inputs at `d` (`st`), the draw decodes (`D`), and whenever `D` proves `S` and `st`'s link points are `mPts`, a
`DrawSetup` at `d` has statement `st`. -/
theorem drawSetup_of_accepts (hh : I.tags.hiddenOutputs = false) (hc1 : I.count ≤ 1) {t : Exec.Transcript}
    (h : Accepts I t)
    (hd : recordDraw t.record = some d) :
    ∃ st D, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st ∧
      Draw.UnitDraw.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : DrawSetup I mPts S d, x.st = st) := by
  obtain ⟨st, hst, hv⟩ := accepts_one hh hc1 h
  rw [hd] at hst
  obtain ⟨-, sch, -, -, hf, -, -⟩ := verify_one_ok hv
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  exact ⟨st, D, hst, hD, fun hS hm => drawSetup_of_setupH ht hc ho hst hf hD hS hm⟩

/-- **At a draw flock-verify accepts, the headline's table is `stmtOf` of the statement it accepted**: when the
session's record carries the challenger's draw `dj S`, that draw proves `S`, and setupH's statement at it has link
points `mPts`. -/
theorem tableAt_of_accepts (hh : I.tags.hiddenOutputs = false) (hc1 : I.count ≤ 1) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)}
    (x₀ : DrawSetup I mPts S₀ (dj S₀)) {t : Exec.Transcript} (h : Accepts I t)
    (hd : recordDraw t.record = some (dj S))
    (hS : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
      .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program = .ok st ∧
      ∃ hwf : StmtWF st, (tableAt ht dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  obtain ⟨st, D, hst, hD, hx⟩ := drawSetup_of_accepts ht hc ho hh hc1 h hd
  obtain ⟨x, rfl⟩ := hx (hS D hD) (hmPts _ hst)
  obtain ⟨y, hy, e⟩ := tableAt_stmtOf ht dj x₀ x
  exact ⟨x.st, hst, hy ▸ ⟨y.wf.1, e⟩⟩

end FlockSoundness.Discharge.Integrate
