import Definitions.Core.Accel.Witness
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.NNRat
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# The non-vacuity witness, proved

`witness : Witness`.

The run of a plan issues exactly the plan's instructions, tick by tick (`run_plan`), since each thread's text is one
instruction and each thread steps once, and its stores are the plan's instructions applied in order (`run_mem`). The
kernel's length is within the bound (`kernel_length_le`). Every tick is within every resource's rate but
`tensorLong`'s; for `tensorLong`, 21 consecutive ticks hold an idle one (`block_long`), so a window of `L` ticks uses at
most `524,288 (L - ⌊L/21⌋)` units (`window_long`), within `tensorLong`'s 540,672 a tick, and within the 1830 MHz cap's
499,712 with a backlog of a tick, should a clock check on the H200 reinstate that cap: the run meets G2's rates
(`kernel_rate`).

The values: the loads copy `A` and `B` to shared memory (`loads_exec`); the idle ticks do nothing, so the MMA ticks run
the MMAs in order (`mmas_ticks`); MMA `κ · blocks n + e` adds slice `κ`'s products to block `e`'s accumulators and
touches nothing else (`mma_reg`, `mma_other`), so after `κ` slices every accumulator holds its first `32κ` products
(`MmaInv`, `mma_slices`) and after all `n/32` its whole dot, wrapped to a word (`mmas_exec`); the stores copy the
accumulators to `C` (`stores_exec`).
-/

open scoped NNRat

namespace Verity.Accel.H200.Resident

section Run

variable {ι : Type} {m : Machine ι} {Θ : Type} [DecidableEq Θ]

theorem step_spec (P : m.Prog Θ) (σ : m.State Θ) (θ : Θ) (h : σ.pc θ = 0) :
    (Machine.step P σ θ).2 = (P θ)[0]? ∧ ∀ θ', θ' ≠ θ → (Machine.step P σ θ).1.pc θ' = σ.pc θ' := by
  unfold Machine.step
  rw [h]
  split <;> rename_i heq <;> refine ⟨by simp [heq], fun θ' hθ' => ?_⟩ <;> simp [Function.update_of_ne hθ']

theorem tick_foldl (P : m.Prog Θ) : ∀ (θs : List Θ) (σ : m.State Θ) (l : List m.Instr), θs.Nodup →
    (∀ θ ∈ θs, σ.pc θ = 0) →
    (θs.foldl (fun acc θ => let r := Machine.step P acc.1 θ; (r.1, acc.2 ++ r.2.toList)) (σ, l)).2 =
        l ++ θs.filterMap (fun θ => (P θ)[0]?) ∧
      ∀ θ ∉ θs,
        (θs.foldl (fun acc θ => let r := Machine.step P acc.1 θ; (r.1, acc.2 ++ r.2.toList)) (σ, l)).1.pc θ =
          σ.pc θ
  | [], σ, l, _, _ => by simp
  | θ :: θs, σ, l, hnd, hpc => by
    obtain ⟨h1, h2⟩ := step_spec P σ θ (hpc θ (by simp))
    have hnd' := List.nodup_cons.mp hnd
    have ih := tick_foldl P θs (Machine.step P σ θ).1 (l ++ (Machine.step P σ θ).2.toList) hnd'.2
      (fun θ' hθ' => by
        rw [h2 θ' (fun e => hnd'.1 (e ▸ hθ'))]
        exact hpc θ' (by simp [hθ']))
    simp only [List.foldl_cons]
    refine ⟨?_, fun θ' hθ' => ?_⟩
    · rw [ih.1, h1]
      rcases h : (P θ)[0]? with _ | i <;> simp [h]
    · rw [ih.2 θ' (fun e => hθ' (by simp [e]))]
      exact h2 θ' (fun e => hθ' (by simp [e]))

/-- An instruction's effect on the stores (a branch has none). -/
def instrExec : m.Instr → m.Mem → m.Mem
  | .op o src dst pred, M => Machine.execOp o src dst pred M
  | .bra _ _, M => M

theorem step_mem (P : m.Prog Θ) (σ : m.State Θ) (θ : Θ) (h : σ.pc θ = 0) :
    (Machine.step P σ θ).1.mem = ((P θ)[0]?.toList).foldl (fun M i => instrExec i M) σ.mem := by
  unfold Machine.step
  rw [h]
  split <;> rename_i heq <;> simp [heq, instrExec]

theorem tick_mem (P : m.Prog Θ) : ∀ (θs : List Θ) (σ : m.State Θ) (l : List m.Instr), θs.Nodup →
    (∀ θ ∈ θs, σ.pc θ = 0) →
    (θs.foldl (fun acc θ => let r := Machine.step P acc.1 θ; (r.1, acc.2 ++ r.2.toList)) (σ, l)).1.mem =
      (θs.filterMap fun θ => (P θ)[0]?).foldl (fun M i => instrExec i M) σ.mem
  | [], σ, l, _, _ => by simp
  | θ :: θs, σ, l, hnd, hpc => by
    obtain ⟨_, h2⟩ := step_spec P σ θ (hpc θ (by simp))
    have hnd' := List.nodup_cons.mp hnd
    have ih := tick_mem P θs (Machine.step P σ θ).1 (l ++ (Machine.step P σ θ).2.toList) hnd'.2
      (fun θ' hθ' => by
        rw [h2 θ' (fun e => hnd'.1 (e ▸ hθ'))]
        exact hpc θ' (by simp [hθ']))
    simp only [List.foldl_cons]
    rw [ih, step_mem P σ θ (hpc θ (by simp))]
    rcases hθ : (P θ)[0]? with _ | i <;> simp [hθ]

end Run

/-- The schedule of a plan's ticks from tick `t₀` on. -/
def schedFrom : ℕ → List (List Ins) → List (List (ℕ × ℕ))
  | _, [] => []
  | t₀, is :: rest => ((List.range is.length).map fun s => (t₀, s)) :: schedFrom (t₀ + 1) rest

theorem mapIdx_schedFrom : ∀ (plan : List (List Ins)) (t₀ : ℕ),
    plan.mapIdx (fun t is => (List.range is.length).map fun s => (t₀ + t, s)) = schedFrom t₀ plan
  | [], _ => rfl
  | is :: rest, t₀ => by
    rw [List.mapIdx_cons, schedFrom, ← mapIdx_schedFrom rest (t₀ + 1)]
    simp only [Nat.add_zero, Nat.add_assoc, Nat.add_comm 1]

theorem sched_eq (plan : List (List Ins)) : sched plan = schedFrom 0 plan := by
  rw [← mapIdx_schedFrom plan 0]
  simp [sched]

theorem filterMap_range_getElem {α β : Type} (g : α → β) : ∀ is : List α,
    (List.range is.length).filterMap (fun s => is[s]?.map g) = is.map g
  | [] => rfl
  | a :: is => by
    rw [List.length_cons, List.range_succ_eq_map, List.filterMap_cons, List.filterMap_map]
    simpa [Function.comp_def] using filterMap_range_getElem g is

variable {ι : Type} {x : Open} {cap : ι → ℕ} {oracle : ℕ}

theorem run_schedFrom (plan : List (List Ins)) : ∀ (rest : List (List Ins)) (t₀ : ℕ)
    (σ : (machine x cap oracle).State (ℕ × ℕ)), plan.drop t₀ = rest → (∀ θ, t₀ ≤ θ.1 → σ.pc θ = 0) →
    (Machine.run (prog plan) σ (schedFrom t₀ rest)).2 = rest.map (List.map Ins.instr)
  | [], _, _, _, _ => rfl
  | is :: rest, t₀, σ, hd, hpc => by
    have hget : plan.getD t₀ [] = is := by
      have := congrArg (·[0]?) hd
      simp only [List.getElem?_drop, Nat.add_zero, List.getElem?_cons_zero] at this
      simp [List.getD_eq_getElem?_getD, this]
    have hd' : plan.drop (t₀ + 1) = rest := by
      rw [← List.drop_drop, hd]; rfl
    have hnd : ((List.range is.length).map fun s => (t₀, s)).Nodup :=
      (List.nodup_range).map (fun a b h => (Prod.mk.inj h).2)
    obtain ⟨h1, h2⟩ := tick_foldl (prog plan) _ σ [] hnd (fun θ hθ => by
      obtain ⟨s, _, rfl⟩ := List.mem_map.mp hθ
      exact hpc _ le_rfl)
    simp only [schedFrom, Machine.run, Machine.tick, List.map_cons]
    rw [h1, run_schedFrom plan rest (t₀ + 1) _ hd' (fun θ hθ => by
      rw [h2 θ (fun hm => by
        obtain ⟨s, _, rfl⟩ := List.mem_map.mp hm
        exact absurd hθ (by simp))]
      exact hpc θ (by omega))]
    congr 1
    have hfun : ((fun θ => (prog (x := x) (cap := cap) (oracle := oracle) plan θ)[0]?) ∘ fun s => (t₀, s)) =
        fun s => is[s]?.map Ins.instr := by
      funext s
      simp only [Function.comp, prog, hget]
      cases is[s]? <;> rfl
    rw [List.nil_append, List.filterMap_map, hfun, filterMap_range_getElem]

theorem tick_filterMap (plan : List (List Ins)) (t₀ : ℕ) (is : List Ins) (hget : plan.getD t₀ [] = is) :
    ((List.range is.length).map fun s => (t₀, s)).filterMap
      (fun θ => (prog (x := x) (cap := cap) (oracle := oracle) plan θ)[0]?) = is.map Ins.instr := by
  have hfun : ((fun θ => (prog (x := x) (cap := cap) (oracle := oracle) plan θ)[0]?) ∘ fun s => (t₀, s)) =
      fun s => is[s]?.map Ins.instr := by
    funext s
    simp only [Function.comp, prog, hget]
    cases is[s]? <;> rfl
  rw [List.filterMap_map, hfun, filterMap_range_getElem]

/-- A plan's effect on the stores: its ticks in order, each tick's instructions in order. -/
def execPlan (plan : List (List Ins)) (M : (machine x cap oracle).Mem) : (machine x cap oracle).Mem :=
  plan.foldl (fun M is => is.foldl (fun M a => instrExec a.instr M) M) M

theorem run_mem_schedFrom (plan : List (List Ins)) : ∀ (rest : List (List Ins)) (t₀ : ℕ)
    (σ : (machine x cap oracle).State (ℕ × ℕ)), plan.drop t₀ = rest → (∀ θ, t₀ ≤ θ.1 → σ.pc θ = 0) →
    (Machine.run (prog plan) σ (schedFrom t₀ rest)).1.mem = execPlan rest σ.mem
  | [], _, _, _, _ => rfl
  | is :: rest, t₀, σ, hd, hpc => by
    have hget : plan.getD t₀ [] = is := by
      have := congrArg (·[0]?) hd
      simp only [List.getElem?_drop, Nat.add_zero, List.getElem?_cons_zero] at this
      simp [List.getD_eq_getElem?_getD, this]
    have hd' : plan.drop (t₀ + 1) = rest := by
      rw [← List.drop_drop, hd]; rfl
    have hnd : ((List.range is.length).map fun s => (t₀, s)).Nodup :=
      (List.nodup_range).map (fun a b h => (Prod.mk.inj h).2)
    have hpc0 : ∀ θ ∈ (List.range is.length).map (fun s => (t₀, s)), σ.pc θ = 0 := fun θ hθ => by
      obtain ⟨s, _, rfl⟩ := List.mem_map.mp hθ
      exact hpc _ le_rfl
    obtain ⟨_, h2⟩ := tick_foldl (prog plan) _ σ [] hnd hpc0
    have h3 := tick_mem (prog plan) _ σ [] hnd hpc0
    simp only [schedFrom, Machine.run, Machine.tick]
    rw [run_mem_schedFrom plan rest (t₀ + 1) _ hd' (fun θ hθ => by
      rw [h2 θ (fun hm => by
        obtain ⟨s, _, rfl⟩ := List.mem_map.mp hm
        exact absurd hθ (by simp))]
      exact hpc θ (by omega)), h3, tick_filterMap plan t₀ is hget, List.foldl_map]
    rfl

theorem run_mem (plan : List (List Ins)) (σ : (machine x cap oracle).State (ℕ × ℕ)) (hpc : ∀ θ, σ.pc θ = 0) :
    (Machine.run (prog plan) σ (sched plan)).1.mem = execPlan plan σ.mem := by
  rw [sched_eq]
  exact run_mem_schedFrom plan plan 0 σ List.drop_zero (fun θ _ => hpc θ)

theorem run_plan (plan : List (List Ins)) (σ : (machine x cap oracle).State (ℕ × ℕ)) (hpc : ∀ θ, σ.pc θ = 0) :
    (Machine.run (prog plan) σ (sched plan)).2 = plan.map (List.map Ins.instr) := by
  rw [sched_eq]
  exact run_schedFrom plan plan 0 σ List.drop_zero (fun θ _ => hpc θ)

/-! ## Length -/

theorem blocks_mul (n : ℕ) (h : 256 ∣ n) : mmas n * 524288 = n ^ 3 := by
  obtain ⟨q, rfl⟩ := h
  simp only [mmas, blocks]
  rw [show 256 * q / 32 = 8 * q by omega, show 256 * q / 64 = 4 * q by omega, show 256 * q / 256 = q by omega]
  ring

theorem kernel_length (n : ℕ) :
    (kernel n).length = 2 * n + (mmas n + mmas n / 20) + (n * n / 4 + 150) / 151 := by
  simp [kernel, Nat.add_assoc]

theorem kernel_length_le (n : ℕ) (h : 256 ∣ n) :
    ((sched (kernel n)).length : ℚ≥0) ≤
      2 * (n : ℚ≥0) + 21 * (n : ℚ≥0) ^ 3 / 10485760 + (n : ℚ≥0) ^ 2 / 604 + 1 := by
  have h3 : (((mmas n + mmas n / 20 : ℕ)) : ℚ≥0) ≤ 21 * (n : ℚ≥0) ^ 3 / 10485760 := by
    rw [le_div_iff₀ (by norm_num)]
    have hb := blocks_mul n h
    have : (mmas n + mmas n / 20) * 10485760 ≤ 21 * n ^ 3 := by rw [← hb]; omega
    exact_mod_cast this
  have hs : (n * n / 4 + 150) / 151 * 604 ≤ n * n + 604 := by
    generalize n * n = N
    omega
  have hs' : (((n * n / 4 + 150) / 151 : ℕ) : ℚ≥0) ≤ (n : ℚ≥0) ^ 2 / 604 + 1 := by
    have : (((n * n / 4 + 150) / 151 : ℕ) : ℚ≥0) * 604 ≤ (n : ℚ≥0) ^ 2 + 604 := by
      have := (Nat.cast_le (α := ℚ≥0)).mpr hs
      push_cast at this
      rw [sq]
      exact this
    rw [div_add_one (by norm_num), le_div_iff₀ (by positivity)]
    exact this
  have hl : ((sched (kernel n)).length : ℚ≥0) =
      2 * (n : ℚ≥0) + ((mmas n + mmas n / 20 : ℕ) : ℚ≥0) + (((n * n / 4 + 150) / 151 : ℕ) : ℚ≥0) := by
    rw [sched, List.length_mapIdx, kernel_length]
    push_cast
    ring
  rw [hl]
  have := add_le_add (add_le_add (le_refl (2 * (n : ℚ≥0))) h3) hs'
  simpa only [add_assoc] using this

/-! ## Rate -/

/-- The units of resource `r` a tick's instructions use. -/
def tickUse (x : Open) (is : List Ins) (r : Res) : ℚ≥0 :=
  (is.map fun a => opUse (card.weightOf x) (card.scale x) x.madHi a.1.1 r).sum

theorem usage_plan (plan : List (List Ins)) (t : ℕ) (r : Res) :
    Machine.usage (m := machine x cap oracle) (plan.map (List.map Ins.instr)) t r =
      tickUse x (plan.getD t []) r := by
  simp only [Machine.usage, tickUse, List.getD_eq_getElem?_getD, List.getElem?_map]
  cases plan[t]? <;> simp only [Option.map_none, Option.map_some, Option.getD_none, Option.getD_some,
    List.map_map] <;> rfl

theorem dram_ge : (604 : ℚ≥0) ≤ ρ .dram := by
  rw [ρ, le_div_iff₀ (by norm_num)]
  norm_num

theorem inv32_le : (32 : ℚ≥0)⁻¹ ≤ ρ .issue := by
  rw [← one_div, div_le_iff₀ (by norm_num)]
  norm_num [ρ]

theorem smem_ge : (512 : ℚ≥0) ≤ ρ .smem := by
  norm_num [ρ]

theorem load_use (n c : ℕ) (hn : n ≤ 2048) (r : Res) : tickUse x [load n c] r ≤ ρ r := by
  have h4 : ((n / 4 : ℕ) : ℚ≥0) ≤ 512 := by exact_mod_cast (by omega : n / 4 ≤ 512)
  unfold load
  split <;> cases r <;> simp [tickUse, copyN, opUse, port, rd] <;>
    first
    | exact inv32_le
    | exact h4.trans ((by norm_num : (512 : ℚ≥0) ≤ 604).trans dram_ge)
    | exact h4.trans smem_ge

theorem mma_use (n p : ℕ) (r : Res) (hr : r ≠ .tensorLong) : tickUse x [mma n p] r ≤ ρ r := by
  have ha : packedWords 64 (Fmt.s8.rowLen 32) Fmt.s8.bits = 512 := by decide
  have hb : packedWords 256 (Fmt.s8.rowLen 32) Fmt.s8.bits = 2048 := by decide
  cases r <;> simp [tickUse, mma, mmaOp, opUse, port, rd, ρ, Card.weightOf, card, weight, Shape.wa, Shape.wb, ha, hb] at hr ⊢ <;>
    norm_num

theorem mma_long (n p : ℕ) : tickUse x [mma n p] .tensorLong = 524288 := by
  simp [tickUse, mma, mmaOp, opUse, port, rd, Card.weightOf, card, weight]
  norm_num

theorem mmaTick_use (n j : ℕ) (r : Res) (hr : r ≠ .tensorLong) : tickUse x (mmaTick n j) r ≤ ρ r := by
  unfold mmaTick
  split
  · simp [tickUse]
  · exact mma_use n _ r hr

theorem mmaTick_long (n j : ℕ) :
    tickUse x (mmaTick n j) .tensorLong = if j % 21 = 20 then 0 else 524288 := by
  unfold mmaTick
  split
  · simp [tickUse]
  · exact mma_long n _

theorem store_tick (n u : ℕ) (r : Res) : tickUse x ((chunk (n * n / 4) 151 u).map (store n)) r =
    (chunk (n * n / 4) 151 u).length * opUse (card.weightOf x) (card.scale x) x.madHi (.copy .reg .dram 4) r := by
  simp only [tickUse, List.map_map]
  rw [show ((fun a : Ins => opUse (card.weightOf x) (card.scale x) x.madHi a.1.1 r) ∘ store n) =
    fun _ => opUse (card.weightOf x) (card.scale x) x.madHi (.copy .reg .dram 4) r from rfl,
    List.map_const', List.sum_replicate, nsmul_eq_mul]

theorem store_use (n u : ℕ) (r : Res) : tickUse x ((chunk (n * n / 4) 151 u).map (store n)) r ≤ ρ r := by
  have hc : ((chunk (n * n / 4) 151 u).length : ℚ≥0) ≤ 151 := by
    exact_mod_cast (by simp [chunk] : (chunk (n * n / 4) 151 u).length ≤ 151)
  have h := store_tick (x := x) n u r
  rw [h]
  cases r <;> simp [opUse, port, rd]
  · rw [← one_div, ← mul_div_assoc, mul_one, div_le_iff₀ (by norm_num)]
    exact hc.trans (by norm_num [ρ])
  · exact (mul_le_mul_of_nonneg_right hc (by norm_num)).trans ((by norm_num : (151 * 4 : ℚ≥0) ≤ 604).trans dram_ge)

theorem kernel_tick_use (n : ℕ) (hn : n ≤ 2048) (t : ℕ) (r : Res) (hr : r ≠ .tensorLong) :
    tickUse x ((kernel n).getD t []) r ≤ ρ r := by
  rcases h : (kernel n)[t]? with _ | is
  · simp [List.getD_eq_getElem?_getD, h, tickUse]
  · rw [List.getD_eq_getElem?_getD, h, Option.getD_some]
    have hm := List.mem_of_getElem? h
    simp only [kernel, List.mem_append, List.mem_map, List.mem_range] at hm
    rcases hm with (⟨c, _, rfl⟩ | ⟨j, _, rfl⟩) | ⟨u, _, rfl⟩
    · exact load_use n c hn r
    · exact mmaTick_use n j r hr
    · exact store_use n u r

theorem load_long (n c : ℕ) : tickUse x [load n c] .tensorLong = 0 := by
  unfold load; split <;> simp [tickUse, copyN, opUse, port, rd]

theorem store_long (n u : ℕ) : tickUse x ((chunk (n * n / 4) 151 u).map (store n)) .tensorLong = 0 := by
  rw [store_tick]; simp [opUse, port, rd]

/-- The kernel's `tensorLong` units at tick `t`: none at the MMAs' idle ticks (`t + 19n ≡ 20` mod 21, as
`t - 2n ≡ 20` there), at most a `wgmma`'s elsewhere. -/
theorem kernel_long (n t : ℕ) :
    tickUse x ((kernel n).getD t []) .tensorLong ≤ if (t + 19 * n) % 21 = 20 then 0 else 524288 := by
  simp only [kernel, List.getD_eq_getElem?_getD]
  by_cases h1 : t < 2 * n
  · rw [List.getElem?_append_left (by simp; omega), List.getElem?_append_left (by simp; omega)]
    simp [h1, load_long]
  · by_cases h2 : t < 2 * n + (mmas n + mmas n / 20)
    · rw [List.getElem?_append_left (by simp; omega), List.getElem?_append_right (by simp; omega)]
      simp only [List.length_map, List.length_range, List.getElem?_map, List.getElem?_range (by omega : t - 2 * n < mmas n + mmas n / 20),
        Option.map_some, Option.getD_some, mmaTick_long]
      have : (t - 2 * n) % 21 = 20 ↔ (t + 19 * n) % 21 = 20 := by omega
      simp only [this, le_refl]
    · rw [List.getElem?_append_right (by simp; omega)]
      simp only [List.length_append, List.length_map, List.length_range, List.getElem?_map]
      rcases (List.range ((n * n / 4 + 150) / 151))[t - (2 * n + (mmas n + mmas n / 20))]? with _ | u
      · simp [tickUse]
      · simp [store_long]

/-- Any 21 consecutive ticks hold one where `(t + b) % 21 = 20`. -/
theorem block_long (a b : ℕ) :
    ∑ t ∈ Finset.Ico a (a + 21), (if (t + b) % 21 = 20 then (0 : ℚ≥0) else 524288) ≤ 20 * 524288 := by
  set i₀ := a + (41 - (a + b) % 21) % 21
  have hi : i₀ ∈ Finset.Ico a (a + 21) := by simp only [Finset.mem_Ico]; omega
  have h0 : (i₀ + b) % 21 = 20 := by omega
  rw [← Finset.add_sum_erase _ _ hi, ite_eq_left_iff.mpr (fun h => absurd h0 h), zero_add]
  calc ∑ t ∈ (Finset.Ico a (a + 21)).erase i₀, (if (t + b) % 21 = 20 then (0 : ℚ≥0) else 524288)
      ≤ ((Finset.Ico a (a + 21)).erase i₀).card • (524288 : ℚ≥0) :=
        Finset.sum_le_card_nsmul _ _ _ fun t _ => by split <;> norm_num
    _ = 20 * 524288 := by rw [Finset.card_erase_of_mem hi]; simp

/-- A window of `L` ticks holds at most `L - L/21` ticks off the residue. -/
theorem window_long (b : ℕ) : ∀ L a : ℕ,
    ∑ t ∈ Finset.Ico a (a + L), (if (t + b) % 21 = 20 then (0 : ℚ≥0) else 524288) ≤
      524288 * ((L - L / 21 : ℕ) : ℚ≥0)
  | L, a => by
    by_cases hL : L < 21
    · calc _ ≤ (Finset.Ico a (a + L)).card • (524288 : ℚ≥0) :=
            Finset.sum_le_card_nsmul _ _ _ fun t _ => by split <;> norm_num
        _ = 524288 * ((L - L / 21 : ℕ) : ℚ≥0) := by
            rw [Nat.card_Ico, show a + L - a = L by omega, show L - L / 21 = L by omega, nsmul_eq_mul, mul_comm]
    · have ih := window_long b (L - 21) (a + 21)
      rw [← Finset.sum_Ico_consecutive _ (by omega : a ≤ a + 21) (by omega : a + 21 ≤ a + L),
        show a + L = a + 21 + (L - 21) by omega]
      calc _ ≤ 20 * 524288 + 524288 * ((L - 21 - (L - 21) / 21 : ℕ) : ℚ≥0) := add_le_add (block_long a b) ih
        _ = 524288 * ((L - L / 21 : ℕ) : ℚ≥0) := by
            rw [show L - L / 21 = 20 + (L - 21 - (L - 21) / 21) by omega]
            push_cast
            ring
termination_by L => L

/-- **G2's rates**: every tick of the kernel is within every resource's rate but `tensorLong`'s, and every window
is within `tensorLong`'s with a tick of backlog. -/
theorem kernel_rate (n : ℕ) (hn : n ≤ 2048) (hs : 1 ≤ x.slack) (σ : (machine x cap oracle).State (ℕ × ℕ))
    (hpc : ∀ θ, σ.pc θ = 0) :
    Machine.Rate (Machine.run (prog (kernel n)) σ (sched (kernel n))).2 := by
  intro w (r : Res)
  rw [run_plan _ σ hpc]
  by_cases hr : r = .tensorLong
  · subst hr
    calc ∑ t ∈ w.ticks, Machine.usage (m := machine x cap oracle) ((kernel n).map (List.map Ins.instr)) t .tensorLong
        ≤ ∑ t ∈ w.ticks, (if (t + 19 * n) % 21 = 20 then (0 : ℚ≥0) else 524288) :=
          Finset.sum_le_sum fun t _ => by rw [usage_plan]; exact kernel_long n t
      _ ≤ 524288 * ((w.len - w.len / 21 : ℕ) : ℚ≥0) := window_long (19 * n) w.len w.start
      _ ≤ ρ .tensorLong * ((w.len + x.slack : ℕ) : ℚ≥0) := by
          have h : 524288 * (w.len - w.len / 21) ≤ 540672 * (w.len + x.slack) := by omega
          rw [show ρ .tensorLong = 540672 by norm_num [ρ]]
          exact_mod_cast h
  · calc ∑ t ∈ w.ticks, Machine.usage (m := machine x cap oracle) ((kernel n).map (List.map Ins.instr)) t r
        ≤ ∑ _t ∈ w.ticks, ρ r :=
          Finset.sum_le_sum fun t _ => by rw [usage_plan]; exact kernel_tick_use n hn t r hr
      _ = ρ r * (w.len : ℚ≥0) := by simp [Accounting.Window.ticks, mul_comm]
      _ ≤ ρ r * ((w.len + x.slack : ℕ) : ℚ≥0) :=
          mul_le_mul_of_nonneg_left (by exact_mod_cast Nat.le_add_right _ _) zero_le

/-! ## Code -/

theorem sum_range_getD {β : Type} [AddCommMonoid β] (f : List Ins → β) (hf : f [] = 0) :
    ∀ plan : List (List Ins), ∑ t ∈ Finset.range plan.length, f (plan.getD t []) = (plan.map f).sum
  | [] => by simp
  | is :: rest => by
    rw [List.length_cons, Finset.sum_range_succ', List.map_cons, List.sum_cons, add_comm]
    simp only [List.getD_cons_succ, List.getD_cons_zero, sum_range_getD f hf rest]

theorem schedFrom_map_length : ∀ (plan : List (List Ins)) (t₀ : ℕ),
    (schedFrom t₀ plan).map List.length = plan.map List.length
  | [], _ => rfl
  | is :: rest, t₀ => by simp [schedFrom, schedFrom_map_length rest]

theorem code_le (plan : List (List Ins)) :
    Machine.code (prog (x := x) (cap := cap) (oracle := oracle) plan) (sched plan) ≤ (plan.map List.length).sum := by
  unfold Machine.code
  have h1 : ∀ θ, (prog (x := x) (cap := cap) (oracle := oracle) plan θ).length ≤ 1 := fun θ => by
    simp only [prog]
    cases ((plan.getD θ.1 [])[θ.2]?.map Ins.instr) <;> simp
  calc ((sched plan).flatten.dedup.map fun θ => (prog plan θ).length).sum
      ≤ ((sched plan).flatten.map fun θ => (prog (x := x) (cap := cap) (oracle := oracle) plan θ).length).sum :=
        ((List.dedup_sublist _).map _).sum_le_sum (fun _ _ => Nat.zero_le _)
    _ ≤ ((sched plan).flatten.map fun _ => 1).sum := List.sum_le_sum fun θ _ => h1 θ
    _ = (plan.map List.length).sum := by
        rw [List.map_const', List.sum_replicate, nsmul_eq_mul, Nat.cast_id, mul_one, List.length_flatten,
          sched_eq, schedFrom_map_length]

theorem chunk_sum (len : ℕ) : ∀ m : ℕ,
    ((List.range m).map fun u => (chunk len 151 u).length).sum = min len (151 * m)
  | 0 => by simp
  | m + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, chunk_sum len m]
    simp [chunk]
    omega

theorem mmaTick_len (n : ℕ) : ∀ m : ℕ, ((List.range m).map fun j => (mmaTick n j).length).sum = m - m / 21
  | 0 => by simp
  | m + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, mmaTick_len n m]
    by_cases h : m % 21 = 20 <;> simp [mmaTick, h] <;> omega

theorem kernel_instrs (n : ℕ) :
    ((kernel n).map List.length).sum = 2 * n + mmas n + n * n / 4 := by
  have h := chunk_sum (n * n / 4) ((n * n / 4 + 150) / 151)
  rw [min_eq_left (by omega)] at h
  simp only [kernel, List.map_append, List.sum_append, List.map_map, Function.comp_def, List.length_map,
    List.length_cons, List.length_nil, h, mmaTick_len]
  simp
  omega

/-! ## Code fetch -/

theorem chunk_sum4 (len : ℕ) : ∀ m : ℕ,
    ((List.range m).map fun u => (chunk len 151 u).length * 4).sum = min len (151 * m) * 4
  | 0 => by simp
  | m + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, chunk_sum4 len m]
    simp [chunk]
    omega

theorem kernel_dram (n : ℕ) :
    ((kernel n).map fun is => tickUse x is .dram).sum = ((2 * n * (n / 4) + n * n / 4 * 4 : ℕ) : ℚ≥0) := by
  have h := chunk_sum4 (n * n / 4) ((n * n / 4 + 150) / 151)
  rw [min_eq_left (by omega)] at h
  have hl : ∀ c, tickUse x [load n c] .dram = ((n / 4 : ℕ) : ℚ≥0) := by
    intro c; unfold load; split <;> simp [tickUse, copyN, opUse, port, rd]
  have hm : ∀ j, tickUse x (mmaTick n j) .dram = 0 := by
    intro j; unfold mmaTick; split <;> simp [tickUse, mma, mmaOp, opUse, port, rd]
  have hs : ∀ u, tickUse x ((chunk (n * n / 4) 151 u).map (store n)) .dram =
      (((chunk (n * n / 4) 151 u).length * 4 : ℕ) : ℚ≥0) := by
    intro u; rw [store_tick]; simp [opUse, port, rd]
  simp only [kernel, List.map_append, List.sum_append, List.map_map, Function.comp_def, hl, hm, hs]
  rw [show ((List.range ((n * n / 4 + 150) / 151)).map fun u =>
      (((chunk (n * n / 4) 151 u).length * 4 : ℕ) : ℚ≥0)) =
    ((List.range ((n * n / 4 + 150) / 151)).map fun u => (chunk (n * n / 4) 151 u).length * 4).map (Nat.cast)
    by rw [List.map_map]; rfl, ← Nat.cast_list_sum, h]
  simp only [List.map_const', List.sum_replicate, List.length_range, nsmul_eq_mul]
  push_cast
  ring

theorem fetch_nat (n : ℕ) (h256 : 256 ∣ n) (hn : n ≤ 2048) :
    (2 * n * (n / 4) + n * n / 4 * 4 + 4 * (2 * n + mmas n + n * n / 4)) * 33 ≤
      20000 * (2 * n + (mmas n + mmas n / 20) + (n * n / 4 + 150) / 151) := by
  obtain ⟨q, rfl⟩ := h256
  have hq : q ≤ 8 := by omega
  interval_cases q <;> norm_num [mmas, blocks]

/-- **G2's code fetch**: the stores' HBM words and the code's, fetched once, fit in HBM's port over the run. -/
theorem kernel_codeFetch (n : ℕ) (h256 : 256 ∣ n) (hn : n ≤ 2048)
    (σ : (machine x cap oracle).State (ℕ × ℕ)) (hpc : ∀ θ, σ.pc θ = 0) :
    Machine.CodeFetch (prog (kernel n)) σ (sched (kernel n)) := by
  unfold Machine.CodeFetch
  have hlen : (sched (kernel n)).length = (kernel n).length := List.length_mapIdx
  have hsum : ∑ t ∈ Finset.range (sched (kernel n)).length,
      Machine.usage (m := machine x cap oracle) (Machine.run (prog (kernel n)) σ (sched (kernel n))).2 t Res.dram =
      ((2 * n * (n / 4) + n * n / 4 * 4 : ℕ) : ℚ≥0) := by
    rw [run_plan _ σ hpc, hlen]
    simp only [usage_plan]
    rw [sum_range_getD (fun is => tickUse x is .dram) (by simp [tickUse]), kernel_dram]
  have hc := code_le (x := x) (cap := cap) (oracle := oracle) (kernel n)
  rw [kernel_instrs] at hc
  have hc' : ((Machine.code (prog (x := x) (cap := cap) (oracle := oracle) (kernel n)) (sched (kernel n)) : ℕ) :
      ℚ≥0) ≤ ((2 * n + mmas n + n * n / 4 : ℕ) : ℚ≥0) := by exact_mod_cast hc
  have hnat := fetch_nat n h256 hn
  have key : ((2 * n * (n / 4) + n * n / 4 * 4 : ℕ) : ℚ≥0) +
      4 * ((Machine.code (prog (x := x) (cap := cap) (oracle := oracle) (kernel n)) (sched (kernel n)) : ℕ) : ℚ≥0) ≤
      ρ .dram * (((sched (kernel n)).length + x.slack : ℕ) : ℚ≥0) := by
    calc _ ≤ ((2 * n * (n / 4) + n * n / 4 * 4 : ℕ) : ℚ≥0) + 4 * ((2 * n + mmas n + n * n / 4 : ℕ) : ℚ≥0) :=
          add_le_add le_rfl (mul_le_mul_of_nonneg_left hc' zero_le)
      _ ≤ ρ .dram * ((sched (kernel n)).length : ℚ≥0) := by
          rw [hlen, kernel_length, ρ, div_mul_eq_mul_div, le_div_iff₀ (by norm_num)]
          exact_mod_cast hnat
      _ ≤ _ := mul_le_mul_of_nonneg_left (by exact_mod_cast Nat.le_add_right _ _) zero_le
  show ∑ t ∈ Finset.range (sched (kernel n)).length, Machine.usage (m := machine x cap oracle)
      (Machine.run (prog (kernel n)) σ (sched (kernel n))).2 t Res.dram + 4 * _ ≤ ρ .dram * _
  rw [hsum]
  exact key

/-! ## Fits -/

/-- **G1's code**: the kernel's texts fit in the H200's code capacity. -/
theorem kernel_progFits (n : ℕ) (h256 : 256 ∣ n) (hn : n ≤ 2048) :
    (prog (x := x) (cap := cap) (oracle := oracle) (kernel n)).Fits (sched (kernel n)) := by
  refine (code_le (kernel n)).trans ?_
  rw [kernel_instrs]
  show _ ≤ 141 * 2 ^ 30 / 16
  obtain ⟨q, rfl⟩ := h256
  have hq : q ≤ 8 := by omega
  interval_cases q <;> norm_num [mmas, blocks]

/-- The kernel meets the model: G1's code, G2's rates and G2's code fetch. -/
theorem kernel_fits (n : ℕ) (h256 : 256 ∣ n) (hn : n ≤ 2048) (hs : 1 ≤ x.slack)
    (σ : (machine x cap oracle).State (ℕ × ℕ)) (hpc : ∀ θ, σ.pc θ = 0) :
    Machine.Fits (prog (kernel n)) σ (sched (kernel n)) :=
  ⟨kernel_progFits n h256 hn, kernel_rate n hn hs σ hpc, kernel_codeFetch n h256 hn σ hpc⟩

/-! ## Writes -/

section Writes

variable {α β γ : Type} [DecidableEq β]

theorem foldl_update_not_mem (f : α → β) (g : α → γ) (b : β) :
    ∀ (l : List α) (M : β → γ), (∀ a ∈ l, f a ≠ b) →
      l.foldl (fun M' a => Function.update M' (f a) (g a)) M b = M b
  | [], _, _ => rfl
  | a :: l, M, h => by
    rw [List.foldl_cons, foldl_update_not_mem f g b l _ (fun a' ha' => h a' (by simp [ha'])),
      Function.update_of_ne (h a (by simp)).symm]

theorem foldl_update_mem (f : α → β) (g : α → γ) :
    ∀ (l : List α) (M : β → γ), (∀ a ∈ l, ∀ a' ∈ l, f a = f a' → a = a') → l.Nodup → ∀ a ∈ l,
      l.foldl (fun M' a => Function.update M' (f a) (g a)) M (f a) = g a
  | [], _, _, _, _, h => absurd h (by simp)
  | a₀ :: l, M, hinj, hnd, a, ha => by
    rw [List.foldl_cons]
    have hnd' := List.nodup_cons.mp hnd
    rcases List.mem_cons.mp ha with rfl | ha'
    · rw [foldl_update_not_mem f g (f a) l _ (fun a' ha' e =>
        hnd'.1 (hinj a' (by simp [ha']) a (by simp) e ▸ ha')), Function.update_self]
    · exact foldl_update_mem f g l _ (fun a ha a' ha' => hinj a (by simp [ha]) a' (by simp [ha'])) hnd'.2 a ha'

theorem execOp_direct {ι : Type} {m : Machine ι} (o : m.Op) (src : Fin (m.nIn o) → Machine.Operand)
    (d : Fin (m.nOut o) → ℕ) (M : m.Mem) :
    Machine.execOp o src (fun i => .direct (d i)) none M =
      (List.finRange (m.nOut o)).foldl (fun M' i => Function.update M' (m.loc (m.outStore o i) (d i))
        (m.eval o (fun i => (src i).readIn (m.inStore o i) M) i)) M := by
  simp [Machine.execOp, Machine.Operand.writeIn, Machine.Operand.locIn]

theorem loc_eq_iff {ι : Type} {m : Machine ι} (s s' : m.Store) (q q' : ℕ) :
    m.loc s q = m.loc s' q' ↔ s = s' ∧ q % m.size s = q' % m.size s' := by
  constructor
  · intro h
    obtain rfl : s = s' := congrArg Sigma.fst h
    exact ⟨rfl, congrArg Fin.val (eq_of_heq (Sigma.mk.inj h).2)⟩
  · rintro ⟨rfl, h⟩
    simp only [Machine.loc, h]

end Writes

/-! ## Copies -/

theorem copy_exec (s t : Store) (len : ℕ) (h : Avail (.copy s t len)) (a b : ℕ)
    (M : (machine x cap oracle).Mem) :
    instrExec (copyN s t len h a b).instr M =
      (List.finRange len).foldl (fun M' (i : Fin len) => Function.update M'
        ((machine x cap oracle).loc t (b + i.val)) (M ((machine x cap oracle).loc s (a + i.val)))) M := by
  simp only [instrExec, Ins.instr, copyN]
  refine (execOp_direct (m := machine x cap oracle) ⟨.copy s t len, h⟩ _ (fun i => b + i.val) M).trans ?_
  congr 1
  funext M' i
  have hi : (i : ℕ) < len := i.isLt
  simp [Card.machine, machine, Opcode.eval, getD, Opcode.outStore, Opcode.inStore, Opcode.nIn, Machine.Operand.readIn,
    Machine.Operand.locIn, hi]

theorem copy_at (s t : Store) (len : ℕ) (h : Avail (.copy s t len)) (a b : ℕ)
    (M : (machine x cap oracle).Mem) (u : Store) (q : ℕ) (hq : q < (machine x cap oracle).size u)
    (hb : b + len ≤ (machine x cap oracle).size t) :
    instrExec (copyN s t len h a b).instr M ((machine x cap oracle).loc u q) =
      if u = t ∧ b ≤ q ∧ q < b + len then M ((machine x cap oracle).loc s (a + (q - b)))
      else M ((machine x cap oracle).loc u q) := by
  rw [copy_exec]
  split_ifs with hin
  · obtain ⟨rfl, h1, h2⟩ := hin
    have := foldl_update_mem (fun i : Fin len => (machine x cap oracle).loc u (b + i.val))
      (fun i => M ((machine x cap oracle).loc s (a + i.val))) (List.finRange len) M
      (fun i _ j _ e => by
        have e2 := ((loc_eq_iff (m := machine x cap oracle) u u _ _).mp e).2
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)] at e2
        exact Fin.ext (by omega))
      (List.nodup_finRange len) ⟨q - b, by omega⟩ (List.mem_finRange _)
    simp only at this
    rwa [show b + (q - b) = q by omega] at this
  · apply foldl_update_not_mem
    intro i _ e
    obtain ⟨rfl, e2⟩ := (loc_eq_iff (m := machine x cap oracle) t u _ _).mp e
    rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt hq] at e2
    exact hin ⟨rfl, by omega, by omega⟩

/-! ## The loads -/

theorem load_eq (n c : ℕ) (h4 : 4 ∣ n) :
    load n c = copyN .dram .smem (n / 4) (Or.inr (Or.inr ⟨rfl, by decide⟩)) (c * (n / 4)) (c * (n / 4)) := by
  unfold load
  split
  · simp [addrA]
  · rename_i hc
    obtain ⟨k, rfl⟩ := h4
    obtain ⟨d, rfl⟩ : ∃ d, c = 4 * k + d := ⟨c - 4 * k, by omega⟩
    have e1 : 4 * k / 4 = k := by omega
    have e2 : 4 * k * (4 * k) / 4 = 4 * k * k := by
      rw [show 4 * k * (4 * k) = 4 * (4 * k * k) by ring]
      exact Nat.mul_div_cancel_left _ (by norm_num)
    simp only [addrB, e1, e2, Nat.add_sub_cancel_left]
    congr 1 <;> ring

/-- One tick's effect on the stores. -/
def tickExec (M : (machine x cap oracle).Mem) (is : List Ins) : (machine x cap oracle).Mem :=
  is.foldl (fun M a => instrExec a.instr M) M

theorem loads_exec (n : ℕ) (h4 : 4 ∣ n) (hsz : 2 * n * (n / 4) ≤ (machine x cap oracle).size .smem)
    (hsd : (machine x cap oracle).size .smem ≤ (machine x cap oracle).size .dram)
    (M : (machine x cap oracle).Mem) :
    ∀ c ≤ 2 * n, ∀ (u : Store) (q : ℕ), q < (machine x cap oracle).size u →
      ((List.range c).map fun c => [load n c]).foldl tickExec M ((machine x cap oracle).loc u q) =
        if u = .smem ∧ q < c * (n / 4) then M ((machine x cap oracle).loc .dram q)
        else M ((machine x cap oracle).loc u q)
  | 0, _, u, q, _ => by simp
  | c + 1, hc, u, q, hq => by
    have hb : c * (n / 4) + n / 4 ≤ (machine x cap oracle).size .smem := by
      have := Nat.mul_le_mul_right (n / 4) hc
      rw [add_mul, one_mul] at this
      omega
    rw [List.range_succ, List.map_append, List.foldl_append, List.map_cons, List.map_nil, List.foldl_cons,
      List.foldl_nil, tickExec, List.foldl_cons, List.foldl_nil, load_eq n c h4]
    erw [copy_at .dram .smem (n / 4) _ (c * (n / 4)) (c * (n / 4)) _ u q hq hb]
    have ih := loads_exec n h4 hsz hsd M c (by omega)
    by_cases hu : u = .smem
    · subst hu
      by_cases h1 : c * (n / 4) ≤ q ∧ q < c * (n / 4) + n / 4
      · rw [ite_eq_left ⟨rfl, h1⟩, ih .dram _ (by omega), ite_eq_right (by simp),
          ite_eq_left ⟨rfl, by rw [add_mul, one_mul]; omega⟩, show c * (n / 4) + (q - c * (n / 4)) = q by omega]
      · rw [ite_eq_right (fun h => h1 h.2), ih .smem q hq]
        by_cases h2 : q < c * (n / 4)
        · rw [ite_eq_left ⟨rfl, h2⟩, ite_eq_left ⟨rfl, by rw [add_mul, one_mul]; omega⟩]
        · rw [ite_eq_right (fun h => h2 h.2), ite_eq_right (fun h => by rw [add_mul, one_mul] at h; omega)]
    · rw [ite_eq_right (fun h => hu h.1), ih u q hq, ite_eq_right (fun h => hu h.1), ite_eq_right (fun h => hu h.1)]

/-! ## The stores -/

theorem store_eq (n w : ℕ) (h4 : 4 ∣ n) :
    store n w = copyN .reg .dram 4 (Or.inl le_rfl) (4 * w) (n * n / 2 + 4 * w) := by
  obtain ⟨k, rfl⟩ := h4
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · simp [store, acc, dramC]
  have e1 : 4 * k / 4 = k := by omega
  have e2 : w / k * (4 * k) + 4 * (w % k) = 4 * w := by
    have := Nat.div_add_mod w k
    rw [show w / k * (4 * k) = 4 * (k * (w / k)) by ring]
    omega
  simp only [store, acc, dramC, e1, e2, Nat.add_assoc]

/-- The stores' instructions in order. -/
theorem stores_exec (n : ℕ) (h4 : 4 ∣ n) (hsd : n * n / 2 + n * n ≤ (machine x cap oracle).size .dram)
    (hsr : n * n ≤ (machine x cap oracle).size .reg) (M : (machine x cap oracle).Mem) :
    ∀ m ≤ n * n / 4, ∀ (u : Store) (q : ℕ), q < (machine x cap oracle).size u →
      ((List.range m).map (store n)).foldl (fun M a => instrExec a.instr M) M ((machine x cap oracle).loc u q) =
        if u = .dram ∧ n * n / 2 ≤ q ∧ q < n * n / 2 + 4 * m then M ((machine x cap oracle).loc .reg (q - n * n / 2))
        else M ((machine x cap oracle).loc u q)
  | 0, _, u, q, _ => by simp
  | m + 1, hm, u, q, hq => by
    have hb : n * n / 2 + 4 * m + 4 ≤ (machine x cap oracle).size .dram := by omega
    rw [List.range_succ, List.map_append, List.foldl_append, List.map_cons, List.map_nil, List.foldl_cons,
      List.foldl_nil, store_eq n m h4]
    erw [copy_at .reg .dram 4 _ (4 * m) (n * n / 2 + 4 * m) _ u q hq hb]
    have ih := stores_exec n h4 hsd hsr M m (by omega)
    by_cases hu : u = .dram
    · subst hu
      by_cases h1 : n * n / 2 + 4 * m ≤ q ∧ q < n * n / 2 + 4 * m + 4
      · rw [ite_eq_left ⟨rfl, h1⟩, ih .reg _ (by omega), ite_eq_right (by simp), ite_eq_left ⟨rfl, by omega, by omega⟩,
          show 4 * m + (q - (n * n / 2 + 4 * m)) = q - n * n / 2 by omega]
      · rw [ite_eq_right (fun h => h1 h.2), ih .dram q hq]
        by_cases h2 : n * n / 2 ≤ q ∧ q < n * n / 2 + 4 * m
        · rw [ite_eq_left ⟨rfl, h2⟩, ite_eq_left ⟨rfl, h2.1, by omega⟩]
        · rw [ite_eq_right (fun h => h2 h.2), ite_eq_right (fun h => by omega)]
    · rw [ite_eq_right (fun h => hu h.1), ih u q hq, ite_eq_right (fun h => hu h.1), ite_eq_right (fun h => hu h.1)]

theorem chunk_flatten (len : ℕ) : ∀ m : ℕ,
    ((List.range m).map (chunk len 151)).flatten = List.range (min len (151 * m))
  | 0 => by simp
  | m + 1 => by
    rw [List.range_succ, List.map_append, List.flatten_append, chunk_flatten len m]
    simp only [List.map_cons, List.map_nil, List.flatten_cons, List.flatten_nil, List.append_nil, chunk]
    rw [List.range_eq_range', List.range_eq_range']
    by_cases h : len ≤ 151 * m
    · rw [min_eq_left h, min_eq_left (by omega : len ≤ 151 * (m + 1)), show len - 151 * m = 0 by omega]
      simp
    · have := List.range'_append (s := 0) (m := 151 * m) (n := min 151 (len - 151 * m)) (step := 1)
      rw [min_eq_right (by omega : 151 * m ≤ len),
        show min len (151 * (m + 1)) = 151 * m + min 151 (len - 151 * m) by omega]
      simpa using this

/-- The store ticks run the stores in order. -/
theorem stores_ticks (n : ℕ) (M : (machine x cap oracle).Mem) :
    ((List.range ((n * n / 4 + 150) / 151)).map fun u => (chunk (n * n / 4) 151 u).map (store n)).foldl tickExec M =
      ((List.range (n * n / 4)).map (store n)).foldl (fun M a => instrExec a.instr M) M := by
  have hf := chunk_flatten (n * n / 4) ((n * n / 4 + 150) / 151)
  rw [min_eq_left (by omega)] at hf
  rw [← hf, List.map_flatten, List.foldl_flatten, List.map_map]
  rfl

/-! ## The MMAs -/

theorem mmaTick_flatten (n : ℕ) : ∀ L : ℕ,
    ((List.range L).map (mmaTick n)).flatten = (List.range (L - L / 21)).map (mma n)
  | 0 => by simp
  | L + 1 => by
    rw [List.range_succ, List.map_append, List.flatten_append, mmaTick_flatten n L]
    by_cases h : L % 21 = 20
    · rw [show L + 1 - (L + 1) / 21 = L - L / 21 by omega]
      simp [mmaTick, h]
    · rw [show L + 1 - (L + 1) / 21 = L - L / 21 + 1 by omega, List.range_succ, List.map_append]
      simp [mmaTick, h]

/-- The MMA ticks run the MMAs in order: the idle ticks do nothing. -/
theorem mmas_ticks (n : ℕ) (M : (machine x cap oracle).Mem) :
    ((List.range (mmas n + mmas n / 20)).map (mmaTick n)).foldl tickExec M =
      ((List.range (mmas n)).map (mma n)).foldl (fun M a => instrExec a.instr M) M := by
  have hf := mmaTick_flatten n (mmas n + mmas n / 20)
  rw [show mmas n + mmas n / 20 - (mmas n + mmas n / 20) / 21 = mmas n by omega] at hf
  rw [← hf, List.foldl_flatten]
  rfl

/-- MMA `p`'s output `o`: its accumulator's index. -/
def accOut (n p o : ℕ) : ℕ :=
  acc n (64 * (p % blocks n / (n / 256)) + o / 256) (256 * (p % blocks n % (n / 256)) + o % 256)

theorem mma_exec (n p : ℕ) (M : (machine x cap oracle).Mem) :
    instrExec (mma n p).instr M =
      (List.finRange ((machine x cap oracle).nOut (mma n p).1)).foldl (fun M' o => Function.update M'
        ((machine x cap oracle).loc .reg (accOut n p o.val))
        ((machine x cap oracle).eval (mma n p).1
          (fun i => ((mma n p).2.1 i.val).readIn ((machine x cap oracle).inStore (mma n p).1 i) M) o)) M := by
  simp only [instrExec, Ins.instr]
  exact execOp_direct (m := machine x cap oracle) _ _ (fun o => accOut n p o.val) M

theorem mma_dot (fa fb : Fmt) (sat : Bool) (k : ℕ) (h : intPair fa fb = true) :
    card.dot x fa fb sat k = intDot fa fb sat := by
  funext c a b
  have h2 : fp8 fa = false := by
    cases fa <;> cases fb <;> simp_all [intPair, fp8]
  simp [Card.dot, card, captured, intCaptured, h, h2]

theorem mma_eval (n p : ℕ) (v : Fin ((machine x cap oracle).nIn (mma n p).1) → Word)
    (o : Fin ((machine x cap oracle).nOut (mma n p).1)) :
    (machine x cap oracle).eval (mma n p).1 v o =
      intDot .s8 .s8 false (getD v (2560 + o.val)) (rowA .s8 ⟨64, 256, 32⟩ (getD v) (o.val / 256))
        (colB .s8 .s8 ⟨64, 256, 32⟩ (getD v) (o.val % 256)) := by
  show mmaEval (card.dot x .s8 .s8 false 32) .s8 .s8 ⟨64, 256, 32⟩ (getD v) o.val = _
  rw [mma_dot .s8 .s8 false 32 rfl]
  rfl

/-- MMA `p`'s input words in `M`, as its opcode reads them. -/
def mmaV (n p : ℕ) (M : (machine x cap oracle).Mem) : ℕ → Word :=
  getD (fun i : Fin ((machine x cap oracle).nIn (mma n p).1) =>
    ((mma n p).2.1 i.val).readIn ((machine x cap oracle).inStore (mma n p).1 i) M)

theorem mma_in (n p : ℕ) (M : (machine x cap oracle).Mem) (i : ℕ) (hi : i < 18944) :
    mmaV n p M i =
      if i < 512 then
        M ((machine x cap oracle).loc .smem (addrA n (64 * (p % blocks n / (n / 256)) + i / 8) (8 * (p / blocks n) + i % 8)))
      else if i < 2560 then
        M ((machine x cap oracle).loc .smem
          (addrB n (256 * (p % blocks n % (n / 256)) + (i - 512) / 8) (8 * (p / blocks n) + (i - 512) % 8)))
      else if p / blocks n = 0 then 0
      else M ((machine x cap oracle).loc .reg
        (acc n (64 * (p % blocks n / (n / 256)) + (i - 2560) / 256) (256 * (p % blocks n % (n / 256)) + (i - 2560) % 256))) := by
  have hn : (machine x cap oracle).nIn (mma n p).1 = 18944 := rfl
  have hwa : packedWords 64 (Fmt.s8.rowLen 32) Fmt.s8.bits = 512 := by decide
  have hwb : packedWords 256 (Fmt.s8.rowLen 32) Fmt.s8.bits = 2048 := by decide
  simp only [mmaV, getD, hn, hi, dite_true]
  simp only [mma, mmaOp, Card.machine, machine, Opcode.inStore, Shape.wa, Shape.wb, hwa, hwb, Machine.Operand.readIn,
    Machine.Operand.locIn]
  by_cases h1 : i < 512
  · simp [h1]
  · by_cases h2 : i < 2560
    · simp [h1, h2]
    · by_cases h3 : p / blocks n = 0 <;> simp [h1, h2, h3]

/-! ## Bytes -/

theorem byte_toInt (w : Word) (k : ℕ) : Fmt.s8.intVal (w.toNat / 2 ^ (8 * k) % 2 ^ 8) = (byteOf w k).toInt := by
  rw [byteOf, BitVec.toInt_eq_toNat_cond, BitVec.toNat_ofNat]
  simp only [Fmt.intVal]
  have h : w.toNat / 2 ^ (8 * k) % 2 ^ 8 < 256 := Nat.mod_lt _ (by norm_num)
  split_ifs <;> push_cast <;> omega

/-- Entry `(i, l)` of `A` as MMA operands read it from shared memory in `M`. -/
def aOf (n : ℕ) (M : (machine x cap oracle).Mem) (i l : ℕ) : ℤ :=
  (byteOf (M ((machine x cap oracle).loc .smem (addrA n i (l / 4)))) (l % 4)).toInt

/-- Entry `(l, j)` of `B` as MMA operands read it from shared memory in `M`. -/
def bOf (n : ℕ) (M : (machine x cap oracle).Mem) (l j : ℕ) : ℤ :=
  (byteOf (M ((machine x cap oracle).loc .smem (addrB n j (l / 4)))) (l % 4)).toInt

theorem zipWith_ofFn_sum (f : ℕ → ℕ → ℤ) (F G : ℕ → ℕ) : ∀ k : ℕ,
    (List.zipWith f (List.ofFn fun l : Fin k => F l) (List.ofFn fun l : Fin k => G l)).sum =
      ∑ l ∈ Finset.range k, f (F l) (G l)
  | 0 => by simp
  | k + 1 => by
    simp only [List.ofFn_succ', Fin.val_castSucc, Fin.val_last, List.concat_eq_append]
    rw [List.zipWith_append (by simp), List.sum_append, zipWith_ofFn_sum f F G k, Finset.sum_range_succ]
    simp

/-- MMA `p`'s products for output `o = 256 r + s`: `A`'s row `i₀ + r` against `B`'s column `j₀ + s` over slice `κ`. -/
theorem mma_sum (n p : ℕ) (M : (machine x cap oracle).Mem) (r s : ℕ) (hr : r < 64) (hs : s < 256) :
    (List.zipWith (fun x y => Fmt.s8.intVal x * Fmt.s8.intVal y)
      (rowA .s8 ⟨64, 256, 32⟩ (mmaV n p M) r) (colB .s8 .s8 ⟨64, 256, 32⟩ (mmaV n p M) s)).sum =
      ∑ l ∈ Finset.range 32, aOf n M (64 * (p % blocks n / (n / 256)) + r) (32 * (p / blocks n) + l) *
        bOf n M (32 * (p / blocks n) + l) (256 * (p % blocks n % (n / 256)) + s) := by
  have hwa : packedWords 64 (Fmt.s8.rowLen 32) Fmt.s8.bits = 512 := by decide
  simp only [rowA, colB, Shape.wa, hwa]
  rw [zipWith_ofFn_sum (fun x y => Fmt.s8.intVal x * Fmt.s8.intVal y)
    (fun l => elem Fmt.s8.bits (mmaV n p M) (r * Fmt.s8.rowLen 32 + l))
    (fun l => elem Fmt.s8.bits (fun w => mmaV n p M (512 + w)) (s * Fmt.s8.rowLen 32 + l)) (Fmt.s8.rowLen 32)]
  simp only [show Fmt.s8.rowLen 32 = 32 from rfl, show Fmt.s8.bits = 8 from rfl]
  refine Finset.sum_congr rfl fun l hl => ?_
  have hl : l < 32 := Finset.mem_range.mp hl
  simp only [elem]
  rw [show (r * 32 + l) * 8 / 32 = 8 * r + l / 4 by omega, show (r * 32 + l) * 8 % 32 = 8 * (l % 4) by omega,
    show (s * 32 + l) * 8 / 32 = 8 * s + l / 4 by omega, show (s * 32 + l) * 8 % 32 = 8 * (l % 4) by omega,
    mma_in n p M _ (by omega), mma_in n p M _ (by omega), byte_toInt, byte_toInt]
  simp only [show 8 * r + l / 4 < 512 by omega, show ¬ 512 + (8 * s + l / 4) < 512 by omega,
    show 512 + (8 * s + l / 4) < 2560 by omega, ite_true, ite_false, aOf, bOf]
  rw [show (8 * r + l / 4) / 8 = r by omega, show (8 * r + l / 4) % 8 = l / 4 by omega,
    show (512 + (8 * s + l / 4) - 512) / 8 = s by omega, show (512 + (8 * s + l / 4) - 512) % 8 = l / 4 by omega,
    show (32 * (p / blocks n) + l) / 4 = 8 * (p / blocks n) + l / 4 by omega,
    show (32 * (p / blocks n) + l) % 4 = l % 4 by omega]

/-! ## One MMA's effect -/

theorem divmod_unique {a b c d Q : ℕ} (hb : b < Q) (hd : d < Q) (h : a * Q + b = c * Q + d) :
    a = c ∧ b = d := by
  have hQ : 0 < Q := by omega
  have h1 : (a * Q + b) / Q = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hQ, Nat.div_eq_of_lt hb, zero_add]
  have h2 : (c * Q + d) / Q = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hQ, Nat.div_eq_of_lt hd, zero_add]
  have h3 : (a * Q + b) % Q = b := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]
  have h4 : (c * Q + d) % Q = d := by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]
  exact ⟨by rw [← h1, ← h2, h], by rw [← h3, ← h4, h]⟩

theorem acc_lt (n i j : ℕ) (hi : i < n) (hj : j < n) : acc n i j < n * n := by
  simp only [acc]
  nlinarith

theorem size_reg : (machine x cap oracle).size .reg = 132 * (65536 + 4494) := rfl

theorem sq_le_reg (n : ℕ) (hn : n ≤ 2048) : n * n ≤ (machine x cap oracle).size .reg := by
  rw [size_reg]
  nlinarith

/-- The block `(i/64, j/256)` of entry `(i, j)`, numbered as the MMAs number them. -/
def blk (n i j : ℕ) : ℕ := i / 64 * (n / 256) + j / 256

theorem blocks_eq (Q : ℕ) : blocks (256 * Q) = 4 * Q * Q := by
  simp only [blocks]
  rw [show 256 * Q / 64 = 4 * Q by omega, show 256 * Q / 256 = Q by omega]

theorem blk_lt (n i j : ℕ) (h256 : 256 ∣ n) (hi : i < n) (hj : j < n) : blk n i j < blocks n := by
  obtain ⟨Q, rfl⟩ := h256
  rw [blocks_eq]
  simp only [blk, show 256 * Q / 256 = Q by omega]
  have h1 : i / 64 < 4 * Q := by omega
  have h2 : j / 256 < Q := by omega
  nlinarith

/-- Entry `(i, j)` is in block `e` exactly when its block coordinates are `e`'s. -/
theorem blk_iff (n i j e : ℕ) (h256 : 256 ∣ n) (hj : j < n) :
    blk n i j = e ↔ i / 64 = e / (n / 256) ∧ j / 256 = e % (n / 256) := by
  obtain ⟨Q, rfl⟩ := h256
  simp only [blk, show 256 * Q / 256 = Q by omega]
  have hQ : 0 < Q := by omega
  constructor
  · rintro rfl
    have hj' : j / 256 < Q := by omega
    exact ⟨by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hQ, Nat.div_eq_of_lt hj', zero_add],
      by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj']⟩
  · rintro ⟨h1, h2⟩
    rw [h1, h2, Nat.div_add_mod']

theorem blk_geom (n e : ℕ) (h256 : 256 ∣ n) (he : e < blocks n) :
    64 * (e / (n / 256)) + 64 ≤ n ∧ 256 * (e % (n / 256)) + 256 ≤ n := by
  obtain ⟨Q, rfl⟩ := h256
  rw [blocks_eq] at he
  rw [show 256 * Q / 256 = Q by omega]
  have hQ : 0 < Q := by
    rcases Nat.eq_zero_or_pos Q with rfl | h
    · simp at he
    · exact h
  have h1 : e / Q < 4 * Q := (Nat.div_lt_iff_lt_mul hQ).mpr (by nlinarith)
  have h2 : e % Q < Q := Nat.mod_lt _ hQ
  omega

theorem blocks_pos (n : ℕ) (h256 : 256 ∣ n) (hn : 0 < n) : 0 < blocks n := by
  obtain ⟨Q, rfl⟩ := h256
  rw [blocks_eq]
  have : 0 < Q := by omega
  positivity

/-- An MMA writes no store but the registers. -/
theorem mma_other (n p : ℕ) (M : (machine x cap oracle).Mem) (u : Store) (hu : u ≠ .reg) (q : ℕ) :
    instrExec (mma n p).instr M ((machine x cap oracle).loc u q) = M ((machine x cap oracle).loc u q) := by
  rw [mma_exec]
  exact foldl_update_not_mem _ _ _ _ _ fun o _ h =>
    hu ((loc_eq_iff (m := machine x cap oracle) _ _ _ _).mp h).1.symm

/-- **One MMA on the accumulators**: MMA `p` (slice `κ = p / blocks n`, block `e = p % blocks n`) adds its slice's
products to each accumulator of block `e`, from zero at `κ = 0`, and leaves every other accumulator. -/
theorem mma_reg (n p : ℕ) (h256 : 256 ∣ n) (hn0 : 0 < n) (hn : n ≤ 2048) (M : (machine x cap oracle).Mem)
    (i j : ℕ) (hi : i < n) (hj : j < n) :
    instrExec (mma n p).instr M ((machine x cap oracle).loc .reg (acc n i j)) =
      if blk n i j = p % blocks n then
        BitVec.ofInt 32 ((if p / blocks n = 0 then 0 else (M ((machine x cap oracle).loc .reg (acc n i j))).toInt) +
          ∑ l ∈ Finset.range 32, aOf n M i (32 * (p / blocks n) + l) * bOf n M (32 * (p / blocks n) + l) j)
      else M ((machine x cap oracle).loc .reg (acc n i j)) := by
  have he : p % blocks n < blocks n := Nat.mod_lt _ (blocks_pos n h256 hn0)
  obtain ⟨gi, gj⟩ := blk_geom n (p % blocks n) h256 he
  set i₀ := 64 * (p % blocks n / (n / 256)) with hi₀
  set j₀ := 256 * (p % blocks n % (n / 256)) with hj₀
  have hsz := sq_le_reg (x := x) (cap := cap) (oracle := oracle) n hn
  have hout : ∀ o : ℕ, o < 16384 → accOut n p o < n * n := fun o ho => acc_lt n _ _ (by omega) (by omega)
  have hacc := acc_lt n i j hi hj
  rw [mma_exec]
  by_cases hb : blk n i j = p % blocks n
  · rw [ite_eq_left hb]
    obtain ⟨h1, h2⟩ := (blk_iff n i j _ h256 hj).mp hb
    have ho : (i % 64) * 256 + j % 256 < (machine x cap oracle).nOut (mma n p).1 := by
      show _ < 64 * 256
      omega
    have hf : accOut n p ((i % 64) * 256 + j % 256) = acc n i j := by
      simp only [accOut, acc]
      rw [← hi₀, ← hj₀, show (i % 64 * 256 + j % 256) / 256 = i % 64 by omega,
        show (i % 64 * 256 + j % 256) % 256 = j % 256 by omega, show i₀ + i % 64 = i by omega,
        show j₀ + j % 256 = j by omega]
    have key := foldl_update_mem (fun o : Fin ((machine x cap oracle).nOut (mma n p).1) =>
        (machine x cap oracle).loc .reg (accOut n p o.val))
      (fun o => (machine x cap oracle).eval (mma n p).1
        (fun i => ((mma n p).2.1 i.val).readIn ((machine x cap oracle).inStore (mma n p).1 i) M) o)
      (List.finRange _) M
      (fun a _ a' _ e => by
        have e2 := ((loc_eq_iff (m := machine x cap oracle) _ _ _ _).mp e).2
        have ha : a.val < 16384 := a.isLt
        have ha' : a'.val < 16384 := a'.isLt
        rw [Nat.mod_eq_of_lt ((hout _ ha).trans_le hsz), Nat.mod_eq_of_lt ((hout _ ha').trans_le hsz)] at e2
        simp only [accOut, acc] at e2
        obtain ⟨e3, e4⟩ := divmod_unique (by omega) (by omega) e2
        exact Fin.ext (by omega))
      (List.nodup_finRange _) ⟨_, ho⟩ (List.mem_finRange _)
    simp only [hf] at key
    rw [key, mma_eval]
    change intDot .s8 .s8 false (mmaV n p M (2560 + _)) (rowA .s8 ⟨64, 256, 32⟩ (mmaV n p M) _)
      (colB .s8 .s8 ⟨64, 256, 32⟩ (mmaV n p M) _) = _
    simp only [intDot, Bool.false_eq_true, ite_false]
    rw [mma_sum n p M _ _ (by omega) (by omega), mma_in n p M _ (by omega)]
    rw [show (i % 64 * 256 + j % 256) / 256 = i % 64 by omega,
      show (i % 64 * 256 + j % 256) % 256 = j % 256 by omega, ← hi₀, ← hj₀, show i₀ + i % 64 = i by omega,
      show j₀ + j % 256 = j by omega, ite_eq_right (by omega : ¬ 2560 + (i % 64 * 256 + j % 256) < 512),
      ite_eq_right (by omega : ¬ 2560 + (i % 64 * 256 + j % 256) < 2560),
      show (2560 + (i % 64 * 256 + j % 256) - 2560) / 256 = i % 64 by omega,
      show (2560 + (i % 64 * 256 + j % 256) - 2560) % 256 = j % 256 by omega, show i₀ + i % 64 = i by omega,
      show j₀ + j % 256 = j by omega]
    split_ifs <;> simp
  · rw [ite_eq_right hb]
    apply foldl_update_not_mem
    intro o _ e
    have e2 := ((loc_eq_iff (m := machine x cap oracle) _ _ _ _).mp e).2
    have ho : o.val < 16384 := o.isLt
    rw [Nat.mod_eq_of_lt ((hout _ ho).trans_le hsz), Nat.mod_eq_of_lt (hacc.trans_le hsz)] at e2
    simp only [accOut, acc] at e2
    obtain ⟨e3, e4⟩ := divmod_unique (by omega) (by omega) e2
    exact hb ((blk_iff n i j _ h256 hj).mpr ⟨by omega, by omega⟩)

/-! ## The MMAs' invariant -/

/-- After the MMAs of the slices before `κ` and of the first `e` blocks of slice `κ` from `M₀`: every store but the
registers is `M₀`'s, and accumulator `(i, j)` holds the sum over the slices its block has had (anything before its
first). -/
def MmaInv (n : ℕ) (M₀ M : (machine x cap oracle).Mem) (κ e : ℕ) : Prop :=
  (∀ u, u ≠ .reg → ∀ q, M ((machine x cap oracle).loc u q) = M₀ ((machine x cap oracle).loc u q)) ∧
  ∀ i j, i < n → j < n → (blk n i j < e ∨ 1 ≤ κ) →
    M ((machine x cap oracle).loc .reg (acc n i j)) =
      BitVec.ofInt 32 (∑ l ∈ Finset.range (32 * (κ + if blk n i j < e then 1 else 0)), aOf n M₀ i l * bOf n M₀ l j)

theorem ofInt_toInt_add (S T : ℤ) :
    BitVec.ofInt 32 ((BitVec.ofInt 32 S).toInt + T) = BitVec.ofInt 32 (S + T) := by
  rw [BitVec.ofInt_add, BitVec.ofInt_toInt, ← BitVec.ofInt_add]

theorem mma_step (n κ e : ℕ) (h256 : 256 ∣ n) (hn0 : 0 < n) (hn : n ≤ 2048) (he : e < blocks n)
    (M₀ M : (machine x cap oracle).Mem) (h : MmaInv n M₀ M κ e) :
    MmaInv n M₀ (instrExec (mma n (κ * blocks n + e)).instr M) κ (e + 1) := by
  obtain ⟨hother, hreg⟩ := h
  have hB := blocks_pos n h256 hn0
  have hdiv : (κ * blocks n + e) / blocks n = κ := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt he, zero_add]
  have hmod : (κ * blocks n + e) % blocks n = e := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt he]
  have hA : aOf n M = aOf n M₀ := by
    funext i l
    simp only [aOf, hother .smem (show (Store.smem : Store) ≠ .reg by decide)]
  have hB' : bOf n M = bOf n M₀ := by
    funext l j
    simp only [bOf, hother .smem (show (Store.smem : Store) ≠ .reg by decide)]
  refine ⟨fun u hu q => by rw [mma_other n _ M u hu q]; exact hother u hu q, fun i j hi hj hc => ?_⟩
  rw [mma_reg n _ h256 hn0 hn M i j hi hj, hdiv, hmod, hA, hB']
  by_cases hb : blk n i j = e
  · rw [ite_eq_left hb, ite_eq_left (show blk n i j < e + 1 by omega)]
    by_cases hκ : κ = 0
    · subst hκ
      simp
    · rw [ite_eq_right hκ, hreg i j hi hj (Or.inr (by omega)), ite_eq_right (show ¬ blk n i j < e by omega),
        ofInt_toInt_add, Nat.add_zero, show 32 * (κ + 1) = 32 * κ + 32 by ring, Finset.sum_range_add]
  · rw [ite_eq_right hb, hreg i j hi hj (by omega)]
    congr 3
    by_cases h1 : blk n i j < e
    · rw [ite_eq_left h1, ite_eq_left (show blk n i j < e + 1 by omega)]
    · rw [ite_eq_right h1, ite_eq_right (show ¬ blk n i j < e + 1 by omega)]

theorem mma_next (n κ : ℕ) (h256 : 256 ∣ n) (M₀ M : (machine x cap oracle).Mem)
    (h : MmaInv n M₀ M κ (blocks n)) : MmaInv n M₀ M (κ + 1) 0 := by
  refine ⟨h.1, fun i j hi hj _ => ?_⟩
  have hb := blk_lt n i j h256 hi hj
  rw [h.2 i j hi hj (Or.inl hb), ite_eq_left hb, ite_eq_right (Nat.not_lt_zero _)]

theorem mma_slices (n : ℕ) (h256 : 256 ∣ n) (hn0 : 0 < n) (hn : n ≤ 2048) (M₀ : (machine x cap oracle).Mem) :
    ∀ κ, MmaInv n M₀ (((List.range (κ * blocks n)).map (mma n)).foldl (fun M a => instrExec a.instr M) M₀) κ 0
  | 0 => ⟨fun _ _ _ => by simp, fun i j _ _ h => by omega⟩
  | κ + 1 => by
    have ih := mma_slices n h256 hn0 hn M₀ κ
    have hblocks : ∀ e, e ≤ blocks n →
        MmaInv n M₀ (((List.range (κ * blocks n + e)).map (mma n)).foldl (fun M a => instrExec a.instr M) M₀) κ e := by
      intro e
      induction e with
      | zero => intro _; simpa using ih
      | succ e ihe =>
        intro he
        rw [show κ * blocks n + (e + 1) = (κ * blocks n + e) + 1 by ring, List.range_succ, List.map_append,
          List.foldl_append, List.map_cons, List.map_nil, List.foldl_cons, List.foldl_nil]
        exact mma_step n κ e h256 hn0 hn (by omega) M₀ _ (ihe (by omega))
    rw [show (κ + 1) * blocks n = κ * blocks n + blocks n by ring]
    exact mma_next n κ h256 M₀ _ (hblocks (blocks n) le_rfl)

/-- **The MMA phase**: from `M₀`, the kernel's MMAs leave every store but the registers as it was, and accumulator
`(i, j)` holding row `i` of `A` dotted with column `j` of `B`, as shared memory holds them in `M₀`, wrapped to a word. -/
theorem mmas_exec (n : ℕ) (h256 : 256 ∣ n) (hn0 : 0 < n) (hn : n ≤ 2048) (M₀ : (machine x cap oracle).Mem) :
    (∀ u, u ≠ .reg → ∀ q,
      ((List.range (mmas n)).map (mma n)).foldl (fun M a => instrExec a.instr M) M₀ ((machine x cap oracle).loc u q) =
        M₀ ((machine x cap oracle).loc u q)) ∧
    ∀ i j, i < n → j < n →
      ((List.range (mmas n)).map (mma n)).foldl (fun M a => instrExec a.instr M) M₀
          ((machine x cap oracle).loc .reg (acc n i j)) =
        BitVec.ofInt 32 (∑ l ∈ Finset.range n, aOf n M₀ i l * bOf n M₀ l j) := by
  have h := mma_slices n h256 hn0 hn M₀ (n / 32)
  have h32 : 1 ≤ n / 32 := by omega
  refine ⟨h.1, fun i j hi hj => ?_⟩
  rw [mmas, h.2 i j hi hj (Or.inr h32), ite_eq_right (Nat.not_lt_zero _), Nat.add_zero,
    show 32 * (n / 32) = n by omega]

/-! ## The witness -/

/-- The kernel's effect: the loads, then the MMAs, then the stores. -/
theorem kernel_exec (n : ℕ) (M : (machine x cap oracle).Mem) :
    execPlan (kernel n) M =
      ((List.range (n * n / 4)).map (store n)).foldl (fun M a => instrExec a.instr M)
        (((List.range (mmas n)).map (mma n)).foldl (fun M a => instrExec a.instr M)
          (((List.range (2 * n)).map fun c => [load n c]).foldl tickExec M)) := by
  show List.foldl tickExec M (kernel n) = _
  rw [kernel, List.foldl_append, List.foldl_append, mmas_ticks, stores_ticks]

theorem idx_lt {a b i w : ℕ} (hi : i < a) (hw : w < b) : i * b + w < a * b := by nlinarith

theorem witness : Witness := by
  intro ι x cap oracle n hs hn0 h256 hn A B σ hpc hP
  refine ⟨kernel_fits n h256 hn hs σ hpc, fun i j => ?_, kernel_length_le n h256⟩
  have h4 : 4 ∣ n := Nat.dvd_trans (by norm_num) h256
  have hnn : n * n / 4 = n * (n / 4) := Nat.mul_div_assoc n h4
  have h44 : 4 * (n * n / 4) = n * n := Nat.mul_div_cancel' (Dvd.dvd.mul_left h4 n)
  have h2n : 2 * n * (n / 4) = 2 * (n * (n / 4)) := by ring
  have hsz : 2 * n * (n / 4) ≤ (machine x cap oracle).size .smem := by
    show _ ≤ 132 * 65536
    have := Nat.mul_le_mul (Nat.mul_le_mul_left 2 hn) (show n / 4 ≤ 512 by omega)
    omega
  have hsd : (machine x cap oracle).size .smem ≤ (machine x cap oracle).size .dram := by
    show 132 * 65536 ≤ 141 * 2 ^ 30 / 4
    norm_num
  have hsd' : n * n / 2 + n * n ≤ (machine x cap oracle).size .dram := by
    show _ ≤ 141 * 2 ^ 30 / 4
    have := Nat.mul_le_mul hn hn
    omega
  have hL := loads_exec n h4 hsz hsd σ.mem (2 * n) le_rfl
  have hM := mmas_exec n h256 hn0 hn (((List.range (2 * n)).map fun c => [load n c]).foldl tickExec σ.mem)
  have hS := stores_exec n h4 hsd' (sq_le_reg n hn)
    (((List.range (mmas n)).map (mma n)).foldl (fun M a => instrExec a.instr M)
      (((List.range (2 * n)).map fun c => [load n c]).foldl tickExec σ.mem)) (n * n / 4) le_rfl
  have hacc := acc_lt n i j i.isLt j.isLt
  have hdc : dramC n i j - n * n / 2 = acc n i j := by simp only [dramC, acc]; omega
  rw [run_mem _ σ hpc, kernel_exec, hS .dram (dramC n i j) (by simp only [dramC, acc] at hacc ⊢; omega),
    ite_eq_left ⟨rfl, by simp only [dramC]; omega, by simp only [dramC, acc] at hacc ⊢; omega⟩, hdc,
    hM.2 i j i.isLt j.isLt, ← Fin.sum_univ_eq_sum_range]
  congr 1
  refine Finset.sum_congr rfl fun l _ => ?_
  have hl4 : (l : ℕ) / 4 < n / 4 := by omega
  have ha := idx_lt i.isLt hl4
  have hb := idx_lt j.isLt hl4
  simp only [aOf, bOf]
  rw [hL .smem _ (by simp only [addrA]; omega), ite_eq_left ⟨rfl, by simp only [addrA]; omega⟩,
    hL .smem _ (by simp only [addrB]; omega), ite_eq_left ⟨rfl, by simp only [addrB]; omega⟩, hP.1 i l, hP.2 l j]

end Verity.Accel.H200.Resident
