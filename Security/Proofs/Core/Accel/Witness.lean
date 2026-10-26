import Definitions.Core.Accel.Witness
import Mathlib.Algebra.Order.Ring.NNRat
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# The non-vacuity witness, proved (in progress)

The run of a plan issues exactly the plan's instructions, tick by tick (`run_kernel`), since each thread's text is one
instruction and each thread steps once. The kernel's length is within the bound (`kernel_length_le`).
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

theorem run_plan (plan : List (List Ins)) (σ : (machine x cap oracle).State (ℕ × ℕ)) (hpc : ∀ θ, σ.pc θ = 0) :
    (Machine.run (prog plan) σ (sched plan)).2 = plan.map (List.map Ins.instr) := by
  rw [sched_eq]
  exact run_schedFrom plan plan 0 σ List.drop_zero (fun θ _ => hpc θ)

/-! ## Length -/

theorem blocks_mul (n : ℕ) (h : 256 ∣ n) : n / 32 * blocks n * 524288 = n ^ 3 := by
  obtain ⟨q, rfl⟩ := h
  simp only [blocks]
  rw [show 256 * q / 32 = 8 * q by omega, show 256 * q / 64 = 4 * q by omega, show 256 * q / 256 = q by omega]
  ring

theorem kernel_length (n : ℕ) :
    (kernel n).length = 2 * n + n / 32 * blocks n + (n * n / 4 + 150) / 151 := by
  simp [kernel, Nat.add_assoc]

theorem kernel_length_le (n : ℕ) (h : 256 ∣ n) :
    ((sched (kernel n)).length : ℚ≥0) ≤
      2 * (n : ℚ≥0) + (n : ℚ≥0) ^ 3 / 524288 + (n : ℚ≥0) ^ 2 / 604 + 1 := by
  have h3 : ((n / 32 * blocks n : ℕ) : ℚ≥0) = (n : ℚ≥0) ^ 3 / 524288 := by
    rw [eq_div_iff (by norm_num)]
    exact_mod_cast blocks_mul n h
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
      2 * (n : ℚ≥0) + ((n / 32 * blocks n : ℕ) : ℚ≥0) + (((n * n / 4 + 150) / 151 : ℕ) : ℚ≥0) := by
    rw [sched, List.length_mapIdx, kernel_length]
    push_cast
    ring
  have := add_le_add (le_refl (2 * (n : ℚ≥0) + (n : ℚ≥0) ^ 3 / 524288)) hs'
  rw [hl, h3]
  simpa only [add_assoc] using this

end Verity.Accel.H200.Resident
