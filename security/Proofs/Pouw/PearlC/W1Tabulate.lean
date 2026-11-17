module

public import Definitions.Assumptions.Pouw
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.PearlC.W1Stop
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Prob

/-! A cheap winning W1 program refutes TT_OUT at W1, and W1 charges only the branch a run takes. -/

@[expose] public section

namespace Pouw.PearlC.W1Tabulate

open Pouw Pouw.PearlC Pouw.PearlC.W1 Pouw.PearlC.Assumptions

variable {Q R S : Type}

theorem exec_ite_untaken (dv : W1Device) (t : Bool) (p : Addr) (a a' b k : W1.Code) (st : St)
    (h : w32 (st.mem p) = 0) :
    exec dv t (.ite p a b k) st = exec dv t (.ite p a' b k) st := by
  simp [exec, h]

/-- A complete binary tree of `ite`s of depth `n` on the words at `(10, n − 1)`, …, `(10, 0)`, with `leaf path` at the
end of each of its `2^n` paths (`path` lists the branches taken, nonzero as `true`, the last one first). -/
def tree (leaf : List Bool → W1.Code) : ℕ → List Bool → W1.Code
  | 0, path => leaf path
  | n + 1, path => .ite (10, n) (tree leaf n (true :: path)) (tree leaf n (false :: path)) .done

/-- The leaf a memory reaches in `tree leaf n path`. -/
def pathOf (M : Mem) : ℕ → List Bool → List Bool
  | 0, path => path
  | n + 1, path => pathOf M n (decide (w32 (M (10, n)) ≠ 0) :: path)

theorem exec_tree (dv : W1Device) (leaf : List Bool → W1.Code) :
    ∀ (n : ℕ) (path : List Bool) (st : St),
      exec dv true (tree leaf n path) st =
        exec dv true (leaf (pathOf st.mem n path)) ⟨st.mem, st.fetched, st.cost + n * dv.prices.int32⟩
  | 0, path, st => by cases st; simp [tree, pathOf]
  | n + 1, path, st => by
      have hf : ([((10 : ℕ), n)] : List Addr).filter preSalt = [] := by simp [preSalt]
      have hc : st.cost + dv.prices.int32 + (n : ℚ) * dv.prices.int32 =
          st.cost + ((n + 1 : ℕ) : ℚ) * dv.prices.int32 := by
        push_cast; ring
      by_cases h : w32 (st.mem (10, n)) = 0
      · have e : exec dv true (tree leaf (n + 1) path) st =
            exec dv true (tree leaf n (false :: path)) ⟨st.mem, st.fetched, st.cost + dv.prices.int32⟩ := by
          simp [tree, exec, h, hf]
        have hp : pathOf st.mem (n + 1) path = pathOf st.mem n (false :: path) := by simp [pathOf, h]
        rw [e, exec_tree dv leaf n]
        try dsimp only
        rw [hc, hp]
      · have e : exec dv true (tree leaf (n + 1) path) st =
            exec dv true (tree leaf n (true :: path)) ⟨st.mem, st.fetched, st.cost + dv.prices.int32⟩ := by
          simp [tree, exec, h, hf]
        have hp : pathOf st.mem (n + 1) path = pathOf st.mem n (true :: path) := by simp [pathOf, h]
        rw [e, exec_tree dv leaf n]
        try dsimp only
        rw [hc, hp]

theorem exec_tree_untainted (dv : W1Device) (leaf : List Bool → W1.Code) (n : ℕ) (path : List Bool) (st : St) :
    exec dv false (tree leaf (n + 1) path) st = exec dv true (tree leaf (n + 1) path) st := by
  simp [tree, exec]

theorem one_le_codeSize (c : W1.Code) : 1 ≤ c.size := by
  cases c <;> simp only [W1.Code.size] <;> omega

theorem size_tree (leaf : List Bool → W1.Code) : ∀ (n : ℕ) (path : List Bool), 2 ^ n ≤ (tree leaf n path).size
  | 0, path => by simpa [tree] using one_le_codeSize (leaf path)
  | n + 1, path => by
      have h1 := size_tree leaf n (true :: path)
      have h2 := size_tree leaf n (false :: path)
      have h3 : 2 ^ (n + 1) = 2 ^ n + 2 ^ n := by ring
      simp only [tree, W1.Code.size]
      omega

theorem tree_not_code (leaf : List Bool → W1.Code) {n : ℕ} (hn : 64 < n) (path : List Bool) (pr : CodeImage)
    (L : Layout) : pr.code L ≠ tree leaf n path := by
  intro h
  have h1 := (pr.bounded L).1
  rw [h] at h1
  have h2 : 2 ^ 65 ≤ 2 ^ n := Nat.pow_le_pow_right (by norm_num) (by omega)
  exact absurd (h2.trans ((size_tree leaf n path).trans h1)) (by norm_num)

theorem not_ttOutW1_of_win [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {dv : W1Device}
    {salt : S → List ℕ} {P : Protocol Q R S} {D : Layout → Prop} {g₀ : ℝ} {δ : ℕ → ℕ → ℝ}
    (U : Workload) (hU : U.InDomain P D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ) (hq : q ≤ 2 ^ 64)
    (A : Adversary (W1.w1Priced dv salt : CostModel Q R S)) (hA : A.Bounded U T q)
    (hwin : δ q U.layout.N < pr (fun x : (Q → R) × S =>
      T / (1 - g₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.trace U x.1 x.2) u))) :
    ¬ TTOutW1 dv salt P D g₀ δ := fun h =>
  absurd (h U hU T hT q hq A hA) (not_le.mpr hwin)

theorem not_gGap_of_win [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {D : Layout → Prop} {g : ℝ} {δ : ℕ → ℕ → ℝ}
    (U : Workload) (hU : U.InDomain P D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ) (hq : q ≤ 2 ^ 64)
    (A : Adversary CM) (hA : A.Bounded U T q)
    (hwin : δ q U.layout.N < pr (fun x : (Q → R) × S => Wins P U g T (A.trace U x.1 x.2) x.1 x.2)) :
    ¬ GGap CM P D g δ := fun h =>
  absurd (h U hU T hT q hq A hA) (not_le.mpr hwin)

theorem not_gGapPerCoin_of_win [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {D : Layout → Prop} {g : ℝ} {δ : ℕ → ℕ → ℝ}
    (U : Workload) (hU : U.InDomain P D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ) (hq : q ≤ 2 ^ 64)
    (A : Adversary CM)
    (hcalls : ∀ (H : Q → R) (s : S), A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q)
    (hwin : δ q U.layout.N < pr (fun x : (Q → R) × S =>
      CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2 ≤ T ∧ Wins P U g T (A.trace U x.1 x.2) x.1 x.2)) :
    ¬ GGapPerCoin CM P D g δ := fun h =>
  absurd (h U hU T hT q hq A hcalls) (not_le.mpr hwin)

def tabU : Workload := ⟨⟨1, fun _ => ⟨1, 1, 1⟩, fun _ => 0⟩, fun _ _ _ => 0⟩

/-- A protocol whose reference words are `0`, the words a fresh memory holds, whose rules and cap always pass, and
which credits each unit `1` and charges it `1` of honest work. -/
def tabP : Protocol Unit Unit Unit where
  ticket := fun _ _ _ _ _ _ _ => Fp8Atom.wordQ 0
  useful := fun _ _ _ _ _ _ _ => Fp8Atom.wordQ 0
  actOK := fun _ _ => True
  fixedOK := fun _ _ _ => True
  credit := fun _ _ _ _ _ => 1
  inertBoundOK := fun _ _ _ _ _ => True
  Wref := fun _ _ => 1

def emptyProg : CodeImage := ⟨fun _ => [], fun _ => .done, fun _ => by norm_num [W1.Code.size]⟩

def emptyAdv (dv : W1Device) : Adversary (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit) :=
  ⟨.done [], emptyProg, fun _ _ _ => 0⟩

theorem emptyAdv_cost (dv : W1Device) (H : Unit → Unit) (s : Unit) :
    (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit).cost (emptyAdv dv).online H tabU
      (emptyAdv dv).act ((emptyAdv dv).pre.eval H) s = 0 := by
  simp [W1.w1Priced, w1Read, final, emptyAdv, emptyProg, exec, QTree.eval]

theorem emptyAdv_bounded (dv : W1Device) : (emptyAdv dv).Bounded tabU 0 0 := by
  intro H s
  refine ⟨(emptyAdv_cost dv H s).le, ?_⟩
  simp [emptyAdv, W1.w1Priced, QTree.calls]

theorem emptyAdv_mem (dv : W1Device) (H : Unit → Unit) (s : Unit) :
    0 ∈ correctSet tabP tabU H s ((emptyAdv dv).trace tabU H s) := by
  simp [correctSet, Correct, Adversary.trace, W1.w1Priced, final, readout, emptyAdv, emptyProg, exec, init,
    tabP, tabU, ticketAt, usefulAt, QTree.eval]

theorem emptyAdv_correctSet (dv : W1Device) (H : Unit → Unit) (s : Unit) :
    correctSet tabP tabU H s ((emptyAdv dv).trace tabU H s) = {0} := by
  ext u
  rw [Finset.mem_singleton]
  constructor
  · intro hu
    simp only [correctSet, Finset.mem_filter, Finset.mem_range] at hu
    have h1 : u < 1 := hu.1
    omega
  · rintro rfl
    exact emptyAdv_mem dv H s

theorem tabU_inDomain : tabU.InDomain tabP (fun _ => True) := by
  simp [Workload.InDomain, tabU]

/-- **TT_OUT at W1, as a schema, fails**: at `tabP` the empty program wins on every coin at cost `0`. -/
theorem schema_not_ttOutW1 (dv : W1Device) (g₀ : ℝ) {δ : ℕ → ℕ → ℝ} (hδ : δ 0 1 < 1) :
    ¬ TTOutW1 dv (fun _ : Unit => []) tabP (fun _ => True) g₀ δ := by
  refine not_ttOutW1_of_win tabU tabU_inDomain 0 le_rfl 0 (by norm_num) (emptyAdv dv) (emptyAdv_bounded dv) ?_
  rw [pr_eq_one]
  · exact hδ
  · intro x
    simp only [emptyAdv_correctSet]
    simp [tabP]

theorem schema_not_gGap (dv : W1Device) (g : ℝ) {δ : ℕ → ℕ → ℝ} (hδ : δ 0 1 < 1) :
    ¬ GGap (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit) tabP (fun _ => True) g δ := by
  refine not_gGap_of_win tabU tabU_inDomain 0 le_rfl 0 (by norm_num) (emptyAdv dv) (emptyAdv_bounded dv) ?_
  rw [pr_eq_one]
  · exact hδ
  · intro x
    simp only [Wins, emptyAdv_correctSet]
    simp [tabP]

theorem schema_not_gGapPerCoin (dv : W1Device) (g : ℝ) {δ : ℕ → ℕ → ℝ} (hδ : δ 0 1 < 1) :
    ¬ GGapPerCoin (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit) tabP (fun _ => True) g δ := by
  refine not_gGapPerCoin_of_win tabU tabU_inDomain 0 le_rfl 0 (by norm_num) (emptyAdv dv) ?_ ?_
  · intro H s
    simp [emptyAdv, W1.w1Priced, QTree.calls]
  · rw [pr_eq_one]
    · exact hδ
    · intro x
      refine ⟨(emptyAdv_cost dv x.1 x.2).le, ?_⟩
      simp only [Wins, emptyAdv_correctSet]
      simp [tabP]

end Pouw.PearlC.W1Tabulate
