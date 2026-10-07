import Definitions.Pouw.Barrier.Pinned
import Proofs.Pouw.NCP.Proofs
import Proofs.Pouw.Prob
import Models.Pouw.Guarantees.Barrier

/-!
# The degenerate-input barrier: proofs

* `genericPr`, `generic`: the barrier, from the definition of `Wins`.
* In the abstract model: the zero-input adversaries of classes (a), (b) and (c), each a plan per unit, and
  `H_ref` plans for class (a) at `2·W_mm` and `3·W_mm` and for class (b) at `t·W_mm`.
-/

namespace Pouw.Barrier.Proofs

open Finset Pouw.Proofs

variable {Q R S : Type}

/-! ## The generic barrier -/

theorem wins_of_all_correct (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) {γ c₀ : ℝ}
    (hc : 0 ≤ c₀) (hW : 0 < totalWref P U.layout) (hγ : γ < 1 - c₀ / totalWref P U.layout)
    (hK : correctSet P U H s τ = range U.layout.N) : Wins P U γ c₀ τ H s := by
  unfold Wins
  rw [hK]
  change c₀ / (1 - γ) < totalWref P U.layout
  have h1 : c₀ / totalWref P U.layout < 1 - γ := by linarith
  have hpos : 0 < 1 - γ := lt_of_le_of_lt (div_nonneg hc hW.le) h1
  rw [div_lt_iff₀ hpos]
  rw [div_lt_iff₀ hW] at h1
  linarith [mul_comm (1 - γ) (totalWref P U.layout)]

/-- **The generic barrier, in probability.** -/
theorem genericPr : Pinned.GenericPr := by
  intro Q R S _ _ _ _ CM P D U A c₀ q hU hc hq hA hW γ hγ η hG
  exact le_trans (pr_mono fun x hx => wins_of_all_correct P U x.1 x.2 _ hc hW hγ hx) (hG U hU c₀ hc q hq A hA)

/-- **The generic barrier.** -/
theorem generic : Pouw.Guarantees.Barrier.Generic := by
  intro Q R S _ _ _ _ _ _ CM P D U A c₀ q hpt hW γ hγ η hG
  obtain ⟨hU, hc, hq, hA, hall⟩ := hpt
  have h := genericPr Q R S CM P D U A c₀ q hU hc hq hA hW γ hγ η hG
  have h1 : pr (fun x : (Q → R) × S => correctSet P U x.1 x.2 (A.transcript U x.1 x.2) = range U.layout.N) = 1 :=
    pr_eq_one fun x => hall x.1 x.2
  rwa [h1] at h

theorem gamma_ratio {γ c₀ W T a b : ℝ} (hT : 0 < T) (hb : 0 < b) (ha : 0 ≤ a) (hc : c₀ = a * T) (hW : b * T ≤ W)
    (hγ : γ < 1 - a / b) : γ < 1 - c₀ / W := by
  have hW0 : 0 < W := lt_of_lt_of_le (mul_pos hb hT) hW
  have : c₀ / W ≤ a / b := by
    rw [div_le_div_iff₀ hW0 hb, hc]
    nlinarith [mul_le_mul_of_nonneg_left hW ha]
  linarith

/-! ## The abstract model -/

theorem regs_eq (p : Plan) (A B : Mat) (nz : ℕ → Mat) (t : ℕ) :
    p.regs A B nz t =
      if t < p.nReg then (p.reg t).eval A B nz (fun i => if i < t then p.regs A B nz i else 0) else 0 := by
  rw [Plan.regs]
  simp only [dite_eq_ite]

/-- An adversary that commits the plan `f (shape u)` on every unit (activation input `0`, no preprocessing). -/
def planAdv (N : NoiseSrc Q R S) (f : Shape → Plan) : Adversary (gemmModel N) :=
  ⟨.done [], fun L u => if u < L.N then some (f (L.shape u)) else none, fun _ => 0⟩

theorem planAdv_bounded (N : NoiseSrc Q R S) (f : Shape → Plan) (U : Workload) :
    (planAdv N f).Bounded U (∑ u ∈ range U.layout.N, ((f (U.layout.shape u)).cost : ℝ)) 0 := fun H s => by
  refine ⟨le_of_eq ?_, by simp [planAdv, gemmModel, QTree.calls]⟩
  simp only [gemmModel]
  exact Finset.sum_congr rfl fun u hu => by simp [planAdv, Finset.mem_range.1 hu]

theorem planAdv_correct (N : NoiseSrc Q R S) (f : Shape → Plan) (P : Protocol Q R S) (U : Workload)
    (hc : ∀ (H : Q → R) (s : S), ∀ u < U.layout.N,
      Correct P U H s u ((f (U.layout.shape u)).commit N H s U (fun _ => 0) u)) :
    ∀ (H : Q → R) (s : S), correctSet P U H s ((planAdv N f).transcript U H s) = range U.layout.N := by
  intro H s
  ext u
  simp only [correctSet, Finset.mem_filter, Finset.mem_range, and_iff_left_iff_imp]
  intro hu
  exact ⟨_, by simp [Adversary.transcript, gemmModel, planAdv, hu], hc H s u hu⟩

theorem zero_actOK (sh : Shape) : ActOK sh (0 : Mat) := fun _ _ _ _ => by unfold InInt7; norm_num

theorem sum_Ico_zero (k : ℕ) (f : ℕ → ℤ) : ∑ l ∈ Ico 0 (0 + k), f l = ∑ l ∈ range k, f l := by
  rw [zero_add, Finset.range_eq_Ico]

/-! ## Class (a) -/

/-- The zero-input plan of class (a): commit `0`, one product `(0 + E)(B + F)`, output `0`. -/
def addPlan (sh : Shape) : Plan :=
  ⟨some 0, 1, fun _ => .gemm sh 0 (.add .act (.noise 0)) (.add .wt (.noise 1)), .const 0⟩

theorem addPlan_cost (sh : Shape) : (addPlan sh).cost = sh.Wmm := by
  simp [addPlan, Plan.cost, MExpr.cost]

theorem addPlan_correct (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (U : Workload) (H : Q → R) (s : S) (u : ℕ) :
    Correct (addProto N W) U H s u ((addPlan (U.layout.shape u)).commit N H s U (fun _ => 0) u) := by
  refine ⟨⟨zero_actOK _, fun x hx => ?_⟩, fun i hi j hj => ?_⟩
  · simp only [addProto, Finset.mem_product, Finset.mem_singleton, Finset.mem_range] at hx
    obtain ⟨h0, hi, hj⟩ := hx
    simp only [Plan.commit, addProto]
    rw [h0, regs_eq]
    simp [addPlan, MExpr.eval, hi, hj, mm]
  · simp [Plan.commit, addPlan, MExpr.eval, mm]

/-- **Class (a).** -/
theorem additive : Pinned.Additive := by
  intro Q R S _ _ _ _ _ _ N W p D U hp hU hS hW γ hγ η hG
  have hB := planAdv_bounded N addPlan U
  simp only [addPlan_cost] at hB
  have hpt : CheapPoint (gemmModel N) (addProto N W) D U (planAdv N addPlan)
      (∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) 0 :=
    ⟨hU, by positivity, by norm_num, hB,
      planAdv_correct N addPlan _ U fun H s u _ => addPlan_correct N W U H s u⟩
  have hWt : p * ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ) ≤ totalWref (addProto N W) U.layout := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun u hu => hW u (Finset.mem_range.1 hu)
  exact generic Q R S _ _ D U _ _ 0 hpt (lt_of_lt_of_le (by positivity) hWt) γ
    (gamma_ratio hS (by linarith) zero_le_one (one_mul _).symm hWt hγ) η hG

/-! ## Class (a): `H_ref` plans -/

theorem addProto_complete (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (D : Layout → Prop) :
    (addProto N W).Complete D := by
  intro U _ u _ H s A _
  refine ⟨fun x hx => ?_, fun i hi j hj => ?_⟩
  · simp only [addProto, Finset.mem_product, Finset.mem_singleton, Finset.mem_range] at hx ⊢
    obtain ⟨h0, hi, hj⟩ := hx
    simp [h0, corner, hi, hj]
  · simp [addProto, corner, hi, hj]

/-- The checked product of class (a) as a plan register. -/
def addBase (sh : Shape) : MExpr := .gemm sh 0 (.add .act (.noise 0)) (.add .wt (.noise 1))

/-- `H_ref` at `2·W_mm`: the checked product and `A·B`. -/
def refPlan2 (sh : Shape) : Plan := ⟨none, 1, fun _ => addBase sh, .gemm sh 0 .act .wt⟩

/-- `H_ref` at `3·W_mm`: the checked product and the standard decode `C' − A·F − E·(B + F)`. -/
def refPlan3 (sh : Shape) : Plan :=
  ⟨none, 1, fun _ => addBase sh,
    .add (.add (.reg 0) (.smul (-1) (.gemm sh 0 .act (.noise 1))))
      (.smul (-1) (.gemm sh 0 (.noise 0) (.add .wt (.noise 1))))⟩

theorem refPlan_checked (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (U : Workload) (H : Q → R) (s : S)
    (act : ℕ → Mat) (u : ℕ) (p : Plan) (hp : p.nReg = 1) (hr : p.reg 0 = addBase (U.layout.shape u)) :
    p.regs (act u) (wtOf U u) (N H s U u (act u)) = (addProto N W).refChecked H s U u (act u) := by
  funext t i j
  rw [regs_eq, hp]
  by_cases ht : t = 0
  · subst ht
    simp only [zero_lt_one, ite_true, hr, addProto, addBase, MExpr.eval, corner, mm, sum_Ico_zero, Pi.add_apply]
  · simp [addProto, ht]

theorem refPlan_run (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (f : Shape → Plan) (U : Workload) (act : ℕ → Mat)
    (H : Q → R) (s : S) (hc : ∀ u < U.layout.N, (f (U.layout.shape u)).commit N H s U act u
      = ⟨act u, (addProto N W).refChecked H s U u (act u), (addProto N W).refOut H s U u (act u)⟩) :
    (gemmModel N).run (fun L u => if u < L.N then some (f (L.shape u)) else none) H U act [] s
      = honestTranscript (addProto N W) H s U act := by
  funext u
  by_cases hu : u < U.layout.N
  · simp only [gemmModel, honestTranscript, hu, ite_true, Option.map_some]
    rw [hc u hu]
  · simp [gemmModel, honestTranscript, hu]

theorem refPlan_cost (N : NoiseSrc Q R S) (f : Shape → Plan) (c : ℕ) (hf : ∀ sh, (f sh).cost = c * sh.Wmm)
    (U : Workload) (act : ℕ → Mat) (H : Q → R) (s : S) :
    (gemmModel N).cost (fun L u => if u < L.N then some (f (L.shape u)) else none) H U act [] s
      = ∑ u ∈ range U.layout.N, ((c * (U.layout.shape u).Wmm : ℕ) : ℝ) := by
  simp only [gemmModel]
  exact Finset.sum_congr rfl fun u hu => by simp [Finset.mem_range.1 hu, hf]

theorem refPlan_commit (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (U : Workload) (H : Q → R) (s : S)
    (act : ℕ → Mat) (u : ℕ) (p : Plan) (hact : p.act = none) (hp : p.nReg = 1)
    (hr : p.reg 0 = addBase (U.layout.shape u))
    (hout : p.out.eval (act u) (wtOf U u) (N H s U u (act u)) ((addProto N W).refChecked H s U u (act u))
      = (addProto N W).refOut H s U u (act u)) :
    p.commit N H s U act u
      = ⟨act u, (addProto N W).refChecked H s U u (act u), (addProto N W).refOut H s U u (act u)⟩ := by
  have hc := refPlan_checked N W U H s act u p hp hr
  simp only [Plan.commit, hact, Option.getD_none]
  rw [hc, hout]

theorem refPlan2_out (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (U : Workload) (H : Q → R) (s : S) (A : Mat)
    (u : ℕ) (rg : ℕ → Mat) :
    (refPlan2 (U.layout.shape u)).out.eval A (wtOf U u) (N H s U u A) rg = (addProto N W).refOut H s U u A := by
  funext i j
  simp [refPlan2, addProto, MExpr.eval, corner, mm]

theorem refPlan3_out (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (U : Workload) (H : Q → R) (s : S) (A : Mat)
    (u : ℕ) :
    (refPlan3 (U.layout.shape u)).out.eval A (wtOf U u) (N H s U u A) ((addProto N W).refChecked H s U u A)
      = (addProto N W).refOut H s U u A := by
  funext i j
  simp only [refPlan3, MExpr.eval, addProto, corner, ite_true]
  split_ifs with h
  · simp only [mm, sum_Ico_zero, Pi.add_apply, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun l _ => by ring
  · ring

/-- **Class (a)'s witness.** -/
theorem additiveWitness : Pinned.AdditiveWitness := by
  intro Q R S N D
  refine ⟨addProto_complete N _ D, ⟨fun L u => if u < L.N then some (refPlan2 (L.shape u)) else none,
    fun U _ act _ H s => ⟨refPlan_run N _ refPlan2 U act H s fun u _ => ?_, ?_, le_rfl⟩⟩,
    ⟨fun L u => if u < L.N then some (refPlan3 (L.shape u)) else none,
    fun U _ act _ H s => ⟨refPlan_run N _ refPlan3 U act H s fun u _ => ?_, ?_, le_rfl⟩⟩⟩
  · exact refPlan_commit N _ U H s act u _ rfl rfl rfl (refPlan2_out N _ U H s (act u) u _)
  · rw [refPlan_cost N refPlan2 2 (fun sh => by simp [refPlan2, addBase, Plan.cost, MExpr.cost]; ring)]
    rfl
  · exact refPlan_commit N _ U H s act u _ rfl rfl rfl (refPlan3_out N _ U H s (act u) u)
  · rw [refPlan_cost N refPlan3 3 (fun sh => by simp [refPlan3, addBase, Plan.cost, MExpr.cost]; ring)]
    rfl

/-! ## Class (b) -/

/-- The sum `f 0 + ⋯ + f (n − 1)` as an expression. -/
def sumE : ℕ → (ℕ → MExpr) → MExpr
  | 0, _ => .const 0
  | n + 1, f => .add (sumE n f) (f n)

theorem eval_sumE (A B : Mat) (nz rg : ℕ → Mat) (n : ℕ) (f : ℕ → MExpr) :
    (sumE n f).eval A B nz rg = fun a b => ∑ x ∈ range n, (f x).eval A B nz rg a b := by
  induction n with
  | zero => funext a b; simp [sumE, MExpr.eval]
  | succ n ih => funext a b; simp only [sumE, MExpr.eval, ih, Finset.sum_range_succ]

theorem cost_sumE (n : ℕ) (f : ℕ → MExpr) : (sumE n f).cost = ∑ x ∈ range n, (f x).cost := by
  induction n with
  | zero => simp [sumE, MExpr.cost]
  | succ n ih => simp [sumE, MExpr.cost, ih, Finset.sum_range_succ]

/-- Checked product `i` as a plan register. -/
def relBase (sh : Shape) (i : ℕ) : MExpr := .gemm sh 0 (.add .act (.noise (2 * i))) (.add .wt (.noise (2 * i + 1)))

/-- The zero-input plan of class (b): commit `0`, the first `t − 1` products, the last one from the relation, output
`0`. -/
def relPlan (t : ℕ) (c : ℕ → ℤ) (sh : Shape) : Plan :=
  ⟨some 0, t, fun i => if i + 1 < t then relBase sh i
    else .smul (-c (t - 1)) (sumE (t - 1) fun i => .smul (c i) (.reg i)), .const 0⟩

theorem relPlan_act (t : ℕ) (c : ℕ → ℤ) (sh : Shape) : (relPlan t c sh).act = some 0 := rfl

theorem relPlan_nReg (t : ℕ) (c : ℕ → ℤ) (sh : Shape) : (relPlan t c sh).nReg = t := rfl

theorem relPlan_reg (t : ℕ) (c : ℕ → ℤ) (sh : Shape) (i : ℕ) :
    (relPlan t c sh).reg i = if i + 1 < t then relBase sh i
      else .smul (-c (t - 1)) (sumE (t - 1) fun i => .smul (c i) (.reg i)) := rfl

theorem relPlan_cost (t : ℕ) (c : ℕ → ℤ) (sh : Shape) (ht : 1 ≤ t) : (relPlan t c sh).cost = (t - 1) * sh.Wmm := by
  obtain ⟨t', rfl⟩ : ∃ t', t = t' + 1 := ⟨t - 1, by omega⟩
  have h1 : ∀ i ∈ range t', ((relPlan (t' + 1) c sh).reg i).cost = sh.Wmm := fun i hi => by
    rw [relPlan_reg, ite_eq_left (by simpa using Finset.mem_range.1 hi)]
    simp [relBase, MExpr.cost]
  have h2 : ((relPlan (t' + 1) c sh).reg t').cost = 0 := by
    rw [relPlan_reg, ite_eq_right (by omega)]
    simp [MExpr.cost, cost_sumE]
  rw [Plan.cost, relPlan_nReg, Finset.sum_range_succ, Finset.sum_congr rfl h1, h2]
  simp [relPlan, MExpr.cost]

theorem rel_zero {t : ℕ} {c : ℕ → ℤ} {N : NoiseSrc Q R S} (hN : RelNoise t c N) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (a b : ℕ) : ∑ x ∈ range t, c x * relProd N H s U u 0 x a b = 0 := by
  obtain ⟨-, hN⟩ := hN
  obtain ⟨hE, -, hEF⟩ := hN H s U u 0
  have h1 : ∑ x ∈ range t, c x * ∑ l ∈ range (U.layout.shape u).k, N H s U u 0 (2 * x) a l * wtOf U u l b = 0 := by
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_eq_zero fun l _ => ?_
    rw [show ∑ x ∈ range t, c x * (N H s U u 0 (2 * x) a l * wtOf U u l b)
        = (∑ x ∈ range t, c x * N H s U u 0 (2 * x) a l) * wtOf U u l b by
          rw [Finset.sum_mul]; exact Finset.sum_congr rfl fun _ _ => by ring,
      hE a l, zero_mul]
  calc ∑ x ∈ range t, c x * relProd N H s U u 0 x a b
      = ∑ x ∈ range t, (c x * ∑ l ∈ range (U.layout.shape u).k, N H s U u 0 (2 * x) a l * wtOf U u l b
          + c x * mm (U.layout.shape u).k (N H s U u 0 (2 * x)) (N H s U u 0 (2 * x + 1)) a b) :=
        Finset.sum_congr rfl fun x _ => by
          simp only [relProd, mm, Pi.add_apply, Pi.zero_apply, zero_add, mul_add, Finset.sum_add_distrib]
    _ = 0 := by rw [Finset.sum_add_distrib, h1, hEF a b, add_zero]

theorem relPlan_regs_base {t : ℕ} (c : ℕ → ℤ) (N : NoiseSrc Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    {i : ℕ} (hi : i + 1 < t) {a b : ℕ} (ha : a < (U.layout.shape u).m) (hb : b < (U.layout.shape u).n) :
    (relPlan t c (U.layout.shape u)).regs 0 (wtOf U u) (N H s U u 0) i a b = relProd N H s U u 0 i a b := by
  rw [regs_eq, relPlan_nReg, ite_eq_left (by omega), relPlan_reg, ite_eq_left hi]
  simp only [relBase, MExpr.eval, ha, hb, and_self, ite_true, relProd, mm, sum_Ico_zero, Pi.add_apply]

theorem relPlan_correct {t : ℕ} {c : ℕ → ℤ} {N : NoiseSrc Q R S} (W : Layout → ℕ → ℕ) (ht : 1 ≤ t)
    (hc : c (t - 1) * c (t - 1) = 1) (hN : RelNoise t c N) (U : Workload) (H : Q → R) (s : S) (u : ℕ) :
    Correct (relProto t c N W) U H s u ((relPlan t c (U.layout.shape u)).commit N H s U (fun _ => 0) u) := by
  refine ⟨⟨zero_actOK _, fun x hx => ?_⟩, fun i hi j hj => ?_⟩
  · simp only [relProto, Finset.mem_product, Finset.mem_range] at hx
    obtain ⟨hxt, ha, hb⟩ := hx
    simp only [Plan.commit, relProto, relPlan_act, Option.getD_some]
    by_cases hlast : x.1 + 1 < t
    · exact relPlan_regs_base c N H s U u hlast ha hb
    · have hx1 : x.1 = t - 1 := by omega
      rw [hx1, regs_eq, relPlan_nReg, ite_eq_left (by omega), relPlan_reg, ite_eq_right (by omega)]
      simp only [MExpr.eval, eval_sumE]
      have hs : ∑ y ∈ range (t - 1), c y * (if y < t - 1 then (relPlan t c (U.layout.shape u)).regs 0
            (U.weight (U.layout.wid u)) (N H s U u 0) y else 0) x.2.1 x.2.2
          = ∑ y ∈ range (t - 1), c y * relProd N H s U u 0 y x.2.1 x.2.2 :=
        Finset.sum_congr rfl fun y hy => by
          rw [ite_eq_left (Finset.mem_range.1 hy), relPlan_regs_base c N H s U u (by have := Finset.mem_range.1 hy; omega)
            ha hb]
      rw [hs]
      have hz := rel_zero hN H s U u x.2.1 x.2.2
      rw [show t = t - 1 + 1 by omega, Finset.sum_range_succ] at hz
      have : ∑ y ∈ range (t - 1), c y * relProd N H s U u 0 y x.2.1 x.2.2
          = -(c (t - 1) * relProd N H s U u 0 (t - 1) x.2.1 x.2.2) := by linarith
      rw [this]
      linear_combination (relProd N H s U u 0 (t - 1) x.2.1 x.2.2) * hc
  · simp [Plan.commit, relPlan, MExpr.eval, mm]

/-- **Class (b).** -/
theorem relation : Pinned.Relation := by
  intro Q R S _ _ _ _ _ _ t c N W D U ht hc hN hU hS hW γ hγ η hG
  have hB := planAdv_bounded N (relPlan t c) U
  simp only [relPlan_cost t c _ ht] at hB
  push_cast [Nat.cast_sub ht] at hB
  rw [← Finset.mul_sum] at hB
  have hpt : CheapPoint (gemmModel N) (relProto t c N W) D U (planAdv N (relPlan t c))
      (((t : ℝ) - 1) * ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) 0 :=
    ⟨hU, mul_nonneg (by have : (1 : ℝ) ≤ t := (by exact_mod_cast ht); linarith) hS.le, by norm_num, hB,
      planAdv_correct N _ _ U fun H s u _ => relPlan_correct W ht hc hN U H s u⟩
  have hWt : (t : ℝ) * ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)
      ≤ totalWref (relProto t c N W) U.layout := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun u hu => hW u (Finset.mem_range.1 hu)
  have ht' : (1 : ℝ) ≤ t := by exact_mod_cast ht
  refine generic Q R S _ _ D U _ _ 0 hpt (lt_of_lt_of_le (by positivity) hWt) γ
    (gamma_ratio hS (by linarith) (by linarith) rfl hWt ?_) η hG
  rw [show 1 - ((t : ℝ) - 1) / t = 1 / t by field_simp; ring]
  exact hγ

/-- **The lead's noise with output-only checks is class (b).** -/
theorem relationWitness : Pinned.RelationWitness := by
  intro Q R S e f r
  refine ⟨⟨by simp [Finset.sum_range_succ], fun H s U u A => ⟨fun a b => ?_, fun a b => ?_, fun a b => ?_⟩⟩,
    by norm_num⟩
  · simp [Finset.sum_range_succ, cancelNoise, NCP.E3]
  · simp [Finset.sum_range_succ, cancelNoise, NCP.F3]
  · have h := congrFun (congrFun (NCP.Proofs.cancel (U.layout.shape u).k 0 0 (e H s U u A) (f H s U u A)
      (r H s U u A)) a) b
    simp only [zero_add, Pi.add_apply, Pi.sub_apply] at h
    simp only [Finset.sum_range_succ, Finset.sum_range_zero, cancelNoise]
    have h0 : mm (U.layout.shape u).k (0 : Mat) 0 a b = 0 := by simp [mm]
    rw [h0] at h
    norm_num
    linarith

/-! ## Class (b): completeness and `H_ref` -/

theorem rel_sum {t : ℕ} {c : ℕ → ℤ} {N : NoiseSrc Q R S} (hN : RelNoise t c N) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (A : Mat) (a b : ℕ) :
    ∑ x ∈ range t, c x * relProd N H s U u A x a b = mm (U.layout.shape u).k A (wtOf U u) a b := by
  obtain ⟨h1, hN⟩ := hN
  obtain ⟨hE, hF, hEF⟩ := hN H s U u A
  have hx : ∀ x, relProd N H s U u A x a b = mm (U.layout.shape u).k A (wtOf U u) a b
      + ∑ l ∈ range (U.layout.shape u).k, A a l * N H s U u A (2 * x + 1) l b
      + ∑ l ∈ range (U.layout.shape u).k, N H s U u A (2 * x) a l * wtOf U u l b
      + mm (U.layout.shape u).k (N H s U u A (2 * x)) (N H s U u A (2 * x + 1)) a b := by
    intro x
    simp only [relProd, mm, Pi.add_apply, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun l _ => by ring
  simp_rw [hx, mul_add, Finset.sum_add_distrib]
  have t1 : ∑ x ∈ range t, c x * mm (U.layout.shape u).k A (wtOf U u) a b
      = mm (U.layout.shape u).k A (wtOf U u) a b := by rw [← Finset.sum_mul, h1, one_mul]
  have t2 : ∑ x ∈ range t, c x * ∑ l ∈ range (U.layout.shape u).k, A a l * N H s U u A (2 * x + 1) l b = 0 := by
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_eq_zero fun l _ => ?_
    rw [show ∑ x ∈ range t, c x * (A a l * N H s U u A (2 * x + 1) l b)
        = A a l * ∑ x ∈ range t, c x * N H s U u A (2 * x + 1) l b by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring,
      hF l b, mul_zero]
  have t3 : ∑ x ∈ range t, c x * ∑ l ∈ range (U.layout.shape u).k, N H s U u A (2 * x) a l * wtOf U u l b = 0 := by
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_eq_zero fun l _ => ?_
    rw [show ∑ x ∈ range t, c x * (N H s U u A (2 * x) a l * wtOf U u l b)
        = (∑ x ∈ range t, c x * N H s U u A (2 * x) a l) * wtOf U u l b by
          rw [Finset.sum_mul]; exact Finset.sum_congr rfl fun _ _ => by ring,
      hE a l, zero_mul]
  rw [t1, t2, t3, hEF a b]
  ring

theorem relProto_complete {t : ℕ} {c : ℕ → ℤ} {N : NoiseSrc Q R S} (hN : RelNoise t c N) (W : Layout → ℕ → ℕ)
    (D : Layout → Prop) : (relProto t c N W).Complete D := by
  intro U _ u _ H s A _
  refine ⟨fun x hx => ?_, fun i hi j hj => ?_⟩
  · simp only [relProto, Finset.mem_product, Finset.mem_range] at hx ⊢
    obtain ⟨ht, hi, hj⟩ := hx
    simp [ht, corner, hi, hj]
  · simp only [relProto, corner, hi, hj, and_self, ite_true]
    exact rel_sum hN H s U u A i j

/-- `H_ref` of class (b): the `t` checked products, and the useful output as their combination `∑ c_i·P_i`. -/
def relRef (t : ℕ) (c : ℕ → ℤ) (sh : Shape) : Plan := ⟨none, t, relBase sh, sumE t fun i => .smul (c i) (.reg i)⟩

theorem relRef_act (t : ℕ) (c : ℕ → ℤ) (sh : Shape) : (relRef t c sh).act = none := rfl

theorem relRef_nReg (t : ℕ) (c : ℕ → ℤ) (sh : Shape) : (relRef t c sh).nReg = t := rfl

theorem relRef_reg (t : ℕ) (c : ℕ → ℤ) (sh : Shape) : (relRef t c sh).reg = relBase sh := rfl

theorem relRef_out (t : ℕ) (c : ℕ → ℤ) (sh : Shape) :
    (relRef t c sh).out = sumE t fun i => .smul (c i) (.reg i) := rfl

theorem relRef_cost (c : ℕ → ℤ) (sh : Shape) : (relRef 3 c sh).cost = 3 * sh.Wmm := by
  simp [Plan.cost, relRef_nReg, relRef_reg, relRef_out, cost_sumE, relBase, MExpr.cost]

theorem relRef_regs (t : ℕ) (c : ℕ → ℤ) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (U : Workload) (H : Q → R)
    (s : S) (A : Mat) (u : ℕ) :
    (relRef t c (U.layout.shape u)).regs A (wtOf U u) (N H s U u A) = (relProto t c N W).refChecked H s U u A := by
  funext x i j
  rw [regs_eq, relRef_nReg, relRef_reg]
  by_cases hx : x < t
  · simp only [hx, ite_true, relBase, MExpr.eval, relProto, corner, relProd, mm, sum_Ico_zero, Pi.add_apply]
  · simp [relProto, hx]

theorem relRef_eval_out (t : ℕ) (c : ℕ → ℤ) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) (U : Workload) (H : Q → R)
    (s : S) (A : Mat) (u : ℕ) :
    (relRef t c (U.layout.shape u)).out.eval A (wtOf U u) (N H s U u A) ((relProto t c N W).refChecked H s U u A)
      = (relProto t c N W).refOut H s U u A := by
  funext i j
  rw [relRef_out, eval_sumE]
  by_cases h : i < (U.layout.shape u).m ∧ j < (U.layout.shape u).n
  · simp only [MExpr.eval, relProto, corner, h, and_self, ite_true]
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [ite_eq_left (Finset.mem_range.1 hx)]
    simp only [corner]
    rw [ite_eq_left h]
  · simp only [MExpr.eval, relProto, corner]
    rw [ite_eq_right h]
    refine Finset.sum_eq_zero fun x hx => ?_
    rw [ite_eq_left (Finset.mem_range.1 hx)]
    simp only [corner]
    rw [ite_eq_right h, mul_zero]

theorem plan_run (N : NoiseSrc Q R S) (P : Protocol Q R S) (f : Shape → Plan) (U : Workload) (act : ℕ → Mat)
    (H : Q → R) (s : S) (hc : ∀ u < U.layout.N, (f (U.layout.shape u)).commit N H s U act u
      = ⟨act u, P.refChecked H s U u (act u), P.refOut H s U u (act u)⟩) :
    (gemmModel N).run (fun L u => if u < L.N then some (f (L.shape u)) else none) H U act [] s
      = honestTranscript P H s U act := by
  funext u
  by_cases hu : u < U.layout.N
  · simp only [gemmModel, honestTranscript, hu, ite_true, Option.map_some]
    rw [hc u hu]
  · simp [gemmModel, honestTranscript, hu]

/-- **Class (b) with the lead's noise is a real protocol.** -/
theorem relationAdmits : Pinned.RelationAdmits := by
  intro Q R S e f r D
  have hN := (relationWitness Q R S e f r).1
  refine ⟨relProto_complete hN _ D,
    ⟨fun L u => if u < L.N then some (relRef 3 (fun i => if i = 2 then -1 else 1) (L.shape u)) else none,
    fun U _ act _ H s => ⟨plan_run _ _ _ U act H s fun u _ => ?_, ?_, le_rfl⟩⟩⟩
  · simp only [Plan.commit, relRef_act, Option.getD_none]
    rw [relRef_regs 3 _ _ (fun L u => 3 * (L.shape u).Wmm), relRef_eval_out]
  · rw [refPlan_cost _ (relRef 3 (fun i => if i = 2 then -1 else 1)) 3 (relRef_cost _)]
    rfl

/-! ## Class (c) -/

/-- Step `t` of the running sums at the zero input: `E[:, S_t]·F[S_t, :]`. -/
def segStep (sh : Shape) (d t : ℕ) : MExpr := .gemm ⟨sh.m, d, sh.n⟩ (t * d) (.noise 0) (.noise 1)

/-- The zero-input plan of class (c): commit `0`, the first `L − 1` running sums step by step, the known last one
(`0`), output `0`. -/
def segPlan (L d : ℕ) (sh : Shape) : Plan :=
  ⟨some 0, L, fun t => if t + 1 < L then .add (.reg (t - 1)) (segStep sh d t) else .const 0, .const 0⟩

theorem segPlan_act (L d : ℕ) (sh : Shape) : (segPlan L d sh).act = some 0 := rfl

theorem segPlan_nReg (L d : ℕ) (sh : Shape) : (segPlan L d sh).nReg = L := rfl

theorem segPlan_reg (L d : ℕ) (sh : Shape) (t : ℕ) :
    (segPlan L d sh).reg t = if t + 1 < L then .add (.reg (t - 1)) (segStep sh d t) else .const 0 := rfl

theorem segPlan_cost (L d : ℕ) (sh : Shape) (hL : 1 ≤ L) : (segPlan L d sh).cost = (L - 1) * (sh.m * d * sh.n) := by
  obtain ⟨L', rfl⟩ : ∃ L', L = L' + 1 := ⟨L - 1, by omega⟩
  have h1 : ∀ i ∈ range L', ((segPlan (L' + 1) d sh).reg i).cost = sh.m * d * sh.n := fun i hi => by
    rw [segPlan_reg, ite_eq_left (by simpa using Finset.mem_range.1 hi)]
    simp [segStep, MExpr.cost, Shape.Wmm]
  have h2 : ((segPlan (L' + 1) d sh).reg L').cost = 0 := by
    rw [segPlan_reg, ite_eq_right (by omega)]
    simp [MExpr.cost]
  rw [Plan.cost, segPlan_nReg, Finset.sum_range_succ, Finset.sum_congr rfl h1, h2]
  simp [segPlan, MExpr.cost]

theorem segPlan_regs (L d : ℕ) (sh : Shape) (A B : Mat) (nz : ℕ → Mat) {a b : ℕ} (ha : a < sh.m) (hb : b < sh.n) :
    ∀ t, t + 1 < L → (segPlan L d sh).regs A B nz t a b = ∑ l ∈ range ((t + 1) * d), nz 0 a l * nz 1 l b := by
  intro t
  induction t with
  | zero =>
    intro h
    rw [regs_eq, segPlan_nReg, ite_eq_left (by omega), segPlan_reg, ite_eq_left h]
    simp only [MExpr.eval, lt_irrefl, ite_false, Pi.zero_apply, zero_add, segStep, ha, hb, and_self, ite_true,
      zero_mul, one_mul, Finset.range_eq_Ico]
  | succ t ih =>
    intro h
    rw [regs_eq, segPlan_nReg, ite_eq_left (by omega), segPlan_reg, ite_eq_left h]
    simp only [MExpr.eval, Nat.add_sub_cancel, show t < t + 1 by omega, ite_true, segStep, ha, hb, and_self]
    rw [ih (by omega), Finset.range_eq_Ico, Finset.range_eq_Ico,
      Finset.sum_Ico_consecutive _ (Nat.zero_le _) (show (t + 1) * d ≤ (t + 1) * d + d by omega)]
    congr 2
    ring

theorem segPlan_correct {L d : ℕ} {α β : Mat → Mat} {N : NoiseSrc Q R S} (W : Layout → ℕ → ℕ) (hL : 1 ≤ L)
    (hα : α 0 = 0) (hβ : β 0 = 0)
    (hcan : ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ), ∀ a < (U.layout.shape u).m, ∀ b < (U.layout.shape u).n,
      mm (L * d) (N H s U u 0 0) (N H s U u 0 1) a b = 0)
    (Lay : Layout) (H : Q → R) (s : S) (u : ℕ) :
    Correct (segProto L d α β N W) ⟨Lay, fun _ => 0⟩ H s u
      ((segPlan L d (Lay.shape u)).commit N H s ⟨Lay, fun _ => 0⟩ (fun _ => 0) u) := by
  refine ⟨⟨zero_actOK _, fun x hx => ?_⟩, fun i hi j hj => ?_⟩
  · simp only [segProto, Finset.mem_product, Finset.mem_range] at hx
    obtain ⟨hxt, ha, hb⟩ := hx
    simp only [Plan.commit, segProto, segPlan_act, Option.getD_some, wtOf, hα, hβ, zero_add]
    by_cases hlast : x.1 + 1 < L
    · rw [segPlan_regs L d _ _ _ _ ha hb x.1 hlast]
      simp [mm]
    · have hx1 : x.1 = L - 1 := by omega
      rw [hx1, regs_eq, segPlan_nReg, ite_eq_left (by omega), segPlan_reg, ite_eq_right (by omega)]
      simp only [MExpr.eval]
      rw [show L - 1 + 1 = L by omega]
      exact (hcan H s ⟨Lay, fun _ => 0⟩ u _ ha _ hb).symm
  · simp [Plan.commit, segPlan, MExpr.eval, mm]

/-- **Class (c).** -/
theorem segment : Pinned.Segment := by
  intro Q R S _ _ _ _ _ _ L d α β N W D Lay hL hα hβ hcan hU hS hW γ hγ η hG
  have hB := planAdv_bounded N (segPlan L d) ⟨Lay, fun _ => 0⟩
  simp only [segPlan_cost L d _ hL] at hB
  push_cast [Nat.cast_sub hL] at hB
  rw [← Finset.mul_sum] at hB
  have hL' : (1 : ℝ) ≤ L := by exact_mod_cast hL
  have hS' : 0 < ∑ u ∈ range Lay.N, (((Lay.shape u).m : ℝ) * d * (Lay.shape u).n) := by
    simpa using hS
  have hpt : CheapPoint (gemmModel N) (segProto L d α β N W) D ⟨Lay, fun _ => 0⟩ (planAdv N (segPlan L d))
      (((L : ℝ) - 1) * ∑ u ∈ range Lay.N, (((Lay.shape u).m : ℝ) * d * (Lay.shape u).n)) 0 :=
    ⟨hU, mul_nonneg (by linarith) hS'.le, by norm_num, hB,
      planAdv_correct N _ _ _ fun H s u _ => segPlan_correct W hL hα hβ hcan Lay H s u⟩
  have hWt : (L : ℝ) * ∑ u ∈ range Lay.N, (((Lay.shape u).m : ℝ) * d * (Lay.shape u).n)
      ≤ totalWref (segProto L d α β N W) Lay := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun u hu => le_trans (le_of_eq ?_) (hW u (Finset.mem_range.1 hu))
    push_cast
    ring
  refine generic Q R S _ _ D _ _ _ 0 hpt (lt_of_lt_of_le (by positivity) hWt) γ
    (gamma_ratio hS' (by linarith) (by linarith) rfl hWt ?_) η hG
  rw [show 1 - ((L : ℝ) - 1) / L = 1 / L by field_simp; ring]
  exact hγ

/-! ## The generic barrier's hypotheses are satisfiable -/

/-- One unit of shape `(1, 1, 1)` with zero weights. -/
def oneUnit : Workload := ⟨⟨1, fun _ => ⟨1, 1, 1⟩, fun _ => 0⟩, fun _ => 0⟩

/-- **`Generic`'s hypotheses are jointly satisfiable.** -/
theorem genericWitness : Pinned.GenericWitness := by
  refine ⟨fun _ _ _ _ _ _ => 0, addProto (fun _ _ _ _ _ _ => 0) fun L u => 2 * (L.shape u).Wmm, fun _ => True,
    oneUnit, planAdv _ addPlan, 1, addProto_complete _ _ _, (additiveWitness Unit Unit Unit _ _).2.1, ⟨?_, by norm_num,
    by norm_num, ?_, planAdv_correct _ addPlan _ _ fun H s u _ => addPlan_correct _ _ oneUnit H s u⟩, ?_, ?_⟩
  · refine ⟨by norm_num [oneUnit], trivial, fun u hu l _ j _ => ?_⟩
    unfold InInt7; norm_num [oneUnit]
  · have hB := planAdv_bounded (Q := Unit) (R := Unit) (S := Unit) (fun _ _ _ _ _ _ => 0) addPlan oneUnit
    simpa [addPlan_cost, oneUnit, Shape.Wmm] using hB
  · simp [totalWref, addProto, oneUnit, Shape.Wmm]
  · simp [totalWref, addProto, oneUnit, Shape.Wmm]

/-! ## Class (a′): entrywise encodings -/

theorem affine_of_separable (s t g h : ℤ → ℤ) (α : ℤ) (hα : α ≠ 0)
    (hsep : ∀ x y, InInt7 x → InInt7 y → s x * t y = α * x * y + g x + h y) :
    ∀ x, InInt7 x → s x = s 0 + x * (s 1 - s 0) := by
  have p0 : InInt7 0 := by unfold InInt7; omega
  have p1 : InInt7 1 := by unfold InInt7; omega
  intro x hx
  have e1 := hsep x 1 hx p1
  have e0 := hsep x 0 hx p0
  have a1 := hsep 1 1 p1 p1
  have a0 := hsep 1 0 p1 p0
  have z1 := hsep 0 1 p0 p1
  have z0 := hsep 0 0 p0 p0
  have hD : t 1 - t 0 ≠ 0 := by
    intro hD
    have ht : t 1 = t 0 := by omega
    rw [ht] at a1 z1
    exact hα (by linarith)
  have key : (s x - s 0) * (t 1 - t 0) = (x * (s 1 - s 0)) * (t 1 - t 0) := by
    linear_combination e1 - e0 - z1 + z0 - x * (a1 - a0 - z1 + z0)
  have := mul_right_cancel₀ hD key
  linarith

/-- **Class (a′).** -/
theorem tablesAffine : Pinned.TablesAffine := fun s t g h α hα hsep =>
  ⟨affine_of_separable s t g h α hα hsep,
    affine_of_separable t s h g α hα fun y x hy hx => by linarith [hsep x y hx hy, mul_comm (s x) (t y)]⟩

/-- **`TablesAffine`'s witness.** -/
theorem tablesAffineWitness : Pinned.TablesAffineWitness :=
  ⟨id, id, fun _ => 0, fun _ => 0, 1, one_ne_zero, fun x y _ _ => by simp⟩

end Pouw.Barrier.Proofs
