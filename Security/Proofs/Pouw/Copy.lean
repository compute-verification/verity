import Definitions.Pouw.Assumptions.Pinned
import Proofs.Pouw.Prob

/-!
# R_copy (tt-attacks §6): the bound under Nz, and Nz for independent uniform bits
-/

namespace Pouw.Proofs

open Finset

variable {α Ω V : Type} [Fintype α]

theorem pr_or_disjoint (P P' : α → Prop) (h : ∀ x, ¬ (P x ∧ P' x)) :
    pr (fun x => P x ∨ P' x) = pr P + pr P' := by
  classical
  rw [pr_eq_card (fun x => P x ∨ P' x) (univ.filter P ∪ univ.filter P') (fun x => by simp),
    pr_eq_card P (univ.filter P) (fun x => by simp), pr_eq_card P' (univ.filter P') (fun x => by simp),
    Finset.card_union_of_disjoint (Finset.disjoint_filter.2 fun x _ hx hx' => h x ⟨hx, hx'⟩), Nat.cast_add, add_div]

namespace RCopy

variable (n : ℕ → ℕ) (X : Ω → ℕ → ℕ → V)

theorem classes_card_le (A : Copy.Adversary) (u : ℕ) : (Copy.classes n A u).card ≤ n u :=
  Finset.card_image_le.trans (Finset.card_range _).le

/-- A unit with `c` classes is correct with probability at most `2^−(n_u − c)`: every position that is not its
class's least position must equal that one. -/
theorem unit_le [Fintype Ω] (hNz : Copy.Nz n X) (A : Copy.Adversary) (u : ℕ) :
    pr (fun ω => Copy.Correct n X A ω u) ≤ (1 / 2 : ℝ) ^ (n u - (Copy.classes n A u).card) := by
  classical
  let repOf : ℕ → ℕ := fun c => if h : ∃ q, q < n u ∧ A.cls u q = c then Nat.find h else 0
  let rep : ℕ → ℕ := fun p => repOf (A.cls u p)
  have rep_spec : ∀ p < n u, rep p < n u ∧ A.cls u (rep p) = A.cls u p := fun p hp => by
    have h : ∃ q, q < n u ∧ A.cls u q = A.cls u p := ⟨p, hp, rfl⟩
    show repOf (A.cls u p) < n u ∧ A.cls u (repOf (A.cls u p)) = A.cls u p
    simp only [repOf, h, ↓reduceDIte]
    exact Nat.find_spec h
  have rep_rep : ∀ p < n u, rep (rep p) = rep p := fun p hp => congrArg repOf (rep_spec p hp).2
  let P := (range (n u)).filter (fun p => rep p ≠ p)
  have hg : ∀ p ∈ P, ∀ x y : ℕ → V, (∀ q, q ∉ P → x q = y q) → x (rep p) = y (rep p) := fun p hp x y h => by
    have hp' := Finset.mem_range.1 (Finset.mem_filter.1 hp).1
    refine h _ fun hr => ?_
    exact (Finset.mem_filter.1 hr).2 (rep_rep p hp')
  have h1 := hNz u P (Finset.filter_subset _ _) (fun p x => x (rep p)) hg
  have h2 : pr (fun ω => Copy.Correct n X A ω u) ≤ pr (fun ω => ∀ p ∈ P, X ω u p = X ω u (rep p)) :=
    pr_mono fun ω hc p hp => by
      have hp' := Finset.mem_range.1 (Finset.mem_filter.1 hp).1
      exact hc p hp' (rep p) (rep_spec p hp').1 (rep_spec p hp').2.symm
  have h3 : n u - (Copy.classes n A u).card ≤ P.card := by
    have hreps : ((range (n u)).filter (fun p => rep p = p)).card ≤ (Copy.classes n A u).card := by
      refine Finset.card_le_card_of_injOn (A.cls u) (fun p hp => ?_) (fun p hp p' hp' heq => ?_)
      · exact Finset.mem_image_of_mem _ (Finset.mem_filter.1 hp).1
      · have e1 : rep p = p := (Finset.mem_filter.1 hp).2
        have e2 : rep p' = p' := (Finset.mem_filter.1 hp').2
        have e3 : rep p = rep p' := congrArg repOf heq
        rw [← e1, ← e2, e3]
    have hsplit : ((range (n u)).filter (fun p => rep p = p)).card + P.card = n u := by
      have := Finset.card_filter_add_card_filter_not (s := range (n u)) (fun p => rep p = p)
      rw [Finset.card_range] at this
      exact this
    omega
  calc _ ≤ _ := h2
    _ ≤ _ := h1
    _ ≤ _ := pow_le_pow_of_le_one (by norm_num) (by norm_num) h3

theorem half_pow_le_rpow (k : ℕ) (x : ℝ) (h : x ≤ k) : (1 / 2 : ℝ) ^ k ≤ (2 : ℝ) ^ (-x) := by
  have e : (1 / 2 : ℝ) ^ k = (2 : ℝ) ^ (-(k : ℝ)) := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast, one_div, inv_pow]
  rw [e]
  exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)

end RCopy

/-- **R_copy's bound.** -/
theorem copyBound : Pinned.CopyBound := by
  intro Ω V _ n X N A γ₀ hγ0 hγ1 hNz
  classical
  have h1γ : 0 < 1 - γ₀ := by linarith
  let few := (range N).filter (fun u => ((Copy.classes n A u).card : ℝ) < (1 - γ₀) * n u)
  have hsub : ∀ ω, (Copy.cost n N A : ℝ) / (1 - γ₀) < 16 * ∑ u ∈ Copy.correctSet n X N A ω, (n u : ℝ) →
      ∃ u ∈ few, Copy.Correct n X A ω u := by
    intro ω h
    by_contra hne
    push Not at hne
    refine absurd h (not_lt.2 ?_)
    have hK : ∀ u ∈ Copy.correctSet n X N A ω, (n u : ℝ) ≤ (Copy.classes n A u).card / (1 - γ₀) := by
      intro u hu
      have hu' := Finset.mem_filter.1 hu
      have : u ∉ few := fun hf => hne u hf hu'.2
      have hle : (1 - γ₀) * n u ≤ (Copy.classes n A u).card := by
        by_contra hlt
        exact this (Finset.mem_filter.2 ⟨hu'.1, not_le.1 hlt⟩)
      rw [le_div_iff₀ h1γ]
      linarith
    calc 16 * ∑ u ∈ Copy.correctSet n X N A ω, (n u : ℝ)
        ≤ 16 * ∑ u ∈ Copy.correctSet n X N A ω, ((Copy.classes n A u).card : ℝ) / (1 - γ₀) := by
          gcongr with u hu; exact hK u hu
      _ ≤ 16 * ∑ u ∈ range N, ((Copy.classes n A u).card : ℝ) / (1 - γ₀) := by
          refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
          exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => by positivity
      _ = (Copy.cost n N A : ℝ) / (1 - γ₀) := by
          unfold Copy.cost
          push_cast
          simp only [div_eq_mul_inv]
          rw [← Finset.sum_mul]
          ring
  calc _ ≤ pr (fun ω => ∃ u ∈ few, Copy.Correct n X A ω u) := pr_mono hsub
    _ ≤ ∑ u ∈ few, pr (fun ω => Copy.Correct n X A ω u) := pr_exists_le_sum _ _
    _ ≤ ∑ u ∈ few, (2 : ℝ) ^ (-(γ₀ * n u)) := by
        refine Finset.sum_le_sum fun u hu => (RCopy.unit_le n X hNz A u).trans (RCopy.half_pow_le_rpow _ _ ?_)
        have hlt := (Finset.mem_filter.1 hu).2
        rw [Nat.cast_sub (RCopy.classes_card_le n A u)]
        linarith
    _ ≤ ∑ u ∈ range N, (2 : ℝ) ^ (-(γ₀ * n u)) :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => by positivity

namespace RCopy

/-- Flipping bit `a`. -/
def flip {M : ℕ} (a : Fin M) : (Fin M → Bool) ≃ (Fin M → Bool) where
  toFun ω := Function.update ω a (!ω a)
  invFun ω := Function.update ω a (!ω a)
  left_inv ω := by funext i; by_cases h : i = a <;> simp [Function.update, h]
  right_inv ω := by funext i; by_cases h : i = a <;> simp [Function.update, h]

theorem bits_flip_ne {M : ℕ} (a : Fin M) (ω : Fin M → Bool) (q : ℕ) (hq : q ≠ a) :
    Pouw.Copy.bits (flip a ω) q = Pouw.Copy.bits ω q := by
  unfold Pouw.Copy.bits
  split
  · next h =>
    have : (⟨q, h⟩ : Fin M) ≠ a := fun e => hq (congrArg Fin.val e)
    simp [flip, Function.update_of_ne this]
  · rfl

theorem bits_flip_self {M : ℕ} (a : Fin M) (ω : Fin M → Bool) :
    Pouw.Copy.bits (flip a ω) a ≠ Pouw.Copy.bits ω a := by
  unfold Pouw.Copy.bits
  simp only [a.isLt, ↓reduceDIte, Fin.eta, flip, Equiv.coe_fn_mk, Function.update_self]
  cases ω a <;> simp

/-- Independent uniform bits satisfy Nz: flipping one constrained bit maps the event for `P` into the event for `P`
without that bit, and out of the event for `P`. -/
theorem bits_nz (M : ℕ) (P : Finset ℕ) :
    P ⊆ range M → ∀ g : ℕ → (ℕ → ℕ) → ℕ,
      (∀ p ∈ P, ∀ x y : ℕ → ℕ, (∀ q, q ∉ P → x q = y q) → g p x = g p y) →
      pr (fun ω : Fin M → Bool => ∀ p ∈ P, Pouw.Copy.bits ω p = g p (Pouw.Copy.bits ω)) ≤ (1 / 2 : ℝ) ^ P.card := by
  classical
  refine Finset.induction_on P ?_ ?_
  · intro _ _ _
    simpa using pr_le_one _
  · intro a s ha ih hP g hg
    have haM : a < M := Finset.mem_range.1 (hP (Finset.mem_insert_self a s))
    let σ := flip (⟨a, haM⟩ : Fin M)
    let E := fun ω : Fin M → Bool => ∀ p ∈ insert a s, Pouw.Copy.bits ω p = g p (Pouw.Copy.bits ω)
    let E' := fun ω : Fin M → Bool => ∀ p ∈ s, Pouw.Copy.bits ω p = g p (Pouw.Copy.bits ω)
    have hagree : ∀ ω, ∀ q, q ∉ insert a s → Pouw.Copy.bits (σ ω) q = Pouw.Copy.bits ω q := fun ω q hq =>
      bits_flip_ne _ ω q fun e => hq (e ▸ Finset.mem_insert_self a s)
    have hIH : pr E' ≤ (1 / 2 : ℝ) ^ s.card :=
      ih (fun p hp => hP (Finset.mem_insert_of_mem hp)) g fun p hp x y hxy =>
        hg p (Finset.mem_insert_of_mem hp) x y fun q hq => hxy q fun hs => hq (Finset.mem_insert_of_mem hs)
    have hcover : ∀ ω, E ω ∨ E (σ ω) → E' ω := by
      rintro ω (h | h) p hp
      · exact h p (Finset.mem_insert_of_mem hp)
      · have hpa : p ≠ a := fun e => ha (e ▸ hp)
        rw [← bits_flip_ne (⟨a, haM⟩ : Fin M) ω p hpa, h p (Finset.mem_insert_of_mem hp)]
        exact hg p (Finset.mem_insert_of_mem hp) _ _ (hagree ω)
    have hdisj : ∀ ω, ¬ (E ω ∧ E (σ ω)) := by
      rintro ω ⟨h, h'⟩
      apply bits_flip_self (⟨a, haM⟩ : Fin M) ω
      show Pouw.Copy.bits (σ ω) a = Pouw.Copy.bits ω a
      rw [h a (Finset.mem_insert_self a s), h' a (Finset.mem_insert_self a s)]
      exact hg a (Finset.mem_insert_self a s) _ _ (hagree ω)
    have h2 : 2 * pr E ≤ pr E' := by
      have := pr_mono (P := fun ω => E ω ∨ E (σ ω)) hcover
      rw [pr_or_disjoint _ _ hdisj, pr_equiv σ E] at this
      linarith
    rw [Finset.card_insert_of_notMem ha, pow_succ]
    linarith

end RCopy

/-- **Nz is satisfiable.** -/
theorem copyNzWitness : Pinned.CopyNzWitness := fun M _ P hP g hg => RCopy.bits_nz M P hP g hg

end Pouw.Proofs
