module

public import Definitions.Core.Interaction
public import Proofs.Flock.Soundness.Audit.Periods
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Prob

/-! The monitoring period's step where a period's bound holds one more charge, paid once. -/

@[expose] public section

namespace FlockSoundness.Audit.Periods

open Claims

theorem Ex_le_of_forall {α : Type} [Fintype α] [Nonempty α] (f : α → ℝ) (E : ℝ) (h : ∀ a, f a ≤ E) : Ex f ≤ E :=
  (Ex_mono h).trans_eq (Ex_const E)

theorem Ex_fin_cons {T : ℕ} {Ω : Fin (T + 1) → Type} [∀ t, Fintype (Ω t)] (F : ((i : Fin (T + 1)) → Ω i) → ℝ) :
    Ex F = Ex fun a : Ω 0 => Ex fun ρ : (i : Fin T) → Ω i.succ => F (Fin.cons a ρ) := by
  have hc : (Fintype.card ((i : Fin (T + 1)) → Ω i) : ℝ) =
      Fintype.card (Ω 0) * Fintype.card ((i : Fin T) → Ω i.succ) := by
    rw [Fintype.card_pi, Fintype.card_pi, Fin.prod_univ_succ, Nat.cast_mul]
  have hs : ∑ ω : (i : Fin (T + 1)) → Ω i, F ω =
      ∑ a : Ω 0, ∑ ρ : (i : Fin T) → Ω i.succ, F (Fin.cons a ρ) := by
    exact ((Fin.consEquiv Ω).sum_comp F).symm.trans (Fintype.sum_prod_type _)
  unfold Ex
  rw [hc, hs, ← Finset.sum_div, div_div]
  congr 1
  ring

theorem ex_prod_add_le : ∀ {T : ℕ} {Ω : Fin T → Type} [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (u g w : (t : Fin T) → ((i : Fin T) → Ω i) → ℝ), Adapted u → Predictable g → Adapted w →
    (∀ t ω, 0 ≤ u t ω) → (∀ t ω, u t ω ≤ 1) → (∀ t ω, 0 ≤ g t ω) → (∀ t ω, 0 ≤ w t ω) →
    (∀ t ω, Ex (fun a : Ω t => u t (Function.update ω t a)) ≤
      g t ω + Ex fun a : Ω t => w t (Function.update ω t a)) →
    ∀ c : ℝ, 0 ≤ c →
    Ex (fun ω => (∏ t, u t ω) * if ∏ t, g t ω ≤ c then 1 else 0) ≤ c + Ex fun ω => ∑ t, w t ω
  | 0, Ω, _, _, u, g, w, _, _, _, _, _, _, _, _, c, hc => by
      simp only [Finset.univ_eq_empty, Finset.prod_empty, Finset.sum_empty, one_mul]
      rw [Ex_const, Ex_const]
      split_ifs with h1
      · linarith
      · linarith
  | T + 1, Ω, _, _, u, g, w, hu, hg, hw, hu0, hu1, hg0, hw0, h, c, hc => by
      let ω₀ : (i : Fin (T + 1)) → Ω i := fun i => Classical.arbitrary (Ω i)
      have hga : ∀ ω, g 0 ω = g 0 ω₀ := fun ω => hg.zero ω ω₀
      have hua : ∀ a ρ, u 0 (Fin.cons a ρ : (i : Fin (T + 1)) → Ω i) = u 0 (Function.update ω₀ 0 a) :=
        fun a ρ => hu.zero _ _ (by simp)
      have hwa : ∀ a ρ, w 0 (Fin.cons a ρ : (i : Fin (T + 1)) → Ω i) = w 0 (Function.update ω₀ 0 a) :=
        fun a ρ => hw.zero _ _ (by simp)
      have hP1 : ∀ (a : Ω 0) (ρ : (i : Fin T) → Ω i.succ), ∏ s : Fin T, u s.succ (Fin.cons a ρ) ≤ 1 :=
        fun a ρ => Finset.prod_le_one₀ (fun s _ => hu0 _ _) fun s _ => hu1 _ _
      have ih : ∀ (a : Ω 0) (c' : ℝ), 0 ≤ c' → Ex (fun ρ : (i : Fin T) → Ω i.succ =>
          (∏ s : Fin T, u s.succ (Fin.cons a ρ)) *
            if ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c' then 1 else 0) ≤
          c' + Ex fun ρ : (i : Fin T) → Ω i.succ => ∑ s : Fin T, w s.succ (Fin.cons a ρ) := fun a c' hc' =>
        ex_prod_add_le (Ω := fun i => Ω i.succ) _ _ _ (hu.tail a) (hg.tail a) (hw.tail a)
          (fun s ρ => hu0 _ _) (fun s ρ => hu1 _ _) (fun s ρ => hg0 _ _) (fun s ρ => hw0 _ _)
          (fun s ρ => by simpa only [Fin.cons_update] using h s.succ (Fin.cons a ρ)) c' hc'
      have hin1 : ∀ (a : Ω 0) (c' : ℝ), Ex (fun ρ : (i : Fin T) → Ω i.succ =>
          (∏ s : Fin T, u s.succ (Fin.cons a ρ)) *
            if ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c' then 1 else 0) ≤ 1 := fun a c' =>
        (Ex_mono (g := fun _ => (1 : ℝ)) fun ρ => by
          split_ifs
          · rw [mul_one]
            exact hP1 a ρ
          · rw [mul_zero]
            exact zero_le_one).trans (Ex_const 1).le
      have hBn : ∀ a : Ω 0, 0 ≤ Ex fun ρ : (i : Fin T) → Ω i.succ => ∑ s : Fin T, w s.succ (Fin.cons a ρ) :=
        fun a => Ex_nonneg fun ρ => Finset.sum_nonneg fun s _ => hw0 _ _
      have hb0 : 0 ≤ Ex fun a : Ω 0 => w 0 (Function.update ω₀ 0 a) := Ex_nonneg fun a => hw0 _ _
      have hBB : 0 ≤ Ex fun a : Ω 0 => Ex fun ρ : (i : Fin T) → Ω i.succ => ∑ s : Fin T, w s.succ (Fin.cons a ρ) :=
        Ex_nonneg hBn
      have hR : Ex (fun ω : (i : Fin (T + 1)) → Ω i => ∑ t, w t ω) =
          Ex (fun a : Ω 0 => w 0 (Function.update ω₀ 0 a)) +
            Ex fun a : Ω 0 => Ex fun ρ : (i : Fin T) → Ω i.succ => ∑ s : Fin T, w s.succ (Fin.cons a ρ) := by
        have h1 : ∀ a : Ω 0, Ex (fun ρ : (i : Fin T) → Ω i.succ => ∑ t, w t (Fin.cons a ρ)) =
            w 0 (Function.update ω₀ 0 a) +
              Ex fun ρ : (i : Fin T) → Ω i.succ => ∑ s : Fin T, w s.succ (Fin.cons a ρ) := fun a => by
          have h2 : (fun ρ : (i : Fin T) → Ω i.succ => ∑ t, w t (Fin.cons a ρ)) =
              fun ρ => w 0 (Function.update ω₀ 0 a) + ∑ s : Fin T, w s.succ (Fin.cons a ρ) := funext fun ρ => by
            rw [Fin.sum_univ_succ, hwa a ρ]
          rw [h2, Ex_add, Ex_const]
        rw [Ex_fin_cons, funext h1, Ex_add]
      have hL : Ex (fun ω : (i : Fin (T + 1)) → Ω i => (∏ t, u t ω) * if ∏ t, g t ω ≤ c then 1 else 0) =
          Ex fun a : Ω 0 => u 0 (Function.update ω₀ 0 a) * Ex fun ρ : (i : Fin T) → Ω i.succ =>
            (∏ s : Fin T, u s.succ (Fin.cons a ρ)) *
              if g 0 ω₀ * ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c then 1 else 0 := by
        rw [Ex_fin_cons]
        refine congrArg Ex (funext fun a => ?_)
        rw [← Ex_mul_left]
        refine congrArg Ex (funext fun ρ => ?_)
        rw [Fin.prod_univ_succ, Fin.prod_univ_succ, hua a ρ, hga (Fin.cons a ρ)]
        ring
      have h00 := h 0 ω₀
      rw [hL, hR]
      rcases (hg0 0 ω₀).eq_or_lt with h0 | h0
      ·
        have hpt : ∀ a : Ω 0, u 0 (Function.update ω₀ 0 a) * Ex (fun ρ : (i : Fin T) → Ω i.succ =>
            (∏ s : Fin T, u s.succ (Fin.cons a ρ)) *
              if g 0 ω₀ * ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c then 1 else 0) ≤
            u 0 (Function.update ω₀ 0 a) := fun a =>
          mul_le_of_le_one_right (hu0 _ _) ((Ex_mono (g := fun _ => (1 : ℝ)) fun ρ => by
            split_ifs
            · rw [mul_one]
              exact hP1 a ρ
            · rw [mul_zero]
              exact zero_le_one).trans (Ex_const 1).le)
        have hsum := Ex_mono hpt
        linarith
      · have hiff : ∀ x : ℝ, g 0 ω₀ * x ≤ c ↔ x ≤ c / g 0 ω₀ := fun x => (le_div_iff₀' h0).symm
        have hk0 : 0 ≤ c / g 0 ω₀ := div_nonneg hc h0.le
        have hmin0 : 0 ≤ min 1 (c / g 0 ω₀) := le_min zero_le_one hk0
        have hin : ∀ a : Ω 0, Ex (fun ρ : (i : Fin T) → Ω i.succ =>
            (∏ s : Fin T, u s.succ (Fin.cons a ρ)) *
              if g 0 ω₀ * ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c then 1 else 0) ≤
            min 1 (c / g 0 ω₀) +
              Ex fun ρ : (i : Fin T) → Ω i.succ => ∑ s : Fin T, w s.succ (Fin.cons a ρ) := fun a => by
          simp only [hiff]
          rcases le_total 1 (c / g 0 ω₀) with h1 | h1
          · rw [min_eq_left h1]
            have h2 := hin1 a (c / g 0 ω₀)
            have h3 := hBn a
            linarith
          · rw [min_eq_right h1]
            exact ih a _ hk0
        have hpt : ∀ a : Ω 0, u 0 (Function.update ω₀ 0 a) * Ex (fun ρ : (i : Fin T) → Ω i.succ =>
            (∏ s : Fin T, u s.succ (Fin.cons a ρ)) *
              if g 0 ω₀ * ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c then 1 else 0) ≤
            min 1 (c / g 0 ω₀) * u 0 (Function.update ω₀ 0 a) +
              Ex fun ρ : (i : Fin T) → Ω i.succ => ∑ s : Fin T, w s.succ (Fin.cons a ρ) := fun a => by
          have h1 := mul_le_mul_of_nonneg_left (hin a) (hu0 0 (Function.update ω₀ 0 a))
          have h2 := mul_le_of_le_one_left (hBn a) (hu1 0 (Function.update ω₀ 0 a))
          linarith
        have hsum : Ex (fun a : Ω 0 => u 0 (Function.update ω₀ 0 a) * Ex (fun ρ : (i : Fin T) → Ω i.succ =>
            (∏ s : Fin T, u s.succ (Fin.cons a ρ)) *
              if g 0 ω₀ * ∏ s : Fin T, g s.succ (Fin.cons a ρ) ≤ c then 1 else 0)) ≤
            min 1 (c / g 0 ω₀) * Ex (fun a : Ω 0 => u 0 (Function.update ω₀ 0 a)) +
              Ex fun a : Ω 0 => Ex fun ρ : (i : Fin T) → Ω i.succ => ∑ s : Fin T, w s.succ (Fin.cons a ρ) := by
          refine (Ex_mono hpt).trans_eq ?_
          rw [Ex_add, Ex_mul_left]
        have h3 : min 1 (c / g 0 ω₀) * g 0 ω₀ ≤ c :=
          (mul_le_mul_of_nonneg_right (min_le_right _ _) h0.le).trans_eq (div_mul_cancel₀ c h0.ne')
        have h4 : min 1 (c / g 0 ω₀) * Ex (fun a : Ω 0 => w 0 (Function.update ω₀ 0 a)) ≤
            Ex fun a : Ω 0 => w 0 (Function.update ω₀ 0 a) := mul_le_of_le_one_left hb0 (min_le_left _ _)
        have h5 := mul_le_mul_of_nonneg_left h00 hmin0
        linarith

end FlockSoundness.Audit.Periods
