import Definitions.Claims.RecursionLin
import Proofs.Flock.Claims.Recursion.InnerAbort

/-!
# The plurality, linearly: a link finder in place of two forks

`Pr_plur_sq_le` bounds a draw's miss of the plurality by the root of two disagreeing draws, and every `√` in the
recursion's `γe` comes from it. This file bounds the same miss linearly, by a finder whose loop the honest finder can
run: draw `u`; if `g u`, draw `u₁ … u_M` until the first where `g` holds, and succeed if its `f` differs from `f u`
(`firstDiffers`). Its success is exactly `Σ_u [g u]·Pr[g ∧ f ≠ f u]·G_M`, with `q·G_M = 1 − (1 − q)^M` at `q = Pr[g]`
(`Pr_firstDiffers`, `geomS_mul`), so a miss costs at most the link plus `q(1 − q)^M ≤ 1/(M+1)` (`Pr_plur_link`). The
finder's expected cost is at most twice one draw's (`Ex_linkCost_le`), so under `ExpectedCR` at it the miss is at most
`2·E[c]/2^256 + 1/(M+1)` (`Pr_plur_ecr`): no root, and `M` is the finder's to choose.
-/

namespace Claims.Recursion

open Claims

section PlurLin

variable {U : Type} [Fintype U]

theorem Pr_top_eq_one {α : Type} [Fintype α] [Nonempty α] : Pr (fun _ : α => True) = 1 := by
  rw [← Ex_ind (fun _ : α => True)]
  simpa using Ex_const (α := α) (1 : ℝ)

theorem Pr_neg_eq {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) : Pr (fun a => ¬ p a) = 1 - Pr p := by
  classical
  have h : (fun a => (if ¬ p a then (1 : ℝ) else 0) + (if p a then 1 else 0)) = fun _ => 1 :=
    funext fun a => by by_cases h : p a <;> simp [h]
  have h2 := Ex_add (fun a => if ¬ p a then (1 : ℝ) else 0) (fun a => if p a then 1 else 0)
  rw [h, Ex_const, Ex_ind, Ex_ind] at h2
  linarith

theorem Ex_if_zero {α : Type} [Fintype α] (p : α → Prop) [DecidablePred p] (c : ℝ) :
    Ex (fun a => if p a then c else 0) = c * Pr p := by
  rw [← Ex_ind p, ← Ex_mul_left]
  congr 1
  funext a
  by_cases h : p a <;> simp [h]

/-- **The first of `M` draws where `g` holds has `f` other than `a`.** -/
def firstDiffers (g : U → Prop) (f : U → Msg) (a : Msg) : (M : ℕ) → (Fin M → U) → Prop
  | 0, _ => False
  | M + 1, us => (g (us 0) ∧ f (us 0) ≠ a) ∨ (¬ g (us 0) ∧ firstDiffers g f a M (Fin.tail us))

omit [Fintype U] in
theorem firstPick_of_firstDiffers {g : U → Prop} {f : U → Msg} {a : Msg} :
    ∀ M (us : Fin M → U), firstDiffers g f a M us → ∃ v, firstPick g M us = some v ∧ g v ∧ f v ≠ a
  | 0, _, h => h.elim
  | M + 1, us, h => by
    rcases h with ⟨hg, hf⟩ | ⟨hg, h⟩
    · exact ⟨us 0, by simp [firstPick, hg], hg, hf⟩
    · obtain ⟨v, hv, hgv, hfv⟩ := firstPick_of_firstDiffers M (Fin.tail us) h
      exact ⟨v, by simp [firstPick, hg, hv], hgv, hfv⟩

/-- `G_M = Σ_{i<M} r^i`. -/
noncomputable def geomS (r : ℝ) : ℕ → ℝ
  | 0 => 0
  | M + 1 => 1 + r * geomS r M

theorem geomS_nonneg {r : ℝ} (hr : 0 ≤ r) : ∀ M, 0 ≤ geomS r M
  | 0 => le_refl 0
  | M + 1 => by
    have := geomS_nonneg hr M
    show 0 ≤ 1 + r * geomS r M
    positivity

theorem geomS_mul (q : ℝ) : ∀ M, q * geomS (1 - q) M = 1 - (1 - q) ^ M
  | 0 => by simp [geomS]
  | M + 1 => by
    have ih := geomS_mul q M
    show q * (1 + (1 - q) * geomS (1 - q) M) = 1 - (1 - q) ^ (M + 1)
    linear_combination (1 - q) * ih

/-- **The geometric tail**: `q(1 − q)^M ≤ 1/(M+1)` for `q ≤ 1`. -/
theorem mul_pow_le_inv {q : ℝ} (h1 : q ≤ 1) (M : ℕ) : q * (1 - q) ^ M ≤ 1 / ((M : ℝ) + 1) := by
  have key : ∀ M : ℕ, (1 - q) ^ M * (1 + M * q) ≤ 1 := by
    intro M
    induction M with
    | zero => simp
    | succ n ih =>
      have ht : 0 ≤ (1 - q) ^ n := pow_nonneg (by linarith) n
      push_cast
      rw [pow_succ]
      nlinarith [mul_nonneg (mul_nonneg ht (sq_nonneg q)) (by positivity : (0 : ℝ) ≤ n + 1)]
  have ht : 0 ≤ (1 - q) ^ M := pow_nonneg (by linarith) M
  rw [le_div_iff₀ (by positivity)]
  nlinarith [key M, mul_nonneg ht (sub_nonneg.2 h1)]

/-- The first of `M + 1` draws and the rest. -/
def splitFirst (M : ℕ) : (Fin (M + 1) → U) ≃ U × (Fin M → U) where
  toFun us := (us 0, Fin.tail us)
  invFun p := Fin.cons p.1 p.2
  left_inv us := Fin.cons_self_tail us
  right_inv p := by simp

theorem Pr_splitFirst {M : ℕ} (P : U → (Fin M → U) → Prop) :
    Pr (fun us : Fin (M + 1) → U => P (us 0) (Fin.tail us)) = Ex fun u => Pr fun vs => P u vs :=
  (Pr_equiv (splitFirst M) (fun p : U × (Fin M → U) => P p.1 p.2)).trans (Pr_prod _)

theorem Ex_splitFirst {M : ℕ} (F : U → (Fin M → U) → ℝ) :
    Ex (fun us : Fin (M + 1) → U => F (us 0) (Fin.tail us)) = Ex fun u => Ex fun vs => F u vs :=
  (Ex_equiv (splitFirst M) (fun p : U × (Fin M → U) => F p.1 p.2)).trans (Ex_prod _)

/-- **The loop finds a differing draw with probability `Pr[g ∧ f ≠ a]·G_M`** at `r = 1 − Pr[g]`. -/
theorem Pr_firstDiffers [Nonempty U] (g : U → Prop) (f : U → Msg) (a : Msg) :
    ∀ M, Pr (firstDiffers g f a M) = Pr (fun u => g u ∧ f u ≠ a) * geomS (1 - Pr g) M
  | 0 => by
    rw [Pr_congr (p := firstDiffers g f a 0) (q := fun _ => False) fun _ => by simp [firstDiffers], Pr_false]
    simp [geomS]
  | M + 1 => by
    classical
    have ih := Pr_firstDiffers g f a M
    have hpt : ∀ u, (Pr fun vs => (g u ∧ f u ≠ a) ∨ (¬ g u ∧ firstDiffers g f a M vs)) =
        (if g u ∧ f u ≠ a then 1 else 0) +
          (if ¬ g u then 1 else 0) * (Pr (fun u => g u ∧ f u ≠ a) * geomS (1 - Pr g) M) := by
      intro u
      by_cases hg : g u
      · by_cases hf : f u = a
        · rw [Pr_congr (q := fun _ => False) fun vs => by simp [hg, hf], Pr_false]
          simp [hg, hf]
        · rw [Pr_congr (q := fun _ => True) fun vs => by simp [hg, hf], Pr_top_eq_one]
          simp [hg, hf]
      · rw [Pr_congr (q := firstDiffers g f a M) fun vs => by simp [hg], ih]
        simp [hg]
    calc Pr (firstDiffers g f a (M + 1))
        = Ex fun u => Pr fun vs => (g u ∧ f u ≠ a) ∨ (¬ g u ∧ firstDiffers g f a M vs) :=
          Pr_splitFirst (fun u vs => (g u ∧ f u ≠ a) ∨ (¬ g u ∧ firstDiffers g f a M vs))
      _ = _ := by
          rw [congrArg Ex (funext hpt), Ex_add, Ex_mul_right, Ex_ind, Ex_ind, Pr_neg_eq]
          show _ = _ * (1 + (1 - Pr g) * geomS (1 - Pr g) M)
          ring

/-- **Off the plurality, at most the link and the geometric tail**: a draw where `g` holds misses the plurality at most
as often as the link finder (a draw `u` where `g` holds, then `M` draws, the first of them where `g` holds differing
from `f u`) succeeds, plus `1/(M+1)`. Linear, where `Pr_plur_sq_le` is squared. -/
theorem Pr_plur_link (g : U → Prop) (f : U → Msg) (M : ℕ) :
    Pr (fun u => g u ∧ f u ≠ plur g f) ≤
      Pr (fun p : U × (Fin M → U) => g p.1 ∧ firstDiffers g f (f p.1) M p.2) + 1 / ((M : ℝ) + 1) := by
  classical
  have hL0 : 0 ≤ Pr (fun p : U × (Fin M → U) => g p.1 ∧ firstDiffers g f (f p.1) M p.2) := Pr_nonneg _
  have hT0 : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1) := by positivity
  by_cases hne : ∃ u, g u
  · obtain ⟨u₀, -, hp, hmax⟩ := plur_spec g f hne
    have : Nonempty U := ⟨u₀⟩
    rw [hp]
    have hq0 : 0 ≤ Pr g := Pr_nonneg _
    have hq1 : Pr g ≤ 1 := Pr_le_one _
    have hS0 : 0 ≤ geomS (1 - Pr g) M := geomS_nonneg (by linarith) M
    have hL : Pr (fun p : U × (Fin M → U) => g p.1 ∧ firstDiffers g f (f p.1) M p.2) =
        Ex fun u => if g u then Pr (fun u' => g u' ∧ f u' ≠ f u) * geomS (1 - Pr g) M else 0 := by
      rw [Pr_prod]
      refine congrArg Ex (funext fun u => ?_)
      by_cases hu : g u
      · simp only [hu, true_and, ↓reduceIte]
        exact Pr_firstDiffers g f (f u) M
      · simp only [hu, false_and, ↓reduceIte]
        exact Pr_false
    have hpt : ∀ u, (if g u then Pr (fun u' => g u' ∧ f u' ≠ f u₀) * geomS (1 - Pr g) M else 0) ≤
        if g u then Pr (fun u' => g u' ∧ f u' ≠ f u) * geomS (1 - Pr g) M else 0 := by
      intro u
      by_cases hu : g u
      · simp only [hu, ↓reduceIte]
        have h1 := Pr_split g f (f u)
        have h2 := Pr_split g f (f u₀)
        exact mul_le_mul_of_nonneg_right (by linarith [hmax u hu]) hS0
      · simp [hu]
    have hlow : Pr (fun u => g u ∧ f u ≠ f u₀) * geomS (1 - Pr g) M * Pr g ≤
        Pr (fun p : U × (Fin M → U) => g p.1 ∧ firstDiffers g f (f p.1) M p.2) := by
      rw [hL, ← Ex_if_zero]
      exact Ex_mono hpt
    have hgm := geomS_mul (Pr g) M
    have hdq : Pr (fun u => g u ∧ f u ≠ f u₀) ≤ Pr g := Pr_mono fun u h => h.1
    have ht : 0 ≤ (1 - Pr g) ^ M := pow_nonneg (by linarith) M
    have hdt := mul_le_mul_of_nonneg_right hdq ht
    have htail := mul_pow_le_inv hq1 M
    have he : Pr (fun u => g u ∧ f u ≠ f u₀) * geomS (1 - Pr g) M * Pr g =
        Pr (fun u => g u ∧ f u ≠ f u₀) * (1 - (1 - Pr g) ^ M) := by
      rw [← hgm]; ring
    nlinarith
  · have h0 : Pr (fun u => g u ∧ f u ≠ plur g f) = 0 := by
      refine le_antisymm ((Pr_mono fun u h => ?_).trans_eq Pr_false) (Pr_nonneg _)
      exact hne ⟨u, h.1⟩
    rw [h0]
    linarith

theorem Ex_loopCost [Nonempty U] (g : U → Prop) (c : U → ℕ) :
    ∀ M, Ex (fun us : Fin M → U => (loopCost g c M us : ℝ)) = Ex (fun u => (c u : ℝ)) * geomS (1 - Pr g) M
  | 0 => by
    show Ex (fun _ : Fin 0 → U => ((0 : ℕ) : ℝ)) = _
    simp [geomS, Ex_const]
  | M + 1 => by
    classical
    have ih := Ex_loopCost g c M
    have hfun : (fun us : Fin (M + 1) → U => (loopCost g c (M + 1) us : ℝ)) = fun us =>
        (c (us 0) : ℝ) + if g (us 0) then 0 else (loopCost g c M (Fin.tail us) : ℝ) := by
      funext us
      simp only [loopCost, Nat.cast_add, Nat.cast_ite, Nat.cast_zero]
    have hpt : ∀ u, (Ex fun vs : Fin M → U => (c u : ℝ) + if g u then 0 else (loopCost g c M vs : ℝ)) =
        (c u : ℝ) + if ¬ g u then Ex (fun vs : Fin M → U => (loopCost g c M vs : ℝ)) else 0 := by
      intro u
      by_cases hu : g u
      · simp [hu, Ex_const]
      · simp only [hu, ↓reduceIte, not_false_eq_true]
        rw [Ex_add, Ex_const]
    rw [hfun, Ex_splitFirst (fun u vs => (c u : ℝ) + if g u then 0 else (loopCost g c M vs : ℝ)),
      congrArg Ex (funext hpt), Ex_add, Ex_if_zero, Pr_neg_eq, ih]
    show _ = _ * (1 + (1 - Pr g) * geomS (1 - Pr g) M)
    ring

/-- **The link finder costs at most two draws in expectation** (Wald): one draw, then `G_M` more where `g` holds,
and `Pr[g]·G_M ≤ 1`. -/
theorem Ex_linkCost_le [Nonempty U] (g : U → Prop) (c : U → ℕ) (M : ℕ) :
    Ex (fun p : U × (Fin M → U) => (linkCost g c M p : ℝ)) ≤ 2 * Ex (fun u => (c u : ℝ)) := by
  classical
  have hpt : ∀ u, (Ex fun vs : Fin M → U => (linkCost g c M (u, vs) : ℝ)) =
      (c u : ℝ) + if g u then Ex (fun vs : Fin M → U => (loopCost g c M vs : ℝ)) else 0 := by
    intro u
    by_cases hu : g u
    · simp only [linkCost, hu, ↓reduceIte, Nat.cast_add]
      rw [Ex_add, Ex_const]
    · simp [linkCost, hu, Ex_const]
  rw [Ex_prod, congrArg Ex (funext hpt), Ex_add, Ex_if_zero, Ex_loopCost]
  have hc : 0 ≤ Ex (fun u => (c u : ℝ)) := Ex_nonneg fun u => Nat.cast_nonneg _
  have hq0 : 0 ≤ Pr g := Pr_nonneg _
  have hq1 : Pr g ≤ 1 := Pr_le_one _
  have hgm := geomS_mul (Pr g) M
  have ht : 0 ≤ (1 - Pr g) ^ M := pow_nonneg (by linarith) M
  have he : Ex (fun u => (c u : ℝ)) * geomS (1 - Pr g) M * Pr g = Ex (fun u => (c u : ℝ)) * (1 - (1 - Pr g) ^ M) := by
    rw [← hgm]; ring
  nlinarith [mul_nonneg hc ht]

/-- **The link finder**: on `(u, u₁ … u_M)`, the collision its output names, at most `linkCost` calls. -/
def linkFinder [Nonempty U] (M : ℕ) (out : U × (Fin M → U) → Option (Msg × Msg)) (cost : U × (Fin M → U) → ℕ) :
    Finder where
  S := U × (Fin M → U)
  out := out
  cost := cost

/-- **Off the plurality under `ExpectedCR`, linearly**: where the link's success names a SHA-512 collision and its
declared cost is at most `linkCost` (one draw's `c`, then the loop's), a draw where `g` holds misses the plurality with
probability at most `2·E[c]/2^256 + 1/(M+1)`. -/
theorem Pr_plur_ecr [Nonempty U] (H : Msg → Msg) (g : U → Prop) (f : U → Msg) (c : U → ℕ) (M : ℕ)
    (out : U × (Fin M → U) → Option (Msg × Msg)) (cost : U × (Fin M → U) → ℕ)
    (hout : ∀ p, g p.1 → firstDiffers g f (f p.1) M p.2 → ∃ a b, out p = some (a, b) ∧ Collision H a b)
    (hcost : ∀ p, cost p ≤ linkCost g c M p) (hcr : ExpectedCR H (linkFinder M out cost)) :
    Pr (fun u => g u ∧ f u ≠ plur g f) ≤ 2 * Ex (fun u => (c u : ℝ)) / 2 ^ (256 : ℝ) + 1 / ((M : ℝ) + 1) := by
  have hlink : Pr (fun p : U × (Fin M → U) => g p.1 ∧ firstDiffers g f (f p.1) M p.2) ≤
      (linkFinder M out cost).Pr H :=
    Pr_mono fun p h => hout p h.1 h.2
  have hc : Ex (fun p : U × (Fin M → U) => (cost p : ℝ)) ≤ 2 * Ex (fun u => (c u : ℝ)) :=
    (Ex_mono fun p => Nat.cast_le.2 (hcost p)).trans (Ex_linkCost_le g c M)
  have hcr' : (linkFinder M out cost).Pr H ≤ 2 * Ex (fun u => (c u : ℝ)) / 2 ^ (256 : ℝ) :=
    hcr.trans (div_le_div_of_nonneg_right hc (by positivity))
  linarith [Pr_plur_link g f M]

/-- **Off the plurality at each coin prefix, linearly**: over a tape and a draw, the draw's `f` misses the plurality
over the draws with its first `j` coins at most as often as the link at the fork at `j` succeeds (a tape, a prefix
`r`, then `M + 1` draws with `r`'s first `j` coins), plus `1/(M+1)`: `Pr_plur_link` at each prefix, averaged. Linear,
where `Pr_plur_fork` is a root. -/
theorem Pr_plur_fork_lin {Coin : Type} [Fintype Coin] [Nonempty Coin] {Ω C : Type} [Fintype Ω] [Nonempty Ω]
    [Fintype C] [Nonempty C] {N : ℕ} (j : ℕ) (g : Ω → (Fin N → Coin) × C → Prop) (f : Ω → (Fin N → Coin) × C → Msg)
    (M : ℕ) :
    Pr (fun s : Ω × (Fin N → Coin) × C => g s.1 s.2 ∧
      f s.1 s.2 ≠ plur (fun u : (Fin N → Coin) × C => g s.1 (fork j s.2.1 u.1, u.2))
        (fun u => f s.1 (fork j s.2.1 u.1, u.2))) ≤
      Pr (fun t : Ω × (Fin N → Coin) × ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
        g t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2) ∧
          firstDiffers (fun u : (Fin N → Coin) × C => g t.1 (fork j t.2.1 u.1, u.2))
            (fun u => f t.1 (fork j t.2.1 u.1, u.2)) (f t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2)) M t.2.2.2) +
        1 / ((M : ℝ) + 1) := by
  have hoff : PrefixDet (k := N) (Coin := Coin) fun _ => j := fun _ _ _ => rfl
  let Gr : Ω → (Fin N → Coin) → (Fin N → Coin) × C → Prop := fun ω r u => g ω (fork j r u.1, u.2)
  let Fr : Ω → (Fin N → Coin) → (Fin N → Coin) × C → Msg := fun ω r u => f ω (fork j r u.1, u.2)
  let a : Ω → (Fin N → Coin) → ℝ := fun ω r => Pr fun u => Gr ω r u ∧ Fr ω r u ≠ plur (Gr ω r) (Fr ω r)
  let b : Ω → (Fin N → Coin) → ℝ := fun ω r =>
    Pr fun p : ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
      Gr ω r p.1 ∧ firstDiffers (Gr ω r) (Fr ω r) (Fr ω r p.1) M p.2
  have hGr : ∀ ω r u, Gr ω (fork j r u) = Gr ω r := fun ω r u => funext fun v => by simp only [Gr, fork_fork]
  have hFr : ∀ ω r u, Fr ω (fork j r u) = Fr ω r := fun ω r u => funext fun v => by simp only [Fr, fork_fork]
  have hL : Pr (fun s : Ω × (Fin N → Coin) × C => g s.1 s.2 ∧
      f s.1 s.2 ≠ plur (fun u : (Fin N → Coin) × C => g s.1 (fork j s.2.1 u.1, u.2))
        (fun u => f s.1 (fork j s.2.1 u.1, u.2))) = Ex fun ω => Ex fun r => a ω r := by
    rw [Pr_prod]
    refine congrArg Ex (funext fun ω => ?_)
    rw [Pr_prod]
    rw [← Ex_fork hoff (fun r => Pr fun c : C => g ω (r, c) ∧ f ω (r, c) ≠ plur (Gr ω r) (Fr ω r)), Ex_prod]
    refine congrArg Ex (funext fun r => ?_)
    simp only [hGr, hFr]
    show _ = Pr (fun u : (Fin N → Coin) × C => g ω (fork j r u.1, u.2) ∧ f ω (fork j r u.1, u.2) ≠ plur (Gr ω r) (Fr ω r))
    rw [Pr_prod (fun u : (Fin N → Coin) × C => g ω (fork j r u.1, u.2) ∧ f ω (fork j r u.1, u.2) ≠ plur (Gr ω r) (Fr ω r))]
  have hR : Pr (fun t : Ω × (Fin N → Coin) × ((Fin N → Coin) × C) × (Fin M → (Fin N → Coin) × C) =>
      g t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2) ∧
        firstDiffers (fun u : (Fin N → Coin) × C => g t.1 (fork j t.2.1 u.1, u.2))
          (fun u => f t.1 (fork j t.2.1 u.1, u.2)) (f t.1 (fork j t.2.1 t.2.2.1.1, t.2.2.1.2)) M t.2.2.2) =
      Ex fun ω => Ex fun r => b ω r := by
    rw [Pr_prod]
    refine congrArg Ex (funext fun ω => ?_)
    rw [Pr_prod]
  have hab : ∀ ω r, a ω r ≤ b ω r + 1 / ((M : ℝ) + 1) := fun ω r => Pr_plur_link (Gr ω r) (Fr ω r) M
  rw [hL, hR]
  calc (Ex fun ω => Ex fun r => a ω r) ≤ Ex fun ω => Ex fun r => b ω r + 1 / ((M : ℝ) + 1) :=
        Ex_mono fun ω => Ex_mono fun r => hab ω r
    _ = (Ex fun ω => Ex fun r => b ω r) + 1 / ((M : ℝ) + 1) := by
        simp only [Ex_add, Ex_const]

end PlurLin

end Claims.Recursion
