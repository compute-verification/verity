import Proofs.Flock.Claims.Recursion.Compose
import Proofs.Flock.Claims.Aggregate

/-!
# The composed extractor's failure, rerun by rerun

`servedERep_bad_le` charges the outer extractor `E₂`'s failure once, at the judged run. The inner stage reruns at many
draws, and `E₂` serves each, so `E₂` can fail at any of them. This file charges every one, by Wald's identity
(`erepProg_cost_exp`, at the indicator of `E₂`'s failure as the weight). The composed extractor fails on an accepted run
at most `(1 + m·k)` times as often as `E₂` fails (`OuterFails`), plus the inner stage's failure when `E₂` succeeds at
every draw it reruns at (`ServedGood`, `PrBadGood`): `servedERep_bad_le_good`. On that event every view the inner stage
sees is the one `dec` reads off a successful outer extraction.

* `Ex_sum`, `Pr_exists_le` (the union bound), `Ex_pi_coord` and `Pr_pi_coord` (one coordinate of a dependent product
  of coins), `Pr_equiv` (coins regrouped): the averages the recursion's proof uses.
* `sqrt_add_lin_any`: `sqrt_add_lin` at any `C`, since `√` of a negative is `0`.
-/

namespace Claims

/-! ## Averages -/

theorem Ex_sum {ι α : Type} [Fintype α] (s : Finset ι) (f : ι → α → ℝ) :
    Ex (fun a => ∑ i ∈ s, f i a) = ∑ i ∈ s, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

/-- **The union bound**: `Pr[∃ i, p i] ≤ Σ_i Pr[p i]`. -/
theorem Pr_exists_le {ι α : Type} [Fintype ι] [Fintype α] (p : ι → α → Prop) :
    Pr (fun a => ∃ i, p i a) ≤ ∑ i, Pr (p i) := by
  classical
  have h : ∀ i, Pr (p i) = Ex fun a => if p i a then (1 : ℝ) else 0 := fun i => (Ex_ind (p i)).symm
  rw [← Ex_ind, Finset.sum_congr rfl fun i _ => h i, ← Ex_sum]
  refine Ex_mono fun a => ?_
  split_ifs with ha
  · obtain ⟨i, hi⟩ := ha
    calc (1 : ℝ) = if p i a then (1 : ℝ) else 0 := by simp [hi]
      _ ≤ ∑ j, if p j a then (1 : ℝ) else 0 :=
          Finset.single_le_sum (f := fun j => if p j a then (1 : ℝ) else 0)
            (fun j _ => by split_ifs <;> norm_num) (Finset.mem_univ i)
  · exact Finset.sum_nonneg fun j _ => by split_ifs <;> norm_num

/-- **One coordinate of a dependent product**: a function of coordinate `i` averages over the product as over `C i`. -/
theorem Ex_pi_coord {ι : Type} [Fintype ι] [DecidableEq ι] {C : ι → Type} [∀ i, Fintype (C i)]
    [∀ i, Nonempty (C i)] (i : ι) (f : C i → ℝ) : Ex (fun c : (j : ι) → C j => f (c i)) = Ex f := by
  have hrest : (0 : ℝ) < Fintype.card ((j : { j // j ≠ i }) → C j) := by exact_mod_cast Fintype.card_pos
  unfold Ex
  rw [← (Equiv.piSplitAt i C).symm.sum_comp, Fintype.card_congr (Equiv.piSplitAt i C), Fintype.sum_prod_type,
    Fintype.card_prod]
  have e : ∀ a : C i, (∑ g : (j : { j // j ≠ i }) → C j, f ((Equiv.piSplitAt i C).symm (a, g) i)) =
      Fintype.card ((j : { j // j ≠ i }) → C j) * f a := by
    intro a
    simp [Equiv.piSplitAt_symm_apply]
  rw [Finset.sum_congr rfl fun a _ => e a, ← Finset.mul_sum]
  push_cast
  field_simp

/-- **One coordinate of a dependent product of coins**, after two others: an event of the first two coordinates and
coordinate `i` of the third has its probability over `α × β × C i`. -/
theorem Pr_pi_coord {α β ι : Type} [Fintype α] [Fintype β] [Fintype ι] [DecidableEq ι] {C : ι → Type}
    [∀ i, Fintype (C i)] [∀ i, Nonempty (C i)] (i : ι) (p : α → β → C i → Prop) :
    Pr (fun s : α × β × ((j : ι) → C j) => p s.1 s.2.1 (s.2.2 i)) = Pr (fun s : α × β × C i => p s.1 s.2.1 s.2.2) := by
  classical
  rw [← Ex_ind, ← Ex_ind, Ex_prod, Ex_prod]
  refine congrArg Ex (funext fun a => ?_)
  rw [Ex_prod, Ex_prod]
  refine congrArg Ex (funext fun b => ?_)
  exact Ex_pi_coord i fun c => if p a b c then (1 : ℝ) else 0

/-- Regrouped coins: an event has its probability over any regrouping of its coins. -/
theorem Pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Pr (fun a => p (e a)) = Pr p := by
  classical
  rw [Pr_eq_sum (fun a => p (e a)), Pr_eq_sum p, Fintype.card_congr e]
  congr 1
  exact e.sum_comp fun b => if p b then (1 : ℝ) else 0

/-- **Linear terms into the one square root, at any `C`**: `√(C·p) + β·p ≤ √((√C + β)²·p)` for `0 ≤ p ≤ 1`. -/
theorem sqrt_add_lin_any {C β p : ℝ} (hβ : 0 ≤ β) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    Real.sqrt (C * p) + β * p ≤ Real.sqrt ((Real.sqrt C + β) ^ 2 * p) := by
  by_cases hC : 0 ≤ C
  · exact sqrt_add_lin hC hβ hp hp1
  · rw [not_le] at hC
    rw [Real.sqrt_eq_zero'.2 (mul_nonpos_of_nonpos_of_nonneg hC.le hp), Real.sqrt_eq_zero'.2 hC.le]
    simpa using sqrt_add_lin (C := 0) le_rfl hβ hp hp1

/-! ## The outer extraction at every rerun -/

section Hybrid

variable {Coin W W₂ Op : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`E₂` fails at a draw**: the prover's run at the draw's coins accepts and `E₂` there, at the draw's own coins,
fails. -/
def OuterFails (H : Msg → Msg) (V : Verifier Coin) (R₂ : Rel W₂) (E₂ : Extractor Coin W₂ V) (x : Msg) {Ω : Type}
    (P : Prover Ω) (ω : Ω) (d : (Fin (V.k x) → Coin) × E₂.C x) : Prop :=
  (run H P V x ω d.1).out = .acc ∧ ¬ R₂.ext x (E₂.eval H P x ω d.1 d.2).1

/-- **`E₂` succeeds at every draw the inner stage reruns at**, the judged draw among them, at the judged run `(ω, r₀)`
and the composed extractor's coins `c`. -/
def ServedGood (H : Msg → Msg) (V : Verifier Coin) (R₂ : Rel W₂) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r₀ : Fin (V.k x) → Coin) (c : (servedERep V E₂ dec m k K opens assemble check).C x) : Prop :=
  ∀ d ∈ (erepProg (m x) (k x) (K x) opens (assemble x) (check x) (r₀, c.1) c.2).rerunsAt H
      (servedRun H (outerStage E₂ dec x) (run H P V x ω)), ¬ OuterFails H V R₂ E₂ x P ω d

/-- **The inner stage's failure when every outer extraction succeeds**:
`Pr[run = acc ∧ ServedGood ∧ ¬R(x, E(x))]`, `E` the composed extractor. -/
noncomputable def PrBadGood (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (R₂ : Rel W₂) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} [Fintype Ω]
    (P : Prover Ω) : ℝ :=
  Pr fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
    (run H P V x s.1 s.2.1).out = .acc ∧ ServedGood H V R₂ E₂ dec m k K opens assemble check x P s.1 s.2.1 s.2.2 ∧
      ¬ R.ext x ((servedERep V E₂ dec m k K opens assemble check).eval H P x s.1 s.2.1 s.2.2).1

/-- **Some outer extraction fails at most `1 + m·k` times as often as `E₂` does** (Wald's identity at the indicator of
`E₂`'s failure). -/
theorem Pr_not_servedGood_le (H : Msg → Msg) (V : Verifier Coin) (R₂ : Rel W₂) (E₂ : Extractor Coin W₂ V)
    (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ) (opens : Seen → List Op)
    (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type} [Fintype Ω]
    (P : Prover Ω) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
      ¬ ServedGood H V R₂ E₂ dec m k K opens assemble check x P s.1 s.2.1 s.2.2) ≤
      (1 + m x * k x) * PrBad H V R₂ E₂ x P := by
  classical
  let w : Ω → (Fin (V.k x) → Coin) × E₂.C x → ℝ := fun ω d => if OuterFails H V R₂ E₂ x P ω d then 1 else 0
  have hw : ∀ ω d, 0 ≤ w ω d := fun ω d => by simp only [w]; split_ifs <;> norm_num
  let L : Ω → ((Fin (V.k x) → Coin) × E₂.C x) × (Fin (m x) → Fin (K x) → (Fin (V.k x) → Coin) × E₂.C x) → ℝ :=
    fun ω t => (((erepProg (m x) (k x) (K x) opens (assemble x) (check x) t.1 t.2).rerunsAt H
      (servedRun H (outerStage E₂ dec x) (run H P V x ω))).map (w ω)).sum
  have hpt : ∀ s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x,
      (if ¬ ServedGood H V R₂ E₂ dec m k K opens assemble check x P s.1 s.2.1 s.2.2 then (1 : ℝ) else 0) ≤
        L s.1 ((s.2.1, s.2.2.1), s.2.2.2) := by
    intro s
    have hnn : ∀ y ∈ ((erepProg (m x) (k x) (K x) opens (assemble x) (check x) (s.2.1, s.2.2.1) s.2.2.2).rerunsAt H
        (servedRun H (outerStage E₂ dec x) (run H P V x s.1))).map (w s.1), 0 ≤ y := by
      intro y hy
      obtain ⟨d, _, rfl⟩ := List.mem_map.1 hy
      exact hw _ _
    by_cases h : ServedGood H V R₂ E₂ dec m k K opens assemble check x P s.1 s.2.1 s.2.2
    · simp only [h, not_true_eq_false, ↓reduceIte]
      exact List.sum_nonneg hnn
    · simp only [h, not_false_eq_true, ↓reduceIte]
      simp only [ServedGood, not_forall, not_not] at h
      obtain ⟨d, hd, hb⟩ := h
      have h1 : w s.1 d = 1 := by simp [w, hb]
      rw [← h1]
      exact List.single_le_sum hnn _ (List.mem_map.2 ⟨d, hd, rfl⟩)
  have hPrBad : PrBad H V R₂ E₂ x P = Ex fun ω => Ex (w ω) := by
    unfold PrBad
    rw [← Ex_ind, Ex_prod]
    exact congrArg Ex (funext fun ω => congrArg Ex (funext fun d => if_congr Iff.rfl rfl rfl))
  calc Pr (fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
        ¬ ServedGood H V R₂ E₂ dec m k K opens assemble check x P s.1 s.2.1 s.2.2)
      = Ex (fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
          if ¬ ServedGood H V R₂ E₂ dec m k K opens assemble check x P s.1 s.2.1 s.2.2 then (1 : ℝ) else 0) :=
        (Ex_ind _).symm
    _ ≤ Ex (fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
          L s.1 ((s.2.1, s.2.2.1), s.2.2.2)) := Ex_mono hpt
    _ = Ex fun ω => Ex (L ω) := by
        rw [Ex_prod]
        exact congrArg Ex (funext fun ω => (Ex_assoc (fun t => L ω ((t.1, t.2.1), t.2.2))).trans rfl)
    _ ≤ Ex fun ω => (1 + m x * k x) * Ex (w ω) :=
        Ex_mono fun ω => erepProg_cost_exp H _ (hw ω) _ _ _ _ _ _
    _ = (1 + m x * k x) * PrBad H V R₂ E₂ x P := by rw [Ex_mul_left, hPrBad]

/-- **The composition's failure, every rerun charged**: the composed extractor fails on an accepted run at most
`1 + m·k` times as often as `E₂` does, plus the inner stage's failure when every outer extraction succeeds. -/
theorem servedERep_bad_le_good (H : Msg → Msg) (V : Verifier Coin) (R : Rel W) (R₂ : Rel W₂)
    (E₂ : Extractor Coin W₂ V) (dec : (x : Msg) → (Fin (V.k x) → Coin) → Option W₂ → Seen) (m k K : Msg → ℕ)
    (opens : Seen → List Op) (assemble : Msg → List Op → Option W) (check : Msg → W → Bool) (x : Msg) {Ω : Type}
    [Fintype Ω] (P : Prover Ω) :
    PrBad H V R (servedERep V E₂ dec m k K opens assemble check) x P ≤
      (1 + m x * k x) * PrBad H V R₂ E₂ x P + PrBadGood H V R R₂ E₂ dec m k K opens assemble check x P := by
  have hng := Pr_not_servedGood_le H V R₂ E₂ dec m k K opens assemble check x P
  have hor : PrBad H V R (servedERep V E₂ dec m k K opens assemble check) x P ≤
      Pr (fun s : Ω × (Fin (V.k x) → Coin) × (servedERep V E₂ dec m k K opens assemble check).C x =>
        ¬ ServedGood H V R₂ E₂ dec m k K opens assemble check x P s.1 s.2.1 s.2.2) +
      PrBadGood H V R R₂ E₂ dec m k K opens assemble check x P := by
    refine (Pr_mono fun s h => ?_).trans (Pr_or_le _ _)
    by_cases hg : ServedGood H V R₂ E₂ dec m k K opens assemble check x P s.1 s.2.1 s.2.2
    · exact Or.inr ⟨h.1, hg, h.2⟩
    · exact Or.inl hg
  linarith

end Hybrid

end Claims
