import Proofs.Population.Exfiltration
import Proofs.Population.Determinacy

/-!
# Exfiltration before detection: tightness, and what fails without each assumption

* `tight_cum`, `tight_exists`: **the stopping-time bound is attained.** A strategy that leaks `ℓ` every window and
  escapes each window with mass exactly `exp (-κ ℓ)` reaches `(j + 1) ℓ` with probability exactly `exp (-κ j ℓ)`, which
  is `tail_cum`'s bound at `x = (j + 1) ℓ`; for every `κ ≥ 0` and `ℓ > 0` there is such a draw.
* `no_pressure_cum`, `not_pressure_univ`: **without detection pressure** (a leak no draw catches: no faulty proof block,
  a draw known before the commitment, or an uncertified link) `n` windows leak `n ℓ` with probability one.
* `no_halt_cumUpTo`, `pressure_empty`: **without a halt**, a strategy caught in every window still leaks `n ℓ`, and it
  meets detection pressure at every `κ`.
* `seedCopy_checked`, `seed_egress`: **without bound inputs**, a checked population whose units copy a prover-chosen
  seed to the egress shows all `2 ^ k` values with no faulty proof block, so no draw sees it.
* `not_trustedModel_of_reveal`: **without a trusted model**, an honest model whose outputs determine the target gives it
  to any guesser.
-/

namespace Population.Exfil

open Finset

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω] {μ : Ω → ℝ}

/-- Leak `ℓ` in every window and escape on the coins `s`. -/
def constStrategy (ℓ : ℝ) (s : Finset Ω) : Strategy Ω := ⟨fun _ => ℓ, fun _ => s⟩

omit [Fintype Ω] [DecidableEq Ω] in
@[simp] theorem const_leak (ℓ : ℝ) (s : Finset Ω) (h : List Ω) : (constStrategy ℓ s).leak h = ℓ := rfl

omit [Fintype Ω] [DecidableEq Ω] in
@[simp] theorem const_esc (ℓ : ℝ) (s : Finset Ω) (h : List Ω) : (constStrategy ℓ s).esc h = s := rfl

omit [Fintype Ω] [DecidableEq Ω] in
theorem const_leakBounded {ℓ : ℝ} (hℓ : 0 ≤ ℓ) (s : Finset Ω) : LeakBounded ℓ (constStrategy ℓ s) :=
  fun _ => ⟨hℓ, le_rfl⟩

omit [DecidableEq Ω] in
theorem prob_true (hμ : IsDist μ) (n : ℕ) : prob μ n (fun _ => True) = 1 := by
  simp only [prob, ite_true]
  exact ex_const hμ n 1

/-! ## Tightness -/

/-- **The constant strategy, exactly**: it reaches `(j + 1) ℓ` within `n > j` windows exactly when its first `j` coins
escape, with probability `a ^ j`, `a` the acceptance probability. -/
theorem tight_cum (hμ : IsDist μ) {ℓ : ℝ} (hℓ : 0 < ℓ) (s : Finset Ω) :
    ∀ (n j : ℕ) (h : List Ω), j < n →
      prob μ n (fun cs => ((j : ℝ) + 1) * ℓ ≤ (constStrategy ℓ s).cum n h cs) = (∑ c ∈ s, μ c) ^ j
  | 0, _, _, hj => absurd hj (Nat.not_lt_zero _)
  | n + 1, 0, h, _ => by
    rw [pow_zero]
    refine (prob_congr (E' := fun _ => True) fun cs => ?_).trans (prob_true hμ (n + 1))
    simp only [Nat.cast_zero, zero_add, one_mul, iff_true]
    rw [← Fin.cons_self_tail cs, cum_cons, const_leak, const_esc]
    have := cum_nonneg (const_leakBounded hℓ.le s) n (cs 0 :: h) (Fin.tail cs)
    split_ifs <;> linarith
  | n + 1, j + 1, h, hj => by
    have key : ∀ c : Ω, prob μ n (fun cs => (((j + 1 : ℕ) : ℝ) + 1) * ℓ ≤
        (constStrategy ℓ s).cum (n + 1) h (Fin.cons c cs)) = if c ∈ s then (∑ c ∈ s, μ c) ^ j else 0 := by
      intro c
      by_cases hc : c ∈ s
      · rw [ite_eq_left hc, ← tight_cum hμ hℓ s n j (c :: h) (by omega)]
        refine prob_congr fun cs => ?_
        rw [cum_cons, const_leak, const_esc, ite_eq_left hc]
        push_cast
        constructor <;> intro <;> linarith
      · rw [ite_eq_right hc, ← prob_false μ n]
        refine prob_congr fun cs => ?_
        rw [cum_cons, const_leak, const_esc, ite_eq_right hc]
        push_cast
        simp only [add_zero, iff_false, not_le]
        nlinarith
    rw [prob_succ]
    simp only [key]
    rw [sum_ite_esc, pow_succ']

/-- **`tail_cum` is attained**: when every window's acceptance probability is exactly `exp (-κ ℓ)`, the constant
strategy reaches `(j + 1) ℓ` with probability `exp (-κ ((j + 1) ℓ - ℓ))`, the bound itself. -/
theorem tight_tail (hμ : IsDist μ) {κ ℓ : ℝ} (hℓ : 0 < ℓ) {s : Finset Ω} (hs : ∑ c ∈ s, μ c = Real.exp (-(κ * ℓ)))
    {n j : ℕ} (h : List Ω) (hj : j < n) :
    prob μ n (fun cs => ((j : ℝ) + 1) * ℓ ≤ (constStrategy ℓ s).cum n h cs) =
      Real.exp (-(κ * (((j : ℝ) + 1) * ℓ - ℓ))) := by
  rw [tight_cum hμ hℓ s n j h hj, hs, ← Real.exp_nat_mul]
  ring_nf

/-- The two-coin draw that catches with probability `1 - a`. -/
noncomputable def coin (a : ℝ) : Bool → ℝ := fun b => if b then a else 1 - a

theorem coin_isDist {a : ℝ} (h0 : 0 ≤ a) (h1 : a ≤ 1) : IsDist (coin a) :=
  ⟨fun b => by cases b <;> simp [coin] <;> linarith, by simp [coin]⟩

/-- **The bound is tight for every `κ ≥ 0` and `ℓ > 0`**: some draw and some strategy meet detection pressure `κ` and the
per-window bound `ℓ`, and reach `(j + 1) ℓ` with probability exactly `tail_cum`'s bound, at every horizon `n > j`. -/
theorem tight_exists {κ ℓ : ℝ} (hκ : 0 ≤ κ) (hℓ : 0 < ℓ) :
    ∃ (μ : Bool → ℝ) (A : Strategy Bool), IsDist μ ∧ DetectionPressure μ κ A ∧ LeakBounded ℓ A ∧
      ∀ (n j : ℕ), j < n → prob μ n (fun cs => ((j : ℝ) + 1) * ℓ ≤ A.cum n [] cs) =
        Real.exp (-(κ * (((j : ℝ) + 1) * ℓ - ℓ))) := by
  have h1 : Real.exp (-(κ * ℓ)) ≤ 1 := Real.exp_le_one_iff.2 (by nlinarith)
  have hs : ∑ c ∈ {true}, coin (Real.exp (-(κ * ℓ))) c = Real.exp (-(κ * ℓ)) := by simp [coin]
  refine ⟨coin (Real.exp (-(κ * ℓ))), constStrategy ℓ {true}, coin_isDist (Real.exp_pos _).le h1,
    fun _ => hs.le, const_leakBounded hℓ.le _, fun n j hj => ?_⟩
  exact tight_tail (coin_isDist (Real.exp_pos _).le h1) hℓ hs [] hj

/-! ## Without detection pressure -/

/-- **No draw catches it**: a strategy that escapes on every coin leaks `ℓ` in each of the `n` windows. -/
theorem no_pressure_cum (ℓ : ℝ) : ∀ (n : ℕ) (h : List Ω) (cs : Fin n → Ω),
    (constStrategy ℓ univ).cum n h cs = n * ℓ
  | 0, _, _ => by simp [Strategy.cum]
  | n + 1, h, cs => by
    rw [← Fin.cons_self_tail cs, cum_cons, const_leak, const_esc, ite_eq_left (mem_univ _), no_pressure_cum ℓ n]
    push_cast
    ring

omit [DecidableEq Ω] in
/-- A positive leak that every coin lets escape meets no detection pressure `κ > 0`. -/
theorem not_pressure_univ (hμ : IsDist μ) {κ ℓ : ℝ} (hκ : 0 < κ) (hℓ : 0 < ℓ) :
    ¬ DetectionPressure μ κ (constStrategy ℓ univ) := by
  intro hP
  have h := hP []
  simp only [constStrategy, hμ.2] at h
  have : Real.exp (-(κ * ℓ)) < 1 := Real.exp_lt_one_iff.2 (by nlinarith)
  linarith

/-! ## Without a halt -/

omit [Fintype Ω] in
/-- **Caught every window, still leaking**: with no halt, a strategy that every coin catches leaks `ℓ` per window up
to the `n`-th detection. -/
theorem no_halt_cumUpTo (ℓ : ℝ) : ∀ (n : ℕ) (h : List Ω) (cs : Fin n → Ω),
    (constStrategy ℓ ∅).cumUpTo n n h cs = n * ℓ
  | 0, _, _ => by simp [Strategy.cumUpTo]
  | n + 1, h, cs => by
    rw [← Fin.cons_self_tail cs, cumUpTo_cons, const_leak, const_esc, ite_eq_right (notMem_empty _),
      no_halt_cumUpTo ℓ n]
    push_cast
    ring

omit [Fintype Ω] [DecidableEq Ω] in
/-- It meets detection pressure at every `κ`: its acceptance probability is zero. -/
theorem pressure_empty (κ ℓ : ℝ) : DetectionPressure μ κ (constStrategy ℓ (∅ : Finset Ω)) :=
  fun _ => by simpa [constStrategy] using (Real.exp_pos _).le

omit [Fintype Ω] in
/-- With the halt, the same strategy leaks one window's `ℓ`. -/
theorem halt_cum (ℓ : ℝ) (n : ℕ) (h : List Ω) (cs : Fin (n + 1) → Ω) :
    (constStrategy ℓ (∅ : Finset Ω)).cum (n + 1) h cs = ℓ := by
  rw [← Fin.cons_self_tail cs, cum_cons]
  simp [constStrategy]

/-! ## Without a trusted model -/

open Classical in
/-- **A model that reveals the target**: if the honest outputs determine `S`, a guesser gets it with probability one,
so no `g < 1` is a trusted-model bound. -/
theorem not_trustedModel_of_reveal {W H T : Type} [Fintype W] {ρ : W → ℝ} (hρ : IsDist ρ) {S : W → T}
    {hon : W → H} (dec : H → T) (hdec : ∀ w, dec (hon w) = S w) {g : ℝ} (hg : g < 1) :
    ¬ TrustedModel ρ S hon g := by
  intro hT
  have := hT dec
  simp only [hdec, ite_true, mul_one, hρ.2] at this
  linarith

end Population.Exfil

/-! ## Without bound inputs: a prover-chosen seed -/

namespace Population

open Finset

/-- Unit `i` reads seed slot `inl i` and writes egress slot `inr i`. The seeds are inputs. -/
def seedCopy (k : ℕ) : Dataflow (Fin k) (Fin k ⊕ Fin k) where
  input := univ.image Sum.inl
  reads i := {Sum.inl i}
  writes i := {Sum.inr i}

/-- Each unit copies its seed. -/
def seedFns (k : ℕ) : (seedCopy k).Fns Bool := fun i r _ => r ⟨Sum.inl i, mem_singleton_self _⟩

theorem seedCopy_checked (k : ℕ) : (seedCopy k).Checked := by
  refine ⟨fun s => ⟨fun hs u hu => ?_, fun hs => ?_⟩, fun u hu => ?_⟩
  · simp only [seedCopy, mem_image, mem_univ, true_and, mem_singleton] at hs hu
    obtain ⟨i, rfl⟩ := hs
    cases hu
  · cases s with
    | inl i => exact absurd (by simp [seedCopy]) hs
    | inr i => exact ⟨i, by simp [seedCopy], fun j hj => by simpa [seedCopy, eq_comm] using hj⟩
  · obtain ⟨_, ⟨s, hsw, hsr⟩, -⟩ := Relation.TransGen.head'_iff.1 hu
    simp only [seedCopy, mem_singleton] at hsw hsr
    rw [hsw] at hsr
    cases hsr

/-- **No faulty proof block, every seed on the egress**: with the seed slots left to the prover (`X` = the inputs), the
assignments with no faulty proof block show all `2 ^ k` egress values. -/
theorem seed_egress (k : ℕ) :
    2 ^ k ≤ ((seedCopy k).egress (seedFns k) (fun _ => false) (seedCopy k).input (univ.image Sum.inr) ∅).card := by
  classical
  have hmem : ∀ w : Fin k → Bool, (fun o : ↥(univ.image Sum.inr : Finset (Fin k ⊕ Fin k)) => Sum.elim w w o.1) ∈
      (seedCopy k).egress (seedFns k) (fun _ => false) (seedCopy k).input (univ.image Sum.inr) ∅ := by
    intro w
    simp only [Dataflow.egress, mem_image, mem_filter, mem_univ, true_and, subset_empty]
    refine ⟨Sum.elim w w, ⟨fun s hs hsX => absurd hs hsX, ?_⟩, rfl⟩
    refine (seedCopy k).wrong_eq_empty fun i s => ?_
    obtain ⟨s, hs⟩ := s
    simp only [seedCopy, mem_singleton] at hs
    subst hs
    rfl
  have hinj : Function.Injective fun (w : Fin k → Bool)
      (o : ↥(univ.image Sum.inr : Finset (Fin k ⊕ Fin k))) => Sum.elim w w o.1 := by
    intro w w' h
    funext i
    exact congrFun h ⟨Sum.inr i, mem_image_of_mem _ (mem_univ i)⟩
  have := card_le_card_of_injOn _ (s := univ) (fun w _ => hmem w) hinj.injOn
  simpa using this

end Population
