module

public import Definitions.Poub.Guarantees.RemoteBinding
public import Proofs.Poub.Lemmas.RemoteBinding
public import Proofs.Hash.RowBinding
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Star.Real
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Pouw.TS.Sampled
import Proofs.Pouw.RelativeSecurity
import Proofs.Pous.Prob

/-! The proof of the receipts' binding bound, per receipt by plurality and over the window by Cauchy–Schwarz. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.Remote

open Finset ResourceAccounting.TS ResourceAccounting.Remote.Sample Pouw.RelativeSecurity
open ResourceAccounting.SecurityProofs.TS.Plurality (Frk)

namespace Binding

theorem sum_prob_le {Y P : Type} [Fintype Y] (L : Finset P) (F : P → Y → Prop) (A : Y → Finset P) (k : ℕ)
    (hA : ∀ p y, F p y → p ∈ A y) (hk : ∀ y, (A y).card ≤ k) :
    ∑ p ∈ L, prob (F p) ≤ k * prob (fun y => ∃ p ∈ L, F p y) := by
  classical
  have key : ∀ y, (∑ p ∈ L, if F p y then 1 else 0 : ℕ) ≤ k * (if ∃ p ∈ L, F p y then 1 else 0) := by
    intro y
    split_ifs with h
    · rw [mul_one, ← Finset.card_filter]
      exact (Finset.card_le_card fun p hp => hA p y (Finset.mem_filter.1 hp).2).trans (hk y)
    · rw [mul_zero, Nat.le_zero]
      exact Finset.sum_eq_zero fun p hp => by simp [show ¬ F p y from fun hF => h ⟨p, hp, hF⟩]
  have hn : (∑ p ∈ L, (univ.filter (F p)).card : ℕ) ≤ k * (univ.filter fun y => ∃ p ∈ L, F p y).card := by
    calc (∑ p ∈ L, (univ.filter (F p)).card : ℕ) = ∑ p ∈ L, ∑ y, (if F p y then 1 else 0 : ℕ) := by
          simp only [Finset.card_filter]
      _ = ∑ y, ∑ p ∈ L, (if F p y then 1 else 0 : ℕ) := Finset.sum_comm
      _ ≤ ∑ y, k * (if ∃ p ∈ L, F p y then 1 else 0) := Finset.sum_le_sum fun y _ => key y
      _ = k * (univ.filter fun y => ∃ p ∈ L, F p y).card := by rw [← Finset.mul_sum, Finset.card_filter]
  unfold prob Claims.prUniform
  rw [← Finset.sum_div, mul_div_assoc']
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have := (Nat.cast_le (α := ℝ)).2 hn
  push_cast at this
  convert this using 2
  congr 2
  ext y
  simp

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

theorem cont_left (t : Fin n) (ω ζ ζ' : SampleCoins n κ X) : cont t (cont t ω ζ) ζ' = cont t ω ζ' := by
  unfold cont
  refine Prod.ext rfl (funext fun j => ?_)
  by_cases h : j < t <;> simp [h]

theorem cont_swap (t : Fin n) (ω ζ : SampleCoins n κ X) : cont t (cont t ω ζ) (cont t ζ ω) = ω := by
  unfold cont
  refine Prod.ext rfl (funext fun j => ?_)
  by_cases h : j < t <;> simp [h]

theorem held_cont [Fintype X] (D : Remote.Sample.Run n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω ζ : SampleCoins n κ X) (p : ℕ) :
    R.held D prf rest t (cont t ω ζ) p = R.held D prf rest t ω p :=
  ResourceAccounting.SecurityProofs.TS.Plurality.plural_sp (opens := R.Shows D prf rest t p) (cont_left t) ω ζ

theorem tally_le_held [Fintype X] (D : Remote.Sample.Run n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω : SampleCoins n κ X) (p : ℕ) (v : Hash.Row) :
    tally (cont t) (R.Shows D prf rest t p) ω v ≤
      tally (cont t) (R.Shows D prf rest t p) ω (R.held D prf rest t ω p) :=
  ResourceAccounting.SecurityProofs.TS.Plurality.tally_le_plural ω v

theorem toReal_pr {α : Type} [Fintype α] (P : α → Prop) : (Pous.pr P).toReal = prob P := by
  unfold prob Claims.prUniform; simp only [Pous.pr_eq_card]
  rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem claims_pr {α : Type} [Fintype α] (P : α → Prop) : Claims.Pr P = prob P := by
  unfold Claims.Pr prob Claims.prUniform
  congr

theorem card_opened_le (D : Remote.Sample.Run n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω : SampleCoins n κ X) : (D.opened prf rest t ω).card ≤ s t := by
  unfold Run.opened openings
  exact Finset.card_image_le.trans ((Finset.card_range _).le.trans (by unfold Run.samples; exact min_le_left _ _))

theorem window_le [Fintype X] (D : Remote.Sample.Run n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ) (hL : ∀ t ω, D.opened prf rest t ω ⊆ L t)
    (t : Fin n) :
    prob (fun ω => ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p) ≤
      √((L t).card * s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal)) := by
  set a : ℕ → ℝ := fun p => prob (ResourceAccounting.SecurityProofs.TS.Plurality.Mov (cont t) (R.Shows D prf rest t p))
    with ha
  set Fk : ℕ → SampleCoins n κ X × SampleCoins n κ X → Prop := fun p y => Frk (cont t) (R.Shows D prf rest t p) y.1 y.2
    with hFk
  have h1 : prob (fun ω => ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p) ≤ ∑ p ∈ L t, a p := by
    refine (prob_mono fun ω hω => ?_).trans
      (prob_exists_le (L t) fun p ω => ResourceAccounting.SecurityProofs.TS.Plurality.Mov (cont t) (R.Shows D prf rest t p) ω)
    obtain ⟨p, hp, hacc, hne⟩ := hω
    exact ⟨p, hL t ω hp, (R.opened t ω p).row, ⟨hp, hacc, rfl⟩, hne⟩
  have hsq : ∀ p, a p ^ 2 ≤ prob (Fk p) := fun p =>
    ResourceAccounting.SecurityProofs.TS.Plurality.moved_le_forked (opens := R.Shows D prf rest t p) (cont_left t)
      (cont_swap t)
  have hsum : ∑ p ∈ L t, prob (Fk p) ≤ s t * prob (fun y => ∃ p ∈ L t, Fk p y) :=
    sum_prob_le (L t) Fk (fun y => D.opened prf rest t y.1) (s t) (fun p y h => by
      obtain ⟨v, v', ⟨hp, -⟩, -⟩ := h
      exact hp) (fun y => card_opened_le D prf rest t y.1)
  have hbreak : prob (fun y => ∃ p ∈ L t, Fk p y) ≤
      (R.fork D prf rest t).toReal + (R.split D prf rest t).toReal := by
    rw [Receipts.fork, Receipts.split, toReal_pr, toReal_pr]
    refine (prob_mono fun y hy => ?_).trans (prob_or_le _ _)
    obtain ⟨p, -, v, v', ⟨hp, hac, rfl⟩, ⟨hp', hac', rfl⟩, hne⟩ := hy
    by_cases hk : R.verifier t (cont t y.1 y.2) p = R.verifier t y.1 p ∧
        R.root t (cont t y.1 y.2) p = R.root t y.1 p ∧ R.rank t (cont t y.1 y.2) p = R.rank t y.1 p
    · left
      obtain ⟨hv, hr, hk'⟩ := hk
      rw [hv, hr, hk'] at hac'
      rcases Hash.SecurityProofs.RowBinding R.hash R.hash _ _ _ _ _ hac hac' with h | h | h
      · exact absurd h hne
      · exact ⟨p, hp, hp', Or.inl (by rw [hv, hk']; exact h)⟩
      · exact ⟨p, hp, hp', Or.inr (by rw [hv]; exact h)⟩
    · exact Or.inr ⟨p, hp, hp', hk⟩
  have hA : 0 ≤ ∑ p ∈ L t, a p := Finset.sum_nonneg fun p _ => prob_nonneg _
  have hfin : (∑ p ∈ L t, a p) ^ 2 ≤
      (L t).card * s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal) := by
    calc (∑ p ∈ L t, a p) ^ 2 ≤ (L t).card * ∑ p ∈ L t, a p ^ 2 := sq_sum_le_card_mul_sum_sq
      _ ≤ (L t).card * ∑ p ∈ L t, prob (Fk p) :=
          mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun p _ => hsq p) (Nat.cast_nonneg _)
      _ ≤ (L t).card * (s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal)) :=
          mul_le_mul_of_nonneg_left (hsum.trans (mul_le_mul_of_nonneg_left hbreak (Nat.cast_nonneg _)))
            (Nat.cast_nonneg _)
      _ = _ := by ring
  exact h1.trans ((Real.sqrt_sq hA).symm.le.trans (Real.sqrt_le_sqrt hfin))

theorem fork_le_finder [Fintype X] [Nonempty X] (D : Remote.Sample.Run n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (q : ℕ) :
    (R.fork D prf rest t).toReal ≤ (R.finder D prf rest t q).Pr (fun u => (R.hash u).bytes) := by
  rw [Receipts.fork, toReal_pr, Claims.Finder.Pr, claims_pr]
  refine prob_mono (Ω := SampleCoins n κ X × SampleCoins n κ X) fun y hy => ?_
  have hex : ∃ uv, R.Finds D prf rest t y.1 y.2 uv := by
    obtain ⟨p, hp, hp', h | h⟩ := hy
    · obtain ⟨u, hu, v, hv, hc⟩ := h
      exact ⟨(u, v), hc, p, hp, hp', Or.inl ⟨hu, hv⟩⟩
    · obtain ⟨u, hu, v, hv, hc⟩ := h
      exact ⟨(u, v), hc, p, hp, hp', Or.inr ⟨hu, hv⟩⟩
  refine ⟨hex.choose.1, hex.choose.2, ?_, hex.choose_spec.1⟩
  simp [Receipts.finder, hex]

end Binding

theorem ReceiptsBindingOfFork : ResourceAccounting.Guarantees.Remote.ReceiptsBindingOfFork := by
  intro n m s κ X _ D R prf rest L hL
  rw [Receipts.binding, Binding.toReal_pr]
  refine (prob_mono fun ω hω => ?_).trans ((prob_exists_le Finset.univ fun t ω =>
    ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p).trans
      (Finset.sum_le_sum fun t _ => Binding.window_le D R prf rest L hL t))
  obtain ⟨t, h⟩ := hω
  exact ⟨t, Finset.mem_univ _, h⟩

/-- **The receipts' binding term under `cr/sha-512`** (`ResourceAccounting.Guarantees.Remote.ReceiptsBindingOfCR`). -/
theorem ReceiptsBindingOfCR : ResourceAccounting.Guarantees.Remote.ReceiptsBindingOfCR := by
  intro n m s κ X _ _ D R prf rest L q hL hq
  refine (ReceiptsBindingOfFork D R prf rest L hL).trans (Finset.sum_le_sum fun t _ => Real.sqrt_le_sqrt ?_)
  refine mul_le_mul_of_nonneg_left (add_le_add ?_ le_rfl) (by positivity)
  exact (Binding.fork_le_finder D R prf rest t (q t)).trans (hq t (q t) fun _ => le_rfl)

namespace Binding

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

theorem binding_le_of_fixed [Fintype X] [Nonempty X] (D : Remote.Sample.Run n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : SampleCoins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ)
    (q : Fin n → ℕ) (hL : ∀ t ω, D.opened prf rest t ω ⊆ L t)
    (hq : ∀ t, Claims.StrictBirthday (fun u => (R.hash u).bytes) (R.finder D prf rest t (q t)))
    (hF : ∀ t ω ζ p, R.verifier t (cont t ω ζ) p = R.verifier t ω p ∧ R.root t (cont t ω ζ) p = R.root t ω p ∧
      R.rank t (cont t ω ζ) p = R.rank t ω p) :
    (R.binding D prf rest).toReal ≤ ∑ t, √((L t).card * s t) * q t / 2 ^ (256.5 : ℝ) := by
  refine (ReceiptsBindingOfCR D R prf rest L q hL hq).trans (le_of_eq (Finset.sum_congr rfl fun t _ => ?_))
  have hs : (R.split D prf rest t).toReal = 0 := by
    rw [Receipts.split, toReal_pr]
    have : (fun y : SampleCoins n κ X × SampleCoins n κ X => R.Split D prf rest t y.1 y.2) = fun _ => False := by
      funext y
      exact propext ⟨fun ⟨p, _, _, h⟩ => h (hF t y.1 y.2 p), False.elim⟩
    rw [this]
    unfold prob Claims.prUniform
    simp
  have h2 : √((2 : ℝ) ^ (513 : ℝ)) = 2 ^ (256.5 : ℝ) := by
    rw [Real.sqrt_eq_rpow, ← Real.rpow_mul (by norm_num)]
    norm_num
  rw [hs, add_zero, Real.sqrt_mul (by positivity), Real.sqrt_div (by positivity), Real.sqrt_sq (by positivity), h2,
    mul_div_assoc]

end Binding

end ResourceAccounting.SecurityProofs.Remote
