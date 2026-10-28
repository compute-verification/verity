import Definitions.Accounting.Guarantees.PoubBinding
import Proofs.Accounting.TS.Drawn
import Proofs.Hash.RowBinding

/-!
# The receipts' binding bound: the proof

Fix window `t` and a receipt `p`, and read a cut of coins and its continuations `cont t` (the other coins and the
earlier windows' openings coins kept, window `t`'s and the later ones fresh), which have `spliceAt`'s two laws
(`Binding.cont_left`, `Binding.cont_swap`).

* **The held row** (`Binding.held_cont`, `Binding.tally_le_held`). `Receipts.held` is the plurality of the rows the
  receipt shows over the cut's continuations (`Accounting.TS.plural`), so every continuation holds the cut's row, and
  no row is shown alone on more of them.
* **Per receipt** (`Plurality.moved_le_forked`, at `cont t`). A move shows the presented row against the plurality
  (`Plurality.Mov`), and the moves squared are at most the pairs of continuations showing two different rows
  (`Plurality.Frk`): `Pr[p moved]² ≤ Pr[p forked]`.
* **Over the window's receipts** (`Binding.window_le`). The union bound and Cauchy–Schwarz over `L_t` give
  `(Σ_p Pr[p moved])² ≤ |L_t|·Σ_p Pr[p forked]`. A run opens at most `s_t` receipts (`Binding.card_opened_le`), so
  `Σ_p Pr[p forked] ≤ s_t·Pr[some p forked]` (`Binding.sum_prob_le`). A forked receipt is accepted in both runs
  with two different rows: at one verifier, root and rank that is a SHA-512 collision among what the two checks hashed
  (`Hash.SecurityProofs.RowBinding`), the fork's break (`Receipts.Broken`), and otherwise the roots split
  (`Receipts.Split`).
* **Under `cr/sha-512`** (`Binding.fork_le_finder`). A break is a collision the window's fork finder outputs, so
  `fork_t ≤ Pr[finder succeeds] ≤ q_t²/2^513`.
* **Fixed roots** (`Binding.binding_le_of_fixed`). Where the window's cut fixes its receipts' roots, ranks and
  verifiers, `split_t = 0`, and `√(|L_t|·s_t·q_t²/2^513) = √(|L_t|·s_t)·q_t/2^256.5`.
-/

namespace Accounting.SecurityProofs.Poub

open Finset Accounting.TS Accounting.Poub.Draw Pouw.RelativeSecurity
open Accounting.SecurityProofs.TS.Plurality (Frk)

namespace Binding

/-- Events each of which names a member of a set of at most `k`: their probabilities sum to at most `k` times that of
some one of them. -/
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
  unfold prob
  rw [← Finset.sum_div, mul_div_assoc']
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have := (Nat.cast_le (α := ℝ)).2 hn
  push_cast at this
  convert this using 2
  congr 2
  ext y
  simp

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

theorem cont_left (t : Fin n) (ω ζ ζ' : Coins n κ X) : cont t (cont t ω ζ) ζ' = cont t ω ζ' := by
  unfold cont
  refine Prod.ext rfl (funext fun j => ?_)
  by_cases h : j < t <;> simp [h]

theorem cont_swap (t : Fin n) (ω ζ : Coins n κ X) : cont t (cont t ω ζ) (cont t ζ ω) = ω := by
  unfold cont
  refine Prod.ext rfl (funext fun j => ?_)
  by_cases h : j < t <;> simp [h]

/-- **The held row reads only the cut**: every continuation of `ω` from window `t`'s cut holds `ω`'s row. -/
theorem held_cont [Fintype X] (D : Session n m s κ X) (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω ζ : Coins n κ X) (p : ℕ) :
    R.held D prf rest t (cont t ω ζ) p = R.held D prf rest t ω p :=
  Accounting.SecurityProofs.TS.Plurality.plural_sp (opens := R.Shows D prf rest t p) (cont_left t) ω ζ

/-- **The held row is the likeliest**: no row is shown alone on more of window `t`'s continuations of `ω`. -/
theorem tally_le_held [Fintype X] (D : Session n m s κ X) (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω : Coins n κ X) (p : ℕ) (v : Hash.Row) :
    tally (cont t) (R.Shows D prf rest t p) ω v ≤
      tally (cont t) (R.Shows D prf rest t p) ω (R.held D prf rest t ω p) :=
  Accounting.SecurityProofs.TS.Plurality.tally_le_plural ω v

theorem toReal_pr {α : Type} [Fintype α] (P : α → Prop) : (Pous.pr P).toReal = prob P := by
  unfold Pous.pr prob
  rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]

theorem claims_pr {α : Type} [Fintype α] (P : α → Prop) : Claims.Pr P = prob P := by
  unfold Claims.Pr prob
  congr

/-- A window opens at most `s_t` receipts: its draws, `min(s_t, U_t)`. -/
theorem card_opened_le (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (ω : Coins n κ X) : (D.opened prf rest t ω).card ≤ s t := by
  unfold Session.opened openings
  exact Finset.card_image_le.trans ((Finset.card_range _).le.trans (by unfold Session.draws; exact min_le_left _ _))

/-- **Window `t`'s share of the binding term** is at most `√(|L_t|·s_t·(fork_t + split_t))`. -/
theorem window_le [Fintype X] (D : Session n m s κ X) (R : Receipts n κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ) (hL : ∀ t ω, D.opened prf rest t ω ⊆ L t)
    (t : Fin n) :
    prob (fun ω => ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p) ≤
      √((L t).card * s t * ((R.fork D prf rest t).toReal + (R.split D prf rest t).toReal)) := by
  set a : ℕ → ℝ := fun p => prob (Accounting.SecurityProofs.TS.Plurality.Mov (cont t) (R.Shows D prf rest t p))
    with ha
  set Fk : ℕ → Coins n κ X × Coins n κ X → Prop := fun p y => Frk (cont t) (R.Shows D prf rest t p) y.1 y.2
    with hFk
  have h1 : prob (fun ω => ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p) ≤ ∑ p ∈ L t, a p := by
    refine (prob_mono fun ω hω => ?_).trans
      (prob_exists_le (L t) fun p ω => Accounting.SecurityProofs.TS.Plurality.Mov (cont t) (R.Shows D prf rest t p) ω)
    obtain ⟨p, hp, hacc, hne⟩ := hω
    exact ⟨p, hL t ω hp, (R.opened t ω p).row, ⟨hp, hacc, rfl⟩, hne⟩
  have hsq : ∀ p, a p ^ 2 ≤ prob (Fk p) := fun p =>
    Accounting.SecurityProofs.TS.Plurality.moved_le_forked (opens := R.Shows D prf rest t p) (cont_left t)
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

/-- A fork break is a collision the window's fork finder outputs. -/
theorem fork_le_finder [Fintype X] [Nonempty X] (D : Session n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (q : ℕ) :
    (R.fork D prf rest t).toReal ≤ (R.finder D prf rest t q).Pr (fun u => (R.hash u).bytes) := by
  rw [Receipts.fork, toReal_pr, Claims.Finder.Pr, claims_pr]
  refine prob_mono (Ω := Coins n κ X × Coins n κ X) fun y hy => ?_
  have hex : ∃ uv, R.Finds D prf rest t y.1 y.2 uv := by
    obtain ⟨p, hp, hp', h | h⟩ := hy
    · obtain ⟨u, hu, v, hv, hc⟩ := h
      exact ⟨(u, v), hc, p, hp, hp', Or.inl ⟨hu, hv⟩⟩
    · obtain ⟨u, hu, v, hv, hc⟩ := h
      exact ⟨(u, v), hc, p, hp, hp', Or.inr ⟨hu, hv⟩⟩
  refine ⟨hex.choose.1, hex.choose.2, ?_, hex.choose_spec.1⟩
  simp [Receipts.finder, hex]

end Binding

/-- **The receipts' binding term, in its fork form** (`Accounting.Guarantees.Poub.ReceiptsBindingOfFork`). -/
theorem ReceiptsBindingOfFork : Accounting.Guarantees.Poub.ReceiptsBindingOfFork := by
  intro n m s κ X _ D R prf rest L hL
  rw [Receipts.binding, Binding.toReal_pr]
  refine (prob_mono fun ω hω => ?_).trans ((prob_exists_le Finset.univ fun t ω =>
    ∃ p ∈ D.opened prf rest t ω, R.Moved D prf rest t ω p).trans
      (Finset.sum_le_sum fun t _ => Binding.window_le D R prf rest L hL t))
  obtain ⟨t, h⟩ := hω
  exact ⟨t, Finset.mem_univ _, h⟩

/-- **The receipts' binding term under `cr/sha-512`** (`Accounting.Guarantees.Poub.ReceiptsBindingOfCR`). -/
theorem ReceiptsBindingOfCR : Accounting.Guarantees.Poub.ReceiptsBindingOfCR := by
  intro n m s κ X _ _ D R prf rest L q hL hq
  refine (ReceiptsBindingOfFork D R prf rest L hL).trans (Finset.sum_le_sum fun t _ => Real.sqrt_le_sqrt ?_)
  refine mul_le_mul_of_nonneg_left (add_le_add ?_ le_rfl) (by positivity)
  exact (Binding.fork_le_finder D R prf rest t (q t)).trans (hq t (q t) fun _ => le_rfl)

namespace Binding

variable {n m : ℕ} {s : Fin n → ℕ} {κ : ℕ} {X : Type}

/-- **Fixed roots**: where every window's cut fixes its receipts' verifiers, roots and ranks, the roots never split,
and under `cr/sha-512` the binding term is at most `Σ_t √(|L_t|·s_t)·q_t/2^256.5`, `ledger.binding`'s charge. -/
theorem binding_le_of_fixed [Fintype X] [Nonempty X] (D : Session n m s κ X) (R : Receipts n κ X)
    (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ) (L : Fin n → Finset ℕ)
    (q : Fin n → ℕ) (hL : ∀ t ω, D.opened prf rest t ω ⊆ L t)
    (hq : ∀ t, Claims.StrictBirthday (fun u => (R.hash u).bytes) (R.finder D prf rest t (q t)))
    (hF : ∀ t ω ζ p, R.verifier t (cont t ω ζ) p = R.verifier t ω p ∧ R.root t (cont t ω ζ) p = R.root t ω p ∧
      R.rank t (cont t ω ζ) p = R.rank t ω p) :
    (R.binding D prf rest).toReal ≤ ∑ t, √((L t).card * s t) * q t / 2 ^ (256.5 : ℝ) := by
  refine (ReceiptsBindingOfCR D R prf rest L q hL hq).trans (le_of_eq (Finset.sum_congr rfl fun t _ => ?_))
  have hs : (R.split D prf rest t).toReal = 0 := by
    rw [Receipts.split, toReal_pr]
    have : (fun y : Coins n κ X × Coins n κ X => R.Split D prf rest t y.1 y.2) = fun _ => False := by
      funext y
      exact propext ⟨fun ⟨p, _, _, h⟩ => h (hF t y.1 y.2 p), False.elim⟩
    rw [this]
    unfold prob
    simp
  have h2 : √((2 : ℝ) ^ (513 : ℝ)) = 2 ^ (256.5 : ℝ) := by
    rw [Real.sqrt_eq_rpow, ← Real.rpow_mul (by norm_num)]
    norm_num
  rw [hs, add_zero, Real.sqrt_mul (by positivity), Real.sqrt_div (by positivity), Real.sqrt_sq (by positivity), h2,
    mul_div_assoc]

end Binding

end Accounting.SecurityProofs.Poub
