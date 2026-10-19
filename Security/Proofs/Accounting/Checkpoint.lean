import Models.Accounting.Guarantees.Checkpoint
import Proofs.Accounting.HMu.Luck
import Proofs.Pouw.RelativeSecurity

/-!
# The whole-unit guess from its checkpoints

`prob_prefix_le` is the chain rule: the first `k` checkpoints together with the mode have probability at most `q^k` times
the mode's, one checkpoint at a time. `WholeGuessOfCheckpoints` reads it at `k = n` on the event `same`, which lies
inside it. A negative `q` leaves nothing to prove: the first checkpoint's bound then makes the mode null.

`CheckpointGuessOfFresh` splits set `s`'s coins off the rest (`splitAt`) and bounds each slice: with the view and the
other sets' coins fixed, the earlier checkpoints and the mode are fixed too, and set `s`'s coins alone move checkpoint
`s`.

`crossRead` is why a round's tiles compound only in an order, fixed before the round's coins: two tiles that each meet
`CheckpointGuessOfFresh`'s hypotheses, the other tile's coin as the view, are lucky together as often as each is alone.
-/

namespace Accounting.SecurityProofs.Checkpoint

open Accounting.TS Accounting.Assumptions.Checkpoint Accounting.HMuProofs
open Pouw.RelativeSecurity (prob)

theorem prefix_succ_iff {C : Type} (E : ℕ → C → Prop) (W : C → Prop) (k : ℕ) (c : C) :
    ((∀ r < k + 1, E r c) ∧ W c) ↔ (E k c ∧ (∀ r < k, E r c) ∧ W c) := by
  constructor
  · rintro ⟨hE, hW⟩
    exact ⟨hE k (Nat.lt_succ_self k), fun r hr => hE r (Nat.lt_succ_of_lt hr), hW⟩
  · rintro ⟨hk, hE, hW⟩
    refine ⟨fun r hr => ?_, hW⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hr with hr | rfl
    · exact hE r hr
    · exact hk

/-- **The chain rule over checkpoints.** -/
theorem prob_prefix_le {C : Type} [Fintype C] {q : ℝ} (hq : 0 ≤ q) {n : ℕ} {E : ℕ → C → Prop} {W : C → Prop}
    (h : CheckpointGuessAt q n E W) (k : ℕ) (hk : k ≤ n) :
    prob (fun c => (∀ r < k, E r c) ∧ W c) ≤ q ^ k * prob W := by
  induction k with
  | zero =>
      rw [pow_zero, one_mul]
      exact prob_mono fun c hc => hc.2
  | succ k ih =>
      calc prob (fun c => (∀ r < k + 1, E r c) ∧ W c)
          = prob (fun c => E k c ∧ (∀ r < k, E r c) ∧ W c) := prob_congr (prefix_succ_iff E W k)
        _ ≤ q * prob (fun c => (∀ r < k, E r c) ∧ W c) := h k hk
        _ ≤ q * (q ^ k * prob W) := mul_le_mul_of_nonneg_left (ih (Nat.le_of_succ_le hk)) hq
        _ = q ^ (k + 1) * prob W := by ring

theorem WholeGuessOfCheckpoints : Accounting.Guarantees.Checkpoint.WholeGuessOfCheckpoints := by
  intro C M _ _ mode s right same E n q hsame h
  have hd : ∀ c, (0 < (if mode c = s then (1 : ℝ) else 0)) ↔ mode c = s := fun c => by
    by_cases hc : mode c = s <;> simp [hc]
  have hP : prob (fun c => same c ∧ 0 < (if mode c = s then (1 : ℝ) else 0)) ≤
      prob (fun c => (∀ r < n, E r c) ∧ mode c = s) :=
    prob_mono fun c hc => ⟨hsame c hc.1, (hd c).1 hc.2⟩
  have hW : prob (fun c => 0 < (if mode c = s then (1 : ℝ) else 0)) = prob (fun c => mode c = s) := prob_congr hd
  show (∀ c, 0 ≤ (if mode c = s then (1 : ℝ) else 0) ∧ (if mode c = s then (1 : ℝ) else 0) ≤ 1) ∧
      prob (fun c => same c ∧ 0 < (if mode c = s then (1 : ℝ) else 0)) ≤
        q ^ n * prob (fun c => 0 < (if mode c = s then (1 : ℝ) else 0)) ∧
      ∀ _ : C, True
  refine ⟨fun c => ?_, ?_, fun _ => trivial⟩
  · by_cases hc : mode c = s <;> simp [hc]
  rw [hW]
  by_cases hq : 0 ≤ q
  · exact hP.trans (prob_prefix_le hq h n le_rfl)
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · rw [pow_zero, one_mul]
    exact hP.trans (prob_mono fun c hc => hc.2)
  rw [not_le] at hq
  have h0 := h 0 hn
  have h0' : prob (fun c => (∀ r < 0, E r c) ∧ mode c = s) = prob (fun c => mode c = s) :=
    prob_congr fun c => by simp
  have hz : prob (fun c => mode c = s) = 0 :=
    le_antisymm
      (by nlinarith [prob_nonneg (fun c => E 0 c ∧ (∀ r < 0, E r c) ∧ mode c = s), prob_nonneg (fun c => mode c = s)])
      (prob_nonneg _)
  rw [hz, mul_zero]
  exact hP.trans ((prob_mono fun c hc => hc.2).trans hz.le)

/-- Set `i`'s coins split off the view and the other sets'. -/
def splitAt {V X : Type} {n : ℕ} (i : Fin n) : V × (Fin n → X) ≃ (V × ({j // j ≠ i} → X)) × X where
  toFun p := ((p.1, fun j => p.2 j.1), p.2 i)
  invFun p := (p.1.1, fun j => if h : j = i then p.2 else p.1.2 ⟨j, h⟩)
  left_inv p := by
    rcases p with ⟨v, x⟩
    refine Prod.ext rfl (funext fun j => ?_)
    by_cases h : j = i
    · subst h; simp
    · simp [h]
  right_inv p := by
    rcases p with ⟨⟨v, o⟩, ξ⟩
    refine Prod.ext (Prod.ext rfl (funext fun j => ?_)) (by simp)
    simp [j.2]

theorem CheckpointGuessOfFresh : Accounting.Guarantees.Checkpoint.CheckpointGuessOfFresh := by
  intro V X _ _ _ n E whole q hE hq s hs
  classical
  let i : Fin n := ⟨s, hs⟩
  rw [Pouw.RelativeSecurity.prob_equiv (splitAt (V := V) (X := X) i),
    Pouw.RelativeSecurity.prob_equiv (splitAt (V := V) (X := X) i)]
  apply prob_le_of_slices
  rintro ⟨v, o⟩
  obtain ⟨ξ0⟩ := ‹Nonempty X›
  let x0 : Fin n → X := ((splitAt (V := V) i).symm ((v, o), ξ0)).2
  have hpair : ∀ ξ, (splitAt (V := V) i).symm ((v, o), ξ) = (v, Function.update x0 i ξ) := fun ξ => by
    refine Prod.ext rfl (funext fun j => ?_)
    by_cases h : j = i
    · subst h; simp [splitAt]
    · simp [splitAt, x0, h]
  have hpre : ∀ ξ, (∀ r < s, E r (v, Function.update x0 i ξ)) ↔ ∀ r < s, E r (v, x0) := fun ξ =>
    forall₂_congr fun r hr => hE r (hr.trans hs) v _ _ fun j hj =>
      Function.update_of_ne (fun h => by subst h; exact absurd hj (by simp [i]; omega)) _ _
  simp only [hpair]
  by_cases hW : whole v
  · by_cases hP : ∀ r < s, E r (v, x0)
    · rw [prob_eq_one (p := fun ξ => (∀ r < s, E r (v, Function.update x0 i ξ)) ∧ whole v)
        fun ξ => ⟨(hpre ξ).2 hP, hW⟩, mul_one]
      exact (prob_mono fun ξ h => h.1).trans (hq s hs v hW x0)
    · rw [prob_eq_zero fun ξ h => hP ((hpre ξ).1 h.2.1),
        prob_eq_zero fun ξ h => hP ((hpre ξ).1 h.1), mul_zero]
  · rw [prob_eq_zero fun ξ h => hW h.2.2, prob_eq_zero fun ξ h => hW h.2, mul_zero]

/-- **Per-tile luck is not joint luck: the cross-read.** Two tiles skipped whole, with independent uniform one-bit set
coins `c.1` and `c.2`, each tile's honest checkpoint its own coin. Tile 1 commits tile 2's coin and tile 2 commits
tile 1's. Each commitment reads only the other tile's coin, so each tile on its own meets `CheckpointGuessOfFresh`'s
hypotheses with that coin as its view, at its set's point mass `½` (the first conjunct), and keeps its checkpoint
with probability `½`. The two events are one, so both tiles keep theirs with probability `½`, not `¼`. A bound over a
round's tiles jointly needs an order, fixed before the round's coins, in which no tile's commitments read a later
tile's coins. -/
theorem crossRead :
    (∀ v : Bool, prob (fun ξ : Bool => v = ξ) = 1 / 2) ∧
      prob (fun c : Bool × Bool => c.2 = c.1) = 1 / 2 ∧ prob (fun c : Bool × Bool => c.1 = c.2) = 1 / 2 ∧
      prob (fun c : Bool × Bool => c.2 = c.1 ∧ c.1 = c.2) = 1 / 2 := by
  refine ⟨fun v => ?_, ?_, ?_, ?_⟩
  · rw [prob_eq_card, show (Finset.univ.filter fun ξ : Bool => v = ξ).card = 1 by cases v <;> decide]
    norm_num
  · rw [prob_eq_card, show (Finset.univ.filter fun c : Bool × Bool => c.2 = c.1).card = 2 by decide]
    norm_num
  · rw [prob_eq_card, show (Finset.univ.filter fun c : Bool × Bool => c.1 = c.2).card = 2 by decide]
    norm_num
  · rw [prob_eq_card, show (Finset.univ.filter fun c : Bool × Bool => c.2 = c.1 ∧ c.1 = c.2).card = 2 by decide]
    norm_num

end Accounting.SecurityProofs.Checkpoint
