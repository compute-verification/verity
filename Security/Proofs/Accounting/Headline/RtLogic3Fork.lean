import Definitions.Accounting.TS.ProofLayer

/-!
RT3-LOGIC-1 witness (red team, not for landing): binding's fork form (`ProofLayer.Forked`) resamples round `w`'s coin
and compares round `w`'s openings, so an honest proof layer whose round-`w` openings read that round's coin forks
with probability `1 − 1/|Ω|` (`½` on `Bool`). `coinLayer n` is such a layer over `n + 1` rounds: every proof accepted,
proof `0` of round `w` opening `coinAt x w`, the coin `x` drew at round `w`, as a real round's proof carries the
receipt's keys and the draw (`LiUnits.k ω`, `window.instance_ts`). It has every round's relation `True`, so it is
sound and pinned at any `δ_ZK`. Builds on main.
-/

namespace Accounting.SecurityProofs.TS.RtLogic3

open Finset Pouw.RelativeSecurity Accounting.TS

/-- The coin a path drew at round `w` (`false` past its end). -/
def coinAt : {n : ℕ} → Paths Bool n → ℕ → Bool
  | 0, _, _ => false
  | _ + 1, x, 0 => x.1
  | _ + 1, x, w + 1 => coinAt x.2 w

/-- An honest proof layer whose round-`w` opening is round `w`'s coin. -/
def coinLayer (n : ℕ) : ProofLayer Bool (n + 1) 1 Bool where
  accepted := fun _ _ _ => True
  opened := fun x w _ => coinAt x w
  rel := fun _ _ _ => True
  right := fun _ => True

theorem coinLayer_forked_zero (n : ℕ) (x z : Paths Bool (n + 1)) :
    (coinLayer n).Forked 0 x z ↔ x.1 ≠ z.1 := by
  simp [ProofLayer.Forked, ProofLayer.Proved, coinLayer, splice, coinAt]

/-- The all-`false` path. -/
def falsePath : (n : ℕ) → Paths Bool n
  | 0 => ()
  | n + 1 => (false, falsePath n)

/-- Flipping the first path's round-0 coin: an involution on pairs of paths. -/
def flip1 (n : ℕ) : Paths Bool (n + 1) × Paths Bool (n + 1) ≃ Paths Bool (n + 1) × Paths Bool (n + 1) where
  toFun p := ((!p.1.1, p.1.2), p.2)
  invFun p := ((!p.1.1, p.1.2), p.2)
  left_inv p := by simp
  right_inv p := by simp

theorem card_ne_twice (n : ℕ) [DecidablePred fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1]
    [DecidablePred fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => ¬ p.1.1 ≠ p.2.1] :
    (univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1).card * 2 =
      Fintype.card (Paths Bool (n + 1) × Paths Bool (n + 1)) := by
  have hsw : (univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1).card =
      (univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => ¬ p.1.1 ≠ p.2.1).card := by
    refine Finset.card_bij (fun p _ => flip1 n p) ?_ ?_ ?_
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, flip1, Equiv.coe_fn_mk] at hp ⊢
      cases h1 : p.1.1 <;> cases h2 : p.2.1 <;> simp_all
    · intro a _ b _ h
      exact (flip1 n).injective h
    · intro b hb
      refine ⟨flip1 n b, ?_, ?_⟩
      · simp only [Finset.mem_filter, Finset.mem_univ, true_and, flip1, Equiv.coe_fn_mk] at hb ⊢
        cases h1 : b.1.1 <;> cases h2 : b.2.1 <;> simp_all
      · simp [flip1]
  have hsum := Finset.card_filter_add_card_filter_not
    (s := (univ : Finset (Paths Bool (n + 1) × Paths Bool (n + 1))))
    (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1)
  rw [Finset.card_univ] at hsum
  omega

/-- Two paths' round-0 coins differ with probability `½`. -/
theorem prob_ne_half (n : ℕ) :
    prob (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1) = 1 / 2 := by
  classical
  have hpos : (0 : ℝ) < Fintype.card (Paths Bool (n + 1) × Paths Bool (n + 1)) := by
    exact_mod_cast Fintype.card_pos_iff.mpr ⟨(falsePath (n + 1), falsePath (n + 1))⟩
  have hN : ((univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1).card : ℝ) =
      1 / 2 * Fintype.card (Paths Bool (n + 1) × Paths Bool (n + 1)) := by
    have h := card_ne_twice n
    have h' : ((univ.filter fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => p.1.1 ≠ p.2.1).card : ℝ) * 2 =
        Fintype.card (Paths Bool (n + 1) × Paths Bool (n + 1)) := by exact_mod_cast h
    linarith
  unfold prob
  rw [div_eq_iff hpos.ne']
  convert hN

/-- **The honest coin layer forks at round 0 with probability `½`.** -/
theorem coinLayer_forked_half (n : ℕ) :
    prob (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => (coinLayer n).Forked 0 p.1 p.2) = 1 / 2 := by
  rw [← prob_ne_half n]
  congr 1
  funext p
  exact propext (coinLayer_forked_zero n p.1 p.2)

/-- Binding's fork form fails for the honest coin layer at any `δ_fork < ½` (`BindingForkAt`'s body, at round 0). -/
theorem coinLayer_not_fork_le (n : ℕ) (δ : ℝ) (hδ : δ < 1 / 2) :
    ¬ ∀ w, prob (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => (coinLayer n).Forked w p.1 p.2) ≤ δ := by
  intro h
  have := h 0
  rw [coinLayer_forked_half] at this
  linarith

end Accounting.SecurityProofs.TS.RtLogic3
