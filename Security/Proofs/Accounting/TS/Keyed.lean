import Definitions.Accounting.TS.Drawn
import Proofs.Pouw.RelativeSecurity

/-!
# An honest coin-keyed proof layer meets (I8)'s premises

The positive twin of RT3-LOGIC-1's `coinLayer`: a served round's unit takes the round's salt σ_w and outputs a receipt
keyed by it (`window.round_key`, `h1tile.live_salt`), committed after σ_w in a tree of its own, while the round's leaves,
committed before σ_w, hold its operands. `keyedLayer n` is such a window over `n + 1` rounds, one proof a round: each
proof accepted, opening its round's operand (`false`), its relation reading σ_w and the receipt the proof gives on the
path (the operand keyed by σ_w). The receipt moves with σ_w (`keyedLayer_receipt_moves`), so as a leaf it would fork
with every redraw of σ_w, as `coinLayer` does. Read by the relation instead, binding (both forms) and proof soundness
hold at `0` and the relations pin (`keyedLayer_meets_budget`), so the restated premises hold for an honest
coin-dependent layer at the budget `S·(√δ_fork + n_π·δ_ZK) ≤ 2^-41`, `S = 48,000` (`keyedLayer_meets_budget_dc`).

`guess_rel_last`: for a prover that guesses round `w`'s receipt (`true`) instead of keying its operand, the committed
values satisfy every relation on one path and fail the last round's on its continuation that redraws the last coin, so
`ProofLayer.Of` no longer has the ideal game accept every last coin (RT3-LOGIC's `game_accepts_last`): the relation
reads the coin the receipt is keyed by.
-/

namespace Accounting.SecurityProofs.TS.Keyed

open Finset Accounting.TS Pouw.RelativeSecurity

theorem coinAt_splice {Ω : Type} : ∀ {n : ℕ} (w : Fin n) (x z : Paths Ω n),
    coinAt (splice (w : ℕ) x z) w = coinAt z w
  | 0, w, _, _ => w.elim0
  | _ + 1, ⟨0, _⟩, _, _ => rfl
  | _ + 1, ⟨w + 1, h⟩, x, z => coinAt_splice ⟨w, Nat.lt_of_succ_lt_succ h⟩ x.2 z.2

theorem coinAt_const {Ω : Type} (a : Ω) : ∀ {n : ℕ} (w : Fin n), coinAt (Paths.const a n) w = a
  | 0, w => w.elim0
  | _ + 1, ⟨0, _⟩ => rfl
  | _ + 1, ⟨w + 1, h⟩ => coinAt_const a ⟨w, Nat.lt_of_succ_lt_succ h⟩

theorem prob_zero {Ω : Type} [Fintype Ω] {p : Ω → Prop} (h : ∀ ω, ¬ p ω) : prob p = 0 := by
  classical
  rw [prob_eq_card, filter_false_of_mem fun ω _ => h ω, card_empty, Nat.cast_zero, zero_div]

/-- A unit's receipt: its operand keyed by the round's coin. -/
def key (v c : Bool) : Bool := xor v c

theorem key_inj {v c : Bool} (h : key false c = key v c) : v = false := by
  revert h
  cases v <;> cases c <;> decide

theorem key_flip (c : Bool) : key false c ≠ key false (!c) := by
  cases c <;> decide

/-- **An honest coin-keyed layer** over `n + 1` rounds, one proof a round: every proof accepted, opening its round's
leaf, the operand `false`, committed before σ_w; its relation, on the path, that the receipt the proof gives there (the
honest one, the operand keyed by σ_w) is the opened operand keyed by σ_w; a window right when every operand is the
honest one. -/
def keyedLayer (n : ℕ) : ProofLayer Bool (n + 1) 1 Bool where
  accepted _ _ _ := True
  opened _ _ _ := false
  rel x w _ v := key false (coinAt x w) = key v (coinAt x w)
  right _ c := ∀ w i, c w i = false

theorem keyedLayer_committed (n : ℕ) (x : Paths Bool (n + 1)) (w : Fin (n + 1)) (i : Fin 1) :
    (keyedLayer n).committed x w i = false := rfl

/-- **The receipt moves with σ_w**: on every path, redrawing round `w`'s coin changes round `w`'s receipt. -/
theorem keyedLayer_receipt_moves (n : ℕ) (w : Fin (n + 1)) (x : Paths Bool (n + 1)) :
    ∃ z, key false (coinAt x w) ≠ key false (coinAt (splice (w : ℕ) x z) w) := by
  refine ⟨Paths.const (!coinAt x w) (n + 1), ?_⟩
  rw [coinAt_splice, coinAt_const]
  exact key_flip _

theorem keyedLayer_binding (n : ℕ) : (keyedLayer n).BindingAt 0 := by
  intro w
  refine le_of_eq (prob_zero fun x h => ?_)
  obtain ⟨_, i, hi⟩ := h
  exact hi (keyedLayer_committed n x w i).symm

theorem keyedLayer_fork (n : ℕ) (w : Fin (n + 1)) :
    prob (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => (keyedLayer n).Forked w p.1 p.2) ≤ 0 := by
  refine le_of_eq (prob_zero fun p h => ?_)
  obtain ⟨_, _, i, hi⟩ := h
  exact hi rfl

theorem keyedLayer_sound (n : ℕ) : (keyedLayer n).SoundAt 0 := by
  intro w i
  refine le_of_eq (prob_zero fun x h => ?_)
  exact h.2 rfl

theorem keyedLayer_pins (n : ℕ) : (keyedLayer n).Pins := by
  intro x c h w i
  exact key_inj (h w i)

/-- **The honest coin-keyed layer meets the restated premises at the budget**, over `S = n + 1` rounds: binding at
`δ_bind = 0` and its fork form at `δ_fork = 0`, proof soundness at `δ_ZK = 0` and pinning, with
`S·(δ_bind + n_π·δ_ZK)` and `S·(√δ_fork + n_π·δ_ZK)` both within `2^-41`. -/
theorem keyedLayer_meets_budget (n : ℕ) :
    ∃ δbind δfork δzk : ℝ, (keyedLayer n).BindingAt δbind ∧
      (∀ w, prob (fun p : Paths Bool (n + 1) × Paths Bool (n + 1) => (keyedLayer n).Forked w p.1 p.2) ≤ δfork) ∧
      (keyedLayer n).SoundAt δzk ∧ (keyedLayer n).Pins ∧
      ((n + 1 : ℕ) : ℝ) * (δbind + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧
      ((n + 1 : ℕ) : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 := by
  refine ⟨0, 0, 0, keyedLayer_binding n, keyedLayer_fork n, keyedLayer_sound n, keyedLayer_pins n, ?_, ?_⟩
  · simp only [mul_zero, add_zero]
    positivity
  · simp only [Real.sqrt_zero, mul_zero, add_zero]
    positivity

/-- `keyedLayer_meets_budget` at the sm_120 and DC headlines' `S = 48,000`. -/
theorem keyedLayer_meets_budget_dc :
    ∃ δbind δfork δzk : ℝ, (keyedLayer 47999).BindingAt δbind ∧
      (∀ w, prob (fun p : Paths Bool 48000 × Paths Bool 48000 => (keyedLayer 47999).Forked w p.1 p.2) ≤ δfork) ∧
      (keyedLayer 47999).SoundAt δzk ∧ (keyedLayer 47999).Pins ∧
      (48000 : ℝ) * (δbind + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 ∧
      (48000 : ℝ) * (Real.sqrt δfork + ((1 : ℕ) : ℝ) * δzk) ≤ (1 / 2) ^ 41 := by
  obtain ⟨δb, δf, δz, hb, hf, hz, hp, h1, h2⟩ := keyedLayer_meets_budget 47999
  have e : ((47999 + 1 : ℕ) : ℝ) = 48000 := by norm_num
  rw [e] at h1 h2
  exact ⟨δb, δf, δz, hb, hf, hz, hp, h1, h2⟩

/-- A prover that guesses each round's receipt (`true`) rather than keying its operand: proofs accepted, the operand
`false` opened, and the relation that the guessed receipt is the operand keyed by σ_w. -/
def guessLayer (n : ℕ) : ProofLayer Bool (n + 1) 1 Bool where
  accepted _ _ _ := True
  opened _ _ _ := false
  rel x w _ v := true = key v (coinAt x w)
  right _ c := ∀ w i, c w i = false

/-- **The relation reads the last coin**: on the all-`true` path the guessing prover's committed values satisfy every
relation, and on its continuation that redraws the last round's coin as `false` they fail the last round's. So `Of`'s
tie to the ideal game reads the coin the receipt is keyed by, and a prover deficient in the last round is accepted by
the ideal game on its lucky coins only (RT3-LOGIC's `game_accepts_last` had it accepted on every last coin). -/
theorem guess_rel_last (n : ℕ) :
    (∀ w i, (guessLayer n).rel (Paths.const true (n + 1)) w i
        ((guessLayer n).committed (Paths.const true (n + 1)) w i)) ∧
      ¬ ∀ w i, (guessLayer n).rel (splice n (Paths.const true (n + 1)) (Paths.const false (n + 1))) w i
        ((guessLayer n).committed (splice n (Paths.const true (n + 1)) (Paths.const false (n + 1))) w i) := by
  refine ⟨fun w _ => ?_, fun h => ?_⟩
  · show true = key false (coinAt (Paths.const true (n + 1)) w)
    rw [coinAt_const]
    rfl
  · have hl := h (Fin.last n) 0
    change true = key false (coinAt (splice ((Fin.last n : Fin (n + 1)) : ℕ) (Paths.const true (n + 1))
      (Paths.const false (n + 1))) (Fin.last n)) at hl
    rw [coinAt_splice, coinAt_const] at hl
    exact absurd hl (by decide)

end Accounting.SecurityProofs.TS.Keyed
