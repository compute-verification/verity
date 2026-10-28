import Definitions.Accounting.Guarantees.TS.Receipt
import Proofs.Accounting.TS.Rounds
import Proofs.Accounting.TS.ReceiptLi
import Proofs.Accounting.TS.ReceiptDuoCP
import Proofs.Pouw.Receipt.Fx1

/-!
# (I7) from the receipt: proofs

The bridge (`luck_of_receipt`), round by round: with `A = Pr[right ∧ d > 0]`, `X = Pr[¬right ∧ d > 0]` and
`D = Pr[d > 0] = A + X`, a pinned round passes only if `same`, so
`Pr[pass ∧ d > 0] ≤ A + Pr[same ∧ ¬right ∧ d > 0] ≤ A + pr·X`, and
`(p + (1 − p)·pr)·D − A − pr·X = (1 − pr)·(p·D − A) ≥ 0` for `pr ≤ 1` and `A ≤ p·D`.

Lemma B in case (a) (`receiptLuck_caseA`): in each unit the run's words are the honest ones with one replaced, so equal
final states from the pinned start give the replaced word right (`duoRun_update_injective`, and for fx1
`fx1Run_update_injective`); for li, duo and fx1 checkpoint units, equal checkpoints segment by segment give each
segment's word right (`LiLemmas.right_of_same`, `DuoCPLemmas.right_of_same`, `DuoCPLemmas.fx1_right_of_same`). The
event `same ∧ ¬right` is empty.

The keyed rounds (`operandLuck_wrongWords`): a share positive only where a deficient word is wrong makes
`right ∧ d > 0` empty, the operand luck at 0.

The operand luck at ¾ for degree-2 operands (`operandLuck_cube`): conditioned on the coins before the bits, a deficient
round's error is a nonzero degree-2 function, which vanishes at uniform bits with probability at most ¾ (`luck_quad`).
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity Pouw.Receipt

namespace ReceiptLemmas

theorem prob_and_add_and_not {Ω : Type*} [Fintype Ω] (p q : Ω → Prop) :
    prob (fun ω => p ω ∧ q ω) + prob (fun ω => ¬p ω ∧ q ω) = prob q := by
  classical
  rw [prob_eq_expect, prob_eq_expect, prob_eq_expect q, ← Rounds.expect_add]
  congr 1
  funext ω
  by_cases hp : p ω <;> by_cases hq : q ω <;> simp [hp, hq]

theorem DuoUnits.right_of_same {Ω U : Type} {n L : ℕ} (D : DuoUnits Ω U n L) (ω : Ω) (h : D.same ω) :
    D.right ω := fun u =>
  Pouw.ReceiptProofs.duoRun_update_injective (D.k u) (D.a u) (D.r u) (D.start ω u) (D.honest ω u) (D.pos ω u)
    (by simp only [Function.update_eq_self]; exact h u)

theorem Fx1Units.right_of_same {Ω U : Type} {n L : ℕ} (D : Fx1Units Ω U n L) (ω : Ω) (h : D.same ω) :
    D.right ω := fun u =>
  Pouw.ReceiptProofs.fx1Run_update_injective (D.k u) (D.a u) (D.r u) (D.m u) (D.start ω u) (D.honest ω u)
    (D.pos ω u) (by simp only [Function.update_eq_self]; exact h u)

end ReceiptLemmas

open ReceiptLemmas

/-- The bridge: the operand luck at `p` and the receipt luck at `pr ≤ 1` give (L) at `p + (1 − p)·pr` on a pinned
game. -/
theorem luck_of_receipt {Ω : Type} [Fintype Ω] :
    ∀ {S : ℕ} (G : ReceiptGame Ω S) (p pr : ℝ), G.Pinned → G.OperandLuck p → G.ReceiptLuck pr → pr ≤ 1 →
      RoundGame.LuckAt (p + (1 - p) * pr) G.game
  | _, .done, _, _, _, _, _, _ => trivial
  | _, .round d pass right same next, p, pr, hpin, hop, hrl, hpr => by
    obtain ⟨hpin, hpinN⟩ := hpin
    obtain ⟨hd, hop, hopN⟩ := hop
    obtain ⟨hrl, hrlN⟩ := hrl
    refine And.intro hd (And.intro ?_ fun ω => luck_of_receipt (next ω) p pr (hpinN ω) (hopN ω) (hrlN ω) hpr)
    have hsplit := prob_and_add_and_not right (fun ω => 0 < d ω)
    have h1 : prob (fun ω => pass ω ∧ 0 < d ω) ≤
        prob (fun ω => right ω ∧ 0 < d ω) + prob (fun ω => same ω ∧ ¬right ω ∧ 0 < d ω) := by
      refine (prob_mono fun ω h => ?_).trans (prob_or_le _ _)
      by_cases hr : right ω
      · exact Or.inl ⟨hr, h.2⟩
      · exact Or.inr ⟨hpin ω h.1, hr, h.2⟩
    rw [← hsplit] at hop ⊢
    nlinarith [mul_nonneg (sub_nonneg.mpr hpr) (sub_nonneg.mpr hop)]

/-- Lemma B in case (a): the receipt luck is 0. -/
theorem receiptLuck_caseA {Ω : Type} [Fintype Ω] :
    ∀ {S : ℕ} (G : ReceiptGame Ω S), G.CaseA → G.ReceiptLuck 0
  | _, .done, _ => trivial
  | _, .round d pass right same next, h => by
    obtain ⟨hA, hAN⟩ := h
    refine And.intro ?_ fun ω => receiptLuck_caseA (next ω) (hAN ω)
    have hsr : ∀ ω, same ω → right ω := by
      rcases hA with ⟨U, w, L, D, hD⟩ | ⟨U, w, L, D, hD⟩ | ⟨U, w, S, W, C, H, D, hD⟩ | ⟨U, w, S, W, C, H, D, hD⟩ |
        ⟨U, w, S, W, C, H, D, hD⟩
      · exact fun ω h => (hD ω).1.mpr (DuoUnits.right_of_same D ω ((hD ω).2.mp h))
      · exact fun ω h => (hD ω).1.mpr (Fx1Units.right_of_same D ω ((hD ω).2.mp h))
      · exact fun ω h => (hD ω).1.mpr (LiLemmas.right_of_same D ω ((hD ω).2.mp h))
      · exact fun ω h => (hD ω).1.mpr (DuoCPLemmas.right_of_same D ω ((hD ω).2.mp h))
      · exact fun ω h => (hD ω).1.mpr (DuoCPLemmas.fx1_right_of_same D ω ((hD ω).2.mp h))
    rw [zero_mul]
    exact (prob_mono (q := fun _ => False) fun ω h => h.2.1 (hsr ω h.1)).trans Rounds.prob_false.le

/-- The operand luck at ¾ for degree-2 operands: slice at the other coins `a`; a deficient `a` has a nonzero degree-2
error, which vanishes at uniform bits with probability at most ¾ (`prob_zero_le_of_cubeQuad`). -/
theorem operandLuck_cube (R : Type) [CommRing R] {α V : Type} [Fintype α] [Fintype V] [DecidableEq V] :
    ∀ {S : ℕ} (G : ReceiptGame (α × Finset V) S), G.CubeOperands R → G.OperandLuck (3 / 4)
  | _, .done, _ => trivial
  | _, .round d _ right _ next, h => by
    classical
    obtain ⟨hd, ⟨δ, e, he, hδ, hne, hright⟩, hN⟩ := h
    refine And.intro hd (And.intro ?_ fun ω => operandLuck_cube R (next ω) (hN ω))
    have hpos : prob (fun ω : α × Finset V => 0 < d ω) = prob (fun a => 0 < δ a) := by
      rw [← prob_fst (β := Finset V) (fun a => 0 < δ a)]
      congr 1
      funext ω
      rw [hδ ω]
    rw [hpos]
    calc prob (fun ω : α × Finset V => right ω ∧ 0 < d ω)
        ≤ prob (fun ω : α × Finset V => e ω.1 (bitInd ω.2) = 0 ∧ 0 < δ ω.1) :=
          prob_mono fun ω h => ⟨hright ω h.1, by rw [← hδ ω]; exact h.2⟩
      _ = expect fun a => prob (fun S : Finset V => e a (bitInd S) = 0 ∧ 0 < δ a) := by
          rw [prob_eq_expect (fun ω : α × Finset V => e ω.1 (bitInd ω.2) = 0 ∧ 0 < δ ω.1), expect_prod]
          congr 1
          funext a
          rw [prob_eq_expect]
      _ ≤ expect fun a => 3 / 4 * (if 0 < δ a then (1 : ℝ) else 0) := by
          refine expect_mono fun a => ?_
          split_ifs with ha
          · rw [mul_one]
            exact (prob_mono (q := fun S : Finset V => e a (bitInd S) = 0) fun S h => h.1).trans
              (prob_zero_le_of_cubeQuad (he a) (hne a ha))
          · rw [mul_zero]
            exact (prob_mono (q := fun _ : Finset V => False) fun S h => ha h.2).trans Rounds.prob_false.le
      _ = 3 / 4 * prob (fun a => 0 < δ a) := by
          rw [expect_mul_left, prob_eq_expect]

/-- The keyed rounds' share: a share positive only where a deficient word is wrong gives the operand luck at 0. -/
theorem operandLuck_wrongWords {Ω : Type} [Fintype Ω] :
    ∀ {S : ℕ} (G : ReceiptGame Ω S), G.WrongWords → G.OperandLuck 0
  | _, .done, _ => trivial
  | _, .round d _ right _ next, h => by
    obtain ⟨hd, hw, hN⟩ := h
    refine And.intro hd (And.intro ?_ fun ω => operandLuck_wrongWords (next ω) (hN ω))
    rw [zero_mul]
    exact (prob_mono (q := fun _ => False) fun ω h => hw ω h.2 h.1).trans Rounds.prob_false.le

theorem LuckOfReceipt : Accounting.Guarantees.TS.LuckOfReceipt :=
  fun G p pr => luck_of_receipt G p pr

theorem ReceiptLuckCaseA : Accounting.Guarantees.TS.ReceiptLuckCaseA :=
  fun G => receiptLuck_caseA G

theorem LuckCaseA : Accounting.Guarantees.TS.LuckCaseA := fun G p hpin hA hop => by
  have h := luck_of_receipt G p 0 hpin hop (receiptLuck_caseA G hA) zero_le_one
  rwa [mul_zero, add_zero] at h

theorem OperandLuckCube : Accounting.Guarantees.TS.OperandLuckCube :=
  fun R _ _ _ _ _ _ _ G => operandLuck_cube R G

theorem LuckCaseACube : Accounting.Guarantees.TS.LuckCaseACube :=
  fun R _ _ _ _ _ _ _ G hpin hA hcube => LuckCaseA G (3 / 4) hpin hA (operandLuck_cube R G hcube)

theorem LiSegmentsBind : Accounting.Guarantees.TS.LiSegmentsBind :=
  fun D ω => LiLemmas.right_of_same D ω

theorem LiWindowUnbound : Accounting.Guarantees.TS.LiWindowUnbound :=
  fun n D ω => LiLemmas.windowUnbound n D ω

theorem DuoSegmentsBind : Accounting.Guarantees.TS.DuoSegmentsBind :=
  fun D ω => DuoCPLemmas.right_of_same D ω

theorem Fx1SegmentsBind : Accounting.Guarantees.TS.Fx1SegmentsBind :=
  fun D ω => DuoCPLemmas.fx1_right_of_same D ω

theorem LuckCaseAKeyed : Accounting.Guarantees.TS.LuckCaseAKeyed :=
  fun G hpin hA hw => LuckCaseA G 0 hpin hA (operandLuck_wrongWords G hw)

end Accounting.SecurityProofs.TS
