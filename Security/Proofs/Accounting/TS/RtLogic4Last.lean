import Proofs.Accounting.TS.Keyed
import Proofs.Accounting.TS.ProofsSound

/-!
# RT4-LOGIC: the last-round cheater under the restated `ProofLayer` (#2855)

Red-team scratch, not for landing.

* `guessLayer_not_pins`: `Keyed.guessLayer`, the guessing prover `guess_rel_last` reads, fails `ProofLayer.Pins` at
  `right` = "every operand is the honest one", the `right` its file gives it: on the all-`false` path the operand
  `true` satisfies every relation (the guess `true` is `true` keyed by `false`). So `ProofsSoundOfPinned` never applies
  to it, and `guess_rel_last` doesn't show that the restated statement covers a last-round cheater.
* `lastLayer`, `lastPlay`: a pinning check (each relation recomputes the operand, `v = false`, and reads the receipt the
  proof gives on the path keyed by σ_w) against a prover honest in every round but the last, where it skips the round's
  words and publishes the guess `true`. A sound verifier accepts a proof exactly when its opening satisfies its
  relation, so the last round passes on its lucky coin only. With a `Play` whose ideal game is a pinned `ReceiptGame`
  at receipt luck `½` and one deficient round, it meets every premise of `ProofsSoundOfPinned` (`Of`, `BindingAt 0`,
  `SoundAt 0`, `Pins`, `RightFixed`), and on its accepted path its fixed words are `(1 − 1/S)·N_w < N_w`, the bound
  `ProofsSound` gives (`lastPlay_proofsSound`, `lastPlay_tight`). The restated statement bounds this cheater rather
  than excluding it.
* `keyedPlay`: `Keyed.keyedLayer` with an honest `Play`, which its file doesn't give, so `Of` and `RightFixed` are met
  by an honest run too (`keyedPlay_proofsSound`).
-/

namespace Accounting.SecurityProofs.TS.RtLogic4

open Finset Accounting.TS Accounting.Assumptions.TS Pouw.RelativeSecurity
open Accounting.SecurityProofs.TS.Keyed

theorem key_false (c : Bool) : key false c = c := by cases c <;> rfl

/-- **`guessLayer` fails `Pins`**: on the all-`false` path, the operand `true` in every round satisfies every
relation, and it is not the honest operand. -/
theorem guessLayer_not_pins (n : ℕ) : ¬ (guessLayer n).Pins := by
  intro h
  have hr := h (Paths.const false (n + 1)) (fun _ _ => true) (fun w _ => by
    show true = key true (coinAt (Paths.const false (n + 1)) w)
    rw [coinAt_const]
    rfl)
  have h0 : true = false := hr 0 0
  cases h0

/-! ## A pinning check against a last-round guesser -/

/-- The receipt round `w` publishes on the path: the honest one (the operand `false` keyed by σ_w) in every round but
the last, where the prover skips the round's words and publishes the guess `true`. -/
def rcpt (n : ℕ) (x : Paths Bool (n + 1)) (w : Fin (n + 1)) : Bool :=
  if (w : ℕ) = n then true else key false (coinAt x w)

/-- **A pinning check** over `n + 1` rounds, one proof a round: round `w`'s relation recomputes the operand
(`v = false`) and reads the receipt the proof gives on the path, keyed by σ_w; a sound verifier accepts the proof
exactly when its opening satisfies the relation; a window right when every operand is the honest one. -/
def lastLayer (n : ℕ) : ProofLayer Bool (n + 1) 1 Bool where
  accepted x w _ := rcpt n x w = key false (coinAt x w)
  opened _ _ _ := false
  rel x w _ v := v = false ∧ rcpt n x w = key v (coinAt x w)
  right _ c := ∀ w i, c w i = false

/-- The ideal game split at the receipt: every round but the last honest (share `0`, passing), the last deficient
(share `1`, its words wrong), passing and keeping the honest committed value exactly when its coin is `true`. -/
noncomputable def lastRG : (n : ℕ) → ReceiptGame Bool (n + 1)
  | 0 => .round (fun _ => 1) (fun ω => ω = true) (fun _ => False) (fun ω => ω = true) fun _ => .done
  | n + 1 => .round (fun _ => 0) (fun _ => True) (fun _ => True) (fun _ => True) fun _ => lastRG n

/-- The cheater's play: accepted when every proof is, its run's fixed words `(1 − 1/S)·N_w` at `N_w = 1`. -/
noncomputable def lastPlay (n : ℕ) : Play Bool (n + 1) Unit Unit Unit Unit where
  game := (lastRG n).game
  accepts x := ∀ w, (lastLayer n).Proved x w
  run _ :=
    { run := { n := fun _ => 0, cycles := 0, clock := 0 }, nword := fun _ _ => 0, nfold := fun _ => 0,
      nfill := fun _ => 0, words := fun _ => 0, fixed := 1 - 1 / ((n + 1 : ℕ) : ℝ) }

theorem lastRG_accepts : ∀ (n : ℕ) (x : Paths Bool (n + 1)),
    (lastRG n).game.accepts x ↔ coinAt x (Fin.last n) = true
  | 0, x => by
    show (x.1 = true ∧ True) ↔ x.1 = true
    exact ⟨fun h => h.1, fun h => ⟨h, trivial⟩⟩
  | n + 1, x => by
    show (True ∧ (lastRG n).game.accepts x.2) ↔ coinAt x.2 (Fin.last n) = true
    rw [true_and]
    exact lastRG_accepts n x.2

theorem lastRG_deficiency : ∀ (n : ℕ) (x : Paths Bool (n + 1)), (lastRG n).game.deficiency x = 1
  | 0, _ => by
    show (1 : ℝ) + 0 = 1
    norm_num
  | n + 1, x => by
    show (0 : ℝ) + (lastRG n).game.deficiency x.2 = 1
    rw [zero_add]
    exact lastRG_deficiency n x.2

theorem rcpt_last (n : ℕ) (x : Paths Bool (n + 1)) : rcpt n x (Fin.last n) = true := by
  unfold rcpt
  simp

theorem lastLayer_of (n : ℕ) : (lastLayer n).Of (lastPlay n) := by
  refine ⟨fun _ hx => hx, fun x hx => ?_⟩
  obtain ⟨_, h1⟩ := hx (Fin.last n) 0
  have h2 : rcpt n x (Fin.last n) = key false (coinAt x (Fin.last n)) := h1
  rw [rcpt_last, key_false] at h2
  exact (lastRG_accepts n x).2 h2.symm

theorem lastLayer_binding (n : ℕ) : (lastLayer n).BindingAt 0 := by
  intro w
  refine le_of_eq (prob_zero fun x h => ?_)
  obtain ⟨_, i, hi⟩ := h
  exact hi rfl

theorem lastLayer_sound (n : ℕ) : (lastLayer n).SoundAt 0 := by
  intro w i
  refine le_of_eq (prob_zero fun x h => ?_)
  exact h.2 ⟨rfl, h.1⟩

theorem lastLayer_pins (n : ℕ) : (lastLayer n).Pins := by
  intro x c h w i
  exact (h w i).1

theorem lastPlay_rightFixed (n : ℕ) : RightFixed (lastPlay n) (lastLayer n) 1 := by
  intro x _ _
  show (1 - (lastRG n).game.deficiency x / ((n + 1 : ℕ) : ℝ)) * 1 ≤ 1 - 1 / ((n + 1 : ℕ) : ℝ)
  exact le_of_eq (by rw [lastRG_deficiency, mul_one])

theorem lastRG_pinned : ∀ n, (lastRG n).Pinned
  | 0 => ⟨fun _ h => h, fun _ => trivial⟩
  | n + 1 => ⟨fun _ _ => trivial, fun _ => lastRG_pinned n⟩

theorem prob_bool_true : prob (fun ω : Bool => ω = true) = 1 / 2 := by
  rw [prob_eq_card]
  have h1 : (Finset.univ.filter fun ω : Bool => ω = true).card = 1 := by decide
  rw [h1, Fintype.card_bool]
  norm_num

theorem prob_bool_all : prob (fun _ : Bool => True) = 1 := by
  rw [prob_eq_card, Finset.filter_true_of_mem fun _ _ => trivial, Finset.card_univ, Fintype.card_bool]
  norm_num

/-- **Receipt luck at `½`**: the deficient round keeps the honest committed value on half its coins. -/
theorem lastRG_receiptLuck : ∀ n, (lastRG n).ReceiptLuck (1 / 2)
  | 0 => by
    refine ⟨?_, fun _ => trivial⟩
    have e1 : (fun ω : Bool => ω = true ∧ ¬ False ∧ (0 : ℝ) < 1) = fun ω => ω = true :=
      funext fun ω => propext ⟨fun h => h.1, fun h => ⟨h, not_false, one_pos⟩⟩
    have e2 : (fun _ : Bool => ¬ False ∧ (0 : ℝ) < 1) = fun _ => True :=
      funext fun _ => propext ⟨fun _ => trivial, fun _ => ⟨not_false, one_pos⟩⟩
    show prob (fun ω : Bool => ω = true ∧ ¬ False ∧ (0 : ℝ) < 1) ≤
      1 / 2 * prob (fun _ : Bool => ¬ False ∧ (0 : ℝ) < 1)
    rw [e1, e2, prob_bool_true, prob_bool_all]
    norm_num
  | n + 1 => by
    refine ⟨?_, fun _ => lastRG_receiptLuck n⟩
    show prob (fun _ : Bool => True ∧ ¬ True ∧ (0 : ℝ) < 0) ≤
      1 / 2 * prob (fun _ : Bool => ¬ True ∧ (0 : ℝ) < 0)
    have h1 : prob (fun _ : Bool => True ∧ ¬ True ∧ (0 : ℝ) < 0) = 0 := prob_zero fun _ h => h.2.1 trivial
    have h2 : prob (fun _ : Bool => ¬ True ∧ (0 : ℝ) < 0) = 0 := prob_zero fun _ h => h.1 trivial
    rw [h1, h2]
    norm_num

/-- **The restated (I8) applies to the last-round cheater**: every premise of `ProofsSoundOfPinned` holds, so
`ProofsSound` holds at `δ_bind = δ_ZK = 0`. -/
theorem lastPlay_proofsSound (n : ℕ) : ProofsSound (lastPlay n) 1 0 0 1 :=
  _root_.Accounting.SecurityProofs.TS.ProofsSoundOfPinned Bool (n + 1) 1 (lastPlay n) (lastLayer n) 1 0 0
    (lastLayer_of n) (lastLayer_binding n) (lastLayer_sound n) (lastLayer_pins n) (lastPlay_rightFixed n)

/-- **It is a cheater the statement is about**: the all-`true` path is accepted, the ideal game charges it one
deficient round, and its fixed words are `(1 − 1/S)·N_w`, below `N_w` and equal to `ProofsSound`'s bound there. -/
theorem lastPlay_tight (n : ℕ) :
    (lastPlay n).accepts (Paths.const true (n + 1)) ∧
      (lastPlay n).game.deficiency (Paths.const true (n + 1)) = 1 ∧
      ((lastPlay n).run (Paths.const true (n + 1))).fixed < 1 ∧
      (1 - (lastPlay n).game.deficiency (Paths.const true (n + 1)) / ((n + 1 : ℕ) : ℝ)) * 1 =
        ((lastPlay n).run (Paths.const true (n + 1))).fixed := by
  refine ⟨fun w i => ?_, lastRG_deficiency n _, ?_, ?_⟩
  · show rcpt n (Paths.const true (n + 1)) w = key false (coinAt (Paths.const true (n + 1)) w)
    unfold rcpt
    rw [coinAt_const]
    split <;> rfl
  · show 1 - 1 / ((n + 1 : ℕ) : ℝ) < 1
    have : (0 : ℝ) < 1 / ((n + 1 : ℕ) : ℝ) := by positivity
    linarith
  · show (1 - (lastRG n).game.deficiency (Paths.const true (n + 1)) / ((n + 1 : ℕ) : ℝ)) * 1 =
      1 - 1 / ((n + 1 : ℕ) : ℝ)
    rw [lastRG_deficiency, mul_one]

/-- And the redrawn last coin is rejected: on the path whose last coin is `false` the window is not accepted. -/
theorem lastPlay_rejects_unlucky (n : ℕ) :
    ¬ (lastPlay n).accepts (splice n (Paths.const true (n + 1)) (Paths.const false (n + 1))) := by
  intro h
  have h2 : rcpt n (splice n (Paths.const true (n + 1)) (Paths.const false (n + 1))) (Fin.last n) =
      key false (coinAt (splice ((Fin.last n : Fin (n + 1)) : ℕ) (Paths.const true (n + 1))
        (Paths.const false (n + 1))) (Fin.last n)) := h (Fin.last n) 0
  rw [rcpt_last, coinAt_splice, coinAt_const, key_false] at h2
  cases h2

/-! ## `keyedLayer` with an honest play -/

/-- The honest ideal game: every round's share `0`, every round passing. -/
noncomputable def honestRG : (n : ℕ) → ReceiptGame Bool n
  | 0 => .done
  | n + 1 => .round (fun _ => 0) (fun _ => True) (fun _ => True) (fun _ => True) fun _ => honestRG n

/-- `keyedLayer`'s honest play: accepted when every proof is, its run's fixed words `N_w = 1`. -/
noncomputable def keyedPlay (n : ℕ) : Play Bool (n + 1) Unit Unit Unit Unit where
  game := (honestRG (n + 1)).game
  accepts x := ∀ w, (keyedLayer n).Proved x w
  run _ :=
    { run := { n := fun _ => 0, cycles := 0, clock := 0 }, nword := fun _ _ => 0, nfold := fun _ => 0,
      nfill := fun _ => 0, words := fun _ => 0, fixed := 1 }

theorem honestRG_accepts : ∀ (n : ℕ) (x : Paths Bool n), (honestRG n).game.accepts x
  | 0, _ => trivial
  | n + 1, x => ⟨trivial, honestRG_accepts n x.2⟩

theorem honestRG_deficiency : ∀ (n : ℕ) (x : Paths Bool n), (honestRG n).game.deficiency x = 0
  | 0, _ => rfl
  | n + 1, x => by
    show (0 : ℝ) + (honestRG n).game.deficiency x.2 = 0
    rw [zero_add]
    exact honestRG_deficiency n x.2

theorem keyedPlay_proofsSound (n : ℕ) : ProofsSound (keyedPlay n) 1 0 0 1 :=
  _root_.Accounting.SecurityProofs.TS.ProofsSoundOfPinned Bool (n + 1) 1 (keyedPlay n) (keyedLayer n) 1 0 0
    ⟨fun _ hx => hx, fun x _ => honestRG_accepts (n + 1) x⟩ (keyedLayer_binding n) (keyedLayer_sound n)
    (keyedLayer_pins n) (fun x _ _ => by
      show (1 - (honestRG (n + 1)).game.deficiency x / ((n + 1 : ℕ) : ℝ)) * 1 ≤ 1
      exact le_of_eq (by rw [honestRG_deficiency, zero_div, sub_zero, mul_one]))

end Accounting.SecurityProofs.TS.RtLogic4

#print axioms Accounting.SecurityProofs.TS.RtLogic4.guessLayer_not_pins
#print axioms Accounting.SecurityProofs.TS.RtLogic4.lastPlay_proofsSound
#print axioms Accounting.SecurityProofs.TS.RtLogic4.lastPlay_tight
#print axioms Accounting.SecurityProofs.TS.RtLogic4.lastPlay_rejects_unlucky
#print axioms Accounting.SecurityProofs.TS.RtLogic4.lastRG_receiptLuck
#print axioms Accounting.SecurityProofs.TS.RtLogic4.lastRG_pinned
#print axioms Accounting.SecurityProofs.TS.RtLogic4.keyedPlay_proofsSound
