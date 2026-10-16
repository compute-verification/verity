import Proofs.Flock.Claims.Bridge.OfGame
import Proofs.Flock.Claims.Aggregate

/-!
# The transport: `Claims.run`'s probabilities are the live game's

For the public-coin verifier of a game (`ofGame k g I d`) and every prover `P*` and tape `ω`, the run on uniform
coins `r : Fin (k x) → Coin` and the game `g x` played by the decoded strategy `stratG` have one law:

* `ex_outcome`: `Ex_r f(outcome r) = expect f (g x) (stratG …)` for every `f`: each game coin uniform, independent,
  drawn after the message it answers (the coin law, `ex_playOn`);
* `Pr_acc_and`, `Pr_acc`: `Pr_r[run = acc ∧ Q(outcome)]` is the game's `expect` (and `prob`) of `d x` accepting with
  `Q`, for every event `Q` of the outcome; `Pr_acc_replay` reads the event off the run's transcript (`replay`), so it
  holds for every event of the transcript the game sees: acceptance with a given transcript is `Q := (· = some b₀)`;
* `ex_fork_run`, `forkPr_ofGame`: two runs forked at round `j` (`Claims.forkCoins`) have the law of one play to round
  `j` and two independent continuations (`forkX`), so `B`'s collision probability at its round guess `j`
  (`Claims.forkPr`) is `forkX` of the collision of the two decisions' `H` logs, averaged over the tape.

Each needs the game to end within `k x` rounds for the prover's strategy (`RoundsLe`).
-/

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

open Classical in
theorem Pr_eq_Ex {α : Type} [Fintype α] (p : α → Prop) : Pr p = Ex fun a => ind p a := by
  rw [Pr_eq_sum]
  rfl

/-- **The strategy the run plays in the game**: the prover's bytes, decoded. -/
abbrev stratG {β : Msg → Type} (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (g : (x : Msg) → Game (β x))
    (I : ∀ x, Iface Coin (g x)) (x : Msg) (ω : Ω) : Strategy (g x) :=
  toStrat (g x) (I x) (stratOf H P x ω (g x) (I x) [] none)

section Transport

variable {β : Msg → Type} (H : Msg → Msg) {Ω : Type} (P : Prover Ω) (k : Msg → ℕ) (g : (x : Msg) → Game (β x))
  (I : ∀ x, Iface Coin (g x)) (d : (x : Msg) → β x → OC Bool) (x : Msg)

/-- **The coin law for the run**: the run's outcome on uniform coins has the game's law against `stratG`. -/
theorem ex_outcome (ω : Ω) (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x))
    (f : β x → ℝ) :
    Ex (fun r : Fin (k x) → Coin => f (outcome H P k g I x ω r)) = expect f (g x) (stratG H P g I x ω) :=
  ex_playOn_root f (g x) (I x) _ (k x) hR

open Classical in
/-- **Acceptance with an event of the outcome**: `Pr_r[run = acc ∧ Q(outcome)]` is the game's probability that
`d x` accepts the outcome and `Q` holds, against `stratG`. -/
theorem Pr_acc_and (ω : Ω) (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x))
    (Q : β x → Prop) :
    Pr (fun r : Fin ((ofGame k g I d).k x) → Coin =>
        (run H P (ofGame k g I d) x ω r).out = .acc ∧ Q (outcome H P k g I x ω r)) =
      expect (ind fun b => ((d x b).eval H).1 = true ∧ Q b) (g x) (stratG H P g I x ω) := by
  rw [Pr_eq_Ex, ← ex_outcome H P k g I x ω hR]
  refine congrArg Ex (funext fun r => ?_)
  simp only [ind, (run_ofGame H P k g I d x ω r hR).1, toOut_eq_acc]
  congr

open Classical in
/-- **Acceptance**, as the game's `prob`. -/
theorem Pr_acc (ω : Ω) (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x)) :
    Pr (fun r : Fin ((ofGame k g I d).k x) → Coin => (run H P (ofGame k g I d) x ω r).out = .acc) =
      (prob (fun b => ((d x b).eval H).1 = true) (g x) (stratG H P g I x ω)).toReal := by
  rw [prob_toReal, ← ex_outcome H P k g I x ω hR, Pr_eq_Ex]
  refine congrArg Ex (funext fun r => ?_)
  simp only [ind, (run_ofGame H P k g I d x ω r hR).1, toOut_eq_acc]
  congr

open Classical in
/-- **Acceptance with an event of the transcript**: an event `Q` of what the game reads off the run's messages
(`replay`) has the game's probability; at `Q := (· = some b₀)`, acceptance with the transcript of outcome `b₀`. -/
theorem Pr_acc_replay (ω : Ω) (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x))
    (Q : Option (β x) → Prop) :
    Pr (fun r : Fin ((ofGame k g I d).k x) → Coin =>
        (run H P (ofGame k g I d) x ω r).out = .acc ∧ Q (replay ⟨g x, I x⟩ (run H P (ofGame k g I d) x ω r).msgs)) =
      expect (ind fun b => ((d x b).eval H).1 = true ∧ Q (some b)) (g x) (stratG H P g I x ω) := by
  rw [← Pr_acc_and H P k g I d x ω hR (fun b => Q (some b))]
  refine Pr_congr fun r => ?_
  rw [(run_ofGame H P k g I d x ω r hR).2.2]

/-- **The fork law for the run**: two runs sharing their first `j` coins have the law `forkX` of the game. -/
theorem ex_fork_run (ω : Ω) (hR : RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x))
    (j : Fin (k x)) (F : β x → β x → ℝ) :
    Ex (fun rr : (Fin (k x) → Coin) × (Fin (k x) → Coin) =>
        F (outcome H P k g I x ω rr.1) (outcome H P k g I x ω (forkCoins j rr.1 rr.2))) =
      forkX F j (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) none :=
  ex_fork F (g x) (I x) _ none (k x) j.val j.isLt hR

/-- The indicator that the two decisions' `H` logs hold a collision, `B`'s output on the two outcomes. -/
noncomputable def collOf {γ : Type} (H : Msg → Msg) (d : γ → OC Bool) (b b' : γ) : ℝ :=
  if (findColl (((d b).eval H).2 ++ ((d b').eval H).2)).isSome then 1 else 0

/-- **`B`'s fork, in the game**: the probability that `B`'s two runs, forked at round `j`, collide is the average over
the tape of `forkX` of the collision of the two decisions' `H` logs. -/
theorem forkPr_ofGame [Fintype Ω] [Nonempty Ω]
    (hR : ∀ ω, RoundsLe (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) false (k x))
    (hk : 0 < (ofGame k g I d).k x) (j : Fin ((ofGame k g I d).k x)) :
    forkPr H (ofGame k g I d) P x hk j =
      Ex fun ω : Ω => forkX (collOf H (d x)) j (g x) (I x) (stratOf H P x ω (g x) (I x) [] none) none := by
  classical
  unfold forkPr
  rw [Pr_eq_Ex, Ex_prod]
  refine congrArg Ex (funext fun ω => ?_)
  rw [← ex_fork_run H P k g I x ω (hR ω) j (collOf H (d x))]
  refine congrArg Ex (funext fun rr => ?_)
  have h1 : (run H P (ofGame k g I d) x ω rr.1).vLog = ((d x (outcome H P k g I x ω rr.1)).eval H).2 :=
    (run_ofGame H P k g I d x ω _ (hR ω)).2.1
  have h2 : (run H P (ofGame k g I d) x ω (forkCoins j rr.1 rr.2)).vLog =
      ((d x (outcome H P k g I x ω (forkCoins j rr.1 rr.2))).eval H).2 :=
    (run_ofGame H P k g I d x ω _ (hR ω)).2.1
  simp only [ind, B, collOf, h1, h2]

end Transport

end Claims.Bridge
