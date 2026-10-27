import Proofs.Flock.Claims.GameFinder
import Proofs.Core.Game.Expect
import Proofs.Core.Game.Prob

/-!
# A strategy's play as a `Claims.Finder`

The audit's play (`Verity.Vocab.game`) is a `Verity.Game`. A collision event of the play becomes a `Claims.Finder`
by running the strategy on a tape that holds a coin for every coin node of the tree it plays (`Game.Tape`,
`Game.run`). The finder's success is exactly the event's probability in the play (`tapeFinder_Pr`,
`prob_eq_ofReal_Pr`), so a bound on a collision event of the play is a bound on one fixed finder's success.
This is `FlockSoundness.Game.tapeFinder` (`Proofs.Flock.Claims.GameFinder`) for `Verity.Game`.
-/

namespace Verity.Game

open scoped ENNReal

variable {α : Type}

/-- **The strategy's coin tape**: a coin for every coin node of the tree `s` plays, drawn whether or not the run reaches
it. -/
def Tape : (g : Game α) → Strategy g → Type
  | .ret _, _ => Unit
  | .send _ k, s => Tape (k s.1) s.2
  | @Game.coin _ C _ _ k, s => C × ((c : C) → Tape (k c) (s c))

noncomputable instance instFintypeTape : (g : Game α) → (s : Strategy g) → Fintype (Tape g s)
  | .ret _, _ => inferInstanceAs (Fintype Unit)
  | .send _ k, s => instFintypeTape (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
      haveI := fun c => instFintypeTape (k c) (s c)
      haveI := Classical.decEq C
      inferInstanceAs (Fintype (C × ((c : C) → Tape (k c) (s c))))

instance instNonemptyTape : (g : Game α) → (s : Strategy g) → Nonempty (Tape g s)
  | .ret _, _ => inferInstanceAs (Nonempty Unit)
  | .send _ k, s => instNonemptyTape (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
      haveI := fun c => instNonemptyTape (k c) (s c)
      inferInstanceAs (Nonempty (C × ((c : C) → Tape (k c) (s c))))

/-- **The run of `s` on a tape**: at each coin node, the tape's coin there. -/
def run : (g : Game α) → (s : Strategy g) → Tape g s → α
  | .ret a, _, _ => a
  | .send _ k, s, t => run (k s.1) s.2 t
  | @Game.coin _ C _ _ k, s, t =>
      let t' : C × ((c : C) → Tape (k c) (s c)) := t
      run (k t'.1) (s t'.1) (t'.2 t'.1)

/-- **Averages over the tape are the play's expectations.** -/
theorem Ex_run (f : α → ℝ) (g : Game α) :
    ∀ s : Strategy g, Claims.Ex (fun t : Tape g s => f (run g s t)) = expect f g s := by
  induction g with
  | ret a =>
    intro s
    exact (Claims.Ex_inst (β := Unit) (instFintypeTape (.ret a) s) inferInstance (fun _ => f a)).trans
      (Claims.Ex_const (f a))
  | send M k ih =>
    intro s
    exact ih s.1 s.2
  | @coin C iF iN k ih =>
    intro s
    classical
    calc Claims.Ex (fun t : Tape (Game.coin C k) s => f (run (Game.coin C k) s t))
        = @Claims.Ex (C × ((c : C) → Tape (k c) (s c))) inferInstance
            (fun t : C × ((c : C) → Tape (k c) (s c)) => f (run (k t.1) (s t.1) (t.2 t.1))) :=
          Claims.Ex_inst (β := C × ((c : C) → Tape (k c) (s c))) (instFintypeTape (Game.coin C k) s) _ _
      _ = Claims.Ex (fun c : C =>
            Claims.Ex (fun t : (c : C) → Tape (k c) (s c) => f (run (k c) (s c) (t c)))) :=
          Claims.Ex_prod _
      _ = Claims.Ex (fun c : C => expect f (k c) (s c)) := by
          congr 1
          funext c
          exact (Claims.Ex_pi_eval (T := fun c => Tape (k c) (s c)) c (fun x => f (run (k c) (s c) x))).trans
            (ih c (s c))
      _ = expect f (Game.coin C k) s := rfl

/-- **An event's probability is its frequency on the tape.** -/
theorem Pr_run (p : α → Prop) (g : Game α) (s : Strategy g) :
    Claims.Pr (fun t : Tape g s => p (run g s t)) = (prob p g s).toReal := by
  classical
  rw [prob_toReal, ← Ex_run, ← Claims.Ex_ind]
  congr 1
  funext t
  unfold ind
  exact if_congr Iff.rfl rfl rfl

/-- **The play as a `Claims.Finder`**: `s`'s run on a tape, with output `out` and cost `cost` there. -/
noncomputable def tapeFinder {β : Type} (g : Game β) (s : Strategy g) (out : β → Option (Claims.Msg × Claims.Msg))
    (cost : β → ℕ) : Claims.Finder where
  S := Tape g s
  out t := out (run g s t)
  cost t := cost (run g s t)

/-- **The tape finder succeeds as the play does**: its probability of outputting a collision of `H` is the play's. -/
theorem tapeFinder_Pr {β : Type} (H : Claims.Msg → Claims.Msg) (g : Game β) (s : Strategy g)
    (out : β → Option (Claims.Msg × Claims.Msg)) (cost : β → ℕ) :
    (tapeFinder g s out cost).Pr H =
      (prob (fun b => ∃ u v, out b = some (u, v) ∧ Claims.Collision H u v) g s).toReal :=
  Pr_run (fun b => ∃ u v, out b = some (u, v) ∧ Claims.Collision H u v) g s

/-- `tapeFinder_Pr` in `ℝ≥0∞`, the form `prob` bounds are stated in. -/
theorem prob_eq_ofReal_Pr {β : Type} (H : Claims.Msg → Claims.Msg) (g : Game β) (s : Strategy g)
    (out : β → Option (Claims.Msg × Claims.Msg)) (cost : β → ℕ) :
    prob (fun b => ∃ u v, out b = some (u, v) ∧ Claims.Collision H u v) g s =
      ENNReal.ofReal ((tapeFinder g s out cost).Pr H) := by
  rw [tapeFinder_Pr, ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ g s))]

theorem tapeFinder_cost_le {β : Type} (g : Game β) (s : Strategy g) (out : β → Option (Claims.Msg × Claims.Msg))
    (cost : β → ℕ) (b : ℕ) (h : ∀ x, cost x ≤ b) (x : (tapeFinder g s out cost).S) :
    (tapeFinder g s out cost).cost x ≤ b :=
  h _

end Verity.Game
