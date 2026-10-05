import Mathlib.Data.Fintype.Pi
import Verity.SecurityProofs.Game
import RunModel.Audit

/-!
# The run's guarantees, stated over the parties' programs

A party's program (`RunModel.Prog`, built with the Lean toolchain alone) is a game against its environment in core's
model (`Verity.Game`): a request is a node where the environment sends any reply within the request's bound, and a coin
is a uniform draw. So the auditor's soundness over the whole run is core's `Sound` of its program's game, with the
environment (the developer, every worker, the network, the auditor's own Python) as the prover's strategy, and the
run's soundness composes from its parts' by core's `SoundBind` (`runSound_bind`).

Zero knowledge needs one more thing the auditor's side doesn't: the developer's environment is in two parts that reach
each other only through the developer's program. `inner` (the GPU rack and the developer's workers) holds the witness
and answers the program's requests on the inner channels; `outer` (the auditor, the network, everything else) answers
the rest and sees only what the program sends it. Which channels are inner is the deployment's egress premise.

This module is a sketch: the definitions type-check against core's model, and the two lemmas show the bridge carries
core's composition guarantees over. Neither guarantee is proved of a real program here.
-/

namespace RunModel.Security

open Verity
open scoped ENNReal

/-- `n` bytes of coins, as a finite type. -/
abbrev Bytes (n : Nat) := Fin n → Fin 256

def Bytes.toByteArray {n : Nat} (b : Bytes n) : ByteArray := ⟨(List.ofFn fun i => (b i).val.toUInt8).toArray⟩

/-- A party's program as a game against its environment. -/
def _root_.RunModel.Prog.toGame {α : Type} : Prog α → Game α
  | .ret a => .ret a
  | .ask r k => .send {b : ByteArray // b.size ≤ r.max} fun b => Prog.toGame (k b.1)
  | .coin n k => .coin (Bytes n) fun c => Prog.toGame (k c.toByteArray)

theorem _root_.RunModel.Prog.toGame_bind {α β : Type} (p : Prog α) (f : α → Prog β) :
    (p.bind f).toGame = p.toGame.bind fun a => (f a).toGame := by
  induction p with
  | ret a => rfl
  | ask r k ih => simp only [RunModel.Prog.bind, Prog.toGame, Game.bind, ih]
  | coin n k ih => simp only [RunModel.Prog.bind, Prog.toGame, Game.bind, ih]

/-- **Soundness of a run, for the auditor**: whatever its environment does, the auditor's program accepts while the
claim fails with probability at most `δ`. -/
def RunSound {α : Type} (auditor : Prog α) (accept claim : α → Prop) (δ : ℝ≥0∞) : Prop :=
  Sound auditor.toGame accept claim δ

/-- A program that, after any first part, continues with a sound program is sound: the run's soundness from its
sessions'. -/
theorem runSound_bind {α β : Type} (p : Prog α) (f : α → Prog β) (accept claim : β → Prop) (δ : ℝ≥0∞)
    (h : ∀ a, RunSound (f a) accept claim δ) : RunSound (p.bind f) accept claim δ := by
  unfold RunSound
  rw [Prog.toGame_bind]
  exact SecurityProofs.SoundBind p.toGame (fun a => (f a).toGame) accept claim δ h

/-- The developer's environment: `inner` answers the inner channels and may read the witness and every inner request;
`outer` answers the rest and sees only the outer requests. -/
structure SplitEnv (W : Type) where
  inner : W → List Req → ByteArray
  outer : List Req → ByteArray

open Classical in
/-- The probability, over the program's coins, that the outer environment's view (the requests sent to it, in order)
passes the test `P`. `hin` and `hout` are the requests each part has seen, latest first. -/
noncomputable def outerView {W α : Type} (isInner : String → Bool) (e : SplitEnv W) (w : W) (P : List Req → Prop) :
    Prog α → List Req → List Req → ℝ≥0∞
  | .ret _, _, hout => if P hout.reverse then 1 else 0
  | .ask r k, hin, hout =>
    if isInner r.chan then outerView isInner e w P (k (e.inner w (r :: hin))) (r :: hin) hout
    else outerView isInner e w P (k (e.outer (r :: hout))) hin (r :: hout)
  | .coin n k, hin, hout =>
    ∑ c : Bytes n, (Fintype.card (Bytes n) : ℝ≥0∞)⁻¹ * outerView isInner e w P (k c.toByteArray) hin hout

/-- **Zero knowledge of a run, for the developer** (statistical, error `ε`): a simulator that never sees the witness
gives every outer environment, whatever the inner one does, a view within `ε` of the developer's program's on any
witness, for every test of the view. -/
def RunZK {W β γ : Type} (isInner : String → Bool) (dev : W → Prog β) (sim : Prog γ) (ε : ℝ≥0∞) : Prop :=
  ∀ (e : SplitEnv W) (P : List Req → Prop) (w : W),
    let real := outerView isInner e w P (dev w) [] []
    let ideal := outerView isInner e w P sim [] []
    real ≤ ideal + ε ∧ ideal ≤ real + ε

/-- The toy audit's soundness has this shape: the auditor's program over the run, against any environment. Its claim
needs the committed table, which the Merkle root binds only computationally, so the statement is left as a shape. -/
example (d k : Nat) (claim : Audit.Verdict → Prop) (δ : ℝ≥0∞) : Prop :=
  RunSound (Audit.auditor d k) (·.accepted = true) claim δ

end RunModel.Security
