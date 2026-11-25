module

public import Definitions.Core.Game
public import Definitions.Slowdown.SecurityWitness

/-!
# An accepted run is not prohibited

The security argument over a run's coins, in the tree's one form of soundness (`Verity.Sound`). A run is a
public-coin interaction `g` (`Verity.Game`), and `world a` is the computation the machines did behind its outcome
`a`: the outcome carries it, so a developer's strategy picks it. The developer's strategies are the feasible ones:
every outcome's computation is in `H`.

Each audit has a slot: its own acceptance and, at each window it covers, its clause of `𝒜`
(`HeldAtLeast`, `DeliveredAtLeast`, `DeliveredAtMost`, `WorkAtLeast`), sound with its own error. The slots are
hypotheses, so what an audit's guarantee must conclude to fill one is read off the statement. The verifier accepts
where the four audits do.

`RunSecure`: where each audit is sound for its clause, the verifier accepts a run whose computation is prohibited
with probability at most the audits' errors added over the windows. `Secure` is the step from `𝒜` to `ℬ` under it.

Its witnesses are at the run of `Definitions.Slowdown.SecurityWitness`, as an interaction with no message and no
coin. The honest one is the statement's completeness at that run: every audit's clause holds of the run with every
gate the declared circuit's. That a strategy each audit accepts outside its own error is accepted outside their sum
is `Slowdown.SecurityProofs.RunComplete`.
-/

@[expose] public section

namespace Slowdown.Guarantees

open Verity Slowdown.Security Slowdown.Security.Witness

open scoped ENNReal

/-- **An accepted run is not prohibited, outside the audits' errors.** `g` is the run's interaction, `world a` the
computation behind its outcome `a`, every one of them feasible on the frame's hardware, and the verifier accepts
where each of the four audits does. The amounts `x_s a`, `x_b a`, `y_b a` and `x_w a` are what the audits certify at
the outcome `a`. Where memory's audit is sound for `HeldAtLeast` at `x_s`, bandwidth's for `DeliveredAtLeast` at `x_b`
and the count for `DeliveredAtMost` at `y_b` at each covered window, and work's for `WorkAtLeast` at `x_w` at each
missed one, with errors `δ_s`, `δ_b`, `δ_y` and `δ_w`, the run is sound for "the computation is not in `B F p (Y a)`",
for every `Y a` at least the fit bound at the outcome's amounts (`F.BoundAtMost`), with error
`|missed| δ_w + (n J − |missed|) (δ_s + δ_b + δ_y)`. -/
def RunSecure : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] {α : Type} (g : Game α) (world : α → Computation L E)
    (F : Frame L E) (x_s x_b x_w y_b : α → ℕ) (p : Recipe) (Y : α → ℝ) (acc_s acc_b acc_y acc_w : α → Prop)
    (δ_s δ_b δ_y δ_w : ℝ≥0∞),
    (∀ a, F.BoundAtMost (x_s a) (x_b a) (x_w a) (y_b a) p (Y a)) → (∀ a, H F.hw (world a)) →
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_s (fun a => world a ∈ HeldAtLeast F i j (x_s a)) δ_s) →
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_b (fun a => world a ∈ DeliveredAtLeast F i j (x_b a)) δ_b) →
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_y (fun a => world a ∈ DeliveredAtMost F i j (y_b a)) δ_y) →
    (∀ i j, finProdFinEquiv (i, j) ∈ F.missed →
      Sound g acc_w (fun a => world a ∈ WorkAtLeast F i j (x_w a)) δ_w) →
    Sound g (fun a => acc_s a ∧ acc_b a ∧ acc_y a ∧ acc_w a) (fun a => ¬ B F p (Y a) (world a))
      (F.missed.card * δ_w + (F.n * F.cuts.J - F.missed.card : ℕ) * (δ_s + δ_b + δ_y))

/-- The run with no message and no coin: one outcome. -/
abbrev still : Game Unit := .ret ()

/-- **`RunSecure`'s premises hold together** at the run with every gate the declared circuit's, on the hardware of
one unit of work a tick, one bit of memory and one bit a tick of link, with the window covered and every audit
accepting: the run is a run (`Faithful`, `Holding`), its server a location of the hardware, the fit bound is `0`,
the computation is feasible, and each covered window's three clauses hold with no error. -/
def RunSecure_honest : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ ({true} : Finset Bool) ∈ (hw 1 1 1).units ∧
    (∀ _ : Unit, (frame 1 1 1 ∅).BoundAtMost 1 1 1 1 recipe 0) ∧ (∀ _ : Unit, H (hw 1 1 1) served) ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame 1 1 1 ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ HeldAtLeast (frame 1 1 1 ∅) i j 1) 0) ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame 1 1 1 ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ DeliveredAtLeast (frame 1 1 1 ∅) i j 1) 0) ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame 1 1 1 ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ DeliveredAtMost (frame 1 1 1 ∅) i j 1) 0)

/-- **`RunSecure` without the audits**: at the run with no gate the declared circuit's, feasible and at the fit
bound `0` of audits that account for everything, memory's audit is not sound for its clause at any error below 1,
and the verifier accepts a prohibited computation with probability 1. -/
def RunSecure_nontrivial : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ ({true} : Finset Bool) ∈ (hw 1 1 1).units ∧
    (∀ _ : Unit, (frame 1 1 1 ∅).BoundAtMost 1 1 1 1 recipe 0) ∧ (∀ _ : Unit, H (hw 1 1 1) trained) ∧
    (∀ δ : ℝ≥0∞, δ < 1 → ∀ i j,
      ¬ Sound still (fun _ => True) (fun _ => trained ∈ HeldAtLeast (frame 1 1 1 ∅) i j 1) δ) ∧
    ∀ δ : ℝ≥0∞, δ < 1 → ¬ Sound still (fun _ => True) (fun _ => ¬ B (frame 1 1 1 ∅) recipe 0 trained) δ

end Slowdown.Guarantees
