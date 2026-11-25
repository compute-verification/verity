module

public import Definitions.Core.Game
public import Definitions.Slowdown.SecurityWitness

/-!
# An accepted run is not prohibited

The security argument over a run's coins, in the tree's one form of soundness (`Verity.Sound`). A run is a
public-coin interaction `g` (`Verity.Game`), and `world a` is the computation the machines did behind its outcome
`a`: the outcome carries it, so a developer's strategy picks it. In `RunSecure` the developer's strategies are the
feasible ones: every outcome's computation is in `H`.

Each audit has a slot: its own acceptance and, at each window it covers, its clause of `𝒜`
(`HeldAtLeast`, `DeliveredAtLeast`, `DeliveredAtMost`, `WorkAtLeast`), sound with its own error. The slots are
hypotheses, so what an audit's guarantee must conclude to fill one is read off the statement. The verifier accepts
where the four audits do.

`RunAccounted`: where each audit is sound for its clause, the verifier accepts a run whose computation is outside `𝒜`
with probability at most the audits' errors added over the windows. It is what the system establishes, and it reads
no hardware and no training bound. `RunSecure` is `Secure` applied to it: at a hardware every outcome's computation
is feasible on and a training bound that holds, the same holds of a run whose computation is prohibited.

Its witnesses are at the run of `Definitions.Slowdown.SecurityWitness`, as an interaction with no message and no
coin. The honest one is the statement's completeness at that run: every audit's clause holds of the run with every
gate the declared circuit's. That a strategy each audit accepts outside its own error is accepted outside their sum
is `Slowdown.SecurityProofs.RunComplete`.
-/

@[expose] public section

namespace Slowdown.Guarantees

open Verity Slowdown.Security Slowdown.Security.Witness

open scoped ENNReal

/-- **An accepted run is accounted for, outside the audits' errors.** `g` is the run's interaction, `world a` the
computation behind its outcome `a`, and the verifier accepts where each of the four audits does. The amounts `x_s a`,
`x_b a`, `y_b a` and `x_w a` are what the audits certify at the outcome `a`. Where memory's audit is sound for
`HeldAtLeast` at `x_s`, bandwidth's for `DeliveredAtLeast` at `x_b` and the count for `DeliveredAtMost` at `y_b` at
each covered window, and work's for `WorkAtLeast` at `x_w` at each missed one, with errors `δ_s`, `δ_b`, `δ_y` and
`δ_w`, the run is sound for "the computation is in `A F` at the outcome's amounts", with error
`|missed| δ_w + (n J − |missed|) (δ_s + δ_b + δ_y)`. It reads no hardware and no training bound. -/
def RunAccounted : Prop :=
  ∀ {L E : Type*} [DecidableEq L] {α : Type} (g : Game α) (world : α → Computation L E)
    (F : Frame L) (x_s x_b x_w y_b : α → ℕ) (acc_s acc_b acc_y acc_w : α → Prop) (δ_s δ_b δ_y δ_w : ℝ≥0∞),
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_s (fun a => world a ∈ HeldAtLeast F i j (x_s a)) δ_s) →
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_b (fun a => world a ∈ DeliveredAtLeast F i j (x_b a)) δ_b) →
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_y (fun a => world a ∈ DeliveredAtMost F i j (y_b a)) δ_y) →
    (∀ i j, finProdFinEquiv (i, j) ∈ F.missed →
      Sound g acc_w (fun a => world a ∈ WorkAtLeast F i j (x_w a)) δ_w) →
    Sound g (fun a => acc_s a ∧ acc_b a ∧ acc_y a ∧ acc_w a)
      (fun a => A F (x_s a) (x_b a) (x_w a) (y_b a) (world a))
      (F.missed.card * δ_w + (F.n * F.cuts.J - F.missed.card : ℕ) * (δ_s + δ_b + δ_y))

/-- **An accepted run is not prohibited, outside the audits' errors**: `RunAccounted` and `Secure` together. At a
hardware `hw` the frame's servers are locations of, with every outcome's computation feasible on it, a training
bound `tb` that holds and every `Y a` at least the bound on training at the outcome's amounts (`F.BoundAtMost`), the
run is sound for "the computation is not in `B F tb sh (Y a)`", with `RunAccounted`'s error. -/
def RunSecure : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] {α : Type} (g : Game α) (world : α → Computation L E)
    (F : Frame L) (x_s x_b x_w y_b : α → ℕ) (hw : ResourceAccounting.Hardware L E) (tb : Training) (sh : Shared)
    (Y : α → ℝ) (acc_s acc_b acc_y acc_w : α → Prop) (δ_s δ_b δ_y δ_w : ℝ≥0∞), F.On hw → tb.Holds →
    (∀ a, F.BoundAtMost hw (x_s a) (x_b a) (x_w a) (y_b a) tb sh (Y a)) → (∀ a, H hw (world a)) →
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_s (fun a => world a ∈ HeldAtLeast F i j (x_s a)) δ_s) →
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_b (fun a => world a ∈ DeliveredAtLeast F i j (x_b a)) δ_b) →
    (∀ i j, finProdFinEquiv (i, j) ∉ F.missed →
      Sound g acc_y (fun a => world a ∈ DeliveredAtMost F i j (y_b a)) δ_y) →
    (∀ i j, finProdFinEquiv (i, j) ∈ F.missed →
      Sound g acc_w (fun a => world a ∈ WorkAtLeast F i j (x_w a)) δ_w) →
    Sound g (fun a => acc_s a ∧ acc_b a ∧ acc_y a ∧ acc_w a) (fun a => ¬ B F tb sh (Y a) (world a))
      (F.missed.card * δ_w + (F.n * F.cuts.J - F.missed.card : ℕ) * (δ_s + δ_b + δ_y))

/-- The run with no message and no coin: one outcome. -/
abbrev still : Game Unit := .ret ()

/-- **`RunSecure`'s premises hold together** at the run with every gate the declared circuit's, on the hardware of
one unit of work a tick, one bit of memory and one bit a tick of link, with the window covered and every audit
accepting: the run is a run (`Faithful`, `Holding`), its server a location of the hardware, the fit bound holds
and its count `0`, the computation is feasible, and each covered window's three clauses hold with no error. -/
def RunSecure_honest : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ (frame ∅).On (hw 1 1 1) ∧ fitted.Holds ∧
    (∀ _ : Unit, (frame ∅).BoundAtMost (hw 1 1 1) 1 1 1 1 fitted apart 0) ∧
    (∀ _ : Unit, H (hw 1 1 1) served) ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ HeldAtLeast (frame ∅) i j 1) 0) ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ DeliveredAtLeast (frame ∅) i j 1) 0) ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ DeliveredAtMost (frame ∅) i j 1) 0)

/-- **`RunSecure` without the audits**: at the run with no gate the declared circuit's, feasible and at the fit
bound `0` of audits that account for everything, memory's audit is not sound for its clause at any error below 1,
and the verifier accepts a prohibited computation with probability 1. -/
def RunSecure_nontrivial : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ (frame ∅).On (hw 1 1 1) ∧ fitted.Holds ∧
    (∀ _ : Unit, (frame ∅).BoundAtMost (hw 1 1 1) 1 1 1 1 fitted apart 0) ∧
    (∀ _ : Unit, H (hw 1 1 1) trained) ∧
    (∀ δ : ℝ≥0∞, δ < 1 → ∀ i j,
      ¬ Sound still (fun _ => True) (fun _ => trained ∈ HeldAtLeast (frame ∅) i j 1) δ) ∧
    ∀ δ : ℝ≥0∞, δ < 1 → ¬ Sound still (fun _ => True) (fun _ => ¬ B (frame ∅) fitted apart 0 trained) δ

/-- **`RunAccounted`'s premises hold together** at the run with every gate the declared circuit's, with the window
covered and every audit accepting: the run is a run (`Faithful`, `Holding`) and each covered window's three clauses
hold with no error. -/
def RunAccounted_honest : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ HeldAtLeast (frame ∅) i j 1) 0) ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ DeliveredAtLeast (frame ∅) i j 1) 0) ∧
    (∀ i j, finProdFinEquiv (i, j) ∉ (frame ∅).missed →
      Sound still (fun _ => True) (fun _ => served ∈ DeliveredAtMost (frame ∅) i j 1) 0)

/-- **`RunAccounted` without the audits**: at the run with no gate the declared circuit's, memory's audit is not
sound for its clause at any error below 1, and the verifier accepts a computation the audits' amounts don't account
for with probability 1. -/
def RunAccounted_nontrivial : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧
    (∀ δ : ℝ≥0∞, δ < 1 → ∀ i j,
      ¬ Sound still (fun _ => True) (fun _ => trained ∈ HeldAtLeast (frame ∅) i j 1) δ) ∧
    ∀ δ : ℝ≥0∞, δ < 1 → ¬ Sound still (fun _ => True) (fun _ => A (frame ∅) 1 1 1 1 trained) δ

end Slowdown.Guarantees
