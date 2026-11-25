module

public import Definitions.Slowdown.SecurityWitness

/-!
# No computation is feasible, accounted for and prohibited

The security argument's one statement over the three sets of `Definitions.Slowdown.Security`: `𝓗 ∩ 𝒜 ∩ ℬ = ∅`.

`Secure` is of the lock's `exclusion` kind, and has its witnesses at the run of `Definitions.Slowdown.SecurityWitness`,
listed under the lock's `witnesses`:

* `Secure_honest`: its premises hold together, at the run with every gate the declared circuit's;
* `Secure_nontrivial`, `Secure_nontrivial_accounted`, `Secure_nontrivial_bound`: at the run with no gate the declared
  circuit's, which holds a reference training algorithm of one counted multiplication, each of its three premises fails
  in turn while the other two hold, and so does its conclusion. So the statement is false with the feasible set, the
  audits' amounts or the count dropped.
-/

@[expose] public section

namespace Slowdown.Guarantees

open Slowdown.Security Slowdown.Security.Witness

/-- **No computation is feasible, accounted for and prohibited**:
`M ∈ H F.hw → M ∈ A F x_s x_b x_w y_b → M ∉ B F p Y`.
At a frame `F`, amounts `x_s`, `x_b`, `x_w` and `y_b`, a recipe `p` and a count `Y` at least the fit bound
(`F.BoundAtMost`): a computation that is feasible on the frame's hardware (`H F.hw`) and consistent with what the audits
account for (`A F x_s x_b x_w y_b`) holds no reference training algorithm of more than `Y` counted multiplications
that meets the recipe's premises (`B F p Y`).

Each set is written applied to `M`, which is membership in it unfolded, so that the lock reads `H` and `A` as the
statement's premises by name. -/
def Secure : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (F : Frame L E) (x_s x_b x_w y_b : ℕ) (p : Recipe) (Y : ℝ),
    F.BoundAtMost x_s x_b x_w y_b p Y → ∀ M : Computation L E, H F.hw M → A F x_s x_b x_w y_b M → ¬ B F p Y M

/-- **`Secure`'s premises hold together** at the run with every gate the declared circuit's, on the hardware of
one unit of work a tick, one bit of memory and one bit a tick of link, with the window covered: the run is a run
(`Faithful`, `Holding`), its server a location of the hardware, the audits account for all the server's memory, all
its link carries and all its work, and the fit bound is `0`. -/
def Secure_honest : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ ({true} : Finset Bool) ∈ (hw 1 1 1).units ∧
    (frame 1 1 1 ∅).BoundAtMost 1 1 1 1 recipe 0 ∧ H (hw 1 1 1) served ∧ A (frame 1 1 1 ∅) 1 1 1 1 served

/-- **`Secure` without the feasible set**: on a server that carries no bit, the run with no gate the declared
circuit's is consistent with audits that account for nothing, at the fit bound `1/4`; it is not feasible, and it is
prohibited at that count. -/
def Secure_nontrivial : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ ({true} : Finset Bool) ∈ (hw 1 0 1).units ∧
    (frame 1 0 1 ∅).BoundAtMost 0 0 0 1 recipe (1 / 4) ∧ A (frame 1 0 1 ∅) 0 0 0 1 trained ∧
    ¬ H (hw 1 0 1) trained ∧ ¬ ¬ B (frame 1 0 1 ∅) recipe (1 / 4) trained

/-- **`Secure` without the audits' amounts**: on the server that carries one bit, the run with no gate the declared
circuit's is feasible, at the fit bound `0` of audits that account for everything; it is not consistent with those
audits, and it is prohibited at that count. -/
def Secure_nontrivial_accounted : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ ({true} : Finset Bool) ∈ (hw 1 1 1).units ∧
    (frame 1 1 1 ∅).BoundAtMost 1 1 1 1 recipe 0 ∧ H (hw 1 1 1) trained ∧
    ¬ A (frame 1 1 1 ∅) 1 1 1 1 trained ∧ ¬ ¬ B (frame 1 1 1 ∅) recipe 0 trained

/-- **`Secure` without the count**: on the server that carries one bit, the run with no gate the declared circuit's
is feasible and consistent with audits that account for nothing; the count `0` is below their fit bound, and the run
is prohibited at it. -/
def Secure_nontrivial_bound : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ ({true} : Finset Bool) ∈ (hw 1 1 1).units ∧
    H (hw 1 1 1) trained ∧ A (frame 1 1 1 ∅) 0 0 0 1 trained ∧
    ¬ (frame 1 1 1 ∅).BoundAtMost 0 0 0 1 recipe 0 ∧ ¬ ¬ B (frame 1 1 1 ∅) recipe 0 trained

end Slowdown.Guarantees
