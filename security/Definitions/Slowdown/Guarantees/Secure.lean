module

public import Definitions.Slowdown.SecurityWitness

/-!
# No computation is feasible, accounted for and prohibited

The security argument's one statement over the three sets of `Definitions.Slowdown.Security`: `𝓗 ∩ 𝒜 ∩ ℬ = ∅`.

`Secure` is of the lock's `exclusion` kind, and has its witnesses at the run of `Definitions.Slowdown.SecurityWitness`,
listed under the lock's `witnesses`:

* `Secure_honest`: its premises hold together, at the run with every gate the declared circuit's and the fit bound;
* `Secure_nontrivial`, `Secure_nontrivial_accounted`, `Secure_nontrivial_bound`, `Secure_nontrivial_holds`: at the run
  with no gate the declared circuit's, which holds a reference training algorithm of one counted multiplication, each
  of its four premises fails in turn while the others hold, and so does its conclusion. So the statement is false
  with the feasible set, the audits' amounts, the count or the training bound's proof dropped.
-/

@[expose] public section

namespace Slowdown.Guarantees

open Slowdown.Security Slowdown.Security.Witness

/-- **No computation is feasible, accounted for and prohibited**:
`M ∈ H hw ∧ M ∈ A F x_s x_b x_w y_b → M ∉ B F tb sh Y`.
At a frame `F`, amounts `x_s`, `x_b`, `x_w` and `y_b`, a hardware `hw` the frame's servers are locations of (`F.On`),
a training bound `tb` that holds (`Training.Holds`), shared amounts `sh` and a count `Y` at least the bound on
training on that hardware (`F.BoundAtMost`): a computation that is feasible on the hardware (`H hw`) and consistent
with what the audits account for (`A F x_s x_b x_w y_b`) holds no reference training algorithm of more than `Y`
counted multiplications that meets the training bound's premises (`B F tb sh Y`).

`A` reads the frame and the amounts and nothing of `hw`, `tb`, `sh` or `Y`: what a run establishes
(`RunAccounted`) is `M ∈ A`, and this statement is applied to it afterwards, at each hardware and each training bound
one asks about. The training bound is the statement's one parameter for the route to a bound on training: the fit
bound is `Training.fit` at a recipe.

Each set is written applied to `M`, which is membership in it unfolded. -/
def Secure : Prop :=
  ∀ {L E : Type*} [Fintype E] [DecidableEq L] (F : Frame L) (x_s x_b x_w y_b : ℕ)
    (hw : ResourceAccounting.Hardware L E) (tb : Training) (sh : Shared) (Y : ℝ), F.On hw → tb.Holds → F.BoundAtMost hw x_s x_b x_w y_b tb sh Y →
    ∀ M : Computation L E, H hw M ∧ A F x_s x_b x_w y_b M → ¬ B F tb sh Y M

/-- **`Secure`'s premises hold together** at the run with every gate the declared circuit's, on the hardware of
one unit of work a tick, one bit of memory and one bit a tick of link, with the window covered: the run is a run
(`Faithful`, `Holding`), its server a location of the hardware, the audits account for all the server's memory, all
its link carries and all its work, the fit bound holds, and its count is `0`. -/
def Secure_honest : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ (frame ∅).On (hw 1 1 1) ∧ fitted.Holds ∧
    (frame ∅).BoundAtMost (hw 1 1 1) 1 1 1 1 fitted apart 0 ∧ H (hw 1 1 1) served ∧ A (frame ∅) 1 1 1 1 served

/-- **`Secure` without the feasible set**: on a server that carries no bit, the run with no gate the declared
circuit's is consistent with audits that account for nothing, at the fit bound `1/4`; it is not feasible, and it is
prohibited at that count. -/
def Secure_nontrivial : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ (frame ∅).On (hw 1 0 1) ∧ fitted.Holds ∧
    (frame ∅).BoundAtMost (hw 1 0 1) 0 0 0 1 fitted apart (1 / 4) ∧ A (frame ∅) 0 0 0 1 trained ∧
    ¬ H (hw 1 0 1) trained ∧ ¬ ¬ B (frame ∅) fitted apart (1 / 4) trained

/-- **`Secure` without the audits' amounts**: on the server that carries one bit, the run with no gate the declared
circuit's is feasible, at the fit bound `0` of audits that account for everything; it is not consistent with those
audits, and it is prohibited at that count. -/
def Secure_nontrivial_accounted : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ (frame ∅).On (hw 1 1 1) ∧ fitted.Holds ∧
    (frame ∅).BoundAtMost (hw 1 1 1) 1 1 1 1 fitted apart 0 ∧ H (hw 1 1 1) trained ∧
    ¬ A (frame ∅) 1 1 1 1 trained ∧ ¬ ¬ B (frame ∅) fitted apart 0 trained

/-- **`Secure` without the count**: on the server that carries one bit, the run with no gate the declared circuit's
is feasible and consistent with audits that account for nothing; the count `0` is below their fit bound, and the run
is prohibited at it. -/
def Secure_nontrivial_bound : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ (frame ∅).On (hw 1 1 1) ∧ fitted.Holds ∧
    H (hw 1 1 1) trained ∧ A (frame ∅) 0 0 0 1 trained ∧
    ¬ (frame ∅).BoundAtMost (hw 1 1 1) 0 0 0 1 fitted apart 0 ∧ ¬ ¬ B (frame ∅) fitted apart 0 trained

/-- **`Secure` without the training bound's proof**: at the training bound with no premise and the count `0`
(`free`), the run with no gate the declared circuit's is feasible and consistent with audits that account for
nothing, and the bound on training is `0`; the training bound does not hold, and the run is prohibited at that
count. -/
def Secure_nontrivial_holds : Prop :=
  run.Faithful actual.val ∧ run.Holding actual.held ∧ (frame ∅).On (hw 1 1 1) ∧
    (frame ∅).BoundAtMost (hw 1 1 1) 0 0 0 1 free apart 0 ∧ H (hw 1 1 1) trained ∧
    A (frame ∅) 0 0 0 1 trained ∧
    ¬ free.Holds ∧ ¬ ¬ B (frame ∅) free apart 0 trained

end Slowdown.Guarantees
