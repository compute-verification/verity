import Definitions.Core.NaiveStrassen
import Definitions.Core.Guarantees.NaiveStrassen
import Proofs.Core.NaiveStrassen.Eval

/-!
# Naive tiled against Strassen: the statements and their non-vacuity witnesses

Theorem 1 at `L = 12` and its size form at `c = 4096` are proved on all four instances from one table, the RTX PRO
6000 data sheet's, which every other instance's atoms map onto (`Packed.Dom`). The table is computed and checked by
`decide +kernel` (`NaiveStrassen/Eval.lean`): a pure kernel check, with no `native_decide` and no compiled code. Its
soundness is the induction on `Sched` in `NaiveStrassen/Sound.lean`.

Theorem 2 (root CRT, `NaiveBeatsCrtStrassen`) is open: its statement stays in `Definitions/Core/NaiveStrassen.lean`,
with its hypotheses' witness `crt_hypotheses` here, and nothing cites it.

The witnesses are proved:
* the schemes satisfy Brent's equations;
* the hypotheses of each statement hold of some schedule;
* Theorem 1's bound is attained at `l = 1`;
* without `fast` the bound fails (naive tiled itself).
-/

namespace Verity.Accel.NaiveStrassen

theorem naive_beats_strassen :
    NaiveBeatsStrassen h200Sheet 12 ∧ NaiveBeatsStrassen h200Measured 12 ∧ NaiveBeatsStrassen rtx6000Sheet 12 ∧
      NaiveBeatsStrassen rtx6000Measured 12 :=
  Packed.naiveBeatsStrassen_all

theorem naive_beats_strassen_up_to :
    NaiveBeatsStrassenUpTo h200Sheet 4096 ∧ NaiveBeatsStrassenUpTo h200Measured 4096 ∧
      NaiveBeatsStrassenUpTo rtx6000Sheet 4096 ∧ NaiveBeatsStrassenUpTo rtx6000Measured 4096 :=
  Packed.naiveBeatsStrassenUpTo_all

/-- Two s8 residues, mod 256 and mod 255, meet `crtValid` at `n = 2` on every instance. -/
theorem crt_hypotheses :
    (instances.all fun fs => crtValid fs 2 0 [(256, .atom s8), (255, .atom s8)]) = true ∧
      8 * 2 ^ 3 < 2 ^ 3 * (([(256, Sched.atom s8), (255, Sched.atom s8)].map fun r => r.2.cost 0).foldl (· + ·) 0) := by
  decide

end Verity.Accel.NaiveStrassen

namespace Verity.SecurityProofs

theorem NaiveBeatsStrassenServed : Verity.Guarantees.NaiveBeatsStrassenServed :=
  Verity.Accel.NaiveStrassen.naive_beats_strassen

theorem NaiveBeatsStrassenServedSize : Verity.Guarantees.NaiveBeatsStrassenServedSize :=
  Verity.Accel.NaiveStrassen.naive_beats_strassen_up_to

end Verity.SecurityProofs
