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

theorem schemes_correct : strassen.correct = true ∧ winograd.correct = true ∧ trivial.correct = true := by
  decide

/-- `winogradOnce` with the f16 atom of the measured instances (`k = 16`; `16 · 510 · 128 ≤ 2^24`). -/
def winogradOnceMeasured : Sched :=
  .scheme winograd fun t => if t.val < 2 then .atom s8 else .atom ⟨16, 4097, some (16, 24)⟩

theorem winogradOnce_attains :
    (winogradOnce.valid h200Sheet schemes 0 1 256 256 = true ∧
      winogradOnce.valid rtx6000Sheet schemes 0 1 256 256 = true ∧
      winogradOnceMeasured.valid h200Measured schemes 0 1 256 256 = true ∧
      winogradOnceMeasured.valid rtx6000Measured schemes 0 1 256 256 = true) ∧
    winogradOnce.fast = true ∧ winogradOnceMeasured.fast = true ∧
    winogradOnce.cost 1 = 8 ^ 2 + 32 ∧ winogradOnceMeasured.cost 1 = 8 ^ 2 + 32 := by
  decide

/-- Naive tiled is valid on every instance and costs `8^(l+1)`, below Theorem 1's bound: `fast` is what it rests on. -/
theorem naive_below_bound :
    (instances.all fun fs => (Sched.atom s8).valid fs schemes 0 1 256 256) = true ∧
    (Sched.atom s8).cost 1 = 8 ^ 2 ∧ 8 ^ 2 < 8 ^ 2 + 32 := by
  decide

theorem up_to_hypotheses :
    winogradOnce.valid h200Sheet schemes 0 1 256 256 = true ∧ winogradOnce.fast = true ∧
      1 ≤ 1 ∧ 1 ≤ 2 ^ 1 * 1 ∧ 1 ≤ 4096 * 1 ∧ 8 * 1 ^ 3 < 1 ^ 3 * winogradOnce.cost 1 := by
  decide

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
