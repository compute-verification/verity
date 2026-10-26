import Definitions.Core.NaiveStrassen

/-!
# Naive tiled against Strassen: the statements and their non-vacuity witnesses

ACCEL NAIVE-STRASSEN (CMP-110; note:compute-research/20261026T0140Z-draft-accel-naive-strassen). The three statements
are open here: their paper proofs are the note's Theorems 1 and 2, by the exact cost tables of
`benchmarks/pouw/machine_model/verity_machine_model/strassen`
(art:a3e04d2dcf334f79a8e00e4801f786c3d678c85e142491a28c85413a4d23c1a6). The witnesses are proved: the schemes satisfy Brent's equations, the hypotheses of each statement hold of some
schedule, Theorem 1's bound is attained at `l = 1`, and without `fast` it fails (naive tiled itself).
-/

namespace Verity.Accel.NaiveStrassen

theorem naive_beats_strassen : ∀ fs ∈ instances, NaiveBeatsStrassen fs := by
  sorry

theorem naive_beats_strassen_up_to : NaiveBeatsStrassenUpTo h200Sheet 16113 := by
  sorry

theorem naive_beats_crt_strassen : ∀ fs ∈ instances, NaiveBeatsCrtStrassen fs := by
  sorry

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
      1 ≤ 1 ∧ 1 ≤ 2 ^ 1 * 1 ∧ 1 ≤ 16113 * 1 ∧ 8 * 1 ^ 3 < 1 ^ 3 * winogradOnce.cost 1 := by
  decide

/-- Two s8 residues, mod 256 and mod 255, meet `crtValid` at `n = 2` on every instance. -/
theorem crt_hypotheses :
    (instances.all fun fs => crtValid fs 2 0 [(256, .atom s8), (255, .atom s8)]) = true ∧
      8 * 2 ^ 3 < 2 ^ 3 * (([(256, Sched.atom s8), (255, Sched.atom s8)].map fun r => r.2.cost 0).foldl (· + ·) 0) := by
  decide

end Verity.Accel.NaiveStrassen
