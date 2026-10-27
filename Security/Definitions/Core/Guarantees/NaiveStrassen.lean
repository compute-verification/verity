import Definitions.Core.NaiveStrassen

/-!
# Naive tiled against Strassen: D4's guarantees

Theorem 1 at the served depth, and its size form, on all four instances (`Verity.Accel.NaiveStrassen`): the H200
SXM and the RTX PRO 6000, each at its data sheet's and its measured instruction `k`. Served dimensions are at most
65536, and a leaf is at least 16 (sm_120 has no `wgmma`, and the H200 is taken at `Open.sync = 1`), so a recursion
on a served product is at most `log₂(65536 / 16) = 12` deep, and the size form's `c` is `65536 / 16 = 4096`. Both are
about int8 operands at `(256, 256)`.
-/

namespace Verity.Guarantees

open Verity.Accel.NaiveStrassen

/-- **Theorem 1 at the served depth**: on each instance, at every depth `l ≤ 12`, every schedule for int8 operands
that applies Strassen or Winograd somewhere costs at least naive tiled's `8^(l+1)` plus 32. -/
def NaiveBeatsStrassenServed : Prop :=
  NaiveBeatsStrassen h200Sheet 12 ∧ NaiveBeatsStrassen h200Measured 12 ∧ NaiveBeatsStrassen rtx6000Sheet 12 ∧
    NaiveBeatsStrassen rtx6000Measured 12

/-- **Theorem 1 in size**: on each instance, an `n × n` int8 product padded to `2^l` leaves of size `b`, at any
depth `l`, with Strassen or Winograd somewhere, costs more than naive tiled's `8 n³` whenever `n ≤ 4096 b`. -/
def NaiveBeatsStrassenServedSize : Prop :=
  NaiveBeatsStrassenUpTo h200Sheet 4096 ∧ NaiveBeatsStrassenUpTo h200Measured 4096 ∧
    NaiveBeatsStrassenUpTo rtx6000Sheet 4096 ∧ NaiveBeatsStrassenUpTo rtx6000Measured 4096

end Verity.Guarantees
