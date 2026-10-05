import FlockSoundness.Soundness
import FlockLevel3
import FlockSoundness.Protocol.Instance

/-!
# The executable's arithmetic satisfies `Arith.Correct`

`execArith` is the arithmetic the executable verifier computes with (lane `flock-verifier`,
package `level3`): `F = GF(2^128)` in GHASH form, `K = GF(2^256)`, the Lagrange weights of the
univariate-skip nodes, the pinned zerocheck coordinates, the packing basis `x^v`, the embedding
`F ⊂ K` and `u`, the Reed–Solomon domain `ω_q` and novel basis `Ŵ_i`, and `lo`. Every field of
`Arith.Correct` is a theorem of `FlockLevel3` about those functions (`execArith_correct`), so
the soundness theorems hold for the executable's arithmetic with no arithmetic hypothesis
(`table_sound_exec`).
-/

namespace FlockSoundness

open Model Accounting

/-- **One table's session is sound, for the executable's arithmetic.** `table_sound` with
`execArith_correct`, so it has no named assumption. -/
theorem table_sound_exec (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (σ : Game.Strategy (table execArith S sch hm ptLocal mPts)) :
    Game.prob (fun out => out.1 = true ∧ ¬ Committed execArith S sch out.2)
        (table execArith S sch hm ptLocal mPts) σ ≤
      ENNReal.ofReal (tableError sch (S.shape mPts)) :=
  table_sound execArith execArith_correct S sch hsch hm ptLocal mPts hlay σ

/-- `table_sound_exec` for `m ∈ [22, 33]` under the `fast100` schedule: at most `2^-205`. -/
theorem table_sound_exec_fast100 (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm1 : 22 ≤ S.m) (hm2 : S.m ≤ 33) (ptLocal mPts : ℕ)
    (hshape : (S.shape mPts).InRange) (hlay : S.LinkLayout ptLocal mPts)
    (σ : Game.Strategy (table execArith S sch (by omega) ptLocal mPts)) :
    Game.prob (fun out => out.1 = true ∧ ¬ Committed execArith S sch out.2)
        (table execArith S sch (by omega) ptLocal mPts) σ ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  table_sound_fast100 execArith execArith_correct S sch hsch hm1 hm2 ptLocal mPts hshape hlay σ

/-- `table_sound_exec` for `m ∈ {34, 35}`: at most `2^-205`. -/
theorem table_sound_exec_fast100_34_35 (S : Statement)
    (sch : Schedule) (hsch : fast100 S.m = some sch) (hm1 : 34 ≤ S.m) (hm2 : S.m ≤ 35)
    (ptLocal mPts : ℕ) (hshape : (S.shape mPts).InRange) (hlay : S.LinkLayout ptLocal mPts)
    (σ : Game.Strategy (table execArith S sch (by omega) ptLocal mPts)) :
    Game.prob (fun out => out.1 = true ∧ ¬ Committed execArith S sch out.2)
        (table execArith S sch (by omega) ptLocal mPts) σ ≤
      ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  table_sound_fast100_34_35 execArith execArith_correct S sch hsch hm1 hm2 ptLocal mPts hshape
    hlay σ

end FlockSoundness
