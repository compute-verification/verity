import Proofs.Accounting.Headline.Sm120Measured

/-!
# RT-LOGIC-14: sm_120's measured headline rests on (I3) at 9/8 (red-team witness, not for landing)

Builds on #2482's head (`cursor/lean-headline-sm120-ea24` @ d9dc188d78), not on main.

`Sm120MeasuredHeadline` takes (I3) `ScalarRoute` at `q = q_T = 9/8`, just above the point (`q ≈ 1.120`) where the
vector route stops undercutting IMMA16832's word on JI3s_0. What A6 proves on any machine is `½`
(`SheetScalarRouteHalf`). `vec_bound`: ♭ with its IMMA16832 instruction (128 words) replaced by two FFMA, the vector
route at `½` lane-instruction a word, has every VF3r-16 row at most `0.92·B_♭`, so no certificate proves the reading
below 8% at `½`. `vec_route`: the vector route heads its 128 words at `q = ½` and not at `9/8`.
-/

namespace Accounting.SecurityProofs.TS.Headline.RtLogicM

open Accounting.TS Accounting.TS.Headline Accounting.SecurityProofs.TS.Headline

/-- ♭ on VF3r-16 with its 128 words on the vector route at `½` (two FFMA a unit), its other word production, fold and
skippable work as ♭'s. -/
def vecAll : Sparse := [(13, 2, 1), (61, 25641, 1000000)] ++ Sm120.flatFold ++ Sm120Measured.flatSkip

theorem vec_rows : ∀ i < 109, dot (vfCoef i) 122 vecAll ≤ 92 / 100 * BqM := by decide +kernel

/-- The model's floor for that program on VF3r-16 is at most `0.92·B_♭`. -/
theorem vec_bound :
    Sm120Measured.model.G.bound (Sparse.counts vecAll) ≤ 92 / 100 * Sm120Measured.model.bnd Sm120Measured.flat := by
  rw [sm120m_bnd]
  unfold Machine.bound
  refine ciSup_le fun r => ?_
  rw [load_counts _ vfRow sm120m_hG]
  have h' : ((dot (vfRow r) 122 vecAll : ℚ) : ℝ) ≤ ((92 / 100 * BqM : ℚ) : ℝ) :=
    Rat.cast_le.mpr (vec_rows r.val r.isLt)
  push_cast at h'
  exact h'

/-- Two FFMA (64 lane instructions) meet SCALAR-ROUTE for 128 vector-headed words at `q = ½` and not at `9/8`. -/
theorem vec_route : (1 / 2 : ℚ) * 128 ≤ 32 * 2 ∧ ¬ ((9 / 8 : ℚ) * 128 ≤ 32 * 2) := by norm_num

end Accounting.SecurityProofs.TS.Headline.RtLogicM

#print axioms Accounting.SecurityProofs.TS.Headline.RtLogicM.vec_bound
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogicM.vec_route
