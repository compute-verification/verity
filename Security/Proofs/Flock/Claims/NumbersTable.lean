import Proofs.Flock.Soundness.Discharge.FailClosed.Numeric
import Proofs.Flock.Soundness.Discharge.ZkSession.Numbers

/-!
# Flock's table error at the deployed parameters: `ε_table ≤ 2^{−205}`, with and without `--zk`

`κ`'s first term is `4·ε_table`, the table's soundness error (`Claims.Flock.κ`). At the deployed parameters it is at
most `2^{−205}` per table, at every statement in scope:

* without `--zk`, at every `fast100` schedule (`22 ≤ m ≤ 35`) and every shape in `Stmt.InRangeWide`, the shapes of every
  table flock-verify plans (`εtable_plain`, `tableError_le_205_wide`); the rational bound's worst case is `2^{−205.011}`,
  at `m = 34`;
* with `--zk`, at C-Flock ZK's parameters (`TabZK.M1`: the padded level 0 of `m1Schedule m`, `25 ≤ m ≤ 35`, and the
  inner proof with 168 queries, `N ≤ 2^13` and `δ ≥ 7/16`), `padLinkError + (padRepError + εInner)² ≤ 2^{−205}`
  (`εtable_zk`). `TabZK.bound_le_m1` gave `2^{−203}`, from `(a + b)² ≤ 2a² + 2b²`; here the square is kept, with
  `εInner ≤ (2^13 + 3)/2^128 + (9/16)^168 ≈ 2^{−115}` (`innerUp`, `εInner_le_innerUp`) next to
  `padRepError ≲ 2^{−102.5}`, and one kernel check per `m` (`m1_check_zk`). Its worst case is `2^{−205.008}`, at
  `m = 34`.
-/

namespace Claims.Flock.Numbers

open FlockSoundness FlockSoundness.Accounting FlockSoundness.Discharge.ZkSession
open scoped ENNReal NNReal

/-! ## Without `--zk` -/

/-- **`ε_table ≤ 2^{−205}` without `--zk`**: at every `fast100` schedule and every shape of a planned table. -/
theorem εtable_plain {m : ℕ} {s : Schedule} (hs : fast100 m = some s) (st : Stmt) (hst : st.InRangeWide) :
    tableError s st ≤ (2 : ℝ) ^ (-(205 : ℝ)) :=
  tableError_le_205_wide hs st hst

/-! ## With `--zk` -/

/-- The inner proof's error at C-Flock ZK's parameters, rationally: `(2^13 + 3)/2^128 + (9/16)^168`. -/
def innerUp : ℚ := (2 ^ 13 + 3) / 2 ^ 128 + (9 / 16) ^ 168

theorem innerUp_nonneg : (0 : ℝ) ≤ (innerUp : ℝ) := by
  unfold innerUp
  push_cast
  positivity

/-- `εInner ≤ innerUp` over `F` of order `2^128`, with 168 queries, `N ≤ 2^13` and `δ ≥ 7/16`. -/
theorem εInner_le_innerUp {F : Type} [Fintype F] {N : ℕ} (hF : Fintype.card F = 2 ^ 128) (hN : N ≤ 2 ^ 13)
    {δ : ℝ≥0} (hδ : 7 / 16 ≤ δ) : εInner N F 168 δ ≤ ENNReal.ofReal (innerUp : ℝ) := by
  unfold εInner
  rw [hF]
  have h1 : (N + 3 : ℝ≥0∞) / ((2 ^ 128 : ℕ) : ℝ≥0∞) = ENNReal.ofReal ((N + 3 : ℝ) / 2 ^ 128) := by
    rw [ENNReal.ofReal_div_of_pos (by positivity)]
    congr 1
    · rw [ENNReal.ofReal_add (by positivity) (by norm_num), ENNReal.ofReal_natCast]
      norm_num
    · rw [show ((2 : ℝ) ^ 128) = ((2 ^ 128 : ℕ) : ℝ) by norm_num, ENNReal.ofReal_natCast]
  have h2 : ((1 - δ : ℝ≥0) : ℝ≥0∞) ^ 168 = ENNReal.ofReal (((1 - δ : ℝ≥0) : ℝ) ^ 168) := by
    rw [ENNReal.ofReal_pow (NNReal.coe_nonneg _), ENNReal.ofReal_coe_nnreal]
  rw [h1, h2, ← ENNReal.ofReal_add (by positivity) (by positivity)]
  refine ENNReal.ofReal_le_ofReal ?_
  have hδ' : ((1 - δ : ℝ≥0) : ℝ) ≤ 9 / 16 := by
    have : (1 - δ : ℝ≥0) ≤ 9 / 16 := by
      rw [tsub_le_iff_right]
      calc (1 : ℝ≥0) = 9 / 16 + 7 / 16 := by norm_num
        _ ≤ 9 / 16 + δ := by gcongr
    exact_mod_cast this
  have hN' : (N : ℝ) ≤ 2 ^ 13 := by exact_mod_cast hN
  calc (N + 3 : ℝ) / 2 ^ 128 + ((1 - δ : ℝ≥0) : ℝ) ^ 168 ≤ (2 ^ 13 + 3) / 2 ^ 128 + (9 / 16) ^ 168 := by
        gcongr
    _ = (innerUp : ℝ) := by unfold innerUp; push_cast; ring

/-- The rational check with the inner proof's error inside the square, for `25 ≤ m ≤ 33`. -/
theorem m1_check_zk_le_33 : ∀ m, 25 ≤ m → m ≤ 33 →
    padLinkUp (m1Schedule m) (m1Pad m) + (padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) + innerUp) ^ 2 ≤
      (1 / 2) ^ 205 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- The rational check with the inner proof's error inside the square, for `m = 34, 35`. -/
theorem m1_check_zk_34_35 : ∀ m, 34 ≤ m → m ≤ 35 →
    padLinkUp (m1Schedule m) (m1Pad m) + (padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) + innerUp) ^ 2 ≤
      (1 / 2) ^ 205 := by
  intro m h1 h2
  interval_cases m <;> decide +kernel

/-- The rational check with the inner proof's error inside the square, at `25 ≤ m ≤ 35`. -/
theorem m1_check_zk (m : ℕ) (h1 : 25 ≤ m) (h2 : m ≤ 35) :
    padLinkUp (m1Schedule m) (m1Pad m) + (padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) + innerUp) ^ 2 ≤
      (1 / 2) ^ 205 := by
  rcases le_or_gt m 33 with h | h
  · exact m1_check_zk_le_33 m h1 h
  · exact m1_check_zk_34_35 m (by omega) h2

/-- `padRepError ≤ padRepUp` at any schedule of two levels or more. -/
theorem padRepError_le_of (s : Schedule) (st : Stmt) (hst : st.InRange) (t e : ℕ) (sbar : ℚ)
    (h2 : 2 ≤ s.levels.length) (hp : PadNumOK (s.levels.headD default) t sbar) (hok : ∀ l ∈ s.levels, l.Ok) :
    padRepError s st t e ≤ (padRepUp s t e sbar : ℝ) := by
  obtain ⟨m, k0, levels, ext⟩ := s
  rcases levels with _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩
  · simp at h2
  · simp at h2
  · exact padRepError_le m k0 l₀ l₁ rest ext t e sbar hp (fun l hl => hok l (List.mem_cons_of_mem _ hl)) st hst

/-- **`ε_table ≤ 2^{−205}` with `--zk`**: a `--zk` table's bound at C-Flock ZK's parameters. -/
theorem εtable_zk {F : Type} [Field F] [Fintype F] [DecidableEq F] {mPts : ℕ} (hF : Fintype.card F = 2 ^ 128)
    (T : TabZK F mPts) (hT : T.M1) : T.bound ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) := by
  obtain ⟨m, h1, h2, hsch, ht, he, hst, hN, hq, hδ⟩ := hT
  have hf := m1Facts_spec (m1Facts_all m h1 h2)
  have hl0 : 0 ≤ padLinkError T.sch (T.S.shape mPts) T.t := by unfold padLinkError; positivity
  have hl : padLinkError T.sch (T.S.shape mPts) T.t ≤ (padLinkUp (m1Schedule m) (m1Pad m) : ℝ) := by
    rw [hsch, ht]
    exact padLinkError_le _ _ _ hst
  have hr0 : 0 ≤ padRepError T.sch (T.S.shape mPts) T.t T.e := by
    rw [hsch, ht, he]
    exact m1_rep_nonneg m h1 h2 _
  have hr : padRepError T.sch (T.S.shape mPts) T.t T.e ≤
      (padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) : ℝ) := by
    rw [hsch, ht, he]
    exact padRepError_le_of _ _ hst _ _ _ hf.1 hf.2.2.1 hf.2.2.2.2
  have hb : εInner T.N F T.q T.δ ≤ ENNReal.ofReal (innerUp : ℝ) := by
    rw [hq]
    exact εInner_le_innerUp hF hN hδ
  have hc := Rat.cast_le (K := ℝ) |>.2 (m1_check_zk m h1 h2)
  push_cast at hc
  have hi := innerUp_nonneg
  unfold TabZK.bound
  calc ENNReal.ofReal (padLinkError T.sch (T.S.shape mPts) T.t) +
        (ENNReal.ofReal (padRepError T.sch (T.S.shape mPts) T.t T.e) + εInner T.N F T.q T.δ) ^ 2
      ≤ ENNReal.ofReal (padLinkError T.sch (T.S.shape mPts) T.t) +
          (ENNReal.ofReal (padRepError T.sch (T.S.shape mPts) T.t T.e) + ENNReal.ofReal (innerUp : ℝ)) ^ 2 := by
        gcongr
    _ = ENNReal.ofReal (padLinkError T.sch (T.S.shape mPts) T.t +
          (padRepError T.sch (T.S.shape mPts) T.t T.e + (innerUp : ℝ)) ^ 2) := by
        rw [← ENNReal.ofReal_add hr0 hi, ← ENNReal.ofReal_pow (by positivity),
          ← ENNReal.ofReal_add hl0 (by positivity)]
    _ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) := by
        refine ENNReal.ofReal_le_ofReal (le_trans ?_ half_pow_205_le)
        have hsq : (padRepError T.sch (T.S.shape mPts) T.t T.e + (innerUp : ℝ)) ^ 2 ≤
            ((padRepUp (m1Schedule m) (m1Pad m) 2 (m1SqrtUp m) : ℝ) + innerUp) ^ 2 := by
          gcongr
        linarith

end Claims.Flock.Numbers
