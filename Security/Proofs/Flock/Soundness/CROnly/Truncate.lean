import Proofs.Flock.Soundness.CROnly.Defs

/-!
# Expected time from strict time: a finder stopped at `q`

A finder `(g, s, out, cost)` that stops adaptively has no worst-case budget, so `SHA512CRStrict` doesn't apply to it. Its
stopped form (`truncOut`, `truncCost`) has one: it runs as the finder does and stops at the `q`-th evaluation. The two
output the same pair wherever the finder makes at most `q` evaluations, and by Markov's inequality the finder makes more
with probability at most `T/q`, for `T` its expected cost. So `SHA512CRStrict` of the stopped finder gives
$$p \le \frac{T}{q} + \frac{q^2}{2^{513}}$$
(`expected_of_strict`). With `ε = T/q` this is `ε + T²/(ε²·2^513)`. At its best budget, `q³ = T·2^512`, it is
`(3/2)·q²/2^512 = (3/2)·T^{2/3}/2^{512/3} ≈ T^{2/3}/2^{170.08}` (`trunc_at_cube`).

The stopped finder is built from the finder, so from the prover when the finder is: the hypothesis stays one about the
finders of the prover bounded, as both forms are.
-/

namespace FlockSoundness.CROnly

open Game

/-- **Expected-time collision resistance from strict-time, for one finder**: if the finder stopped at `q` evaluations
satisfies `SHA512CRStrict`, the finder finds a collision with probability at most `T/q + q²/2^513`, for `T` its expected
number of evaluations. -/
theorem expected_of_strict {Dg β : Type} (H : List UInt8 → Dg) (g : Game β) (s : Strategy g)
    (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (hc : ∀ b, 0 ≤ cost b) {q : ℝ} (hq : 0 < q)
    (hS : Assumptions.SHA512CRStrict H g s (truncOut out cost q) (truncCost cost q) q) :
    (prob (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s).toReal ≤
      expect cost g s / q + q ^ 2 / 2 ^ (513 : ℝ) := by
  classical
  have h1 := hS fun b => min_le_right _ _
  rw [prob_toReal] at h1 ⊢
  have hpt : ∀ b, ind (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) b ≤
      ind (fun b => ∃ x y, truncOut out cost q b = some (x, y) ∧ x ≠ y ∧ H x = H y) b + q⁻¹ * cost b := by
    intro b
    have hq' : 0 ≤ q⁻¹ * cost b := mul_nonneg (inv_nonneg.2 hq.le) (hc b)
    by_cases hb : cost b ≤ q
    · have : truncOut out cost q b = out b := by simp only [truncOut, hb, ↓reduceIte]
      have e : ind (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) b =
          ind (fun b => ∃ x y, truncOut out cost q b = some (x, y) ∧ x ≠ y ∧ H x = H y) b := by
        simp only [ind, this]
      rw [e]
      linarith
    · have h1 : 1 ≤ q⁻¹ * cost b := by
        rw [← div_eq_inv_mul, le_div_iff₀ hq]
        linarith [not_le.1 hb]
      linarith [ind_le_one (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) b,
        ind_nonneg (fun b => ∃ x y, truncOut out cost q b = some (x, y) ∧ x ≠ y ∧ H x = H y) b]
  calc expect (ind fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s
      ≤ expect (fun b => ind (fun b => ∃ x y, truncOut out cost q b = some (x, y) ∧ x ≠ y ∧ H x = H y) b +
          q⁻¹ * cost b) g s := expect_mono hpt g s
    _ = expect (ind fun b => ∃ x y, truncOut out cost q b = some (x, y) ∧ x ≠ y ∧ H x = H y) g s +
          expect cost g s / q := by
        rw [expect_add, expect_const_mul, div_eq_inv_mul]
    _ ≤ q ^ 2 / 2 ^ (513 : ℝ) + expect cost g s / q := add_le_add h1 le_rfl
    _ = _ := add_comm _ _

/-- **The stopped finder at its best budget**: at `q³ = T·2^512`, `T/q + q²/2^513 = (3/2)·q²/2^512`, which is
`(3/2)·T^{2/3}/2^{512/3}`. -/
theorem trunc_at_cube {T q : ℝ} (hq : 0 < q) (hcube : q ^ 3 = T * 2 ^ (512 : ℝ)) :
    T / q + q ^ 2 / 2 ^ (513 : ℝ) = 3 / 2 * (q ^ 2 / 2 ^ (512 : ℝ)) := by
  have h513 : (2 : ℝ) ^ (513 : ℝ) = 2 * 2 ^ (512 : ℝ) := by
    rw [show (513 : ℝ) = 1 + 512 by norm_num, Real.rpow_add (by norm_num : (0 : ℝ) < 2), Real.rpow_one]
  have h2 : (0 : ℝ) < 2 ^ (512 : ℝ) := by positivity
  have hT : T = q ^ 3 / 2 ^ (512 : ℝ) := by rw [hcube]; field_simp
  rw [hT, h513]
  field_simp
  ring

end FlockSoundness.CROnly
