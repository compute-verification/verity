import Mathlib

/-!
# Red team of G: the relay term prices every side bit at `b/h`

Scratch, not built by any `lean_lib`. G's error carries `n·P·G·C(N, m)·2^(L + C + a)/2^(h·m)` and its `ε` carries
`m·b/(N·b) = m/N`. `relay_forces`: the term is below 1 only when `h·m > L + C + a`, so `ε > (L + C + a)/(h·N)`, which
is `b/h = 64` times the side bits' share `(L + C + a)/(N·b)` at `b = 32768`, `h = 512`.

Instances, all with `X ≥ L = 400 Gb/s × 61 ms = 24.4·10^9` bits (a pass over the store at its measured read rate):

* one GPU, 94 GB at 1,539 GB/s, `N = 22,949,218` blocks: for every `m` the relay term is at least 1, or `m > N` and
  `ε > 1`. G says nothing.
* the node, 789 GB at 12.92 TB/s, `N = 192,626,953` blocks: a relay term below 1 forces `m/N > 0.247`.
-/

open scoped ENNReal

namespace GRedTeam

theorem relay_forces {N m h X : ℕ} {k : ℝ≥0∞} (hk : 1 ≤ k) (hmN : m ≤ N)
    (hlt : k * ((N.choose m : ℝ≥0∞) * 2 ^ X / 2 ^ (h * m)) < 1) : X < h * m := by
  by_contra hle
  replace hle := not_lt.mp hle
  refine absurd hlt (not_lt.mpr ?_)
  have h1 : (1 : ℝ≥0∞) ≤ N.choose m := by exact_mod_cast Nat.choose_pos hmN
  have h2 : (2 : ℝ≥0∞) ^ (h * m) ≤ 2 ^ X := pow_le_pow_right₀ (by norm_num) hle
  have h3 : (1 : ℝ≥0∞) ≤ 2 ^ X / 2 ^ (h * m) := by
    rw [ENNReal.le_div_iff_mul_le (Or.inl (by positivity)) (Or.inl (by simp))]
    simpa using h2
  calc (1 : ℝ≥0∞) = 1 * (1 * 1) := by simp
    _ ≤ k * ((N.choose m : ℝ≥0∞) * (2 ^ X / 2 ^ (h * m))) := by gcongr
    _ = k * ((N.choose m : ℝ≥0∞) * 2 ^ X / 2 ^ (h * m)) := by rw [mul_div_assoc]

/-- `ε`'s share `m·b/(N·b)` exceeds `X/(h·N)` once `X < h·m`. -/
theorem share_gt {N m h X b : ℕ} (hN : 0 < N) (hb : 0 < b) (hh : 0 < h) (hX : X < h * m) :
    (X : ℝ) / (h * N) < (m * b : ℝ) / (N * b) := by
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  have hh' : (0 : ℝ) < h := by exact_mod_cast hh
  rw [mul_div_mul_right _ _ hb'.ne', div_lt_div_iff₀ (by positivity) hN']
  have : (X : ℝ) < h * m := by exact_mod_cast hX
  nlinarith

/-- **One GPU: G is vacuous.** For every `m`, either the relay term is at least 1, or `m > N`, so that `ε ≥ m/N > 1`. -/
theorem oneGpu_vacuous (m X : ℕ) (k : ℝ≥0∞) (hk : 1 ≤ k) (hX : 24400000000 ≤ X) :
    1 ≤ k * (((22949218 : ℕ).choose m : ℝ≥0∞) * 2 ^ X / 2 ^ (512 * m)) ∨ 22949218 < m := by
  by_contra hcon
  rw [not_or, not_le, not_lt] at hcon
  have := relay_forces hk hcon.2 hcon.1
  omega

/-- **The node with one 400 Gb/s link: `ε > 0.247`.** A relay term below 1 forces `m/N > 0.247`. -/
theorem node_share (m X : ℕ) (k : ℝ≥0∞) (hk : 1 ≤ k) (hX : 24400000000 ≤ X) (hmN : m ≤ 192626953)
    (hlt : k * (((192626953 : ℕ).choose m : ℝ≥0∞) * 2 ^ X / 2 ^ (512 * m)) < 1) :
    (0.247 : ℝ) < (m : ℝ) / 192626953 := by
  have hm : 47656250 < m := by have := relay_forces hk hmN hlt; omega
  have hm' : (47656251 : ℝ) ≤ m := by exact_mod_cast hm
  rw [lt_div_iff₀ (by norm_num)]
  linarith

end GRedTeam

#print axioms GRedTeam.relay_forces
#print axioms GRedTeam.share_gt
#print axioms GRedTeam.oneGpu_vacuous
#print axioms GRedTeam.node_share
