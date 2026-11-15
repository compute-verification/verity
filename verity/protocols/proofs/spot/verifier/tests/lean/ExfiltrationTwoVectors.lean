import Proofs.Flock.SamplingGeneric.TwoTable
import Mathlib.Algebra.BigOperators.Fin

/-!
# `card_influenceTwo_le_uniform`'s count on two-stage designs, evaluated by Lean

Vectors for `verity.core.protocols.sampled_proofs.consumers.exfiltration_bound` on a design with a replay stage
(`verifier/tests/fixtures/exfiltration_two_lean.json`). A two-stage vector `(R, N, k, num, den, e)` is a design of `R` replay blocks
of `N` proof blocks each, in one stratum, `k` drawn in each sampled replay block, the first stage at rate `num/den` (Lean's
`bernoulli:NUM/DEN` rule, or any first stage missing each replay block with probability at most `(den − num)/den`,
independently or less often), at `δ = 2^-e`. Its count `K` is the last entry of row `R` of `TwoTable.table` above
`(den·C(N, k))^R / 2^e`, and the kernel checks every later entry is at most that (`gridTwo_meets`), so every set admissible
at `δ` has at most `K` proof blocks (`TwoTable.card_le_of_table`). With `b` committed bits per proof block, `H = b·K`, and
the served outputs at any inputs number at most `2^H · Σ_{j≤K} C(R·N, j)` (`gridTwo_influence`,
`card_influenceTwo_le_uniform`). A replay block's factor at `m` incorrect proof blocks is `accept[m] / accept_den`
(`TwoTable.factor_eq`).

Regenerate with `verifier/tests/lean/generate.sh`.
-/

namespace Claims.Sampling.TwoStageExfiltrationVectors

open Claims.Sampling.TwoTable

/-- `C(m, k)` as `m.descFactorial k / k!`, which the kernel evaluates in `k` steps (`Nat.choose` unfolds Pascal's rule). -/
def chooseD (m k : ℕ) : ℕ := m.descFactorial k / k.factorial

theorem chooseD_eq (m k : ℕ) : chooseD m k = m.choose k := (Nat.choose_eq_descFactorial_div_factorial m k).symm

/-- `∑ j ∈ range (K + 1), n.choose j` and `n.choose K`, each term from the last: `C(n, j + 1) = C(n, j) · (n − j) / (j + 1)`. -/
def sumChoose (n : ℕ) : ℕ → ℕ × ℕ
  | 0 => (1, 1)
  | K + 1 => let (s, c) := sumChoose n K; (s + c * (n - K) / (K + 1), c * (n - K) / (K + 1))

theorem sumChoose_eq (n K : ℕ) : sumChoose n K = (∑ j ∈ Finset.range (K + 1), n.choose j, n.choose K) := by
  induction K with
  | zero => simp [sumChoose]
  | succ K ih =>
    have hc : n.choose K * (n - K) / (K + 1) = n.choose (K + 1) := by
      rw [← Nat.choose_succ_right_eq]; exact Nat.mul_div_cancel _ (Nat.succ_pos K)
    simp only [sumChoose, ih, hc, Finset.sum_range_succ _ (K + 1)]

/-- `TwoTable.factor` with `chooseD`, which the kernel evaluates in `k` steps. -/
def factorD (N k num den c : ℕ) : ℕ := (den - num) * chooseD N k + num * chooseD (N - c) k

theorem factorD_eq (N k num den : ℕ) : factorD N k num den = factor N k num den := by
  funext c
  simp only [factorD, factor, chooseD_eq]

/-- The vector's count: the entries of row `R` above `(den·C(N, k))^R / 2^e`, less one (they fall with the index). -/
def countTwo (R N k num den e : ℕ) : ℕ :=
  ((table (factorD N k num den) N R).takeWhile fun x => (den * chooseD N k) ^ R < x * 2 ^ e).length - 1

/-- What the kernel checks of a two-stage vector: `0 < den`, `num ≤ den`, `k ≤ N`, and every entry of row `R` past `K`
is at most `(den·C(N, k))^R / 2^e`. -/
def meetsTwo (R N k num den e K : ℕ) : Bool :=
  decide (0 < den ∧ num ≤ den ∧ k ≤ N) &&
    ((table (factorD N k num den) N R).drop (K + 1)).all fun x => decide (x * 2 ^ e ≤ (den * chooseD N k) ^ R)

theorem table_of_meetsTwo {R N k num den e K : ℕ} (h : meetsTwo R N k num den e K = true) :
    0 < den ∧ num ≤ den ∧ k ≤ N ∧
      ∀ m, K + 1 ≤ m → (table (factor N k num den) N R).getD m 0 * 2 ^ e ≤ (den * N.choose k) ^ R := by
  simp only [meetsTwo, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, factorD_eq, chooseD_eq] at h
  obtain ⟨⟨hd, hn, hk⟩, hall⟩ := h
  exact ⟨hd, hn, hk, getD_of_drop (P := fun x => x * 2 ^ e ≤ (den * N.choose k) ^ R) (by simp) _ _ hall⟩

/-- `(R, N, k, num, den, e)`: every replay block replayed (`num = den`), the first stage at a rate, and every proof block
admissible (the last). -/
def gridTwo : List (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  [(16, 16, 1, 1, 1, 4), (8, 32, 4, 1, 1, 10), (32, 8, 1, 1, 1, 20), (4, 32, 2, 1, 1, 20), (8, 8, 1, 1, 2, 4),
    (64, 4, 1, 1, 2, 4), (32, 8, 1, 1, 2, 20), (32, 16, 2, 1, 2, 20), (32, 8, 2, 3, 4, 20), (24, 12, 3, 2, 3, 30),
    (16, 16, 2, 1, 4, 10)]

/-- Every two-stage vector's count meets `meetsTwo`, evaluated by the kernel. -/
theorem gridTwo_meets :
    gridTwo.all (fun (R, N, k, num, den, e) => meetsTwo R N k num den e (countTwo R N k num den e)) = true := by
  decide +kernel

/-- **Each two-stage vector's bound**: on a statement in scope whose design has the vector's shape and whose proof blocks
each commit at most `b` bits, the served outputs at any input values number at most `2^{b·K} · Σ_{j≤K} C(R·N, j)`. -/
theorem gridTwo_influence {R N k num den e : ℕ} (hv : (R, N, k, num, den, e) ∈ gridTwo)
    (H : Msg → Msg) (Cm : CommitmentScheme Bool) {s : AuditStatement} {Pt : Partition} {d : Design.Two}
    (hsc : InScopeTwo Pt d s) (hR : d.part.blocks.length = R)
    (hrep : ∀ T ⊆ Finset.range R, d.replay.Pa d.part T ≤ (((den - num : ℕ) : ℝ) / den) ^ T.card)
    (hs : Pt.strata.length = 1) (hk : d.k.getD 0 0 = k) (hN : ∀ ρ < R, (cell Pt d.part ρ 0).card = N)
    (hcov : ∀ v < Pt.blocks.length, ∃ ρ < R, v ∈ cell Pt d.part ρ 0) {b : ℕ}
    (hb : ∀ v < Pt.blocks.length, committed s.prog (Pt.blocks.getD v []) ≤ b) (a : ℕ → Option Bool) :
    (influenceTwo H Cm s Pt d (((2 : ℝ) ^ e)⁻¹) a).card ≤
      2 ^ (b * countTwo R N k num den e) *
        ∑ j ∈ Finset.range (countTwo R N k num den e + 1), (R * N).choose j := by
  have h : meetsTwo R N k num den e (countTwo R N k num den e) = true := List.all_eq_true.1 gridTwo_meets _ hv
  obtain ⟨hd, hn, hkN, htab⟩ := table_of_meetsTwo h
  exact influence_le_of_table H Cm hsc hd hn hR hrep hs hk hkN hN hcov hb htab a

open Lean in
run_meta do
  for d in [``gridTwo_meets, ``gridTwo_influence, ``sumChoose_eq] do
    for a in ← collectAxioms d do
      unless a ∈ [``propext, ``Classical.choice, ``Quot.sound] do
        throwError "{d} uses {a}, beyond propext, Classical.choice and Quot.sound"

/-- A two-stage vector's row at `b` bits per proof block: its shape, `K`, `H = b·K`, the replay block's factors
`accept[m]` over `accept_den`, and `admissible = Σ_{j≤K} C(R·N, j)` (`sumChoose_eq`). -/
def rowTwo (b : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ → String
  | (R, N, k, num, den, e) =>
    let K := countTwo R N k num den e
    "{" ++ s!"\"replay_blocks\": {R}, \"n_rho\": {N}, \"k\": {k}, \"rate\": [{num}, {den}], \"delta_log2\": -{e}, " ++
      s!"\"bits\": {b}, \"K\": {K}, \"H\": {b * K}, " ++
      s!"\"accept\": {(List.range (N + 1)).map (factorD N k num den)}, \"accept_den\": {den * chooseD N k}, " ++
      s!"\"admissible\": {(sumChoose (R * N) K).1}" ++ "}"

end Claims.Sampling.TwoStageExfiltrationVectors

open Claims.Sampling.TwoStageExfiltrationVectors in
def main : IO Unit :=
  IO.println ("{\"source\": \"verifier/tests/lean/ExfiltrationTwoVectors.lean via verifier/tests/lean/generate.sh\", " ++
    "\"statement\": \"Claims.Sampling.card_influenceTwo_le_uniform\", \"vectors\": [\n" ++
    ",\n".intercalate (gridTwo.flatMap fun v => [rowTwo 1 v, rowTwo 16 v]) ++ "\n]}")
