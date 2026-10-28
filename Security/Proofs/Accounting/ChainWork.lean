import Proofs.Pous.B5.ExPostFacto

/-!
# Chain work: a hash-chain receipt takes one oracle query a label

The core lemma of a costed-random-oracle receipt: the receipt is the last of
`N + 1` data-carrying labels on a line, `ℓ_i = H(i, ℓ_{i−1}) ⊕ d_i` (`Pous.Labelling.label` on `line`), and an oracle
program that outputs it after at most `N` queries, fewer than one a label, is right with probability at most
`2^m · 4(N + 1)³ / 2^w`, with `m` bits of advice computed from the oracle and the data. Rounds of no queries and the
order of the queries are free: only their number is charged. It is B5 (`PousB5.b5_ex_post_facto_bounded`) at one
challenge, no initial pebbles and depth `N`, applied to the program cut at `N` queries (`cut`).

The bound says nothing once `m ≥ w − log₂(4(N + 1)³)`: advice is paid for by a union over its values. Precomputing
the oracle before the data exist is not this lemma's to exclude; the receipt's data are fresh each round.

What turns a query into time is not here: that is a price floor on the oracle's implementation, a named assumption the
note proposes and no guarantee states.
-/

namespace Accounting.ChainWork

open Pous Pous.Labelling
open scoped ENNReal

/-- The line on `Fin n`: node `i + 1`'s one parent is `i`. -/
def line (n : ℕ) : TopoDAG n where
  par i := Finset.univ.filter fun j => j.val + 1 = i.val
  par_lt i j hj := by
    have h := (Finset.mem_filter.1 hj).2
    show j.val < i.val
    omega

theorem mem_par_line {n : ℕ} {i j : Fin n} : j ∈ (line n).par i ↔ j.val + 1 = i.val := by
  simp [line]

/-- From no pebbles, `r` rounds of greedy pebbling cover only the first `r` nodes of the line. -/
theorem reach_line_lt (n : ℕ) : ∀ (r : ℕ) (v : Fin n), v ∈ reach (line n) ∅ r → v.val < r
  | 0, v, hv => by simp [reach] at hv
  | r + 1, v, hv => by
    rw [reach, Finset.mem_union] at hv
    rcases hv with hv | hv
    · exact Nat.lt_succ_of_lt (reach_line_lt n r v hv)
    · have hsub := (Finset.mem_filter.1 hv).2
      have hlt := v.isLt
      rcases Nat.eq_zero_or_pos v.val with h0 | hpos
      · omega
      · have hj : (⟨v.val - 1, by omega⟩ : Fin n) ∈ (line n).par v :=
          mem_par_line.2 (show v.val - 1 + 1 = v.val by omega)
        have h : v.val - 1 < r := reach_line_lt n r _ (hsub hj)
        omega

/-- The line on `N + 1` nodes is `(0, N, 1)`-hard at its last node: from no pebbles it is reached only in round
`N + 1`. -/
theorem line_hard (N : ℕ) : PebblingHard (line (N + 1)) {Fin.last N} 0 N 1 := by
  intro P₀ hP₀
  rw [Finset.card_eq_zero.1 (Nat.le_zero.1 hP₀), Nat.lt_one_iff, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
  intro c hc hreach
  rw [Finset.mem_singleton.1 hc] at hreach
  have h := reach_line_lt (N + 1) N _ hreach
  simp at h

section Cut

variable {M : IdealModel} {α : Type}

/-- `p` on a budget of `b` queries: a round of no queries is skipped, a round past the budget stops with `a₀`, and
every other round is `p`'s. -/
def cut (a₀ : α) : ℕ → OracleAlgorithm M α → OracleAlgorithm M α
  | _, .ret a => .ret a
  | b, .round [] k => cut a₀ b (k [])
  | b, .round (q :: qs) k =>
    if qs.length + 1 ≤ b then .round (q :: qs) fun as => cut a₀ (b - (qs.length + 1)) (k as) else .ret a₀

theorem work_cut (a₀ : α) (f : M.Qry → M.Ans) : ∀ (p : OracleAlgorithm M α) (b : ℕ), (cut a₀ b p).work f ≤ b
  | .ret a, b => by simp [cut, OracleAlgorithm.work]
  | .round [] k, b => by
    simp only [cut]
    exact work_cut a₀ f (k []) b
  | .round (q :: qs) k, b => by
    simp only [cut]
    split_ifs with h
    · have ih := work_cut a₀ f (k ((q :: qs).map f)) (b - (qs.length + 1))
      simp only [OracleAlgorithm.work, List.length_cons]
      omega
    · simp [OracleAlgorithm.work]

theorem depth_cut (a₀ : α) (f : M.Qry → M.Ans) : ∀ (p : OracleAlgorithm M α) (b : ℕ), (cut a₀ b p).depth f ≤ b
  | .ret a, b => by simp [cut, OracleAlgorithm.depth]
  | .round [] k, b => by
    simp only [cut]
    exact depth_cut a₀ f (k []) b
  | .round (q :: qs) k, b => by
    simp only [cut]
    split_ifs with h
    · have ih := depth_cut a₀ f (k ((q :: qs).map f)) (b - (qs.length + 1))
      simp only [OracleAlgorithm.depth]
      omega
    · simp [OracleAlgorithm.depth]

/-- The cut program is `(b, b)`-bounded against every primitive. -/
theorem bounded_cut (a₀ : α) (b : ℕ) (p : OracleAlgorithm M α) : (cut a₀ b p).Bounded b b := by
  intro ω
  exact ⟨depth_cut a₀ _ p b, work_cut a₀ _ p b⟩

/-- Within its budget the cut program outputs what `p` does. -/
theorem run_cut (a₀ : α) (f : M.Qry → M.Ans) :
    ∀ (p : OracleAlgorithm M α) (b : ℕ), p.work f ≤ b → (cut a₀ b p).run f = p.run f
  | .ret a, b, _ => by simp [cut, OracleAlgorithm.run]
  | .round [] k, b, h => by
    simp only [cut, OracleAlgorithm.run, List.map_nil]
    exact run_cut a₀ f (k []) b (by simpa [OracleAlgorithm.work] using h)
  | .round (q :: qs) k, b, h => by
    have hw : qs.length + 1 + (k ((q :: qs).map f)).work f ≤ b := by simpa [OracleAlgorithm.work] using h
    have hb : qs.length + 1 ≤ b := by omega
    simp only [cut]
    rw [if_pos hb]
    simp only [OracleAlgorithm.run]
    exact run_cut a₀ f _ _ (by omega)

end Cut

/-- **Chain work.** On the line of `N + 1` data-carrying labels, an oracle program with `m` bits of advice that outputs
the last label after at most `N` queries is right with probability at most `2^m · 4(N + 1)³ / 2^w`. -/
theorem chain_work (N w m : ℕ) (d : Fin (N + 1) → Bits w) (A₁ : (LabelQry (N + 1) w → Bits w) → Bits m)
    (P : Bits m → OracleAlgorithm (IdealModel.randomOracle (LabelQry (N + 1) w) (Bits w)) (Bits w)) :
    Pous.pr (fun H : LabelQry (N + 1) w → Bits w =>
        (P (A₁ H)).work H ≤ N ∧ (P (A₁ H)).run H = label (line (N + 1)) d H (Fin.last N)) ≤
      2 ^ m * ((4 * (N + 1) ^ 3 : ℕ) : ℝ≥0∞) / 2 ^ w := by
  classical
  have hB5 := PousB5.b5_ex_post_facto_bounded (N + 1) w m 0 N N 1 (line (N + 1)) {Fin.last N} d (line_hard N) A₁
    (fun σ _ => cut (fun _ => false) N (P σ)) (fun σ _ => bounded_cut _ N (P σ))
  refine le_trans (Pous.pr_mono ?_) (hB5.trans_eq ?_)
  · intro H hH
    exact Finset.card_pos.2 ⟨Fin.last N, Finset.mem_filter.2 ⟨Finset.mem_singleton_self _,
      (run_cut _ H (P (A₁ H)) N hH.1).trans hH.2⟩⟩
  · simp only [Finset.card_singleton, one_mul, zero_add, pow_one]
    rw [mul_comm ((N + 1 : ℕ) : ℝ≥0∞), mul_assoc, ← mul_div_assoc, ← mul_div_assoc, ← Nat.cast_mul,
      show (N + 1) * (N + 1 + (N + 1)) ^ 2 = 4 * (N + 1) ^ 3 by ring]

end Accounting.ChainWork
