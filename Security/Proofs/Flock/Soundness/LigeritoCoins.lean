import Proofs.Flock.Soundness.ZerocheckPhase

/-!
# Ligerito's coins (PROTOCOL.md §13)

The pieces of the Ligerito phase that do not depend on its weight bookkeeping:

* a `K` coin (`drawK`, two `F` limbs) is uniform on `K` (`prCoin_ofLimbs`, from
  `Arith.Correct.ofLimbs_bijective`), so a step on it costs `prCoin bad` over `K`
  (`value_drawK_le`);
* a Ligerito round's quadratic `from(msg, s)` sums to `s` by construction (§13.2), so a false
  claim survives a `K` coin with probability at most `2/|K|` (`prCoin_fromMsg_le`);
* the consistency coins `α` batch per-query mismatches through `eq(α)`: a nonzero mismatch survives
  with probability at most `⌈log2 Q⌉/|F|` (`prCoin_eqList_le`).
-/

namespace FlockSoundness

open Game Model Accounting Polynomial
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## `K` coins -/

/-- A `K` coin is uniform on `K`: its two limbs are a bijection `F × F → K`. -/
theorem prCoin_ofLimbs (A : Arith F K) (hA : A.Correct) (P : K → Prop) :
    prCoin (fun w : Fin 2 → F => P (A.ofLimbs (w 0) (w 1))) = prCoin P := by
  classical
  let e : (Fin 2 → F) ≃ K :=
    (piFinTwoEquiv fun _ => F).trans (Equiv.ofBijective _ hA.ofLimbs_bijective)
  have hcard : Fintype.card (Fin 2 → F) = Fintype.card K := Fintype.card_congr e
  simp only [prCoin, avg]
  rw [hcard, ← e.sum_comp]
  rfl

/-- A step on a `K` coin: leaving the good set costs `prCoin bad` over `K`. -/
theorem value_drawK_le (A : Arith F K) (hA : A.Correct) {α : Type} (p : α → Prop)
    (k : K → Game α) (bad : K → Prop) (ε ε' : ℝ≥0∞) (hbad : prCoin bad ≤ ε)
    (hgood : ∀ c, ¬ bad c → value p (k c) ≤ ε') :
    value p ((drawK A).bind k) ≤ ε + ε' := by
  show value p (Game.coin (Fin 2 → F) fun w => k (A.ofLimbs (w 0) (w 1))) ≤ ε + ε'
  exact value_coin_le p _ _ (fun w => bad (A.ofLimbs (w 0) (w 1))) ε ε'
    (by rw [prCoin_ofLimbs A hA]; exact hbad) (fun w hw => hgood _ hw)

/-! ## A Ligerito round -/

/-- A round polynomial as a polynomial. -/
noncomputable def quadPoly (q : Quad K) : K[X] :=
  C q.1 + C q.2.1 * Polynomial.X + C q.2.2 * Polynomial.X ^ 2

omit [Fintype K] [DecidableEq K] in
theorem quadPoly_eval (q : Quad K) (x : K) : (quadPoly q).eval x = q.eval x := by
  simp only [quadPoly, eval_add, eval_mul, eval_C, eval_X, eval_pow, Quad.eval]

omit [Fintype K] [DecidableEq K] in
theorem quadPoly_natDegree_le (q : Quad K) : (quadPoly q).natDegree ≤ 2 := by
  unfold quadPoly
  refine (natDegree_add_le _ _).trans (max_le ((natDegree_add_le _ _).trans (max_le ?_ ?_)) ?_)
  · simp
  · exact (natDegree_C_mul_le _ _).trans (by simp)
  · exact (natDegree_C_mul_le _ _).trans (by simp)

omit [Fintype K] [DecidableEq K] in
/-- `from(msg, s)` sums to `s` over `{0, 1}` (§13.2). -/
theorem fromMsg_sum [CharP K 2] (msg : K × K) (s : K) :
    (fromMsg msg s).eval 0 + (fromMsg msg s).eval 1 = s := by
  simp only [fromMsg, Quad.eval, mul_zero, zero_pow two_ne_zero, add_zero, mul_one, one_pow]
  have h1 := Rounds.add_self msg.1
  have h2 := Rounds.add_self msg.2
  linear_combination h1 + h2

/-- **A Ligerito round.** If the running claim is not `T(0) + T(1)` for the true quadratic `T`,
then whatever message the prover sends, the new claim `from(msg, s)(r)` equals `T(r)` at a uniform
`r ∈ K` with probability at most `2/|K|`. -/
theorem prCoin_fromMsg_le [CharP K 2] (msg : K × K) (s : K) (T : K[X]) (hT : T.natDegree ≤ 2)
    (hs : s ≠ T.eval 0 + T.eval 1) :
    prCoin (fun r : K => (fromMsg msg s).eval r = T.eval r) ≤ 2 / Fintype.card K := by
  simp_rw [← quadPoly_eval]
  refine Rounds.sumRound_sound s _ T (quadPoly_natDegree_le _) hT ?_ hs
  rw [quadPoly_eval, quadPoly_eval, fromMsg_sum]

/-! ## The consistency coins -/

omit [Fintype F] [DecidableEq F] in
theorem eqList_ofFn {R : Type} [CommRing R] {k : ℕ} (α : Fin k → R) (j : ℕ) :
    eqList (List.ofFn α) j = eqAt α j := by
  unfold eqList eqAt
  rw [← List.prod_ofFn]
  congr 1
  apply List.ext_getElem
  · simp
  · intro i h1 h2
    simp [List.getElem_zipIdx]

/-- **The consistency coins.** A nonzero vector of per-query mismatches, weighted by `eq(α)` over
`k ≥ ⌈log2 Q⌉` coordinates, sums to zero with probability at most `k/|F|`. -/
theorem prCoin_eqList_le [CharP F 2] {k Q : ℕ} (hQ : Q ≤ 2 ^ k) (d : ℕ → F) (j₀ : ℕ)
    (hj₀ : j₀ < Q) (hd : d j₀ ≠ 0) :
    prCoin (fun α : Fin k → F => ∑ j ∈ Finset.range Q, eqList (List.ofFn α) j * d j = 0) ≤
      (k : ℝ≥0∞) / Fintype.card F := by
  classical
  let d' : Fin (2 ^ k) → F := fun u => if u.val < Q then d u.val else 0
  have hsum : ∀ α : Fin k → F, ∑ j ∈ Finset.range Q, eqList (List.ofFn α) j * d j =
      ∑ u : Fin (2 ^ k), d' u * eqAt α u.val := by
    intro α
    rw [Fin.sum_univ_eq_sum_range (fun u => (if u < Q then d u else 0) * eqAt α u) (2 ^ k)]
    rw [← Finset.sum_range_add_sum_Ico _ hQ]
    rw [Finset.sum_eq_zero (s := Finset.Ico Q (2 ^ k)) (fun u hu => by
      rw [if_neg (by simp at hu; omega), zero_mul]), add_zero]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [if_pos (Finset.mem_range.1 hj), eqList_ofFn, mul_comm]
  simp_rw [hsum]
  exact prCoin_eqSum_le d' ⟨j₀, by omega⟩ (by simp [d', hj₀, hd])

end FlockSoundness
