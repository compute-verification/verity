import FlockSoundness.RepSound

/-!
# Soundness of Flock's interactive protocol, as PROTOCOL.md specifies it

**`table_sound`.** For every prover strategy, the probability that the verifier accepts a table
(both reps, on one committed root, with live coins) while no codeword within the level-0 radius
of the committed table packs a witness satisfying the statement, is at most
`tableError (fast100 m) shape`.

**`table_sound_fast100`.** With the numbers: at most `2^-195.5` for `22 ≤ m ≤ 33` and `2^-195.4`
for `m = 34, 35`, for every statement shape in `Stmt.InRange`.

This is the interactive-oracle layer, a16z's security stage 1a (`DESIGN.md` §7): the
committed tables are oracles. The Merkle layer (`Merkle`, stages 1b and 1c with live coins) adds
SHA-256's collision term for the compiled protocol.

**Status.** `table_sound` is proved from `rep_sound` (one rep from a doomed start, `RepSound`,
itself proved from four phase lemmas) and `link_sound` (the link points), with the phase lemmas
and `link_sound` still `sorry`; see `ASSUMPTIONS.md` for the plan.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **The link points** (§16.2): if the committed values do not satisfy the statement, the two
link points leave some rep start undoomed with probability at most `linkError`.
*Status: to prove* (Schwartz–Zippel on each region's multilinear difference, twice). -/
theorem link_sound (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (ptLocal mPts : ℕ) (C : Oracle F)
    (hC : ¬ Committed A S sch C) :
    prCoin (fun pts : Fin 2 → Fin mPts → F =>
        ¬ RepDoomed A S sch C (extraClaims S ptLocal mPts pts)) ≤
      ENNReal.ofReal (linkError sch (S.shape mPts)) := by
  sorry

/-- **Soundness of one table** (interactive-oracle layer). For every prover strategy, the
verifier accepts both reps while the committed values do not satisfy the statement with
probability at most `tableError`. -/
theorem table_sound (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (σ : Strategy (table A S sch hm ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2) (table A S sch hm ptLocal mPts) σ
      ≤ ENNReal.ofReal (tableError sch (S.shape mPts)) := by
  refine (prob_le_value _ _ σ).trans ?_
  set ε := repError sch (S.shape mPts)
  have hε : 0 ≤ ε := repError_nonneg sch _ (fast100_levels_ok S.m sch hsch)
  have hlink0 : 0 ≤ linkError sch (S.shape mPts) := by
    unfold linkError qF; positivity
  simp only [table, bind_eq, pure_eq, recv, draw, bind_send, bind_ret, bind_coin, value_send]
  refine iSup_le fun C => ?_
  by_cases hC : Committed A S sch C
  · -- every outcome carries `C`, so the event never happens
    have h0 : value (fun out : Bool × Oracle F => out.1 = true ∧ ¬ Committed A S sch out.2)
        (.coin (Fin 2 → Fin mPts → F) fun pts =>
          ((rep A S sch hm C (extraClaims S ptLocal mPts pts)).interleave
              (rep A S sch hm C (extraClaims S ptLocal mPts pts))).bind
            fun accepts => .ret (accepts.1 && accepts.2, C)) = 0 := by
      rw [value_coin]
      refine Finset.sum_eq_zero fun pts _ => ?_
      rw [value_bind_ret, value_congr (q := fun _ => False) (fun acc => by simp [hC]),
        value_false, mul_zero]
    rw [h0]
    exact zero_le
  · refine (value_coin_le _ _ _
      (fun pts => ¬ RepDoomed A S sch C (extraClaims S ptLocal mPts pts))
      (ENNReal.ofReal (linkError sch (S.shape mPts))) (ENNReal.ofReal ε * ENNReal.ofReal ε)
      (link_sound A hA S sch hsch ptLocal mPts C hC) fun pts hpts => ?_).trans ?_
    · rw [not_not] at hpts
      rw [value_bind_ret]
      have h := rep_sound hMCA A hA S sch hsch hm mPts C _
        (extraClaims_length S ptLocal mPts pts) hpts
      calc value (fun acc : Bool × Bool =>
              (acc.1 && acc.2, C).1 = true ∧ ¬ Committed A S sch (acc.1 && acc.2, C).2)
              ((rep A S sch hm C (extraClaims S ptLocal mPts pts)).interleave
                (rep A S sch hm C (extraClaims S ptLocal mPts pts)))
          = value (fun acc : Bool × Bool => acc.1 = true ∧ acc.2 = true)
              ((rep A S sch hm C (extraClaims S ptLocal mPts pts)).interleave
                (rep A S sch hm C (extraClaims S ptLocal mPts pts))) :=
            value_congr (fun acc => by simp [hC]) _
        _ ≤ _ := value_interleave_le (fun b : Bool => b = true) (fun b : Bool => b = true) _ _
        _ ≤ ENNReal.ofReal ε * ENNReal.ofReal ε := mul_le_mul' h h
    · rw [tableError, ENNReal.ofReal_add hlink0 (by positivity), sq, ENNReal.ofReal_mul hε]

/-- **The theorem with the numbers.** `fast100` run twice on one root with live coins: at most
`2^-195.5` per table for `22 ≤ m ≤ 33`, for every statement shape in range. -/
theorem table_sound_fast100 (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm1 : 22 ≤ S.m)
    (hm2 : S.m ≤ 33) (ptLocal mPts : ℕ) (hshape : (S.shape mPts).InRange)
    (σ : Strategy (table A S sch (by omega) ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2)
      (table A S sch (by omega) ptLocal mPts) σ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(195.5 : ℝ))) :=
  (table_sound hMCA A hA S sch hsch (by omega) ptLocal mPts σ).trans
    (ENNReal.ofReal_le_ofReal (tableError_le_of_le_33 hm1 hm2 hsch _ hshape))

end FlockSoundness
