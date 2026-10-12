import Proofs.Flock.Soundness.RepSound
import Proofs.Flock.Soundness.LinkSound

/-!
# Soundness of Flock's interactive protocol, as PROTOCOL.md specifies it

**`table_sound`.** For every prover strategy, the probability that the verifier accepts a table
(both reps, on one committed root, with live coins) while no codeword within the level-0 radius
of the committed table packs a witness satisfying the statement, is at most
`tableError (fast100 m) shape`.

**`table_sound_fast100`, `table_sound_fast100_34_35`.** With the numbers: at most `2^-205` for
`22 ≤ m ≤ 33` and `2^-205` for `m = 34, 35`, for every statement shape in `Stmt.InRange`.

This is the interactive-oracle layer, a16z's security stage 1a (`DESIGN.md` §7): the
committed tables are oracles. The Merkle layer (`Merkle`, stages 1b and 1c with live coins) adds
SHA-512's collision term for the compiled protocol (`table_sound_compiled`), under `SHA512CRStrict`
(`StrictCR.table_sound_compiled_strict`).

**Status.** Proved. `table_sound` follows from `rep_sound` (one rep from a doomed start,
`RepSound`, itself proved from four phase lemmas, Ligerito's in `LigeritoPhase`) and `link_sound`
(the link points, `LinkSound`). Mutual correlated agreement up to the Johnson bound is proved
(`mcaError_le_bchks25`, from DKT26), so there is no named assumption; `tools/lean/audit.py` audits every declaration's axioms.

Both theorems assume the link points' layout (`Statement.LinkLayout`): each region's free bits
are distinct and read link coordinates below `PT_LOCAL`, and the block coordinates `PT_LOCAL …`
fit inside the point, as the statements of PROTOCOL.md §16 have them.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- **The link points** (§16.2): if the committed values do not satisfy the statement, the two
link points leave some rep start undoomed with probability at most `linkError`: Schwartz–Zippel
on a failing region's difference, at two independent points, for every codeword near the
committed table (`LinkSound`, `link_sound_of_card`, with `closeMsgs_card_le`). -/
theorem link_sound (A : Arith F K) (hA : A.Correct) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (ptLocal mPts : ℕ) (C : Oracle F)
    (hC : ¬ Committed A S sch C) (hlay : S.LinkLayout ptLocal mPts) :
    prCoin (fun pts : Fin 2 → Fin mPts → F =>
        ¬ RepDoomed A S sch C (extraClaims S ptLocal mPts pts)) ≤
      ENNReal.ofReal (linkError sch (S.shape mPts)) := by
  classical
  refine (link_sound_of_card A hA S sch ptLocal mPts C hC hlay _
    (closeMsgs_card_le A hA sch S.m hsch C)).trans (le_of_eq ?_)
  rw [hA.card_F, linkError]
  simp only [Statement.shape]
  rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_mul (by positivity),
    ENNReal.ofReal_natCast, ENNReal.ofReal_natCast, ENNReal.ofReal_pow (by unfold qF; positivity),
    ofReal_nat_div_qF]
  push_cast
  ring

/-- **Soundness of one table, for the optimal prover** (interactive-oracle layer): the verifier
accepts both reps while the committed values do not satisfy the statement with probability at most
`tableError`. `table_sound` is its form for every prover strategy; this form composes with other
tables through `Game.value_interleave_le`. -/
theorem table_value_sound (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts) :
    value (fun out => out.1 = true ∧ ¬ Committed A S sch out.2) (table A S sch hm ptLocal mPts)
      ≤ ENNReal.ofReal (tableError sch (S.shape mPts)) := by
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
      (link_sound A hA S sch hsch ptLocal mPts C hC hlay) fun pts hpts => ?_).trans ?_
    · rw [not_not] at hpts
      rw [value_bind_ret]
      have h := rep_sound A hA S sch hsch hm mPts C _
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

/-- **Soundness of one table** (interactive-oracle layer). For every prover strategy, the
verifier accepts both reps while the committed values do not satisfy the statement with
probability at most `tableError`. -/
theorem table_sound (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (table A S sch hm ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2) (table A S sch hm ptLocal mPts) σ
      ≤ ENNReal.ofReal (tableError sch (S.shape mPts)) :=
  (prob_le_value _ _ σ).trans (table_value_sound A hA S sch hsch hm ptLocal mPts hlay)

/-- **The theorem with the numbers.** `fast100` run twice on one root with live coins: at most
`2^-205` per table for `22 ≤ m ≤ 33`, for every statement shape in range. -/
theorem table_sound_fast100 (A : Arith F K) (hA : A.Correct)
    (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch) (hm1 : 22 ≤ S.m)
    (hm2 : S.m ≤ 33) (ptLocal mPts : ℕ) (hshape : (S.shape mPts).InRange)
    (hlay : S.LinkLayout ptLocal mPts) (σ : Strategy (table A S sch (by omega) ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2)
      (table A S sch (by omega) ptLocal mPts) σ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  (table_sound A hA S sch hsch (by omega) ptLocal mPts hlay σ).trans
    (ENNReal.ofReal_le_ofReal (tableError_le_of_le_33 hm1 hm2 hsch _ hshape))

/-- **The numbers, `m = 34, 35`.** -/
theorem table_sound_fast100_34_35 (A : Arith F K)
    (hA : A.Correct) (S : Statement) (sch : Schedule) (hsch : fast100 S.m = some sch)
    (hm1 : 34 ≤ S.m) (hm2 : S.m ≤ 35) (ptLocal mPts : ℕ) (hshape : (S.shape mPts).InRange)
    (hlay : S.LinkLayout ptLocal mPts) (σ : Strategy (table A S sch (by omega) ptLocal mPts)) :
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2)
      (table A S sch (by omega) ptLocal mPts) σ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ))) :=
  (table_sound A hA S sch hsch (by omega) ptLocal mPts hlay σ).trans
    (ENNReal.ofReal_le_ofReal (tableError_le_of_34_35 hm1 hm2 hsch _ hshape))

end FlockSoundness
