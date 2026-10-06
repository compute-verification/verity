import Definitions.Flock

/-!
# Soundness of one table (trusted)

The guarantees of `Proofs.Flock.Soundness.Soundness`: for every prover strategy, the verifier accepts a table (both reps,
on one committed root, with live coins) while no codeword within the level-0 radius of the committed table packs a
witness satisfying the statement with probability at most `tableError (fast100 m) shape` (`Sound`), and with the
numbers, `2^-205` for `22 ≤ m ≤ 33` (`SoundFast100`) and for `m = 34, 35` (`SoundFast100_34_35`), for every statement
shape in `Stmt.InRange`. Each assumes the link points' layout (`Statement.LinkLayout`). This is the interactive-oracle
layer: the committed tables are oracles. Each keeps the text and the docstring of the theorem it was in its source module;
its proof is `Flock.SecurityProofs.Table.G`, that theorem under its old name.
-/

namespace Flock.Guarantees.Table

open FlockSoundness FlockSoundness.Game FlockSoundness.Model FlockSoundness.Accounting

/-- **Soundness of one table** (interactive-oracle layer). For every prover strategy, the
verifier accepts both reps while the committed values do not satisfy the statement with
probability at most `tableError`. -/
def Sound : Prop :=
  ∀ {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (A : Arith F K), A.Correct → ∀ (S : Statement) (sch : Schedule), fast100 S.m = some sch → ∀ (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ), S.LinkLayout ptLocal mPts → ∀ σ : Strategy (table A S sch hm ptLocal mPts),
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2) (table A S sch hm ptLocal mPts) σ
      ≤ ENNReal.ofReal (tableError sch (S.shape mPts))

/-- **The theorem with the numbers.** `fast100` run twice on one root with live coins: at most
`2^-205` per table for `22 ≤ m ≤ 33`, for every statement shape in range. -/
def SoundFast100 : Prop :=
  ∀ {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (A : Arith F K), A.Correct → ∀ (S : Statement) (sch : Schedule), fast100 S.m = some sch → ∀ (hm1 : 22 ≤ S.m)
    (hm2 : S.m ≤ 33) (ptLocal mPts : ℕ), (S.shape mPts).InRange → S.LinkLayout ptLocal mPts →
    ∀ σ : Strategy (table A S sch (by omega) ptLocal mPts),
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2)
      (table A S sch (by omega) ptLocal mPts) σ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ)))

/-- **The numbers, `m = 34, 35`.** -/
def SoundFast100_34_35 : Prop :=
  ∀ {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (A : Arith F K), A.Correct → ∀ (S : Statement) (sch : Schedule), fast100 S.m = some sch → ∀ (hm1 : 34 ≤ S.m)
    (hm2 : S.m ≤ 35) (ptLocal mPts : ℕ), (S.shape mPts).InRange → S.LinkLayout ptLocal mPts →
    ∀ σ : Strategy (table A S sch (by omega) ptLocal mPts),
    prob (fun out => out.1 = true ∧ ¬ Committed A S sch out.2)
      (table A S sch (by omega) ptLocal mPts) σ ≤ ENNReal.ofReal ((2 : ℝ) ^ (-(205 : ℝ)))

end Flock.Guarantees.Table
