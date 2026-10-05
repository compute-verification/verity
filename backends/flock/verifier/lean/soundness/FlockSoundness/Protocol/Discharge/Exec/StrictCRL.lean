import FlockSoundness.Protocol.Discharge.Exec.FlockBatchedL
import FlockSoundness.Protocol.StrictCR

/-!
# The salted compiled and knowledge terms under strict-time collision resistance

`StrictCR.lean`'s finders over a leaf scheme `Lf`: the rewinding finders (`twoRuns`, `forkFinder`) and the self-clash
finder (`oneRun`) run the salted prover, and every pair they count is two verifying salted openings of one position to
different rows, a collision of `H` (`op₀L_conflict_collision`, `opRL_conflict_collision`, `SelfClashL.collision`). So:

* `table_sound_compiled_strictL`: the salted compiled theorem at its finders' budgets;
* `ksBoundAccB_le_strictL`: a salted table's knowledge bound in a session is at most `StrictCR.ksStrict`;
* `ksAvgBE_le_strictL`: the salted audit's knowledge term is at most `StrictCR.ksAvgStrict`, the same closed form as
  the unsalted one: it reads only the plan, the law and the budgets.

The budgets are the hypotheses' to state: the prover is a strategy, and Lean counts no evaluations.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting StrictCR
open scoped ENNReal

/-! ## The compiled theorem -/

section Compiled

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

end Compiled

/-! ## The knowledge term -/

section Session

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
  [Inhabited D] [Inhabited Sl] {mPts : ℕ}

/-- **`Adv_{r,ℓ}`'s finder in a session**: along the session, at the table's rep `r`'s level-`ℓ` cap, two continuations,
and a level-`ℓ` position they open to different rows. -/
noncomputable def advRSFinderL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D) (r ℓ : ℕ)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    Finder Pair :=
  forkFinder (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
    (fun d => conflictPick H
      (fun outs : List (OutCL F D Sl) => opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
      fun x _ h => opRL_conflict_collision (x := (x.1.getD pre.length dfltOutL, x.2.getD pre.length dfltOutL)) h)
    none (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post default cap₀) σ

/-- **`Adv₀`'s finder in a session**: two runs after `Commit`, and a level-0 position of the table they open to different
rows. -/
noncomputable def adv₀SFinderL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl)))
    (cap₀ : Cap D) (σ : Strategy (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀)) : Finder Pair :=
  twoRuns (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀) σ
    (conflictPick H (fun outs : List (OutCL F D Sl) => op₀L H E Lf sch cap₀ (outs.getD pre.length dfltOutL))
      fun x _ h => op₀L_conflict_collision (x := (x.1.getD pre.length dfltOutL, x.2.getD pre.length dfltOutL)) h)

/-- **The self-clash finder in a session**: one run after `Commit`, and a position of the table it opens two ways. -/
noncomputable def selfSFinderL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl)))
    (cap₀ : Cap D) (σ : Strategy (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀)) : Finder Pair :=
  oneRun (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀) σ
    (onePick H (fun outs : List (OutCL F D Sl) => SelfClashL H E Lf sch (outs.getD pre.length dfltOutL))
      fun _ h => h.collision)

/-- **`SHA512CRStrict` for a table's finders in a session**: the rewinding finders at budget `qF`, the self-clash finder
at budget `qS`. -/
structure SessCRL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ)
    (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀))
    (qF qS : ℝ) : Prop where
  fork : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
    (advRSFinderL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ).CR H qF
  level0 : (adv₀SFinderL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀ σ).CR H qF
  self : (selfSFinderL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀ σ).CR H qS

/-- **`SHA512CRStrict` for table `j`'s finders in a batched session**, with the prover's session strategy `τ`. -/
abbrev TableCRL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (τ : Strategy (sessionBL A H E Lf ts)) (qF qS : ℝ) : Prop :=
  SessCRL A H E Lf (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest (ts.get j).ext
    (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm (ts.get j).ptLocal mPts
    ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j) (stratBL A H E Lf ts j τ) qF qS

end Session

end FlockSoundness.Discharge.Exec
