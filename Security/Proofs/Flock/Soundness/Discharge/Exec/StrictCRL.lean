import Proofs.Flock.Soundness.Discharge.Exec.FlockBatchedAccL
import Proofs.Flock.Soundness.Discharge.Exec.CompiledTableL
import Proofs.Flock.Soundness.StrictCR

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

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

/-! ## The compiled theorem -/

section Compiled

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-- **`Adv₀`'s finder**: two continuations after the prover's level-0 cap, and a level-0 position they open to different
rows. -/
noncomputable def adv₀FinderL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (σ : Strategy (tableCL A H E Lf S sch hm ptLocal mPts)) :
    Finder Pair :=
  twoRuns (tableCmpL A H E Lf S sch hm ptLocal mPts σ.1) σ.2
    (conflictPick H (op₀L H E Lf sch σ.1) fun _ _ h => op₀L_conflict_collision h)

/-- **`Adv_{r,ℓ}`'s finder**: along the run, at rep `r`'s level-`ℓ` cap, two continuations, and a level-`ℓ` position
they open to different rows. -/
noncomputable def advRFinderL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : Finder Pair :=
  forkFinder (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (fun d => conflictPick H (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) fun _ _ h => opRL_conflict_collision h)
    none
    (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2

/-- **The self-clash finder**: one run, and a position it opens two ways. -/
noncomputable def selfFinderL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (σ : Strategy (tableCL A H E Lf S sch hm ptLocal mPts)) :
    Finder Pair :=
  oneRun (tableCL A H E Lf S sch hm ptLocal mPts) σ (onePick H (SelfClashL H E Lf sch) fun _ h => h.collision)

/-- **`SHA512CRStrict` for the compiled theorem's finders**: the rewinding finders at budget `qF`, the self-clash finder at
budget `qS`. -/
structure CompiledCRL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) (qF qS : ℝ) : Prop where
  fork : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
    (advRFinderL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ).CR H qF
  level0 : (adv₀FinderL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ).CR H qF
  self : (selfFinderL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ).CR H qS

omit [Fintype K] in
theorem adv₀_le_strictL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (σ : Strategy (tableCL A H E Lf S sch hm ptLocal mPts))
    {q : ℝ} (h : (adv₀FinderL A H E Lf S sch hm ptLocal mPts σ).CR H q) :
    adv₀L A H E Lf S sch hm ptLocal mPts σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect₂_conflict_le H (tableCmpL A H E Lf S sch hm ptLocal mPts σ.1) σ.2 (op₀L H E Lf sch σ.1)
    fun _ _ h => op₀L_conflict_collision h).trans h.le

theorem advR_le_strictL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) {q : ℝ}
    (h : (advRFinderL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ).CR H q) :
    advRL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (forkE_conflict_le H (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ) (fun _ _ _ h => opRL_conflict_collision h)
    (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2).trans h.le

omit [Fintype K] in
theorem self_le_strictL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (σ : Strategy (tableCL A H E Lf S sch hm ptLocal mPts))
    {q : ℝ} (h : (selfFinderL A H E Lf S sch hm ptLocal mPts σ).CR H q) :
    expect (ind (SelfClashL H E Lf sch)) (tableCL A H E Lf S sch hm ptLocal mPts) σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect_one_le H _ σ (SelfClashL H E Lf sch) fun _ h => h.collision).trans h.le

/-- **The compiled theorem under `SHA512CRStrict`** (`table_sound_compiledL` at its finders). If the rewinding finders
each satisfy `SHA512CRStrict` at budget `qF` and the self-clash finder at `qS`, a compiled run accepts while the table
`P*` committed to is not a codeword of the statement with probability at most
$$\text{tableError}+\sqrt{2^{\log N_0}\cdot 2Q_0\cdot\frac{q_F^2}{2^{513}}}+\sum_{r<2}\sum_{\ell}
\sqrt{2^{\log N_\ell}Q_\ell\cdot\frac{q_F^2}{2^{513}}}+\frac{q_S^2}{2^{513}}.$$ -/
theorem table_sound_compiled_strictL (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) {qF qS : ℝ}
    (hCR : CompiledCRL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ qF qS) :
    expect (ind fun out : OutCL F D Sl => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) * (qF ^ 2 / 2 ^ (513 : ℝ))) +
      ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
          (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
      qS ^ 2 / 2 ^ (513 : ℝ) :=
  (table_sound_compiledL A hA H E Lf S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay σ).trans
    (add_le_add (add_le_add (add_le_add le_rfl
      (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
        (adv₀_le_strictL A H E Lf S _ hm ptLocal mPts σ hCR.level0) (by positivity))))
      (Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => Real.sqrt_le_sqrt
        (mul_le_mul_of_nonneg_left (advR_le_strictL A H E Lf S m k0 l₀ l₁ rest ext _ hm ptLocal mPts r ℓ σ
          (hCR.fork r hr ℓ hℓ)) (by positivity))))
      (self_le_strictL A H E Lf S _ hm ptLocal mPts σ hCR.self))

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

theorem advRS_le_strictL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D) (r ℓ : ℕ)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) {q : ℝ}
    (h : (advRSFinderL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ).CR H q) :
    advRSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (forkE_conflict_le H (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
    (fun d (outs : List (OutCL F D Sl)) => opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
    (fun _ x _ h => opRL_conflict_collision (x := (x.1.getD pre.length dfltOutL, x.2.getD pre.length dfltOutL)) h)
    (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post default cap₀) σ).trans h.le

omit [Fintype K] in
theorem adv₀S_le_strictL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl)))
    (cap₀ : Cap D) (σ : Strategy (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀)) {q : ℝ}
    (h : (adv₀SFinderL A H E Lf S sch hm ptLocal mPts pre post cap₀ σ).CR H q) :
    adv₀SL A H E Lf S sch hm ptLocal mPts pre post cap₀ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect₂_conflict_le H (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀) σ
    (fun outs : List (OutCL F D Sl) => op₀L H E Lf sch cap₀ (outs.getD pre.length dfltOutL))
    fun x _ h => op₀L_conflict_collision (x := (x.1.getD pre.length dfltOutL, x.2.getD pre.length dfltOutL)) h).trans
    h.le

omit [Fintype K] in
theorem selfS_le_strictL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl)))
    (cap₀ : Cap D) (σ : Strategy (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀)) {q : ℝ}
    (h : (selfSFinderL A H E Lf S sch hm ptLocal mPts pre post cap₀ σ).CR H q) :
    expect (ind fun outs : List (OutCL F D Sl) => SelfClashL H E Lf sch (outs.getD pre.length dfltOutL))
      (sessAfterL A H E Lf S sch hm ptLocal mPts pre post cap₀) σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect_one_le H _ σ (fun outs : List (OutCL F D Sl) => SelfClashL H E Lf sch (outs.getD pre.length dfltOutL))
    fun _ h => h.collision).trans h.le

/-- **`ε_c⁻` in the session under `SHA512CRStrict`.** -/
theorem epsS_le_strictL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) {qF qS : ℝ}
    (h : SessCRL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ σ qF qS) :
    epsSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ σ ≤
      tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
        ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
          Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
        qS ^ 2 / 2 ^ (513 : ℝ) :=
  add_le_add (add_le_add le_rfl (Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ =>
      Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (advRS_le_strictL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal
        mPts pre post cap₀ r ℓ σ (h.fork r hr ℓ hℓ)) (by positivity))))
    (selfS_le_strictL A H E Lf S _ hm ptLocal mPts pre post cap₀ σ h.self)

/-- **`SHA512CRStrict` for table `j`'s finders in a batched session**, with the prover's session strategy `τ`. -/
abbrev TableCRL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (τ : Strategy (sessionBL A H E Lf ts)) (qF qS : ℝ) : Prop :=
  SessCRL A H E Lf (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest (ts.get j).ext
    (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm (ts.get j).ptLocal mPts
    ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j) (stratBL A H E Lf ts j τ) qF qS

theorem ksBoundAccB_le_strictL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionBL A H E Lf ts)) {qF qS : ℝ}
    (h : TableCRL A H E Lf ts j τ qF qS) : ksBoundAccBL A H E Lf ts j k τ ≤ ksStrict ts j k qF qS :=
  add_le_add
    (epsS_le_strictL A H E Lf (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest
      (ts.get j).ext (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm (ts.get j).ptLocal mPts
      ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j) (stratBL A H E Lf ts j τ) h)
    (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
      (adv₀S_le_strictL A H E Lf (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
        ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j) (stratBL A H E Lf ts j τ) h.level0)
      (Nat.cast_nonneg k)))

end Session

/-- **The knowledge term with the expected-time extractor, under `SHA512CRStrict`**: `ksAvgBEL` is at most `ksAvgStrict`
at a prover each of whose tables' finders, after each draw, satisfies `SHA512CRStrict` at budgets `qF`, `qS`
(`TableCRL`). -/
theorem ksAvgBE_le_strictL {n : ℕ} {Reg D Sl : Type} [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) {L : Audit.Law n} (k : ℕ) {qF qS : ℝ} (R : Reg)
    (τ : Audit.Cont L Reg (batchedSessionL H E Lf plan) R)
    (hCR : ∀ ω j, TableCRL execArith H E Lf (plan (L.draw ω) R) j (batchStratL H E Lf plan (τ ω)) qF qS) :
    ksAvgBEL H E Lf plan k R τ ≤ ksAvgStrict plan L k qF qS R := by
  refine avg_mono fun ω => iSup_mono fun j => ?_
  have ha := ksBoundAccB_nonnegL execArith H E Lf (plan (L.draw ω) R) j k (batchStratL H E Lf plan (τ ω))
  rcases le_or_gt (1 - rateB (plan (L.draw ω) R) j k) 0 with h0 | h0
  · exact (le_of_eq (ENNReal.ofReal_of_nonpos (div_nonpos_of_nonneg_of_nonpos ha h0))).trans zero_le
  · exact ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right
      (ksBoundAccB_le_strictL execArith H E Lf (plan (L.draw ω) R) j k _ (hCR ω j)) h0.le)

end FlockSoundness.Discharge.Exec
