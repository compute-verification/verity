import Proofs.Flock.Soundness.Discharge.Exec.SessionSoundL
import Proofs.Flock.Soundness.SessionBatch

/-!
# Several salted tables in one batched session

`SessionBatch.lean` over a leaf scheme `Lf` (`sessionBL`). For each table `j`, everything after `Commit` is a batch
with table `j` at position `j` (`sessAfterB_gamesBL`), so `session_knowledge_soundL` applies to it: `joint_le_BL`,
`joint_le_BL'` (`Pr[the session accepts] · failProbBL ≤ ksBoundBL`) and the audit's `cover`
(`one_le_fail_add_BL`, `one_le_fail_add_BL'`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type} {mPts : ℕ}

omit [Fintype K] in
theorem sessAfterB_eqL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (T : TabSpec mPts) (cap₀ : Cap D) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) :
    sessAfterBL (pre ++ tabAfterL A H E Lf T cap₀ :: post) =
      sessAfterL A H E Lf T.S T.sch T.hm T.ptLocal mPts pre post cap₀ := by
  simp only [sessAfterBL, sessAfterL, List.map_append, List.map_cons]
  rfl

omit [Fintype K] in
/-- **Table `j`'s place in the session**: everything after `Commit` is a batch with table `j` at
position `j`. -/
theorem sessAfterB_gamesBL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (ts : List (TabSpec mPts)) (caps : Fin ts.length → Cap D) (j : Fin ts.length) :
    sessAfterBL (gamesBL A H E Lf ts caps) =
      sessAfterL A H E Lf (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
        ((gamesBL A H E Lf ts caps).take j) ((gamesBL A H E Lf ts caps).drop (j + 1)) (caps j) := by
  have hs : gamesBL A H E Lf ts caps = (gamesBL A H E Lf ts caps).take j ++
      tabAfterL A H E Lf (ts.get j) (caps j) :: (gamesBL A H E Lf ts caps).drop (j + 1) :=
    ofFn_split (fun i => tabAfterL A H E Lf (ts.get i) (caps i)) j
  exact (congrArg sessAfterBL hs).trans (sessAfterB_eqL A H E Lf (ts.get j) (caps j) _ _)

omit [Fintype K] in
theorem take_gamesB_lengthL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (ts : List (TabSpec mPts)) (caps : Fin ts.length → Cap D) (j : Fin ts.length) :
    ((gamesBL A H E Lf ts caps).take j).length = j := by
  simp [gamesBL, j.isLt.le]

/-- The session strategy after `Commit`, seen with table `j` at its position. -/
noncomputable def stratBL (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (ts : List (TabSpec mPts)) (j : Fin ts.length) (τ : Strategy (sessionBL A H E Lf ts)) :
    Strategy (sessAfterL A H E Lf (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
      ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j)) :=
  cast (congrArg Strategy (sessAfterB_gamesBL A H E Lf ts τ.1 j)) τ.2

/-! ## Per table: the extractor's probabilities and the knowledge bound -/

open Classical in
/-- **An event of table `j`'s extraction**: the probability, over `Kr` reruns of everything after
`Commit`, that the table the extractor fills from table `j`'s openings satisfies `G`. -/
noncomputable def extractProbBL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) (G : Oracle F → Prop) : ℝ :=
  expectN (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 Kr
    (ind fun bs => G (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs))

/-- **Table `j`'s extraction fails**: no codeword near the extracted table packs a satisfying
witness. -/
noncomputable def failProbBL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) : ℝ :=
  extractProbBL A H E Lf ts j Kr τ fun T => ¬ Committed A (ts.get j).S (ts.get j).sch T

/-- **Table `j`'s knowledge bound in the session**: `ε_c⁻ + K·adv₀L + N₀/(eK)`, every term the
table's own, computed in the session. -/
noncomputable def ksBoundBL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) : ℝ :=
  epsSL A H E Lf (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest
      (ts.get j).ext (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm
      (ts.get j).ptLocal mPts ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1))
      (τ.1 j) (stratBL A H E Lf ts j τ) +
    (Kr * adv₀SL A H E Lf (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
        ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j)
        (stratBL A H E Lf ts j τ) +
      2 ^ (ts.get j).l₀.logLen / (Real.exp 1 * Kr))

omit [Fintype K] in
theorem extractProbB_nonnegL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) (G : Oracle F → Prop) :
    0 ≤ extractProbBL A H E Lf ts j Kr τ G :=
  expectN_nonneg _ _ (fun _ => ind_nonneg _ _) _

open Classical in
/-- **Knowledge soundness of table `j` in a batched session**, in the product form: table `j`
accepts in a fresh run after `Commit`, while the extraction from `Kr` independent reruns fails,
with probability at most `ε_c⁻ + K·adv₀L + N₀/(eK)`. -/
theorem joint_le_BL (A : Arith F K) (hA : A.Correct)
    [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (Kr : ℕ) (hK : 1 ≤ Kr) (τ : Strategy (sessionBL A H E Lf ts)) :
    expect (ind fun outs : List (OutCL F D Sl) => (outs.getD j dfltOutL).accepted = true)
        (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 * failProbBL A H E Lf ts j Kr τ ≤
      ksBoundBL A H E Lf ts j Kr τ := by
  have h := session_knowledge_soundL A hA H E Lf (ts.get j).S (ts.get j).m (ts.get j).k0
    (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest (ts.get j).ext (ts.get j).hsch (ts.get j).hm
    (ts.get j).ptLocal mPts (ts.get j).hlay ((gamesBL A H E Lf ts τ.1).take j)
    ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j) (stratBL A H E Lf ts j τ) Kr hK
  rw [take_gamesB_lengthL] at h
  unfold stratBL at h
  rw [expectN_expect_cast (sessAfterB_gamesBL A H E Lf ts τ.1 j) τ.2 Kr] at h
  unfold failProbBL extractProbBL ksBoundBL
  rw [expect_mul_expectN]
  exact h

open Classical in
/-- **The same, for the session's verdict**, in the audit's `ℝ≥0∞` form:
`Pr[the session accepts] · Pr[table j's extraction fails] ≤ ksBoundBL`. -/
theorem joint_le_BL' (A : Arith F K) (hA : A.Correct)
    [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (Kr : ℕ) (hK : 1 ≤ Kr) (τ : Strategy (sessionBL A H E Lf ts)) :
    prob (acceptedBL ts) (sessionBL A H E Lf ts) τ * ENNReal.ofReal (failProbBL A H E Lf ts j Kr τ) ≤
      ENNReal.ofReal (ksBoundBL A H E Lf ts j Kr τ) := by
  have hmono : prob (acceptedBL ts) (sessionBL A H E Lf ts) τ ≤
      prob (fun outs : List (OutCL F D Sl) => (outs.getD j dfltOutL).accepted = true)
        (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 :=
    prob_mono (fun outs h => h j) _ _
  have hfin : prob (fun outs : List (OutCL F D Sl) => (outs.getD j dfltOutL).accepted = true)
      (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  calc prob (acceptedBL ts) (sessionBL A H E Lf ts) τ * ENNReal.ofReal (failProbBL A H E Lf ts j Kr τ)
      ≤ prob (fun outs : List (OutCL F D Sl) => (outs.getD j dfltOutL).accepted = true)
          (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 * ENNReal.ofReal (failProbBL A H E Lf ts j Kr τ) :=
        mul_le_mul' hmono le_rfl
    _ = ENNReal.ofReal ((prob (fun outs : List (OutCL F D Sl) => (outs.getD j dfltOutL).accepted = true)
          (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2).toReal * failProbBL A H E Lf ts j Kr τ) := by
        rw [ENNReal.ofReal_mul ENNReal.toReal_nonneg, ENNReal.ofReal_toReal hfin]
    _ ≤ ENNReal.ofReal (ksBoundBL A H E Lf ts j Kr τ) := by
        refine ENNReal.ofReal_le_ofReal ?_
        rw [prob_toReal]
        exact joint_le_BL A hA H E Lf ts j Kr hK τ

omit [Fintype K] in
open Classical in
/-- **The audit's `cover`, per table.** If commitment implies `G`, every extracted table is
uncommitted or satisfies `G`, so table `j`'s failure and `G` have probabilities adding up to at
least `1`. -/
theorem one_le_fail_add_BL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) (G : Oracle F → Prop)
    (hG : ∀ T, Committed A (ts.get j).S (ts.get j).sch T → G T) :
    1 ≤ failProbBL A H E Lf ts j Kr τ + extractProbBL A H E Lf ts j Kr τ G := by
  unfold failProbBL extractProbBL
  rw [← expectN_add]
  calc (1 : ℝ) = expectN (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 Kr (fun _ => 1) :=
        (expectN_const _ _ 1 Kr).symm
    _ ≤ _ := expectN_mono _ _ (fun bs => by
        by_cases hc : Committed A (ts.get j).S (ts.get j).sch
            (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)
        · have hg := hG _ hc
          unfold ind
          rw [if_neg (not_not.2 hc), if_pos hg]
          norm_num
        · have h1 : ind (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
              (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs = 1 := by
            unfold ind; rw [if_pos hc]
          rw [h1]
          linarith [ind_nonneg (fun bs => G (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs]) Kr

/-- `one_le_fail_add_BL` in the audit's `ℝ≥0∞` form. -/
theorem one_le_fail_add_BL' [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) (G : Oracle F → Prop)
    (hG : ∀ T, Committed A (ts.get j).S (ts.get j).sch T → G T) :
    1 ≤ ENNReal.ofReal (failProbBL A H E Lf ts j Kr τ) + ENNReal.ofReal (extractProbBL A H E Lf ts j Kr τ G) := by
  have h0 : 0 ≤ failProbBL A H E Lf ts j Kr τ := extractProbB_nonnegL A H E Lf ts j Kr τ _
  rw [← ENNReal.ofReal_add h0 (extractProbB_nonnegL A H E Lf ts j Kr τ G)]
  exact ENNReal.one_le_ofReal.2 (one_le_fail_add_BL A H E Lf ts j Kr τ G hG)

end FlockSoundness.Discharge.Exec
