import Proofs.Flock.Soundness.Discharge.Exec.CompiledL
import Proofs.Flock.Soundness.CompiledAlign

/-!
# The salted compiled bound for any level-0 table

`CompiledTable.lean` over a leaf scheme `Lf`. The salted locksteps for two level-0 tables are aligned
(`aligned_lockTableL`), so the reps' fork probabilities do not depend on the level-0 table (`advRTL_eq`), and

> `Pr[accepted ∧ T uncommitted] ≤ ε_c⁻ + Pr[a run opens a level-0 position off T]`
> (`table_sound_compiled_of_tableL`),

with `ε_c⁻ = tableError + Σ √(N_ℓ·Q_ℓ·Adv_{r,ℓ}) + Pr[self-clash]` (`epsCminusL`), one number for every `T`.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type}

/-- **The salted table's locksteps for two level-0 tables are aligned.** -/
theorem aligned_lockTableL [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₁ C₂ : Oracle F)
    (cap₀ : Cap D) :
    Lock.Aligned (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₁ cap₀)
      (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₂ cap₀) := by
  refine .coin _ fun pts => Lock.Aligned.bind ?_ (Lock.Aligned.interleaveLock disjoint_reps
    (aligned_lockRep A 0 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₁ C₂ _)
    (aligned_lockRep A 1 (by norm_num) S m k0 l₀ l₁ rest ext hk0 hm C₁ C₂ _))
  intro _ _ _ _ _ _ _
  exact .send _ fun _ => .ret _ _

/-- `Adv_{r,ℓ}` computed through the salted lockstep for the level-0 table `T`. -/
noncomputable def advRTL [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (T : Oracle F)
    (r ℓ : ℕ) (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : ℝ :=
  Lock.forkE (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (fun d b b' => ind (fun x : OutCL F D Sl × OutCL F D Sl =>
      ∃ p, Rewinding.ConflictAt (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) x p) (b, b'))
    (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts T σ.1) σ.2

theorem advRTL_C0starL [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    advRTL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
        (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) r ℓ σ =
      advRL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ :=
  rfl

open Classical in
/-- **The salted bad runs, cap by cap, for any level-0 table `C₀`**, given a bound `x₀` on the level-0 rewinding
term. -/
theorem bad_le_genL [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₀ : Oracle F)
    (σ : Strategy (Game.send (Cap D)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))) (x₀ : ℝ)
    (h0 : expect (ind fun b => ∃ p, Rewinding.Off (op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (rowT _ _ C₀) b p)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 ≤ x₀) :
    Lock.leafE (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) (out : OutCL F D Sl) => ind (fun out : OutCL F D Sl =>
          out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
            AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) out)
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2 ≤
      expect (ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
          (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ + x₀ +
        ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
          Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
            advRTL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ r ℓ σ) := by
  have hsplit := Lock.leafE_mono_rel (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (fun i a b hR => bad_pointwiseL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ _ σ.1 i a b hR)
    (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2
  rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at hsplit
  simp only [Lock.leafE_sum] at hsplit
  have eS : expect (ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) b => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p) b)
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2 ≤
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRTL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ r ℓ σ) := by
    intro r hr ℓ _
    have := Lock.leafE_off_le (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ)
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
      (slotOf r ℓ) (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries
      (fun d b => (opOfL_count _ _ _ _ _ _ _ _).trans (slotOpensL_length _ _ _ _))
      (plurCL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
      (fun d g σ' => plurCL_plurality H E Lf _ r ℓ (Finset.mem_range.1 hr) d g σ')
      (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2
      (getPair_init_none _)
    rw [Fintype.card_fin] at this
    unfold advRTL
    simpa [Nat.cast_pow] using this
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  rw [eS]
  linarith

/-- **The reps' salted fork probabilities do not depend on the level-0 table.** -/
theorem advRTL_eq [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (T : Oracle F) (r ℓ : ℕ)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    advRTL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts T r ℓ σ =
      advRL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ :=
  Lock.forkE_congr _ _ _ (aligned_lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts T
    (C0starL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2

open Classical in
/-- **`ε_c⁻`** of a salted table: `tableError`, the reps' cap terms and the self-clash probability; the compiled
error without its level-0 rewinding term. -/
noncomputable def epsCminusL [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level)
    (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : ℝ :=
  tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
    ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ) +
    expect (ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ

open Classical in
/-- A salted run opens a level-0 position, with a verifying path, to a row other than `T`'s. -/
def Off₀L (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (sch : Schedule)
    (cap₀ : Cap D) (T : Oracle F) (out : OutCL F D Sl) : Prop :=
  ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT (lvOf sch 0).lanes (lvOf sch 0).logLen T) out p

open Classical in
/-- **Salted compiled soundness for any level-0 table.** For every prover `P*` of `tableCL A H E Lf` and every table
`T`: `Pr[accepted ∧ no codeword near T packs a satisfying witness]` is at most `ε_c⁻` plus the probability that a
run opens a level-0 position off `T`. -/
theorem table_sound_compiled_of_tableL (A : Arith F K)
    (hA : A.Correct) [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (T : Oracle F)
    (σ : Strategy (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    expect (ind fun out : OutCL F D Sl => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ ≤
    epsCminusL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ +
      expect (ind (Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 T))
        (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := by
  revert σ
  show ∀ σ : Strategy (Game.send (Cap D)
    (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)), _
  intro σ
  have h1 := compiled_le_badL A hA H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hsch
    hm ptLocal mPts hlay T σ.1 σ.2
  have h2 := bad_le_genL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts T
    σ _ le_rfl
  have h3 : (∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRTL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts T r ℓ σ)) =
    ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts r ℓ σ) :=
    Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun ℓ _ => by
      congr 1
      congr 1
      exact advRTL_eq A H E Lf S m k0 l₀ l₁ rest ext _ hm ptLocal mPts T r ℓ σ
  have e : expect (ind fun out : OutCL F D Sl => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind fun out : OutCL F D Sl => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  rw [e]
  unfold epsCminusL Off₀L
  linarith

end FlockSoundness.Discharge.Exec
