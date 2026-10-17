import Proofs.Flock.Soundness.Discharge.Exec.SessionSoundL

/-!
# Off the table without a self-clash

`session_sound_of_table_accL` with its level-0 term read only on runs that do not self-clash: for an event `Acc` of the
session's outputs that implies the table accepts, and any level-0 table `T`,
`Pr[Acc ∧ T uncommitted] ≤ ε_c⁻ + Pr[Acc ∧ no self-clash ∧ a run opens a level-0 position off T]`
(`session_sound_of_table_accL_sc`). `ε_c⁻` already counts the self-clash, so a bad run that self-clashes is charged
there (`bad_pointwise_ofL_sc`).
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

open Classical in
/-- `bad_pointwise_ofL` with the level-0 term only where the run does not self-clash. -/
theorem bad_pointwise_ofL_sc {F : Type} [Field F] [Fintype F] [DecidableEq F] {D Sl : Type} [Inhabited D]
    [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (C₀ : Oracle F) (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (n : ℕ)
    (Acc : List (OutCL F D Sl) → Prop) (hAcc : ∀ outs, Acc outs → (outs.getD n dfltOutL).accepted = true)
    (b : List (OutCL F D Sl)) (hR : TopRelL H E Lf sch C₀ cap₀ i a (b.getD n dfltOutL)) :
    ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
      ¬ (AgreeRep 0 C₀ i.1 (outs.getD n dfltOutL).pend.1.reqs (outs.getD n dfltOutL).opens ∧
        AgreeRep 1 C₀ i.2 (outs.getD n dfltOutL).pend.2.reqs (outs.getD n dfltOutL).opens)) b ≤
    ind (SelfClashL H E Lf sch) (b.getD n dfltOutL) +
      ind (fun outs : List (OutCL F D Sl) => Acc outs ∧ ¬ SelfClashL H E Lf sch (outs.getD n dfltOutL) ∧
        ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOutL) p) b +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) b p) (b.getD n dfltOutL) := by
  have h := bad_pointwise_ofL H E Lf sch C₀ cap₀ i a n Acc hAcc b hR
  have h2 := ind_nonneg (fun outs : List (OutCL F D Sl) => Acc outs ∧
    ¬ SelfClashL H E Lf sch (outs.getD n dfltOutL) ∧
      ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOutL) p) b
  have h3 : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
      ind (fun b => ∃ t d, Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
        ∃ p, Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ t) b p) (b.getD n dfltOutL) :=
    Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _
  by_cases hsc : SelfClashL H E Lf sch (b.getD n dfltOutL)
  · have e1 : ind (SelfClashL H E Lf sch) (b.getD n dfltOutL) = 1 := by unfold ind; exact ite_eq_left hsc
    have h1 : ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
        ¬ (AgreeRep 0 C₀ i.1 (outs.getD n dfltOutL).pend.1.reqs (outs.getD n dfltOutL).opens ∧
          AgreeRep 1 C₀ i.2 (outs.getD n dfltOutL).pend.2.reqs (outs.getD n dfltOutL).opens)) b ≤ 1 := by
      unfold ind; split_ifs <;> norm_num
    linarith
  · have e2 : ind (fun outs : List (OutCL F D Sl) => Acc outs ∧ ¬ SelfClashL H E Lf sch (outs.getD n dfltOutL) ∧
          ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOutL) p) b =
        ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
          ∃ p, Rewinding.Off (op₀L H E Lf sch cap₀) (rowT _ _ C₀) (outs.getD n dfltOutL) p) b := by
      unfold ind
      split_ifs with h1 h2 h2
      · rfl
      · exact absurd ⟨h1.1, h1.2.2⟩ h2
      · exact absurd ⟨h2.1, hsc, h2.2⟩ h1
      · rfl
    rw [e2]
    exact h

open Classical in
/-- **Compiled soundness for any level-0 table, in a batched session, off the table without a self-clash.** For every
event `Acc` of the session's outputs that implies the table accepts, every session strategy after `Commit` and every
table `T`: `Acc` holds with `T` uncommitted with probability at most `ε_c⁻` plus the probability that `Acc` holds in a
run that does not self-clash and opens a level-0 position off `T`. -/
theorem session_sound_of_table_accL_sc {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] {D Sl : Type} (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (Acc : List (OutCL F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOutL).accepted = true) (T : Oracle F)
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
    epsSL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post cap₀ σ +
      expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
          ¬ SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL) ∧
          Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOutL))
        (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := by
  have hk0 := (fast100_shape S.m _ hsch).2.1
  have hcpl := Lock.expect_le_coupled (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
    (fun outs : List (OutCL F D Sl) => Acc outs ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
    (fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true)
    (fun (i : IRep F D × IRep F D) (outs : List (OutCL F D Sl)) =>
      Acc outs ∧
        ¬ (AgreeRep 0 T i.1 (outs.getD pre.length dfltOutL).pend.1.reqs (outs.getD pre.length dfltOutL).opens ∧
          AgreeRep 1 T i.2 (outs.getD pre.length dfltOutL).pend.2.reqs (outs.getD pre.length dfltOutL).opens))
    (fun i a b hR hE hB => by
      obtain ⟨x, hx, ha2, -, hacc, -, -, -⟩ := hR
      rw [hx]
      have h1 : x.1 = true := by
        by_contra hne
        exact hB ⟨hE.1, fun h => hne (hacc (hAcc _ hE.1) h.1 h.2)⟩
      simp only [orcOutL, h1, Bool.true_and, decide_eq_true_eq]
      rw [ha2]
      exact hE.2)
    (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
  have horc : expect (ind fun outs : List (OutCL F D Sl) => (outs.getD pre.length dfltOutL).accepted = true)
      (orcSessL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post T cap₀)
      (Lock.oracleStrat (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
        (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ) ≤
      tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) := by
    rw [← prob_toReal]
    exact ENNReal.toReal_le_of_le_ofReal (tableError_nonneg _ _)
      ((prob_le_value _ _ _).trans (orcSess_value_leL A hA S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hsch hm ptLocal mPts hlay
        pre post T cap₀))
  have hsplit := Lock.leafE_mono_rel (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
    (F := fun (i : IRep F D × IRep F D) (outs : List (OutCL F D Sl)) => ind (fun outs : List (OutCL F D Sl) =>
      Acc outs ∧
        ¬ (AgreeRep 0 T i.1 (outs.getD pre.length dfltOutL).pend.1.reqs
            (outs.getD pre.length dfltOutL).opens ∧
          AgreeRep 1 T i.2 (outs.getD pre.length dfltOutL).pend.2.reqs
            (outs.getD pre.length dfltOutL).opens)) outs)
    (fun i a b hR => by
      obtain ⟨x, -, hrel⟩ := hR
      exact bad_pointwise_ofL_sc H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T cap₀ i x pre.length Acc hAcc b hrel)
    (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
  rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at hsplit
  simp only [Lock.leafE_sum] at hsplit
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
        (fun (i : IRep F D × IRep F D) (outs : List (OutCL F D Sl)) => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p)
          (outs.getD pre.length dfltOutL))
        (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ ≤
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ) := by
    intro r hr ℓ _
    have := Lock.leafE_off_le
      (fun d (outs : List (OutCL F D Sl)) =>
        opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
      (slotOf r ℓ) (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries
      (fun d b => (opOfL_count _ _ _ _ _ _ _ _).trans (slotOpensL_length _ _ _ _))
      (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
      (fun d g σ' => plurS_pluralityL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length r ℓ (Finset.mem_range.1 hr)
        d g σ')
      (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ
      (getPair_init_none _)
    rw [Fintype.card_fin] at this
    have heq : advRSL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ =
        Lock.forkE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
          (fun d b b' => ind (fun x : List (OutCL F D Sl) × List (OutCL F D Sl) => ∃ p,
            Rewinding.ConflictAt (fun outs : List (OutCL F D Sl) =>
              opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL)) x p) (b, b'))
          (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ :=
      Lock.forkE_congr _ _ _ (aligned_lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre
        post default T cap₀) σ
    rw [heq]
    show Lock.leafE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
        (fun (i : IRep F D × IRep F D) (b : List (OutCL F D Sl)) => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (fun outs : List (OutCL F D Sl) =>
              opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
              (rowT _ _ t) b p) b)
        (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ ≤ _
    simpa [Nat.cast_pow] using this
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  have e0 : expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
        ¬ SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL) ∧
        Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOutL))
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ =
    expect (fun outs : List (OutCL F D Sl) =>
        ind (fun outs : List (OutCL F D Sl) => Acc outs ∧
          ¬ SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL) ∧
          ∃ p, Rewinding.Off (op₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀) (rowT _ _ T)
            (outs.getD pre.length dfltOutL) p) outs)
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := rfl
  have eS : expect (ind fun outs : List (OutCL F D Sl) =>
        SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ =
    expect (fun outs : List (OutCL F D Sl) =>
        ind (SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (outs.getD pre.length dfltOutL))
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ := rfl
  unfold epsSL
  rw [e0, eS]
  linarith

end FlockSoundness.Discharge.Exec
