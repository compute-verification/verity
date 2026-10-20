import Proofs.Flock.Claims.KS.SessionSC
import Proofs.Flock.Soundness.Game.LockOff

/-!
# Off the table without a self-clash, with each slot's term raw

`session_sound_of_table_accL_sc` bounds each rep cap's slot term by `Lock.leafE_off_le`'s root, which needs the
plurality chooser `plurSL`. Before that step, the bound holds at any chooser, with each slot's term the raw mass of runs
that open a position off the table chosen there (`session_sound_of_table_accL_raw`). At `plurSL` that mass is at most
the per-position off mass at the slot's node (`Lock.nodeOff`), which is at most the mass off any other table there
(`Lock.nodeOff_le_of_plurality`): `session_sound_of_table_accL_nodeOff`.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

open Classical in
/-- **Compiled soundness for any level-0 table, in a batched session, with each slot's term raw**, at any chooser:
`Acc` with `T` uncommitted has probability at most `tableError`, plus the self-clash, plus `Acc` without a self-clash
off `T` at level 0, plus, per rep and level, the mass of runs that open a position off the table chosen at the rep's
cap. -/
theorem session_sound_of_table_accL_raw {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] {D Sl : Type} (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutCL F D Sl))) (cap₀ : Cap D)
    (Acc : List (OutCL F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOutL).accepted = true) (T : Oracle F)
    (plur : Lock.Chooser (Oracle F) (Cap D) (List (OutCL F D Sl)))
    (σ : Strategy (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      expect (ind fun outs : List (OutCL F D Sl) =>
          SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
        (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ +
      expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
          ¬ SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL) ∧
          Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOutL))
        (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Lock.leafE plur
          (fun (i : IRep F D × IRep F D) (outs : List (OutCL F D Sl)) => ind (fun b => ∃ t d,
            Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
              ∃ p, Rewinding.Off (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p)
            (outs.getD pre.length dfltOutL))
          (lockSessL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post T cap₀)
          σ := by
  have hk0 := (fast100_shape S.m _ hsch).2.1
  have hcpl := Lock.expect_le_coupled plur
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
      (Lock.oracleStrat plur
        (lockSessL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post T cap₀) σ) ≤
      tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) := by
    rw [← prob_toReal]
    exact ENNReal.toReal_le_of_le_ofReal (tableError_nonneg _ _)
      ((prob_le_value _ _ _).trans (orcSess_value_leL A hA S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hsch hm ptLocal mPts hlay
        pre post T cap₀))
  have hsplit := Lock.leafE_mono_rel plur
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
  rw [e0, eS]
  linarith

open Classical in
/-- **Compiled soundness for any level-0 table, in a batched session, with each slot's off mass at its node**: at
`plurSL`, each slot's raw term is at most the per-position off mass at the rep cap's node (`Lock.nodeOff`), over the
prefixes; that is at most the mass off any other table there (`Lock.nodeOff_le_of_plurality`, `plurS_pluralityL`). -/
theorem session_sound_of_table_accL_nodeOff {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K]
    [Fintype K] [DecidableEq K] {D Sl : Type} (A : Arith F K) (hA : A.Correct) [Inhabited D] [Inhabited Sl]
    (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement)
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
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      expect (ind fun outs : List (OutCL F D Sl) =>
          SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
        (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ +
      expect (ind fun outs : List (OutCL F D Sl) => Acc outs ∧
          ¬ SelfClashL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL) ∧
          Off₀L H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ cap₀ T (outs.getD pre.length dfltOutL))
        (sessAfterL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀) σ +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Lock.nodeE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
          (Lock.nodeOff (fun d (outs : List (OutCL F D Sl)) =>
              opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
            (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
            (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ))
          (lockSessL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post T cap₀)
          σ := by
  have h := session_sound_of_table_accL_raw A hA H E Lf S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay pre post cap₀
    Acc hAcc T (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) σ
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
        (fun (i : IRep F D × IRep F D) (outs : List (OutCL F D Sl)) => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p)
          (outs.getD pre.length dfltOutL))
        (lockSessL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post T cap₀)
        σ ≤
      Lock.nodeE (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
        (Lock.nodeOff (fun d (outs : List (OutCL F D Sl)) =>
            opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
          (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
          (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ))
        (lockSessL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post T cap₀)
        σ := fun r _ ℓ _ =>
    Lock.leafE_off_le_nodeOff
      (fun d (outs : List (OutCL F D Sl)) =>
        opRL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOutL))
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
      (slotOf r ℓ) (plurSL H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length)
      (lockSessL A H E Lf S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts pre post T cap₀) σ
      (getPair_init_none _)
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  linarith

end FlockSoundness.Discharge.Exec
