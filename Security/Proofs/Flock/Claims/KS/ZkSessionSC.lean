import Proofs.Flock.Soundness.Discharge.ZkLink.SessionZ

/-!
# Off the masked table without a self-clash

`session_sound_of_table_accZ` with its level-0 term read only on runs that do not self-clash, as `SessionSC` has it at
the plain record: for an event `Acc` of the session's outputs that implies the table accepts, and any level-0 oracle
`C`, `Pr[Acc ∧ C uncommitted] ≤ ε_c⁻ + Pr[Acc ∧ no self-clash ∧ a run opens a level-0 position off C]`
(`session_sound_of_table_accZ_sc`). `ε_c⁻` already counts the self-clash, so a bad run that self-clashes is charged
there (`bad_pointwise_ofZ_sc`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

open Classical in
/-- `bad_pointwise_ofZ` with the level-0 term only where the run does not self-clash. -/
theorem bad_pointwise_ofZ_sc [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (C₀ : Oracle F)
    (cap₀ : Cap D) (i : IZ F D × IZ F D) (a : Bool × Oracle F) (n : ℕ) (Acc : List (OutZC F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD n dfltOutZC).accepted = true) (b : List (OutZC F D Sl))
    (hR : TopRelZ Hs T C₀ cap₀ i a (b.getD n dfltOutZC)) :
    ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ AgreeAllZ Hs T C₀ i (outs.getD n dfltOutZC)) b ≤
      ind (SelfClashZ Hs T) (b.getD n dfltOutZC) +
        ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ SelfClashZ Hs T (outs.getD n dfltOutZC) ∧
          ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C₀) (outs.getD n dfltOutZC) p) b +
        ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
          ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (slotOf r ℓ + 4) = some (t, d) ∧
            ∃ p, Rewinding.Off (opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d) (rowT _ _ t) b p) (b.getD n dfltOutZC) +
        ∑ r ∈ Finset.range 2,
          ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i r = some (t, d) ∧
            ∃ p, Rewinding.Off (opPadsZ Hs T.N T.q r d) (rowPads T.N t) b p) (b.getD n dfltOutZC) +
        ∑ r ∈ Finset.range 2,
          ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (2 + r) = some (t, d) ∧
            ∃ p, Rewinding.Off (opTauZ Hs r d) (rowTau t) b p) (b.getD n dfltOutZC) := by
  have h := bad_pointwise_ofZ Hs T C₀ cap₀ i a n Acc hAcc b hR
  by_cases hsc : SelfClashZ Hs T (b.getD n dfltOutZC)
  · have e1 : ind (SelfClashZ Hs T) (b.getD n dfltOutZC) = 1 := ind_eq_oneZ hsc
    have h1 := ind_le_one (fun outs : List (OutZC F D Sl) => Acc outs ∧
      ¬ AgreeAllZ Hs T C₀ i (outs.getD n dfltOutZC)) b
    have h2 := ind_nonneg (fun outs : List (OutZC F D Sl) => Acc outs ∧
      ¬ SelfClashZ Hs T (outs.getD n dfltOutZC) ∧
        ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C₀) (outs.getD n dfltOutZC) p) b
    have h3 : 0 ≤ ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
        ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (slotOf r ℓ + 4) = some (t, d) ∧
          ∃ p, Rewinding.Off (opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d) (rowT _ _ t) b p) (b.getD n dfltOutZC) :=
      Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _
    have h4 : 0 ≤ ∑ r ∈ Finset.range 2,
        ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i r = some (t, d) ∧
          ∃ p, Rewinding.Off (opPadsZ Hs T.N T.q r d) (rowPads T.N t) b p) (b.getD n dfltOutZC) :=
      Finset.sum_nonneg fun _ _ => ind_nonneg _ _
    have h5 : 0 ≤ ∑ r ∈ Finset.range 2,
        ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (2 + r) = some (t, d) ∧
          ∃ p, Rewinding.Off (opTauZ Hs r d) (rowTau t) b p) (b.getD n dfltOutZC) :=
      Finset.sum_nonneg fun _ _ => ind_nonneg _ _
    linarith
  · have e2 : ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ SelfClashZ Hs T (outs.getD n dfltOutZC) ∧
          ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C₀) (outs.getD n dfltOutZC) p) b =
        ind (fun outs : List (OutZC F D Sl) => Acc outs ∧
          ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C₀) (outs.getD n dfltOutZC) p) b := by
      unfold ind
      split_ifs with h1 h2 h2
      · rfl
      · exact absurd ⟨h1.1, h1.2.2⟩ h2
      · exact absurd ⟨h2.1, hsc, h2.2⟩ h1
      · rfl
    rw [e2]
    exact h

open Classical in
/-- **Compiled soundness of a `--zk` table in a batched session, off the table without a self-clash.** For every
event `Acc` of the session's outputs that implies the table accepts, every session strategy after `Commit` and every
level-0 oracle `C`: `Acc` holds with `C` uncommitted with probability at most `ε_c⁻` plus the probability that `Acc`
holds in a run that does not self-clash and opens a level-0 position off `C`. -/
theorem session_sound_of_table_accZ_sc [Inhabited D] [Inhabited Sl] (A : Arith F K) (hA : A.Correct)
    (Hs : Hashes F D Sl) (T : TabZK F mPts) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (cap₀ : Cap D) (Acc : List (OutZC F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOutZC).accepted = true) (C : Oracle F)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
        (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ ≤
      epsSZ A Hs T pre post cap₀ σ +
        expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
            ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
            ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C) (outs.getD pre.length dfltOutZC) p)
          (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ := by
  have hcpl : expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
        (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) ≤
      expect (ind fun outs : List (OutZC F D Sl) => (outs.getD pre.length dfltOutZC).accepted = true)
          (orcSessZ A T pre post C cap₀) (Lock.oracleStrat (plurSZ Hs T pre.length)
            (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ)) +
        Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (b : List (OutZC F D Sl)) =>
            ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ AgreeAllZ Hs T C i (outs.getD pre.length dfltOutZC)) b)
          (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) :=
    Lock.expect_le_coupled (plurSZ Hs T pre.length)
      (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
      (fun outs : List (OutZC F D Sl) => (outs.getD pre.length dfltOutZC).accepted = true)
      (fun (i : IZ F D × IZ F D) (outs : List (OutZC F D Sl)) =>
        Acc outs ∧ ¬ AgreeAllZ Hs T C i (outs.getD pre.length dfltOutZC))
      (fun i a b hR hE hB => by
        obtain ⟨x, hx, ha2, -, hacc, -, -, -⟩ := hR
        rw [hx]
        have h1 : x.1 = true := by
          by_contra hne
          refine hB ⟨hE.1, fun h => hne ?_⟩
          obtain ⟨h0, h1, p0, p1, t0, t1⟩ := h
          exact hacc (hAcc _ hE.1) h0 h1 p0 p1 t0 t1
        simp only [orcOutZ, h1, Bool.true_and, decide_eq_true_eq]
        rw [ha2]
        exact hE.2)
      (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ)
  have horc : expect (ind fun outs : List (OutZC F D Sl) => (outs.getD pre.length dfltOutZC).accepted = true)
      (orcSessZ A T pre post C cap₀) (Lock.oracleStrat (plurSZ Hs T pre.length)
        (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ)) ≤ T.bound.toReal := by
    rw [← prob_toReal]
    exact ENNReal.toReal_mono (bound_ne_topZ T)
      ((prob_le_value _ _ _).trans (orcSessZ_value_le A hA T pre post C cap₀))
  have hsplit : Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (b : List (OutZC F D Sl)) =>
        ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ AgreeAllZ Hs T C i (outs.getD pre.length dfltOutZC)) b)
        (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) ≤
      expect (fun outs : List (OutZC F D Sl) => ind (SelfClashZ Hs T) (outs.getD pre.length dfltOutZC))
          (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) +
        expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
            ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
            ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C) (outs.getD pre.length dfltOutZC) p)
          (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) +
        ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
          Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (outs : List (OutZC F D Sl)) =>
            ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (slotOf r ℓ + 4) = some (t, d) ∧
              ∃ p, Rewinding.Off (opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d) (rowT _ _ t) b p)
              (outs.getD pre.length dfltOutZC))
            (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) +
        ∑ r ∈ Finset.range 2,
          Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (outs : List (OutZC F D Sl)) =>
            ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i r = some (t, d) ∧
              ∃ p, Rewinding.Off (opPadsZ Hs T.N T.q r d) (rowPads T.N t) b p) (outs.getD pre.length dfltOutZC))
            (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) +
        ∑ r ∈ Finset.range 2,
          Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (outs : List (OutZC F D Sl)) =>
            ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (2 + r) = some (t, d) ∧
              ∃ p, Rewinding.Off (opTauZ Hs r d) (rowTau t) b p) (outs.getD pre.length dfltOutZC))
            (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) := by
    have h := Lock.leafE_mono_rel (plurSZ Hs T pre.length)
      (F := fun (i : IZ F D × IZ F D) (b : List (OutZC F D Sl)) =>
        ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ AgreeAllZ Hs T C i (outs.getD pre.length dfltOutZC)) b)
      (fun i a b hR => by
        obtain ⟨x, -, hrel⟩ := hR
        exact bad_pointwise_ofZ_sc Hs T C cap₀ i x pre.length Acc hAcc b hrel)
      (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ)
    rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at h
    simp only [Lock.leafE_sum] at h
    exact h
  have hlev : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
      Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (outs : List (OutZC F D Sl)) =>
          ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (slotOf r ℓ + 4) = some (t, d) ∧
            ∃ p, Rewinding.Off (opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d) (rowT _ _ t) b p)
            (outs.getD pre.length dfltOutZC))
          (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) ≤
        Real.sqrt (2 ^ (lvOf T.sch ℓ).logLen * (lvOf T.sch ℓ).queries * advRSZ A Hs T pre post cap₀ r ℓ σ) := by
    intro r hr ℓ _
    have h := Lock.leafE_off_le
      (fun d (outs : List (OutZC F D Sl)) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC))
      (rowT (lvOf T.sch ℓ).lanes (lvOf T.sch ℓ).logLen) (slotOf r ℓ + 4) (lvOf T.sch ℓ).queries
      (fun d b => (Exec.opOfL_count _ _ _ _ _ _ _ _).trans (slotOpensZ_length _ _ _ _))
      (plurSZ Hs T pre.length)
      (fun d g σ'' => plurSZ_lev Hs T pre.length r ℓ (Finset.mem_range.1 hr) d g σ'')
      (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) (getPair_empty_none _)
    simp only [Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat] at h
    have heq : advRSZ A Hs T pre post cap₀ r ℓ σ =
        Lock.forkE (plurSZ Hs T pre.length) (slotOf r ℓ + 4)
          (fun d b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p, Rewinding.ConflictAt
            (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC)) x p)
            (b, b'))
          (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) :=
      Lock.forkE_congr _ _ _ (aligned_lockSessZ A Hs T pre post default C cap₀) _
    rw [heq]
    exact h
  have hpads : ∀ r ∈ Finset.range 2,
      Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (outs : List (OutZC F D Sl)) =>
          ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i r = some (t, d) ∧
            ∃ p, Rewinding.Off (opPadsZ Hs T.N T.q r d) (rowPads T.N t) b p) (outs.getD pre.length dfltOutZC))
          (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) ≤
        Real.sqrt (T.N * T.N * advPadsSZ A Hs T pre post cap₀ r σ) := by
    intro r hr
    have h := Lock.leafE_off_le
      (fun d (outs : List (OutZC F D Sl)) => opPadsZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC))
      (rowPads T.N) r T.N (fun d b => (Finset.card_filter_le _ _).trans (Finset.card_fin T.N).le)
      (plurSZ Hs T pre.length)
      (fun d g σ'' => plurSZ_pads Hs T pre.length r (Finset.mem_range.1 hr) d g σ'')
      (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) (getPair_empty_none _)
    rw [Fintype.card_fin] at h
    have heq : advPadsSZ A Hs T pre post cap₀ r σ =
        Lock.forkE (plurSZ Hs T pre.length) r
          (fun d b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p, Rewinding.ConflictAt
            (fun outs : List (OutZC F D Sl) => opPadsZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
          (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) :=
      Lock.forkE_congr _ _ _ (aligned_lockSessZ A Hs T pre post default C cap₀) _
    rw [heq]
    exact h
  have htau : ∀ r ∈ Finset.range 2,
      Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (outs : List (OutZC F D Sl)) =>
          ind (fun b => ∃ t d, Lock.getPair (getZ 0) (getZ 1) i (2 + r) = some (t, d) ∧
            ∃ p, Rewinding.Off (opTauZ Hs r d) (rowTau t) b p) (outs.getD pre.length dfltOutZC))
          (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) ≤
        Real.sqrt (advTauSZ A Hs T pre post cap₀ r σ) := by
    intro r hr
    have h := Lock.leafE_off_le
      (fun d (outs : List (OutZC F D Sl)) => opTauZ Hs r d (outs.getD pre.length dfltOutZC))
      rowTau (2 + r) 1 (fun d b => (Finset.card_filter_le _ _).trans (by simp))
      (plurSZ Hs T pre.length)
      (fun d g σ'' => plurSZ_tau Hs T pre.length r (Finset.mem_range.1 hr) d g σ'')
      (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) (getPair_empty_none _)
    simp only [Fintype.card_unit, Nat.cast_one, one_mul] at h
    have heq : advTauSZ A Hs T pre post cap₀ r σ =
        Lock.forkE (plurSZ Hs T pre.length) (2 + r)
          (fun d b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p, Rewinding.ConflictAt
            (fun outs : List (OutZC F D Sl) => opTauZ Hs r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
          (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) :=
      Lock.forkE_congr _ _ _ (aligned_lockSessZ A Hs T pre post default C cap₀) _
    rw [heq]
    exact h
  have hsumL := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hlev r hr ℓ hℓ
  have hsumP := Finset.sum_le_sum fun r hr => hpads r hr
  have hsumT := Finset.sum_le_sum fun r hr => htau r hr
  have eL : expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
        (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ =
      expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
        (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) :=
    expect_sessAtZ_push A Hs T cap₀ pre post _ σ
  have e0 : expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
        ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
        ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C) (outs.getD pre.length dfltOutZC) p)
        (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ =
      expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
        ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
        ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C) (outs.getD pre.length dfltOutZC) p)
        (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) :=
    expect_sessAtZ_push A Hs T cap₀ pre post _ σ
  have eS : expect (ind fun outs : List (OutZC F D Sl) => SelfClashZ Hs T (outs.getD pre.length dfltOutZC))
        (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ =
      expect (fun outs : List (OutZC F D Sl) => ind (SelfClashZ Hs T) (outs.getD pre.length dfltOutZC))
        (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) :=
    expect_sessAtZ_push A Hs T cap₀ pre post _ σ
  unfold epsSZ
  rw [eL, e0, eS]
  linarith

end FlockSoundness.Discharge.ZkLink
