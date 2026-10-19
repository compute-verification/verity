import Proofs.Flock.Claims.KS.ZkPads

/-!
# Off the masked table on a witness lane

`session_sound_of_table_accZ_sc` compares an opened level-0 row with the oracle on all `width₀` lanes, while a table of
`Flock.KS`'s masked columns fixes only the witness lanes (`ZkPads`). Here the extra lanes are the plurality of the
openings whose witness lanes are `C`'s (`Rewinding.exists_pluralityE`), at the cost the old single-table bound paid at
level 0 (`bad_leL`, by `Rewinding.expect_bad_le`). For an event `Acc` of the session's outputs that implies the table
accepts and any level-0 oracle `C`, `Acc` holds with `C` uncommitted with probability at most `ε_c⁻`, plus
`√(2^{d₀}·2Q₀·Adv₀)`, plus that of `Acc` in a run that does not self-clash and opens a level-0 position off `C` on a
witness lane (`session_sound_of_table_accZ_w`). `Adv₀` is two independent runs after `Commit` opening one level-0
position against the witness cap to different rows (`adv₀SZ`). Being committed reads only the witness lanes
(`FlockDeploy.KS.committed_congr_lanes`), so the oracle the session's bound is read at is committed as `C` is.

The fork events `Flock.joinEscZ` joins have no level-0 term. The weight `C` lists level 0's `2·2^{d₀}·Q₀`, but the
masked tables' weight spends it as the `2` that covers `k`'s ceiling (`FlockDeploy.KS.cZ_sum_le_tableWeight`, the pads'
weight at the deployed pads' domain), so a level-0 event needs the weight to grow.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-- **A masked output opens a level-0 position off `C` on a witness lane**: a verifying row at `p` whose lane
`i < 2^k0` is not `C`'s. -/
def OffW₀Z (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D) (C : Oracle F) (out : OutZC F D Sl) : Prop :=
  ∃ p v, op₀Z Hs.H Hs.E Hs.Lf T cap₀ out p = some v ∧
    ∃ i : Fin (width₀ T), i.val < 2 ^ T.sch.k0 ∧ v i ≠ C i.val p.val

open Classical in
/-- The level-0 openings whose witness lanes are `C`'s. -/
noncomputable def opW₀Z (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D) (C : Oracle F) (out : OutZC F D Sl)
    (p : Fin (2 ^ (lvOf T.sch 0).logLen)) : Option (Fin (width₀ T) → F) :=
  (op₀Z Hs.H Hs.E Hs.Lf T cap₀ out p).filter fun v =>
    decide (∀ i : Fin (width₀ T), i.val < 2 ^ T.sch.k0 → v i = C i.val p.val)

theorem op₀Z_of_opW₀Z {Hs : Hashes F D Sl} {T : TabZK F mPts} {cap₀ : Cap D} {C : Oracle F} {out : OutZC F D Sl}
    {p : Fin (2 ^ (lvOf T.sch 0).logLen)} {v : Fin (width₀ T) → F} (h : opW₀Z Hs T cap₀ C out p = some v) :
    op₀Z Hs.H Hs.E Hs.Lf T cap₀ out p = some v := by
  unfold opW₀Z at h
  cases h' : op₀Z Hs.H Hs.E Hs.Lf T cap₀ out p with
  | none => rw [h'] at h; simp at h
  | some w =>
    rw [h'] at h
    simp only [Option.filter] at h
    split at h
    · exact h
    · simp at h

theorem opW₀Z_of_op₀Z {Hs : Hashes F D Sl} {T : TabZK F mPts} {cap₀ : Cap D} {C : Oracle F} {out : OutZC F D Sl}
    {p : Fin (2 ^ (lvOf T.sch 0).logLen)} {v : Fin (width₀ T) → F} (h : op₀Z Hs.H Hs.E Hs.Lf T cap₀ out p = some v)
    (hw : ∀ i : Fin (width₀ T), i.val < 2 ^ T.sch.k0 → v i = C i.val p.val) :
    opW₀Z Hs T cap₀ C out p = some v := by
  unfold opW₀Z
  rw [h]
  simpa [Option.filter] using hw

open Classical in
/-- **A masked output opens at most both reps' `Q₀` level-0 positions.** -/
theorem op₀Z_count (Hs : Hashes F D Sl) (T : TabZK F mPts) (cap₀ : Cap D) (out : OutZC F D Sl) :
    (Finset.univ.filter fun p => (op₀Z Hs.H Hs.E Hs.Lf T cap₀ out p).isSome).card ≤
      2 * (lvOf T.sch 0).queries := by
  calc (Finset.univ.filter fun p => (op₀Z Hs.H Hs.E Hs.Lf T cap₀ out p).isSome).card
      ≤ ((opens₀Z T.sch out).map fun x => x.1).toFinset.card := by
        refine Finset.card_le_card_of_injOn (fun p => p.val) (fun p hp => ?_)
          (fun p _ q _ h => Fin.ext h)
        simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hp
        unfold op₀Z at hp
        rw [Option.isSome_map] at hp
        obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hp
        have hmem := List.mem_of_mem_head? hx
        rw [List.mem_filter, decide_eq_true_eq] at hmem
        simp only [List.coe_toFinset, List.mem_map, Set.mem_ofPred_eq]
        exact ⟨x, hmem.1, hmem.2.1⟩
    _ ≤ ((opens₀Z T.sch out).map fun x => x.1).length := List.toFinset_card_le _
    _ ≤ 2 * (lvOf T.sch 0).queries := by
        rw [List.length_map]
        unfold opens₀Z
        rw [List.length_append]
        have h1 := slotOpensZ_length out 0 0 (lvOf T.sch 0).queries
        have h2 := slotOpensZ_length out 1 0 (lvOf T.sch 0).queries
        omega

open Classical in
/-- **Compiled soundness of a `--zk` table in a batched session, off the table on a witness lane.** For every event
`Acc` of the session's outputs that implies the table accepts, every session strategy after `Commit` and every
level-0 oracle `C`: `Acc` holds with `C` uncommitted with probability at most `ε_c⁻`, plus the level-0 term
`√(2^{d₀}·2Q₀·Adv₀)`, plus the probability that `Acc` holds in a run that does not self-clash and opens a level-0
position off `C` on a witness lane. -/
theorem session_sound_of_table_accZ_w [Inhabited D] [Inhabited Sl] (A : Arith F K) (hA : A.Correct)
    (Hs : Hashes F D Sl) (T : TabZK F mPts) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (cap₀ : Cap D) (Acc : List (OutZC F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOutZC).accepted = true) (C : Oracle F)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
        (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ ≤
      epsSZ A Hs T pre post cap₀ σ +
        Real.sqrt (2 ^ (lvOf T.sch 0).logLen * (2 * (lvOf T.sch 0).queries) * adv₀SZ A Hs T pre post cap₀ σ) +
        expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
            ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
            OffW₀Z Hs T cap₀ C (outs.getD pre.length dfltOutZC))
          (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ := by
  set op : List (OutZC F D Sl) → Fin (2 ^ (lvOf T.sch 0).logLen) → Option (Fin (width₀ T) → F) :=
    fun outs => opW₀Z Hs T cap₀ C (outs.getD pre.length dfltOutZC) with hop
  obtain ⟨vstar, hv⟩ := Rewinding.exists_pluralityE op (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ
  set Cx : Oracle F := fun lane q =>
    if lane < 2 ^ T.sch.k0 then C lane q else tabOfW (width₀ T) (lvOf T.sch 0).logLen vstar lane q with hCx
  have hcomm : T.Committed A C ↔ T.Committed A Cx :=
    FlockDeploy.KS.committed_congr_lanes A T fun ℓ hℓ q => by simp [hCx, hℓ]
  have h1 := session_sound_of_table_accZ_sc A hA Hs T pre post cap₀ Acc hAcc Cx σ
  have e1 : (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C) =
      fun outs => Acc outs ∧ ¬ T.Committed A Cx := by
    funext outs
    rw [hcomm]
  have hpt : ∀ outs : List (OutZC F D Sl),
      ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
        ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ Cx) (outs.getD pre.length dfltOutZC) p) outs ≤
      ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
        OffW₀Z Hs T cap₀ C (outs.getD pre.length dfltOutZC)) outs +
      ind (fun outs => ∃ p, Rewinding.Off op vstar outs p) outs := by
    intro outs
    by_cases hE : Acc outs ∧ ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
        ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ Cx) (outs.getD pre.length dfltOutZC) p
    · obtain ⟨hA', hsc, p, hs, hne⟩ := hE
      obtain ⟨v, hvp⟩ := Option.isSome_iff_exists.1 hs
      by_cases hw : ∀ i : Fin (width₀ T), i.val < 2 ^ T.sch.k0 → v i = C i.val p.val
      · have hW : op outs p = some v := opW₀Z_of_op₀Z hvp hw
        have hoff : ∃ p, Rewinding.Off op vstar outs p := by
          refine ⟨p, by rw [hW]; rfl, fun he => hne ?_⟩
          rw [hW, Option.some.injEq] at he
          rw [hvp, Option.some.injEq]
          funext i
          show v i = Cx i.val p.val
          simp only [hCx]
          split
          · exact hw i (by assumption)
          · simp only [tabOfW, p.isLt, i.isLt, and_self, dite_true, he]
        have e2 : ind (fun outs => ∃ p, Rewinding.Off op vstar outs p) outs = 1 := by
          simp only [ind]
          exact ite_eq_left_of_eq_true _ _ (eq_true hoff)
        have := ind_le_one (fun outs : List (OutZC F D Sl) => Acc outs ∧
          ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
            ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ Cx) (outs.getD pre.length dfltOutZC) p) outs
        have := ind_nonneg (fun outs : List (OutZC F D Sl) => Acc outs ∧
          ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧ OffW₀Z Hs T cap₀ C (outs.getD pre.length dfltOutZC)) outs
        linarith
      · obtain ⟨i, hi, hne'⟩ : ∃ i : Fin (width₀ T), i.val < 2 ^ T.sch.k0 ∧ v i ≠ C i.val p.val := by
          by_contra hc
          exact hw fun i hi => by_contra fun hne => hc ⟨i, hi, hne⟩
        have e2 : ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
            OffW₀Z Hs T cap₀ C (outs.getD pre.length dfltOutZC)) outs = 1 := by
          simp only [ind]
          exact ite_eq_left_of_eq_true _ _ (eq_true ⟨hA', hsc, p, v, hvp, i, hi, hne'⟩)
        have := ind_le_one (fun outs : List (OutZC F D Sl) => Acc outs ∧
          ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
            ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ Cx) (outs.getD pre.length dfltOutZC) p) outs
        have := ind_nonneg (fun outs => ∃ p, Rewinding.Off op vstar outs p) outs
        linarith
    · have e0 : ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧
          ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ Cx) (outs.getD pre.length dfltOutZC) p) outs =
            0 := by
        simp only [ind]
        exact ite_eq_right_of_eq_false _ _ (eq_false hE)
      have := ind_nonneg (fun outs : List (OutZC F D Sl) => Acc outs ∧
        ¬ SelfClashZ Hs T (outs.getD pre.length dfltOutZC) ∧ OffW₀Z Hs T cap₀ C (outs.getD pre.length dfltOutZC)) outs
      have := ind_nonneg (fun outs => ∃ p, Rewinding.Off op vstar outs p) outs
      linarith
  have h2 := expect_mono hpt (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ
  rw [expect_add] at h2
  have hQ : ∀ outs, (Finset.univ.filter fun p => (op outs p).isSome).card ≤ 2 * (lvOf T.sch 0).queries := by
    intro outs
    refine le_trans (Finset.card_le_card fun p hp => ?_) (op₀Z_count Hs T cap₀ (outs.getD pre.length dfltOutZC))
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
    obtain ⟨v, hv'⟩ := Option.isSome_iff_exists.1 hp
    rw [op₀Z_of_opW₀Z hv']
    rfl
  have h3 := Rewinding.expect_bad_le op (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ vstar hv _ hQ
  have h4 : expect₂ (fun b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) =>
        ∃ p, Rewinding.ConflictAt op x p) (b, b')) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ ≤
      adv₀SZ A Hs T pre post cap₀ σ := by
    unfold adv₀SZ expect₂
    refine expect_mono (fun b => expect_mono (fun b' => ind_mono (fun x hx => ?_) _) _ σ) _ σ
    obtain ⟨p, h₁, h₂, hne⟩ := hx
    obtain ⟨v₁, hv₁⟩ := Option.isSome_iff_exists.1 h₁
    obtain ⟨v₂, hv₂⟩ := Option.isSome_iff_exists.1 h₂
    refine ⟨p, ?_, ?_, ?_⟩
    · show (op₀Z Hs.H Hs.E Hs.Lf T cap₀ (x.1.getD pre.length dfltOutZC) p).isSome
      rw [op₀Z_of_opW₀Z hv₁]
      rfl
    · show (op₀Z Hs.H Hs.E Hs.Lf T cap₀ (x.2.getD pre.length dfltOutZC) p).isSome
      rw [op₀Z_of_opW₀Z hv₂]
      rfl
    · show op₀Z Hs.H Hs.E Hs.Lf T cap₀ (x.1.getD pre.length dfltOutZC) p ≠
        op₀Z Hs.H Hs.E Hs.Lf T cap₀ (x.2.getD pre.length dfltOutZC) p
      rw [op₀Z_of_opW₀Z hv₁, op₀Z_of_opW₀Z hv₂]
      rw [hv₁, hv₂] at hne
      exact hne
  have h5 : Real.sqrt (Fintype.card (Fin (2 ^ (lvOf T.sch 0).logLen)) * ((2 * (lvOf T.sch 0).queries : ℕ) : ℝ) *
        expect₂ (fun b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) =>
          ∃ p, Rewinding.ConflictAt op x p) (b, b')) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ) ≤
      Real.sqrt (2 ^ (lvOf T.sch 0).logLen * (2 * (lvOf T.sch 0).queries) * adv₀SZ A Hs T pre post cap₀ σ) := by
    rw [Fintype.card_fin]
    push_cast
    exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left h4 (by positivity))
  rw [e1]
  linarith

end FlockSoundness.Discharge.ZkLink
