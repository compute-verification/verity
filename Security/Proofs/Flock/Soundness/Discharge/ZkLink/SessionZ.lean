import Proofs.Flock.Soundness.Discharge.ZkLink.BadZ
import Proofs.Flock.Soundness.Discharge.ZkLink.Analysis
import Proofs.Flock.Soundness.SessionBatchAcc

/-!
# Knowledge soundness of a compiled `--zk` table in the session, from accepted reruns

`SessionSoundL`, `SessionAccL` and `SessionBatchAccL` for the compiled `--zk` session. The table at position
`pre.length` is replaced by its primed form (`expect_sessAtZ_push`), whose lockstep with the oracle session
(`lockSessZ`) fixes a table at every committed object: each level's cap, each rep's pads root and `τ`'s commitment. A
bad run is split by `bad_splitZ`, and the mass of runs off a fixed table is a fork term (`Lock.leafE_off_le`):

* `session_sound_of_table_accZ`: for an event `Acc` of the session's outputs that implies the table accepts, and any
  level-0 oracle `C`, `Pr[Acc ∧ C uncommitted] ≤ ε_c⁻ + Pr[Acc ∧ a run opens a level-0 position off C]`, with `ε_c⁻ =
  epsSZ`: the table's bound, the levels', pads roots' and `τ` commitments' fork terms, and the self-clash;
* `session_knowledge_sound_accZ`: `ε · Pr[the table extracted from k ≥ 1 accepted reruns is uncommitted] ≤
  ε_c⁻ + N₀·ε/(ek) + √(k·adv₀)`;
* `jointZC_ksBound`: `JointZC` at `ksBoundAccZ`, table `j`'s `ε_c⁻ + √(k·adv₀)` in the session, for `k ≥ 1`.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Small facts -/

theorem bound_ne_topZ (T : TabZK F mPts) : T.bound ≠ ⊤ := by
  unfold TabZK.bound εInner
  have hF : (Fintype.card F : ℝ≥0∞) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  refine ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, ENNReal.pow_ne_top (ENNReal.add_ne_top.2
    ⟨ENNReal.ofReal_ne_top, ENNReal.add_ne_top.2 ⟨ENNReal.div_ne_top ?_ hF, ENNReal.pow_ne_top ENNReal.coe_ne_top⟩⟩)⟩
  simp

theorem ind_eq_oneZ {α : Type} {p : α → Prop} {a : α} (h : p a) : ind p a = 1 := by
  simp [ind, h]

theorem ind_eq_zeroZ {α : Type} {p : α → Prop} {a : α} (h : ¬ p a) : ind p a = 0 := by
  simp [ind, h]

theorem le_add5Z {x a b c d e : ℝ} (hx : x ≤ 1) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (he : 0 ≤ e)
    (h : x ≤ 0 ∨ 1 ≤ a ∨ 1 ≤ b ∨ 1 ≤ c ∨ 1 ≤ d ∨ 1 ≤ e) : x ≤ a + b + c + d + e := by
  rcases h with h | h | h | h | h | h <;> linarith

theorem one_le_sumZ {s : Finset ℕ} {f : ℕ → ℝ} (hf : ∀ r, 0 ≤ f r) {r : ℕ} (hr : r ∈ s) (h : f r = 1) :
    1 ≤ ∑ r ∈ s, f r :=
  h.symm.le.trans (Finset.single_le_sum (fun r _ => hf r) hr)

theorem one_le_sum2Z {s t : Finset ℕ} {f : ℕ → ℕ → ℝ} (hf : ∀ r ℓ, 0 ≤ f r ℓ) {r ℓ : ℕ} (hr : r ∈ s) (hℓ : ℓ ∈ t)
    (h : f r ℓ = 1) : 1 ≤ ∑ r ∈ s, ∑ ℓ ∈ t, f r ℓ :=
  h.symm.le.trans ((Finset.single_le_sum (fun ℓ _ => hf r ℓ) hℓ).trans
    (Finset.single_le_sum (f := fun r => ∑ ℓ ∈ t, f r ℓ) (fun r _ => Finset.sum_nonneg fun ℓ _ => hf r ℓ) hr))

/-! ## The bad runs, pointwise -/

open Classical in
/-- **A bad run, pointwise**: for an event `Acc` of the session's outputs that implies the table at position `n`
accepts, a run where `Acc` holds while the table's openings disagree with the fixed tables clashes within the run, or
opens a level-0 position off `C₀`, or a position off one of the reps' fixed tables: a level's, the pads' or `τ`'s. -/
theorem bad_pointwise_ofZ [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (C₀ : Oracle F)
    (cap₀ : Cap D) (i : IZ F D × IZ F D) (a : Bool × Oracle F) (n : ℕ) (Acc : List (OutZC F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD n dfltOutZC).accepted = true) (b : List (OutZC F D Sl))
    (hR : TopRelZ Hs T C₀ cap₀ i a (b.getD n dfltOutZC)) :
    ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ AgreeAllZ Hs T C₀ i (outs.getD n dfltOutZC)) b ≤
      ind (SelfClashZ Hs T) (b.getD n dfltOutZC) +
        ind (fun outs : List (OutZC F D Sl) => Acc outs ∧
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
  refine le_add5Z (ind_le_one _ _) (ind_nonneg _ _) (ind_nonneg _ _)
    (Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => ind_nonneg _ _)
    (Finset.sum_nonneg fun _ _ => ind_nonneg _ _) (Finset.sum_nonneg fun _ _ => ind_nonneg _ _) ?_
  by_cases hB : Acc b ∧ ¬ AgreeAllZ Hs T C₀ i (b.getD n dfltOutZC)
  · refine Or.inr ?_
    rcases bad_splitZ Hs T C₀ cap₀ i a _ hR (hAcc b hB.1) hB.2 with hc | hc | ⟨r, hr, ℓ, hℓ, t, d, hg, hc⟩ |
        ⟨r, hr, t, d, hg, hc⟩ | ⟨r, hr, t, d, hg, hc⟩
    · exact Or.inl (ge_of_eq (ind_eq_oneZ hc))
    · exact Or.inr (Or.inl (ge_of_eq (ind_eq_oneZ (p := fun outs : List (OutZC F D Sl) => Acc outs ∧
        ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C₀) (outs.getD n dfltOutZC) p) ⟨hB.1, hc⟩)))
    · exact Or.inr (Or.inr (Or.inl (one_le_sum2Z (fun _ _ => ind_nonneg _ _) hr hℓ
        (ind_eq_oneZ ⟨t, d, hg, hc⟩))))
    · exact Or.inr (Or.inr (Or.inr (Or.inl (one_le_sumZ (fun _ => ind_nonneg _ _) hr
        (ind_eq_oneZ ⟨t, d, hg, hc⟩)))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (one_le_sumZ (fun _ => ind_nonneg _ _) hr
        (ind_eq_oneZ ⟨t, d, hg, hc⟩)))))
  · exact Or.inl (le_of_eq (ind_eq_zeroZ hB))

/-! ## Rewinding in the session -/

open Classical in
/-- **The tables `P̃` sends in the session**: at each committed object of the table at position `n`, the plurality of
the table's openings there: rep `s`'s pads at slot `s < 2`, rep `s - 2`'s `τ` at `s < 4`, and rep `(s - 4) % 2`'s level
`(s - 4) / 2` above. -/
noncomputable def plurSZ [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n : ℕ) :
    Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl)) :=
  fun s d g σ =>
    if s < 2 then padsTab T.N (Classical.choose (Rewinding.exists_pluralityE
      (fun outs : List (OutZC F D Sl) => opPadsZ Hs T.N T.q s d (outs.getD n dfltOutZC)) g σ))
    else if s < 4 then tauTab (Classical.choose (Rewinding.exists_pluralityE
      (fun outs : List (OutZC F D Sl) => opTauZ Hs (s - 2) d (outs.getD n dfltOutZC)) g σ))
    else tabOf (lvOf T.sch ((s - 4) / 2)).lanes (lvOf T.sch ((s - 4) / 2)).logLen
      (Classical.choose (Rewinding.exists_pluralityE
        (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch ((s - 4) % 2) ((s - 4) / 2) d
          (outs.getD n dfltOutZC)) g σ))

theorem plurSZ_pads [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n r : ℕ) (hr : r < 2)
    (d : Cap D) (g : Game (List (OutZC F D Sl))) (σ : Strategy g) :
    Rewinding.IsPluralityE (fun outs : List (OutZC F D Sl) => opPadsZ Hs T.N T.q r d (outs.getD n dfltOutZC)) g σ
      (rowPads T.N (plurSZ Hs T n r d g σ)) := by
  unfold plurSZ
  simp only [hr, ↓reduceIte, rowPads_padsTab]
  exact Classical.choose_spec (Rewinding.exists_pluralityE _ g σ)

theorem plurSZ_tau [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n r : ℕ) (hr : r < 2)
    (d : Cap D) (g : Game (List (OutZC F D Sl))) (σ : Strategy g) :
    Rewinding.IsPluralityE (fun outs : List (OutZC F D Sl) => opTauZ Hs r d (outs.getD n dfltOutZC)) g σ
      (rowTau (plurSZ Hs T n (2 + r) d g σ)) := by
  have key : ∀ s, 2 ≤ s → s < 4 → Rewinding.IsPluralityE
      (fun outs : List (OutZC F D Sl) => opTauZ Hs (s - 2) d (outs.getD n dfltOutZC)) g σ
      (rowTau (plurSZ Hs T n s d g σ)) := by
    intro s h2 h4
    unfold plurSZ
    simp only [show ¬ s < 2 by omega, h4, ↓reduceIte, rowTau_tauTab]
    exact Classical.choose_spec (Rewinding.exists_pluralityE _ g σ)
  have := key (2 + r) (by omega) (by omega)
  rwa [show 2 + r - 2 = r by omega] at this

theorem plurSZ_lev [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n r ℓ : ℕ) (hr : r < 2)
    (d : Cap D) (g : Game (List (OutZC F D Sl))) (σ : Strategy g) :
    Rewinding.IsPluralityE (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD n dfltOutZC))
      g σ (rowT (lvOf T.sch ℓ).lanes (lvOf T.sch ℓ).logLen (plurSZ Hs T n (slotOf r ℓ + 4) d g σ)) := by
  have key : ∀ s, 4 ≤ s → Rewinding.IsPluralityE
      (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch ((s - 4) % 2) ((s - 4) / 2) d
        (outs.getD n dfltOutZC)) g σ
      (rowT (lvOf T.sch ((s - 4) / 2)).lanes (lvOf T.sch ((s - 4) / 2)).logLen (plurSZ Hs T n s d g σ)) := by
    intro s hs
    unfold plurSZ
    simp only [show ¬ s < 2 by omega, show ¬ s < 4 by omega, ↓reduceIte, rowT_tabOf]
    exact Classical.choose_spec (Rewinding.exists_pluralityE _ g σ)
  have h1 : (slotOf r ℓ + 4 - 4) % 2 = r := by unfold slotOf; omega
  have h2 : (slotOf r ℓ + 4 - 4) / 2 = ℓ := by unfold slotOf; omega
  have := key (slotOf r ℓ + 4) (by omega)
  rw [h1, h2] at this
  exact this

section Terms

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)

/-- **`Adv_{r,ℓ}` in the session**: the collision finder at the table's rep `r`'s level-`ℓ` cap, along the primed
prover. -/
noncomputable def advRSZ (r ℓ : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : ℝ :=
  Lock.forkE (plurSZ Hs T pre.length) (slotOf r ℓ + 4)
    (fun d b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch r ℓ d (outs.getD pre.length dfltOutZC)) x p)
      (b, b'))
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv^pads_r` in the session**: the collision finder at the table's rep `r`'s pads root. -/
noncomputable def advPadsSZ (r : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : ℝ :=
  Lock.forkE (plurSZ Hs T pre.length) r
    (fun d b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC F D Sl) => opPadsZ Hs T.N T.q r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv^τ_r` in the session**: the collision finder at the table's rep `r`'s `τ` commitment. -/
noncomputable def advTauSZ (r : ℕ) (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : ℝ :=
  Lock.forkE (plurSZ Hs T pre.length) (2 + r)
    (fun d b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p, Rewinding.ConflictAt
      (fun outs : List (OutZC F D Sl) => opTauZ Hs r d (outs.getD pre.length dfltOutZC)) x p) (b, b'))
    (lockSessZ A Hs T pre post default cap₀) (pushStrat A Hs T cap₀ pre post σ)

/-- **`Adv₀` in the session**: two runs after `Commit` open a level-0 position of the table to different rows. -/
noncomputable def adv₀SZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : ℝ :=
  expect₂ (fun b b' => ind (fun x : List (OutZC F D Sl) × List (OutZC F D Sl) => ∃ p,
      Rewinding.ConflictAt (fun outs : List (OutZC F D Sl) =>
        op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD pre.length dfltOutZC)) x p) (b, b'))
    (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ

/-- **`ε_c⁻` in the session**: the table's bound, the fork terms of its reps' level caps, pads roots and `τ`
commitments, and its self-clash. -/
noncomputable def epsSZ (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) : ℝ :=
  T.bound.toReal +
    ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 T.sch.levels.length,
      Real.sqrt (2 ^ (lvOf T.sch ℓ).logLen * (lvOf T.sch ℓ).queries * advRSZ A Hs T pre post cap₀ r ℓ σ) +
    ∑ r ∈ Finset.range 2, Real.sqrt (T.N * T.N * advPadsSZ A Hs T pre post cap₀ r σ) +
    ∑ r ∈ Finset.range 2, Real.sqrt (advTauSZ A Hs T pre post cap₀ r σ) +
    expect (ind fun outs : List (OutZC F D Sl) => SelfClashZ Hs T (outs.getD pre.length dfltOutZC))
      (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ

theorem epsSZ_nonneg (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    0 ≤ epsSZ A Hs T pre post cap₀ σ := by
  unfold epsSZ
  exact add_nonneg (add_nonneg (add_nonneg (add_nonneg ENNReal.toReal_nonneg
    (Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _))
    (Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _)) (Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _))
    (expect_nonneg (ind_nonneg _) _ _)

end Terms

/-! ## The compiled bound for any level-0 oracle, in the session -/

open Classical in
/-- **Compiled soundness of a `--zk` table in a batched session, for an acceptance event.** For every event `Acc` of
the session's outputs that implies the table accepts, every session strategy after `Commit` and every level-0 oracle
`C`: `Acc` holds with `C` uncommitted with probability at most `ε_c⁻` plus the probability that `Acc` holds in a run
that opens a level-0 position off `C`. -/
theorem session_sound_of_table_accZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (hA : A.Correct)
    (Hs : Hashes F D Sl) (T : TabZK F mPts) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (cap₀ : Cap D) (Acc : List (OutZC F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOutZC).accepted = true) (C : Oracle F)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
        (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ ≤
      epsSZ A Hs T pre post cap₀ σ +
        expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
            ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C) (outs.getD pre.length dfltOutZC) p)
          (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ := by
  -- the coupling, along the primed prover
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
  -- the oracle side
  have horc : expect (ind fun outs : List (OutZC F D Sl) => (outs.getD pre.length dfltOutZC).accepted = true)
      (orcSessZ A T pre post C cap₀) (Lock.oracleStrat (plurSZ Hs T pre.length)
        (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ)) ≤ T.bound.toReal := by
    rw [← prob_toReal]
    exact ENNReal.toReal_mono (bound_ne_topZ T)
      ((prob_le_value _ _ _).trans (orcSessZ_value_le A hA T pre post C cap₀))
  -- the bad runs, by committed object
  have hsplit : Lock.leafE (plurSZ Hs T pre.length) (fun (i : IZ F D × IZ F D) (b : List (OutZC F D Sl)) =>
        ind (fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ AgreeAllZ Hs T C i (outs.getD pre.length dfltOutZC)) b)
        (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ) ≤
      expect (fun outs : List (OutZC F D Sl) => ind (SelfClashZ Hs T) (outs.getD pre.length dfltOutZC))
          (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) +
        expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
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
        exact bad_pointwise_ofZ Hs T C cap₀ i x pre.length Acc hAcc b hrel)
      (lockSessZ A Hs T pre post C cap₀) (pushStrat A Hs T cap₀ pre post σ)
    rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at h
    simp only [Lock.leafE_sum] at h
    exact h
  -- each level's cap
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
  -- each rep's pads root
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
  -- each rep's `τ` commitment
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
  -- back to the compiled prover
  have eL : expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
        (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ =
      expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧ ¬ T.Committed A C)
        (sessAtZ pre post (tabAfterZCP A Hs T cap₀)) (pushStrat A Hs T cap₀ pre post σ) :=
    expect_sessAtZ_push A Hs T cap₀ pre post _ σ
  have e0 : expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
        ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ C) (outs.getD pre.length dfltOutZC) p)
        (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ =
      expect (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
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

/-! ## Knowledge soundness in the session -/

open Classical in
/-- **Knowledge soundness of a `--zk` table in a batched session, from `k` accepted reruns.** For an event `Acc` of
the session's outputs that implies the table accepts, with probability `ε`: `ε` times the probability that the table
extracted from `k ≥ 1` reruns of everything after `Commit` in which `Acc` holds is uncommitted, is at most
`ε_c⁻ + N₀·ε/(ek) + √(k·adv₀)`. -/
theorem session_knowledge_sound_accZ [Inhabited D] [Inhabited Sl] (A : Arith F K) (hA : A.Correct)
    (Hs : Hashes F D Sl) (T : TabZK F mPts) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (cap₀ : Cap D) (Acc : List (OutZC F D Sl) → Prop)
    (hAcc : ∀ outs, Acc outs → (outs.getD pre.length dfltOutZC).accepted = true)
    (σ : Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) (k : ℕ) (hk : 1 ≤ k) :
    expect (ind Acc) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ *
        Extract.expectAcc (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ Acc k
          (ind fun bs => ¬ T.Committed A (extractTableZ Hs T pre.length cap₀ bs)) ≤
      epsSZ A Hs T pre post cap₀ σ +
        2 ^ (lvOf T.sch 0).logLen * expect (ind Acc) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ /
          (Real.exp 1 * k) +
        Real.sqrt (k * adv₀SZ A Hs T pre post cap₀ σ) := by
  have hX : ∀ bs : List (List (OutZC F D Sl)), ¬ T.Committed A (extractTableZ Hs T pre.length cap₀ bs) →
      expect (ind Acc) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ ≤
        epsSZ A Hs T pre post cap₀ σ +
          expect (ind fun ω : List (OutZC F D Sl) => Acc ω ∧
              ∃ p, Rewinding.Off (fun outs : List (OutZC F D Sl) =>
                op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD pre.length dfltOutZC))
                (Extract.firstTable (fun outs : List (OutZC F D Sl) =>
                  op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD pre.length dfltOutZC)) (fun _ _ => 0) bs) ω p)
            (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ := by
    intro bs hbs
    have h := session_sound_of_table_accZ A hA Hs T pre post cap₀ Acc hAcc
      (extractTableZ Hs T pre.length cap₀ bs) σ
    have e1 : (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
        ¬ T.Committed A (extractTableZ Hs T pre.length cap₀ bs)) = ind Acc := by
      funext outs
      simp only [ind, hbs, not_false_eq_true, and_true]
    have e2 : (ind fun outs : List (OutZC F D Sl) => Acc outs ∧
        ∃ p, Rewinding.Off (op₀Z Hs.H Hs.E Hs.Lf T cap₀) (rowTW _ _ (extractTableZ Hs T pre.length cap₀ bs))
          (outs.getD pre.length dfltOutZC) p) =
        ind fun ω : List (OutZC F D Sl) => Acc ω ∧
          ∃ p, Rewinding.Off (fun outs : List (OutZC F D Sl) =>
            op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD pre.length dfltOutZC))
            (Extract.firstTable (fun outs : List (OutZC F D Sl) =>
              op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD pre.length dfltOutZC)) (fun _ _ => 0) bs) ω p := by
      funext outs
      unfold extractTableZ
      rw [rowTW_tabOfW]
    rw [e1, e2] at h
    exact h
  have hjoint := Extract.joint_acc_le (fun outs : List (OutZC F D Sl) =>
      op₀Z Hs.H Hs.E Hs.Lf T cap₀ (outs.getD pre.length dfltOutZC)) (fun _ _ => 0)
    (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ Acc
    (fun bs => T.Committed A (extractTableZ Hs T pre.length cap₀ bs))
    _ (epsSZ_nonneg A Hs T pre post cap₀ σ) hX k hk
  rw [Fintype.card_fin] at hjoint
  refine hjoint.trans (le_of_eq ?_)
  rw [Nat.cast_pow, Nat.cast_ofNat]
  all_goals rfl

/-! ## Several tables in one batched session -/

section Batch

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts))

/-- Table `j`'s strategy in the session, read at its position. -/
noncomputable def stratZ (j : Fin ts.length) (τ : Strategy (sessionZC A Hs ts)) :
    Strategy (sessAtZ ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1))
      (tabAfterZC A Hs (ts.get j) (τ.1 j))) :=
  cast (congrArg Strategy (sessAfterZC_split A Hs ts τ.1 j)) τ.2

/-- **Table `j`'s knowledge bound from accepted reruns**, without its relative term: `ε_c⁻ + √(k·adv₀)`, every term the
table's own, computed in the session. -/
noncomputable def ksBoundAccZ (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionZC A Hs ts)) : ℝ :=
  epsSZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j)
      (stratZ A Hs ts j τ) +
    Real.sqrt (k * adv₀SZ A Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1))
      (τ.1 j) (stratZ A Hs ts j τ))

theorem ksBoundAccZ_nonneg (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionZC A Hs ts)) :
    0 ≤ ksBoundAccZ A Hs ts j k τ :=
  add_nonneg (epsSZ_nonneg _ _ _ _ _ _ _) (Real.sqrt_nonneg _)

open Classical in
/-- **Knowledge soundness of table `j` in a compiled `--zk` session, from accepted reruns**, in the product form:
`Pr[the session accepts] · Pr[table j's extraction fails] ≤ ε_c⁻ + √(k·adv₀) + N₀/(ek) · Pr[the session accepts]`, over
`k ≥ 1` reruns in which the session accepts. -/
theorem joint_le_accZ (hA : A.Correct) (j : Fin ts.length) (k : ℕ) (hk : 1 ≤ k) (τ : Strategy (sessionZC A Hs ts)) :
    accProbZC A Hs ts τ * failProbAccZ A Hs ts j k τ ≤
      ksBoundAccZ A Hs ts j k τ + rateZ ts j k * accProbZC A Hs ts τ := by
  have hAcc : ∀ outs : List (OutZC F D Sl), acceptedZC ts outs →
      (outs.getD ((gamesZC A Hs ts τ.1).take j).length dfltOutZC).accepted = true := by
    intro outs h
    rw [take_gamesZC_length]
    exact h j
  have h := session_knowledge_sound_accZ A hA Hs (ts.get j) ((gamesZC A Hs ts τ.1).take j)
    ((gamesZC A Hs ts τ.1).drop (j + 1)) (τ.1 j) (acceptedZC ts) hAcc (stratZ A Hs ts j τ) k hk
  rw [take_gamesZC_length] at h
  unfold stratZ at h
  rw [expect_cast' (sessAfterZC_split A Hs ts τ.1 j) τ.2,
    expectAcc_cast (sessAfterZC_split A Hs ts τ.1 j) τ.2] at h
  calc accProbZC A Hs ts τ * failProbAccZ A Hs ts j k τ ≤ _ := h
    _ = _ := by unfold ksBoundAccZ rateZ stratZ accProbZC; ring

end Batch

/-- **The compiled joint inequality, discharged** at the knowledge bound `ksBoundAccZ`, for `k ≥ 1` accepted reruns. -/
theorem jointZC_ksBound [Inhabited D] [Inhabited Sl] (A : Arith F K) (hA : A.Correct) (Hs : Hashes F D Sl) {n : ℕ}
    {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts)) {k : ℕ} (hk : 1 ≤ k) :
    JointZC A Hs planZ k (fun S R τ j => ksBoundAccZ A Hs (planZ S R) j k τ) :=
  fun S R τ j => ⟨ksBoundAccZ_nonneg A Hs (planZ S R) j k τ, joint_le_accZ A Hs (planZ S R) hA j k hk τ⟩

end FlockSoundness.Discharge.ZkLink
