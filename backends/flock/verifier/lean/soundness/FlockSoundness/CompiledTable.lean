import FlockSoundness.CompiledSound

/-!
# The compiled bound for any level-0 table

`table_sound_compiled` is stated for `T*(P*)`, the plurality of the level-0 openings (`C0star`).
The same coupling works for any table `T` the oracle prover sends at the level-0 cap
(`compiled_le_bad` is stated for any). Only the level-0 rewinding term changes: a run that opens a
position off `T` either opens it off `T*`, or opens a position where `T` and `T*` differ with
`T*`'s row (`expect_off_table_le`). So `T` costs `dev₀`, the probability of opening a position
where its rows differ from `T*`'s. At a position no run opens, `T` is free.

The reps' cap terms are computed through the lockstep for `T` (`advRT`); at `T = C0star` they are
`advR`.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

open Classical in
/-- **The level-0 deviation** of a table `T` from `T*(P*)`: the probability mass of opening a
position where their rows differ, summed over those positions. -/
noncomputable def dev₀ (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (T : Oracle F)
    (σ : Strategy (tableC A H E S sch hm ptLocal mPts)) : ℝ :=
  ∑ p ∈ Finset.univ.filter (fun p => rowT (lvOf sch 0).lanes (lvOf sch 0).logLen T p ≠
      rowT (lvOf sch 0).lanes (lvOf sch 0).logLen (C0star A H E S sch hm ptLocal mPts σ.1 σ.2) p),
    expect (ind fun out : OutC F D => (op₀ H E sch σ.1 out p).isSome)
      (tableCmp A H E S sch hm ptLocal mPts σ.1) σ.2

/-- `Adv_{r,ℓ}` computed through the lockstep for the level-0 table `T`. -/
noncomputable def advRT [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (T : Oracle F) (r ℓ : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : ℝ :=
  Lock.forkE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (fun d b b' => ind (fun x : OutC F D × OutC F D =>
      ∃ p, Rewinding.ConflictAt (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) x p) (b, b'))
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts T σ.1) σ.2

theorem advRT_C0star [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level)
    (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    advRT A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
        (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) r ℓ σ =
      advR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ :=
  rfl

open Classical in
/-- **Off `T`, off `T*`, or where they differ.** A run that opens a position off the table `v`
opens one off `v'`, or opens a position where `v` and `v'` differ. -/
theorem expect_off_table_le {P V : Type} [Fintype P] [DecidableEq V] {α : Type}
    (op : α → P → Option V) (g : Game α) (s : Strategy g) (v v' : P → V) :
    expect (ind fun b => ∃ p, Rewinding.Off op v b p) g s ≤
      expect (ind fun b => ∃ p, Rewinding.Off op v' b p) g s +
        ∑ p ∈ Finset.univ.filter (fun p => v p ≠ v' p),
          expect (ind fun b => (op b p).isSome) g s := by
  rw [← expect_sum, ← expect_add]
  refine expect_mono (fun b => ?_) g s
  have hsum : 0 ≤ ∑ p ∈ Finset.univ.filter (fun p => v p ≠ v' p),
      ind (fun b => (op b p).isSome) b :=
    Finset.sum_nonneg fun _ _ => ind_nonneg _ _
  by_cases h : ∃ p, Rewinding.Off op v b p
  · obtain ⟨p, hsome, hne⟩ := h
    rw [ind, if_pos ⟨p, hsome, hne⟩]
    by_cases h' : ∃ p, Rewinding.Off op v' b p
    · rw [ind, if_pos h']
      linarith
    · have hv' : op b p = some (v' p) := by
        by_contra hc
        exact h' ⟨p, hsome, hc⟩
      have hvv : v p ≠ v' p := fun heq => hne (by rw [heq]; exact hv')
      have h1 : ind (fun b => (op b p).isSome) b = 1 := if_pos hsome
      have h2 := Finset.single_le_sum (s := Finset.univ.filter (fun p => v p ≠ v' p))
        (f := fun p => ind (fun b => (op b p).isSome) b)
        (fun _ _ => ind_nonneg _ _) (Finset.mem_filter.2 ⟨Finset.mem_univ p, hvv⟩)
      have h3 := ind_nonneg (fun b => ∃ p, Rewinding.Off op v' b p) b
      linarith
  · rw [ind, if_neg h]
    exact add_nonneg (ind_nonneg _ _) hsum

open Classical in
/-- The level-0 rewinding term at `T*(P*)`: at most `√(N₀ · 2Q₀ · Adv₀)`. -/
theorem off_C0star_le (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ)
    (σ : Strategy (Game.send (Cap D)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))) :
    expect (ind fun b => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (rowT _ _ (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2)) b p)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 ≤
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
        adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) := by
  have hpl : Rewinding.IsPluralityE (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2
      (rowT _ _ (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2)) := by
    unfold C0star
    rw [rowT_tabOf]
    exact Classical.choose_spec _
  have hQ : ∀ b, (Finset.univ.filter fun p =>
      (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1 b p).isSome).card ≤ 2 * l₀.queries := by
    intro b
    refine (opOf_count _ _ _ _ _ _ _).trans ?_
    unfold opens₀
    rw [List.length_append]
    have h1 := slotOpens_length b 0 0 (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0).queries
    have h2 := slotOpens_length b 1 0 (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ 0).queries
    simp only [lvOf, List.getD_cons_zero] at h1 h2 ⊢
    omega
  have := Rewinding.expect_bad_le (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1) _ σ.2 _ hpl _ hQ
  rw [Fintype.card_fin] at this
  unfold adv₀
  simpa [lvOf, Nat.cast_pow, Nat.cast_mul] using this

open Classical in
/-- **The bad runs, cap by cap, for any level-0 table `C₀`**, given a bound `x₀` on the level-0
rewinding term. -/
theorem bad_le_gen [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (C₀ : Oracle F)
    (σ : Strategy (Game.send (Cap D)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))) (x₀ : ℝ)
    (h0 : expect (ind fun b => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (rowT _ _ C₀) b p)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 ≤ x₀) :
    Lock.leafE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) (out : OutC F D) => ind (fun out : OutC F D =>
          out.accepted = true ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
            AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) out)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2 ≤
      expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
          (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ + x₀ +
        ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
          Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
            advRT A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ r ℓ σ) := by
  have hsplit := Lock.leafE_mono_rel (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
    (fun i a b hR => bad_pointwise H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ _ σ.1 i a b hR)
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2
  rw [Lock.leafE_add, Lock.leafE_add, Lock.leafE_const, Lock.leafE_const] at hsplit
  simp only [Lock.leafE_sum] at hsplit
  have eS : expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  have hR : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
      Lock.leafE (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
        (fun (i : IRep F D × IRep F D) b => ind (fun b => ∃ t d,
          Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = some (t, d) ∧
            ∃ p, Rewinding.Off (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) (rowT _ _ t) b p) b)
        (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2 ≤
      Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
        (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
        advRT A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ r ℓ σ) := by
    intro r hr ℓ _
    have := Lock.leafE_off_le (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ)
      (rowT (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).lanes (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen)
      (slotOf r ℓ) (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries
      (fun d b => (opOf_count _ _ _ _ _ _ _).trans (slotOpens_length _ _ _ _))
      (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩)
      (fun d g σ' => plurC_plurality H E _ r ℓ (Finset.mem_range.1 hr) d g σ')
      (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ σ.1) σ.2
      (getPair_init_none _)
    rw [Fintype.card_fin] at this
    unfold advRT
    simpa [Nat.cast_pow] using this
  have hsum := Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => hR r hr ℓ hℓ
  rw [eS]
  linarith

open Classical in
/-- **Compiled soundness for any level-0 table** (ASSUMPTIONS.md §1.2, generalized). For every
compiled prover `P*` and every table `T`: `Pr[accepted ∧ no codeword near T packs a satisfying
witness]` is at most `table_sound_compiled`'s bound (with the reps' cap terms through the lockstep
for `T`), plus `dev₀`, the probability of opening a level-0 position where `T`'s rows differ from
`T*(P*)`'s. -/
theorem table_sound_compiled_of_table (hMCA : Assumptions.BCHKS25Thm46) (A : Arith F K)
    (hA : A.Correct) [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (T : Oracle F)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) :
    expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
        adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) +
      dev₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts T σ +
      ∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
          (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries *
          advRT A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts T r ℓ σ) +
      expect (ind (SelfClash H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩))
        (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ := by
  revert σ
  show ∀ σ : Strategy (Game.send (Cap D)
    (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)), _
  intro σ
  have h1 := compiled_le_bad hMCA A hA H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hsch
    hm ptLocal mPts hlay T σ.1 σ.2
  have h0 : expect (ind fun b => ∃ p, Rewinding.Off (op₀ H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ σ.1)
        (rowT _ _ T) b p)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 ≤
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) *
        adv₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ) +
      dev₀ A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts T σ := by
    refine (expect_off_table_le _ _ _ _
      (rowT _ _ (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))).trans ?_
    exact add_le_add (off_C0star_le A H E S m k0 l₀ l₁ rest ext hm ptLocal mPts σ) le_rfl
  have h2 := bad_le_gen A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts T
    σ _ h0
  have e : expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ T)
      (tableCmp A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  rw [e]
  linarith

end FlockSoundness
