import Proofs.Flock.Recursive.Target

/-!
# Recursion: the compiled table on a clash-free set of outcomes

`table_prob_clashFree`: on a statement no witness satisfies, for a set `W` of compiled-table outcomes none of which
self-clashes and no two of which open a position of a cap to different rows, every prover makes the compiled table end
accepted in `W` with probability at most `tableError`. It is `compiled_le_badL`'s coupling with the oracle table, with
the tables the oracle prover sends read off `W` instead of the pluralities: an accepted run in `W` then agrees with
them, so the coupling leaves no bad run and no rewinding term.

* `rowOf`: at each position, the row some outcome of `W` opens there; `tab₀W` and `tabRW` are the level-0 table and a
  rep's table at a slot made of those rows.
* `leafE_le_of_fixed`: along a lockstep whose tables are a fixed function of the slot and the cap, a leaf bound that
  holds wherever the recorded tables are that function's bounds the expectation.
* `bad_zero_of_fixed`: an accepted run in `W` that disagrees with `W`'s tables is a self-clash or a conflict of two runs
  in `W`.
* `compiled_le_bad_on`: `compiled_le_badL` for any chooser, restricted to `W`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Model Accounting Exec
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}

/-! ## The tables of `W` -/

open Classical in
/-- The row some outcome in `W` opens at `p` under `op`, and an arbitrary row where none does. -/
noncomputable def rowOf {Ω P V : Type} [Nonempty V] (op : Ω → P → Option V) (W : Ω → Prop) (p : P) : V :=
  if h : ∃ o v, W o ∧ op o p = some v then h.choose_spec.choose else Classical.ofNonempty

theorem rowOf_spec {Ω P V : Type} [Nonempty V] (op : Ω → P → Option V) (W : Ω → Prop) {p : P}
    (h : ∃ o v, W o ∧ op o p = some v) : ∃ o, W o ∧ op o p = some (rowOf op W p) := by
  unfold rowOf
  rw [dif_pos h]
  exact ⟨h.choose, h.choose_spec.choose_spec.1, h.choose_spec.choose_spec.2⟩

omit [Fintype F] [DecidableEq F] in
/-- A run in `W` that opens a position off `W`'s rows conflicts there with a run in `W`. -/
theorem conflict_of_off {n d : ℕ} (op : OutCL F D Sl → Fin (2 ^ d) → Option (Fin (2 ^ n) → F))
    (W : OutCL F D Sl → Prop) {b : OutCL F D Sl} {p : Fin (2 ^ d)} (hb : W b)
    (hoff : Rewinding.Off op (rowOf op W) b p) : ∃ o, W o ∧ Rewinding.ConflictAt op (b, o) p := by
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hoff.1
  obtain ⟨o, ho, hop⟩ := rowOf_spec op W ⟨b, v, hb, hv⟩
  refine ⟨o, ho, ?_⟩
  show (op b p).isSome ∧ (op o p).isSome ∧ op b p ≠ op o p
  rw [hop]
  exact ⟨hoff.1, rfl, hoff.2⟩

/-- The level-0 table of `W` at the cap `cap`: at each position the row an outcome of `W` opens there under `cap`. -/
noncomputable def tab₀W (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (W : OutCL F D Sl → Prop) (cap : Cap D) : Oracle F :=
  tabOf (lvOf sch 0).lanes (lvOf sch 0).logLen (rowOf (op₀L H E Lf sch cap) W)

/-- The table of `W` at a rep's slot `s` and cap `d`: at each position the row an outcome of `W` opens there. -/
noncomputable def tabRW (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (W : OutCL F D Sl → Prop) (s : ℕ) (d : Cap D) : Oracle F :=
  tabOf (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen (rowOf (opRL H E Lf sch (s % 2) (s / 2) d) W)

theorem conflict₀_of_off (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (W : OutCL F D Sl → Prop) {cap : Cap D} {b : OutCL F D Sl} (hb : W b)
    {p : Fin (2 ^ (lvOf sch 0).logLen)}
    (hoff : Rewinding.Off (op₀L H E Lf sch cap) (rowT _ _ (tab₀W H E Lf sch W cap)) b p) :
    ∃ o, W o ∧ Rewinding.ConflictAt (op₀L H E Lf sch cap) (b, o) p := by
  rw [tab₀W, rowT_tabOf] at hoff
  exact conflict_of_off _ W hb hoff

theorem conflictR_of_off (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (W : OutCL F D Sl → Prop) (r : ℕ) (hr : r < 2) (ℓ : ℕ) (d : Cap D) {b : OutCL F D Sl}
    (hb : W b) {p : Fin (2 ^ (lvOf sch ℓ).logLen)}
    (hoff : Rewinding.Off (opRL H E Lf sch r ℓ d) (rowT _ _ (tabRW H E Lf sch W (slotOf r ℓ) d)) b p) :
    ∃ o, W o ∧ Rewinding.ConflictAt (opRL H E Lf sch r ℓ d) (b, o) p := by
  have key : ∀ s (p : Fin (2 ^ (lvOf sch (s / 2)).logLen)),
      Rewinding.Off (opRL H E Lf sch (s % 2) (s / 2) d)
          (rowT (lvOf sch (s / 2)).lanes (lvOf sch (s / 2)).logLen (tabRW H E Lf sch W s d)) b p →
        ∃ o, W o ∧ Rewinding.ConflictAt (opRL H E Lf sch (s % 2) (s / 2) d) (b, o) p := by
    intro s p h
    rw [tabRW, rowT_tabOf] at h
    exact conflict_of_off _ W hb h
  have h1 : slotOf r ℓ % 2 = r := by unfold slotOf; omega
  have h2 : slotOf r ℓ / 2 = ℓ := by unfold slotOf; omega
  have := key (slotOf r ℓ)
  rw [h1, h2] at this
  exact this p hoff

/-! ## A lockstep whose tables are fixed -/

/-- **A lockstep's tables fixed by the slot and the cap.** With the chooser `c`, which reads only the slot and the cap,
every table a run records at a slot is `c`'s there; so a leaf bound that holds wherever that is so bounds the
expectation. -/
theorem leafE_le_of_fixed {T₀ D₀ I α β : Type} {get : I → ℕ → Option (T₀ × D₀)} {R : I → α → β → Prop}
    (c : ℕ → D₀ → T₀) {f f' : I → β → ℝ}
    (h : ∀ i a b, R i a b → (∀ s t d, get i s = some (t, d) → t = c s d) → f i b ≤ f' i b) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T₀ D₀ get R i g g') (σ' : Strategy g'),
      (∀ s t d, get i s = some (t, d) → t = c s d) →
      Lock.leafE (fun s d _ _ => c s d) f L σ' ≤ Lock.leafE (fun s d _ _ => c s d) f' L σ'
  | _, _, _, @Lock.ret _ _ _ _ _ _ _ i a b hr, _, hi => h i a b hr hi
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ', hi => by
      simp only [Lock.leafE]
      exact div_le_div_of_nonneg_right
        (Finset.sum_le_sum fun x _ => leafE_le_of_fixed c h (L x) (σ' x) hi) (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ', hi => leafE_le_of_fixed c h (L σ'.1) σ'.2 hi
  | _, _, _, .table slot _ _ hset hkeep L, σ', hi =>
      leafE_le_of_fixed c h (L (c slot σ'.1) σ'.1) σ'.2 fun s t d hs => by
        by_cases hsl : s = slot
        · subst hsl
          rw [hset] at hs
          obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hs)
          rfl
        · rw [hkeep _ _ s hsl] at hs
          exact hi s t d hs

theorem getPair_slot (i : IRep F D × IRep F D) (r : ℕ) (hr : r < 2) (ℓ : ℕ) (hℓ : 1 ≤ ℓ) :
    Lock.getPair (getRep 0) (getRep 1) i (slotOf r ℓ) = (if r = 0 then i.1 else i.2) ℓ := by
  rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
  · have h1 : getRep 1 i.2 (slotOf 0 ℓ) = none := by
      unfold getRep slotOf; rw [if_neg (by omega)]
    rw [if_pos rfl]
    unfold Lock.getPair
    rw [getRep_slot 0 (by norm_num) i.1 ℓ hℓ, h1]
    cases i.1 ℓ <;> rfl
  · have h0 : getRep 0 i.1 (slotOf 1 ℓ) = none := by
      unfold getRep slotOf; rw [if_neg (by omega)]
    rw [if_neg (by decide : (1 : ℕ) ≠ 0)]
    simp only [Lock.getPair, h0, Option.orElse_none, getRep_slot 1 (by norm_num) i.2 ℓ hℓ]

/-! ## No run in `W` is bad -/

/-- **An accepted run in `W` agrees with `W`'s tables**: where the tables the run records are `W`'s, disagreeing would
be a self-clash, or a conflict of the run with a run of `W` at the level-0 cap or at a cap the run records. -/
theorem bad_zero_of_fixed (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (sch : Schedule) (W : OutCL F D Sl → Prop)
    (hW₀ : ∀ o, W o → ¬ SelfClashL H E Lf sch o)
    (hW₁ : ∀ o o', W o → W o' → ∀ p, ¬ Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p)
    (hW₂ : ∀ o o', W o → W o' → ∀ i < 2, ∀ ℓ, 1 ≤ ℓ → ∀ cap,
      (if i = 0 then o.pend.1 else o.pend.2).caps[ℓ - 1]? = some cap →
      ∀ p, ¬ Rewinding.ConflictAt (opRL H E Lf sch i ℓ cap) (o, o') p)
    (cap₀ : Cap D) (i : IRep F D × IRep F D) (a : Bool × Oracle F) (b : OutCL F D Sl)
    (hR : TopRelL H E Lf sch (tab₀W H E Lf sch W cap₀) cap₀ i a b)
    (hfix : ∀ s t d, Lock.getPair (getRep 0) (getRep 1) i s = some (t, d) → t = tabRW H E Lf sch W s d) :
    ind (fun out : OutCL F D Sl => out.accepted = true ∧ W out ∧
      ¬ (AgreeRep 0 (tab₀W H E Lf sch W cap₀) i.1 out.pend.1.reqs out.opens ∧
        AgreeRep 1 (tab₀W H E Lf sch W cap₀) i.2 out.pend.2.reqs out.opens)) b ≤ 0 := by
  have hnot : ¬ (b.accepted = true ∧ W b ∧
      ¬ (AgreeRep 0 (tab₀W H E Lf sch W cap₀) i.1 b.pend.1.reqs b.opens ∧
        AgreeRep 1 (tab₀W H E Lf sch W cap₀) i.2 b.pend.2.reqs b.opens)) := by
    rintro ⟨hacc, hb, hbad⟩
    obtain ⟨-, hcap, -, hok, hs₀, hs₁⟩ := hR
    obtain ⟨hok₀, hok₁⟩ := hok hacc
    have key : ∀ r (hr : r < 2),
        RecStruct sch (if r = 0 then i.1 else i.2) (if r = 0 then b.pend.1 else b.pend.2) →
        OpensOKL H E Lf cap₀ r (if r = 0 then b.pend.1 else b.pend.2) b.opens b.salts →
        ¬ AgreeRep r (tab₀W H E Lf sch W cap₀) (if r = 0 then i.1 else i.2)
          (if r = 0 then b.pend.1 else b.pend.2).reqs b.opens → False := by
      intro r hr hs hok hn
      rcases rep_badL H E Lf sch (tab₀W H E Lf sch W cap₀) cap₀ b r hr _ hs hok hn with
        hcl | ⟨p, hoff⟩ | ⟨ℓ, hℓ, t, d, hi, hc, ⟨p, hoff⟩ | hcl⟩
      · exact hW₀ b hb (Or.inl (hcap ▸ hcl))
      · obtain ⟨o, ho, hconf⟩ := conflict₀_of_off H E Lf sch W hb hoff
        exact hW₁ b o hb ho p (hcap ▸ hconf)
      · have ht : t = tabRW H E Lf sch W (slotOf r ℓ) d :=
          hfix _ t d (by rw [getPair_slot i r hr ℓ (Finset.mem_Icc.1 hℓ).1]; exact hi)
        subst ht
        obtain ⟨o, ho, hconf⟩ := conflictR_of_off H E Lf sch W r hr ℓ d hb hoff
        exact hW₂ b o hb ho r hr ℓ (Finset.mem_Icc.1 hℓ).1 d hc p hconf
      · exact hW₀ b hb (Or.inr ⟨r, hr, ℓ, (Finset.mem_Icc.1 hℓ).1, d, hc, hcl⟩)
    by_cases h₀ : AgreeRep 0 (tab₀W H E Lf sch W cap₀) i.1 b.pend.1.reqs b.opens
    · have h₁ : ¬ AgreeRep 1 (tab₀W H E Lf sch W cap₀) i.2 b.pend.2.reqs b.opens := fun h₁ => hbad ⟨h₀, h₁⟩
      exact key 1 (by norm_num) (by simpa using hs₁) (by simpa using hok₁) (by simpa using h₁)
    · exact key 0 (by norm_num) (by simpa using hs₀) (by simpa using hok₀) (by simpa using h₀)
  refine le_of_eq ?_
  unfold ind
  rw [if_neg hnot]

/-! ## The coupling for any chooser -/

/-- **`compiled_le_badL` for any chooser, on `W`.** Whatever table `C₀` the oracle prover sends at the level-0 cap and
whatever tables `ch` chooses at the reps' caps, a salted compiled run ends accepted in `W` with `C₀` uncommitted with
probability at most `tableError` plus the probability of an accepted run in `W` that disagrees with the tables. -/
theorem compiled_le_bad_on (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts) (ch : Lock.Chooser (Oracle F) (Cap D) (OutCL F D Sl))
    (W : OutCL F D Sl → Prop) (C₀ : Oracle F) (cap₀ : Cap D)
    (σ₁ : Strategy (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀)) :
    expect (ind fun out : OutCL F D Sl => out.accepted = true ∧ W out ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts cap₀) σ₁ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      Lock.leafE ch
        (fun (i : IRep F D × IRep F D) (out : OutCL F D Sl) => ind (fun out : OutCL F D Sl =>
          out.accepted = true ∧ W out ∧ ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧
            AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens)) out)
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁ := by
  have hcpl := Lock.expect_le_coupled ch
    (fun out : OutCL F D Sl => out.accepted = true ∧ W out ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ C₀)
    (fun a : Bool × Oracle F => a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
    (fun (i : IRep F D × IRep F D) (out : OutCL F D Sl) => out.accepted = true ∧ W out ∧
      ¬ (AgreeRep 0 C₀ i.1 out.pend.1.reqs out.opens ∧ AgreeRep 1 C₀ i.2 out.pend.2.reqs out.opens))
    (fun i a b hR hE hB => by
      obtain ⟨ha2, -, hacc, -, -, -⟩ := hR
      refine ⟨?_, by rw [ha2]; exact hE.2.2⟩
      by_contra hn
      exact hB ⟨hE.1, hE.2.1, fun h => hn (hacc hE.1 h.1 h.2)⟩)
    (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁
  have horc : expect (ind fun a : Bool × Oracle F =>
        a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
      (tableOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts C₀)
      (Lock.oracleStrat ch (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁) ≤
        tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) := by
    have h1 := expect_cast (ind fun a : Bool × Oracle F =>
        a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
      (withDummy_tableL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)
      (⟨C₀, Lock.oracleStrat ch
        (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁⟩ :
        Strategy (Game.send (Oracle F)
          (tableOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)))
    calc _ = expect (ind fun a : Bool × Oracle F =>
            a.1 = true ∧ ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ a.2)
          (Game.send (Oracle F)
            (tableOrcL (D := D) (Sl := Sl) A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts))
          ⟨C₀, Lock.oracleStrat ch
            (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts C₀ cap₀) σ₁⟩ := rfl
      _ = _ := h1.symm
      _ = _ := Lock.expect_withDummy _ _ _ _
      _ = _ := (prob_toReal _ _ _).symm
      _ ≤ _ := ENNReal.toReal_le_of_le_ofReal (tableError_nonneg _ _)
        (table_sound A hA S _ hsch hm ptLocal mPts hlay _)
  linarith

/-! ## The theorem -/

/-- **The compiled table on a clash-free set of outcomes, against one prover.** On a statement no witness satisfies,
for every set `W` of outcomes none of which self-clashes, no two of which open a position of the first's level-0 cap to
different rows, and no two of which open a position of a rep's level-`ℓ` cap that the first records to different rows:
every prover makes the compiled table end accepted in `W` with probability at most `tableError`. The oracle prover
sends `W`'s tables (`tab₀W` at the prover's cap, `tabRW` at each rep's), so `bad_zero_of_fixed` leaves
`compiled_le_bad_on` no bad term. -/
theorem table_prob_clashFree (A : Arith F K) (hA : A.Correct) [Inhabited D] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (S : Statement) (sch : Schedule)
    (hsch : fast100 S.m = some sch) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (hlay : S.LinkLayout ptLocal mPts)
    (hno : ¬ ∃ z, S.Satisfies z) (W : OutCL F D Sl → Prop)
    (hW₀ : ∀ o, W o → ¬ SelfClashL H E Lf sch o)
    (hW₁ : ∀ o o', W o → W o' → ∀ p, ¬ Rewinding.ConflictAt (op₀L H E Lf sch o.cap₀) (o, o') p)
    (hW₂ : ∀ o o', W o → W o' → ∀ i < 2, ∀ ℓ, 1 ≤ ℓ → ∀ cap,
      (if i = 0 then o.pend.1 else o.pend.2).caps[ℓ - 1]? = some cap →
      ∀ p, ¬ Rewinding.ConflictAt (opRL H E Lf sch i ℓ cap) (o, o') p)
    (σ : Strategy (tableCL A H E Lf S sch hm ptLocal mPts)) :
    prob (fun o : OutCL F D Sl => o.accepted = true ∧ W o) (tableCL A H E Lf S sch hm ptLocal mPts) σ ≤
      ENNReal.ofReal (tableError sch (S.shape mPts)) := by
  obtain ⟨hlen, hk0, -, -⟩ := fast100_shape S.m sch hsch
  obtain ⟨m, k0, levels, ext⟩ := sch
  obtain _ | ⟨l₀, _ | ⟨l₁, rest⟩⟩ := levels
  · simp at hlen
  · simp at hlen
  revert σ
  show ∀ σ : Strategy (Game.send (Cap D)
    (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)), _
  intro σ
  have hfix₀ : ∀ s t d, Lock.getPair (getRep 0) (getRep 1)
      (((fun _ => none), (fun _ => none)) : IRep F D × IRep F D) s = some (t, d) →
      t = tabRW H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W s d := by
    intro s t d h
    simp [Lock.getPair, getRep] at h
  have hbad : Lock.leafE (fun s d _ _ => tabRW H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W s d)
      (fun (i : IRep F D × IRep F D) (out : OutCL F D Sl) => ind (fun out : OutCL F D Sl =>
        out.accepted = true ∧ W out ∧
          ¬ (AgreeRep 0 (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W σ.1) i.1 out.pend.1.reqs out.opens ∧
            AgreeRep 1 (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W σ.1) i.2 out.pend.2.reqs out.opens)) out)
      (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
        (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W σ.1) σ.1) σ.2 ≤ 0 := by
    have h := leafE_le_of_fixed (tabRW H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W) (f' := fun _ _ => 0)
      (fun i a b hR hi => bad_zero_of_fixed H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W hW₀ hW₁ hW₂ σ.1 i a b hR hi)
      (lockTableL A H E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
        (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W σ.1) σ.1) σ.2 hfix₀
    rw [Lock.leafE_const _ (fun _ => 0), expect_const] at h
    exact h
  have h1 := compiled_le_bad_on A hA H E Lf S m k0 l₀ l₁ rest ext hk0 hsch hm ptLocal mPts hlay
    (fun s d _ _ => tabRW H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W s d) W
    (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W σ.1) σ.1 σ.2
  have hmono : expect (ind fun o : OutCL F D Sl => o.accepted = true ∧ W o)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 ≤
    expect (ind fun out : OutCL F D Sl => out.accepted = true ∧ W out ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W σ.1))
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 :=
    expect_mono (f := ind fun o : OutCL F D Sl => o.accepted = true ∧ W o)
      (f' := ind fun out : OutCL F D Sl => out.accepted = true ∧ W out ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (tab₀W H E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ W σ.1))
      (ind_mono fun _ h => ⟨h.1, h.2, fun hc => hno (satisfiable_of_committed hc)⟩) _ _
  have e : expect (ind fun o : OutCL F D Sl => o.accepted = true ∧ W o)
      (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ =
    expect (ind fun o : OutCL F D Sl => o.accepted = true ∧ W o)
      (tableCmpL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1) σ.2 := rfl
  have hp := prob_toReal (fun o : OutCL F D Sl => o.accepted = true ∧ W o)
    (tableCL A H E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ
  refine (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))
    (tableError_nonneg _ _)).2 ?_
  linarith

end FlockSoundness.Discharge.Recursion
