import Proofs.Flock.Claims.KS.TableBound

/-!
# The old fixed-table bound's terms, refit to `TableBoundθ`

`TableBoundθ` (`KS.Joint`) bounds a run's acceptance at a fixed table by `ε_table`, escapes that fit the weight with
one more fork event of weight `k`, and the probability of accepting while looking up a column of the word off the
table. The old session theorems (`session_sound_of_table_accL`, `session_sound_of_table_accZ`) bound it by their
`ε_c⁻` and the probability of accepting while opening a level-0 position off the oracle. This file refits the old
terms one by one, at one table.

* **The event.** A run that opens a level-0 position off the oracle of a word `C`, where every opened row is the
  lanes of a column the run looks up there, looks up a column other than `C`'s (`exists_ne_of_off`; at the salted
  and masked openings `exists_ne_of_off₀L`, `exists_ne_of_off₀Z`), so the old event's probability over the deployed
  runs is at most the spec's (`Pr_le_of_off`).
* **The table error.** At one table, the old `tableError` at table 0's model statement's shape is the deployed
  `ε_table` (`εtable_one_plain`, `shape_model`); under `--zk` the old padded table's bound is at most it
  (`bound_le_εtable_zk`), since the deployed error bounds the inner proof's by `innerUp`.
* **The fork and self-clash terms.** The old fork terms, one per rep and level `ℓ ≥ 1` with weight `2^{d_ℓ}·Q_ℓ`, and
  the self-clash probability are at most the term of escapes with those weights and one one-run event of weight `1`
  (`oldEsc`, `oldEsc_term`), given fork and one-run events whose probabilities bound them. With at most one fork
  event per round they fit the plain weight with the fork event of weight `k` (`oldEsc_fitsFork_plain`), with level
  0's `2·2^{d₀}·Q₀` of it to spare (`plain_fork_room`), and the salted session's `ε_c⁻` is at most `ε_table` plus
  their term (`epsSL_le`). Under `--zk` each rep adds a pads root's
  fork term of weight `N_p²` and `τ`'s of weight `1` (`IZ`, `cZ`, `oldEscZ`, `oldEscZ_term`); they fit the masked
  weight when `N_p` is at most the deployed pads' domain, level 0's `2·2^{d₀}·Q₀ ≥ 2` paying for `k`'s rounding
  (`oldEscZ_fitsFork_zk`), and the masked session's `ε_c⁻` is at most `ε_table` plus their term (`epsSZ_le`).
* **Mixtures.** Averaged over a mixture of session strategies, `ε_c⁻` is still at most `ε_table` plus the escapes'
  term when the mean `Adv`s are at most the fork events' probabilities, since a mean of roots is at most the root of
  the mean (`mean_sqrt_le`; `mean_epsSL_le`, `mean_epsSZ_le`).
-/

namespace Claims.Flock

/-! ## The event -/

/-- **An opening off the old oracle is a looked-up column off the word**: on a level whose domain is the word's `2^d`
positions, a run that opens a position off the oracle of `C`, where every opened row of `w` lanes is the lanes of a
column the run looks up there, looks up a column other than `C`'s. -/
theorem exists_ne_of_off {F : Type} [Zero F] {N : ℕ} (w d : ℕ) (hN : N = 2 ^ d) (C : Fin N → List F)
    (op : Fin (2 ^ d) → Option (Fin w → F)) (look : ℕ → Option (List F))
    (hlink : ∀ p v, op p = some v → ∃ u, look p.val = some u ∧ (fun lane : Fin w => u.getD lane.val 0) = v)
    (hoff : ∃ p, (op p).isSome ∧ op p ≠ some fun lane : Fin w => oracleOf C lane.val p.val) :
    ∃ (p : Fin N) (u : List F), look p.val = some u ∧ u ≠ C p := by
  obtain ⟨p, hs, hne⟩ := hoff
  obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
  obtain ⟨u, hu, hρ⟩ := hlink p v hv
  have hp : p.val < N := hN ▸ p.isLt
  refine ⟨⟨p.val, hp⟩, u, hu, fun h => hne ?_⟩
  rw [hv, ← hρ, h]
  congr 1
  funext lane
  unfold oracleOf
  rw [dite_eq_left hp]

/-- **The old salted level-0 event is a looked-up column off the word** (`Off₀L` at the oracle of `C`), where every
row the run opens at level 0 is the lanes of a column looked up there. -/
theorem exists_ne_of_off₀L {F D Sl : Type} [Field F] (H : List UInt8 → D) (E : FlockSoundness.Merkle.Enc (List F) D)
    (Lf : FlockSoundness.Merkle.Leaf (List F) Sl D H) (sch : FlockSoundness.Accounting.Schedule)
    (cap₀ : FlockSoundness.Model.Cap D) (out : FlockSoundness.Model.OutCL F D Sl) {N : ℕ}
    (hN : N = 2 ^ (FlockSoundness.lvOf sch 0).logLen) (C : Fin N → List F) (look : ℕ → Option (List F))
    (hlink : ∀ p v, FlockSoundness.Discharge.Exec.op₀L H E Lf sch cap₀ out p = some v →
      ∃ u, look p.val = some u ∧ FlockSoundness.rowFin (FlockSoundness.lvOf sch 0).lanes u = v)
    (hoff : FlockSoundness.Discharge.Exec.Off₀L H E Lf sch cap₀ (oracleOf C) out) :
    ∃ (p : Fin N) (u : List F), look p.val = some u ∧ u ≠ C p :=
  exists_ne_of_off _ _ hN C _ look hlink hoff

/-- **The old masked level-0 event is a looked-up column off the word**: a run that opens a level-0 position off the
oracle of `C` (`op₀Z`, at the table's `width₀` lanes), where every row it opens at level 0 is the lanes of a column
looked up there. -/
theorem exists_ne_of_off₀Z {F D Sl : Type} [Field F] [Fintype F] [DecidableEq F] {mPts : ℕ} (H : List UInt8 → D)
    (E : FlockSoundness.Merkle.Enc (List F) D) (Lf : FlockSoundness.Merkle.Leaf (List F) Sl D H)
    (T : FlockSoundness.Discharge.ZkSession.TabZK F mPts) (cap₀ : FlockSoundness.Model.Cap D)
    (out : FlockSoundness.Discharge.ZkLink.OutZC F D Sl) {N : ℕ} (hN : N = 2 ^ (FlockSoundness.lvOf T.sch 0).logLen)
    (C : Fin N → List F) (look : ℕ → Option (List F))
    (hlink : ∀ p v, FlockSoundness.Discharge.ZkLink.op₀Z H E Lf T cap₀ out p = some v →
      ∃ u, look p.val = some u ∧ (fun lane : Fin (FlockSoundness.Discharge.ZkLink.width₀ T) => u.getD lane.val 0) = v)
    (hoff : ∃ p, FlockSoundness.Rewinding.Off (FlockSoundness.Discharge.ZkLink.op₀Z H E Lf T cap₀)
      (FlockSoundness.Discharge.ZkLink.rowTW _ _ (oracleOf C)) out p) :
    ∃ (p : Fin N) (u : List F), look p.val = some u ∧ u ≠ C p :=
  exists_ne_of_off _ _ hN C _ look hlink hoff

/-- **The old event's probability over the deployed runs is at most the spec's**: when, on every accepted run, every
row opened at a level-0 position is the lanes of the column the run looks up there. -/
theorem Pr_le_of_off {Coin T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {F : Type} [Zero F]
    (H : Msg → Msg) (D : Deploy Coin (List F) T) (N : Msg → ℕ) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (w d : ℕ) (hN : N x = 2 ^ d) (C : Fin (N x) → List F)
    (op : (Fin (D.V.k x) → Coin) → Fin (2 ^ d) → Option (Fin w → F))
    (hlink : ∀ r p v, (run H P D.V x ω r).out = .acc → op r p = some v →
      ∃ u, lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ (fun lane : Fin w => u.getD lane.val 0) = v) :
    (Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
      ∃ p, (op r p).isSome ∧ op r p ≠ some fun lane : Fin w => oracleOf C lane.val p.val) ≤
    Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
      ∃ (p : Fin (N x)) (u : List F), lookCol (D.cols (run H P D.V x ω r).seen 0) p.val = some u ∧ u ≠ C p :=
  Pr_mono fun r ⟨ha, hoff⟩ => ⟨ha, exists_ne_of_off w d hN C (op r) _ (fun p v => hlink r p v ha) hoff⟩

/-! ## The fork and self-clash terms -/

/-- `Σ_{ℓ ∈ [1, L]} g ℓ` over `Fin L`, shifted by one. -/
theorem sum_Icc_one (g : ℕ → ℝ) : ∀ L : ℕ, ∑ ℓ ∈ Finset.Icc 1 L, g ℓ = ∑ ℓ : Fin L, g (ℓ.val + 1)
  | 0 => by simp
  | L + 1 => by
    rw [Finset.sum_Icc_succ_top (by omega), sum_Icc_one g L, Fin.sum_univ_castSucc]
    rfl

/-- **A list's sum is its head and its shifted terms**, for a function that is `0` at the default. -/
theorem head_add_sum_shift {α : Type} [Inhabited α] (f : α → ℝ) (h0 : f default = 0) (ls : List α) :
    f (ls.headD default) + ∑ ℓ : Fin ls.length, f (ls.getD (ℓ.val + 1) default) = (ls.map f).sum := by
  cases ls with
  | nil => simp [h0]
  | cons a t =>
    rw [List.length_cons, Fin.sum_univ_castSucc, List.map_cons, List.sum_cons, ← Fin.sum_univ_fun_getElem t f]
    have e : ∀ i : Fin t.length, (a :: t).getD ((Fin.castSucc i).val + 1) default = t[i.val] := fun i => by
      simp [List.getD_eq_getElem?_getD]
    have e2 : (a :: t).getD ((Fin.last t.length).val + 1) default = default := by simp
    simp only [e, e2, h0, List.headD_cons, add_zero]

/-- A level's fork weight in the old bound, `2^{d_ℓ}·Q_ℓ`. -/
noncomputable def cOf (sch : FlockSoundness.Accounting.Schedule) (ℓ : ℕ) : ℝ :=
  2 ^ (FlockSoundness.lvOf sch ℓ).logLen * (FlockSoundness.lvOf sch ℓ).queries

theorem cOf_nonneg (sch : FlockSoundness.Accounting.Schedule) (ℓ : ℕ) : 0 ≤ cOf sch ℓ := by
  unfold cOf
  positivity

/-- **The old fork and self-clash terms as escapes**: one fork event per rep `r < 2` and level `ℓ + 1` of the schedule,
at round `j r ℓ` with that level's weight `2^{d}·Q`, and one one-run event of weight `1`. -/
noncomputable def oldEsc {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (sch : FlockSoundness.Accounting.Schedule) (j : Fin 2 → Fin sch.levels.length → Fin (V.k x))
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog) : Escapes H V P x where
  I := Fin 2 × Fin sch.levels.length
  c i := cOf sch (i.2.val + 1)
  j i := j i.1 i.2
  ev i := ev i.1 i.2
  L := Unit
  β _ := 1
  ev₁ _ := ev₁
  c_nonneg _ := cOf_nonneg sch _
  β_nonneg _ := zero_le_one
  tied i := tied i.1 i.2
  tied₁ _ := tied₁

/-- **The old fork and self-clash terms are at most the escapes' term**, at a tape where each fork term's `Adv` is at
most its fork event's probability and the self-clash probability at most the one-run event's. -/
theorem oldEsc_term {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin)
    (P : Prover Ω) (x : Msg) (sch : FlockSoundness.Accounting.Schedule)
    (j : Fin 2 → Fin sch.levels.length → Fin (V.k x))
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog) (ω : Ω) (adv : ℕ → ℕ → ℝ) (sc : ℝ)
    (hadv : ∀ r ℓ, adv r.val (ℓ.val + 1) ≤
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev r ℓ ω s.1 s.2)
    (hsc : sc ≤ Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    (∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        Real.sqrt (2 ^ (FlockSoundness.lvOf sch ℓ).logLen * (FlockSoundness.lvOf sch ℓ).queries * adv r ℓ)) + sc ≤
      (oldEsc H V P x sch j ev ev₁ tied tied₁).term ω := by
  show _ ≤ (∑ i : Fin 2 × Fin sch.levels.length, Real.sqrt (cOf sch (i.2.val + 1) *
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev i.1 i.2 ω s.1 s.2)) +
    ∑ _l : Unit, (1 : ℝ) * Pr fun s : Fin (V.k x) → Coin => ev₁ ω s
  have hu : (∑ _l : Unit, (1 : ℝ) * Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) =
      Pr fun s : Fin (V.k x) → Coin => ev₁ ω s := by
    simp
  rw [Fintype.sum_prod_type, Finset.sum_range, hu]
  refine add_le_add (Finset.sum_le_sum fun r _ => ?_) hsc
  rw [sum_Icc_one]
  exact Finset.sum_le_sum fun ℓ _ =>
    Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (hadv r ℓ) (cOf_nonneg sch _))

/-- The old masked bound's fork events: one per rep and level `ℓ + 1`, one per rep's pads root, one per rep's `τ`. -/
abbrev IZ (L : ℕ) : Type := (Fin 2 × Fin L) ⊕ (Fin 2 ⊕ Fin 2)

/-- Their weights: the level's `2^{d}·Q`, the pads root's `N_p²` and `τ`'s `1`. -/
noncomputable def cZ (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) : IZ sch.levels.length → ℝ
  | .inl i => cOf sch (i.2.val + 1)
  | .inr (.inl _) => (Np : ℝ) * Np
  | .inr (.inr _) => 1

theorem cZ_nonneg (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) (i : IZ sch.levels.length) : 0 ≤ cZ sch Np i :=
  match i with
  | .inl _ => cOf_nonneg sch _
  | .inr (.inl _) => mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  | .inr (.inr _) => zero_le_one

/-- **The old masked fork and self-clash terms as escapes**: the fork events of `IZ` at rounds `j` with weights `cZ`,
and one one-run event of weight `1`. -/
noncomputable def oldEscZ {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω)
    (x : Msg) (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) (j : IZ sch.levels.length → Fin (V.k x))
    (ev : IZ sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog) : Escapes H V P x where
  I := IZ sch.levels.length
  c := cZ sch Np
  j := j
  ev := ev
  L := Unit
  β _ := 1
  ev₁ _ := ev₁
  c_nonneg := cZ_nonneg sch Np
  β_nonneg _ := zero_le_one
  tied := tied
  tied₁ _ := tied₁

/-- **The old masked fork and self-clash terms are at most the escapes' term**, at a tape where each fork term's
`Adv` (a level's, a pads root's, `τ`'s) is at most its fork event's probability and the self-clash probability at
most the one-run event's. -/
theorem oldEscZ_term {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin)
    (P : Prover Ω) (x : Msg) (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ)
    (j : IZ sch.levels.length → Fin (V.k x))
    (ev : IZ sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → TiedColl H (run H P V x ω s).vLog) (ω : Ω) (adv : ℕ → ℕ → ℝ) (advP advT : ℕ → ℝ)
    (sc : ℝ)
    (hadv : ∀ r ℓ, adv r.val (ℓ.val + 1) ≤
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inl (r, ℓ)) ω s.1 s.2)
    (hP : ∀ r : Fin 2, advP r.val ≤
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inr (.inl r)) ω s.1 s.2)
    (hT : ∀ r : Fin 2, advT r.val ≤
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inr (.inr r)) ω s.1 s.2)
    (hsc : sc ≤ Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    (∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        Real.sqrt (2 ^ (FlockSoundness.lvOf sch ℓ).logLen * (FlockSoundness.lvOf sch ℓ).queries * adv r ℓ)) +
      ∑ r ∈ Finset.range 2, Real.sqrt (Np * Np * advP r) + ∑ r ∈ Finset.range 2, Real.sqrt (advT r) + sc ≤
      (oldEscZ H V P x sch Np j ev ev₁ tied tied₁).term ω := by
  show _ ≤ (∑ i : (Fin 2 × Fin sch.levels.length) ⊕ (Fin 2 ⊕ Fin 2), Real.sqrt (cZ sch Np i *
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev i ω s.1 s.2)) +
    ∑ _l : Unit, (1 : ℝ) * Pr fun s : Fin (V.k x) → Coin => ev₁ ω s
  have hu : (∑ _l : Unit, (1 : ℝ) * Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) =
      Pr fun s : Fin (V.k x) → Coin => ev₁ ω s := by
    simp
  rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_prod_type, Finset.sum_range, Finset.sum_range,
    Finset.sum_range, hu, ← add_assoc]
  refine add_le_add (add_le_add (add_le_add (Finset.sum_le_sum fun r _ => ?_)
    (Finset.sum_le_sum fun r _ => ?_)) (Finset.sum_le_sum fun r _ => ?_)) hsc
  · rw [sum_Icc_one]
    exact Finset.sum_le_sum fun ℓ _ =>
      Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (hadv r ℓ) (cOf_nonneg sch _))
  · exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (hP r) (cZ_nonneg sch Np (.inr (.inl r))))
  · exact Real.sqrt_le_sqrt ((hT r).trans_eq (one_mul _).symm)

/-! ## Means over a mixture -/

/-- **A mean of roots is at most the root of the mean** (Cauchy–Schwarz). -/
theorem mean_sqrt_le {τ : Type} [Fintype τ] [Nonempty τ] (a : τ → ℝ) (ha : ∀ t, 0 ≤ a t) :
    (∑ t, Real.sqrt (a t)) / Fintype.card τ ≤ Real.sqrt ((∑ t, a t) / Fintype.card τ) := by
  have hn : (0 : ℝ) < Fintype.card τ := Nat.cast_pos.2 Fintype.card_pos
  have cs := Real.sum_sqrt_mul_sqrt_le Finset.univ ha fun _ => zero_le_one
  simp only [Real.sqrt_one, mul_one, Finset.sum_const, Finset.card_univ, nsmul_eq_mul] at cs
  rw [div_le_iff₀ hn, Real.sqrt_div' _ hn.le, div_mul_eq_mul_div, le_div_iff₀ (Real.sqrt_pos.2 hn)]
  calc (∑ t, Real.sqrt (a t)) * Real.sqrt (Fintype.card τ)
      ≤ Real.sqrt (∑ t, a t) * Real.sqrt (Fintype.card τ) * Real.sqrt (Fintype.card τ) :=
        mul_le_mul_of_nonneg_right cs (Real.sqrt_nonneg _)
    _ = Real.sqrt (∑ t, a t) * Fintype.card τ := by
        rw [mul_assoc, Real.mul_self_sqrt hn.le]

theorem mean_sum_sqrt_le {τ ι : Type} [Fintype τ] [Nonempty τ] (s : Finset ι) (a : ι → τ → ℝ)
    (ha : ∀ i t, 0 ≤ a i t) :
    (∑ t, ∑ i ∈ s, Real.sqrt (a i t)) / Fintype.card τ ≤ ∑ i ∈ s, Real.sqrt ((∑ t, a i t) / Fintype.card τ) := by
  rw [Finset.sum_comm, Finset.sum_div]
  exact Finset.sum_le_sum fun i _ => mean_sqrt_le _ (ha i)

theorem mean_sum_sqrt_mul_le {τ ι : Type} [Fintype τ] [Nonempty τ] (s : Finset ι) (c : ι → ℝ)
    (hc : ∀ i, 0 ≤ c i) (a : ι → τ → ℝ) (ha : ∀ i t, 0 ≤ a i t) :
    (∑ t, ∑ i ∈ s, Real.sqrt (c i * a i t)) / Fintype.card τ ≤
      ∑ i ∈ s, Real.sqrt (c i * ((∑ t, a i t) / Fintype.card τ)) := by
  refine (mean_sum_sqrt_le s (fun i t => c i * a i t) fun i t => mul_nonneg (hc i) (ha i t)).trans_eq ?_
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Finset.mul_sum, mul_div_assoc]

theorem mean_add {τ : Type} [Fintype τ] (f g : τ → ℝ) :
    (∑ t, (f t + g t)) / Fintype.card τ = (∑ t, f t) / Fintype.card τ + (∑ t, g t) / Fintype.card τ := by
  rw [Finset.sum_add_distrib, add_div]

theorem mean_const {τ : Type} [Fintype τ] [Nonempty τ] (a : ℝ) : (∑ _t : τ, a) / Fintype.card τ = a := by
  have hn : (0 : ℝ) < Fintype.card τ := Nat.cast_pos.2 Fintype.card_pos
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_div_cancel_left₀ _ hn.ne']

end Claims.Flock

namespace FlockDeploy.KS

open Flock

/-! ## The table error at one table -/

theorem εtable_one {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables zk x = [(spec, st)]) :
    εtable zk x = tableErr zk spec st := by
  unfold εtable
  rw [h, List.map_singleton, List.sum_singleton]

/-- **The old statement's shape is the deployed table's**: table 0's model statement at `regionsOf` has the table's
`k_log` and regions. -/
theorem shape_model {st : Stmt} (L : FlockSoundness.Layout st) (spec : SessionSpec) :
    (L.model (FlockSoundness.Refine.regionsOf st)).shape spec.mPts = acct spec st := by
  simp only [FlockSoundness.Model.Statement.shape, FlockSoundness.Layout.model, FlockSoundness.Refine.regionsOf,
    List.length_map, Array.length_toList, acct]

/-- **The plain table error at one table**: the old `tableError` at table 0's model statement's shape and its
`fast100` schedule is the deployed `ε_table`. -/
theorem εtable_one_plain {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)])
    {sch : FlockSoundness.Accounting.Schedule} (hsch : FlockSoundness.Accounting.fast100 st.m = some sch)
    (L : FlockSoundness.Layout st) :
    FlockSoundness.Accounting.tableError sch ((L.model (FlockSoundness.Refine.regionsOf st)).shape spec.mPts) =
      εtable false x := by
  rw [εtable_one h, shape_model]
  simp only [tableErr, Bool.false_eq_true, ↓reduceIte, hsch, Option.getD_some]

open FlockSoundness.Discharge.ZkSession in
/-- **The masked table error at one table**: the old padded table's bound, at C-Flock ZK's parameters at table 0's
`m` and shape over a field of order `2^128`, is at most the deployed `ε_table`, whose inner proof's error is `innerUp`
(`εInner_le_innerUp`). -/
theorem bound_le_εtable_zk {F : Type} [Field F] [Fintype F] [DecidableEq F] (hF : Fintype.card F = 2 ^ 128)
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables true x = [(spec, st)]) {mPts : ℕ}
    (T : TabZK F mPts) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35)
    (hsch : T.sch = FlockSoundness.Accounting.m1Schedule st.m) (ht : T.t = FlockSoundness.Accounting.m1Pad st.m)
    (he : T.e = 2) (hS : T.S.shape mPts = acct spec st) (hN : T.N ≤ 2 ^ 13) (hq : T.q = 168)
    (hδ : 7 / 16 ≤ T.δ) : T.bound.toReal ≤ εtable true x := by
  rw [εtable_one h]
  have hl0 : 0 ≤ FlockSoundness.Accounting.padLinkError (FlockSoundness.Accounting.m1Schedule st.m) (acct spec st)
      (FlockSoundness.Accounting.m1Pad st.m) := by
    unfold FlockSoundness.Accounting.padLinkError
    positivity
  have hr0 := FlockSoundness.Accounting.m1_rep_nonneg st.m h25 h35 (acct spec st)
  have hi := Claims.Flock.Numbers.innerUp_nonneg
  have hb : εInner T.N F T.q T.δ ≤ ENNReal.ofReal (Claims.Flock.Numbers.innerUp : ℝ) := by
    rw [hq]
    exact Claims.Flock.Numbers.εInner_le_innerUp hF hN hδ
  unfold tableErr TabZK.bound
  simp only [↓reduceIte]
  rw [hS, hsch, ht, he]
  set l := FlockSoundness.Accounting.padLinkError (FlockSoundness.Accounting.m1Schedule st.m) (acct spec st)
    (FlockSoundness.Accounting.m1Pad st.m)
  set r := FlockSoundness.Accounting.padRepError (FlockSoundness.Accounting.m1Schedule st.m) (acct spec st)
    (FlockSoundness.Accounting.m1Pad st.m) 2
  refine ENNReal.toReal_le_of_le_ofReal (add_nonneg hl0 (sq_nonneg _)) ?_
  calc ENNReal.ofReal l + (ENNReal.ofReal r + εInner T.N F T.q T.δ) ^ 2
      ≤ ENNReal.ofReal l + (ENNReal.ofReal r + ENNReal.ofReal (Claims.Flock.Numbers.innerUp : ℝ)) ^ 2 := by
        gcongr
    _ = ENNReal.ofReal (l + (r + (Claims.Flock.Numbers.innerUp : ℝ)) ^ 2) := by
        rw [← ENNReal.ofReal_add hr0 hi, ← ENNReal.ofReal_pow (by positivity),
          ← ENNReal.ofReal_add hl0 (by positivity)]

/-! ## The escapes fit the plain weight -/

/-- The weight's level sum and pads term, in `ℝ`. -/
theorem weightNat_cast (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) :
    (Claims.Flock.Numbers.weightNat sch Np : ℝ) =
      2 * (sch.levels.map fun l => (2 : ℝ) ^ l.logLen * l.queries).sum + 2 * (Np : ℝ) ^ 2 + 2 := by
  unfold Claims.Flock.Numbers.weightNat
  simp [Nat.cast_list_sum, List.map_map, Function.comp_def]

/-- A table's weight is at most the session's, at table 0. -/
theorem tableWeight_le_weight {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables zk x).head? = some (spec, st)) : tableWeight zk st ≤ weight zk x := by
  unfold weight
  cases ht : tables zk x with
  | nil => rw [ht] at h0; cases h0
  | cons p rest =>
    rw [ht, List.head?_cons, Option.some.injEq] at h0
    subst h0
    simp only [List.map_cons, List.sum_cons]
    have : 0 ≤ (rest.map fun p => tableWeight zk p.2).sum :=
      List.sum_nonneg fun a ha => by
        obtain ⟨q, -, rfl⟩ := List.mem_map.1 ha
        exact tableWeight_nonneg zk q.2
    linarith

/-- **Table 0's weight and the other tables' share of `k`** fit in the weight (`lens_le_weights`). -/
theorem weight_split {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt}
    (h0 : (tables zk x).head? = some (spec, st)) :
    tableWeight zk st + 4 * (N₀ zk x - tableLen zk st) / Real.exp 1 ≤ weight zk x := by
  unfold weight N₀
  cases ht : tables zk x with
  | nil => rw [ht] at h0; cases h0
  | cons p rest =>
    rw [ht, List.head?_cons, Option.some.injEq] at h0
    subst h0
    simp only [List.map_cons, List.sum_cons, add_sub_cancel_left]
    have := (lens_le_weights zk rest).1
    have : (0 : ℝ) ≤ rest.length := Nat.cast_nonneg _
    linarith

/-- **The plain weight's room past the old escapes**: the fork events' weights `2·Σ_{ℓ ≥ 1} 2^{d_ℓ}·Q_ℓ` and the fork
of weight `k`, at two fork events a round, leave level 0's `2·2^{d₀}·Q₀` of the weight unused, since `k < 4·N₀/e + 1`
fits in the weight's `2 + 4·2^{d₀}/e` (`kFits_all`): escapes of total weight up to `2·2^{d₀}·Q₀` more, with at most
two fork events a round in all, still fit. -/
theorem plain_fork_room (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) {spec : SessionSpec}
    {st : Stmt} (h0 : (tables false x).head? = some (spec, st)) {sch : FlockSoundness.Accounting.Schedule}
    (hsch : FlockSoundness.Accounting.fast100 st.m = some sch) :
    ((2 : ℕ) : ℝ) * ((deployPlain rel).k x + ∑ i : Fin 2 × Fin sch.levels.length, Claims.Flock.cOf sch (i.2.val + 1)) +
        4 * ((2 : ℝ) ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries) ≤
      2 * (deployPlain rel).weight x := by
  have hgs : (FlockSoundness.Accounting.fast100 st.m).getD default = sch := by rw [hsch]; rfl
  set f : FlockSoundness.Accounting.Level → ℝ := fun l => 2 ^ l.logLen * l.queries with hf
  have hsum : ∑ i : Fin 2 × Fin sch.levels.length, Claims.Flock.cOf sch (i.2.val + 1) =
      2 * ∑ ℓ : Fin sch.levels.length, f (sch.levels.getD (ℓ.val + 1) default) := by
    simp only [Fintype.sum_prod_type, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      Nat.cast_ofNat]
    rfl
  have hd : (default : FlockSoundness.Accounting.Level).queries = 0 := rfl
  have hhead := Claims.Flock.head_add_sum_shift f (by simp [hf, hd]) sch.levels
  have hfh : f (sch.levels.headD default) =
      (2 : ℝ) ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries := rfl
  have htw : tableWeight false st = Claims.Flock.Numbers.weightOf sch 0 := by
    simp only [tableWeight, Bool.false_eq_true, ↓reduceIte, hgs]
  have hlen : tableLen false st = 2 ^ (Claims.Flock.plainLevel0 st.m).d := rfl
  have hfits := kFits_all false st.m
  simp only [level0, wSched, Bool.false_eq_true, ↓reduceIte, hgs] at hfits
  have he := Real.exp_pos 1
  have hpow : (2 : ℝ) ^ (Claims.Flock.plainLevel0 st.m).d ≤ 2 ^ (sch.levels.headD default).logLen :=
    pow_le_pow_right₀ (by norm_num) hfits
  have hdiv : 4 * (2 : ℝ) ^ (Claims.Flock.plainLevel0 st.m).d / Real.exp 1 ≤
      4 * 2 ^ (sch.levels.headD default).logLen / Real.exp 1 := by
    gcongr
  have hceil := Nat.ceil_lt_add_one (div_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 4) (N₀_nonneg false x)) he.le)
  have hk : ((deployPlain rel).k x : ℝ) = (⌈4 * N₀ false x / Real.exp 1⌉₊ : ℝ) := rfl
  have hsplit := weight_split h0
  rw [hlen] at hsplit
  have e : 4 * (N₀ false x - 2 ^ (Claims.Flock.plainLevel0 st.m).d) / Real.exp 1 =
      4 * N₀ false x / Real.exp 1 - 4 * 2 ^ (Claims.Flock.plainLevel0 st.m).d / Real.exp 1 := by ring
  rw [e] at hsplit
  show ((2 : ℕ) : ℝ) * ((deployPlain rel).k x + ∑ i : Fin 2 × Fin sch.levels.length,
    Claims.Flock.cOf sch (i.2.val + 1)) + 4 * ((2 : ℝ) ^ (sch.levels.headD default).logLen *
      (sch.levels.headD default).queries) ≤ 2 * weight false x
  rw [hsum, hk, ← hfh]
  rw [htw, Claims.Flock.Numbers.weightOf, weightNat_cast] at hsplit
  push_cast at hsplit ⊢
  linarith

/-- **The old escapes fit the plain weight with the fork event of weight `k`**, at a statement whose table 0 has
schedule `sch`, when no two of them share a round: the fork events' weights are `2·Σ_{ℓ ≥ 1} 2^{d_ℓ}·Q_ℓ`, within the
weight's `2·Σ_ℓ 2^{d_ℓ}·Q_ℓ`, and `k < 4·N₀/e + 1` within its `2 + 4·2^{d₀}/e` (`kFits_all`). -/
theorem oldEsc_fitsFork_plain (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} {st : Stmt} (h0 : (tables false x).head? = some (spec, st))
    {sch : FlockSoundness.Accounting.Schedule} (hsch : FlockSoundness.Accounting.fast100 st.m = some sch)
    (j : Fin 2 → Fin sch.levels.length → Fin ((deployPlain rel).V.k x))
    (hj : ∀ r ℓ r' ℓ', j r ℓ = j r' ℓ' → r = r' ∧ ℓ = ℓ')
    (ev : Fin 2 → Fin sch.levels.length → Ω → (Fin ((deployPlain rel).V.k x) → Coin) →
      (Fin ((deployPlain rel).V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((deployPlain rel).V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployPlain rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployPlain rel).V x ω (Claims.forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployPlain rel).V x ω s).vLog) :
    (Claims.Flock.oldEsc sha512 (deployPlain rel).V P x sch j ev ev₁ tied tied₁).FitsFork
      ((deployPlain rel).weight x) ((deployPlain rel).k x) := by
  refine ⟨2, fun r => ?_, ?_, ?_⟩
  · have hc : (Finset.univ.filter fun i : Fin 2 × Fin sch.levels.length => j i.1 i.2 = r).card ≤ 1 :=
      Finset.card_le_one.2 fun a ha b hb => by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
        obtain ⟨h1, h2⟩ := hj _ _ _ _ (ha.trans hb.symm)
        exact Prod.ext h1 h2
    have hr : (if r.val = 0 then 1 else 0) ≤ 1 := by split <;> omega
    exact Nat.add_le_add hc hr
  · have hroom := plain_fork_room rel x h0 hsch
    have hf0 : 0 ≤ 4 * ((2 : ℝ) ^ (sch.levels.headD default).logLen * (sch.levels.headD default).queries) := by
      positivity
    show ((2 : ℕ) : ℝ) * ((deployPlain rel).k x + ∑ i : Fin 2 × Fin sch.levels.length,
      Claims.Flock.cOf sch (i.2.val + 1)) ≤ 2 * (deployPlain rel).weight x
    linarith
  · show ∑ _l : Unit, (1 : ℝ) ≤ (V false).k x
    rw [Fintype.sum_unique]
    exact_mod_cast V_k_pos false x

open FlockSoundness.Discharge.Exec in
/-- **The salted session's `ε_c⁻` at one table is at most `ε_table` and the old escapes' term**: its `tableError` at
table 0's model statement is the deployed `ε_table` (`εtable_one_plain`), and its fork and self-clash terms are at most
the term of escapes whose fork and one-run events bound them at the tape (`oldEsc_term`). -/
theorem epsSL_le {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : FlockSoundness.Model.Arith F K) (Hc : List UInt8 → D)
    (E : FlockSoundness.Merkle.Enc (List F) D) (Lf : FlockSoundness.Merkle.Leaf (List F) Sl D Hc) {x : Claims.Msg}
    {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) (L : FlockSoundness.Layout st)
    (m k0 : ℕ) (l₀ l₁ : FlockSoundness.Accounting.Level) (rest : List FlockSoundness.Accounting.Level) (ext : ℕ)
    (hsch : FlockSoundness.Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (FlockSoundness.Refine.regionsOf st)).m) (ptLocal : ℕ)
    (pre post : List ((Fin 2 → Fin spec.mPts → F) → FlockSoundness.Game (FlockSoundness.Model.OutCL F D Sl)))
    (cap₀ : FlockSoundness.Model.Cap D)
    (σ : FlockSoundness.Game.Strategy (sessAfterL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st))
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts pre post cap₀))
    {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (V : Claims.Verifier Coin)
    (P : Claims.Prover Ω) (j : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Fin (V.k x))
    (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl H
      ((Claims.run H P V x ω s).vLog ++ (Claims.run H P V x ω (Claims.forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl H (Claims.run H P V x ω s).vLog) (ω : Ω)
    (hadv : ∀ r ℓ, advRSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm ptLocal
        spec.mPts pre post cap₀ r.val (ℓ.val + 1) σ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev r ℓ ω s.1 s.2)
    (hsc : FlockSoundness.Game.expect (FlockSoundness.Game.ind fun outs : List (FlockSoundness.Model.OutCL F D Sl) =>
        SelfClashL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
      (sessAfterL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal
        spec.mPts pre post cap₀) σ ≤ Claims.Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    epsSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm ptLocal spec.mPts pre
        post cap₀ σ ≤
      εtable false x + (Claims.Flock.oldEsc H V P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ j ev ev₁ tied tied₁).term ω := by
  have key := Claims.Flock.oldEsc_term H V P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ j ev ev₁ tied tied₁ ω
    (fun r ℓ => advRSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm ptLocal
      spec.mPts pre post cap₀ r ℓ σ) _ hadv hsc
  unfold epsSL
  rw [εtable_one_plain h hsch L, add_assoc]
  exact add_le_add le_rfl key

open FlockSoundness.Discharge.Exec in
theorem advRSL_nonneg {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : FlockSoundness.Model.Arith F K) (Hc : List UInt8 → D)
    (E : FlockSoundness.Merkle.Enc (List F) D) (Lf : FlockSoundness.Merkle.Leaf (List F) Sl D Hc)
    (S : FlockSoundness.Model.Statement) (m k0 : ℕ) (l₀ l₁ : FlockSoundness.Accounting.Level)
    (rest : List FlockSoundness.Accounting.Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → FlockSoundness.Game (FlockSoundness.Model.OutCL F D Sl)))
    (cap₀ : FlockSoundness.Model.Cap D) (r ℓ : ℕ)
    (σ : FlockSoundness.Game.Strategy (sessAfterL A Hc E Lf S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post
      cap₀)) :
    0 ≤ advRSL A Hc E Lf S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ := by
  unfold advRSL
  exact FlockSoundness.Game.Lock.forkE_nonneg _ _ (fun _ _ _ => FlockSoundness.Game.ind_nonneg _ _) _ _

open FlockSoundness.Discharge.Exec in
/-- **The salted session's `ε_c⁻`, averaged over a mixture of strategies, at one table**, is at most `ε_table` and the
old escapes' term, when each fork term's mean `Adv` is at most its fork event's probability and the mean self-clash
at most the one-run event's: the mean of each root is at most the root of the mean (`mean_sum_sqrt_mul_le`). -/
theorem mean_epsSL_le {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : FlockSoundness.Model.Arith F K) (Hc : List UInt8 → D)
    (E : FlockSoundness.Merkle.Enc (List F) D) (Lf : FlockSoundness.Merkle.Leaf (List F) Sl D Hc) {x : Claims.Msg}
    {spec : SessionSpec} {st : Stmt} (h : tables false x = [(spec, st)]) (L : FlockSoundness.Layout st)
    (m k0 : ℕ) (l₀ l₁ : FlockSoundness.Accounting.Level) (rest : List FlockSoundness.Accounting.Level) (ext : ℕ)
    (hsch : FlockSoundness.Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ (L.model (FlockSoundness.Refine.regionsOf st)).m) (ptLocal : ℕ)
    (pre post : List ((Fin 2 → Fin spec.mPts → F) → FlockSoundness.Game (FlockSoundness.Model.OutCL F D Sl)))
    (cap₀ : FlockSoundness.Model.Cap D) {τ : Type} [Fintype τ] [Nonempty τ]
    (σ : τ → FlockSoundness.Game.Strategy (sessAfterL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st))
      ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal spec.mPts pre post cap₀))
    {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (V : Claims.Verifier Coin)
    (P : Claims.Prover Ω) (j : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Fin (V.k x))
    (ev : Fin 2 → Fin (l₀ :: l₁ :: rest).length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ r ℓ ω s s', ev r ℓ ω s s' → Claims.TiedColl H
      ((Claims.run H P V x ω s).vLog ++ (Claims.run H P V x ω (Claims.forkCoins (j r ℓ) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl H (Claims.run H P V x ω s).vLog) (ω : Ω)
    (hadv : ∀ r ℓ, (∑ t, advRSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
        ptLocal spec.mPts pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev r ℓ ω s.1 s.2)
    (hsc : (∑ t, FlockSoundness.Game.expect
        (FlockSoundness.Game.ind fun outs : List (FlockSoundness.Model.OutCL F D Sl) =>
          SelfClashL Hc E Lf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (outs.getD pre.length dfltOutL))
        (sessAfterL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm
          ptLocal spec.mPts pre post cap₀) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    (∑ t, epsSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm ptLocal
        spec.mPts pre post cap₀ (σ t)) / Fintype.card τ ≤
      εtable false x + (Claims.Flock.oldEsc H V P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ j ev ev₁ tied tied₁).term ω := by
  have key := Claims.Flock.oldEsc_term H V P x ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ j ev ev₁ tied tied₁ ω
    (fun r ℓ => (∑ t, advRSL A Hc E Lf (L.model (FlockSoundness.Refine.regionsOf st)) m k0 l₀ l₁ rest ext hk0 hm
      ptLocal spec.mPts pre post cap₀ r ℓ (σ t)) / Fintype.card τ) _ hadv hsc
  unfold epsSL
  rw [Claims.Flock.mean_add, Claims.Flock.mean_add, Claims.Flock.mean_const, εtable_one_plain h hsch L, add_assoc]
  refine add_le_add le_rfl (le_trans (add_le_add ?_ le_rfl) key)
  rw [Finset.sum_comm, Finset.sum_div]
  exact Finset.sum_le_sum fun r _ => Claims.Flock.mean_sum_sqrt_mul_le _ _
    (fun ℓ => Claims.Flock.cOf_nonneg _ ℓ) _ fun ℓ t => advRSL_nonneg A Hc E Lf _ m k0 l₀ l₁ rest ext hk0 hm ptLocal
      spec.mPts pre post cap₀ r ℓ (σ t)

/-! ## The masked escapes fit the masked weight -/

/-- C-Flock ZK's level 0 reads at least one position at every `m` it serves. -/
theorem m1_queries₀_pos : ∀ m : Fin 11,
    1 ≤ ((FlockSoundness.Accounting.m1Schedule (m.val + 25)).levels.headD default).queries := by
  decide +kernel

/-- **The old masked escapes fit the masked weight with the fork event of weight `k`**, at a statement whose table 0
has C-Flock ZK's schedule, when no two of them share a round and the pads' weight `N_p` is at most the deployed pads'
domain: the fork events' weights are `2·Σ_{ℓ ≥ 1} 2^{d_ℓ}·Q_ℓ + 2·N_p² + 2`, within the weight's
`2·Σ_ℓ 2^{d_ℓ}·Q_ℓ + 2·N² + 2`, and `k < 4·N₀/e + 1` within `4·2^{d₀}/e` and level 0's `2·2^{d₀}·Q₀ ≥ 2`. -/
theorem oldEscZ_fitsFork_zk (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} {st : Stmt} (h0 : (tables true x).head? = some (spec, st))
    (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) {sch : FlockSoundness.Accounting.Schedule}
    (hsch : sch = FlockSoundness.Accounting.m1Schedule st.m) (Np : ℕ)
    (hNp : Np ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0))
    (j : Claims.Flock.IZ sch.levels.length → Fin ((deployZk rel).V.k x)) (hj : Function.Injective j)
    (ev : Claims.Flock.IZ sch.levels.length → Ω → (Fin ((deployZk rel).V.k x) → Coin) →
      (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployZk rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployZk rel).V x ω (Claims.forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployZk rel).V x ω s).vLog) :
    (Claims.Flock.oldEscZ sha512 (deployZk rel).V P x sch Np j ev ev₁ tied tied₁).FitsFork
      ((deployZk rel).weight x) ((deployZk rel).k x) := by
  refine ⟨2, fun r => ?_, ?_, ?_⟩
  · have hc : (Finset.univ.filter fun i : Claims.Flock.IZ sch.levels.length => j i = r).card ≤ 1 :=
      Finset.card_le_one.2 fun a ha b hb => by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
        exact hj (ha.trans hb.symm)
    have hr : (if r.val = 0 then 1 else 0) ≤ 1 := by split <;> omega
    exact Nat.add_le_add hc hr
  · set f : FlockSoundness.Accounting.Level → ℝ := fun l => 2 ^ l.logLen * l.queries with hf
    have hsum : ∑ i : Claims.Flock.IZ sch.levels.length, Claims.Flock.cZ sch Np i =
        2 * ∑ ℓ : Fin sch.levels.length, f (sch.levels.getD (ℓ.val + 1) default) + 2 * ((Np : ℝ) * Np) + 2 := by
      show ∑ i : (Fin 2 × Fin sch.levels.length) ⊕ (Fin 2 ⊕ Fin 2), Claims.Flock.cZ sch Np i = _
      rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_prod_type]
      simp only [Claims.Flock.cZ, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
        Nat.cast_ofNat, mul_one]
      rw [add_assoc]
      rfl
    have hd : (default : FlockSoundness.Accounting.Level).queries = 0 := rfl
    have hhead := Claims.Flock.head_add_sum_shift f (by simp [hf, hd]) sch.levels
    have hq0 : 1 ≤ (sch.levels.headD default).queries := by
      have := m1_queries₀_pos ⟨st.m - 25, by omega⟩
      simp only [Nat.sub_add_cancel h25] at this
      rw [hsch]
      exact this
    have hf1 : 1 ≤ f (sch.levels.headD default) := by
      simp only [hf]
      have h1 : (1 : ℝ) ≤ 2 ^ (sch.levels.headD default).logLen := one_le_pow₀ (by norm_num)
      have h2 : (1 : ℝ) ≤ (sch.levels.headD default).queries := by exact_mod_cast hq0
      nlinarith
    set pd := Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0)
    have htw : tableWeight true st = Claims.Flock.Numbers.weightOf sch pd := by
      simp only [tableWeight, ↓reduceIte, hsch, pd]
    have hNp2 : (Np : ℝ) * Np ≤ (pd : ℝ) ^ 2 := by
      have := Nat.mul_le_mul hNp hNp
      rw [sq]
      exact_mod_cast this
    have hlen : tableLen true st = 2 ^ (Claims.Flock.zkLevel0 st.m).d := rfl
    have hfits := kFits_all true st.m
    simp only [level0, wSched, ↓reduceIte, ← hsch] at hfits
    have he := Real.exp_pos 1
    have hpow : (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d ≤ 2 ^ (sch.levels.headD default).logLen :=
      pow_le_pow_right₀ (by norm_num) hfits
    have hdiv : 4 * (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d / Real.exp 1 ≤
        4 * 2 ^ (sch.levels.headD default).logLen / Real.exp 1 := by
      gcongr
    have hceil := Nat.ceil_lt_add_one (div_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 4) (N₀_nonneg true x)) he.le)
    have hk : ((deployZk rel).k x : ℝ) = (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ) := rfl
    have hsplit := weight_split h0
    rw [hlen] at hsplit
    have e : 4 * (N₀ true x - 2 ^ (Claims.Flock.zkLevel0 st.m).d) / Real.exp 1 =
        4 * N₀ true x / Real.exp 1 - 4 * 2 ^ (Claims.Flock.zkLevel0 st.m).d / Real.exp 1 := by ring
    rw [e] at hsplit
    show ((2 : ℕ) : ℝ) * ((deployZk rel).k x + ∑ i : Claims.Flock.IZ sch.levels.length, Claims.Flock.cZ sch Np i) ≤
      2 * weight true x
    rw [hsum, hk]
    rw [htw, Claims.Flock.Numbers.weightOf, weightNat_cast] at hsplit
    push_cast at hsplit ⊢
    linarith
  · show ∑ _l : Unit, (1 : ℝ) ≤ (V true).k x
    rw [Fintype.sum_unique]
    exact_mod_cast V_k_pos true x

open FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession in
/-- **The masked session's `ε_c⁻` at one table is at most `ε_table` and the old masked escapes' term**: its padded
table's bound is at most the deployed `ε_table` (`bound_le_εtable_zk`), and its fork terms (levels, pads roots, `τ`)
and self-clash are at most the term of escapes, at the table's schedule and pads' domain, whose fork and one-run
events bound them at the tape (`oldEscZ_term`). -/
theorem epsSZ_le {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (hF : Fintype.card F = 2 ^ 128) {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : FlockSoundness.Model.Arith F K)
    (Hs : Hashes F D Sl) {mPts : ℕ} (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → FlockSoundness.Game (OutZC F D Sl)))
    (cap₀ : FlockSoundness.Model.Cap D) (σ : FlockSoundness.Game.Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables true x = [(spec, st)]) (h25 : 25 ≤ st.m)
    (h35 : st.m ≤ 35) (hsch : T.sch = FlockSoundness.Accounting.m1Schedule st.m)
    (ht : T.t = FlockSoundness.Accounting.m1Pad st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st)
    (hN : T.N ≤ 2 ^ 13) (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ)
    {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (V : Claims.Verifier Coin)
    (P : Claims.Prover Ω) (j : Claims.Flock.IZ T.sch.levels.length → Fin (V.k x))
    (ev : Claims.Flock.IZ T.sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl H
      ((Claims.run H P V x ω s).vLog ++ (Claims.run H P V x ω (Claims.forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl H (Claims.run H P V x ω s).vLog) (ω : Ω)
    (hadv : ∀ r ℓ, advRSZ A Hs T pre post cap₀ r.val (ℓ.val + 1) σ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inl (r, ℓ)) ω s.1 s.2)
    (hP : ∀ r : Fin 2, advPadsSZ A Hs T pre post cap₀ r.val σ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inr (.inl r)) ω s.1 s.2)
    (hT : ∀ r : Fin 2, advTauSZ A Hs T pre post cap₀ r.val σ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inr (.inr r)) ω s.1 s.2)
    (hsc : FlockSoundness.Game.expect (FlockSoundness.Game.ind fun outs : List (OutZC F D Sl) =>
        SelfClashZ Hs T (outs.getD pre.length dfltOutZC)) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) σ ≤
      Claims.Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    epsSZ A Hs T pre post cap₀ σ ≤
      εtable true x + (Claims.Flock.oldEscZ H V P x T.sch T.N j ev ev₁ tied tied₁).term ω := by
  have key := Claims.Flock.oldEscZ_term H V P x T.sch T.N j ev ev₁ tied tied₁ ω
    (fun r ℓ => advRSZ A Hs T pre post cap₀ r ℓ σ) (fun r => advPadsSZ A Hs T pre post cap₀ r σ)
    (fun r => advTauSZ A Hs T pre post cap₀ r σ) _ hadv hP hT hsc
  have hb := bound_le_εtable_zk hF h T h25 h35 hsch ht he hS hN hq hδ
  unfold epsSZ
  rw [add_assoc, add_assoc, add_assoc]
  exact add_le_add hb (by simpa only [add_assoc] using key)

open FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession in
theorem advSZ_nonneg {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : FlockSoundness.Model.Arith F K) (Hs : Hashes F D Sl) {mPts : ℕ}
    (T : TabZK F mPts) (pre post : List ((Fin 2 → Fin mPts → F) → FlockSoundness.Game (OutZC F D Sl)))
    (cap₀ : FlockSoundness.Model.Cap D) (r ℓ : ℕ)
    (σ : FlockSoundness.Game.Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀))) :
    0 ≤ advRSZ A Hs T pre post cap₀ r ℓ σ ∧ 0 ≤ advPadsSZ A Hs T pre post cap₀ r σ ∧
      0 ≤ advTauSZ A Hs T pre post cap₀ r σ :=
  ⟨FlockSoundness.Game.Lock.forkE_nonneg _ _ (fun _ _ _ => FlockSoundness.Game.ind_nonneg _ _) _ _,
    FlockSoundness.Game.Lock.forkE_nonneg _ _ (fun _ _ _ => FlockSoundness.Game.ind_nonneg _ _) _ _,
    FlockSoundness.Game.Lock.forkE_nonneg _ _ (fun _ _ _ => FlockSoundness.Game.ind_nonneg _ _) _ _⟩

open FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession in
/-- **The masked session's `ε_c⁻`, averaged over a mixture of strategies, at one table**, is at most `ε_table` and
the old masked escapes' term, when each fork term's mean `Adv` (a level's, a pads root's, `τ`'s) is at most its fork
event's probability and the mean self-clash at most the one-run event's. -/
theorem mean_epsSZ_le {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (hF : Fintype.card F = 2 ^ 128) {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : FlockSoundness.Model.Arith F K)
    (Hs : Hashes F D Sl) {mPts : ℕ} (T : TabZK F mPts)
    (pre post : List ((Fin 2 → Fin mPts → F) → FlockSoundness.Game (OutZC F D Sl)))
    (cap₀ : FlockSoundness.Model.Cap D) {τ : Type} [Fintype τ] [Nonempty τ]
    (σ : τ → FlockSoundness.Game.Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} (h : tables true x = [(spec, st)]) (h25 : 25 ≤ st.m)
    (h35 : st.m ≤ 35) (hsch : T.sch = FlockSoundness.Accounting.m1Schedule st.m)
    (ht : T.t = FlockSoundness.Accounting.m1Pad st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st)
    (hN : T.N ≤ 2 ^ 13) (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ)
    {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (V : Claims.Verifier Coin)
    (P : Claims.Prover Ω) (j : Claims.Flock.IZ T.sch.levels.length → Fin (V.k x))
    (ev : Claims.Flock.IZ T.sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl H
      ((Claims.run H P V x ω s).vLog ++ (Claims.run H P V x ω (Claims.forkCoins (j i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl H (Claims.run H P V x ω s).vLog) (ω : Ω)
    (hadv : ∀ r ℓ, (∑ t, advRSZ A Hs T pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inl (r, ℓ)) ω s.1 s.2)
    (hP : ∀ r : Fin 2, (∑ t, advPadsSZ A Hs T pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inr (.inl r)) ω s.1 s.2)
    (hT : ∀ r : Fin 2, (∑ t, advTauSZ A Hs T pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inr (.inr r)) ω s.1 s.2)
    (hsc : (∑ t, FlockSoundness.Game.expect (FlockSoundness.Game.ind fun outs : List (OutZC F D Sl) =>
        SelfClashZ Hs T (outs.getD pre.length dfltOutZC)) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) (σ t)) /
        Fintype.card τ ≤ Claims.Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    (∑ t, epsSZ A Hs T pre post cap₀ (σ t)) / Fintype.card τ ≤
      εtable true x + (Claims.Flock.oldEscZ H V P x T.sch T.N j ev ev₁ tied tied₁).term ω := by
  have key := Claims.Flock.oldEscZ_term H V P x T.sch T.N j ev ev₁ tied tied₁ ω
    (fun r ℓ => (∑ t, advRSZ A Hs T pre post cap₀ r ℓ (σ t)) / Fintype.card τ)
    (fun r => (∑ t, advPadsSZ A Hs T pre post cap₀ r (σ t)) / Fintype.card τ)
    (fun r => (∑ t, advTauSZ A Hs T pre post cap₀ r (σ t)) / Fintype.card τ) _ hadv hP hT hsc
  have hb := bound_le_εtable_zk hF h T h25 h35 hsch ht he hS hN hq hδ
  unfold epsSZ
  rw [Claims.Flock.mean_add, Claims.Flock.mean_add, Claims.Flock.mean_add, Claims.Flock.mean_add,
    Claims.Flock.mean_const, add_assoc, add_assoc, add_assoc]
  simp only [add_assoc] at key
  refine add_le_add hb (le_trans (add_le_add ?_ (add_le_add ?_ (add_le_add ?_ le_rfl))) key)
  · rw [Finset.sum_comm, Finset.sum_div]
    exact Finset.sum_le_sum fun r _ => Claims.Flock.mean_sum_sqrt_mul_le _ _
      (fun ℓ => Claims.Flock.cOf_nonneg _ ℓ) _ fun ℓ t => (advSZ_nonneg A Hs T pre post cap₀ r ℓ (σ t)).1
  · exact Claims.Flock.mean_sum_sqrt_mul_le _ (fun _ => (T.N : ℝ) * T.N)
      (fun _ => mul_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)) _
      fun r t => (advSZ_nonneg A Hs T pre post cap₀ r 0 (σ t)).2.1
  · exact Claims.Flock.mean_sum_sqrt_le _ _ fun r t => (advSZ_nonneg A Hs T pre post cap₀ r 0 (σ t)).2.2

end FlockDeploy.KS
