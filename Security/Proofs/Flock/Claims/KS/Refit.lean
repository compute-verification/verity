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
  event per round they fit the plain weight with the fork event of weight `k` (`oldEsc_fitsFork_plain`), and the
  salted session's `ε_c⁻` is at most `ε_table` plus their term (`epsSL_le`).
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
      ∃ u, lookCol (D.cols (run H P D.V x ω r).seen) p.val = some u ∧ (fun lane : Fin w => u.getD lane.val 0) = v) :
    (Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
      ∃ p, (op r p).isSome ∧ op r p ≠ some fun lane : Fin w => oracleOf C lane.val p.val) ≤
    Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
      ∃ (p : Fin (N x)) (u : List F), lookCol (D.cols (run H P D.V x ω r).seen) p.val = some u ∧ u ≠ C p :=
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

/-- The weight's level sum, in `ℝ`. -/
theorem weightNat_cast (sch : FlockSoundness.Accounting.Schedule) :
    (Claims.Flock.Numbers.weightNat sch 0 : ℝ) =
      2 * (sch.levels.map fun l => (2 : ℝ) ^ l.logLen * l.queries).sum + 2 := by
  unfold Claims.Flock.Numbers.weightNat
  simp [Nat.cast_list_sum, List.map_map, Function.comp_def]

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
  · have hgs : (FlockSoundness.Accounting.fast100 st.m).getD default = sch := by rw [hsch]; rfl
    set f : FlockSoundness.Accounting.Level → ℝ := fun l => 2 ^ l.logLen * l.queries with hf
    have hsum : ∑ i : Fin 2 × Fin sch.levels.length, Claims.Flock.cOf sch (i.2.val + 1) =
        2 * ∑ ℓ : Fin sch.levels.length, f (sch.levels.getD (ℓ.val + 1) default) := by
      simp only [Fintype.sum_prod_type, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
        Nat.cast_ofNat]
      rfl
    have hd : (default : FlockSoundness.Accounting.Level).queries = 0 := rfl
    have hhead := Claims.Flock.head_add_sum_shift f (by simp [hf, hd]) sch.levels
    have hf0 : 0 ≤ f (sch.levels.headD default) := by simp only [hf]; positivity
    have hw : tableWeight false st ≤ weight false x := by
      unfold weight
      cases ht : tables false x with
      | nil => rw [ht] at h0; cases h0
      | cons p rest =>
        rw [ht, List.head?_cons, Option.some.injEq] at h0
        subst h0
        simp only [List.map_cons, List.sum_cons]
        have : 0 ≤ (rest.map fun p => tableWeight false p.2).sum :=
          List.sum_nonneg fun a ha => by
            obtain ⟨q, -, rfl⟩ := List.mem_map.1 ha
            exact tableWeight_nonneg false q.2
        linarith
    have htw : tableWeight false st = Claims.Flock.Numbers.weightOf sch 0 := by
      simp only [tableWeight, Bool.false_eq_true, ↓reduceIte, hgs]
    have hN : N₀ false x = 2 ^ (Claims.Flock.plainLevel0 st.m).d := by
      unfold N₀
      rw [h0]
      rfl
    have hfits := kFits_all false st.m
    simp only [level0, wSched, Bool.false_eq_true, ↓reduceIte, hgs] at hfits
    have he := Real.exp_pos 1
    have hpow : (2 : ℝ) ^ (Claims.Flock.plainLevel0 st.m).d ≤ 2 ^ (sch.levels.headD default).logLen :=
      pow_le_pow_right₀ (by norm_num) hfits
    have hdiv : 4 * (2 : ℝ) ^ (Claims.Flock.plainLevel0 st.m).d / Real.exp 1 ≤
        4 * 2 ^ (sch.levels.headD default).logLen / Real.exp 1 := by
      gcongr
    have hceil := Nat.ceil_lt_add_one (show (0 : ℝ) ≤ 4 * N₀ false x / Real.exp 1 by rw [hN]; positivity)
    rw [hN] at hceil
    have hk : ((deployPlain rel).k x : ℝ) = (⌈4 * N₀ false x / Real.exp 1⌉₊ : ℝ) := rfl
    rw [hN] at hk
    show ((2 : ℕ) : ℝ) * ((deployPlain rel).k x + ∑ i : Fin 2 × Fin sch.levels.length,
      Claims.Flock.cOf sch (i.2.val + 1)) ≤ 2 * weight false x
    rw [hsum, hk]
    rw [htw, Claims.Flock.Numbers.weightOf, weightNat_cast] at hw
    push_cast
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

end FlockDeploy.KS
