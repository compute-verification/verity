import Proofs.Flock.Claims.KS.ZkTables
import Proofs.Flock.Claims.KS.ZkKS

/-!
# Every masked table's bound from masked sessions after `Commit`

`ZkBound` at every table of the masked record. The tables are batched: one `Commit`, one link-point draw, then every
table's reps at once, so the tables' old fork events share rounds. Escapes concatenated table by table would put as
many fork events at one round as there are tables, more than the weight pays for; joined by round (`joinEscZ`: at
round `r`, the weight of every table's fork event there, and the event that one of them happens) there is one per
round, and the weights sum to every table's old ones (`joinEscZ_c_sum`). Each table's weight pays for its old fork
events and its share of `k` (`cZ_sum_le_tableWeight`), so the joined escapes fit the deployed weight
(`joinEscZ_fitsFork_zk`), and every table's old escapes' term is at most theirs (`oldEscZ_term_le_join`).

At table `j`, everything `ZkBound` proves at table 0 holds as it is: the old masked session theorems speak of the table
at position `pre.length` of a batch after `Commit` (`session_sound_of_table_accZ`, `epsSZ`), the old padded table's
bound is at most table `j`'s error (`bound_le_tableErr_zk`), which is at most `ε_table` (`tableErr_le_εtable`), and a
table of witness-lane columns that does not extract table `j` is not committed to table `j`'s model statement
(`zk_not_committedJ`), since a satisfying old witness of it reads as a witness of table `j`'s own statement
(`relS_tableJ`).
-/

namespace Claims.Flock

/-! ## Escapes joined by round -/

section Join

variable {Ω Coin : Type} [Inhabited Coin] (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)

/-- **The old masked escapes of `n` tables, joined by round**: at round `r`, the weight of every table's old fork
events there, and the event that one of them happens; one one-run event, that some table's happens. -/
noncomputable def joinEscZ (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZ (sch j).levels.length → Fin (V.k x))
    (ev : (j : ℕ) → IZ (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog) : Escapes H V P x where
  I := Fin (V.k x)
  c r := ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r), cZ (sch j) (Np j) i
  j r := r
  ev r ω s s' := ∃ j < n, ∃ i, rd j i = r ∧ ev j i ω s s'
  L := Unit
  β _ := 1
  ev₁ _ ω s := ∃ j < n, ev₁ j ω s
  c_nonneg _ := Finset.sum_nonneg fun j _ => Finset.sum_nonneg fun i _ => cZ_nonneg _ _ i
  β_nonneg _ := zero_le_one
  tied r ω s s' := by
    rintro ⟨j, -, i, rfl, h⟩
    exact tied j i ω s s' h
  tied₁ _ ω s := by
    rintro ⟨j, -, h⟩
    exact tied₁ j ω s h

variable {H V P x}

/-- **The joined weights are every table's old ones.** -/
theorem joinEscZ_c_sum (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZ (sch j).levels.length → Fin (V.k x))
    (ev : (j : ℕ) → IZ (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog) :
    ∑ r, (joinEscZ H V P x n sch Np rd ev ev₁ tied tied₁).c r =
      ∑ j ∈ Finset.range n, ∑ i, cZ (sch j) (Np j) i := by
  show ∑ r : Fin (V.k x), ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r),
    cZ (sch j) (Np j) i = _
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun j _ => Finset.sum_fiberwise _ _ _

/-- **The joined escapes fit a weight** that pays for every table's old fork events and the fork event of weight
`c`: one fork event per round, two at round `0` with the extra one. -/
theorem joinEscZ_fitsFork (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZ (sch j).levels.length → Fin (V.k x))
    (ev : (j : ℕ) → IZ (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog) (C c : ℝ)
    (h : c + ∑ j ∈ Finset.range n, ∑ i, cZ (sch j) (Np j) i ≤ C) (hk : 1 ≤ V.k x) :
    (joinEscZ H V P x n sch Np rd ev ev₁ tied tied₁).FitsFork C c := by
  refine ⟨2, fun r => ?_, ?_, ?_⟩
  · have hc : (Finset.univ.filter fun i : Fin (V.k x) => i = r).card ≤ 1 :=
      Finset.card_le_one.2 fun a ha b hb => by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
        rw [ha, hb]
    have hr : (if r.val = 0 then 1 else 0) ≤ 1 := by split <;> omega
    exact Nat.add_le_add hc hr
  · rw [joinEscZ_c_sum]
    push_cast
    linarith
  · show ∑ _l : Unit, (1 : ℝ) ≤ V.k x
    rw [Fintype.sum_unique]
    exact_mod_cast hk

/-- **Each table's old escapes' term is at most the joined escapes'**, at a table of the statement whose fork events
are at distinct rounds. -/
theorem oldEscZ_term_le_join [Fintype Coin] (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZ (sch j).levels.length → Fin (V.k x))
    (ev : (j : ℕ) → IZ (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog) {j : ℕ} (hj : j < n)
    (hinj : Function.Injective (rd j)) (ω : Ω) :
    (oldEscZ H V P x (sch j) (Np j) (rd j) (ev j) (ev₁ j) (tied j) (tied₁ j)).term ω ≤
      (joinEscZ H V P x n sch Np rd ev ev₁ tied tied₁).term ω := by
  set c : Fin (V.k x) → ℝ := fun r =>
    ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r), cZ (sch j) (Np j) i with hc
  set g : Fin (V.k x) → ℝ := fun r => Real.sqrt (c r *
    Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ∃ j < n, ∃ i, rd j i = r ∧ ev j i ω s.1 s.2) with hg
  show (∑ i : IZ (sch j).levels.length, Real.sqrt (cZ (sch j) (Np j) i *
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2)) +
    ∑ _l : Unit, (1 : ℝ) * (Pr fun s : Fin (V.k x) → Coin => ev₁ j ω s) ≤
    (∑ r : Fin (V.k x), g r) + ∑ _l : Unit, (1 : ℝ) * (Pr fun s : Fin (V.k x) → Coin => ∃ j < n, ev₁ j ω s)
  refine add_le_add ?_ (Finset.sum_le_sum fun _ _ =>
    mul_le_mul_of_nonneg_left (Pr_mono fun s h => ⟨j, hj, h⟩) zero_le_one)
  have hle : ∀ i, Real.sqrt (cZ (sch j) (Np j) i *
      Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2) ≤ g (rd j i) := by
    intro i
    have hci : cZ (sch j) (Np j) i ≤ c (rd j i) := by
      have h1 : cZ (sch j) (Np j) i ≤
          ∑ i' ∈ Finset.univ.filter (fun i' => rd j i' = rd j i), cZ (sch j) (Np j) i' :=
        Finset.single_le_sum (f := cZ (sch j) (Np j)) (fun i' _ => cZ_nonneg _ _ i')
          (by simp only [Finset.mem_filter, Finset.mem_univ, true_and])
      have h2 : (∑ i' ∈ Finset.univ.filter (fun i' => rd j i' = rd j i), cZ (sch j) (Np j) i') ≤ c (rd j i) :=
        Finset.single_le_sum (f := fun j' => ∑ i' ∈ Finset.univ.filter (fun i' => rd j' i' = rd j i),
            cZ (sch j') (Np j') i')
          (fun j' _ => Finset.sum_nonneg fun i' _ => cZ_nonneg _ _ i') (Finset.mem_range.2 hj)
      exact h1.trans h2
    have hpi : (Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2) ≤
        Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) =>
          ∃ j' < n, ∃ i', rd j' i' = rd j i ∧ ev j' i' ω s.1 s.2 :=
      Pr_mono fun s h => ⟨j, hj, i, rfl, h⟩
    exact Real.sqrt_le_sqrt (mul_le_mul hci hpi (Pr_nonneg _)
      ((cZ_nonneg _ _ i).trans hci))
  calc (∑ i : IZ (sch j).levels.length, Real.sqrt (cZ (sch j) (Np j) i *
          Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2))
      ≤ ∑ i : IZ (sch j).levels.length, g (rd j i) := Finset.sum_le_sum fun i _ => hle i
    _ = ∑ r ∈ Finset.univ.image (rd j), g r := (Finset.sum_image fun a _ b _ h => hinj h).symm
    _ ≤ ∑ r, g r := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        fun _ _ _ => Real.sqrt_nonneg _

end Join

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

/-! ## The table error at table `j` -/

/-- **The masked table error at a table**: the old padded table's bound, at C-Flock ZK's parameters at the table's `m`
and shape over a field of order `2^128`, is at most the table's deployed error (`bound_le_εtable_zk` at any table). -/
theorem bound_le_tableErr_zk {F : Type} [Field F] [Fintype F] [DecidableEq F] (hF : Fintype.card F = 2 ^ 128)
    {spec : SessionSpec} {st : Stmt} {mPts : ℕ} (T : TabZK F mPts) (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35)
    (hsch : T.sch = Accounting.m1Schedule st.m) (ht : T.t = Accounting.m1Pad st.m) (he : T.e = 2)
    (hS : T.S.shape mPts = acct spec st) (hN : T.N ≤ 2 ^ 13) (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ) :
    T.bound.toReal ≤ tableErr true spec st := by
  have hl0 : 0 ≤ Accounting.padLinkError (Accounting.m1Schedule st.m) (acct spec st) (Accounting.m1Pad st.m) := by
    unfold Accounting.padLinkError
    positivity
  have hr0 := Accounting.m1_rep_nonneg st.m h25 h35 (acct spec st)
  have hi := Claims.Flock.Numbers.innerUp_nonneg
  have hb : εInner T.N F T.q T.δ ≤ ENNReal.ofReal (Claims.Flock.Numbers.innerUp : ℝ) := by
    rw [hq]
    exact Claims.Flock.Numbers.εInner_le_innerUp hF hN hδ
  unfold tableErr TabZK.bound
  simp only [↓reduceIte]
  rw [hS, hsch, ht, he]
  set l := Accounting.padLinkError (Accounting.m1Schedule st.m) (acct spec st) (Accounting.m1Pad st.m)
  set r := Accounting.padRepError (Accounting.m1Schedule st.m) (acct spec st) (Accounting.m1Pad st.m) 2
  refine ENNReal.toReal_le_of_le_ofReal (add_nonneg hl0 (sq_nonneg _)) ?_
  calc ENNReal.ofReal l + (ENNReal.ofReal r + εInner T.N F T.q T.δ) ^ 2
      ≤ ENNReal.ofReal l + (ENNReal.ofReal r + ENNReal.ofReal (Claims.Flock.Numbers.innerUp : ℝ)) ^ 2 := by
        gcongr
    _ = ENNReal.ofReal (l + (r + (Claims.Flock.Numbers.innerUp : ℝ)) ^ 2) := by
        rw [← ENNReal.ofReal_add hr0 hi, ← ENNReal.ofReal_pow (by positivity),
          ← ENNReal.ofReal_add hl0 (by positivity)]

/-- **A table's error is at most `ε_table`**, the tables' errors summed. -/
theorem tableErr_le_εtable {zk : Bool} {x : Claims.Msg} {spec : SessionSpec} {st : Stmt} {j : ℕ}
    (h : (tables zk x)[j]? = some (spec, st)) : tableErr zk spec st ≤ εtable zk x := by
  have hm : tableErr zk spec st ∈ (tables zk x).map fun p => tableErr zk p.1 p.2 :=
    List.mem_map.2 ⟨(spec, st), List.mem_of_getElem? h, rfl⟩
  have hnn : ∀ a ∈ (tables zk x).map fun p => tableErr zk p.1 p.2, 0 ≤ a := fun a ha => by
    obtain ⟨p, -, rfl⟩ := List.mem_map.1 ha
    exact tableErr_nonneg zk p.1 p.2
  exact List.single_le_sum hnn _ hm

/-- **The masked session's `ε_c⁻`, averaged over a mixture of strategies, at table `j`**, is at most `ε_table` and
the old masked escapes' term (`mean_epsSZ_le` at any table). -/
theorem mean_epsSZ_leJ {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (hF : Fintype.card F = 2 ^ 128) {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : Model.Arith F K)
    (Hs : Hashes F D Sl) {mPts : ℕ} (T : TabZK F mPts) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (cap₀ : Model.Cap D) {τ : Type} [Fintype τ] [Nonempty τ]
    (σ : τ → Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    {x : Claims.Msg} {j : ℕ} {spec : SessionSpec} {st : Stmt} (h : (tables true x)[j]? = some (spec, st))
    (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hsch : T.sch = Accounting.m1Schedule st.m)
    (ht : T.t = Accounting.m1Pad st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st)
    (hN : T.N ≤ 2 ^ 13) (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ)
    {Ω Coin : Type} [Fintype Coin] [Inhabited Coin] (H : Claims.Msg → Claims.Msg) (V : Claims.Verifier Coin)
    (P : Claims.Prover Ω) (rd : Claims.Flock.IZ T.sch.levels.length → Fin (V.k x))
    (ev : Claims.Flock.IZ T.sch.levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ i ω s s', ev i ω s s' → Claims.TiedColl H
      ((Claims.run H P V x ω s).vLog ++ (Claims.run H P V x ω (Claims.forkCoins (rd i) s s')).vLog))
    (tied₁ : ∀ ω s, ev₁ ω s → Claims.TiedColl H (Claims.run H P V x ω s).vLog) (ω : Ω)
    (hadv : ∀ r ℓ, (∑ t, advRSZ A Hs T pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inl (r, ℓ)) ω s.1 s.2)
    (hP : ∀ r : Fin 2, (∑ t, advPadsSZ A Hs T pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inr (.inl r)) ω s.1 s.2)
    (hT : ∀ r : Fin 2, (∑ t, advTauSZ A Hs T pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev (.inr (.inr r)) ω s.1 s.2)
    (hsc : (∑ t, expect (ind fun outs : List (OutZC F D Sl) =>
        SelfClashZ Hs T (outs.getD pre.length dfltOutZC)) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) (σ t)) /
        Fintype.card τ ≤ Claims.Pr fun s : Fin (V.k x) → Coin => ev₁ ω s) :
    (∑ t, epsSZ A Hs T pre post cap₀ (σ t)) / Fintype.card τ ≤
      εtable true x + (Claims.Flock.oldEscZ H V P x T.sch T.N rd ev ev₁ tied tied₁).term ω := by
  have key := Claims.Flock.oldEscZ_term H V P x T.sch T.N rd ev ev₁ tied tied₁ ω
    (fun r ℓ => (∑ t, advRSZ A Hs T pre post cap₀ r ℓ (σ t)) / Fintype.card τ)
    (fun r => (∑ t, advPadsSZ A Hs T pre post cap₀ r (σ t)) / Fintype.card τ)
    (fun r => (∑ t, advTauSZ A Hs T pre post cap₀ r (σ t)) / Fintype.card τ) _ hadv hP hT hsc
  have hb := (bound_le_tableErr_zk hF T h25 h35 hsch ht he hS hN hq hδ).trans (tableErr_le_εtable h)
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

/-! ## The joined escapes fit the masked weight -/

/-- A list's sum, read through a family that gives each position's entry. -/
theorem sum_map_eq_range {α : Type} (l : List α) (g : ℕ → α) (hg : ∀ j < l.length, l[j]? = some (g j))
    (f : α → ℝ) : (l.map f).sum = ∑ j ∈ Finset.range l.length, f (g j) := by
  rw [← Fin.sum_univ_fun_getElem l f, Finset.sum_range]
  refine Finset.sum_congr rfl fun j _ => ?_
  have h := hg j j.isLt
  rw [List.getElem?_eq_getElem j.isLt, Option.some.injEq] at h
  rw [h]

/-- **A table's weight pays for its old masked fork events and its share of `k`**, with `2` to spare: the fork events'
weights are `2·Σ_{ℓ ≥ 1} 2^{d_ℓ}·Q_ℓ + 2·N_p² + 2`, within the table's `2·Σ_ℓ 2^{d_ℓ}·Q_ℓ + 2·N² + 2 + 4·2^{d₀}/e`,
when `N_p` is at most the deployed pads' domain; level 0's `2·2^{d₀}·Q₀ ≥ 2` is the `2`, and `4·2^{d₀}/e` the
share of `k` (`kFits_all`). -/
theorem cZ_sum_le_tableWeight {st : Stmt} (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35)
    {sch : Accounting.Schedule} (hsch : sch = Accounting.m1Schedule st.m) (Np : ℕ)
    (hNp : Np ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new st.m st.c.kLog (Claims.Flock.zkSched st.m).1.k0)) :
    4 * tableLen true st / Real.exp 1 + ∑ i : Claims.Flock.IZ sch.levels.length, Claims.Flock.cZ sch Np i + 2 ≤
      tableWeight true st := by
  set f : Accounting.Level → ℝ := fun l => 2 ^ l.logLen * l.queries with hf
  have hsum : ∑ i : Claims.Flock.IZ sch.levels.length, Claims.Flock.cZ sch Np i =
      2 * ∑ ℓ : Fin sch.levels.length, f (sch.levels.getD (ℓ.val + 1) default) + 2 * ((Np : ℝ) * Np) + 2 := by
    show ∑ i : (Fin 2 × Fin sch.levels.length) ⊕ (Fin 2 ⊕ Fin 2), Claims.Flock.cZ sch Np i = _
    rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_prod_type]
    simp only [Claims.Flock.cZ, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      Nat.cast_ofNat, mul_one]
    rw [add_assoc]
    rfl
  have hd : (default : Accounting.Level).queries = 0 := rfl
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
  have hpow : (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d ≤ 2 ^ (sch.levels.headD default).logLen :=
    pow_le_pow_right₀ (by norm_num) hfits
  have hdiv : 4 * (2 : ℝ) ^ (Claims.Flock.zkLevel0 st.m).d / Real.exp 1 ≤
      4 * 2 ^ (sch.levels.headD default).logLen / Real.exp 1 := by
    gcongr
  rw [hsum, hlen, htw, Claims.Flock.Numbers.weightOf, weightNat_cast, ← hhead]
  linarith

/-- **The joined escapes fit the masked weight with the fork event of weight `k`**, at a statement whose tables
`st j` are C-Flock ZK's, when each table's pads' weight is at most its deployed pads' domain: each table's weight
pays for its fork events and its share of `k` with `2` to spare (`cZ_sum_le_tableWeight`), and
`k < 4·N₀/e + 1`. -/
theorem joinEscZ_fitsFork_zk (rel : Readable.Parsed → Readable.Relation) {Ω : Type} (P : Claims.Prover Ω)
    (x : Claims.Msg) {spec : SessionSpec} (st : ℕ → Stmt)
    (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j))
    (h25 : ∀ j < (tables true x).length, 25 ≤ (st j).m) (h35 : ∀ j < (tables true x).length, (st j).m ≤ 35)
    (sch : ℕ → Accounting.Schedule) (hsch : ∀ j < (tables true x).length, sch j = Accounting.m1Schedule (st j).m)
    (Np : ℕ → ℕ) (hNp : ∀ j < (tables true x).length,
      Np j ≤ Claims.Flock.Numbers.padDomain (Zk.PadLayout.new (st j).m (st j).c.kLog
        (Claims.Flock.zkSched (st j).m).1.k0))
    (rd : (j : ℕ) → Claims.Flock.IZ (sch j).levels.length → Fin ((deployZk rel).V.k x))
    (ev : (j : ℕ) → Claims.Flock.IZ (sch j).levels.length → Ω → (Fin ((deployZk rel).V.k x) → Coin) →
      (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin ((deployZk rel).V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (deployZk rel).V x ω s).vLog ++
        (Claims.run sha512 P (deployZk rel).V x ω (Claims.forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (deployZk rel).V x ω s).vLog) :
    (Claims.Flock.joinEscZ sha512 (deployZk rel).V P x (tables true x).length sch Np rd ev ev₁ tied tied₁).FitsFork
      ((deployZk rel).weight x) ((deployZk rel).k x) := by
  refine Claims.Flock.joinEscZ_fitsFork _ _ _ _ _ _ _ _ _ _ ?_ (V_k_pos true x)
  set n := (tables true x).length with hn
  have hw : weight true x = ∑ j ∈ Finset.range n, tableWeight true (st j) :=
    sum_map_eq_range (tables true x) (fun j => (spec, st j)) hst _
  have hN : N₀ true x = ∑ j ∈ Finset.range n, tableLen true (st j) :=
    sum_map_eq_range (tables true x) (fun j => (spec, st j)) hst _
  have hk : ((deployZk rel).k x : ℝ) = (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ) := rfl
  show ((deployZk rel).k x : ℝ) + _ ≤ weight true x
  rcases Nat.eq_zero_or_pos n with h0 | hpos
  · have := k_le_weight true x
    rw [h0, Finset.sum_range_zero, add_zero, hk]
    exact this
  have hper : ∀ j ∈ Finset.range n, 4 * tableLen true (st j) / Real.exp 1 +
      ∑ i, Claims.Flock.cZ (sch j) (Np j) i + 2 ≤ tableWeight true (st j) := fun j hj => by
    have hj := Finset.mem_range.1 hj
    exact cZ_sum_le_tableWeight (h25 j hj) (h35 j hj) (hsch j hj) (Np j) (hNp j hj)
  have hs := Finset.sum_le_sum hper
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.sum_div, ← Finset.mul_sum, ← hN, ← hw,
    Finset.sum_const, Finset.card_range, nsmul_eq_mul] at hs
  have he := Real.exp_pos 1
  have hceil := Nat.ceil_lt_add_one (div_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 4) (N₀_nonneg true x)) he.le)
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast hpos
  rw [hk]
  linarith

/-! ## A committed table extracts, at table `j` -/

section Committed

open Model Refine Claims.Flock

/-- **At table `j`'s index**, masked: table `j`'s word is `2^d` long and its columns have the `2^lanes` witness
lanes. -/
theorem zk_atJ {x : Claims.Msg} {j : ℕ} {spec : SessionSpec} {st : _root_.Flock.Stmt}
    (h : (tables true x)[j]? = some (spec, st)) {i : ZkIdx} (hi : i.m = st.m) :
    wordLenJ true x j = 2 ^ zkLen i ∧ ∀ u, wideJ true x j u ↔ u.length = 2 ^ zkLanes i := by
  refine ⟨?_, fun u => ?_⟩
  · unfold wordLenJ
    rw [h]
    simp [zkLen, hi]
  · unfold wideJ
    rw [h]
    simp [zkLanes, hi]

/-- **A close table extracts table `j`** (`good_of_close_zk` at table `j`). -/
theorem good_of_close_zkJ (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) (j : ℕ)
    (s : Claims.Flock.Stmt) (hs : (deployZk rel).table x j = some s) (i : ZkIdx)
    (hN : wordLenJ true x j = 2 ^ zkLen i) (C : Fin (wordLenJ true x j) → List 𝔽)
    (hw : ∀ q, (C q).length = 2 ^ zkLanes i) (R : Fin (2 ^ zkLanes i) → Fin (2 ^ zkCols i + zkPad i) → 𝔽)
    (hclose : CloseRows execArith (zkLen i) (2 ^ zkCols i + zkPad i) (zkThetaD ⟨i, zkOfRows padC R⟩)
      (fun (ℓ : Fin (2 ^ zkLanes i)) (q : Fin (2 ^ zkLen i)) => oracleOf C ℓ q) R)
    (hrel : Claims.Flock.RelS s (readTrace execArith i.m (zkLanes i) (zkCols i) (padWit (e := 0) padC R))) :
    GoodTableAt (deployZk rel) (wordLenJ true) x j C := by
  refine ⟨⟨i, zkOfRows padC R⟩, s, hs, ?_, hN.ge, ?_⟩
  · show Claims.Flock.RelS s (readTables zkReadD [⟨i, zkOfRows padC R⟩])
    rw [readTables_singleton, zkReadD_eq_padWit, zkRowsOf_zkOfRows]
    exact hrel
  · have hf : ∀ q : Fin (2 ^ zkLen i), zkEncD ⟨i, zkOfRows padC R⟩ q =
        List.ofFn fun ℓ => encAt execArith (zkLen i) (2 ^ zkCols i + zkPad i) (R ℓ) q := by
      intro q
      rw [zkEncD_eq]
      refine congrArg List.ofFn (funext fun ℓ => ?_)
      rw [zkEnc_eq_encAt, zkRowsOf_zkOfRows]
    have hc : Close (zkThetaD ⟨i, zkOfRows padC R⟩)
        (fun (ℓ : Fin (2 ^ zkLanes i)) (q : Fin (2 ^ zkLen i)) => oracleOf C ℓ q)
        (fun ℓ q => encAt execArith (zkLen i) (2 ^ zkCols i + zkPad i) (R ℓ) q.val) := by
      unfold Close
      unfold CloseRows at hclose
      convert hclose using 4
    have := agree_of_close hN C hw _ _ hf hc
    show (1 - zkThetaD ⟨i, zkOfRows padC R⟩) * ((2 ^ zkLen i : ℕ) : ℝ) ≤
      ((Finset.univ : Finset (Fin (2 ^ zkLen i))).filter fun q =>
        zkEncD ⟨i, zkOfRows padC R⟩ q.val = extTab C q.val).card
    push_cast
    exact this

/-- **A committed table extracts table `j`** (`good_of_committed_zk` at table `j`): a table of witness-lane columns
whose oracle is `CommittedPad` for the old statement `S` at the deployed masked schedule and padding, when a satisfying
old witness of `S` reads as a witness of table `j`'s statement. -/
theorem good_of_committed_zkJ (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) (j : ℕ)
    (s : Claims.Flock.Stmt) (hs : (deployZk rel).table x j = some s) (i : ZkIdx)
    (hN : wordLenJ true x j = 2 ^ zkLen i) (C : Fin (wordLenJ true x j) → List 𝔽)
    (hw : ∀ q, (C q).length = 2 ^ zkLanes i) (S : Statement)
    (hwit : ∀ P : Message 𝔽 (zkLanes i) (zkCols i),
      S.Satisfies (witnessOf execArith S.m (zkLanes i) (zkCols i) P) →
        Claims.Flock.RelS s (readTrace execArith i.m (zkLanes i) (zkCols i) P))
    (hc : CommittedPad execArith S (schedOf (zkSched i.m).1) (zkPad i) padC (oracleOf C)) :
    GoodTableAt (deployZk rel) (wordLenJ true) x j C := by
  have key : ∀ (k0 : ℕ) (l : FlockSoundness.Accounting.Level), k0 = zkLanes i → l = levelOf (zkLevel0 i.m) →
      ∀ R ∈ padList execArith l (zkPad i) k0 0 0 (oracleOf C),
        S.Satisfies (witnessOf execArith S.m k0 l.logCols (padWit padC R)) →
      GoodTableAt (deployZk rel) (wordLenJ true) x j C := by
    intro k0 l hk hl R hR hsat
    subst hk hl
    simp only [padList, Finset.mem_filter, Finset.mem_univ, true_and] at hR
    refine good_of_close_zkJ rel x j s hs i hN C hw R ?_ (hwit _ hsat)
    rw [logLen_zk, ← radius_zk ⟨i, zkOfRows padC R⟩] at hR
    unfold CloseRows at hR ⊢
    convert hR using 5 with q
    simp only [padRows, padLane_zero]
    exact Iff.rfl
  obtain ⟨P, ⟨R, hR, rfl⟩, hsat⟩ := hc
  exact key _ _ (zkLevel_table i).2 (zkLevel_table i).1 R hR hsat

/-- **An uncommitted old table at table `j`** (`not_committed_zk` at table `j`). -/
theorem not_committed_zkJ (rel : Readable.Parsed → Readable.Relation) (x : Claims.Msg) (j : ℕ)
    (s : Claims.Flock.Stmt) (hs : (deployZk rel).table x j = some s) {spec : SessionSpec} {st : _root_.Flock.Stmt}
    (hst : (tables true x)[j]? = some (spec, st)) (i : ZkIdx) (hi : i.m = st.m)
    (C : Fin (wordLenJ true x j) → List 𝔽) (hw : ∀ q, wideJ true x j (C q)) (S : Statement)
    (hwit : ∀ P : Message 𝔽 (zkLanes i) (zkCols i),
      S.Satisfies (witnessOf execArith S.m (zkLanes i) (zkCols i) P) →
        Claims.Flock.RelS s (readTrace execArith i.m (zkLanes i) (zkCols i) P))
    (hg : ¬ GoodTableAt (deployZk rel) (wordLenJ true) x j C) :
    ¬ CommittedPad execArith S (schedOf (zkSched i.m).1) (zkPad i) padC (oracleOf C) := fun hc =>
  hg (good_of_committed_zkJ rel x j s hs i (zk_atJ hst hi).1 C (fun q => ((zk_atJ hst hi).2 _).1 (hw q)) S hwit hc)

end Committed

/-- **Table `j` of a masked statement in scope**: C-Flock ZK's `m` and `k_log` (every table has table 0's,
`tables_fit`), its layout and its link layout at `PT_LOCAL` (`zk_table0` at table `j`). -/
theorem zk_scopeJ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) {j : ℕ} (hj : j < sts.size) :
    25 ≤ sts[j].m ∧ sts[j].m ≤ 35 ∧ sts[j].c.kLog ≤ 27 ∧
      ∃ L : Layout sts[j], (L.model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts := by
  have h0 : 0 < sts.size := by omega
  have hm := parse_m hs hset h0
  have hkl := parse_kLog hs hset h0
  have h25 : 25 ≤ sts[0].m := by have := hin.1; simp only [↓reduceIte] at this; omega
  have h35 : sts[0].m ≤ 35 := by have := hin.2.1; omega
  have hk : sts[0].c.kLog ≤ 27 := by have := hin.2.2.2.1; omega
  obtain ⟨p, hp, rfl, hP, -, -⟩ := parsed_of_parse hs
  obtain ⟨hpst, hpr, hsr⟩ := parseBytes_setup hp hset
  subst hpst
  obtain ⟨-, -, -, -, -, ht, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at ht hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  have hfit := Readable.tables_fit (Readable.tablesOf_of_parses hP) ht hpi' hj
  have hmj : p.sts[j].m = p.sts[0].m := hfit.1
  have hkj : p.sts[j].c.kLog = p.sts[0].c.kLog := hfit.2.1
  obtain ⟨-, L, hwf, hr⟩ := Readable.tableAt (Readable.tablesOf_of_parses hP) ht hpi' hj
  rw [← hpr] at hsr
  unfold Readable.Read.setup at hsr
  rw [hpi'] at hsr
  have hpts := (Discharge.HiddenExec.setupTables_pointsAny hsr).2 j hj
  refine ⟨by omega, by omega, by omega, L, (Refine.stmtOf_linkLayout _ hwf hr).1, ?_⟩
  show PT_LOCAL + (p.sts[j].m - p.sts[j].c.kLog) ≤ spec.mPts
  have h1 := hr.mPts
  have h2 := hr.hm
  omega

/-- **At table `j`, a message satisfying its model statement reads as a witness of table `j`'s own statement**
(`relS_of_table0` at table `j`): table `j`'s statement is the parsed one with table `j` alone (`Parsed.only`), whose
relation is table `j`'s model statement at its one table's share (`relS_relation`). -/
theorem relS_tableJ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) {j : ℕ} (hj : j < sts.size) {t : Claims.Flock.Stmt}
    (ht : table true Readable.relation x j = some t) (L : Layout sts[j]) {m lanes cols : ℕ} (hm : m = sts[j].m)
    (M : Model.Message 𝔽 lanes cols)
    (hsat : (L.model (Refine.regionsOf sts[j])).Satisfies
      (Model.witnessOf execArith (L.model (Refine.regionsOf sts[j])).m lanes cols M)) :
    Claims.Flock.RelS t (Claims.Flock.readTrace execArith m lanes cols M) := by
  obtain ⟨p, hp, rfl, hP, -, htab⟩ := parsed_of_parse hs
  obtain ⟨hpst, -, -⟩ := parseBytes_setup hp hset
  subst hpst
  rw [htab j, Option.some.injEq] at ht
  subst ht
  obtain ⟨-, -, -, -, -, htg, -, -, -, hpi, -⟩ := hin
  simp only [Readable.toReadable, Readable.Read.form, Bool.not_eq_true] at htg hpi
  have hpi' : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hpi])
  have hfit := Readable.tables_fit (Readable.tablesOf_of_parses hP) htg hpi' hj
  have hj' : (p.only j).sts = #[p.sts[j]] := by
    show #[p.sts[j]!] = _
    rw [getElem!_pos p.sts j hj]
  have hm1 : (Readable.toReadable Readable.relation (p.only j)).m = p.sts[j].m := by
    simp [Readable.toReadable, hj']
  have hk1 : (Readable.toReadable Readable.relation (p.only j)).kLog = p.sts[j].c.kLog := by
    simp [Readable.toReadable, hj']
  have h00 : (p.only j).sts[0]! = p.sts[j] := by rw [hj']; rfl
  have hsz : (p.only j).sts.size = 1 := by rw [hj']; rfl
  rw [Readable.relS_relation]
  intro c hc
  obtain rfl : c = 0 := by omega
  rw [hsz, h00, hm1, hk1]
  refine ⟨⟨rfl, rfl, hfit.2.2⟩, ?_⟩
  subst hm
  rw [zOf_readTrace rfl]
  exact hsat

/-- **At the masked record, a table that does not extract table `j` is not committed** to table `j`'s model
statement, at the old padded theorems' schedule and padding (`zk_not_committed` at table `j`, at `relS_tableJ`): at a
statement in scope that the deployed masked parser reads at the deployed relation and the verifier sets up, for every
table of witness-lane columns at table `j`. -/
theorem zk_not_committedJ {x : Claims.Msg} {s : Claims.Flock.Stmt}
    (hs : (deployZk Readable.relation).parse x = some s) (hin : Claims.Flock.InScope true s) {ins : Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : setupOf true x = some (ins, spec, sts)) {j : ℕ}
    (hj : j < sts.size) (L : Layout sts[j]) (C : Fin (wordLenJ true x j) → List 𝔽)
    (hw : ∀ q, wideJ true x j (C q))
    (hg : ¬ Claims.Flock.GoodTableAt (deployZk Readable.relation) (wordLenJ true) x j C) :
    ¬ CommittedPad execArith (L.model (Refine.regionsOf sts[j]))
      (Refine.schedOf (Claims.Flock.zkSched sts[j].m).1) (Claims.Flock.zkSched sts[j].m).2 padC
      (Claims.Flock.oracleOf C) := by
  obtain ⟨h25, h35, -, -⟩ := zk_scopeJ hs hin hset hj
  have hi : Claims.Flock.ZkIdx.m ⟨sts[j].m - 25, by omega⟩ = sts[j].m := by
    simp only [Claims.Flock.ZkIdx.m]
    omega
  obtain ⟨t, ht⟩ : ∃ t, table true Readable.relation x j = some t := by
    obtain ⟨p, -, -, -, -, htab⟩ := parsed_of_parse hs
    exact ⟨_, htab j⟩
  have := not_committed_zkJ Readable.relation x j t ht (tables_getElem? hset (Array.getElem?_eq_getElem hj))
    ⟨sts[j].m - 25, by omega⟩ hi C hw (L.model (Refine.regionsOf sts[j]))
    (fun P hsat => relS_tableJ hs hin hset hj ht L hi P hsat) hg
  unfold Claims.Flock.zkPad at this
  rwa [hi] at this

end FlockDeploy.KS
