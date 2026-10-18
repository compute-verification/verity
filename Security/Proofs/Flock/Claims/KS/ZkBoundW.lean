import Proofs.Flock.Claims.KS.ZkFitsW
import Proofs.Flock.Claims.KS.ZkSessionW

/-!
# Every masked table's bound off the table on a witness lane

`zk_le_sessionsJ` reads the sessions' acceptance with no level-0 opening off `C` on any lane (`Off₀Z`), which a table of
`Flock.KS`'s masked columns, fixing only the witness lanes, does not decide. Here it reads no opening off `C` on a
witness lane (`OffW₀Z`), through `session_sound_of_table_accZ_w`, whose level-0 term `√(2^{d₀}·2Q₀·Adv₀)` is one more
fork event per table: at one tape and table, the deployed runs accept with no column of table `j` off `C` with
probability at most `ε_table` and the joined escapes with level 0's (`joinEscW`, through `oldEscW_term_le_join`)
(`zk_le_sessionsJW`). With the escapes counted per round (`TableBoundJAccAtR`, which `TableBoundJAccAt` gives:
`TableBoundJAccAt.tableBoundJAccAtR`), the masked sessions after `Commit` that bound every table at that verdict and
whose events are on accepted runs (`ZkSessWJ`) give every masked table's bound (`zk_tableBoundJAccAtR_of_sessionsW`,
the weight from `joinEscW_fitsForkR_zk`).
-/

namespace Claims.Flock

/-! ## Each table's escapes with level 0's, under the joined ones -/

section Join

variable {Ω Coin : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}

/-- **Each table's escapes with level 0's are at most the joined ones'**, at a table whose fork events are at distinct
rounds. -/
theorem oldEscW_term_le_join [Fintype Coin] (n : ℕ) (sch : ℕ → FlockSoundness.Accounting.Schedule) (Np : ℕ → ℕ)
    (rd : (j : ℕ) → IZW (sch j).levels.length → Fin (V.k x))
    (ev : (j : ℕ) → IZW (sch j).levels.length → Ω → (Fin (V.k x) → Coin) → (Fin (V.k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin (V.k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' →
      TiedColl H ((run H P V x ω s).vLog ++ (run H P V x ω (forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → TiedColl H (run H P V x ω s).vLog) {j : ℕ} (hj : j < n)
    (hinj : Function.Injective (rd j)) (ω : Ω) :
    (∑ i : IZW (sch j).levels.length, Real.sqrt (cZW (sch j) (Np j) i *
        Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2)) +
      (Pr fun s : Fin (V.k x) → Coin => ev₁ j ω s) ≤
      (joinEscW H V P x n sch Np rd ev ev₁ tied tied₁).term ω := by
  set c : Fin (V.k x) → ℝ := fun r =>
    ∑ j ∈ Finset.range n, ∑ i ∈ Finset.univ.filter (fun i => rd j i = r), cZW (sch j) (Np j) i with hc
  set g : Fin (V.k x) → ℝ := fun r => Real.sqrt (c r *
    Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ∃ j < n, ∃ i, rd j i = r ∧ ev j i ω s.1 s.2) with hg
  show _ ≤ (∑ r : Fin (V.k x), g r) + ∑ _l : Unit, (1 : ℝ) * (Pr fun s : Fin (V.k x) → Coin => ∃ j < n, ev₁ j ω s)
  refine add_le_add ?_ ?_
  · have hle : ∀ i, Real.sqrt (cZW (sch j) (Np j) i *
        Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2) ≤ g (rd j i) := by
      intro i
      have hci : cZW (sch j) (Np j) i ≤ c (rd j i) := by
        have h1 : cZW (sch j) (Np j) i ≤
            ∑ i' ∈ Finset.univ.filter (fun i' => rd j i' = rd j i), cZW (sch j) (Np j) i' :=
          Finset.single_le_sum (f := cZW (sch j) (Np j)) (fun i' _ => cZW_nonneg _ _ i')
            (by simp only [Finset.mem_filter, Finset.mem_univ, true_and])
        have h2 : (∑ i' ∈ Finset.univ.filter (fun i' => rd j i' = rd j i), cZW (sch j) (Np j) i') ≤ c (rd j i) :=
          Finset.single_le_sum (f := fun j' => ∑ i' ∈ Finset.univ.filter (fun i' => rd j' i' = rd j i),
              cZW (sch j') (Np j') i')
            (fun j' _ => Finset.sum_nonneg fun i' _ => cZW_nonneg _ _ i') (Finset.mem_range.2 hj)
        exact h1.trans h2
      have hpi : (Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2) ≤
          Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) =>
            ∃ j' < n, ∃ i', rd j' i' = rd j i ∧ ev j' i' ω s.1 s.2 :=
        Pr_mono fun s h => ⟨j, hj, i, rfl, h⟩
      exact Real.sqrt_le_sqrt (mul_le_mul hci hpi (Pr_nonneg _) ((cZW_nonneg _ _ i).trans hci))
    calc (∑ i : IZW (sch j).levels.length, Real.sqrt (cZW (sch j) (Np j) i *
            Pr fun s : (Fin (V.k x) → Coin) × (Fin (V.k x) → Coin) => ev j i ω s.1 s.2))
        ≤ ∑ i : IZW (sch j).levels.length, g (rd j i) := Finset.sum_le_sum fun i _ => hle i
      _ = ∑ r ∈ Finset.univ.image (rd j), g r := (Finset.sum_image fun a _ b _ h => hinj h).symm
      _ ≤ ∑ r, g r := Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
          fun _ _ _ => Real.sqrt_nonneg _
  · rw [Fintype.sum_unique, one_mul]
    exact Pr_mono fun s h => ⟨j, hj, h⟩

end Join

/-! ## Every table's bound, the escapes counted per round -/

section At

variable {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`TableBoundJAccAt` with the escapes counted per round** (`Escapes.FitsForkR`). -/
def TableBoundJAccAtR (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop)
    (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.OnAcc ∧
    e.FitsForkR (D.weight x) (D.k x) ∧ ∀ (ω : Ω) (j : ℕ), j < nTab D x → ∀ C : Fin (N x j) → Col,
      (∀ p, Ok x j (C p)) → ¬ GoodTableAt D N x j C →
      accPr H D.V P x ω ≤ D.εtable x + e.term ω +
        Pr fun r : Fin (D.V.k x) → Coin => (run H P D.V x ω r).out = .acc ∧
          ∃ (p : Fin (N x j)) (u : Col), lookCol (D.cols (run H P D.V x ω r).seen j) p.val = some u ∧ u ≠ C p

theorem TableBoundJAccAt.tableBoundJAccAtR {H : Msg → Msg} {D : Deploy Coin Col T} {N : Msg → ℕ → ℕ}
    {Ok : Msg → ℕ → Col → Prop} {x : Msg} (h : TableBoundJAccAt H D N Ok x) (hk : 0 < D.V.k x) :
    TableBoundJAccAtR H D N Ok x := by
  intro Ω _ _ P
  obtain ⟨e, ha, hf, hb⟩ := h Ω P
  exact ⟨e, ha, hf.fitsForkR (Nat.cast_nonneg _) hk, hb⟩

end At

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

open Classical in
/-- **The masked bound at one tape and table, off the table on a witness lane**: `zk_le_sessionsJ` where the mixture
`σ` bounds the deployed runs accepting with no column of table `j` off `C` by its acceptance with no level-0 opening
off `C` on a witness lane unless it self-clashes, and table `j`'s level-0 event bounds its `Adv₀`. Each session's
acceptance there is at most its `ε_c⁻` and level-0 term (`session_sound_of_table_accZ_w`), whose mean is at most
`ε_table`, table `j`'s old escapes' term (`mean_epsSZ_leJ`) and `√(2^{d₀}·2Q₀·Pr[ev₀])`, at most the joined escapes'
with level 0's (`oldEscW_term_le_join`). -/
theorem zk_le_sessionsJW {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (hA : A.Correct) (Hs : Hashes 𝔽 D Sl) {x : Claims.Msg} {spec : SessionSpec}
    (st : ℕ → Stmt) (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j))
    (T : ℕ → TabZK 𝔽 spec.mPts)
    (hT : ∀ j < (tables true x).length, 25 ≤ (st j).m ∧ (st j).m ≤ 35 ∧
      (T j).sch = Accounting.m1Schedule (st j).m ∧ (T j).t = Accounting.m1Pad (st j).m ∧ (T j).e = 2 ∧
      (T j).S.shape spec.mPts = acct spec (st j) ∧ (T j).N ≤ 2 ^ 13 ∧ (T j).q = 168 ∧ 7 / 16 ≤ (T j).δ ∧
      (T j).N ≤ Claims.Flock.Numbers.padDomain
        (Zk.PadLayout.new (st j).m (st j).c.kLog (Claims.Flock.zkSched (st j).m).1.k0))
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω)
    (rd : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Fin ((V true).k x))
    (hrd : ∀ j, Function.Injective (rd j))
    (ev : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
      (Fin ((V true).k x) → Coin) → Prop)
    (ev₁ : ℕ → Ω → (Fin ((V true).k x) → Coin) → Prop)
    (tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
      ((Claims.run sha512 P (V true) x ω s).vLog ++
        (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).vLog))
    (tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog)
    (ω : Ω) {j : ℕ} (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽)
    (hTc : ¬ (T j).Committed A (Claims.Flock.oracleOf C)) {τ : Type} [Fintype τ] [Nonempty τ]
    (pre post : List ((Fin 2 → Fin spec.mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
    (σ : τ → Strategy (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)))
    (hacc : (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
            u ≠ C p) ≤
      (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
          (outs.getD pre.length dfltOutZC).accepted = true ∧
            ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
              ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
        (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ)
    (hadv : ∀ r ℓ, (∑ t, advRSZ A Hs (T j) pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inl (r, ℓ))) ω s.1 s.2)
    (hP : ∀ r : Fin 2, (∑ t, advPadsSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inr (.inl r))) ω s.1 s.2)
    (hTau : ∀ r : Fin 2, (∑ t, advTauSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inr (.inr r))) ω s.1 s.2)
    (h₀ : (∑ t, adv₀SZ A Hs (T j) pre post cap₀ (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) => ev j (.inr ()) ω s.1 s.2)
    (hsc : (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
        SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))
      (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s) :
    Claims.accPr sha512 (V true) P x ω ≤ εtable true x +
      (Claims.Flock.joinEscW sha512 (V true) P x (tables true x).length (fun j => (T j).sch)
        (fun j => (T j).N) rd ev ev₁ tied tied₁).term ω +
      Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
            u ≠ C p := by
  obtain ⟨h25, h35, hsch, ht, he, hS, hN, hq, hδ, -⟩ := hT j hj
  set c₀ : ℝ := 2 ^ (lvOf (T j).sch 0).logLen * (2 * (lvOf (T j).sch 0).queries) with hc₀
  have hc₀0 : 0 ≤ c₀ := by positivity
  have hper : ∀ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
        (outs.getD pre.length dfltOutZC).accepted = true ∧
          ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
            ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
      (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t) ≤
        epsSZ A Hs (T j) pre post cap₀ (σ t) + Real.sqrt (c₀ * adv₀SZ A Hs (T j) pre post cap₀ (σ t)) := by
    intro t
    have hs := session_sound_of_table_accZ_w A hA Hs (T j) pre post cap₀
      (fun outs => (outs.getD pre.length dfltOutZC).accepted = true ∧
        ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
          ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (OutZC 𝔽 D Sl) =>
        ((outs.getD pre.length dfltOutZC).accepted = true ∧
          ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
            ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))) ∧
          ¬ (T j).Committed A (Claims.Flock.oracleOf C)) = ind fun outs =>
        (outs.getD pre.length dfltOutZC).accepted = true ∧
          ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
            ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)) :=
      congrArg ind (funext fun _ => propext (and_iff_left hTc))
    have e₀ : (ind fun outs : List (OutZC 𝔽 D Sl) =>
        ((outs.getD pre.length dfltOutZC).accepted = true ∧
          ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
            ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))) ∧
          ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC) ∧
          OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC)) = fun _ => 0 := by
      funext outs
      unfold ind
      split_ifs with h0
      · exact absurd ⟨h0.2.2, h0.2.1⟩ h0.1.2
      · rfl
    rw [e, e₀, expect_const, add_zero] at hs
    exact hs
  have hmean := mean_epsSZ_leJ FlockLevel3.card_F A Hs (T j) pre post cap₀ σ (hst j hj) h25 h35 hsch ht he hS hN
    hq hδ sha512 (V true) P (fun i => rd j (.inl i)) (fun i => ev j (.inl i)) (ev₁ j) (fun i => tied j (.inl i))
    (tied₁ j) ω hadv hP hTau hsc
  have hadv₀ : ∀ t, 0 ≤ adv₀SZ A Hs (T j) pre post cap₀ (σ t) := fun t => by
    unfold adv₀SZ expect₂
    exact expect_nonneg (fun _ => expect_nonneg (fun _ => ind_nonneg _ _) _ _) _ _
  have hlev₀ : (∑ t, Real.sqrt (c₀ * adv₀SZ A Hs (T j) pre post cap₀ (σ t))) / Fintype.card τ ≤
      Real.sqrt (c₀ * Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inr ()) ω s.1 s.2) := by
    refine (Claims.Flock.mean_sqrt_le _ fun t => mul_nonneg hc₀0 (hadv₀ t)).trans (Real.sqrt_le_sqrt ?_)
    rw [← Finset.mul_sum, mul_div_assoc]
    exact mul_le_mul_of_nonneg_left h₀ hc₀0
  have hjoin := Claims.Flock.oldEscW_term_le_join (tables true x).length (fun j => (T j).sch) (fun j => (T j).N) rd
    ev ev₁ tied tied₁ hj (hrd j) ω
  have hold : (Claims.Flock.oldEscZ sha512 (V true) P x (T j).sch (T j).N (fun i => rd j (.inl i))
      (fun i => ev j (.inl i)) (ev₁ j) (fun i => tied j (.inl i)) (tied₁ j)).term ω +
      Real.sqrt (c₀ * Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inr ()) ω s.1 s.2) =
      (∑ i : Claims.Flock.IZW (T j).sch.levels.length, Real.sqrt (Claims.Flock.cZW (T j).sch (T j).N i *
        Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) => ev j i ω s.1 s.2)) +
      Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s := by
    show (∑ i : Claims.Flock.IZ (T j).sch.levels.length, Real.sqrt (Claims.Flock.cZ (T j).sch (T j).N i *
        Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) => ev j (.inl i) ω s.1 s.2)) +
      ∑ _l : Unit, (1 : ℝ) * (Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s) + _ = _
    rw [Fintype.sum_sum_type (α₁ := Claims.Flock.IZ (T j).sch.levels.length) (α₂ := Unit)]
    simp only [Fintype.sum_unique, one_mul, Claims.Flock.cZW, hc₀]
    ring
  have hsum := Finset.sum_le_sum fun t (_ : t ∈ Finset.univ) => hper t
  have hcard : (0 : ℝ) ≤ Fintype.card τ := Nat.cast_nonneg _
  have hdiv := div_le_div_of_nonneg_right hsum hcard
  rw [Claims.Flock.mean_add] at hdiv
  have hsplit : Claims.accPr sha512 (V true) P x ω ≤
      (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
            u ≠ C p) +
      Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
            u ≠ C p := by
    unfold Claims.accPr
    refine le_trans (Claims.Pr_mono fun r hr => ?_) (Claims.Pr_or_le _ _)
    by_cases hq : ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
        Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧ u ≠ C p
    · exact Or.inr ⟨hr, hq⟩
    · exact Or.inl ⟨hr, hq⟩
  linarith [hsplit, hacc, hmean, hdiv, hjoin, hold, hlev₀]

/-- **The masked tables' sessions after `Commit`, off the table on a witness lane, with events on accepted runs**:
`ZkSessJ` with level 0's fork event at each table (`IZW`), every event on accepted deployed runs, and the mixture
bounding the deployed runs by its acceptance with no level-0 opening off `C` on a witness lane unless it
self-clashes. -/
def ZkSessWJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (Ok : Claims.Msg → ℕ → List 𝔽 → Prop) (x : Claims.Msg) {mPts : ℕ}
    (T : ℕ → TabZK 𝔽 mPts) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j))
      (ev : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Ω → (Fin ((V true).k x) → Coin) →
        (Fin ((V true).k x) → Coin) → Prop)
      (ev₁ : ℕ → Ω → (Fin ((V true).k x) → Coin) → Prop)
      (_tied : ∀ j i ω s s', ev j i ω s s' → Claims.TiedColl sha512
        ((Claims.run sha512 P (V true) x ω s).vLog ++
          (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).vLog))
      (_tied₁ : ∀ j ω s, ev₁ j ω s → Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω s).vLog)
      (_acc : ∀ j i ω s s', ev j i ω s s' → (Claims.run sha512 P (V true) x ω s).out = .acc ∧
        (Claims.run sha512 P (V true) x ω (Claims.forkCoins (rd j i) s s')).out = .acc)
      (_acc₁ : ∀ j ω s, ev₁ j ω s → (Claims.run sha512 P (V true) x ω s).out = .acc),
      ∀ ω (j : ℕ), j < (tables true x).length → ∀ C : Fin (wordLenJ true x j) → List 𝔽, (∀ p, Ok x j (C p)) →
        ∃ (τ : Type) (_ : Fintype τ) (_ : Nonempty τ)
          (pre post : List ((Fin 2 → Fin mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
          (σ : τ → Strategy (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀))),
          (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
              ¬ ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
                Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
                  u ≠ C p) ≤
            (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
                (outs.getD pre.length dfltOutZC).accepted = true ∧
                  ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
                    ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
              (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ∧
          (∀ r ℓ, (∑ t, advRSZ A Hs (T j) pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) ∧
          (∀ r : Fin 2, (∑ t, advPadsSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inl r))) ω s.1 s.2) ∧
          (∀ r : Fin 2, (∑ t, advTauSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inr r))) ω s.1 s.2) ∧
          ((∑ t, adv₀SZ A Hs (T j) pre post cap₀ (σ t)) / Fintype.card τ ≤
            Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr ()) ω s.1 s.2) ∧
          (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
              SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))
            (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

/-- **Every masked table's bound at a statement, off the table on a witness lane, the escapes counted per round**:
where table `j` of the setup is `st j` and `T j` is C-Flock ZK's padded table there, when the masked sessions bound
every table at that verdict with events on accepted runs (`ZkSessWJ`) and no table `Ok` allows that does not extract
table `j` is committed to `T j`. The escapes are the tables' with level 0's joined by round (`joinEscW`), on accepted
runs (`joinEscW_onAcc`), fitting the deployed weight per round (`joinEscW_fitsForkR_zk`), and at each tape and table
the bound is `zk_le_sessionsJW`. -/
theorem zk_tableBoundJAccAtR_of_sessionsW {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D]
    [Inhabited Sl] (A : Model.Arith 𝔽 K) (hA : A.Correct) (Hs : Hashes 𝔽 D Sl)
    (rel : Readable.Parsed → Readable.Relation) (Ok : Claims.Msg → ℕ → List 𝔽 → Prop) {x : Claims.Msg}
    (hn : Claims.Flock.nTab (deployZk rel) x = (tables true x).length) {spec : SessionSpec} (st : ℕ → Stmt)
    (hst : ∀ j < (tables true x).length, (tables true x)[j]? = some (spec, st j)) (T : ℕ → TabZK 𝔽 spec.mPts)
    (hT : ∀ j < (tables true x).length, 25 ≤ (st j).m ∧ (st j).m ≤ 35 ∧
      (T j).sch = Accounting.m1Schedule (st j).m ∧ (T j).t = Accounting.m1Pad (st j).m ∧ (T j).e = 2 ∧
      (T j).S.shape spec.mPts = acct spec (st j) ∧ (T j).N ≤ 2 ^ 13 ∧ (T j).q = 168 ∧ 7 / 16 ≤ (T j).δ ∧
      (T j).N ≤ Claims.Flock.Numbers.padDomain
        (Zk.PadLayout.new (st j).m (st j).c.kLog (Claims.Flock.zkSched (st j).m).1.k0))
    (hnc : ∀ j < (tables true x).length, ∀ C : Fin (wordLenJ true x j) → List 𝔽, (∀ p, Ok x j (C p)) →
      ¬ Claims.Flock.GoodTableAt (deployZk rel) (wordLenJ true) x j C →
        ¬ (T j).Committed A (Claims.Flock.oracleOf C))
    (hmix : ZkSessWJ A Hs Ok x T) :
    Claims.Flock.TableBoundJAccAtR sha512 (deployZk rel) (wordLenJ true) Ok x := by
  intro Ω _ _ P
  obtain ⟨rd, hrd, ev, ev₁, tied, tied₁, hacc, hacc₁, hω⟩ := hmix Ω P
  refine ⟨Claims.Flock.joinEscW sha512 (V true) P x (tables true x).length (fun j => (T j).sch)
      (fun j => (T j).N) rd ev ev₁ tied tied₁,
    Claims.Flock.joinEscW_onAcc (tables true x).length (fun j => (T j).sch) (fun j => (T j).N) rd ev ev₁ tied tied₁
      hacc hacc₁,
    joinEscW_fitsForkR_zk rel P x st hst (fun j hj => (hT j hj).1) (fun j hj => (hT j hj).2.1) _
      (fun j hj => (hT j hj).2.2.1) _ (fun j hj => (hT j hj).2.2.2.2.2.2.2.2.2) rd hrd ev ev₁ tied tied₁,
    fun ω j hj C hok hg => ?_⟩
  rw [hn] at hj
  obtain ⟨τ, _, _, pre, post, cap₀, σ, hacc', hadv, hP, hTau, h₀, hsc⟩ := hω ω j hj C hok
  exact zk_le_sessionsJW A hA Hs st hst T hT P rd hrd ev ev₁ tied tied₁ ω hj C (hnc j hj C hok hg) pre post cap₀ σ
    hacc' hadv hP hTau h₀ hsc

end FlockDeploy.KS
