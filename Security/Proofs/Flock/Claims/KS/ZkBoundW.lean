import Proofs.Flock.Claims.KS.ZkSlackW
import Proofs.Flock.Claims.KS.ZkSessionW

/-!
# Every masked table's bound off the table on a witness lane

`zk_le_sessionsJ` reads the sessions' acceptance with no level-0 opening off `C` on any lane (`Off₀Z`), which a table of
`Flock.KS`'s masked columns, fixing only the witness lanes, does not decide. Here it reads no opening off `C` on a
witness lane (`OffW₀Z`), through `session_sound_of_table_accZ_w`, whose level-0 term `√(2^{d₀}·2Q₀·Adv₀)` is one more
fork event per table. Each fork term's mean `Adv` is bounded by its fork event's probability and the fork bound's slack
at its round (`Claims.Flock.forkSlack`): at one tape and table, the deployed runs accept with no column of table `j`
off `C` with probability at most `ε_table` and the slack's escapes, the tables' with level 0's and the slack's merged
by round (`Claims.Flock.slackEscW`, through `mean_epsSZ_le_rawJ` and `Claims.Flock.slackEscW_term`)
(`zk_le_sessionsJW`). With the escapes counted per round on accepted first runs (`TableBoundJAccAtR`, which
`TableBoundJAccAt` gives: `TableBoundJAccAt.tableBoundJAccAtR`), the masked sessions after `Commit` that bound every
table at that verdict, whose events are on accepted runs, whose tables' level-0 events alone are at round `0` and
whose slack fits the room (`Claims.Flock.SlackFitW`) (`ZkSessWJ`) give every masked table's bound
(`zk_tableBoundJAccAtR_of_sessionsW`, the weight from `slackEscW_fitsForkR_zk`).
-/

namespace Claims.Flock

/-- **The old masked fork and self-clash terms are at most their roots at bounds `b`**: `oldEscZ_term` with each fork
term's `Adv` (a level's, a pads root's, `τ`'s) at most its bound and the self-clash probability at most `b₁`. -/
theorem oldEscZ_terms_le (sch : FlockSoundness.Accounting.Schedule) (Np : ℕ) (adv : ℕ → ℕ → ℝ) (advP advT : ℕ → ℝ)
    (sc : ℝ) (b : IZ sch.levels.length → ℝ) (b₁ : ℝ) (hadv : ∀ r ℓ, adv r.val (ℓ.val + 1) ≤ b (.inl (r, ℓ)))
    (hP : ∀ r : Fin 2, advP r.val ≤ b (.inr (.inl r))) (hT : ∀ r : Fin 2, advT r.val ≤ b (.inr (.inr r)))
    (hsc : sc ≤ b₁) :
    (∑ r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 sch.levels.length,
        Real.sqrt (2 ^ (FlockSoundness.lvOf sch ℓ).logLen * (FlockSoundness.lvOf sch ℓ).queries * adv r ℓ)) +
      ∑ r ∈ Finset.range 2, Real.sqrt (Np * Np * advP r) + ∑ r ∈ Finset.range 2, Real.sqrt (advT r) + sc ≤
      (∑ i, Real.sqrt (cZ sch Np i * b i)) + b₁ := by
  show _ ≤ (∑ i : (Fin 2 × Fin sch.levels.length) ⊕ (Fin 2 ⊕ Fin 2), Real.sqrt (cZ sch Np i * b i)) + b₁
  rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_prod_type, Finset.sum_range, Finset.sum_range,
    Finset.sum_range, ← add_assoc]
  refine add_le_add (add_le_add (add_le_add (Finset.sum_le_sum fun r _ => ?_)
    (Finset.sum_le_sum fun r _ => ?_)) (Finset.sum_le_sum fun r _ => ?_)) hsc
  · rw [sum_Icc_one]
    exact Finset.sum_le_sum fun ℓ _ =>
      Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (hadv r ℓ) (cOf_nonneg sch _))
  · exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (hP r) (cZ_nonneg sch Np (.inr (.inl r))))
  · exact Real.sqrt_le_sqrt ((hT r).trans_eq (one_mul _).symm)

/-! ## Every table's bound, the escapes counted per round -/

section At

variable {Coin Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`TableBoundJAccAt` with the escapes counted per round** (`Escapes.FitsForkR`), on accepted first runs
(`Escapes.OnAcc₁`). -/
def TableBoundJAccAtR (H : Msg → Msg) (D : Deploy Coin Col T) (N : Msg → ℕ → ℕ) (Ok : Msg → ℕ → Col → Prop)
    (x : Msg) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.OnAcc₁ ∧
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
  exact ⟨e, ha.onAcc₁, hf.fitsForkR (Nat.cast_nonneg _) hk, hb⟩

end At

end Claims.Flock

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

namespace Gen

variable [S : ZkSetup]

/-- **The masked session's `ε_c⁻`, averaged over a mixture of strategies, at table `j`, at bounds `b`**: at most
`ε_table` and the old masked terms' roots at bounds of the fork terms' mean `Adv`s and the mean self-clash
(`mean_epsSZ_leJ` at `Claims.Flock.oldEscZ_terms_le`). -/
theorem mean_epsSZ_le_rawJ {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (hF : Fintype.card F = 2 ^ 128) {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : Model.Arith F K)
    (Hs : Hashes F D Sl) {mPts : ℕ} (T : TabZK F mPts) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (cap₀ : Model.Cap D) {τ : Type} [Fintype τ] [Nonempty τ]
    (σ : τ → Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    {x : Claims.Msg} {j : ℕ} {spec : SessionSpec} {st : Stmt} (h : (tables true x)[j]? = some (spec, st))
    (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hsch : T.sch = Accounting.m1Schedule st.m)
    (ht : T.t = Accounting.m1Pad st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st)
    (hN : T.N ≤ 2 ^ 13) (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ) (b : Claims.Flock.IZ T.sch.levels.length → ℝ) (b₁ : ℝ)
    (hadv : ∀ r ℓ, (∑ t, advRSZ A Hs T pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤ b (.inl (r, ℓ)))
    (hP : ∀ r : Fin 2, (∑ t, advPadsSZ A Hs T pre post cap₀ r.val (σ t)) / Fintype.card τ ≤ b (.inr (.inl r)))
    (hT : ∀ r : Fin 2, (∑ t, advTauSZ A Hs T pre post cap₀ r.val (σ t)) / Fintype.card τ ≤ b (.inr (.inr r)))
    (hsc : (∑ t, expect (ind fun outs : List (OutZC F D Sl) =>
        SelfClashZ Hs T (outs.getD pre.length dfltOutZC)) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) (σ t)) /
        Fintype.card τ ≤ b₁) :
    (∑ t, epsSZ A Hs T pre post cap₀ (σ t)) / Fintype.card τ ≤
      εtable true x + ((∑ i, Real.sqrt (Claims.Flock.cZ T.sch T.N i * b i)) + b₁) := by
  have key := Claims.Flock.oldEscZ_terms_le T.sch T.N
    (fun r ℓ => (∑ t, advRSZ A Hs T pre post cap₀ r ℓ (σ t)) / Fintype.card τ)
    (fun r => (∑ t, advPadsSZ A Hs T pre post cap₀ r (σ t)) / Fintype.card τ)
    (fun r => (∑ t, advTauSZ A Hs T pre post cap₀ r (σ t)) / Fintype.card τ) _ b b₁ hadv hP hT hsc
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

end Gen

/-- **The masked session's `ε_c⁻`, averaged over a mixture of strategies, at table `j`, at bounds `b`**: at most
`ε_table` and the old masked terms' roots at bounds of the fork terms' mean `Adv`s and the mean self-clash
(`mean_epsSZ_leJ` at `Claims.Flock.oldEscZ_terms_le`). -/
theorem mean_epsSZ_le_rawJ {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
    (hF : Fintype.card F = 2 ^ 128) {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : Model.Arith F K)
    (Hs : Hashes F D Sl) {mPts : ℕ} (T : TabZK F mPts) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl)))
    (cap₀ : Model.Cap D) {τ : Type} [Fintype τ] [Nonempty τ]
    (σ : τ → Strategy (sessAtZ pre post (tabAfterZC A Hs T cap₀)))
    {x : Claims.Msg} {j : ℕ} {spec : SessionSpec} {st : Stmt} (h : (tables true x)[j]? = some (spec, st))
    (h25 : 25 ≤ st.m) (h35 : st.m ≤ 35) (hsch : T.sch = Accounting.m1Schedule st.m)
    (ht : T.t = Accounting.m1Pad st.m) (he : T.e = 2) (hS : T.S.shape mPts = acct spec st)
    (hN : T.N ≤ 2 ^ 13) (hq : T.q = 168) (hδ : 7 / 16 ≤ T.δ) (b : Claims.Flock.IZ T.sch.levels.length → ℝ) (b₁ : ℝ)
    (hadv : ∀ r ℓ, (∑ t, advRSZ A Hs T pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤ b (.inl (r, ℓ)))
    (hP : ∀ r : Fin 2, (∑ t, advPadsSZ A Hs T pre post cap₀ r.val (σ t)) / Fintype.card τ ≤ b (.inr (.inl r)))
    (hT : ∀ r : Fin 2, (∑ t, advTauSZ A Hs T pre post cap₀ r.val (σ t)) / Fintype.card τ ≤ b (.inr (.inr r)))
    (hsc : (∑ t, expect (ind fun outs : List (OutZC F D Sl) =>
        SelfClashZ Hs T (outs.getD pre.length dfltOutZC)) (sessAtZ pre post (tabAfterZC A Hs T cap₀)) (σ t)) /
        Fintype.card τ ≤ b₁) :
    (∑ t, epsSZ A Hs T pre post cap₀ (σ t)) / Fintype.card τ ≤
      εtable true x + ((∑ i, Real.sqrt (Claims.Flock.cZ T.sch T.N i * b i)) + b₁) :=
  Gen.mean_epsSZ_le_rawJ (S := S₀) (F := F) (K := K) (hF := hF) (D := D) (Sl := Sl) (A := A) (Hs := Hs) (mPts := mPts)
    (T := T) (pre := pre) (post := post) (cap₀ := cap₀) (τ := τ) (σ := σ) (x := x) (j := j) (spec := spec) (st := st)
    (h := h) (h25 := h25) (h35 := h35) (hsch := hsch) (ht := ht) (he := he) (hS := hS) (hN := hN) (hq := hq) (hδ := hδ)
    (b := b) (b₁ := b₁) (hadv := hadv) (hP := hP) (hT := hT) (hsc := hsc)

namespace Gen

variable [S : ZkSetup]

open Classical in
/-- **The masked bound at one tape and table, off the table on a witness lane, at the fork bound's slack**:
`zk_le_sessionsJ` where the mixture `σ` bounds the deployed runs accepting with no column of table `j` off `C` by its
acceptance at table `j` and every table before it (`AccUpTo`) with no level-0 opening off `C` on a witness lane unless
it self-clashes, each fork term's mean `Adv` at
table `j` (level 0's `Adv₀` among them) is at most its fork event's probability and the slack of the fork at its round
(`Claims.Flock.forkSlack`), and the mean self-clash at most the one-run event's. Each session's acceptance there is at
most its `ε_c⁻` and level-0 term (`session_sound_of_table_accZ_w`), whose mean is at most `ε_table`, table `j`'s old
roots at those bounds (`mean_epsSZ_le_rawJ`) and level 0's, at most the slack's escapes' term
(`Claims.Flock.slackEscW_term`). -/
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
    (δ : ℕ → ℝ) (hδs : ∀ t, 0 ≤ δ t)
    (ω : Ω) {j : ℕ} (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽)
    (hTc : ¬ (T j).Committed A (Claims.Flock.oracleOf C)) {τ : Type} [Fintype τ] [Nonempty τ]
    (pre post : List ((Fin 2 → Fin spec.mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
    (σ : τ → Strategy (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)))
    (hacc : (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
            u ≠ C p) ≤
      (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
          AccUpTo pre.length outs ∧
            ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
              ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
        (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ)
    (hadv : ∀ r ℓ, (∑ t, advRSZ A Hs (T j) pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω)
    (hP : ∀ r : Fin 2, (∑ t, advPadsSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inr (.inl r))) ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω)
    (hTau : ∀ r : Fin 2, (∑ t, advTauSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inr (.inr r))) ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω)
    (h₀ : (∑ t, adv₀SZ A Hs (T j) pre post cap₀ (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) => ev j (.inr ()) ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω)
    (hsc : (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
        SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))
      (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s) :
    Claims.accPr sha512 (V true) P x ω ≤ εtable true x +
      (Claims.Flock.slackEscW sha512 (V true) P x (tables true x).length (fun j => (T j).sch)
        (fun j => (T j).N) rd ev ev₁ tied tied₁ δ).term ω +
      Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
            u ≠ C p := by
  obtain ⟨h25, h35, hsch, ht, he, hS, hN, hq, hδ, -⟩ := hT j hj
  set b : Claims.Flock.IZW (T j).sch.levels.length → ℝ := fun i =>
    (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) => ev j i ω s.1 s.2) +
      Claims.Flock.forkSlack sha512 (V true) P x δ (rd j i).val ω
  set c₀ : ℝ := 2 ^ (lvOf (T j).sch 0).logLen * (2 * (lvOf (T j).sch 0).queries) with hc₀
  have hc₀0 : 0 ≤ c₀ := by positivity
  have hper : ∀ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
        AccUpTo pre.length outs ∧
          ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
            ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
      (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t) ≤
        epsSZ A Hs (T j) pre post cap₀ (σ t) + Real.sqrt (c₀ * adv₀SZ A Hs (T j) pre post cap₀ (σ t)) := by
    intro t
    have hs := session_sound_of_table_accZ_w A hA Hs (T j) pre post cap₀
      (fun outs => AccUpTo pre.length outs ∧
        ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
          ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
      (fun _ h => h.1) (Claims.Flock.oracleOf C) (σ t)
    have e : (ind fun outs : List (OutZC 𝔽 D Sl) =>
        (AccUpTo pre.length outs ∧
          ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
            ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))) ∧
          ¬ (T j).Committed A (Claims.Flock.oracleOf C)) = ind fun outs =>
        AccUpTo pre.length outs ∧
          ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
            ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)) :=
      congrArg ind (funext fun _ => propext (and_iff_left hTc))
    have e₀ : (ind fun outs : List (OutZC 𝔽 D Sl) =>
        (AccUpTo pre.length outs ∧
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
  have hmean : (∑ t, epsSZ A Hs (T j) pre post cap₀ (σ t)) / Fintype.card τ ≤ εtable true x +
      ((∑ i, Real.sqrt (Claims.Flock.cZ (T j).sch (T j).N i * b (.inl i))) +
        Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s) :=
    mean_epsSZ_le_rawJ FlockLevel3.card_F A Hs (T j) pre post cap₀ σ (hst j hj) h25 h35 hsch ht he hS hN hq hδ
      (fun i => b (.inl i)) _ hadv hP hTau hsc
  have hadv₀ : ∀ t, 0 ≤ adv₀SZ A Hs (T j) pre post cap₀ (σ t) := fun t => by
    unfold adv₀SZ expect₂
    exact expect_nonneg (fun _ => expect_nonneg (fun _ => ind_nonneg _ _) _ _) _ _
  have hlev₀ : (∑ t, Real.sqrt (c₀ * adv₀SZ A Hs (T j) pre post cap₀ (σ t))) / Fintype.card τ ≤
      Real.sqrt (c₀ * b (.inr ())) := by
    refine (Claims.Flock.mean_sqrt_le _ fun t => mul_nonneg hc₀0 (hadv₀ t)).trans (Real.sqrt_le_sqrt ?_)
    rw [← Finset.mul_sum, mul_div_assoc]
    exact mul_le_mul_of_nonneg_left h₀ hc₀0
  have hterm := Claims.Flock.slackEscW_term (tables true x).length (fun j => (T j).sch) (fun j => (T j).N) rd ev
    ev₁ tied tied₁ δ hδs hj (hrd j) ω b fun _ => le_rfl
  have hold : ((∑ i : Claims.Flock.IZ (T j).sch.levels.length, Real.sqrt (Claims.Flock.cZ (T j).sch (T j).N i *
      b (.inl i))) + Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s) + Real.sqrt (c₀ * b (.inr ())) =
      (∑ i : Claims.Flock.IZW (T j).sch.levels.length, Real.sqrt (Claims.Flock.cZW (T j).sch (T j).N i * b i)) +
        Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s := by
    rw [Fintype.sum_sum_type (α₁ := Claims.Flock.IZ (T j).sch.levels.length) (α₂ := Unit)]
    simp only [Fintype.sum_unique, Claims.Flock.cZW, hc₀]
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
  linarith [hsplit, hacc, hmean, hdiv, hterm, hold, hlev₀]

end Gen

open Classical in
/-- **The masked bound at one tape and table, off the table on a witness lane, at the fork bound's slack**:
`zk_le_sessionsJ` where the mixture `σ` bounds the deployed runs accepting with no column of table `j` off `C` by its
acceptance at table `j` and every table before it (`AccUpTo`) with no level-0 opening off `C` on a witness lane unless
it self-clashes, each fork term's mean `Adv` at
table `j` (level 0's `Adv₀` among them) is at most its fork event's probability and the slack of the fork at its round
(`Claims.Flock.forkSlack`), and the mean self-clash at most the one-run event's. Each session's acceptance there is at
most its `ε_c⁻` and level-0 term (`session_sound_of_table_accZ_w`), whose mean is at most `ε_table`, table `j`'s old
roots at those bounds (`mean_epsSZ_le_rawJ`) and level 0's, at most the slack's escapes' term
(`Claims.Flock.slackEscW_term`). -/
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
    (δ : ℕ → ℝ) (hδs : ∀ t, 0 ≤ δ t)
    (ω : Ω) {j : ℕ} (hj : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽)
    (hTc : ¬ (T j).Committed A (Claims.Flock.oracleOf C)) {τ : Type} [Fintype τ] [Nonempty τ]
    (pre post : List ((Fin 2 → Fin spec.mPts → 𝔽) → Game (OutZC 𝔽 D Sl))) (cap₀ : Model.Cap D)
    (σ : τ → Strategy (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)))
    (hacc : (Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ¬ ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
            u ≠ C p) ≤
      (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
          AccUpTo pre.length outs ∧
            ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
              ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
        (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ)
    (hadv : ∀ r ℓ, (∑ t, advRSZ A Hs (T j) pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω)
    (hP : ∀ r : Fin 2, (∑ t, advPadsSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inr (.inl r))) ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω)
    (hTau : ∀ r : Fin 2, (∑ t, advTauSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
        ev j (.inl (.inr (.inr r))) ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω)
    (h₀ : (∑ t, adv₀SZ A Hs (T j) pre post cap₀ (σ t)) / Fintype.card τ ≤
      (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) => ev j (.inr ()) ω s.1 s.2) +
        Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω)
    (hsc : (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
        SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))
      (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ≤
      Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s) :
    Claims.accPr sha512 (V true) P x ω ≤ εtable true x +
      (Claims.Flock.slackEscW sha512 (V true) P x (tables true x).length (fun j => (T j).sch)
        (fun j => (T j).N) rd ev ev₁ tied tied₁ δ).term ω +
      Claims.Pr fun r : Fin ((V true).k x) → Coin => (Claims.run sha512 P (V true) x ω r).out = .acc ∧
        ∃ (p : Fin (wordLenJ true x j)) (u : List 𝔽),
          Claims.Flock.lookCol (cols true (Claims.run sha512 P (V true) x ω r).seen j) p.val = some u ∧
            u ≠ C p :=
  Gen.zk_le_sessionsJW (S := S₀) (K := K) (D := D) (Sl := Sl) (A := A) (hA := hA) (Hs := Hs) (x := x) (spec := spec)
    (st := st) (hst := hst) (T := T) (hT := hT) (Ω := Ω) (P := P) (rd := rd) (hrd := hrd) (ev := ev) (ev₁ := ev₁)
    (tied := tied) (tied₁ := tied₁) (δ := δ) (hδs := hδs) (ω := ω) (j := j) (hj := hj) (C := C) (hTc := hTc) (τ := τ)
    (pre := pre) (post := post) (cap₀ := cap₀) (σ := σ) (hacc := hacc) (hadv := hadv) (hP := hP) (hTau := hTau)
    (h₀ := h₀) (hsc := hsc)

/-- **The masked tables' sessions after `Commit`, off the table on a witness lane, with events on accepted runs**:
`ZkSessJ` with level 0's fork event at each table (`IZW`), every event on accepted deployed runs, and the mixture
bounding the deployed runs by its acceptance up to table `j` with no level-0 opening off `C` on a witness lane unless
it self-clashes. Each fork term's mean `Adv` is at most its fork event's probability and the slack of the fork at its
round (`Claims.Flock.forkSlack` at `δ ≥ 0`), only level 0's event of a table is at round `0`, and the slack fits the
masked weight's room (`Claims.Flock.SlackFitW`). -/
def ZkSessWJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (Ok : Claims.Msg → ℕ → List 𝔽 → Prop) (x : Claims.Msg) {mPts : ℕ}
    (T : ℕ → TabZK 𝔽 mPts) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j)) (_ : ∀ j i, (rd j i).val = 0 → i = .inr ()) (δ : ℕ → ℝ)
      (_ : ∀ t, 0 ≤ δ t)
      (_ : Claims.Flock.SlackFitW (tables true x).length (fun j => (T j).sch) (fun j => (T j).N) rd δ)
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
                AccUpTo pre.length outs ∧
                  ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
                    ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
              (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ∧
          (∀ r ℓ, (∑ t, advRSZ A Hs (T j) pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω) ∧
          (∀ r : Fin 2, (∑ t, advPadsSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inl r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω) ∧
          (∀ r : Fin 2, (∑ t, advTauSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inr r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω) ∧
          ((∑ t, adv₀SZ A Hs (T j) pre post cap₀ (σ t)) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr ()) ω s.1 s.2) + Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω) ∧
          (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
              SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))
            (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

namespace Gen

variable [S : ZkSetup]

/-- **The masked tables' sessions after `Commit`, off the table on a witness lane, with events on accepted runs**:
`ZkSessJ` with level 0's fork event at each table (`IZW`), every event on accepted deployed runs, and the mixture
bounding the deployed runs by its acceptance up to table `j` with no level-0 opening off `C` on a witness lane unless
it self-clashes. Each fork term's mean `Adv` is at most its fork event's probability and the slack of the fork at its
round (`Claims.Flock.forkSlack` at `δ ≥ 0`), only level 0's event of a table is at round `0`, and the slack fits the
masked weight's room (`Claims.Flock.SlackFitW`). -/
def ZkSessWJ {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
    (A : Model.Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (Ok : Claims.Msg → ℕ → List 𝔽 → Prop) (x : Claims.Msg) {mPts : ℕ}
    (T : ℕ → TabZK 𝔽 mPts) : Prop :=
  ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
    ∃ (rd : (j : ℕ) → Claims.Flock.IZW (T j).sch.levels.length → Fin ((V true).k x))
      (_ : ∀ j, Function.Injective (rd j)) (_ : ∀ j i, (rd j i).val = 0 → i = .inr ()) (δ : ℕ → ℝ)
      (_ : ∀ t, 0 ≤ δ t)
      (_ : Claims.Flock.SlackFitW (tables true x).length (fun j => (T j).sch) (fun j => (T j).N) rd δ)
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
                AccUpTo pre.length outs ∧
                  ¬ (OffW₀Z Hs (T j) cap₀ (Claims.Flock.oracleOf C) (outs.getD pre.length dfltOutZC) ∧
                    ¬ SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC)))
              (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ∧
          (∀ r ℓ, (∑ t, advRSZ A Hs (T j) pre post cap₀ r.val (ℓ.val + 1) (σ t)) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inl (r, ℓ))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inl (r, ℓ)))).val ω) ∧
          (∀ r : Fin 2, (∑ t, advPadsSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inl r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inl r)))).val ω) ∧
          (∀ r : Fin 2, (∑ t, advTauSZ A Hs (T j) pre post cap₀ r.val (σ t)) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inl (.inr (.inr r))) ω s.1 s.2) +
              Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inl (.inr (.inr r)))).val ω) ∧
          ((∑ t, adv₀SZ A Hs (T j) pre post cap₀ (σ t)) / Fintype.card τ ≤
            (Claims.Pr fun s : (Fin ((V true).k x) → Coin) × (Fin ((V true).k x) → Coin) =>
              ev j (.inr ()) ω s.1 s.2) + Claims.Flock.forkSlack sha512 (V true) P x δ (rd j (.inr ())).val ω) ∧
          (∑ t, expect (ind fun outs : List (OutZC 𝔽 D Sl) =>
              SelfClashZ Hs (T j) (outs.getD pre.length dfltOutZC))
            (sessAtZ pre post (tabAfterZC A Hs (T j) cap₀)) (σ t)) / Fintype.card τ ≤
            Claims.Pr fun s : Fin ((V true).k x) → Coin => ev₁ j ω s

/-- **Every masked table's bound at a statement, off the table on a witness lane, the escapes counted per round**:
where table `j` of the setup is `st j` and `T j` is C-Flock ZK's padded table there, when the masked sessions bound
every table at that verdict with events on accepted runs (`ZkSessWJ`) and no table `Ok` allows that does not extract
table `j` is committed to `T j`. The escapes are the tables' with level 0's and the slack's merged by round
(`Claims.Flock.slackEscW`), on accepted first runs (`Claims.Flock.slackEscW_onAcc₁`), fitting the deployed weight per
round (`slackEscW_fitsForkR_zk`), and at each tape and table the bound is `zk_le_sessionsJW`. -/
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
  obtain ⟨rd, hrd, hr0, δ, hδ, hfit, ev, ev₁, tied, tied₁, hacc, hacc₁, hω⟩ := hmix Ω P
  refine ⟨Claims.Flock.slackEscW sha512 (V true) P x (tables true x).length (fun j => (T j).sch)
      (fun j => (T j).N) rd ev ev₁ tied tied₁ δ,
    Claims.Flock.slackEscW_onAcc₁ (tables true x).length (fun j => (T j).sch) (fun j => (T j).N) rd ev ev₁ tied
      tied₁ δ hacc hacc₁,
    slackEscW_fitsForkR_zk rel P x st hst (fun j => (T j).sch) (fun j hj => (hT j hj).2.2.1) (fun j => (T j).N)
      (fun j hj => (hT j hj).2.2.2.2.2.2.2.2.2) rd hr0 ev ev₁ tied tied₁ δ hfit,
    fun ω j hj C hok hg => ?_⟩
  rw [hn] at hj
  obtain ⟨τ, _, _, pre, post, cap₀, σ, hacc', hadv, hP, hTau, h₀, hsc⟩ := hω ω j hj C hok
  exact zk_le_sessionsJW A hA Hs st hst T hT P rd hrd ev ev₁ tied tied₁ δ hδ ω hj C (hnc j hj C hok hg) pre post
    cap₀ σ hacc' hadv hP hTau h₀ hsc

end Gen

/-- **Every masked table's bound at a statement, off the table on a witness lane, the escapes counted per round**:
where table `j` of the setup is `st j` and `T j` is C-Flock ZK's padded table there, when the masked sessions bound
every table at that verdict with events on accepted runs (`ZkSessWJ`) and no table `Ok` allows that does not extract
table `j` is committed to `T j`. The escapes are the tables' with level 0's and the slack's merged by round
(`Claims.Flock.slackEscW`), on accepted first runs (`Claims.Flock.slackEscW_onAcc₁`), fitting the deployed weight per
round (`slackEscW_fitsForkR_zk`), and at each tape and table the bound is `zk_le_sessionsJW`. -/
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
    Claims.Flock.TableBoundJAccAtR sha512 (deployZk rel) (wordLenJ true) Ok x :=
  Gen.zk_tableBoundJAccAtR_of_sessionsW (S := S₀) (K := K) (D := D) (Sl := Sl) (A := A) (hA := hA) (Hs := Hs)
    (rel := rel) (Ok := Ok) (x := x) (hn := hn) (spec := spec) (st := st) (hst := hst) (T := T) (hT := hT) (hnc := hnc)
    (hmix := hmix)

end FlockDeploy.KS
