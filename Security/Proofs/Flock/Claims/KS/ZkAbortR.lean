import Proofs.Flock.Claims.KS.ZkSimW
import Proofs.Flock.Claims.Recursion.FlockKSRecord

/-!
# The recursion's KS over the masked record, from the masked table bound counted per round

The recursion's chain at the abort law (`KSAbort_of_tableBoundOnAcc`) reads the table bound with escapes on accepted
runs fitting as `Escapes.FitsFork` asks, every fork event charged at the busiest round's count, which the masked
escapes with level 0's fork event do not fit at the deployed weight (`ZkFitsW`). Counted per round
(`Claims.Recursion.TableBoundθOnAccR`, at every statement `Flock.TableBoundθAccAtR`, which `TableBoundθOnAcc` gives:
`Claims.Recursion.tableBoundθOnAccR_of_onAcc`), the same chain holds once the abort-law escapes with the conflict fork
are merged by round (`Claims.Recursion.EscapesD.mergeR`, as `Escapes.mergeR`): the abort-law bound carries the
per-round fit (`Claims.Recursion.TableBoundθAbortR`, `Claims.Recursion.tableBoundθAbortR_of_onAccR`), the merged
escapes fit as `OneWaitAbort` asks with a term no smaller
(`Claims.Recursion.EscapesD.fitsR_addFork`, `Claims.Recursion.EscapesD.FitsR.fits_mergeR`,
`Claims.Recursion.EscapesD.term_le_mergeR`, `Claims.Recursion.oneWaitAbort_of_tableBoundR`), and so Flock meets
`Inner.KSAbort` (`Claims.Recursion.KSAbort_of_tableBoundOnAccR`). At the masked record the claims read that is
`FlockDeploy.KS.zk_KSAbortR`, and the recursion's KS over it `FlockDeploy.KS.zk_KSFlockR`, each from the bound over
full-width tables at the untrimmed record carried by padding back (`Claims.Recursion.tableBoundθOnAccR_mapCols`) and
one table wherever some run accepts; the bound follows from a simulator of each masked prover
(`FlockDeploy.KS.zk_tableBoundθOnAccR_of_simWJ`, `FlockDeploy.KS.zk_KSAbort_of_simWJ`,
`FlockDeploy.KS.zk_KSFlock_of_simWJ`).
-/

namespace Claims.Recursion

open Claims Claims.Flock FlockSoundness FlockSoundness.Game FlockSoundness.Extract FlockSoundness.Rewinding

/-! ## Fork events over draws, merged by round -/

section MergeD

variable {Ω Coin A : Type} {H : Msg → Msg} {n : ℕ} {tr : Ω → (Fin n → Coin) × A → Trace}

/-- **The escapes fit `C` counted per round**: `Σ_i m_i c_i ≤ 2C`, with `m_i` the fork events at `i`'s round, and
`Σ_l β_l ≤ n`. -/
def EscapesD.FitsR (e : EscapesD H n tr) (C : ℝ) : Prop :=
  ∑ i, ((Finset.univ.filter fun i' => e.j i' = e.j i).card : ℝ) * e.c i ≤ 2 * C ∧ ∑ l, e.β l ≤ n

/-- **The escapes fit `C` with one more fork event** of weight `c` at round `0`, each counted at its own round:
`(m_0 + 1)·c + Σ_i m'_i c_i ≤ 2C`, with `m_0` the fork events at round `0` and `m'_i` those at `i`'s round counting the
extra one, and `Σ_l β_l ≤ n`. -/
def EscapesD.FitsForkR (e : EscapesD H n tr) (C c : ℝ) : Prop :=
  (((Finset.univ.filter fun i => (e.j i).val = 0).card : ℝ) + 1) * c +
      ∑ i, (((Finset.univ.filter fun i' => e.j i' = e.j i).card : ℝ) + if (e.j i).val = 0 then 1 else 0) * e.c i ≤
    2 * C ∧ ∑ l, e.β l ≤ n

/-- **With the extra fork event at round `0`, escapes fitting with it per round fit per round.** -/
theorem EscapesD.fitsR_addFork (e : EscapesD H n tr) (C c : ℝ) (hc : 0 ≤ c) (j₀ : Fin n) (hj₀ : j₀.val = 0)
    (ev : Ω → (Fin n → Coin) × A → (Fin n → Coin) × A → Prop) (tied) (h : e.FitsForkR C c) :
    (e.addFork c hc j₀ ev tied).FitsR C := by
  obtain ⟨hmc, hβ⟩ := h
  refine ⟨le_trans (le_of_eq ?_) hmc, hβ⟩
  have hcard : ∀ r : Fin n, ((Finset.univ : Finset (Option e.I)).filter fun o => o.elim j₀ e.j = r).card =
      (if r.val = 0 then 1 else 0) + (Finset.univ.filter fun i => e.j i = r).card := fun r => by
    refine (Claims.Flock.card_filter_option _ (fun o : Option e.I => o.elim j₀ e.j) r).trans ?_
    have hj : j₀ = r ↔ r.val = 0 := by rw [Fin.ext_iff, hj₀, eq_comm]
    simp only [Option.elim_none, hj]
    rfl
  show ∑ o : Option e.I,
      (((Finset.univ : Finset (Option e.I)).filter fun o' => o'.elim j₀ e.j = o.elim j₀ e.j).card : ℝ) *
        o.elim c e.c = _
  rw [Claims.Flock.sum_option_any]
  show (((Finset.univ : Finset (Option e.I)).filter fun o => o.elim j₀ e.j = j₀).card : ℝ) * c +
    ∑ a, (((Finset.univ : Finset (Option e.I)).filter fun o => o.elim j₀ e.j = e.j a).card : ℝ) * e.c a = _
  rw [hcard]
  simp only [hcard]
  have h0 : (Finset.univ.filter fun i => e.j i = j₀) = Finset.univ.filter fun i => (e.j i).val = 0 :=
    Finset.filter_congr fun i _ => by rw [Fin.ext_iff, hj₀]
  rw [h0, hj₀]
  simp only [↓reduceIte]
  push_cast
  exact congrArg₂ (· + ·) (by ring) (Finset.sum_congr rfl fun i _ => by ring)

/-- **The escapes' fork events merged by round**: at each round one event, that some event at the round holds, of
weight the round's count of events times their total weight. -/
def EscapesD.mergeR (e : EscapesD H n tr) : EscapesD H n tr where
  I := Fin n
  c r := ((Finset.univ.filter fun i => e.j i = r).card : ℝ) * ∑ i ∈ Finset.univ.filter (fun i => e.j i = r), e.c i
  j r := r
  ev r ω d d' := ∃ i, e.j i = r ∧ e.ev i ω d d'
  L := e.L
  β := e.β
  ev₁ := e.ev₁
  c_nonneg _ := mul_nonneg (Nat.cast_nonneg _) (Finset.sum_nonneg fun i _ => e.c_nonneg i)
  β_nonneg := e.β_nonneg
  tied r ω d d' := by
    rintro ⟨i, rfl, h⟩
    exact e.tied i ω d d' h
  tied₁ := e.tied₁

/-- **Merged by round, escapes that fit `C` per round fit it** with one event per round. -/
theorem EscapesD.FitsR.fits_mergeR {e : EscapesD H n tr} {C : ℝ} (h : e.FitsR C) : e.mergeR.Fits C := by
  refine ⟨1, fun r => ?_, ?_, h.2⟩
  · show (Finset.univ.filter fun r' : Fin n => r' = r).card ≤ 1
    exact Finset.card_le_one.2 fun a ha b hb => by
      rw [(Finset.mem_filter.1 ha).2, (Finset.mem_filter.1 hb).2]
  · show ((1 : ℕ) : ℝ) * ∑ r : Fin n, ((Finset.univ.filter fun i => e.j i = r).card : ℝ) *
        ∑ i ∈ Finset.univ.filter (fun i => e.j i = r), e.c i ≤ 2 * C
    rw [Nat.cast_one, one_mul]
    refine le_trans (le_of_eq ?_) h.1
    rw [← Finset.sum_fiberwise Finset.univ e.j
      (fun i => ((Finset.univ.filter fun i' => e.j i' = e.j i).card : ℝ) * e.c i)]
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i hi => by rw [(Finset.mem_filter.1 hi).2]

/-- **Merging by round only raises the escapes' term**: at each round, by Cauchy–Schwarz,
`Σ_i √(c_i·Pr[ev_i]) ≤ √(n·Σ_i c_i·Pr[∃ i, ev_i])` over its `n` events. -/
theorem EscapesD.term_le_mergeR [Fintype Coin] [Fintype A] (e : EscapesD H n tr) (ω : Ω) :
    e.term ω ≤ e.mergeR.term ω := by
  unfold EscapesD.term
  refine add_le_add ?_ le_rfl
  show _ ≤ ∑ r : Fin n, Real.sqrt ((((Finset.univ.filter fun i => e.j i = r).card : ℝ) *
      ∑ i ∈ Finset.univ.filter (fun i => e.j i = r), e.c i) *
      Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => ∃ i, e.j i = r ∧ e.ev i ω s.1 s.2)
  rw [← Finset.sum_fiberwise Finset.univ e.j]
  refine Finset.sum_le_sum fun r _ => ?_
  set F := Finset.univ.filter fun i => e.j i = r with hF
  set Q := Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => ∃ i, e.j i = r ∧ e.ev i ω s.1 s.2 with hQ
  have hQ0 : 0 ≤ Q := Pr_nonneg _
  have hc0 : 0 ≤ ∑ i ∈ F, e.c i := Finset.sum_nonneg fun i _ => e.c_nonneg i
  calc ∑ i ∈ F, Real.sqrt (e.c i * Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => e.ev i ω s.1 s.2)
      ≤ ∑ i ∈ F, Real.sqrt 1 * Real.sqrt (e.c i) * Real.sqrt Q := Finset.sum_le_sum fun i hi => by
        rw [Real.sqrt_one, one_mul, ← Real.sqrt_mul (e.c_nonneg i)]
        exact Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
          (Pr_mono fun s h => ⟨i, (Finset.mem_filter.1 hi).2, h⟩) (e.c_nonneg i))
    _ = (∑ i ∈ F, Real.sqrt 1 * Real.sqrt (e.c i)) * Real.sqrt Q := by rw [Finset.sum_mul]
    _ ≤ (Real.sqrt (∑ _i ∈ F, (1 : ℝ)) * Real.sqrt (∑ i ∈ F, e.c i)) * Real.sqrt Q :=
        mul_le_mul_of_nonneg_right (Real.sum_sqrt_mul_sqrt_le F (fun _ => zero_le_one) fun i => e.c_nonneg i)
          (Real.sqrt_nonneg _)
    _ = Real.sqrt (((F.card : ℝ) * ∑ i ∈ F, e.c i) * Q) := by
        rw [Finset.sum_const, nsmul_eq_mul, mul_one, Real.sqrt_mul (mul_nonneg (Nat.cast_nonneg _) hc0),
          Real.sqrt_mul (Nat.cast_nonneg _)]

end MergeD

/-! ## The chain at the abort law, counted per round -/

section TableBound

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **`TableBoundθOnAcc` with the escapes counted per round** (`Escapes.FitsForkR`): `Flock.TableBoundθAccAtR` at
every statement. -/
def TableBoundθOnAccR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) : Prop :=
  ∀ x, Flock.TableBoundθAccAtR H D N Ok x

/-- **`TableBoundθOnAcc` gives it**: an event charged at the busiest round's count is charged at least its own
round's (`TableBoundθAccAt.tableBoundθAccAtR`). -/
theorem tableBoundθOnAccR_of_onAcc {H : Msg → Msg} {D : Flock.Deploy Coin Col T} {N : Msg → ℕ}
    {Ok : Msg → Col → Prop} (htab : TableBoundθOnAcc H D N Ok) (hVk : ∀ x, 0 < D.V.k x) :
    TableBoundθOnAccR H D N Ok :=
  fun x => Flock.TableBoundθAccAt.tableBoundθAccAtR (fun Ω _ _ P => htab x Ω P) (hVk x)

/-- **`TableBoundθAbort` with the escapes counted per round** (`EscapesD.FitsForkR`). -/
def TableBoundθAbortR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (Ok : Msg → Col → Prop) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (A : Type) [Fintype A] [Nonempty A]
    (keep : Ω → (Fin (D.V.k x) → Coin) → A → Bool),
    ∃ e : EscapesD H (D.V.k x) (abortRun H P D.V x keep), e.FitsForkR (D.weight x) (D.k x) ∧
      ∀ (ω : Ω) (C : Fin (N x) → Col), (∀ p, Ok x (C p)) → ¬ GoodTable D N x C →
        accPrD (abortRun H P D.V x keep ω) ≤ D.εtable x + e.term ω +
          Pr fun d : (Fin (D.V.k x) → Coin) × A => (abortRun H P D.V x keep ω d).out = .acc ∧
            ∃ (p : Fin (N x)) (u : Col),
              lookCol (D.cols (abortRun H P D.V x keep ω d).seen 0) p.val = some u ∧ u ≠ C p

/-- **The abort-law table bound counted per round, from the table bound with escapes on accepted runs counted per
round**: `tableBoundθAbort_of_onAcc`, whose escapes at the abort law have the same rounds and weights. -/
theorem tableBoundθAbortR_of_onAccR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (Ok : Msg → Col → Prop) (decC : Msg → Msg → Coin) (htab : TableBoundθOnAccR H D N Ok) (hVk : ∀ x, 0 < D.V.k x)
    (hpc : CoinsShown H D.V decC) (hlast : AccLast H D.V) (hfree : DecideFree H D.V) (href : Refusable H D.V) :
    TableBoundθAbortR H D N Ok := by
  classical
  intro x Ω _ _ P A _ _ keep
  obtain ⟨m₀, hm₀⟩ := href x
  obtain ⟨e, ⟨hon, hon₁⟩, hfit, hb⟩ :=
    htab x (Ω × ((Fin (D.V.k x) → Coin) → Coin × A)) (refuser P x (D.V.k x) decC keep m₀)
  refine ⟨Escapes.toAbortD e, hfit, fun ω C hC hg => ?_⟩
  exact accPrD_le_of_refuser H P D.V x decC (hVk x) hpc hlast hfree keep m₀ hm₀ e hon hon₁ (D.εtable x)
    (fun s => ∃ (p : Fin (N x)) (u : Col), lookCol (D.cols s 0) p.val = some u ∧ u ≠ C p) ω
    (fun g => hb (ω, g) C hC hg)

/-- **Flock's one-wait bound at the abort law, from the table's bound there counted per round**:
`oneWaitAbort_of_tableBound`, the escapes with the conflict fork merged by round (`EscapesD.mergeR`). -/
theorem oneWaitAbort_of_tableBoundR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x) (hVk : ∀ x, 0 < D.V.k x)
    (hk : ∀ x, 1 ≤ D.k x ∨ NoAcc H D.V x)
    (hone : ∀ x, (∃ s, D.parse x = some s ∧ s.tables = 1) ∨ NoAcc H D.V x) (Ok : Msg → Col → Prop) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N) (htab : TableBoundθAbortR H D N Ok) :
    OneWaitAbort H D := by
  intro x Ω _ _ P A _ _ keep
  obtain ⟨e₀, hfit, hb⟩ := htab x Ω P A keep
  refine ⟨(e₀.addFork (D.k x) (Nat.cast_nonneg _) ⟨0, hVk x⟩
      (fun ω d d' => ∃ p, ConflictAt (opD D N x (abortRun H P D.V x keep ω)) (d, d') p)
      (opD_tied H D N hconf P x keep ⟨0, hVk x⟩ rfl)).mergeR,
    (e₀.fitsR_addFork (D.weight x) (D.k x) (Nat.cast_nonneg _) ⟨0, hVk x⟩ rfl
      (fun ω d d' => ∃ p, ConflictAt (opD D N x (abortRun H P D.V x keep ω)) (d, d') p)
      (opD_tied H D N hconf P x keep ⟨0, hVk x⟩ rfl) hfit).fits_mergeR, fun ω hε => ?_⟩
  have hm := (e₀.addFork (D.k x) (Nat.cast_nonneg _) ⟨0, hVk x⟩
    (fun ω d d' => ∃ p, ConflictAt (opD D N x (abortRun H P D.V x keep ω)) (d, d') p)
    (opD_tied H D N hconf P x keep ⟨0, hVk x⟩ rfl)).term_le_mergeR ω
  have := oneWaitAbort_le_of_tableBound H D N hN hεt hVk hk hone Ok hz hok hconf x P keep e₀ hb ω hε
  linarith

/-- **Flock as the inner protocol meets `Inner.KSAbort`, from the KS chain's obligations counted per round**:
`KSAbort_of_tableBoundOnAcc` at `TableBoundθOnAccR`. -/
theorem KSAbort_of_tableBoundOnAccR (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ)
    (Ok : Msg → Col → Prop) (decC : Msg → Msg → Coin) (hN : ∀ x, (N x : ℝ) ≤ D.N₀ x) (hεt : ∀ x, 0 ≤ D.εtable x)
    (hVk : ∀ x, 0 < D.V.k x) (hK : ∀ x, 0 < D.cap x) (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x)
    (hone : ∀ x, (∃ s, D.parse x = some s ∧ s.tables = 1) ∨ NoAcc H D.V x) (hz : ∀ x, Ok x 0)
    (hok : ColsOk H D N Ok) (hconf : ConflictTied H D N) (htab : TableBoundθOnAccR H D N Ok)
    (hpc : CoinsShown H D.V decC) (hlast : AccLast H D.V) (hfree : DecideFree H D.V) (href : Refusable H D.V)
    (decode : List (ℕ × Col) → List T) (hdec : decOK D decode) : (Inner.ofFlock D decode).KSAbort H :=
  KSAbort_of_oneWaitAbort H D hK hk hεt
    (oneWaitAbort_of_tableBoundR H D N hN hεt hVk (fun x => (hk x).imp (fun h => by omega) id) hone Ok hz hok hconf
      (tableBoundθAbortR_of_onAccR H D N Ok decC htab hVk hpc hlast hfree href)) decode hdec

/-- **`TableBoundθOnAccR` through `f`, by padding back**, as `tableBoundθOnAcc_mapCols`. -/
theorem tableBoundθOnAccR_mapCols (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (N : Msg → ℕ) (f : Col → Col)
    (hf : f 0 = 0) (Ok : Msg → Col → Prop) (hok : ColsOk H D N Ok)
    (hinj : ∀ x u v, Ok x u → Ok x v → f u = f v → u = v) (htab : TableBoundθOnAccR H D N Ok) :
    TableBoundθOnAccR H (D.mapCols f) N (OkThrough f Ok) := by
  intro x Ω _ _ P
  obtain ⟨e, hon, hfit, hb⟩ := htab x Ω P
  refine ⟨e, hon, hfit, fun ω C' hC' hg' => ?_⟩
  choose C hC hfC using hC'
  have hg : ¬ GoodTable D N x C := fun h => hg' (by
    have := goodTable_mapCols D N x f hf C h
    rwa [show (fun p => f (C p)) = C' from funext hfC] at this)
  refine (hb ω C hC hg).trans (add_le_add le_rfl (Pr_mono fun r ⟨ha, p, u, hu, hne⟩ => ⟨ha, p, f u, ?_, ?_⟩))
  · show lookCol ((D.cols (run H P D.V x ω r).seen 0).map fun q => (q.1, f q.2)) p.val = some (f u)
    rw [lookCol_map, hu]
    rfl
  · rw [← hfC p]
    exact fun he => hne (hinj x u (C p) (hok x Ω P ω r p.val u ha p.isLt hu) (hC p) he)

end TableBound

end Claims.Recursion

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **The masked (`--zk`) record the claims read meets `Inner.KSAbort`**, at SHA-512, at every decoder within the
radius: `zk_KSAbort` from the table bound with escapes on accepted runs counted per round over full-width tables at the
untrimmed instance, and one table wherever the verifier accepts some run. -/
theorem zk_KSAbortR (rel : Readable.Parsed → Readable.Relation)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (htab : TableBoundθOnAccR sha512 (deployZk rel) (wordLen true) (wide true))
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace)
    (hdec : decOK (deployZkTrim rel) decode) : (Inner.ofFlock (deployZkTrim rel) decode).KSAbort sha512 :=
  KSAbort_of_tableBoundOnAccR sha512 (deployZkTrim rel) (wordLen true) (trimWide true) decCoin
    (wordLen_le_N₀ true) (fun x => εtable_nonneg true x) (V_k_pos true) (fun _ => cap_pos)
    (zk_kSplits rel hone) hone (trimWide_nil true)
    (Claims.Flock.colsOk_mapCols sha512 (deployZk rel) (wordLen true) trimCol (wide true) (zk_colsOk sha512 rel))
    (Claims.Flock.conflictTied_mapCols sha512 (deployZk rel) (wordLen true) trimCol (zk_conflictTied rel))
    (tableBoundθOnAccR_mapCols sha512 (deployZk rel) (wordLen true) trimCol rfl (wide true)
      (zk_colsOk sha512 rel) (wide_trimCol_inj true) htab)
    (coinsShown_V sha512 true) (accLast_V_sha512 true) (decideFree_V sha512 true) (refusable_V sha512 true)
    decode hdec

/-- **KS of the recursion over the masked (`--zk`) record the claims read as the inner protocol**, at SHA-512:
`zk_KSFlock` with `Inner.KSAbort` from `zk_KSAbortR`. -/
theorem zk_KSFlockR (rel : Readable.Parsed → Readable.Relation) (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hone : ∀ x, (∃ s, (deployZk rel).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk rel).V x)
    (htab : TableBoundθOnAccR sha512 (deployZk rel) (wordLen true) (wide true)) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim rel).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim rel) decI) DO) (hS : SessionAccepts D)
    (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim rel) decI) DO) (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim rel) DO :=
  (KSFlockA_iff_KSFlock _ _ _ _).1 fun decI h => KS_of_flockKS sha512 D _ DO opening decCoin hF
    (zk_KSAbortR rel hone htab decI h) hRef hRounds hN hε (hRd decI) hS (hR decI) (hV decI) (hT decI) (hL decI)
    (hO decI) (hAcc decI) (coinsShown_V sha512 true) hN0

/-- **The recursion's masked table bound counted per round**, at full-width columns, from a simulator of each prover
at the statements some run accepts (`zk_tableBoundθAccAtR_of_simWJ`). -/
theorem zk_tableBoundθOnAccR_of_simWJ (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x) :
    TableBoundθOnAccR sha512 (deployZk Readable.relation) (wordLen true) (wide true) :=
  zk_tableBoundθAccAtR_of_simWJ hs

/-- **The masked record the claims read meets `Inner.KSAbort`** at the deployed relation, from a simulator of each
prover at the statements some run accepts and one table wherever some run accepts. -/
theorem zk_KSAbort_of_simWJ
    (hone : ∀ x, (∃ s, (deployZk Readable.relation).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk Readable.relation).V x)
    (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x)
    (decode : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace)
    (hdec : decOK (deployZkTrim Readable.relation) decode) :
    (Inner.ofFlock (deployZkTrim Readable.relation) decode).KSAbort sha512 :=
  zk_KSAbortR Readable.relation hone (zk_tableBoundθOnAccR_of_simWJ hs) decode hdec

/-- **KS of the recursion over the masked record the claims read** at the deployed relation, from a simulator of each
prover at the statements some run accepts and one table wherever some run accepts (`zk_KSFlockR`). -/
theorem zk_KSFlock_of_simWJ (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hone : ∀ x, (∃ s, (deployZk Readable.relation).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployZk Readable.relation).V x)
    (hs : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimWJ x) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim Readable.relation).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim Readable.relation) DO :=
  zk_KSFlockR Readable.relation D DO opening hF hone (zk_tableBoundθOnAccR_of_simWJ hs) hRef hRounds hN hε hRd hS hR
    hV hT hL hO hAcc hN0

end FlockDeploy.KS
