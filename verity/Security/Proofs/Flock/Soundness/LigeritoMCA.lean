import Proofs.Flock.Soundness.LigeritoQueries
import Proofs.Flock.Soundness.JohnsonMCA
import VCVio.OracleComp.Constructions.SampleableType.NativeMeasure

/-!
# Mutual correlated agreement at Ligerito's folds (PROTOCOL.md §13.3)

From mutual correlated agreement up to the Johnson bound (`mcaError_le_bchks25`, proved from DKT26)
to Ligerito's coins:

* **The coin bound** (`prCoin_mca_le`): for a Reed–Solomon code in the Johnson range over any finite
  field, whatever two words the prover fixes, a uniform coin makes the MCA event happen with
  probability at most `a/|L|`. This is `mcaError_le_bchks25` read through `prCoin`
  (VCVio's `prEvent_uniformSample`).
* **One fold round** (`rowsAgreeOn_of_fold`, `prCoin_fold_mca_le`): a round folds lanes `2ℓ` and
  `2ℓ + 1` into `R(2ℓ) + r·(R(2ℓ+1) + R(2ℓ))`, an affine line. Outside the rounds' MCA events, if
  every folded row agrees with a codeword on a large set `T`, so does every row, on the same `T`.
  One round costs (number of pairs)·`a/|L|`.
* **All rounds** (`rowsAgreeOn_of_foldAll`): folding `2^k` lanes to one by `k` challenges, lowest
  lane bit first; the fold is the `eq`-weighted sum of the lanes (`foldAllRows_eq_sum`, the weights
  of `Model.foldColumn`). Paper Lemma 9's row union is the sum of the rounds' pair counts,
  `2^k − 1`.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal NNReal

/-! ## The coin bound -/

/-- **Mutual correlated agreement, read as a coin bound.** For a Reed–Solomon code of `k`
coefficients on `n` points and a radius `0 ≤ δ < 1 − √((k − 1)/n)`, whatever two words the prover
fixes, a uniform `γ` makes the MCA event happen with probability at most `a/|L|`, BCHKS25's printed
bound (`mcaError_le_bchks25`). -/
theorem prCoin_mca_le {n : ℕ} {L : Type} [Field L] [Fintype L] [DecidableEq L]
    (domain : Fin n ↪ L) (k : ℕ) (δ : ℝ≥0) (hk : 1 < k)
    (hδJ : (δ : ℝ) < 1 - ((((k - 1 : ℕ) : ℝ) / Fintype.card (Fin n)) ^ ((1 : ℝ) / 2)))
    (U : Fin 2 → Fin n → L) :
    prCoin (fun γ : L => IsMCA (AffineLineGenerator L) (ReedSolomon.code domain k) γ U δ) ≤
      ENNReal.ofReal
        (let n : ℝ := Fintype.card (Fin n)
         let ρ : ℝ := (k - 1 : ℕ) / n
         let m : ℝ := max ⌈(ρ ^ ((1 : ℝ) / 2)) / (1 - ρ ^ ((1 : ℝ) / 2) - δ)⌉ 3
         ((2 * (m + 1/2) ^ 5 + 3 * (m + 1/2) * δ * ρ) / (3 * ρ ^ ((3 : ℝ) / 2)) * n
          + (m + 1/2) / ρ ^ ((1 : ℝ) / 2)) / (Fintype.card L : ℝ)) := by
  classical
  letI : SampleableType L := SampleableType.ofFintype L
  have h := mcaError_le_bchks25 domain k δ hk hδJ
  refine le_trans ?_ h
  have hU : Pr{let γ ← $ᵗ L}[IsMCA (AffineLineGenerator L) (ReedSolomon.code domain k) γ U δ] ≤
      mcaError (AffineLineGenerator L) (ReedSolomon.code domain k) (δ : ℝ) :=
    le_iSup (fun U : Fin 2 → (Fin n → L) =>
      Pr{let γ ← $ᵗ L}[IsMCA (AffineLineGenerator L) (ReedSolomon.code domain k) γ U δ]) U
  refine le_trans (le_of_eq ?_) hU
  rw [SampleableType.prEvent_uniformSample, prCoin_eq_card]

/-! ## One fold round -/

variable {ι : Type} [Fintype ι] {L : Type} [Field L] [Fintype L] [DecidableEq L]

/-- Every row agrees with a codeword on `T`. -/
def RowsAgreeOn (code : Submodule L (ι → L)) (T : Finset ι) {m : ℕ} (R : Fin m → ι → L) : Prop :=
  ∀ ℓ, ∃ c ∈ code, ∀ i ∈ T, c i = R ℓ i

/-- One fold round: lanes `2ℓ` and `2ℓ + 1` fold into `R(2ℓ) + r·(R(2ℓ+1) + R(2ℓ))`. -/
def foldRows {m : ℕ} (r : L) (R : Fin (2 * m) → ι → L) : Fin m → ι → L := fun ℓ i =>
  R ⟨2 * ℓ.val, by omega⟩ i + r * (R ⟨2 * ℓ.val + 1, by omega⟩ i + R ⟨2 * ℓ.val, by omega⟩ i)

/-- The two words folded into row `ℓ`, on the affine line `u₀ + r·u₁`. -/
def foldPair {m : ℕ} (R : Fin (2 * m) → ι → L) (ℓ : Fin m) : Fin 2 → ι → L :=
  ![R ⟨2 * ℓ.val, by omega⟩, fun i => R ⟨2 * ℓ.val + 1, by omega⟩ i + R ⟨2 * ℓ.val, by omega⟩ i]

omit [Fintype L] [DecidableEq L] in
theorem mem_projected_of_agree (code : Submodule L (ι → L)) (T : Finset ι) (v c : ι → L)
    (hc : c ∈ code) (h : ∀ i ∈ T, c i = v i) :
    LinearCode.projectedWord v T ∈ LinearCode.projectedCodeSubmod code T :=
  Submodule.mem_map.2 ⟨c, hc, funext fun i => h i i.property⟩

omit [DecidableEq L] in
/-- **One fold round, deterministically.** If no pair of this round is an MCA event at `r`, and
every folded row agrees with a codeword on a set `T` of at least `(1 − δ)n` positions, then every
row agrees with a codeword on `T`. -/
theorem rowsAgreeOn_of_fold [CharP L 2] (code : Submodule L (ι → L)) (δ : ℝ) {m : ℕ} (r : L)
    (R : Fin (2 * m) → ι → L) (hr : ∀ ℓ, ¬ IsMCA (AffineLineGenerator L) code r (foldPair R ℓ) δ)
    (T : Finset ι) (hT : (T.card : ℝ) ≥ Fintype.card ι * (1 - δ))
    (hfold : RowsAgreeOn code T (foldRows r R)) : RowsAgreeOn code T R := by
  have hpair : ∀ ℓ : Fin m, ∃ p : Fin 2 → code, ∀ j i, i ∈ T → (p j).val i = foldPair R ℓ j i := by
    intro ℓ
    obtain ⟨c, hc, hcT⟩ := hfold ℓ
    refine (not_isMCA_iff_forall_exists_codewords (AffineLineGenerator L) code r (foldPair R ℓ) δ).1
      (hr ℓ) T hT (mem_projected_of_agree code T _ c hc fun i hi => ?_)
    rw [hcT i hi]
    simp [foldRows, foldPair, Fin.sum_univ_two]
  intro ℓ₀
  obtain ⟨p, hp⟩ := hpair ⟨ℓ₀.val / 2, by omega⟩
  rcases Nat.even_or_odd ℓ₀.val with ⟨q, hq⟩ | ⟨q, hq⟩
  · refine ⟨(p 0).val, (p 0).property, fun i hi => ?_⟩
    rw [hp 0 i hi]
    simp only [foldPair, Matrix.cons_val_zero]
    congr 1
    ext
    simp only
    omega
  · refine ⟨(p 1).val + (p 0).val, code.add_mem (p 1).property (p 0).property, fun i hi => ?_⟩
    rw [Pi.add_apply, hp 1 i hi, hp 0 i hi]
    simp only [foldPair, Matrix.cons_val_one, Matrix.cons_val_zero]
    have h1 : (⟨2 * (ℓ₀.val / 2) + 1, by omega⟩ : Fin (2 * m)) = ℓ₀ := Fin.ext (by simp only; omega)
    rw [h1, add_assoc, Rounds.add_self, add_zero]

/-- **One fold round, the coin.** Given a bound `ε` on each pair's MCA event, the round's MCA event
over all `m` pairs has probability at most `m·ε`. -/
theorem prCoin_fold_mca_le (code : Submodule L (ι → L)) (δ : ℝ) {m : ℕ} (R : Fin (2 * m) → ι → L)
    (ε : ℝ≥0∞)
    (hε : ∀ U : Fin 2 → ι → L, prCoin (fun γ : L => IsMCA (AffineLineGenerator L) code γ U δ) ≤ ε) :
    prCoin (fun r : L => ∃ ℓ : Fin m, IsMCA (AffineLineGenerator L) code r (foldPair R ℓ) δ) ≤
      m * ε := by
  calc prCoin (fun r : L => ∃ ℓ : Fin m, IsMCA (AffineLineGenerator L) code r (foldPair R ℓ) δ)
      ≤ prCoin (fun r : L => ∃ ℓ ∈ (Finset.univ : Finset (Fin m)),
          IsMCA (AffineLineGenerator L) code r (foldPair R ℓ) δ) :=
        prCoin_mono fun r ⟨ℓ, h⟩ => ⟨ℓ, Finset.mem_univ ℓ, h⟩
    _ ≤ ∑ ℓ ∈ (Finset.univ : Finset (Fin m)), prCoin (fun r : L =>
          IsMCA (AffineLineGenerator L) code r (foldPair R ℓ) δ) :=
        prCoin_exists_le _ _
    _ ≤ ∑ _ℓ ∈ (Finset.univ : Finset (Fin m)), ε := Finset.sum_le_sum fun ℓ _ => hε _
    _ = m * ε := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-! ## All rounds -/

theorem two_pow_succ' (k : ℕ) : 2 ^ (k + 1) = 2 * 2 ^ k := by ring

/-- `2^(k+1)` rows as `2·2^k` rows. -/
def splitRows {k : ℕ} (R : Fin (2 ^ (k + 1)) → ι → L) : Fin (2 * 2 ^ k) → ι → L :=
  fun ℓ => R (Fin.cast (two_pow_succ' k).symm ℓ)

/-- Fold `2^k` rows to one by the challenges `ch`, lowest lane bit first (as `Model.foldColumn`
weighs them). -/
def foldAllRows : (k : ℕ) → List L → (Fin (2 ^ k) → ι → L) → (ι → L)
  | 0, _, R => R ⟨0, by norm_num⟩
  | _ + 1, [], _ => 0
  | k + 1, r :: rs, R => foldAllRows k rs (foldRows r (splitRows R))

/-- No round of the fold by `ch` is an MCA event. -/
def MCAGood (code : Submodule L (ι → L)) (δ : ℝ) : (k : ℕ) → List L → (Fin (2 ^ k) → ι → L) → Prop
  | 0, _, _ => True
  | _ + 1, [], _ => False
  | k + 1, r :: rs, R =>
      (∀ ℓ, ¬ IsMCA (AffineLineGenerator L) code r (foldPair (splitRows R) ℓ) δ) ∧
        MCAGood code δ k rs (foldRows r (splitRows R))

omit [DecidableEq L] in
/-- **All rounds, deterministically.** If no round is an MCA event and the fully folded row agrees
with a codeword on a set `T` of at least `(1 − δ)n` positions, every row agrees with a codeword on
`T`. -/
theorem rowsAgreeOn_of_foldAll [CharP L 2] (code : Submodule L (ι → L)) (δ : ℝ) :
    ∀ (k : ℕ) (ch : List L) (R : Fin (2 ^ k) → ι → L), MCAGood code δ k ch R →
      ∀ T : Finset ι, (T.card : ℝ) ≥ Fintype.card ι * (1 - δ) →
        (∃ c ∈ code, ∀ i ∈ T, c i = foldAllRows k ch R i) → RowsAgreeOn code T R
  | 0, ch, R, _, T, _, ⟨c, hc, hcT⟩ => fun ℓ => ⟨c, hc, fun i hi => by
      rw [hcT i hi]
      simp only [foldAllRows]
      congr 1
      exact Fin.ext (by have := ℓ.isLt; simp only [pow_zero, Nat.lt_one_iff] at this; omega)⟩
  | _ + 1, [], _, hgood, _, _, _ => absurd hgood (by simp [MCAGood])
  | k + 1, r :: rs, R, ⟨hr, hgood⟩, T, hT, hfin => by
      have h1 := rowsAgreeOn_of_foldAll code δ k rs _ hgood T hT hfin
      have h2 := rowsAgreeOn_of_fold code δ r (splitRows R) hr T hT h1
      intro ℓ
      obtain ⟨c, hc, hcT⟩ := h2 (Fin.cast (two_pow_succ' k) ℓ)
      exact ⟨c, hc, fun i hi => by rw [hcT i hi]; rfl⟩

omit [Fintype ι] [Fintype L] [DecidableEq L] in
theorem eqList_cons {R : Type} [CommRing R] (a : R) (l : List R) (j : ℕ) :
    eqList (a :: l) j = (if j.testBit 0 then a else 1 + a) * eqList l (j / 2) := by
  unfold eqList
  rw [List.zipIdx_cons, List.map_cons, List.prod_cons]
  congr 1
  rw [List.zipIdx_succ, List.map_map]
  congr 1
  refine List.map_congr_left fun x _ => ?_
  simp only [Function.comp, Nat.testBit_succ]

omit [Fintype ι] [Fintype L] [DecidableEq L] in
/-- The full fold is the `eq`-weighted sum of the rows (the weights of `Model.foldColumn`). -/
theorem foldAllRows_eq_sum [CharP L 2] : ∀ (k : ℕ) (ch : List L), ch.length = k →
    ∀ (R : Fin (2 ^ k) → ι → L) (i : ι),
      foldAllRows k ch R i = ∑ ℓ : Fin (2 ^ k), eqList ch ℓ.val * R ℓ i
  | 0, ch, hch, R, i => by
      rw [List.length_eq_zero_iff.1 hch]
      simp [foldAllRows, eqList]
  | k + 1, [], hch, R, i => by simp at hch
  | k + 1, r :: rs, hch, R, i => by
      simp only [List.length_cons, add_left_inj] at hch
      rw [foldAllRows, foldAllRows_eq_sum k rs hch]
      set Rn : ℕ → L := fun j => if h : j < 2 ^ (k + 1) then R ⟨j, h⟩ i else 0 with hRn
      have hR : ∀ j : Fin (2 ^ (k + 1)), R j i = Rn j.val := fun j => by simp [hRn, j.isLt]
      have hL : ∀ ℓ : Fin (2 ^ k), foldRows r (splitRows R) ℓ i =
          Rn (2 * ℓ.val) + r * (Rn (2 * ℓ.val + 1) + Rn (2 * ℓ.val)) := fun ℓ => by
        simp only [foldRows, splitRows]
        rw [hR, hR]
        rfl
      simp_rw [hL, hR]
      rw [Fin.sum_univ_eq_sum_range (fun j => eqList (r :: rs) j * Rn j) (2 ^ (k + 1)),
        sum_range_pairs,
        Fin.sum_univ_eq_sum_range
          (fun y => eqList rs y * (Rn (2 * y) + r * (Rn (2 * y + 1) + Rn (2 * y)))) (2 ^ k)]
      refine Finset.sum_congr rfl fun y _ => ?_
      have h0 : (2 * y).testBit 0 = false := by simp [Nat.testBit_zero]
      have h1 : (2 * y + 1).testBit 0 = true := by simp [Nat.testBit_zero]
      rw [eqList_cons, eqList_cons, h0, h1, show 2 * y / 2 = y by omega,
        show (2 * y + 1) / 2 = y by omega]
      simp only [Bool.false_eq_true, if_false, if_true]
      ring

/-! ## The folds' MCA events in the game -/

section Game

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

theorem foldRounds_succ (A : Arith F K) (k : ℕ) (s : K) (q : Quad K) :
    foldRounds A (k + 1) s q = (drawK A).bind fun r =>
      Game.send (K × K) fun msg =>
        (foldRounds A k (q.eval r) (fromMsg msg (q.eval r))).bind fun out =>
          Game.ret (r :: out.1, out.2.1, out.2.2) := by
  rw [foldRounds]
  rfl

/-- **The folds' MCA events.** Over the `k` rounds of `foldRounds` on fixed rows, some round is an
MCA event with probability at most `(2^k − 1)·ε`, whatever the prover sends: the rows are fixed
before the folds, so each round's event depends only on the earlier challenges. -/
theorem value_foldRounds_mca_le (A : Arith F K) (hA : A.Correct) {ι : Type} [Fintype ι]
    (code : Submodule K (ι → K)) (δ : ℝ) (ε : ℝ≥0∞)
    (hε : ∀ U : Fin 2 → ι → K, prCoin (fun γ : K => IsMCA (AffineLineGenerator K) code γ U δ) ≤ ε) :
    ∀ (k : ℕ) (R : Fin (2 ^ k) → ι → K) (s : K) (q : Quad K),
      value (fun out => ¬ MCAGood code δ k out.1 R) (foldRounds A k s q) ≤
        ((2 ^ k - 1 : ℕ) : ℝ≥0∞) * ε
  | 0, R, s, q => by
      show value _ (Game.ret ([], s, q)) ≤ _
      simp [MCAGood]
  | k + 1, R, s, q => by
      rw [foldRounds_succ]
      refine (value_drawK_le A hA _ _ (fun r => ∃ ℓ : Fin (2 ^ k),
          IsMCA (AffineLineGenerator K) code r (foldPair (splitRows R) ℓ) δ)
          ((2 ^ k : ℕ) * ε) (((2 ^ k - 1 : ℕ) : ℝ≥0∞) * ε)
          (prCoin_fold_mca_le code δ (splitRows R) ε hε) fun r hr => ?_).trans ?_
      · rw [value_send]
        refine iSup_le fun msg => ?_
        rw [value_bind_ret]
        refine (value_mono (fun out h => ?_) _).trans
          (value_foldRounds_mca_le A hA code δ ε hε k (foldRows r (splitRows R)) _ _)
        simp only [MCAGood, not_and] at h
        exact h (fun ℓ hℓ => hr ⟨ℓ, hℓ⟩)
      · rw [← add_mul, ← Nat.cast_add]
        gcongr
        have : 1 ≤ 2 ^ k := Nat.one_le_two_pow
        rw [pow_succ]
        omega

end Game

end FlockSoundness
