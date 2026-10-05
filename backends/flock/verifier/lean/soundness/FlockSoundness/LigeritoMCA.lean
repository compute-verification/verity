import FlockSoundness.LigeritoQueries
import FlockSoundness.JohnsonMCA
import VCVio.OracleComp.Constructions.SampleableType.NativeMeasure
import FlockSoundness.Protocol.LigeritoMCA

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
