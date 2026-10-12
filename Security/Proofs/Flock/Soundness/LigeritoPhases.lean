import Proofs.Flock.Soundness.LigeritoDecode
import Proofs.Flock.Soundness.LigeritoFolds
import Proofs.Flock.Soundness.LigeritoOOD
import Proofs.Flock.Soundness.LigeritoQueries

/-!
# Ligerito level by level (PROTOCOL.md §13.3)

The doomed states between a level's steps:
* `PreFold`: before a level's folds, the running claim is false for every message of the level's
  list, against the weight on its split table (`preW`);
* `PostFold`: after the folds by `ch`, it is false for every fold of those messages, against the
  next stage's weight (`resAt`), and no fold round was an MCA event.

`phaseB` is the folds (sumcheck and MCA), costing `foldError`.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- Before level `l`'s folds, with `e + 1` variables on its split table. -/
structure PreFold (A : Arith F K) (n : ℕ) (b : ℕ → F) (initial : List K) (l : Level)
    (C : Oracle F) (e : ℕ) (st : LigState F K) : Prop where
  sum : st.q.eval 0 + st.q.eval 1 = st.s
  ok : PreOK initial st.levels st.oods st.cons e
  dim : e + 1 = l.lanes + l.logCols
  doom : ∀ M ∈ closeL A l.radius l.lanes l.logCols l.logLen C,
    st.s ≠ ∑ y ∈ Finset.range (2 ^ (e + 1)),
      A.embed (flat M y) * preW A n b initial st.levels st.oods st.cons e y

/-- After level `l`'s folds by `ch` (the last entry of `st.levels`, or `initial` at level 0). -/
structure PostFold (A : Arith F K) (hA : A.Correct) (n : ℕ) (b : ℕ → F) (initial : List K)
    (l : Level) (hf : l.Fits) (lanes : ℕ) (C : Oracle F) (ch : List K) (st : LigState F K) :
    Prop where
  sum : st.q.eval 0 + st.q.eval 1 = st.s
  len : ch.length = lanes
  ok : PostOK initial st.levels st.oods st.cons l.logCols
  mca : MCAGood (codeK A hA l hf) l.radius lanes ch (rowsK A lanes l.logLen C)
  doom : ∀ M ∈ closeL A l.radius lanes l.logCols l.logLen C,
    st.s ≠ ∑ x ∈ Finset.range (2 ^ l.logCols),
      foldMsg A ch M x * resAt A n b initial st.levels st.oods st.cons l.logCols x

theorem value_foldRounds_length (A : Arith F K) (hA : A.Correct) :
    ∀ (k : ℕ) (s : K) (q : Quad K),
      value (fun out : List K × K × Quad K => out.1.length ≠ k) (foldRounds A k s q) = 0
  | 0, s, q => by
      show value _ (Game.ret ([], s, q)) = 0
      simp
  | k + 1, s, q => by
      rw [foldRounds_succ]
      refine le_antisymm ((value_drawK_le A hA _ _ (fun _ => False) 0 0 (by simp [prCoin, avg])
        fun r _ => ?_).trans (by simp)) zero_le
      rw [value_send]
      refine iSup_le fun msg => ?_
      rw [value_bind_ret]
      refine le_of_eq ?_
      rw [← value_foldRounds_length A hA k (Quad.eval q r) (fromMsg msg (Quad.eval q r))]
      exact value_congr (fun out => by simp) _

theorem foldError_bound (l : Level) (L : ℕ) (hL : 0 ≤ l.mcaA) :
    ((2 * l.lanes * L : ℕ) : ℝ≥0∞) / Fintype.card K +
        ((2 ^ l.lanes - 1 : ℕ) : ℝ≥0∞) * ENNReal.ofReal (l.mcaA / qK) ≤
      ENNReal.ofReal (l.foldError L) ∨ Fintype.card K ≠ 2 ^ 256 := by
  by_cases hK : Fintype.card K = 2 ^ 256
  · left
    unfold Level.foldError
    have hq : (0 : ℝ) < qK := by unfold qK; positivity
    rw [ENNReal.ofReal_add (by positivity) (div_nonneg (mul_nonneg (by
      have : (1 : ℝ) ≤ 2 ^ l.lanes := one_le_pow₀ (by norm_num)
      linarith) hL) hq.le)]
    gcongr
    · rw [hK, show ((2 ^ 256 : ℕ) : ℝ≥0∞) = ENNReal.ofReal qK by
        unfold qK; rw [ENNReal.ofReal_pow (by norm_num)]; norm_num,
        ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_div_of_pos hq]
      refine ENNReal.ofReal_le_ofReal (le_of_eq ?_)
      push_cast; ring
    · rw [← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity)]
      refine ENNReal.ofReal_le_ofReal (le_of_eq ?_)
      rw [Nat.cast_sub Nat.one_le_two_pow]
      push_cast; ring
  · exact Or.inr hK

theorem mcaA_nonneg (l : Level) (hl : l.Ok) : 0 ≤ l.mcaA := (mcaA_bounds l hl.2.1 hl.2.2).1

/-- **A level's folds.** From a doomed claim before the folds, after them the claim is false for
every fold of the level's list and no round was an MCA event, except with probability
`foldError`. -/
theorem phaseB (A : Arith F K) (hA : A.Correct) (n : ℕ)
    (b : ℕ → F) (initial : List K) (l : Level) (hl : l.Ok) (hf : l.Fits) (C : Oracle F) (e : ℕ)
    (st : LigState F K) (hpre : PreFold A n b initial l C e st) (hlanes : 1 ≤ l.lanes) :
    value (fun out : List K × K × Quad K => ¬ PostFold A hA n b initial l hf l.lanes C out.1
        { st with s := out.2.1, q := out.2.2, levels := st.levels ++ [out.1] })
      (foldRounds A l.lanes st.s st.q) ≤ ENNReal.ofReal (l.foldError l.listBound) := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  set S := closeL A l.radius l.lanes l.logCols l.logLen C
  have hS : S.card ≤ l.listBound := card_close_le A hA l hf l.lanes (tbl l.lanes l.logLen C)
  set u : Message F l.lanes l.logCols → ℕ → K := fun M y => A.embed (flat M y)
  set v := preW A n b initial st.levels st.oods st.cons e
  have hdim : e + 1 = l.lanes + l.logCols := hpre.dim
  have himp : ∀ out : List K × K × Quad K, ¬ PostFold A hA n b initial l hf l.lanes C out.1
      { st with s := out.2.1, q := out.2.2, levels := st.levels ++ [out.1] } →
      (out.1.length ≠ l.lanes ∨ out.2.2.eval 0 + out.2.2.eval 1 ≠ out.2.1) ∨
        (¬ MCAGood (codeK A hA l hf) l.radius l.lanes out.1 (rowsK A l.lanes l.logLen C) ∨
          ∃ M ∈ S, out.2.1 = ∑ y ∈ Finset.range (2 ^ l.logCols),
            foldLoL out.1 (u M) y * foldLoL out.1 v y) := by
    intro out hno
    by_contra hall
    simp only [not_or, not_not, not_exists, not_and] at hall
    obtain ⟨⟨hlen, hsum⟩, hmca, hdoom⟩ := hall
    obtain ⟨c, cs, hch⟩ : ∃ c cs, out.1 = c :: cs := by
      cases h : out.1 with
      | nil => rw [h] at hlen; simp at hlen; omega
      | cons c cs => exact ⟨c, cs, rfl⟩
    have hcs : e = cs.length + l.logCols := by
      rw [hch] at hlen; simp at hlen; omega
    refine hno ⟨hsum, hlen, ?_, hmca, fun M hM heq => hdoom M hM ?_⟩
    · rw [hch]; exact postOK_of_preOK initial st.levels st.oods st.cons e c cs l.logCols hcs hpre.ok
    · have heq' : out.2.1 = _ := heq
      rw [heq']
      refine Finset.sum_congr rfl fun y hy => ?_
      rw [hch, foldLoL_preW A n b initial st.levels st.oods st.cons e c cs l.logCols hcs hpre.ok y
        (Finset.mem_range.1 hy)]
      rfl
  have hmcaε : ∀ U : Fin 2 → Fin (2 ^ l.logLen) → K,
      prCoin (fun γ : K => IsMCA (AffineLineGenerator K) (codeK A hA l hf) γ U l.radius) ≤
        ENNReal.ofReal (l.mcaA / qK) := prCoin_level_mca_le A hA l hl hf
  have hsumε := value_foldRounds_sum_le A hA S u v l.lanes l.logCols st.s st.q hpre.sum
    (fun M hM => by rw [← hdim]; exact hpre.doom M hM)
  calc value _ (foldRounds A l.lanes st.s st.q)
      ≤ value (fun out : List K × K × Quad K =>
          (out.1.length ≠ l.lanes ∨ out.2.2.eval 0 + out.2.2.eval 1 ≠ out.2.1) ∨
            (¬ MCAGood (codeK A hA l hf) l.radius l.lanes out.1 (rowsK A l.lanes l.logLen C) ∨
              ∃ M ∈ S, out.2.1 = ∑ y ∈ Finset.range (2 ^ l.logCols),
                foldLoL out.1 (u M) y * foldLoL out.1 v y)) (foldRounds A l.lanes st.s st.q) :=
        value_mono himp _
    _ ≤ (value (fun out : List K × K × Quad K => out.1.length ≠ l.lanes) _ +
          value (fun out : List K × K × Quad K => out.2.2.eval 0 + out.2.2.eval 1 ≠ out.2.1) _) +
        (value (fun out : List K × K × Quad K =>
            ¬ MCAGood (codeK A hA l hf) l.radius l.lanes out.1 (rowsK A l.lanes l.logLen C)) _ +
          value (fun out : List K × K × Quad K => ∃ M ∈ S, out.2.1 =
            ∑ y ∈ Finset.range (2 ^ l.logCols), foldLoL out.1 (u M) y * foldLoL out.1 v y) _) :=
        (value_or_le _ _ _).trans (add_le_add (value_or_le _ _ _) (value_or_le _ _ _))
    _ ≤ (0 + 0) + (((2 ^ l.lanes - 1 : ℕ) : ℝ≥0∞) * ENNReal.ofReal (l.mcaA / qK) +
          ((2 * l.lanes * S.card : ℕ) : ℝ≥0∞) / Fintype.card K) :=
        add_le_add (add_le_add (value_foldRounds_length A hA _ _ _).le
          (value_foldRounds_sum_invariant A hA _ _ _ hpre.sum).le)
          (add_le_add (value_foldRounds_mca_le A hA (codeK A hA l hf) l.radius _ hmcaε l.lanes _ _ _)
            hsumε)
    _ ≤ ENNReal.ofReal (l.foldError l.listBound) := by
        rw [zero_add, zero_add, add_comm]
        rcases foldError_bound (K := K) l l.listBound (mcaA_nonneg l hl) with h | h
        · refine le_trans (add_le_add ?_ le_rfl) h
          gcongr
        · exact absurd hA.card_K h

end FlockSoundness
