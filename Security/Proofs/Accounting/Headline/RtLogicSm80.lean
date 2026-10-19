import Proofs.Accounting.Headline.Sm80

/-!
# RT-LOGIC-10: sm_80's FOLD-OPT fails at the IMAD rotate (red-team witness, not for landing)

Duo's step `z ← rotl(u, r) + y` computed as `t = u·2^r + y` (IMAD) and `z = ⌊u·2^r / 2^32⌋ + t` (IMAD.HI.U32, whose
addend pair is `(0, t)`) moves its ALU instruction to the FMA pipe; it is the same function for every `r < 32`. nvcc
13.0 at `-arch=sm_80` compiles 128 words of it to LOP3, IMAD, IMAD, IMAD.HI.U32 and one MOV (the pair's low half) a
word, against the menu's LOP3, IMAD and LEA.HI (r20261019-062436-826f). Priced on DS-80 at π*, the ALU facet:

* `imadRot_alu`: 128 words of it cost 8 on the ALU facet; the menu's route costs 16 (`N_w·φ*`, `sm80_phiStar`).
* `mixAll`: ♭ with 3/16 of its words folded so; `mix_rows`, `mix_bound`: every row of it is at most `0.912·B_♭`.
* `not_foldOpt_mix`: no labelling of that run meets `FoldOptFolded` at any `folded ≥ 117` of the 128 words, so
  `Sm80SheetHeadline` says nothing about an honest program the model floors at 91.2% of `T_cred`.
-/

namespace Accounting.SecurityProofs.TS.Headline.RtLogic

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Accounting.SecurityProofs.TS.Headline

/-- 128 words of duo at the IMAD rotate, in warp instructions: 4 ALU (LOP3), 12 IMAD (IMAD, IMAD, IMAD.HI.U32) and
4 NOROW (MOV). -/
def imadRotFold : Sparse := [(0, 4, 1), (16, 12, 1), (32, 4, 1)]

theorem imadRot_alu : dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 imadRotFold = 8 := by decide +kernel

theorem menu_alu : dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 Sm80.flatFold = 16 := by decide +kernel

/-- ♭'s fold with 3/16 of its 128 words at the IMAD rotate: `13/16` of `flatFold` plus `3/16` of `imadRotFold`. -/
def mixFold : Sparse := [(0, 29, 4), (16, 11, 2), (32, 3, 4)]

/-- The honest program at the mixed fold: ♭'s word production and skippable work, the fold `mixFold`. -/
def mixAll : Sparse := Sm80.flatWords ++ mixFold ++ [] ++ Sm80.flatSkip

theorem mix_rows (r : SheetRow Sm80.Pipe) : dot (sheetCoef Sm80.ds80 r) 35 mixAll ≤ 912 / 1000 * Bq80 := by
  cases r with
  | issue => decide +kernel
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

theorem mix_alu : dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 mixAll < 1453 / 100 := by decide +kernel

/-- The model's floor for the mixed program is at most `0.912·B_♭`. -/
theorem mix_bound : Sm80.model.G.bound (Sparse.counts mixAll) ≤ 912 / 1000 * Sm80.model.bnd Sm80.flat := by
  rw [sm80_bnd]
  unfold Machine.bound
  refine ciSup_le fun r => ?_
  rw [load_counts _ (sheetCoef Sm80.ds80) sm80_hG]
  have h := mix_rows r
  have h' : ((dot (sheetCoef Sm80.ds80 r) 35 mixAll : ℚ) : ℝ) ≤ ((912 / 1000 * Bq80 : ℚ) : ℝ) := Rat.cast_le.mpr h
  push_cast at h'
  exact h'

/-- No labelling of the mixed program meets FOLD-OPT per word folded at `folded ≥ 117`. -/
theorem not_foldOpt_mix (A : Anatomy (Fin 35) (Fin 23) Unit Empty) (hn : A.run.n = Sparse.counts mixAll)
    (hA : A.Split) (folded : ℝ) (hf : 117 ≤ folded) : ¬ FoldOptFolded Sm80.model A folded := by
  intro h
  have h1 := h (onFacet Sm80.piStar) (isCert_onFacet _)
  have hle : ∀ k, A.nfold k ≤ A.run.n k := fun k => by
    obtain ⟨hw, -, hx, -, hs⟩ := hA
    have := hs k
    have h0 : 0 ≤ ∑ hd, A.nword hd k := Finset.sum_nonneg fun hd _ => hw hd k
    linarith [hx k]
  have hp : Sm80.model.G.price (onFacet Sm80.piStar) A.nfold ≤ 1453 / 100 := by
    have e : Sm80.model.G.price (onFacet Sm80.piStar) A.nfold = Sm80.model.G.load A.nfold Sm80.piStar :=
      demandPrice_onFacet Sm80.piStar (Sm80.model.G.load A.nfold)
    rw [e]
    calc Sm80.model.G.load A.nfold Sm80.piStar ≤ Sm80.model.G.load A.run.n Sm80.piStar := by
          unfold Machine.load
          exact Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hle k) (sm80_a_nonneg _ _)
      _ = ((dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 mixAll : ℚ) : ℝ) := by
          rw [hn, load_counts _ (sheetCoef Sm80.ds80) sm80_hG]
      _ ≤ 1453 / 100 := by
          have h2 : ((dot (sheetCoef Sm80.ds80 Sm80.piStar) 35 mixAll : ℚ) : ℝ) ≤ ((1453 / 100 : ℚ) : ℝ) :=
            Rat.cast_le.mpr mix_alu.le
          push_cast at h2
          exact h2
  have hmu : Sm80.model.muF = 0 := rfl
  have hφ := sm80_phiStar
  rw [hmu, sub_zero, one_mul] at h1
  nlinarith [mul_le_mul hf hφ (by norm_num : (0 : ℝ) ≤ 1 / 8) (by linarith : (0 : ℝ) ≤ folded)]

end Accounting.SecurityProofs.TS.Headline.RtLogic

#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic.imadRot_alu
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic.menu_alu
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic.mix_rows
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic.mix_bound
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic.not_foldOpt_mix
