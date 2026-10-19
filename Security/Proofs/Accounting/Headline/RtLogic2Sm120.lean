import Proofs.Accounting.Headline.Sm120Measured

/-!
# RT2-LOGIC-1: sm_120's FOLD-OPT is asked at every certificate, and the IMAD rotate fails it (red-team witness, not for landing)

Builds on #2482's head (`cursor/lean-headline-sm120-ea24` @ a495f9c5dd), not on main.

`FoldOptFolded` quantifies over every certificate; `price_of_parts_folded` reads it at the statement's one
(`hF lam h`, `Proofs/Accounting/TS/Price.lean:152`), π* = DS_issue for both sm_120 headlines. fx1's step
`z ← rotl(u, r) + y'` computed as `t = u·2^r + y'` (IMAD) and `z = ⌊u·2^r / 2^32⌋ + t` (IMAD.HI.U32) is the same
function for every `r < 32` (RT-LOGIC-10); a word is then LOP3 and four IMAD-pipe instructions, against the menu's
LOP3, LEA.HI and two IMADs. On DS-120:

* `rot_issue`, `menu_issue`: at π* the rotate costs 20 a unit against the menu's 16, so it is slower there, and
  `rot_slower`: the honest program with that fold has an issue load above `B_♭`.
* `rot_mix`, `phiStar_lamMix`: at `λ = ½·issue + ½·lop` it costs 14 against `N_w·φ*(λ) = 16`.
* `not_foldOpt_rot`: no labelling of the honest program with the rotate fold meets `FoldOptFolded` at any
  `folded ≥ 117` of the 128 words.
* `not_premises_rot`: nor does an unfolded share rescue it (round 1's `share_zero`): both headlines' premise sets
  exclude a bit-exact honest program that runs slower than ♭ at π*.

On VF3r-16 (the measured headline), the same at its π* (JI3s_0) and at `λ = ½·issue + ½·A`:

* `rotM_slower`: at π* the honest program with the rotate fold has a load above `B_♭`;
* `rotM_mix`, `phiStar_lamMixM`: at `λ` the rotate costs 14 against `N_w·φ*(λ) = 16`;
* `not_foldOpt_rotM`, `not_premises_rotM`: as on DS-120.
-/

namespace Accounting.SecurityProofs.TS.Headline.RtLogic2

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Accounting.SecurityProofs.TS.Headline
  Pouw.RelativeSecurity Accounting.SecurityProofs.TS

/-- 128 words of fx1 at the IMAD rotate, in warp instructions: 4 ALU (LOP3) and 16 IMAD (IMAD, IMAD, IMAD,
IMAD.HI.U32). -/
def rotFold : Sparse := [(0, 4, 1), (24, 16, 1)]

/-- The honest program with the rotate fold: ♭'s word production and skippable work, the fold `rotFold`. -/
def rotAll : Sparse := Sm120.flatWords ++ rotFold ++ [] ++ Sm120.flatSkip

theorem rot_issue : dot (sheetCoef Sm120.ds120 .issue) 122 rotFold = 20 := by decide +kernel

theorem rot_lop : dot (sheetCoef Sm120.ds120 (.pipe .lop)) 122 rotFold = 8 := by decide +kernel

theorem menu_issue : dot (sheetCoef Sm120.ds120 .issue) 122 Sm120.flatFold = 16 := by decide +kernel

theorem menu_lop : dot (sheetCoef Sm120.ds120 (.pipe .lop)) 122 Sm120.flatFold = 16 := by decide +kernel

/-- At π* the honest program with the rotate fold is slower than ♭: its issue load is ♭'s plus 4. -/
theorem rot_slower : Bq + 4 = dot (sheetCoef Sm120.ds120 .issue) 122 rotAll := by decide +kernel

theorem rotAll_mix :
    1 / 2 * dot (sheetCoef Sm120.ds120 .issue) 122 rotAll + 1 / 2 * dot (sheetCoef Sm120.ds120 (.pipe .lop)) 122 rotAll
      < 1455 / 100 := by
  decide +kernel

theorem fold_mix :
    ∀ i < 2, (1 / 8 : ℚ) ≤ 1 / 2 * dot (sheetCoef Sm120.ds120 .issue) 122 (Sm120.h1fx1.folds.getD i []) +
      1 / 2 * dot (sheetCoef Sm120.ds120 (.pipe .lop)) 122 (Sm120.h1fx1.folds.getD i []) := by
  decide +kernel

/-- The certificate halfway between the issue facet and the LOP pipe's. -/
noncomputable def lamMix : SheetRow Sm120.Pipe → ℝ :=
  fun r => 1 / 2 * onFacet SheetRow.issue r + 1 / 2 * onFacet (SheetRow.pipe Sm120.Pipe.lop) r

theorem demandPrice_lamMix (d : SheetRow Sm120.Pipe → ℝ) :
    demandPrice lamMix d = 1 / 2 * d .issue + 1 / 2 * d (.pipe .lop) := by
  have h1 := demandPrice_onFacet (SheetRow.issue : SheetRow Sm120.Pipe) d
  have h2 := demandPrice_onFacet (SheetRow.pipe Sm120.Pipe.lop) d
  have e : demandPrice lamMix d = 1 / 2 * demandPrice (onFacet SheetRow.issue) d +
      1 / 2 * demandPrice (onFacet (SheetRow.pipe Sm120.Pipe.lop)) d := by
    simp only [demandPrice, lamMix, add_mul, Finset.sum_add_distrib, Finset.mul_sum, mul_assoc]
  rw [e, h1, h2]

theorem isCert_lamMix : IsCert lamMix := by
  obtain ⟨h1, s1⟩ := isCert_onFacet (SheetRow.issue : SheetRow Sm120.Pipe)
  obtain ⟨h2, s2⟩ := isCert_onFacet (SheetRow.pipe Sm120.Pipe.lop)
  refine ⟨fun r => ?_, ?_⟩
  · have a := h1 r
    have b := h2 r
    show 0 ≤ 1 / 2 * onFacet SheetRow.issue r + 1 / 2 * onFacet (SheetRow.pipe Sm120.Pipe.lop) r
    linarith
  · have e : ∑ r, lamMix r = 1 / 2 * ∑ r, onFacet (SheetRow.issue : SheetRow Sm120.Pipe) r +
        1 / 2 * ∑ r, onFacet (SheetRow.pipe Sm120.Pipe.lop) r := by
      simp only [lamMix, Finset.sum_add_distrib, Finset.mul_sum]
    rw [e, s1, s2]
    norm_num

/-- `φ*(λ) ≥ 1/8` at the mixed certificate: each fold route's mixed price. -/
theorem phiStar_lamMix : 1 / 8 ≤ Sm120.model.phiStar lamMix := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_lamMix]
  show (1 : ℝ) / 8 ≤ 1 / 2 * Sm120.model.G.load (Sparse.counts (Sm120.h1fx1.folds.getD φ.val [])) .issue +
    1 / 2 * Sm120.model.G.load (Sparse.counts (Sm120.h1fx1.folds.getD φ.val [])) (.pipe .lop)
  rw [load_counts _ (sheetCoef Sm120.ds120) sm120_hG, load_counts _ (sheetCoef Sm120.ds120) sm120_hG]
  have h : ((1 / 8 : ℚ) : ℝ) ≤ ((1 / 2 * dot (sheetCoef Sm120.ds120 .issue) 122 (Sm120.h1fx1.folds.getD φ.val []) +
      1 / 2 * dot (sheetCoef Sm120.ds120 (.pipe .lop)) 122 (Sm120.h1fx1.folds.getD φ.val []) : ℚ) : ℝ) :=
    Rat.cast_le.mpr (fold_mix φ.val φ.isLt)
  push_cast at h
  linarith

/-- The rotate fold's price at the mixed certificate: 14 a unit, under `N_w·φ*(λ) = 16`. -/
theorem rot_mix : Sm120.model.G.price lamMix (Sparse.counts rotFold) = 14 := by
  show demandPrice lamMix (Sm120.model.G.load (Sparse.counts rotFold)) = 14
  rw [demandPrice_lamMix, load_counts _ (sheetCoef Sm120.ds120) sm120_hG,
    load_counts _ (sheetCoef Sm120.ds120) sm120_hG, rot_issue, rot_lop]
  norm_num

/-- No labelling of the honest program with the rotate fold meets FOLD-OPT per word folded at `folded ≥ 117`. -/
theorem not_foldOpt_rot (A : Anatomy (Fin 122) (Fin 17) Unit Empty) (hn : A.run.n = Sparse.counts rotAll)
    (hA : A.Split) (folded : ℝ) (hf : 117 ≤ folded) : ¬ FoldOptFolded Sm120.model A folded := by
  intro h
  have h1 := h lamMix isCert_lamMix
  have hle : ∀ k, A.nfold k ≤ A.run.n k := fun k => by
    obtain ⟨hw, -, hx, -, hs⟩ := hA
    have := hs k
    have h0 : 0 ≤ ∑ hd, A.nword hd k := Finset.sum_nonneg fun hd _ => hw hd k
    linarith [hx k]
  have hrow : ∀ r, Sm120.model.G.load A.nfold r ≤ Sm120.model.G.load A.run.n r := fun r => by
    unfold Machine.load
    exact Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hle k) (sm120_a_nonneg _ _)
  have hp : Sm120.model.G.price lamMix A.nfold ≤ 1455 / 100 := by
    have e : Sm120.model.G.price lamMix A.nfold =
        1 / 2 * Sm120.model.G.load A.nfold .issue + 1 / 2 * Sm120.model.G.load A.nfold (.pipe .lop) :=
      demandPrice_lamMix (Sm120.model.G.load A.nfold)
    have hi := hrow .issue
    have hl := hrow (.pipe .lop)
    rw [hn, load_counts _ (sheetCoef Sm120.ds120) sm120_hG] at hi
    rw [hn, load_counts _ (sheetCoef Sm120.ds120) sm120_hG] at hl
    have h2 : ((1 / 2 * dot (sheetCoef Sm120.ds120 .issue) 122 rotAll +
        1 / 2 * dot (sheetCoef Sm120.ds120 (.pipe .lop)) 122 rotAll : ℚ) : ℝ) ≤ ((1455 / 100 : ℚ) : ℝ) :=
      Rat.cast_le.mpr rotAll_mix.le
    push_cast at h2
    rw [e]
    linarith
  have hmu : Sm120.model.muF = 0 := rfl
  have hφ := phiStar_lamMix
  rw [hmu, sub_zero, one_mul] at h1
  nlinarith [mul_le_mul hf hφ (by norm_num : (0 : ℝ) ≤ 1 / 8) (by linarith : (0 : ℝ) ≤ folded)]

theorem prob_univ {Ω : Type} [Fintype Ω] [Nonempty Ω] {p : Ω → Prop} (h : ∀ ω, p ω) : prob p = 1 := by
  classical
  have e : prob p = ((Finset.univ.filter p).card : ℝ) / Fintype.card Ω := by
    unfold prob
    congr
  rw [e, Finset.filter_true_of_mem fun ω _ => h ω, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)

/-- Every round's ideal check passes on every coin, whatever the past. -/
def AlwaysPasses {Ω : Type} : {n : ℕ} → ReceiptGame Ω n → Prop
  | _, .done => True
  | _, .round _ pass _ _ next => (∀ ω, pass ω) ∧ ∀ ω, AlwaysPasses (next ω)

noncomputable def somePath {Ω : Type} [Nonempty Ω] : (n : ℕ) → Paths Ω n
  | 0 => (() : Unit)
  | n + 1 => ((Classical.arbitrary Ω, somePath n) : Ω × Paths Ω n)

theorem alwaysPasses_of_accepts {Ω : Type} [Nonempty Ω] :
    ∀ {n : ℕ} (G : ReceiptGame Ω n), (∀ x, G.game.accepts x) → AlwaysPasses G
  | _, .done, _ => trivial
  | n + 1, .round d pass right same next, h => by
      show (∀ ω, pass ω) ∧ ∀ ω, AlwaysPasses (next ω)
      refine ⟨fun ω => ?_, fun ω => alwaysPasses_of_accepts (next ω) fun y => ?_⟩
      · exact (show pass ω ∧ _ from h (ω, somePath n)).1
      · exact (show pass ω ∧ _ from h (ω, y)).2

/-- At a pinned game whose every check always passes, `WholeLuck ½` holds only at unfolded share 0. -/
theorem share_zero {Ω : Type} [Fintype Ω] [Nonempty Ω] :
    ∀ {n : ℕ} (G : ReceiptGame Ω n) (Uf : Unfolded Ω n), G.Pinned → AlwaysPasses G → G.WholeLuck (1 / 2) Uf →
      ∀ x, Uf.share x = 0
  | _, .done, .done, _, _, _, _ => rfl
  | _, .round d pass right same next, .round e vnext, hPin, hA, hW, x => by
      obtain ⟨hpin, hpin'⟩ := hPin
      obtain ⟨hpass, hpass'⟩ := hA
      obtain ⟨⟨he0, -⟩, hsame, hW'⟩ := hW
      have he : e = 0 := by
        by_contra hne
        have h1 := hsame (lt_of_le_of_ne he0 (Ne.symm hne))
        have h2 : prob same = 1 := prob_univ fun ω => hpin ω (hpass ω)
        linarith
      show e + (vnext x.1).share x.2 = 0
      rw [he, zero_add]
      exact share_zero (next x.1) (vnext x.1) (hpin' x.1) (hpass' x.1) (hW' x.1) x.2

/-- **Both sm_120 headlines' premises exclude the honest program with the rotate fold.** For a play of it whose
ideal game accepts every path, and any receipt game and unfolded share the headlines take with it (`Pl.game = G.game`,
`Pinned`, `WholeLuckAt ½`), FOLD-OPT per word folded fails on some path, though at π* the program is slower than ♭. -/
theorem not_premises_rot (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) (hn : ∀ x, (Pl.run x).run.n = Sparse.counts rotAll)
    (hA : ∀ x, (Pl.run x).Split) (hacc : ∀ x, Pl.game.accepts x) (hG : Pl.game = G.game) (hPin : G.Pinned)
    (hWL : WholeLuckAt (1 / 2) G Uf) :
    ¬ ∀ x, FoldOptFolded Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw) := by
  intro h
  have hP : AlwaysPasses G := alwaysPasses_of_accepts G fun x => by
    rw [← hG]
    exact hacc x
  have h0 : Uf.share (somePath Sm120.checks) = 0 := share_zero G Uf hPin hP hWL _
  refine not_foldOpt_rot (Pl.run (somePath Sm120.checks)) (hn _) (hA _) _ ?_ (h _)
  rw [h0, sm120_Nw]
  norm_num

/-! ## VF3r-16: rows 0 (`issue`) and 6 (`A`, ALU at 2), π* row 86 (`JI3s_0`) -/

/-- The honest program with the rotate fold on VF3r-16: ♭'s word production and skippable work there. -/
def rotAllM : Sparse := Sm120.flatWords ++ rotFold ++ [] ++ Sm120Measured.flatSkip

theorem rotM_slower : BqM < dot (vfRow Sm120Measured.piStar) 122 rotAllM := by decide +kernel

theorem rotM_issue : dot (vfRow 0) 122 rotFold = 20 := by decide +kernel

theorem rotM_A : dot (vfRow 6) 122 rotFold = 8 := by decide +kernel

theorem rotAllM_mix : 1 / 2 * dot (vfRow 0) 122 rotAllM + 1 / 2 * dot (vfRow 6) 122 rotAllM < 1456 / 100 := by
  decide +kernel

theorem foldM_mix :
    ∀ i < 2, (1 / 8 : ℚ) ≤ 1 / 2 * dot (vfRow 0) 122 (Sm120Measured.h1fx1.folds.getD i []) +
      1 / 2 * dot (vfRow 6) 122 (Sm120Measured.h1fx1.folds.getD i []) := by decide +kernel

noncomputable def lamMixM : Fin 109 → ℝ :=
  fun r => 1 / 2 * onFacet (0 : Fin 109) r + 1 / 2 * onFacet (6 : Fin 109) r

theorem demandPrice_lamMixM (d : Fin 109 → ℝ) : demandPrice lamMixM d = 1 / 2 * d 0 + 1 / 2 * d 6 := by
  have h1 := demandPrice_onFacet (0 : Fin 109) d
  have h2 := demandPrice_onFacet (6 : Fin 109) d
  have e : demandPrice lamMixM d = 1 / 2 * demandPrice (onFacet (0 : Fin 109)) d +
      1 / 2 * demandPrice (onFacet (6 : Fin 109)) d := by
    simp only [demandPrice, lamMixM, add_mul, Finset.sum_add_distrib, Finset.mul_sum, mul_assoc]
  rw [e, h1, h2]

theorem isCert_lamMixM : IsCert lamMixM := by
  obtain ⟨h1, s1⟩ := isCert_onFacet (0 : Fin 109)
  obtain ⟨h2, s2⟩ := isCert_onFacet (6 : Fin 109)
  refine ⟨fun r => ?_, ?_⟩
  · have a := h1 r
    have b := h2 r
    show 0 ≤ 1 / 2 * onFacet (0 : Fin 109) r + 1 / 2 * onFacet (6 : Fin 109) r
    linarith
  · have e : ∑ r, lamMixM r = 1 / 2 * ∑ r, onFacet (0 : Fin 109) r + 1 / 2 * ∑ r, onFacet (6 : Fin 109) r := by
      simp only [lamMixM, Finset.sum_add_distrib, Finset.mul_sum]
    rw [e, s1, s2]
    norm_num

theorem phiStar_lamMixM : 1 / 8 ≤ Sm120Measured.model.phiStar lamMixM := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_lamMixM]
  show (1 : ℝ) / 8 ≤ 1 / 2 * Sm120Measured.model.G.load (Sparse.counts (Sm120Measured.h1fx1.folds.getD φ.val [])) 0 +
    1 / 2 * Sm120Measured.model.G.load (Sparse.counts (Sm120Measured.h1fx1.folds.getD φ.val [])) 6
  rw [load_counts _ vfRow sm120m_hG, load_counts _ vfRow sm120m_hG]
  have h : ((1 / 8 : ℚ) : ℝ) ≤ ((1 / 2 * dot (vfRow 0) 122 (Sm120Measured.h1fx1.folds.getD φ.val []) +
      1 / 2 * dot (vfRow 6) 122 (Sm120Measured.h1fx1.folds.getD φ.val []) : ℚ) : ℝ) :=
    Rat.cast_le.mpr (foldM_mix φ.val φ.isLt)
  push_cast at h
  linarith

theorem rotM_mix : Sm120Measured.model.G.price lamMixM (Sparse.counts rotFold) = 14 := by
  show demandPrice lamMixM (Sm120Measured.model.G.load (Sparse.counts rotFold)) = 14
  rw [demandPrice_lamMixM, load_counts _ vfRow sm120m_hG, load_counts _ vfRow sm120m_hG, rotM_issue, rotM_A]
  norm_num

theorem not_foldOpt_rotM (A : Anatomy (Fin 122) (Fin 15) Unit Empty) (hn : A.run.n = Sparse.counts rotAllM)
    (hA : A.Split) (folded : ℝ) (hf : 117 ≤ folded) : ¬ FoldOptFolded Sm120Measured.model A folded := by
  intro h
  have h1 := h lamMixM isCert_lamMixM
  have hle : ∀ k, A.nfold k ≤ A.run.n k := fun k => by
    obtain ⟨hw, -, hx, -, hs⟩ := hA
    have := hs k
    have h0 : 0 ≤ ∑ hd, A.nword hd k := Finset.sum_nonneg fun hd _ => hw hd k
    linarith [hx k]
  have hrow : ∀ r, Sm120Measured.model.G.load A.nfold r ≤ Sm120Measured.model.G.load A.run.n r := fun r => by
    unfold Machine.load
    exact Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hle k) (sm120m_a_nonneg _ _)
  have hp : Sm120Measured.model.G.price lamMixM A.nfold ≤ 1456 / 100 := by
    have e : Sm120Measured.model.G.price lamMixM A.nfold =
        1 / 2 * Sm120Measured.model.G.load A.nfold 0 + 1 / 2 * Sm120Measured.model.G.load A.nfold 6 :=
      demandPrice_lamMixM (Sm120Measured.model.G.load A.nfold)
    have hi := hrow 0
    have hl := hrow 6
    rw [hn, load_counts _ vfRow sm120m_hG] at hi
    rw [hn, load_counts _ vfRow sm120m_hG] at hl
    have h2 : ((1 / 2 * dot (vfRow 0) 122 rotAllM + 1 / 2 * dot (vfRow 6) 122 rotAllM : ℚ) : ℝ) ≤
        ((1456 / 100 : ℚ) : ℝ) :=
      Rat.cast_le.mpr rotAllM_mix.le
    push_cast at h2
    rw [e]
    linarith
  have hmu : Sm120Measured.model.muF = 0 := rfl
  have hφ := phiStar_lamMixM
  rw [hmu, sub_zero, one_mul] at h1
  nlinarith [mul_le_mul hf hφ (by norm_num : (0 : ℝ) ≤ 1 / 8) (by linarith : (0 : ℝ) ≤ folded)]

/-- **The measured headline's premises exclude the honest program with the rotate fold**, as `not_premises_rot`. -/
theorem not_premises_rotM (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty) (G : ReceiptGame Ω Sm120.checks)
    (Uf : Unfolded Ω Sm120.checks) (hn : ∀ x, (Pl.run x).run.n = Sparse.counts rotAllM)
    (hA : ∀ x, (Pl.run x).Split) (hacc : ∀ x, Pl.game.accepts x) (hG : Pl.game = G.game) (hPin : G.Pinned)
    (hWL : WholeLuckAt (1 / 2) G Uf) :
    ¬ ∀ x, FoldOptFolded Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) := by
  intro h
  have hP : AlwaysPasses G := alwaysPasses_of_accepts G fun x => by
    rw [← hG]
    exact hacc x
  have h0 : Uf.share (somePath Sm120.checks) = 0 := share_zero G Uf hPin hP hWL _
  refine not_foldOpt_rotM (Pl.run (somePath Sm120.checks)) (hn _) (hA _) _ ?_ (h _)
  rw [h0, sm120m_Nw]
  norm_num

end Accounting.SecurityProofs.TS.Headline.RtLogic2

#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.rot_slower
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.isCert_lamMix
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.phiStar_lamMix
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.rot_mix
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.not_foldOpt_rot
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.not_premises_rot
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.rotM_slower
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.phiStar_lamMixM
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.rotM_mix
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.not_foldOpt_rotM
#print axioms Accounting.SecurityProofs.TS.Headline.RtLogic2.not_premises_rotM
