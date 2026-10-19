import Definitions.Accounting.HeadlineDc
import Models.Accounting.Assumptions.Fill
import Models.Accounting.Assumptions.TS
import Proofs.Accounting.Headline.Encoding
import Proofs.Accounting.TS.Price
import Proofs.Accounting.TS.Sheet

/-!
# RT-LOGIC-12 and -13: #2699's B200 fold and H200 filler are priced at work no check reads, and the H200's measured menu is a cut (red-team witness, not for landing)

Builds on #2699's head (`cursor/int-hb-ea24` @ f3c3237e8a; e12b32444a's data are the same), not on main.

* The B200 (`Sm100`, DS-100). The design's fold is li with 3/8 of an extra keyed round a word (ALU 29/1024, IMAD
  7/256, SHFL 3/4096 a word; ♭'s 928 ALU, 896 IMAD, 24 SHFL a unit). The checks read plain li (`PouwLiAbsorb`, in
  `PouwWarpCheck_v1` and `_v2`), which the honest kernel `dc100-licpw-redux-l32` runs at 544 ALU, 512 IMAD, 24 SHFL a
  unit. `plain_bound`: the model floors that program at `0.589·B_♭`. `not_foldOpt_plain`: no labelling of it meets
  `FoldOptFolded` at `folded ≥ 19300` of the 32768 words. `not_premises_plain`: with a pinned game whose every check
  passes, `WholeLuckAt ½` forces the unfolded share to 0, so `Sm100SheetHeadline`'s FOLD-OPT premise fails on it.
* The H200 (`Sm90`, DS-90). The design credits ffill at `p = 0` ("LI-CP-W's last checkpoint absorbs" its two chain
  words), but no Definition absorbs them. `noFill_bound`: ♭ without its filler is floored at `0.939·B_♭`.
  `not_optPair_noFill`: no labelling of it meets both FOLD-OPT at `N_w` words and FILL-OPT at share 1.
  `not_premises_noFill`: with the same game, `Sm90SheetHeadline`'s two premises fail on it together.
  `b1_unrated` (RT-LOGIC-15): the two b1 tensor forms are writers at tensor cost `0 / 0 = 0`; `b1_pistar`: π* reads no
  tensor row, so that zero moves no headline price.
* The H200 on DC-90-ri (`Sm90Measured`). Its writer kinds are honest's `IGMMA_S8_N256` and `UNI`; the rows price every
  other tensor form at the scalar coefficient, so `NoSum` carries them. `imma_bound` and `n128_bound`: ♭ moved mostly to
  mma.sync m16n8k32 s8, or wholly to wgmma s8 at N = 128, is floored at `0.875·B_♭` and `0.72·B_♭`.
  `not_noSum_at`: `NoSum` fails on any program whose JI90_1 load is below its parts' price there, under every
  labelling A6, FOLD-OPT and FILL-OPT allow; `not_noSum_imma` and `not_noSum_n128` are its two instances.
-/

namespace Accounting.SecurityProofs.TS.HeadlineDc.RtLogic

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Accounting.SecurityProofs.TS.Headline
  Pouw.RelativeSecurity

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

/-! ### The B200 at DS-100 -/

abbrev iss100 : SheetRow Sm100.Pipe := .issue

/-- li as the checks read it and the honest kernel runs it, a unit of 32768 words: 544 ALU (0.53125 a lane word),
512 IMAD (0.5), 24 SHFL (the redux and the store). -/
def plainFold : Sparse := [(0, 544, 1), (2, 512, 1), (14, 24, 1)]

/-- ♭ with the fold the checks read: its word production and skippable work, the fold `plainFold`. -/
def plainAll : Sparse := Sm100.flatWords ++ plainFold ++ [] ++ Sm100.flatSkip

def flatAll100 : Sparse := Sm100.flatWords ++ Sm100.flatFold ++ Sm100.flatFill ++ Sm100.flatSkip

theorem sm100_hG : ∀ r k, Sm100.model.G.a r k = (sheetCoef Sm100.ds100 r k.val : ℝ) := sheet_a Sm100.ds100 26

theorem sm100_a_nonneg : ∀ r k, 0 ≤ Sm100.model.G.a r k := fun r k => by
  rw [sm100_hG]
  exact Rat.cast_nonneg.mpr (sheetCoef_nonneg _ _ _)

theorem sm100_Nw : Sm100.model.Nw = 32768 := by
  show ((32768 : ℕ) : ℝ) = 32768
  norm_num

/-- ♭'s bound, on the ALU facet. -/
def Bq100 : ℚ := dot (sheetCoef Sm100.ds100 (.pipe .alu)) 26 flatAll100

theorem sm100_max (r : SheetRow Sm100.Pipe) : dot (sheetCoef Sm100.ds100 r) 26 flatAll100 ≤ Bq100 := by
  cases r with
  | issue => decide +kernel
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

theorem sm100_bnd : Sm100.model.bnd Sm100.flat = (Bq100 : ℝ) := by
  unfold Model.bnd
  rw [show Sm100.flat.n = Sparse.counts flatAll100 from
    labelled_n 26 Sm100.flatWords Sm100.flatFold Sm100.flatFill Sm100.flatSkip]
  exact bound_eq _ (sheetCoef Sm100.ds100) sm100_hG flatAll100 (.pipe .alu) sm100_max

theorem plain_rows (r : SheetRow Sm100.Pipe) : dot (sheetCoef Sm100.ds100 r) 26 plainAll ≤ 589 / 1000 * Bq100 := by
  cases r with
  | issue => decide +kernel
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

/-- The model's floor for ♭ at the checked fold is at most `0.589·B_♭`. -/
theorem plain_bound : Sm100.model.G.bound (Sparse.counts plainAll) ≤ 589 / 1000 * Sm100.model.bnd Sm100.flat := by
  rw [sm100_bnd]
  unfold Machine.bound
  refine ciSup_le fun r => ?_
  rw [load_counts _ (sheetCoef Sm100.ds100) sm100_hG]
  have h := plain_rows r
  have h' : ((dot (sheetCoef Sm100.ds100 r) 26 plainAll : ℚ) : ℝ) ≤ ((589 / 1000 * Bq100 : ℚ) : ℝ) :=
    Rat.cast_le.mpr h
  push_cast at h'
  exact h'

theorem sm100_fold_issue : ∀ i < 2, (231 / 4096 : ℚ) ≤ dot (sheetCoef Sm100.ds100 iss100) 26 (Sm100.d2s.folds.getD i []) := by
  decide +kernel

/-- At the issue facet the menu's cheapest route is the heavy lane fold, `231/4096` a word. -/
theorem sm100_phiStar_issue : 231 / 4096 ≤ Sm100.model.phiStar (onFacet iss100) := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_onFacet]
  show (231 : ℝ) / 4096 ≤ Sm100.model.G.load (Sparse.counts (Sm100.d2s.folds.getD φ.val [])) iss100
  rw [load_counts _ (sheetCoef Sm100.ds100) sm100_hG]
  have h : ((231 / 4096 : ℚ) : ℝ) ≤
      ((dot (sheetCoef Sm100.ds100 iss100) 26 (Sm100.d2s.folds.getD φ.val []) : ℚ) : ℝ) :=
    Rat.cast_le.mpr (sm100_fold_issue φ.val φ.isLt)
  push_cast at h
  linarith

theorem plain_issue : dot (sheetCoef Sm100.ds100 iss100) 26 plainAll < 1085 := by decide +kernel

/-- No labelling of ♭ at the checked fold meets FOLD-OPT per word folded at `folded ≥ 19300`. -/
theorem not_foldOpt_plain (A : Anatomy (Fin 26) (Fin 12) Unit Empty) (hn : A.run.n = Sparse.counts plainAll)
    (hA : A.Split) (folded : ℝ) (hf : 19300 ≤ folded) : ¬ FoldOptFolded Sm100.model A folded := by
  intro h
  have h1 := h (onFacet iss100) (isCert_onFacet _)
  have hle : ∀ k, A.nfold k ≤ A.run.n k := fun k => by
    obtain ⟨hw, -, hx, -, hs⟩ := hA
    have := hs k
    have h0 : 0 ≤ ∑ hd, A.nword hd k := Finset.sum_nonneg fun hd _ => hw hd k
    linarith [hx k]
  have hp : Sm100.model.G.price (onFacet iss100) A.nfold ≤ 1085 := by
    have e : Sm100.model.G.price (onFacet iss100) A.nfold = Sm100.model.G.load A.nfold iss100 :=
      demandPrice_onFacet iss100 (Sm100.model.G.load A.nfold)
    rw [e]
    calc Sm100.model.G.load A.nfold iss100 ≤ Sm100.model.G.load A.run.n iss100 := by
          unfold Machine.load
          exact Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hle k) (sm100_a_nonneg _ _)
      _ = ((dot (sheetCoef Sm100.ds100 iss100) 26 plainAll : ℚ) : ℝ) := by
          rw [hn, load_counts _ (sheetCoef Sm100.ds100) sm100_hG]
      _ ≤ 1085 := by
          have h2 : ((dot (sheetCoef Sm100.ds100 iss100) 26 plainAll : ℚ) : ℝ) ≤ ((1085 : ℚ) : ℝ) :=
            Rat.cast_le.mpr plain_issue.le
          push_cast at h2
          exact h2
  have hmu : Sm100.model.muF = 0 := rfl
  have hφ := sm100_phiStar_issue
  rw [hmu, sub_zero, one_mul] at h1
  nlinarith [mul_le_mul hf hφ (by norm_num : (0 : ℝ) ≤ 231 / 4096) (by linarith : (0 : ℝ) ≤ folded)]

/-- **`Sm100SheetHeadline`'s premises exclude every program that runs the fold the checks read.** For a play of it
whose ideal game accepts every path, and any receipt game and unfolded share the headline would take with it,
FOLD-OPT per word folded fails on some path. -/
theorem not_premises_plain (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm100.checks (Fin 26) (Fin 12) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) (hn : ∀ x, (Pl.run x).run.n = Sparse.counts plainAll)
    (hA : ∀ x, (Pl.run x).Split) (hacc : ∀ x, Pl.game.accepts x) (hG : Pl.game = G.game) (hPin : G.Pinned)
    (hWL : WholeLuckAt (1 / 2) G Uf) :
    ¬ ∀ x, FoldOptFolded Sm100.model (Pl.run x) ((1 - Uf.share x / Sm100.checks) * Sm100.model.Nw) := by
  intro h
  have hP : AlwaysPasses G := alwaysPasses_of_accepts G fun x => by
    rw [← hG]
    exact hacc x
  have h0 : Uf.share (somePath Sm100.checks) = 0 := share_zero G Uf hPin hP hWL _
  refine not_foldOpt_plain (Pl.run (somePath Sm100.checks)) (hn _) (hA _) _ ?_ (h _)
  rw [h0, sm100_Nw]
  norm_num

/-! ### The H200 at DS-90 -/

abbrev iss90 : SheetRow Sm90.Pipe := .issue

/-- ♭ without its filler: the word production, the fold and the skippable work. -/
def noFillAll : Sparse := Sm90.flatWords ++ Sm90.flatFold ++ [] ++ Sm90.flatSkip

def flatAll90 : Sparse := Sm90.flatWords ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90.flatSkip

theorem sm90_hG : ∀ r k, Sm90.model.G.a r k = (sheetCoef Sm90.ds90 r k.val : ℝ) := sheet_a Sm90.ds90 28

theorem sm90_a_nonneg : ∀ r k, 0 ≤ Sm90.model.G.a r k := fun r k => by
  rw [sm90_hG]
  exact Rat.cast_nonneg.mpr (sheetCoef_nonneg _ _ _)

theorem sm90_Nw : Sm90.model.Nw = 4096 := by
  show ((4096 : ℕ) : ℝ) = 4096
  norm_num

/-- ♭'s bound, on the issue facet. -/
def Bq90 : ℚ := dot (sheetCoef Sm90.ds90 iss90) 28 flatAll90

theorem sm90_max (r : SheetRow Sm90.Pipe) : dot (sheetCoef Sm90.ds90 r) 28 flatAll90 ≤ Bq90 := by
  cases r with
  | issue => decide +kernel
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

theorem sm90_bnd : Sm90.model.bnd Sm90.flat = (Bq90 : ℝ) := by
  unfold Model.bnd
  rw [show Sm90.flat.n = Sparse.counts flatAll90 from
    labelled_n 28 Sm90.flatWords Sm90.flatFold Sm90.flatFill Sm90.flatSkip]
  exact bound_eq _ (sheetCoef Sm90.ds90) sm90_hG flatAll90 iss90 sm90_max

theorem noFill_rows (r : SheetRow Sm90.Pipe) : dot (sheetCoef Sm90.ds90 r) 28 noFillAll ≤ 939 / 1000 * Bq90 := by
  cases r with
  | issue => decide +kernel
  | tensor => decide +kernel
  | pipe p => cases p <;> decide +kernel
  | smem => decide +kernel

/-- The model's floor for ♭ without its filler is at most `0.939·B_♭`. -/
theorem noFill_bound : Sm90.model.G.bound (Sparse.counts noFillAll) ≤ 939 / 1000 * Sm90.model.bnd Sm90.flat := by
  rw [sm90_bnd]
  unfold Machine.bound
  refine ciSup_le fun r => ?_
  rw [load_counts _ (sheetCoef Sm90.ds90) sm90_hG]
  have h := noFill_rows r
  have h' : ((dot (sheetCoef Sm90.ds90 r) 28 noFillAll : ℚ) : ℝ) ≤ ((939 / 1000 * Bq90 : ℚ) : ℝ) :=
    Rat.cast_le.mpr h
  push_cast at h'
  exact h'

theorem sm90_fold_issue : ∀ i < 2, (135 / 4096 : ℚ) ≤ dot (sheetCoef Sm90.ds90 iss90) 28 (Sm90.d2s.folds.getD i []) := by
  decide +kernel

theorem sm90_phiStar_issue : 135 / 4096 ≤ Sm90.model.phiStar (onFacet iss90) := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_onFacet]
  show (135 : ℝ) / 4096 ≤ Sm90.model.G.load (Sparse.counts (Sm90.d2s.folds.getD φ.val [])) iss90
  rw [load_counts _ (sheetCoef Sm90.ds90) sm90_hG]
  have h : ((135 / 4096 : ℚ) : ℝ) ≤
      ((dot (sheetCoef Sm90.ds90 iss90) 28 (Sm90.d2s.folds.getD φ.val []) : ℚ) : ℝ) :=
    Rat.cast_le.mpr (sm90_fold_issue φ.val φ.isLt)
  push_cast at h
  linarith

theorem fill_issue : (9 : ℚ) ≤ dot (sheetCoef Sm90.ds90 iss90) 28 Sm90.flatFill := by decide +kernel

/-- ffill's one route costs 9 issue slots a unit. -/
theorem sm90_chiStar_issue : 9 ≤ Sm90.model.chiStar (onFacet iss90) := by
  unfold Model.chiStar
  refine le_ciInf fun ψ => ?_
  rw [demandPrice_onFacet]
  show (9 : ℝ) ≤ Sm90.model.G.load (Sparse.counts Sm90.flatFill) iss90
  rw [load_counts _ (sheetCoef Sm90.ds90) sm90_hG]
  have h : ((9 : ℚ) : ℝ) ≤ ((dot (sheetCoef Sm90.ds90 iss90) 28 Sm90.flatFill : ℚ) : ℝ) := Rat.cast_le.mpr fill_issue
  push_cast at h
  linarith

theorem noFill_issue : dot (sheetCoef Sm90.ds90 iss90) 28 noFillAll < 138 := by decide +kernel

/-- No labelling of ♭ without its filler meets FOLD-OPT at `folded ≥ 4096` and FILL-OPT at `filled ≥ 1` together. -/
theorem not_optPair_noFill (A : Anatomy (Fin 28) (Fin 24) Unit Empty) (hn : A.run.n = Sparse.counts noFillAll)
    (hA : A.Split) (folded filled : ℝ) (hf : 4096 ≤ folded) (hx : 1 ≤ filled) :
    ¬ (FoldOptFolded Sm90.model A folded ∧ FillOptFilled Sm90.model A filled) := by
  rintro ⟨hF, hX⟩
  have h1 := hF (onFacet iss90) (isCert_onFacet _)
  have h2 := hX (onFacet iss90) (isCert_onFacet _)
  have hle : ∀ k, A.nfold k + A.nfill k ≤ A.run.n k := fun k => by
    obtain ⟨hw, -, -, -, hs⟩ := hA
    have := hs k
    have h0 : 0 ≤ ∑ hd, A.nword hd k := Finset.sum_nonneg fun hd _ => hw hd k
    linarith
  have hp : Sm90.model.G.price (onFacet iss90) A.nfold + Sm90.model.G.price (onFacet iss90) A.nfill ≤ 138 := by
    have e1 : Sm90.model.G.price (onFacet iss90) A.nfold = Sm90.model.G.load A.nfold iss90 :=
      demandPrice_onFacet iss90 (Sm90.model.G.load A.nfold)
    have e2 : Sm90.model.G.price (onFacet iss90) A.nfill = Sm90.model.G.load A.nfill iss90 :=
      demandPrice_onFacet iss90 (Sm90.model.G.load A.nfill)
    rw [e1, e2]
    calc Sm90.model.G.load A.nfold iss90 + Sm90.model.G.load A.nfill iss90 ≤ Sm90.model.G.load A.run.n iss90 := by
          unfold Machine.load
          rw [← Finset.sum_add_distrib]
          exact Finset.sum_le_sum fun k _ => by
            rw [← mul_add]
            exact mul_le_mul_of_nonneg_left (hle k) (sm90_a_nonneg _ _)
      _ = ((dot (sheetCoef Sm90.ds90 iss90) 28 noFillAll : ℚ) : ℝ) := by
          rw [hn, load_counts _ (sheetCoef Sm90.ds90) sm90_hG]
      _ ≤ 138 := by
          have h3 : ((dot (sheetCoef Sm90.ds90 iss90) 28 noFillAll : ℚ) : ℝ) ≤ ((138 : ℚ) : ℝ) :=
            Rat.cast_le.mpr noFill_issue.le
          push_cast at h3
          exact h3
  have hmuF : Sm90.model.muF = 0 := rfl
  have hmuX : Sm90.model.muX = 0 := rfl
  rw [hmuF, sub_zero, one_mul] at h1
  rw [hmuX, sub_zero, one_mul] at h2
  have hφ := sm90_phiStar_issue
  have hχ := sm90_chiStar_issue
  have m1 := mul_le_mul hf hφ (by norm_num : (0 : ℝ) ≤ 135 / 4096) (by linarith : (0 : ℝ) ≤ folded)
  have m2 := mul_le_mul hx hχ (by norm_num : (0 : ℝ) ≤ 9) (by linarith : (0 : ℝ) ≤ filled)
  linarith

/-- **`Sm90SheetHeadline`'s premises exclude every program that leaves ffill out**, which every check passes since no
Definition reads ffill's words: FOLD-OPT and FILL-OPT fail together on some path. -/
theorem not_premises_noFill (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm90.checks (Fin 28) (Fin 24) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) (hn : ∀ x, (Pl.run x).run.n = Sparse.counts noFillAll)
    (hA : ∀ x, (Pl.run x).Split) (hacc : ∀ x, Pl.game.accepts x) (hG : Pl.game = G.game) (hPin : G.Pinned)
    (hWL : WholeLuckAt (1 / 2) G Uf) :
    ¬ ((∀ x, FoldOptFolded Sm90.model (Pl.run x) ((1 - Uf.share x / Sm90.checks) * Sm90.model.Nw)) ∧
      ∀ x, FillOptFilled Sm90.model (Pl.run x) (1 - Uf.share x / Sm90.checks)) := by
  rintro ⟨hF, hX⟩
  have hP : AlwaysPasses G := alwaysPasses_of_accepts G fun x => by
    rw [← hG]
    exact hacc x
  have h0 : Uf.share (somePath Sm90.checks) = 0 := share_zero G Uf hPin hP hWL _
  refine not_optPair_noFill (Pl.run (somePath Sm90.checks)) (hn _) (hA _) _ _ ?_ ?_ ⟨hF _, hX _⟩
  · rw [h0, sm90_Nw]
    norm_num
  · rw [h0]
    norm_num

/-! ### RT-LOGIC-15: DS-90's b1 tensor forms are writers the tensor facet doesn't charge -/

/-- `BGMMA_B1_N256` and `BMMA168256_B1` have `peak = 0` on DS-90, so A3's term and the tensor row are `0 / 0 = 0`;
both are writer kinds, while honest's `IGMMA_S8_N256` costs 128 there. π* puts no weight on the tensor row, so the
zero moves no price the headline reads. -/
theorem b1_unrated : sheetCoef Sm90.ds90 .tensor 1 = 0 ∧ sheetCoef Sm90.ds90 .tensor 2 = 0 ∧
    sheetCoef Sm90.ds90 .tensor 14 = 128 ∧ (1, 4096) ∈ Sm90.d2s.writers ∧ (2, 128) ∈ Sm90.d2s.writers := by
  decide +kernel

theorem b1_pistar : Sm90.lamStar .tensor = 0 := rfl

/-! ### RT-LOGIC-13: the H200 on DC-90-ri, where every tensor form but honest's is off the menu -/

/-- DC-90-ri's row `r` as rationals, by class index. -/
def dcCoef (r i : ℕ) : ℚ := Sparse.coeff (Sm90Measured.dc90ri.getD r []) i

def dcRow (r : Fin 15) (i : ℕ) : ℚ := dcCoef r.val i

def flatAllDc : Sparse := Sm90.flatWords ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

/-- ♭'s bound on DC-90-ri, on JI90_0 (row 5). -/
def BqDc : ℚ := dot (dcCoef 5) 28 flatAllDc

theorem dc_max : ∀ i < 15, dot (dcCoef i) 28 flatAllDc ≤ BqDc := by decide +kernel

theorem dc_hG : ∀ r k, Sm90Measured.model.G.a r k = (dcRow r k.val : ℝ) := fun _ _ => rfl

theorem dc_bnd : Sm90Measured.model.bnd Sm90Measured.flat = (BqDc : ℝ) := by
  unfold Model.bnd
  rw [show Sm90Measured.flat.n = Sparse.counts flatAllDc from
    labelled_n 28 Sm90.flatWords Sm90.flatFold Sm90.flatFill Sm90Measured.flatSkip]
  exact bound_eq _ dcRow dc_hG flatAllDc 5 fun r => dc_max r.val r.isLt

/-- ♭ with `817/1000` of its words written by mma.sync m16n8k32 s8 (`IMMA16832_S8`, 128 words an instruction, the
s32 words honest's wgmma writes): `183/1000` of honest's IGMMA a unit and `3268/125` IMMA16832, with ♭'s fold, filler
and skippable work. -/
def immaAll : Sparse :=
  [(14, 183, 1000), (22, 3268, 125)] ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

theorem imma_rows : ∀ i < 15, dot (dcCoef i) 28 immaAll ≤ 875 / 1000 * BqDc := by decide +kernel

/-- The model's floor for that program on DC-90-ri is at most `0.875·B_♭`. -/
theorem imma_bound :
    Sm90Measured.model.G.bound (Sparse.counts immaAll) ≤ 875 / 1000 * Sm90Measured.model.bnd Sm90Measured.flat := by
  rw [dc_bnd]
  unfold Machine.bound
  refine ciSup_le fun r => ?_
  rw [load_counts _ dcRow dc_hG]
  have h' : ((dot (dcRow r) 28 immaAll : ℚ) : ℝ) ≤ ((875 / 1000 * BqDc : ℚ) : ℝ) :=
    Rat.cast_le.mpr (imma_rows r.val r.isLt)
  push_cast at h'
  exact h'

/-- `IMMA16832_S8` is a writer kind of DS-90's model at 128 words an instruction, and no writer kind of DC-90-ri's. -/
theorem imma_off_menu : (22, 128) ∈ Sm90.d2s.writers ∧ 22 ∉ Sm90Measured.d2s.writers.map Prod.fst := by
  decide +kernel

/-- ♭ with its words written by honest's instruction at half the tile, wgmma s8 m64n128k32 (`IGMMA_S8_N128`, 2048
words an instruction): two a unit, with ♭'s fold, filler and skippable work. -/
def n128All : Sparse :=
  [(11, 2, 1)] ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

theorem n128_rows : ∀ i < 15, dot (dcCoef i) 28 n128All ≤ 72 / 100 * BqDc := by decide +kernel

/-- The model's floor for that program on DC-90-ri is at most `0.72·B_♭`. -/
theorem n128_bound :
    Sm90Measured.model.G.bound (Sparse.counts n128All) ≤ 72 / 100 * Sm90Measured.model.bnd Sm90Measured.flat := by
  rw [dc_bnd]
  unfold Machine.bound
  refine ciSup_le fun r => ?_
  rw [load_counts _ dcRow dc_hG]
  have h' : ((dot (dcRow r) 28 n128All : ℚ) : ℝ) ≤ ((72 / 100 * BqDc : ℚ) : ℝ) :=
    Rat.cast_le.mpr (n128_rows r.val r.isLt)
  push_cast at h'
  exact h'

/-- `IGMMA_S8_N128` is a writer kind of DS-90's model at 2048 words an instruction, and no writer kind of DC-90-ri's. -/
theorem n128_off_menu : (11, 2048) ∈ Sm90.d2s.writers ∧ 11 ∉ Sm90Measured.d2s.writers.map Prod.fst := by
  decide +kernel

/-! The exclusion, at the JI90_1 certificate, where `UNI` costs 0.90 a word and the vector route `(1/2)/32` of a lane
instruction a word. -/

abbrev ji1 : Fin 15 := 6

/-- The vector classes' least coefficient on JI90_1. -/
def v1 : ℚ := dcCoef 6 0

/-- A lower bound on `w*` at JI90_1: the vector route's `(1/2)/32·v1`. -/
def w1 : ℚ := v1 / 64

def phi1 : ℚ := dot (dcCoef 6) 28 (Sm90.d2s.folds.getD 0 [])

def chi1 : ℚ := dot (dcCoef 6) 28 Sm90.flatFill

theorem dc_Bpos : (0 : ℚ) < BqDc := by decide +kernel

theorem dc_data :
    (∀ i < 2, 0 < (Sm90Measured.d2s.writers.getD i (0, 0)).2 ∧
      w1 * (Sm90Measured.d2s.writers.getD i (0, 0)).2 ≤
        dcCoef 6 ((Sm90Measured.d2s.writers.getD i (0, 0)).1 % 28)) ∧
    (∀ j ∈ Sm90Measured.d2s.vec, v1 ≤ dcCoef 6 (j % 28)) ∧
    (∀ i < 2, phi1 ≤ dot (dcCoef 6) 28 (Sm90Measured.d2s.folds.getD i [])) ∧
    0 ≤ v1 := by
  decide +kernel

theorem dc_a_nonneg : ∀ r k, 0 ≤ Sm90Measured.model.G.a r k := fun r k => by
  rw [dc_hG]
  exact Rat.cast_nonneg.mpr (coeff_nonneg _ _)

theorem dc_Nw : Sm90Measured.model.Nw = 4096 := by
  show ((4096 : ℕ) : ℝ) = 4096
  norm_num

theorem dc_q : Sm90Measured.model.q = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem dc_qT : Sm90Measured.model.qT = 1 / 2 := by
  show ((1 / 2 : ℚ) : ℝ) = 1 / 2
  norm_num

theorem dc_wf : Sm90Measured.model.Wf Sm90Measured.flat := by
  refine ⟨dc_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun ψ r => ?_,
    le_rfl, ?_, le_rfl, ?_⟩
  · show (0 : ℝ) < ((1980 : ℕ) : ℝ)
    norm_num
  · rw [dc_Nw]
    norm_num
  · rw [dc_bnd]
    exact_mod_cast dc_Bpos
  · show (0 : ℝ) < ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    exact_mod_cast (dc_data.1 t.val t.isLt).1
  · rw [dc_q]
    norm_num
  · rw [dc_qT]
    norm_num
  · show 0 ≤ Sm90Measured.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) r
    exact load_counts_nonneg _ dc_a_nonneg _ _
  · show 0 ≤ Sm90Measured.model.G.load (Sparse.counts Sm90.flatFill) r
    exact load_counts_nonneg _ dc_a_nonneg _ _
  · show (0 : ℝ) ≤ 1
    norm_num
  · show (0 : ℝ) ≤ 1
    norm_num

theorem dc_vecMin : (v1 : ℝ) ≤ Sm90Measured.model.vecMin ji1 := by
  have : Nonempty ↥Sm90Measured.model.vec :=
    ⟨⟨idx 28 0, List.mem_toFinset.mpr (List.mem_map.mpr ⟨0, by decide, rfl⟩)⟩⟩
  unfold Model.vecMin
  refine le_ciInf fun k => ?_
  obtain ⟨j, hj, hk⟩ := List.mem_map.mp (List.mem_toFinset.mp k.2)
  rw [dc_hG, ← hk]
  exact Rat.cast_le.mpr (dc_data.2.1 j hj)

theorem dc_wStar : (w1 : ℝ) ≤ Sm90Measured.model.wStar (onFacet ji1) := by
  have hvec : (w1 : ℝ) ≤ 1 / 2 / 32 * Sm90Measured.model.vecMin ji1 := by
    have h : (w1 : ℝ) = 1 / 2 / 32 * (v1 : ℝ) := by
      unfold w1
      push_cast
      ring
    rw [h]
    exact mul_le_mul_of_nonneg_left dc_vecMin (by norm_num)
  have : Nonempty (Head (Fin 2) Unit Empty) := ⟨.vector⟩
  unfold Model.wStar
  refine le_ciInf fun h => ?_
  rw [demandPrice_onFacet]
  cases h with
  | writer t =>
    have hw := dc_data.1 t.val t.isLt
    show (w1 : ℝ) ≤ (dcCoef 6 ((Sm90Measured.d2s.writers.getD t.val (0, 0)).1 % 28) : ℝ) /
        ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℝ)
    rw [le_div_iff₀ (by exact_mod_cast hw.1)]
    have h : ((w1 * ((Sm90Measured.d2s.writers.getD t.val (0, 0)).2 : ℚ) : ℚ) : ℝ) ≤
        ((dcCoef 6 ((Sm90Measured.d2s.writers.getD t.val (0, 0)).1 % 28) : ℚ) : ℝ) :=
      Rat.cast_le.mpr hw.2
    push_cast at h
    linarith
  | vector =>
    show (w1 : ℝ) ≤ Sm90Measured.model.q / 32 * Sm90Measured.model.vecMin ji1
    rw [dc_q]
    exact hvec
  | table l =>
    show (w1 : ℝ) ≤ max (Sm90Measured.model.qT / 32 * Sm90Measured.model.vecMin ji1)
      (Sm90Measured.model.bT * Sm90Measured.model.G.a ji1 (Sm90Measured.model.bytes l))
    refine le_trans ?_ (le_max_left _ _)
    rw [dc_qT]
    exact hvec
  | side u => exact u.elim

theorem dc_phiStar : (phi1 : ℝ) ≤ Sm90Measured.model.phiStar (onFacet ji1) := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_onFacet]
  show (phi1 : ℝ) ≤ Sm90Measured.model.G.load (Sparse.counts (Sm90Measured.d2s.folds.getD φ.val [])) ji1
  rw [load_counts _ dcRow dc_hG]
  exact Rat.cast_le.mpr (dc_data.2.2.1 φ.val φ.isLt)

theorem dc_chiStar : (chi1 : ℝ) ≤ Sm90Measured.model.chiStar (onFacet ji1) := by
  unfold Model.chiStar
  refine le_ciInf fun ψ => ?_
  rw [demandPrice_onFacet]
  show (chi1 : ℝ) ≤ Sm90Measured.model.G.load (Sparse.counts Sm90.flatFill) ji1
  rw [load_counts _ dcRow dc_hG]
  exact le_rfl

theorem dc_fillOpt (A : Anatomy (Fin 28) (Fin 2) Unit Empty) (h : FillOptFilled Sm90Measured.model A 1) :
    FillOpt Sm90Measured.model A := fun lam hl => by
  have := h lam hl
  rwa [mul_one] at this

/-- **`NoSum` fails on any program whose JI90_1 load is below its words', fold's and filler's price there, under
every labelling the other premises allow.** With A6 (`SheetWords`), FOLD-OPT at all 4096 words, FILL-OPT at share 1
and its 4096 committed words right and fixed (it computes them exactly, so `RightFixed` gives all of them at a zero
unfolded share), the JI90_1 certificate prices the run's words, fold and filler above its whole load there. -/
theorem not_noSum_at (s : Sparse) (hs : dot (dcCoef 6) 28 s < 4096 * w1 + 4096 * phi1 + chi1)
    (A : Anatomy (Fin 28) (Fin 2) Unit Empty) (hn : A.run.n = Sparse.counts s)
    (hA : A.Split) (hW : SheetWords Sm90Measured.model A) (hF : FoldOptFolded Sm90Measured.model A 4096)
    (hX : FillOptFilled Sm90Measured.model A 1) (hfix : 4096 ≤ A.fixed) : ¬ NoSum A := by
  intro hN
  have hSR : ScalarRoute Sm90Measured.model A :=
    scalarRoute_half _ A hA hW (le_of_eq dc_q) (le_of_eq dc_qT) le_rfl
  have key := price_of_parts_folded Sm90Measured.model Sm90Measured.flat A (onFacet ji1) 4096 (isCert_onFacet _)
    dc_wf hA hW.1 hSR (fun u => u.elim) hN hF (dc_fillOpt A hX)
  have hp : Sm90Measured.model.G.price (onFacet ji1) A.run.n = ((dot (dcRow ji1) 28 s : ℚ) : ℝ) := by
    rw [show Sm90Measured.model.G.price (onFacet ji1) A.run.n = Sm90Measured.model.G.load A.run.n ji1 from
      demandPrice_onFacet ji1 (Sm90Measured.model.G.load A.run.n), hn, load_counts _ dcRow dc_hG]
  have hlt : ((dot (dcRow ji1) 28 s : ℚ) : ℝ) < ((4096 * w1 + 4096 * phi1 + chi1 : ℚ) : ℝ) :=
    Rat.cast_lt.mpr hs
  push_cast at hlt
  have hmuF : Sm90Measured.model.muF = 0 := rfl
  have hmuX : Sm90Measured.model.muX = 0 := rfl
  rw [hmuF, hmuX, hp] at key
  have hw0 : (0 : ℝ) ≤ w1 := by
    have : (0 : ℚ) ≤ w1 := div_nonneg dc_data.2.2.2 (by norm_num)
    exact_mod_cast this
  have hw := mul_le_mul hfix dc_wStar hw0 (by linarith)
  have hφ := dc_phiStar
  have hχ := dc_chiStar
  linarith

/-- `NoSum` fails on the IMMA16832 program: its words have no head on the menu. -/
theorem not_noSum_imma (A : Anatomy (Fin 28) (Fin 2) Unit Empty) (hn : A.run.n = Sparse.counts immaAll)
    (hA : A.Split) (hW : SheetWords Sm90Measured.model A) (hF : FoldOptFolded Sm90Measured.model A 4096)
    (hX : FillOptFilled Sm90Measured.model A 1) (hfix : 4096 ≤ A.fixed) : ¬ NoSum A :=
  not_noSum_at immaAll (by decide +kernel) A hn hA hW hF hX hfix

/-- `NoSum` fails on the N128 program: honest's instruction at half the tile heads no word on the menu. -/
theorem not_noSum_n128 (A : Anatomy (Fin 28) (Fin 2) Unit Empty) (hn : A.run.n = Sparse.counts n128All)
    (hA : A.Split) (hW : SheetWords Sm90Measured.model A) (hF : FoldOptFolded Sm90Measured.model A 4096)
    (hX : FillOptFilled Sm90Measured.model A 1) (hfix : 4096 ≤ A.fixed) : ¬ NoSum A :=
  not_noSum_at n128All (by decide +kernel) A hn hA hW hF hX hfix

end Accounting.SecurityProofs.TS.HeadlineDc.RtLogic

#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.plain_bound
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.not_foldOpt_plain
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.not_premises_plain
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.noFill_bound
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.not_optPair_noFill
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.not_premises_noFill
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.imma_bound
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.imma_off_menu
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.not_noSum_imma
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.n128_bound
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.n128_off_menu
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.not_noSum_n128
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.b1_unrated
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic.b1_pistar
