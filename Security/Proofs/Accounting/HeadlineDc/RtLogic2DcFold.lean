import Definitions.Accounting.HeadlineDc
import Proofs.Accounting.Headline.Encoding

/-!
# RT2-LOGIC-1 on the H200 and the B200: FOLD-OPT at every certificate excludes li with its multiply as shift-adds (red-team witness, not for landing)

Builds on #2699's head (`cursor/int-hb-ea24` @ fd6a05c0fc), not on main.

All four statements take FOLD-OPT per word folded (`FoldOptFolded`), which quantifies over every certificate; the
proofs read it at π* only. li's IMAD computes the low word of `u·a + d` for a 32-bit key `a`, and so do 32 shift-adds
`@P_i LEA acc, u, acc, i` (bit `i` of `a` in `P_i`, set once a window), all ALU. The program with that fold, `lea*`,
is ♭'s word production and skippable work with ALU in place of every IMAD of ♭'s fold, 32 to one.

* DS-90 and DS-100, at `λ = 1/20·issue + 19/20·IMAD`: both fold routes cost at least `2567/81920` a word, so
  `N_w·φ*(λ)` is about 128.35 (H200) and 1026.8 (B200), while the whole program with the LEA fold costs about 106.06
  and 847.84 (`lea90_sep`, `lea100_sep`). `not_foldOpt_lea90`, `not_foldOpt_lea100`: no labelling meets FOLD-OPT
  once it folds 3400 of 4096 words (27,100 of 32,768).
* DC-90-ri and DC-100-ri, at `λ = 1/20·I + 19/20·UP`: the lane route costs on I and UNI's on UP, and a program with
  neither IMAD nor UNI costs 0 there (`leaM90_zero`, `leaM100_zero`), so FOLD-OPT fails at any word folded.
* `lea90_slower` and the others: the program is slower than ♭, its issue (ALU) load above every facet's load of ♭.
* `not_premises_lea90` and the others: with an ideal game that accepts every path and the statement's `Pinned` and
  `WholeLuckAt ½`, the unfolded share is 0, so each statement's FOLD-OPT premise is false on some path of the play.
-/

namespace Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Accounting.SecurityProofs.TS.Headline
  Pouw.RelativeSecurity

/-! ## Two-facet certificates and FOLD-OPT's failure, in general -/

/-- The certificate `s` on facet `r0`, the rest on `r1`. -/
noncomputable def mix {R : Type} [DecidableEq R] (s : ℚ) (r0 r1 : R) : R → ℝ :=
  fun r => (s : ℝ) * onFacet r0 r + (1 - (s : ℝ)) * onFacet r1 r

theorem demandPrice_mix {R : Type} [Fintype R] [DecidableEq R] (s : ℚ) (r0 r1 : R) (d : R → ℝ) :
    demandPrice (mix s r0 r1) d = (s : ℝ) * d r0 + (1 - (s : ℝ)) * d r1 := by
  have h1 := demandPrice_onFacet r0 d
  have h2 := demandPrice_onFacet r1 d
  have e : demandPrice (mix s r0 r1) d = (s : ℝ) * demandPrice (onFacet r0) d +
      (1 - (s : ℝ)) * demandPrice (onFacet r1) d := by
    simp only [demandPrice, mix, add_mul, Finset.sum_add_distrib, Finset.mul_sum, mul_assoc]
  rw [e, h1, h2]

theorem isCert_mix {R : Type} [Fintype R] [DecidableEq R] (s : ℚ) (hs0 : 0 ≤ s) (hs1 : s ≤ 1) (r0 r1 : R) :
    IsCert (mix s r0 r1) := by
  obtain ⟨h1, s1⟩ := isCert_onFacet r0
  obtain ⟨h2, s2⟩ := isCert_onFacet r1
  have t0 : (0 : ℝ) ≤ (s : ℝ) := by exact_mod_cast hs0
  have t1 : (s : ℝ) ≤ 1 := by exact_mod_cast hs1
  refine ⟨fun r => ?_, ?_⟩
  · have a := mul_nonneg t0 (h1 r)
    have b := mul_nonneg (sub_nonneg.mpr t1) (h2 r)
    show 0 ≤ (s : ℝ) * onFacet r0 r + (1 - (s : ℝ)) * onFacet r1 r
    linarith
  · have e : ∑ r, mix s r0 r1 r = (s : ℝ) * ∑ r, onFacet r0 r + (1 - (s : ℝ)) * ∑ r, onFacet r1 r := by
      simp only [mix, Finset.sum_add_distrib, Finset.mul_sum]
    rw [e, s1, s2]
    ring

/-- A program's price at a two-facet certificate, on rational rows. -/
theorem price_mix_counts {n : ℕ} {R : Type} [Fintype R] [DecidableEq R] (G : Machine (Fin n) R) (c : R → ℕ → ℚ)
    (hG : ∀ r k, G.a r k = (c r k.val : ℝ)) (s : ℚ) (r0 r1 : R) (v : Sparse) :
    G.price (mix s r0 r1) (Sparse.counts v) = ((s * dot (c r0) n v + (1 - s) * dot (c r1) n v : ℚ) : ℝ) := by
  show demandPrice (mix s r0 r1) (G.load (Sparse.counts v)) = _
  rw [demandPrice_mix, load_counts G c hG, load_counts G c hG]
  push_cast
  ring

/-- `φ*` at a two-facet certificate is at least `p` when each of the design's two fold routes costs `p` there. -/
theorem phiStar_mix_ge {n nt : ℕ} [NeZero n] {R : Type} [Fintype R] [DecidableEq R] (D : DesignData)
    (G : Machine (Fin n) R) (c : R → ℕ → ℚ) (hG : ∀ r k, G.a r k = (c r k.val : ℝ)) (s : ℚ) (r0 r1 : R) (p : ℚ)
    (h : ∀ i < 2, p ≤ s * dot (c r0) n (D.folds.getD i []) + (1 - s) * dot (c r1) n (D.folds.getD i [])) :
    (p : ℝ) ≤ (D.model n nt 2 G).phiStar (mix s r0 r1) := by
  unfold Model.phiStar
  refine le_ciInf fun φ => ?_
  rw [demandPrice_mix]
  show (p : ℝ) ≤ (s : ℝ) * G.load (Sparse.counts (D.folds.getD φ.val [])) r0 +
    (1 - (s : ℝ)) * G.load (Sparse.counts (D.folds.getD φ.val [])) r1
  rw [load_counts G c hG, load_counts G c hG]
  have h' : ((p : ℚ) : ℝ) ≤ ((s * dot (c r0) n (D.folds.getD φ.val []) +
      (1 - s) * dot (c r1) n (D.folds.getD φ.val []) : ℚ) : ℝ) :=
    Rat.cast_le.mpr (h φ.val φ.isLt)
  push_cast at h'
  linarith

/-- FOLD-OPT per word folded fails at `folded` once a certificate prices the whole run under `folded·φ`, `φ ≤ φ*`. -/
theorem not_foldOpt_of {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] (Z : Model K R T L U F X) (A : Anatomy K T L U) (hA : A.Split)
    (ha : ∀ r k, 0 ≤ Z.G.a r k) (lam : R → ℝ) (hlam : IsCert lam) (hmu : Z.muF = 0) (φ c folded : ℝ)
    (hφ : φ ≤ Z.phiStar lam) (hc : Z.G.price lam A.run.n ≤ c) (hf : 0 ≤ folded) (hlt : c < folded * φ) :
    ¬ FoldOptFolded Z A folded := by
  intro h
  have h1 := h lam hlam
  rw [hmu, sub_zero, one_mul] at h1
  have hle : ∀ k, A.nfold k ≤ A.run.n k := fun k => by
    obtain ⟨hw, -, hx, -, hs⟩ := hA
    have := hs k
    have h0 : 0 ≤ ∑ hd, A.nword hd k := Finset.sum_nonneg fun hd _ => hw hd k
    linarith [hx k]
  have hp : Z.G.price lam A.nfold ≤ Z.G.price lam A.run.n := by
    unfold Machine.price Machine.load
    exact Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left
      (Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_left (hle k) (ha r k)) (hlam.1 r)
  have := mul_le_mul_of_nonneg_left hφ hf
  linarith

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

/-- **A statement's FOLD-OPT premise is false on a play** whose ideal game accepts every path, at a pinned receipt game
with `WholeLuckAt ½`, once FOLD-OPT fails at all `N_w` words on one of its runs. -/
theorem not_premises_of {Ω : Type} [Fintype Ω] [Nonempty Ω] {K R T L U F X : Type} [Fintype K] [Fintype R]
    [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X] {S : ℕ} (Z : Model K R T L U F X)
    (Pl : Play Ω S K T L U) (G : ReceiptGame Ω S) (Uf : Unfolded Ω S) (hacc : ∀ x, Pl.game.accepts x)
    (hG : Pl.game = G.game) (hPin : G.Pinned) (hWL : WholeLuckAt (1 / 2) G Uf)
    (hno : ¬ FoldOptFolded Z (Pl.run (somePath S)) Z.Nw) :
    ¬ ∀ x, FoldOptFolded Z (Pl.run x) ((1 - Uf.share x / S) * Z.Nw) := by
  intro h
  have hP : AlwaysPasses G := alwaysPasses_of_accepts G fun x => by
    rw [← hG]
    exact hacc x
  have h0 : Uf.share (somePath S) = 0 := share_zero G Uf hPin hP hWL _
  have h1 := h (somePath S)
  rw [h0, zero_div, sub_zero, one_mul] at h1
  exact hno h1

/-- Both fold routes' least price at `1/20` on one facet and `19/20` on another, on DS-90 and DS-100. -/
def phiS : ℚ := 2567 / 81920

/-! ## H200, DS-90 (`Sm90SheetHeadline`) -/

/-- li's fold for 4096 words with each IMAD as 32 LEAs: ALU `68 + 32·64`, SHFL 3. -/
def leaFold90 : Sparse := [(0, 2116, 1), (26, 3, 1)]

def leaAll90 : Sparse := Sm90.flatWords ++ leaFold90 ++ Sm90.flatFill ++ Sm90.flatSkip

def flatAll90 : Sparse := Sm90.flatWords ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90.flatSkip

theorem sm90_hG : ∀ r k, Sm90.model.G.a r k = (sheetCoef Sm90.ds90 r k.val : ℝ) := sheet_a Sm90.ds90 28

theorem sm90_a_nonneg : ∀ r k, 0 ≤ Sm90.model.G.a r k := fun r k => by
  rw [sm90_hG]
  exact Rat.cast_nonneg.mpr (sheetCoef_nonneg _ _ _)

theorem sm90_Nw : Sm90.model.Nw = 4096 := by
  show ((4096 : ℕ) : ℝ) = 4096
  norm_num

theorem lea90_slower :
    ∀ r ∈ [SheetRow.issue, SheetRow.tensor, SheetRow.pipe Sm90.Pipe.alu, SheetRow.pipe Sm90.Pipe.imad,
      SheetRow.pipe Sm90.Pipe.shfl, SheetRow.smem],
      dot (sheetCoef Sm90.ds90 r) 28 flatAll90 < dot (sheetCoef Sm90.ds90 .issue) 28 leaAll90 := by
  decide +kernel

theorem fold90_mix :
    ∀ i < 2, phiS ≤ 1 / 20 * dot (sheetCoef Sm90.ds90 .issue) 28 (Sm90.d2s.folds.getD i []) +
      (1 - 1 / 20) * dot (sheetCoef Sm90.ds90 (.pipe .imad)) 28 (Sm90.d2s.folds.getD i []) := by
  decide +kernel

def c90 : ℚ := 1 / 20 * dot (sheetCoef Sm90.ds90 .issue) 28 leaAll90 +
  (1 - 1 / 20) * dot (sheetCoef Sm90.ds90 (.pipe .imad)) 28 leaAll90

theorem lea90_sep : c90 < 3400 * phiS := by decide +kernel

theorem not_foldOpt_lea90 (A : Anatomy (Fin 28) (Fin 24) Unit Empty) (hn : A.run.n = Sparse.counts leaAll90)
    (hA : A.Split) (folded : ℝ) (hf : 3400 ≤ folded) : ¬ FoldOptFolded Sm90.model A folded := by
  have hφ : ((phiS : ℚ) : ℝ) ≤ Sm90.model.phiStar (mix (1 / 20) SheetRow.issue (SheetRow.pipe Sm90.Pipe.imad)) :=
    phiStar_mix_ge Sm90.d2s Sm90.sheet.machine (sheetCoef Sm90.ds90) sm90_hG _ _ _ _ fold90_mix
  have hc : Sm90.model.G.price (mix (1 / 20) SheetRow.issue (SheetRow.pipe Sm90.Pipe.imad)) A.run.n =
      ((c90 : ℚ) : ℝ) := by
    rw [hn]
    exact price_mix_counts _ (sheetCoef Sm90.ds90) sm90_hG _ _ _ _
  have h1 : ((c90 : ℚ) : ℝ) < ((3400 * phiS : ℚ) : ℝ) := Rat.cast_lt.mpr lea90_sep
  have h2 : (0 : ℝ) ≤ ((phiS : ℚ) : ℝ) := Rat.cast_nonneg.mpr (by decide +kernel)
  push_cast at h1
  have h3 := mul_le_mul_of_nonneg_right hf h2
  exact not_foldOpt_of Sm90.model A hA sm90_a_nonneg _ (isCert_mix _ (by norm_num) (by norm_num) _ _) rfl _ _
    folded hφ hc.le (by linarith) (by linarith)

/-- **`Sm90SheetHeadline`'s FOLD-OPT premise excludes the LEA program.** -/
theorem not_premises_lea90 (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm90.checks (Fin 28) (Fin 24) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) (hn : ∀ x, (Pl.run x).run.n = Sparse.counts leaAll90)
    (hA : ∀ x, (Pl.run x).Split) (hacc : ∀ x, Pl.game.accepts x) (hG : Pl.game = G.game) (hPin : G.Pinned)
    (hWL : WholeLuckAt (1 / 2) G Uf) :
    ¬ ∀ x, FoldOptFolded Sm90.model (Pl.run x) ((1 - Uf.share x / Sm90.checks) * Sm90.model.Nw) :=
  not_premises_of Sm90.model Pl G Uf hacc hG hPin hWL
    (not_foldOpt_lea90 _ (hn _) (hA _) _ (by rw [sm90_Nw]; norm_num))

/-! ## B200, DS-100 (`Sm100SheetHeadline`) -/

/-- li's fold for 32768 words with each IMAD as 32 LEAs: ALU `544 + 32·512`, SHFL 24. -/
def leaFold100 : Sparse := [(0, 16928, 1), (14, 24, 1)]

def leaAll100 : Sparse := Sm100.flatWords ++ leaFold100 ++ Sm100.flatFill ++ Sm100.flatSkip

def flatAll100 : Sparse := Sm100.flatWords ++ Sm100.flatFold ++ Sm100.flatFill ++ Sm100.flatSkip

theorem sm100_hG : ∀ r k, Sm100.model.G.a r k = (sheetCoef Sm100.ds100 r k.val : ℝ) := sheet_a Sm100.ds100 26

theorem sm100_a_nonneg : ∀ r k, 0 ≤ Sm100.model.G.a r k := fun r k => by
  rw [sm100_hG]
  exact Rat.cast_nonneg.mpr (sheetCoef_nonneg _ _ _)

theorem sm100_Nw : Sm100.model.Nw = 32768 := by
  show ((32768 : ℕ) : ℝ) = 32768
  norm_num

theorem lea100_slower :
    ∀ r ∈ [SheetRow.issue, SheetRow.tensor, SheetRow.pipe Sm100.Pipe.alu, SheetRow.pipe Sm100.Pipe.imad,
      SheetRow.pipe Sm100.Pipe.shfl, SheetRow.smem],
      dot (sheetCoef Sm100.ds100 r) 26 flatAll100 < dot (sheetCoef Sm100.ds100 .issue) 26 leaAll100 := by
  decide +kernel

theorem fold100_mix :
    ∀ i < 2, phiS ≤ 1 / 20 * dot (sheetCoef Sm100.ds100 .issue) 26 (Sm100.d2s.folds.getD i []) +
      (1 - 1 / 20) * dot (sheetCoef Sm100.ds100 (.pipe .imad)) 26 (Sm100.d2s.folds.getD i []) := by
  decide +kernel

def c100 : ℚ := 1 / 20 * dot (sheetCoef Sm100.ds100 .issue) 26 leaAll100 +
  (1 - 1 / 20) * dot (sheetCoef Sm100.ds100 (.pipe .imad)) 26 leaAll100

theorem lea100_sep : c100 < 27100 * phiS := by decide +kernel

theorem not_foldOpt_lea100 (A : Anatomy (Fin 26) (Fin 13) Unit Empty) (hn : A.run.n = Sparse.counts leaAll100)
    (hA : A.Split) (folded : ℝ) (hf : 27100 ≤ folded) : ¬ FoldOptFolded Sm100.model A folded := by
  have hφ : ((phiS : ℚ) : ℝ) ≤
      Sm100.model.phiStar (mix (1 / 20) SheetRow.issue (SheetRow.pipe Sm100.Pipe.imad)) :=
    phiStar_mix_ge Sm100.d2s Sm100.sheet.machine (sheetCoef Sm100.ds100) sm100_hG _ _ _ _ fold100_mix
  have hc : Sm100.model.G.price (mix (1 / 20) SheetRow.issue (SheetRow.pipe Sm100.Pipe.imad)) A.run.n =
      ((c100 : ℚ) : ℝ) := by
    rw [hn]
    exact price_mix_counts _ (sheetCoef Sm100.ds100) sm100_hG _ _ _ _
  have h1 : ((c100 : ℚ) : ℝ) < ((27100 * phiS : ℚ) : ℝ) := Rat.cast_lt.mpr lea100_sep
  have h2 : (0 : ℝ) ≤ ((phiS : ℚ) : ℝ) := Rat.cast_nonneg.mpr (by decide +kernel)
  push_cast at h1
  have h3 := mul_le_mul_of_nonneg_right hf h2
  exact not_foldOpt_of Sm100.model A hA sm100_a_nonneg _ (isCert_mix _ (by norm_num) (by norm_num) _ _) rfl _ _
    folded hφ hc.le (by linarith) (by linarith)

/-- **`Sm100SheetHeadline`'s FOLD-OPT premise excludes the LEA program.** -/
theorem not_premises_lea100 (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm100.checks (Fin 26) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) (hn : ∀ x, (Pl.run x).run.n = Sparse.counts leaAll100)
    (hA : ∀ x, (Pl.run x).Split) (hacc : ∀ x, Pl.game.accepts x) (hG : Pl.game = G.game) (hPin : G.Pinned)
    (hWL : WholeLuckAt (1 / 2) G Uf) :
    ¬ ∀ x, FoldOptFolded Sm100.model (Pl.run x) ((1 - Uf.share x / Sm100.checks) * Sm100.model.Nw) :=
  not_premises_of Sm100.model Pl G Uf hacc hG hPin hWL
    (not_foldOpt_lea100 _ (hn _) (hA _) _ (by rw [sm100_Nw]; norm_num))

/-! ## H200, DC-90-ri (`Sm90MeasuredHeadline`): rows 2 (`A`), 3 (`I`) and 13 (`UP`) -/

def coefM90 (r i : ℕ) : ℚ := Sparse.coeff (Sm90Measured.dc90ri.getD r []) i

def rowM90 (r : Fin 15) (i : ℕ) : ℚ := coefM90 r.val i

def leaAllM90 : Sparse := Sm90.flatWords ++ leaFold90 ++ Sm90.flatFill ++ Sm90Measured.flatSkip

def flatAllM90 : Sparse := Sm90.flatWords ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

theorem m90_hG : ∀ r k, Sm90Measured.model.G.a r k = (rowM90 r k.val : ℝ) := fun _ _ => rfl

theorem m90_a_nonneg : ∀ r k, 0 ≤ Sm90Measured.model.G.a r k := fun r k => by
  rw [m90_hG]
  exact Rat.cast_nonneg.mpr (coeff_nonneg _ _)

theorem m90_Nw : Sm90Measured.model.Nw = 4096 := by
  show ((4096 : ℕ) : ℝ) = 4096
  norm_num

theorem leaM90_slower : ∀ i < 15, dot (coefM90 i) 28 flatAllM90 < dot (coefM90 2) 28 leaAllM90 := by
  decide +kernel

/-- The lane route's price at the mix, which UNI's exceeds. -/
def phiM90 : ℚ := 1 / 20 * dot (rowM90 3) 28 (Sm90.d2s.folds.getD 0 [])

theorem foldM90_mix :
    ∀ i < 2, phiM90 ≤ 1 / 20 * dot (rowM90 3) 28 (Sm90Measured.d2s.folds.getD i []) +
      (1 - 1 / 20) * dot (rowM90 13) 28 (Sm90Measured.d2s.folds.getD i []) := by
  decide +kernel

theorem leaM90_zero :
    1 / 20 * dot (rowM90 3) 28 leaAllM90 + (1 - 1 / 20) * dot (rowM90 13) 28 leaAllM90 = 0 := by decide +kernel

theorem phiM90_pos : 0 < phiM90 := by decide +kernel

theorem not_foldOpt_leaM90 (A : Anatomy (Fin 28) (Fin 2) Unit Empty) (hn : A.run.n = Sparse.counts leaAllM90)
    (hA : A.Split) (folded : ℝ) (hf : 0 < folded) : ¬ FoldOptFolded Sm90Measured.model A folded := by
  have hφ : ((phiM90 : ℚ) : ℝ) ≤ Sm90Measured.model.phiStar (mix (1 / 20) (3 : Fin 15) (13 : Fin 15)) :=
    phiStar_mix_ge Sm90Measured.d2s Sm90Measured.machine rowM90 m90_hG _ _ _ _ foldM90_mix
  have hc : Sm90Measured.model.G.price (mix (1 / 20) (3 : Fin 15) (13 : Fin 15)) A.run.n = 0 := by
    rw [hn, price_mix_counts _ rowM90 m90_hG, leaM90_zero, Rat.cast_zero]
  have h2 : (0 : ℝ) < ((phiM90 : ℚ) : ℝ) := Rat.cast_pos.mpr phiM90_pos
  exact not_foldOpt_of Sm90Measured.model A hA m90_a_nonneg _ (isCert_mix _ (by norm_num) (by norm_num) _ _) rfl
    _ _ folded hφ hc.le hf.le (mul_pos hf h2)

/-- **`Sm90MeasuredHeadline`'s FOLD-OPT premise excludes the LEA program.** -/
theorem not_premises_leaM90 (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm90.checks (Fin 28) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) (hn : ∀ x, (Pl.run x).run.n = Sparse.counts leaAllM90)
    (hA : ∀ x, (Pl.run x).Split) (hacc : ∀ x, Pl.game.accepts x) (hG : Pl.game = G.game) (hPin : G.Pinned)
    (hWL : WholeLuckAt (1 / 2) G Uf) :
    ¬ ∀ x, FoldOptFolded Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw) :=
  not_premises_of Sm90Measured.model Pl G Uf hacc hG hPin hWL
    (not_foldOpt_leaM90 _ (hn _) (hA _) _ (by rw [m90_Nw]; norm_num))

/-! ## B200, DC-100-ri (`Sm100MeasuredHeadline`): rows 2 (`A`), 3 (`I`) and 10 (`UP`) -/

def coefM100 (r i : ℕ) : ℚ := Sparse.coeff (Sm100Measured.dc100ri.getD r []) i

def rowM100 (r : Fin 12) (i : ℕ) : ℚ := coefM100 r.val i

def leaAllM100 : Sparse := Sm100.flatWords ++ leaFold100 ++ Sm100.flatFill ++ Sm100Measured.flatSkip

def flatAllM100 : Sparse := Sm100.flatWords ++ Sm100.flatFold ++ Sm100.flatFill ++ Sm100Measured.flatSkip

theorem m100_hG : ∀ r k, Sm100Measured.model.G.a r k = (rowM100 r k.val : ℝ) := fun _ _ => rfl

theorem m100_a_nonneg : ∀ r k, 0 ≤ Sm100Measured.model.G.a r k := fun r k => by
  rw [m100_hG]
  exact Rat.cast_nonneg.mpr (coeff_nonneg _ _)

theorem m100_Nw : Sm100Measured.model.Nw = 32768 := by
  show ((32768 : ℕ) : ℝ) = 32768
  norm_num

theorem leaM100_slower : ∀ i < 12, dot (coefM100 i) 26 flatAllM100 < dot (coefM100 2) 26 leaAllM100 := by
  decide +kernel

def phiM100 : ℚ := 1 / 20 * dot (rowM100 3) 26 (Sm100.d2s.folds.getD 0 [])

theorem foldM100_mix :
    ∀ i < 2, phiM100 ≤ 1 / 20 * dot (rowM100 3) 26 (Sm100Measured.d2s.folds.getD i []) +
      (1 - 1 / 20) * dot (rowM100 10) 26 (Sm100Measured.d2s.folds.getD i []) := by
  decide +kernel

theorem leaM100_zero :
    1 / 20 * dot (rowM100 3) 26 leaAllM100 + (1 - 1 / 20) * dot (rowM100 10) 26 leaAllM100 = 0 := by decide +kernel

theorem phiM100_pos : 0 < phiM100 := by decide +kernel

theorem not_foldOpt_leaM100 (A : Anatomy (Fin 26) (Fin 13) Unit Empty) (hn : A.run.n = Sparse.counts leaAllM100)
    (hA : A.Split) (folded : ℝ) (hf : 0 < folded) : ¬ FoldOptFolded Sm100Measured.model A folded := by
  have hφ : ((phiM100 : ℚ) : ℝ) ≤ Sm100Measured.model.phiStar (mix (1 / 20) (3 : Fin 12) (10 : Fin 12)) :=
    phiStar_mix_ge Sm100Measured.d2s Sm100Measured.machine rowM100 m100_hG _ _ _ _ foldM100_mix
  have hc : Sm100Measured.model.G.price (mix (1 / 20) (3 : Fin 12) (10 : Fin 12)) A.run.n = 0 := by
    rw [hn, price_mix_counts _ rowM100 m100_hG, leaM100_zero, Rat.cast_zero]
  have h2 : (0 : ℝ) < ((phiM100 : ℚ) : ℝ) := Rat.cast_pos.mpr phiM100_pos
  exact not_foldOpt_of Sm100Measured.model A hA m100_a_nonneg _ (isCert_mix _ (by norm_num) (by norm_num) _ _) rfl
    _ _ folded hφ hc.le hf.le (mul_pos hf h2)

/-- **`Sm100MeasuredHeadline`'s FOLD-OPT premise excludes the LEA program.** -/
theorem not_premises_leaM100 (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm100.checks (Fin 26) (Fin 13) Unit Empty) (G : ReceiptGame Ω Sm100.checks)
    (Uf : Unfolded Ω Sm100.checks) (hn : ∀ x, (Pl.run x).run.n = Sparse.counts leaAllM100)
    (hA : ∀ x, (Pl.run x).Split) (hacc : ∀ x, Pl.game.accepts x) (hG : Pl.game = G.game) (hPin : G.Pinned)
    (hWL : WholeLuckAt (1 / 2) G Uf) :
    ¬ ∀ x, FoldOptFolded Sm100Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm100.checks) * Sm100Measured.model.Nw) :=
  not_premises_of Sm100Measured.model Pl G Uf hacc hG hPin hWL
    (not_foldOpt_leaM100 _ (hn _) (hA _) _ (by rw [m100_Nw]; norm_num))

end Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold

#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold.lea90_slower
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold.not_premises_lea90
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold.lea100_slower
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold.not_premises_lea100
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold.leaM90_slower
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold.not_premises_leaM90
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold.leaM100_slower
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2Fold.not_premises_leaM100
