import Definitions.Accounting.HeadlineDc
import Proofs.Accounting.Headline.Encoding
import Proofs.Accounting.Headline.Luck

/-!
# RT2-LOGIC-5: the H200 measured headline's `NoSum` is a menu cut (red-team witness, not for landing)

Builds on #2699's head (`cursor/int-hb-ea24` @ fd6a05c0fc), not on main. Round 1's `not_noSum_n128`
(note:20261019T0515Z-draft-rt-logic, RT-LOGIC-13) at that head, where the design has no filler and no FILL-OPT.

`Sm90MeasuredHeadline` reads `Sm90Measured.model`, whose writer kinds are honest's `IGMMA_S8_N256` and `UNI` only.
Every other exact s8 tensor form writes the same s32 words and is off the menu. Three of them, in place of honest's
MMA with ♭'s fold and skippable work, a unit:

* `n128All`: wgmma s8 m64n128k32 (`IGMMA_S8_N128`), two a unit;
* `rsAll`: wgmma s8 m64n256k32 with A from registers (`IGMMA_S8_N256_RS`), one a unit;
* `immaAll`: mma.sync m16n8k32 s8 (`IMMA16832_S8`) for `817/1000` of the words.

`n128_bound`, `rs_bound`, `imma_bound`: on DC-90-ri's rows their floors are at most `0.70`, `0.69` and `0.90` of
`B_♭` (♭'s bound is on JPA90_0). `not_noSum_at`: at the JI90_1 certificate, `NoSum` fails on any program whose load
there is below `4096·w* + 4096·φ*`, under every labelling A6 and FOLD-OPT at all 4096 words allow, once its 4096
words are fixed. `deficiency_zero`: on a play whose ideal game passes every path, the statement's luck premises force
every round's deficiency to 0. `not_premises_at` and its three instances: so for a play of any of these programs
whose ideal game passes every path and which has one accepted path with right committed values, the statement's
premises (Split, A6, FOLD-OPT per word folded, the game, `Pinned`, `OperandHMu ¾`, `ReceiptLuckAt ½`,
`WholeLuckAt ½`, `RightFixed`) leave `NoSum` false on that path.
-/

namespace Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2

open Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Accounting.SecurityProofs.TS.Headline
  Accounting.SecurityProofs.TS Pouw.RelativeSecurity

/-- DC-90-ri's row `r` as rationals, by class index. -/
def dcCoef (r i : ℕ) : ℚ := Sparse.coeff (Sm90Measured.dc90ri.getD r []) i

def dcRow (r : Fin 15) (i : ℕ) : ℚ := dcCoef r.val i

def flatAllDc : Sparse := Sm90.flatWords ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

/-- ♭'s bound on DC-90-ri, on JPA90_0 (row 10). -/
def BqDc : ℚ := dot (dcCoef 10) 28 flatAllDc

theorem dc_max : ∀ i < 15, dot (dcCoef i) 28 flatAllDc ≤ BqDc := by decide +kernel

theorem dc_hG : ∀ r k, Sm90Measured.model.G.a r k = (dcRow r k.val : ℝ) := fun _ _ => rfl

theorem dc_bnd : Sm90Measured.model.bnd Sm90Measured.flat = (BqDc : ℝ) := by
  unfold Model.bnd
  rw [show Sm90Measured.flat.n = Sparse.counts flatAllDc from
    labelled_n 28 Sm90.flatWords Sm90.flatFold Sm90.flatFill Sm90Measured.flatSkip]
  exact bound_eq _ dcRow dc_hG flatAllDc 10 fun r => dc_max r.val r.isLt

/-- ♭ with its words written by wgmma s8 m64n128k32, two a unit. -/
def n128All : Sparse := [(11, 2, 1)] ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

/-- ♭ with its words written by wgmma s8 m64n256k32 with A from registers. -/
def rsAll : Sparse := [(15, 1, 1)] ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

/-- ♭ with `817/1000` of its words written by mma.sync m16n8k32 s8: `183/1000` of honest's IGMMA and `3268/125`
IMMA16832 a unit. -/
def immaAll : Sparse :=
  [(14, 183, 1000), (22, 3268, 125)] ++ Sm90.flatFold ++ Sm90.flatFill ++ Sm90Measured.flatSkip

theorem n128_rows : ∀ i < 15, dot (dcCoef i) 28 n128All ≤ 70 / 100 * BqDc := by decide +kernel

theorem rs_rows : ∀ i < 15, dot (dcCoef i) 28 rsAll ≤ 69 / 100 * BqDc := by decide +kernel

theorem imma_rows : ∀ i < 15, dot (dcCoef i) 28 immaAll ≤ 90 / 100 * BqDc := by decide +kernel

theorem bound_of_rows (s : Sparse) (c : ℚ) (h : ∀ i < 15, dot (dcCoef i) 28 s ≤ c * BqDc) :
    Sm90Measured.model.G.bound (Sparse.counts s) ≤ c * Sm90Measured.model.bnd Sm90Measured.flat := by
  rw [dc_bnd]
  unfold Machine.bound
  refine ciSup_le fun r => ?_
  rw [load_counts _ dcRow dc_hG]
  have h' : ((dot (dcRow r) 28 s : ℚ) : ℝ) ≤ ((c * BqDc : ℚ) : ℝ) := Rat.cast_le.mpr (h r.val r.isLt)
  push_cast at h'
  exact h'

/-- The model's floor for the N128 program on DC-90-ri is at most `0.70·B_♭`. -/
theorem n128_bound :
    Sm90Measured.model.G.bound (Sparse.counts n128All) ≤ 70 / 100 * Sm90Measured.model.bnd Sm90Measured.flat := by
  have h := bound_of_rows n128All (70 / 100) n128_rows
  push_cast at h
  exact h

/-- The model's floor for the register-sourced program on DC-90-ri is at most `0.69·B_♭`. -/
theorem rs_bound :
    Sm90Measured.model.G.bound (Sparse.counts rsAll) ≤ 69 / 100 * Sm90Measured.model.bnd Sm90Measured.flat := by
  have h := bound_of_rows rsAll (69 / 100) rs_rows
  push_cast at h
  exact h

/-- The model's floor for the IMMA16832 program on DC-90-ri is at most `0.90·B_♭`. -/
theorem imma_bound :
    Sm90Measured.model.G.bound (Sparse.counts immaAll) ≤ 90 / 100 * Sm90Measured.model.bnd Sm90Measured.flat := by
  have h := bound_of_rows immaAll (90 / 100) imma_rows
  push_cast at h
  exact h

/-- Each of the three is a writer kind of DS-90's model, and none is a writer kind of DC-90-ri's. -/
theorem off_menu :
    (11, 2048) ∈ Sm90.d2s.writers ∧ 11 ∉ Sm90Measured.d2s.writers.map Prod.fst ∧
    (15, 4096) ∈ Sm90.d2s.writers ∧ 15 ∉ Sm90Measured.d2s.writers.map Prod.fst ∧
    (22, 128) ∈ Sm90.d2s.writers ∧ 22 ∉ Sm90Measured.d2s.writers.map Prod.fst := by
  decide +kernel

/-! ### The exclusion, at the JI90_1 certificate (row 6), where the vector route's `(1/2)/32` of a lane instruction
is the cheapest head of a word -/

abbrev ji1 : Fin 15 := 6

/-- The vector classes' least coefficient on JI90_1. -/
def v1 : ℚ := dcCoef 6 0

/-- A lower bound on `w*` at JI90_1: the vector route's `(1/2)/32·v1`. -/
def w1 : ℚ := v1 / 64

/-- The lane fold's load on JI90_1, a word. -/
def phi1 : ℚ := dot (dcCoef 6) 28 (Sm90.d2s.folds.getD 0 [])

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
  refine ⟨dc_a_nonneg, ?_, ?_, ?_, fun t => ?_, ?_, ?_, le_rfl, fun u => u.elim, fun φ r => ?_, fun _ _ => le_rfl,
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

/-- FILL-OPT on any split run: the model has no filler, so `χ* = 0`. -/
theorem dc_fillOpt (A : Anatomy (Fin 28) (Fin 2) Unit Empty) (hA : A.Split) : FillOpt Sm90Measured.model A := by
  intro lam hlam
  have hchi : Sm90Measured.model.chiStar lam ≤ 0 := by
    refine (iInf_le_price lam Sm90Measured.model.fill ()).trans (le_of_eq ?_)
    show ∑ r, lam r * (0 : ℝ) = 0
    simp
  have hprice : 0 ≤ Sm90Measured.model.G.price lam A.nfill := by
    unfold Machine.price Machine.load
    exact Finset.sum_nonneg fun r _ => mul_nonneg (hlam.1 r)
      (Finset.sum_nonneg fun k _ => mul_nonneg (dc_a_nonneg r k) (hA.2.2.1 k))
  show (1 - (0 : ℝ)) * Sm90Measured.model.chiStar lam ≤ _
  rw [sub_zero, one_mul]
  exact hchi.trans hprice

/-- **`NoSum` fails on any program whose JI90_1 load is below its words' and fold's price there.** With A6, FOLD-OPT
at all 4096 words and the 4096 words fixed, the JI90_1 certificate prices the run's words and fold above its whole
load there. -/
theorem not_noSum_at (s : Sparse) (hs : dot (dcCoef 6) 28 s < 4096 * w1 + 4096 * phi1)
    (A : Anatomy (Fin 28) (Fin 2) Unit Empty) (hn : A.run.n = Sparse.counts s)
    (hA : A.Split) (hW : SheetWords Sm90Measured.model A) (hF : FoldOptFolded Sm90Measured.model A 4096)
    (hfix : 4096 ≤ A.fixed) : ¬ NoSum A := by
  intro hN
  have hSR : ScalarRoute Sm90Measured.model A :=
    scalarRoute_half _ A hA hW (le_of_eq dc_q) (le_of_eq dc_qT) le_rfl
  have key := price_of_parts_folded Sm90Measured.model Sm90Measured.flat A (onFacet ji1) 4096 (isCert_onFacet _)
    dc_wf hA hW.1 hSR (fun u => u.elim) hN hF (dc_fillOpt A hA)
  obtain ⟨-, -, -, hχ⟩ := stars_nonneg Sm90Measured.model Sm90Measured.flat (onFacet ji1) dc_wf (isCert_onFacet _)
  have hp : Sm90Measured.model.G.price (onFacet ji1) A.run.n = ((dot (dcRow ji1) 28 s : ℚ) : ℝ) := by
    rw [show Sm90Measured.model.G.price (onFacet ji1) A.run.n = Sm90Measured.model.G.load A.run.n ji1 from
      demandPrice_onFacet ji1 (Sm90Measured.model.G.load A.run.n), hn, load_counts _ dcRow dc_hG]
  have hlt : ((dot (dcRow ji1) 28 s : ℚ) : ℝ) < ((4096 * w1 + 4096 * phi1 : ℚ) : ℝ) :=
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
  linarith

/-! ### The statement's premises, path by path -/

theorem prob_univ {Ω : Type} [Fintype Ω] [Nonempty Ω] {p : Ω → Prop} (h : ∀ ω, p ω) : prob p = 1 := by
  classical
  have e : prob p = ((Finset.univ.filter p).card : ℝ) / Fintype.card Ω := by
    unfold prob
    congr
  rw [e, Finset.filter_true_of_mem fun ω _ => h ω, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)

theorem prob_pos_of {Ω : Type} [Fintype Ω] {p : Ω → Prop} {ω : Ω} (h : p ω) : 0 < prob p := by
  classical
  have e : prob p = ((Finset.univ.filter p).card : ℝ) / Fintype.card Ω := by
    unfold prob
    congr
  rw [e]
  have h1 : 0 < (Finset.univ.filter p).card :=
    Finset.card_pos.mpr ⟨ω, Finset.mem_filter.mpr ⟨Finset.mem_univ _, h⟩⟩
  have h2 : 0 < Fintype.card Ω := Fintype.card_pos_iff.mpr ⟨ω⟩
  exact div_pos (Nat.cast_pos.mpr h1) (Nat.cast_pos.mpr h2)

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

/-- On a game whose every path is accepted, the luck at `7/8` holds only at deficiency 0. -/
theorem deficiency_zero {Ω : Type} [Fintype Ω] [Nonempty Ω] :
    ∀ {n : ℕ} (g : RoundGame Ω n), RoundGame.LuckAt (7 / 8) g → (∀ x, g.accepts x) → ∀ x, g.deficiency x = 0
  | _, .done, _, _, _ => rfl
  | n + 1, .round d pass next, hL, hacc, x => by
      obtain ⟨hd01, hl, hL'⟩ := hL
      have hpass : ∀ ω, pass ω := fun ω => (show pass ω ∧ _ from hacc (ω, somePath n)).1
      have hd : ∀ ω, d ω = 0 := by
        intro ω
        by_contra hne
        have hpos : 0 < d ω := lt_of_le_of_ne (hd01 ω).1 (Ne.symm hne)
        have h1 : prob (fun ω => 0 < d ω) ≤ prob (fun ω => pass ω ∧ 0 < d ω) :=
          prob_mono fun ω h => ⟨hpass ω, h⟩
        have h2 := prob_pos_of (p := fun ω => 0 < d ω) hpos
        linarith
      show d x.1 + (next x.1).deficiency x.2 = 0
      rw [hd x.1, zero_add]
      exact deficiency_zero (next x.1) (hL' x.1) (fun y => (show pass x.1 ∧ _ from hacc (x.1, y)).2) x.2

/-- **`Sm90MeasuredHeadline`'s premises leave `NoSum` false on a play of any program `s` that `not_noSum_at`
separates.** For a play whose ideal game accepts every path, with one accepted path `x0` whose committed values are
right: the luck premises force share and deficiency 0 (`share_zero`, `deficiency_zero`), so FOLD-OPT is at all 4096
words and `RightFixed` fixes all 4096 of them on `x0`, and `NoSum` fails there. -/
theorem not_premises_at (s : Sparse) (hs : dot (dcCoef 6) 28 s < 4096 * w1 + 4096 * phi1)
    (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm90.checks (Fin 28) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V)
    (hn : ∀ x, (Pl.run x).run.n = Sparse.counts s) (hgame : ∀ x, Pl.game.accepts x)
    (x0 : Paths Ω Sm90.checks) (hx0 : Pl.accepts x0) (hr : pf.right (pf.committed x0))
    (hA : ∀ x, (Pl.run x).Split) (hW : ∀ x, SheetWords Sm90Measured.model (Pl.run x))
    (hF : ∀ x, FoldOptFolded Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw))
    (hG : Pl.game = G.game) (hPin : G.Pinned) (hOp : OperandHMu (3 / 4) G) (hRL : ReceiptLuckAt (1 / 2) G)
    (hWL : WholeLuckAt (1 / 2) G Uf) (hRF : RightFixed Pl pf Sm90Measured.model.Nw) :
    ¬ ∀ x, NoSum (Pl.run x) := by
  intro hN
  have hP : AlwaysPasses G := alwaysPasses_of_accepts G fun x => by
    rw [← hG]
    exact hgame x
  have h0 : Uf.share x0 = 0 := share_zero G Uf hPin hP hWL x0
  have hd0 : Pl.game.deficiency x0 = 0 :=
    deficiency_zero Pl.game (luck_of_receipt_half Pl.game G hG hPin hOp hRL) hgame x0
  have e : (1 - (0 : ℝ) / (Sm90.checks : ℝ)) * 4096 = 4096 := by norm_num
  have hfix : 4096 ≤ (Pl.run x0).fixed := by
    have h := hRF x0 hx0 hr
    rw [hd0, dc_Nw, e] at h
    exact h
  have hF0 : FoldOptFolded Sm90Measured.model (Pl.run x0) 4096 := by
    have h := hF x0
    rw [h0, dc_Nw, e] at h
    exact h
  exact not_noSum_at s hs (Pl.run x0) (hn x0) (hA x0) (hW x0) hF0 hfix (hN x0)

theorem n128_sep : dot (dcCoef 6) 28 n128All < 4096 * w1 + 4096 * phi1 := by decide +kernel

theorem rs_sep : dot (dcCoef 6) 28 rsAll < 4096 * w1 + 4096 * phi1 := by decide +kernel

theorem imma_sep : dot (dcCoef 6) 28 immaAll < 4096 * w1 + 4096 * phi1 := by decide +kernel

/-- The N128 program: honest's instruction at half the tile heads no word on the menu. -/
theorem not_premises_n128 (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm90.checks (Fin 28) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V)
    (hn : ∀ x, (Pl.run x).run.n = Sparse.counts n128All) (hgame : ∀ x, Pl.game.accepts x)
    (x0 : Paths Ω Sm90.checks) (hx0 : Pl.accepts x0) (hr : pf.right (pf.committed x0))
    (hA : ∀ x, (Pl.run x).Split) (hW : ∀ x, SheetWords Sm90Measured.model (Pl.run x))
    (hF : ∀ x, FoldOptFolded Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw))
    (hG : Pl.game = G.game) (hPin : G.Pinned) (hOp : OperandHMu (3 / 4) G) (hRL : ReceiptLuckAt (1 / 2) G)
    (hWL : WholeLuckAt (1 / 2) G Uf) (hRF : RightFixed Pl pf Sm90Measured.model.Nw) :
    ¬ ∀ x, NoSum (Pl.run x) :=
  not_premises_at n128All n128_sep Ω Pl G Uf pf hn hgame x0 hx0 hr hA hW hF hG hPin hOp hRL hWL hRF

/-- The register-sourced program. -/
theorem not_premises_rs (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm90.checks (Fin 28) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V)
    (hn : ∀ x, (Pl.run x).run.n = Sparse.counts rsAll) (hgame : ∀ x, Pl.game.accepts x)
    (x0 : Paths Ω Sm90.checks) (hx0 : Pl.accepts x0) (hr : pf.right (pf.committed x0))
    (hA : ∀ x, (Pl.run x).Split) (hW : ∀ x, SheetWords Sm90Measured.model (Pl.run x))
    (hF : ∀ x, FoldOptFolded Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw))
    (hG : Pl.game = G.game) (hPin : G.Pinned) (hOp : OperandHMu (3 / 4) G) (hRL : ReceiptLuckAt (1 / 2) G)
    (hWL : WholeLuckAt (1 / 2) G Uf) (hRF : RightFixed Pl pf Sm90Measured.model.Nw) :
    ¬ ∀ x, NoSum (Pl.run x) :=
  not_premises_at rsAll rs_sep Ω Pl G Uf pf hn hgame x0 hx0 hr hA hW hF hG hPin hOp hRL hWL hRF

/-- The IMMA16832 program. -/
theorem not_premises_imma (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (Pl : Play Ω Sm90.checks (Fin 28) (Fin 2) Unit Empty) (G : ReceiptGame Ω Sm90.checks)
    (Uf : Unfolded Ω Sm90.checks) {V : Type} {nπ : ℕ} (pf : ProofLayer Ω Sm90.checks nπ V)
    (hn : ∀ x, (Pl.run x).run.n = Sparse.counts immaAll) (hgame : ∀ x, Pl.game.accepts x)
    (x0 : Paths Ω Sm90.checks) (hx0 : Pl.accepts x0) (hr : pf.right (pf.committed x0))
    (hA : ∀ x, (Pl.run x).Split) (hW : ∀ x, SheetWords Sm90Measured.model (Pl.run x))
    (hF : ∀ x, FoldOptFolded Sm90Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm90.checks) * Sm90Measured.model.Nw))
    (hG : Pl.game = G.game) (hPin : G.Pinned) (hOp : OperandHMu (3 / 4) G) (hRL : ReceiptLuckAt (1 / 2) G)
    (hWL : WholeLuckAt (1 / 2) G Uf) (hRF : RightFixed Pl pf Sm90Measured.model.Nw) :
    ¬ ∀ x, NoSum (Pl.run x) :=
  not_premises_at immaAll imma_sep Ω Pl G Uf pf hn hgame x0 hx0 hr hA hW hF hG hPin hOp hRL hWL hRF

end Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2

#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.n128_bound
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.rs_bound
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.imma_bound
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.off_menu
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.not_noSum_at
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.deficiency_zero
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.not_premises_at
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.not_premises_n128
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.not_premises_rs
#print axioms Accounting.SecurityProofs.TS.HeadlineDc.RtLogic2.not_premises_imma
