import Proofs.Accounting.Headline.FpG1
import Proofs.Accounting.Headline.Sm120Measured

/-!
# FP8-g1's and W4A8-g1's headlines on sm_120's measured vector, and an honest g1 window that meets every premise

`Sm120FpG1MeasuredHeadline` and `Sm120W4a8G1MeasuredHeadline` are the measured record's argument with binding at
`δ_bind` (`sm120MeasuredHeadline_bind`, which `Sm120MeasuredHeadline` takes at `√δ_fork`) at `V = Leaves × W` and
`n_π + 1` proofs a round: the g1 premise is dropped and the argument applies as it stands. The g1 premise does no work
in the bound. The 0.591% rests on the measured record's premises, (I3) at `9/8` (`ScalarRoute`) among them, which is
assumed, as it is for the record itself: at A6's proved `½` floor the same bound is 7.791%. Beside the g1 premise, which
narrows the windows covered, the statements take no premise the measured record doesn't: binding at `δ_bind` is what
the record's fork form gives at `√δ_fork`.

The witness is `Proofs.Accounting.Headline.FpG1`'s honest window with its run on the measured data:

* its receipt game, unfolded share and proof layer are `FpG1`'s (`honestGame`, `honestUnfolded`, `honestLayer`);
* its run (`honestRunM`) is ♭'s counts on VF3s-16 at ♭'s bound `B_♭` there (`BqM`, the load at π* = JI3t_0) cycles and
  `f_max`, its word production IMMA16832's (writer kind 6, `ω = 128`) with ♭'s operand floor, its fold ♭'s lane fold, no
  filler, and its 128 committed words a unit fixed and headed by IMMA16832. The facet ceilings hold since every
  VF3s-16 row's load of ♭ is at most `B_♭` (`sm120m_max`), and every facet is SM-clocked; no word is vector- or
  table-headed, so SCALAR-ROUTE holds with nothing to route; and FOLD-OPT at `π*` holds since ♭'s fold is 128 lane
  folds, priced `128 φ_q` there, and `φ* ≤ φ_q` (`phiStarM_le`), `φ_q` the lane fold's load at `π*` (`phiqM`).

`MeasuredPremises` is the measured statements' hypotheses but the g1 premise, as one conjunction, and
`measured_bound_of_premises` feeds it to `sm120MeasuredHeadline_bind`. `measured_honest_premises` proves it at that
window, for any `v` in H1's ranges; `sm120FpG1Measured_honest` and `sm120W4a8G1Measured_honest` add the g1 premise at
the honest encoders' leaves, and `sm120FpG1Measured_honest_applies` and `sm120W4a8G1Measured_honest_applies` feed them
to the statements. The honest window is accepted on every path, its time exactly the credit
(`measured_honest_accepted`). `measured_offGrid_premises`: at `Pouw.FpG1.offGrid`'s leaves the other premises all hold
and FP8-g1's fails.

The witness isn't a served window, for `FpG1`'s reasons: its relation checks H1's ranges, its receipt game is the
honest one, and its time is exactly the credit, so it shows only that every premise can hold at once.
-/

namespace Accounting.SecurityProofs.FpG1

open Finset Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Pouw.FpG1
  Accounting.SecurityProofs.TS.Headline

/-- **FP8-g1's headline on sm_120's measured vector**: `sm120MeasuredHeadline_bind` at `V = Leaves × W` and `n_π + 1`
proofs a round. The 0.591% rests on the measured record's premises, `ScalarRoute` at `9/8` among them (assumed; at
A6's proved `½` floor the same bound is 7.791%). -/
theorem Sm120FpG1MeasuredHeadline : Accounting.Guarantees.FpG1.Sm120FpG1MeasuredHeadline := by
  intro Ω _ Pl G Uf W nπ pf δbind δzk _
  exact Accounting.SecurityProofs.TS.sm120MeasuredHeadline_bind Pl G Uf pf δbind δzk

/-- **W4A8-g1's headline on sm_120's measured vector**: `sm120MeasuredHeadline_bind` at `V = Leaves × W` and
`n_π + 1` proofs a round. The 0.591% rests on the measured record's premises, `ScalarRoute` at `9/8` among them
(assumed; at A6's proved `½` floor the same bound is 7.791%). -/
theorem Sm120W4a8G1MeasuredHeadline : Accounting.Guarantees.FpG1.Sm120W4a8G1MeasuredHeadline := by
  intro Ω _ Pl G Uf W nπ pf δbind δzk _
  exact Accounting.SecurityProofs.TS.sm120MeasuredHeadline_bind Pl G Uf pf δbind δzk

/-! ## The honest run on VF3s-16 -/

/-- ♭'s word production, the MMA and its operand floor, headed by IMMA16832 (writer kind 6). -/
noncomputable def honestNwordM : Head (Fin 15) Unit Empty → Fin 122 → ℝ
  | .writer t, k => if t = 6 then Sparse.counts Sm120.flatWords k else 0
  | .vector, _ => 0
  | .table _, _ => 0
  | .side _, _ => 0

/-- The 128 committed words a unit, every one IMMA16832's. -/
def honestWordsM : Head (Fin 15) Unit Empty → ℝ
  | .writer t => if t = 6 then 128 else 0
  | .vector => 0
  | .table _ => 0
  | .side _ => 0

/-- **The honest run on VF3s-16**: ♭'s counts at ♭'s bound `B_♭` cycles and `f_max`, its word production
IMMA16832's, its fold ♭'s lane fold, no filler, and its 128 committed words a unit fixed. -/
noncomputable def honestRunM : Anatomy (Fin 122) (Fin 15) Unit Empty where
  run := ⟨Sparse.counts flatAllM, (BqM : ℝ), Sm120Measured.model.G.fmax⟩
  nword := honestNwordM
  nfold := Sparse.counts Sm120.flatFold
  nfill := 0
  words := honestWordsM
  fixed := 128

theorem honestNwordM_ne {h : Head (Fin 15) Unit Empty} (hb : h ≠ .writer 6) (k : Fin 122) :
    honestNwordM h k = 0 := by
  rcases h with t | _ | _ | _
  · have ht : t ≠ 6 := fun e => hb (by rw [e])
    show (if t = 6 then Sparse.counts Sm120.flatWords k else 0) = 0
    rw [ite_eq_right ht]
  · rfl
  · rfl
  · rfl

theorem honestWordsM_ne {h : Head (Fin 15) Unit Empty} (hb : h ≠ .writer 6) : honestWordsM h = 0 := by
  rcases h with t | _ | _ | _
  · have ht : t ≠ 6 := fun e => hb (by rw [e])
    show (if t = 6 then (128 : ℝ) else 0) = 0
    rw [ite_eq_right ht]
  · rfl
  · rfl
  · rfl

theorem honestNwordM_sum (k : Fin 122) : ∑ h, honestNwordM h k = Sparse.counts Sm120.flatWords k := by
  rw [Finset.sum_eq_single (Head.writer 6) (fun _ _ hb => honestNwordM_ne hb k)
    (fun h => absurd (Finset.mem_univ _) h)]
  show (if (6 : Fin 15) = 6 then Sparse.counts Sm120.flatWords k else 0) = _
  rw [ite_eq_left rfl]

theorem honestWordsM_sum : ∑ h, honestWordsM h = 128 := by
  rw [Finset.sum_eq_single (Head.writer 6) (fun _ _ hb => honestWordsM_ne hb)
    (fun h => absurd (Finset.mem_univ _) h)]
  show (if (6 : Fin 15) = 6 then (128 : ℝ) else 0) = 128
  rw [ite_eq_left rfl]

theorem honestRunM_split : honestRunM.Split := by
  refine ⟨fun h k => ?_, fun k => Rat.cast_nonneg.mpr (coeff_nonneg _ _), fun _ => le_rfl, fun h => ?_, fun k => ?_⟩
  · rcases h with t | _ | _ | _
    · show 0 ≤ if t = 6 then Sparse.counts Sm120.flatWords k else 0
      split_ifs
      · exact Rat.cast_nonneg.mpr (coeff_nonneg _ _)
      · exact le_rfl
    · exact le_rfl
    · exact le_rfl
    · exact le_rfl
  · rcases h with t | _ | _ | _
    · show (0 : ℝ) ≤ if t = 6 then 128 else 0
      split_ifs <;> norm_num
    · exact le_rfl
    · exact le_rfl
    · exact le_rfl
  · show ∑ h, honestNwordM h k + Sparse.counts Sm120.flatFold k + (0 : Fin 122 → ℝ) k ≤ Sparse.counts flatAllM k
    rw [honestNwordM_sum]
    have e : Sparse.coeff flatAllM k.val = Sparse.coeff Sm120.flatWords k.val + Sparse.coeff Sm120.flatFold k.val +
        Sparse.coeff [] k.val + Sparse.coeff Sm120Measured.flatSkip k.val := by
      unfold flatAllM
      rw [coeff_append, coeff_append, coeff_append]
    have h0 : Sparse.coeff [] k.val = 0 := by simp [Sparse.coeff]
    have hs := coeff_nonneg Sm120Measured.flatSkip k.val
    have hq : Sparse.coeff Sm120.flatWords k.val + Sparse.coeff Sm120.flatFold k.val ≤
        Sparse.coeff flatAllM k.val := by
      rw [e, h0]
      linarith
    show (Sparse.coeff Sm120.flatWords k.val : ℝ) + (Sparse.coeff Sm120.flatFold k.val : ℝ) + 0 ≤
      (Sparse.coeff flatAllM k.val : ℝ)
    rw [add_zero]
    exact_mod_cast hq

/-- (I1) at the honest run: every VF3s-16 row's load of ♭ is at most `B_♭` (`sm120m_max`), its counts are `≥ 0`, its
clock is `f_max`, and every facet is SM-clocked. -/
theorem honestRunM_ceilings : FacetCeilings Sm120Measured.machine honestRunM.run := by
  refine ⟨fun k => Rat.cast_nonneg.mpr (coeff_nonneg _ _), ?_, ?_, le_rfl, fun r => ⟨fun _ => ?_, fun h => ?_⟩⟩
  · show (0 : ℝ) ≤ (BqM : ℝ)
    exact_mod_cast sm120m_Bpos.le
  · show (0 : ℝ) < ((2430 : ℕ) : ℝ)
    norm_num
  · show Sm120Measured.model.G.load (Sparse.counts flatAllM) r ≤ (BqM : ℝ)
    rw [load_counts _ vfRow sm120m_hG]
    exact Rat.cast_le.mpr (sm120m_max r.val r.isLt)
  · exact absurd (show true = false from h) (by decide)

theorem honestRunM_words : WordsWritten Sm120Measured.model honestRunM := by
  intro t
  show (if t = 6 then (128 : ℝ) else 0) ≤
    Sm120Measured.model.omega t *
      (if t = 6 then Sparse.counts Sm120.flatWords (Sm120Measured.model.cls t) else 0)
  split_ifs with ht
  · subst ht
    have h1 : Sparse.coeff Sm120.flatWords 27 = 1 := by decide +kernel
    show (128 : ℝ) ≤ ((128 : ℕ) : ℝ) * ((Sparse.coeff Sm120.flatWords 27 : ℚ) : ℝ)
    rw [h1]
    norm_num
  · simp

theorem honestRunM_scalar : ScalarRoute Sm120Measured.model honestRunM := by
  refine ⟨?_, fun _ => ⟨?_, ?_⟩⟩
  · show Sm120Measured.model.q * (0 : ℝ) ≤ 32 * ∑ _k ∈ Sm120Measured.model.vec, (0 : ℝ)
    simp
  · show Sm120Measured.model.qT * (0 : ℝ) ≤ 32 * ∑ _k ∈ Sm120Measured.model.vec, (0 : ℝ)
    simp
  · show Sm120Measured.model.bT * (0 : ℝ) ≤ 0
    simp

theorem honestRunM_noSum : NoSum honestRunM := by
  show (128 : ℝ) ≤ ∑ h, honestWordsM h
  exact honestWordsM_sum.ge

/-- `φ*(π*) ≤ φ_q`, the lane fold's load at `π*` (`phiqM`). -/
theorem phiStarM_le : Sm120Measured.model.phiStar (onFacet Sm120Measured.piStar) ≤ (phiqM : ℝ) := by
  refine (Accounting.SecurityProofs.TS.iInf_le_price (onFacet Sm120Measured.piStar) Sm120Measured.model.fold 0).trans
    (le_of_eq ?_)
  rw [demandPrice_onFacet]
  show Sm120Measured.model.G.load (Sparse.counts (Sm120Measured.h1fx1.folds.getD 0 [])) Sm120Measured.piStar = _
  rw [load_counts _ vfRow sm120m_hG]
  rfl

/-- ♭'s fold is 128 lane folds, so it costs `128 φ_q` a unit at `π*`. -/
theorem honestM_foldPrice :
    Sm120Measured.model.G.price (onFacet Sm120Measured.piStar) (Sparse.counts Sm120.flatFold) =
      128 * (phiqM : ℝ) := by
  show demandPrice (onFacet Sm120Measured.piStar) (Sm120Measured.model.G.load (Sparse.counts Sm120.flatFold)) = _
  rw [demandPrice_onFacet, load_counts _ vfRow sm120m_hG]
  have h : dot (vfRow Sm120Measured.piStar) 122 Sm120.flatFold = 128 * phiqM := by decide +kernel
  rw [h]
  norm_num

theorem honestRunM_foldOpt (Ω : Type) (n : ℕ) (x : Paths Ω n) :
    FoldOptFoldedAt Sm120Measured.model honestRunM
      ((1 - (honestUnfolded Ω n).share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar) := by
  unfold FoldOptFoldedAt
  rw [honestUnfolded_share, show Sm120Measured.model.muF = 0 from rfl, sm120m_Nw]
  show _ ≤ Sm120Measured.model.G.price (onFacet Sm120Measured.piStar) (Sparse.counts Sm120.flatFold)
  rw [honestM_foldPrice]
  have := phiStarM_le
  simp only [sub_zero, zero_div, one_mul]
  linarith

/-- The honest window's time on VF3s-16 is exactly the credit `B_♭/f_max`. -/
theorem honestRunM_time : honestRunM.run.time / Sm120Measured.model.credit Sm120Measured.flat = 1 := by
  have hB : (0 : ℝ) < BqM := by exact_mod_cast sm120m_Bpos
  have hf : (0 : ℝ) < Sm120Measured.model.G.fmax := by
    show (0 : ℝ) < ((2430 : ℕ) : ℝ)
    norm_num
  unfold Model.credit Run.time
  rw [sm120m_bnd]
  show (BqM : ℝ) / Sm120Measured.model.G.fmax / ((BqM : ℝ) / Sm120Measured.model.G.fmax) = 1
  exact div_self (div_pos hB hf).ne'

/-! ## The honest play -/

/-- **The honest play on the measured model's heads**: the honest game's acceptance, the window accepted on every path,
the run `A` on each. -/
noncomputable def honestPlayM (Ω : Type) (n : ℕ) (A : Anatomy (Fin 122) (Fin 15) Unit Empty) :
    Play Ω n (Fin 122) (Fin 15) Unit Empty :=
  ⟨(honestGame Ω n).game, fun _ => True, fun _ => A⟩

theorem honestPlayM_game (Ω : Type) (n : ℕ) (A : Anatomy (Fin 122) (Fin 15) Unit Empty) :
    (honestPlayM Ω n A).game = (honestGame Ω n).game :=
  rfl

theorem honestLayer_ofM (Ω : Type) [Fintype Ω] (n : ℕ) (v : Leaves) (A : Anatomy (Fin 122) (Fin 15) Unit Empty) :
    (honestLayer Ω n v).Of (honestPlayM Ω n A) :=
  ⟨fun _ _ _ _ => trivial, fun x _ => honestGame_accepts Ω n x⟩

theorem honestLayer_rightFixedM (Ω : Type) [Fintype Ω] (v : Leaves) :
    RightFixed (honestPlayM Ω Sm120.checks honestRunM) (honestLayer Ω Sm120.checks v) Sm120Measured.model.Nw := by
  intro x _ _
  show (1 - (honestGame Ω Sm120.checks).game.deficiency x / Sm120.checks) * Sm120Measured.model.Nw ≤ (128 : ℝ)
  rw [honestGame_deficiency, sm120m_Nw]
  norm_num

/-! ## The witnesses -/

/-- **The measured statements' premises at a window but the g1 premise**, `sm120MeasuredHeadline_bind`'s hypotheses
in its order (`measured_bound_of_premises` feeds them to it): `Sm120MeasuredHeadline`'s, with binding at `δ_bind` in
place of its fork form. -/
def MeasuredPremises {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty)
    (G : ReceiptGame Ω Sm120.checks) (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ}
    (pf : ProofLayer Ω Sm120.checks nπ V) (δbind δzk : ℝ) : Prop :=
  (Sm120.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
    (∀ x, FacetCeilings Sm120Measured.machine (Pl.run x).run) ∧
    (∀ x, WordsWritten Sm120Measured.model (Pl.run x)) ∧ (∀ x, ScalarRoute Sm120Measured.model (Pl.run x)) ∧
    (∀ x, NoSum (Pl.run x)) ∧
    (∀ x, FoldOptFoldedAt Sm120Measured.model (Pl.run x)
      ((1 - Uf.share x / Sm120.checks) * Sm120Measured.model.Nw) (onFacet Sm120Measured.piStar)) ∧
    Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
    pf.Of Pl ∧ pf.BindingAt δbind ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧ RightFixed Pl pf Sm120Measured.model.Nw

/-- The measured statements' conclusion at a window whose premises but the g1 premise hold. -/
theorem measured_bound_of_premises {Ω : Type} [Fintype Ω] {Pl : Play Ω Sm120.checks (Fin 122) (Fin 15) Unit Empty}
    {G : ReceiptGame Ω Sm120.checks} {Uf : Unfolded Ω Sm120.checks} {V : Type} {nπ : ℕ}
    {pf : ProofLayer Ω Sm120.checks nπ V} {δbind δzk : ℝ} (h : MeasuredPremises Pl G Uf pf δbind δzk) :
    prob (fun x => Pl.accepts x ∧
      (Pl.run x).run.time / Sm120Measured.model.credit Sm120Measured.flat < 1 - 591 / 100000) ≤ (1 / 2) ^ 40 := by
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ := h
  exact Accounting.SecurityProofs.TS.sm120MeasuredHeadline_bind Pl G Uf pf δbind δzk h0 h1 h2 h3 h4 h5 h6 h7 h8 h9
    h10 h11 h12 h13 h14 h15 h16

/-- **Every premise but the g1 premise holds at the honest window on VF3s-16**, for any leaves `v` in H1's ranges:
the honest play on `honestRunM`, receipt game and unfolded share over `S = 48,000` rounds, the honest proof layer at
`v`, and `δ_bind = δ_ZK = 0`. The witness isn't a served window: its relation is H1's ranges, not a served
Program's. -/
theorem measured_honest_premises (Ω : Type) [Fintype Ω] {v : Leaves} (hv : v.InH1) :
    MeasuredPremises (honestPlayM Ω Sm120.checks honestRunM) (honestGame Ω Sm120.checks)
      (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks v) 0 0 :=
  ⟨honest_budget, fun _ => honestRunM_split, fun _ => honestRunM_ceilings, fun _ => honestRunM_words,
    fun _ => honestRunM_scalar, fun _ => honestRunM_noSum, fun x => honestRunM_foldOpt Ω Sm120.checks x,
    honestPlayM_game Ω Sm120.checks honestRunM, honestGame_pinned Ω Sm120.checks,
    honestGame_hmu Ω (3 / 4) Sm120.checks, honestGame_receiptLuck Ω (by norm_num) Sm120.checks,
    honestGame_wholeLuck Ω (1 / 2) Sm120.checks, honestLayer_ofM Ω Sm120.checks v honestRunM,
    honestLayer_binding Ω Sm120.checks v, honestLayer_sound Ω Sm120.checks hv, honestLayer_pins Ω Sm120.checks v,
    honestLayer_rightFixedM Ω v⟩

/-- **The honest window on VF3s-16 is accepted on every path, its time exactly the credit.** -/
theorem measured_honest_accepted (Ω : Type) (x : Paths Ω Sm120.checks) :
    (honestPlayM Ω Sm120.checks honestRunM).accepts x ∧
      ((honestPlayM Ω Sm120.checks honestRunM).run x).run.time / Sm120Measured.model.credit Sm120Measured.flat = 1 :=
  ⟨trivial, honestRunM_time⟩

/-- **Every premise of FP8-g1's measured statement holds at an honest FP8-g1 window on VF3s-16**: the honest
encoder's leaves on H1's grids, at any scales and model values, committed at every round, with the other premises.
The witness isn't a served window (`measured_honest_premises`). -/
theorem sm120FpG1Measured_honest (Ω : Type) [Fintype Ω] (sa sw : ℕ → ℚ) (x y : ℕ → ℕ → ℚ) :
    (∀ p w i, ((honestLayer Ω Sm120.checks (encodeFp8 Grids.h1 sa sw x y)).committed p w i).1.Fp8 Grids.h1) ∧
      MeasuredPremises (honestPlayM Ω Sm120.checks honestRunM) (honestGame Ω Sm120.checks)
        (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks (encodeFp8 Grids.h1 sa sw x y)) 0 0 :=
  ⟨fun _ _ _ => encodeFp8_fp8 Grids.h1 sa sw x y,
    measured_honest_premises Ω (Leaves.fp8_inH1 (encodeFp8_fp8 _ sa sw x y))⟩

/-- **Every premise of W4A8-g1's measured statement holds at an honest W4A8-g1 window on VF3s-16**: the honest
encoder's leaves on H1's grids, at any scales, block scales and model values, committed at every round, with the
other premises. The witness isn't a served window (`measured_honest_premises`). -/
theorem sm120W4a8G1Measured_honest (Ω : Type) [Fintype Ω] (sa sw : ℕ → ℚ) (m : ℕ → ℕ → ℕ) (x y : ℕ → ℕ → ℚ) :
    (∀ p w i, ((honestLayer Ω Sm120.checks (encodeW4a8 Grids.h1 sa sw m x y)).committed p w i).1.W4a8 Grids.h1) ∧
      MeasuredPremises (honestPlayM Ω Sm120.checks honestRunM) (honestGame Ω Sm120.checks)
        (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks (encodeW4a8 Grids.h1 sa sw m x y)) 0 0 :=
  ⟨fun _ _ _ => encodeW4a8_w4a8 sa sw m x y,
    measured_honest_premises Ω (Leaves.w4a8_inH1 (encodeW4a8_w4a8 sa sw m x y))⟩

/-- **FP8-g1's measured headline applies at the honest FP8-g1 window**: `sm120FpG1Measured_honest`'s premises fed to
it. -/
theorem sm120FpG1Measured_honest_applies (Ω : Type) [Fintype Ω] (sa sw : ℕ → ℚ) (x y : ℕ → ℕ → ℚ) :
    prob (fun p => (honestPlayM Ω Sm120.checks honestRunM).accepts p ∧
      ((honestPlayM Ω Sm120.checks honestRunM).run p).run.time / Sm120Measured.model.credit Sm120Measured.flat <
        1 - 591 / 100000) ≤ (1 / 2) ^ 40 := by
  obtain ⟨hg, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ :=
    sm120FpG1Measured_honest Ω sa sw x y
  exact Sm120FpG1MeasuredHeadline Ω (honestPlayM Ω Sm120.checks honestRunM) (honestGame Ω Sm120.checks)
    (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks (encodeFp8 Grids.h1 sa sw x y)) 0 0 hg h0 h1 h2 h3 h4
    h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16

/-- **W4A8-g1's measured headline applies at the honest W4A8-g1 window**: `sm120W4a8G1Measured_honest`'s premises fed
to it. -/
theorem sm120W4a8G1Measured_honest_applies (Ω : Type) [Fintype Ω] (sa sw : ℕ → ℚ) (m : ℕ → ℕ → ℕ)
    (x y : ℕ → ℕ → ℚ) :
    prob (fun p => (honestPlayM Ω Sm120.checks honestRunM).accepts p ∧
      ((honestPlayM Ω Sm120.checks honestRunM).run p).run.time / Sm120Measured.model.credit Sm120Measured.flat <
        1 - 591 / 100000) ≤ (1 / 2) ^ 40 := by
  obtain ⟨hg, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ :=
    sm120W4a8G1Measured_honest Ω sa sw m x y
  exact Sm120W4a8G1MeasuredHeadline Ω (honestPlayM Ω Sm120.checks honestRunM) (honestGame Ω Sm120.checks)
    (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks (encodeW4a8 Grids.h1 sa sw m x y)) 0 0 hg h0 h1 h2
    h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16

/-- **The other premises don't imply the g1 premise on VF3s-16 either**: at `offGrid`'s leaves, in H1's ranges with
a weight off E4M3's grid, every premise of the measured statement but the g1 premise holds, so the bound does, and
FP8-g1's premise fails at every round's commitment. -/
theorem measured_offGrid_premises (Ω : Type) [Fintype Ω] :
    MeasuredPremises (honestPlayM Ω Sm120.checks honestRunM) (honestGame Ω Sm120.checks)
        (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks offGrid) 0 0 ∧
      ∀ p w i, ¬ ((honestLayer Ω Sm120.checks offGrid).committed p w i).1.Fp8 Grids.h1 :=
  ⟨measured_honest_premises Ω offGrid_inH1, fun _ _ _ => offGrid_not_fp8⟩

end Accounting.SecurityProofs.FpG1
