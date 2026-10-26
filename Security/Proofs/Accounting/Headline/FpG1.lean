import Proofs.Accounting.Headline.Sm120
import Proofs.Pouw.FpG1.H1
import Definitions.Accounting.Guarantees.FpG1

/-!
# FP8-g1's and W4A8-g1's headlines on sm_120, and an honest g1 window that meets every premise

`Sm120FpG1SheetHeadline` and `Sm120W4a8G1SheetHeadline` are the record's argument with binding at `δ_bind`
(`sm120SheetHeadline_bind`, which `Sm120SheetHeadline` takes at `√δ_fork`) at `V = Leaves × W` and `n_π + 1` proofs a
round: the g1 premise is dropped and the argument applies as it stands. The g1 premise does no work in the bound.

The witness is an honest window on any finite coins `Ω`, generic in the rounds `n` and taken at `S = 48,000`:

* its receipt game (`honestGame`) has no deficient round, every check passes, every word is right and every
  committed value is the honest one; no word's fold is skipped (`honestUnfolded`);
* its run (`honestRun`) is ♭'s counts at ♭'s bound `B_♭` cycles and `f_max`, its word production IMMA16832's (writer
  kind 6, `ω = 128`) with ♭'s operand floor, its fold ♭'s lane fold, no filler, and its 128 committed words a unit
  fixed and headed by IMMA16832. A1–A7 hold at the sheet's coefficients, issue being ♭'s largest facet, and FOLD-OPT
  at `π*` (`c = 3564/3761` on issue, `197/3761` on shared memory) since the lane fold costs `16c = 128 · c/8` there,
  16 issue slots and no shared memory, and `φ* ≤ c/8`;
* its proof layer (`honestLayer`) is one proof a round, accepted on every path, opening the round's leaves to one
  `Leaves` value `v`; the proof's relation is H1's committed ranges (`Leaves.InH1`), and a window's values are right
  when every leaf is in them. Binding and soundness hold at `δ_bind = δ_ZK = 0`.

`Premises` is the statements' hypotheses but the g1 premise, as one conjunction, and `bound_of_premises` feeds it to
`sm120SheetHeadline_bind`. `honest_premises` proves it at that window, for any `v` in H1's ranges. `sm120FpG1_honest`
and `sm120W4a8G1_honest` add the g1 premise at the honest encoders' leaves (`encodeFp8`, `encodeW4a8`), which are g1's
on H1's grids and so in H1's ranges (`Leaves.fp8_inH1`, `Leaves.w4a8_inH1`): every premise of each statement holds
there, and `sm120FpG1_honest_applies` and `sm120W4a8G1_honest_applies` feed them to the statements. The honest window
is accepted on every path, its time exactly the credit (`honest_accepted`, `honestRun_time`).

`offGrid_premises`: at `Pouw.FpG1.offGrid`'s leaves, in H1's ranges but off E4M3's grid, the other premises all hold
and FP8-g1's fails (`offGrid_not_fp8`), so the other premises don't imply the g1 premise.

The witness isn't a served window. Its relation checks H1's ranges; it is no model of the served Programs, which no
Lean here models (`ProofsPin`'s docstring). Its receipt game is the honest one, so it doesn't exercise H1's coin-keyed
receipt check, and its time is exactly the credit, so it shows only that every premise can hold at once.
-/

namespace Accounting.SecurityProofs.FpG1

open Finset Accounting.TS Accounting.TS.Headline Accounting.Assumptions.TS Pouw.RelativeSecurity Pouw.FpG1
  Accounting.SecurityProofs.TS.Headline

/-- **FP8-g1's headline on sm_120**: `sm120SheetHeadline_bind` at `V = Leaves × W` and `n_π + 1` proofs a round. -/
theorem Sm120FpG1SheetHeadline : Accounting.Guarantees.FpG1.Sm120FpG1SheetHeadline := by
  intro Ω _ Pl G Uf W nπ pf δbind δzk _
  exact Accounting.SecurityProofs.TS.sm120SheetHeadline_bind Pl G Uf pf δbind δzk

/-- **W4A8-g1's headline on sm_120**: `sm120SheetHeadline_bind` at `V = Leaves × W` and `n_π + 1` proofs a round. -/
theorem Sm120W4a8G1SheetHeadline : Accounting.Guarantees.FpG1.Sm120W4a8G1SheetHeadline := by
  intro Ω _ Pl G Uf W nπ pf δbind δzk _
  exact Accounting.SecurityProofs.TS.sm120SheetHeadline_bind Pl G Uf pf δbind δzk

theorem prob_le_of_none {Ω : Type} [Fintype Ω] {p : Ω → Prop} {c : ℝ} (hc : 0 ≤ c) (h : ∀ ω, ¬ p ω) :
    prob p ≤ c :=
  (Accounting.HMuProofs.prob_eq_zero h).le.trans hc

/-! ## The honest receipt game -/

/-- The honest receipt game: no round deficient, every check passed, every deficient word right and every committed
value the honest one. -/
def honestGame (Ω : Type) : (n : ℕ) → ReceiptGame Ω n
  | 0 => .done
  | n + 1 => .round (fun _ => 0) (fun _ => True) (fun _ => True) (fun _ => True) fun _ => honestGame Ω n

/-- No word's fold is skipped. -/
def honestUnfolded (Ω : Type) : (n : ℕ) → Unfolded Ω n
  | 0 => .done
  | n + 1 => .round 0 fun _ => honestUnfolded Ω n

theorem honestGame_pinned (Ω : Type) : ∀ n, (honestGame Ω n).Pinned
  | 0 => trivial
  | n + 1 => ⟨fun _ _ => trivial, fun _ => honestGame_pinned Ω n⟩

theorem honestGame_hmu (Ω : Type) [Fintype Ω] (μ : ℝ) : ∀ n, (honestGame Ω n).OperandPointMass μ
  | 0 => trivial
  | n + 1 => by
    refine ⟨fun _ => ⟨le_rfl, zero_le_one⟩, ?_, fun _ => honestGame_hmu Ω μ n⟩
    refine ⟨Unit, Unit, ⟨fun _ => (), fun _ => (), fun _ => 0, fun _ => (), fun _ u => u⟩, ?_, fun _ _ => rfl,
      fun _ => rfl, fun _ _ => rfl⟩
    intro _ h
    exact (lt_irrefl _ h).elim

theorem honestGame_receiptLuck (Ω : Type) [Fintype Ω] {pr : ℝ} (hpr : 0 ≤ pr) :
    ∀ n, (honestGame Ω n).ReceiptLuck pr
  | 0 => trivial
  | n + 1 => by
    refine ⟨?_, fun _ => honestGame_receiptLuck Ω hpr n⟩
    refine prob_le_of_none (mul_nonneg hpr (prob_nonneg _)) fun _ h => ?_
    exact lt_irrefl _ h.2.2

theorem honestGame_wholeLuck (Ω : Type) [Fintype Ω] (p0 : ℝ) :
    ∀ n, (honestGame Ω n).WholeLuck p0 (honestUnfolded Ω n)
  | 0 => trivial
  | n + 1 => by
    refine ⟨⟨le_rfl, zero_le_one⟩, fun h => ?_, fun _ => honestGame_wholeLuck Ω p0 n⟩
    exact (lt_irrefl _ h).elim

theorem honestGame_accepts (Ω : Type) : ∀ n (x : Paths Ω n), (honestGame Ω n).game.accepts x
  | 0, _ => trivial
  | n + 1, x => ⟨trivial, honestGame_accepts Ω n x.2⟩

theorem honestGame_deficiency (Ω : Type) : ∀ n (x : Paths Ω n), (honestGame Ω n).game.deficiency x = 0
  | 0, _ => rfl
  | n + 1, x => by
    show (0 : ℝ) + (honestGame Ω n).game.deficiency x.2 = 0
    rw [honestGame_deficiency Ω n x.2, add_zero]

theorem honestUnfolded_share (Ω : Type) : ∀ n (x : Paths Ω n), (honestUnfolded Ω n).share x = 0
  | 0, _ => rfl
  | n + 1, x => by
    show (0 : ℝ) + (honestUnfolded Ω n).share x.2 = 0
    rw [honestUnfolded_share Ω n x.2, add_zero]

/-! ## The honest run -/

/-- ♭'s word production, the MMA and its operand floor, headed by IMMA16832 (writer kind 6). -/
noncomputable def honestNword : Head (Fin 17) Unit Empty → Fin 122 → ℝ
  | .writer t, k => if t = 6 then Sparse.counts Sm120.flatWords k else 0
  | .vector, _ => 0
  | .table _, _ => 0
  | .side _, _ => 0

/-- The 128 committed words a unit, every one IMMA16832's. -/
def honestWords : Head (Fin 17) Unit Empty → ℝ
  | .writer t => if t = 6 then 128 else 0
  | .vector => 0
  | .table _ => 0
  | .side _ => 0

/-- **The honest run**: ♭'s counts at ♭'s bound `B_♭` cycles and `f_max`, its word production IMMA16832's, its fold
♭'s lane fold, no filler, and its 128 committed words a unit fixed. -/
noncomputable def honestRun : Anatomy (Fin 122) (Fin 17) Unit Empty where
  run := ⟨Sparse.counts flatAll, (Bq : ℝ), Sm120.sheet.fmax⟩
  nword := honestNword
  nfold := Sparse.counts Sm120.flatFold
  nfill := 0
  words := honestWords
  fixed := 128

theorem honestNword_ne {h : Head (Fin 17) Unit Empty} (hb : h ≠ .writer 6) (k : Fin 122) : honestNword h k = 0 := by
  rcases h with t | _ | _ | _
  · have ht : t ≠ 6 := fun e => hb (by rw [e])
    show (if t = 6 then Sparse.counts Sm120.flatWords k else 0) = 0
    rw [ite_eq_right ht]
  · rfl
  · rfl
  · rfl

theorem honestWords_ne {h : Head (Fin 17) Unit Empty} (hb : h ≠ .writer 6) : honestWords h = 0 := by
  rcases h with t | _ | _ | _
  · have ht : t ≠ 6 := fun e => hb (by rw [e])
    show (if t = 6 then (128 : ℝ) else 0) = 0
    rw [ite_eq_right ht]
  · rfl
  · rfl
  · rfl

theorem honestNword_sum (k : Fin 122) : ∑ h, honestNword h k = Sparse.counts Sm120.flatWords k := by
  rw [Finset.sum_eq_single (Head.writer 6) (fun _ _ hb => honestNword_ne hb k)
    (fun h => absurd (Finset.mem_univ _) h)]
  show (if (6 : Fin 17) = 6 then Sparse.counts Sm120.flatWords k else 0) = _
  rw [ite_eq_left rfl]

theorem honestWords_sum : ∑ h, honestWords h = 128 := by
  rw [Finset.sum_eq_single (Head.writer 6) (fun _ _ hb => honestWords_ne hb) (fun h => absurd (Finset.mem_univ _) h)]
  show (if (6 : Fin 17) = 6 then (128 : ℝ) else 0) = 128
  rw [ite_eq_left rfl]

theorem honestRun_split : honestRun.Split := by
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
  · show ∑ h, honestNword h k + Sparse.counts Sm120.flatFold k + (0 : Fin 122 → ℝ) k ≤ Sparse.counts flatAll k
    rw [honestNword_sum]
    have e : Sparse.coeff flatAll k.val = Sparse.coeff Sm120.flatWords k.val + Sparse.coeff Sm120.flatFold k.val +
        Sparse.coeff [] k.val + Sparse.coeff Sm120.flatSkip k.val := by
      unfold flatAll
      rw [coeff_append, coeff_append, coeff_append]
    have h0 : Sparse.coeff [] k.val = 0 := by simp [Sparse.coeff]
    have hs := coeff_nonneg Sm120.flatSkip k.val
    have hq : Sparse.coeff Sm120.flatWords k.val + Sparse.coeff Sm120.flatFold k.val ≤ Sparse.coeff flatAll k.val := by
      rw [e, h0]
      linarith
    show (Sparse.coeff Sm120.flatWords k.val : ℝ) + (Sparse.coeff Sm120.flatFold k.val : ℝ) + 0 ≤
      (Sparse.coeff flatAll k.val : ℝ)
    rw [add_zero]
    exact_mod_cast hq

theorem honest_load (r : SheetRow Sm120.Pipe) : Sm120.model.G.load (Sparse.counts flatAll) r ≤ (Bq : ℝ) := by
  rw [load_counts _ (sheetCoef Sm120.ds120) sm120_hG]
  exact_mod_cast sm120_max r

theorem honestRun_clock : SheetClock Sm120.sheet honestRun.run := by
  refine ⟨?_, le_rfl⟩
  show (0 : ℝ) < ((2430 : ℕ) : ℝ)
  norm_num

theorem honestRun_dispatch : SheetDispatch honestRun.run := by
  have e : Sm120.model.G.load (Sparse.counts flatAll) .issue = ∑ k : Fin 122, Sparse.counts flatAll k := by
    unfold Machine.load
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [sm120_hG]
    simp [sheetCoef]
  show ∑ k : Fin 122, Sparse.counts flatAll k ≤ (Bq : ℝ)
  rw [← e]
  exact honest_load .issue

theorem honestRun_tensor : SheetTensor Sm120.sheet honestRun.run := by
  show Sm120.model.G.load (Sparse.counts flatAll) .tensor ≤ (Bq : ℝ)
  exact honest_load .tensor

theorem honestRun_pipe : SheetPipe Sm120.sheet honestRun.run := by
  intro p
  have hw : (0 : ℝ) < Sm120.sheet.width p := by
    show (0 : ℝ) < ((Sm120.Pipe.width p : ℕ) : ℝ)
    exact_mod_cast (by cases p <;> decide : 0 < Sm120.Pipe.width p)
  have hw0 := hw.ne'
  have e : ∑ k, (if Sm120.sheet.pipe k = some p then (32 : ℝ) else 0) * Sparse.counts flatAll k =
      Sm120.sheet.width p / 4 * Sm120.model.G.load (Sparse.counts flatAll) (.pipe p) := by
    unfold Machine.load
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    show _ = _ * ((if Sm120.sheet.pipe k = some p then 128 / Sm120.sheet.width p else 0) * _)
    split_ifs
    · field_simp
      ring
    · ring
  show ∑ k, (if Sm120.sheet.pipe k = some p then (32 : ℝ) else 0) * Sparse.counts flatAll k ≤
    Sm120.sheet.width p / 4 * (Bq : ℝ)
  rw [e]
  exact mul_le_mul_of_nonneg_left (honest_load (.pipe p)) (by positivity)

theorem honestRun_smem : SheetSmem Sm120.sheet honestRun.run := by
  have e : ∑ k, Sm120.sheet.smem k * Sparse.counts flatAll k =
      32 * Sm120.model.G.load (Sparse.counts flatAll) .smem := by
    unfold Machine.load
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    show _ = 32 * (Sm120.sheet.smem k / 32 * _)
    ring
  show ∑ k, Sm120.sheet.smem k * Sparse.counts flatAll k ≤ 32 * (Bq : ℝ)
  rw [e]
  exact mul_le_mul_of_nonneg_left (honest_load .smem) (by norm_num)

theorem honestRun_sheetWords : SheetWords Sm120.model honestRun := by
  refine ⟨fun t => ?_, ?_, fun _ => ?_⟩
  · show (if t = 6 then (128 : ℝ) else 0) ≤
      Sm120.model.omega t * (if t = 6 then Sparse.counts Sm120.flatWords (Sm120.model.cls t) else 0)
    split_ifs with ht
    · subst ht
      have h1 : Sparse.coeff Sm120.flatWords 27 = 1 := by decide +kernel
      show (128 : ℝ) ≤ ((128 : ℕ) : ℝ) * ((Sparse.coeff Sm120.flatWords 27 : ℚ) : ℝ)
      rw [h1]
      norm_num
    · simp
  · show (0 : ℝ) ≤ 2 * (32 * ∑ _k ∈ Sm120.model.vec, (0 : ℝ))
    simp
  · show (0 : ℝ) ≤ 2 * (32 * ∑ _k ∈ Sm120.model.vec, (0 : ℝ))
    simp

theorem honestRun_semantics : SheetSemantics Sm120.model honestRun := by
  refine ⟨?_, fun u => u.elim⟩
  show (128 : ℝ) ≤ ∑ h, honestWords h
  exact honestWords_sum.ge

/-- `φ*(π*) ≤ c/8`, `c = 3564/3761`: the lane fold's issue count a word, with no shared memory. -/
theorem phiStar_le : Sm120.model.phiStar Sm120.piStar ≤ 3564 / 3761 / 8 := by
  refine (Accounting.SecurityProofs.TS.iInf_le_price Sm120.piStar Sm120.model.fold 0).trans (le_of_eq ?_)
  rw [demandPrice_piStar]
  show 3564 / 3761 * Sm120.model.G.load (Sparse.counts [(0, 1, 16), (24, 1, 16)]) .issue +
      197 / 3761 * Sm120.model.G.load (Sparse.counts [(0, 1, 16), (24, 1, 16)]) .smem = 3564 / 3761 / 8
  rw [load_counts _ (sheetCoef Sm120.ds120) sm120_hG, load_counts _ (sheetCoef Sm120.ds120) sm120_hG]
  have hi : dot (sheetCoef Sm120.ds120 .issue) 122 [(0, 1, 16), (24, 1, 16)] = 1 / 8 := by decide +kernel
  have hs : dot (sheetCoef Sm120.ds120 .smem) 122 [(0, 1, 16), (24, 1, 16)] = 0 := by decide +kernel
  rw [hi, hs]
  norm_num

/-- ♭'s fold costs `16c` a unit at `π*`, `c = 3564/3761`: 16 issue slots and no shared memory. -/
theorem honest_foldPrice :
    Sm120.model.G.price Sm120.piStar (Sparse.counts Sm120.flatFold) = 3564 / 3761 * 16 := by
  show demandPrice Sm120.piStar (Sm120.model.G.load (Sparse.counts Sm120.flatFold)) = 3564 / 3761 * 16
  rw [demandPrice_piStar, load_counts _ (sheetCoef Sm120.ds120) sm120_hG,
    load_counts _ (sheetCoef Sm120.ds120) sm120_hG]
  have hi : dot (sheetCoef Sm120.ds120 .issue) 122 Sm120.flatFold = 16 := by decide +kernel
  have hs : dot (sheetCoef Sm120.ds120 .smem) 122 Sm120.flatFold = 0 := by decide +kernel
  rw [hi, hs]
  norm_num

theorem honestRun_foldOpt (Ω : Type) (n : ℕ) (x : Paths Ω n) :
    FoldOptFoldedAt Sm120.model honestRun ((1 - (honestUnfolded Ω n).share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar := by
  unfold FoldOptFoldedAt
  rw [honestUnfolded_share, show Sm120.model.muF = 0 from rfl, sm120_Nw]
  show _ ≤ Sm120.model.G.price Sm120.piStar (Sparse.counts Sm120.flatFold)
  rw [honest_foldPrice]
  have := phiStar_le
  simp only [sub_zero, zero_div, one_mul]
  linarith

/-- The honest window's time is exactly the credit `B_♭/f_max`. -/
theorem honestRun_time : honestRun.run.time / Sm120.model.credit Sm120.flat = 1 := by
  have hB : (0 : ℝ) < Bq := by exact_mod_cast sm120_Bpos
  have hf : (0 : ℝ) < Sm120.sheet.fmax := honestRun_clock.1
  unfold Model.credit Run.time
  rw [sm120_bnd]
  show (Bq : ℝ) / Sm120.sheet.fmax / ((Bq : ℝ) / Sm120.sheet.fmax) = 1
  exact div_self (div_pos hB hf).ne'

/-! ## The honest play and proof layer -/

/-- **The honest play**: the honest game's acceptance, the window accepted on every path, the honest run on each. -/
noncomputable def honestPlay (Ω : Type) (n : ℕ) (A : Anatomy (Fin 122) (Fin 17) Unit Empty) :
    Play Ω n (Fin 122) (Fin 17) Unit Empty :=
  ⟨(honestGame Ω n).game, fun _ => True, fun _ => A⟩

theorem honestPlay_game (Ω : Type) (n : ℕ) (A : Anatomy (Fin 122) (Fin 17) Unit Empty) :
    (honestPlay Ω n A).game = (honestGame Ω n).game :=
  rfl

/-- **The honest proof layer**: one proof a round, accepted on every path, opening the round's leaves to `v` and
nothing else beside them; its relation is H1's committed ranges on every path, and the window's values are right on a
path when every leaf is in them. -/
def honestLayer (Ω : Type) (n : ℕ) (v : Leaves) : ProofLayer Ω n (0 + 1) (Leaves × Unit) where
  accepted _ _ _ := True
  opened _ _ _ := (v, ())
  rel _ _ _ c := c.1.InH1
  right _ c := ∀ w i, (c w i).1.InH1

theorem honestLayer_committed (Ω : Type) [Fintype Ω] (n : ℕ) (v : Leaves) (x : Paths Ω n) (w : Fin n)
    (i : Fin (0 + 1)) : ((honestLayer Ω n v).committed x w i).1 = v :=
  rfl

theorem honestLayer_of (Ω : Type) [Fintype Ω] (n : ℕ) (v : Leaves) (A : Anatomy (Fin 122) (Fin 17) Unit Empty) :
    (honestLayer Ω n v).Of (honestPlay Ω n A) :=
  ⟨fun _ _ _ _ => trivial, fun x _ => honestGame_accepts Ω n x⟩

theorem honestLayer_binding (Ω : Type) [Fintype Ω] (n : ℕ) (v : Leaves) : (honestLayer Ω n v).BindingAt 0 := by
  intro w
  refine prob_le_of_none le_rfl fun x h => ?_
  obtain ⟨-, i, hi⟩ := h
  exact hi rfl

theorem honestLayer_sound (Ω : Type) [Fintype Ω] (n : ℕ) {v : Leaves} (hv : v.InH1) :
    ProofSoundAt (honestLayer Ω n v) 0 := by
  intro w i
  exact prob_le_of_none le_rfl fun _ h => h.2 hv

theorem honestLayer_pins (Ω : Type) (n : ℕ) (v : Leaves) : ProofsPin (honestLayer Ω n v) :=
  fun _ _ h => h

theorem honestLayer_rightFixed (Ω : Type) [Fintype Ω] (v : Leaves) :
    RightFixed (honestPlay Ω Sm120.checks honestRun) (honestLayer Ω Sm120.checks v) Sm120.model.Nw := by
  intro x _ _
  show (1 - (honestGame Ω Sm120.checks).game.deficiency x / Sm120.checks) * Sm120.model.Nw ≤ (128 : ℝ)
  rw [honestGame_deficiency, sm120_Nw]
  norm_num

theorem honest_budget : (Sm120.checks : ℝ) * (0 + ((0 + 1 : ℕ) : ℝ) * 0) ≤ (1 / 2) ^ 41 := by
  rw [mul_zero, add_zero, mul_zero]
  positivity

/-! ## The witnesses -/

/-- **The statements' premises at a window but the g1 premise**, `sm120SheetHeadline_bind`'s hypotheses in its order
(`bound_of_premises` feeds them to it): `Sm120SheetHeadline`'s, with binding at `δ_bind` in place of its fork form. -/
def Premises {Ω : Type} [Fintype Ω] (Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty)
    (G : ReceiptGame Ω Sm120.checks) (Uf : Unfolded Ω Sm120.checks) {V : Type} {nπ : ℕ}
    (pf : ProofLayer Ω Sm120.checks nπ V) (δbind δzk : ℝ) : Prop :=
  (Sm120.checks : ℝ) * (δbind + nπ * δzk) ≤ (1 / 2) ^ 41 ∧ (∀ x, (Pl.run x).Split) ∧
    (∀ x, SheetClock Sm120.sheet (Pl.run x).run) ∧ (∀ x, SheetDispatch (Pl.run x).run) ∧
    (∀ x, SheetTensor Sm120.sheet (Pl.run x).run) ∧ (∀ x, SheetPipe Sm120.sheet (Pl.run x).run) ∧
    (∀ x, SheetSmem Sm120.sheet (Pl.run x).run) ∧ (∀ x, SheetWords Sm120.model (Pl.run x)) ∧
    (∀ x, SheetSemantics Sm120.model (Pl.run x)) ∧
    (∀ x, FoldOptFoldedAt Sm120.model (Pl.run x) ((1 - Uf.share x / Sm120.checks) * Sm120.model.Nw)
      Sm120.piStar) ∧
    Pl.game = G.game ∧ G.Pinned ∧ OperandHMu (3 / 4) G ∧ ReceiptLuckAt (1 / 2) G ∧ WholeLuckAt (1 / 2) G Uf ∧
    pf.Of Pl ∧ pf.BindingAt δbind ∧ ProofSoundAt pf δzk ∧ ProofsPin pf ∧ RightFixed Pl pf Sm120.model.Nw

/-- The statements' conclusion at a window whose premises but the g1 premise hold. -/
theorem bound_of_premises {Ω : Type} [Fintype Ω] {Pl : Play Ω Sm120.checks (Fin 122) (Fin 17) Unit Empty}
    {G : ReceiptGame Ω Sm120.checks} {Uf : Unfolded Ω Sm120.checks} {V : Type} {nπ : ℕ}
    {pf : ProofLayer Ω Sm120.checks nπ V} {δbind δzk : ℝ} (h : Premises Pl G Uf pf δbind δzk) :
    prob (fun x => Pl.accepts x ∧ (Pl.run x).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40 := by
  obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h
  exact Accounting.SecurityProofs.TS.sm120SheetHeadline_bind Pl G Uf pf δbind δzk h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    h11 h12 h13 h14 h15 h16 h17 h18 h19

/-- **Every premise but the g1 premise holds at the honest window**, for any leaves `v` in H1's ranges: the honest
play, receipt game and unfolded share over `S = 48,000` rounds, the honest proof layer at `v`, and
`δ_bind = δ_ZK = 0`. The witness isn't a served window: its relation is H1's ranges, not a served Program's. -/
theorem honest_premises (Ω : Type) [Fintype Ω] {v : Leaves} (hv : v.InH1) :
    Premises (honestPlay Ω Sm120.checks honestRun) (honestGame Ω Sm120.checks) (honestUnfolded Ω Sm120.checks)
      (honestLayer Ω Sm120.checks v) 0 0 :=
  ⟨honest_budget, fun _ => honestRun_split, fun _ => honestRun_clock, fun _ => honestRun_dispatch,
    fun _ => honestRun_tensor, fun _ => honestRun_pipe, fun _ => honestRun_smem, fun _ => honestRun_sheetWords,
    fun _ => honestRun_semantics, fun x => honestRun_foldOpt Ω Sm120.checks x,
    honestPlay_game Ω Sm120.checks honestRun, honestGame_pinned Ω Sm120.checks,
    honestGame_hmu Ω (3 / 4) Sm120.checks, honestGame_receiptLuck Ω (by norm_num) Sm120.checks,
    honestGame_wholeLuck Ω (1 / 2) Sm120.checks, honestLayer_of Ω Sm120.checks v honestRun,
    honestLayer_binding Ω Sm120.checks v, honestLayer_sound Ω Sm120.checks hv, honestLayer_pins Ω Sm120.checks v,
    honestLayer_rightFixed Ω v⟩

/-- **The honest window is accepted on every path, its time exactly the credit.** -/
theorem honest_accepted (Ω : Type) (x : Paths Ω Sm120.checks) :
    (honestPlay Ω Sm120.checks honestRun).accepts x ∧
      ((honestPlay Ω Sm120.checks honestRun).run x).run.time / Sm120.model.credit Sm120.flat = 1 :=
  ⟨trivial, honestRun_time⟩

/-- **Every premise of FP8-g1's statement holds at an honest FP8-g1 window**: the honest encoder's leaves on H1's
grids, at any scales and model values, committed at every round, with the other premises. The witness isn't a
served window (`honest_premises`). -/
theorem sm120FpG1_honest (Ω : Type) [Fintype Ω] (sa sw : ℕ → ℚ) (x y : ℕ → ℕ → ℚ) :
    (∀ p w i, ((honestLayer Ω Sm120.checks (encodeFp8 Grids.h1 sa sw x y)).committed p w i).1.Fp8 Grids.h1) ∧
      Premises (honestPlay Ω Sm120.checks honestRun) (honestGame Ω Sm120.checks)
        (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks (encodeFp8 Grids.h1 sa sw x y)) 0 0 :=
  ⟨fun _ _ _ => encodeFp8_fp8 Grids.h1 sa sw x y, honest_premises Ω (Leaves.fp8_inH1 (encodeFp8_fp8 _ sa sw x y))⟩

/-- **Every premise of W4A8-g1's statement holds at an honest W4A8-g1 window**: the honest encoder's leaves on H1's
grids, at any scales, block scales and model values, committed at every round, with the other premises. The
witness isn't a served window (`honest_premises`). -/
theorem sm120W4a8G1_honest (Ω : Type) [Fintype Ω] (sa sw : ℕ → ℚ) (m : ℕ → ℕ → ℕ) (x y : ℕ → ℕ → ℚ) :
    (∀ p w i, ((honestLayer Ω Sm120.checks (encodeW4a8 Grids.h1 sa sw m x y)).committed p w i).1.W4a8 Grids.h1) ∧
      Premises (honestPlay Ω Sm120.checks honestRun) (honestGame Ω Sm120.checks)
        (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks (encodeW4a8 Grids.h1 sa sw m x y)) 0 0 :=
  ⟨fun _ _ _ => encodeW4a8_w4a8 sa sw m x y, honest_premises Ω (Leaves.w4a8_inH1 (encodeW4a8_w4a8 sa sw m x y))⟩

/-- **FP8-g1's headline applies at the honest FP8-g1 window**: `sm120FpG1_honest`'s premises fed to it. -/
theorem sm120FpG1_honest_applies (Ω : Type) [Fintype Ω] (sa sw : ℕ → ℚ) (x y : ℕ → ℕ → ℚ) :
    prob (fun p => (honestPlay Ω Sm120.checks honestRun).accepts p ∧
      ((honestPlay Ω Sm120.checks honestRun).run p).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40 := by
  obtain ⟨hg, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ :=
    sm120FpG1_honest Ω sa sw x y
  exact Sm120FpG1SheetHeadline Ω (honestPlay Ω Sm120.checks honestRun) (honestGame Ω Sm120.checks)
    (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks (encodeFp8 Grids.h1 sa sw x y)) 0 0 hg h0 h1 h2 h3 h4
    h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19

/-- **W4A8-g1's headline applies at the honest W4A8-g1 window**: `sm120W4a8G1_honest`'s premises fed to it. -/
theorem sm120W4a8G1_honest_applies (Ω : Type) [Fintype Ω] (sa sw : ℕ → ℚ) (m : ℕ → ℕ → ℕ) (x y : ℕ → ℕ → ℚ) :
    prob (fun p => (honestPlay Ω Sm120.checks honestRun).accepts p ∧
      ((honestPlay Ω Sm120.checks honestRun).run p).run.time / Sm120.model.credit Sm120.flat < 1 - 5860 / 100000) ≤
      (1 / 2) ^ 40 := by
  obtain ⟨hg, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ :=
    sm120W4a8G1_honest Ω sa sw m x y
  exact Sm120W4a8G1SheetHeadline Ω (honestPlay Ω Sm120.checks honestRun) (honestGame Ω Sm120.checks)
    (honestUnfolded Ω Sm120.checks) (honestLayer Ω Sm120.checks (encodeW4a8 Grids.h1 sa sw m x y)) 0 0 hg h0 h1 h2 h3
    h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h17 h18 h19

/-- **The other premises don't imply the g1 premise**: at `offGrid`'s leaves, in H1's ranges with a weight off E4M3's
grid, every premise but the g1 premise holds, so the bound does, and FP8-g1's premise fails at every round's
commitment. -/
theorem offGrid_premises (Ω : Type) [Fintype Ω] :
    Premises (honestPlay Ω Sm120.checks honestRun) (honestGame Ω Sm120.checks) (honestUnfolded Ω Sm120.checks)
        (honestLayer Ω Sm120.checks offGrid) 0 0 ∧
      ∀ p w i, ¬ ((honestLayer Ω Sm120.checks offGrid).committed p w i).1.Fp8 Grids.h1 :=
  ⟨honest_premises Ω offGrid_inH1, fun _ _ _ => offGrid_not_fp8⟩

end Accounting.SecurityProofs.FpG1
