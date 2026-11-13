module

public import Proofs.Accounting.RateAudit.SketchHonest
public import Proofs.Accounting.RateAudit.SketchInteractive
public import Proofs.Accounting.RateAudit.Witnesses

/-!
# The sketch rate audit's guarantees: each one's honest and nontrivial lemmas

For `RateAuditSketch` and `RateAuditSketchInteractiveRO`, at one tile of one one-bit sector whose digest is the sector
(`RateSketch.firstSector`), one sign, a one-bit tag, no certified stream and one sampled tile, and for
`RateAuditTensorSketchRank` at one tile of one 32-byte sector, no sign rows and rank `r = 0`:

* **The honest lemma** (`<guarantee>_honest`): the honest device (`RateSketch.Completeness.honest`; as an interactive
  device, `RateSketch.honestOne`, which reads the sector and tags it) is accepted by the audit's own check, within one
  tag's compute, with its footprint at the floor, with probability 1.
* **The nontrivial lemma** (`<guarantee>_nontrivial`): with the check replaced by `True`, a device that reads nothing
  (`RateSketch.blank`, `RateSketch.blankOne`) is short of the floor with probability 1, above the guarantee's error:
  `2^-1`, the fiber and rank terms `0`.
-/

@[expose] public section

namespace Accounting.RateSketch

open Pous (Bits pr)

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- The digest of a one-sector tile: the sector. -/
def firstSector : Unit → (Fin 1 → Bool) → Bool := fun _ x => x 0

/-- The device that reads nothing: it holds no sector and commits to all-`false` tags at once. -/
def blank (N m ρ h S L q w : ℕ) (X Sig Dg : Type) : Prover N m ρ h S L q w X Sig Dg where
  held := ∅
  held_le := by simp
  chip _ _ := .done fun _ _ => false
  bounded _ _ _ _ := ⟨by simp [Chip.linkSteps], by simp [Chip.queries]⟩

theorem blank_compute (cTag cEnc : ℕ) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    (blank N m ρ h S L q w X Sig Dg).compute cTag cEnc σ O D = 0 := by
  simp [Prover.compute, Prover.prog, blank, Chip.queries, Chip.rebuilds]

theorem blank_footprint (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) :
    (blank N m ρ h S L q w X Sig Dg).footprint σ O D = 0 := by
  simp [Prover.footprint, Prover.prog, blank, Chip.mems]

/-- `blank` as an interactive device, at one tile of one one-bit sector. -/
def blankOne : InteractiveProver 1 1 0 1 0 0 0 1 Bool Unit Bool where
  held := ∅
  held_le := by simp
  addr _ := .inl (0, 0)
  chip _ _ := .done fun _ _ => false
  bounded _ _ _ _ := ⟨by simp [IChip.linkSteps], by simp [IChip.queries]⟩

theorem blankOne_compute (cTag cEnc : ℕ) (σ : Unit) (O : Oracle 1 0 1 Bool Bool) (D : Data 1 1 0 Bool) :
    blankOne.compute cTag cEnc σ O D = 0 := by
  simp [InteractiveProver.compute, InteractiveProver.prog, blankOne, IChip.queries, IChip.rebuilds]

theorem blankOne_footprint (σ : Unit) (O : Oracle 1 0 1 Bool Bool) (D : Data 1 1 0 Bool) :
    blankOne.footprint σ O D = 0 := by
  simp [InteractiveProver.footprint, InteractiveProver.prog, blankOne, IChip.mems]

/-- The honest interactive device at one tile of one one-bit sector: it reads the sector from device memory, queries
the oracle at it, and never sends. -/
def honestOne : InteractiveProver 1 1 0 1 0 0 1 1 Bool Unit Bool where
  held := ∅
  held_le := by simp
  addr _ := .inl (0, 0)
  chip _ _ := .mem (0, 0) fun x => .tag ((fun r => r.elim0), 0, x) fun t => .done fun _ => t
  bounded _ _ _ _ := ⟨by simp [IChip.linkSteps], by simp [IChip.queries]⟩

theorem honestOne_accepts (σ : Unit) (O : Oracle 1 0 1 Bool Bool) (D : Data 1 1 0 Bool) (I : Fin 1 → Fin 1) :
    honestOne.Accepts firstSector σ O D I ∧ honestOne.footprint σ O D = 1 ∧ honestOne.compute 1 1 σ O D = 1 := by
  refine ⟨fun i => ?_, ?_, ?_⟩
  · show O ((fun r : Fin 0 => r.elim0), 0, D (.inl (0, 0))) = O (D.stream, I i, firstSector σ (D.tile (I i)))
    refine congrArg O (Prod.ext (funext fun r => r.elim0) (Prod.ext (Subsingleton.elim _ _) ?_))
    show D (.inl (0, 0)) = D (.inl (I i, 0))
    rw [Subsingleton.elim (I i) 0]
  · simp [InteractiveProver.footprint, InteractiveProver.prog, honestOne, IChip.mems]
  · simp [InteractiveProver.compute, InteractiveProver.prog, honestOne, IChip.queries, IChip.rebuilds]

/-- At a one-bit sector the digest's fiber is at most the sector's two values. -/
theorem fib_firstSector_le (σ : Unit) : fib firstSector σ ≤ 2 := by
  unfold fib
  exact max_le (by norm_num) (Finset.sup_le fun p _ => (Finset.card_le_univ _).trans (by simp))

end Accounting.RateSketch

namespace Accounting.SecurityProofs

open Accounting.RateSketch
open Pous (Bits pr)
open scoped ENNReal

theorem pr_zero {α : Type} [Fintype α] {P : α → Prop} (h : ∀ x, ¬ P x) : pr P = 0 := by
  classical
  simp only [Pous.pr_eq_card]
  rw [Finset.filter_false_of_mem fun x _ => h x, Finset.card_empty, Nat.cast_zero, ENNReal.zero_div]

/-- The sampling and guess terms of the sketch audits' error at `N = 1`, `k = 1`, `s = 1`, `h = 1`: `2^-1`. -/
theorem sketch_error_lt :
    (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 < 1 := by
  have t1 : (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 = 0 := by simp
  have t2 : ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 = 2⁻¹ := by simp
  rw [t1, t2, zero_add]
  exact ENNReal.inv_lt_one.2 ENNReal.one_lt_two

/-- **`RateAuditSketch`'s honest lemma**: the honest device, at one tile of one one-bit sector, is accepted within one
tag's compute with its footprint at the floor, with probability 1. -/
theorem RateAuditSketch_honest :
    pr (fun x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool =>
        (RateSketch.Completeness.honest 1 1 0 1 1 Bool Unit Bool firstSector).Accepts firstSector x.2.1.1 x.2.1.2 x.1
            x.2.2 ∧
          (RateSketch.Completeness.honest 1 1 0 1 1 Bool Unit Bool firstSector).compute 1 1 x.2.1.1 x.2.1.2 x.1 ≤ 1 ∧
          ¬ (RateSketch.Completeness.honest 1 1 0 1 1 Bool Unit Bool firstSector).footprint x.2.1.1 x.2.1.2 x.1 * 1 <
            RateSketch.floor 1 1 0 0 0 1 1 1 1) = 1 :=
  pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ :=
      RateSketch.Completeness.honest_accepts 1 firstSector 1 1 x.2.1.1 x.2.1.2 x.1 x.2.2
    refine ⟨ha, by omega, ?_⟩
    rw [hf]
    unfold Accounting.RateSketch.floor
    omega

/-- **`RateAuditSketch`'s nontrivial lemma**: with the check accepting everything, the device that reads nothing is
within its compute and short of the floor on every coin, above the error `2^-1` (the fiber term is `0` at `g = 2`). -/
theorem RateAuditSketch_nontrivial :
    ¬ pr (fun x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool =>
        True ∧ (blank 1 1 0 1 0 0 0 1 Bool Unit Bool).compute 0 1 x.2.1.1 x.2.1.2 x.1 ≤ 0 ∧
          (blank 1 1 0 1 0 0 0 1 Bool Unit Bool).footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 1 0 0 0 0 0 1 1)
      ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
        pr (fun σ : Unit => 2 < fib firstSector σ) + ((0 * 2 : ℕ) : ℝ≥0∞) / (Fintype.card Bool : ℝ≥0∞) := by
  have hev : ∀ x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool,
      True ∧ (blank 1 1 0 1 0 0 0 1 Bool Unit Bool).compute 0 1 x.2.1.1 x.2.1.2 x.1 ≤ 0 ∧
        (blank 1 1 0 1 0 0 0 1 Bool Unit Bool).footprint x.2.1.1 x.2.1.2 x.1 * 1 <
          RateSketch.floor 1 1 0 0 0 0 0 1 1 := fun x =>
    ⟨trivial, (blank_compute _ _ _ _ _).le, by
      rw [blank_footprint]
      unfold Accounting.RateSketch.floor
      omega⟩
  have hfib : pr (fun σ : Unit => 2 < fib firstSector σ) = 0 := pr_zero fun σ => not_lt.2 (fib_firstSector_le σ)
  have t4 : ((0 * 2 : ℕ) : ℝ≥0∞) / (Fintype.card Bool : ℝ≥0∞) = 0 := by simp
  rw [pr_one hev, hfib, t4, add_zero, add_zero, not_le]
  exact sketch_error_lt

/-- **`RateAuditSketchInteractiveRO`'s honest lemma**: the honest interactive device, at one tile of one one-bit
sector, is accepted within one tag's compute with its footprint at the floor, with probability 1. -/
theorem RateAuditSketchInteractiveRO_honest :
    pr (fun x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool =>
        honestOne.Accepts firstSector x.2.1.1 x.2.1.2 x.1 x.2.2 ∧ honestOne.compute 1 1 x.2.1.1 x.2.1.2 x.1 ≤ 1 ∧
          ¬ honestOne.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 1 0 0 0 1 1 1 1) = 1 :=
  pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ := honestOne_accepts x.2.1.1 x.2.1.2 x.1 x.2.2
    refine ⟨ha, by omega, ?_⟩
    rw [hf]
    unfold Accounting.RateSketch.floor
    omega

/-- **`RateAuditSketchInteractiveRO`'s nontrivial lemma**: with the check accepting everything, the interactive device
that reads nothing is short of the floor on every coin, above the error `2^-1`. -/
theorem RateAuditSketchInteractiveRO_nontrivial :
    ¬ pr (fun x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool =>
        True ∧ blankOne.compute 0 1 x.2.1.1 x.2.1.2 x.1 ≤ 0 ∧
          blankOne.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 1 0 0 0 0 0 1 1)
      ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
        pr (fun σ : Unit => 2 < fib firstSector σ) + ((0 * 2 : ℕ) : ℝ≥0∞) / (Fintype.card Bool : ℝ≥0∞) := by
  have hev : ∀ x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool,
      True ∧ blankOne.compute 0 1 x.2.1.1 x.2.1.2 x.1 ≤ 0 ∧
        blankOne.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 1 0 0 0 0 0 1 1 := fun x =>
    ⟨trivial, (blankOne_compute _ _ _ _ _).le, by
      rw [blankOne_footprint]
      unfold Accounting.RateSketch.floor
      omega⟩
  have hfib : pr (fun σ : Unit => 2 < fib firstSector σ) = 0 := pr_zero fun σ => not_lt.2 (fib_firstSector_le σ)
  have t4 : ((0 * 2 : ℕ) : ℝ≥0∞) / (Fintype.card Bool : ℝ≥0∞) = 0 := by simp
  rw [pr_one hev, hfib, t4, add_zero, add_zero, not_le]
  exact sketch_error_lt

/-- **`RateAuditTensorSketchRank`'s honest lemma**: the honest device, at one tile of one 32-byte sector under the
tensor-core sketch with no sign rows, is accepted within one tag's compute with its footprint at the floor, with
probability 1. -/
theorem RateAuditTensorSketchRank_honest :
    pr (fun x : Data 1 1 0 Sector × Coins 1 0 1 1 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)) =>
        (RateSketch.Completeness.honest 1 1 0 1 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)) sketch).Accepts
            sketch x.2.1.1 x.2.1.2 x.1 x.2.2 ∧
          (RateSketch.Completeness.honest 1 1 0 1 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32))
            sketch).compute 1 1 x.2.1.1 x.2.1.2 x.1 ≤ 1 ∧
          ¬ (RateSketch.Completeness.honest 1 1 0 1 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32))
            sketch).footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 256 0 0 0 1 1 1 1) = 1 :=
  pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ := RateSketch.Completeness.honest_accepts 256 sketch 1 1 x.2.1.1 x.2.1.2 x.1 x.2.2
    refine ⟨ha, by omega, ?_⟩
    rw [hf]
    unfold Accounting.RateSketch.floor
    omega

/-- **`RateAuditTensorSketchRank`'s nontrivial lemma**: with the check accepting everything, the device that reads
nothing is short of the floor on every coin, above the error `2^-1` at `r = 0` (the rank and guess terms are `0`). -/
theorem RateAuditTensorSketchRank_nontrivial :
    ¬ pr (fun x : Data 1 1 0 Sector × Coins 1 0 1 1 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)) =>
        True ∧ (blank 1 1 0 1 0 0 0 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32))).compute 0 1 x.2.1.1
            x.2.1.2 x.1 ≤ 0 ∧
          (blank 1 1 0 1 0 0 0 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32))).footprint x.2.1.1 x.2.1.2 x.1 *
            1 < RateSketch.floor 1 256 0 0 0 0 0 1 1)
      ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
        pr (fun σ : SketchSigns 0 1 => ∃ u, ((Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank < 0) +
        ((0 : ℕ) : ℝ≥0∞) / 2 ^ (8 * 0) := by
  have hev : ∀ x : Data 1 1 0 Sector × Coins 1 0 1 1 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)),
      True ∧ (blank 1 1 0 1 0 0 0 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32))).compute 0 1 x.2.1.1
          x.2.1.2 x.1 ≤ 0 ∧
        (blank 1 1 0 1 0 0 0 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32))).footprint x.2.1.1 x.2.1.2 x.1 *
          1 < RateSketch.floor 1 256 0 0 0 0 0 1 1 := fun x =>
    ⟨trivial, (blank_compute _ _ _ _ _).le, by
      rw [blank_footprint]
      unfold Accounting.RateSketch.floor
      omega⟩
  have hrank : pr (fun σ : SketchSigns 0 1 =>
      ∃ u, ((Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank < 0) = 0 :=
    pr_zero fun _ ⟨_, hu⟩ => Nat.not_lt_zero _ hu
  have t4 : ((0 : ℕ) : ℝ≥0∞) / 2 ^ (8 * 0) = 0 := by simp
  rw [pr_one hev, hrank, t4, add_zero, add_zero, not_le]
  exact sketch_error_lt

end Accounting.SecurityProofs
