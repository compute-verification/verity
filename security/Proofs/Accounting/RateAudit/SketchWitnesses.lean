module

public import Proofs.Accounting.RateAudit.SketchHonest
public import Proofs.Accounting.RateAudit.SketchInteractive
public import Proofs.Accounting.RateAudit.Witnesses

/-!
# The sketch rate audit's guarantees: each one's honest and nontrivial lemmas

For `RateAuditSketch` and `RateAuditSketchInteractiveRO`, at one tile of one one-bit sector whose digest is the sector
(`RateSketch.firstSector`), one sign, a one-bit tag, no certified stream and one sampled tile, and for
`RateAuditTensorSketchRank` at one tile of one 32-byte sector, no sign rows and rank `r = 0`, and for
`RateAuditTensorSketch` at one tile of one 32-byte sector under `33` sign rows, which bind it at `RateSketch.bindSigns`:

* **The honest lemma** (`<guarantee>_honest`): the honest device (`RateSketch.Completeness.honest`; as an interactive
  device, `RateSketch.honestOne`, which reads the sector and tags it) is accepted by the audit's own check, within one
  tag's compute, with its footprint at the floor, with probability 1.
* **The nontrivial lemma** (`<guarantee>_nontrivial`): with the check replaced by `True`, a device that reads nothing
  (`RateSketch.blank`, `RateSketch.blankOne`) is short of the floor with probability 1, above the guarantee's error:
  `2^-1`, the fiber and rank terms `0`; for `RateAuditTensorSketch`, `2^-1057` and the signs' term, below `1 − 2^-1056`.
Each lemma states that some device (and, for the first two guarantees, some digest) has its property and proves it of
the named one.
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

/-- Signs that bind a sector: row `0` is all `+1`, and row `l + 1` is `+1` but for `−1` at byte `l`, so rows `0` and
`l + 1` differ by `2·int8(x l)`. -/
def bindSigns : SketchSigns 33 1 := fun i _ l => decide ((i : ℕ) = l + 1)

theorem bindSigns_zero (y : Fin 32 → ℤ) (u : Fin 1) : Tag.project (bindSigns.block u) y 0 = ∑ j, y j := by
  simp [Tag.project, SketchSigns.block, bindSigns, Tag.sgn]

theorem bindSigns_succ (y : Fin 32 → ℤ) (u : Fin 1) (l : Fin 32) :
    Tag.project (bindSigns.block u) y l.succ = (∑ j, y j) - 2 * y l := by
  have h : ∀ j : Fin 32, Tag.sgn (bindSigns.block u l.succ j) * y j = y j - 2 * (if l = j then y j else 0) := by
    intro j
    by_cases hj : l = j
    · subst hj
      simp [SketchSigns.block, bindSigns, Tag.sgn]
      ring
    · have hv : (l : ℕ) ≠ j := fun e => hj (Fin.ext e)
      simp [SketchSigns.block, bindSigns, Tag.sgn, Fin.val_succ, hv, hj]
  simp only [Tag.project, h, Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_ite_eq, Finset.mem_univ, ite_true]

theorem int8_injective : Function.Injective int8 := by
  intro a b h
  have := a.isLt
  have := b.isLt
  unfold int8 at h
  apply Fin.ext
  split_ifs at h <;> omega

theorem bindSigns_binds : SectorsBind bindSigns := by
  intro u a b hab
  have hab' : Tag.project (bindSigns.block u) (fun l => int8 (a l)) =
      Tag.project (bindSigns.block u) (fun l => int8 (b l)) := hab
  have h0 := congrFun hab' 0
  rw [bindSigns_zero, bindSigns_zero] at h0
  funext l
  have hl := congrFun hab' l.succ
  rw [bindSigns_succ, bindSigns_succ] at hl
  exact int8_injective (by omega)

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
    ∃ (dig : Unit → (Fin 1 → Bool) → Bool) (A : Prover 1 1 0 1 0 0 1 1 Bool Unit Bool),
      pr (fun x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool =>
          A.Accepts dig x.2.1.1 x.2.1.2 x.1 x.2.2 ∧
            A.compute 1 1 x.2.1.1 x.2.1.2 x.1 ≤ 1 ∧
            ¬ A.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 1 0 0 0 1 1 1 1) = 1 :=
  ⟨firstSector, RateSketch.Completeness.honest 1 1 0 1 1 Bool Unit Bool firstSector, pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ :=
      RateSketch.Completeness.honest_accepts 1 firstSector 1 1 x.2.1.1 x.2.1.2 x.1 x.2.2
    refine ⟨ha, by omega, ?_⟩
    rw [hf]
    unfold Accounting.RateSketch.floor
    omega⟩

/-- **`RateAuditSketch`'s nontrivial lemma**: with the check accepting everything, the device that reads nothing is
within its compute and short of the floor on every coin, above the error `2^-1` (the fiber term is `0` at `g = 2`). -/
theorem RateAuditSketch_nontrivial :
    ∃ (dig : Unit → (Fin 1 → Bool) → Bool) (A : Prover 1 1 0 1 0 0 0 1 Bool Unit Bool),
      ¬ pr (fun x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool =>
          True ∧ A.compute 0 1 x.2.1.1 x.2.1.2 x.1 ≤ 0 ∧
            A.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 1 0 0 0 0 0 1 1)
        ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
          pr (fun σ : Unit => 2 < fib dig σ) + ((0 * 2 : ℕ) : ℝ≥0∞) / (Fintype.card Bool : ℝ≥0∞) := by
  refine ⟨firstSector, blank 1 1 0 1 0 0 0 1 Bool Unit Bool, ?_⟩
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
    ∃ (dig : Unit → (Fin 1 → Bool) → Bool) (A : InteractiveProver 1 1 0 1 0 0 1 1 Bool Unit Bool),
      pr (fun x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool =>
          A.Accepts dig x.2.1.1 x.2.1.2 x.1 x.2.2 ∧ A.compute 1 1 x.2.1.1 x.2.1.2 x.1 ≤ 1 ∧
            ¬ A.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 1 0 0 0 1 1 1 1) = 1 :=
  ⟨firstSector, honestOne, pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ := honestOne_accepts x.2.1.1 x.2.1.2 x.1 x.2.2
    refine ⟨ha, by omega, ?_⟩
    rw [hf]
    unfold Accounting.RateSketch.floor
    omega⟩

/-- **`RateAuditSketchInteractiveRO`'s nontrivial lemma**: with the check accepting everything, the interactive device
that reads nothing is short of the floor on every coin, above the error `2^-1`. -/
theorem RateAuditSketchInteractiveRO_nontrivial :
    ∃ (dig : Unit → (Fin 1 → Bool) → Bool) (A : InteractiveProver 1 1 0 1 0 0 0 1 Bool Unit Bool),
      ¬ pr (fun x : Data 1 1 0 Bool × Coins 1 0 1 1 Bool Unit Bool =>
          True ∧ A.compute 0 1 x.2.1.1 x.2.1.2 x.1 ≤ 0 ∧
            A.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 1 0 0 0 0 0 1 1)
        ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
          pr (fun σ : Unit => 2 < fib dig σ) + ((0 * 2 : ℕ) : ℝ≥0∞) / (Fintype.card Bool : ℝ≥0∞) := by
  refine ⟨firstSector, blankOne, ?_⟩
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
    ∃ A : Prover 1 1 0 1 0 0 1 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)),
      pr (fun x : Data 1 1 0 Sector × Coins 1 0 1 1 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)) =>
          A.Accepts sketch x.2.1.1 x.2.1.2 x.1 x.2.2 ∧
            A.compute 1 1 x.2.1.1 x.2.1.2 x.1 ≤ 1 ∧
            ¬ A.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 256 0 0 0 1 1 1 1) = 1 :=
  ⟨RateSketch.Completeness.honest 1 1 0 1 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)) sketch,
    pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ := RateSketch.Completeness.honest_accepts 256 sketch 1 1 x.2.1.1 x.2.1.2 x.1 x.2.2
    refine ⟨ha, by omega, ?_⟩
    rw [hf]
    unfold Accounting.RateSketch.floor
    omega⟩

/-- **`RateAuditTensorSketchRank`'s nontrivial lemma**: with the check accepting everything, the device that reads
nothing is short of the floor on every coin, above the error `2^-1` at `r = 0` (the rank and guess terms are `0`). -/
theorem RateAuditTensorSketchRank_nontrivial :
    ∃ A : Prover 1 1 0 1 0 0 0 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)),
      ¬ pr (fun x : Data 1 1 0 Sector × Coins 1 0 1 1 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)) =>
          True ∧ A.compute 0 1 x.2.1.1 x.2.1.2 x.1 ≤ 0 ∧
            A.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 256 0 0 0 0 0 1 1)
        ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
          pr (fun σ : SketchSigns 0 1 => ∃ u, ((Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank < 0) +
          ((0 : ℕ) : ℝ≥0∞) / 2 ^ (8 * 0) := by
  refine ⟨blank 1 1 0 1 0 0 0 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)), ?_⟩
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

/-- **`RateAuditTensorSketch`'s honest lemma**: as `RateAuditTensorSketchRank_honest`, under `RateAuditTensorSketch`'s
error. -/
theorem RateAuditTensorSketch_honest :
    ∃ A : Prover 1 1 0 1 0 0 1 256 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)),
      pr (fun x : Data 1 1 0 Sector × Coins 1 0 1 1 Sector (SketchSigns 0 1) (Fin 0 → ZMod (2 ^ 32)) =>
          A.Accepts sketch x.2.1.1 x.2.1.2 x.1 x.2.2 ∧
            A.compute 1 1 x.2.1.1 x.2.1.2 x.1 ≤ 1 ∧
            ¬ A.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 256 0 0 0 1 1 1 1) = 1 :=
  RateAuditTensorSketchRank_honest

/-- An event and its complement have probabilities summing to `1`. -/
theorem pr_add_pr_not_one {α : Type} [Fintype α] [Nonempty α] (P : α → Prop) :
    pr P + pr (fun x => ¬ P x) = 1 := by
  classical
  simp only [Pous.pr_eq_card]
  rw [← ENNReal.add_div, ← Nat.cast_add, Finset.card_filter_add_card_filter_not, Finset.card_univ]
  exact ENNReal.div_self (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- An event that holds somewhere has probability at least one over the size of the type. -/
theorem inv_card_le_pr {α : Type} [Fintype α] {P : α → Prop} (a : α) (ha : P a) :
    (Fintype.card α : ℝ≥0∞)⁻¹ ≤ pr P := by
  classical
  rw [Pous.pr_eq_card, ENNReal.div_eq_inv_mul]
  refine le_mul_of_one_le_right zero_le ?_
  exact Nat.one_le_cast.2 (Finset.card_pos.2 ⟨a, Finset.mem_filter.2 ⟨Finset.mem_univ _, ha⟩⟩)

/-- `RateAuditTensorSketch`'s error at `N = k = s = 1`, `h = 1057`, `q = 0` and `33` sign rows over one sector is below
`1`: `bindSigns` binds, so the signs' term is at most `1 − 2^-1056`. -/
theorem tensor_error_lt :
    (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1057 +
        pr (fun σ : SketchSigns 33 1 => ¬ SectorsBind σ) + ((0 : ℕ) : ℝ≥0∞) / 2 ^ 256 < 1 := by
  have t1 : (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 = 0 := by simp
  have t4 : ((0 : ℕ) : ℝ≥0∞) / 2 ^ 256 = 0 := by simp
  rw [t1, t4, zero_add, add_zero]
  have hsum := pr_add_pr_not_one fun σ : SketchSigns 33 1 => SectorsBind σ
  have hn : Fintype.card (SketchSigns 33 1) < 2 ^ 1057 := by
    rw [Fintype.card_fun, Fintype.card_fun, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Fintype.card_fin,
      Fintype.card_fin, ← pow_mul, ← pow_mul]
    exact Nat.pow_lt_pow_right (by norm_num) (by norm_num)
  have hn' : (Fintype.card (SketchSigns 33 1) : ℝ≥0∞) < 2 ^ 1057 := by
    have := (Nat.cast_lt (α := ℝ≥0∞)).2 hn
    rwa [Nat.cast_pow, Nat.cast_ofNat] at this
  have hbind : ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1057 < pr (fun σ : SketchSigns 33 1 => SectorsBind σ) := by
    rw [Nat.cast_one, one_div]
    exact (ENNReal.inv_lt_inv.2 hn').trans_le (inv_card_le_pr bindSigns bindSigns_binds)
  calc ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1057 + pr (fun σ : SketchSigns 33 1 => ¬ SectorsBind σ)
      < pr (fun σ : SketchSigns 33 1 => SectorsBind σ) + pr (fun σ : SketchSigns 33 1 => ¬ SectorsBind σ) :=
        ENNReal.add_lt_add_right (ne_top_of_le_ne_top ENNReal.one_ne_top (le_add_self.trans hsum.le)) hbind
    _ = 1 := hsum

/-- **`RateAuditTensorSketch`'s nontrivial lemma**: with the check accepting everything, the device that reads nothing
is short of the floor on every coin, above the error at `33` sign rows over one sector, where the signs bind but for a
set of chance below `1 − 2^-1056` (`bindSigns`), and `h = 1057`. -/
theorem RateAuditTensorSketch_nontrivial :
    ∃ A : Prover 1 1 0 1057 0 0 0 256 Sector (SketchSigns 33 1) (Fin 33 → ZMod (2 ^ 32)),
      ¬ pr (fun x : Data 1 1 0 Sector × Coins 1 0 1057 1 Sector (SketchSigns 33 1) (Fin 33 → ZMod (2 ^ 32)) =>
          True ∧ A.compute 0 1 x.2.1.1 x.2.1.2 x.1 ≤ 0 ∧
            A.footprint x.2.1.1 x.2.1.2 x.1 * 1 < RateSketch.floor 1 256 0 0 0 0 0 1 1)
        ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1057 +
          pr (fun σ : SketchSigns 33 1 => ¬ SectorsBind σ) + ((0 : ℕ) : ℝ≥0∞) / 2 ^ 256 := by
  refine ⟨blank 1 1 0 1057 0 0 0 256 Sector (SketchSigns 33 1) (Fin 33 → ZMod (2 ^ 32)), ?_⟩
  have hev : ∀ x : Data 1 1 0 Sector × Coins 1 0 1057 1 Sector (SketchSigns 33 1) (Fin 33 → ZMod (2 ^ 32)),
      True ∧ (blank 1 1 0 1057 0 0 0 256 Sector (SketchSigns 33 1) (Fin 33 → ZMod (2 ^ 32))).compute 0 1 x.2.1.1
          x.2.1.2 x.1 ≤ 0 ∧
        (blank 1 1 0 1057 0 0 0 256 Sector (SketchSigns 33 1) (Fin 33 → ZMod (2 ^ 32))).footprint x.2.1.1 x.2.1.2
          x.1 * 1 < RateSketch.floor 1 256 0 0 0 0 0 1 1 := fun x =>
    ⟨trivial, (blank_compute _ _ _ _ _).le, by
      rw [blank_footprint]
      unfold Accounting.RateSketch.floor
      omega⟩
  rw [pr_one hev, not_le]
  exact tensor_error_lt

end Accounting.SecurityProofs
