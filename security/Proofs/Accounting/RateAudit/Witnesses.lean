module

public import Proofs.Accounting.RateAudit.Honest
public import Proofs.Accounting.RateAudit.Interactive
public import Proofs.Accounting.RateAudit.Partial
public import Proofs.Pous.Prob

/-!
# The rate audit's guarantees: each one's honest and nontrivial lemmas

For `RateAudit`, `RateAuditWindows`, `RateAuditInteractiveRO`, `RateAuditInteractive` and `RateAuditPartial`, each at
a store of one word of one bit (`N = b = 1`, two at `RateAuditInteractive`'s), a one-bit tag, no certified stream and
one sampled word:

* **The honest lemma** (`<guarantee>_honest`): the honest device (`Rate.Completeness.honest`; as an interactive device,
  `Prover.toInteractive`; as a partial one, `Prover.toPartial`) is accepted by the audit's own check, within the compute
  budget, with its footprint at the floor, with probability 1: the guarantee's event is not one no device passes.
* **The nontrivial lemma** (`<guarantee>_nontrivial`): with the check replaced by `True`, a device that reads nothing
  (`Rate.blank`) has its footprint short of the floor with probability 1, above the guarantee's error.
-/

@[expose] public section

namespace Accounting.Rate

open Pous (Bits pr)
open scoped ENNReal

variable {N b h ρ S L q : ℕ}

/-- The device that reads nothing: no state, nothing in device memory or the side sources, and a chip that commits to
all-`false` tags at once. -/
def blank (N b h ρ S L q : ℕ) : Prover N b h ρ S L q where
  state _ := fun _ => false
  mem _ _ := false
  side _ _ := false
  chip _ := .done fun _ _ => false
  bounded _ _ := ⟨Nat.zero_le _, Nat.zero_le _⟩

theorem blank_compute (cTag cEnc : ℕ) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    (blank N b h ρ S L q).compute cTag cEnc E R O = 0 := by
  simp [Prover.compute, blank, Chip.queries, Chip.rebuilds]

theorem blank_footprint (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    (blank N b h ρ S L q).footprint E R O = 0 := by
  simp [Prover.footprint, blank, Chip.addrs]

theorem Prover.toInteractive_footprint (A : Prover N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.toInteractive.footprint E R O = A.footprint E R O :=
  congrArg Finset.card (Chip.toIChip_addrs _ _ _)

/-- The honest device at one word of one bit, a one-bit tag and no stream: accepted, the store as its footprint, one
tag as its compute. -/
theorem honest_one (E : Store 1 1) (R : Bits 0) (O : Oracle 1 1 1 0) (I : Fin 1 → Fin 1) :
    (Completeness.honest 1 1 1 0).Accepts E R O I ∧ (Completeness.honest 1 1 1 0).footprint E R O = 1 * 1 ∧
      (Completeness.honest 1 1 1 0).compute 1 1 E R O = 1 * 1 := by
  obtain ⟨a, f, -, c⟩ := Completeness.honest_accepts (h := 1) 1 1 E R O I
  exact ⟨a, f, c⟩

/-- `honest_one` as a partial device, at every price of a completion. -/
theorem honest_one_partial (cost : ℕ → ℕ) (E : Store 1 1) (R : Bits 0) (O : Oracle 1 1 1 0) (I : Fin 1 → Fin 1) :
    (Completeness.honest 1 1 1 0).toPartial.Accepts E R O I ∧
      (Completeness.honest 1 1 1 0).toPartial.footprint E R O = 1 * 1 ∧
      (Completeness.honest 1 1 1 0).toPartial.compute 1 cost E R O = 1 * 1 := by
  obtain ⟨hacc, hfp, -, -⟩ := Completeness.honest_accepts (h := 1) 1 0 E R O I
  obtain ⟨-, -, -, -, h5, h6⟩ := Completeness.chip_eq (h := 1) ((Completeness.honest 1 1 1 0).world E R O)
  obtain ⟨t1, t2, -, -, t5⟩ := (Completeness.honest 1 1 1 0).toPartial_eq E R O
  have hc : (Completeness.honest 1 1 1 0).chip ((Completeness.honest 1 1 1 0).state E) =
      Completeness.chip 1 1 1 0 := rfl
  refine ⟨fun i => ?_, t2.trans hfp, ?_⟩
  · rw [t1]
    exact hacc i
  · rw [t5 1 cost, hc, h5, h6, zero_mul, add_zero]

end Accounting.Rate

namespace Accounting.SecurityProofs

open Accounting.Rate
open Pous (Bits pr)
open scoped ENNReal

theorem pr_one {α : Type} [Fintype α] [Nonempty α] {P : α → Prop} (h : ∀ x, P x) : pr P = 1 := by
  classical
  simp only [Pous.pr_eq_card]
  rw [Finset.filter_true_of_mem fun x _ => h x, Finset.card_univ]
  exact ENNReal.div_self (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

/-- The error of `RateAudit` (and of `RateAuditPartial`) at `N = 1`, `k = 1`, `s = 1`, `h = 1`, `q = 0`, `λ = 0` is
`2^-1`. -/
theorem error_one_lt :
    (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
      ((Nat.choose 0 1 : ℕ) : ℝ≥0∞) / 2 ^ 0 < 1 := by
  have t1 : (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 = 0 := by simp
  have t2 : ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 = 2⁻¹ := by simp
  have t3 : ((Nat.choose 0 1 : ℕ) : ℝ≥0∞) / 2 ^ 0 = 0 := by
    rw [Nat.choose_eq_zero_of_lt (by norm_num : 0 < 1)]
    simp
  rw [t1, t2, t3, zero_add, add_zero]
  exact ENNReal.inv_lt_one.2 ENNReal.one_lt_two

theorem one_mul_cast (a : ℝ≥0∞) : ((1 : ℕ) : ℝ≥0∞) * a = a := by
  rw [Nat.cast_one, one_mul]

theorem natCast_div_pow_succ_lt (a n : ℕ) (h : a < 2 ^ n) : (a : ℝ≥0∞) / 2 ^ (n + 1) < 2⁻¹ := by
  rw [ENNReal.div_lt_iff (Or.inl (pow_ne_zero _ two_ne_zero)) (Or.inl (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)),
    pow_succ', ← mul_assoc, ENNReal.inv_mul_cancel two_ne_zero ENNReal.ofNat_ne_top, one_mul]
  exact_mod_cast h

/-- **`RateAudit`'s honest lemma**: the honest device, at one word of one bit, is accepted within one tag's compute
with its footprint at the floor, with probability 1. -/
theorem RateAudit_honest :
    pr (fun x : Store 1 1 × Coins 1 1 1 0 1 =>
        (Completeness.honest 1 1 1 0).Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧
          (Completeness.honest 1 1 1 0).compute 1 1 x.1 x.2.1.1 x.2.1.2 ≤ 1 ∧
          ¬ (Completeness.honest 1 1 1 0).footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 1 1 1 1 0) = 1 :=
  pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ := honest_one x.1 x.2.1.1 x.2.1.2 x.2.2
    refine ⟨ha, by omega, ?_⟩
    rw [hf]
    unfold floor
    omega

/-- **`RateAudit`'s nontrivial lemma**: with the check accepting everything, the device that reads nothing is within
its compute and short of the floor on every coin, above the error `2^-1`. -/
theorem RateAudit_nontrivial :
    ¬ pr (fun x : Store 1 1 × Coins 1 1 1 0 1 =>
        True ∧ (blank 1 1 1 0 0 0 0).compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
          (blank 1 1 1 0 0 0 0).footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 0 0 1 1 0)
      ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
        ((Nat.choose 0 1 : ℕ) : ℝ≥0∞) / 2 ^ 0 := by
  have hev : ∀ x : Store 1 1 × Coins 1 1 1 0 1,
      True ∧ (blank 1 1 1 0 0 0 0).compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
        (blank 1 1 1 0 0 0 0).footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 0 0 1 1 0 := fun x =>
    ⟨trivial, (blank_compute _ _ _ _ _).le, by rw [blank_footprint]; unfold floor; omega⟩
  rw [pr_one hev, not_le]
  exact error_one_lt

/-- `RateAudit_honest`'s device as the device of one window. -/
def honestDevices : Devices 1 1 1 1 0 0 0 1 1 := fun _ _ => Completeness.honest 1 1 1 0

/-- `RateAudit_nontrivial`'s device as the device of one window. -/
def blankDevices : Devices 1 1 1 1 0 0 0 0 1 := fun _ _ => blank 1 1 1 0 0 0 0

/-- **`RateAuditWindows`'s honest lemma**: one window of `RateAudit_honest`'s device, at `m = 1`. -/
theorem RateAuditWindows_honest :
    pr (fun x : Store 1 1 × (Fin 1 → Coins 1 1 1 0 1) =>
        (∀ i, (honestDevices.dev i x.2).Accepts x.1
            (x.2 i).1.1 (x.2 i).1.2 (x.2 i).2 ∧
          (honestDevices.dev i x.2).compute 1 1 x.1
            (x.2 i).1.1 (x.2 i).1.2 ≤ 1) ∧
        ¬ ∑ i, (honestDevices.dev i x.2).footprint x.1
            (x.2 i).1.1 (x.2 i).1.2 * 1 < (1 + 1 - 1) * floor 1 0 0 0 1 1 1 1 0) = 1 :=
  pr_one fun x => by
    refine ⟨fun i => ?_, ?_⟩
    · obtain ⟨ha, -, hc⟩ := honest_one x.1 (x.2 i).1.1 (x.2 i).1.2 (x.2 i).2
      exact ⟨ha, (show (Completeness.honest 1 1 1 0).compute 1 1 x.1 (x.2 i).1.1 (x.2 i).1.2 ≤ 1 by omega)⟩
    · obtain ⟨-, hf, -⟩ := honest_one x.1 (x.2 0).1.1 (x.2 0).1.2 (x.2 0).2
      rw [Fin.sum_univ_one]
      change ¬ (Completeness.honest 1 1 1 0).footprint x.1 (x.2 0).1.1 (x.2 0).1.2 * 1 < _
      rw [hf]
      unfold floor
      omega

/-- **`RateAuditWindows`'s nontrivial lemma**: one window of `RateAudit_nontrivial`'s device, at `m = 1`, with the
check accepting everything. -/
theorem RateAuditWindows_nontrivial :
    ¬ ((1 : ℕ) : ℝ≥0∞) * pr (fun x : Store 1 1 × (Fin 1 → Coins 1 1 1 0 1) =>
        (∀ i, True ∧ (blankDevices.dev i x.2).compute 0 1 x.1
            (x.2 i).1.1 (x.2 i).1.2 ≤ 0) ∧
        ∑ i, (blankDevices.dev i x.2).footprint x.1
            (x.2 i).1.1 (x.2 i).1.2 * 1 < (1 + 1 - 1) * floor 1 0 0 0 0 0 1 1 0)
      ≤ ((1 : ℕ) : ℝ≥0∞) * ((((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
        ((Nat.choose 0 1 : ℕ) : ℝ≥0∞) / 2 ^ 0) := by
  have hev : ∀ x : Store 1 1 × (Fin 1 → Coins 1 1 1 0 1),
      (∀ i, True ∧ (blankDevices.dev i x.2).compute 0 1 x.1
          (x.2 i).1.1 (x.2 i).1.2 ≤ 0) ∧
      ∑ i, (blankDevices.dev i x.2).footprint x.1
          (x.2 i).1.1 (x.2 i).1.2 * 1 < (1 + 1 - 1) * floor 1 0 0 0 0 0 1 1 0 := fun x => by
    refine ⟨fun i => ⟨trivial, ?_⟩, ?_⟩
    · change (blank 1 1 1 0 0 0 0).compute 0 1 x.1 (x.2 i).1.1 (x.2 i).1.2 ≤ 0
      exact (blank_compute _ _ _ _ _).le
    · rw [Fin.sum_univ_one]
      change (blank 1 1 1 0 0 0 0).footprint x.1 (x.2 0).1.1 (x.2 0).1.2 * 1 < _
      rw [blank_footprint]
      unfold floor
      omega
  rw [pr_one hev, one_mul_cast, one_mul_cast, not_le]
  exact error_one_lt

/-- **`RateAuditInteractiveRO`'s honest lemma**: the honest device as an interactive device, whose side sources ignore
the oracle, is accepted within one tag's compute with its footprint at the floor, with probability 1. -/
theorem RateAuditInteractiveRO_honest :
    (∀ E R O O', (Completeness.honest 1 1 1 0).toInteractive.side E R O =
      (Completeness.honest 1 1 1 0).toInteractive.side E R O') ∧
    pr (fun x : Store 1 1 × Coins 1 1 1 0 1 =>
        (Completeness.honest 1 1 1 0).toInteractive.Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧
          (Completeness.honest 1 1 1 0).toInteractive.compute 1 1 x.1 x.2.1.1 x.2.1.2 ≤ 1 ∧
          ¬ (Completeness.honest 1 1 1 0).toInteractive.footprint x.1 x.2.1.1 x.2.1.2 * 1 <
            floor 1 0 0 0 1 1 1 1 0) = 1 :=
  ⟨fun _ _ _ _ => rfl, pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ := honest_one x.1 x.2.1.1 x.2.1.2 x.2.2
    refine ⟨(Prover.toInteractive_accepts _ _ _ _ _).2 ha, ?_, ?_⟩
    · rw [Prover.toInteractive_compute]
      omega
    · rw [Prover.toInteractive_footprint, hf]
      unfold floor
      omega⟩

/-- **`RateAuditInteractiveRO`'s nontrivial lemma**: with the check accepting everything, the device that reads
nothing, as an interactive device (whose side sources ignore the oracle), is short of the floor on every coin. -/
theorem RateAuditInteractiveRO_nontrivial :
    ¬ pr (fun x : Store 1 1 × Coins 1 1 1 0 1 =>
        True ∧ (blank 1 1 1 0 0 0 0).toInteractive.compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
          (blank 1 1 1 0 0 0 0).toInteractive.footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 0 0 1 1 0)
      ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
        ((Nat.choose 0 1 : ℕ) : ℝ≥0∞) / 2 ^ 0 := by
  have hev : ∀ x : Store 1 1 × Coins 1 1 1 0 1,
      True ∧ (blank 1 1 1 0 0 0 0).toInteractive.compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
        (blank 1 1 1 0 0 0 0).toInteractive.footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 0 0 1 1 0 := fun x =>
    ⟨trivial, ((Prover.toInteractive_compute _ _ _ _ _ _).trans (blank_compute _ _ _ _ _)).le,
      by rw [Prover.toInteractive_footprint, blank_footprint]; unfold floor; omega⟩
  rw [pr_one hev, not_le]
  exact error_one_lt

/-- **`RateAuditInteractive`'s honest lemma**: `RateAuditInteractiveRO_honest`'s device at `m = 0`. -/
theorem RateAuditInteractive_honest :
    pr (fun x : Store 1 1 × Coins 1 1 1 0 1 =>
        (Completeness.honest 1 1 1 0).toInteractive.Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧
          (Completeness.honest 1 1 1 0).toInteractive.compute 1 1 x.1 x.2.1.1 x.2.1.2 ≤ 1 ∧
          ¬ (Completeness.honest 1 1 1 0).toInteractive.footprint x.1 x.2.1.1 x.2.1.2 * 1 <
            floor 1 0 0 0 1 1 1 (1 - 0) 0) = 1 :=
  pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ := honest_one x.1 x.2.1.1 x.2.1.2 x.2.2
    refine ⟨(Prover.toInteractive_accepts _ _ _ _ _).2 ha, ?_, ?_⟩
    · rw [Prover.toInteractive_compute]
      omega
    · rw [Prover.toInteractive_footprint, hf]
      unfold floor
      omega

/-- **`RateAuditInteractive`'s nontrivial lemma**: with the check accepting everything, the device that reads nothing,
as an interactive device, is short of the floor at `k − m = 1` of `k = 2` words on every coin, above the error
`(1/2)^1 + C(2, 1)·2^0/2^3 + 0 = 3/4` at `N = 2`, `h = 3`, `m = 1`. -/
theorem RateAuditInteractive_nontrivial :
    ¬ pr (fun x : Store 2 1 × Coins 2 1 3 0 1 =>
        True ∧ (blank 2 1 3 0 0 0 0).toInteractive.compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
          (blank 2 1 3 0 0 0 0).toInteractive.footprint x.1 x.2.1.1 x.2.1.2 * 1 < floor 1 0 0 0 0 0 1 (2 - 1) 0)
      ≤ (((2 - 1 : ℕ) : ℝ≥0∞) / ((2 : ℕ) : ℝ≥0∞)) ^ 1 + ((Nat.choose 2 1 : ℕ) : ℝ≥0∞) * 2 ^ 0 / 2 ^ (3 * 1) +
        ((Nat.choose 0 (2 - 1) : ℕ) : ℝ≥0∞) / 2 ^ 0 := by
  have hev : ∀ x : Store 2 1 × Coins 2 1 3 0 1,
      True ∧ (blank 2 1 3 0 0 0 0).toInteractive.compute 0 1 x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
        (blank 2 1 3 0 0 0 0).toInteractive.footprint x.1 x.2.1.1 x.2.1.2 * 1 <
          floor 1 0 0 0 0 0 1 (2 - 1) 0 := fun x =>
    ⟨trivial, ((Prover.toInteractive_compute _ _ _ _ _ _).trans (blank_compute _ _ _ _ _)).le,
      by rw [Prover.toInteractive_footprint, blank_footprint]; unfold floor; omega⟩
  rw [pr_one hev, not_le]
  have t1 : (((2 - 1 : ℕ) : ℝ≥0∞) / ((2 : ℕ) : ℝ≥0∞)) ^ 1 = 2⁻¹ := by simp
  have t2 : ((Nat.choose 2 1 : ℕ) : ℝ≥0∞) * 2 ^ 0 / 2 ^ (3 * 1) < 2⁻¹ := by
    rw [show ((Nat.choose 2 1 : ℕ) : ℝ≥0∞) * 2 ^ 0 = ((2 : ℕ) : ℝ≥0∞) by simp,
      show (3 * 1 : ℕ) = 2 + 1 from rfl]
    exact natCast_div_pow_succ_lt 2 2 (by norm_num)
  have t3 : ((Nat.choose 0 (2 - 1) : ℕ) : ℝ≥0∞) / 2 ^ 0 = 0 := by
    rw [Nat.choose_eq_zero_of_lt (by norm_num : 0 < 2 - 1)]
    simp
  rw [t1, t3, add_zero]
  calc 2⁻¹ + ((Nat.choose 2 1 : ℕ) : ℝ≥0∞) * 2 ^ 0 / 2 ^ (3 * 1) < 2⁻¹ + 2⁻¹ :=
        ENNReal.add_lt_add_left (ENNReal.inv_ne_top.2 two_ne_zero) t2
    _ = 1 := ENNReal.inv_two_add_inv_two

/-- **`RateAuditPartial`'s honest lemma**: the honest device as a partial device, completing nothing, is accepted
within one tag's compute with its footprint at the floor, with probability 1, at `rounds = c_root = 1` and every price
of a completion. -/
theorem RateAuditPartial_honest :
    pr (fun x : Store 1 1 × Coins 1 1 1 0 1 =>
        (Completeness.honest 1 1 1 0).toPartial.Accepts x.1 x.2.1.1 x.2.1.2 x.2.2 ∧
          (Completeness.honest 1 1 1 0).toPartial.compute 1 (fun _ => 0) x.1 x.2.1.1 x.2.1.2 ≤ 1 ∧
          ¬ (Completeness.honest 1 1 1 0).toPartial.footprint x.1 x.2.1.1 x.2.1.2 * (1 * 1) <
            floor 1 0 0 0 1 1 (1 * 1) 1 0) = 1 :=
  pr_one fun x => by
    obtain ⟨ha, hf, hc⟩ := honest_one_partial (fun _ => 0) x.1 x.2.1.1 x.2.1.2 x.2.2
    refine ⟨ha, by omega, ?_⟩
    rw [hf]
    unfold floor
    omega

/-- **`RateAuditPartial`'s nontrivial lemma**: with the check accepting everything, the device that reads nothing, as a
partial device, is short of the floor on every coin. -/
theorem RateAuditPartial_nontrivial :
    ¬ pr (fun x : Store 1 1 × Coins 1 1 1 0 1 =>
        True ∧ (blank 1 1 1 0 0 0 0).toPartial.compute 0 (fun _ => 0) x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
          (blank 1 1 1 0 0 0 0).toPartial.footprint x.1 x.2.1.1 x.2.1.2 * (1 * 1) <
            floor 1 0 0 0 0 0 (1 * 1) 1 0)
      ≤ (((1 - 1 : ℕ) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ^ 1 + ((1 : ℕ) : ℝ≥0∞) / 2 ^ 1 +
        ((Nat.choose 0 1 : ℕ) : ℝ≥0∞) / 2 ^ 0 := by
  have hev : ∀ x : Store 1 1 × Coins 1 1 1 0 1,
      True ∧ (blank 1 1 1 0 0 0 0).toPartial.compute 0 (fun _ => 0) x.1 x.2.1.1 x.2.1.2 ≤ 0 ∧
        (blank 1 1 1 0 0 0 0).toPartial.footprint x.1 x.2.1.1 x.2.1.2 * (1 * 1) <
          floor 1 0 0 0 0 0 (1 * 1) 1 0 := fun x => by
    obtain ⟨-, t2, -, -, t5⟩ := (blank 1 1 1 0 0 0 0).toPartial_eq x.1 x.2.1.1 x.2.1.2
    refine ⟨trivial, ?_, ?_⟩
    · rw [t5 0 (fun _ => 0)]
      simp [blank, Chip.queries, Chip.rebuilds]
    · rw [t2, blank_footprint]
      unfold floor
      omega
  rw [pr_one hev, not_le]
  exact error_one_lt

end Accounting.SecurityProofs
