module

public import Mathlib
public import Definitions.Poub.Guarantees.RateAuditSketch
public import Proofs.Poub.RateAudit.Sketch

/-! The sketch rate audit with interactive side sources that ignore the oracle, by reading an interactive device as
a `RateSketch.Prover`. -/

@[expose] public section

namespace ResourceAccounting.RateSketch

open Pous (Bits pr)

namespace IChip

variable {N m ρ h : ℕ} {X Dg α : Type}

def toChip (addr : List Bool → Pos N m ρ) : List Bool → IChip N m ρ h X Dg α → Chip N m ρ h X Dg α
  | _, done a => .done a
  | v, tag x k => .tag x (fun y => toChip addr v (k y))
  | v, mem p k => .mem p (fun x => toChip addr v (k x))
  | v, side k => .link (addr v) (fun x => toChip addr v (k (addr v, x)))
  | v, rebuild p k => .rebuild p (fun x => toChip addr v (k x))
  | v, send β p => toChip addr (β :: v) p

theorem toChip_eq (addr : List Bool → Pos N m ρ) (O : Oracle N ρ h X Dg) (D : Data N m ρ X)
    (c : IChip N m ρ h X Dg α) (v : List Bool) :
    (toChip addr v c).run O D = c.run addr O D v ∧ (toChip addr v c).queries O D = c.queries addr O D v ∧
      (toChip addr v c).mems O D = c.mems addr O D v ∧ (toChip addr v c).linkSteps O D = c.linkSteps addr O D v ∧
      (toChip addr v c).rebuilds O D = c.rebuilds addr O D v := by
  induction c generalizing v with
  | done a => exact ⟨rfl, rfl, rfl, rfl, rfl⟩
  | tag x k ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih (O x) v
    simp only [toChip, Chip.run, Chip.queries, Chip.mems, Chip.linkSteps, Chip.rebuilds, run, queries, mems,
      linkSteps, rebuilds]
    exact ⟨h1, by rw [h2], h3, h4, h5⟩
  | mem p k ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih (D (.inl p)) v
    simp only [toChip, Chip.run, Chip.queries, Chip.mems, Chip.linkSteps, Chip.rebuilds, run, queries, mems,
      linkSteps, rebuilds]
    exact ⟨h1, h2, by rw [h3], h4, h5⟩
  | side k ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih (addr v, D (addr v)) v
    simp only [toChip, Chip.run, Chip.queries, Chip.mems, Chip.linkSteps, Chip.rebuilds, run, queries, mems,
      linkSteps, rebuilds]
    exact ⟨h1, h2, h3, by rw [h4], h5⟩
  | rebuild p k ih =>
    obtain ⟨h1, h2, h3, h4, h5⟩ := ih (D (.inl p)) v
    simp only [toChip, Chip.run, Chip.queries, Chip.mems, Chip.linkSteps, Chip.rebuilds, run, queries, mems,
      linkSteps, rebuilds]
    exact ⟨h1, h2, h3, h4, by rw [h5]⟩
  | send β p ih =>
    simp only [toChip, run, queries, mems, linkSteps, rebuilds]
    exact ih (β :: v)

end IChip

namespace InteractiveProver

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

def toProver (A : InteractiveProver N m ρ h S L q w X Sig Dg) : Prover N m ρ h S L q w X Sig Dg where
  held := A.held
  held_le := A.held_le
  chip σ v := IChip.toChip A.addr [] (A.chip σ v)
  bounded σ v O D := by
    obtain ⟨-, h2, -, h4, -⟩ := IChip.toChip_eq A.addr O D (A.chip σ v) []
    rw [h2, h4]
    exact A.bounded σ v O D

theorem toProver_eq (A : InteractiveProver N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg)
    (cTag cEnc : ℕ) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) {s : ℕ} (I : Fin s → Fin N) :
    (A.toProver.Accepts dig σ O D I ↔ A.Accepts dig σ O D I) ∧
      A.toProver.compute cTag cEnc σ O D = A.compute cTag cEnc σ O D ∧
      A.toProver.footprint σ O D = A.footprint σ O D := by
  obtain ⟨h1, h2, h3, -, h5⟩ := IChip.toChip_eq A.addr O D (A.prog σ D) []
  have ht : A.toProver.tags σ O D = A.tags σ O D := h1
  have hq : (A.toProver.prog σ D).queries O D = (A.prog σ D).queries A.addr O D [] := h2
  have hm : (A.toProver.prog σ D).mems O D = (A.prog σ D).mems A.addr O D [] := h3
  have hr : (A.toProver.prog σ D).rebuilds O D = (A.prog σ D).rebuilds A.addr O D [] := h5
  refine ⟨?_, ?_, ?_⟩
  · simp only [Prover.Accepts, Accepts, ht]
  · simp only [Prover.compute, compute, hq, hr]
  · simp only [Prover.footprint, footprint, hm]

end InteractiveProver

end ResourceAccounting.RateSketch

/-- **The sketch rate audit with interactive side sources that ignore the oracle**
(`ResourceAccounting.Guarantees.RateAuditSketchInteractiveRO`), from `RateAuditSketch` on the interactive device as a device. -/
theorem ResourceAccounting.SecurityProofs.RateAuditSketchInteractiveRO :
    ResourceAccounting.Guarantees.RateAuditSketchInteractiveRO := by
  intro N m ρ h S L q w s k F cTag cEnc g X Sig Dg _ _ _ _ _ dig A
  have hev : (fun x : ResourceAccounting.RateSketch.Data N m ρ X × ResourceAccounting.RateSketch.Coins N ρ h s X Sig Dg =>
      A.Accepts dig x.2.1.1 x.2.1.2 x.1 x.2.2 ∧ A.compute cTag cEnc x.2.1.1 x.2.1.2 x.1 ≤ F ∧
        A.footprint x.2.1.1 x.2.1.2 x.1 * cEnc < ResourceAccounting.RateSketch.floor m w ρ S L F cTag cEnc k) =
      fun x => A.toProver.Accepts dig x.2.1.1 x.2.1.2 x.1 x.2.2 ∧ A.toProver.compute cTag cEnc x.2.1.1 x.2.1.2 x.1 ≤ F ∧
        A.toProver.footprint x.2.1.1 x.2.1.2 x.1 * cEnc < ResourceAccounting.RateSketch.floor m w ρ S L F cTag cEnc k := by
    funext x
    obtain ⟨e1, e2, e3⟩ := A.toProver_eq dig cTag cEnc x.2.1.1 x.2.1.2 x.1 x.2.2
    rw [e2, e3, propext e1]
  rw [hev]
  exact ResourceAccounting.SecurityProofs.RateAuditSketch N m ρ h S L q w s k F cTag cEnc g X Sig Dg dig A.toProver
