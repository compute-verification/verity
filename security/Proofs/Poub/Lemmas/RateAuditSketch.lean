module

public import Definitions.Poub.Guarantees.RateAuditSketch

/-! The sketch rate audit's honest device and its table outside the verbatim class; neither is a guarantee, and no
record pins them. -/

@[expose] public section

namespace ResourceAccounting.Guarantees

open ResourceAccounting ResourceAccounting.RateSketch

def RateSketchHonest : Prop :=
  ∀ (N m ρ h w : ℕ) (X Sig Dg : Type) [Inhabited X] (dig : Sig → (Fin m → X) → Dg),
    ∃ A : RateSketch.Prover N m ρ h 0 (ρ * w) N w X Sig Dg,
      ∀ (cTag cEnc : ℕ) (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X) (s : ℕ) (I : Fin s → Fin N),
        A.Accepts dig σ O D I ∧ A.footprint σ O D = N * m * w ∧ A.compute cTag cEnc σ O D = N * cTag

/-- Outside the verbatim class the sketch rate audit charges a prover that holds a table of digests only for the
digests it reads. A bound for an arbitrary prover must bound its memory, and near the store's size that is a
static cell-probe lower bound for random `±1` sketches, which this statement neither proves nor refutes. -/
def RateSketchTable : Prop :=
  ∀ (N m ρ h d : ℕ) (X Sig Dg : Type) [Inhabited X] [Fintype Sig] [Fintype Dg] (dig : Sig → (Fin m → X) → Dg),
    Fintype.card Dg ≤ 2 ^ d →
    ∃ A : LayoutDevice N m ρ h X Sig Dg,
      (∀ (E : Fin N × Fin m → X) (a : ℕ), Fintype.card Sig * (N * d) ≤ a → A.mem E a = false) ∧
      ∀ (σ : Sig) (O : Oracle N ρ h X Dg) (D : Data N m ρ X),
        (∀ t, A.tags σ O D t = O (D.stream, t, dig σ (D.tile t))) ∧ (A.queries σ O D).length = N ∧
          A.reads σ O D = N * d ∧ A.linkSteps σ O D = ρ

def RateTensorSketchTable : Prop :=
  ∀ (N m ρ h ks : ℕ), ∃ A : LayoutDevice N m ρ h Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)),
    (∀ (E : Fin N × Fin m → Sector) (a : ℕ), 2 ^ (ks * m * 32) * (N * (32 * ks)) ≤ a → A.mem E a = false) ∧
    ∀ (σ : SketchSigns ks m) (O : Oracle N ρ h Sector (Fin ks → ZMod (2 ^ 32))) (D : Data N m ρ Sector),
      (∀ t, A.tags σ O D t = O (D.stream, t, sketch σ (D.tile t))) ∧ (A.queries σ O D).length = N ∧
        A.reads σ O D = N * (32 * ks) ∧ A.linkSteps σ O D = ρ

end ResourceAccounting.Guarantees
