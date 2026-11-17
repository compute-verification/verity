module

public import Definitions.Pouw.Window

/-! A window's ε from its calls' (M2), with a sample that depends on the registration (C1), stated and not proved; `ServedTie`'s `δ_in` is 1 because no Program reads the `-h3` leaf. -/

@[expose] public section

namespace Pouw.Guarantees.Window

open Pouw.Window
open scoped ENNReal

/-- Luck is not in the event: a deficient unit whose receipt came out right isn't incorrect, and PoUW adds luck apart. -/
def ExecutionPhaseDelta : Prop :=
  ∀ {Reg Tr : Type} {C : ℕ} {n : Fin C → ℕ} (L : Reg → (c : Fin C) → SamplingDesign (Fin (n c))) (Lw : Reg → SamplingDesign (WUnit n))
    (η : ℝ≥0∞) (sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool) (U : UnitStmts Tr n)
    (committed : (R : Reg) → Cont Lw (executionPhaseSession sess) R → Tr) (A : (c : Fin C) → CallAnalysis U c (sess c))
    (κ : Fin C → ℝ≥0∞) (g : (c : Fin C) → (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess c S R) → ℝ≥0∞)
    (εbind : Fin C → (R : Reg) → Cont Lw (executionPhaseSession sess) R → ℝ≥0∞) (𝓑 : Finset (WUnit n) → Prop),
    MissProbWithin Lw L 𝓑 η →
    (∀ c, (A c).KnowledgeSound (κ c) (g c)) →
    (∀ c, BindSoundIn Lw sess committed A c (εbind c)) →
    ∀ σ : Strategy (auditReg Lw (executionPhaseSession sess)),
      prob (fun o => o.1 = true ∧ 𝓑 (U.incorrect (committed (reg σ) (cont σ)))) (auditReg Lw (executionPhaseSession sess)) σ ≤
        (⨆ (B : Finset (WUnit n)) (_ : 𝓑 B), (SamplingDesign.prod (L (reg σ))).missProb B) + η + (⨆ c, κ c) +
          coll Lw sess g (reg σ) (cont σ) + ∑ c, εbind c (reg σ) (cont σ)

def ServiceMissProb : Prop :=
  ∀ {Reg : Type} (N K cap : ℕ), 0 < cap → K ≤ N →
    MissProbWithin (fun _ : Reg => serviceSamplingDesign N K cap) (fun _ σ => sessionSamplingDesign N K cap σ.val) CopyClosed 0

end Pouw.Guarantees.Window
