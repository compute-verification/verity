module

public import Definitions.Pouw.PearlC.SaltDead

/-! §5's forming and P3's salt-dead predicate over a `Pipeline`; it imports no proofs, so the named protocol may read it. -/

@[expose] public section

namespace Pouw.PearlC.FormingP

open Pouw.Fp8Atom

/-- **§5's atom-mix on the step `p`**: `D = p.step(f32(α·x), E′, F)` with `E′ = lineCodes β e` and `F`'s codes, read by
the cast. -/
def atomMixP (p : Pipeline) : FormChain 32 :=
  ⟨fun α x β e f => wordVal (p.step (f32Word (α * x)) (List.ofFn (lineCodes β e)) (List.ofFn f))⟩

/-- **P3's salt-dead predicate on the step `p`**, of `PearlCSem.saltDead`'s type: element `l` of row `x` is salt-dead
when its code under `atomMixP p` has the same value at every pair of `SaltDead.support`. -/
def saltDeadP (p : Pipeline) (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) : (k : ℕ) → (Fin k → ℚ) → Fin k → Prop :=
  fun k x l => ∀ q ∈ SaltDead.support, ∀ q' ∈ SaltDead.support,
    codeVal ((atomMixP p).elem (alpha k x) (x l) (beta k x) q.1 q.2) =
      codeVal ((atomMixP p).elem (alpha k x) (x l) (beta k x) q'.1 q'.2)

end Pouw.PearlC.FormingP
