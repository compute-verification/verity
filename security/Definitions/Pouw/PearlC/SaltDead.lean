module

public import Definitions.Pouw.PearlC.Defs

/-! P3's salt-dead predicate over a flat-line support that lies in the image of Π's lines, so the debit can only over-count; it imports no proofs, so the named protocol may read it. -/

@[expose] public section

namespace Pouw.PearlC.SaltDead

/-- A flat `E` line (norm 256): all 32 entries the codes of `±44`, negative where `s` is set. -/
def flatLine (s : Fin 32 → Bool) : Fin 32 → Fp8Atom.Code := fun t => e4m3 (if s t then -44 else 44)

/-- A flat `F` line (Π's norm 16): all 32 entries the codes of `±11/4`, negative where `s` is set. -/
def flatLineF (s : Fin 32 → Bool) : Fin 32 → Fp8Atom.Code := fun t => e4m3 (if s t then -(11 / 4) else 11 / 4)

/-- The noise support of one element: the pairs `(E row, F row)` of flat lines. -/
def support : Set ((Fin 32 → Fp8Atom.Code) × (Fin 32 → Fp8Atom.Code)) :=
  {p | ∃ s s' : Fin 32 → Bool, p = (flatLine s, flatLineF s')}

/-- **P3's salt-dead predicate**, of `PearlCSem.saltDead`'s type: element `l` of row `x` (of length `k`) is salt-dead
when its code under the atom-mix, `FormChain.atomMix.elem (alpha k x) (x l) (beta k x) e f`, has the same value at every
pair `(e, f)` of `support`. -/
def saltDead (alpha beta : (k : ℕ) → (Fin k → ℚ) → ℚ) : (k : ℕ) → (Fin k → ℚ) → Fin k → Prop :=
  fun k x l => ∀ p ∈ support, ∀ q ∈ support,
    codeVal (FormChain.atomMix.elem (alpha k x) (x l) (beta k x) p.1 p.2) =
      codeVal (FormChain.atomMix.elem (alpha k x) (x l) (beta k x) q.1 q.2)

end Pouw.PearlC.SaltDead
