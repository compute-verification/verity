module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed

/-! The named assumption `FragDraw`, which only the fragment corollary `ttOutRowSeed_skipClass` takes; P2's main line takes no named assumption. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

open Pouw.PearlC Pouw.Fp8Atom

variable {Q R S : Type}

/-- **`FragDraw` per row** at `Λ`, assumed: a row's own sample leaves at most `Λ` fragments except with probability
`2^−128`. Nothing proves it: it is a claim about the noise distribution at the record's pipeline. -/
def FragDraw [Fintype R] [DecidableEq Q] (sem : PearlCSem Q R S) (p : Pipeline) (G : ℕ) (s : S) (U : Workload) (u : ℕ)
    (key : ℕ → (Fin (U.layout.shape u).k → ℕ) → Q) (Λ : ℕ) : Prop :=
  ∀ (H : Q → R) (act : Codes) (i : Fin (U.layout.shape u).m),
    sem.OperandOK 1 (U.layout.shape u).k (rowCodes act i) →
      pr (fun r : R => Λ < fragCount p G
        (sem.unitChain (Function.update H (key i (rowWords (U.layout.shape u).k act i)) r) s U u act).1
        (sem.unitChain H s U u act).2 i) ≤ 1 / 2 ^ 128

end Pouw.PearlC.Assumptions
