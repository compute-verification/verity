import Definitions.Pouw.PearlC.SaltDead

/-!
# P3 over a `Pipeline` (definitions only)

§5's forming and P3's salt-dead predicate with the noise atom's step a parameter, so that a device whose FP8 step is
another `Pipeline` (sm_120's `mma.sync` m16n8k32, `⟨[32], 26, −133⟩`, `BLACKWELL_SM120_E4M3_M16N8K32`) has its own
chain and predicate beside Π's. At `hopperE4m3K32` both are Π's: `atomMixP hopperE4m3K32 = FormChain.atomMix` and
`saltDeadP hopperE4m3K32 = SaltDead.saltDead` (`Pouw.PearlC.FormingP`, by `rfl`). This module imports no proofs, so the
named protocol may read it.

* **The chain** (`atomMixP p`): one step of `p`, `D = p.step(C, E′, F)`, with accumulator input `C = f32(α·x)`, the
  noise line `E′ = lineCodes β e` and the `F` row's codes; the cast reads `D`.
* **The predicate** (`saltDeadP p alpha beta`): `SaltDead.saltDead` with `atomMixP p` for the atom-mix, on the same
  support (`SaltDead.support`, the flat pairs).
-/

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
