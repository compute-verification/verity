import Definitions.Pouw.PearlC.Defs

/-!
# Pearl-C's salt-dead elements (definitions only)

P3's per-element predicate for the named protocol's forming debit (`PearlCSem.unitDebit`): element `l` of a committed
activation row `x` is *salt-dead* when the value of its `A′` code is the same at every noise pair of the support, under
§5's atom-mix (`FormChain.atomMix`) at the row's scale `α` and noise scale `β`. That it is `Forming.saltDead` at Π, that
its support has `Forming.FlatPairs`, and that ρ_D = 1/64 bounds it are proved in `Pouw.PearlC.Forming` and
`Pouw.PearlC.AtomErr`; this module imports no proofs, so the named protocol may read it.

* **The support** (`flatLine`, `flatLineF`, `support`): the pairs `(E row, F row)` of flat lines of codes, the `E` row
  flat at Pearl's norm 256 (all 32 entries `±44`) and the `F` row flat at Π's norm 16 (all 32 entries `±11/4`).
  Pearl's line rule gives a flat line whenever the 32 raw magnitudes are equal (checked on #449's `sample_line`: all
  128 flat raw inputs give `±44` at norm 256 and `±2.75` at norm 16), so these pairs lie in the image of Π's lines; an
  element salt-dead over that image is salt-dead here, and the debit can only over-count. They are the pairs
  `FlatPairs` is about and the ones ρ_D is proved on.
* **The predicate** (`saltDead alpha beta`): on the code's value, so a salt that only flips the sign of a zero (`0x00`
  against `0x80`) counts as dead; it changes the code but not the product. `alpha` and `beta` are the row's scales as the
  forming computes them (`PearlCSem.alpha` and `PearlCSem.beta`: α from the model's quantizer, β from `σ = δ·ρ` with ρ
  under `Stat.every8`, and Π's folding `√32/(256·16)` of the two line norms), so the predicate judges the forming that
  ran.
-/

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
