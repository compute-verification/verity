import Definitions.Pouw.PearlC.Fp4Chain
import Proofs.Pouw.PearlC.Fp4

/-! The proof-only part of `Definitions.Pouw.PearlC.Fp4Chain`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC.Fp4

open Pouw.Fp8Atom Pouw.PearlC

/-- A chain vector: the accumulator word, `n` firings' packed codes and scale bytes, and the word `verity.ml.tc`'s
`chain` computes (`none` where it raises). -/
structure ChainVec where
  acc : Nat
  n : Nat
  a : Nat
  b : Nat
  sa : Nat
  sb : Nat
  d : Option Nat

/-- A chain vector holds for the atom `p`. -/
def ChainVec.holds (p : Fp4Atom) (v : ChainVec) : Bool :=
  p.chain v.acc (nibs (v.n * p.k) v.a) (nibs (v.n * p.k) v.b) (bytes (v.n * p.scales) v.sa)
    (bytes (v.n * p.scales) v.sb) == v.d

/-- A noise-atom vector: the words of `α₀`, `x` and `β`, the packed lines `E′` and `F` (32 codes each), and the word the
model writes for `f32(α₀·x)`, `E′ ‖ 0³²` at the byte `e4m3(β)` and `F ‖ 0³²` at `2^−4` (`none` where it raises). -/
structure NoiseVec where
  α : Nat
  x : Nat
  β : Nat
  e : Nat
  f : Nat
  d : Option Nat

/-- The 32 codes packed little-endian in `A`, as a line. -/
def line (A : Nat) : Fin 32 → Nib := fun i => ⟨(A >>> (4 * i.val)) % 16, Nat.mod_lt _ (by decide)⟩

/-- A noise-atom vector holds for `nvf4NoiseElem`. -/
def NoiseVec.holds (v : NoiseVec) : Bool :=
  nvf4NoiseElem (wordVal v.α) (wordVal v.x) (wordVal v.β) (line v.e) (line v.f) == v.d

end Pouw.PearlC.Fp4
