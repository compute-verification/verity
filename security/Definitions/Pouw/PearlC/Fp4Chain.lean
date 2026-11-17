module

public import Definitions.Pouw.PearlC.Defs
public import Definitions.Pouw.PearlC.Fp4

/-! Pearl-C4's chain and noise atom on the block-scaled FP4 step; MXFP4's cast of `β` to UE8M0 is not modelled, so `noiseAtom` takes the byte. -/

@[expose] public section

namespace Pouw.PearlC.Fp4

open Pouw.Fp8Atom Pouw.PearlC

/-- `n` firings in ascending K: each takes the next `k` code pairs and `k / block` scale bytes per operand. -/
def Fp4Atom.chainN (p : Fp4Atom) : Nat → Nat → List Nib → List Nib → List (Fin 256) → List (Fin 256) → Option Nat
  | 0, c, _, _, _, _ => some c
  | n + 1, c, a, b, sa, sb => do
    let d ← p.step c (a.take p.k) (b.take p.k) (sa.take p.scales) (sb.take p.scales)
    p.chainN n d (a.drop p.k) (b.drop p.k) (sa.drop p.scales) (sb.drop p.scales)

/-- **The chain** (`chain`): from the word `c` over `K` code pairs, `K` a multiple of `k`, and `K / k · k / block` scale
bytes per operand; `none` if the lengths don't fit or a firing is outside the model. -/
def Fp4Atom.chain (p : Fp4Atom) (c : Nat) (a b : List Nib) (sa sb : List (Fin 256)) : Option Nat :=
  if b.length = a.length ∧ a.length % p.k = 0 ∧ sa.length = a.length / p.k * p.scales ∧
      sb.length = a.length / p.k * p.scales then
    p.chainN (a.length / p.k) c a b sa sb
  else none

/-- **C̃**: the chain from `+0` over the `K / k` atoms, with no promotion. -/
def cTilde (p : Fp4Atom) (a b : List Nib) (sa sb : List (Fin 256)) : Option Nat := p.chain 0 a b sa sb

/-- A noise line in lanes 0–31 and zeros in the atom's other lanes (`E′ ‖ 0³²`, `F ‖ 0³²`). -/
def noiseOperand (p : Fp4Atom) (e : Fin 32 → Nib) : List Nib := List.ofFn e ++ List.replicate (p.k - 32) 0

/-- The operand's scale bytes: one byte for every block. -/
def noiseScales (p : Fp4Atom) (s : Fin 256) : List (Fin 256) := List.replicate p.scales s

/-- `F`'s scale byte, `2^−4`: UE4M3 `0x18` (`8·2^(3 − 10)`), UE8M0 `0x7B` (`2^(123 − 127)`). -/
def fScale : ScaleFmt → Fin 256
  | .ue4m3 => 0x18
  | .ue8m0 => 0x7B

/-- **The noise atom**: `D = step(C, E′ ‖ 0³², F ‖ 0³²)`, with `E′`'s blocks at the byte `sE` and `F`'s at `fScale`. -/
def noiseAtom (p : Fp4Atom) (c : Nat) (e f : Fin 32 → Nib) (sE : Fin 256) : Option Nat :=
  p.step c (noiseOperand p e) (noiseOperand p f) (noiseScales p sE) (noiseScales p (fScale p.fmt))

/-- **NVFP4's noise element**: the noise atom on `sm120Nvf4` from `C = f32(α₀·x)`, with `E′`'s UE4M3 scale `e4m3(β)`. -/
def nvf4NoiseElem (α₀ x β : ℚ) (e f : Fin 32 → Nib) : Option Nat :=
  noiseAtom sm120Nvf4 (f32Word (α₀ * x)) e f (e4m3 β)

structure ChainVec where
  acc : Nat
  n : Nat
  a : Nat
  b : Nat
  sa : Nat
  sb : Nat
  d : Option Nat

def ChainVec.holds (p : Fp4Atom) (v : ChainVec) : Bool :=
  p.chain v.acc (nibs (v.n * p.k) v.a) (nibs (v.n * p.k) v.b) (bytes (v.n * p.scales) v.sa)
    (bytes (v.n * p.scales) v.sb) == v.d

structure NoiseVec where
  α : Nat
  x : Nat
  β : Nat
  e : Nat
  f : Nat
  d : Option Nat

/-- The 32 codes packed little-endian in `A`, as a line. -/
def line (A : Nat) : Fin 32 → Nib := fun i => ⟨(A >>> (4 * i.val)) % 16, Nat.mod_lt _ (by decide)⟩

def NoiseVec.holds (v : NoiseVec) : Bool :=
  nvf4NoiseElem (wordVal v.α) (wordVal v.x) (wordVal v.β) (line v.e) (line v.f) == v.d

end Pouw.PearlC.Fp4
