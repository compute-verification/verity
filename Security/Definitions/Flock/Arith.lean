import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Ring.Hom.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Nat.Bitwise
import Mathlib.LinearAlgebra.Matrix.Defs

/-!
# The arithmetic the verifier uses

The model of the Flock verifier (PROTOCOL.md §9–§13) is written over two fields `F`
(GF(2^128), GHASH form, §2.1) and `K` (GF(2^256), §2.4), and uses a handful of fixed
arithmetic functions: the Lagrange weights of the univariate-skip nodes, the pinned zerocheck
coordinates, the packing of 128 bits into one `F` element, the embedding `F ⊂ K`, and the
Reed–Solomon domain and novel basis. `Arith` collects them as data, so that the executable
verifier (lane `flock-verifier`, `Flock.Field.*`, §20) instantiates the same functions this
model uses. `Arith.Correct` lists the mathematical facts the soundness proof needs about them;
they are facts about concrete functions, to be proved for the GHASH instance, not assumptions.
-/

namespace FlockSoundness.Model

/-- The equality tensor at index `b` (bits least-significant first):
`eq(r, b) = Π_j (r_j if bit_j(b) = 1 else 1 + r_j)` (§1; `1 + r = 1 − r` in characteristic 2). -/
def eqAt {R : Type} [CommRing R] {n : ℕ} (r : Fin n → R) (b : ℕ) : R :=
  ∏ j : Fin n, if b.testBit j then r j else 1 + r j

/-- The arithmetic of §2, §9, §12 and §13 that the verifier's equations use. -/
structure Arith (F K : Type) [Field F] [Field K] where
  /-- `L_s(ζ)`: the Lagrange weight of skip node `φ8(s)`, `s < 64`, at `ζ` (§9). -/
  lagS : F → Fin 64 → F
  /-- `L^Λ_{φ8(64+i)}(ζ)`: the Lagrange weight of output-coset node `φ8(64+i)` at `ζ` (§9.2). -/
  lagΛ : F → Fin 64 → F
  /-- The weights of `Pcomb(ζ)`: `Pcomb(ζ) = Σ_i combW ζ i · v_i` is the polynomial of degree
  `< 128` that is `0` on `S` and `v_i` at `φ8(64+i)` (§9.2). -/
  combW : F → Fin 64 → F
  /-- The pinned zerocheck coordinates at address bits 6–12 (§9.2 table). -/
  pinned : Fin 7 → F
  /-- The packing `f(y) = Σ_v z[128y+v]·x^v` of 128 bits into one `F` element (§12.1). -/
  pack : (Fin 128 → ZMod 2) → F
  /-- The bits of an `F` element in the packing basis. -/
  unpack : F → Fin 128 → ZMod 2
  /-- The embedding `F ⊂ K`. -/
  embed : F →+* K
  /-- `u` with `K = F[u] / (u² + u + x⁻¹)`: a `K` element `(c₀, c₁)` is `c₀ + c₁·u` (§2.4). -/
  u : K
  /-- The Reed–Solomon domain point `ω_q` (§13.1). -/
  omega : ℕ → F
  /-- The novel-basis value `Ŵ_i(ω)` (§13.1, §13.4). -/
  What : ℕ → F → F
  /-- The low 64-bit word `lo` of an `F` element (§2.1); query positions read its low bits
  (§13.5). -/
  lo : F → ℕ

namespace Arith

variable {F K : Type} [Field F] [Field K] (A : Arith F K)

/-- A `K` element from its two `F` limbs. -/
def ofLimbs (c₀ c₁ : F) : K := A.embed c₀ + A.embed c₁ * A.u

/-- The novel-basis monomial `X̂_c(ω) = Π_{i ∈ bits(c)} Ŵ_i(ω)` (§13.1). -/
def Xhat (c : ℕ) (L : ℕ) (ω : F) : F :=
  ∏ i : Fin L, if c.testBit i then A.What i ω else 1

/-- The Reed–Solomon encoding of one lane (row) of a level with `2^L` message columns, at
codeword position `q`: `C[q] = Σ_c M[c]·X̂_c(ω_q)` (§13.1). -/
def encodeAt (L : ℕ) (M : Fin (2 ^ L) → F) (q : ℕ) : F :=
  ∑ c : Fin (2 ^ L), M c * A.Xhat c L (A.omega q)

end Arith

end FlockSoundness.Model
