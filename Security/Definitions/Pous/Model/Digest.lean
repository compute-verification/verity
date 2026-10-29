import Mathlib.Algebra.BigOperators.Fin
import Definitions.Pous.Game.Prob

/-!
# The keyed digest of a region

The fill check of the secure-erasure audit (`Security/Definitions/Pous/Model/SecureErasure.lean`) asks for a
digest of the whole filled region under a fresh key: a keyed two-level polynomial hash over a finite field `F` (the bulk
read's digest, `benchmarks/pous/hbm_audit/digest.py`). A block is `W` words in `F`; the key is two evaluation
points `(g, z)`. `Pous.Guarantees.DigestAU` bounds how often two different regions share a digest.

The digest only: the bulk read's responder and a host-link bandwidth assumption are not modelled here.
-/

namespace Pous.Digest

/-- A block: `W` words, each an element of the field `F`. -/
abbrev Blk (W : ℕ) (F : Type) : Type := Fin W → F

/-- The digest key: the two evaluation points `(g, z)`. -/
abbrev Key (F : Type) : Type := F × F

variable {F : Type} [CommRing F]

/-- A block's hash at `g`: `P_g(x) = Σ_j x_j · g^(W − 1 − j)`, Horner's rule from the first word
(`reference_block_hash` in `benchmarks/pous/hbm_audit/digest.py`). -/
def blockHash {W : ℕ} (g : F) (x : Blk W F) : F :=
  ∑ j : Fin W, x j * g ^ (W - 1 - (j : ℕ))

/-- The digest of `R` blocks, in order, at the key `(g, z)`: `Σ_r P_g(x_r) · z^r` (`reference_digest`). -/
def digest {R W : ℕ} (g z : F) (xs : Fin R → Blk W F) : F :=
  ∑ r : Fin R, blockHash g (xs r) * z ^ (r : ℕ)

end Pous.Digest
