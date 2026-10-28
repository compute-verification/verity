import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.ZMod.Defs
import Mathlib.Probability.Distributions.Uniform
import Definitions.Accounting.Tag.Prob

/-!
# The keyed polynomial hash over `GF(2^31 − 1)` (trusted)

The tag of the bandwidth audit's uncertified fast check: `benchmarks/pous/hbm_audit/digest.py`, `gf31-poly/v1`. No
statement about it certifies bandwidth. A device with memory to spare can keep a structure computed from each block
(a Kedlaya–Umans evaluation table) and answer the hash without reading the block's words, so the certified tag is a
random oracle's (`Accounting.Guarantees.Remote.AuditRO`). The statements about this hash are inputs to a possible
version in the joint model, under a capacity bound.

A block is `W` little-endian `uint32` words (`WORDS = 16384` for a 64 KiB block). Its hash at a key point `g` is
Horner's rule over the words' residues mod `p = 2^31 − 1` (`reference_block_hash`), and the hashes of `R` blocks are
folded at a second point `z` (`reference_digest`).

A word enters the field as its residue: `int(w) % MOD` in the reference, one Mersenne fold `(w & p) + (w >> 31)` on the
GPU, which agrees mod `p`. Since `2^32 > 2p`, every word has another word with its residue (`w` and `w ± p`), so the
hash binds residues, not words: two blocks whose words agree mod `p` hash alike at every key.
-/

namespace Accounting.Tag

open scoped ENNReal

/-- The Mersenne prime `2^31 − 1`, the field's modulus (`MOD` in `digest.py`). -/
abbrev P : ℕ := 2147483647

/-- A 32-bit word, as `digest.py` reads a block (little-endian `uint32`). -/
abbrev Word : Type := Fin (2 ^ 32)

/-- A block of `W` words. -/
abbrev Block (W : ℕ) : Type := Fin W → Word

/-- A word's residue mod `P`, the value the hash reads (`int(w) % MOD`). -/
def residue (w : Word) : ZMod P := ((w : ℕ) : ZMod P)

/-- A block's residues, word by word. -/
def residues {W : ℕ} (x : Block W) : Fin W → ZMod P := fun j => residue (x j)

/-- **The block hash** at key `g`: `P_g(x) = Σ_j (x_j mod p) · g^(W − 1 − j)`, the value Horner's rule from the first
word computes (`reference_block_hash`, `hornerHash`). -/
def blockHash {W : ℕ} (g : ZMod P) (x : Block W) : ZMod P :=
  ∑ j : Fin W, residue (x j) * g ^ (W - 1 - (j : ℕ))

/-- `reference_block_hash` as written: `acc ← acc · g + (w mod p)` over the words in order, from `acc = 0`. -/
def hornerHash (g : ZMod P) (ws : List Word) : ZMod P :=
  ws.foldl (fun acc w => acc * g + residue w) 0

/-- **The fold** of a list of block hashes at `z`: `Σ_r h_r · z^r` (`reference_digest`'s `acc += h · zp; zp *= z`). -/
def fold (z : ZMod P) (hs : List (ZMod P)) : ZMod P :=
  ∑ r : Fin hs.length, hs[r] * z ^ (r : ℕ)

/-- **The digest** of `R` blocks, in order, at the key `(g, z)`: the fold at `z` of their hashes at `g`
(`reference_digest` with `lane_points`' two points). -/
def digest {R W : ℕ} (g z : ZMod P) (xs : Fin R → Block W) : ZMod P :=
  fold z (List.ofFn fun r => blockHash g (xs r))

/-- **The best guess of word `j`** of a block drawn from `μ`, by one who knows every other word of the block: the largest
conditional probability `μ(x) / μ(blocks that agree with x off j)` (`0` where that class has probability `0`). Its
`−log₂` is word `j`'s conditional min-entropy given the rest of the block: `2^-32` for a uniform word independent of the
rest, `1` for a word the rest determines. -/
noncomputable def wordGuess {W : ℕ} (μ : PMF (Block W)) (j : Fin W) : ℝ≥0∞ :=
  ⨆ x : Block W, μ x / ∑ a : Word, μ (Function.update x j a)

/-- The blocks that agree with `known` outside `U`: what a view that read every word outside `U` leaves possible. -/
def consistent {W : ℕ} (known : Block W) (U : Finset (Fin W)) : Finset (Block W) :=
  Finset.univ.filter fun x => ∀ i, i ∉ U → x i = known i

/-- **A partially read block**: the words outside `U` are `known`'s, the words in `U` independent and uniform. -/
noncomputable def partialRead {W : ℕ} (known : Block W) (U : Finset (Fin W)) : PMF (Block W) :=
  PMF.uniformOfFinset (consistent known U) (show (consistent known U).Nonempty from ⟨known, by simp [consistent]⟩)

end Accounting.Tag
