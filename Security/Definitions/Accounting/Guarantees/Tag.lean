import Mathlib.Probability.Distributions.Uniform
import Definitions.Accounting.Tag.PolyHash

/-!
# The block tags' statements (trusted)

The tag of the bandwidth audit's uncertified fast check (`gf31-poly/v1`): the keyed polynomial hash over
`GF(2^31 − 1)` of `benchmarks/pous/hbm_audit/digest.py` (`Security/Definitions/Accounting/Tag/PolyHash.lean`). None of
these statements certifies bandwidth: the bandwidth audit's guarantee rests on a random oracle's tag
(`Accounting.Guarantees.Remote.AuditRO`). They are inputs to a possible version in the joint model, under a capacity
bound.

* `PolyNonCollision`, `PolyNonCollisionKeyed`: two blocks fixed before the key, with different residues, hash alike with
  probability at most `(W − 1)/p` over a uniform key (at most `W − 1` times the largest point mass of any key
  distribution).
* `PolyUnpredictable`: unpredictability under partial reads. Whoever knows every word of a block but one, and the key,
  guesses the hash with probability at most `3` times its best guess of that word (`wordGuess`): `3 · 2^-32 ≈ 1.5/p`
  when the word is uniform. The `3` is the reduction: three 32-bit words share each of the residues `0` and `1`.
* `PartialReadGuess`: for a block whose unread words are uniform and independent of what was read (`partialRead`), the
  best guess of an unread word is `2^-32`, so `PolyUnpredictable` bounds the guess of the hash by `3 · 2^-32`, for a
  guesser whose view is words of the block.
* `FoldNonCollision`: changing the list of block hashes (dropping, altering, reordering or inserting one) changes the
  fold except with probability at most `(n − 1)/p` over a uniform `z`, `n` the longer list's length.
* `DigestNonCollision`: the two together, over a uniform `(g, z)`.
* `HornerIsBlockHash`: `reference_block_hash`'s Horner loop computes `blockHash`.

Each takes no hypothesis: a condition on its inputs (the blocks differ, the lists differ) is part of the event it
bounds. The proofs are in `Security/Proofs/Accounting/Tag/`.
-/

namespace Accounting.Guarantees.Tag

open Accounting.Tag
open scoped ENNReal

/-- **Non-collision for blocks fixed before the key.** Two blocks of `W` words whose residues mod `p = 2^31 − 1` differ
hash alike at a uniform key `g ∈ GF(p)` with probability at most `(W − 1)/p`: their difference is a nonzero polynomial
in `g` of degree at most `W − 1`. Blocks whose words agree mod `p` (a word `w` and `w ± p`) hash alike at every key, so
the event asks that the residues differ. -/
def PolyNonCollision : Prop :=
  ∀ (W : ℕ) (x y : Block W),
    prob (PMF.uniformOfFintype (ZMod P)) (fun g => residues x ≠ residues y ∧ blockHash g x = blockHash g y) ≤
      ((W - 1 : ℕ) : ℝ≥0∞) / P

/-- **Non-collision at any key distribution.** As `PolyNonCollision`, with the key drawn from any distribution `ν` on
`GF(p)`: the probability is at most `W − 1` times `ν`'s largest point mass. `lane_points` in `digest.py` draws
`g ∈ [2, p − 1]` from a uniform 31-bit value, three points twice as likely as the rest, so its largest point mass is
`2^-30`. -/
def PolyNonCollisionKeyed : Prop :=
  ∀ (W : ℕ) (ν : PMF (ZMod P)) (x y : Block W),
    prob ν (fun g => residues x ≠ residues y ∧ blockHash g x = blockHash g y) ≤ ((W - 1 : ℕ) : ℝ≥0∞) * ⨆ g, ν g

/-- **Unpredictability under partial reads.** Draw a block from any distribution `μ` (the stored block, conditioned on
everything its guesser has seen) and fix any word `j` (one it did not read). For every nonzero key `g` and every guess
`t`, which may depend on the key and on that view, the hash at `g` equals `t` with probability at most `3 · wordGuess μ j`,
three times the best guess of word `j` from every other word of the block.

Given the other words the hash is `c + (x_j mod p) · g^(W − 1 − j)` with a nonzero coefficient, so it equals `t` only when
`x_j` lies in one residue class mod `p`, which holds at most three 32-bit words. For a word uniform and independent of the
rest, `wordGuess μ j = 2^-32` and the bound is `3 · 2^-32 ≈ 1.5/p`; for a word of conditional min-entropy `h` it is
`3 · 2^-h`. A word that other words, read or unread, determine has `wordGuess = 1`: the bound needs one unread word that
the rest of the block leaves unpredictable. The `2^-32` reading needs the guesser's view to be words of the block. A view
of anything computed from the block is outside that model: a Kedlaya–Umans evaluation table, read in a few places, gives
the hash, so conditioned on it `wordGuess μ j = 1` and the bound is vacuous. That case is why no bandwidth guarantee
rests on this statement. -/
def PolyUnpredictable : Prop :=
  ∀ (W : ℕ) (μ : PMF (Block W)) (j : Fin W) (g : (ZMod P)ˣ) (t : ZMod P),
    prob μ (fun x => blockHash (g : ZMod P) x = t) ≤ 3 * wordGuess μ j

/-- **A partially read block's words.** For a block whose words outside `U` are fixed (`known`'s, what was read) and whose
words in `U` are independent and uniform (`partialRead known U`), the best guess of word `j` from every other word is
`2^-32` when `j` is unread (`j ∈ U`), and `1` when it was read. With `PolyUnpredictable`, a guesser whose view is the
words it read guesses the hash of a block with an unread word with probability at most `3 · 2^-32`, at every nonzero
key. The view here is words of the block. A view of anything computed from the block (a Kedlaya–Umans evaluation table,
say) is outside the model and can give the hash outright, which is why no bandwidth guarantee rests on this
statement. -/
def PartialReadGuess : Prop :=
  ∀ (W : ℕ) (known : Block W) (U : Finset (Fin W)) (j : Fin W),
    wordGuess (partialRead known U) j = if j ∈ U then (2 ^ 32 : ℝ≥0∞)⁻¹ else 1

/-- **The fold.** Two lists of block hashes that differ once padded with zero hashes fold alike at a uniform `z ∈ GF(p)`
with probability at most `(n − 1)/p`, `n` the longer list's length: dropping, altering, reordering or inserting a block's
hash changes the fold except with that probability. Dropping trailing hashes equal to `0` changes nothing, since the fold
is `Σ_r h_r · z^r`. -/
def FoldNonCollision : Prop :=
  ∀ (hs hs' : List (ZMod P)),
    prob (PMF.uniformOfFintype (ZMod P))
        (fun z => (∃ r, hs.getD r 0 ≠ hs'.getD r 0) ∧ fold z hs = fold z hs') ≤
      ((max hs.length hs'.length - 1 : ℕ) : ℝ≥0∞) / P

/-- **The digest.** Two lists of `R` blocks of `W` words, some block's residues differing, have the same digest at a
uniform key `(g, z)` with probability at most `((W − 1) + (R − 1))/p`: either some block hash differs, and the folds agree
at at most `R − 1` points `z`, or every block hash agrees, which the block that differs allows at at most `W − 1` points
`g`. -/
def DigestNonCollision : Prop :=
  ∀ (R W : ℕ) (xs ys : Fin R → Block W),
    prob (PMF.uniformOfFintype (ZMod P × ZMod P))
        (fun k => (∃ r, residues (xs r) ≠ residues (ys r)) ∧ digest k.1 k.2 xs = digest k.1 k.2 ys) ≤
      ((W - 1 + (R - 1) : ℕ) : ℝ≥0∞) / P

/-- **The reference computes the hash.** `reference_block_hash`'s loop, `acc ← acc · g + (w mod p)` over the block's words
in order from `acc = 0` (`hornerHash`), returns `blockHash g x`. -/
def HornerIsBlockHash : Prop :=
  ∀ (W : ℕ) (g : ZMod P) (x : Block W), hornerHash g (List.ofFn x) = blockHash g x

end Accounting.Guarantees.Tag
