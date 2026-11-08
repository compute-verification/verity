module

public import Definitions.Core.Game
public import Definitions.Core.MemCheck

/-!
# What a fingerprint collision costs a memory check

A memory check accepts when two multisets of packed records have equal fingerprints at coins drawn uniformly after both
are committed (`Verity.MemCheck.fingerprint`), and keys its memory by prints at a coin of their own
(`Verity.MemCheck.element`). Each bound is the probability, over the coins, that two different objects pass as one.
Both are about records already packed into `m` chunks of a finite field `F`: a packing that is injective keeps two
different multisets of records, or two different keys, different.

`FingerprintCollision`: two different multisets `A` and `B` of records have equal fingerprints at independent uniform
`α` and `γ` with probability at most `max |A| |B| · max 1 (m - 1) / |F|`, so at most `(N + N² (m - 1)) / |F|` for sides
of at most `N` records. Of two chunks over GF(2^128), `N / 2^128`.

`PrintCollision`: some two different keys of a set `S` have equal prints at a uniform `δ` with probability at most
`C(|S|, 2) · (m - 1) / |F|`. Of `2^20` keys of four chunks over GF(2^128), below `2^-87`.

Both are bounds on polynomials' roots: a fingerprint is a polynomial of total degree at most `|A| · max 1 (m - 1)` in
`α` and `γ` (Schwartz–Zippel), and a print one of degree at most `m - 1` in `δ`.
-/

@[expose] public section

namespace Verity.Guarantees

open Verity.MemCheck
open scoped ENNReal

/-- **Two different multisets collide with probability at most `max |A| |B| · max 1 (m - 1) / |F|`.** For every finite
field `F`, number of chunks `m` and multisets `A` and `B` of records packed into `m` chunks: over independent uniform
coins `α` and `γ`, the probability that `A ≠ B` and their fingerprints are equal is at most
`max |A| |B| · max 1 (m - 1) / |F|`. -/
def FingerprintCollision : Prop :=
  ∀ {F : Type} [Field F] [Fintype F] (m : ℕ) (A B : Multiset (Fin m → F)),
    Game.prCoin (fun c : F × F => A ≠ B ∧ fingerprint c.1 c.2 A = fingerprint c.1 c.2 B) ≤
      ((max A.card B.card * max 1 (m - 1) : ℕ) : ℝ≥0∞) / Fintype.card F

/-- **Some two keys of a set share a print with probability at most `C(|S|, 2) · (m - 1) / |F|`.** For every finite
field `F`, number of chunks `m` and set `S` of keys packed into `m` chunks: over a uniform coin `δ`, the probability that
two different keys of `S` have equal prints at `δ` is at most `C(|S|, 2) · (m - 1) / |F|`. -/
def PrintCollision : Prop :=
  ∀ {F : Type} [Field F] [Fintype F] (m : ℕ) (S : Finset (Fin m → F)),
    Game.prCoin (fun δ : F => ∃ k ∈ S, ∃ k' ∈ S, k ≠ k' ∧ element δ k = element δ k') ≤
      ((S.card.choose 2 * (m - 1) : ℕ) : ℝ≥0∞) / Fintype.card F

end Verity.Guarantees
