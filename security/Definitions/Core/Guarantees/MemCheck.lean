module

public import Definitions.Core.Game
public import Definitions.Core.MemCheck

/-! What an accepted memory check establishes (`MemoryChecked`), and what a fingerprint or print collision costs it. -/

@[expose] public section

namespace Verity.Guarantees

open Verity.MemCheck
open scoped ENNReal

/-- **A log offline memory checking accepts read its memory correctly.** For all types of addresses `A` and values `V`,
initial values `v0`, logs, sets of touched addresses and final contents `fin`: if the check accepts them, each access
read the value and time its address held before it, the final contents are what the log leaves, and the log touches only
touched addresses. -/
def MemoryChecked : Prop :=
  ∀ {A V : Type} [DecidableEq A] (v0 : A → V) (log : List (Access A V)) (touched : Finset A) (fin : A → V × ℕ),
    Checked v0 log touched fin → ReadsCorrect v0 log touched fin

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
