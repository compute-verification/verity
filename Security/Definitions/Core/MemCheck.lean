module

public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Algebra.BigOperators.Group.Multiset.Defs
public import Mathlib.Algebra.Field.Defs
public import Mathlib.Data.Fintype.Card

/-!
# Offline memory checking's fingerprint

Offline memory checking (Blum, Evans, Gemmell, Kannan and Naor) reduces a log of memory accesses to one equality of
multisets of records (address, value, time): the records the reads consume with the final contents, against the records
the writes produce with the initial contents. The verifier tests that equality by a fingerprint over a finite field `F`,
at coins it draws after both multisets are committed. A record is packed into `m` chunks `c : Fin m → F`; its element at
a coin `γ` is `∑ i, γ^i · c i` (`element`), and a multiset of packed records has the fingerprint
`∏ c, (α + element γ c)` at coins `α` and `γ` (`fingerprint`). The same `element`, at a coin of its own, prints a key
packed into `m` chunks as one field element. Packing is the caller's: an injective one keeps different multisets of
records different.

What a collision of each costs is `Verity.Guarantees.FingerprintCollision` and `Verity.Guarantees.PrintCollision`.
-/

@[expose] public section

namespace Verity.MemCheck

/-- A record packed into `m` chunks `c`, read at the coin `γ`: `∑ i, γ^i · c i`. -/
def element {F : Type} [Field F] {m : ℕ} (γ : F) (c : Fin m → F) : F :=
  ∑ i : Fin m, γ ^ (i : ℕ) * c i

/-- The fingerprint of a multiset `A` of records, each packed into `m` chunks, at the coins `α` and `γ`:
`∏ c ∈ A, (α + element γ c)`. -/
def fingerprint {F : Type} [Field F] {m : ℕ} (α γ : F) (A : Multiset (Fin m → F)) : F :=
  (A.map fun c => α + element γ c).prod

end Verity.MemCheck
