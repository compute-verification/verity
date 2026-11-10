module

public import Definitions.Accounting.Guarantees.FillerLuck

/-!
# The filler's per-check luck: lemmas

The statements of `Definitions.Accounting.Guarantees.FillerLuck` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.

* `ParityLuck`: Lemma P2 at a cell, in any word ring read mod 2 and any column of uniform parity: a value with `R`
  two-sided products equals the cell's word with probability at most `luck (K − R) = ½ + 2^(−(K−R)/2−1)`.
* `FillerParityBalanced`: those four columns have uniform parity, so the statements below, stated at any balanced
  column, hold at each of them.
-/

@[expose] public section

namespace Accounting.Guarantees.FillerLuck

open Accounting.Filler Accounting.TS Pouw.RelativeSecurity

/-- **Lemma P2 at a cell**. Let the row's bits `x ∈ {0,1}^K` be uniform and the column's noise `y` uniform on `Y`, with
its entries' parity uniform. A value with `R` two-sided products over `A`, its factors any functions of their side,
equals the cell's word `Σ_k x_k·c_k(y)` with probability at most `luck (K − R) = ½ + 2^(−(K−R)/2−1)`, at every `K`. -/
def ParityLuck : Prop :=
  ∀ {Y A : Type} [Fintype Y] [CommRing A] (φ : A →+* ZMod 2) {K : ℕ} (c : Y → Fin K → A) (R : ℕ)
    (P : (Fin K → ZMod 2) → Y → A),
    ParityBalanced φ c → Products R P →
      prob (fun xy : (Fin K → ZMod 2) × Y => P xy.1 xy.2 = cellWord c xy.1 xy.2) ≤ luck (K - R)

/-- **The filler's columns have uniform parity**: pfill-bits' (its bits) and pfill-bU's (`uOf t ≡ t₀ ⊕ t₆`), in the s32
word and over `ℤ`, at every `K`. -/
def FillerParityBalanced : Prop :=
  ∀ K : ℕ, ParityBalanced toF2 (bitsCol (K := K)) ∧ ParityBalanced (Int.castRingHom (ZMod 2)) (bitsCol (K := K)) ∧
    ParityBalanced toF2 (buCol (K := K)) ∧ ParityBalanced (Int.castRingHom (ZMod 2)) (buCol (K := K))

end Accounting.Guarantees.FillerLuck
