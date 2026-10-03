import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Basic.NNReal.Basic
import Pous.Protocol.Game.Oracle

/-!
# The bulk read: is the store in device memory?

The timed audit (`Pous/Protocol/Model/Audit.lean`) certifies that the server holds `C`. It cannot tell `C` in device
memory (HBM) from `C` in host memory, behind the host link (PCIe). The bulk read is the placement check that can
(drafted in `benchmarks/pous/hbm_audit/README.md`, PR #935). The verifier names `R` distinct blocks of `C` and a fresh
key `(g, z)`; the server returns the digest of those blocks, a keyed two-level polynomial hash over a finite field `F`,
within a deadline `T`. A block that only host memory holds must cross the host link to be read, and the named
assumption `Pous.Assumptions.TwoTierBandwidth` bounds what the link carries in time `T`.

**The model.** A block is `W` words in `F` (the benchmark: 16,384 `uint32` words taken mod `p = 2^31 − 1`). A
responder is a program that reads the store block by block (`Prog` over `memory`, one block per query), adaptively,
and returns a field element; the store is its only input about `C`, and it gets nothing else about `C`. Its elapsed
time is a physical quantity, which only the named assumption constrains. `host` is the set of blocks the device does
not hold: reading one of them crosses the host link. The guarantees average over a uniform store, the model of `C`'s
pseudorandom labels.
-/

namespace Pous.Bandwidth

open scoped NNReal

/-- A block: `W` words, each an element of the field `F`. -/
abbrev Blk (W : ℕ) (F : Type) : Type := Fin W → F

/-- The store `C`: `N` blocks. -/
abbrev Store (N W : ℕ) (F : Type) : Type := Fin N → Blk W F

/-- The verifier's key for one lane: the two evaluation points `(g, z)`. -/
abbrev Key (F : Type) : Type := F × F

section Digest

variable {F : Type} [CommRing F]

/-- A block's hash at `g`: `P_g(x) = Σ_j x_j · g^(W − 1 − j)`, Horner's rule from the first word
(`reference_block_hash` in `benchmarks/pous/hbm_audit/digest.py`). -/
def blockHash {W : ℕ} (g : F) (x : Blk W F) : F :=
  ∑ j : Fin W, x j * g ^ (W - 1 - (j : ℕ))

/-- The digest of `R` blocks, in order, at the key `(g, z)`: `Σ_r P_g(x_r) · z^r` (`reference_digest`). -/
def digest {R W : ℕ} (g z : F) (xs : Fin R → Blk W F) : F :=
  ∑ r : Fin R, blockHash g (xs r) * z ^ (r : ℕ)

end Digest

section Responder

variable {F : Type} [Fintype F] [Nonempty F]

/-- The store as an oracle: a query names a block, and the answer is its content. A `Prog (memory N W F) α` reads
blocks of the store in rounds of parallel reads, each round chosen from the blocks read before. -/
abbrev memory (N W : ℕ) (F : Type) [Fintype F] [Nonempty F] : IdealModel where
  Ω := Store N W F
  Qry := Fin N
  Ans := Blk W F
  answer := fun C i => C i

/-- The blocks a program reads on the path it takes when the store is `C`. -/
def reads {N W : ℕ} {α : Type} (C : Store N W F) : Prog (memory N W F) α → Finset (Fin N)
  | .ret _ => ∅
  | .round qs k => qs.toFinset ∪ reads C (k (qs.map ((memory N W F).answer C)))

/-- A responder to the bulk read. `prog k S` answers the key `k` and the challenge `S` (the `R` block indices, in
order) by reading blocks of the store and returning a digest. `time C k S` is the elapsed time of that response, from
the challenge to the answer, when the store is `C`. -/
structure BulkResponder (N R W : ℕ) (F : Type) [Fintype F] [Nonempty F] where
  prog : Key F → (Fin R → Fin N) → Prog (memory N W F) F
  time : Store N W F → Key F → (Fin R → Fin N) → ℝ≥0

end Responder

/-- The verifier accepts the response to the challenge `S` under the key `k` when the store is `C`: it is the digest of
the challenged blocks, in challenge order, and it arrives within the deadline `T`. -/
def Accepts {N R W : ℕ} {F : Type} [Field F] [Fintype F] (A : BulkResponder N R W F) (T : ℝ≥0)
    (S : Fin R → Fin N) (C : Store N W F) (k : Key F) : Prop :=
  (A.prog k S).run ((memory N W F).answer C) = digest k.1 k.2 (fun r => C (S r)) ∧ A.time C k S ≤ T

end Pous.Bandwidth
