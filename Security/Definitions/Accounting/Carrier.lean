import Definitions.Accounting.RateAuditSketchDeployed

/-!
# Carriers: device memory whose post-seed contents are a circuit's values (trusted)

A verbatim device (`RateAuditSketch.lean`) obtains a store sector by holding it on chip, reading it from device memory,
taking it over the side link or rebuilding it. Stacking adds a fifth way: a carrier, device memory whose contents were
chosen after the seed (a KV page the audit credits, say), holds other blocks' encodings, and the device decodes them
from it instead of reading them. This file is a carrier's vocabulary; `Models/Accounting/Guarantees/Stacking.lean`
bounds what carriers hold and prices it as on-chip state.

**Provenance.** A carrier's contents are values of a circuit (the served model as an exact-integer Program). In the
ideal encoding model the store's sectors are uniform, they are the blocks' encodings, and the circuit's values depend
on them in two ways only:

* `input x f`: `x` bits the prover feeds the circuit, any function `f` of the store (the prover knows all of it);
* `encode t x _ f`: `x ≤ m·256` bits of tile `t`'s encoding, any `x`-bit function `f` of it, which the circuit
  computes itself (exact-integer ops on the tile's payload, the tile's other bits among what it may hold), at a cost in
  roundings that depends only on `x`.

A `Provenance` is the circuit seen through those two channels: every other gate is a function of the values so far,
folded into the continuations, so the gates themselves are not here. It covers every value a carrier's contents
depend on, back to values that depend on no store sector (a request's start, under the property's isolation).

**A carrier** of the store sectors `carried` is a provenance whose values are those sectors, whole and as they are, at
every store (the device's decoding is its last step). The verbatim class delivers every sector this way, so a
carrier's sectors reach the chip as `held` sectors do: whole, as they are, and at no read.
-/

namespace Accounting.RateSketch

open Pous (Bits)

/-- How a circuit's values over the store, `N` tiles of `m` sectors, came to depend on it, ending in the values (of
type `α`). Each step waits on `x` bits that depend on the store: an input the prover feeds, any function of the store
(`input`), or `x ≤ m·256` bits of a tile's encoding, any function of it, which the circuit computes (`encode`). -/
inductive Provenance (N m : ℕ) (α : Type) : Type where
  | done : α → Provenance N m α
  | input : (x : ℕ) → (SketchStore N m → Bits x) → (Bits x → Provenance N m α) → Provenance N m α
  | encode : Fin N → (x : ℕ) → x ≤ m * 256 → ((Fin m → Sector) → Bits x) → (Bits x → Provenance N m α) →
      Provenance N m α

namespace Provenance

variable {N m : ℕ} {α : Type}

/-- The circuit's values at store `E`. -/
def run (E : SketchStore N m) : Provenance N m α → α
  | done a => a
  | input _ f k => (k (f E)).run E
  | encode t _ _ f k => (k (f fun u => E (t, u))).run E

/-- The bits fed as inputs at store `E`: `x` per `input` step of `x` bits. -/
def inputs (E : SketchStore N m) : Provenance N m α → ℕ
  | done _ => 0
  | input x f k => (k (f E)).inputs E + x
  | encode t _ _ f k => (k (f fun u => E (t, u))).inputs E

/-- The bits encoded at store `E`: `x` per `encode` step of `x` bits. -/
def encoded (E : SketchStore N m) : Provenance N m α → ℕ
  | done _ => 0
  | input _ f k => (k (f E)).encoded E
  | encode t x _ f k => (k (f fun u => E (t, u))).encoded E + x

/-- The roundings the encodes cost at store `E`, at `cost x` per `encode` step of `x` bits. The circuit's other
roundings only add to its count, so a budget on its roundings bounds this. -/
def roundings (cost : ℕ → ℕ) (E : SketchStore N m) : Provenance N m α → ℕ
  | done _ => 0
  | input _ f k => (k (f E)).roundings cost E
  | encode t x _ f k => (k (f fun u => E (t, u))).roundings cost E + cost x

end Provenance

/-- A carrier of the store sectors `carried`: a provenance whose values are those sectors, whole and as they are, at
every store. -/
structure Carrier (N m : ℕ) where
  carried : Finset (Fin N × Fin m)
  provenance : Provenance N m (carried → Sector)
  holds : ∀ E, provenance.run E = fun p => E p.1

end Accounting.RateSketch
