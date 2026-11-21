module

public import Definitions.Pouw.H1Zy.Lane
public import Mathlib.Data.BitVec

/-!
# `PouwH1Tile_v3`'s word pieces (trusted)

`PouwH1Tile_v3` (`verity/protocols/accounting/pouw/prover/circuit/h1tile.py`, `_body_v3`) makes a cell's z from its row's
residue sums `Y_ρ = Σ_s X_{32 s + ρ}` (`PouwSx8`, then `PouwResAdd` per segment: `resSum`), each split into digits
(`h1tile._digits`: `PouwDigitLo`, `PouwDigitHi`, the last coin digit through `PouwDigitZ`, the top digit `PouwLow8`:
`digitsFrom`). These are those primitives' word views, and `segFold`, a fold over segments.
-/

@[expose] public section

namespace Pouw.H1Zy

/-! ### Segments -/

/-- A fold over segments: `f s` after segment `s`'s predecessors, from `init`. -/
def segFold (f : ℕ → BitVec 32 → BitVec 32) (init : BitVec 32) : ℕ → BitVec 32
  | 0 => init
  | n + 1 => f n (segFold f init n)

/-! ### The primitives' word views -/

/-- `PouwSx8_v1`: an int8 as a 32-bit word. -/
def sx8 (x : BitVec 8) : BitVec 32 := BitVec.ofInt 32 x.toInt

/-- `PouwResAdd_v1`: `acc + x mod 2^32`, `x` int8. -/
def resAdd (acc : BitVec 32) (x : BitVec 8) : BitVec 32 := acc + sx8 x

/-- `PouwDigitLo_v1`: a digit's int8 encoding, `(r mod 128) - 128 z`. -/
def digitLo (r : BitVec 32) (z : Bool) : BitVec 8 := BitVec.ofNat 8 (r.toNat % 128 + if z then 128 else 0)

/-- `PouwDigitHi_v1`: the rest after a digit, `⌊r / 128⌋ + z mod 2^32`, `r` an s32. -/
def digitHi (r : BitVec 32) (z : Bool) : BitVec 32 := r.sshiftRight 7 + if z then 1 else 0

/-- `PouwDigitZ_v1`: the last coin digit's `z`, `z` but `0` where `⌊r / 128⌋ = 127`. -/
def digitZ (r : BitVec 32) (z : Bool) : Bool := if r.sshiftRight 7 = 127 then false else z

/-- `PouwLow8_v1`: `r mod 2^8`, the top digit's int8 encoding. -/
def low8 (r : BitVec 32) : BitVec 8 := r.setWidth 8

/-! ### The digits -/

/-- **`h1tile._digits`' digits of one residue sum** `r`, from digit `d` on with `n` coin digits left, at the coins `z`:
`PouwDigitLo` of the rest at the coin, the rest `PouwDigitHi` of it; the last coin digit's coin through `PouwDigitZ`;
the top digit `PouwLow8` of the last rest. -/
def digitsFrom (z : ℕ → Bool) : ℕ → ℕ → BitVec 32 → List (BitVec 8)
  | 0, _, r => [low8 r]
  | 1, d, r => [digitLo r (digitZ r (z d)), low8 (digitHi r (digitZ r (z d)))]
  | n + 2, d, r => digitLo r (z d) :: digitsFrom z (n + 1) (d + 1) (digitHi r (z d))

/-- **A residue sum** `Y_ρ` of an entry-indexed row after its segments `0 .. n`: `PouwSx8` of segment 0's entry `ρ`,
then `PouwResAdd` of each later segment's. -/
def resSum (x : ℕ → BitVec 8) (ρ : Fin 32) : ℕ → BitVec 32
  | 0 => sx8 (x ρ.val)
  | n + 1 => resAdd (resSum x ρ n) (x (32 * (n + 1) + ρ.val))


end Pouw.H1Zy
