module

public import Definitions.Pouw.H1Zy.Lane
public import Mathlib.Data.BitVec

/-!
# `PouwH1TileZy_v1{TILE}` on words, over `PouwH1Tile_v1` or `PouwH1Tile_v3` (trusted)

The H1 tile unit with its z/y record checked (`verity/protocols/accounting/pouw/prover/circuit/h1zy.py`, `PouwH1TileZy`, `_body`)
reads its `R` A rows and `C` B rows (`K` s8 codes and an f32 scale each), its `R × C` record (cell `(i, j)` its s32 z
and bf16 y, row-major), its salt, position and roots, and returns `{receipt, ok}`. TILE on the rows' codes gives each
cell's z (and the receipt, which `ok` doesn't read), and `ok` is `PouwZyCheck_v1{R, C}` (`zyCheck`) of the record
against that z at the rows' scales (`tileOk`). TILE is one of `h1zy.TILES` over its full grid in row-major order
(`_refuse_tile`), each with `EXACT = GemmAccumulatorS8_v2{K, sm120}` (`h1window.h1_tile`):

* **v1** (`h1tile.PouwH1Tile`, `_body`): cell `(i, j)`'s z is EXACT on row `i`'s and column `j`'s codes (`gemmAccS8`).
* **v3** (`h1tile.PouwH1Tile3`, `_body_v3`, `_one_stream`, `_cell_z`): cell `(i, j)`'s z is made from the words its
  lane's chain absorbs (`streamZ`), on the primitives' word views:
  * row `i`'s `X = A + b` (`PouwI8AddBit`, `b` its dither bits), and its residue sums `Y_ρ = Σ_s X_{32 s + ρ}`
    (`PouwSx8`, then `PouwResAdd` per segment) split into `m = nDigits (K / 32)` digits (`h1tile._digits`:
    `PouwDigitLo`, `PouwDigitHi`, the last coin digit through `PouwDigitZ`, the top digit `PouwLow8`);
  * column `j`'s `U` (`PouwU129` of its noise bytes), `-U` (`PouwI8Neg`) and `W′ = B + c + U` (`PouwI8AddBit`, then
    `PouwI8Add` with `U` repeated over `K`);
  * the cell's prefix word `T` over every segment (`PouwSegDot0`, then `PouwSegDot`), `O′ = T + (-U) · A_0`
    (`PouwSegDot`) and `D_d = U · A_d` (`PouwSegDot0`), `h1tile._chain`'s words for the entry of its lane at the
    cell's row and column slots;
  * the recombination `O′ - Σ_d 128^d D_d` (`PouwOneStream2`, `PouwOneStream3`: `oneStream`), then the wrap correction
    per segment (`PouwWrapBits` of `B`, `c` and `U`, `PouwWrapSeg`), then the dither peel per segment (`PouwXWrap` of
    `A` and `b`, `PouwPeelSeg`).

  Each lane's entry `e` is its row slot's row and column slot's column, and the lanes' entries are the tile's cells once
  each, so cell `(i, j)`'s words are computed from row `i` and column `j` alone; `streamZ` states them per cell. The
  noise (`TileNoise`: each row's dither bits and digit coins, each column's coin bits and noise bytes for `U`) is
  seed_j's stream from the unit's salt, position and roots (`PouwUnitSeeds`, `noise_blocks`); it is the parameter
  `noise`, a function of the public inputs, and not stated here.

Every word view is its primitive's, with indices past `K` read as `0` (`codeAt`, `bitAt`), which only `K`'s that the
tile doesn't bind at reach. `PouwOneStream` is stated at any digit count (`one_stream`), where the Definition has it
only at `m = 2, 3`. Proved in `Proofs.Pouw.H1Zy` to give the exact dot in either form.
-/

@[expose] public section

namespace Pouw.H1Zy

/-! ### The tile's inputs and noise -/

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

/-! ### The tile's z -/

end Pouw.H1Zy
