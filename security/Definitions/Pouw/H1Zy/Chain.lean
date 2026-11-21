module

public import Definitions.Pouw.H1Zy.LaneChunk
public import Definitions.Pouw.H1Zy.Tile
public import Definitions.Pouw.PearlC.Sha512

/-!
# A lane's fx1 chain on words: `h1tile._chain` at a 32 × 32 leaf (trusted)

Lane `l` of a sampled leaf over `PouwH1Tile_v1` folds `KS + m` sets of 32 words into an fx1 state keyed from σ_R, and its
partial is the state's two receipt words after each set (`h1tile.fold`, `h1zy.lane_partial` before the admit mask). This
states that chain as the step machine `ChainSteps` (`h1Chain K`), so a lane's chain words are `chainOf (h1Chain K)` and a
chunk's carry is its state. On the primitives' word views:

* **Seeds and noise** (`seeds.py`). seed_j and σ_R are SHA-512 of the position's 8 big-endian bytes, then the tag, salt
  and roots, each byte XORed with byte `i mod 8` of the position (`tagged`); block `t` of a seed's noise stream is
  SHA-512 of `NOISE_TAG ‖ seed ‖ u64(t)` (`noiseChunk`), and its bit `β` is bit `β mod 32` of big-endian half
  `⌊(β mod 512) / 32⌋` of block `⌊β / 512⌋` (`streamBit`). The salt, position and roots are the `pub` port's fields
  (`SeedIn`): field bit `t` is bit `t mod 32` of the field's 32-bit half `⌊t / 32⌋`, halves big-endian (`seeds.leaves`).
* **The lane's noise** (`h1tile.noise`, `fold_keys`, at the layout `h1tile.Layout(32, 32, K)`, `laneNoise`): row `i`'s
  stream starts at block `i · RB` and holds its `K` dither bits then `m - 1` digit coins of 32 bits; column `j`'s at
  block `32 RB + j · CB` holds its `K` coin bits then U's 32 bytes, each `PouwU129` of the byte's value; the lane's fold
  keys are block `l` of σ_R's stream, 16 words. Its rows and columns are `laneRow l s` and `laneCol l t`.
* **A segment** (`noiseChain`'s `seg`, `h1tile._row`, `_col` and `_chain`): `X = PouwI8AddBit(A, dither)` and
  `W′ = PouwI8Add(PouwI8AddBit(B, coin), U)` at the segment's 32 entries; each of the 32 entries `e` takes
  `PouwSegDot(T_e, X[row_slot e], W′[col_slot e])`; each row slot's residue sums `PouwResAdd(Y, X)`; the state absorbs
  the 32 new T (`PouwFx1Absorb`) and the chain gains `PouwXinFinal` of its y half and of its z half. From `init`'s zero T
  and Y this is `h1tile`'s first segment too: `PouwSegDot0` is `PouwSegDot` from 0, `PouwSx8` is `PouwResAdd` from 0.
* **The end** (`fin`, `h1tile._digits` and `_chain`'s tail): each row slot's digits of its residue sums (`PouwDigitLo`,
  `PouwDigitHi`, `PouwDigitZ` at the last coin digit, `PouwLow8`), then `O′ = PouwSegDot(T_last, PouwI8Neg(U), A_0)` and
  `D_d = PouwSegDot(0, U, A_d)` for `d = 1 .. m - 1`, each absorbed as a set. The partial's word `w` is the chain's word
  `w`, `0` past its `2 (KS + m)` words.
-/

@[expose] public section

namespace Pouw.H1Zy

open Pouw.PearlC.Sha512 (digest beBytes beNat ascii padTo)

/-! ## Seeds and the noise stream (`seeds.py`) -/

/-- The public seed inputs as the lane's `pub` port holds them: the salt (512 bits), the position (64) and the roots
(1024), each a field whose bit `t` is bit `t mod 32` of its 32-bit half `⌊t / 32⌋`. -/
structure SeedIn where
  salt : ℕ
  position : ℕ
  roots : ℕ

/-- A field's bytes: its first `n` 32-bit halves, each big-endian. -/
def halvesBytes (n v : ℕ) : List ℕ := (List.range n).flatMap fun h => beBytes 4 (v / 2 ^ (32 * h) % 2 ^ 32)

/-- `seeds.SEED_TAG`, `RECEIPT_TAG` and `NOISE_TAG`: ASCII, zero-padded to 24 bytes. -/
def seedTag : List ℕ := padTo 24 (ascii "verity/pouw/seed/v2")
def receiptTag : List ℕ := padTo 24 (ascii "verity/pouw/receipt/v1")
def noiseTag : List ℕ := padTo 24 (ascii "verity/pouw/noise/v1")

/-- **`seeds._tagged`**: SHA-512 of the position's 8 big-endian bytes, then `tag ‖ salt ‖ root_a ‖ root_b` with byte `i`
XORed with byte `i mod 8` of the position. -/
def tagged (tag : List ℕ) (x : SeedIn) : List ℕ :=
  let p := halvesBytes 2 x.position
  let data := tag ++ halvesBytes 16 x.salt ++ halvesBytes 32 x.roots
  digest (p ++ (List.range data.length).map fun i => data.getD i 0 ^^^ p.getD (i % 8) 0)

/-- seed_j (`seeds.seed`). -/
def seedJ (x : SeedIn) : List ℕ := tagged seedTag x

/-- σ_R (`seeds.receipt_seed`). -/
def sigmaR (x : SeedIn) : List ℕ := tagged receiptTag x

/-- Block `t` of a seed's noise stream (`seeds.noise_block`). -/
def noiseChunk (seed : List ℕ) (t : ℕ) : List ℕ := digest (noiseTag ++ seed ++ beBytes 8 t)

/-- Bit `β` of a block's 512 bits: bit `β mod 32` of its big-endian half `⌊β / 32⌋`. -/
def chunkBit (blk : List ℕ) (β : ℕ) : Bool := beNat ((blk.drop (4 * (β / 32))).take 4) / 2 ^ (β % 32) % 2 = 1

/-- Bit `β` of a seed's noise stream from block `first` (`seeds.stream`, `h1tile._stream`). -/
def streamBit (seed : List ℕ) (first β : ℕ) : Bool := chunkBit (noiseChunk seed (first + β / 512)) (β % 512)

/-- The value of `n` stream bits from bit `β`, least significant first (`h1tile._value`). -/
def streamVal (seed : List ℕ) (first β n : ℕ) : ℕ :=
  ((List.range n).map fun i => if streamBit seed first (β + i) then 2 ^ i else 0).sum

/-! ## The layout (`h1tile.Layout` at a 32 × 32 leaf) -/

/-- `m` at contraction `K`. -/
def digitsK (K : ℕ) : ℕ := nDigits (K / 32)

/-- A row's stream blocks: `⌈(K + 32 (m - 1)) / 512⌉`. -/
def rowBlocks (K : ℕ) : ℕ := (K + 32 * (digitsK K - 1) + 511) / 512

/-- A column's stream blocks: `⌈(K + 256) / 512⌉`. -/
def colBlocks (K : ℕ) : ℕ := (K + 256 + 511) / 512

/-- Row `i`'s first block. -/
def rowFirst (K i : ℕ) : ℕ := i * rowBlocks K

/-- Column `j`'s first block, past the 32 rows'. -/
def colFirst (K j : ℕ) : ℕ := 32 * rowBlocks K + j * colBlocks K

/-! ## duo and fx1 (`segment.py`)

The segment and digit pieces' word views (`i8AddBit`, `i8Add`, `i8Neg`, `u129`, `segDot`, `resAdd`, `digitLo`,
`digitHi`, `digitZ`, `low8`, `digits`, `nDigits`) are `Tile`'s. -/

/-! ## The chain -/

/-- **A lane's noise**: each row slot's dither bits and digit coins (`digit s d ρ`, coin `ρ` of digit `d`), each column
slot's coin bits and U, and its fold keys. -/
structure LaneNoise where
  dither : Fin 4 → ℕ → Bool
  digit : Fin 4 → ℕ → Fin 32 → Bool
  coin : Fin 8 → ℕ → Bool
  u : Fin 8 → Fin 32 → BitVec 8
  key : Fin 16 → BitVec 32

/-- **Lane `l`'s noise** at contraction `K` (`h1tile.noise`, `fold_keys`): its rows' and columns' streams of seed_j and
block `l` of σ_R's. -/
def laneNoise (K : ℕ) (x : SeedIn) (l : Fin 32) : LaneNoise where
  dither s k := streamBit (seedJ x) (rowFirst K (laneRow l s)) k
  digit s d ρ := streamBit (seedJ x) (rowFirst K (laneRow l s)) (K + 32 * d + ρ.val)
  coin t k := streamBit (seedJ x) (colFirst K (laneCol l t)) k
  u t j := u129 (BitVec.ofNat 8 (streamVal (seedJ x) (colFirst K (laneCol l t)) (K + 8 * j.val) 8))
  key w := BitVec.ofNat 32 (streamVal (sigmaR x) l.val (32 * w.val) 32)

end Pouw.H1Zy
