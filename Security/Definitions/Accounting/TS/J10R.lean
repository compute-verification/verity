import Definitions.Pouw.PearlC.Sha512

/-!
# J10-R: H1's noise from one seed a call, at call scale (trusted; definitions only)

One SHA-512 stream per call row (its dither bits b and digit coins z) and one per call column (its coin bits c and `U`),
seeded once a call; a unit reads its rows' and its columns' slices, and its fold keys stay its own. These are the
bits, op for op, of `verity/core/protocols/pouw/circuit/seeds.py` (`call_seed`, `receipt_seed`, `noise_block`, `leaves`)
and `h1tile.py` (`TileShape`, `_noise_at`, `j10r_blocks`, `j10r_noise`, `fold_keys`, `unit_compressions`,
`j10r_compressions`); the units that compute them are `PouwH1Tile_v5` and `_v6`, beside today's per-unit form, which no
scheme replaces.

- `tagged tag salt pos rootA rootB`: SHA-512 of `u64 pos ‖ (tag ‖ salt ‖ rootA ‖ rootB) ⊕ (u64 pos)*`, the bytes XORed with
  the position's eight cyclically (`seeds._tagged`). `seed` and `receiptSeed` are today's seed_j and σ_R; `callSeed` is
  J10-R's seed_c, at the position of the call's first unit and under the call's registered roots, root(a/<K>@<s>) and
  root(b/<K>).
- `noiseBlock s t`: block `t` of seed `s`'s stream, `SHA-512(NOISE_TAG ‖ s ‖ u64 t)`; `bits` a digest's bits as a unit
  reads them (16 big-endian halves, each least significant bit first); `stream s first n` the first `n` bits from block
  `first` on.
- `rowAt s k first`, `colAt s k first`: a row's (dither bits, digit coins) and a column's (coin bits, `U` as int8 codes,
  `u129`) from block `first` of `s`'s stream. Their widths are functions of `K` alone (`rb k`, `cb k`).
- `TileShape r c k`: a tile or a call. `row i = i RB`, `col j = r RB + j CB`; `noise s L` is today's noise of a unit
  (`h1tile.noise`) and, at a call's layout, the call-scale noise.
- `unitBlocks M k R C I J`: unit `(I, J)`'s public block bases in a call of `M` rows at `R × C` tiles, call row `R I`'s
  first block and call column `C J`'s; `j10rNoise s L b` the unit's noise from bases `b`, `unitNoise` with
  `s = callSeed salt call rootA rootB`, `unitKeys` its fold keys, σ_R at its own position.
- `unitCompressions L`: SHA-512 compressions today's unit takes (its blocks, a key block a lane, two seeds of two each).
  `callCompressionsNum M N k R C` is J10-R's over the call's `units M N R C` units when each block is derived once a
  call: the call's blocks and seed_c's two, and each unit's σ_R (two) and its lanes' key blocks.
-/

namespace Accounting.TS.J10R

open Pouw.PearlC.Sha512

/-- A 24-byte tag: the ASCII name, zero-padded. -/
def tag (s : String) : List ℕ := padTo 24 (ascii s)

def seedTag : List ℕ := tag "verity/pouw/seed/v2"
def receiptTag : List ℕ := tag "verity/pouw/receipt/v1"
def noiseTag : List ℕ := tag "verity/pouw/noise/v1"
def callSeedTag : List ℕ := tag "verity/pouw/call-seed/v1"

/-- A position or a block counter as 8 big-endian bytes. -/
def u64 (x : ℕ) : List ℕ := beBytes 8 x

/-- SHA-512 of `u64 pos` and the tag, the salt and the roots, each byte XORed with byte `i mod 8` of `u64 pos`. -/
def tagged (tg salt : List ℕ) (pos : ℕ) (rootA rootB : List ℕ) : List ℕ :=
  let p := u64 pos
  let data := tg ++ salt ++ rootA ++ rootB
  digest (p ++ List.zipWith (fun x b => x ^^^ b) data (List.replicate (data.length / 8 + 1) p).flatten)

/-- Today's seed_j of the unit at `pos`. -/
def seed (salt : List ℕ) (pos : ℕ) (rootA rootB : List ℕ) : List ℕ := tagged seedTag salt pos rootA rootB

/-- σ_R of the unit at `pos`: its fold keys' seed, under J10-R as today. -/
def receiptSeed (salt : List ℕ) (pos : ℕ) (rootA rootB : List ℕ) : List ℕ := tagged receiptTag salt pos rootA rootB

/-- **J10-R's seed_c** of the call whose first unit is at `call`, under its registered roots `rootA` (root(a/<K>@<s>))
and `rootB` (root(b/<K>)). -/
def callSeed (salt : List ℕ) (call : ℕ) (rootA rootB : List ℕ) : List ℕ := tagged callSeedTag salt call rootA rootB

/-- Block `t` of seed `s`'s noise stream. -/
def noiseBlock (s : List ℕ) (t : ℕ) : List ℕ := digest (noiseTag ++ s ++ u64 t)

/-- A digest's 512 bits as a unit reads them: its 16 big-endian 32-bit halves, each least significant bit first. -/
def bits (d : List ℕ) : List ℕ :=
  (List.range 16).flatMap fun h =>
    let w := beNat ((d.drop (4 * h)).take 4)
    (List.range 32).map fun j => w / 2 ^ j % 2

/-- The first `n` bits of seed `s`'s stream from block `first` on. -/
def stream (s : List ℕ) (first n : ℕ) : List ℕ :=
  ((List.range ((n + 511) / 512)).flatMap fun t => bits (noiseBlock s (first + t))).take n

/-- The number of bits least significant first. -/
def value (bs : List ℕ) : ℕ := bs.foldr (fun b acc => b % 2 + 2 * acc) 0

/-- A noise byte as H1's keyed word in ±64: `-64 + x mod 129`, as its int8 encoding. -/
def u129 (x : ℕ) : ℕ := (192 + x % 256 % 129) % 256

/-- H1's digit count at `ks` segments (`segment.n_digits`), searched from `m` with `f` more tries. -/
def nDigitsFrom (ks : ℕ) : ℕ → ℕ → ℕ
  | 0, m => m
  | f + 1, m => if 128 * ks ≤ 128 ^ m ∧ 127 * ks / 128 ^ (m - 1) ≤ 127 then m else nDigitsFrom ks f (m + 1)

/-- H1's digit count `m` at contraction `k`: the least `m` with `128^m ≥ 128 KS` and `127 KS / 128^(m-1) ≤ 127`. -/
def nDigits (k : ℕ) : ℕ := nDigitsFrom (k / 32) (k / 32 + 1) 1

/-- A row's noise bits at contraction `k`: `k` dither bits, then 32 a digit coin. -/
def rowBits (k : ℕ) : ℕ := k + 32 * (nDigits k - 1)

/-- A column's noise bits: `k` coin bits, then `U`'s 32 bytes. -/
def colBits (k : ℕ) : ℕ := k + 256

/-- A row's blocks, RB. -/
def rb (k : ℕ) : ℕ := (rowBits k + 511) / 512

/-- A column's blocks, CB. -/
def cb (k : ℕ) : ℕ := (colBits k + 511) / 512

/-- A row's noise from block `first` of `s`'s stream: (its `k` dither bits, its digit coins of 32 bits). -/
def rowAt (s : List ℕ) (k first : ℕ) : List ℕ × List (List ℕ) :=
  let z := stream s first (rowBits k)
  (z.take k, (List.range (nDigits k - 1)).map fun d => (z.drop (k + 32 * d)).take 32)

/-- A column's noise from block `first`: (its `k` coin bits, `U`'s 32 int8 codes). -/
def colAt (s : List ℕ) (k first : ℕ) : List ℕ × List ℕ :=
  let z := stream s first (colBits k)
  (z.take k, (List.range 32).map fun t => u129 (value ((z.drop (k + 8 * t)).take 8)))

/-- An `r × c` tile, or a call, at contraction `k`. -/
structure TileShape where
  r : ℕ
  c : ℕ
  k : ℕ

namespace TileShape

/-- Row `i`'s first block. -/
def row (L : TileShape) (i : ℕ) : ℕ := i * rb L.k

/-- Column `j`'s first block, after every row's. -/
def col (L : TileShape) (j : ℕ) : ℕ := L.r * rb L.k + j * cb L.k

/-- The lanes, `r/4 · c/8`, a key block each. -/
def lanes (L : TileShape) : ℕ := L.r / 4 * (L.c / 8)

/-- The noise blocks: the rows' and the columns'. -/
def blocks (L : TileShape) : ℕ := L.r * rb L.k + L.c * cb L.k

end TileShape

/-- The noise of layout `L` from seed `s`, rows then columns: today's unit noise (`h1tile.noise`) at a unit's layout, the
call-scale noise at a call's. -/
def noise (s : List ℕ) (L : TileShape) : List (List ℕ × List (List ℕ)) × List (List ℕ × List ℕ) :=
  ((List.range L.r).map fun i => rowAt s L.k (L.row i), (List.range L.c).map fun j => colAt s L.k (L.col j))

/-- **Unit `(I, J)`'s public block bases** in a call of `M` rows at `R × C` tiles and contraction `k`: the first block of
call row `R I` and of call column `C J` in the call-scale layout (`h1tile.j10r_blocks`). -/
def unitBlocks (M k R C I J : ℕ) : ℕ × ℕ := ((TileShape.mk M C k).row (R * I), (TileShape.mk M C k).col (C * J))

/-- A unit's noise from seed `s` at bases `b`: row `i` from block `b.1 + i RB`, column `j` from `b.2 + j CB`. -/
def j10rNoise (s : List ℕ) (L : TileShape) (b : ℕ × ℕ) : List (List ℕ × List (List ℕ)) × List (List ℕ × List ℕ) :=
  ((List.range L.r).map fun i => rowAt s L.k (b.1 + i * rb L.k), (List.range L.c).map fun j => colAt s L.k (b.2 + j * cb L.k))

/-- Each lane's 16 fold keys from seed `sigma`: lane `ln`'s block, as 32-bit words. -/
def foldKeys (sigma : List ℕ) (L : TileShape) : List (List ℕ) :=
  (List.range L.lanes).map fun ln =>
    let z := stream sigma ln 512
    (List.range 16).map fun w => value ((z.drop (32 * w)).take 32)

/-- **A unit's noise under J10-R**: seed_c of its call (first unit at `call`) at its bases `b`. No seed is an input. -/
def unitNoise (salt rootA rootB : List ℕ) (call : ℕ) (L : TileShape) (b : ℕ × ℕ) :
    List (List ℕ × List (List ℕ)) × List (List ℕ × List ℕ) :=
  j10rNoise (callSeed salt call rootA rootB) L b

/-- A unit's fold keys under J10-R, as today: σ_R at its own position `pos`. -/
def unitKeys (salt rootA rootB : List ℕ) (pos : ℕ) (L : TileShape) : List (List ℕ) :=
  foldKeys (receiptSeed salt pos rootA rootB) L

/-- SHA-512 compressions of today's unit: its noise blocks, a key block a lane, and its two seeds of two each. -/
def unitCompressions (L : TileShape) : ℕ := L.blocks + L.lanes + 4

/-- The units of an `M × N` call at `R × C` tiles. -/
def units (M N R C : ℕ) : ℕ := M / R * (N / C)

/-- J10-R's SHA-512 compressions over the call's `units` when each block is derived once a call: the call's blocks and
seed_c's two, and each unit's σ_R (two) and its lanes' key blocks. A unit's share is this over `units M N R C`. -/
def callCompressionsNum (M N k R C : ℕ) : ℕ :=
  (TileShape.mk M N k).blocks + 2 + units M N R C * (2 + (TileShape.mk R C k).lanes)

end Accounting.TS.J10R
