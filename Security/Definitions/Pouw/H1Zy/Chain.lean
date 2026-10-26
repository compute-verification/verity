import Definitions.Pouw.H1Zy.LaneChunk
import Definitions.Pouw.H1Zy.Tile
import Definitions.Pouw.PearlC.Sha512

/-!
# A lane's fx1 chain on words: `h1tile._chain` at a 32 × 32 leaf (trusted)

Lane `l` of a drawn leaf over `PouwH1Tile_v1` folds `KS + m` sets of 32 words into an fx1 state keyed from σ_R, and its
partial is the state's two receipt words after each set (`h1tile.fold`, `h1zy.lane_partial` before the admit mask). This
states that chain as the step machine `ChainSteps` (`h1Chain K`), so a lane's chain words are `chainOf (h1Chain K)` and a
chunk's carry is its state. On the primitives' word views:

* **Seeds and noise** (`seeds.py`). seed_j and σ_R are SHA-512 of the position's 8 big-endian bytes, then the tag, salt
  and roots, each byte XORed with byte `i mod 8` of the position (`tagged`); block `t` of a seed's noise stream is
  SHA-512 of `NOISE_TAG ‖ seed ‖ u64(t)` (`noiseBlock`), and its bit `β` is bit `β mod 32` of big-endian half
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
def noiseBlock (seed : List ℕ) (t : ℕ) : List ℕ := digest (noiseTag ++ seed ++ beBytes 8 t)

/-- Bit `β` of a block's 512 bits: bit `β mod 32` of its big-endian half `⌊β / 32⌋`. -/
def blockBit (blk : List ℕ) (β : ℕ) : Bool := beNat ((blk.drop (4 * (β / 32))).take 4) / 2 ^ (β % 32) % 2 = 1

/-- Bit `β` of a seed's noise stream from block `first` (`seeds.stream`, `h1tile._stream`). -/
def streamBit (seed : List ℕ) (first β : ℕ) : Bool := blockBit (noiseBlock seed (first + β / 512)) (β % 512)

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

/-- `segment.mix32`. -/
def mix32 (x : BitVec 32) : BitVec 32 :=
  let x := x ^^^ (x >>> 16)
  let x := x * 0x7FEB352D
  let x := x ^^^ (x >>> 15)
  let x := x * 0x846CA68B
  x ^^^ (x >>> 16)

/-- duo's multiplier `KA[t]`, XOR key `KK[t]` and rotation `ROT[t]` at position `t < 8`. -/
def duoKa (t : ℕ) : BitVec 32 := mix32 (BitVec.ofNat 32 (0xA0000000 + 2 * t)) ||| 1
def duoKk (t : ℕ) : BitVec 32 := mix32 (BitVec.ofNat 32 (0xB0000000 + 2 * t + 1))
def duoRot (t : ℕ) : ℕ := (t * 7 + 5) % 31 + 1

/-- fx1's multiplier of entry `q` (`segment.fx1_key`). -/
def fx1Key (q : ℕ) : BitVec 32 := mix32 (BitVec.ofNat 32 (0xA0000000 + 2 * (16 + 32 * q))) ||| 1

/-- One step of duo at position `t` and entry `e` (`segment._duo_absorb`'s inner loop): state word `i = 4 (t mod 2) + e`
of y and of z, `u = y_i ⊕ p ⊕ KK[t]`, `y_i ← u · KA[t] + z_i`, `z_i ← rotl(u, ROT[t]) + y_i`. -/
def duoStep (st : Fin 16 → BitVec 32) (t e : ℕ) (p : BitVec 32) : Fin 16 → BitVec 32 :=
  let i := 4 * (t % 2) + e
  let u := st ⟨i % 16, Nat.mod_lt _ (by omega)⟩ ^^^ p ^^^ duoKk t
  let y := u * duoKa t + st ⟨(8 + i) % 16, Nat.mod_lt _ (by omega)⟩
  let z := u.rotateLeft (duoRot t) + y
  fun j => if j.val = i then y else if j.val = 8 + i then z else st j

/-- **`PouwDuoAbsorb`** on words: duo's 16-word state after a set's 32 words, position `t` taking words `4 t .. 4 t + 3`. -/
def duoAbsorb (st : Fin 16 → BitVec 32) (p : Fin 32 → BitVec 32) : Fin 16 → BitVec 32 :=
  (List.range 8).foldl (fun st t => (List.range 4).foldl (fun st e =>
    duoStep st t e (p ⟨(4 * t + e) % 32, Nat.mod_lt _ (by omega)⟩)) st) st

/-- **`PouwFx1Absorb`** on words: duo's step on each word `q` times `fx1Key q`. -/
def fx1Absorb (st : Fin 16 → BitVec 32) (p : Fin 32 → BitVec 32) : Fin 16 → BitVec 32 :=
  duoAbsorb st fun q => p q * fx1Key q.val

/-- **`PouwXinFinal`** on words: the XOR of 8 words. -/
def xinFinal (w : Fin 8 → BitVec 32) : BitVec 32 := (List.ofFn w).foldl (· ^^^ ·) 0

/-- Entry `q`'s row slot (`segment.row_slot`): `2 ⌊q / 16⌋ + ⌊(q mod 4) / 2⌋`. -/
def rowSlot (q : Fin 32) : Fin 4 := ⟨2 * (q.val / 16) + q.val % 4 / 2, by omega⟩

/-- Entry `q`'s column slot (`segment.col_slot`): `2 (⌊q / 4⌋ mod 4) + q mod 2`. -/
def colSlot (q : Fin 32) : Fin 8 := ⟨2 * (q.val / 4 % 4) + q.val % 2, by omega⟩

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

/-- The chain's state after absorbing a set: the fx1 state, and the chain's two new words. -/
def absorbSet (σ : ChainSt) (p : Fin 32 → BitVec 32) : ChainSt :=
  let st := fx1Absorb σ.state p
  { σ with state := st, words := σ.words ++ [xinFinal fun i => st (Fin.castLE (by omega) i),
      xinFinal fun i => st ⟨8 + i.val, by omega⟩] }

/-- **Segment `s` on a lane's noise** (`h1tile._row`, `_col`, `_chain`'s prefix sets). -/
def segStep (N : LaneNoise) (s : ℕ) (a : Fin 4 → Fin 32 → BitVec 8) (b : Fin 8 → Fin 32 → BitVec 8) (σ : ChainSt) :
    ChainSt :=
  let x : Fin 4 → Fin 32 → BitVec 8 := fun r ρ => i8AddBit (a r ρ) (N.dither r (32 * s + ρ.val))
  let w : Fin 8 → Fin 32 → BitVec 8 := fun t ρ => i8Add (i8AddBit (b t ρ) (N.coin t (32 * s + ρ.val))) (N.u t ρ)
  let t' : Fin 32 → BitVec 32 := fun e => segDot (σ.t e) (x (rowSlot e)) (w (colSlot e))
  absorbSet { σ with t := t', res := fun r ρ => resAdd (σ.res r ρ) (x r ρ) } t'

/-- **The end on a lane's noise**: the row slots' digits, then the `O′` and `D_1 .. D_(m-1)` sets absorbed. -/
def finSets (m : ℕ) (N : LaneNoise) (σ : ChainSt) : ChainSt :=
  let A : Fin 4 → ℕ → Fin 32 → BitVec 8 := fun r d ρ => (digits m (fun d => N.digit r d ρ) (σ.res r ρ)).getD d 0
  let op : Fin 32 → BitVec 32 := fun e => segDot (σ.t e) (fun ρ => i8Neg (N.u (colSlot e) ρ)) (A (rowSlot e) 0)
  let D : ℕ → Fin 32 → BitVec 32 := fun d e => segDot 0 (N.u (colSlot e)) (A (rowSlot e) d)
  ((List.range (m - 1)).map (· + 1)).foldl (fun σ d => absorbSet σ (D d)) (absorbSet σ op)

/-- **The chain on a lane's noise**: from its fold keys with zero T and Y, its segments, and `m` digit sets at the end. -/
def noiseChain {P : Type} (nw m : ℕ) (N : P → Fin 32 → LaneNoise) : ChainSteps P nw where
  init x l := ⟨(N x l).key, 0, 0, []⟩
  seg x l s a b σ := segStep (N x l) s a b σ
  fin x l σ w := (finSets m (N x l) σ).words.getD w.val 0

/-- **`h1tile._chain` at contraction `K`**: the chain on `laneNoise K`, with `m = digitsK K`. -/
def h1Chain (K nw : ℕ) : ChainSteps SeedIn nw := noiseChain nw (digitsK K) (laneNoise K)

end Pouw.H1Zy
