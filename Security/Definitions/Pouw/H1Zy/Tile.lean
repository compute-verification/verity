import Definitions.Pouw.H1Zy.Lane

/-!
# `PouwH1TileZy_v1{TILE}` on words, over `PouwH1Tile_v1` or `PouwH1Tile_v3` (trusted)

The H1 tile unit with its z/y record checked (`verity/core/protocols/pouw/circuit/h1zy.py`, `PouwH1TileZy`, `_body`)
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

namespace Pouw.H1Zy

/-! ### The tile's inputs and noise -/

/-- **The tile unit's inputs on words** (`PouwH1TileZy_v1`'s signature): its `R` A rows' codes and scales, its `C` B
rows', its record (cell `(i, j)`'s z and y) and its public inputs (salt, position and roots, of type `P`). -/
structure TileIn (R C K : ℕ) (P : Type) where
  a : Fin R → Fin K → BitVec 8
  sa : Fin R → BitVec 32
  b : Fin C → Fin K → BitVec 8
  sw : Fin C → BitVec 32
  zr : Fin R → Fin C → BitVec 32
  yr : Fin R → Fin C → BitVec 16
  pub : P

/-- **The v3 tile's noise** (`h1tile.noise`, its blocks of seed_j's stream): row `i`'s dither bits `bd i` (one per
entry) and digit coins `zc i d ρ` (digit `d`'s 32 bits, residue `ρ`'s), column `j`'s coin bits `cc j` (one per entry)
and its 32 noise bytes `ub j`, of which `U` is `PouwU129`'s. -/
structure TileNoise (R C K : ℕ) where
  bd : Fin R → Fin K → Bool
  zc : Fin R → ℕ → Fin 32 → Bool
  cc : Fin C → Fin K → Bool
  ub : Fin C → Fin 32 → BitVec 8

/-- Bit `k` of a row of `K` bits, `false` past its end. -/
def bitAt {K : ℕ} (x : Fin K → Bool) (k : ℕ) : Bool := if h : k < K then x ⟨k, h⟩ else false

/-- Segment `s` of an entry-indexed row: entries `32 s .. 32 s + 31`. -/
def seg {α : Type} (f : ℕ → α) (s : ℕ) : Fin 32 → α := fun e => f (32 * s + e.val)

/-- A fold over segments: `f s` after segment `s`'s predecessors, from `init`. -/
def segFold (f : ℕ → BitVec 32 → BitVec 32) (init : BitVec 32) : ℕ → BitVec 32
  | 0 => init
  | n + 1 => f n (segFold f init n)

/-! ### The primitives' word views -/

/-- `PouwI8AddBit_v1`: `b + d mod 2^8`. -/
def i8AddBit (b : BitVec 8) (d : Bool) : BitVec 8 := b + if d then 1 else 0

/-- `PouwI8Add_v1`: `a + u mod 2^8`. -/
def i8Add (a u : BitVec 8) : BitVec 8 := a + u

/-- `PouwI8Neg_v1`: `-x mod 2^8`. -/
def i8Neg (x : BitVec 8) : BitVec 8 := -x

/-- `PouwU129_v1`: a noise byte as H1's keyed word, `-64 + x mod 129` as an int8. -/
def u129 (x : BitVec 8) : BitVec 8 := BitVec.ofInt 8 (-64 + ((x.toNat % 129 : ℕ) : ℤ))

/-- `PouwSegDot_v1`: `acc + Σ_k a_k b_k mod 2^32`, `a` and `b` int8. `PouwSegDot0_v1` is it from the zero word. -/
def segDot (acc : BitVec 32) (a b : Fin 32 → BitVec 8) : BitVec 32 :=
  acc + BitVec.ofInt 32 (∑ k, (a k).toInt * (b k).toInt)

/-- `PouwSegDot0_v1`: `Σ_k a_k b_k mod 2^32`. -/
def segDot0 (a b : Fin 32 → BitVec 8) : BitVec 32 := segDot 0 a b

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

/-- `one_stream` (`PouwOneStream2_v1`, `PouwOneStream3_v1` at one and two words): `O′ - Σ_t 128^(t+1) D_t mod 2^32`. -/
def oneStream {n : ℕ} (op : BitVec 32) (d : Fin n → BitVec 32) : BitVec 32 :=
  op - ∑ t, d t <<< (7 * (t.val + 1))

/-- `PouwWrapBits_v1`: `W′ = B + c + U`'s int8 wraps, `(wb⁺, wb⁻)`: whether `B + c + U > 127`, whether it is
`< -128`. -/
def wrapBits (b : BitVec 8) (c : Bool) (u : BitVec 8) : Bool × Bool :=
  let s := b.toInt + (if c then 1 else 0) + u.toInt
  (decide (127 < s), decide (s < -128))

/-- `PouwWrapSeg_v1`: `acc + 256 Σ_e x_e (wb⁺_e - wb⁻_e) mod 2^32`, `x` int8. -/
def wrapSeg (acc : BitVec 32) (x : Fin 32 → BitVec 8) (wb : Fin 32 → Bool × Bool) : BitVec 32 :=
  acc + BitVec.ofInt 32 (256 * ∑ e, (x e).toInt * ((if (wb e).1 then 1 else 0) - (if (wb e).2 then 1 else 0)))

/-- `PouwXWrap_v1`: `X = A + b`'s int8 wrap, `A = 127` and `b`. -/
def xWrap (a : BitVec 8) (b : Bool) : Bool := decide (a = 127) && b

/-- `PouwPeelSeg_v1`: `acc - Σ_e (c_e x_e + bd_e w_e - 256 xa_e w_e) mod 2^32`, `x` and `w` int8. -/
def peelSeg (acc : BitVec 32) (x : Fin 32 → BitVec 8) (bd xa : Fin 32 → Bool) (w : Fin 32 → BitVec 8)
    (c : Fin 32 → Bool) : BitVec 32 :=
  acc - BitVec.ofInt 32 (∑ e, ((x e).toInt * (if c e then 1 else 0) +
    (w e).toInt * ((if bd e then 1 else 0) - 256 * (if xa e then 1 else 0))))

/-! ### The digits -/

/-- **H1's digit count** `m` at `KS` segments (`segment.n_digits`, `PouwWindow.Lanes.nDigits`): the least `m ≥ 1` with
`-128^m ≤ -128 KS`, that is `KS ≤ 128^(m-1)` (which gives its second condition, `127 KS / 128^(m-1) ≤ 127`). -/
def nDigits (ks : ℕ) : ℕ := go ks 1 1
where
  /-- `p = 128^(m-1)`; the fuel bounds the search, since `m ≤ KS + 1`. -/
  go : ℕ → ℕ → ℕ → ℕ
    | 0, m, _ => m
    | fuel + 1, m, p => if ks ≤ p then m else go fuel (m + 1) (128 * p)

/-- **`h1tile._digits`' digits of one residue sum** `r`, from digit `d` on with `n` coin digits left, at the coins `z`:
`PouwDigitLo` of the rest at the coin, the rest `PouwDigitHi` of it; the last coin digit's coin through `PouwDigitZ`;
the top digit `PouwLow8` of the last rest. -/
def digitsFrom (z : ℕ → Bool) : ℕ → ℕ → BitVec 32 → List (BitVec 8)
  | 0, _, r => [low8 r]
  | 1, d, r => [digitLo r (digitZ r (z d)), low8 (digitHi r (digitZ r (z d)))]
  | n + 2, d, r => digitLo r (z d) :: digitsFrom z (n + 1) (d + 1) (digitHi r (z d))

/-- A residue sum's `m` digits `A_0 .. A_(m-1)` at the coins `z`. -/
def digits (m : ℕ) (z : ℕ → Bool) (r : BitVec 32) : List (BitVec 8) := digitsFrom z (m - 1) 0 r

/-- **A residue sum** `Y_ρ` of an entry-indexed row after its segments `0 .. n`: `PouwSx8` of segment 0's entry `ρ`,
then `PouwResAdd` of each later segment's. -/
def resSum (x : ℕ → BitVec 8) (ρ : Fin 32) : ℕ → BitVec 32
  | 0 => sx8 (x ρ.val)
  | n + 1 => resAdd (resSum x ρ n) (x (32 * (n + 1) + ρ.val))

/-! ### The tile's z -/

/-- Row `i`'s `X = A + b` at entry `k` (`PouwI8AddBit`). -/
def xOf {R C K : ℕ} {P : Type} (n : TileNoise R C K) (x : TileIn R C K P) (i : Fin R) (k : ℕ) : BitVec 8 :=
  i8AddBit (codeAt (x.a i) k) (bitAt (n.bd i) k)

/-- Column `j`'s `U`, residue `ρ`'s (`PouwU129`). -/
def uOf {R C K : ℕ} (n : TileNoise R C K) (j : Fin C) (ρ : Fin 32) : BitVec 8 := u129 (n.ub j ρ)

/-- Column `j`'s `W′ = B + c + U` at entry `k` (`PouwI8AddBit`, then `PouwI8Add` with `U` repeated every 32 entries). -/
def wOf {R C K : ℕ} {P : Type} (n : TileNoise R C K) (x : TileIn R C K P) (j : Fin C) (k : ℕ) : BitVec 8 :=
  i8Add (i8AddBit (codeAt (x.b j) k) (bitAt (n.cc j) k)) (uOf n j ⟨k % 32, Nat.mod_lt _ (by omega)⟩)

/-- Row `i`'s digit `d` at residue `ρ` (`h1tile._digits` of its residue sums over its `K / 32` segments). -/
def digitOf {R C K : ℕ} {P : Type} (n : TileNoise R C K) (x : TileIn R C K P) (i : Fin R) (d : ℕ) (ρ : Fin 32) :
    BitVec 8 :=
  (digits (nDigits (K / 32)) (fun d' => n.zc i d' ρ) (resSum (xOf n x i) ρ (K / 32 - 1))).getD d 0

/-- **`PouwH1Tile_v3`'s z of cell `(i, j)`** (`h1tile._cell_z` on `_chain`'s words) at the noise `n`: the prefix word
`T` over the `K / 32` segments of `X` and `W′`, `O′` and `D_1 .. D_(m-1)` against the row's digits, `oneStream`, then
the wrap correction and the dither peel over each segment. -/
def streamZ {R C K : ℕ} {P : Type} (n : TileNoise R C K) (x : TileIn R C K P) (i : Fin R) (j : Fin C) : BitVec 32 :=
  let T := segFold (fun s acc => segDot acc (seg (xOf n x i) s) (seg (wOf n x j) s)) 0 (K / 32)
  let op := segDot T (fun ρ => i8Neg (uOf n j ρ)) (digitOf n x i 0)
  let D : Fin (nDigits (K / 32) - 1) → BitVec 32 := fun t => segDot0 (uOf n j) (digitOf n x i (t.val + 1))
  let wb : ℕ → Bool × Bool := fun k =>
    wrapBits (codeAt (x.b j) k) (bitAt (n.cc j) k) (uOf n j ⟨k % 32, Nat.mod_lt _ (by omega)⟩)
  let xa : ℕ → Bool := fun k => xWrap (codeAt (x.a i) k) (bitAt (n.bd i) k)
  let zw := segFold (fun s acc => wrapSeg acc (seg (xOf n x i) s) (seg wb s)) (oneStream op D) (K / 32)
  segFold (fun s acc => peelSeg acc (seg (xOf n x i) s) (seg (bitAt (n.bd i)) s) (seg xa s)
    (seg (codeAt (x.b j)) s) (seg (bitAt (n.cc j)) s)) zw (K / 32)

/-- **TILE**: `PouwH1Tile_v1`, whose z is EXACT, or `PouwH1Tile_v3` at its noise, a function of the public inputs. -/
inductive TileForm (R C K : ℕ) (P : Type) where
  | exact : TileForm R C K P
  | stream (noise : P → TileNoise R C K) : TileForm R C K P

/-- **TILE's z** of cell `(i, j)`: v1's `GemmAccumulatorS8_v2{K, sm120}` on row `i`'s and column `j`'s codes, or v3's
`streamZ` at the noise of the public inputs. -/
def TileForm.z {R C K : ℕ} {P : Type} : TileForm R C K P → TileIn R C K P → Fin R → Fin C → BitVec 32
  | .exact, x, i, j => gemmAccS8 (x.a i) (x.b j)
  | .stream noise, x, i, j => streamZ (noise x.pub) x i j

/-- **`PouwH1TileZy_v1{TILE}`'s `ok`** (`h1zy._body`) at the dequant `dqW`: `PouwZyCheck_v1{R, C}` of the record against
TILE's z at the A rows' and B rows' scales. -/
def tileOk {R C K : ℕ} {P : Type} (form : TileForm R C K P) (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16)
    (x : TileIn R C K P) : Bool :=
  zyCheck dqW x.zr x.yr (form.z x) x.sa x.sw

end Pouw.H1Zy
