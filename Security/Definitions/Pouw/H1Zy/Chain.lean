import Definitions.Pouw.H1Zy.Lane
import Definitions.Pouw.PearlC.Sha512

/-!
# A lane's chain words (trusted)

`h1zy.lane_partial`'s fold, ungated (`verity/core/protocols/pouw/circuit/h1zy.py`, `h1tile.py`, `seeds.py`,
`segment.py`): lane `l` of a 32 × 32 H1 leaf at contraction `K`, from its 4 A rows and 8 B rows by slot (s8 encodings),
and the leaf's salt, position and roots. Everything is on naturals and lists, with SHA-512 the spec's own
(`Pouw.PearlC.Sha512.digest`), so that the kernel evaluates it (`Proofs.Pouw.H1Zy.Vectors`).

* **The public inputs** are numbers as their port fields read them, least significant bit first: a digest's 16 halves,
  half `h` the bits `32 h .. 32 h + 31`, each half the big-endian word of its 4 bytes (`seeds.leaves`), and the position
  `_u64(position)`'s 2 halves alike (`portBytes`).
* **The seeds** (`seeds._tagged`): seed_j under `SEED_TAG`, σ_R under `RECEIPT_TAG`, each the SHA-512 of the position's
  8 bytes and the tag, salt and roots XORed byte by byte with them; noise block `t` of a seed is the SHA-512 of
  `NOISE_TAG`, the seed and `u64(t)`, read as its halves' bits, least significant first (`stream`).
* **The layout** (`h1tile.Layout(32, 32, K)`): `KS = K/32` segments, `m = n_digits(KS)` digits, a row's `K + 32 (m - 1)`
  noise bits (its dither bits, then its digit coins) from block `i rb`, a column's `K + 256` (its coin bits, then U's
  32 bytes, each `u129`) from block `32 rb + j cb`, lane `l`'s 16 fold keys block `l` of σ_R's stream.
* **The sets** (`segment.call_words`): the prefixes `T_0 .. T_(KS-1)` over `X = A + b` and `W' = B + c + U`, `O'` and
  `D_1 .. D_(m-1)` from the digits of the residue sums.
* **The fold** (`h1tile.fold`, fold `fx1`): from the keys, after each set the XOR of the state's y half and of its z
  half, `2 (KS + m)` words.
-/

namespace Pouw.H1Zy

open Pouw.PearlC.Sha512 (digest beBytes beNat ascii padTo)

/-! ### Words -/

/-- `segment.mix32`, fgemm.cu's mix32. -/
def mix32 (x : ℕ) : ℕ :=
  let x := x % 2 ^ 32
  let x := x ^^^ (x >>> 16)
  let x := x * 0x7FEB352D % 2 ^ 32
  let x := x ^^^ (x >>> 15)
  let x := x * 0x846CA68B % 2 ^ 32
  x ^^^ (x >>> 16)

/-- `segment.rotl` on a 32-bit word, `0 < r < 32`. -/
def rotl32 (x r : ℕ) : ℕ := ((x <<< r) ||| (x >>> (32 - r))) % 2 ^ 32

/-- `segment._sx8`: an int8 encoding's value. -/
def sx8 (v : ℕ) : ℤ := if v % 256 < 128 then (v % 256 : ℕ) else (v % 256 : ℕ) - 256

/-- `segment.seg_dot`: `acc` plus the s8 dot of `a` and `b`, mod `2^32`. -/
def segDot (acc : ℕ) (a b : List ℕ) : ℕ := ((acc + (List.zipWith (fun x w => sx8 x * sx8 w) a b).sum) % 2 ^ 32).toNat

/-- `segment.row_slot`: entry `q`'s row slot. -/
def rowSlot (q : ℕ) : ℕ := 2 * (q / 16) + q % 4 / 2

/-- `segment.col_slot`: entry `q`'s column slot. -/
def colSlot (q : ℕ) : ℕ := 2 * (q / 4 % 4) + q % 2

/-! ### The seeds and the noise stream -/

/-- A port field's number as its bytes: `n` halves, half `h` the bits `32 h .. 32 h + 31`, each its 4 big-endian
bytes. -/
def portBytes (n x : ℕ) : List ℕ := (List.range n).flatMap fun h => beBytes 4 (x >>> (32 * h) % 2 ^ 32)

/-- A tag zero-padded to 24 bytes. -/
def tag24 (s : String) : List ℕ := padTo 24 (ascii s)

/-- `seeds.SEED_TAG`. -/
def seedTag : List ℕ := tag24 "verity/pouw/seed/v2"

/-- `seeds.RECEIPT_TAG`. -/
def receiptTag : List ℕ := tag24 "verity/pouw/receipt/v1"

/-- `seeds.NOISE_TAG`. -/
def noiseTag : List ℕ := tag24 "verity/pouw/noise/v1"

/-- `seeds._tagged`: SHA-512 of the position's 8 bytes `p`, then the tag, salt and roots, byte `i` XORed with
`p[i mod 8]`. -/
def tagged (tg salt p roots : List ℕ) : List ℕ :=
  let data := tg ++ salt ++ roots
  digest (p ++ List.zipWith (· ^^^ ·) data ((List.range data.length).map fun i => p.getD (i % 8) 0))

/-- `seeds.noise_block`: block `t` of a seed's noise stream. -/
def noiseBlock (seed : List ℕ) (t : ℕ) : List ℕ := digest (noiseTag ++ seed ++ beBytes 8 t)

/-- `seeds.leaves(_, True)` of a 64-byte block: its 16 big-endian halves' bits, least significant first. -/
def blockBits (blk : List ℕ) : List Bool :=
  (List.range 16).flatMap fun h =>
    let w := beNat ((blk.drop (4 * h)).take 4)
    (List.range 32).map fun β => w.testBit β

/-- `h1tile._stream`: `n` bits of a seed's noise stream from block `block` on. -/
def streamFrom (seed : List ℕ) (block n : ℕ) : List Bool :=
  ((List.range ((n + 511) / 512)).flatMap fun t => blockBits (noiseBlock seed (block + t))).take n

/-! ### The layout -/

/-- `segment.n_digits` from `m` on, with `fuel` more tries: the least `m' ≥ m` with `128 KS ≤ 128^m'` and
`⌊127 KS / 128^(m'-1)⌋ ≤ 127`. -/
def nDigitsFrom (ks : ℕ) : ℕ → ℕ → ℕ
  | 0, m => m
  | fuel + 1, m => if 128 * ks ≤ 128 ^ m ∧ 127 * ks / 128 ^ (m - 1) ≤ 127 then m else nDigitsFrom ks fuel (m + 1)

/-- **`segment.n_digits`**: H1's digit count at `KS` segments (`m = KS + 1` always qualifies). -/
def nDigits (ks : ℕ) : ℕ := nDigitsFrom ks (ks + 1) 1

/-- `h1tile.Layout(32, 32, K)`'s numbers. -/
structure Lay where
  k : ℕ
  ks : ℕ
  m : ℕ
  rowBits : ℕ
  colBits : ℕ
  rb : ℕ
  cb : ℕ

/-- `h1tile.Layout(32, 32, K)`: `KS = K / 32` segments, `m` digits, a row's and a column's noise bits and blocks. -/
def lay (K : ℕ) : Lay :=
  let ks := K / 32
  let m := nDigits ks
  let rowBits := K + 32 * (m - 1)
  let colBits := K + 256
  ⟨K, ks, m, rowBits, colBits, (rowBits + 511) / 512, (colBits + 511) / 512⟩

/-- `h1tile.u129`: a noise byte as H1's keyed word, `-64 + x mod 129` as its int8 encoding. -/
def u129 (x : ℕ) : ℕ := (x % 256 % 129 + 192) % 256

/-- The number of bits, least significant first. -/
def bitsNat (bs : List Bool) : ℕ := bs.foldr (fun b n => (if b then 1 else 0) + 2 * n) 0

/-- `h1tile.noise`'s row `i`: its `K` dither bits and its `m - 1` digit coins (32 bits each). -/
def rowNoise (seedJ : List ℕ) (L : Lay) (i : ℕ) : List Bool × List (List Bool) :=
  let s := streamFrom seedJ (i * L.rb) L.rowBits
  (s.take L.k, (List.range (L.m - 1)).map fun d => (s.drop (L.k + 32 * d)).take 32)

/-- `h1tile.noise`'s column `j`: its `K` coin bits and U's 32 int8 encodings. -/
def colNoise (seedJ : List ℕ) (L : Lay) (j : ℕ) : List Bool × List ℕ :=
  let s := streamFrom seedJ (32 * L.rb + j * L.cb) L.colBits
  (s.take L.k, (List.range 32).map fun t => u129 (bitsNat ((s.drop (L.k + 8 * t)).take 8)))

/-- `h1tile.fold_keys`' lane `l`: block `l` of σ_R's stream as 16 words. -/
def foldKeys (sigmaR : List ℕ) (l : ℕ) : List ℕ :=
  let s := streamFrom sigmaR l 512
  (List.range 16).map fun w => bitsNat ((s.drop (32 * w)).take 32)

/-! ### The sets -/

/-- Segment `s` of a row: entries `32 s .. 32 s + 31`. -/
def seg {α : Type} (x : List α) (s : ℕ) : List α := (x.drop (32 * s)).take 32

/-- `segment.prefixes`: `T_0 .. T_(KS-1)` from `X` and `W'` by segment (`X[s][r]`, `W'[s][k]`, 32 entries each). -/
def prefixes (X W : List (List (List ℕ))) : List (List ℕ) :=
  ((List.zip X W).foldl (fun (acc, out) xw =>
    let acc := (List.range 32).map fun q =>
      segDot (acc.getD q 0) ((xw.1).getD (rowSlot q) []) ((xw.2).getD (colSlot q) [])
    (acc, out ++ [acc])) (List.replicate 32 0, [])).2

/-- `segment.digit_words`' one residue: from the sum `R` and its coins `z_0 .. z_(m-2)`, the digits `A_0 .. A_(m-1)`
(int8 encodings): `A_d = (R_d mod 128) + 128 z_d` and `R_(d+1) = ⌊R_d / 128⌋ + z_d`, `z_(m-2)` taken as `0` where
`⌊R / 128⌋ = 127`, and `A_(m-1) = R_(m-1) mod 256`. -/
def digitsOf (m : ℕ) (R : ℤ) (z : List Bool) : List ℕ :=
  let (R, out) := (List.range (m - 1)).foldl (fun (R, out) d =>
    let q := R / 128
    let zd : ℤ := if d = m - 2 ∧ q = 127 then 0 else if z.getD d false then 1 else 0
    (q + zd, out ++ [(R % 128 + 128 * zd).toNat])) (R, [])
  out ++ [(R % 256).toNat]

/-- **`segment.call_words`** at a lane: its sets `T_0 .. T_(KS-1)`, `O'`, `D_1 .. D_(m-1)` (32 words each), from its
rows' codes `a` (4, by row slot) and their dither bits `bd` and coins `zc` (`zc[r][d]`), and its columns' codes `w` (8,
by column slot) and their coin bits `c` and U's encodings `u`. -/
def callWords (L : Lay) (a : List (List ℕ)) (bd : List (List Bool)) (zc : List (List (List Bool)))
    (w : List (List ℕ)) (c : List (List Bool)) (u : List (List ℕ)) : List (List ℕ) :=
  let X := (List.range L.ks).map fun s => (List.range 4).map fun r =>
    List.zipWith (fun v b => (v + if b then 1 else 0) % 256) (seg (a.getD r []) s) (seg (bd.getD r []) s)
  let W := (List.range L.ks).map fun s => (List.range 8).map fun k =>
    List.zipWith (fun (v : ℕ) (cu : Bool × ℕ) => (v + (if cu.1 then 1 else 0) + cu.2) % 256) (seg (w.getD k []) s)
      (List.zip (seg (c.getD k []) s) (u.getD k []))
  let T := prefixes X W
  let A : List (List (List ℕ)) := (List.range 4).map fun r => (List.range 32).map fun ρ =>
    digitsOf L.m (X.map fun xs => sx8 ((xs.getD r []).getD ρ 0)).sum ((zc.getD r []).map fun zd => zd.getD ρ false)
  let digit (d r : ℕ) : List ℕ := (A.getD r []).map fun ds => ds.getD d 0
  let neg := u.map fun row => row.map fun v => (256 - v % 256) % 256
  let last := T.getLastD (List.replicate 32 0)
  let op := (List.range 32).map fun q => segDot (last.getD q 0) (neg.getD (colSlot q) []) (digit 0 (rowSlot q))
  T ++ [op] ++ (List.range (L.m - 1)).map fun d =>
    (List.range 32).map fun q => segDot 0 (u.getD (colSlot q) []) (digit (d + 1) (rowSlot q))

/-! ### The fold -/

/-- `segment.KA[t]`: duo's multiplier at position `t`. -/
def duoKA (t : ℕ) : ℕ := mix32 (0xA0000000 + 2 * t) ||| 1

/-- `segment.KK[t]`: duo's XOR key at position `t`. -/
def duoKK (t : ℕ) : ℕ := mix32 (0xB0000000 + 2 * t + 1)

/-- `segment.ROT[t]`: duo's rotation at position `t`. -/
def duoROT (t : ℕ) : ℕ := (7 * t + 5) % 31 + 1

/-- `segment.fx1_key`: fx1's multiplier of entry `q`. -/
def fx1Key (q : ℕ) : ℕ := mix32 (0xA0000000 + 2 * (16 + 32 * q)) ||| 1

/-- `segment._duo_absorb`: one set `p` (32 words) into the state `y ++ z` (8 words each), position `t` by position, each
of its 4 entries `e` into chain `4 (t mod 2) + e`. -/
def duoAbsorb (st p : List ℕ) : List ℕ :=
  let yz := ((List.range 8).flatMap fun t => (List.range 4).map fun e => (t, e)).foldl (fun (yz : List ℕ × List ℕ) te =>
    let (y, z) := yz
    let (t, e) := te
    let i := 4 * (t % 2) + e
    let u := y.getD i 0 ^^^ p.getD (4 * t + e) 0 % 2 ^ 32 ^^^ duoKK t
    let yi := (u * duoKA t + z.getD i 0) % 2 ^ 32
    (y.set i yi, z.set i ((rotl32 u (duoROT t) + yi) % 2 ^ 32)))
    ((st.take 8).map (· % 2 ^ 32), ((st.drop 8).take 8).map (· % 2 ^ 32))
  yz.1 ++ yz.2

/-- `segment._fx1_absorb`: duo's absorb of the set with entry `q` times `fx1Key q`. -/
def fx1Absorb (st p : List ℕ) : List ℕ :=
  duoAbsorb st ((List.range p.length).map fun q => p.getD q 0 * fx1Key q % 2 ^ 32)

/-- The XOR of a list of words. -/
def xorAll (ws : List ℕ) : ℕ := ws.foldl (· ^^^ ·) 0

/-- **`h1tile.fold`** (fold `fx1`): from the 16 keys, after each set the XOR of the state's words `0 .. 7` and of its
words `8 .. 15`. -/
def foldSets (key : List ℕ) (sets : List (List ℕ)) : List ℕ :=
  (sets.foldl (fun (st, out) p =>
    let st := fx1Absorb st p
    (st, out ++ [xorAll (st.take 8), xorAll ((st.drop 8).take 8)])) (key, [])).2

/-! ### The lane's partial -/

/-- **`h1zy.lane_partial`, ungated**: lane `l`'s fold at contraction `K` from its A rows `a` (by row slot) and B rows
`b` (by column slot), s8 encodings, and the salt, position and roots as their port fields' numbers. -/
def lanePartial (K salt pos roots l : ℕ) (a b : List (List ℕ)) : List ℕ :=
  let L := lay K
  let p := portBytes 2 pos
  let sr := portBytes 16 salt
  let rt := portBytes 32 roots
  let seedJ := tagged seedTag sr p rt
  let sigmaR := tagged receiptTag sr p rt
  let rows := (List.range 4).map fun s => rowNoise seedJ L (l / 4 + 8 * s)
  let cols := (List.range 8).map fun t => colNoise seedJ L (2 * (l % 4) + 8 * (t / 2) + t % 2)
  foldSets (foldKeys sigmaR l)
    (callWords L a (rows.map (·.1)) (rows.map (·.2)) b (cols.map (·.1)) (cols.map (·.2)))

/-- **The lane's chain words** (`LeafChecks`' `chain`): `lanePartial`'s word `w` at the rows' codes, `0` past its
`2 (KS + m)` words. -/
def chainWords {K nw : ℕ} (salt pos roots : ℕ) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8)
    (b : Fin 8 → Fin K → BitVec 8) (w : Fin nw) : BitVec 32 :=
  BitVec.ofNat 32 ((lanePartial K salt pos roots l.val (List.ofFn fun s => List.ofFn fun k => (a s k).toNat)
    (List.ofFn fun t => List.ofFn fun k => (b t k).toNat)).getD w.val 0)

end Pouw.H1Zy
