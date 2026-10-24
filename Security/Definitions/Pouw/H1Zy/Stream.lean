import Definitions.Pouw.H1Zy.Chain

/-!
# `PouwH1Tile_v3`'s z of a cell on words (trusted)

`h1tile.reference(stream=True)`'s z (`verity/core/protocols/pouw/circuit/h1tile.py`, `_cell_z`, `stream_cell`): cell
`(i, j)` of a 32 × 32 tile at contraction `K` from row `i`'s codes `a`, its dither bits `bd` and its digit coins `zc`
(`zc d ρ`, bit `ρ` of coin `d`), and column `j`'s codes `w`, its coin bits `c` and its 32 noise bytes `ub` (U is `u129`
of them), entry `t` of each at `t`:

* the cell's O' and D words, entry `e` of its lane's sets (`callWords`), which read only the cell's row and column:
  the prefix `T` over `X = A + b` and `W' = B + c + U` segment by segment (`PouwSegDot`), the digits `A_d` of the
  residue sums `Y_ρ = Σ_s X[32 s + ρ]` (`digitsOf`), `O' = T + (-U) A_0` and `D_d = U A_d`;
* `PouwOneStream{m}` of them (`oneStream`: `O' - Σ_d 128^d D_d`), then `PouwWrapSeg` over each segment of `X` and the
  column's wrap bits (`wrapBits`), then `PouwPeelSeg` over each segment with the row's dither bits and `X` wrap bits
  (`xWrap`) and the column's codes and coin bits, all mod `2^32`.

The noise is arbitrary here: these are the words for any bits the seed's stream gives (`rowNoise`, `colNoise`).
-/

namespace Pouw.H1Zy

/-- `h1tile.one_stream`: `O' - Σ_d 128^d D_d mod 2^32`, `d = 1 .. m - 1`. -/
def oneStream (op : ℕ) (ds : List ℕ) : ℕ :=
  (((op : ℤ) - ((List.range ds.length).map fun t => (2 ^ (7 * (t + 1)) * ds.getD t 0 : ℤ)).sum) % 2 ^ 32).toNat

/-- `h1tile.wrap_bits`: `W' = B + c + U`'s int8 wrap at one entry, `(wb⁺, wb⁻)`: above `127`, below `-128`. -/
def wrapBits (b : ℕ) (c : Bool) (u : ℕ) : Bool × Bool :=
  let s := sx8 b + (if c then 1 else 0) + sx8 u
  (decide (127 < s), decide (s < -128))

/-- `h1tile.wrap_seg`: `acc + 256 Σ_e x_e (wb⁺_e - wb⁻_e) mod 2^32` over one segment. -/
def wrapSeg (acc : ℕ) (x : List ℕ) (wb : List (Bool × Bool)) : ℕ :=
  ((acc + 256 * (List.zipWith (fun (v : ℕ) (p : Bool × Bool) => sx8 v * ((if p.1 then 1 else 0) - (if p.2 then 1 else 0)))
    x wb).sum)
    % 2 ^ 32).toNat

/-- `h1tile.x_wrap`: `X = A + b`'s int8 wrap at one entry, `A = 127` with its dither bit set. -/
def xWrap (a : ℕ) (b : Bool) : Bool := decide (a % 256 = 127) && b

/-- One entry's term of `h1tile.peel_seg`: `c x + bd w - 256 xa w`. -/
def peelTerm (x : ℕ) (bd xa : Bool) (w : ℕ) (c : Bool) : ℤ :=
  sx8 x * (if c then 1 else 0) + sx8 w * ((if bd then 1 else 0) - 256 * (if xa then 1 else 0))

/-- `h1tile.peel_seg`: `acc - Σ_e (c_e x_e + bd_e w_e - 256 xa_e w_e) mod 2^32` over one segment. -/
def peelSeg (acc : ℕ) (x : List ℕ) (bd xa : List Bool) (w : List ℕ) (c : List Bool) : ℕ :=
  ((acc - ((List.range x.length).map fun e =>
    peelTerm (x.getD e 0) (bd.getD e false) (xa.getD e false) (w.getD e 0) (c.getD e false)).sum) % 2 ^ 32).toNat

/-- Segment `s` of an entry function: entries `32 s .. 32 s + 31`. -/
def segOf {α : Type} (f : ℕ → α) (s : ℕ) : List α := (List.range 32).map fun e => f (32 * s + e)

/-- **`PouwH1Tile_v3`'s z of a cell** (`h1tile.stream_cell` of its lane's O' and D words): from its row's codes `a`,
dither bits `bd` and digit coins `zc`, and its column's codes `w`, coin bits `c` and noise bytes `ub`. -/
def streamZ (K : ℕ) (a : ℕ → ℕ) (bd : ℕ → Bool) (zc : ℕ → ℕ → Bool) (w : ℕ → ℕ) (c : ℕ → Bool) (ub : ℕ → ℕ) : ℕ :=
  let L := lay K
  let u : ℕ → ℕ := fun ρ => u129 (ub ρ)
  let X : ℕ → ℕ := fun t => (a t + if bd t then 1 else 0) % 256
  let W : ℕ → ℕ := fun t => (w t + (if c t then 1 else 0) + u (t % 32)) % 256
  let T := (List.range L.ks).foldl (fun acc s => segDot acc (segOf X s) (segOf W s)) 0
  let A : ℕ → List ℕ := fun ρ =>
    digitsOf L.m ((List.range L.ks).map fun s => sx8 (X (32 * s + ρ))).sum ((List.range (L.m - 1)).map fun d => zc d ρ)
  let digit : ℕ → List ℕ := fun d => (List.range 32).map fun ρ => (A ρ).getD d 0
  let op := segDot T ((List.range 32).map fun ρ => (256 - u ρ % 256) % 256) (digit 0)
  let z := oneStream op ((List.range (L.m - 1)).map fun d => segDot 0 ((List.range 32).map u) (digit (d + 1)))
  let z := (List.range L.ks).foldl (fun acc s =>
    wrapSeg acc (segOf X s) (segOf (fun t => wrapBits (w t) (c t) (u (t % 32))) s)) z
  (List.range L.ks).foldl (fun acc s =>
    peelSeg acc (segOf X s) (segOf bd s) (segOf (fun t => xWrap (a t) (bd t)) s) (segOf w s) (segOf c s)) z

/-- **`PouwH1Tile_v3`'s z of lane `l`'s cell `(s, t)`** at a 32 × 32 leaf (`h1zy._lane_body`'s `H._cell_z`): `streamZ`
on its A row's and B row's codes, with row `laneRow l s`'s dither bits and digit coins (`rowNoise`) and column
`laneCol l t`'s coin bits and noise bytes (`colNoise`) from the leaf's seed_j stream at the salt, position and roots. -/
def laneZ3 {K : ℕ} (salt pos roots : ℕ) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8) (b : Fin 8 → Fin K → BitVec 8)
    (s : Fin 4) (t : Fin 8) : BitVec 32 :=
  let L := lay K
  let seedJ := tagged seedTag (portBytes 16 salt) (portBytes 2 pos) (portBytes 32 roots)
  let rs := streamFrom seedJ ((laneRow l s).val * L.rb) L.rowBits
  let cs := streamFrom seedJ (32 * L.rb + (laneCol l t).val * L.cb) L.colBits
  BitVec.ofNat 32 (streamZ K (fun k => (codeAt (a s) k).toNat) (fun k => rs.getD k false)
    (fun d ρ => rs.getD (L.k + 32 * d + ρ) false) (fun k => (codeAt (b t) k).toNat) (fun k => cs.getD k false)
    (fun ρ => bitsNat ((cs.drop (L.k + 8 * ρ)).take 8)))

/-- **`PouwH1LaneZy_v1{TILE}`'s `ok`** at a 32 × 32 leaf, `laneOk` with its cells' z the tile's `zf` (of the public
inputs, the lane `sel` names and the rows' codes): at `fun _ _ a b s t => gemmAccS8 (a s) (b t)` it is
`laneOk` (`PouwH1Tile_v1`), at `laneZ3` of the public inputs' salt, position and roots `PouwH1Tile_v3`'s. -/
def laneOkZ {K nw : ℕ} {P : Type}
    (zf : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin 4 → Fin 8 → BitVec 32)
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : LaneIn K nw P) : Bool :=
  let picked : Fin 4 → Fin 8 → BitVec 32 × BitVec 16 := fun s t =>
    pick (0, 0) x.sel (List.ofFn fun ln : Fin 32 => (x.zr (laneRow ln s) (laneCol ln t), x.yr (laneRow ln s) (laneCol ln t)))
  let ok := zyCheck dqW (fun s t => (picked s t).1) (fun s t => (picked s t).2) (zf x.pub x.l x.a x.b) x.sa x.sw
  let live := tree orBit false (List.ofFn x.admit)
  let eq := List.ofFn fun w => eq32 (x.partials w) (mask32 (chain x.pub x.l x.a x.b w) live)
  let high := tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩)
  mux high (tree andBit true (ok :: eq)) false

end Pouw.H1Zy
