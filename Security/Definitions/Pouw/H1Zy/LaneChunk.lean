import Definitions.Pouw.H1Zy.Lane

/-!
# `PouwH1LaneZyChunk_v1{TILE, CHUNKS, CHUNK}` on words, at a 32 × 32 leaf over `PouwH1Tile_v1` (trusted)

A lane past `K = 4096` is proved as `n` chunk statements along `K` (`h1zy.lane_statements`): chunk `c` reads the lane's
4 A rows and 8 B rows at entries `[c K_c, (c + 1) K_c)`, `K_c = K / n`, each with its row's scale, and hands what it
computed to chunk `c + 1` in a committed carry row (`h1zy.CARRY`), which the next chunk reads (`h1zy._chunk_body`). On
its primitives' word views:

* **The chain.** The lane's fx1 chain (`h1tile._chain`) runs over its `KS = K / 32` prefix sets, segment by segment,
  then its `O′` set and its `m - 1` digit sets. It is stated as a step machine (`ChainSteps`, a parameter, as `Lane`'s
  `chain` is): its state (`ChainSt`, what `CARRY` holds of it: the fx1 state, the entries' prefix sums T, the row
  slots' residue sums Y, and every chain word so far) starts at `init` (chunk 0's seeds and fold key), takes one `seg`
  per segment at the segment's index in the whole `K` (its rows' dither and columns' coin and U noise at the lane's
  full-`K` layout, `_noise` at offset `c K_c`; `PouwI8AddBit`, `PouwSx8` or `PouwResAdd`, `PouwSegDot0` or `PouwSegDot`,
  `PouwFx1Absorb` and two `PouwXinFinal` words), and ends with `fin`: the `O′` and digit sets from the state and the
  lane's digit coins and U, and the words of every set. So the whole lane's chain words are `chainOf`.
* **z.** Each of the lane's 32 cells (row slot `s`, column slot `t`) runs the chunk's `K_c / 32` steps of
  `BlackwellS8ImmaDot32_v1` (`immaS8`) on its codes, from zero at chunk 0 (`GemmAccumulatorS8_v2{K_c}`) and from the
  carried z after it (`zRun`).
* **The checks.** `ok` is the `PouwAndBit` tree of these, sent to `0` by a high index bit as the lane's is:
  * past chunk 0, the rows' 12 scales are the carried ones (`PouwEq32`);
  * before the last chunk, the carry out is what the chunk computed (`PouwEq32` over its words), its chain state, its
    cells' z and its rows' scales;
  * at the last chunk, the lane's checks on the final state: its cells pass `PouwZyCheck_v1{4, 8}` on the record's cells
    `sel` picks, at the final z and the rows' scales, and the chain's words `fin` gives, masked by the admit bits' OR, are
    the committed partial.

A chunk's inputs (`ChunkIn`) are all of its signature's at any `c`; it reads `carryIn` only past chunk 0, `carryOut` only
before the last chunk, and `admit`, the record and `partials` only at the last, as `_chunk_signature` has them. The carry's
words are a list: honestly `2 (c + 1) K_c / 32` of them, the count `CARRY` gives, so the comparison is `_chunk_body`'s
word by word. At `n = 1` the one chunk is the whole lane (`laneOk` at `chainOf`).
-/

namespace Pouw.H1Zy

/-- **What a lane's chain carries from segment to segment** (`h1zy.CARRY` less z and the scales): the fx1 state (16 u32),
each of the lane's 32 entries' prefix sum T, each row slot's residue sums Y (4 × 32 u32), and every chain word so far. -/
structure ChainSt where
  state : Fin 16 → BitVec 32
  t : Fin 32 → BitVec 32
  res : Fin 4 → Fin 32 → BitVec 32
  words : List (BitVec 32)
  deriving DecidableEq

/-- **The lane's fx1 chain as a step machine**, over the public seed inputs `P` (salt, position, roots) and the lane:
`init` its state before segment 0, `seg s` segment `s` of the whole `K` on the segment's 32 codes of each A and B row, and
`fin` the partial's `nw` words from the state after the last segment. -/
structure ChainSteps (P : Type) (nw : ℕ) where
  init : P → Fin 32 → ChainSt
  seg : P → Fin 32 → ℕ → (Fin 4 → Fin 32 → BitVec 8) → (Fin 8 → Fin 32 → BitVec 8) → ChainSt → ChainSt
  fin : P → Fin 32 → ChainSt → Fin nw → BitVec 32

/-- Segment `j`'s 32 codes of a row of `K` codes, `0` past its end. -/
def segOf {K : ℕ} (x : Fin K → BitVec 8) (j : ℕ) : Fin 32 → BitVec 8 := fun i => codeAt x (32 * j + i.val)

/-- The chain's state after `j` segments of a chunk whose first segment is segment `base` of the whole `K`, from `σ`. -/
def ChainSteps.run {P : Type} {nw K : ℕ} (M : ChainSteps P nw) (p : P) (l : Fin 32) (base : ℕ)
    (a : Fin 4 → Fin K → BitVec 8) (b : Fin 8 → Fin K → BitVec 8) (σ : ChainSt) : ℕ → ChainSt
  | 0 => σ
  | j + 1 => M.seg p l (base + j) (fun s => segOf (a s) j) (fun t => segOf (b t) j) (M.run p l base a b σ j)

/-- **The whole lane's chain words**: `fin` after its `K / 32` segments from `init`. -/
def chainOf {P : Type} {nw K : ℕ} (M : ChainSteps P nw) (p : P) (l : Fin 32) (a : Fin 4 → Fin K → BitVec 8)
    (b : Fin 8 → Fin K → BitVec 8) : Fin nw → BitVec 32 :=
  M.fin p l (M.run p l 0 a b (M.init p l) (K / 32))

/-- A cell's accumulator after `j` more `BlackwellS8ImmaDot32_v1` steps on two rows' codes, from `z`. -/
def zRun {K : ℕ} (x w : Fin K → BitVec 8) (z : BitVec 32) : ℕ → BitVec 32
  | 0 => z
  | j + 1 => immaS8 (zRun x w z j) (segOf x j) (segOf w j)

/-- **`h1zy.CARRY`**: the chain's state, each cell's z (row slot, then column slot) and the rows' scales. -/
structure Carry where
  chain : ChainSt
  z : Fin 4 → Fin 8 → BitVec 32
  sa : Fin 4 → BitVec 32
  sw : Fin 8 → BitVec 32
  deriving DecidableEq

/-- **A chunk's inputs on words** (`_chunk_signature` at a 32 × 32 leaf, `K_c` entries a row): its A and B rows' codes
and scales, the carry in and out, the leaf's admit bits and record, its partials row, the public seed inputs and its
index's 16 bits. -/
structure ChunkIn (Kc nw : ℕ) (P : Type) where
  a : Fin 4 → Fin Kc → BitVec 8
  sa : Fin 4 → BitVec 32
  b : Fin 8 → Fin Kc → BitVec 8
  sw : Fin 8 → BitVec 32
  carryIn : Carry
  carryOut : Carry
  admit : Fin 32 → Bool
  zr : Fin 32 → Fin 32 → BitVec 32
  yr : Fin 32 → Fin 32 → BitVec 16
  partials : Fin nw → BitVec 32
  pub : P
  lane : Fin 16 → Bool

/-- The index's selector bits, its low 5. -/
def ChunkIn.sel {Kc nw : ℕ} {P : Type} (x : ChunkIn Kc nw P) : List Bool :=
  List.ofFn fun t : Fin 5 => x.lane (Fin.castLE (by omega) t)

/-- The lane the selector names. -/
def ChunkIn.l {Kc nw : ℕ} {P : Type} (x : ChunkIn Kc nw P) : Fin 32 :=
  ⟨bitsVal x.sel, by simpa [ChunkIn.sel] using bitsVal_lt x.sel⟩

/-- **What chunk `c` computes**: from `init` and zero z at chunk 0, else from the carry in, its `K_c / 32` segments at
segments `c K_c / 32 ..` of the whole `K`, with its rows' scales. -/
def chunkCarry {Kc nw : ℕ} {P : Type} (M : ChainSteps P nw) (c : ℕ) (x : ChunkIn Kc nw P) : Carry where
  chain := M.run x.pub x.l (c * (Kc / 32)) x.a x.b (if c = 0 then M.init x.pub x.l else x.carryIn.chain) (Kc / 32)
  z := fun s t => zRun (x.a s) (x.b t) (if c = 0 then 0 else x.carryIn.z s t) (Kc / 32)
  sa := x.sa
  sw := x.sw

/-- **`PouwH1LaneZyChunk_v1{PouwH1Tile_v1, n, c}`'s `ok`** at a 32 × 32 leaf (`h1zy._chunk_body`), at the dequant `dqW`
and the chain machine `M`. -/
def chunkOk {Kc nw : ℕ} {P : Type} (M : ChainSteps P nw) (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16)
    (n c : ℕ) (x : ChunkIn Kc nw P) : Bool :=
  let mine := chunkCarry M c x
  let scales := if c = 0 then [] else
    (List.ofFn fun s => eq32 (x.sa s) (x.carryIn.sa s)) ++ List.ofFn fun t => eq32 (x.sw t) (x.carryIn.sw t)
  let picked : Fin 4 → Fin 8 → BitVec 32 × BitVec 16 := fun s t =>
    pick (0, 0) x.sel (List.ofFn fun ln : Fin 32 => (x.zr (laneRow ln s) (laneCol ln t), x.yr (laneRow ln s) (laneCol ln t)))
  let live := tree orBit false (List.ofFn x.admit)
  let tail := if c + 1 = n then
      zyCheck dqW (fun s t => (picked s t).1) (fun s t => (picked s t).2) mine.z x.sa x.sw ::
        List.ofFn fun w => eq32 (x.partials w) (mask32 (M.fin x.pub x.l mine.chain w) live)
    else [decide (x.carryOut = mine)]
  let high := tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩)
  mux high (tree andBit true (scales ++ tail)) false

end Pouw.H1Zy
