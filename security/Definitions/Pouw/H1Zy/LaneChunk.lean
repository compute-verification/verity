module

public import Definitions.Pouw.H1Zy.Lane

/-! `PouwH1LaneZyChunk_v1{TILE, CHUNKS, CHUNK}` on words: a lane past `K = 4096` as chunk statements linked by a committed carry, whose one chunk at `n = 1` is the whole lane. -/

@[expose] public section

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

def zRun {K : ℕ} (x w : Fin K → BitVec 8) (z : BitVec 32) : ℕ → BitVec 32
  | 0 => z
  | j + 1 => immaS8 (zRun x w z j) (segOf x j) (segOf w j)

structure Carry where
  chain : ChainSt
  z : Fin 4 → Fin 8 → BitVec 32
  sa : Fin 4 → BitVec 32
  sw : Fin 8 → BitVec 32
  deriving DecidableEq

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

def ChunkIn.sel {Kc nw : ℕ} {P : Type} (x : ChunkIn Kc nw P) : List Bool :=
  List.ofFn fun t : Fin 5 => x.lane (Fin.castLE (by omega) t)

def ChunkIn.l {Kc nw : ℕ} {P : Type} (x : ChunkIn Kc nw P) : Fin 32 :=
  ⟨bitsVal x.sel, by simpa [ChunkIn.sel] using bitsVal_lt x.sel⟩

def chunkCarry {Kc nw : ℕ} {P : Type} (M : ChainSteps P nw) (c : ℕ) (x : ChunkIn Kc nw P) : Carry where
  chain := M.run x.pub x.l (c * (Kc / 32)) x.a x.b (if c = 0 then M.init x.pub x.l else x.carryIn.chain) (Kc / 32)
  z := fun s t => zRun (x.a s) (x.b t) (if c = 0 then 0 else x.carryIn.z s t) (Kc / 32)
  sa := x.sa
  sw := x.sw

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
