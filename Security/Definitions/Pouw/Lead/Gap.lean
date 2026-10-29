import Definitions.Pouw.Lead.Model

/-!
# The lead's honest work and its gap, as functions of the layout (trusted definitions)

Every quantity is in registers; the W1 work is 32 a register (`H32`'s floor).

* **Stock** (`stockRegs`): the call's unmodified FP8 matmul, `m · n · K / 32` k32 `wgmma` registers. Every share is
  of stock.
* **The honest work** (`honestRegs`): the committed words, one register each (`m · n · S`), and the SHA-512
  compressions of the commitment at ShaWrites' registers per form: `3,515.38` a words compression and `3,449.05` an A
  compression (`shaWordsRegs`, `shaARegs`).
* **The gap** (`gapRegs`) is the sum of six terms, each the bound its piece proves:
  - `headGap`: the words chunks' first-block excess, 23 registers a chunk;
  - `hidingGap`: hiding L1 + (a), 1.6064% of one row's stock a call;
  - `coinBranchGap`: what a prover that branches on the coins keeps by copying whole MMA tiles across the call's
    rows and patching the rest (copy-and-patch at tile grain), 0.084% of stock;
  - `pStarGap`: the coin lanes' guess at `-rot`'s weights, 0.0035% of stock;
  - `networkGap`: work outsourced within the egress cap, 0.036% of stock a 32 bits of cap;
  - `tableGap`: pre-coin tables holding a committed tile, within the node's memory (negligible).
* `gapShare` is the gap over stock, which `Pouw.Guarantees.Lead.GapAt` evaluates at the three points.
-/

namespace Pouw.Lead

/-- Stock: the call's FP8 matmul in k32 `wgmma` registers, `m · n · K / 32`. -/
def stockRegs (L : Layout) : ℚ := (L.m * L.n * L.K : ℚ) / 32

/-- The committed words, one register each: `m · n · S`. -/
def wordsRegs (L : Layout) : ℚ := L.m * L.n * L.S

/-- ShaWrites' registers a words compression. -/
def shaWordsRegs : ℚ := 351538 / 100

/-- ShaWrites' registers an A compression. -/
def shaARegs : ℚ := 344905 / 100

/-- The words' bytes in 128-byte compressions, with one padding block a chunk: `4 m n S / 128 + chunks`. -/
def wordsBlocks (L : Layout) (C : Commit L) : ℚ := 4 * wordsRegs L / 128 + C.chunks

/-- A's FP8 rows in 128-byte compressions: `m · K / 128`. -/
def aBlocks (L : Layout) : ℚ := (L.m * L.K : ℚ) / 128

/-- **The honest work** in registers: the words, and the commitment's compressions at ShaWrites' registers. -/
def honestRegs (L : Layout) (C : Commit L) : ℚ :=
  wordsRegs L + shaWordsRegs * wordsBlocks L C + shaARegs * aBlocks L

/-- The words chunks' first-block excess: 23 registers a chunk. -/
def headGap (L : Layout) (C : Commit L) : ℚ := 23 * C.chunks

/-- Hiding L1 + (a): 1.6064% of one row's stock a call, `0.016064 · n · K / 32`. -/
def hidingGap (L : Layout) : ℚ := 16064 / 1000000 * ((L.n * L.K : ℚ) / 32)

/-- Copy-and-patch across the call's rows at tile grain: 0.084% of stock. -/
def coinBranchGap (L : Layout) : ℚ := 84 / 100000 * stockRegs L

/-- The coin lanes' guess under `-rot`'s weights: 0.0035% of stock. -/
def pStarGap (L : Layout) : ℚ := 35 / 1000000 * stockRegs L

/-- Work outsourced within the egress cap: 0.036% of stock a 32 bits of cap. -/
def networkGap (L : Layout) : ℚ := 36 / 100000 * (L.cap / 32 : ℚ) * stockRegs L

/-- Pre-coin tables and the coin stream holding a committed tile: twice the node's memory in tiles, times the call's
tiles and their words, at `2^-150`, a bound on the chance that a fixed tile is a given committed tile. -/
def tableGap (L : Layout) : ℚ := 2 * L.mem * (L.rowBlocks * L.colBlocks * L.S * tileWords : ℚ) / 2 ^ 150

/-- **The gap** in registers: the sum of the six terms. -/
def gapRegs (L : Layout) (C : Commit L) : ℚ :=
  headGap L C + hidingGap L + coinBranchGap L + pStarGap L + networkGap L + tableGap L

/-- The gap as a share of stock. -/
def gapShare (L : Layout) (C : Commit L) : ℚ := gapRegs L C / stockRegs L

/-- The words chunks of a call: `⌈m n S / chunkWords⌉`, at most `chunkWords` words a leaf. -/
def wordChunks (L : Layout) : ℕ := (L.m * L.n * L.S + L.chunkWords - 1) / L.chunkWords

end Pouw.Lead
