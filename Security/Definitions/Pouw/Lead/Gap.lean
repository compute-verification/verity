import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Definitions.Pouw.Basic.Prob
import Definitions.Pouw.Lead.Model

/-!
# The lead's honest work and its gap, as functions of the layout (trusted definitions)

Every quantity is in registers; the W1 work is 32 a register (`H32`'s floor).

* **Served stock** (`servedRegs`): the plain matmul of the call's served shape, `rows · cols · K / 32` k32 `wgmma`
  registers. Every share of the gap is over served stock (`gapServed`, `gapBound`).
* **Padded stock** (`stockRegs`): the matmul over the committed, padded shape, `m · n · K / 32`. Padding rows and
  columns are committed and hashed, so they are in the honest work, and the terms that scale with the committed tiles
  are shares of it; but they serve nothing. At decode one row is served and 16 are committed.
* **The honest work** (`honestRegs`): the committed words, one register each (`m · n · S`), and the SHA-512
  compressions of the commitment at ShaWrites' registers per form: `3,530.92` a words compression, the lead's 256 KiB
  chunks' figure, and `3,449.05` an A compression (`shaWordsRegs`, `shaARegs`).
* **The gap** (`gapRegs`) is the sum of five terms, each the bound its piece proves:
  - `headGap`: the words chunks' first-block excess, 23 registers a chunk;
  - `hidingGap`: hiding L1 + (a), 1.6064% of one row's padded stock a call;
  - `coinBranchGap`: whole-tile repeats, what copying whole tiles saves, averaged over the coins; 0.084% of padded
    stock, which is slack;
  - `pStarGap`: the coin lanes' guess at `-rot`'s weights, 0.0035% of padded stock;
  - `tableGap`: pre-coin tables and the coin stream holding a committed tile, within the node's storage, from each
    step's coin lanes alone.
* **The network term** is a forward's, not a call's: `Pouw.Guarantees.Lead.G` charges `betaLead` a bit a call's
  execution exchanged, and the network certifier bounds a forward's bits (`Pouw.Lead.Forward.networkRegs`). `betaLead`
  is `n_max = 256/3` MACs a coin bit, every step carrying the three coin lanes, at the call's honest W1 work a MAC
  (`wMac`).
* `gapServed` is the gap over served stock, which `Pouw.Guarantees.Lead.GapAt` evaluates at the three points and
  `GapAtDecode` at decode; `gapBound` is a closed form of it at every admitted call (`GapAll`).
* **The tail form** (`G_tail`) splits the gap into a fixed part (`gapFixed`) and a probability (`etaLead`).
-/

namespace Pouw.Lead

/-- Padded stock: the call's FP8 matmul over its committed, padded shape in k32 `wgmma` registers, `m · n · K / 32`. -/
def stockRegs (L : Layout) : ℚ := (L.m * L.n * L.K : ℚ) / 32

/-- The committed words, one register each: `m · n · S`. -/
def wordsRegs (L : Layout) : ℚ := L.m * L.n * L.S

/-- ShaWrites' registers a words compression, at the lead's 256 KiB words chunks. -/
def shaWordsRegs : ℚ := 353092 / 100

/-- ShaWrites' registers an A compression. -/
def shaARegs : ℚ := 344905 / 100

/-- The words' bytes in 128-byte compressions, with one padding block a chunk: `4 m n S / 128 + chunks`. -/
def wordsBlocks (L : Layout) (C : Commit L) : ℚ := 4 * wordsRegs L / 128 + C.chunks

/-- A's rows, unpadded, in 128-byte compressions: `rows · K / 128`. -/
def aBlocks (L : Layout) : ℚ := (L.rows * L.K : ℚ) / 128

/-- **The honest work** in registers: the words, and the commitment's compressions at ShaWrites' registers. -/
def honestRegs (L : Layout) (C : Commit L) : ℚ :=
  wordsRegs L + shaWordsRegs * wordsBlocks L C + shaARegs * aBlocks L

/-- The words chunks' first-block excess: 23 registers a chunk. -/
def headGap (L : Layout) (C : Commit L) : ℚ := 23 * C.chunks

/-- Hiding L1 + (a): 1.6064% of one row's padded stock a call, `0.016064 · n · K / 32`. -/
def hidingGap (L : Layout) : ℚ := 16064 / 1000000 * ((L.n * L.K : ℚ) / 32)

/-- Whole-tile repeats, what copying whole tiles saves: the committed tiles less their distinct values, in words and
averaged over the coins, are at most this at every admitted call (`Tags.repeats_le`: a tile repeats with chance at
most `4 · 7^-8 + tiles · 2^-56`, from step 0's tag weights and from pairs of tiles' coin lanes). It is slack: under
`TileGrain` a patched tile costs the whole tile, so no copy-and-patch saving is left, and the repeats' average is far
below it. 0.084% of padded stock. -/
def coinBranchGap (L : Layout) : ℚ := 84 / 100000 * stockRegs L

/-- The coin lanes' guess under `-rot`'s weights: 0.0035% of padded stock. -/
def pStarGap (L : Layout) : ℚ := 35 / 1000000 * stockRegs L

/-- MACs a coin bit at most, where every step carries the three coin lanes under `-rot`'s weights: `256/3`. -/
def nMax : ℚ := 256 / 3

/-- The call's honest W1 work a MAC: `32 · honestRegs / (m · n · K)`. -/
def wMac (L : Layout) (C : Commit L) : ℚ := 32 * honestRegs L C / (L.m * L.n * L.K)

/-- The priced W1 work an exchanged bit replaces: `n_max · W_mac`. -/
def betaLead (L : Layout) (C : Commit L) : ℚ := nMax * wMac L C

/-- Pre-coin tables and the coin stream holding a committed tile, in a call of whole tiles: a storage's `2 · mem + 1`
tile values (the tables', zero and the stream's) against each committed tile's words, at every step, step 0
included, from the tile's coin lanes alone. Each column's even and odd rows fix its three lanes, so a tile is a
given value with chance at most `2^-168`; at most three of its columns hold the stream's lanes, so it is a streamed
tile with chance at most `2^-105`; either is within `2^-100`, so a tile and step are held with chance at most
`(2 · mem + 1) · 2^-100`. -/
def tableGap (L : Layout) : ℚ :=
  (L.rowBlocks * L.colBlocks * tileWords : ℚ) * (L.S * (2 * L.mem + 1 : ℚ) / 2 ^ 100)

/-- **The gap** in registers: the sum of the five terms. -/
def gapRegs (L : Layout) (C : Commit L) : ℚ :=
  headGap L C + hidingGap L + coinBranchGap L + pStarGap L + tableGap L

/-- **Served stock**: the plain matmul of the call's served shape, its unpadded MACs in k32 `wgmma` registers,
`rows · cols · K / 32`. Padded stock and the honest work count the padding rows and columns, which are committed and
hashed but serve nothing. -/
def servedRegs (L : Layout) : ℚ := (L.rows * L.cols * L.K : ℚ) / 32

/-- The gap as a share of served stock. -/
def gapServed (L : Layout) (C : Commit L) : ℚ := gapRegs L C / servedRegs L

/-- **The gap's bound at the lead's call of `m × K × n`**, a share of served stock: with `ρ` the padding's inflation
`padRows m · padCols n / (m n)`,
`ρ · (0.0875% + (1 + 34/K) · (23/57,344 + 8 (2^41 + 1) / (7 · 2^100))) + 1.6064% · padCols n / (m n) + 736 / (m n K)`.
The terms are copy-and-patch and `p*` over padded stock, the chunks' heads and the tables at `stepsFor K ≤ (K + 34)/28`
steps, hiding's row, and one partial chunk's head. -/
def gapBound (m n K : ℕ) : ℚ :=
  (padRows m * padCols n : ℚ) / (m * n) *
      (875 / 1000000 + (1 + 34 / (K : ℚ)) * (23 / 57344 + 8 * (2 * 2 ^ 40 + 1) / (7 * 2 ^ 100))) +
    16064 / 1000000 * padCols n / (m * n) + 736 / (m * n * K)

/-- The words chunks of a call: `⌈m n S / chunkWords⌉`, at most `chunkWords` words a leaf. -/
def wordChunks (L : Layout) : ℕ := (L.m * L.n * L.S + L.chunkWords - 1) / L.chunkWords

/-! ## The tail form

`Pouw.Guarantees.Lead.G_tail` bounds the share of the certifier's entropy values at whose coins a strategy is
accepted, hashed no collision and worked, `betaLead` a bit exchanged included, less than the honest work less the
fixed gap (`gapFixed`) by `etaLead`.

* **Fixed** (`gapFixed`), in registers, holding at every accepted coin value: the head (`headGap`), the coin-branch
  count's mean (`coinBranchMean`) and slack (`coinSlack`), and `p*`'s mean (`pStarGap`). The network is charged in
  the work, as in `G`, and bounded per forward. Hiding doesn't enter: the per-coin floor has no hiding term.
* **Probabilities** (`etaLead`): a pre-coin table in the node's storage or the coin stream holding a committed tile
  (`tableTail`, a union bound over tiles and steps), and the call's repeated tiles exceeding their mean by more than
  the slack (`coinTail`). -/

/-- **The coin-branch count's mean**, in registers: `tileWords · rowBlocks · colBlocks · S · (4/7^8 + 2^−16)`. A
tile's value agrees with each of the three other tiles at its step, column block and coin tile on at most `7^−8` of
the coins and with any tile on at most `2^−56`, and a call the verifier admits has at most `2^40` tiles, so averaged
over the coins `tileWords · repeats` is at most this. -/
def coinBranchMean (L : Layout) : ℚ :=
  tileWords * (L.rowBlocks * L.colBlocks * L.S : ℚ) * (4 / 7 ^ 8 + 1 / 2 ^ 16)

/-- **The slack** `t` over the coin-branch mean that the tail allows: 0.08% of padded stock. -/
def coinSlack (L : Layout) : ℚ := 8 / 10000 * stockRegs L

/-- **The tail's fixed gap** in registers: the head, the coin-branch mean and slack, and `p*`. -/
def gapFixed (L : Layout) (C : Commit L) : ℚ :=
  headGap L C + coinBranchMean L + coinSlack L + pStarGap L

/-- The fixed gap as a share of served stock. -/
def gapFixedServed (L : Layout) (C : Commit L) : ℚ := gapFixed L C / servedRegs L

/-- **The tables' tail**: the share of the coins on which a storage fixed before the coins, or the coin stream,
holds some committed tile, by a union bound over the call's tiles and steps. A tile at a step is one of a storage's
`mem + 1` values (the tables' and zero) on at most `2^−168` of the coins each, and one of its `mem` streamed tiles on
at most `2^−105` each. -/
def tableTail (L : Layout) : ℚ :=
  (L.rowBlocks * L.colBlocks * L.S : ℚ) * ((L.mem + 1 : ℚ) / 2 ^ 168 + L.mem / 2 ^ 105)

/-- **The coin-branch tail**: the largest share of the coins, over the call's data, on which its repeated tiles'
words exceed the coin-branch mean by more than the slack. A strategy doesn't enter: `repeats` reads the data and the
coins alone. -/
noncomputable def coinTail (L : Layout) : ℝ :=
  ⨆ κ : Fin L.m → Fin L.S → Fin L.n → ℤ,
    Pouw.pr fun ω : Coins L => coinBranchMean L + coinSlack L < tileWords * (repeats L κ ω : ℚ)

/-- **The tail** `η_lead`: the tables' tail and the coin-branch tail. -/
noncomputable def etaLead (L : Layout) : ℝ := tableTail L + coinTail L

end Pouw.Lead
