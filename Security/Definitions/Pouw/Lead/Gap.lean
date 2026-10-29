import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Definitions.Pouw.Basic.Prob
import Definitions.Pouw.Lead.Model

/-!
# The lead's honest work and its gap, as functions of the layout (trusted definitions)

Every quantity is in registers; the W1 work is 32 a register (`H32`'s floor).

* **Stock** (`stockRegs`): the call's unmodified FP8 matmul over whole MMA tiles, `m · n · K / 32` k32 `wgmma`
  registers with `m` and `n` padded. The honest work and the gap are the padded call's.
* **Served compute** (`servedRegs`): the call's unpadded MACs, `rows · cols · K / 32`, what credit counts. Padding
  rows and columns are committed and hashed, so they are in the honest work, but they are never served compute. At
  decode one row is served and 16 are committed, so the gap is reported as a share of served compute (`gapServed`),
  and its share of stock (`gapShare`) is secondary.
* **The honest work** (`honestRegs`): the committed words, one register each (`m · n · S`), and the SHA-512
  compressions of the commitment at ShaWrites' registers per form: `3,515.38` a words compression and `3,449.05` an A
  compression (`shaWordsRegs`, `shaARegs`).
* **The gap** (`gapRegs`) is the sum of six terms, each the bound its piece proves:
  - `headGap`: the words chunks' first-block excess, 23 registers a chunk;
  - `hidingGap`: hiding L1 + (a), 1.6064% of one row's stock a call;
  - `coinBranchGap`: what a prover that branches on the coins keeps by copying whole MMA tiles across the call's
    rows and patching the rest (copy-and-patch at tile grain), 0.084% of stock;
  - `pStarGap`: the coin lanes' guess at `-rot`'s weights, 0.0035% of stock;
  - `networkGap`: work outsourced within the call's bits (`Layout.callBits`), `betaLead` a bit: `n_max = 256/3` MACs
    a coin bit, every step carrying the three coin lanes, at the call's honest W1 work a MAC (`wMac`);
  - `tableGap`: pre-coin tables and the coin stream holding a committed tile, within the node's memory, from each
    step's coin lanes alone.
* `gapShare` is the gap over stock, which `Pouw.Guarantees.Lead.GapAt` evaluates at the three points; `gapServed` is
  the gap over served compute, which `GapAtDecode` states at decode.
* **The tail form** (`G_tail`) splits the gap into a fixed part (`gapFixed`) and a probability (`etaLead`).
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

/-- A's rows, unpadded, in 128-byte compressions: `rows · K / 128`. -/
def aBlocks (L : Layout) : ℚ := (L.rows * L.K : ℚ) / 128

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

/-- MACs a coin bit at most, where every step carries the three coin lanes under `-rot`'s weights: `256/3`. -/
def nMax : ℚ := 256 / 3

/-- The call's honest W1 work a MAC: `32 · honestRegs / (m · n · K)`. -/
def wMac (L : Layout) (C : Commit L) : ℚ := 32 * honestRegs L C / (L.m * L.n * L.K)

/-- The priced W1 work an exchanged bit replaces: `n_max · W_mac`. -/
def betaLead (L : Layout) (C : Commit L) : ℚ := nMax * wMac L C

/-- Work outsourced within the call's bits, in registers: `betaLead · callBits / 32`. -/
def networkGap (L : Layout) (C : Commit L) : ℚ := betaLead L C * L.callBits / 32

/-- Pre-coin tables and the coin stream holding a committed tile, in a call of whole tiles: a memory's `2 · mem + 1`
tile values (the tables', zero and the stream's) against each committed tile's words, at every step, step 0
included, from the tile's coin lanes alone. Each column's even and odd rows fix its three lanes, so a tile is a
given value with chance at most `2^-168`; at most three of its columns hold the stream's lanes, so it is a streamed
tile with chance at most `2^-105`; either is within `2^-100`, so a tile and step are held with chance at most
`(2 · mem + 1) · 2^-100`. -/
def tableGap (L : Layout) : ℚ :=
  (L.rowBlocks * L.colBlocks * tileWords : ℚ) * (L.S * (2 * L.mem + 1 : ℚ) / 2 ^ 100)

/-- **The gap** in registers: the sum of the six terms. -/
def gapRegs (L : Layout) (C : Commit L) : ℚ :=
  headGap L C + hidingGap L + coinBranchGap L + pStarGap L + networkGap L C + tableGap L

/-- The gap as a share of stock. -/
def gapShare (L : Layout) (C : Commit L) : ℚ := gapRegs L C / stockRegs L

/-- **Served compute**: the call's unpadded MACs in k32 `wgmma` registers, `rows · cols · K / 32`. Credit is served
compute. Stock and the honest work count the padding rows and columns, which are committed and hashed but serve
nothing. -/
def servedRegs (L : Layout) : ℚ := (L.rows * L.cols * L.K : ℚ) / 32

/-- The gap as a share of served compute. -/
def gapServed (L : Layout) (C : Commit L) : ℚ := gapRegs L C / servedRegs L

/-- The words chunks of a call: `⌈m n S / chunkWords⌉`, at most `chunkWords` words a leaf. -/
def wordChunks (L : Layout) : ℕ := (L.m * L.n * L.S + L.chunkWords - 1) / L.chunkWords

/-! ## The tail form

`Pouw.Guarantees.Lead.G_tail` bounds the share of the coins on which a strategy is accepted, hashed no collision and
worked less than the honest work less the fixed gap (`gapFixed`) by `etaLead`.

* **Fixed** (`gapFixed`), in registers, holding at every accepted coin value: the head (`headGap`), the network
  (`networkGap`, under the per-call egress refusal), the coin-branch count's mean (`coinBranchMean`) and slack
  (`coinSlack`), and `p*`'s mean (`pStarGap`). Hiding is 0: the keyed leaf replaces L1 + (a)'s salts.
* **Probabilities** (`etaLead`): a pre-coin table or the coin stream holding a committed tile (`tableTail`, a union
  bound over tiles and steps), and the call's repeated tiles exceeding their mean by more than the slack
  (`coinTail`). -/

/-- **The coin-branch count's mean**, in registers: `tileWords · rowBlocks · colBlocks · S · (4/7^8 + 2^−16)`. A
tile's value agrees with each of the three other tiles at its step, column block and coin tile on at most `7^−8` of
the coins and with any tile on at most `2^−56`, and a call the verifier admits has at most `2^40` tiles, so averaged
over the coins `tileWords · repeats` is at most this. -/
def coinBranchMean (L : Layout) : ℚ :=
  tileWords * (L.rowBlocks * L.colBlocks * L.S : ℚ) * (4 / 7 ^ 8 + 1 / 2 ^ 16)

/-- **The slack** `t` over the coin-branch mean that the tail allows: 0.08% of stock. -/
def coinSlack (L : Layout) : ℚ := 8 / 10000 * stockRegs L

/-- **The tail's fixed gap** in registers: the head, the network, the coin-branch mean and slack, and `p*`. -/
def gapFixed (L : Layout) (C : Commit L) : ℚ :=
  headGap L C + networkGap L C + coinBranchMean L + coinSlack L + pStarGap L

/-- The fixed gap as a share of stock. -/
def gapFixedShare (L : Layout) (C : Commit L) : ℚ := gapFixed L C / stockRegs L

/-- The fixed gap as a share of served compute. -/
def gapFixedServed (L : Layout) (C : Commit L) : ℚ := gapFixed L C / servedRegs L

/-- **The tables' tail**: the share of the coins on which a memory fixed before the coins, or the coin stream, holds
some committed tile, by a union bound over the call's tiles and steps. A tile at a step is one of a memory's
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
