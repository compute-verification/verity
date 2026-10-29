# The ACCEL model

ACCEL is the physical model the accounting reads: a machine 𝓜 (`Accel.Machine`) made of a card's table of numbers
(`Accel.Card`: `Accel.H200`, `Accel.Sm120`) and the card's semantics (`Accel.Open`), which no table states. Programs
are lists of the ACCEL ops (`Accel.Op`) and branches. A run fits a time `T` when its code fits the text (G1) and each
resource's units are at most its rate times `T` (G2), on a schedule fixed before the content (`Machine.Fits`). The
headline holds at every semantics meeting (P1), integer MMAs exact within the table's range, and (P2), every MMA dot a
function of its accumulator and its pairs' products. The card premise says the card's own semantics is one of them.

## The table (`Accel.Card`)

| Field | What it is |
|---|---|
| `size s` | the words of each store: registers (predicates and uniform registers are register words), shared memory with L1, L2, device memory |
| `ρ r` | each resource's units a second at its clock's ceiling: SMs × a SM's units a clock × the clock (the tensor's own clock for `tensor`) |
| `weight f` | the tensor units a MAC of operand format `f` uses (`s8` is 1) |
| `exact` | (P1)'s range: an integer MMA is exact while `|C| + Σ|a·b| < 2^exact` |
| `text` | code space, instructions (device memory at 16 bytes an instruction) |

Resources: `tensor` (s8 MACs), `issue` (warp instructions), the `alu`, `imad`, `shfl`, `f64` and `sfu` lanes, `smem`
(bank words), `l2` (words read or written) and `dram` (words). Each table is a SKU's at its firmware's maximum:

| Table | SKU | SM clock | Tensor clock | Device memory |
|---|---|---|---|---|
| `Accel.H200.card` | H200 SXM 141 GB, 132 SMs | 1980 MHz | 1830 MHz | HBM3e at 3201 MHz on 6144 bits, 4.917 TB/s |
| `Accel.Sm120.card` | RTX PRO 6000 Blackwell Server Edition, 188 SMs | 2430 MHz | 2430 MHz | GDDR7 at 12481 MHz on 512 bits, 1597.6 GB/s |

Both cards' firmware refuses or clamps every clock above these, and both report SM and memory clock offset ranges of
[0, 0]. The H200's tensor rate is taken at its tensor clock's ceiling, 1830 MHz (`Accel.H200.tensorClock`): the clock
its data sheet quotes the tensor peaks at, and the most its SMs' clock reads with the tensor pipe full while the driver
reports 1980. Every rate is a ceiling, not a clock a loaded card holds: under its power limit it holds less. The table also gives the device model's limits (`Card.toDevice`, `Accounting.Device`). A run that fits the
model stays within them summed over a window (`fits_toDevice`), and no per-tick `Accounting.Schedule` is built from a
run.

## The ACCEL ops (`Accel.Op`, `Card.use`)

- `mma` and `mmaSp`: an `m × n × k` tile of one of the card's MMA opcodes, dense or 2:4 sparse. Each costs
  `m·n·max(k, 1)` (or `k/2` kept pairs) products at the format's weight in tensor units, one issue slot, and each
  operand word's read at its store.
- `lane`: one of the card's lane functions on its pipe, reading up to 4 register words (6 on `f64`) and writing 2.
  It costs `1/32` of an issue slot and a lane of the pipe.
- `move`: `n` words of a store read once into `k` copies in another. It costs `max k 1 / 32` issue slots, each word's
  read and each copy's write, and a register-to-register copy is an `alu` lane.
- `atom` and `red`: an atomic on a word, or a reduction of `n` words into `n` words, each with the card's function.
  Each costs `1/32` of an issue slot and an access at each end; a word either writes to registers is an `alu` lane.
- A branch costs `1/32` of an issue slot.

The hash oracle has no op: programs reach it only through the accelerator's priced calls (`Pouw.Dither.Accel.G3`,
`Pouw.Dither.CallsAtPub`). The model fixes semantics only where an honest kernel needs them: a move copies its words,
and an integer MMA is exact within the range (P1).

## The card premise (`Assumptions.CardWithinAccel`)

The card does no more than 𝓜 at its numbers: for some semantics meeting (P1) and (P2), whatever the card computes in
`T` seconds, some program of 𝓜 computes on a schedule fixed before the content, within `T` (`Machine.Fits`). It rests on
one claim: every instruction the card executes is dominated by an ACCEL op (or a branch) that reads and writes at least
its words, at no more units of each resource. The per-instruction evidence is the coverage table,
`benchmarks/pouw/machine_model/data/cover/rows.toml`. At a table, the premise also says the cards are that table's SKU
and firmware.

## Proved

- `WorkSoundAt`, `WorkSoundOnCard` and their instances on both tables (`WorkSoundH200`, `WorkSoundSm120`,
  `WorkSoundOnCardH200`, `WorkSoundOnCardSm120`) follow from Dither's `WorkSound`, with no premise beyond its own
  and the card premise (`Proofs/Pouw/DitherInstance.lean`).
- The premise holds at the honest instance (`gemm_cardWithinAccel`, `Proofs/Core/Accel/Witness.lean`). The deployed
  `s8` GEMM, at every `n` with 256 ∣ n ≤ 2048, fits the H200's table in its seconds (`kernel_fits`) and leaves
  C = A·B in device memory (`kernel_computes`).
- The premise refuses a card that changes memory in no time (`cardWithinAccel_fails`), and one that changes more words
  of shared memory, L2 or device memory in `T` seconds than the table's rate times `T` (`cardWithinAccel_fails_rate`)
  (`Proofs/Core/Accel/Model.lean`).
- The exact `s8` semantics (`s8Open`) meets (P1) and (P2) on both cards, so the headline's hypotheses on the semantics
  are not empty.

## Open

- **Instructions not yet dominated**, outside the premise's claim until an op covers them or an argument does:
  - MATCH, which compares a word across 32 lanes in one instruction, and REDUX, which shares its unit;
  - VOTE, which spends 16 `alu` lanes;
  - MMAs with f16 accumulators;
  - FP6 and block-scaled FP8 and FP6 MMAs, since `Fmt` lacks `e3m2`, `e2m3`, `mxf8` and `mxf6`;
  - clock reads, which make time data;
  - IADD3 at sm_120's 128 lanes a SM-clock, twice `alu`'s 64;
  - the uniform datapath's 4 results a SM-clock beside `alu`'s lanes;
  - BAR.RED, which folds a predicate across the CTA.
- **Exclusions.** The premise excludes the video and JPEG engines, sm_120's RT cores, the texture path's filtering, the
  copy engines and compute data compression. An exclusion is a second adversary class until it has a rate row or a
  proof that the model dominates it. The RT cores (at most 58 integer products a SM-clock, 6% of sm_120's `s8` rate)
  and the texture path (at most 64, 6% on sm_120 and 1.7% on the H200) exceed the headline's served margins, 3.9% on
  sm_120 and 2.6% on the H200.
- **The card numbers.** Nothing the protocol checks attests a node's SKU and firmware. The H200's L2 row is a
  placeholder, neither measured nor a ceiling, and sm_120's is one read pattern's measurement.
- **CUDA-core matmul routes.** With every register word an `alu` lane, the four-Russians table method pays at least an
  `alu` lane a MAC: at most 6.25% of the `s8` tensor rate on sm_120 and 1.7% on the H200 through registers, and 13.5% on
  the H200 through L2 at its placeholder row. IDP.4A on `imad` does 4 MACs a lane. None of these is a public route, so
  `SAKMatmul` as written fails at a dual price that leaves their pipes cheap. The algorithmic premise that would carry
  them, `SAKResidual`, is not yet adopted.
