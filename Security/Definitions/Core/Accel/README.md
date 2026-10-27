# The accelerator model: the envelope ISA and its coverage

The machine 𝓜 (`Accel.Machine`) is a card's table of numbers (`Accel.Card`: `Accel.H200`, `Accel.Sm120`) and its
semantics (`Accel.Open`), which no table states: the headline holds at every semantics meeting (P1) and (P2), and the
coverage premise (`Assumptions.CardWithinAccel`) says the card's own is one. The premise rests on one claim: **every
instruction the card executes is dominated by an envelope opcode** (`Accel.Op`, or a branch) that reads and writes at
least its words, at no more units of each resource. This file is the envelope's opcode list and the map from the card's
instructions onto it; MODEL-COVER (CMP-166) owns the per-opcode table and its sources (CUDA Binary Utilities 12.9
§4.3 and §4.4, PTX ISA 9.0, the DS-120 and DC-90 SASS census).

## The table (`Accel.Card`)

| Field | What it is |
|---|---|
| `size s` | the words of each store: registers (predicates and uniform registers are register words), shared memory with L1, L2, device memory |
| `ρ r` | each resource's units a second at the clock's ceiling: SMs × a SM's units a clock × the clock |
| `weight f` | the tensor units a MAC of operand format `f` uses (`s8` is 1) |
| `exact` | (P1)'s range: an integer MMA is exact while `|C| + Σ|a·b| < 2^exact` |
| `text` | code space, instructions (device memory at 16 bytes an instruction) |

The table refines the device model (`Card.toDevice`, `Accounting.Device`): at a tick of `τ` seconds the device's
on-chip and device-memory bits are the stores', and its tensor MACs, lane ops, oracle queries and device-memory bits
a tick are the rates', rounded up; a run that meets the model stays within them summed over a window (`fits_toDevice`).

Resources: `tensor` (s8 MACs), `issue` (warp instructions), `alu`, `imad`, `shfl`, `f64` and `sfu` lanes, `smem`
(bank words), `l2` (words read) and `dram` (words).

## The envelope opcodes (`Accel.Op`, `Card.use`)

| Opcode | Reads → writes | Units |
|---|---|---|
| `mma o sh sa sb` | an `m × n × k` tile of the card's MMA opcode `o` (formats, 32- or 64-bit accumulators): `A` from `sa`, `B` from `sb`, accumulators in registers → the accumulators | `m·n·k·weight` tensor, one issue slot, each operand word's read at its store |
| `mmaSp o sh sa sb` | as `mma`, 2:4 sparse: `k/2` kept pairs a row, chosen by metadata | `m·n·(k/2)·weight` tensor, one issue slot, reads |
| `lane o` | one of the card's lane functions on its pipe: up to 4 register words (6 on `f64`) → 2 | `1/32` issue slot and a lane of the pipe (`fma` has none: issue-bound) |
| `move s t n k` | `n` words of `s`, read once → `k` copies in `t`, unchanged | `max k 1 / 32` issue, each word's read and each copy's write, a register-to-register copy an `alu` lane |
| `atom s o` | a word of `s` and two registers → the word's new value and a returned word (the card's function) | `1/32` issue, the word's read |
| `red s t n o` | `n` words of `s` into `n` words of `t` (the card's function) | `1/32` issue, a read at each end a word |
| branch | a predicate bit → the program counter, to a fixed, register or immediate target, optionally writing the return index | `1/32` issue |

The hash oracle has no opcode: programs reach it only through the accelerator's priced calls (`Pouw.Dither.Accel.G3`,
`Pouw.Dither.CallsAtPub`). Semantics are fixed by the model only where an honest kernel needs them: a move copies its
words, and an integer MMA is exact within the range (P1). Every MMA dot is a function of its accumulator and its pairs'
products (P2, `ProductForm`).

## Coverage, by class

| The card's instructions | Envelope opcode | Notes |
|---|---|---|
| IMMA, IGMMA (s8, u8) | `mma` | dense `wgmma` and `mma.sync` alike at the tensor rate: `mma.sync`'s lower measured rate only overprices it |
| HMMA, HGMMA (f16, bf16, tf32), QGMMA, QMMA (FP8, e2m1), OMMA (nvf4, mxf4), BMMA, BGMMA (b1) | `mma` | the card's dot at the format's weight; P2 holds by the multiplier structure |
| DMMA | `mma` with 64-bit accumulators | FP64 operands at weight 32 |
| the sparse forms (`mma.sp`, sparse `wgmma`) | `mmaSp` | |
| IADD3, IMAD, IMAD.WIDE, IMAD.HI, LEA, LOP3, SHF, PRMT, POPC, FLO, BREV, ISETP, SEL, IMNMX, IDP, IABS, BMSK, SGXT, the video ops, PLOP3, PSETP, P2R, R2P | `lane` on `alu` or `imad` | each reads at most 4 words and writes at most 2; a 64-bit result is 2 words |
| FADD, FMUL, FFMA, FMNMX, FSET, FSETP, FSEL, FCHK, FSWZADD, HADD2, HFMA2, HMUL2, HMNMX2, HSET2, conversions (F2F, F2I, I2F, I2I, FRND, F2FP) | `lane` on `fma` or `alu` | |
| DADD, DFMA, DMUL, DSETP | `lane` on `f64` | three FP64 operands are 6 words |
| MUFU | `lane` on `sfu` | 16 lanes a SM-clock (CPG Table 7) |
| SHFL | `lane` on `shfl` | registers are pooled, so a lane names another's register |
| the uniform datapath (UIADD3, UIMAD, ULOP3, …, R2UR, S2UR) | `lane`, `move` | uniform registers are register words |
| VOTE, REDUX, ELECT | `lane`s on `fma` | a fold of the card's binary function: 31 lanes are at most the instruction's issue slot |
| MOV, LD, ST, LDG, STG, LDS, STS, LDL, STL, LDSM, STSM, LDC, LDGSTS, TMA and bulk copies | `move` | an address is a register plus an offset; a broadcast load is one read to `k` copies |
| ATOM, ATOMS, ATOMG, RED | `atom` | |
| bulk and TMA reductions, vector RED | `red` | |
| BRA, BRX, JMX, CALL, RET, EXIT | branch | EXIT is a branch past the text's end |
| BAR, SYNCS, MEMBAR, FENCE, DEPBAR, BSSY, BSYNC, WARPSYNC, NANOSLEEP, YIELD | none | no data effect: the schedule orders the threads, and the model has no latency |
| S2R, CS2R (thread and block ids) | `move` from an immediate | a thread's ids are constants of its text |

**Open** (not yet dominated, so outside the premise's claim until added or argued): MATCH (a word compared across 32
lanes in one instruction); MMAs with f16 accumulators; FP6 and block-scaled FP8 MMAs (`kind::mxf8f6f4`); clock reads
(`%clock`, time as data); the texture path's filtering (TEX, TLD4, and surfaces); sm_100's tensor memory and `tcgen05`,
which sm_120 doesn't have. **Excluded** by the premise (no code here uses them, and none computes a committed word or a
hash more cheaply than the model prices): the video and JPEG engines, sm_120's RT cores, the copy engines and compute
data compression.
