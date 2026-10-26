# The accelerator model: opcode coverage

`Accel.Machine` is the model and `Accel.Isa` its instruction set. `Accel.H200` and `Accel.Sm120` are the two cards. The
model claims that **the card can do no more than this machine**. That claim needs every SASS opcode of the card to be
covered: either an opcode of `Accel.Isa` with the same semantics at no fewer units, or a coarser one that lets a
program do at least as much. **The claim is not established yet.** The table below is a first transcription. It must
be checked opcode by opcode against the source, and every *missing* row is a hole in it until that opcode is added.

Sources:
- **BU**: CUDA Binary Utilities 12.9, the instruction-set tables: §4.3 for Hopper (sm_90) and §4.4 for Blackwell (CC
  10.0 to 12.0, sm_120 among them).
- **PTX**: PTX ISA 9.0.
- **Sheet**: the machine model's device sheets for sm_90 and sm_100.
- **Census**: the DS-120 and DC-90 SASS census.

Marks:
- **covered**: the same semantics at the same or fewer units;
- **coarse**: a model opcode that does at least as much, at no more units;
- **missing**: not in the model.

Where the two cards differ, a row says which. Otherwise it holds for both.

## Tensor core

| SASS | PTX | Mark | Model, or what is missing |
|---|---|---|---|
| IMMA.16816, IMMA.16832 (s8, u8, `.SAT`) | `mma.sync` m16n8k16/k32 | covered | `mma` s8/u8; dot pinned (`HOPPER_SM90_S8_K32`, `BLACKWELL_SM120_S8_M16N8K32`). On the H200 the tensor rate is `Open.sync` (CALIB) |
| HMMA.16816 (f16, bf16), HMMA.1688 (tf32) | `mma.sync` | covered | `mma`. The bf16 dot is pinned; f16 and tf32 are `Open.dot`. On the H200 the rate is `Open.sync` |
| HMMA with f16 accumulators | `mma.sync` `.f16` D | missing | f16 accumulators |
| BMMA (b1 `.and.popc`, `.xor.popc`) | `mma.sync` `.b1` | coarse | `mma` b1 at `Open.b1` tensor units and an issue slot. `.xor.popc` is missing |
| HGMMA, IGMMA, QGMMA, BGMMA (sm_90) | `wgmma.mma_async` | covered | `mma` at m = 64, A in registers or shared memory, B in shared memory, its operand words paying bank units (R2) |
| QMMA.16832 (sm_120: e4m3, e5m2, e2m1) | `mma.sync` `kind::f8f6f4` | covered | `mma`. Same-format dots are pinned; mixed pairings are `Open.dot`. FP6 (e3m2, e2m3) is missing |
| OMMA.SF.16864 (sm_120: nvf4, mxf4) | `mma.sync` `kind::mxf4nvf4` `.block_scale` | covered | `mma` nvf4/mxf4. The dot is `Open.dot` (pinned in Python only) |
| QMMA.SF (sm_120 `kind::mxf8f6f4`) | `mma.sync` `.block_scale` | missing | block-scaled FP8 and FP6 |
| HMMA.SP, IMMA.SP, QMMA.SP, OMMA.SP, the sparse GMMA | `mma.sp`, `wgmma.mma_async.sp` | missing | 2:4 sparsity: the same products at the same rate with a free gather of A |
| DMMA (sm_90) | `mma.sync` `.f64` | missing | 128 f64 MACs an SM clock (Sheet); an int8 × int8 product is exact in f64 |
| DMMA (sm_120) | `mma.sync` `.f64` | missing | to check: whether sm_120 has a tensor DMMA at any rate |

## Integer and floating-point lanes

| SASS | Mark | Model, or what is missing |
|---|---|---|
| IADD3, IMAD, IMAD.WIDE, LOP3, SHF, PRMT, POPC, FLO, BREV, ISETP, SEL, IMNMX, LEA | covered | `Lane` (`iadd3`, `mad`, `madWide` coarsely at one lane, `lop3`, `shf`, `prmt`, `popc`, `flo`, `brev`, `setp`, `sel`, `min`, `max`, `lea`) |
| IMAD.HI, LEA.HI | covered, semantics not captured | `Lane.madHi` (PTX `mad.hi`), `Lane.leaHi` (the high word of the funnel shift, plus `b`); added at FOLD-SAT's report (CMP-131); their SASS semantics are to capture, and FOLD-SAT's SMT models are to align |
| IDP.4A (dp4a) | covered | `Lane.dp4a` |
| IDP.2A (dp2a) | missing | |
| IABS, BMSK, SGXT, IMUL.HI (`mul.hi`), ISCADD | missing | each is a short sequence of covered lanes, at more units |
| FADD, FMUL, FFMA | covered | `Lane.fadd`, `fmul`, `ffma` (round to nearest even), issue-bound |
| FMNMX, FSEL, FSET, FSETP, FCHK, FSWZADD | missing | float compare, select and min/max |
| MUFU (rcp, rsq, sqrt, ex2, lg2, sin, cos, tanh) | missing | the SFU pipe and its rate |
| HADD2, HFMA2, HMUL2, HMNMX2, HSET2, HSETP2 | missing | two f16/bf16 results a lane (SIMD in a word) |
| DADD, DFMA, DMUL, DSETP | missing | the FP64 pipe |
| VABSDIFF, VABSDIFF4, VIMNMX (video, SIMD in a word) | missing | |
| F2F, F2I, I2F, I2I, FRND, F2FP (the PACK_AB pack), I2FP, F2IP, I2IP | missing | conversions. F2FP is W1's `cvt` |

## Uniform datapath

| SASS | Mark | Model, or what is missing |
|---|---|---|
| UIADD3, UIMAD, ULOP3, USHF, UPRMT, UPOPC, UFLO, UBREV, UISETP, USEL, ULEA, UMOV | covered | `uni` at one issue slot (measured: Sheet `[uniform_issue]`) |
| R2UR, S2UR, UR2UP, UP2UR, ULDC | coarse | uniform registers are pooled with the register file, so a move between them is a register copy |
| REDUX, VOTEU, UF2FP, UBMSK, USGXT | missing | |

## Moves, memory and atomics

| SASS | Mark | Model, or what is missing |
|---|---|---|
| MOV, SHFL | coarse | a register copy (a shuffle is a cross-lane move here). The shuffle pipe's rate is omitted |
| LDG, STG, LDS, STS, LDL, STL, LD, ST | covered | `copy` of at most 4 words, direct or `indirect` (an address register plus an offset) |
| LDSM, STSM (`ldmatrix`, `stmatrix`) | coarse | 4-word copies. The `.trans` forms move 16-bit halves across lanes, which is missing |
| MOVM (`movmatrix`) | missing | |
| LDGSTS (`cp.async`) | coarse | a copy from device memory or L2 to shared memory, with no latency |
| UTMALDG, UTMASTG, UBLKCP (TMA, `cp.async.bulk`) | coarse | a bulk copy of any length with shared memory at one end |
| UTMAREDG (TMA reduce), TMA multicast | missing | |
| ATOMS, ATOMG, ATOM, RED (add, exch) | covered | `atom` add and exch on shared memory, L2 and device memory. RED is an `atom` whose old value is discarded |
| ATOMS, ATOMG (min, max, and, or, xor, inc, dec, cas), float atomics | missing | |
| LDC (constant bank operands) | missing | constant-bank operands |
| TEX, TLD, TLD4, TXD, TMML, SULD, SUST, SURED, SUATOM | **missing, important** | the texture path has its own bandwidth and filtering arithmetic (weighted sums of texels) outside every resource here |

## Control and synchronization

| SASS | Mark | Model, or what is missing |
|---|---|---|
| BRA, EXIT | covered | `bra` on a register, and a thread past its text's end |
| predication (`@P`) | coarse | `pred` on a register (predicates are registers here) |
| BRX, JMX, CALL, RET, BSSY, BSYNC, WARPSYNC, BREAK | missing | indirect branches and calls, which are emulable by `bra` trees at more units |
| BAR, SYNCS (mbarrier), MEMBAR, FENCE, DEPBAR, ERRBAR, CCTL | coarse | the model is sequential within a tick and latency-free, so every synchronization holds at no cost |
| S2R (`%tid`, `%clock`, `%smid`), CS2R | missing | special registers. A clock read is time as data, a side channel the model doesn't have |
| VOTE, MATCH, ELECT, NANOSLEEP, YIELD, PMTRIG | missing | |
