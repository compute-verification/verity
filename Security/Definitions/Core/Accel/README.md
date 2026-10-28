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
- **Census**: the sm_120 and sm_90 SASS census (`benchmarks/pouw/sass_census/`).

Marks:
- **covered**: the same semantics at the same or fewer units;
- **coarse**: a model opcode that does at least as much, at no more units;
- **over-approximated**: an exact integer version at the unit's rate and in its register layout. It equals the unit on
  its format's exact integers once the card's float codes are read as those integers (`Card.decode`, the encoding
  `Assumptions.CardWithinAccel` refines up to). Use out of that range is outside the claim on the contents where it occurs;
- **missing**: not in the model.

Where the two cards differ, a row says which. Otherwise it holds for both.

## Tensor core

| SASS | PTX | Mark | Model, or what is missing |
|---|---|---|---|
| IMMA.16816, IMMA.16832 (s8, u8, `.SAT`) | `mma.sync` m16n8k16/k32 | covered | `mma` s8/u8; dot pinned (`HOPPER_SM90_S8_K32`, `BLACKWELL_SM120_S8_M16N8K32`). On the H200 the tensor rate is `Open.sync` until measured |
| HMMA.16816 (f16, bf16), HMMA.1688 (tf32) | `mma.sync` | covered | `mma`. The bf16 dot is pinned; f16 and tf32 are `Open.dot`. On the H200 the rate is `Open.sync` |
| HMMA with f16 accumulators | `mma.sync` `.f16` D | missing | f16 accumulators |
| BMMA (b1 `.and.popc`, `.xor.popc`) | `mma.sync` `.b1` | coarse | `mma` b1 at `Open.b1` tensor units and an issue slot. `.xor.popc` is missing |
| HGMMA, IGMMA, QGMMA, BGMMA (sm_90) | `wgmma.mma_async` | covered | `mma` at m = 64, A in registers or shared memory, B in shared memory, its operand words paying bank units (R2) |
| QMMA.16832 (sm_120: e4m3, e5m2, e2m1) | `mma.sync` `kind::f8f6f4` | covered | `mma`. The E4M3 and E5M2 same-format K32 dots are pinned at every accumulator (`Sm120.f8Pipeline`); E2M1 and the mixed pairings are `Open.dot`. FP6 (e3m2, e2m3) is missing |
| OMMA.SF.16864 (sm_120: nvf4, mxf4) | `mma.sync` `kind::mxf4nvf4` `.block_scale` | covered | `mma` nvf4/mxf4. The dot is `Open.dot` (pinned in Python only) |
| QMMA.SF (sm_120 `kind::mxf8f6f4`) | `mma.sync` `.block_scale` | missing | block-scaled FP8 and FP6 |
| HMMA.SP, IMMA.SP, QMMA.SP, the sparse GMMA | `mma.sp`, `wgmma.mma_async.sp` | covered | `Opcode.mmaSp`: the kept pairs (each kept element of A with the element of B its 2-bit selector names) at the dense forms' tensor units a real product, so twice the logical rate; dots `Open.dotSp` under `AccelProductForm` |
| OMMA.SP (sm_120) | `mma.sp` `kind::mxf4nvf4` | missing | block-scaled sparse FP4 |
| DMMA (sm_90) | `mma.sync` `.f64` | over-approximated | `Opcode.dmma`: exact integer MMA on FP64's exact integers, on the tensor cores at 128 MACs an SM clock at 1980 MHz (32 units of `tensor` and `tensorLong` a MAC, as ALGO-COST A18 measured it beside `wgmma`, so it shares the low-precision formats' budget) |
| DMMA (sm_120) | `mma.sync` `.f64` | over-approximated | `Opcode.dmma` at 32 MACs an SM clock; rate undocumented, measured at 1.75 |

## Integer and floating-point lanes

| SASS | Mark | Model, or what is missing |
|---|---|---|
| IADD3, IMAD, IMAD.WIDE, LOP3, SHF, PRMT, POPC, FLO, BREV, ISETP, SEL, IMNMX, LEA | covered | `Lane` (`iadd3`, `mad`, `madWide` at two `imad` lanes (CPG Table 7 footnote 6), `lop3`, `shf`, `prmt`, `popc`, `flo`, `brev`, `setp`, `sel`, `min`, `max`, `lea`) |
| IMAD.HI, LEA.HI | covered, semantics not captured | `Lane.madHi` (PTX `mad.hi`), `Lane.leaHi` (the high word of the funnel shift, plus `b`); their SASS semantics are to capture, and the fold's SMT models are to align with them |
| IDP.4A (dp4a) | covered | `Lane.dp4a` |
| IDP.2A (dp2a) | missing | |
| IABS, BMSK, SGXT, IMUL.HI (`mul.hi`), ISCADD | missing | each is a short sequence of covered lanes, at more units |
| FADD, FMUL, FFMA | covered | `Lane.fadd`, `fmul`, `ffma` (round to nearest even), issue-bound |
| FMNMX, FSEL, FSET, FSETP, FCHK, FSWZADD | missing | float compare, select and min/max |
| MUFU (rcp, rsq, sqrt, ex2, lg2, sin, cos, tanh) | missing | the SFU pipe and its rate |
| HADD2, HFMA2, HMUL2 | over-approximated | `Lane.hfma2`: each half exact on f16's exact integers, both halves in one word |
| HMNMX2, HSET2, HSETP2 | missing | half2 min, max and compares |
| DADD, DFMA, DMUL | over-approximated | `Lane.dfma` on the `f64` pipe: exact on FP64's exact integers, two words out |
| DSETP | missing | FP64 compares |
| VABSDIFF, VABSDIFF4, VIMNMX (video, SIMD in a word) | missing | |
| F2F, F2I, I2F, I2I, FRND, F2FP (the PACK_AB pack), I2FP, F2IP, I2IP | missing | conversions. F2FP is W1's `cvt` |

## Uniform datapath

| SASS | Mark | Model, or what is missing |
|---|---|---|
| UIADD3, UIMAD, ULOP3, USHF, UPRMT, UPOPC, UFLO, UBREV, UISETP, USEL, ULEA, UMOV | covered | `uni` at one issue slot (measured: Sheet `[uniform_issue]`) |
| R2UR, S2UR, UR2UP, UP2UR, ULDC | coarse | uniform registers are pooled with the register file, so a move between them is a register copy |
| REDUX | covered | `Lane.redux` on the uniform datapath, a slot |
| VOTEU, UF2FP, UBMSK, USGXT | missing | |

## Moves, memory and atomics

| SASS | Mark | Model, or what is missing |
|---|---|---|
| MOV | coarse | a register copy on the `alu` |
| SHFL | covered | `Lane.shfl` on the shuffle pipe (`Res.shfl`, 32 lanes a SM-clock), beside the `alu` |
| LDG, STG, LDS, STS, LDL, STL, LD, ST | covered | `copy` of at most 4 words, direct or `indirect` (an address register plus an offset); a load whose lanes read one address is `bcast`, its words read once |
| LDSM, STSM (`ldmatrix`, `stmatrix`) | coarse | 4-word copies. The `.trans` forms move 16-bit halves across lanes, which is missing |
| MOVM (`movmatrix`) | missing | |
| LDGSTS (`cp.async`) | coarse | a copy from device memory or L2 to shared memory, with no latency |
| UTMALDG, UTMASTG, UBLKCP (TMA, `cp.async.bulk`) | coarse | a bulk copy of any length with shared memory at one end |
| UTMAREDG (TMA reduce, `cp.reduce.async.bulk`), integer add, min, max and bitwise | covered | `Opcode.red` from shared memory into L2 or device memory, priced as the bulk copy |
| UTMAREDG 64-bit, float, `inc`, `dec`; TMA multicast | missing | |
| ATOMS, ATOMG, ATOM, RED (add, exch) | covered | `atom` add and exch on shared memory, L2 and device memory. RED is an `atom` whose old value is discarded |
| RED vector forms (`red.v2`, `.v4`), integer | coarse | `Opcode.red` from up to 8 registers, any `Lane.reduces` function (integer forms are scalar on the card) |
| RED float forms (`.f32`, `.f16x2`, `.bf16x2`, `.v8`) | over-approximated | `Opcode.red` by `iadd3` and `add2`: exact integer adds, from up to 8 registers |
| ATOMS, ATOMG (min, max, and, or, xor, inc, dec, cas), float atomics | missing | |
| LDC (constant bank operands) | missing | constant-bank operands |
| TEX, TLD, TLD4, TXD, TMML, SULD, SUST, SURED, SUATOM | **missing, important** | the texture path has its own bandwidth and filtering arithmetic (weighted sums of texels) outside every resource here |

## Control and synchronization

| SASS | Mark | Model, or what is missing |
|---|---|---|
| BRA, EXIT | covered | `bra` on a register, and a thread past its text's end |
| predication (`@P`) | coarse | `pred` on a register (predicates are registers here) |
| BRX, JMX, CALL, RET | missing | indirect branches and calls, which are emulable by `bra` trees at more units |
| BSSY, BSYNC, WARPSYNC, BREAK | coarse | convergence: no data effect, and the schedule is free |
| BAR, SYNCS (mbarrier), MEMBAR, FENCE, DEPBAR, ERRBAR, CCTL | coarse | the model is sequential within a tick and latency-free, so every synchronization holds at no cost |
| S2R (`%tid`, `%clock`, `%smid`), CS2R | missing | special registers. A clock read is time as data, a side channel the model doesn't have |
| VOTE, MATCH, ELECT | missing | |
| NANOSLEEP, YIELD, PMTRIG | coarse | no data effect |

## Every opcode of BU's lists

The rows below take every opcode name in BU §4.3 (Hopper, 204 names) and §4.4 (Blackwell and Rubin, 257 names), as
the page lists them (docs.nvidia.com/cuda/cuda-binary-utilities). §4.4 doesn't say which of its
opcodes sm_120 has, so its rows are the family's: the `tcgen05` and tensor-memory rows are sm_100's. A mark here is by
the opcode's name and BU's one-line description, not yet by each variant's semantics: a *covered* opcode can still have
uncovered variants (`.64`, other atomic operations, `f16` accumulators).

### sm_90 (BU §4.3)

| Opcodes | Mark | Model |
|---|---|---|
| BRA | covered | `Instr.bra` |
| IDP4A | covered | `Lane.dp4a` |
| FADD, FADD32I, FFMA32I, FFMA, FMUL, FMUL32I | covered | `Lane.fadd`/`ffma`/`fmul` (an immediate is an operand) |
| FSEL, BREV, FLO, IADD, IADD3, IADD32I, IMAD, IMNMX, IMUL, IMUL32I, ISCADD, ISCADD32I, ISETP, LEA, LOP, LOP3, LOP32I, POPC, SHF, SHL, SHR, PRMT, SEL | covered | `Lane` integer functions (an immediate is an operand; `IMUL` is `mad` at `c = 0`, `SHL`/`SHR` are `shf`, `ISCADD` is `lea`) |
| ATOM, ATOMS, ATOMG, REDAS, REDG | covered | `Opcode.atom` (the modelled operations; others missing) |
| MOV, MOV32I, LD, LDG, LDL, LDS, LDSM, STSM, ST, STG, STL, STS, R2UR | covered | `Opcode.copy` (a lane load or store, at most 4 words), `Opcode.bcast` (a broadcast load) |
| LDGSTS, STAS, UBLKCP, UTMALDG, UTMASTG | covered | `Opcode.copy` (bulk, shared memory at one end) |
| BMMA, BGMMA | covered | `Opcode.mma` (`b1`, `Open.b1`) |
| QGMMA | covered | `Opcode.mma` (`e4m3`/`e5m2`) |
| HMMA, HGMMA | covered | `Opcode.mma` (`f16`/`bf16`/`tf32` with `f32` accumulators; `f16` accumulators missing) |
| IMMA, IGMMA | covered | `Opcode.mma` (`s8`/`u8`, the integer path) |
| UBREV, UFLO, UIADD3, UIMAD, UISETP, ULEA, ULOP, ULOP3, ULOP32I, UMOV, UPOPC, UPRMT, USEL, USHF, USHL, USHR | covered | `Opcode.uni` (integer, a whole issue slot) |
| PLOP3, PSETP, P2R, R2P, UP2UR, UPLOP3, UPSETP, UR2UP | covered | predicates are registers: `Lane.lop3`, `setp`, register copies |
| LDC, ULDC | coarse | a copy from device memory (constant banks) |
| SHFL | covered | `Lane.shfl` on the shuffle pipe (`Res.shfl`) |
| FENCE, LDGDEPBAR, SYNCS, CCTL, CCTLL, ERRBAR, MEMBAR, CCTLT, UCGABAR_ARV, UCGABAR_WAIT, USETMAXREG, WARPGROUP, WARPGROUPSET, UBLKPF, UTMACCTL, UTMACMDFLUSH, UTMAPF, ACQBULK, BMOV, BPT, BREAK, BSSY, BSYNC, CGAERRBAR, ENDCOLLECTIVE, EXIT, KILL, NANOSLEEP, PREEXIT, WARPSYNC, YIELD, BAR, DEPBAR, NOP, PMTRIG | coarse | no data effect: the schedule is free, so a program may omit it |
| UCLEA, UIADD3.64, ULEPC | missing | 64-bit uniform and immediate moves, `UCLEA`, `ULEPC` |
| FCHK, FMNMX, FSET, FSETP, FSWZADD | missing | FP32 compare, min/max, check, swizzle |
| DADD, DFMA, DMMA, DMUL | over-approximated | `Lane.dfma`, `Opcode.dmma` |
| DSETP | missing | FP64 compares |
| MUFU | missing | MUFU (transcendentals) |
| UBLKRED, UTMAREDG (integer add, min, max, and, or, xor) | covered | `Opcode.red` from shared memory into L2 or device memory, priced as the bulk copy |
| UBLKRED, UTMAREDG (64-bit, float, `inc`, `dec`; into another CTA's shared memory) | missing | other functions |
| LDGMC | missing | multimem loads |
| F2F, F2I, I2F, I2I, I2IP, I2FP, F2IP, FRND | missing | conversions |
| QSPC, ELECT, GETLMEMBASE, LEPC, SETCTAID, SETLMEMBASE | missing | elect, address-space query, local-memory base, block id, work id, virtual count |
| HADD2, HADD2_32I, HFMA2, HFMA2_32I, HMUL2, HMUL2_32I | over-approximated | `Lane.hfma2` |
| HMNMX2, HSET2, HSETP2 | missing | half2 min, max and compares |
| BRX, BRXU, CALL, JMP, JMX, JMXU, RET, RPCMOV | missing | indirect branches, calls, returns, jumps |
| BMSK, IABS, IDP, SGXT, UBMSK, USGXT | missing | integer abs, sign extend, bit mask, `IDP.2A` |
| REDUX | covered | `Lane.redux` |
| MOVM, MATCH, VOTEU, VOTE | missing | matrix move, match, vote |
| S2UR, B2R, CS2R, S2R | missing | special registers (thread and block ids, clocks) |
| SUATOM, SULD, SURED, SUST | missing | surfaces |
| TEX, TLD, TLD4, TMML, TXD, TXQ | missing | texture path |
| UF2FP | missing | uniform FP and conversions (the instances allow integer `uni` only) |
| VABSDIFF, VABSDIFF4, VHMNMX, VIADD, VIADDMNMX, VIMNMX, VIMNMX3 | missing | video and three-way min/max integer ops |

204 opcodes: 85 covered, 38 coarse, 81 missing.

### Blackwell additions over sm_90 (BU §4.4; sm_120 has a subset)

| Opcodes | Mark | Model |
|---|---|---|
| OMMA | covered | `Opcode.mma` (`e2m1`, block-scaled dots `Open.dot`) |
| QMMA | covered | `Opcode.mma` (`e4m3`/`e5m2`) |
| UIMNMX | covered | `Opcode.uni` (integer, a whole issue slot) |
| LDCU | coarse | a copy from device memory (constant banks) |
| SYNCSU, USYNCS, UBLKL2CCTL, UTMAL2CCTL, ACQSHMINIT | coarse | no data effect: the schedule is free, so a program may omit it |
| MOV64IUR, UMOV.64 | missing | 64-bit uniform and immediate moves, `UCLEA`, `ULEPC` |
| FMNMX3 | missing | FP32 compare, min/max, check, swizzle |
| REDS, UREDGR | missing | shared and uniform-register reductions (`Opcode.red`'s forms cover the integer bulk ones on sm_90) |
| UGETNEXTWORKID, UMEMSETS, UVIRTCOUNT | missing | elect, address-space query, local-memory base, block id, work id, virtual count |
| FHADD, FHADD2, FHFMA, FHFMA2, FHMUL2 | missing | half2 and mixed half arithmetic |
| UIABS | missing | integer abs, sign extend, bit mask, `IDP.2A` |
| CREDUX | missing | matrix move, match, vote, reduce |
| FADD2, FFMA2, FMUL2 | missing | paired FP32 |
| DECOMPRESS, SPARSIFY | missing | sparsity, decompression |
| CS2UR | missing | special registers (thread and block ids, clocks) |
| LDT, LDTM, STT, STTM, UTCATOMSWS, UTCBAR, UTCCP, UTCHMMA, UTCIMMA, UTCOMMA, UTCQMMA, UTCSHIFT | missing | tensor memory and `tcgen05` (not on sm_120: PTX ISA gives `tcgen05` to sm_100a, sm_101a and sm_103a) |
| UFADD, UF2F, UF2I, UF2IP, UFFMA, UFMNMX, UFMUL, UFRND, UFSEL, UFSET, UFSETP, UI2F, UI2FP, UI2I, UI2IP | missing | uniform FP and conversions (the instances allow integer `uni` only) |
| USTGR | missing | uniform global store |
| UVIADD, UVIMNMX, VISET | missing | video and three-way min/max integer ops |

61 opcodes: 3 covered, 6 coarse, 52 missing.
