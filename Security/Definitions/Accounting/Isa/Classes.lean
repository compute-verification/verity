import Definitions.Accounting.Isa.Table

/-!
# The classes devices share (trusted)

The instruction classes whose rows don't depend on the device (note:20261019T0520Z-draft-mm-isa): the mma.sync forms
(`Sync`) and every class outside the tensor cores (`Base`), each row the reading of its PTX section. A device's table
(`Isa/Sm120.lean`, `Isa/Sm90.lean`, `Isa/Sm100.lean`) is these and its own tensor classes; a class a device lacks has no
instances there.

The classes are DS-120's census classes (`benchmarks/pouw/facet_break/loads.py`'s `class_of`, the names of
`machine_model/data/census/DS-120.csv` and `floors.csv`), with four that `isa_escape.py` attacks and `class_of` puts in
no class: `BMMA168256` (mma.sync .b1), `IMMAS4` (mma.sync and mma.sp .s4), `REDUX` (redux.sync) and `VSIMD` (the SIMD
video instructions, §9.7.20.2); `LDCU128`, LDCU.128, which `class_of` puts in UNI though it writes four uniform
registers; the heads the census never probed: `DMMA` (mma.sync .f64), global stores (`STG32` … `STG256`), surface
stores and atomics (`SUST`, `SUATOM`) and vote and match (`VOTE`); and the bulk transfers past sm_120's shared memory:
`BULKMAX` and `BULKRED` (a copy or reduction up to sm_90a's and sm_100a's 227 KB) and `UTMAMC` (a multicast tensor
copy, its box into up to 16 CTAs).

**An opcode's class.** `class_of` places the census's forms. Every other opcode that writes a 32-bit word (the
instruction-set lists of the CUDA Binary Utilities, §4.3 Hopper and §4.4 Blackwell, which
`benchmarks/pouw/theory/isa_table.py` places one by one) is in the class of its unit at the next width up (a class's
`words` bounds every form in it):

* memory: LD and LDL in the LDG class of their width; ST, STL and USTGR in the STG class; STS.32 and STS.64 in
  STS128; STAS in DSMST4P; RED.E and UREDGR in RED32 or RED64; REDS in ATOMS32; LDGMC (a reducing load, four words a
  lane at most) in ATOMG64; the mbarrier operations (SYNCS, SYNCSU, USYNCS) in ATOMS64.
* bulk, the U-prefixed opcodes `class_of`'s uniform rule puts in UNI: UBLKRED and UTMAREDG in TMAred512, any larger in
  BULKRED; UTMALDG and UTMASTG in the UTMA class of their box; UBLKCP in UBLKCP4K, any larger in BULKMAX, a multicast
  one in UTMAMC; UMEMSETS in BULKMAX.
* tensor: UTC*MMA in `Tc05`; LDT, LDTM, STT, STTM, UTCCP and UTCSHIFT in `Tmem`; BMMA, which `class_of` puts in XU, in
  BMMA168256; DMMA in DMMA.
* texture: SULD in SULD1 or SULD4, TXQ in TEX1, TMML in TEX2, TXD in TEX4.
* warp and uniform: MATCH in VOTE; CREDUX in REDUX; UTCATOMSWS and every other U-prefixed opcode in UNI or UNI64
  (`class_of`).
* lane: FFMA2, FADD2, FMUL2, FSWZADD and the FH forms in FFMA; FSET, HSET2, FMNMX3, VHMNMX, VIADDMNMX and VIMNMX3 in
  VIMNMX; VIADD, IADD32I, ISCADD, SHL and SHR in ALU; IDP, IDP4A and IMUL in IMAD; VISET in VSIMD; MOV64IUR in MOV64;
  and every other lane instruction that writes at most two words a lane and multiplies nothing (P2R, R2P, IABS, SGXT,
  BMSK, S2R, CS2R, MOVM, ELECT, B2R, BMOV, RPCMOV, GETLMEMBASE, LEPC) in NOROW.

An instruction that writes no 32-bit word (a SETP's predicate, a branch, a barrier, a fence, a prefetch or cache
control) heads no word and is in no class. Blackwell's DECOMPRESS and SPARSIFY, whose widths no document gives, are in
no class yet: a head of either falsifies `Accounting.Assumptions.Isa.IsaComplete`.

A row's `words` is its class's widest form's: a lane class two words a lane (`Row.lane`), an atomic that returns its
old value its memory and register words together (CAS.128: four of each a lane, 256 a warp instruction), a load or
store its bytes over four, a TMA box or bulk copy its largest box. Where `floors.csv` prices a side class at fewer words
an instruction than the row (the atomics), a model reading `floors.csv` doesn't read this table
(`Accounting.Isa.ReadsTable`).
-/

namespace Accounting.Isa

/-- **The mma.sync forms** (PTX ISA 9.4 §9.7.16): DS-120's MMA classes, `BMMA168256`, `IMMAS4` and `DMMA` (mma.sync
.f64). -/
inductive Sync
  | IMMA16832 | IMMA16816 | IMMA8816 | IMMASP16864 | IMMASP16832 | MMA
  | HMMA16816 | HMMASP16832 | HMMA16816F16 | HMMA1688T | HMMA1684T
  | QMMA16832 | QMMASF16832 | QMMASP16864 | OMMASF16864
  | BMMA168256 | IMMAS4 | DMMA
  deriving DecidableEq, Fintype, Repr

/-- **The mma.sync rows** (§9.7.16.5.14, block scale §9.7.16.3, sparse §9.7.16.6): each m16n8 form writes its 32-bit
accumulator, 128 words a warp instruction; m8n8k16 (IMMA.8816) 64; HMMA.16816.F16's f16 accumulator, two to a register,
64; DMMA's m16n8 f64 accumulator, two registers an element, 256. -/
def Sync.row : Sync → Row
  | .IMMA16832 => Row.mma .mma .s8 .s8 .s32 128
  | .IMMA16816 => Row.mma .mma .s8 .s8 .s32 128
  | .IMMA8816 => Row.mma .mma .s8 .s8 .s32 64
  | .IMMASP16864 => Row.mma .mma .s8 .s8 .s32 128
  | .IMMASP16832 => Row.mma .mma .s8 .s8 .s32 128
  | .MMA => Row.mma .mma .s8 .s8 .s32 128
  | .HMMA16816 => Row.mma .mma .f16 .f16 .f32 128
  | .HMMASP16832 => Row.mma .mma .f16 .f16 .f32 128
  | .HMMA16816F16 => Row.mma .mma .f16 .f16 .f16 64
  | .HMMA1688T => Row.mma .mma .tf32 .tf32 .f32 128
  | .HMMA1684T => Row.mma .mma .tf32 .tf32 .f32 128
  | .QMMA16832 => Row.mma .mma .e4m3 .e4m3 .f32 128
  | .QMMASF16832 => Row.mma .mma .e4m3 .e4m3 .f32 128
  | .QMMASP16864 => Row.mma .mma .e4m3 .e4m3 .f32 128
  | .OMMASF16864 => Row.mma .mma .e2m1 .e2m1 .f32 128
  | .BMMA168256 => Row.mma .mma .b1 .b1 .s32 128
  | .IMMAS4 => Row.mma .mma .s4 .s4 .s32 128
  | .DMMA => Row.mma .mma .f64 .f64 .f64 256

/-- **The classes outside the tensor cores**: DS-120's other classes, `REDUX`, `VSIMD`, `LDCU128`, the global and
surface stores, surface atomics and `VOTE`. -/
inductive Base
  | UNI | UNI64 | LDCU128
  | ALU | IMAD | FFMA | HFMA2 | VIMNMX | F2FS | XU | SEL | NOROW | VSIMD
  | IMADW | ALU64 | W2 | DMUL | CVT64 | MOV64
  | SHFL | REDUX | VOTE
  | LDS32 | LDS64 | LDS128
  | LDG32 | LDG64 | LDG128 | LDG128CG | LDG128NA | LDG256 | LDG256CG
  | LDC64 | LDCDIV | LDCUNI | LDGSTS16
  | LDSM_X1 | LDSM_X2 | LDSM_X4 | LDSM_X1T | LDSM_X2T | LDSM_X4T
  | LDSM4_X1 | LDSM4_X2 | LDSM4_X4 | LDSM4T_X1 | LDSM4T_X2
  | LDSM6_X1 | LDSM6_X2 | LDSM6_X4 | LDSM6T_X1 | LDSM6T_X2
  | LDSM8T_X1 | LDSM8T_X2
  | STS128 | DSMST4P | STG32 | STG64 | STG128 | STG256
  | STSM_X1 | STSM_X2 | STSM_X4 | STSM_X1T | STSM_X2T | STSM_X4T
  | STSM8T_X1 | STSM8T_X2 | STSM8T_X4
  | ATOMG32 | ATOMG64 | ATOMS32 | ATOMS64 | ATOME32 | ATOME32P | ATOME64
  | RED32 | RED64 | REDV | REDG32 | REDG64 | REDAS32 | REDAS64
  | TMA512 | TMAred512 | UBLKCP4K
  | UTMA | UTMA8 | UTMA512 | UTMA1K | UTMA2K | UTMA4K | UTMA8K | UTMA16K | UTMA32K | UTMA48K | UTMA64K | UTMABIG
  | BULKMAX | BULKRED | UTMAMC
  | G4_16 | G4_32 | G4_256 | G4_1K
  | TEX1 | TEX2 | TEX4 | TEXBIL
  | TLD1 | TLD2 | TLD2D1 | TLD2D4 | TLD4G | TLD4w
  | SULD1 | SULD4 | SUST | SURED | SUATOM
  deriving DecidableEq, Fintype, Repr

/-- **The rows outside the tensor cores**, family by family (`Family` cites each family's PTX sections):

* `uniform`: a uniform instruction writes one uniform register (UNI), or a pair (UNI64); UIMAD multiplies. LDCU.128
  loads four from a constant bank at a uniform address the program computes (a gather).
* `lane` (§9.7.1–§9.7.9, §9.7.20): at most two words a lane. IMAD.WIDE, IADD.64, DADD/DFMA, DMUL and the 64-bit
  conversions are side units at their two words a lane; MOV.64 copies.
* `warp`: SHFL selects a lane's word (§9.7.10.6), one word a lane; VOTE writes a ballot or match mask, one word a
  lane (§9.7.15.10, §9.7.15.11); REDUX writes one uniform word (§9.7.15.13).
* `memory` (ld §9.7.10.8, st §9.7.10.11, cp.async §9.7.10.28, ldmatrix §9.7.16.5.15, stmatrix §9.7.16.5.16, atom
  §9.7.15.5, red §9.7.15.6, red.async §9.7.15.7): its bytes over four, every lane.
* `bulk` (§9.7.10.28.4–§9.7.10.28.5): its box's or copy's bytes over four, at most the shared memory's (99 KB on
  sm_120, 227 KB on sm_90a and sm_100a), and a multicast copy's times its CTAs (at most 16, `.multicast::cluster`).
* `texture` (tex §9.7.12.3, tld4 §9.7.12.4, suld §9.7.13.1, sust §9.7.13.2, sured §9.7.13.3): its texels' or
  elements' words; a surface atomic (SUATOM, CUDA Binary Utilities' "Atomic Op on Surface Memory") its old value's
  and its memory words, at most four a lane. -/
def Base.row : Base → Row
  | .UNI =>
    { fam := .uniform, exec := .uniform, arith := .mul, dataAddr := true, a := .s32, b := .s32, acc := .s32,
      words := 1, pipe := .issue, route := .writer }
  | .UNI64 =>
    { fam := .uniform, exec := .uniform, arith := .mul, dataAddr := true, a := .s32, b := .s32, acc := .s64,
      words := 2, pipe := .issue, route := .writer }
  | .LDCU128 => Row.side .uniform .uniform .move true 4
  | .ALU => Row.lane .lane .add false .alu
  | .IMAD => Row.lane .lane .mul false .imad .s32 .s32 .s32
  | .FFMA => Row.lane .lane .mul false .issue .f32 .f32 .f32
  | .HFMA2 => Row.lane .lane .mul false .issue .f16 .f16 .f16
  | .VIMNMX => Row.lane .lane .add false .issue
  | .F2FS => Row.lane .lane .convert false .issue
  | .XU => Row.lane .lane .unary false .xu
  | .SEL => Row.lane .lane .move true .issue
  | .NOROW => Row.lane .lane .logic false .issue
  | .VSIMD => Row.lane .lane .add false .issue
  | .IMADW => Row.side .lane .lane .mul false 64 (a := .s32) (b := .s32) (acc := .s64)
  | .ALU64 => Row.side .lane .lane .add false 64
  | .W2 => Row.side .lane .lane .mul false 64 (a := .f64) (b := .f64) (acc := .f64)
  | .DMUL => Row.side .lane .lane .mul false 64 (a := .f64) (b := .f64) (acc := .f64)
  | .CVT64 => Row.side .lane .lane .convert false 64
  | .MOV64 => Row.copy .lane 64
  | .SHFL => Row.lane .warp .move true .shfl
  | .REDUX => Row.side .warp .warp .add false 1
  | .VOTE => Row.lane .warp .logic true .issue
  | .LDS32 => Row.side .memory .lane .move true 32 .smem
  | .LDS64 => Row.side .memory .lane .move true 64 .smem
  | .LDS128 => Row.side .memory .lane .move true 128 .smem
  | .LDG32 => Row.side .memory .lane .move true 32
  | .LDG64 => Row.side .memory .lane .move true 64
  | .LDG128 => Row.side .memory .lane .move true 128
  | .LDG128CG => Row.side .memory .lane .move true 128
  | .LDG128NA => Row.side .memory .lane .move true 128
  | .LDG256 => Row.side .memory .lane .move true 256
  | .LDG256CG => Row.side .memory .lane .move true 256
  | .LDC64 => Row.side .memory .lane .move true 64
  | .LDCDIV => Row.side .memory .lane .move true 32
  | .LDCUNI => Row.side .memory .lane .move true 32
  | .LDGSTS16 => Row.side .memory .lane .move true 128
  | .LDSM_X1 => Row.side .memory .lane .move true 32 .smem
  | .LDSM_X2 => Row.side .memory .lane .move true 64 .smem
  | .LDSM_X4 => Row.side .memory .lane .move true 128 .smem
  | .LDSM_X1T => Row.side .memory .lane .move true 32 .smem
  | .LDSM_X2T => Row.side .memory .lane .move true 64 .smem
  | .LDSM_X4T => Row.side .memory .lane .move true 128 .smem
  | .LDSM4_X1 => Row.side .memory .lane .convert true 32
  | .LDSM4_X2 => Row.side .memory .lane .convert true 64
  | .LDSM4_X4 => Row.side .memory .lane .convert true 128
  | .LDSM4T_X1 => Row.side .memory .lane .convert true 64
  | .LDSM4T_X2 => Row.side .memory .lane .convert true 128
  | .LDSM6_X1 => Row.side .memory .lane .convert true 32
  | .LDSM6_X2 => Row.side .memory .lane .convert true 64
  | .LDSM6_X4 => Row.side .memory .lane .convert true 128
  | .LDSM6T_X1 => Row.side .memory .lane .convert true 64
  | .LDSM6T_X2 => Row.side .memory .lane .convert true 128
  | .LDSM8T_X1 => Row.side .memory .lane .move true 64
  | .LDSM8T_X2 => Row.side .memory .lane .move true 128
  | .STS128 => Row.side .memory .lane .move true 128 .smem
  | .DSMST4P => Row.side .memory .lane .move true 128
  | .STG32 => Row.side .memory .lane .move true 32
  | .STG64 => Row.side .memory .lane .move true 64
  | .STG128 => Row.side .memory .lane .move true 128
  | .STG256 => Row.side .memory .lane .move true 256
  | .STSM_X1 => Row.side .memory .lane .move true 32
  | .STSM_X2 => Row.side .memory .lane .move true 64
  | .STSM_X4 => Row.side .memory .lane .move true 128
  | .STSM_X1T => Row.side .memory .lane .move true 32
  | .STSM_X2T => Row.side .memory .lane .move true 64
  | .STSM_X4T => Row.side .memory .lane .move true 128
  | .STSM8T_X1 => Row.side .memory .lane .move true 32
  | .STSM8T_X2 => Row.side .memory .lane .move true 64
  | .STSM8T_X4 => Row.side .memory .lane .move true 128
  | .ATOMG32 => Row.side .memory .lane .add true 256
  | .ATOMG64 => Row.side .memory .lane .add true 128
  | .ATOMS32 => Row.side .memory .lane .add true 256
  | .ATOMS64 => Row.side .memory .lane .add true 128
  | .ATOME32 => Row.side .memory .lane .add true 256
  | .ATOME32P => Row.side .memory .lane .add true 256
  | .ATOME64 => Row.side .memory .lane .add true 128
  | .RED32 => Row.side .memory .async .add true 32
  | .RED64 => Row.side .memory .async .add true 64
  | .REDV => Row.side .memory .async .add true 128
  | .REDG32 => Row.side .memory .async .add true 32
  | .REDG64 => Row.side .memory .async .add true 64
  | .REDAS32 => Row.side .memory .async .add true 32
  | .REDAS64 => Row.side .memory .async .add true 64
  | .TMA512 => Row.side .bulk .async .move true 128
  | .TMAred512 => Row.side .bulk .async .add true 128
  | .UBLKCP4K => Row.side .bulk .async .move true 1024
  | .UTMA => Row.side .bulk .async .move true 4
  | .UTMA8 => Row.side .bulk .async .move true 8
  | .UTMA512 => Row.side .bulk .async .move true 128
  | .UTMA1K => Row.side .bulk .async .move true 256
  | .UTMA2K => Row.side .bulk .async .move true 512
  | .UTMA4K => Row.side .bulk .async .move true 1024
  | .UTMA8K => Row.side .bulk .async .move true 2048
  | .UTMA16K => Row.side .bulk .async .move true 4096
  | .UTMA32K => Row.side .bulk .async .move true 8192
  | .UTMA48K => Row.side .bulk .async .move true 12288
  | .UTMA64K => Row.side .bulk .async .move true 16384
  | .UTMABIG => Row.side .bulk .async .move true 25344
  | .BULKMAX => Row.side .bulk .async .move true 58112
  | .BULKRED => Row.side .bulk .async .add true 58112
  | .UTMAMC => Row.side .bulk .async .move true 929792
  | .G4_16 => Row.side .bulk .async .move true 16
  | .G4_32 => Row.side .bulk .async .move true 32
  | .G4_256 => Row.side .bulk .async .move true 256
  | .G4_1K => Row.side .bulk .async .move true 1024
  | .TEX1 => Row.side .texture .async .interp true 32
  | .TEX2 => Row.side .texture .async .interp true 64
  | .TEX4 => Row.side .texture .async .interp true 128
  | .TEXBIL => Row.side .texture .async .interp true 128
  | .TLD1 => Row.side .texture .async .convert true 32
  | .TLD2 => Row.side .texture .async .convert true 64
  | .TLD2D1 => Row.side .texture .async .convert true 32
  | .TLD2D4 => Row.side .texture .async .convert true 128
  | .TLD4G => Row.side .texture .async .convert true 128
  | .TLD4w => Row.side .texture .async .convert true 128
  | .SULD1 => Row.side .texture .async .convert true 32
  | .SULD4 => Row.side .texture .async .convert true 128
  | .SUST => Row.side .texture .async .move true 128
  | .SURED => Row.side .texture .async .add true 32
  | .SUATOM => Row.side .texture .async .add true 128

end Accounting.Isa
