"""ISA-TABLE: the Lean instruction table held against the Python and data A7 rested on, and every documented SASS opcode
placed in one of its classes (note:20261019T0520Z-draft-mm-isa).

A7 (`Accounting.Assumptions.TS.SheetSemantics`) is now `Accounting.SecurityProofs.Isa.IsaSheet` over a device's table
(`Security/Definitions/Accounting/Isa/`): Lean decides each row's discipline, and what stays trusted is one premise per
row family and `IsaComplete`, that every head is an instance of a class. This script checks what Lean doesn't read:

  - the table: the Lean source's classes and rows parse (route and words), sm_120's classes are DS-120's census kinds
    and the sixteen `Isa/Classes.lean` adds, sm_90's and sm_100's censuses are classes of their tables, and the one
    class off every route is MOV.64's;
  - the floors: every kind `floors.csv` prices is a class, and a side or writer class's words there are at least its
    row's (`ReadsTable` holds a model's side vectors and ω to the row), but for the atomics, which it prices below
    their old value's words (F6);
  - isa_escape: every candidate `isa_escape.py` closes sits in classes on a priced route, none traced, and the nine
    RT-MACHINE's branch adds (DMMA's among them) are placed ahead of landing (`ESCAPE_AHEAD`);
  - completeness evidence for `IsaComplete`: every opcode the CUDA Binary Utilities 13.4 list (§4.3 Hopper, §4.4
    Blackwell and Rubin; last updated 12 Oct 2026) is placed in classes of its device's table (`PLACE`) or read as
    writing no 32-bit word (`NO_WORD`), and an opcode in neither fails, with no list of exceptions; sm_120's is
    Blackwell's list without its tensor core memory instructions; and where `class_of` (`facet_break/loads.py`) places
    a bare mnemonic it agrees, but for the disagreements `CLASS_OF_WRONG` names, and a dotted form it places in a class
    whose row may not bound it is moved by the map to one that does (`FORMS_MOVED`);
  - the same against RT-MACHINE-2's hand triage (`TRIAGE_ART`, derived to `machine_model/data/rt_machine_triage.json`)
    of the 175 Blackwell opcodes A7's menu didn't name: each is placed on a priced route or writes no word, a traced
    class only for a mover, an opcode it finds can multiply (`TRIAGE_PRICED`) placed, and its opcodes absent on sm_120
    are the tensor core memory list, outside sm_120's table;
  - A6-WIDE (note:20261019T0457Z-draft-fpx-machine): no opcode of the lists' access categories (`ACCESS`: loads,
    LDSM, atomics, texture, surface and tensor-memory copies, the wide destinations among them) is in a vector class,
    whose heads A6 bounds at two words a lane, but MATCH (`ACCESS_LANE`: VOTE's row, one word a lane); Lean decides the
    same of the table's rows (`sm120_lane`, `sm90_lane`, `sm100_lane`);
  - the same against what the libraries emit: RT-ISA's SASS census (`CORPUS_ART`, v3: 164 binaries on two nodes
    under CUDA 13.0, derived to `machine_model/data/rt_isa_corpus.json`), every base on each device placed, read as
    writing no word or named undocumented (`UNDOCUMENTED`: ARRIVES, F2FP, sm_90's USETSHMSZ), its UNDERPRICED forms
    `COUNT_TWO`'s (FFMA2 and FMUL2 count as two FFMA, RT-ISA-6), its word flag against the map's no-word reading
    (`WORD_DISAGREE`), its unpriced MMA forms (GAP-TENSOR: DMMA, UTCOMMA) in writer rows, and each vector class at two
    words a lane exactly where a corpus form writes a pair (`LANE_PAIR`).

A placement is a words bound only: a class's rate is the data sheet's, measured for the forms its census probed.

Run: `python isa_table.py [--mutations] [--verdict PATH]`, or `--write-corpus DIR` (`--write-triage DIR`) to rederive
the corpus (triage) fixture from its artifact fetched to DIR. Stdlib and exact_kit; reads the Lean table, floors.csv,
the censuses and the two fixtures.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from exact_kit.checks import Checks

POUW = Path(__file__).resolve().parents[1]
ROOT = POUW.parents[1]
ISA = ROOT / "Security" / "Definitions" / "Accounting" / "Isa"
DATA = POUW / "machine_model" / "data"
sys.path.insert(0, str(POUW / "facet_break"))
sys.path.insert(0, str(POUW / "theory"))

from loads import class_of  # noqa: E402

#: the classes `Isa/Classes.lean` adds to DS-120's census kinds
ADDED = set("BMMA168256 IMMAS4 DMMA REDUX VSIMD LDCU128 STG32 STG64 STG128 STG256 SUST SUATOM VOTE BULKMAX BULKRED "
            "UTMAMC".split())
#: sm_90's and sm_100's census names for the mma.sync forms, as classes (the wider form where the name leaves the
#: accumulator open)
CENSUS_ALIAS = {"IMMA16832_S8": "IMMA16832", "IMMA16816_S8": "IMMA16816", "BMMA168256_B1": "BMMA168256",
                "HMMA16816_F16": "HMMA16816", "HMMA16816_BF16": "HMMA16816", "HMMA1688_TF32": "HMMA1688T"}
#: the classes floors.csv prices below their row: an atomic returns its old value and writes memory, four words each a
#: lane for CAS.128, where floors.csv counts the returned word (F6)
FLOORS_BELOW = set("ATOMG32 ATOMG64 ATOMS32 ATOMS64 ATOME32 ATOME32P ATOME64".split())

#: isa_escape.py's candidates (by name prefix) and the classes their instances are in
ESCAPE = {
    "redux.sync": "REDUX", "L2 reduction": "RED32 RED64 REDV REDG32 REDG64 ATOMG32 ATOMG64 ATOME32 ATOME64",
    "TMA bulk reduce": "TMAred512 BULKRED", "IDP4A": "IMAD", "IDP2A": "IMAD",
    "sparse IMMA": "IMMASP16864 IMMASP16832", "BMMA": "BMMA168256", "s4 IMMA": "IMMAS4",
    "FP16/BF16 MMA, f32": "HMMA16816 HMMASP16832", "FP16/BF16 MMA, f16": "HMMA16816F16",
    "FP8 MMA": "QMMA16832 QMMASF16832 QMMASP16864", "FP4 MMA": "OMMASF16864", "DPX": "VIMNMX",
    "video / SIMD ALU": "VSIMD VIMNMX", "texture filtering": "TEX1 TEX2 TEX4 TEXBIL TLD1 TLD2 TLD2D1 TLD2D4 TLD4G TLD4w",
    "async copy / TMA bulk copy": "LDGSTS16 TMA512 UBLKCP4K UTMA UTMA512 UTMABIG BULKMAX",
    "ldmatrix / stmatrix": "LDSM_X4 LDSM_X4T STSM_X4 STSM_X4T", "PRMT / FMNMX / LOP3": "ALU VIMNMX",
    "IMAD.WIDE": "IMADW", "uniform datapath": "UNI UNI64",
    "FP6 MMA": "QMMA16832", "FP64 MMA": "DMMA", "IADD.64 / IMNMX.64 / SEL.64": "ALU64 VIMNMX SEL",
    "tcgen05.cp": "UTCCP", "tcgen05.shift": "UTCSHIFT", "tcgen05.ld / tcgen05.st": "LDTM_32X32B_X128 LDTM STTM",
    "block-scaled FP6 MMA": "UTCQMMA_SF_E3M2 UTCQMMA_SF_E2M3", "block-scaled FP4 MMA": "UTCOMMA_SF UTCOMMA_2CTA",
    "weight decompression engine": "NOROW",
}
#: the placements whose candidates are on `cursor/rt-machine-ea24` (RT-MACHINE's, with #2704's B200 word ISA) and not
#: yet in `isa_escape.py` here, so the check passes before and after they land
ESCAPE_AHEAD = {"FP6 MMA", "FP64 MMA", "IADD.64 / IMNMX.64 / SEL.64", "tcgen05.cp", "tcgen05.shift",
                "tcgen05.ld / tcgen05.st", "block-scaled FP6 MMA", "block-scaled FP4 MMA",
                "weight decompression engine"}

#: CUDA Binary Utilities 13.4, §4.3 "Hopper Instruction Set" and §4.4 "Blackwell and Rubin Instruction Set", by category
HOPPER = """
Floating Point: FADD FADD32I FCHK FFMA32I FFMA FMNMX FMUL FMUL32I FSEL FSET FSETP FSWZADD MUFU HADD2 HADD2_32I HFMA2
  HFMA2_32I HMMA HMNMX2 HMUL2 HMUL2_32I HSET2 HSETP2 DADD DFMA DMMA DMUL DSETP
Integer: BMMA BMSK BREV FLO IABS IADD IADD3 IADD32I IDP IDP4A IMAD IMMA IMNMX IMUL IMUL32I ISCADD ISCADD32I ISETP LEA LOP
  LOP3 LOP32I POPC SHF SHL SHR VABSDIFF VABSDIFF4 VHMNMX VIADD VIADDMNMX VIMNMX VIMNMX3
Conversion: F2F F2I I2F I2I I2IP I2FP F2IP FRND
Movement: MOV MOV32I MOVM PRMT SEL SGXT SHFL
Predicate: PLOP3 PSETP P2R R2P
Load/Store: FENCE LD LDC LDG LDGDEPBAR LDGMC LDGSTS LDL LDS LDSM STSM ST STG STL STS STAS SYNCS MATCH QSPC ATOM ATOMS
  ATOMG REDAS REDG CCTL CCTLL ERRBAR MEMBAR CCTLT
Uniform Datapath: R2UR REDUX S2UR UBMSK UBREV UCGABAR_ARV UCGABAR_WAIT UCLEA UF2FP UFLO UIADD3 UIADD3.64 UIMAD UISETP
  ULDC ULEA ULEPC ULOP ULOP3 ULOP32I UMOV UP2UR UPLOP3 UPOPC UPRMT UPSETP UR2UP USEL USETMAXREG USGXT USHF USHL USHR
  VOTEU
Warpgroup: BGMMA HGMMA IGMMA QGMMA WARPGROUP WARPGROUPSET
Tensor Memory Access: UBLKCP UBLKPF UBLKRED UTMACCTL UTMACMDFLUSH UTMALDG UTMAPF UTMAREDG UTMASTG
Texture: TEX TLD TLD4 TMML TXD TXQ
Surface: SUATOM SULD SURED SUST
Control: ACQBULK BMOV BPT BRA BREAK BRX BRXU BSSY BSYNC CALL CGAERRBAR ELECT ENDCOLLECTIVE EXIT JMP JMX JMXU KILL
  NANOSLEEP PREEXIT RET RPCMOV WARPSYNC YIELD
Miscellaneous: B2R BAR CS2R DEPBAR GETLMEMBASE LEPC NOP PMTRIG S2R SETCTAID SETLMEMBASE VOTE
"""
BLACKWELL = """
Floating Point: FADD FADD2 FADD32I FCHK FFMA32I FFMA FFMA2 FHADD FHADD2 FHFMA FHFMA2 FHMUL2 FMNMX FMNMX3 FMUL FMUL2
  FMUL32I FSEL FSET FSETP FSWZADD MUFU HADD2 HADD2_32I HFMA2 HFMA2_32I HMMA HMNMX2 HMUL2 HMUL2_32I HSET2 HSETP2 DADD
  DFMA DMMA DMUL DSETP OMMA QMMA
Integer: BMSK BREV DECOMPRESS FLO IABS IADD IADD3 IADD32I IDP IDP4A IMAD IMMA IMNMX IMUL IMUL32I ISCADD ISCADD32I ISETP
  LEA LOP LOP3 LOP32I POPC SHF SHL SHR SPARSIFY VABSDIFF VABSDIFF4 VHMNMX VIADD VIADDMNMX VIMNMX VIMNMX3
Conversion: F2F F2I I2F I2I I2IP I2FP F2IP FRND
Movement: MOV MOV32I MOV64IUR MOVM PRMT SEL SGXT SHFL
Predicate: PLOP3 PSETP P2R R2P
Load/Store: FENCE LD LDC LDG LDGDEPBAR LDGMC LDGSTS LDL LDS LDSM STSM ST STG STL STS STAS SYNCS SYNCSU USYNCS MATCH QSPC
  ATOM ATOMS ATOMG REDAS REDS REDG CCTL CCTLL ERRBAR MEMBAR CCTLT
Uniform Datapath: CREDUX CS2UR LDCU R2UR REDUX S2UR UBMSK UBREV UCGABAR_ARV UCGABAR_WAIT UCLEA UFADD UF2F UF2FP UF2I
  UF2IP UFFMA UFLO UFMNMX UFMUL UFRND UFSEL UFSET UFSETP UI2F UI2FP UI2I UI2IP UIABS UIMNMX UIADD3 UIADD3.64 UIMAD
  UISETP ULEA ULEPC ULOP ULOP3 ULOP32I UMOV UMOV.64 UP2UR UPLOP3 UPOPC UPRMT UPSETP UR2UP USEL USETMAXREG USGXT USHF
  USHL USHR UGETNEXTWORKID UMEMSETS UREDGR USTGR UVIADD UVIMNMX UVIRTCOUNT VOTEU
Tensor Memory Access: UBLKCP UBLKL2CCTL UBLKPF UBLKRED UTMACCTL UTMACMDFLUSH UTMAL2CCTL UTMALDG UTMAPF UTMAREDG UTMASTG
Tensor Core Memory: LDT LDTM STT STTM UTCATOMSWS UTCBAR UTCCP UTCHMMA UTCIMMA UTCOMMA UTCQMMA UTCSHIFT
Texture: TEX TLD TLD4 TMML TXD TXQ
Surface: SUATOM SULD SURED SUST
Control: ACQBULK ACQSHMINIT BMOV BPT BRA BREAK BRX BRXU BSSY BSYNC CALL CGAERRBAR ELECT ENDCOLLECTIVE EXIT JMP JMX JMXU
  KILL NANOSLEEP PREEXIT RET RPCMOV WARPSYNC YIELD
Miscellaneous: B2R BAR CS2R DEPBAR GETLMEMBASE LEPC NOP PMTRIG S2R SETCTAID SETLMEMBASE VISET VOTE
"""
#: the category sm_120 lacks (it has no tcgen05 or tensor memory)
NOT_SM120 = "Tensor Core Memory"

_LDSM = ("LDSM_X1 LDSM_X2 LDSM_X4 LDSM_X1T LDSM_X2T LDSM_X4T LDSM4_X1 LDSM4_X2 LDSM4_X4 LDSM4T_X1 LDSM4T_X2 LDSM6_X1 "
         "LDSM6_X2 LDSM6_X4 LDSM6T_X1 LDSM6T_X2 LDSM8T_X1 LDSM8T_X2")
_STSM = "STSM_X1 STSM_X2 STSM_X4 STSM_X1T STSM_X2T STSM_X4T STSM8T_X1 STSM8T_X2 STSM8T_X4"
_UTMA = "UTMA UTMA8 UTMA512 UTMA1K UTMA2K UTMA4K UTMA8K UTMA16K UTMA32K UTMA48K UTMA64K UTMABIG BULKMAX"
_IGMMA = ("IGMMA_S8_N8 IGMMA_S8_N16 IGMMA_S8_N32 IGMMA_S8_N64 IGMMA_S8_N128 IGMMA_S8_N192 IGMMA_S8_N256 IGMMA_S8_N256_RS "
          "IGMMA_U8S8_N256 IGMMA_SP_S8")
_LDTM = ("LDTM_32X32B_X1 LDTM_32X32B_X4 LDTM_32X32B_X16 LDTM_32X32B_X32 LDTM_32X32B_X64 LDTM_32X32B_X128 LDTM_16X64B_X16 "
         "LDTM_16X128B_X8 LDTM_16X256B_X4 LDTM")
_UTCIMMA = ("UTCIMMA_M64_N64 UTCIMMA_M64_N128 UTCIMMA_M64_N256 UTCIMMA_M128_N32 UTCIMMA_M128_N64 UTCIMMA_M128_N128 "
            "UTCIMMA_M128_N256")
_HGMMA = ("HGMMA_BF16_N128 HGMMA_BF16_N256 HGMMA_F16_N256 HGMMA_TF32_N256 HGMMA_F16ACC HGMMA_SP_F16 HGMMA_SP_BF16 "
          "HGMMA_SP_TF32")
_UTCQMMA = ("UTCQMMA_E4M3_M128_N128 UTCQMMA_E4M3_M128_N256 UTCQMMA_E5M2 UTCQMMA_E3M2 UTCQMMA_E2M3 UTCQMMA_E2M1 "
            "UTCQMMA_SF UTCQMMA_SF_E5M2 UTCQMMA_SF_E3M2 UTCQMMA_SF_E2M3 UTCQMMA_SF_E2M1 UTCMMA_SP UTCMMA_2CTA")
_UNIFORM = ("R2UR S2UR CS2UR VOTEU UBMSK UBREV UCLEA UF2FP UFLO UIADD3 UIADD3.64 UIMAD ULDC ULEA ULEPC ULOP ULOP3 ULOP32I "
            "UMOV UMOV.64 UP2UR UPOPC UPRMT USEL USGXT USHF USHL USHR UFADD UF2F UF2I UF2IP UFFMA UFMNMX UFMUL UFRND UFSEL "
            "UFSET UI2F UI2FP UI2I UI2IP UIABS UIMNMX UGETNEXTWORKID UVIADD UVIMNMX UVIRTCOUNT UTCATOMSWS")

#: every documented opcode that writes a 32-bit word, and the classes its forms are in (`Isa/Classes.lean`'s map): the
#: lines are mnemonics, then `->`, then classes
PLACE_TEXT = f"""
FADD FADD2 FADD32I FFMA FFMA2 FFMA32I FHADD FHADD2 FHFMA FHFMA2 FHMUL2 FMUL FMUL2 FMUL32I FSWZADD -> FFMA
FMNMX FMNMX3 FSET HMNMX2 HSET2 IMNMX VHMNMX VIADDMNMX VIMNMX VIMNMX3 -> VIMNMX
FSEL SEL -> SEL
MUFU BREV FLO POPC -> XU
HADD2 HADD2_32I HFMA2 HFMA2_32I HMUL2 HMUL2_32I -> HFMA2
HMMA -> HMMA16816 HMMASP16832 HMMA16816F16 HMMA1688T HMMA1684T
DADD DFMA -> W2
DMUL -> DMUL
DMMA -> DMMA
OMMA -> OMMASF16864
QMMA -> QMMA16832 QMMASF16832 QMMASP16864
BMMA -> BMMA168256
BMSK IABS SGXT MOV32I MOVM P2R BMOV ELECT RPCMOV B2R CS2R GETLMEMBASE LEPC S2R DECOMPRESS SPARSIFY -> NOROW
IADD3 IADD32I ISCADD ISCADD32I LEA LOP LOP3 LOP32I SHF SHL SHR VIADD PRMT -> ALU
IADD -> ALU ALU64
IDP IDP4A -> IMAD
IMAD IMUL IMUL32I -> IMAD IMADW
IMMA -> IMMA16832 IMMA16816 IMMA8816 IMMASP16864 IMMASP16832 IMMAS4 MMA
VABSDIFF VABSDIFF4 VISET -> VSIMD
F2F F2I I2F I2I I2IP I2FP F2IP FRND -> F2FS CVT64
MOV -> NOROW MOV64
MOV64IUR -> MOV64
SHFL -> SHFL
LD -> LDG32 LDG64 LDG128 LDG256
LDC -> LDC64 LDCDIV LDCUNI
LDG -> LDG32 LDG64 LDG128 LDG128CG LDG128NA LDG256 LDG256CG
LDGMC -> ATOMG64
LDGSTS -> LDGSTS16
LDL -> LDG32 LDG64 LDG128
LDS -> LDS32 LDS64 LDS128
LDSM -> {_LDSM}
STSM -> {_STSM}
ST STG -> STG32 STG64 STG128 STG256
STL -> STG32 STG64 STG128
STS -> STS128
STAS -> DSMST4P
SYNCS SYNCSU USYNCS -> ATOMS64
MATCH VOTE -> VOTE
ATOM -> ATOME32 ATOME32P ATOME64
ATOMS -> ATOMS32 ATOMS64
ATOMG -> ATOMG32 ATOMG64
REDAS -> REDAS32 REDAS64
REDS -> ATOMS32
REDG -> RED32 RED64 REDV REDG32 REDG64
CREDUX REDUX -> REDUX
LDCU -> UNI UNI64 LDCU128
{_UNIFORM} -> UNI UNI64
UMEMSETS -> BULKMAX
UREDGR -> RED32 RED64
USTGR -> STG32 STG64
BGMMA -> BGMMA_B1_N256
HGMMA -> {_HGMMA}
IGMMA -> {_IGMMA}
QGMMA -> QGMMA_E4M3_N128 QGMMA_E4M3_N256 QGMMA_E5M2_N256 QGMMA_SP_E4M3 QGMMA_SP_E5M2
UBLKCP -> UBLKCP4K BULKMAX UTMAMC
UBLKRED UTMAREDG -> TMAred512 BULKRED
UTMALDG -> TMA512 {_UTMA} UTMAMC G4_16 G4_32 G4_256 G4_1K
UTMASTG -> {_UTMA}
LDT LDTM -> {_LDTM}
STT STTM -> STTM
UTCCP -> UTCCP
UTCSHIFT -> UTCSHIFT
UTCHMMA -> UTCHMMA_F16_M128_N256 UTCHMMA_BF16 UTCHMMA_TF32 UTCMMA_SP UTCMMA_2CTA
UTCIMMA -> {_UTCIMMA} UTCMMA_SP UTCMMA_2CTA
UTCOMMA -> UTCOMMA_SF UTCMMA_SP UTCOMMA_2CTA
UTCQMMA -> {_UTCQMMA}
TEX -> TEX1 TEX2 TEX4 TEXBIL
TLD -> TLD1 TLD2 TLD2D1 TLD2D4 TLD4w
TLD4 -> TLD4G
TMML -> TEX2
TXD -> TEX4
TXQ -> TEX1
SUATOM -> SUATOM
SULD -> SULD1 SULD4
SURED -> SURED
SUST -> SUST
"""

#: the documented opcodes read as writing no 32-bit word, by why
NO_WORD = {
    "a predicate": "FCHK FSETP HSETP2 DSETP ISETP PLOP3 PSETP R2P QSPC UISETP UPLOP3 UPSETP UR2UP UFSETP",
    "control": "ACQBULK ACQSHMINIT BPT BRA BREAK BRX BRXU BSSY BSYNC CALL ENDCOLLECTIVE EXIT JMP JMX JMXU KILL NANOSLEEP "
               "PREEXIT RET WARPSYNC YIELD",
    "a barrier or fence": "BAR DEPBAR LDGDEPBAR FENCE MEMBAR ERRBAR CGAERRBAR UCGABAR_ARV UCGABAR_WAIT WARPGROUP "
                          "WARPGROUPSET UTCBAR",
    "cache, prefetch or TMA control": "CCTL CCTLL CCTLT UBLKPF UBLKL2CCTL UTMACCTL UTMACMDFLUSH UTMAL2CCTL UTMAPF",
    "launch or register state": "NOP PMTRIG SETCTAID SETLMEMBASE USETMAXREG",
}
#: where `class_of` places a bare word-writing mnemonic in a class its forms aren't in, and why
CLASS_OF_WRONG = {
    **{m: "the uniform rule (U-prefixed) takes a bulk, tensor or mbarrier opcode" for m in
       "UBLKCP UBLKRED UTMALDG UTMAREDG UTMASTG UMEMSETS UREDGR USTGR UTCHMMA UTCIMMA UTCOMMA UTCQMMA UTCCP UTCSHIFT "
       "USYNCS".split()},
    **{m: "an FP32 opcode whose name holds HADD2, HFMA2 or HMUL2" for m in ("FHADD2", "FHFMA2", "FHMUL2")},
}

#: a dotted form `class_of` places in a class whose row may not bound it: its class there, the map's, and the words
#: it may write (IMMA.1688's shape no document gives; an m16n8 one writes 128)
FORMS_MOVED = {"IMMA.1688": ("IMMA8816", "IMMA16816", 128)}

#: RT-ISA's SASS census: every opcode the CUDA 13.0 libraries emit on sm_90, sm_100 and sm_120 (v3: both nodes'
#: corpora, 164 binaries), its store artifact, and the fixture `--write-corpus` derives from it (each dotted form:
#: RT-ISA's word flag, its count, its bucket)
CORPUS_ART = "art:d11bc611d0e9ed530b018e1d9b873d124b866a3545cde708022621d604d2ca3a"
CORPUS = DATA / "rt_isa_corpus.json"
CORPUS_DEVICES = {"Sm90": "sm_90", "Sm100": "sm_100", "Sm120": "sm_120"}
#: the corpus's bases no documented list names, and the class the map puts them in: USETSHMSZ (sm_90 only, new in v3,
#: whose destination no document gives) in the wider uniform writer class, on a priced route
UNDOCUMENTED = {"ARRIVES": "ATOMS64", "F2FP": "F2FS", "USETSHMSZ": "UNI64"}
#: the undocumented bases the corpus emits on some devices only (the rest on every device)
UNDOCUMENTED_ON = {"USETSHMSZ": {"Sm90"}}
#: the bases the corpus reads as writing a word and the map as writing none, and why none
WORD_DISAGREE = {
    "R2P": "it writes predicates from a register's bits",
    "UPLOP3": "it writes uniform predicates",
    "LDGDEPBAR": "a dependency barrier on LDGSTS; it writes no register",
    "USETMAXREG": "it changes the warp's register allocation (setmaxnreg)",
    "UTMAPF": "a TMA prefetch into L2",
}
#: each vector class at two words a lane, and a corpus form the map or `class_of` puts in it that writes a pair
#: (IMAD's is the rule: `class_of` puts every IMUL form in IMAD, a .WIDE one included)
LANE_PAIR = {"FFMA": "FFMA2", "VIMNMX": "IMNMX.U64", "F2FS": "F2F.F64.F32", "SEL": "SEL.64", "NOROW": "CS2R",
             "IMAD": None}
#: the paired FP32 forms that count as two instructions of their class (RT-ISA-6, note:20261019T0520Z-draft-rt-isa):
#: each takes two issue slots and two FMA-pipe cycles (B200 r20261019-083352-2e24), the corpus's UNDERPRICED bucket;
#: either count fits the class's row, one instance at two words a lane or two at one
COUNT_TWO = {"FFMA2": "FFMA", "FMUL2": "FFMA"}
#: RT-MACHINE-2's triage (note:20261019T0456Z-draft-rt-machine): each Blackwell opcode no menu file named, classed by
#: hand by what it can do to a committed word, its store artifact, and the fixture `--write-triage` derives from it
TRIAGE_ART = "art:fac984dca8c0141d83e5f147ba04787c02b7d59bd49edcb1ffb64727eeeb7e0c"
TRIAGE = DATA / "rt_machine_triage.json"
#: its classes that can multiply or fold words, which the map must place on a priced route
TRIAGE_PRICED = {"PRICED-LANE-MAC", "LOW-PRECISION", "PRICED-UNIFORM", "REDUCER"}
TRIAGE_CLASSES = TRIAGE_PRICED | {"NO-MULTIPLY", "MOVER", "MEMORY", "CONTROL", "ABSENT-ON-SM120"}

#: the documented lists' access categories: a load, LDSM, an atomic, a texture or surface access or a tensor-memory
#: copy writes memory's content, so none heads a vector or table word (A6-WIDE, note:20261019T0457Z-draft-fpx-machine)
ACCESS = ("Load/Store", "Tensor Memory Access", "Tensor Core Memory", "Texture", "Surface")
#: the access-category opcodes in a vector class, each with its class: MATCH, filed under Load/Store, reads no memory
#: and writes one word a lane (a lane mask), VOTE's row
ACCESS_LANE = {"MATCH": "VOTE"}


def opcode_list(text: str) -> dict[str, list[str]]:
    """A documented list, category to mnemonics."""
    out: dict[str, list[str]] = {}
    cat = None
    for line in text.strip().splitlines():
        if not line.startswith(" "):
            cat, line = line.split(": ", 1)
            out[cat] = []
        out[cat] += line.split()
    return out


def placements(text: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for line in text.strip().splitlines():
        ops, classes = line.split(" -> ")
        for op in ops.split():
            out.setdefault(op, set()).update(classes.split())
    return out


def lean_types(src: str) -> dict[str, list[str]]:
    """Each `inductive` of the Lean source, its constructors."""
    return {m.group(1): re.findall(r"\|\s*([A-Za-z0-9_]+)", m.group(2))
            for m in re.finditer(r"^inductive (\w+)\n(.*?)^\s*deriving", src, re.M | re.S)}


def lean_rows(src: str, ty: str) -> dict[str, tuple[str, int]]:
    """`ty.row`'s arms: each class's route and words."""
    body = re.search(rf"^def {ty}\.row : {ty} → Row\n(.*?)(?=^\S|\Z)", src, re.M | re.S).group(1)
    out = {}
    for k, arm in re.findall(r"^\s*\| \.(\w+) =>(.*?)(?=^\s*\| \.|\Z)", body, re.M | re.S):
        if m := re.search(r"Row\.lane \.\w+ \.\w+ (?:true|false) \.\w+ (\d+)", arm):
            out[k] = ("lane", int(m.group(1)))
        elif m := re.search(r"Row\.mma \.\w+ \.\w+ \.\w+ \.\w+ (\d+)", arm):
            out[k] = ("writer", int(m.group(1)))
        elif m := re.search(r"Row\.side \.\w+ \.\w+ \.\w+ (?:true|false) (\d+)", arm):
            out[k] = ("side", int(m.group(1)))
        elif m := re.search(r"Row\.copy \.\w+ (\d+)", arm):
            out[k] = ("traced", int(m.group(1)))
        elif (r := re.search(r"route := \.(\w+)", arm)) and (w := re.search(r"words := (\d+)", arm)):
            out[k] = (r.group(1), int(w.group(1)))
    return out


def device_parts(src: str, dev: str) -> list[str]:
    """The class types a device's table (`inductive Sm120 | sync (s : Sync) | …`) is made of."""
    return re.findall(r"\| \w+ \(\w+ : (\w+)\)", re.search(rf"^inductive {dev}\n(.*?)deriving", src, re.M | re.S)[1])


def table(sources: dict[str, str]) -> tuple[dict[str, tuple[str, int]], dict[str, set[str]], dict[str, list[str]]]:
    """The classes' rows, each device's classes, and each class type's constructors, from the Lean sources."""
    src = "\n".join(sources.values())
    types = lean_types(src)
    rows: dict[str, tuple[str, int]] = {}
    for ty in ("Sync", "Base", "Gmma", "Tc05", "Tmem"):
        rows.update(lean_rows(src, ty))
    devices = {dev: {k for part in device_parts(src, dev) for k in types[part]} for dev in ("Sm120", "Sm90", "Sm100")}
    return rows, devices, types


def floors_short(floors: list[dict], rows: dict[str, tuple[str, int]]) -> set[str]:
    """The side and writer classes floors.csv prices at fewer words an instruction than their row."""
    return {r["kind"] for r in floors if rows.get(r["kind"], ("lane", 0))[0] in ("side", "writer")
            and int(r["words"]) < rows[r["kind"]][1]}


def off_route(classes: set[str], rows: dict[str, tuple[str, int]]) -> set[str]:
    """The classes that are no row of a table, or are traced (on no priced route)."""
    return {k for k in classes if k not in rows or rows[k][0] == "traced"}


def unplaced(lists: dict[str, dict[str, list[str]]], place: dict[str, set[str]], devices: dict[str, set[str]]):
    """Each device's documented opcodes that are neither placed in its table's classes nor read as writing no word."""
    no_word = {op for ops in NO_WORD.values() for op in ops.split()}
    out = {}
    for dev, by_cat in lists.items():
        ops = [op for ops in by_cat.values() for op in ops]
        out[dev] = sorted(op for op in ops if op not in no_word and not (place.get(op) and place[op] <= devices[dev]))
    return out


def class_of_disagrees(ops: set[str], place: dict[str, set[str]]) -> set[str]:
    return {op for op in ops if op in place and (c := class_of(op)) is not None and c not in place[op]}


def forms_moved_ok(moved: dict[str, tuple[str, str, int]], rows: dict[str, tuple[str, int]], doc: str) -> bool:
    """Each moved form is `class_of`'s in a class whose row writes fewer than its words, the map in `Isa/Classes.lean`
    moves it, and its new class's row writes at least its words."""
    return all(class_of(f) == old and rows[old][1] < w <= rows[new][1]
               and f"{f}, which `class_of` puts in {old}, in {new}" in doc for f, (old, new, w) in moved.items())


def write_corpus(src: Path) -> None:
    """Derive the fixture from RT-ISA's artifact, fetched to `src` (`research data fetch CORPUS_ART --to src`)."""
    import json
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    head = {"source": CORPUS_ART, **{k: man[k] for k in ("schema", "corpus_runs", "toolkit", "binaries", "code")},
            "written_by": "benchmarks/pouw/theory/isa_table.py --write-corpus"}
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()] + [' "devices": {']
    fams = list(CORPUS_DEVICES.values())
    for i, fam in enumerate(fams):
        rows = sorted(json.loads((src / f"opcodes_{fam}.json").read_text(encoding="utf-8"))["rows"],
                      key=lambda r: r["op"])
        lines.append(f"  {json.dumps(fam)}: {{")
        lines += [f"   {json.dumps(r['op'])}: {json.dumps([int(bool(r['word'])), r['count'] or 0, r['bucket']])}"
                  + ("," if j < len(rows) - 1 else "") for j, r in enumerate(rows)]
        lines.append("  }" + ("," if i < len(fams) - 1 else ""))
    CORPUS.write_text("\n".join(lines + [" }", "}"]) + "\n", encoding="utf-8")


def corpus_forms() -> tuple[dict, dict[str, dict[str, list]]]:
    """The fixture's header and each device's dotted forms."""
    import json
    doc = json.loads(CORPUS.read_text(encoding="utf-8"))
    return doc, {dev: doc["devices"][fam] for dev, fam in CORPUS_DEVICES.items()}


def corpus_unplaced(forms: dict[str, list], place: dict[str, set[str]], device: set[str],
                    undocumented: dict[str, str]) -> list[str]:
    """A device's corpus bases neither placed in its table's classes, read as writing no word, nor placed as
    undocumented."""
    no_word = {op for ops in NO_WORD.values() for op in ops.split()}
    bases = {op.split(".")[0] for op in forms}
    return sorted(b for b in bases if b not in no_word and not (place.get(b) and place[b] <= device)
                  and not undocumented.get(b) in device)


def word_disagree(forms: dict[str, list]) -> set[str]:
    """The bases a corpus form of which RT-ISA reads as writing a word, which the map reads as writing none."""
    no_word = {op for ops in NO_WORD.values() for op in ops.split()}
    return {op.split(".")[0] for op, (word, _, _) in forms.items() if word and op.split(".")[0] in no_word}


def gap_off_writer(forms: dict[str, list], place: dict[str, set[str]], rows: dict[str, tuple[str, int]],
                   device: set[str]) -> set[str]:
    """The corpus's GAP-TENSOR forms (an MMA form no sheet peak or census row prices) whose classes aren't all writer
    rows of the device."""
    return {op for op, (_, _, bucket) in forms.items() if bucket == "GAP-TENSOR"
            and not ((cs := place.get(op.split(".")[0])) and cs <= device and all(rows[c][0] == "writer" for c in cs))}


def lane_widths_ok(rows: dict[str, tuple[str, int]], forms: dict[str, list], pair: dict[str, str | None],
                   place: dict[str, set[str]]) -> bool:
    """A vector class is at two words a lane exactly when `pair` names it, and each named form is in the corpus, in
    that class by the map or `class_of`."""
    lanes = {k: w for k, (route, w) in rows.items() if route == "lane"}
    return (all(w in (32, 64) for w in lanes.values()) and {k for k, w in lanes.items() if w == 64} == set(pair)
            and all(f is None or (f in forms and (class_of(f) == k or k in place.get(f.split(".")[0], set())))
                    for k, f in pair.items()))


def access_on_lane(lists: dict[str, dict[str, list[str]]], place: dict[str, set[str]],
                   rows: dict[str, tuple[str, int]]) -> set[str]:
    """The documented access opcodes placed in a vector class, whose heads A6 bounds at two words a lane: a wide load
    there (LDS.128 and LDSM.x4 at 4 words a lane, LDG.E.ENL2.256 at 8, LDTM.32x32b.x128 at 128) would break it."""
    return {op for by_cat in lists.values() for cat in ACCESS for op in by_cat.get(cat, [])
            if any(rows[c][0] == "lane" for c in place.get(op, ()))}


def count_two_ok(forms: dict[str, list], place: dict[str, set[str]], rows: dict[str, tuple[str, int]],
                 count_two: dict[str, str]) -> bool:
    """The corpus's UNDERPRICED bases are `count_two`'s, each placed in its class alone, a vector class at two words a
    lane."""
    under = {op.split(".")[0] for op, (_, _, bucket) in forms.items() if bucket == "UNDERPRICED"}
    return under == set(count_two) and all(place.get(op) == {c} and rows[c] == ("lane", 64)
                                           for op, c in count_two.items())


def write_triage(src: Path) -> None:
    """Derive the fixture from RT-MACHINE-2's artifact, fetched to `src` (`research data fetch TRIAGE_ART --to src`)."""
    import json
    doc = json.loads((src / "triage.json").read_text(encoding="utf-8"))
    head = {"source": TRIAGE_ART, "list": doc["source"], "unnamed_by_menu": doc["unnamed_by_menu"],
            "unclassified": doc["unclassified"], "written_by": "benchmarks/pouw/theory/isa_table.py --write-triage"}
    rows = sorted((r["opcode"], r["class"]) for r in doc["rows"])
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()] + [' "opcodes": {']
    lines += [f"  {json.dumps(op)}: {json.dumps(c)}" + ("," if i < len(rows) - 1 else "")
              for i, (op, c) in enumerate(rows)]
    TRIAGE.write_text("\n".join(lines + [" }", "}"]) + "\n", encoding="utf-8")


def triage() -> tuple[dict, dict[str, str]]:
    """The triage fixture's header and each opcode's class."""
    import json
    doc = json.loads(TRIAGE.read_text(encoding="utf-8"))
    return doc, doc["opcodes"]


def triage_disagree(tri: dict[str, str], place: dict[str, set[str]], rows: dict[str, tuple[str, int]],
                    no_word: set[str]) -> set[str]:
    """The triaged opcodes the map treats otherwise than their class allows: neither placed nor read as writing no
    word, in a traced class but not a mover, or one that can multiply or fold read as writing no word."""
    out = set()
    for op, c in tri.items():
        cs = place.get(op, set())
        if ((not cs and op not in no_word) or (c != "MOVER" and any(rows[k][0] == "traced" for k in cs))
                or (c in TRIAGE_PRICED and not cs)):
            out.add(op)
    return out


def main() -> int:
    if "--write-corpus" in sys.argv:
        write_corpus(Path(sys.argv[sys.argv.index("--write-corpus") + 1]))
        return 0
    if "--write-triage" in sys.argv:
        write_triage(Path(sys.argv[sys.argv.index("--write-triage") + 1]))
        return 0
    checks = Checks()
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted(ISA.glob("*.lean"))}
    rows, devices, types = table(sources)
    sizes = {ty: len(types[ty]) for ty in ("Sync", "Base", "Gmma", "Tc05", "Tmem")}
    checks.check("the Lean class types parse: Sync 18, Base 120, Gmma 24, Tc05 25, Tmem 13",
                 sizes == {"Sync": 18, "Base": 120, "Gmma": 24, "Tc05": 25, "Tmem": 13}, str(sizes))
    missing = [k for ty in sizes for k in types[ty] if k not in rows]
    checks.check("every class has a row with a route and words", not missing and len(rows) == sum(sizes.values()),
                 ", ".join(missing) or f"{len(rows)} rows")
    checks.check("sm_120's table is 138 classes, sm_90a's 162, sm_100a's 176",
                 [len(devices[d]) for d in ("Sm120", "Sm90", "Sm100")] == [138, 162, 176],
                 str({d: len(v) for d, v in devices.items()}))
    traced = sorted(k for k, (route, _) in rows.items() if route == "traced")
    checks.check("the one class off every route is MOV.64's (NoSum's decided part: a copy heads nothing)",
                 traced == ["MOV64"], str(traced))

    ds120 = {r["kind"] for r in csv.DictReader((DATA / "census" / "DS-120.csv").open(encoding="utf-8"))}
    checks.check("sm_120's classes are DS-120's census kinds and the sixteen Isa/Classes.lean adds",
                 devices["Sm120"] == ds120 | ADDED and not ds120 & ADDED,
                 f"DS-120 {len(ds120)}, added {len(ADDED)}")
    for dev, d in (("Sm90", "sm_90"), ("Sm100", "sm_100")):
        kinds = {CENSUS_ALIAS.get(r["kind"], r["kind"])
                 for r in csv.DictReader((DATA / "devices" / d / "census.csv").open(encoding="utf-8"))}
        checks.check(f"{d}'s census kinds are classes of its table", kinds <= devices[dev],
                     ", ".join(sorted(kinds - devices[dev])) or f"{len(kinds)} kinds")

    floors = list(csv.DictReader((DATA / "floors.csv").open(encoding="utf-8")))
    checks.check("every kind floors.csv prices is a class of sm_120's table",
                 {r["kind"] for r in floors} <= devices["Sm120"])
    short = floors_short(floors, rows)
    checks.check("floors.csv prices every side and writer class at its row's words or more, but the atomics (F6)",
                 short == FLOORS_BELOW, ", ".join(sorted(short ^ FLOORS_BELOW)) or f"{len(short)} atomics below")

    import isa_escape  # noqa: E402
    names = [c["name"] for c in isa_escape.CANDIDATES]
    placed = {n: ESCAPE[p].split() for n in names for p in ESCAPE if n.startswith(p)}
    ahead = {p for p in ESCAPE if not any(n.startswith(p) for n in names)}
    checks.check("every isa_escape.py candidate is placed in classes, each placement naming one or ESCAPE_AHEAD's",
                 len(placed) == len(names) and ahead <= ESCAPE_AHEAD,
                 ", ".join(sorted(set(names) - set(placed)) + sorted(ahead - ESCAPE_AHEAD))
                 or f"{len(placed)} of {len(names)}, {len(ahead)} ahead")
    escape_classes = {k for ks in ESCAPE.values() for k in ks.split()}
    checks.check("every candidate's classes are rows of a table, on a priced route (none traced)",
                 not off_route(escape_classes, rows), ", ".join(sorted(off_route(escape_classes, rows))) or None)

    lists = {"Sm90": opcode_list(HOPPER), "Sm100": opcode_list(BLACKWELL)}
    lists["Sm120"] = {c: ops for c, ops in lists["Sm100"].items() if c != NOT_SM120}
    counts = {d: sum(map(len, v.values())) for d, v in lists.items()}
    checks.check("the documented lists: Hopper 204 opcodes, Blackwell 257 (sm_120 without tensor core memory: 245)",
                 counts == {"Sm90": 204, "Sm100": 257, "Sm120": 245}, str(counts))
    every = {op for by_cat in lists.values() for ops in by_cat.values() for op in ops}
    place = placements(PLACE_TEXT)
    no_word = [op for ops in NO_WORD.values() for op in ops.split()]
    checks.check("placed and no-word are disjoint, name only documented opcodes, and leave none out",
                 len(no_word) == len(set(no_word)) and not (set(place) & set(no_word))
                 and set(place) | set(no_word) == every,
                 ", ".join(sorted((set(place) | set(no_word)) ^ every)) or f"{len(every)} opcodes")
    gaps = unplaced(lists, place, devices)
    for dev in ("Sm120", "Sm90", "Sm100"):
        checks.check(f"every documented {dev} opcode that writes a word is placed in classes of its table",
                     not gaps[dev], ", ".join(gaps[dev]) or f"{counts[dev]} opcodes")
    wrong = class_of_disagrees(every, place)
    checks.check("class_of agrees with every placement of a bare mnemonic it places, but CLASS_OF_WRONG's",
                 wrong == set(CLASS_OF_WRONG), ", ".join(sorted(wrong ^ set(CLASS_OF_WRONG))) or f"{len(wrong)} named")
    doc = " ".join(sources["Classes.lean"].split())
    checks.check("a form the map moves (IMMA.1688) is class_of's in a row below its words, and the map's row bounds it",
                 forms_moved_ok(FORMS_MOVED, rows, doc), str(FORMS_MOVED))

    header, corpus = corpus_forms()
    checks.check(f"the corpus fixture is RT-ISA's ({CORPUS_ART[:16]}…, {header['binaries']} binaries, "
                 f"{header['toolkit']})", header["source"] == CORPUS_ART and all(corpus.values()),
                 " ".join(header["corpus_runs"].values()))
    for dev in ("Sm120", "Sm90", "Sm100"):
        gaps = corpus_unplaced(corpus[dev], place, devices[dev], UNDOCUMENTED)
        bases = {op.split(".")[0] for op in corpus[dev]}
        checks.check(f"every {dev} corpus base is placed in classes of its table, read as writing no word, or named",
                     not gaps, ", ".join(gaps) or f"{len(bases)} bases, {len(corpus[dev])} forms")
    undoc = {d: {op.split(".")[0] for op in corpus[d]} - every for d in corpus}
    checks.check("the corpus bases no documented list names are UNDOCUMENTED's: ARRIVES and F2FP on every device, "
                 "USETSHMSZ on sm_90",
                 all(u == {b for b in UNDOCUMENTED if d in UNDOCUMENTED_ON.get(b, {d})} for d, u in undoc.items()),
                 str({d: sorted(u) for d, u in undoc.items()}))
    disagree = set().union(*(word_disagree(corpus[d]) for d in corpus))
    checks.check("where the corpus reads a word and the map none, it is WORD_DISAGREE's, each with its reason",
                 disagree == set(WORD_DISAGREE), ", ".join(sorted(disagree ^ set(WORD_DISAGREE))) or
                 ", ".join(sorted(disagree)))
    off = {d: gap_off_writer(corpus[d], place, rows, devices[d]) for d in corpus}
    gap_forms = sorted({op for d in corpus for op, (_, _, b) in corpus[d].items() if b == "GAP-TENSOR"})
    checks.check("the corpus's unpriced MMA forms (RT-ISA-1's DMMA, RT-ISA-4's UTCOMMA) sit in writer rows only",
                 gap_forms and not any(off.values()), str({d: sorted(o) for d, o in off.items() if o}) if any(
                     off.values()) else " ".join(gap_forms))
    all_forms = {op: v for d in corpus for op, v in corpus[d].items()}
    checks.check("a vector class writes two words a lane exactly where a corpus form needs it (LANE_PAIR)",
                 lane_widths_ok(rows, all_forms, LANE_PAIR, place),
                 " ".join(f"{k}:{w}" for k, (r, w) in sorted(rows.items()) if r == "lane"))
    checks.check("the corpus's UNDERPRICED forms are COUNT_TWO's (RT-ISA-6: FFMA2 and FMUL2 count as two FFMA), in "
                 "FFMA alone at two words a lane", count_two_ok(all_forms, place, rows, COUNT_TWO),
                 " ".join(sorted(op for op, (_, _, b) in all_forms.items() if b == "UNDERPRICED")))

    thead, tri = triage()
    blackwell = {op for ops in lists["Sm100"].values() for op in ops}
    checks.check(f"the triage fixture is RT-MACHINE-2's ({TRIAGE_ART[:16]}…): {thead['unnamed_by_menu']} Blackwell "
                 "opcodes, none unclassed", thead["source"] == TRIAGE_ART and not thead["unclassified"]
                 and len(tri) == thead["unnamed_by_menu"] == 175 and set(tri) <= blackwell
                 and set(tri.values()) <= TRIAGE_CLASSES, ", ".join(sorted(set(tri) - blackwell)) or None)
    tri_off = triage_disagree(tri, place, rows, set(no_word))
    by_class = {c: sum(v == c for v in tri.values()) for c in sorted(set(tri.values()))}
    checks.check("every triaged opcode is placed on a priced route or writes no word, as its class allows",
                 not tri_off, ", ".join(sorted(tri_off)) or " ".join(f"{c} {n}" for c, n in by_class.items()))
    absent = {op for op, c in tri.items() if c == "ABSENT-ON-SM120"}
    checks.check("the opcodes the triage finds absent on sm_120 are Blackwell's tensor core memory list",
                 absent == set(lists["Sm100"][NOT_SM120]), " ".join(sorted(absent ^ set(lists["Sm100"][NOT_SM120])))
                 or f"{len(absent)} opcodes")
    on_lane = access_on_lane(lists, place, rows)
    n_access = len({op for by_cat in lists.values() for cat in ACCESS for op in by_cat.get(cat, [])})
    checks.check("no load, atomic, texture, surface or tensor-memory opcode heads a vector or table word (A6-WIDE): "
                 "each is in side, traced or writer classes, but ACCESS_LANE's at one word a lane",
                 on_lane == set(ACCESS_LANE)
                 and all(place[op] == {c} and rows[c] == ("lane", 32) for op, c in ACCESS_LANE.items()),
                 ", ".join(sorted(on_lane ^ set(ACCESS_LANE))) or f"{n_access} access opcodes")

    if checks.mutations:
        without = {**sources, "Classes.lean": sources["Classes.lean"].replace("| BMMA168256 | IMMAS4 | DMMA",
                                                                              "| BMMA168256 | IMMAS4")}
        r2, d2, _ = table(without)
        checks.mutation("a class dropped from the Lean (DMMA) leaves its opcode unplaced",
                        "DMMA" in unplaced(lists, place, d2)["Sm90"])
        cut = [dict(r, words="64") if r["kind"] == "TEX4" else r for r in floors]
        checks.mutation("a side class priced below its row in floors.csv (TEX4 at 64) is caught",
                        floors_short(cut, rows) != FLOORS_BELOW)
        checks.mutation("an opcode dropped from the map (VOTE) is unplaced on every device",
                        all("VOTE" in unplaced(lists, {k: v for k, v in place.items() if k != "VOTE"}, devices)[d]
                            for d in devices))
        checks.mutation("a Hopper opcode placed in a class sm_90a lacks (UTMALDG in UTCCP) is caught",
                        "UTMALDG" in unplaced(lists, {**place, "UTMALDG": {"UTCCP"}}, devices)["Sm90"])
        checks.mutation("a candidate placed in the traced class (MOV64) is caught",
                        off_route(escape_classes | {"MOV64"}, rows) == {"MOV64"})
        checks.mutation("a class_of disagreement left unnamed (UBLKCP) is caught",
                        class_of_disagrees(every, place) != set(CLASS_OF_WRONG) - {"UBLKCP"})
        checks.mutation("a moved form left in class_of's class (IMMA.1688 in IMMA8816) is caught",
                        not forms_moved_ok({"IMMA.1688": ("IMMA8816", "IMMA8816", 128)}, rows, doc))
        checks.mutation("a corpus base dropped from UNDOCUMENTED (ARRIVES) is unplaced",
                        "ARRIVES" in corpus_unplaced(corpus["Sm90"], place, devices["Sm90"],
                                                     {k: v for k, v in UNDOCUMENTED.items() if k != "ARRIVES"}))
        checks.mutation("a new corpus opcode on no list and in no class is unplaced",
                        corpus_unplaced({**corpus["Sm100"], "FOO.32": [1, 1, "UNKNOWN"]}, place, devices["Sm100"],
                                        UNDOCUMENTED) == ["FOO"])
        checks.mutation("a word the map reads as none, left unnamed (R2P), is caught",
                        word_disagree(corpus["Sm90"]) - (set(WORD_DISAGREE) - {"R2P"}) == {"R2P"})
        checks.mutation("an unpriced MMA form placed in a side class (DMMA in LDS32) is caught",
                        bool(gap_off_writer(corpus["Sm90"], {**place, "DMMA": {"LDS32"}}, rows, devices["Sm90"])))
        checks.mutation("a vector class cut to one word a lane where a corpus form writes a pair (SEL) is caught",
                        not lane_widths_ok({**rows, "SEL": ("lane", 32)}, all_forms, LANE_PAIR, place))
        drop = {k: v for k, v in place.items() if k != "DECOMPRESS"}
        checks.mutation("a documented opcode dropped from the map (DECOMPRESS) fails, with no list to excuse it",
                        all("DECOMPRESS" in unplaced(lists, drop, devices)[d] for d in ("Sm100", "Sm120"))
                        and "DECOMPRESS" in triage_disagree(tri, drop, rows, set(no_word)))
        checks.mutation("a triaged lane multiply placed in the traced class (FFMA2 in MOV64) is caught",
                        triage_disagree(tri, {**place, "FFMA2": {"MOV64"}}, rows, set(no_word)) == {"FFMA2"})
        checks.mutation("a triaged lane multiply read as writing no word (FMUL32I) is caught",
                        triage_disagree(tri, {k: v for k, v in place.items() if k != "FMUL32I"}, rows,
                                        set(no_word) | {"FMUL32I"}) == {"FMUL32I"})
        checks.mutation("a paired FP32 form left out of COUNT_TWO (FMUL2) is caught",
                        not count_two_ok(all_forms, place, rows, {"FFMA2": "FFMA"}))
        checks.mutation("a load placed in a vector class (LDS in ALU) is caught",
                        access_on_lane(lists, {**place, "LDS": {"ALU"}}, rows) - set(ACCESS_LANE) == {"LDS"})

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
