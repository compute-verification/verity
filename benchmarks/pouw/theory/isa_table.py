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
  - isa_escape: every candidate `isa_escape.py` closes sits in classes on a priced route, none traced;
  - completeness evidence for `IsaComplete`: every opcode the CUDA Binary Utilities 13.4 list (§4.3 Hopper, §4.4
    Blackwell and Rubin; last updated 12 Oct 2026) is placed in classes of its device's table (`PLACE`), read as
    writing no 32-bit word (`NO_WORD`), or named unplaced (`UNPLACED`); sm_120's is Blackwell's list without its tensor
    core memory instructions; and where `class_of` (`facet_break/loads.py`) places a bare mnemonic it agrees, but for
    the disagreements `CLASS_OF_WRONG` names, and a dotted form it places in a class whose row may not bound it is
    moved by the map to one that does (`FORMS_MOVED`).

A placement is a words bound only: a class's rate is the data sheet's, measured for the forms its census probed.

Run: `python isa_table.py [--mutations] [--verdict PATH]`. Stdlib and exact_kit; reads the Lean table, floors.csv and
the censuses.
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
}

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
          "IGMMA_U8S8_N256 GMMASP")
_LDTM = ("LDTM_32X32B_X1 LDTM_32X32B_X4 LDTM_32X32B_X16 LDTM_32X32B_X32 LDTM_32X32B_X64 LDTM_32X32B_X128 LDTM_16X64B_X16 "
         "LDTM_16X128B_X8 LDTM_16X256B_X4 LDTM")
_UTCIMMA = ("UTCIMMA_M64_N64 UTCIMMA_M64_N128 UTCIMMA_M64_N256 UTCIMMA_M128_N32 UTCIMMA_M128_N64 UTCIMMA_M128_N128 "
            "UTCIMMA_M128_N256")
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
BMSK IABS SGXT MOV32I MOVM P2R BMOV ELECT RPCMOV B2R CS2R GETLMEMBASE LEPC S2R -> NOROW
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
HGMMA -> HGMMA_BF16_N128 HGMMA_BF16_N256 HGMMA_F16_N256 HGMMA_TF32_N256 HGMMA_F16ACC GMMASP
IGMMA -> {_IGMMA}
QGMMA -> QGMMA_E4M3_N128 QGMMA_E4M3_N256 QGMMA_E5M2_N256 GMMASP
UBLKCP -> UBLKCP4K BULKMAX UTMAMC
UBLKRED UTMAREDG -> TMAred512 BULKRED
UTMALDG -> TMA512 {_UTMA} UTMAMC G4_16 G4_32 G4_256 G4_1K
UTMASTG -> {_UTMA}
LDT LDTM -> {_LDTM}
STT STTM -> STTM
UTCCP -> UTCCP
UTCSHIFT -> UTCSHIFT
UTCHMMA -> UTCHMMA_F16_M128_N256 UTCHMMA_TF32 UTCMMA_SP UTCMMA_2CTA
UTCIMMA -> {_UTCIMMA} UTCMMA_SP UTCMMA_2CTA
UTCOMMA -> UTCOMMA_SF UTCMMA_SP UTCMMA_2CTA
UTCQMMA -> UTCQMMA_E4M3_M128_N128 UTCQMMA_E4M3_M128_N256 UTCQMMA_SF UTCMMA_SP UTCMMA_2CTA
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
#: documented word writers no class holds yet: Blackwell's vector decompress and sparsify, whose widths no document gives
UNPLACED = {"DECOMPRESS", "SPARSIFY"}

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
        if "Row.lane" in arm:
            out[k] = ("lane", 64)
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
        out[dev] = sorted(op for op in ops if op not in no_word and op not in UNPLACED
                          and not (place.get(op) and place[op] <= devices[dev]))
    return out


def class_of_disagrees(ops: set[str], place: dict[str, set[str]]) -> set[str]:
    return {op for op in ops if op in place and (c := class_of(op)) is not None and c not in place[op]}


def forms_moved_ok(moved: dict[str, tuple[str, str, int]], rows: dict[str, tuple[str, int]], doc: str) -> bool:
    """Each moved form is `class_of`'s in a class whose row writes fewer than its words, the map in `Isa/Classes.lean`
    moves it, and its new class's row writes at least its words."""
    return all(class_of(f) == old and rows[old][1] < w <= rows[new][1]
               and f"{f}, which `class_of` puts in {old}, in {new}" in doc for f, (old, new, w) in moved.items())


def main() -> int:
    checks = Checks()
    sources = {p.name: p.read_text(encoding="utf-8") for p in sorted(ISA.glob("*.lean"))}
    rows, devices, types = table(sources)
    sizes = {ty: len(types[ty]) for ty in ("Sync", "Base", "Gmma", "Tc05", "Tmem")}
    checks.check("the Lean class types parse: Sync 18, Base 120, Gmma 19, Tc05 15, Tmem 13",
                 sizes == {"Sync": 18, "Base": 120, "Gmma": 19, "Tc05": 15, "Tmem": 13}, str(sizes))
    missing = [k for ty in sizes for k in types[ty] if k not in rows]
    checks.check("every class has a row with a route and words", not missing and len(rows) == sum(sizes.values()),
                 ", ".join(missing) or f"{len(rows)} rows")
    checks.check("sm_120's table is 138 classes, sm_90a's 157, sm_100a's 166",
                 [len(devices[d]) for d in ("Sm120", "Sm90", "Sm100")] == [138, 157, 166],
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
    checks.check("every isa_escape.py candidate is placed in classes", len(placed) == len(names) == len(ESCAPE),
                 f"{len(placed)} of {len(names)}")
    escape_classes = {k for ks in placed.values() for k in ks}
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
    checks.check("placed, no-word and unplaced are disjoint and name only documented opcodes",
                 len(no_word) == len(set(no_word)) and not (set(place) & set(no_word)) and not (set(place) & UNPLACED)
                 and not (set(no_word) & UNPLACED) and set(place) | set(no_word) | UNPLACED == every,
                 ", ".join(sorted((set(place) | set(no_word) | UNPLACED) ^ every)) or f"{len(every)} opcodes")
    gaps = unplaced(lists, place, devices)
    for dev in ("Sm120", "Sm90", "Sm100"):
        checks.check(f"every documented {dev} opcode that writes a word is placed in classes of its table",
                     not gaps[dev], ", ".join(gaps[dev]) or f"{counts[dev]} opcodes")
    checks.check("the documented word writers no class holds are DECOMPRESS and SPARSIFY (IsaComplete's open part)",
                 UNPLACED <= every and all(class_of(op) is None for op in UNPLACED), " ".join(sorted(UNPLACED)))
    wrong = class_of_disagrees(every, place)
    checks.check("class_of agrees with every placement of a bare mnemonic it places, but CLASS_OF_WRONG's",
                 wrong == set(CLASS_OF_WRONG), ", ".join(sorted(wrong ^ set(CLASS_OF_WRONG))) or f"{len(wrong)} named")
    doc = " ".join(sources["Classes.lean"].split())
    checks.check("a form the map moves (IMMA.1688) is class_of's in a row below its words, and the map's row bounds it",
                 forms_moved_ok(FORMS_MOVED, rows, doc), str(FORMS_MOVED))

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

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
