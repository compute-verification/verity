"""What each SASS base opcode does, as diff_menu.py classifies it against A7's menu: can it form a multiply (head a
fresh product), does it write a word, and which menu route (if any) prices it.

A7 (`SheetSemantics`) says no unit outside the menu heads a fresh committed word (NoSum) and a side unit costs at least
its per-word vector (SideUnits). A committed word is a fresh s32: the running sum of 32 int8 products of noised operands.
So the question for each opcode is: can it form an int8 (or FP8/FP4) product at all (`mul`), and if it only writes a word
without multiplying (`word`, not `mul`), is that word a sum/copy/select of words that already exist (NoSum). The menu's
five routes (isa_escape.py): an MMA writer on the tensor route (A3), the CUDA lane route (q*), the uniform datapath on
issue, table routes by bytes, priced side units at the pi32 floor.

`ROUTE` says which route prices a base opcode:
  - "tensor": an MMA form, priced by A3 (needs an operand-type peak; a form with no sheet peak is priced at macs/0 = 0).
  - "lane": a scalar/vector lane instruction, priced on the q* lane route (`loads.class_of` gives its census class).
  - "uniform": the uniform datapath, priced on issue (one warp-uniform word an instruction).
  - "sidefloor": a side unit (reduction, atomic, texture, bulk) priced at the pi32 floor (floors.csv).
  - "mover": no arithmetic; its output equals an existing word (a copy/transpose/select).
  - "control": branch/barrier/predicate/misc; writes no committed word.
  - None: not priced by any menu route -- `loads.class_of` returns None and it is no tensor/side form. The red-team gap.

This table is keyed on the SASS base opcode (the part before the first `.`). It is the reviewable hand-classification the
diff ranks; `diff_menu.py` cross-checks it against `loads.class_of` (the model's own SASS->census-class map) and flags
any disagreement, so a drift between this table and the model is caught.
"""
from __future__ import annotations

# The tensor-core MMA families. Each heads products on the tensor route (A3). `omega` forms, operand formats and peaks
# are the sheet's; a form whose operand type has no sheet peak is priced at 0 (DMMA FP64, b1, s4 on sm_120) -- a gap A3
# can't price. mul=True, word=True, route="tensor".
MMA_SYNC = ["HMMA", "IMMA", "DMMA", "BMMA", "QMMA", "OMMA"]          # mma.sync (register accumulators)
MMA_WGMMA = ["HGMMA", "IGMMA", "QGMMA", "OGMMA", "BGMMA"]           # Hopper async warpgroup wgmma
MMA_TCGEN05 = ["UTCHMMA", "UTCIMMA", "UTCQMMA", "UTCOMMA", "UTCBMMA"]  # Blackwell tcgen05.mma (TMEM accumulators)
MMA = MMA_SYNC + MMA_WGMMA + MMA_TCGEN05

# The lanes' multiply-accumulate and multiply opcodes: exact integer MAC (the honest route), FP MAC, dot products.
LANE_MUL = {
    "IMAD": "s32 integer multiply-add (the honest lane MAC; IMAD.WIDE -> s64 pair)",
    "IMUL": "s32 integer multiply (high/low)",
    "IMUL32I": "s32 integer multiply by an immediate",
    "IDP": "integer dot product (IDP.4A: 4 int8 MACs; IDP.2A: 2 int16xint8) into s32",
    "IDP4A": "dp4a: 4 int8 MACs an instruction into s32",
    "FFMA": "FP32 fused multiply-add",
    "FHFMA": "FP32 fused multiply-add (half rate)",
    "FFMA32I": "FP32 fused multiply-add with an immediate",
    "FMUL": "FP32 multiply",
    "FMUL32I": "FP32 multiply by an immediate",
    "DFMA": "FP64 fused multiply-add",
    "DMUL": "FP64 multiply",
    "HFMA2": "packed FP16x2 fused multiply-add",
    "HMUL2": "packed FP16x2 multiply",
    "MUFU": "multi-function (rcp/rsqrt/sin/...): an approximation, not an exact product",
}
# Paired / dual lane multiply forms (Blackwell): two FP32 FMAs an instruction. `loads.class_of` maps them to the single
# FFMA class, so a census that uses FFMA's cost under-prices them by 2x where they are native (sm_100a) -- UNDERPRICED.
LANE_MUL_PAIRED = {
    "FFMA2": "two FP32 fused multiply-adds an instruction",
    "FMUL2": "two FP32 multiplies an instruction",
    "FHFMA2": "two FP32 (half-rate) fused multiply-adds an instruction",
    "FHMUL2": "two FP32 (half-rate) multiplies an instruction",
}
# Uniform-datapath multiply: one warp-uniform word an instruction, priced on issue.
UNIFORM_MUL = {"UIMAD": "uniform s32 integer multiply-add", "UFFMA": "uniform FP32 fused multiply-add",
               "UFMUL": "uniform FP32 multiply"}

# Lane / uniform adds, logic, shifts, selects, min/max, conversions: write a word but form no product (NO-MULTIPLY).
# A word they write is a sum/logic/select of words that already exist, so it is not a fresh product (NoSum by exactness
# of the no-multiply argument). Priced on the lane route where `loads.class_of` has a class, else unpriced-but-harmless.
LANE_NOMUL = {
    "IADD": "integer add", "IADD3": "three-input integer add", "IADD32I": "integer add immediate",
    "IABS": "integer abs", "ISCADD": "shift-then-add (scale by a power-of-2 constant, not a free product)",
    "ISCADD32I": "shift-then-add immediate", "LEA": "load effective address (a<<n + b: a constant shift-add)",
    "LOP": "bitwise", "LOP3": "3-input bitwise truth table", "LOP32I": "bitwise immediate", "LOP32I": "bitwise immediate",
    "SHF": "funnel shift", "SHL": "shift left", "SHR": "shift right", "BMSK": "bit-field mask", "BREV": "bit reverse",
    "FLO": "find leading one", "POPC": "population count", "SGXT": "sign/zero extend", "PRMT": "byte permute/select",
    "SEL": "select", "IMNMX": "integer min/max", "ISETP": "integer set predicate", "FSETP": "FP set predicate",
    "FSET": "FP set", "FSEL": "FP select", "FADD": "FP32 add", "FADD32I": "FP32 add immediate", "FADD2": "two FP32 adds",
    "FHADD": "FP32 add (half rate)", "FHADD2": "two FP32 adds (half rate)", "HADD2": "packed FP16x2 add",
    "FMNMX": "FP32 min/max", "FMNMX3": "FP32 3-input min/max", "HMNMX2": "packed FP16 min/max",
    "HSET2": "packed FP16 set", "DADD": "FP64 add", "FRND": "FP round", "SGXT": "sign extend",
    "F2F": "float-to-float convert", "F2I": "float-to-int convert", "I2F": "int-to-float convert",
    "I2I": "int-to-int convert", "F2IP": "float-to-int pack", "I2FP": "int-to-float pack", "F2FP": "float-to-float pack",
    "I2IP": "int-to-int pack", "UF2F": "uniform float convert", "UF2I": "uniform float-to-int", "UI2F": "uniform int-to-float",
    "UI2I": "uniform int convert", "UF2FP": "uniform float pack", "UF2IP": "uniform f2i pack", "UI2FP": "uniform i2f pack",
    "UI2IP": "uniform i2i pack", "UFRND": "uniform FP round",
}
# Uniform-datapath adds/logic/shifts/selects (no multiply), each on issue.
UNIFORM_NOMUL = ["UIADD3", "UIABS", "UIMNMX", "ULOP", "ULOP3", "ULOP32I", "USHF", "USHL", "USHR", "USEL", "UISETP",
                 "UFADD", "UFMNMX", "UFSEL", "UFSET", "UFSETP", "UBMSK", "UBREV", "UFLO", "UPOPC", "USGXT", "UPRMT",
                 "ULEA", "UCLEA", "UPLOP3", "UPSETP", "UMEMSETS", "UVIADD", "UVIMNMX", "UVIRTCOUNT", "UGETNEXTWORKID",
                 "USETMAXREG", "ULEPC"]

# SIMD / video integer: packed add, abs-difference (SAD), min/max. No general product (SAD = sum of |differences|).
SIMD_NOMUL = {
    "VABSDIFF": "sum of absolute differences (SAD), not a product", "VABSDIFF4": "4-way byte SAD accumulate",
    "VIADD": "packed SIMD integer add", "VIADDMNMX": "packed add then min/max", "VIMNMX": "packed min/max",
    "VIMNMX3": "packed 3-input min/max", "VHMNMX": "packed half min/max", "VISET": "packed set", "VSET": "packed set",
    "UVIADD": "uniform packed add", "UVIMNMX": "uniform packed min/max",
}
# Cross-lane / coupled reducers: collapse the warp (or couple into a uniform register) to one result by add/min/max.
# No multiply, and as a FOLD a cross-lane reduction heads one word a warp instruction = 32 lane instructions a word
# (isa_escape FOLD). Some are not named in floors.csv/isa_escape (CREDUX, FSWZADD, REDS) -- unpriced but NoSum/FOLD holds.
CROSSLANE_REDUCE = {
    "REDUX": "warp reduce (add/min/max/and/or/xor) to one result", "CREDUX": "coupled reduction into a uniform register",
    "FSWZADD": "FP32 swizzle-add (a fixed cross-lane add pattern)", "SHFL": "warp shuffle (a mover; data-dependent select)",
}
# Side-unit reducers / atomics: an exact add/min/max into memory, no multiply, priced at the pi32 floor (floors.csv).
SIDE_REDUCE = {
    "RED": "memory reduction", "REDG": "global reduction", "REDS": "shared reduction", "REDAS": "async shared reduction",
    "ATOM": "atomic op (returns the landed word)", "ATOMG": "global atomic", "ATOMS": "shared atomic",
    "SURED": "surface reduction", "SUATOM": "surface atomic", "UBLKRED": "TMA bulk reduce", "UTMAREDG": "TMA reduce to global",
    "UREDGR": "uniform global reduce", "LDGMC": "reducing / multicast load (NVLS)",
}
# Movers: no arithmetic. The output word equals an existing word (a copy / transpose / byte-select / load / store).
MOVERS = {
    "MOV": "register move", "MOV32I": "move immediate", "MOV64IUR": "move 64-bit immediate from UR", "UMOV": "uniform move",
    "MOVM": "matrix/transpose move", "R2UR": "register-to-uniform", "S2UR": "special-to-uniform", "R2P": "register-to-predicate",
    "P2R": "predicate-to-register", "S2R": "special register read", "CS2R": "constant special read", "B2R": "barrier-to-register",
    "LD": "load", "LDG": "global load", "LDS": "shared load", "LDL": "local load", "LDC": "constant load", "LDCU": "uniform constant load",
    "LDSM": "shared matrix load (ldmatrix)", "STSM": "shared matrix store (stmatrix)", "ST": "store", "STG": "global store",
    "STS": "shared store", "STAS": "async store to distributed shared memory", "STL": "local store",
    "LDGSTS": "async global-to-shared copy (cp.async)", "LDGDEPBAR": "cp.async barrier",
    "LDT": "tensor-memory load (tcgen05.ld)", "LDTM": "tensor-memory load", "STT": "tensor-memory store", "STTM": "tensor-memory store",
    "UTCCP": "tcgen05 copy into tensor memory", "UTCSHIFT": "tcgen05 shift within tensor memory", "UTCATOMSWS": "tcgen05 atom swizzle",
    "UBLKCP": "TMA bulk copy", "UBLKPF": "TMA bulk prefetch", "UTMALDG": "TMA load", "UTMASTG": "TMA store", "UTMAPF": "TMA prefetch",
    "UMEMSETS": "shared memset (writes a constant)", "USTGR": "uniform global store", "SPARSIFY": "compress to 2:4 sparse (data reshape)",
    "DECOMPRESS": "expand a 2:4 sparse operand (data reshape)", "LDCU": "uniform constant load",
}
# Texture / surface sampling: a convex low-precision weighted sum over a fixed basis, format-converted and rounded;
# priced at the pi32 floor (LOW-PRECISION / sidefloor). No free exact s32.
TEXTURE = {"TEX": "texture filter", "TLD": "texture load", "TLD4": "texture gather-4", "TXD": "texture derivative",
           "TMML": "texture mip-map level", "TXQ": "texture query", "SULD": "surface load", "SUST": "surface store"}
# Control / predicate / barrier / misc: no committed word at all.
CONTROL = {"BRA", "BRX", "BRXU", "JMP", "JMX", "JMXU", "CALL", "RET", "EXIT", "BSSY", "BSYNC", "BREAK", "BPT", "KILL",
           "YIELD", "NANOSLEEP", "WARPSYNC", "BAR", "MEMBAR", "FENCE", "DEPBAR", "LDGDEPBAR", "NOP", "VOTE", "VOTEU",
           "ELECT", "WARPGROUP", "WARPGROUPSET", "ACQBULK", "ACQSHMINIT", "CGAERRBAR", "ERRBAR", "ENDCOLLECTIVE",
           "PREEXIT", "RPCMOV", "BMOV", "LEPC", "ULEPC", "PMTRIG", "SETCTAID", "GETLMEMBASE", "SETLMEMBASE", "QSPC",
           "MATCH", "SYNCS", "SYNCSU", "USYNCS", "CCTL", "CCTLL", "CCTLT", "CCTLU", "UTMACCTL", "UTMACMDFLUSH",
           "UTCBAR", "UBLKL2CCTL", "UTMAL2CCTL", "UBLKPF", "UTMAPF", "PLOP3", "UPLOP3", "PSETP", "UPSETP", "P2R", "R2P",
           "UP2UR", "UR2UP", "CS2UR", "VISET", "UVIRTCOUNT", "Option", "SETLMEMBASE", "TXQ", "DEPBAR",
           # cp.async.mbarrier.arrive: arrives on a shared-memory mbarrier as prior LDGSTS land (writes barrier state, no
           # register). Absent from NVIDIA's Instruction Set Reference, yet emitted on sm_90/sm_100/sm_120.
           "ARRIVES"}


# Compare / set-predicate / range-check ops: they write a predicate (or a one-slot flag), not a committed s32 word.
PREDICATE = {"ISETP", "FSETP", "DSETP", "HSETP2", "PSETP", "UISETP", "UFSETP", "UPSETP", "VSETP", "VISETP", "FCHK"}


def classify(base: str) -> dict:
    if base in PREDICATE:
        return {"mul": False, "word": False, "route": "control", "note": "compare / set-predicate (writes a predicate)"}
    """The semantic record for a SASS base opcode: `mul` (can form a product), `word` (writes a >=32-bit result that
    could be a committed word), `route` (the menu route that prices it, or None), and a human `note`."""
    if base in MMA:
        fam = "wgmma" if base in MMA_WGMMA else "tcgen05" if base in MMA_TCGEN05 else "mma.sync"
        return {"mul": True, "word": True, "route": "tensor", "note": f"{fam} matrix multiply-accumulate"}
    if base in LANE_MUL:
        return {"mul": base != "MUFU", "word": True, "route": "lane", "note": LANE_MUL[base]}
    if base in LANE_MUL_PAIRED:
        return {"mul": True, "word": True, "route": "lane", "paired": True, "note": LANE_MUL_PAIRED[base]}
    if base in UNIFORM_MUL:
        return {"mul": True, "word": True, "route": "uniform", "note": UNIFORM_MUL[base]}
    if base in SIDE_REDUCE:
        return {"mul": False, "word": True, "route": "sidefloor", "reduce": True, "note": SIDE_REDUCE[base]}
    if base in CROSSLANE_REDUCE:
        mover = base == "SHFL"
        return {"mul": False, "word": True, "route": "mover" if mover else None, "reduce": not mover,
                "note": CROSSLANE_REDUCE[base]}
    if base in TEXTURE:
        return {"mul": False, "word": True, "route": "sidefloor", "note": TEXTURE[base]}
    if base in MOVERS:
        return {"mul": False, "word": True, "route": "mover", "note": MOVERS[base]}
    if base in SIMD_NOMUL:
        return {"mul": False, "word": True, "route": "lane", "note": SIMD_NOMUL[base]}
    if base in LANE_NOMUL:
        return {"mul": False, "word": True, "route": "lane", "note": LANE_NOMUL[base]}
    if base in UNIFORM_NOMUL:
        return {"mul": False, "word": True, "route": "uniform", "note": "uniform-datapath add/logic/shift/select"}
    if base in CONTROL:
        return {"mul": False, "word": False, "route": "control", "note": "control / predicate / barrier / misc"}
    return {"mul": None, "word": None, "route": None, "note": "unclassified base opcode (new to the table)"}
