"""facet_break: a measured program's load on every facet of Π, the facets it beats, and the refit a beaten facet needs.

The standing adversary for the sm_120 data sheet. A facet is a row of the
machine model (`../machine_model/data/`): row r holds Σ_k c_kr·x_k SMSP-cycles for a program that issues x_k warp
instructions of census class k a SMSP-cycle, and the facet is the ceiling a_r·x ≤ 1. A program's **load** on r is
Σ_k c_kr·x_k (dimensionless): a measured program with load > 1 beats the facet, since the rows fitted from rates carry the
2% margin in their coefficients (a point measured on its own facet sits at 0.98). Exact rows (`issue` 1 cycle, `M` 16) are
reported the same way; a load in (1, 1.02] on them is inside timing noise.

The rate of a census class k in a probe's `rows.jsonl` row is the sum, over every timed loop, of that loop's rate of
every SASS opcode of class k (warp instructions/SM/clk), divided by 4 (SMSPs an SM) to get x_k a SMSP-cycle. `CLASS_OF`
maps a SASS opcode to its census class. Only opcodes whose class is certain are counted, so a reported load is a lower
bound and a reported violation is real.

`--full` counts the rest of a single-loop program's timed loop too (the loop its run's gate verdict names, read from the
run's `sass/` listings): each instruction the gate didn't pin runs once an iteration, at a pinned instruction's rate, in
its census class, or on `issue` alone when it has none (BRA, ISETP, NOP). The machine model's `Run.n` counts every
executed instruction, so the full load is the one a row must hold; it is still a lower bound (an unclassed instruction
loads only `issue`).

  python loads.py loads OUT/rows.jsonl... [--epoch E] [--probe NAME] [--top N]   # each program's heaviest facets
  python loads.py summary OUT/rows.jsonl... [--epoch E] [--margin 0] [--json]    # every violation (load > 1 + margin)
  python loads.py refit OUT/rows.jsonl... --row R [--epoch E]                    # the refit line a violation needs
  python loads.py table OUT/rows.jsonl... [--epoch E]                            # every program's loads, a JSON line each
  python loads.py summary OUT/rows.jsonl... --full                               # with every loop instruction counted

`--generic` loads every MMA form through the generic `MMA` lines where its own kind has none (advisory). The generator of
the programs is `generate.py`, the batch `run.sh`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fnmatch import fnmatchcase
from fractions import Fraction
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
MM = HERE.parent / "machine_model"
sys.path.insert(0, str(MM))

from verity_machine_model.model import census, model  # noqa: E402

SMSPS = 4

#: a SASS opcode (the dotted name, no guard) to its census class. The MMA forms map to their own kind (for the joint
#: issue rows, each form's own coefficient); every MMA form also loads the generic-pipe rows through `MMA` (see
#: `mma_kind`). The order matters: the first matching rule wins.
CLASS_RULES: list[tuple[str, str]] = [
    # tensor forms (the exact mma.sync SASS)
    (r"IMMA\.16832", "IMMA16832"), (r"IMMA\.16816", "IMMA16816"), (r"IMMA\.8816|IMMA\.1688", "IMMA8816"),
    (r"IMMA\.SP\.16864", "IMMASP16864"), (r"IMMA\.SP\.16832", "IMMASP16832"),
    (r"HMMA\.16816\.F16", "HMMA16816F16"), (r"HMMA\.16816", "HMMA16816"),
    (r"HMMA\.SP\.16832", "HMMASP16832"), (r"HMMA\.1688", "HMMA1688T"), (r"HMMA\.1684", "HMMA1684T"),
    (r"QMMA\.SF\.16832", "QMMASF16832"), (r"QMMA\.SP\.16864", "QMMASP16864"), (r"QMMA\.16832", "QMMA16832"),
    (r"OMMA\.SF\.16864|OMMA", "OMMASF16864"),
    # loads (the table routes'): a sub-word load is its word's class (floors.csv); the 64- and 256-bit and the
    # L1-skipping (.NA, .STRONG.GPU) loads are classes from VF3q-16, `unpriced` at an epoch without their lines
    (r"^LDS(\.[US](8|16))?$", "LDS32"), (r"^LDS\.64$", "LDS64"), (r"^LDS\.128$", "LDS128"),
    (r"^LDG\.E\.NA\.128$", "LDG128NA"), (r"^LDG\.E\.128\.STRONG\.GPU$", "LDG128CG"),
    (r"^LDG\.E\.ENL2\.256\.STRONG\.GPU$", "LDG256CG"), (r"^LDG\.E\.ENL2\.256(\.CONSTANT)?$", "LDG256"),
    (r"^LDG\.E(\.[US](8|16))?(\.CONSTANT)?$", "LDG32"), (r"^LDG\.E\.64(\.CONSTANT)?$", "LDG64"),
    (r"^LDG\.E\.128(\.CONSTANT)?$", "LDG128"),
    # the side units (floors.csv's RED*, ATOM*, LDSM*, STSM*): a reduction's width is its class, an L2 select op
    # (min, max, and, or, xor) its own; ATOM.E is ATOME32 whatever its window, since SASS doesn't show a peer CTA's
    (r"^REDG\.E\.ADD\.[A-Z0-9_]+x4\b", "REDV"), (r"^REDG\.E\.ADD\.64(\.|$)", "RED64"), (r"^REDG\.E\.ADD(\.STRONG|$)", "RED32"),
    (r"^REDG\.E\.(MIN|MAX|AND|OR|XOR)\b.*\.64(\.|$)", "REDG64"), (r"^REDG\.E\.(MIN|MAX|AND|OR|XOR)\b", "REDG32"),
    (r"^ATOMG\.E\..*\.64(\.|$)", "ATOMG64"), (r"^ATOMG\.E\.", "ATOMG32"),
    (r"^ATOMS\..*\.64$", "ATOMS64"), (r"^ATOMS\.", "ATOMS32"),
    (r"^ATOM\.E\..*\.64(\.|$)", "ATOME64"), (r"^ATOM\.E\.", "ATOME32"),
    (r"^STS\.128$", "STS128"), (r"^LDGSTS\.E\.BYPASS\.128$", "LDGSTS16"),
    # the stores, a sub-word store its word's class
    (r"^STS(\.[US](8|16))?$", "STS32"), (r"^STG\.E(\.[US](8|16))?$", "STG32"), (r"^STG\.E\.128$", "STG128"),
    (r"^LDSM\.16\.M88$", "LDSM_X1"), (r"^LDSM\.16\.M88\.2$", "LDSM_X2"), (r"^LDSM\.16\.M88\.4$", "LDSM_X4"),
    (r"^LDSM\.16\.MT88$", "LDSM_X1T"), (r"^LDSM\.16\.MT88\.2$", "LDSM_X2T"), (r"^LDSM\.16\.MT88\.4$", "LDSM_X4T"),
    (r"^LDSM\.8\.MT1616$", "LDSM8T_X1"), (r"^LDSM\.8\.MT1616\.2$", "LDSM8T_X2"),
    (r"^STSM\.16\.M88$", "STSM_X1"), (r"^STSM\.16\.M88\.2$", "STSM_X2"), (r"^STSM\.16\.M88\.4$", "STSM_X4"),
    (r"^STSM\.16\.MT88$", "STSM_X1T"), (r"^STSM\.16\.MT88\.2$", "STSM_X2T"), (r"^STSM\.16\.MT88\.4$", "STSM_X4T"),
    (r"^STSM\.8\.MT1616$", "STSM8T_X1"), (r"^STSM\.8\.MT1616\.2$", "STSM8T_X2"), (r"^STSM\.8\.MT1616\.4$", "STSM8T_X4"),
    # the warp reduce, the predicate test and sm_100's VIADD (ahead of IADD, which VIADD contains): each its own class,
    # which the H200's and B200's census price on their issue rows alone
    (r"^REDUX\.", "REDUX"), (r"^ISETP\.", "ISETP"), (r"^VIADD\b", "VIADD"),
    # scalar / CUDA lane classes (floors.csv and census share these names)
    (r"IMAD\.WIDE", "IMADW"), (r"IADD\.64", "ALU64"), (r"DFMA|DADD", "W2"), (r"DMUL", "DMUL"),
    (r"^IDP\.", "IMAD"),  # IDP.4A: the IMAD pipe's (rows.csv's I, the dp4a design's IMAD lanes)
    (r"IMAD|IMUL|IADD3|IADD|LEA\b|LEA\.HI", "ALU_OR_IMAD"),  # resolved below
    (r"LOP3|SHF\b|SHF\.|PRMT|LOP\b", "ALU"),
    (r"HFMA2", "HFMA2"), (r"HADD2|HMUL2", "HFMA2"),
    (r"FFMA|FADD|FMUL", "FFMA"),
    (r"VIMNMX|IMNMX|VMNMX|MNMX", "VIMNMX"),
    (r"F2F|F2I|I2F|I2I|FRND|F2FP", "F2FS"),
    (r"MUFU", "XU"),
    (r"SHFL", "SHFL"),
    (r"SEL\b|SEL\.", "SEL"),
    (r"POPC|FLO|BREV", "XU"),   # POPC on the XU (the bit-slice / XOR-popc route's counter)
]

#: mad.lo.u32 disassembles to IMAD on sm_120; IADD3/LOP3/SHF/PRMT are the ALU group. `IMAD` is its own census class.
_IMAD = re.compile(r"\bIMAD\b|\bIMUL\b")
_ALU = re.compile(r"\bIADD3\b|\bIADD\b|\bLOP3\b|\bLOP\b|\bSHF\b|\bPRMT\b|\bLEA\b")


#: a uniform-datapath instruction (`opcode_probe/eps1.py`'s UNIFORM): census class UNI, or UNI64 when it writes a UR pair
#: (UIADD3.64, UMOV.64, UIMAD.WIDE, LDCU.64), ahead of the vector rules that would match ULOP3, USHF, UPRMT or USEL by
#: their suffix
UNIFORM = re.compile(r"U[A-Z].*|LDCU|R2UR|S2UR|CS2UR|VOTEU")
UR_PAIR = re.compile(r"\.(64|WIDE)\b")
#: the bulk-copy and TMA units' shared-memory instructions (`cp.async.bulk`, `cp.reduce.async.bulk`, their `.tensor`
#: forms): one costs its shared-memory bytes on DS_smem (A5b), which its opcode doesn't show, so `bulk_counts` classes
#: it by its bytes and `class_of` takes it ahead of the uniform rule that would call it UNI: unclassed,
#: but for a bulk reduction (`BULK_RED`)
BULK = re.compile(r"UBLK(CP|RED)|UTMA(LDG|STG|REDG)")
#: a bulk reduction (UBLKRED) by its opcode alone is TMAred512, the census's class for one of 512 B: its issue slot and
#: 512 B on DS_smem on DS-120, its own unit's rows on the measured census. That bounds below every bulk reduction the
#: tree runs on sm_120 (side.cu's and smem_side.cu's 512 B to 4 KB, cpw_routes' 1 to 16 KB, fold_bulk's 8 to 32 KB);
#: one under 512 B would be overstated, so it must be counted by its bytes
BULK_RED = "UBLKRED"
#: UTMABIG's bytes on DS-120: 99 KB, the most shared memory a block holds on sm_120 (CUDA C++ Programming Guide,
#: compute capability 12.0), so a bulk instruction's count `S/99 KB` is at most one and a box is never one UTMABIG there
BULK_BYTES = 101376


def bulk_counts(opcode: str, nbytes: int) -> dict[str, Fraction]:
    """A bulk or TMA instruction's census counts (A5b): `S/99 KB` in UTMABIG, `S` its size, so DS-120 charges DS_smem
    exactly `S` and DS_issue at most its one slot.

    `nbytes` is its size in bytes (the SASS size register holds bytes/16: fold_bulk.cu's `0x8000` bytes load as
    `0x800`). Each of these instructions reads or writes `S` bytes of a block's shared memory on at least one side, so
    `S` is a lower bound on its bank bytes (a reduction into shared memory moves `3S`) and at most 99 KB."""
    op = opcode.split("@")[-1].strip()
    if not BULK.match(op.split(".")[0]):
        raise ValueError(f"{opcode} is no bulk or TMA instruction")
    if not 0 < nbytes <= BULK_BYTES:
        raise ValueError(f"{opcode} of {nbytes} B: a bulk instruction moves 1 B to a block's 99 KB of shared memory")
    return {"UTMABIG": Fraction(nbytes, BULK_BYTES)}


def class_of(opcode: str) -> str | None:
    """The census class of a dotted SASS opcode (`IMMA.16832.S8.S8`, `LOP3.LUT`, `SHFL.BFLY`), or None when not certain:
    a bulk copy's or TMA instruction's class is its bytes' (`bulk_counts`), and a bulk reduction's TMAred512
    (`BULK_RED`)."""
    op = opcode.split("@")[-1].strip()
    base = op.split(".")[0]
    if BULK.match(base):
        return "TMAred512" if base.startswith(BULK_RED) else None
    if UNIFORM.fullmatch(base):
        return "UNI64" if UR_PAIR.search(op) else "UNI"
    for pat, kind in CLASS_RULES:
        if re.search(pat, op):
            if kind == "ALU_OR_IMAD":
                return "IMAD" if _IMAD.search(op) else "ALU" if _ALU.search(op) else None
            return kind
    if base == "MOV":
        return "NOROW"
    return None


MMA_FORMS = frozenset({"IMMA16832", "IMMA16816", "IMMA8816", "IMMASP16864", "IMMASP16832", "HMMA16816F16", "HMMA16816",
                       "HMMASP16832", "HMMA1688T", "HMMA1684T", "QMMASF16832", "QMMASP16864", "QMMA16832",
                       "OMMASF16864"})
#: the joint issue rows (JI3n_0..2, JI3o_0..2, …), where a MMA form is priced by its own kind; on every other row it
#: loads through generic `MMA`
JI_ROW = re.compile(r"^JI\d")


#: a TLD's class is its width (the registers it writes), which its operands show and its opcode doesn't: a generated
#: memory program names it (`_tld4_`: a 16-byte texel, `_tld4n_`: four registers from a 4-byte texel)
TLD_BY_NAME = {"tld1": "TLD1", "tld2": "TLD2", "tld4": "TLD4w", "tld4n": "TLD4w"}
_TLD_NAME = re.compile(r"_(tld4n|tld4|tld2|tld1)_")

#: side.cu counts its side unit by a wildcard (`REDG*`, `UBLKRED*`), so its unit's name (`hs_red64`, `sa_red64`) is the
#: class; a unit missing here takes its pattern's `class_of` (sa_blkred4k's 4 KB reduction is one TMAred512, a lower
#: bound)
SIDE_BY_NAME = {"red32": {"REDG*": "RED32"}, "red64": {"REDG*": "RED64"}, "redv4": {"REDG*": "REDV"},
                "redv4h": {"REDG*": "REDV"}, "redblk": {"REDG*": "RED32"}, "atomg32": {"ATOMG*": "ATOMG32"},
                "atomgd": {"ATOMG*": "ATOMG32"}, "ldgsts": {"LDGSTS*": "LDGSTS16"},
                "blkred512": {"UBLKRED*": "TMAred512"}, "blkcp4k": {"UBLKCP*": "UBLKCP4K"},
                "dsm_red": {"ATOM*": "ATOME32P"}, "dsm_st4": {"ST*": "DSMST4P"},
                **{t: {"TLD*": "TLD1"} for t in ("tld1", "tld1c", "tld1d", "tld1cd", "tld1w", "tld1cw")},
                **{t: {"TEX*": "TEXBIL"} for t in ("texbil", "texc", "texd", "texr", "texw")}}
_SIDE_NAME = re.compile(r"^(?:hs|sa)_(\w+)$")


def side_class(probe: str, pat: str) -> str | None:
    m = _SIDE_NAME.match(probe or "")
    return SIDE_BY_NAME.get(m.group(1), {}).get(pat) if m else None


def rates_by_class(row: dict) -> dict[str, float]:
    """x_k a SMSP-cycle for each census class k the probe's loops touched (its `rates`, keyed `loop:pattern`)."""
    out: dict[str, float] = {}
    for key, rate in row["rates"].items():
        pat = key.split(":", 1)[1]
        k = side_class(row.get("probe"), pat) or class_of(pat)
        if k is None and pat.startswith("TLD") and (t := _TLD_NAME.search(row.get("probe") or "")):
            k = TLD_BY_NAME[t.group(1)]
        if k is not None:
            out[k] = out.get(k, 0.0) + rate / SMSPS
    return out


def unpriced(row: dict, c: dict | None = None) -> dict[str, float]:
    """The program's load opcodes with no census class, or (given the epoch's census `c`) a class without lines there,
    x a SMSP-cycle: Π prices them at 0."""
    out: dict[str, float] = {}
    for key, rate in (row.get("rates") or {}).items():
        pat = key.split(":", 1)[1]
        named_tld = pat.startswith("TLD") and _TLD_NAME.search(row.get("probe") or "")
        k = class_of(pat)
        if pat.split(".")[0] in ("LDS", "LDG", "TLD") and not named_tld and (k is None or (c is not None and k not in c)):
            out[pat] = out.get(pat, 0.0) + rate / SMSPS
    return out


#: the table route's L2 facet (`../machine_model/data/table.toml`, what ε's `--lookups` prices a lookup's bytes against):
#: L2 read bytes a second ≤ `l2`. Only loads that skip L1 (.NA, .STRONG.GPU) count, so their bytes are L2's, and each at
#: its own width (a 16-byte load of a 32-byte sector counts 16), so the load is a lower bound.
L2_ROW = "L2tbl"
_L2_BYPASS = re.compile(r"^LDG\..*\.(NA|STRONG\.GPU)(\.|$)")
_L2_WIDTH = ((".256", 32), (".128", 16), (".64", 8))


def l2_default() -> float | None:
    t = MM / "data" / "table.toml"
    if not t.exists():
        return None
    import tomllib
    return float(tomllib.loads(t.read_text())["l2"])


def l2_bytes_per_s(row: dict) -> float:
    """The program's L2 read bytes a second: each L1-skipping load's rate a SM-clock, x 32 lanes x its width, x the SMs
    timed (one block an SM) x the measured clock."""
    b = 0.0
    for key, rate in (row.get("rates") or {}).items():
        pat = key.split(":", 1)[1]
        if _L2_BYPASS.match(pat) and not re.search(r"\.[US](8|16)\b", pat):
            b += rate * 32 * next((w for s, w in _L2_WIDTH if s in pat), 4)
    return b * (row.get("blocks") or 0) * (row.get("sm_mhz") or 0) * 1e6


#: the form the generic `MMA` lines were measured with (the honest kernel's mma.sync): it loads every row its own kind has
#: no line on through `MMA`. Every other form is its own kind with lines on M, issue and the joint issue rows only, so the model prices it
#: at 0 on the rest; `generic=True` loads it through `MMA` there too (advisory: a form's missing line can't be violated).
MMA_MEASURED_WITH = "IMMA16832"


def facet_loads(x: dict[str, float], c: dict[str, dict[str, float]], cycle_u: float,
                generic: bool = False) -> dict[str, float]:
    """A program's load Σ_k c_kr·x_k on every row r. A MMA form takes its own kind's lines and, if it is the form the
    generic `MMA` lines were measured with (or `generic`), `MMA`'s on every row it has none on."""
    coeff: dict[str, dict[str, float]] = {}
    for k, xk in x.items():
        per_cycle = {r: u / cycle_u for r, u in c.get(k, {}).items()}
        if k in MMA_FORMS and (generic or k == MMA_MEASURED_WITH):
            for r, u in c.get("MMA", {}).items():
                if r not in per_cycle and not JI_ROW.match(r):
                    per_cycle[r] = u / cycle_u
        coeff[k] = per_cycle
    load: dict[str, float] = {}
    for k, xk in x.items():
        for r, ck in coeff[k].items():
            load[r] = load.get(r, 0.0) + ck * xk
    return load


def program(row: dict, c, cycle_u: float, generic: bool = False, l2: float | None = None) -> dict:
    x = row["x"] if "x" in row else rates_by_class(row)
    load = facet_loads(x, c, cycle_u, generic)
    if "x_issue" in row and "issue" in load:
        # a full loop's instruction whose class this census has no line for still issues once
        load["issue"] += row["x_issue"] + sum(v for k, v in x.items() if not facet_loads({k: 1.0}, c, cycle_u, generic))
    bps = l2_bytes_per_s(row)
    if l2 and bps > 0:
        load[L2_ROW] = bps / l2
    return {"probe": row.get("probe"), "run": row.get("run"), "sass": (row.get("sass_sha256") or "")[:16],
            "sm_mhz": row.get("sm_mhz"), "tags": row.get("tags", []), "x": x, "load": load, "l2_bps": bps,
            "full": "x_issue" in row}


#: tensor_kinds: each form name's census kind, and each mode's lane op split
#: over census classes (spec/mixu: IADD3 and FFMA alternating; *_iadd3: IADD3 alone)
TK_KIND = {"imma_s8_m16n8k32": "IMMA16832", "imma_u8s8_m16n8k32": "IMMA16832", "imma_s8_m16n8k32_sat": "IMMA16832",
           "imma_s8_m16n8k16": "IMMA16816", "imma_s8_m8n8k16": "IMMA8816", "imma_sp_s8_m16n8k64": "IMMASP16864",
           "imma_sp_s8_m16n8k32": "IMMASP16832", "hmma_f16_m16n8k16_f32": "HMMA16816",
           "hmma_bf16_m16n8k16_f32": "HMMA16816", "hmma_f16_m16n8k16_f16": "HMMA16816F16",
           "hmma_sp_f16_m16n8k32_f32": "HMMASP16832", "hmma_tf32_m16n8k8": "HMMA1688T", "hmma_tf32_m16n8k4": "HMMA1684T",
           "qmma_e4m3_m16n8k32_f32": "QMMA16832", "qmma_e5m2_m16n8k32_f32": "QMMA16832",
           "qmma_e2m1_m16n8k32_f32": "QMMA16832", "qmma_sf_e4m3_m16n8k32_f32": "QMMASF16832",
           "qmma_sp_e4m3_m16n8k64_f32": "QMMASP16864", "omma_nvf4_m16n8k64": "OMMASF16864"}
TK_LANE = {"iadd3": {"ALU": 1.0}, "iadd3_32": {"ALU": 1.0}, "ffma": {"FFMA": 1.0},
           "iadd3_ffma": {"ALU": 0.5, "FFMA": 0.5}}


def from_tensor_kinds(d: dict, run: str, mixed_lane: str = "iadd3_ffma") -> dict | None:
    """A tensor_kinds.jsonl record as a probe row: its MMAs and lane ops a SM over its median block's SM cycles. Only the
    PTX lane ops are counted (not the loop's own IADD3/ISETP/BRA), so the load is a lower bound. `mixed_lane` is the
    lane of the file's spec/mixu modes: IADD3 alone in the first version (de1d918, its opalone form `iadd3_32`)."""
    if "mode" not in d or not d.get("cyc_med"):
        return None
    cyc = d["cyc_med"]
    if d["mode"] == "opalone":
        lane, form = d["form"], None
    else:
        lane, form = ("iadd3" if d["mode"].endswith("_iadd3") else mixed_lane), d["form"]
    x: dict[str, float] = {}
    if form:
        x[TK_KIND[form]] = d["mmas_per_sm"] / cyc / SMSPS
    for k, share in TK_LANE[lane].items():
        if d["ops_per_sm"]:
            x[k] = x.get(k, 0.0) + share * d["ops_per_sm"] / cyc / SMSPS
    name = f"{d['mode']}:{form or lane}:R{round(d.get('ops_per_mma', 0) * 8)}:w{d['mma_warps']}+{d['op_warps']}"
    return {"probe": name, "run": run, "x": x, "sm_mhz": d.get("mhz"), "tags": []}


#: a cuobjdump listing's function header, and an instruction's address and dotted opcode (past any guard predicate)
_SASS_FUNC = re.compile(r"\n\s*Function : (\S+)\n")
_SASS_INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)")


def gate_loops(out: Path) -> dict[str, list[str]]:
    """Each gated program's timed loop (its gate verdict's loop 0, from `gate-build.json` or `gate.json` in the run's
    `out/`) as the opcodes of its instructions in `out/sass/`'s listings; empty when the run kept neither."""
    gate = next((g for g in (out / "gate-build.json", out / "gate.json") if g.exists()), None)
    if gate is None or not (out / "sass").is_dir():
        return {}
    bodies: dict[str, str] = {}
    for s in sorted((out / "sass").glob("*.sass")):
        parts = _SASS_FUNC.split("\n" + s.read_text())
        bodies |= dict(zip(parts[1::2], parts[2::2]))
    loops = {}
    for v in json.loads(gate.read_text())["verdicts"]:
        ver = v.get("verdict") or {}
        body = bodies.get(ver.get("kernel") or v.get("probe"))
        if v.get("passed") and ver.get("loops") and body is not None:
            lo, hi = int(ver["loops"][0]["head"], 16), int(ver["loops"][0]["tail"], 16)
            loops[v["probe"]] = [m.group(2) for m in _SASS_INS.finditer(body) if lo <= int(m.group(1), 16) <= hi]
    return loops


def full_x(row: dict, loop: list[str]) -> tuple[dict[str, float], float] | None:
    """The program's x with every instruction of its timed loop counted, and the x of the loop's instructions with no
    census class (one `issue` cycle each, on no other row; `program` counts a class its census has no line for the
    same way). A loop instruction runs once an iteration, so one the gate
    didn't pin runs at the pinned instructions' rate over their number in the loop. None for a program timed over more
    than one loop, or whose listed loop holds none of its pinned opcodes."""
    rates = row.get("rates") or {}
    if not rates or any(k.split(":", 1)[0] != "0" for k in rates):
        return None
    pats = [k.split(":", 1)[1] for k in rates]
    free = [op for op in loop if not any(fnmatchcase(op, p) for p in pats)]
    if len(free) == len(loop):
        return None
    per = sum(rates.values()) / SMSPS / (len(loop) - len(free))
    x, other = rates_by_class(row), 0.0
    for op in free:
        if (k := class_of(op)) is None:
            other += per
        else:
            x[k] = x.get(k, 0.0) + per
    return x, other


def load_rows(paths: list[Path], full: bool = False) -> list[dict]:
    """Every program in the given files: an opcode probe's `rows.jsonl` (rates by SASS pattern) or tensor_kinds.jsonl.
    With `full`, a single-loop program whose run kept its gate verdicts and listings takes `full_x`'s x."""
    out = []
    for p in paths:
        run = next((s for s in p.parts if s.startswith("r20")), "")
        recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        mixed = "iadd3_32" if any(d.get("mode") == "opalone" and d.get("form") == "iadd3_32" for d in recs) \
            else "iadd3_ffma"
        loops = gate_loops(p.parent) if full else {}
        for d in recs:
            if d.get("rates"):
                f = full_x(d, loops[d["probe"]]) if d.get("probe") in loops else None
                out.append(d if f is None else {**d, "x": f[0], "x_issue": f[1]})
            elif (t := from_tensor_kinds(d, run, mixed)) is not None:
                out.append(t)
    return out


def in_vector(epoch: str) -> set[str]:
    """The rows the named vector's price LP (and CALC's ε LP) reads, so a violation of one is flagged as ε-reading."""
    from verity_machine_model.model import vector, vectors
    for n in vectors():
        v = vector(n)
        if v.epoch == epoch:
            return set(v.rows)
    return set()


def cmd_loads(a) -> int:
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    for row in load_rows(a.rows, a.full):
        p = program(row, c, m.cycle_u, a.generic, a.l2)
        top = sorted(p["load"].items(), key=lambda t: -t[1])[: a.top]
        if a.probe and p["probe"] != a.probe:
            continue
        mix = ", ".join(f"{k} {v:.4f}" for k, v in sorted(p["x"].items(), key=lambda t: -t[1]))
        print(f"{p['probe']:28s} {p['sm_mhz'] or 0:>5}MHz  {mix}")
        print("    " + "  ".join(f"{r} {v:.3f}" for r, v in top))
    return 0


def cmd_table(a) -> int:
    """Every program as a JSON line: its mix, its load on every row and its heaviest row (the results table)."""
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    vrows = in_vector(epoch)
    for row in load_rows(a.rows, a.full):
        p = program(row, c, m.cycle_u, a.generic, a.l2)
        top = max(p["load"].items(), key=lambda t: t[1]) if p["load"] else ("", 0.0)
        print(json.dumps({"epoch": epoch, **p, "max_row": top[0], "max_load": top[1], "max_in_vector": top[0] in vrows,
                          "load": {r: round(v, 6) for r, v in sorted(p["load"].items())}, "unpriced": unpriced(row, c)}))
    return 0


def cmd_summary(a) -> int:
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    vrows = in_vector(epoch)
    viol = []
    for row in load_rows(a.rows, a.full):
        p = program(row, c, m.cycle_u, a.generic, a.l2)
        for r, v in p["load"].items():
            if v > 1 + a.margin:
                viol.append({"row": r, "load": v, "probe": p["probe"], "run": p["run"], "sass": p["sass"],
                             "sm_mhz": p["sm_mhz"], "in_vector": r in vrows, "tags": p["tags"], "x": p["x"],
                             "l2_bps": p["l2_bps"]})
    viol.sort(key=lambda d: -d["load"])
    if a.json:
        print(json.dumps(viol, indent=1))
        return 1 if viol else 0
    rows = load_rows(a.rows, a.full)
    print(f"epoch {epoch}: {len(rows)} programs, margin {a.margin}")
    free: dict[str, tuple[float, str]] = {}
    for row in rows:
        for pat, x in unpriced(row, c).items():
            if x > free.get(pat, (0.0, ""))[0]:
                free[pat] = (x, f"{row.get('probe')} @ {row.get('run')}")
    for pat, (x, by) in sorted(free.items()):
        print(f"unpriced: {pat} (no census class: Π prices it at 0) at up to {x:.4f} a SMSP-cycle, {by}")
    if not viol:
        print("no facet beaten (every program loads <= 1 + margin on every row)")
        return 0
    worst: dict[str, dict] = {}
    for d in viol:
        worst.setdefault(d["row"], d)
    print(f"{len(worst)} rows beaten ({len(viol)} program-row violations); the worst program of each:")
    for d in worst.values():
        n = sum(v["row"] == d["row"] for v in viol)
        print(f"  {d['row']:12s} {d['load']:.4f}{' [ε]' if d['in_vector'] else '    '} x{n:<4d} {d['probe']} @ {d['run']}")
    if a.worst:
        return 1
    print("every violation:")
    for d in viol:
        star = " [ε]" if d["in_vector"] else ""
        mix = ", ".join(f"{k} {x:.4f}" for k, x in sorted(d["x"].items(), key=lambda t: -t[1]))
        if d["row"] == L2_ROW:
            mix = f"L2 reads {d['l2_bps'] / 1e12:.3f} TB/s against l2 {a.l2 / 1e12:.3f}" + (f"; {mix}" if mix else "")
        print(f"  {d['row']:12s} load {d['load']:.4f}{star}  {d['probe']} @ {d['run']} sass {d['sass']} "
              f"({d['sm_mhz']}MHz{', ' + ','.join(d['tags']) if d['tags'] else ''})")
        print(f"      mix: {mix}")
    return 1


def cmd_refit(a) -> int:
    """The refit a beaten row needs. A row over MMA and one other class (R1, KFFMA0, KALU*, EIMAD*, …) gets the upper hull
    of every measured point (x_MMA, x_X) and the old row's MMA-axis end, at the margin (`eps1.facets`, the rule VF3m-16
    and VF3n-16 refit by); any other row gets its coefficients scaled so the worst program loads 1 - margin."""
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    progs = [program(row, c, m.cycle_u, a.generic, a.l2) for row in load_rows(a.rows, a.full)]
    hits = [p for p in progs if a.row in p["load"]]
    if not hits:
        print(f"no program loads {a.row}", file=sys.stderr)
        return 1
    hit = max(hits, key=lambda p: p["load"][a.row])
    worst = hit["load"][a.row]
    print(f"{a.row}: worst load {worst:.4f} by {hit['probe']} @ {hit['run']} (mix {hit['x']})")
    if a.row == L2_ROW:
        print(f"refit: l2 {a.l2:.4g} -> {hit['l2_bps'] / (1 - a.margin):.4g} B/s (the worst program's "
              f"{hit['l2_bps']:.4g} at {1 - a.margin:.2f})")
        return 0
    support = sorted(k for k in c if a.row in c[k])
    others = [k for k in support if k != "MMA" and k not in MMA_FORMS]
    form = a.form or max((k for k in hit["x"] if k in MMA_FORMS or k == "MMA"), key=lambda k: hit["x"][k], default=None)
    if form is None and len(others) >= 2:
        # a row over loads or lanes alone (LDG: LDG32 and LDG128): the worst program's heaviest class is the first axis
        form = max((k for k in hit["x"] if k in others), key=lambda k: hit["x"][k] * c[k][a.row])
        others = [k for k in others if k != form]
    if form in support:
        cf, mma_ok, fname = c[form][a.row], {form}, form
    elif form in (MMA_MEASURED_WITH, "MMA", None) and "MMA" in support:
        cf, mma_ok, fname = c["MMA"][a.row], {MMA_MEASURED_WITH, "MMA"} | (MMA_FORMS if a.generic else set()), "MMA"
    else:
        cf = None
    if cf and others:
        # one tensor axis (the form) and one vector axis: each other class weighted by its coefficient over the row's
        # commonest one, so a row over MMA and one class (R1, KALU*) is that class, and the joint issue rows' vector instructions sum
        sys.path.insert(0, str(HERE.parent / "opcode_probe"))
        from eps1 import facets
        cref = max({c[k][a.row] for k in others}, key=lambda u: sum(c[k][a.row] == u for k in others))
        X = others[0] if len(others) == 1 else "V"
        pts = []
        on_row = set(support) | ({MMA_MEASURED_WITH} if "MMA" in support else set())
        for p in progs:
            # a program with a class the row prices that is on neither axis can't be placed in the section
            if any(v > 0 and k in on_row and k not in mma_ok and k not in others for k, v in p["x"].items()):
                continue
            mm = sum(v for k, v in p["x"].items() if k in mma_ok)
            vv = sum(c[k][a.row] / cref * v for k, v in p["x"].items() if k in others)
            if mm > 0 or vv > 0:
                pts.append((mm, vv, f"{p['probe']}@{p['run']}"))
        s = 1 - a.margin
        pts.append((s * m.cycle_u / cf, 0.0, f"{a.row}:{fname}-axis"))
        pts.append((0.0, s * m.cycle_u / cref, f"{a.row}:{X}-axis"))
        if X == "V":
            print(f"V: every vector class on {a.row} by its coefficient over {cref / m.cycle_u:.4f} SMSP-cycles "
                  f"({len(others)} classes)")
        print(f"refit: the upper hull of {len(pts)} points ({fname}, {X}) a SMSP-cycle, x (1 - {a.margin}):")
        for f in facets(pts, a.margin):
            load = max(f["MMA"] * p[0] + f["X"] * p[1] for p in pts)
            print(f"  {fname} {f['MMA']:.3f} + {X} {f['X']:.3f} SMSP-cycles  (tight at {' and '.join(f['tight'])}; "
                  f"max load {load:.3f})")
        return 0
    scale = (1 - a.margin) / worst
    print(f"refit: scale every {a.row} coefficient by {scale:.5f} (so the worst program loads {1 - a.margin:.2f}):")
    for k in support:
        print(f"  {k}: {c[k][a.row] / m.cycle_u:.4f} -> {c[k][a.row] / m.cycle_u * scale:.4f} SMSP-cycles")
    return 0


def polygon(lines: list[tuple[float, float]], s: float, label: str) -> list[tuple[float, float, str]]:
    """The vertices of {(m, x) >= 0 : a·m + b·x <= 1 for each line (a, b)} but the origin, scaled by s: the points a
    published family was fitted through (a published facet sits at s on its points), both axis ends included."""
    cand = [(1 / max(a for a, _ in lines), 0.0), (0.0, 1 / max(b for _, b in lines))]
    for (a1, b1), (a2, b2) in combinations(lines, 2):
        det = a1 * b2 - a2 * b1
        if abs(det) > 1e-12 and (m := (b2 - b1) / det) >= 0 and (x := (a1 - a2) / det) >= 0:
            cand.append((m, x))
    return [(s * m, s * x, f"{label}:vertex") for m, x in cand if all(a * m + b * x <= 1 + 1e-9 for a, b in lines)]


def _close(m, epoch: str, rows: list[str]) -> list[str]:
    """Each live census line on the given rows, to close at `epoch` (its `until`), by the file it is in."""
    from verity_machine_model.model import entries, holds
    return [f"close: census/{e.since or m.epochs[0]}.csv {e.kind},{e.row} until {epoch}"
            for e in entries(m) if e.row in rows and holds(m, e.since, e.until, epoch)]


def cmd_family(a) -> int:
    """A beaten family of joint rows refitted as one (the VF3m-16 rule, `eps1.py facets --prior`): the upper hull of the
    old rows' vertices and every measured program, at the margin. Over an axis class (`--axis`, default MMA: the generic
    lines, which IMMA16832 takes, and with `--all-forms` every form) and one class (`--kind`); or, with `--forms`, the
    joint issue rows: each MMA form its own
    axis beside the rows' vector classes summed by their coefficients (so `V` is a vector instruction), each form's hull
    taken alone, and every new row at a vector coefficient one of those hulls has, each form at the most it can carry
    there. `--emit EPOCH` prints the rows.csv and census/EPOCH.csv lines and the old lines to close."""
    m = model()
    epoch = a.epoch or m.current
    c = census(m, epoch)
    cu, s = m.cycle_u, 1 - a.margin
    progs = [program(row, c, cu, a.generic, a.l2) for row in load_rows(a.rows, a.full)]
    rows = a.family
    on = {k for k in c if any(r in c[k] for r in rows)}
    sys.path.insert(0, str(HERE.parent / "opcode_probe"))
    from eps1 import facets
    runs = a.runs or " ".join(sorted({p["run"] for p in progs if p["run"]}))
    if not a.forms:
        axis = ({"MMA", MMA_MEASURED_WITH} | (MMA_FORMS if a.all_forms else set())) if a.axis == "MMA" else {a.axis}
        lines = [(c.get(a.axis, {}).get(r, 0.0) / cu, c.get(a.kind, {}).get(r, 0.0) / cu) for r in rows]
        pts = polygon(lines, s, "+".join(rows))
        n = 0
        for p in progs:
            if any(v > 0 and k in on and k not in axis and k != a.kind for k, v in p["x"].items()):
                continue
            mm, xx = sum(v for k, v in p["x"].items() if k in axis), p["x"].get(a.kind, 0.0)
            if mm > 0 or xx > 0:
                pts.append((mm, xx, f"{p['probe']}@{p['run']}"))
                n += 1
        fs = facets(pts, a.margin)
        print(f"{'+'.join(rows)}: {n} programs and {len(pts) - n} vertices; the refit ({a.axis}, {a.kind}) "
              f"SMSP-cycles:")
        for i, f in enumerate(fs):
            tight = " and ".join(t.split("@")[0] for t in f["tight"])
            print(f"  {a.name}{i}: {a.axis} {f['MMA']:.3f} + {a.kind} {f['X']:.3f} (tight at {tight})")
            if a.emit:
                desc = (f"{a.axis} {f['MMA']:.3f} + {a.kind} {f['X']:.3f} SMSP-cycles (FACET-BREAK: the upper hull of "
                        f"{'+'.join(rows)}'s vertices and {n} measured programs, 2% noise margin; tight at {tight})")
                print(f'rows.csv: {a.name}{i},joint,{runs},"{desc}"')
                for k, v in ((a.axis, f["MMA"]), (a.kind, f["X"])):
                    if v > 0:
                        print(f"census/{a.emit}.csv: {k},{a.name}{i},,,{v:.9g},,{a.emit},,{a.cite or runs}")
        if a.emit:
            print("\n".join(_close(m, epoch, rows)))
        return 0
    forms = sorted(k for k in on if k in MMA_FORMS)
    others = sorted(k for k in on if k not in MMA_FORMS and k != "MMA")
    cref = {r: max({c[k][r] for k in others if r in c[k]}, key=lambda u: sum(c[k].get(r) == u for k in others))
            for r in rows}
    w = {k: c[k][rows[0]] / cref[rows[0]] for k in others}
    if any(abs(c[k].get(r, 0.0) / cref[r] - w[k]) > 1e-6 for k in others for r in rows):
        print("--forms: a vector class's share of the row's coefficient differs between the rows", file=sys.stderr)
        return 1

    def V(p: dict) -> float:
        return sum(w[k] * v for k, v in p["x"].items() if k in w)

    hulls, vs = {}, []
    for f in forms:
        pts = polygon([(c[f][r] / cu, cref[r] / cu) for r in rows], s, f"{f}")
        n = 0
        for p in progs:
            if any(v > 0 and k in MMA_FORMS | {"MMA"} and k != f for k, v in p["x"].items()):
                continue
            mm, vv = p["x"].get(f, 0.0), V(p)
            if mm > 0 or vv > 0:
                pts.append((mm, vv, f"{p['probe']}@{p['run']}"))
                n += 1
        hulls[f] = pts
        for g in facets(pts, a.margin):
            if g["MMA"] > 0 and g["X"] > 0:
                vs.append((g["X"], f, " and ".join(t.split("@")[0] for t in g["tight"]), n))
    vs.sort(key=lambda t: -t[0])
    merged = []
    for t in vs:
        if not merged or merged[-1][0][0] - t[0] > 0.002 * vs[0][0]:
            merged.append([t])
        else:
            merged[-1].append(t)

    def carry(f: str, v: float) -> float:
        return min((s - v * p[1]) / p[0] for p in hulls[f] if p[0] > 0)

    old = {r: ({f: c[f][r] / cu for f in forms}, cref[r] / cu) for r in rows}
    print(f"{'+'.join(rows)}: {len(forms)} forms, V = {len(others)} vector classes (UNI at "
          f"{w.get('UNI', 0):.4f}); each row's vector coefficient and its forms' SMSP-cycles:")
    i, close = 0, []
    for grp in merged:
        v = grp[0][0]
        mf = {f: carry(f, v) for f in forms}
        same = next((r for r, (of, ov) in old.items() if abs(ov - v) <= 0.002 * ov and
                     all(abs(mf[f] - of[f]) <= 0.002 * of[f] for f in forms)), None)
        why = "; ".join(f"{f} tight at {t}" for _, f, t, _ in grp if t.count(":vertex") < 2) or "the old rows' vertices"
        if same:
            print(f"  keep {same}: V {v:.4f} ({why})")
            continue
        name = f"{a.name}{i}"
        i += 1
        print(f"  {name}: V {v:.4f}, MMA {min(mf.values()):.3f}; " +
              ", ".join(f"{f} {mf[f]:.3f}" for f in forms) + f" ({why})")
        if a.emit:
            desc = (f"MMA {min(mf.values()):.3f} + every other vector-pipe warp instruction {v:.3f} SMSP-cycles "
                    f"(FACET-BREAK: each MMA form's upper hull of {'+'.join(rows)}'s vertices and the measured "
                    f"programs, 2% noise margin, each form at the most it carries beside V {v:.3f}; MMA is every "
                    f"form's least, each form its own kind; {why})")
            print(f'rows.csv: {name},joint,{runs},"{desc}"')
            for f in forms:
                print(f"census/{a.emit}.csv: {f},{name},,,{mf[f]:.9g},,{a.emit},,{a.cite or runs}")
            print(f"census/{a.emit}.csv: MMA,{name},,,{min(mf.values()):.9g},,{a.emit},,{a.cite or runs}")
            for k in others:
                print(f"census/{a.emit}.csv: {k},{name},,,{w[k] * v:.9g},,{a.emit},,{a.cite or runs}")
    for r in rows:
        of, ov = old[r]
        if any(p[0] * of[f] + p[1] * ov > 1 + 1e-9 for f in forms for p in hulls[f] if ":vertex" not in p[2]):
            close.append(r)
    print(f"beaten (to close): {' '.join(close) or 'none'}")
    if a.emit:
        print("\n".join(_close(m, epoch, close)))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("loads", "summary", "refit", "table", "family"):
        s = sub.add_parser(name)
        s.add_argument("rows", type=Path, nargs="+")
        s.add_argument("--epoch")
        s.add_argument("--generic", action="store_true",
                       help="load every MMA form through MMA's lines where it has none (advisory, not the model)")
        s.add_argument("--full", action="store_true",
                       help="count every instruction of a single-loop program's timed loop (from its run's gate "
                            "verdicts and sass/), not only the gate's pinned opcodes")
        s.add_argument("--l2", type=float, default=l2_default(),
                       help="the L2 facet, read bytes a second (default: machine_model/data/table.toml's l2; none: "
                            "no L2tbl row)")
    sub.choices["loads"].add_argument("--probe")
    sub.choices["loads"].add_argument("--top", type=int, default=4)
    sub.choices["summary"].add_argument("--margin", type=float, default=0.0,
                                        help="report load > 1 + margin (default 0: the rows already carry the 2%%)")
    sub.choices["summary"].add_argument("--json", action="store_true")
    sub.choices["summary"].add_argument("--worst", action="store_true", help="only the worst program of each row")
    sub.choices["refit"].add_argument("--row", required=True)
    sub.choices["refit"].add_argument("--margin", type=float, default=0.02)
    sub.choices["refit"].add_argument("--form", help="the MMA form on the tensor axis (default: the worst program's)")
    fam = sub.choices["family"]
    fam.add_argument("--family", nargs="+", required=True, help="the rows refitted as one (R1 KFFMA0; KALU0 ... KALU3)")
    fam.add_argument("--name", required=True, help="the new rows' prefix (EFFMA: EFFMA0, EFFMA1, ...)")
    fam.add_argument("--axis", default="MMA")
    fam.add_argument("--kind")
    fam.add_argument("--forms", action="store_true", help="the joint issue rows: each MMA form its own axis")
    fam.add_argument("--all-forms", action="store_true",
                     help="every MMA form on the generic MMA axis (ε prices a form without lines of its own there)")
    fam.add_argument("--margin", type=float, default=0.02)
    fam.add_argument("--emit", metavar="EPOCH", help="print the rows.csv and census lines for EPOCH")
    fam.add_argument("--runs", help="the new rows' runs in rows.csv (default: every run read)")
    fam.add_argument("--cite", help="the census lines' runs (default --runs; @NAME cites runs.toml)")
    a = p.parse_args(argv)
    if a.cmd == "family" and not a.forms and not a.kind:
        p.error("family: --kind, or --forms")
    sys.path.insert(0, str(HERE))
    return {"loads": cmd_loads, "summary": cmd_summary, "refit": cmd_refit, "table": cmd_table,
            "family": cmd_family}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
