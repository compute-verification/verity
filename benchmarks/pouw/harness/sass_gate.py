"""The SASS gate: no shipped kernel computes a value the verifier's scalar model can't replay. Before anything is timed,
`bench.py` passes it every binary an arm declares (its build record's ``binaries``) and every library with device code
mapped into the run; NVIDIA's own libraries (cuBLASLt, the runtime, the driver) are listed as not scanned.

Two kinds of SASS instruction are flagged: any with .FTZ among its suffixes (FADD, FFMA, FMUL, and also FSETP, FMNMX, F2I,
F2F, HADD2 and every other), which -ftz=true and --use_fast_math emit and which flush subnormals the model keeps; and any FP32
MUFU (RCP, RSQ, SQRT, EX2, LG2, SIN, COS, TANH), an approximation that --use_fast_math, -prec-div=false and -prec-sqrt=false
emit bare, without .FTZ. FP64's MUFU.RCP64H and MUFU.RSQ64H are counted, not flagged.

ptxas builds div.rn, sqrt.rn, rcp.rn and integer division from flagged instructions in every build, so a flagged instruction
passes only inside one of these sequences, and only in a cubin whose .note.nv.tkinfo names a ptxas pinned in `sass_pins.json`
with its arguments (-v aside) and a numeric run; any other ptxas, other arguments, no note (12.9's sm_90a cubins have none)
or no numeric run, and nothing is exempt:
- a subroutine whose body has one of its toolkit's pinned signatures (each basic block's instruction shapes, registers,
  moves between them and the way an immediate is loaded aside): ``__cuda_sm20_rcp_rn_f32_slowpath``,
  ``__cuda_sm20_sqrt_rn_f32_slowpath``, ``__cuda_sm3x_div_rn_noftz_f32_slowpath``, ``__cuda_sm20_div_u16`` (two at 13.0:
  ptxas specializes it to its call sites), ``__cuda_sm20_div_u64``, ``__cuda_sm20_rem_s64``. ptxas folds a constant
  numerator into a function's copy of the div slow path when it is the function's only division: a body that loads it from
  a kernel parameter's slot is pinned like any other (the call passes that slot), one with an immediate isn't;
- rcp.rn's and sqrt.rn's fast path around a call into a pinned body (`FAST`): the range check, the branch, the call and the
  four-instruction fast block;
- div.rn's fast path (`DIV`): MUFU.RCP b, its five FFMAs and FCHK, and the branch around a call into a pinned body; a
  constant numerator is an immediate in the FFMAs, and FCHK's numerator register must hold the FFMAs' numerator (`value`:
  the same register unchanged, a copy, the same immediate or a load of the same slot of constant bank 0);
- integer division's seed (`match_int_seed`): I2F.RP d, MUFU.RCP, +2 ulp (IADD3 0xffffffe, or 0x1ffffffe at 64 bits), then
  F2I(.FTZ).U32/U64.TRUNC, each step the one reader of the last that only its value reaches; the F2I's input is a normal
  float, so .FTZ can't change it.
ptxas for sm_90a writes the range checks' and the seed's add of an immediate as VIADD R, R, imm or IADD3 R, R, imm, RZ
where sm_120a's writes IADD3 R, PT, PT, R, imm, RZ, and a constant numerator's move from a uniform register as IMAD.U32
R, RZ, RZ, UR: each sequence takes every spelling of the same 32-bit integer operation, and nothing else.
Each fast path's call passes the slow path exactly the fast path's operands (`_passes`): the registers the body may read
before writing them and the constant-bank slots it loads hold, as `value` keys them, x (rcp, sqrt) or a and b (div) each
in its role (the div body's divisor is the one input its MUFU.RCP's operand may derive from, `_sources`), with the return
address besides, so a call block may move them into place or reload them from their slot (ULDC, IMAD.U32).
A BRX branches to its ``BRANCH_TARGETS`` (the cubin's EIATTR_INDIRECT_BRANCH_TARGETS, as nvdisasm prints them); any other
indirect branch is a path the gate can't follow.
rsqrt has no entry: at default flags ptxas builds __frsqrt_rn from bare MUFU.RSQ, so it's refused.
Every operand of a matched sequence is the one definition that reaches it on every path (a sequence may straddle blocks, as
when ptxas hoists a loop's next range check and seed above the last iteration's tail), inputs unchanged between their
readers on every path, and every value the sequence computes other than its result is read only by the sequence until
overwritten, on every path through branches (whose predicates it follows), calls (into the subroutine's body) and returns
(to every call site) (`readers`). A read outside the sequence that another write of the register also reaches passes as a
merge the gate can't rule out, such as a read on a path it can't show infeasible, and is listed (``merged_reads``). The
numeric evidence that each sequence is exact on subnormal inputs and outputs, as each pinned ptxas builds it, is
`ieee_pin.cu`, run by `ieee_pin.py`, whose run each toolkit's ``numeric`` names; its builds are where the pins come from.

A flagged instruction also passes, alone and in any function, when its toolkit's entry lists it under ``exceptions``: an
instruction of `EXCEPTABLE`, in that one form, which a Definition the verifier replays models bit for bit, named in the
entry (``definition``) with the numeric run that held the toolkit's target card to it on every operand word (``numeric``,
its ``runs``). MUFU.EX2 is one under 13.0.1-sm_90a: `verity.ml.circuits.ops.mufu.MufuEx2Ftz` on all 2^32 operands on two
H200s. An entry without both, or an instruction `EXCEPTABLE` doesn't list, fails the pins.

The cubins come from a .cubin, a fatbin, or a host ELF's .nv_fatbin section (parsed here, since ``cuobjdump -xelf`` writes
same-named cubins over each other), each disassembled by nvdisasm; ``cuobjdump -sass`` of the whole binary must list the same
flagged instructions, or the gate fails. A compressed fatbin entry fails too (build with nvcc's ``--no-compress``, or ship the
cubin), as does a PTX entry with an .ftz or .approx instruction or div.full. Passes are cached by the binary's sha256, this
file's, the pins' and the tools' versions (``POUW_SASS_GATE_CACHE``, default ``~/.cache/pouw-harness/sass-gate``; ``none``
disables).

The file also holds the build gate every kernel family's build.sh runs on ptxas's report and its cubin's SASS (`mma_gate`):
the kernels the family wants, compiled for its arch with no ptxas warning, serialization, spill or local memory, and no
tensor-core form but the ones it allows. Each family's own sass_gate.py keeps its WANT table, its SASS pins and its CLI.

  python sass_gate.py BINARY... [--json OUT]     # exit 0: every binary passes; 7: one fails (named, with its instructions)
  python sass_gate.py --pin CUBIN...             # each cubin's tool notes and subroutine signatures
  python sass_gate.py --pin CUBIN... --toolkit NAME   # pin them in sass_pins.json as NAME (its numeric run kept while the
                                                      # subroutines are unchanged)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

VERSION = "sass-gate/v4"
EXIT_FAIL = 7
PINS = Path(__file__).with_name("sass_pins.json")
NVIDIA_LIB = re.compile(r"lib(cublas|cublasLt|cudart|cuda|nvrtc|nvrtc-builtins|nvJitLink|cufft|cufftw|curand|cusparse|cusolver"
                        r"|cudnn\w*|nccl|cupti|nvToolsExt|nvperf_\w+|nvidia-[\w-]+)\.so(\.\d+)*")
#: how /proc/self/maps names shared anonymous memory (mmap MAP_SHARED|MAP_ANONYMOUS, which the CUDA driver maps) and System V
#: shared memory: no file is behind either, so neither is a library; any other deleted mapping but a POSIX semaphore (below)
#: fails the gate
ANON_SHARED = re.compile(r"/dev/zero \(deleted\)|/SYSV[0-9a-f]{8} \(deleted\)")
#: a POSIX named semaphore (sem_open; multiprocessing unlinks it at once): one page holding a sem_t, skipped only when every
#: mapping of it is one page at most and holds neither an ELF nor a fatbin
POSIX_SEM = re.compile(r"/dev/shm/sem\.[^/]+ \(deleted\)")
PAGE = 4096
PTX_BAD = re.compile(r"\.(ftz|approx)\b|\bdiv\.full\b")
FATBIN_MAGIC = 0xBA55ED50
FATBIN_COMPRESSED = 0x2000
EM_CUDA = 190
SHT_NOTE = 7
MAX_LISTED = 20
MAX_STATES = 1 << 20
FP64_MUFU = frozenset({"RCP64H", "RSQ64H"})
#: the flagged instructions a toolkit's ``exceptions`` may list, each in the one form (guard aside) its numeric run measured:
#: a register operand, no modifier
EXCEPTABLE = {"MUFU.EX2": re.compile(r"MUFU\.EX2 R\d+, R\d+")}

_INSN = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
_PRED = re.compile(r"^@!?U?P[T0-9]+\s+")
_GUARD = re.compile(r"^@(!?)(U?P[T0-6])\s+")
_REG = re.compile(r"(?<![\w.$])(UR\d+|R\d+|UP[0-6]|P[0-6]|PR)(\.64)?(?!\w)")
_TARGET = re.compile(r"`\(([^)\s]+)\)")
# nvdisasm's rendering of a BRX's EIATTR_INDIRECT_BRANCH_TARGETS, which ptxas writes into the cubin as it does the tkinfo note
_BRANCH_TARGETS = re.compile(r'\(\*"BRANCH_TARGETS ([^"]+)"\*\)')
_SUBROUTINE = re.compile(r"\$__internal_\d+_\$(\w+)")


class SassGateFailed(RuntimeError):
    pass


def opcode(text: str) -> list[str]:
    """``@!P0 FFMA.RZ.FTZ R0, ...`` -> ``["FFMA", "RZ", "FTZ"]``."""
    t = _PRED.sub("", text.lstrip("{ ").strip())
    return t.split(None, 1)[0].split(".") if t else [""]


def flagged(op: list[str]) -> bool:
    """An instruction the scalar model can't replay unless a pinned sequence covers it: .FTZ, or an FP32 MUFU."""
    return "FTZ" in op[1:] or (op[0] == "MUFU" and (len(op) < 2 or op[1] not in FP64_MUFU))


# ---- instructions: operands, reads and writes ---------------------------------------------------------------------------
# A kill (an overwrite) may be missed and a read invented, never the reverse: either only makes `readers` stricter, and
# `reach` stops at anything that may write the register. So an opcode's first operand is its destination only for the
# opcodes listed, and an unlisted opcode may write every register it names.

DEST = frozenset("""B2R BMSK BREV CS2R CS2UR DADD DFMA DMMA DMNMX DMUL F2F F2FP F2I FADD FCHK FFMA FLO FMNMX FMUL FRND FSEL
FSWZADD HADD2 HFMA2 HGMMA HMMA HMNMX2 HMUL2 I2F I2FP I2I IABS IADD IADD3 IDP IMAD IMMA IMNMX LD LDC LDCU LDG LDL LDS LDSM LEA
LOP3 MATCH MOV MOV32I MUFU OMMA P2R POPC PRMT QGMMA QMMA R2UR REDUX S2R S2UR SEL SGXT SHF SYNCS UF2I UFLO UI2F UI2FP UIADD3
UIMAD ULDC ULEA ULOP3 UMOV UPRMT USEL USETMAXREG USHF UVIMNMX VIADD VIADDMNMX VIMNMX VIMNMX3 VOTE VOTEU""".split())
PRED_PAIR = frozenset("SHFL ATOM ATOMG ATOMS LOP3 ULOP3 PLOP3 UPLOP3 ISETP UISETP FSETP DSETP HSETP2 ELECT VOTE".split())
NODEST = frozenset("""BRA BREAK BRX BSSY BSYNC CALL CCTL CGAERRBAR DEPBAR ENDCOLLECTIVE ERRBAR EXIT FENCE JMP JMX LDGDEPBAR
LDGSTS MEMBAR NOP RED REDG RET ST STG STL STS STSM UBLKCP UTMACCTL UTMALDG UTMAPF UTMASTG WARPGROUP WARPSYNC YIELD""".split())
INDIRECT = frozenset({"BRX", "JMX", "JMP"})
# Memory operations whose bracketed operands are addresses: 32-bit unless written .64, and a desc[UR] pair.
MEMORY = frozenset("LD LDG LDS LDL LDC LDCU ST STG STS STL LDGSTS ATOM ATOMG ATOMS RED REDG LDSM STSM".split())
_PREDS = tuple(f"P{i}" for i in range(7))


def _widths(op: list[str]) -> tuple[int, int]:
    """(the registers a destination spans, at least; the registers a source may span, at most)."""
    o, sfx = op[0], op[1:]
    if o.endswith("MMA"):
        # A warp MMA fragment spans at most 4 registers (16 for FP64); a warpgroup accumulator up to 128.
        d = next((s for s in sfx if s in ("F32", "S32", "F16", "BF16")), None)
        shaped = not o.endswith("GMMA") and any(s.isdigit() and len(s) >= 4 for s in sfx)
        return (4 if d in ("F32", "S32") else 2) if shaped and d else 1, 128 if o.endswith("GMMA") else 16 if o == "DMMA" else 4
    if o.startswith(("UTMA", "UBLK", "UTC", "SYNCS")):
        return 1, 8
    if "128" in sfx:
        return 4, 4
    if o in ("LDSM", "STSM", "MOVM"):
        n = int(sfx[-1]) if sfx and sfx[-1] in ("1", "2", "4") else 1
        return n, 4
    if o in ("F2F", "I2F", "UI2F", "F2I", "UF2I", "I2I", "FRND", "I2FP", "F2FP"):
        wide = {"F64", "S64", "U64"} & set(sfx)
        if o == "F2F":
            return (2 if sfx and sfx[0] == "F64" else 1), (2 if wide else 1)
        if o in ("I2F", "UI2F", "FRND"):
            return (2 if "F64" in sfx else 1), (2 if wide else 1)
        if o in ("F2I", "UF2I"):
            return (2 if {"S64", "U64"} & set(sfx) else 1), (2 if wide else 1)
        return 1, (2 if wide else 1)
    if "64" in sfx or "WIDE" in sfx or o in ("DADD", "DFMA", "DMUL", "DMNMX", "DSETP", "CS2UR", "MOV64") \
            or (o == "CS2R" and "32" not in sfx):
        return 2, 2
    return 1, 1


def _span(tok: str, explicit64: bool, width: int) -> list[str]:
    if tok == "PR":
        return list(_PREDS)
    if tok[0] == "P" or tok.startswith("UP"):
        return [tok]
    kind, n = ("UR", int(tok[2:])) if tok.startswith("UR") else ("R", int(tok[1:]))
    return [f"{kind}{n + k}" for k in range(max(width, 2 if explicit64 else 1))]


def _regs(arg: str, width: int, brackets: bool | None = None, address: bool = False) -> set[str]:
    """The registers `arg` names (inside [...] only, outside only, or both), each spanning `width`; with `address`, one
    inside brackets spans one register (two when written .64 or in desc[...])."""
    out: set[str] = set()
    for m in _REG.finditer(arg):
        before = arg[:m.start()]
        inside = before.count("[") > before.count("]")
        if brackets is None or inside == brackets:
            w = (2 if before[:before.rfind("[")].endswith("desc") else 1) if inside and address else width
            out.update(_span(m.group(1), bool(m.group(2)), w))
    return out


def _split(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        depth += (ch in "[({") - (ch in "])}")
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    return out + [tail] if tail or out else out


class Insn:
    """One instruction: its guard (``!P0``), opcode, operands, the registers it may read, surely writes when it executes
    (``writes``; ``kills`` is the same, empty when guarded) and may write, its branch targets (label names: a BRX's are its
    ``BRANCH_TARGETS``), and whether it branches where the gate can't follow (``indirect``: JMX, JMP, a BRX without them)."""

    __slots__ = ("addr", "text", "guard", "op", "args", "reads", "writes", "kills", "maydef", "targets", "indirect")

    def __init__(self, addr: int, text: str):
        self.addr, self.text = addr, text
        t = text.lstrip("{ ").strip().replace(".reuse", "")
        g = _GUARD.match(t)
        self.guard = (g.group(1) + g.group(2)) if g else None
        t = t[g.end():] if g else t
        head, _, rest = t.partition(" ")
        self.op = head.split(".")
        self.args = _split(rest)
        self.targets = _TARGET.findall(rest)
        bt = _BRANCH_TARGETS.search(rest) if self.op[0] == "BRX" else None
        if bt:
            self.targets = [x.strip() for x in bt.group(1).split(",")]
            self.args = _split(rest[:bt.start()])
        self.indirect = self.op[0] in INDIRECT and not bt
        o, (dw, rw) = self.op[0], _widths(self.op)
        a, mem = self.args, o in MEMORY
        reads = {g.group(2)} - {"PT", "UPT"} if g else set()
        kills: set[str] = set()
        if o in DEST or o in PRED_PAIR:
            first = 2 if o in PRED_PAIR and len(a) > 1 and re.fullmatch(r"!?U?P[T0-6]", a[0]) else 1
            outs = a[:first]
            for arg in outs:
                kills |= _regs(arg, dw, brackets=False)
                reads |= _regs(arg, rw, brackets=True, address=mem)
            srcs = a[first:]
            if o in ("IMAD", "UIMAD") and "WIDE" in self.op and len(srcs) >= 3:
                reads |= _regs(srcs[0], 1) | _regs(srcs[1], 1)
                srcs = srcs[2:]
            elif o.endswith("MMA") and not o.endswith("GMMA"):
                ws = (16, 8, 8) if o == "DMMA" else (4, 2, 4)
                for arg, w in zip(srcs, ws):
                    reads |= _regs(arg, w)
                srcs = srcs[len(ws):]
            elif self.op == ["P2R"] and len(srcs) == 3 and srcs[0] == "PR" and re.fullmatch(r"0x[0-9a-f]+", srcs[2]):
                # P2R R, PR, R', mask reads the predicates whose bits the mask sets
                reads |= {p for i, p in enumerate(_PREDS) if int(srcs[2], 16) >> i & 1}
                srcs = srcs[1:]
            for arg in srcs:
                reads |= _regs(arg, rw, address=mem)
            # F2F.<to>.<from> names its destination's type, so that type's width is all it writes (sm_90a's ptxas
            # schedules an F2F.F32.F64 inside a div.rn fast path that reads the register after its destination)
            types = [s for s in self.op[1:] if s in ("F16", "BF16", "F32", "F64")]
            ow = (2 if types[0] == "F64" else 1) if o == "F2F" and len(types) == 2 else rw
            maydef = set().union(*(_regs(arg, ow, brackets=False) for arg in outs)) if outs else set()
            maydef |= {r for arg in a[1:3] for r in _regs(arg, 1, brackets=False) if r in _PREDS or r.startswith("UP")}
        elif o in NODEST:
            for arg in a:
                reads |= _regs(arg, rw, address=mem)
            maydef = set()
        else:
            for arg in a:
                reads |= _regs(arg, rw, address=mem)
            maydef = set().union(*(_regs(arg, rw, brackets=False) for arg in a)) if a else set()
        self.reads = frozenset(reads)
        self.writes = frozenset(kills)
        self.kills = frozenset() if self.guard else self.writes
        self.maydef = frozenset(maydef | kills)

    @property
    def name(self) -> str:
        return ".".join(self.op)

    def body(self) -> str:
        """The instruction without its guard or reuse flags."""
        return f"{self.name} {', '.join(self.args)}".strip()

    def unconditional(self) -> bool:
        return self.guard is None and len(self.args) == 1


class Function:
    """One .text section of `parse_nvdisasm`: its instructions, labels, ``.size`` ends and subroutine regions."""

    def __init__(self, fn: dict):
        self.name = fn["name"]
        self.insns = [Insn(a, t) for a, t in fn["insns"]]
        self.labels: dict[str, int] = fn["labels"]
        self.label_at = frozenset(self.labels.values())
        self.subroutines: dict[str, tuple[str, int, int]] = {}
        for lab, i in self.labels.items():
            m = _SUBROUTINE.fullmatch(lab)
            end = fn["sizes"].get(lab)
            if m and end in self.labels and self.labels[end] > i:
                self.subroutines[lab] = (m.group(1), i, self.labels[end])
        self.sub_starts = frozenset(s for _, s, _ in self.subroutines.values())
        self._effects: dict[tuple[str, str], tuple[bool, bool] | None] = {}
        self._preds: dict[int, list[int]] | None = None
        self._preds_built = False
        self._guards: dict[str, set[str]] = {}
        for ins in self.insns:
            if ins.guard:
                for r in ins.reads | ins.maydef:
                    self._guards.setdefault(r, set()).add(ins.guard.lstrip("!"))
        self.entries = Counter(self.labels.get(t) for i in self.insns if i.op[0] in ("BRA", "BRX") for t in i.targets)

    def target(self, k: int) -> int | None:
        t = self.insns[k].targets
        return self.labels.get(t[0]) if t else None

    def succ(self, k: int) -> list[int] | None:
        """The instructions that may follow k; None when unknown (an indirect branch)."""
        ins = self.insns[k]
        o = ins.op[0]
        if ins.indirect or (o == "BRA" and not ins.targets):
            return None
        if o in ("BRA", "BRX"):
            t = [self.labels.get(x) for x in ins.targets]
            if None in t:
                return None
            return t if ins.unconditional() else t + [k + 1]
        if o in ("EXIT", "RET") and ins.guard is None:
            return []
        return [k + 1]

    def _pred_map(self) -> dict[int, list[int]] | None:
        """Each instruction's possible predecessors; None when a branch is indirect."""
        if not self._preds_built:
            self._preds_built = True
            preds: dict[int, list[int]] = {}
            for k in range(len(self.insns)):
                s = self.succ(k)
                if s is None:
                    return None
                for t in s:
                    preds.setdefault(t, []).append(k)
            self._preds = preds
        return self._preds

    def before(self, i: int) -> list[int] | None:
        """The instructions that may run just before i: [] at the function's or a subroutine's entry; None when unknown (a
        label, in a function with an indirect branch)."""
        if i in self.sub_starts:
            return []
        preds = self._pred_map()
        if preds is not None:
            return preds.get(i, [])
        if i in self.label_at or i == 0:
            return None
        s = self.succ(i - 1)
        return [i - 1] if s is not None and i in s else []

    def reach(self, k: int, reg: str) -> int | None:
        """The instruction whose result `reg` holds at k on every path to k: None when paths differ, or one meets a call that
        may change it (`preserves`), an entry, a guarded or partial write, or a label it can't see behind (a function with an
        indirect branch)."""
        found, seen, stack = None, set(), [k]
        while stack:
            ps = self.before(stack.pop())
            if not ps:
                return None
            for j in ps:
                if j in seen:
                    continue
                seen.add(j)
                p = self.insns[j]
                if (p.op[0] == "CALL" and not self.preserves(j, reg)) or p.indirect:
                    return None
                if reg in p.maydef:
                    if reg not in p.kills or found not in (None, j):
                        return None
                    found = j
                else:
                    stack.append(j)
        return found

    def steady(self, reg: str, i: int, j: int) -> bool:
        """`reg` holds the same value when j reads it as when i does: every path to j passes i, and nothing from i on may
        write `reg` (a call, unless it `preserves` it)."""
        seen, stack = set(), [j] if i != j else []
        while stack:
            ps = self.before(stack.pop())
            if not ps:
                return False
            for t in ps:
                if t in seen:
                    continue
                seen.add(t)
                p = self.insns[t]
                if reg in p.maydef or (p.op[0] == "CALL" and not self.preserves(t, reg)) or p.indirect:
                    return False
                if t != i:
                    stack.append(t)
        return True

    def holds(self, reg: str, d: int, j: int) -> bool:
        """`reg` still holds, when j reads it, the value d wrote."""
        return self.reach(j, reg) == d

    def callee_effect(self, k: int, reg: str) -> tuple[bool, bool] | None:
        """For the subroutine CALL k enters: (whether it may read `reg` before writing it, whether it writes `reg` on every
        path to its return); None when k enters no subroutine of this section, or the subroutine may leave it other than by
        its return."""
        lab = self.insns[k].targets[0] if self.insns[k].targets else None
        if (lab, reg) not in self._effects:
            sub = self.subroutines.get(lab)
            t = None if sub is None else self._trace([sub[1]], reg, sub)
            self._effects[lab, reg] = None if t is None else (bool(t[0]), not t[1])
        return self._effects[lab, reg]

    def preserves(self, k: int, reg: str) -> bool:
        """The CALL at k returns with `reg` unchanged: it enters a subroutine of this section that may leave only by its
        return, and none of whose instructions may write `reg`."""
        sub = self.subroutines.get(self.insns[k].targets[0] if self.insns[k].targets else None)
        return sub is not None and self.callee_effect(k, reg) is not None \
            and not any(reg in self.insns[t].maydef for t in range(sub[1], sub[2]))

    def defined_elsewhere(self, u: int, reg: str, d: int) -> bool:
        """Whether a write of `reg` besides d reaches u, so that u reads a merge of d's value and another: an unguarded
        write, or a call into a subroutine that writes it on every path. False when a branch is indirect."""
        preds = self._pred_map() or {}
        seen: set[int] = set()
        stack = list(preds.get(u, ()))
        while stack:
            j = stack.pop()
            if j in seen or j == d:
                continue
            seen.add(j)
            ins = self.insns[j]
            if reg in ins.kills or (ins.op[0] == "CALL" and ins.guard is None and self.must_write(j, reg)):
                return True
            stack.extend(preds.get(j, ()))
        return False

    def return_sites(self, i: int) -> list[int]:
        """Where a return at i may continue: after every CALL into the subroutine that contains i ([] when none does)."""
        lab = next((lab for lab, (_, lo, hi) in self.subroutines.items() if lo <= i < hi), None)
        return [k + 1 for k, ins in enumerate(self.insns) if lab and ins.op[0] == "CALL" and ins.targets[:1] == [lab]]

    def must_write(self, k: int, reg: str) -> bool:
        eff = self.callee_effect(k, reg)
        return eff is not None and eff[1]

    def readers(self, k: int, reg: str) -> set[int] | None:
        """Every instruction that may read the value k writes to `reg`, on every path until it is overwritten; None when it
        may escape: a return, an indirect branch, or a call into a subroutine that may read `reg` before writing it."""
        first = self.succ(k)
        t = None if first is None else self._trace(first, reg)
        return None if t is None else t[0]

    def _trace(self, starts: list[int], reg: str, sub: tuple[str, int, int] | None = None) -> tuple[set[int], bool] | None:
        """Follow the value `reg` holds at each start forward until every path overwrites it: (the instructions that may
        read it, whether a path returns with it); None when a path may leave the region or branch indirectly, when a
        subroutine (`sub`) calls or exits, or when a function's path calls a subroutine that may read it or returns from
        one that no CALL in the section enters (a return continues after every CALL into its subroutine).

        A guarded write counts where its guard is known on the path (from a branch on it), for a later read under the same
        guard, and once both polarities of an unchanged predicate have written; only predicates guarding an instruction
        that touches `reg` are tracked."""
        lo, hi = (sub[1], sub[2]) if sub else (0, len(self.insns))
        rel = self._guards.get(reg, frozenset())
        out: set[int] = set()
        returns = False
        seen: set = set()
        stack = [(i, frozenset(), frozenset()) for i in starts]
        while stack:
            st = stack.pop()
            if st in seen:
                continue
            seen.add(st)
            if len(seen) > MAX_STATES:
                return None
            i, facts, pending = st
            if not lo <= i < hi:
                return None
            ins = self.insns[i]
            o, g = ins.op[0], ins.guard
            lit = None if g in (None, "PT", "UPT") else (g.lstrip("!"), not g.startswith("!"))
            if g in ("!PT", "!UPT") or (lit and (lit[0], not lit[1]) in facts):
                stack.append((i + 1, facts, pending))
                continue
            sure = lit is None or lit in facts
            if reg in ins.reads and not (lit and lit in pending):
                out.add(i)
            if ins.indirect or (sub and o in ("CALL", "EXIT")):
                return None
            if o == "CALL":
                eff = self.callee_effect(i, reg)
                if eff is None or eff[0]:
                    return None
                if not (eff[1] and sure):
                    stack.append((i + 1, facts, pending))
                continue
            if reg in ins.writes:
                if sure or (lit[0], not lit[1]) in pending:
                    continue
                pending = pending | {lit}
            gone = {p for p in ins.maydef if p in _PREDS or p.startswith("UP")}
            if gone:
                facts = frozenset(f for f in facts if f[0] not in gone)
                pending = frozenset(f for f in pending if f[0] not in gone)
            if o in ("BRA", "BRX", "RET", "EXIT"):
                if lit:
                    alts = [(lit, True)] + ([] if lit in facts else [((lit[0], not lit[1]), False)])
                else:
                    alts = [(None, True)] if o != "BRA" or ins.unconditional() else [(None, True), (None, False)]
                for fact, taken in alts:
                    f2 = facts | {fact} if fact and fact[0] in rel else facts
                    p2 = frozenset(p for p in pending if (p[0], not p[1]) not in f2)
                    if any(p in f2 for p in p2):
                        continue
                    if not taken:
                        stack.append((i + 1, f2, p2))
                    elif o == "RET":
                        if sub:
                            returns = True
                            continue
                        back = self.return_sites(i)
                        if not back:
                            return None
                        stack.extend((x, f2, p2) for x in back)
                    elif o in ("BRA", "BRX"):
                        t = [self.labels.get(x) for x in ins.targets]
                        if not t or None in t:
                            return None
                        stack.extend((x, f2, p2) for x in t)
                continue
            stack.append((i + 1, facts, pending))
        return out, returns


# ---- the pinned sequences -----------------------------------------------------------------------------------------------

_VAR = re.compile(r"\$(\w+)")
_FLOAT = r"-?\d+(?:\.\d+)?(?:e[+-]\d+)?"
# The $vars that may also be an FP32 immediate (a quotient's constant numerator).
IMMEDIATE = frozenset({"a"})


def unify(pattern: str, ins: Insn, env: dict) -> dict | None:
    """``FFMA $t, -$b, $r, 1`` against an unguarded instruction: each $var one register (or, in IMMEDIATE, an FP32
    immediate), consistent with `env`."""
    if ins.guard is not None:
        return None
    name, _, rest = pattern.partition(" ")
    pargs = [a.strip() for a in rest.split(",")] if rest else []
    if ins.name != name or len(pargs) != len(ins.args):
        return None
    env = dict(env)
    for p, a in zip(pargs, ins.args):
        rx, names, pos = "", [], 0
        for m in _VAR.finditer(p):
            rx += re.escape(p[pos:m.start()])
            v = m.group(1)
            imm = f"|{_FLOAT}" if v in IMMEDIATE else ""
            rx += re.escape(env[v]) if v in env else f"(?P<{v}>U?R\\d+|U?P[0-6]{imm})"
            names += [] if v in env else [v]
            pos = m.end()
        m = re.fullmatch(rx + re.escape(p[pos:]), a)
        if not m:
            return None
        env.update({v: m.group(v) for v in names})
    return env


def _defines(pattern: str) -> str | None:
    m = _VAR.match(pattern.partition(" ")[2])
    return m.group(1) if m else None


def _reads(pattern: str) -> list[str]:
    return _VAR.findall(", ".join(pattern.partition(" ")[2].split(",")[1:]))


def _first(fn: Function, pats: tuple, ins: Insn, env: dict) -> tuple[str, dict] | None:
    for p in pats:
        e = unify(p, ins, env)
        if e is not None:
            return p, e
    return None


def _block(fn: Function, start: int, block: list[tuple], env: dict) -> tuple[dict, dict[int, str]] | None:
    """Consecutive instructions from `start` against `block`, trying every alternative: (env, index -> pattern)."""
    def go(i: int, env: dict, pats: dict) -> tuple[dict, dict] | None:
        if i == len(block):
            return env, pats
        for p in block[i]:
            e = unify(p, fn.insns[start + i], env)
            if e is not None and (r := go(i + 1, e, {**pats, start + i: p})) is not None:
                return r
        return None
    return go(0, env, {})


# A call block's moves of the slow path's operands and return address into place (`_passes` checks what they pass): an
# immediate, a register, or a reload of a fixed slot of constant bank 0 (ULDC, then IMAD.U32 R, RZ, RZ, UR)
SETUP = re.compile(r"(MOV|MOV32I) U?R\d+, (0x[0-9a-f]+|-?\d+|U?R\d+)|IMAD\.MOV\.U32 R\d+, RZ, RZ, (0x[0-9a-f]+|-?\d+|R\d+)"
                   r"|HFMA2(\.MMA)? R\d+, -RZ, RZ, \S+, \S+|ULDC UR\d+, c\[0x0\]\[0x[0-9a-f]+\]|IMAD\.U32 R\d+, RZ, RZ, UR\d+")
COPY = re.compile(r"MOV R\d+, R\d+|IMAD\.MOV\.U32 R\d+, RZ, RZ, R\d+")

FAST = {
    "rcp": {"slowpath": "__cuda_sm20_rcp_rn_f32_slowpath", "negated": False,
            "check": [("ISETP.GT.U32.AND $P, PT, $m, 0x1ffffff, PT",), ("LOP3.LUT $m, $t, 0x7f800000, RZ, 0xc0, !PT",),
                      ("IADD3 $t, PT, PT, $x, 0x1800000, RZ", "IADD3 $t, $x, 0x1800000, RZ",
                       "VIADD $t, $x, 0x1800000")],
            "block": [("MUFU.RCP $y, $x",), ("FFMA $e, $x, $y, -1", "FFMA $e, $y, $x, -1"), ("FADD.FTZ $f, -$e, -RZ",),
                      ("FFMA $z, $y, $f, $y", "FFMA $z, $f, $y, $y")]},
    "sqrt": {"slowpath": "__cuda_sm20_sqrt_rn_f32_slowpath", "negated": True,
             "check": [("ISETP.GT.U32.AND $P, PT, $t, 0x727fffff, PT",),
                       ("IADD3 $t, PT, PT, $x, -0xd000000, RZ", "IADD3 $t, $x, -0xd000000, RZ",
                        "VIADD $t, $x, 0xf3000000")],
             "seed": ("MUFU.RSQ $r, $x",),
             "block": [("FMUL.FTZ $y, $r, $x", "FMUL.FTZ $y, $x, $r"), ("FMUL.FTZ $h, $r, 0.5",), ("FFMA $e, -$y, $y, $x",),
                       ("FFMA $z, $e, $h, $y", "FFMA $z, $h, $e, $y")]},
}
DIV = {"slowpath": "__cuda_sm3x_div_rn_noftz_f32_slowpath",
       "seed": ("MUFU.RCP $r, $b",),
       "chain": [("FFMA $t, -$b, $r, 1", "FFMA $t, $r, -$b, 1"), ("FFMA $u, $r, $t, $r", "FFMA $u, $t, $r, $r"),
                 ("FFMA $q, $a, $u, RZ", "FFMA $q, $u, $a, RZ"), ("FFMA $e, -$b, $q, $a", "FFMA $e, $q, -$b, $a"),
                 ("FFMA $z, $u, $e, $q", "FFMA $z, $e, $u, $q")],
       "check": ("FCHK $P, $n, $b",)}
INT_ROUND_UP = {"0xffffffe", "0x1ffffffe"}


class Match:
    """A matched sequence: its instructions, the values it computes besides its result (index, register), and the reads
    outside it of those values that another write also reaches (index, register, reader)."""

    def __init__(self, kind: str, members: set[int], inner: list[tuple[int, str]]):
        self.kind, self.members, self.inner = kind, members, inner
        self.merged: list[tuple[int, str, int]] = []


def _control(ins: Insn) -> bool:
    return ins.op[0] in ("BRA", "CALL", "EXIT", "RET") or ins.op[0] in INDIRECT


def _call_block(fn: Function, g: int, slowpath: str, bodies: frozenset[str]) -> tuple[list[int], int, list[int], int] | str:
    """From the branch g around a slow-path call into one of `bodies` (the subroutine labels whose bodies are their pins):
    (setup, the call, the copies after it, the index after them)."""
    k, setup = g + 1, []
    while k < len(fn.insns) and k not in fn.label_at and fn.insns[k].guard is None and SETUP.fullmatch(fn.insns[k].body()):
        setup.append(k)
        k += 1
    ins = fn.insns[k] if k < len(fn.insns) else None
    m = _SUBROUTINE.fullmatch(ins.targets[0]) if ins is not None and ins.targets else None
    if ins is None or k in fn.label_at or ins.guard or ins.name != "CALL.REL.NOINC" or not m or m.group(1) != slowpath:
        return f"no call to {slowpath} after the branch"
    if ins.targets[0] not in bodies:
        return f"the {slowpath} it calls isn't the pinned body"
    c, k, copies = k, k + 1, []
    while k < len(fn.insns) and k not in fn.label_at and fn.insns[k].guard is None and COPY.fullmatch(fn.insns[k].body()):
        copies.append(k)
        k += 1
    return setup, c, copies, k


def _returned(fn: Function, c: int, copies: list[int], z: str) -> str | None:
    """The register the slow path at c returns z in (the copy into z, or z itself), if its subroutine writes it on every
    path to its return; else None."""
    into = [t for t in copies if fn.insns[t].args[0] == z]
    if copies and not into:
        return None
    res = fn.insns[into[-1]].args[-1] if into else z
    return res if fn.must_write(c, res) else None


def _flows(fn: Function, defs: dict[str, int], env: dict, pats: dict[int, str]) -> str | None:
    """Every $var a matched instruction reads is the matched definition's value (or, for an input, the same value at every
    reader): None, or what breaks."""
    inputs: dict[str, list[int]] = {}
    for k, p in pats.items():
        for v in _reads(p):
            if v in defs:
                if fn.reach(k, env[v]) != defs[v]:
                    return f"${v} at 0x{fn.insns[k].addr:04x} isn't the value its match defines"
            else:
                inputs.setdefault(v, []).append(k)
    for v, ks in inputs.items():
        if not fn.steady(env[v], min(ks), max(ks)):
            return f"${v} changes between its readers"
    return None


def _escape(fn: Function, m: Match) -> str | None:
    """None when every value the sequence computes besides its result is read only by the sequence, or by a read another
    write of its register also reaches (a merge the gate can't rule out, such as an uninitialized read on a path it can't
    show infeasible: listed in `m.merged`); else what escapes."""
    for k, reg in m.inner:
        r = fn.readers(k, reg)
        if r is None:
            return f"{reg} from 0x{fn.insns[k].addr:04x} may escape (a return, a call or an indirect branch)"
        outside = sorted(r - m.members)
        direct = [u for u in outside if not fn.defined_elsewhere(u, reg, k)]
        if direct:
            return f"{reg} from 0x{fn.insns[k].addr:04x} is read outside the sequence at 0x{fn.insns[direct[0]].addr:04x}"
        m.merged += [(k, reg, u) for u in outside]
    return None


def match_fast(fn: Function, c: int, kind: str, bodies: frozenset[str]) -> Match | str:
    """rcp.rn's or sqrt.rn's fast path around the slow-path call at c (into one of `bodies`)."""
    spec = FAST[kind]
    g = c - 1
    while g >= 0 and g + 1 not in fn.label_at and fn.insns[g].guard is None and SETUP.fullmatch(fn.insns[g].body()):
        g -= 1
    br = fn.insns[g] if g >= 0 else None
    if br is None or g + 1 in fn.label_at or br.name != "BRA" or len(br.args) != 1 or br.guard is None \
            or br.guard.startswith("!") != spec["negated"]:
        return "no guarded branch before the call"
    cb = _call_block(fn, g, spec["slowpath"], bodies)
    if isinstance(cb, str) or cb[1] != c:
        return cb if isinstance(cb, str) else "the call isn't the branch's"
    setup, _, copies, b = cb
    jb = fn.insns[b] if b < len(fn.insns) else None
    fast, n = fn.target(g), len(spec["block"])
    if jb is None or jb.name != "BRA" or not jb.unconditional() or fast != b + 1:
        return "the call block doesn't branch over the fast block"
    if fn.entries[fast] != 1:
        return "another branch enters the fast block"
    if fn.target(b) != fast + n or any(t in fn.label_at for t in range(fast + 1, fast + n)):
        return "the fast block isn't four instructions ending at the join"
    env: dict = {"P": br.guard.lstrip("!")}
    at, members, inner = g, {g, c, b, *setup, *copies, *range(fast, fast + n)}, []
    for alts in spec["check"]:
        v = _defines(alts[0])
        d = fn.reach(at, env[v])
        hit = _first(fn, alts, fn.insns[d], env) if d is not None else None
        if hit is None:
            return f"the range check's ${v} isn't {alts[0]}"
        env, at = hit[1], d
        members.add(d)
    if not fn.steady(env["x"], at, g):
        return "$x changes between the range check and the branch"
    hit = _block(fn, fast, spec["block"], env)
    if hit is None:
        return f"the fast block at 0x{fn.insns[fast].addr:04x} isn't {kind}.rn's"
    env, pats = hit
    if "seed" in spec:
        d = fn.reach(fast, env["r"])
        if d is None or unify(spec["seed"][0], fn.insns[d], env) is None or not fn.steady(env["x"], d, g):
            return "the fast block's $r isn't MUFU.RSQ $r, $x of the branch's $x"
        members.add(d)
        inner.append((d, env["r"]))
    defs = {_defines(p): t for t, p in pats.items()}
    if (e := _flows(fn, defs, env, pats)) is not None:
        return e
    for v in {v for p in pats.values() for v in _reads(p)} - set(defs):
        if not fn.steady(env[v], fast, max(t for t, p in pats.items() if v in _reads(p))):
            return f"${v} changes in the fast block"
    inner += [(t, env[_defines(pats[t])]) for t in range(fast, fast + n - 1)]
    res = _returned(fn, c, copies, env["z"])
    if res is None:
        return "the slow path doesn't return its result in $z on every path"
    if (e := _passes(fn, c, [value(fn, g, env["x"])])) is not None:
        return e
    m = Match(f"{kind}-fast-path", members, inner)
    return _escape(fn, m) or m


def _f32(text: str) -> int | None:
    try:
        return struct.unpack("<I", struct.pack("<f", float(text)))[0]
    except (ValueError, OverflowError):
        return None


def _immediate(ins: Insn) -> int | None:
    """The 32 bits a load of an immediate (MOV, MOV32I, UMOV, IMAD.MOV.U32 RZ, RZ or HFMA2 -RZ, RZ) writes, else None."""
    o, a = ins.name, ins.args
    if (o in ("MOV", "MOV32I", "UMOV") and len(a) == 2) or (o == "IMAD.MOV.U32" and len(a) == 4 and a[1:3] == ["RZ", "RZ"]):
        if a[-1] in ("RZ", "URZ"):
            return 0
        return int(a[-1], 0) & 0xFFFFFFFF if re.fullmatch(r"-?(0x[0-9a-f]+|\d+)", a[-1]) else None
    if o in ("HFMA2", "HFMA2.MMA") and len(a) == 5 and a[1:3] == ["-RZ", "RZ"]:
        try:
            hi, lo = (struct.unpack("<H", struct.pack("<e", float(x)))[0] for x in a[3:])
        except (ValueError, OverflowError):
            return None
        return hi << 16 | lo
    return None


_SLOT = re.compile(r"c\[0x0\]\[(0x[0-9a-f]+)\]")


def _slot(ins: Insn, reg: str) -> str | None:
    """The slot of constant bank 0 whose 32 bits `reg` holds after a load of a fixed slot (LDC or ULDC, or the pair .64
    loads), else None."""
    m = _SLOT.fullmatch(ins.args[1]) if len(ins.args) == 2 and ins.op[0] in ("LDC", "ULDC") else None
    d = re.fullmatch(r"(U?R)(\d+)", ins.args[0]) if ins.args else None
    if m is None or d is None or ins.op[1:] not in ([], ["64"]):
        return None
    regs = [f"{d.group(1)}{int(d.group(2)) + i}" for i in range(2 if ins.op[1:] else 1)]
    return f"c[0x0][0x{int(m.group(1), 16) + 4 * regs.index(reg):x}]" if reg in regs else None


def _copied(ins: Insn) -> str | None:
    """The register a 32-bit copy (MOV, UMOV, IMAD.MOV.U32 or IMAD.U32 R, RZ, RZ, src) reads, else None."""
    a = ins.args
    if ins.name in ("MOV", "UMOV") and len(a) == 2 and re.fullmatch(r"U?R\d+", a[1]):
        return a[1]
    if ins.name in ("IMAD.MOV.U32", "IMAD.U32") and len(a) == 4 and a[1:3] == ["RZ", "RZ"] and re.fullmatch(r"U?R\d+", a[3]):
        return a[3]
    return None


def value(fn: Function, k: int, reg: str, seen: frozenset = frozenset()) -> tuple:
    """What `reg` holds at k, through the copies that set it on every path to k: ``("imm", bits)`` from a load of an
    immediate (RZ is 0), ``("const", slot)`` from a 32-bit LDC or ULDC of a fixed slot of constant bank 0 (the kernel's
    parameters, which nothing writes while it runs), else ``("reg", r, j)``: register r's value when j reads it."""
    if reg in ("RZ", "URZ"):
        return ("imm", 0)
    d = fn.reach(k, reg)
    if d is None or d in seen:
        return ("reg", reg, k)
    ins = fn.insns[d]
    if (src := _copied(ins)) is not None:
        return value(fn, d, src, seen | {d})
    if (v := _immediate(ins)) is not None:
        return ("imm", v)
    if (slot := _slot(ins, reg)) is not None:
        return ("const", slot)
    return ("reg", reg, k)


def same(fn: Function, x: tuple, y: tuple) -> bool:
    """Two `value` keys are the same 32 bits: equal constants, or one register unchanged from one's read to the other's."""
    if x[0] == y[0] == "reg":
        return x[1] == y[1] and (fn.steady(x[1], x[2], y[2]) or fn.steady(x[1], y[2], x[2]))
    return x == y


def _sources(fn: Function, lo: int, hi: int, k: int, reg: str) -> frozenset | None:
    """The inputs of the body [lo, hi) that `reg` may derive from when k reads it: ``("in", r)`` for a register the body
    reads before writing, ``("const", slot)`` for a fixed slot of constant bank 0 it loads. Every operand and the guard of
    each instruction on some path count, so this is a superset; None when a path leaves the body other than by RET."""
    def get(s: dict, r: str) -> frozenset:
        return s.get(r, frozenset({("in", r)}))

    state: dict[int, dict] = {lo: {}}
    work = [lo]
    while work:
        t = work.pop()
        s, ins = state[t], fn.insns[t]
        if ins.indirect or ins.op[0] in ("CALL", "EXIT"):
            return None
        src = frozenset().union(*(get(s, r) for r in ins.reads))
        out = dict(s)
        for r in ins.maydef:
            new = frozenset({("const", sl)}) if (sl := _slot(ins, r)) is not None else src
            out[r] = new if r in ins.kills else get(s, r) | new
        nxt = fn.succ(t)
        if nxt is None:
            return None
        if ins.op[0] == "RET" and ins.guard is None:
            continue
        for u in nxt:
            if not lo <= u < hi:
                return None
            old = state.get(u)
            merged = out if old is None else {r: get(old, r) | get(out, r) for r in old.keys() | out.keys()}
            if merged != old:
                state[u] = merged
                work.append(u)
    if k not in state:
        return frozenset()
    return get(state[k], reg)


def _passes(fn: Function, c: int, want: list[tuple]) -> str | None:
    """None when the slow path the CALL at c enters reads exactly the operands `want` (`value` keys), each in its role: the
    registers its body may read before writing them, the constant-bank slots it loads, and the return address (the
    instruction after c). `want` is [x] (rcp, sqrt) or [a, b] (div): the div body's divisor is the one input its MUFU.RCP's
    operand may derive from (`_sources`), and its numerator the other."""
    lo, hi = fn.subroutines[fn.insns[c].targets[0]][1:]
    inputs, slots = [], set()
    for t in range(lo, hi):
        ins = fn.insns[t]
        if "c[" in ins.text:
            if ins.name not in ("LDC", "ULDC") or (slot := _slot(ins, ins.args[0])) is None:
                return f"the slow path reads the constant bank at 0x{ins.addr:04x} other than one fixed 32-bit slot"
            slots.add(slot)
    for reg in sorted({r for t in range(lo, hi) for r in fn.insns[t].reads}):
        eff = fn.callee_effect(c, reg)
        if eff is None:
            return "the slow path may leave other than by its return"
        if eff[0]:
            inputs.append((("in", reg), value(fn, c, reg)))
    back = ("imm", fn.insns[c + 1].addr) if c + 1 < len(fn.insns) else None
    if back not in [v for _, v in inputs]:
        return "the slow path isn't passed its return address"
    inputs.remove(next(p for p in inputs if p[1] == back))
    inputs += [(("const", s), ("const", s)) for s in sorted(slots)]
    if len(inputs) != len(want):
        return "the slow path isn't passed the fast path's operands"
    if len(want) == 2:
        rcp = [t for t in range(lo, hi) if fn.insns[t].name == "MUFU.RCP" and len(fn.insns[t].args) == 2]
        d = _sources(fn, lo, hi, rcp[0], fn.insns[rcp[0]].args[1]) if len(rcp) == 1 else None
        roles = [o for o, _ in inputs]
        if d is None or len(d & set(roles)) != 1:
            return "the slow path's MUFU.RCP doesn't take exactly one of its inputs as the divisor"
        if roles[0] in d:
            inputs.reverse()
    if not all(same(fn, x, y) for (_, x), y in zip(inputs, want)):
        return "the slow path isn't passed the fast path's operands" + (" in their roles" if len(want) == 2 else "")
    return None


def match_div(fn: Function, k0: int, bodies: frozenset[str]) -> Match | str:
    """div.rn's fast path from its seed MUFU.RCP at k0, around a call into one of `bodies`."""
    env = unify(DIV["seed"][0], fn.insns[k0], {})
    if env is None:
        return "not MUFU.RCP"
    n = len(fn.insns)
    start = k0
    while start > 0 and start not in fn.label_at and not _control(fn.insns[start - 1]):
        start -= 1
    pats, defs, at = {k0: DIV["seed"][0]}, {"r": k0}, k0
    for alts in DIV["chain"]:
        t, hit = at + 1, None
        while t < n and t not in fn.label_at and not _control(fn.insns[t]):
            hit = _first(fn, alts, fn.insns[t], env)
            if hit is not None:
                break
            t += 1
        if hit is None:
            return f"no {alts[0]} after the seed"
        pats[t], env = hit
        defs[_defines(hit[0])] = at = t
    g = at + 1
    while g < n and g not in fn.label_at and not _control(fn.insns[g]):
        g += 1
    qa = next(t for t, p in pats.items() if _defines(p) == "q")
    if re.fullmatch(r"U?R\d+", env["a"]):
        a = value(fn, qa, env["a"])
    elif (bits := _f32(env["a"])) is not None:
        a = ("imm", bits)
    else:
        return f"the numerator {env['a']} isn't an FP32 immediate"
    b = value(fn, k0, env["b"])
    check = DIV["check"][0]
    fchk = [(t, e) for t in range(start, g) if (e := unify(check, fn.insns[t], env)) is not None
            and same(fn, value(fn, t, e["n"]), a)]
    if len(fchk) != 1:
        return f"no {check} in the block whose $n holds $a"
    (t, env), = fchk
    fchk = [t]
    pats[t] = check
    br = fn.insns[g] if g < n else None
    if br is None or g in fn.label_at or br.name != "BRA" or br.guard != "!" + env["P"] or len(br.args) != 1:
        return "no @!$P BRA after the fast path"
    if not fn.holds(env["P"], fchk[0], g):
        return "$P changes between FCHK and the branch"
    if (e := _flows(fn, defs, env, pats)) is not None:
        return e
    cb = _call_block(fn, g, DIV["slowpath"], bodies)
    if isinstance(cb, str):
        return cb
    setup, c, copies, after = cb
    if fn.target(g) != after:
        return "the branch doesn't skip exactly the call block"
    res = _returned(fn, c, copies, env["z"])
    if res is None:
        return "the slow path doesn't return its result in $z on every path"
    if (e := _passes(fn, c, [a, b])) is not None:
        return e
    inner = [(t, env[_defines(p)]) for t, p in pats.items() if _defines(p) not in ("z", "P")]
    m = Match("div-fast-path", {g, c, *setup, *copies, *pats}, inner)
    return _escape(fn, m) or m


def _add_immediate(ins: Insn) -> tuple[str, str] | None:
    """(the register, the immediate) of a 32-bit add of the two, in any of its spellings: IADD3 R, PT, PT, R, imm, RZ
    (or UIADD3 with UPT and URZ), and sm_90a's IADD3 R, R, imm, RZ and VIADD R, R, imm; else None."""
    a = ins.args
    pt, zero = ("UPT", "URZ") if ins.name == "UIADD3" else ("PT", "RZ")
    if ins.name in ("IADD3", "UIADD3") and len(a) == 6 and a[1:3] == [pt, pt] and a[5] == zero:
        return a[3], a[4]
    if (ins.name == "IADD3" and len(a) == 4 and a[3] == "RZ") or (ins.name == "VIADD" and len(a) == 3):
        return a[1], a[2]
    return None


def match_int_seed(fn: Function, k0: int) -> Match | str:
    """Integer division's reciprocal seed from its MUFU.RCP at k0: I2F.RP d -> MUFU.RCP -> +2 ulp -> F2I.TRUNC."""
    ins = fn.insns[k0]
    if ins.guard or ins.name != "MUFU.RCP" or len(ins.args) != 2:
        return "not MUFU.RCP"
    r, f = ins.args
    d = fn.reach(k0, f)
    cv = fn.insns[d] if d is not None else None
    if cv is None or cv.op[0] not in ("I2F", "UI2F") or "RP" not in cv.op or "F64" in cv.op or cv.args[0] != f:
        return "its operand isn't I2F.RP of an integer"
    members, inner = {d, k0}, [(k0, r)]

    def link(k: int, reg: str) -> int | None:
        """The one reader of k's `reg` that only k's value reaches (the chain's next step); `_escape` then judges the
        others."""
        uses = fn.readers(k, reg)
        near = [u for u in uses or () if fn.reach(u, reg) == k]
        return near[0] if len(near) == 1 else None

    at, val = k0, r
    u = link(k0, r)
    if u is None:
        return "the seed has no one next step"
    ui = fn.insns[u]
    if ui.name == "R2UR" and not ui.guard and ui.args[1:] == [r]:
        members.add(u)
        inner.append((u, ui.args[0]))
        at, val = u, ui.args[0]
        u = link(at, val)
        if u is None:
            return "the uniform seed has no one next step"
        ui = fn.insns[u]
    if ui.guard or (added := _add_immediate(ui)) is None or added[0] != val or added[1] not in INT_ROUND_UP:
        return "the seed isn't rounded up by IADD3 +2 ulp"
    s = ui.args[0]
    members.add(u)
    inner.append((u, s))
    q = link(u, s)
    qi = fn.insns[q] if q is not None else None
    if qi is None or qi.guard or qi.op[0] not in ("F2I", "UF2I") or "TRUNC" not in qi.op \
            or not {"U32", "U64"} & set(qi.op) or qi.args[1:] != [s]:
        return "the rounded seed isn't truncated by F2I"
    members.add(q)
    m = Match("int-seed", members, inner)
    return _escape(fn, m) or m


# ---- subroutine signatures ----------------------------------------------------------------------------------------------

_CLASS = re.compile(r"(?<![\w.$])(UR|R|UP|P|B)\d+(?!\w)")


def signature(fn: Function, start: int, end: int) -> list[list[str]]:
    """A subroutine body's shape: each basic block's instructions, sorted, with registers as their class, moves between
    registers dropped (register allocation adds and removes them), each load of an immediate written as ``MOV R, 0x...``
    (the scheduler picks MOV or HFMA2) and branch targets as relative block numbers."""
    body = list(range(start, end))
    while body and (fn.insns[body[-1]].name == "NOP" or (fn.insns[body[-1]].op[0] == "BRA" and fn.target(body[-1]) == body[-1])):
        body.pop()
    if not body:
        return []
    starts = sorted({start} | {t for k in body for lab in fn.insns[k].targets
                               if (t := fn.labels.get(lab)) is not None and start <= t <= body[-1]}
                    | {k + 1 for k in body if fn.insns[k].op[0] in ("BRA", "RET", "EXIT") and k + 1 <= body[-1]})
    block_of = {k: sum(s <= k for s in starts) - 1 for k in body}
    out = []
    for bi, s in enumerate(starts):
        e = starts[bi + 1] if bi + 1 < len(starts) else body[-1] + 1
        items = []
        for k in range(s, e):
            ins = fn.insns[k]
            if ins.guard is None and (COPY.fullmatch(ins.body()) or re.fullmatch(r"UMOV UR\d+, UR\d+|MOV\.64 R\d+, R\d+",
                                                                                 ins.body())):
                continue
            v = _immediate(ins)
            t = (("@" + ins.guard + " ") if ins.guard else "") + (
                ins.body() if v is None else f"MOV {ins.args[0]}, 0x{v:08x}")

            def tgt(m, k=k):
                lab = m.group(1)
                sub = _SUBROUTINE.fullmatch(lab)
                if sub:
                    return f"`({sub.group(1)})"
                i = fn.labels.get(lab)
                if ins.op[0] == "RET":
                    return "`(caller)"
                return f"`(B{block_of[i] - bi:+d})" if i in block_of else "`(END)"

            items.append(_CLASS.sub(lambda m: m.group(1), _TARGET.sub(tgt, t)))
        out.append(sorted(items))
    return out


def signature_id(blocks: list[list[str]]) -> str:
    return hashlib.sha256(json.dumps(blocks).encode()).hexdigest()[:16]


# ---- tool notes and pins ------------------------------------------------------------------------------------------------


def tool_notes(cubin: bytes) -> list[dict]:
    """The cubin's .note.nv.tkinfo records: the tool (ptxas, nvlink), its version, build and arguments."""
    out = []
    for name, (typ, off, size) in _sections(cubin).items():
        if name != ".note.nv.tkinfo" or typ != SHT_NOTE:
            continue
        b, p = cubin[off:off + size], 0
        while p + 12 <= len(b):
            namesz, descsz, _ = struct.unpack_from("<III", b, p)
            q = p + 12 + ((namesz + 3) & ~3)
            desc = b[q:q + descsz]
            if len(desc) < 24:
                raise SassGateFailed(f"a .note.nv.tkinfo record of {len(desc)} bytes")
            strs = desc[24:]

            def at(o):
                end = strs.find(b"\0", o)
                return strs[o:end if end >= 0 else len(strs)].decode("utf-8", "replace")

            tool, version, branch, args = (at(o) for o in struct.unpack_from("<4I", desc, 8))
            out.append({"tool": tool, "version": version, "branch": branch, "args": " ".join(args.split())})
            p = q + ((descsz + 3) & ~3)
    return out


PINS_VERSION = "sass-pins/v2"


def parse_pins(data: bytes | str, where: str) -> dict:
    pins = json.loads(data)
    if pins.get("version") != PINS_VERSION:
        raise SassGateFailed(f"{where} is {pins.get('version')!r}, not {PINS_VERSION}")
    for tk, rec in pins["toolkits"].items():
        for insn, exc in rec.get("exceptions", {}).items():
            if insn not in EXCEPTABLE:
                raise SassGateFailed(f"{where}: {tk} excepts {insn}, which sass_gate.EXCEPTABLE doesn't list (only "
                                     f"{', '.join(EXCEPTABLE)}): add it there with the one form its numeric run measured")
            if not exc.get("definition") or not (exc.get("numeric") or {}).get("runs"):
                raise SassGateFailed(f"{where}: {tk}'s exception for {insn} names no definition or no numeric runs")
    return pins


def load_pins(path: Path = PINS) -> dict:
    return parse_pins(path.read_bytes(), str(path))


def toolkit_of(notes: list[dict], pins: dict) -> tuple[str | None, str | None]:
    """The pinned toolkit every ptxas (and nvlink) note of a cubin names, with pinned ptxas arguments; else (None, why).
    One ptxas build may have an entry per target (13.0.1 for sm_120a, 13.0.1-sm_90a), each with its own arguments and run."""
    if not any(n["tool"] == "ptxas" for n in notes):
        return None, "no ptxas record (.note.nv.tkinfo)"
    unpinned = None
    for tk, rec in pins["toolkits"].items():
        if all(n["version"] == rec["version"] and n["branch"] == rec["branch"] for n in notes):
            bad = [n["args"] for n in notes if n["tool"] == "ptxas"
                   and " ".join(a for a in n["args"].split() if a != "-v") not in rec["args"]]
            if bad:
                unpinned = unpinned or f"ptxas {rec['version']} with unpinned arguments {bad[0]!r}"
                continue
            if not rec.get("numeric"):
                return None, f"ptxas {rec['version']} is pinned without a numeric run"
            return tk, None
    if unpinned:
        return None, unpinned
    tools = sorted({f"{n['tool']} {n['version']!r} ({n['branch']})" for n in notes})
    return None, f"no pinned toolkit built it: {'; '.join(tools)}"


def cubin_verdict(funcs: list[dict], notes: list[dict], pins: dict) -> dict:
    """One cubin's flagged instructions and which pinned sequence covers each; a cubin whose ptxas isn't pinned has none."""
    fns = [Function(f) for f in funcs]
    tk, why = toolkit_of(notes, pins) if any(f.insns for f in fns) else (None, None)
    pinned = pins["toolkits"][tk]["subroutines"] if tk else {}
    excepted = pins["toolkits"][tk].get("exceptions", {}) if tk else {}
    flagged_at, covered, hints, misses, merged = [], {}, {}, [], []
    fp64 = 0
    for fi, fn in enumerate(fns):
        for k, ins in enumerate(fn.insns):
            if flagged(ins.op):
                flagged_at.append((fi, k))
            elif ins.op[0] == "MUFU":
                fp64 += 1
        if tk is None:
            continue
        good = set()
        for lab, (name, s, e) in fn.subroutines.items():
            if name in pinned:
                sid = signature_id(signature(fn, s, e))
                good |= {lab} if sid in pinned[name] else set()
                for k in range(s, e):
                    if sid in pinned[name]:
                        covered[(fi, k)] = name
                    else:
                        hints[(fi, k)] = f"{name}'s body (signature {sid}) isn't a pinned one ({', '.join(pinned[name])})"
        bodies = frozenset(good)
        found: list[Match] = []
        for k, ins in enumerate(fn.insns):
            if ins.name == "CALL.REL.NOINC" and ins.targets and (m := _SUBROUTINE.fullmatch(ins.targets[0])):
                for kind, spec in FAST.items():
                    if m.group(1) == spec["slowpath"]:
                        r = match_fast(fn, k, kind, bodies)
                        if isinstance(r, str):
                            misses.append(f"{fn.name} 0x{ins.addr:04x} {kind}: {r}")
                        else:
                            found.append(r)
            elif ins.name == "MUFU.RCP" and not ins.guard:
                d = match_div(fn, k, bodies)
                r = d if isinstance(d, Match) else match_int_seed(fn, k)
                if isinstance(r, str):
                    hints.setdefault((fi, k), f"div: {d}; int seed: {r}")
                else:
                    found.append(r)
        for m in found:
            for k in m.members:
                if flagged(fn.insns[k].op):
                    covered.setdefault((fi, k), m.kind)
            merged += [{"function": fn.name, "sequence": m.kind, "value": f"{reg} from 0x{fn.insns[k].addr:04x}",
                        "reader": f"0x{fn.insns[u].addr:04x} {fn.insns[u].text}"} for k, reg, u in m.merged]
        for k, ins in enumerate(fn.insns):
            if (fi, k) in covered or ins.name not in EXCEPTABLE:
                continue
            if ins.name not in excepted:
                hints.setdefault((fi, k), f"{tk} lists no exception for {ins.name}")
            elif EXCEPTABLE[ins.name].fullmatch(ins.body()):
                covered[(fi, k)] = ins.name
            else:
                hints.setdefault((fi, k), f"{ins.name} is excepted only as {EXCEPTABLE[ins.name].pattern}")
    bad = [(fi, k) for fi, k in flagged_at if (fi, k) not in covered]
    return {"tool_notes": notes, "toolkit": tk, "unpinned": why, "functions": len(fns),
            "flagged": len(flagged_at), "covered": dict(Counter(covered[x] for x in flagged_at if x in covered)),
            "merged_reads_n": len(merged), "merged_reads": merged[:MAX_LISTED],
            "fp64_mufu": fp64, "unmatched_calls": misses[:MAX_LISTED], "offending_n": len(bad),
            "offending": [{"function": fns[fi].name, "address": f"0x{fns[fi].insns[k].addr:04x}",
                           "instruction": fns[fi].insns[k].text,
                           **({"why": hints[(fi, k)]} if (fi, k) in hints else {})} for fi, k in bad[:MAX_LISTED]],
            "_flagged": Counter((fns[fi].name, fns[fi].insns[k].addr, fns[fi].insns[k].text) for fi, k in flagged_at)}


# ---- the binaries' cubins -----------------------------------------------------------------------------------------------


def _sections(data: bytes) -> dict[str, tuple[int, int, int]]:
    """An ELF64's sections: name -> (type, offset, size)."""
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    heads = [struct.unpack_from("<IIQQQQ", data, shoff + i * shentsize) for i in range(shnum)]
    stroff = heads[shstrndx][4]
    return {data[stroff + h[0]:data.index(b"\0", stroff + h[0])].decode(): (h[1], h[4], h[5]) for h in heads}


def _elf_size(data: bytes) -> int:
    shoff, = struct.unpack_from("<Q", data, 0x28)
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", data, 0x36)
    ends = [shoff + shentsize * shnum, phoff + phentsize * phnum]
    ends += [off + size for typ, off, size in _sections(data).values() if typ != 8]  # SHT_NOBITS takes no file bytes
    return max(ends)


def fatbin_entries(data: bytes, where: str) -> tuple[list[bytes], list[str]]:
    """The ELF cubins and PTX texts of one or more concatenated fatbins; a compressed or unknown entry fails."""
    cubins, ptx, pos = [], [], 0
    while pos < len(data):
        magic, _, hsz, fsz = struct.unpack_from("<IHHQ", data, pos)
        if magic != FATBIN_MAGIC:
            raise SassGateFailed(f"{where}: no fatbin header at offset {pos}")
        e, end = pos + hsz, pos + hsz + fsz
        while e < end:
            kind, _, ehs, psz = struct.unpack_from("<HHIQ", data, e)
            flags, = struct.unpack_from("<Q", data, e + 40)
            payload = data[e + ehs:e + ehs + psz]
            if flags & FATBIN_COMPRESSED:
                raise SassGateFailed(f"{where}: a compressed fatbin entry (kind {kind}) the gate can't read; build with "
                                     "nvcc --no-compress, or ship the cubin")
            if kind == 2 and payload[:4] == b"\x7fELF":
                size = _elf_size(payload)
                if size > psz:
                    raise SassGateFailed(f"{where}: a cubin of {size} bytes in a {psz}-byte fatbin entry")
                cubins.append(payload[:size])
            elif kind == 1:
                ptx.append(payload.rstrip(b"\0").decode("utf-8", "replace"))
            else:
                raise SassGateFailed(f"{where}: an unknown fatbin entry (kind {kind})")
            e += ehs + psz
        pos = end
        while pos < len(data) and data[pos] == 0:
            pos += 1
    return cubins, ptx


def device_code(data: bytes, where: str) -> tuple[list[bytes], list[str]]:
    """A cubin, a fatbin, or a host ELF with a .nv_fatbin section -> (cubins, PTX texts)."""
    if data[:4] == b"\x7fELF":
        if struct.unpack_from("<H", data, 0x12)[0] == EM_CUDA:
            return [data], []
        sec = _sections(data).get(".nv_fatbin")
        if sec is None or not sec[2]:
            raise SassGateFailed(f"{where}: no device code (no .nv_fatbin section)")
        return fatbin_entries(data[sec[1]:sec[1] + sec[2]], where)
    if len(data) >= 4 and struct.unpack_from("<I", data)[0] == FATBIN_MAGIC:
        return fatbin_entries(data, where)
    raise SassGateFailed(f"{where}: not a cubin, a fatbin or an ELF with device code")


def has_device_code(path: str) -> bool:
    """Whether a host ELF has a non-empty .nv_fatbin section, from its headers alone."""
    try:
        with open(path, "rb") as f:
            head = f.read(64)
            if head[:4] != b"\x7fELF" or head[4] != 2:
                return False
            shoff, = struct.unpack_from("<Q", head, 0x28)
            shentsize, shnum, shstrndx = struct.unpack_from("<HHH", head, 0x3A)
            f.seek(shoff)
            table = f.read(shentsize * shnum)
            heads = [struct.unpack_from("<IIQQQQ", table, i * shentsize) for i in range(shnum)]
            f.seek(heads[shstrndx][4])
            names = f.read(heads[shstrndx][5])
    except (OSError, struct.error, IndexError):
        return False
    return any(names[h[0]:names.find(b"\0", h[0])] == b".nv_fatbin" and h[5] for h in heads)


# ---- the listings -------------------------------------------------------------------------------------------------------


def parse_nvdisasm(text: str) -> list[dict]:
    """``nvdisasm -c``'s listing -> one dict per .text section: its name, instructions (address, text), labels (-> the index
    of the instruction after them) and ``.size`` end labels."""
    funcs: list[dict] = []
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"\.section\s", s):
            m = re.match(r"\.section\s+\.text\.([^,\s]+)", s)
            cur = {"name": m.group(1), "insns": [], "labels": {}, "sizes": {}} if m else None
            if cur is not None:
                funcs.append(cur)
        elif cur is None:
            continue
        elif m := re.match(r"\.size\s+(\S+),\((\S+) - (\S+)\)$", s):
            cur["sizes"][m.group(1)] = m.group(2)
        elif m := re.fullmatch(r"([^\s/][^\s]*):", s):
            cur["labels"][m.group(1)] = len(cur["insns"])
        elif m := _INSN.search(s):
            cur["insns"].append((int(m.group(1), 16), " ".join(m.group(2).split())))
    return funcs


def cuobjdump_flagged(listing: str) -> Counter:
    """``cuobjdump -sass``'s flagged instructions, as (function, address, instruction)."""
    fn, got = None, Counter()
    for line in listing.splitlines():
        if m := re.match(r"\s*Function : (\S+)", line):
            fn = m.group(1)
        elif (m := _INSN.search(line)) and flagged(opcode(m.group(2))):
            got[(fn, int(m.group(1), 16), " ".join(m.group(2).split()))] += 1
    return got


def ptx_bad(ptx: list[str]) -> list[str]:
    return [ln.strip() for text in ptx for ln in text.splitlines() if PTX_BAD.search(ln.split("//")[0])]


def judge(cubins: list[tuple[list[dict], list[dict]]], cuobjdump_listing: str, ptx: list[str], pins: dict) -> dict:
    """The verdict on a binary: its cubins (`parse_nvdisasm`'s functions and `tool_notes`), its ``cuobjdump -sass`` listing
    and its PTX texts."""
    recs = [cubin_verdict(f, n, pins) for f, n in cubins]
    listed = cuobjdump_flagged(cuobjdump_listing)
    agree = listed == sum((r.pop("_flagged") for r in recs), Counter())
    bad = [{**o, "cubin": i} for i, r in enumerate(recs) for o in r["offending"]]
    pbad = ptx_bad(ptx)
    covered = sum((Counter(r["covered"]) for r in recs), Counter())
    return {"ptx_entries": len(ptx), "cubins": recs, "flagged": sum(r["flagged"] for r in recs), "covered": dict(covered),
            "merged_reads_n": sum(r["merged_reads_n"] for r in recs),
            "offending_n": sum(r["offending_n"] for r in recs), "offending": bad[:MAX_LISTED],
            "ptx_bad_n": len(pbad), "ptx_bad": pbad[:MAX_LISTED],
            "cross_check": {"cuobjdump_flagged": sum(listed.values()), "agree": agree},
            "ok": agree and not bad and not pbad}


# ---- the gate -----------------------------------------------------------------------------------------------------------


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SassGateFailed(f"{' '.join(cmd)} exited {r.returncode}: {r.stderr.strip()[-400:]}")
    return r.stdout


def find_tools() -> dict[str, dict[str, str]]:
    """cuobjdump and nvdisasm: from POUW_CUDA_BIN, CUDA_HOME/bin, the PATH, then /usr/local/cuda/bin."""
    dirs = [os.environ.get("POUW_CUDA_BIN"), os.environ.get("CUDA_HOME") and os.path.join(os.environ["CUDA_HOME"], "bin")]
    out = {}
    for t in ("cuobjdump", "nvdisasm"):
        cands = [os.path.join(d, t) for d in dirs if d] + [shutil.which(t) or "", f"/usr/local/cuda/bin/{t}"]
        path = next((c for c in cands if c and os.access(c, os.X_OK)), None)
        if path is None:
            raise SassGateFailed(f"no {t} (set POUW_CUDA_BIN or CUDA_HOME): the gate can't read SASS")
        out[t] = {"path": path, "version": _run([path, "--version"]).strip().splitlines()[-1]}
    return out


def _cache_dir() -> Path | None:
    d = os.environ.get("POUW_SASS_GATE_CACHE", str(Path.home() / ".cache" / "pouw-harness" / "sass-gate"))
    return None if d == "none" else Path(d)


def _disassemble(cubins: list[bytes], tools: dict) -> list[tuple[list[dict], list[dict]]]:
    out = []
    with tempfile.TemporaryDirectory() as td:
        for i, c in enumerate(cubins):
            p = Path(td) / f"{i}.cubin"
            p.write_bytes(c)
            out.append((parse_nvdisasm(_run([tools["nvdisasm"]["path"], "-c", str(p)])), tool_notes(c)))
    return out


def scan(path: str | Path, tools: dict, cache: Path | None = None, pins_path: Path = PINS) -> dict:
    """One binary's verdict: its flagged instructions, which pinned sequence covers each, each cubin's ptxas, the
    cross-check, and ``ok``."""
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    pins_bytes = pins_path.read_bytes()
    key = hashlib.sha256(json.dumps([VERSION, hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     hashlib.sha256(pins_bytes).hexdigest(), sha,
                                     sorted((t, v["version"]) for t, v in tools.items())]).encode()).hexdigest()
    hit = cache / f"{key}.json" if cache else None
    if hit and hit.exists():
        return {**json.loads(hit.read_text()), "path": str(path), "cached": True}
    cubins, ptx = device_code(data, str(path))
    if not cubins:
        raise SassGateFailed(f"{path}: no SASS (PTX only: the driver would JIT it); ship sm_120a SASS")
    rec = {"path": str(path), "sha256": sha, "bytes": len(data), "pins_sha256": hashlib.sha256(pins_bytes).hexdigest(),
           **judge(_disassemble(cubins, tools), _run([tools["cuobjdump"]["path"], "-sass", str(path)]), ptx,
                   parse_pins(pins_bytes, str(pins_path))), "cached": False}
    if rec["ok"] and hit:
        hit.parent.mkdir(parents=True, exist_ok=True)
        tmp = hit.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(hit)
    return rec


def why_failed(rec: dict) -> str:
    if rec["offending_n"]:
        o = rec["offending"][0]
        c = rec["cubins"][o["cubin"]]
        where = (f" (cubin {o['cubin']}: {c['unpinned']}, so nothing is exempt)" if c["unpinned"]
                 else f" ({o['why']})" if o.get("why") else "")
        return (f"{rec['offending_n']} flagged instruction(s) (.FTZ or FP32 MUFU) outside a pinned IEEE sequence or "
                f"exception, first "
                f"{o['function']} {o['address']}: {o['instruction']}{where}")
    if rec["ptx_bad_n"]:
        return f"{rec['ptx_bad_n']} .ftz, .approx or div.full PTX instruction(s), first: {rec['ptx_bad'][0]}"
    return (f"nvdisasm and cuobjdump -sass disagree ({rec['flagged']} against {rec['cross_check']['cuobjdump_flagged']} "
            "flagged instructions)")


def declared(arm: str, build: dict) -> list[str]:
    """An arm's build record's ``binaries``: every cubin, fatbin or library its kernels come from (arm interface v0.3)."""
    b = build.get("binaries") if isinstance(build, dict) else None
    if not isinstance(b, list) or not all(isinstance(p, str) for p in b):
        raise SassGateFailed(f"arm {arm}: build() returns no 'binaries' list (arm interface v0.3): name every cubin, fatbin "
                             "or library its kernels are loaded from ([] if they all come from libraries mapped in the "
                             "process, which the gate finds itself)")
    return b


def mapped() -> list[str]:
    """The files mapped into this process, from /proc/self/maps."""
    out = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith("/") and parts[5] not in out:
            out.append(parts[5])
    return out


def regions(path: str) -> list[tuple[int, int]]:
    """Each (start, end) where /proc/self/maps maps `path`."""
    out = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5] == path:
            s, e = parts[0].split("-")
            out.append((int(s, 16), int(e, 16)))
    return out


def read_mapped(start: int, n: int) -> bytes:
    fd = os.open("/proc/self/mem", os.O_RDONLY)
    try:
        return os.pread(fd, n, start)
    finally:
        os.close(fd)


def posix_semaphore(path: str) -> str | None:
    """Why a deleted /dev/shm/sem.* mapping is a semaphore, not device code; None when it may be device code."""
    rs = regions(path)
    if not rs or any(e - s > PAGE for s, e in rs):
        return None
    try:
        data = [read_mapped(s, e - s) for s, e in rs]
    except OSError:
        return None
    if any(d[:4] == b"\x7fELF" or struct.pack("<I", FATBIN_MAGIC) in d for d in data):
        return None
    return "a POSIX semaphore (one page, no ELF or fatbin), not a library"


class SassChecker:
    """The run's gate: every binary it has checked, and the mapped ones it didn't scan (NVIDIA's). `check` raises
    SassGateFailed naming the first failing binary, its function and instruction; `report` goes into the run's JSON."""

    def __init__(self, log=None, tools: dict | None = None, cache: Path | None | str = "default", pins: Path = PINS):
        self.log = log or (lambda s: None)
        self.tools = tools
        self.cache = _cache_dir() if cache == "default" else cache
        self.pins = pins
        self.records: dict[str, dict] = {}
        self.not_scanned: dict[str, str] = {}

    def check(self, paths: list[str], source: str) -> None:
        if self.tools is None:
            self.tools = find_tools()
        failed = []
        for p in paths:
            real = os.path.realpath(p)
            if real in self.records:
                continue
            if not os.path.isfile(real):
                raise SassGateFailed(f"{source}: {p} is not a file")
            rec = {**scan(real, self.tools, self.cache, self.pins), "source": source}
            self.records[real] = rec
            tks = sorted({c["toolkit"] or "unpinned" for c in rec["cubins"] if c["functions"]})
            merged = f", {rec['merged_reads_n']} merged read(s)" if rec["merged_reads_n"] else ""
            self.log(f"sass gate: {'pass' if rec['ok'] else 'FAIL'} {real} (ptxas {', '.join(tks)}; {rec['flagged']} "
                     f"flagged, covered {rec['covered']}{merged}{', cached' if rec['cached'] else ''})")
            if not rec["ok"]:
                failed.append(rec)
        if failed:
            raise SassGateFailed("; ".join(f"{r['path']}: {why_failed(r)}" for r in failed))

    def discover(self) -> list[str]:
        """Mapped files with device code, NVIDIA's libraries aside (recorded as not scanned)."""
        out = []
        for p in mapped():
            if ANON_SHARED.fullmatch(p):
                self.not_scanned[p] = "shared anonymous memory, not a file"
                continue
            if POSIX_SEM.fullmatch(p) and (why := posix_semaphore(p)):
                self.not_scanned[p] = why
                continue
            if p.endswith(" (deleted)"):
                raise SassGateFailed(f"{p} is mapped but deleted: the gate can't read it")
            if os.path.realpath(p) in self.records or p in self.not_scanned or not has_device_code(p):
                continue
            if NVIDIA_LIB.fullmatch(os.path.basename(p)):
                self.not_scanned[p] = "NVIDIA's library"
            else:
                out.append(p)
        return out

    def report(self) -> dict:
        return {"version": VERSION, "tools": self.tools, "pins": str(self.pins), "binaries": list(self.records.values()),
                "not_scanned": [{"path": p, "reason": r} for p, r in self.not_scanned.items()],
                "ok": all(r["ok"] for r in self.records.values())}


def pin_record(paths: list[Path], tools: dict) -> list[dict]:
    """Each cubin's tool notes and each subroutine's signature: what a `sass_pins.json` entry is made from."""
    out = []
    for p in paths:
        cubins, _ = device_code(p.read_bytes(), str(p))
        for i, (funcs, notes) in enumerate(_disassemble(cubins, tools)):
            subs: dict[str, dict] = {}
            for f in map(Function, funcs):
                for name, s, e in f.subroutines.values():
                    blocks = signature(f, s, e)
                    subs.setdefault(name, {})[signature_id(blocks)] = blocks
            out.append({"path": str(p), "cubin": i, "tool_notes": notes, "subroutines": subs})
    return out


def update_pins(records: list[dict], toolkit: str, path: Path = PINS) -> dict:
    """Pin `toolkit` from `pin_record`s of its builds: the one ptxas version and build their notes name, the arguments
    (-v aside), and every signature of each subroutine with a flagged instruction (ptxas specializes a body to its
    callers, so one name may have several). Its numeric run is kept while those signatures are unchanged, and its
    exceptions while the ptxas is."""
    notes = [n for r in records for n in r["tool_notes"] if n["tool"] == "ptxas"]
    builds = {(n["version"], n["branch"]) for n in notes}
    if len(builds) != 1:
        raise SassGateFailed(f"the builds name {len(builds)} ptxas builds, not one")
    (version, branch), = builds
    subs: dict[str, set[str]] = {}
    blocks: dict[str, list] = {}
    for r in records:
        for name, sigs in r["subroutines"].items():
            for sid, b in sigs.items():
                if any(flagged(opcode(t.split(" ", 1)[1] if t.startswith("@") else t)) for blk in b for t in blk):
                    subs.setdefault(name, set()).add(sid)
                    blocks[sid] = b
    pinned = {name: sorted(s) for name, s in sorted(subs.items())}
    pins = load_pins(path) if path.exists() else {"version": "sass-pins/v2", "toolkits": {}, "signatures": {}}
    old = pins["toolkits"].get(toolkit, {})
    pins["version"] = "sass-pins/v2"
    pins["toolkits"][toolkit] = {
        "version": version, "branch": branch,
        "args": sorted({" ".join(a for a in n["args"].split() if a != "-v") for n in notes}),
        "subroutines": pinned,
        "numeric": old.get("numeric") if old.get("subroutines") == pinned else None,
        **({"exceptions": old["exceptions"]} if old.get("exceptions") and (old["version"], old["branch"]) == (version, branch)
           else {})}
    live = {sid for t in pins["toolkits"].values() for sids in t["subroutines"].values() for sid in sids}
    pins["signatures"] = {sid: b for sid, b in sorted({**pins["signatures"], **blocks}.items()) if sid in live}
    path.write_text(json.dumps(pins, indent=1) + "\n")
    return pins["toolkits"][toolkit]


# ---- the build gate: ptxas's report and cuobjdump's SASS of one kernel family -----------------------------------------
# Each kernel family's build.sh gate (pearl_c_sm90/, pearl_c_sm120/, torch_gpu's s8_linear/ and attn_bf16/) states the
# kernels it wants and the tensor-core forms each may run; `mma_gate` reads the build and refuses the rest.
# torch_gpu's units/ reads the build through `ptxas_report` and `sass_functions`.

MMA = re.compile(r"\b([QHOIU]MMA|[QHI]GMMA|UTC\w*MMA)(\.[\w.]+)?")
#: a ptxas line that refuses the build: a warning, an error or a serialized wgmma (C751x)
PTXAS_REFUSED = re.compile(r"warning|error|C751\d|serializ", re.I)
_ENTRY = re.compile(r"Compiling entry function '(\w+)' for '(\w+)'(.*?)(?=Compiling entry function|\Z)", re.S)
_FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")


def ptxas_report(ptxas: str, arch: str, smem: bool = False) -> tuple[dict[str, dict], list[str]]:
    """ptxas -v's report: each entry function's registers, stack frame and spills (None where the report gives none) and,
    with `smem`, its static shared memory; and what it refuses: a `PTXAS_REFUSED` line and an entry compiled for another
    arch."""
    problems = [line for line in ptxas.splitlines() if PTXAS_REFUSED.search(line)]
    funcs: dict[str, dict] = {}
    for m in _ENTRY.finditer(ptxas):
        if m.group(2) != arch:
            problems.append(f"{m.group(1)}: compiled for {m.group(2)}, want {arch}")
        regs, frame = re.search(r"Used (\d+) registers", m.group(3)), _FRAME.search(m.group(3))
        r = funcs[m.group(1)] = {"regs": int(regs.group(1)) if regs else None}
        r.update(zip(("stack", "spill_stores", "spill_loads"), map(int, frame.groups()) if frame else (None,) * 3))
        if smem:
            s = re.search(r"(\d+) bytes smem", m.group(3))
            r["static_smem"] = int(s.group(1)) if s else 0
    return funcs, problems


def sass_functions(sass: str) -> dict[str, list[str]]:
    """cuobjdump -sass's listing: each function's lines after its `Function :` header, in the listing's order (a header
    seen again starts that function's lines over)."""
    out: dict[str, list[str]] = {}
    cur = None
    for line in sass.splitlines():
        fm = re.search(r"Function : (\w+)", line)
        if fm:
            cur = fm.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def mma_gate(ptxas: str, sass: str, want, *, arch: str, record: dict, counts: dict[str, str],
             atom: Callable[[str, str, dict], bool], allowed: Callable[[str], str], line: Callable[[dict, str], None] | None = None,
             check: Callable[[str, dict], list[str]] | None = None, local_ok: dict[str, int] | None = None,
             smem: bool = False, unreported: bool = False) -> tuple[dict, list[str]]:
    """A kernel family's build gate: `ptxas_report`'s records and refusals, then each SASS function's `record` (a fresh copy)
    with `counts`' patterns counted over its lines, `line` run on each, and each MMA form `atom` counts into it (an MMA it
    doesn't is the function's ``other_mma``). Refused: an entry with no ptxas resource line (with `unreported`, also a SASS
    function ptxas didn't report); local memory, a stack frame or a spill (up to `local_ok`'s bytes for a kernel it names,
    and then LDL and STL too); an MMA form other than `allowed`'s; what `check` says of it; and a kernel of `want` missing
    from the build."""
    funcs, problems = ptxas_report(ptxas, arch, smem)
    for name, lines in sass_functions(sass).items():
        r = funcs.setdefault(name, {})
        r.update(copy.deepcopy(record))
        for text in lines:
            for key, pat in counts.items():
                r[key] += len(re.findall(pat, text))
            if line:
                line(r, text)
            for mm in MMA.finditer(text):
                if not atom(name, mm.group(0), r):
                    r["other_mma"].append(mm.group(0))
    for name, r in funcs.items():
        if ("regs" in r or unreported) and (r.get("regs") is None or r.get("stack") is None):
            problems.append(f"{name}: no ptxas resource line")
        bound = (local_ok or {}).get(name, 0)
        if max(r.get(k) or 0 for k in ("stack", "spill_stores", "spill_loads")) > bound or \
                (not bound and (r.get("LDL") or r.get("STL"))):
            problems.append(f"{name}: local memory {r}")
        if r.get("other_mma"):
            problems.append(f"{name}: MMA forms other than {allowed(name)}: {sorted(set(r['other_mma']))}")
        problems += check(name, r) if check else []
    missing = sorted(set(want) - set(funcs))
    if missing:
        problems.append(f"kernels missing from the cubin: {missing}")
    return funcs, problems


def write_verdict(out: str, doc: dict, problems: list[str], passed: Callable[[], str], refused: str = "REFUSED") -> int:
    """A build gate's CLI ending: `doc` as OUT_JSON, then exit 1 with the problems on stderr, or 0 with `passed()`'s line."""
    Path(out).write_text(json.dumps(doc, indent=1))
    if problems:
        print(f"{refused}: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(passed())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("binaries", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="write every binary's record here")
    ap.add_argument("--pin", action="store_true", help="print each cubin's tool notes and subroutine signatures")
    ap.add_argument("--toolkit", help="with --pin: write the builds' ptxas and signatures into sass_pins.json as this")
    args = ap.parse_args(argv)
    if args.pin:
        rec = pin_record(args.binaries, find_tools())
        if args.toolkit:
            print(json.dumps(update_pins(rec, args.toolkit), indent=1))
            return 0
        text = json.dumps(rec, indent=1)
        (args.json.write_text(text) if args.json else print(text))
        return 0
    g = SassChecker(log=lambda s: print(s, file=sys.stderr))
    code, error = 0, None
    for b in args.binaries:
        try:
            g.check([str(b)], "cli")
        except SassGateFailed as ex:
            code, error = EXIT_FAIL, str(ex)
            print(f"FAIL {ex}", file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps({**g.report(), "error": error}, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
