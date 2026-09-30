"""The SASS gate: no shipped kernel contains an FP32 FADD, FFMA or FMUL with .FTZ (server.md 09:34Z, from the assessor's
fp-model/sm120-scalar check `r20260930-091638-9747`). A build with -ftz=true or --use_fast_math flushes subnormal operands and
results to zero, which the scalar model the verifier replays does not. Before anything is timed, `bench.py` passes every
binary an arm declares (its build record's ``binaries``) and every library with device code mapped into the run through it;
NVIDIA's own libraries (cuBLASLt, the runtime, the driver) are listed as not scanned.

Any opcode of FADD, FFMA, FMUL and their packed (``2``) and immediate (``32I``) forms with FTZ among its suffixes, in any
order, is refused, with two exemptions. ptxas expands the IEEE intrinsics (div.rn, sqrt.rn and rcp.rn: ``__fdiv_rn``,
``__fsqrt_rn``, ``__frcp_rn``, ``/``, ``sqrtf``) with a few .FTZ instructions on operands its range checks keep normal, in any
build (Pearl-C's kernels use all three). The gate exempts the instructions of those expansions' non-ftz slow-path subroutines
(`EXEMPT_SUBROUTINES`, by nvdisasm's label and ``.size``) and of the fast path that branches around a call to one, when the
fast path's FP32 arithmetic is exactly that intrinsic's (`FAST_PATHS`). An -ftz=true build calls the ``_ftz_`` slow paths
instead and flushes its own arithmetic, so it fails.

The cubins come from a .cubin, a fatbin, or a host ELF's .nv_fatbin section (parsed here, since ``cuobjdump -xelf`` writes
same-named cubins over each other), each disassembled by nvdisasm; ``cuobjdump -sass`` of the whole binary must list the same
.FTZ instructions, or the gate fails. A compressed fatbin entry fails too (build with nvcc's ``--no-compress``, or ship the
cubin), as does a PTX entry with an ftz f32 add, sub, mul, fma or mad. Passes are cached by the binary's sha256, this
file's and the tools' versions (``POUW_SASS_GATE_CACHE``, default ``~/.cache/pouw-harness/sass-gate``; ``none`` disables).

  python sass_gate.py BINARY... [--json OUT]     # exit 0: every binary passes; 7: one fails (named, with its instructions)
"""

from __future__ import annotations

import argparse
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
from pathlib import Path

VERSION = "sass-gate/v2"
EXIT_FAIL = 7
FP32_ARITH = frozenset({"FADD", "FFMA", "FMUL", "FADD2", "FFMA2", "FMUL2", "FADD32I", "FFMA32I", "FMUL32I"})
EXEMPT_SUBROUTINES = {
    "__cuda_sm20_rcp_rn_f32_slowpath": "rcp",
    "__cuda_sm20_sqrt_rn_f32_slowpath": "sqrt",
    "__cuda_sm3x_div_rn_noftz_f32_slowpath": "div",
}
FAST_PATHS = {
    "rcp": ("FFMA R, R, R, -1", "FADD.FTZ R, -R, -RZ", "FFMA R, R, R, R"),
    "sqrt": ("FMUL.FTZ R, R, R", "FMUL.FTZ R, R, 0.5", "FFMA R, -R, R, R", "FFMA R, R, R, R"),
}
NVIDIA_LIB = re.compile(r"lib(cublas|cublasLt|cudart|cuda|nvrtc|nvrtc-builtins|nvJitLink|cufft|cufftw|curand|cusparse|cusolver"
                        r"|cudnn\w*|nccl|cupti|nvToolsExt|nvperf_\w+|nvidia-[\w-]+)\.so(\.\d+)*")
#: how /proc/self/maps names a MAP_SHARED | MAP_ANONYMOUS mapping: no file behind it, like private anonymous memory
SHARED_ANONYMOUS = "/dev/zero (deleted)"
PTX_FTZ = re.compile(r"\b(add|sub|mul|fma|mad)(\.r[nzmp])?\.ftz(\.sat)?\.f32(x2)?\b")
FATBIN_MAGIC = 0xBA55ED50
FATBIN_COMPRESSED = 0x2000
EM_CUDA = 190
MAX_LISTED = 20

_INSN = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
_PRED = re.compile(r"^@!?U?P[T0-9]+\s+")


class SassGateFailed(RuntimeError):
    pass


def opcode(text: str) -> list[str]:
    """``@!P0 FFMA.RZ.FTZ R0, ...`` -> ``["FFMA", "RZ", "FTZ"]``."""
    t = _PRED.sub("", text.lstrip("{ ").strip())
    return t.split(None, 1)[0].split(".") if t else [""]


def is_ftz_arith(text: str) -> bool:
    op = opcode(text)
    return op[0] in FP32_ARITH and "FTZ" in op[1:]


def shape(text: str) -> str:
    """An instruction with its registers as R (uniform ones as UR) and no reuse flags."""
    t = re.sub(r"\bUR\d+\b", "UR", text.replace(".reuse", ""))
    return " ".join(re.sub(r"\bR\d+\b", "R", t).split())


# ---- the binaries' cubins -------------------------------------------------------------------------------------------------


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


# ---- the listings ---------------------------------------------------------------------------------------------------------


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


def cuobjdump_ftz(listing: str) -> Counter:
    """``cuobjdump -sass``'s FP32 .FTZ instructions, as (function, address, instruction)."""
    fn, got = None, Counter()
    for line in listing.splitlines():
        if m := re.match(r"\s*Function : (\S+)", line):
            fn = m.group(1)
        elif (m := _INSN.search(line)) and is_ftz_arith(m.group(2)):
            got[(fn, int(m.group(1), 16), " ".join(m.group(2).split()))] += 1
    return got


def classify(fn: dict) -> list[tuple[int, str, str | None]]:
    """A function's FP32 .FTZ instructions, each (address, text, the exemption that covers it or None)."""
    insns, labels = fn["insns"], fn["labels"]
    exempt: list[str | None] = [None] * len(insns)
    for lab, i in labels.items():
        m = re.fullmatch(r"\$__internal_\d+_\$(\w+)", lab)
        end = fn["sizes"].get(lab)
        if m and m.group(1) in EXEMPT_SUBROUTINES and end in labels:
            for t in range(i, labels[end]):
                exempt[t] = m.group(1)
    for i, (_, text) in enumerate(insns):
        m = re.search(r"\(\$__internal_\d+_\$(\w+)\)", text)
        if ".".join(opcode(text)) != "CALL.REL.NOINC" or not m or EXEMPT_SUBROUTINES.get(m.group(1)) not in FAST_PATHS:
            continue
        kind = EXEMPT_SUBROUTINES[m.group(1)]
        k = i + 1
        while k < len(insns) and (opcode(insns[k][1]) == ["MOV"] or shape(insns[k][1]) == "IMAD.MOV.U32 R, RZ, RZ, R"):
            k += 1
        b = re.fullmatch(r"BRA `\((\S+)\)", insns[k][1]) if k < len(insns) else None
        j = labels.get(b.group(1)) if b else None
        if j is None or j <= k + 1:
            continue
        block = range(k + 1, j)
        if tuple(shape(insns[t][1]) for t in block if opcode(insns[t][1])[0] in FP32_ARITH) == FAST_PATHS[kind]:
            for t in block:
                exempt[t] = exempt[t] or f"{kind}-fast-path"
    return [(a, text, exempt[i]) for i, (a, text) in enumerate(insns) if is_ftz_arith(text)]


# ---- the gate -------------------------------------------------------------------------------------------------------------


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


def judge(funcs: list[dict], cuobjdump_listing: str, ptx: list[str]) -> dict:
    """The verdict on a binary's functions (`parse_nvdisasm`), its ``cuobjdump -sass`` listing and its PTX texts."""
    found = [(fn["name"], a, text, why) for fn in funcs for a, text, why in classify(fn)]
    other = Counter(op[0] for fn in funcs for _, text in fn["insns"]
                    if "FTZ" in (op := opcode(text))[1:] and op[0] not in FP32_ARITH)
    listed = cuobjdump_ftz(cuobjdump_listing)
    agree = listed == Counter((n, a, t) for n, a, t, _ in found)
    bad = [{"function": n, "address": f"0x{a:04x}", "instruction": t} for n, a, t, why in found if why is None]
    ptx_bad = [ln.strip() for text in ptx for ln in text.splitlines() if PTX_FTZ.search(ln)]
    return {"ptx_entries": len(ptx), "functions": len(funcs), "ftz_fp32": len(found),
            "exempt": dict(Counter(why for *_, why in found if why)), "offending_n": len(bad), "offending": bad[:MAX_LISTED],
            "ptx_ftz_n": len(ptx_bad), "ptx_ftz": ptx_bad[:MAX_LISTED], "other_ftz": dict(other),
            "cross_check": {"cuobjdump_ftz_fp32": sum(listed.values()), "agree": agree},
            "ok": agree and not bad and not ptx_bad}


def _cache_dir() -> Path | None:
    d = os.environ.get("POUW_SASS_GATE_CACHE", str(Path.home() / ".cache" / "pouw-harness" / "sass-gate"))
    return None if d == "none" else Path(d)


def scan(path: str | Path, tools: dict, cache: Path | None = None) -> dict:
    """One binary's verdict: its FP32 .FTZ instructions, which are exempt and why, the cross-check, and ``ok``."""
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    key = hashlib.sha256(json.dumps([VERSION, hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sha,
                                     sorted((t, v["version"]) for t, v in tools.items())]).encode()).hexdigest()
    hit = cache / f"{key}.json" if cache else None
    if hit and hit.exists():
        return {**json.loads(hit.read_text()), "path": str(path), "cached": True}
    cubins, ptx = device_code(data, str(path))
    if not cubins:
        raise SassGateFailed(f"{path}: no SASS (PTX only: the driver would JIT it); ship sm_120a SASS")
    funcs: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        for i, c in enumerate(cubins):
            p = Path(td) / f"{i}.cubin"
            p.write_bytes(c)
            funcs += parse_nvdisasm(_run([tools["nvdisasm"]["path"], "-c", str(p)]))
    rec = {"path": str(path), "sha256": sha, "bytes": len(data), "cubins": len(cubins),
           **judge(funcs, _run([tools["cuobjdump"]["path"], "-sass", str(path)]), ptx), "cached": False}
    if rec["ok"] and hit:
        hit.parent.mkdir(parents=True, exist_ok=True)
        tmp = hit.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(hit)
    return rec


def why_failed(rec: dict) -> str:
    if rec["offending_n"]:
        o = rec["offending"][0]
        return (f"{rec['offending_n']} FP32 .FTZ instruction(s) outside ptxas's IEEE expansions, first {o['function']} "
                f"{o['address']}: {o['instruction']}")
    if rec["ptx_ftz_n"]:
        return f"{rec['ptx_ftz_n']} ftz f32 PTX instruction(s), first: {rec['ptx_ftz'][0]}"
    return (f"nvdisasm and cuobjdump -sass disagree ({rec['ftz_fp32']} against {rec['cross_check']['cuobjdump_ftz_fp32']} "
            "FP32 .FTZ instructions)")


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


class Gate:
    """The run's gate: every binary it has checked, and the mapped ones it didn't scan (NVIDIA's). `check` raises
    SassGateFailed naming the first failing binary, its function and instruction; `report` goes into the run's JSON."""

    def __init__(self, log=None, tools: dict | None = None, cache: Path | None | str = "default"):
        self.log = log or (lambda s: None)
        self.tools = tools
        self.cache = _cache_dir() if cache == "default" else cache
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
            rec = {**scan(real, self.tools, self.cache), "source": source}
            self.records[real] = rec
            self.log(f"sass gate: {'pass' if rec['ok'] else 'FAIL'} {real} ({rec['ftz_fp32']} FP32 .FTZ, "
                     f"exempt {rec['exempt']}{', cached' if rec['cached'] else ''})")
            if not rec["ok"]:
                failed.append(rec)
        if failed:
            raise SassGateFailed("; ".join(f"{r['path']}: {why_failed(r)}" for r in failed))

    def discover(self) -> list[str]:
        """Mapped files with device code, NVIDIA's libraries aside (recorded as not scanned)."""
        out = []
        for p in mapped():
            if p == SHARED_ANONYMOUS:
                self.not_scanned[p] = "shared anonymous memory, not a file (the CUDA driver maps some)"
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
        return {"version": VERSION, "tools": self.tools, "binaries": list(self.records.values()),
                "not_scanned": [{"path": p, "reason": r} for p, r in self.not_scanned.items()],
                "ok": all(r["ok"] for r in self.records.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("binaries", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="write every binary's record here")
    args = ap.parse_args(argv)
    g = Gate(log=lambda s: print(s, file=sys.stderr))
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
