"""rows.sh's gate on the kernels a 64-row pass adds to `session.cu`, and their record: ptxas's report and cuobjdump's
SASS of those eight, through the harness's `mma_gate` (benchmarks/pouw/harness/sass_gate.py). Every other kernel of
the build is held to the base's, instruction for instruction, by `same_sass.py`. Refuses (exit 1, the reasons on
stderr):

- one of the eight missing from the build or from the listing, compiled for another arch, or with a ptxas warning;
- a sketching kernel (`SKETCHING`, the six that compute σ) whose tensor-core instructions aren't the int8 form alone,
  IMMA.16832.S8.S8 (mma.m16n8k32 on s8 operands into s32), in a multiple of 4 (64 rows are four of its 16), and any
  tensor-core instruction in the other two;
- local memory (a stack frame, a spill, LDL or STL) beyond `LOCAL_OK`'s bytes: none in the pass's receipt grid and
  the opening's receipts; the consumers' and the two once-a-pass kernels' frames as they are; and in the two in-place
  kernels none, or 16 bytes where the build caps them at 128 registers (`CAPPED`), which is what that build gives up;
- floating point: an .FTZ that isn't integer division's truncation, a MUFU that isn't the reciprocal of an integer
  divisor, or any float arithmetic. The compiler's unsigned division by a runtime divisor (a tile's group, a
  subtree's chunk) is the only floating point these kernels hold, and it never sees a block's byte.

OUT_JSON holds each kernel's record: its registers, stack frame and spills, its instructions and the counts above.
With --listings, DIR gets each kernel's listing, a line an entry as `same_sass.py` reads it.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON [--arch sm_90a] [--listings DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from same_sass import stem

if TYPE_CHECKING:
    from types import ModuleType


def _harness() -> ModuleType:
    name = "pouw_harness_sass_gate"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parents[1] / "pouw" / "harness" / "sass_gate.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


H = _harness()
ATOM = "IMMA.16832.S8.S8"
#: the kernels that compute σ at 64 rows: the pass's receipt grid, its consumers (a consumer's own tile), an opening's
#: receipts, and the in-place pass's three
SKETCHING = {"sketch64_pass_kernel", "pass_consumer_kernel<64>", "subtree64_receipts_kernel", "mapped64_pass_kernel",
             "mapped64_receipts_kernel", "mapped_consumer_kernel<64>"}
#: a pass's packed signs from its seed, and a chained pass's seed and signs: one launch a pass
ONCE = {"signs_kernel<64>", "chain_prepare_kernel<64>"}
WANT = SKETCHING | ONCE
#: the bytes of stack frame or spill a kernel may hold. The consumers run at the register limit, as their 16-row
#: instances do; the once-a-pass kernels keep a SHA-512 schedule on the stack, as theirs do.
LOCAL_OK = {"pass_consumer_kernel<64>": 224, "mapped_consumer_kernel<64>": 288, "signs_kernel<64>": 256,
            "chain_prepare_kernel<64>": 304}
#: the in-place kernels a build may cap at CAP registers (two CTAs an SM), where each spills up to SPILL bytes
CAPPED, CAP, SPILL = {"mapped64_pass_kernel", "mapped64_receipts_kernel"}, 128, 16
COUNTS = {"LDL": r"\bLDL\b", "STL": r"\bSTL\b", "FTZ": r"\.FTZ\b", "INT_DIV": r"\bF2I\.FTZ\.U32\.TRUNC\.NTZ\b",
          "MUFU": r"\bMUFU\.\w+", "MUFU_RCP": r"\bMUFU\.RCP\b", "I2F_RP": r"\bI2F\.U(?:32|64)\.RP\b",
          "FP": r"\b(?:F(?:ADD|MUL|FMA|CMP|SET|SETP|MNMX|CHK|SWZADD)|F2F|F2FP|D(?:ADD|MUL|FMA)|H(?:ADD|MUL|FMA)2)\b"}
_INSTRUCTION = re.compile(r"\s*/\*[0-9a-f]{4,}\*/\s+\S")
_COMPILING = re.compile(r"Compiling entry function '(\w+)'")


def short(name: str) -> str:
    """A kernel's name as the source writes it: its own name, and its rows where it is a template's instance."""
    own = stem(name)
    return own + next((f"<{k}>" for k in (16, 64) if f"{own}ILi{k}E" in name), "")


def _atom(name: str, form: str, r: dict) -> bool:
    if short(name) not in SKETCHING or form != ATOM:
        return False
    r["IMMA"] += 1
    return True


def _line(r: dict, text: str) -> None:
    r["instructions"] += bool(_INSTRUCTION.match(text))


def _check(name: str, r: dict) -> list[str]:
    out, k = [], short(name)
    if "instructions" not in r:
        return [f"{k}: no listing"]
    if k in SKETCHING and (r["IMMA"] < 4 or r["IMMA"] % 4):
        out.append(f"{k}: {r['IMMA']} {ATOM} (want a multiple of 4, at least 4)")
    if r["FTZ"] != r["INT_DIV"] or r["MUFU"] != r["MUFU_RCP"] or r["MUFU"] != r["I2F_RP"] or r["FP"]:
        out.append(f"{k}: floating point other than integer division's: "
                   f"{({c: r[c] for c in ('FTZ', 'INT_DIV', 'MUFU', 'MUFU_RCP', 'I2F_RP', 'FP')})}")
    return out


def gate(ptxas: str, sass: str, arch: str = "sm_90a") -> tuple[dict, list[str]]:
    """The eight kernels' records by `short` name, and what the gate refuses."""
    listed = H.sass_functions(sass)
    mine = {n for n in [*listed, *_COMPILING.findall(ptxas)] if short(n) in WANT}
    kept, keep = [], False
    for line in ptxas.splitlines():
        m = _COMPILING.search(line)
        keep = m.group(1) in mine if m else keep
        kept += [line] if keep else []
    report = "\n".join(kept)
    regs, _ = H.ptxas_report(report, arch)
    local_ok = {n: LOCAL_OK.get(short(n), 0) for n in mine}
    local_ok.update({n: SPILL for n in mine if short(n) in CAPPED and (regs.get(n, {}).get("regs") or CAP + 1) <= CAP})
    funcs, problems = H.mma_gate(
        report, "\n".join(f"Function : {n}\n" + "\n".join(listed[n]) for n in sorted(mine) if n in listed), mine,
        arch=arch, counts=COUNTS, atom=_atom, line=_line, check=_check, local_ok=local_ok, unreported=True,
        record={**dict.fromkeys(COUNTS, 0), "IMMA": 0, "instructions": 0, "other_mma": []},
        allowed=lambda name: ATOM if short(name) in SKETCHING else "none")
    missing = sorted(WANT - {short(n) for n in mine})
    if missing:
        problems.append(f"kernels missing from the build: {missing}")
    return {short(n): {"symbol": n, "local_ok": local_ok[n], **r} for n, r in funcs.items()}, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ptxas", type=Path)
    ap.add_argument("sass", type=Path)
    ap.add_argument("out")
    ap.add_argument("--arch", default="sm_90a")
    ap.add_argument("--listings", type=Path, metavar="DIR")
    a = ap.parse_args(argv)
    sass = a.sass.read_text()
    funcs, problems = gate(a.ptxas.read_text(), sass, a.arch)
    if a.listings:
        a.listings.mkdir(parents=True, exist_ok=True)
        listed = H.sass_functions(sass)
        for k, r in funcs.items():
            (a.listings / (re.sub(r"\W+", "_", k).strip("_") + ".sass")).write_text(
                "\n".join(" ".join(line.split()) for line in listed.get(r["symbol"], [])) + "\n")
    return H.write_verdict(a.out, {"arch": a.arch, "kernels": funcs, "problems": problems}, problems, lambda: "; ".join(
        f"{k} {r['regs']} registers, {r['IMMA']} {ATOM}, "
        f"{max(r['stack'], r['spill_stores'], r['spill_loads'])} bytes local" for k, r in sorted(funcs.items())))


if __name__ == "__main__":
    sys.exit(main())
