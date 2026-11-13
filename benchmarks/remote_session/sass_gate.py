"""rows.sh's gate on the kernels a 64-row pass adds to `session.cu`, and their record: ptxas's report and cuobjdump's
SASS of those five, through the harness's `mma_gate` (verity/protocols/accounting/pouw/prover/sass_gate.py). Every
other kernel of the build is held to the base's, instruction for instruction, by `same_sass.py`. Refuses (exit 1, the
reasons on stderr):

- one of the five missing from the build or from the listing, compiled for another arch, an arch this gate has no
  table for (`ARCHS`: sm_90a and sm_120a, each as read from its build), or with a ptxas warning;
- a sketching kernel (`SKETCHING`, the three that compute σ) whose tensor-core instructions aren't the int8 form alone,
  IMMA.16832.S8.S8 (mma.m16n8k32 on s8 operands into s32), in a multiple of 4 (64 rows are four of its 16), and any
  tensor-core instruction in the other two;
- local memory (a stack frame, a spill, LDL or STL) beyond the arch's bytes for the kernel (`local`): none in the
  pass's receipt grid and the opening's receipts; the consumers' and the two once-a-pass kernels' frames as they
  are. A build that caps the receipt kernels at 128 registers (-DSK64_PER_SM=2) may add the
  arch's `capped` bytes to a kernel it holds to that, which is what that build gives up;
- floating point: any instruction that reads or writes a float outside the compiler's unsigned division by a
  runtime divisor, whole (`DIV`: its forms by the divisor's width, each a reciprocal's seed and its truncation), and
  the constants sm_120a loads through HFMA2 (`-RZ, RZ` and two immediates). A division's operands are counts and
  positions (a tile's group, a subtree's chunk), never a block's byte.

OUT_JSON holds each kernel's record: its registers, stack frame and spills, its instructions, its tensor-core
instructions, its float instructions by mnemonic (`float`) and the divisions they make (`divisions`). With
--listings, DIR gets each kernel's listing, a line an entry as `same_sass.py` reads it.

  python3 sass_gate.py PTXAS_TXT SASS_TXT OUT_JSON [--arch sm_90a] [--listings DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from collections import Counter
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
            name, Path(__file__).resolve().parents[2] / "verity/protocols/accounting/pouw/prover/sass_gate.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


H = _harness()
ATOM = "IMMA.16832.S8.S8"
#: the kernels that compute σ at 64 rows: the pass's receipt grid, its consumers (a consumer's own tile) and an
#: opening's receipts
SKETCHING = {"sketch64_pass_kernel", "pass_consumer_kernel<64>", "subtree64_receipts_kernel"}
#: a pass's packed signs from its seed, and a chained pass's seed and signs: one launch a pass
ONCE = {"signs_kernel<64>", "chain_prepare_kernel<64>"}
WANT = SKETCHING | ONCE
#: by arch, as CUDA 12.9's ptxas builds them: `local`, the bytes of stack frame or spill a kernel holds (a kernel it
#: doesn't name, none): the consumers run at the register limit, as their 16-row instances do, and the once-a-pass
#: kernels keep a SHA-512 schedule on the stack, as theirs do; and `capped`, what a kernel held to CAP registers (two
#: receipt CTAs an SM) may hold beside
ARCHS = {
    "sm_90a": {"local": {"pass_consumer_kernel<64>": 224, "signs_kernel<64>": 256, "chain_prepare_kernel<64>": 304},
               "capped": {}},
    "sm_120a": {"local": {"pass_consumer_kernel<64>": 160, "signs_kernel<64>": 368, "chain_prepare_kernel<64>": 352},
                "capped": {"sketch64_pass_kernel": 32}},
}
CAP = 128
COUNTS = {"LDL": r"\bLDL\b", "STL": r"\bSTL\b"}
#: the float instructions of the compiler's unsigned division by a runtime divisor, by the divisor's width; a form's
#: truncation (`DIV_OF`) is in no other. A uniform register's instruction (UI2F, UFMUL) counts as its plain one.
DIV = {"u32": {"I2F.U32.RP": 1, "MUFU.RCP": 1, "F2I.FTZ.U32.TRUNC.NTZ": 1},
       "u64": {"I2F.U64.RP": 1, "MUFU.RCP": 1, "F2I.U64.TRUNC": 1},
       "u16": {"I2F.U16": 1, "I2F.U16.RZ": 1, "MUFU.RCP": 1, "FMUL": 2, "FMUL.RZ": 1, "FSEL": 2, "FSETP.GEU.AND": 1,
               "FSETP.GT.AND": 1, "F2I.U32.TRUNC.NTZ": 1}}
DIV_OF = {"u32": "F2I.FTZ.U32.TRUNC.NTZ", "u64": "F2I.U64.TRUNC", "u16": "F2I.U32.TRUNC.NTZ"}
_OP = re.compile(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][\w.]*)\s*([^;]*?)\s*;")
_FLOAT = re.compile(r"U?(?:I2FP?|F2[IF]P?|MUFU|F(?:ADD|MUL|FMA|CMP|SETP?|SEL|MNMX|CHK|SWZADD|RND)|"
                    r"D(?:ADD|MUL|FMA|SETP|MNMX)|H(?:ADD|MUL|FMA|SETP?|MNMX)2)(?:\.|$)")
_CONSTANT = re.compile(r"HFMA2 U?R\d+, -RZ, RZ, [-+.\de]+, [-+.\de]+$")
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
    m = _OP.match(text)
    if not m:
        return
    r["instructions"] += 1
    op, operands = m.groups()
    if _FLOAT.match(op):
        constant = bool(_CONSTANT.match(f"{op} {operands}"))
        r["constants"] += constant
        if not constant:
            r["float"][op] = r["float"].get(op, 0) + 1


def divisions(census: dict) -> dict | None:
    """The whole divisions (`DIV`) a kernel's float instructions make, by form; None if they are anything else."""
    have = Counter()
    for op, n in census.items():
        have[op.removeprefix("U")] += n
    made = {form: have[DIV_OF[form]] for form in DIV}
    want = Counter()
    for form, n in made.items():
        for op, each in DIV[form].items():
            want[op] += n * each
    return made if +have == +want else None


def _check(name: str, r: dict) -> list[str]:
    out, k = [], short(name)
    if "instructions" not in r:
        return [f"{k}: no listing"]
    if k in SKETCHING and (r["IMMA"] < 4 or r["IMMA"] % 4):
        out.append(f"{k}: {r['IMMA']} {ATOM} (want a multiple of 4, at least 4)")
    r["divisions"] = divisions(r["float"])
    if r["divisions"] is None:
        out.append(f"{k}: floating point other than integer division's: {r['float']}")
    return out


def gate(ptxas: str, sass: str, arch: str = "sm_90a") -> tuple[dict, list[str]]:
    """The five kernels' records by `short` name, and what the gate refuses."""
    listed = H.sass_functions(sass)
    mine = {n for n in [*listed, *_COMPILING.findall(ptxas)] if short(n) in WANT}
    kept, keep = [], False
    for line in ptxas.splitlines():
        m = _COMPILING.search(line)
        keep = m.group(1) in mine if m else keep
        kept += [line] if keep else []
    report = "\n".join(kept)
    table = ARCHS.get(arch)
    if table is None:
        return {}, [f"no table for {arch}: read its build's listings (--listings) and add one to ARCHS ({sorted(ARCHS)})"]
    regs, _ = H.ptxas_report(report, arch)
    local_ok = {n: table["local"].get(short(n), 0) for n in mine}
    for n in mine:
        if short(n) in table["capped"] and (regs.get(n, {}).get("regs") or CAP + 1) <= CAP:
            local_ok[n] = table["capped"][short(n)]
    funcs, problems = H.mma_gate(
        report, "\n".join(f"Function : {n}\n" + "\n".join(listed[n]) for n in sorted(mine) if n in listed), mine,
        arch=arch, counts=COUNTS, atom=_atom, line=_line, check=_check, local_ok=local_ok, unreported=True,
        record={**dict.fromkeys(COUNTS, 0), "IMMA": 0, "instructions": 0, "constants": 0, "float": {}, "other_mma": []},
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
