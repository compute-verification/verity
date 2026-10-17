"""The timing gate: what a timed loop executes, checked on the kernel's SASS before it is timed.

`harness/sass_gate.py` is the numeric gate (no .FTZ, no bare FP32 MUFU). This gate checks what a timing measures, and it
reuses that gate's cubin reader, disassembly and instruction model rather than a copy of them. For each kernel `--kernels`
names (fnmatch on the mangled name, or on the name the mangling holds), it checks the following.

- **clock:** the kernel reads SR_CLOCKLO at least twice. The timed region runs from the first read to the last.
- **loops:** every loop (a backward branch) in the timed region is found. None may be nested in another or straddle a clock
  read. There are as many as `--expect` gives, and each matches one `--expect` exactly: OP=N says the loop body holds N
  instructions whose dotted opcode (`LDSM.16.M88.4`, without guard) matches the fnmatch pattern OP. Fewer means ptxas
  merged, hoisted or eliminated some; more means it duplicated them. So the body must be one iteration (`#pragma unroll 1`)
  for the count to be exact. An ELECT loop (`@P ELECT` ... `@P BRA.U.ANY` back to it: ptxas's lowering of a per-thread
  op with uniform operands, such as UBLKRED) is not a loop here: it runs once per active lane, which SASS doesn't show,
  so a probe issues such an op from one lane (side.cu's bulk roles), and the record lists it under `elect`.
- **forbid:** no instruction anywhere in the kernel has a `--forbid` opcode (default LDL,STL: spills).
- **sink:** the timed work is observable. An expected instruction writes memory (a reduction, an atomic, a store), or a
  store after the timed region stores a value an expected instruction computed. The second is traced on the kernel's
  reaching definitions.
- **smem:** with `--smem FILE`, the 32 lanes' byte offsets of every shared-memory access of one iteration, as
  `[{"op": "STSM.16.M88", "lanes": [32 ints or null], "sample": S}, ...]` (or `{kernel: [...]}`), where each `sample` is
  one (warp, iteration) the offsets were taken at. Each sample's opcodes must be exactly the timed loops' shared-memory
  instructions, and no access may conflict (`banks.degree` 1).

A refusal names the kernel, the check and the fix.

  python sass_gate.py BINARY --kernels K... [--expect OP=N[,OP=N...]]... [--forbid LDL,STL] [--smem FILE] [--numeric]
                         [--json OUT]
  # exit 0: every kernel passes; 8: one is refused (named, with its reasons); `--numeric` also runs the harness's gate
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict, deque
from fnmatch import fnmatchcase
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_GATE = HERE.parent / "harness" / "sass_gate.py"


def _load(name: str, path: Path):
    """A module of another directory, under its own name: two gates named sass_gate.py live in this tree."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("pouw_harness_sass_gate", HARNESS_GATE)
banks = _load("gpu_kit_banks", HERE / "banks.py")

VERSION = "gpu-kit-timing-gate/v1"
EXIT_FAIL = 8
FORBID = ("LDL", "STL")
CLOCK = re.compile(r"\bSR_CLOCKLO\b")
SHARED = frozenset({"LDS", "STS", "LDSM", "STSM", "ATOMS"})
#: instructions whose effect outlives the kernel or the iteration: their execution is the sink
WRITES = frozenset({"ST", "STG", "STS", "STSM", "RED", "REDG", "ATOM", "ATOMG", "ATOMS", "SUST", "SURED", "LDGSTS"})
WRITE_PREFIX = ("UBLK", "UTMASTG", "UTMARED", "REDAS")
#: a store after the timed region whose value can carry the loop's results out of the kernel
STORES = frozenset({"ST", "STG", "RED", "REDG", "ATOM", "ATOMG"})
_MANGLED = re.compile(r"_Z(\d+)")


class GateError(RuntimeError):
    pass


def parse_expect(spec: str) -> dict[str, int]:
    """``LDSM.16.M88=8,LOP3.*=8`` -> {pattern: count}: one loop's expected opcode counts."""
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        op, sep, n = part.rpartition("=")
        if not sep or not op or not n.isdigit():
            raise GateError(f"--expect {spec!r}: each entry is OP=N with N a count (got {part!r})")
        out[op] = int(n)
    return out


def base_name(mangled: str) -> str:
    """The name an Itanium mangling holds (``_Z4k_lmILi1EE...`` -> ``k_lm``); an unmangled name as it is."""
    m = _MANGLED.match(mangled)
    return mangled[m.end():m.end() + int(m.group(1))] if m else mangled


def is_write(op: list[str]) -> bool:
    return op[0] in WRITES or op[0].startswith(WRITE_PREFIX)


def sass_text(fn) -> str:
    return "\n".join(f"/*{ins.addr:04x}*/ {ins.text}" for ins in fn.insns)


def sass_sha256(fn) -> str:
    """The kernel's SASS, every instruction with its address: the hash a probe row carries."""
    return hashlib.sha256(sass_text(fn).encode()).hexdigest()


def matching(pats: dict[str, int], ops: Counter) -> dict[str, int]:
    return {p: sum(c for name, c in ops.items() if fnmatchcase(name, p)) for p in pats}


# ---- loops in the timed region -----------------------------------------------------------------------------------------


def clock_reads(fn) -> list[int]:
    return [k for k, ins in enumerate(fn.insns) if any(CLOCK.search(a) for a in ins.args)]


def loops(fn) -> list[tuple[int, int]]:
    """Every loop as (head, tail): a branch at `tail` back to the label at `head` <= tail; back-edges to one head merge."""
    tails: dict[int, int] = {}
    for k, ins in enumerate(fn.insns):
        if ins.op[0] != "BRA":
            continue
        for t in ins.targets:
            j = fn.labels.get(t)
            if j is not None and j <= k:
                tails[j] = max(tails.get(j, k), k)
    return sorted(tails.items())


def elect_loop(fn, j: int, k: int) -> bool:
    """ptxas's one-trip-a-lane loop around a uniform op: it opens with ELECT and closes with BRA.U.ANY."""
    return fn.insns[j].op[0] == "ELECT" and fn.insns[k].op[:3] == ["BRA", "U", "ANY"]


def _addr(fn, k: int) -> str:
    return f"0x{fn.insns[k].addr:04x}"


# ---- reaching definitions and the sink -------------------------------------------------------------------------------------


def reaching(fn):
    """Each instruction's reaching definitions, as a bitset over (instruction, register) definitions; None when a branch is
    indirect. A guarded write reaches without killing; an unguarded one kills the register's other definitions."""
    preds = fn._pred_map()
    if preds is None:
        return None
    n = len(fn.insns)
    defs: list[tuple[int, str]] = []
    by_reg: dict[str, int] = defaultdict(int)
    gen = [0] * n
    for k, ins in enumerate(fn.insns):
        for r in sorted(ins.maydef):
            bit = 1 << len(defs)
            defs.append((k, r))
            gen[k] |= bit
            by_reg[r] |= bit
    kill = [0] * n
    for k, ins in enumerate(fn.insns):
        for r in ins.kills:
            kill[k] |= by_reg[r]
    succ = [fn.succ(k) or [] for k in range(n)]
    inn, out = [0] * n, [0] * n
    work, queued = deque(range(n)), [True] * n
    while work:
        k = work.popleft()
        queued[k] = False
        i = 0
        for p in preds.get(k, ()):
            i |= out[p]
        inn[k] = i
        o = (i & ~kill[k]) | gen[k]
        if o != out[k]:
            out[k] = o
            for s in succ[k]:
                if s < n and not queued[s]:
                    work.append(s)
                    queued[s] = True
    return inn, defs, by_reg


def _same_guard_def(fn, j: int, r: str) -> int | None:
    """The write of `r` that a guarded instruction j reads, when one under the same guard precedes it in its basic block
    with the predicate unchanged between: both run or neither does (``@!P0 CS2R R6`` then ``@!P0 IADD.64 R2, R6``)."""
    g = fn.insns[j].guard
    if g is None:
        return None
    pred = g.lstrip("!")
    for i in range(j - 1, -1, -1):
        ins = fn.insns[i]
        if r in ins.maydef:
            return i if ins.guard == g else None
        if pred in ins.maydef or ins.op[0] in ("BRA", "EXIT", "RET", "CALL") or i in fn.label_at:
            return None
    return None


def feeds(fn, rd, k: int, regs, targets: set[int]) -> int | None:
    """An instruction of `targets` whose result reaches `regs` at instruction k, through any chain of definitions."""
    inn, defs, by_reg = rd
    seen, stack = 0, [(k, r) for r in regs]
    while stack:
        j, r = stack.pop()
        m = inn[j] & by_reg.get(r, 0) & ~seen
        if (d := _same_guard_def(fn, j, r)) is not None:
            m &= sum(1 << b for b, (kd, rr) in enumerate(defs) if kd == d and rr == r)
        seen |= m
        while m:
            low = m & -m
            m ^= low
            kd = defs[low.bit_length() - 1][0]
            if kd in targets:
                return kd
            stack.extend((kd, r2) for r2 in fn.insns[kd].reads)
    return None


def stored_regs(ins) -> set[str]:
    """The registers a store writes to memory: its operands outside brackets."""
    width = H._widths(ins.op)[1]
    return {r for a in ins.args for r in H._regs(a, width, brackets=False)}


# ---- one kernel --------------------------------------------------------------------------------------------------------


def judge_kernel(fn, expect: list[dict[str, int]], forbid=FORBID, smem: list[dict] | None = None) -> dict:
    """The gate's verdict on one kernel (`harness/sass_gate.Function`): its checks, and every reason it is refused."""
    insns, reasons = fn.insns, []
    rec: dict = {"kernel": fn.name, "sass_sha256": sass_sha256(fn), "instructions": len(insns)}

    bad = [(k, ins.text) for k, ins in enumerate(insns) if ins.op[0] in forbid]
    rec["forbidden"] = [f"{_addr(fn, k)} {t}" for k, t in bad[:H.MAX_LISTED]]
    if bad:
        reasons.append(f"forbid: {len(bad)} {'/'.join(sorted({insns[k].op[0] for k, _ in bad}))} (first at "
                       f"{_addr(fn, bad[0][0])}): the kernel spills; cut its live registers or its unroll")

    clocks = clock_reads(fn)
    rec["clock_reads"] = [_addr(fn, k) for k in clocks]
    if len(clocks) < 2:
        reasons.append(f"clock: {len(clocks)} SR_CLOCKLO read(s); bracket the timed loop with probe.cuh's OP_BEGIN and "
                       "OP_END (or bench.cuh's gk_clock()) so the timed region is known")
        rec.update(ok=False, reasons=reasons, loops=[])
        return rec
    lo, hi = clocks[0], clocks[-1]
    rec["timed"] = [_addr(fn, lo), _addr(fn, hi)]

    every = [(j, k) for j, k in loops(fn) if not elect_loop(fn, j, k)]
    rec["elect"] = [_addr(fn, j) for j, k in loops(fn) if elect_loop(fn, j, k) and lo < j and k < hi]
    timed = [(j, k) for j, k in every if lo < j and k < hi]
    straddle = [(j, k) for j, k in every if (j <= lo <= k or j <= hi <= k)]
    for j, k in straddle:
        reasons.append(f"loops: the loop at {_addr(fn, j)}-{_addr(fn, k)} contains a clock read; read the clock once "
                       "before the timed loop and once after it")
    nested = [(a, b) for a in timed for b in timed if a != b and a[0] <= b[0] and b[1] <= a[1]]
    for (j, k), (j2, k2) in nested:
        reasons.append(f"loops: the loop at {_addr(fn, j2)} is nested in the timed loop at {_addr(fn, j)}; unroll the inner "
                       "loop fully (#pragma unroll) so one timed loop holds one iteration's instructions")
    pats = sorted({p for e in expect for p in e})
    body = {jk: Counter(ins.name for ins in insns[jk[0]:jk[1] + 1]) for jk in timed}
    rec["loops"] = [{"head": _addr(fn, j), "tail": _addr(fn, k), "counts": matching(dict.fromkeys(pats, 0), body[j, k]),
                     "top": dict(body[j, k].most_common(8))} for j, k in timed]

    assigned: dict[int, tuple[int, int]] = {}
    if not timed:
        reasons.append("loops: no loop between the clock reads (eliminated, or unrolled into straight code); time a loop "
                       "with probe.cuh's OP_LOOP (#pragma unroll 1)")
    elif expect and len(timed) != len(expect):
        reasons.append(f"loops: {len(timed)} loops between the clock reads, {len(expect)} declared; write the timed loop "
                       "with #pragma unroll 1 (OP_LOOP), so ptxas adds no unrolled copy or remainder loop")
    if expect and len(timed) == len(expect):
        fits = {(i, jk): matching(e, body[jk]) == e for i, e in enumerate(expect) for jk in timed}
        best = next((p for p in permutations(timed) if all(fits[i, jk] for i, jk in enumerate(p))), None)
        if best is not None:
            assigned = dict(enumerate(best))
        else:
            for i, e in enumerate(expect):
                if any(fits[i, jk] for jk in timed):
                    continue
                jk = max(timed, key=lambda jk: sum(min(c, e[p]) for p, c in matching(e, body[jk]).items()))
                got = matching(e, body[jk])
                for p, n in e.items():
                    if got[p] != n:
                        what = ("ptxas merged, hoisted or eliminated some; give every instruction of every iteration "
                                "its own operands (the lmmix ring of addresses, a dependent chain)") if got[p] < n else \
                               "ptxas duplicated them; time one iteration a loop (#pragma unroll 1)"
                        held = ", ".join(f"{c} {o}" for o, c in body[jk].most_common(6))
                        reasons.append(f"loops: {p}: {got[p]} in the loop at {_addr(fn, jk[0])}, {n} expected: {what} "
                                       f"(the loop holds {held})")
    elif not expect and timed:
        assigned = dict(enumerate(timed))

    # the sink: an expected instruction writes memory, or its result reaches a store after the timed region
    target = {k for i, (j, t) in assigned.items() for k in range(j, t + 1)
              if not expect or any(fnmatchcase(insns[k].name, p) for p in expect[i])}
    sink: dict = {}
    side = sorted(k for k in target if is_write(insns[k].op))
    if side:
        sink = {"by": "side effect", "at": _addr(fn, side[0]), "instruction": insns[side[0]].text}
    elif target:
        rd = reaching(fn)
        if rd is None:
            reasons.append("sink: an indirect branch hides the kernel's data flow; the gate can't show the sink is live")
        else:
            for k in range(hi, len(insns)):
                if insns[k].op[0] in STORES:
                    src = feeds(fn, rd, k, stored_regs(insns[k]), target)
                    if src is not None:
                        sink = {"by": "store", "at": _addr(fn, k), "instruction": insns[k].text, "from": _addr(fn, src)}
                        break
            if not sink:
                reasons.append("sink: no store after the timed region stores a value the timed instructions computed, "
                               "so ptxas may delete them; XOR every result into the sink (probe.cuh's OP_SINK)")
    rec["sink"] = sink

    if smem is not None:
        rec["smem"] = []
        loop_ops = Counter(insns[k].name for k in range(len(insns)) if insns[k].op[0] in SHARED and any(
            j <= k <= t for j, t in assigned.values()))
        samples: dict[int, list[dict]] = defaultdict(list)
        for a in smem:
            samples[a.get("sample", 0)].append(a)
        conflicts: dict[str, list[tuple[int, int, int, list[int]]]] = defaultdict(list)
        for s, accs in sorted(samples.items()):
            given = Counter(a["op"] for a in accs)
            if assigned and given != loop_ops:
                reasons.append(f"smem: sample {s} gives offsets for {dict(given)}, the timed loop's shared-memory "
                               f"instructions are {dict(loop_ops)}; give every access's 32 lane offsets "
                               "(probe.cuh's OpProbe::smem_ops and smem)")
            for i, a in enumerate(accs):
                d, ph = banks.worst(a["op"], a["lanes"])
                rec["smem"].append({"sample": s, "slot": i, "op": a["op"], "degree": d})
                if d > 1:
                    conflicts[a["op"]].append((d, s, i, ph))
        for op, cs in conflicts.items():
            d, s, i, ph = max(cs, key=lambda c: c[0])
            n = sum(1 for a in smem if a["op"] == op)
            reasons.append(f"smem: {op}: {len(cs)} of {n} accesses conflict, up to {d}-way (sample {s}, slot {i}: a "
                           f"phase at byte offsets {ph}); put each phase's rows (16 B) or lanes on distinct banks")
    rec.update(ok=not reasons, reasons=reasons)
    return rec


# ---- a binary ----------------------------------------------------------------------------------------------------------


def functions(path: Path, tools: dict | None = None) -> list:
    """Every .text section of every cubin in `path` (a cubin, a fatbin, or a host ELF), as harness `Function`s."""
    tools = tools or H.find_tools()
    cubins, _ = H.device_code(path.read_bytes(), str(path))
    return [H.Function(f) for funcs, _notes in H._disassemble(cubins, tools) for f in funcs]


def select(fns: list, patterns: list[str]) -> tuple[dict[str, list], list[str]]:
    picked, missing = {}, []
    for p in patterns:
        got = [f for f in fns if fnmatchcase(f.name, p) or fnmatchcase(base_name(f.name), p)]
        (picked.setdefault(p, []).extend(got)) if got else missing.append(p)
    return picked, missing


def gate(path: Path, kernels: list[str], expect: list[dict[str, int]], forbid=FORBID, smem=None, numeric: bool = False,
         tools: dict | None = None, fns: list | None = None) -> dict:
    fns = functions(path, tools) if fns is None else fns
    picked, missing = select(fns, kernels)
    recs = []
    for p, fs in picked.items():
        for f in fs:
            s = smem.get(f.name, smem.get(p)) if isinstance(smem, dict) else smem
            recs.append(judge_kernel(f, expect, forbid, s))
    out = {"version": VERSION, "binary": str(path), "kernels": recs, "missing": missing,
           "ok": not missing and bool(recs) and all(r["ok"] for r in recs)}
    if numeric:
        out["numeric"] = H.scan(path, tools or H.find_tools(), cache=None)
        out["ok"] = out["ok"] and out["numeric"]["ok"]
    return out


def refusals(verdict: dict) -> list[str]:
    out = [f"{m}: no such kernel in {verdict['binary']}" for m in verdict["missing"]]
    out += [f"{r['kernel']}: {why}" for r in verdict["kernels"] for why in r["reasons"]]
    if "numeric" in verdict and not verdict["numeric"]["ok"]:
        out.append("numeric: " + H.why_failed(verdict["numeric"]))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("binary", type=Path)
    p.add_argument("--kernels", nargs="+", required=True, help="fnmatch patterns on the kernels' (mangled or base) names")
    p.add_argument("--expect", action="append", default=[], help="one timed loop's OP=N[,OP=N...]; repeat for each loop")
    p.add_argument("--forbid", default=",".join(FORBID), help="opcodes refused anywhere in the kernel (default %(default)s)")
    p.add_argument("--smem", type=Path, help="JSON: each shared-memory access's op and 32 lane byte offsets")
    p.add_argument("--numeric", action="store_true", help="also run harness/sass_gate.py's numeric gate on the binary")
    p.add_argument("--json", type=Path, help="write the verdict here")
    a = p.parse_args(argv)
    try:
        expect = [parse_expect(e) for e in a.expect]
        smem = json.loads(a.smem.read_text()) if a.smem else None
        v = gate(a.binary, a.kernels, expect, tuple(filter(None, a.forbid.split(","))), smem, a.numeric)
    except (GateError, H.SassGateFailed) as e:
        print(f"sass_gate: {e}", file=sys.stderr)
        return EXIT_FAIL
    if a.json:
        a.json.write_text(json.dumps(v, indent=1) + "\n")
    for r in v["kernels"]:
        print(f"{'PASS' if r['ok'] else 'REFUSED'} {r['kernel']} sass {r['sass_sha256'][:16]}")
    for why in refusals(v):
        print(f"  {why}")
    return 0 if v["ok"] else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
