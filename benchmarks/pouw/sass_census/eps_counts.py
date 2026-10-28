"""ε_skip's per-instruction counts from SASS, on any architecture the kernels build for (INT-HB row 20).

    python eps_counts.py OUT --src TAG=APV_CU [--src ...] [--json FILE]

eps_budget.py prices a peel design's skippable work with the checked kernels' and the D2S kernels' exact ncu counts on
sm_120: APV_TAIL (apv_l7qs's tail stage, warp instructions a 128-cell tile by class: IMMA 4, IMAD 4.0, the rest 11.5,
which ncu read as apv_l7qs − apv_cc), c_apv_serve (the served f over a plain epilogue, 4.4 a tile) and c_vfold (vfold −
vfold_x, 0.6884 / 0.7265 / 0.7221 warp instructions an entry at 2K / 8K / 32K). Off sm_120 it reads the same numbers.
This counts the same differences from each architecture's SASS (eps_counts.sh's `nvdisasm -c -gi` dumps): every
instruction a converged warp can reach (`reach`: `BRA.DIV`'s out-of-line paths never run) is placed by the source lines
its inline chain passes through, and each region is weighted by how often a warp runs it at the shape ncu measured
(Llama-3.1-8B, 32 query and 8 KV heads):

- the tail: apv_l7qs's instructions inlined at apv_body's atail call, less apv_cc's at the `cabs_all` it runs instead
  (the fold both kernels do, which ncu's difference cancels), once a warp-tile (8 tiles of 128 cells), without atail's
  `y2s` block (s <= 4096 skips it); `issue`'s (the TMA refill, thread 0's) at 1/16, its warp of the CTA's 16;
- the served epilogue: apv_body's `OUT != 2` block, apv_l7qs's less apv_plain's, a warp-tile;
- the sums: vfold<1, D3> less vfold<0, 0> over a warp's s / 2 entries (a CTA's 8 warps take 4 channels x s keys): the
  key-chunk loop's body s / 1024 times, its `threadIdx.x < 64` block on 2 of the 8 warps, the digest loop on the warps
  whose threads hash (min(nC, 256) of them, nC = s / 128) and everything else once. D3 is on past 4096 tokens.

A static count runs a spin-wait once, and a branch any lane takes counts for every warp, so sm_120a's numbers here are
the check on the rule against ncu's (`RECORDED`: tail 20.36 against 20.26, serve 4.50 against 4.4, the sums 2–5% high);
the other architectures are read against sm_120a's from the same tree, by the same rule (`against`).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: eps_budget.py's sm_120 values, from the checked kernels' and the D2S kernels' runs
RECORDED = {"tail": {"IMMA": 4.0, "IMAD": 4.0, "ALU": 11.5}, "serve": 4.4,
            "vfold": {2048: 0.6884, 8192: 0.7265, 32768: 0.7221}}
SHAPES = (2048, 8192, 32768)
TILES_A_WARP = 8      # a warp's 32 x 32 cells: MI 2 x NI 4 m16n8 tiles
WARPS_A_CTA = 16      # apv's 512 threads; thread 0 issues the TMA refills
VF_WARPS = 8          # vfold's 256 threads
KERNELS = {"apv_l7qs": r"_Z8apv_l7qs6ApArgs$", "apv_cc": r"_Z6apv_cc6ApArgs$", "apv_plain": r"_Z9apv_plain6ApArgs$",
           "vfold_d3": r"_Z5vfoldILb1ELb1EE", "vfold": r"_Z5vfoldILb1ELb0EE", "vfold_x": r"_Z5vfoldILb0ELb0EE"}

_FUNC = re.compile(r"^\s*\.text\.(\S+?):\s*$")
_LABEL = re.compile(r"^\s*\.(L_x_\d+):")
_LOC = re.compile(r'"([^"]+)", line (\d+)')
_INSTR = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
_TARGET = re.compile(r"`\(\.(L_x_\d+)\)")


@dataclass(frozen=True)
class Instr:
    addr: int
    op: str
    frames: tuple[tuple[str, int], ...]   # (file's basename, line), innermost first
    live: bool = True                     # reached by a converged warp (`reach`)


def parse(text: str, keep: str | None = None) -> dict[str, list[Instr]]:
    """nvdisasm -c -gi's SASS a function (those whose name matches `keep`): each instruction with its inline chain.
    nvdisasm writes the chain as the run of `//## File` lines above the instruction, one a frame ("line N inlined at
    ..."), and an instruction with none above it has the one before it's."""
    funcs: dict[str, list] = {}
    cur, frames, run, labels = None, (), None, []
    rx = re.compile(keep) if keep else None
    for line in text.splitlines():
        if m := _FUNC.match(line):
            cur, frames, run, labels = m.group(1), (), None, []
            if rx and not rx.search(cur):
                cur = None
            else:
                funcs.setdefault(cur, [])
        elif cur is None:
            continue
        elif "//##" in line:
            run = (run or ()) + tuple((Path(f).name, int(n)) for f, n in _LOC.findall(line))
        elif cur is not None and (m := _LABEL.match(line)):
            labels.append(m.group(1))
        elif cur is not None and (m := _INSTR.search(line)):
            if run is not None:
                frames, run = tuple(dict.fromkeys(run)), None
            body = m.group(2).strip()
            pred = body.startswith("@")
            if pred:
                body = body.split(None, 1)[1] if " " in body else ""
            op = body.split(None, 1)[0] if body else ""
            if op:
                t = _TARGET.search(body)
                funcs[cur].append((Instr(int(m.group(1), 16), op, frames), pred, t.group(1) if t else None, labels))
                labels = []
    return {f: reach(v) for f, v in funcs.items()}


def reach(fn: list) -> list[Instr]:
    """Marks what a converged warp can run: from the entry, through fall-through and branch targets, with `BRA.DIV` (taken
    only by a diverged warp, into the out-of-line shuffle and vote paths ptxas puts after the EXIT) never taken. An
    indirect branch leaves everything live."""
    at = {lab: k for k, (_, _, _, labs) in enumerate(fn) for lab in labs}
    if any(base(i.op) in ("BRX", "JMX") for i, *_ in fn):
        return [i for i, *_ in fn]
    seen, todo = set(), [0] if fn else []
    while todo:
        k = todo.pop()
        if k in seen or k >= len(fn):
            continue
        seen.add(k)
        i, pred, target, _ = fn[k]
        b, div = base(i.op), ".DIV" in i.op
        if b in ("BRA", "CALL") and target is not None and not div:
            todo.append(at[target])
        if pred or div or b not in ("BRA", "EXIT", "RET"):
            todo.append(k + 1)
    return [Instr(i.addr, i.op, i.frames, k in seen) for k, (i, *_) in enumerate(fn)]


def block(lines: list[str], pattern: str, start: int = 0) -> tuple[int, int]:
    """1-based (first, last) lines of the brace block that opens on the first line from `start` matching `pattern`."""
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            depth, opened = 0, False
            for j in range(i, len(lines)):
                text = re.sub(r"//.*", "", lines[j])
                if j == i:
                    text = text[rx.search(text).end():]
                for ch in text:
                    if ch == "{":
                        depth, opened = depth + 1, True
                    elif ch == "}":
                        depth -= 1
                        if opened and depth == 0:
                            return i + 1, j + 1
            break
    raise ValueError(f"no block for {pattern!r}")


def line(lines: list[str], pattern: str) -> tuple[int, int]:
    """The one line matching `pattern`, as a range."""
    hits = [k + 1 for k, x in enumerate(lines) if re.search(pattern, x)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} lines match {pattern!r}")
    return hits[0], hits[0]


def regions(src: str) -> dict[str, tuple[int, int]]:
    """apv.cu's line ranges the counts read; `y2s` and the vfold ones only where the tree has them (the checked kernels
    and the D2S kernels added them). An empty range matches nothing. `tail_call` is apv_body's call of atail (OUT 0, 1)
    and `cc_fold` the `cabs_all` apv_cc runs in its place (OUT 3): every instruction inlined there has the line in its
    chain."""
    ln = src.splitlines()
    none = (0, -1)
    atail = block(ln, r"DEV void atail\(")
    out = {"atail": atail, "issue": block(ln, r"DEV void issue\(\)"),
           "tail_call": line(ln, r"ml\.template atail<OUT>\("), "cc_fold": line(ln, r"else cabs_all<L>\(ch, acc\);"),
           "epilogue": block(ln, r"else if constexpr \(OUT != 2\)")}
    y2s = block(ln, r"if \(y2s\) \{", atail[0] - 1) if any("if (y2s) {" in x for x in ln) else none
    out["y2s"] = y2s if atail[0] <= y2s[0] <= atail[1] else none
    if any("c0 < nC; c0 += 8" in x for x in ln):
        loop = block(ln, r"for \(int c0 = 0; c0 < nC; c0 \+= 8\)")
        out.update(vf_loop=loop, vf_tid64=block(ln, r"if \(threadIdx\.x < 64\)", loop[0] - 1),
                   vf_hash=block(ln, r"for \(int i = threadIdx\.x; i < nC; i \+= VF_NT\)"))
    else:
        out.update(vf_loop=none, vf_tid64=none, vf_hash=none)
    return out


def within(ins: Instr, r: tuple[int, int], file: str = "apv.cu") -> bool:
    return any(f == file and r[0] <= n <= r[1] for f, n in ins.frames)


def base(op: str) -> str:
    return op.split(".", 1)[0]


def collapse(c: Counter) -> dict[str, float]:
    """eps_budget's APV_TAIL classes: IMMA, IMAD and the rest (priced as ALU's)."""
    out = {"IMMA": 0.0, "IMAD": 0.0, "ALU": 0.0}
    for op, n in c.items():
        out[base(op) if base(op) in ("IMMA", "IMAD") else "ALU"] += n
    return out


def at(fn: list[Instr], R: dict, where: str, skip: str | None = None) -> Counter:
    """A warp-tile's instructions inlined at region `where` (less `skip`), a tile; `issue`'s at 1/16."""
    c: Counter = Counter()
    for i in fn:
        if i.live and within(i, R[where]) and not (skip and within(i, R[skip])):
            c[base(i.op)] += (1 / WARPS_A_CTA if within(i, R["issue"]) else 1.0) / TILES_A_WARP
    return c


def tail(l7qs: list[Instr], cc: list[Instr], R: dict) -> Counter:
    """apv_l7qs − apv_cc a tile, as ncu read it: the tail stage less the fold apv_cc runs instead of it."""
    return diff(at(l7qs, R, "tail_call", "y2s"), at(cc, R, "cc_fold"))


def epilogue(fn: list[Instr], R: dict) -> Counter:
    return at(fn, R, "epilogue")


def vf_weight(i: Instr, R: dict, s: int) -> float:
    """How many times an average warp of vfold's CTA runs instruction `i` at s tokens."""
    nC = s // 128
    if within(i, R["vf_loop"]):
        return (s // 1024) * (2 / VF_WARPS if within(i, R["vf_tid64"]) else 1.0)
    if within(i, R["vf_hash"]):
        return min(-(-min(nC, 256) // 32), VF_WARPS) / VF_WARPS
    return 1.0


def vf_count(fn: list[Instr], R: dict, s: int) -> Counter:
    c: Counter = Counter()
    for i in fn:
        if i.live:
            c[base(i.op)] += vf_weight(i, R, s)
    return c


def diff(a: Counter, b: Counter, scale: float = 1.0) -> Counter:
    out: Counter = Counter()
    for k in set(a) | set(b):
        v = (a.get(k, 0.0) - b.get(k, 0.0)) * scale
        if abs(v) > 1e-12:
            out[k] = v
    return out


def kernels(funcs: dict[str, list[Instr]]) -> dict[str, list[Instr]]:
    out = {}
    for name, rx in KERNELS.items():
        hits = [f for f in funcs if re.search(rx, f)]
        if len(hits) > 1:
            raise ValueError(f"{name}: {len(hits)} functions match {rx}: {hits}")
        if hits:
            out[name] = funcs[hits[0]]
    return out


def ptxas(log: str) -> dict[str, dict]:
    """ptxas -v's registers and spill bytes a kernel."""
    out, cur = {}, None
    for line in log.splitlines():
        if m := re.search(r"Compiling entry function '(\S+)'", line):
            cur = m.group(1)
        elif cur and (m := re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)):
            out.setdefault(cur, {}).update(spill_stores=int(m.group(1)), spill_loads=int(m.group(2)))
        elif cur and (m := re.search(r"Used (\d+) registers", line)):
            out.setdefault(cur, {})["registers"] = int(m.group(1))
    return out


def counts(sass: str, src: str, log: str = "") -> dict:
    """One build's counts: the tail and served epilogue a tile, the sums an entry at each shape, with their parts."""
    R, ks = regions(src), kernels(parse(sass, "|".join(KERNELS.values())))
    out: dict = {"regions": R, "kernels": {k: {"instructions": len(v), "live": sum(i.live for i in v)}
                                           for k, v in ks.items()}}
    if "apv_l7qs" in ks:
        if "apv_cc" in ks:
            t = tail(ks["apv_l7qs"], ks["apv_cc"], R)
            out["tail"] = {"by_opcode": dict(t), "classes": collapse(t), "total": sum(t.values()),
                           "stage": sum(at(ks["apv_l7qs"], R, "tail_call", "y2s").values()),
                           "cc_fold": sum(at(ks["apv_cc"], R, "cc_fold").values())}
        if "apv_plain" in ks:
            e, p = epilogue(ks["apv_l7qs"], R), epilogue(ks["apv_plain"], R)
            sv = diff(e, p)
            out["serve"] = {"by_opcode": dict(sv), "total": sum(sv.values()), "epilogue": sum(e.values()),
                            "plain_epilogue": sum(p.values())}
            if "apv_cc" in ks:
                out["serve"]["cc_epilogue"] = sum(epilogue(ks["apv_cc"], R).values())
    if "vfold_x" in ks:
        out["vfold"] = {}
        for s in SHAPES:
            k = "vfold_d3" if s > 4096 else "vfold"
            if k in ks:
                d = diff(vf_count(ks[k], R, s), vf_count(ks["vfold_x"], R, s), 2 / s)
                out["vfold"][s] = {"by_opcode": dict(d), "total": sum(d.values())}
    regs = ptxas(log)
    out["ptxas"] = {k: v for f, v in regs.items() for k, rx in KERNELS.items() if re.search(rx, f)}
    return out


def table(res: dict) -> str:
    rows = ["| source | arch | tail IMMA | tail IMAD | tail other | tail total | serve | vfold 2K | vfold 8K | vfold 32K |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    r = RECORDED
    rows.append(f"| ncu (eps_budget) | sm_120 | {r['tail']['IMMA']:.2f} | {r['tail']['IMAD']:.2f} | {r['tail']['ALU']:.2f} "
                f"| {sum(r['tail'].values()):.2f} | {r['serve']:.2f} | "
                + " | ".join(f"{r['vfold'][s]:.4f}" for s in SHAPES) + " |")
    for tag, by in res.items():
        for arch, c in by.items():
            t = c.get("tail", {}).get("classes", {})
            vf = c.get("vfold", {})
            cell = lambda v, f=".2f": format(v, f) if v is not None else "–"  # noqa: E731
            rows.append(f"| {tag} | {arch} | {cell(t.get('IMMA'))} | {cell(t.get('IMAD'))} | {cell(t.get('ALU'))} | "
                        f"{cell(c.get('tail', {}).get('total'))} | {cell(c.get('serve', {}).get('total'))} | "
                        + " | ".join(cell(vf.get(s, {}).get("total"), ".4f") for s in SHAPES) + " |")
    return "\n".join(rows)


def flat(c: dict) -> dict[str, float]:
    """eps_budget's numbers out of one build's counts, by name."""
    out = {f"tail.{k}": v for k, v in c.get("tail", {}).get("classes", {}).items()}
    if "tail" in c:
        out["tail.total"] = c["tail"]["total"]
    if "serve" in c:
        out["serve"] = c["serve"]["total"]
    out.update({f"vfold.{s}": v["total"] for s, v in c.get("vfold", {}).items()})
    return out


def against(res: dict, ref: str = "sm_120a") -> dict:
    """Each architecture against `ref` from the same tree, by the same rule: the difference, and sm_120's ncu count plus
    it (what eps_budget would read there)."""
    rec = {f"tail.{k}": v for k, v in RECORDED["tail"].items()}
    rec.update({"tail.total": sum(RECORDED["tail"].values()), "serve": RECORDED["serve"]})
    rec.update({f"vfold.{s}": v for s, v in RECORDED["vfold"].items()})
    out: dict = {}
    for tag, by in res.items():
        if ref not in by:
            continue
        r = flat(by[ref])
        for arch, c in by.items():
            out.setdefault(tag, {})[arch] = {k: {"static": v, "delta": v - r[k], "ratio": v / r[k] if r[k] else None,
                                                 "estimate": rec[k] + v - r[k]} for k, v in flat(c).items() if k in r}
    return out


def delta_table(d: dict) -> str:
    keys = ["tail.IMAD", "tail.ALU", "tail.total", "serve"] + [f"vfold.{s}" for s in SHAPES]
    rows = ["| source | arch | " + " | ".join(keys) + " |", "|---|---|" + "---:|" * len(keys)]
    for tag, by in d.items():
        for arch, c in by.items():
            rows.append(f"| {tag} | {arch} | " + " | ".join(
                f"{c[k]['delta']:+.3f} ({c[k]['estimate']:.3f})" if k in c else "–" for k in keys) + " |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out", type=Path, help="eps_counts.sh's OUT (TAG/ARCH/apv.gi.sass.gz)")
    p.add_argument("--src", action="append", required=True, help="TAG=path to that tree's apv.cu")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    res: dict = {}
    for spec in a.src:
        tag, path = spec.split("=", 1)
        src = Path(path).read_text()
        for d in sorted((a.out / tag).iterdir()):
            f = d / "apv.gi.sass.gz"
            if f.exists():
                log = (d / "build.log").read_text() if (d / "build.log").exists() else ""
                res.setdefault(tag, {})[d.name] = counts(gzip.decompress(f.read_bytes()).decode(), src, log)
    d = against(res)
    print(table(res) + "\n\nAgainst sm_120a, same tree and rule: the difference (sm_120's ncu count plus it)\n\n"
          + delta_table(d))
    if a.json:
        a.json.write_text(json.dumps({"recorded_sm120": RECORDED, "counts": res, "against_sm120a": d}, indent=1,
                                     default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
