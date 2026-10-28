"""`flock-rows`, C-Flock's Lean rows, against a unit's size: the flat MoE classes (no calls, no table reads; 41k to 2M
ANDs) and the derive vectors' fan-out unit at M placed calls and M placed reads (`test_derive.fanout_case`).

    python benchmarks/verifier_cost/flock_rows.py --out DIR --bin LABEL=PATH [--bin LABEL=PATH ...] [--target NAME ...]
        [--fanout M,..] [--modes rows,unit,delta,logical,check] [--cap SECONDS]
    python benchmarks/verifier_cost/flock_rows.py --table FLOCK_ROWS_JSON     # its before/after table, as Markdown

Each target's classes (`partition_units` under Q_word, `lower_class`) are written as a flat circuit type and as a one-layout
archive, and every binary runs every mode on them: `rows` is the default path (`CircuitType.check`, then `Flock.Derive.derive`,
as `flock-rows TYPES --root D --name N` prints them), the others the archive's parts (`--archive FILE --part P`: the
verifier's `DeriveAll.deriveChecked`). Each output is held to the Python mirror: `ir_lower.netlist` for `rows`, the SHA-512
of `verity_flock.derive`'s unit rows, Δ and logical list for the parts, and `ok` for `check`. `flock_rows.json`
(`verity/verifier-cost-flock-rows/v1`) has one row per unit, binary and mode: its ANDs, input ports and bits, wall and CPU
seconds, peak RSS, exit code and whether the output equals the mirror's (null past `--cap`). `flock_rows.sh` builds the binaries
before and after a change and runs this on a node.

The MoE targets need circuits' static MoE forms (`verity.ml.circuits.ops.moe`) and C-Flock's MoE select pieces where their
definitions do; each target imports only its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.setrecursionlimit(100000)

SCHEMA = "verity/verifier-cost-flock-rows/v1"
MODES = ("rows", "unit", "delta", "logical", "check")


def _qwen3(name: str, **kw):
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops import prims as P

    if name.startswith("registry"):
        from verity.ml.definitions.registry import moe as VM

        d = VM.MoeExpertCoordinateWV2 if name.endswith("-w") else VM.MoeExpertCoordinateV2
    else:
        from verity.ml.circuits.ops import moe as M

        d = M.MoePairCoordinateW if name.endswith("-w") else M.MoePairCoordinate
    return bind(d, E=128, DOT=P.HopperBF16WgmmaDot16, **kw)


def _glm(form: str, **kw):
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.prims import HopperE4m3QgmmaDot32
    from verity.ml.definitions.registry import fp8_moe_experts as X

    return bind(getattr(X, form), E=256, G=128, KB=128, SWAP=True, DOT=HopperE4m3QgmmaDot32, **kw)


def _select(e: int):
    from verity.ml.circuits.ops import moe as M

    return M.select_bf16(e)


def _gather(v: int):
    from verity.ml.definitions.registry import fp8_moe_experts as X

    return X.gather_e4m3(v)


#: the MoE targets: Qwen3-30B-A3B (E = 128, w13's K 2048, w2's K 768) and GLM-5.3 at TP8 rank 0 (E = 256, the rank's w2 K
#: 256), their small-K classes, and the selects. Each lowers to one flat class but `glm-fp8-gemm` (two).
TARGETS = {
    "select-x3": lambda: _select(3),
    "select-x128": lambda: _select(128),
    "select-x256": lambda: _select(256),
    "glm-gather-e4m3-x256": lambda: _gather(256),
    "qwen3-pair-w13-k16": lambda: _qwen3("pair", K=16),
    "qwen3-pair-w2-k16": lambda: _qwen3("pair-w", K=16),
    "qwen3-registry-coordinate-k16": lambda: _qwen3("registry", K=16),
    "qwen3-registry-coordinate-w-k16": lambda: _qwen3("registry-w", K=16),
    "glm-fp8-gemm": lambda: _glm("MoeExpertFp8Gemm", K=256, N=16),
    "glm-fp8-pair-w13-k128": lambda: _glm("MoeExpertFp8Coordinate", K=128),
    "glm-fp8-pair-w2-k128": lambda: _glm("MoeExpertFp8CoordinateW", K=128),
    "glm-fp8-pair-w2": lambda: _glm("MoeExpertFp8CoordinateW", K=256),
    "qwen3-registry-coordinate-w": lambda: _qwen3("registry-w", K=768),
}


def _sha(s: str | bytes) -> str:
    return hashlib.sha512(s.encode() if isinstance(s, str) else s).hexdigest()


def _parts(types: dict, layouts: list, unit: int, tables: dict) -> dict:
    """The mirror's archive parts: the unit's rows, Δ and logical list, by SHA-512."""
    from verity_flock import derive as D

    dv = D.Derivation(types, layouts, unit, tables)
    return {"unit": _sha(D.netlist(dv.unit, layouts[unit].digest()[:16])), "delta": _sha(D.delta_text(dv)),
            "logical": _sha(D.logical_text(dv))}


def write_target(name: str, scratch: Path) -> list[dict]:
    """Each class of `name` as `STEM.types.json` and `STEM.archive.json` in `scratch`, with the mirror's outputs."""
    from verity.ml.circuits import library
    from verity_flock import circuit_types as CT
    from verity_flock import ir_lower as IL
    from verity_flock import layouts as LY
    from verity_flock import partition_units as PU

    units = PU.partition_units(PU.one_call(TARGETS[name]()), "q-word")
    out = []
    for k, (_, cl) in enumerate(sorted(units.classes().items(), key=lambda kv: len(kv[1].representative.gates))):
        c = PU.lower_class(units.activation(cl.definition), cl.representative)
        if c.circuit is None:
            raise SystemExit(f"flock_rows: {name} class {k} does not lower: {c.reason} {sorted(c.missing)}")
        C, U = c.circuit, cl.representative
        ins = [f"in{j}" for j in range(len(U.in_widths))]
        outs = [f"out{j}" for j in range(len(U.out_widths))]
        t = CT.from_gf2(C, [b for p in ins for b in C.inputs[p]], [f for p in outs for f in C.outputs[p]],
                        [(1, w) for w in U.in_widths], [(1, w) for w in U.out_widths], library.PROGRAM, {})
        stem = f"{name}.{k}"
        (scratch / f"{stem}.types.json").write_text(json.dumps({"types": [t.content()]}))
        (scratch / f"{stem}.netlist").write_text(IL.netlist(IL.layout(C, ins, outs), f"moe/{name}"))
        lay = LY.BlockLayout(t.digest(), None, (), (), ("derive",), (), ())
        (scratch / f"{stem}.archive.json").write_text(json.dumps(
            {"types": [t.content()], "tables": {}, "layouts": [lay.to_json()], "unit_layout": 0}))
        out.append({"unit": stem, "definition": U.definition, "ands": c.counts["ands"], "in_ports": len(U.in_widths),
                    "in_bits": sum(U.in_widths), "out_bits": sum(U.out_widths), "digest": t.digest(),
                    "name": f"moe/{name}", "mirror": _parts({t.digest(): t}, [lay], 0, {})})
    return out


def write_fanout(m: int, scratch: Path) -> dict:
    """`test_derive.fanout_case(m, m)`: a unit of m placed calls of a one-AND callee and m placed reads of a 4-bit table."""
    from verity.ml.circuits import library
    from verity_flock import circuit_types as CT
    from verity_flock import layouts as LY

    def lay(t, rl, calls=(), reads=(), own=("derive",)):
        return LY.BlockLayout(t.digest() if t is not None else None, rl, (), (), own, tuple(calls), tuple(reads))

    callee = CT.CircuitType(library.PROGRAM, ((1, 1),), ((1, 1),), (), (), ((CT.AND, ((0,), 0), ((0,), 1)),), (((1,), 0),))
    ws = [(i * 7 + 3) & 0xF for i in range(16)]
    sha = _sha(b"".join(w.to_bytes(4, "little") for w in ws))
    items = [(CT.CALL, 0, (((i,), 0),)) for i in range(m)] + [(CT.READ, 0, tuple(((i % m,), 0) for _ in range(4)))
                                                              for i in range(m)]
    outs = [((m + i,), 0) for i in range(m)] + [((2 * m + 4 * i + b,), 0) for i in range(m) for b in range(4)]
    unit = CT.CircuitType(library.PROGRAM, ((m, 1),), ((len(outs), 1),), (callee.digest(),), ((sha, 4, 4),), tuple(items),
                          tuple(outs))
    layouts = [lay(callee, 10), lay(None, 10, own=("gen", sha, 4, 4, 2)),
               lay(unit, None, calls=[(i, 0, i) for i in range(m)], reads=[(m + i, 1, i) for i in range(m)])]
    stem = f"fanout-{m}"
    (scratch / f"{stem}.archive.json").write_text(json.dumps(
        {"types": [callee.content(), unit.content()], "tables": {sha: {"words": ws}},
         "layouts": [x.to_json() for x in layouts], "unit_layout": 2}))
    return {"unit": stem, "definition": f"fan-out {m} calls, {m} reads", "ands": None, "in_ports": 1, "in_bits": m,
            "out_bits": len(outs), "calls": m, "reads": m,
            "mirror": _parts({callee.digest(): callee, unit.digest(): unit}, layouts, 2, {sha: ws})}


WRAP = ("import resource, subprocess, sys; r = subprocess.run(sys.argv[1:]); u = resource.getrusage(resource.RUSAGE_CHILDREN); "
        "sys.stderr.write('\\n%.3f %d\\n' % (u.ru_utime + u.ru_stime, u.ru_maxrss)); sys.exit(r.returncode)")


def timed(cmd: list[str], cap: float | None) -> tuple[float, float | None, int, int | None, bytes, bytes]:
    """Wall seconds, CPU seconds (user and system) and peak RSS in KiB (of `cmd` alone: a fresh wrapper is its only parent),
    exit code (None past `cap`), stdout and stderr. CPU seconds are what a single-threaded run costs on a shared machine,
    where wall seconds count the time it waited for a core."""
    t0 = time.monotonic()
    with open(os.devnull, "rb") as nul:
        p = subprocess.Popen([sys.executable, "-c", WRAP, *cmd], stdin=nul, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True)
    try:
        out, err = p.communicate(timeout=cap)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        out, err = p.communicate()
        rc = None
    dt = time.monotonic() - t0
    err, _, last = err.rstrip(b"\n").rpartition(b"\n")
    cpu, _, rss = last.decode(errors="replace").partition(" ")
    if not rss.isdigit():
        return dt, None, 0, rc, out, err + b"\n" + last
    return dt, float(cpu), int(rss), rc, out, err


def measure(u: dict, label: str, binp: str, mode: str, scratch: Path, cap: float | None) -> dict:
    stem = u["unit"]
    if mode == "rows":
        cmd = [binp, str(scratch / f"{stem}.types.json"), "--root", u["digest"], "--name", u["name"]]
    else:
        cmd = [binp, "--archive", str(scratch / f"{stem}.archive.json"), "--part", mode]
    dt, cpu, rss, rc, out, err = timed(cmd, cap)
    if rc is None:
        equal = None
    elif mode == "rows":
        equal = rc == 0 and out == (scratch / f"{stem}.netlist").read_bytes()
    elif mode == "check":
        equal = rc == 0 and out == b"ok\n"
    else:
        equal = rc == 0 and _sha(out) == u["mirror"][mode]
    return {**{k: v for k, v in u.items() if k not in ("mirror", "digest", "name")}, "bin": label, "mode": mode,
            "seconds": round(dt, 3), "cpu_seconds": cpu, "peak_rss_mib": round(rss / 1024, 1), "rc": rc, "equal": equal,
            "stderr": err.decode(errors="replace")[-300:]}


def table(doc: dict, cols: tuple[str, ...] = ("rows", "check")) -> str:
    """`doc`'s wall seconds as Markdown, one line per unit and a column per mode in `cols` and binary: the MoE units by size,
    then the fan-out units' `check`."""
    def cell(r: dict | None) -> str:
        if r is None:
            return "-"
        if r["rc"] is None:
            return f"> {r['seconds']:,.0f} (cap)"
        return f"{r['seconds']:,.2f}" if r["seconds"] < 10 else f"{r['seconds']:,.1f}"

    bins = list(doc["bins"])
    units: dict[str, dict] = {}
    for r in doc["rows"]:
        units.setdefault(r["unit"], {"row": r})[(r["bin"], r["mode"])] = r
    head = [f"{m}, {b}" for m in cols for b in bins]
    lines = ["| unit | ANDs | input ports | " + " | ".join(head) + " | peak RSS |",
             "| --- | ---: | ---: | " + " | ".join("---:" for _ in head) + " | ---: |"]
    for name, u in units.items():
        if u["row"]["ands"] is None:
            continue
        rss = max(r["peak_rss_mib"] for k, r in u.items() if k != "row" and k[0] == bins[-1])
        lines.append(f"| {u['row']['definition']} | {u['row']['ands']:,} | {u['row']['in_ports']:,} | "
                     + " | ".join(cell(u.get((b, m))) for m in cols for b in bins) + f" | {rss / 1024:.1f} GiB |")
    fan = [(u["row"]["calls"], u) for u in units.values() if u["row"]["ands"] is None]
    if fan:
        lines += ["", "| placed calls and reads | " + " | ".join(f"check, {b}" for b in bins) + " |",
                  "| ---: | " + " | ".join("---:" for _ in bins) + " |"]
        lines += [f"| {m:,} | " + " | ".join(cell(u.get((b, "check"))) for b in bins) + " |" for m, u in sorted(fan)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--bin", action="append", help="LABEL=PATH of a flock-rows binary")
    ap.add_argument("--target", action="append", choices=sorted(TARGETS), help="default: every target")
    ap.add_argument("--fanout", default="", help="M,..: the fan-out unit at each M")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--cap", type=float, help="seconds per run; a run past it is killed and recorded with rc null")
    ap.add_argument("--cap-bin", action="append", default=[], help="LABEL=SECONDS: a cap for one binary's runs")
    ap.add_argument("--table", type=Path, metavar="FLOCK_ROWS_JSON", help="print a flock_rows.json as Markdown, and run nothing")
    a = ap.parse_args(argv)
    if a.table:
        print(table(json.loads(a.table.read_text())))
        return 0
    if not (a.out and a.bin):
        ap.error("--out and --bin are required to measure")
    bins = [b.split("=", 1) for b in a.bin]
    caps = {k: float(v) for k, v in (c.split("=", 1) for c in a.cap_bin)}
    modes = a.modes.split(",")
    a.out.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "bins": dict(bins), "modes": modes, "rows": []}
    path = a.out / "flock_rows.json"
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        units = [u for name in (a.target or list(TARGETS)) for u in write_target(name, scratch)]
        units += [write_fanout(int(m), scratch) for m in a.fanout.split(",") if m]
        for u in sorted(units, key=lambda u: (u.get("calls", 0), u["ands"] or 0)):
            for mode in modes:
                if mode == "rows" and "digest" not in u:
                    continue
                for label, binp in bins:
                    row = measure(u, label, binp, mode, scratch, caps.get(label, a.cap))
                    bad += row["equal"] is False
                    doc["rows"].append(row)
                    print(json.dumps({k: row[k] for k in ("unit", "ands", "bin", "mode", "seconds", "cpu_seconds",
                                                          "peak_rss_mib", "equal")}),
                          flush=True)
                    path.write_text(json.dumps(doc, indent=1))
    doc["mismatches"] = bad
    path.write_text(json.dumps(doc, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
