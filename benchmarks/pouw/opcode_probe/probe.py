"""The opcode-probe harness: build the probes, gate every probe's SASS, time the ones that pass, and write their price rows.

A probe is an `OP_KERNEL` and its `OpProbe` table entry (probe.cuh; README.md has how one is written). `gate` judges
every probe of a binary with `gpu_kit/sass_gate.py`: the loop counts its `expect` declares, no spill, a live sink, and the
bank check on its own shared-memory addresses. A probe that declares `refuse = CHECK` is a known-invalid one, kept as the
gate's regression test: it must be refused, by that check. `run` gates first and times nothing unless every verdict is
the one declared; then it times the passed probes interleaved (gpu_kit/bench.cuh, after each sink's poisoned pass and
its no-write control) and writes, under --out:

- `census.csv`: in RS-W′'s census schema (rows.py), a line for each (kind, row) a timed probe prices, the fastest of
  its probes (the class costs what its cheapest form does), its `runs` the run id;
- `rows.jsonl`: one line a timed probe: class, rate, MACs a word, run id, SASS hash, its census line, clocks and tags;
- `gate.json`, the verdicts, and `times.jsonl`, the binary's own lines (every block's cycles, every rep).

The rate is the gate's count: a loop that `expect` says holds N instructions of the class, checked on the SASS, run
`iters` times by `warps` warps of each of the blocks (one an SM), over the block's cycles; the median block, then the
median rep.

  python probe.py build [--out DIR] [--arch SM] [SOURCE...]        # probes/*.cu -> DIR/STEM, SM's SASS only (sm_120a)
  python probe.py gate BINARY... [--json OUT]                      # CPU only; exit 8 when a verdict isn't the declared one
  python probe.py run BINARY... --out DIR [--reps N] [--scale X] [--since EPOCH] [--run-id ID]   # one GPU
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXIT_FAIL = 8
NVCC_FLAGS = ("-std=c++20", "-O3", "-ldl")
#: the one SASS target a binary carries; the datacenter probes (`../dc/`) build for sm_90a and sm_100a
ARCHES = ("sm_120a", "sm_90a", "sm_100a")


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("gpu_kit_sass_gate", HERE.parent / "gpu_kit" / "sass_gate.py")
rows = _load("opcode_probe_rows", HERE / "rows.py")


# ---- build -------------------------------------------------------------------------------------------------------------


def nvcc() -> str:
    for c in (os.environ.get("NVCC"), os.environ.get("CUDA_HOME") and os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"),
              shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and os.access(c, os.X_OK):
            return c
    raise SystemExit("probe: no nvcc (set NVCC or CUDA_HOME)")


def build(out: Path, sources: list[Path], arch: str = "sm_120a") -> list[Path]:
    """Each source for `arch`'s SASS. The host side sees the target as PROBE_ARCH (90, 100, 120), so a probe table can
    declare the counts that target's ptxas lowers to."""
    out.mkdir(parents=True, exist_ok=True)
    gencode = f"arch=compute_{arch.removeprefix('sm_')},code={arch}"
    define = f"-DPROBE_ARCH={arch.removeprefix('sm_').rstrip('a')}"
    built = []
    for src in sources:
        dst = out / src.stem
        subprocess.run([nvcc(), *NVCC_FLAGS, define, "-gencode", gencode, "-o", str(dst), str(src)], check=True)
        built.append(dst)
    return built


# ---- gate --------------------------------------------------------------------------------------------------------------


def listing(binary: Path) -> list[dict]:
    """The binary's probe table (`--list`: host code only, no GPU)."""
    return json.loads(subprocess.run([str(binary), "--list"], check=True, capture_output=True, text=True).stdout)


def expects(p: dict) -> list[dict[str, int]]:
    return [G.parse_expect(e) for e in p["expect"].split(";") if e.strip()]


def judge(binary: Path, probes: list[dict], fns: list | None = None, tools: dict | None = None) -> list[dict]:
    """Each probe's gate verdict, and whether it is the declared one (passed; or refused by its `refuse` check)."""
    fns = G.functions(binary, tools) if fns is None else fns
    out = []
    for p in probes:
        fn = p.get("fn") or p["name"]
        v = G.gate(binary, [fn], expects(p), smem=p["smem"] if p["smem_ops"] else None, fns=fns)
        rec = v["kernels"][0] if v["kernels"] else {"kernel": fn, "ok": False, "sass_sha256": None,
                                                     "reasons": [f"no kernel {fn} in {binary}"]}
        if p["refuse"]:
            correct = not rec["ok"] and any(r.startswith(p["refuse"] + ":") for r in rec["reasons"])
        else:
            correct = rec["ok"]
        out.append({"binary": binary.name, "probe": p["name"], "refuse": p["refuse"], "passed": rec["ok"],
                    "correct": correct, "sass_sha256": rec["sass_sha256"], "reasons": rec["reasons"], "verdict": rec})
    return out


def verdict_lines(verdicts: list[dict]) -> list[str]:
    out = []
    for v in verdicts:
        head = f"{'PASS' if v['passed'] else 'REFUSED'} {v['binary']}:{v['probe']}"
        if v["refuse"]:
            head += f" (known-invalid, refuse by {v['refuse']}: {'as declared' if v['correct'] else 'NOT AS DECLARED'})"
        elif not v["correct"]:
            head += " (must pass)"
        out.append(head + (f" sass {v['sass_sha256'][:16]}" if v["sass_sha256"] else ""))
        out += [f"  {r}" for r in v["reasons"]]
    return out


# ---- rows --------------------------------------------------------------------------------------------------------------


def rate_of(p: dict, times: list[dict]) -> dict:
    """The probe's rates from its timed reps: every expected pattern of every loop, and the counted one's."""
    ex = expects(p)
    warps = [int(w) for w in p["warps"].split(";")] if p["warps"] else [p["tpb"] // 32] * len(ex)
    counted = p["counted"] or next(iter(ex[0]))
    loop = next(i for i, e in enumerate(ex) if counted in e)
    until = p.get("until", 0) - 1
    per_rep: dict[str, list[float]] = {}
    stop_gap = []
    for t in times:
        cyc = statistics.median(t["cyc"])
        if until >= 0:
            # the OP_UNTIL loop ran only while the fixed role did: its trips over the stop's cycles, block by block
            per_block = [tr / st for tr, st in zip(t["trips"], t["stop"]) if st > 0]
            stop_gap.append(statistics.median([1 - st / c for st, c in zip(t["stop"], t["cyc"]) if c > 0]))
        for i, e in enumerate(ex):
            for pat, n in e.items():
                r = n * statistics.median(per_block) if i == until else n * t["iters"] * warps[i] / cyc
                per_rep.setdefault(f"{i}:{pat}", []).append(r)
    rates = {k: statistics.median(v) for k, v in per_rep.items()}
    reps = per_rep[f"{loop}:{counted}"]
    mhz = [t["sm_mhz"]["during"] or t["sm_mhz"]["after"] for t in times]
    mhz = [m for m in mhz if m]
    return {"counted": counted, "rate": rates[f"{loop}:{counted}"], "rate_reps": reps,
            "spread": (max(reps) - min(reps)) / statistics.median(reps), "rates": rates,
            "sm_mhz": statistics.median(mhz) if mhz else None,
            "distinct_sms": min(len(set(t["smid"])) for t in times), "blocks": times[0]["blocks"],
            "throttled": any(int(t["throttle"], 16) & 0xEC for t in times),
            "stop_gap": statistics.median(stop_gap) if stop_gap else None}


def make_rows(binary: Path, probes: list[dict], verdicts: list[dict], lines: list[dict], run_id: str,
              since: str) -> list[dict]:
    """A sidecar row a timed probe, each with the census line it would write (`census`; None for a joint point)."""
    by = {v["probe"]: v for v in verdicts if v["binary"] == binary.name}
    side = []
    gates = {ln["probe"]: ln for ln in lines if ln.get("op") == "gate"}
    for p in probes:
        times = [ln for ln in lines if ln.get("op") == "time" and ln["probe"] == p["name"]]
        if not times:
            continue
        r = rate_of(p, times)
        kind, row = p["kind"], p["row"] or (f"ROW_{p['kind']}" if p["kind"] else "")
        line = None
        if kind:
            try:
                line = rows.census_line(kind, row, r["rate"], run_id, since)
            except rows.RowError as e:
                r["census_refused"] = str(e)
        tags = [t for t, on in (("throttled", r["throttled"]), ("spread", r["spread"] > 0.005),
                                ("sms", r["distinct_sms"] < r["blocks"])) if on]
        side.append({"class": kind or None, "rate": r["rate"],
                     "macs_a_word": rows.macs_a_word(r["rate"], p["words"]) if p["words"] else None,
                     "run": run_id or None, "sass_sha256": by[p["name"]]["sass_sha256"], "probe": p["name"],
                     "binary": binary.name, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                     "row": row or None, "words": p["words"] or None, "census": line, "tags": tags,
                     "sink": gates.get(p["name"]), "note": p["note"] or None, "blocks_asked": p.get("blocks") or None,
                     **r})
    return side


def one_a_row(side: list[dict]) -> list[dict[str, str]]:
    """The census: a line for each (kind, row), since machine_model refuses two at one epoch. Of the probes that price
    one, the fastest form's (the class costs what its cheapest form does); each sidecar row names it (`census_by`)."""
    best: dict[tuple[str, str], dict] = {}
    for s in side:
        if s["census"]:
            k = (s["census"]["kind"], s["census"]["row"])
            if k not in best or s["rate"] > best[k]["rate"]:
                best[k] = s
    for s in side:
        if s["census"]:
            s["census_by"] = best[(s["census"]["kind"], s["census"]["row"])]["probe"]
    return [b["census"] for b in best.values()]


def row_line(s: dict) -> str:
    mw = f" macs/word={s['macs_a_word']:.1f}" if s["macs_a_word"] else ""
    return (f"ROW {s['class'] or '(joint)'} {s['row'] or ''} rate={s['rate']:.6f}{mw} run={s['run']} "
            f"sass={(s['sass_sha256'] or '')[:16]} sm_mhz={s['sm_mhz']}" + "".join(f" [{t}]" for t in s["tags"]))


# ---- the command ---------------------------------------------------------------------------------------------------------


def gate_all(binaries: list[Path]) -> tuple[dict[Path, list[dict]], list[dict]]:
    tools = G.H.find_tools()
    tables = {b: listing(b) for b in binaries}
    verdicts = [v for b in binaries for v in judge(b, tables[b], tools=tools)]
    return tables, verdicts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, default=Path("build"))
    b.add_argument("--arch", choices=ARCHES, default="sm_120a")
    b.add_argument("sources", nargs="*", type=Path)
    g = sub.add_parser("gate")
    g.add_argument("binaries", nargs="+", type=Path)
    g.add_argument("--json", type=Path)
    r = sub.add_parser("run")
    r.add_argument("binaries", nargs="+", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--reps", type=int, default=10)
    r.add_argument("--scale", type=float, default=1.0, help="multiply every probe's iterations")
    r.add_argument("--since", default="", help="the census epoch the lines hold from")
    r.add_argument("--run-id", default=os.environ.get("RESEARCH_RUN_ID", ""))
    r.add_argument("--only", default="", help="time only the probes whose name matches this regex")
    r.add_argument("--passed-only", action="store_true",
                   help="time the probes that pass even when others are refused (the refusals stay in gate.json)")
    a = p.parse_args(argv)

    if a.cmd == "build":
        for out in build(a.out, a.sources or sorted((HERE / "probes").glob("*.cu")), a.arch):
            print(out)
        return 0

    tables, verdicts = gate_all(a.binaries)
    for line in verdict_lines(verdicts):
        print(line)
    wrong = [v for v in verdicts if not v["correct"]]
    if a.cmd == "gate":
        if a.json:
            a.json.write_text(json.dumps({"version": G.VERSION, "verdicts": verdicts}, indent=1) + "\n")
        return EXIT_FAIL if wrong else 0

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gate.json").write_text(json.dumps({"version": G.VERSION, "verdicts": verdicts}, indent=1) + "\n")
    if wrong and not a.passed_only:
        print(f"probe: {len(wrong)} verdicts are not the declared ones; nothing timed", file=sys.stderr)
        return EXIT_FAIL
    if wrong:
        print(f"probe: {len(wrong)} verdicts are not the declared ones; timing the rest (--passed-only)", file=sys.stderr)
    only = re.compile(a.only) if a.only else None
    side = []
    with (a.out / "times.jsonl").open("w") as raw:
        for binary in a.binaries:
            names = [v["probe"] for v in verdicts if v["binary"] == binary.name and v["passed"] and v["correct"]
                     and (only is None or only.search(v["probe"]))]
            if not names:
                continue
            res = subprocess.run([str(binary), "--time", *names, "--reps", str(a.reps), "--scale", str(a.scale)],
                                 capture_output=True, text=True)
            raw.write(res.stdout)
            sys.stderr.write(res.stderr)
            if res.returncode:
                print(f"probe: {binary} --time exited {res.returncode} (3: a sink failed its poisoned pass or "
                      "control)", file=sys.stderr)
                return EXIT_FAIL
            lines = [json.loads(ln) for ln in res.stdout.splitlines() if ln.startswith("{")]
            side += make_rows(binary, tables[binary], verdicts, lines, a.run_id, a.since)
    census = one_a_row(side)
    (a.out / "census.csv").write_text(rows.census_csv(census))
    (a.out / "rows.jsonl").write_text("".join(json.dumps(s) + "\n" for s in side))
    for s in side:
        print(row_line(s))
    if not a.run_id:
        print("probe: no run id (run under research run): rows.jsonl only, no census line", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
