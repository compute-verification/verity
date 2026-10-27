"""Serve a workload on the simulated datacenter, then check N3 in the clear.

    python -m workload.run --out /tmp/netsim-run                        # the MLP on 3 blocks of 2 devices, in one process
    python -m workload.run --out /tmp/netsim-run --transport tcp        # one localhost TCP process per unit
    python -m workload.run --out /tmp/netsim-run --tamper b0:fill       # a block writes into a frame's zero fill
    python -m workload.run --out /tmp/netsim-run --graph leaf_spine:2   # on `datacenter.examples.leaf_spine(2)`
    python -m workload.run --out /tmp/netsim-run --workload served --devices 1 --frame-bytes 1024
                                                                        # verity_vllm's serving Program, a stage a block

(run from `examples/network_sim/`, or with it on `PYTHONPATH`). The prover's side commits the placement (`statement`),
the units serve and commit their values, and the check reads what `run` writes to `--out`: `public.json` (the statement:
the Program's digest, the hardware, B, the placement's and units' roots, the occupancy profile), `witness.json` (the
placement's and units' rows and salts), `records.json` (each certified link direction's frames) and `n3.json` (the
check's report). It prints a summary and exits 1 when the check fails.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from verity.ml.circuits.codec import program_digest

from . import model as M
from . import n3
from .block import TAMPERS, Context, MemoryNet
from .hardware import OUTSIDE, Hardware

HERE = Path(__file__).resolve().parent
DATACENTER = str(HERE.parents[2] / "benchmarks" / "datacenter")


def declaration(spec: str):
    """`NAME[:ARG,…]`: the `datacenter.Datacenter` that `datacenter.examples.NAME` declares with those integer arguments,
    e.g. `leaf_spine:2`."""
    if DATACENTER not in sys.path:
        sys.path.insert(0, DATACENTER)
    from datacenter import examples

    name, _, args = spec.partition(":")
    fn = getattr(examples, name, None) if not name.startswith("_") else None
    if not callable(fn):
        raise SystemExit(f"--graph is NAME[:ARG,…], NAME a declaration of datacenter.examples, not {spec!r}")
    return fn(*(int(a) for a in args.split(",") if a))


def statement(ctx: Context, results: dict[str, dict]) -> tuple[dict, dict]:
    """The public statement and the witness of a run: the placement committed once, each unit's committed values."""
    reg = ctx.placement.commit()
    held = {u: r["values"] for u, r in sorted(results.items()) if r["values"] is not None}
    public = {"program": program_digest(ctx.model.program), "workload": ctx.workload, "spec": ctx.spec.to_json(),
              "hardware": ctx.hw.to_json(),
              "frame_bytes": ctx.phi.frame_bytes, "placement": reg.root.hex(),
              "values": {u: {"root": v["root"], "count": v["count"]} for u, v in held.items()},
              "occupancy": ctx.phi.occupancy()}
    witness = {"placement": {"rows": [r.hex() for r in reg.rows], "salts": [s.hex() for s in reg.salts]},
               "values": {u: {"rows": v["rows"], "salts": v["salts"]} for u, v in held.items()}}
    return public, witness


def run_memory(ctx: Context) -> dict[str, dict]:
    return asyncio.run(MemoryNet(ctx).run())


def _free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def run_tcp(ctx: Context, out: Path) -> dict[str, dict]:
    """One `python -m workload.block` process per unit on localhost; start them together once all listen."""
    units = ctx.units()
    config = dict(ctx.config, addresses={u: ["127.0.0.1", p] for u, p in zip(units, _free_ports(len(units)))})
    path = out / "config.json"
    path.write_text(json.dumps(config, indent=1))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(HERE.parent), os.environ.get("PYTHONPATH")])))
    procs = {u: subprocess.Popen([sys.executable, "-m", "workload.block", "--config", str(path), "--unit", u,
                                  "--out", str(out / f"unit-{u}.json")],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
             for u in units}
    try:
        for u, p in procs.items():
            if p.stdout.readline().strip() != "ready":
                raise RuntimeError(f"unit {u} exited before listening (code {p.wait()})")
        for p in procs.values():
            p.stdin.write("go\n")
            p.stdin.flush()
        for u, p in procs.items():
            if p.wait() != 0:
                raise RuntimeError(f"unit {u} failed with code {p.returncode}")
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
                p.wait()
    return {u: json.loads((out / f"unit-{u}.json").read_text()) for u in units}


def crossing_bits(ctx: Context) -> dict[str, tuple[int, int]]:
    """(values, bits) of the crossing edges, by the kind of value that crosses."""
    out: dict[str, tuple[int, int]] = {}
    for e in ctx.edges:
        n, b = out.get(ctx.model.kind[e.value], (0, 0))
        out[ctx.model.kind[e.value]] = (n + 1, b + e.width)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve a workload on the simulated datacenter and check N3 in the clear.")
    ap.add_argument("--out", required=True, type=Path, help="the directory to write the run's files to")
    ap.add_argument("--workload", choices=M.WORKLOADS, default="mlp",
                    help="the MLP, or verity_vllm's serving Program for a truncated SmolLM2 (`served`)")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--devices", type=int, default=2, help="devices in each block")
    ap.add_argument("--d", type=int, default=4, help="the MLP's width")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--groups", type=int, default=1, help="row groups a layer is split into (Calls per layer and request)")
    ap.add_argument("--hidden", type=int, default=64, help="served: the model's width")
    ap.add_argument("--intermediate", type=int, default=128, help="served: the MLP's inner width")
    ap.add_argument("--vocab", type=int, default=32, help="served: the vocabulary")
    ap.add_argument("--prompt", type=int, default=2, help="served: prompt tokens")
    ap.add_argument("--steps", type=int, default=1, help="served: greedy decode steps")
    ap.add_argument("--plan", choices=("pipeline", "tensor"), default="pipeline")
    ap.add_argument("--frame-bytes", type=int, default=64)
    ap.add_argument("--bucket-ticks", type=int, default=1, help="the mock's bucket, in ticks")
    ap.add_argument("--step-ticks", type=int, help="ticks between engine steps (default: one bucket)")
    ap.add_argument("--graph", metavar="NAME[:ARG,…]",
                    help="a declaration of datacenter.examples to run on instead of the mock, e.g. leaf_spine:2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transport", choices=("memory", "tcp"), default="memory")
    ap.add_argument("--tamper", help=f"UNIT:MODE, MODE one of {', '.join(TAMPERS)}")
    args = ap.parse_args(argv)

    spec = ({"d": args.d, "layers": args.layers, "requests": args.requests, "groups": args.groups} if args.workload == "mlp"
            else {"layers": args.layers, "hidden": args.hidden, "intermediate": args.intermediate, "vocab": args.vocab,
                  "prompt": args.prompt, "steps": args.steps})
    config = {"workload": args.workload, "spec": spec, "blocks": args.blocks, "devices": args.devices, "plan": args.plan,
              "frame_bytes": args.frame_bytes, "bucket_ticks": args.bucket_ticks, "seed": args.seed}
    if args.graph:
        config["hardware"] = Hardware.from_datacenter(declaration(args.graph)).to_json()
    if args.step_ticks:
        config["step_ticks"] = args.step_ticks
    if args.tamper:
        unit, _, mode = args.tamper.partition(":")
        config["tamper"] = {"unit": unit, "mode": mode}
    ctx = Context(config)
    args.out.mkdir(parents=True, exist_ok=True)
    results = run_tcp(ctx, args.out) if args.transport == "tcp" else run_memory(ctx)
    public, witness = statement(ctx, results)
    records, disagreements = n3.merge(results)
    report = n3.check(public, witness, records)
    report.violations[:0] = disagreements

    for name, obj in (("public", public), ("witness", witness), ("n3", report.to_json()),
                      ("records", {k: [[b, f.hex()] for b, f in v] for k, v in sorted(records.items())})):
        (args.out / f"{name}.json").write_text(json.dumps(obj, indent=1))
    answers = results[OUTSIDE]["answers"]
    ref = ctx.model.reference(ctx.seed)
    hw = ctx.hw
    print(f"program {public['program'][:16]}… {ctx.model.n} gates; placement root {public['placement'][:16]}… "
          f"({args.plan} on {len(hw.units)} blocks of {', '.join(str(len(hw.devices(u))) for u in sorted(hw.units))} "
          f"devices; ℓ across units {hw.cross_latency} ticks, a bucket {hw.bucket_ticks})")
    oracle = "integer reference" if args.workload == "mlp" else "Program evaluated in one process"
    print(f"answers {'equal' if answers == ref else 'differ from'} the {oracle}: {answers}")
    by_kind = ", ".join(f"{k} {n} values / {b} bits" for k, (n, b) in sorted(crossing_bits(ctx).items()))
    print(f"{len(ctx.edges)} crossing edges ({by_kind}) in {len(ctx.phi.messages)} messages, {args.frame_bytes}-byte frames")
    for key, counts in public["occupancy"].items():
        print(f"  occupancy {key:<22} {counts}")
    s = report.stats
    print(f"N3 {'holds' if report.ok else 'fails'}: {s['frames']} frames recorded, {s['explained']} φ frames and "
          f"{s['declared']} HELLOs explained, {s['appearances']} value appearances checked against {s['values']} "
          f"opened values")
    for code, n in sorted(report.codes().items()):
        print(f"  {code}: {n}")
    for v in report.violations[:8]:
        print(f"  {v.code} {v.where}: {v.detail}")
    print(f"files in {args.out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
