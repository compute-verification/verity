"""``verity bench verifier_cost COMMAND``: a verifier of record's time against the size of the description it reads and
the size of the circuit that description unrolls to (README.md).

  measure --out DIR [--config NAME ...] [--ir word|boolean ...] [--unrolled-max GATES] [--exe PATH]
      Serve_v1 at each config, word-level and on bits (`lift.on_bits`), as an inference-only check instance (θ its registered inputs, one
      IU per Call), checked by `nci-check --policy` (`NciCheck.Desc`) and, where the unrolled circuit has at most GATES
      gates, by `nci-check --policy --unrolled` (`NciCheck.check`) on the instance expanded gate by gate;
      DIR/results.json.
  partition --out DIR [--config NAME ...] [--ir word|boolean ...] [--unrolled-max GATES] [--exe PATH]
      the one-stage verifier's partition (`sampled-proofs-difftest`'s `partition`: `Partition.units`, `strata`, the law
      and the digest) on Serve_v1 under the template query of its Calls, and under `Q_word` v1 and `Q_call` v1 at X = 16
      where its Calls' distinct Definitions have at most GATES gates in all; and on the served shape, the population
      program of Serve_v1's largest Definition at N instances (`Q_template_instance`); DIR/partition.json.
  store RUN [--runs-dir DIR]
      a `job.sh` run's results.json or partition.json (`research fetch RUN --all`) as one evidence/v1 record, naming
      the run and its commit; prints its art: id. From a VM with the store's keys: a node's shell has none.
  plot ART --to PNG
      the results of `art:ART` (a results.json or partition.json in the store), time against description bytes and
      against unrolled gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NCI = ROOT / "verity/core/protocols/network/nci/lean"
EXE = NCI / ".lake/build/bin/nci-check"
SCHEMA = "verity/verifier-cost/v1"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "benchmarks/network_sim"))

_TINY = dict(layers=1, hidden=8, heads=1, kv_heads=1, head_dim=8, intermediate=16, vocab=8, prompt=1, steps=1, max_pos=8)
# name -> Config keyword arguments, or ("b0", layers) for `Config.b0`
CONFIGS: dict[str, dict | tuple[str, int]] = {
    "tiny": _TINY,
    "tiny-2": _TINY | {"layers": 2},
    "small": {"layers": 1},
    "small-3": {"layers": 3},
    "mid": {"layers": 1, "hidden": 128, "intermediate": 256, "vocab": 64},
    **{f"b0-{n}": ("b0", n) for n in (1, 2, 4, 8, 16)},
    "b0": ("b0", 30),
}
BOOLEAN = ("tiny", "tiny-2", "small", "mid", "b0-1", "b0")


def _reference():
    spec = importlib.util.spec_from_file_location("nci_reference", NCI / "scripts/nci_reference.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


def serve(name: str, ir: str):
    """Serve_v1 at config `name` as a Program, word-level or on bits."""
    from workload.served import Config, _derive

    from verity.ml.circuits.program import Program

    c = CONFIGS[name]
    cfg = Config.b0(layers=c[1]) if isinstance(c, tuple) else Config(**c)
    fn = _derive(cfg).program.fn
    if ir == "boolean":
        import lift

        fn = lift.on_bits(fn)
    return Program(fn, registered=("weights",))


def text_of(prog) -> str:
    from verity.ml.circuits import codec

    return codec.canonical_json(codec.encode_program(prog)).decode()


def program(name: str, ir: str) -> str:
    """Serve_v1 at config `name` as descriptor text, word-level or on bits."""
    return text_of(serve(name, ir))


def instance(ref, text: str) -> tuple[dict, dict]:
    """`NciCheck.Desc`'s instance (θ the registered inputs, one IU per Call, a gate-less node in none) and the
    description's sizes."""
    inp, reg, spans = ref.root_nodes(text)
    desc = json.loads(text)
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    theta = [k for k in range(len(nodes)) if reg[k]]
    owner, u = [], 0
    for k, (lo, hi) in enumerate(spans):
        if inp[k] or hi == lo:
            owner.append("input")
        else:
            owner.append(u)
            u += 1
    gates = spans[-1][1] if spans else 0
    body_nodes = sum(len(d.get("body", {}).get("nodes", [])) for d in desc["definitions"].values())
    sizes = {"description_bytes": len(text.encode()), "definitions": len(desc["definitions"]), "body_nodes": body_nodes,
             "root_nodes": len(nodes), "ius": u, "unrolled_gates": gates}
    return {"program": text, "theta": theta, "owner": owner, "gamma": gates, "alpha": [10**12, 1]}, sizes


#: Linux gives a child the high-water RSS of the process it forked from, so a verifier forked from this process would
#: report this process's peak (gigabytes once B0 is built). It is forked from this launcher instead, a fresh
#: interpreter, which times it and reports its peak (at least the launcher's own, about 10 MB).
LAUNCHER = ("import json, os, sys, time\n"
            "t = time.perf_counter()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    try:\n"
            "        os.execvp(sys.argv[2], sys.argv[2:])\n"
            "    finally:\n"
            "        os._exit(127)\n"
            "_, status, ru = os.wait4(pid, 0)\n"
            "s = time.perf_counter() - t\n"
            "with open(sys.argv[1], 'w') as f:\n"
            "    json.dump({'seconds': s, 'max_rss_kb': ru.ru_maxrss, 'rc': os.waitstatus_to_exitcode(status)}, f)\n")


def timed(argv: list[str], stdin: Path | None = None) -> dict:
    """One run of `argv` (reading `stdin`, a file): wall seconds, peak RSS and stdout as JSON."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, \
            open(stdin or os.devnull, "rb") as inp, tempfile.NamedTemporaryFile(suffix=".json") as stats:
        subprocess.run([sys.executable, "-c", LAUNCHER, stats.name, *argv], stdin=inp, stdout=out, stderr=err, check=True)
        r = json.loads(Path(stats.name).read_text())
        out.seek(0)
        err.seek(0)
        text, etext = out.read().decode(), err.read().decode()
    if r["rc"] != 0:
        return r | {"stderr": etext[-400:]}
    return r | {"verdict": json.loads(text)}


def _digest(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def measure(args) -> int:
    ref = _reference()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.config or list(CONFIGS):
            for ir in args.ir or ("word", "boolean"):
                if ir == "boolean" and not args.config and name not in BOOLEAN:
                    continue
                t = time.perf_counter()
                text = program(name, ir)
                inst, row = instance(ref, text)
                row = {"config": name, "ir": ir, "build_seconds": round(time.perf_counter() - t, 3), **row}
                path = Path(tmp) / f"{name}-{ir}.json"
                path.write_text(json.dumps(inst))
                runs = [timed([str(args.exe), "--policy", str(path)]) for _ in range(args.repeat)]
                d = min(runs, key=lambda r: r["seconds"])
                row["desc"] = {k: v for k, v in d.items() if k != "verdict"} | {"verdict_sha256": _digest(d.get("verdict"))}
                row["verdict"] = d.get("verdict")
                if row["unrolled_gates"] <= args.unrolled_max:
                    upath = Path(tmp) / f"{name}-{ir}-unrolled.json"
                    t = time.perf_counter()
                    upath.write_text(json.dumps(ref.expand(inst)))
                    expand_s = time.perf_counter() - t
                    u = timed([str(args.exe), "--policy", "--unrolled", str(upath)])
                    upath.unlink()
                    row["unrolled"] = {k: v for k, v in u.items() if k != "verdict"} | {
                        "expand_seconds": round(expand_s, 3), "agrees": u.get("verdict") == d.get("verdict")}
                else:
                    row["unrolled"] = {"skipped": f"more than --unrolled-max {args.unrolled_max} gates"}
                path.unlink()
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("config", "ir", "description_bytes", "unrolled_gates")}
                                 | {"desc_s": round(row["desc"]["seconds"], 3),
                                    "unrolled_s": row["unrolled"].get("seconds"), "agrees": row["unrolled"].get("agrees")}),
                      flush=True)
    result = {"schema": SCHEMA, "verifier": "nci-check", "workload": "Serve_v1", "repeat": args.repeat,
              "unrolled_max": args.unrolled_max, "rows": rows}
    (out / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    bad = [r["config"] + "/" + r["ir"] for r in rows
           if r["desc"]["rc"] != 0 or r["unrolled"].get("agrees") is False]
    if bad:
        print("disagreements or failures:", ", ".join(bad), file=sys.stderr)
        return 1
    return 0


PIECES = ROOT / "verity/core/service/sampled_proofs/.lake/build/bin/sampled-proofs-difftest"
PARTITION_SCHEMA = "verity/verifier-cost-partition/v1"
POPULATION = (1, 30, 1000, 1_000_000)


def _partition_case(desc: dict, query: dict) -> dict:
    params = {"anchors": [], "delta": "1/10", "law": {"law": "subset", "k": 1}, "leaf_layer": None, "log": [],
              "partition": None, "ports": {}, "program": desc, "query": query, "registered": None,
              "scheme": "verifier_cost", "work_table": None}
    return {"fn": "partition", "input": {"params": params}}


def _partition_row(exe: Path, tmp: str, prog, query: dict, repeat: int) -> dict:
    """The `partition` piece on `prog` under `query`, best of `repeat`, with the sizes it is measured against."""
    from verity.ml.circuits import partition_object as PO

    text = text_of(prog)
    path = Path(tmp) / "case.jsonl"
    path.write_text(json.dumps(_partition_case(json.loads(text), query)) + "\n")
    runs = [timed([str(exe)], stdin=path) for _ in range(repeat)]
    path.unlink()
    best = min(runs, key=lambda r: r["seconds"])
    line = best.pop("verdict", None) or {}
    out = line.get("output") or {}
    distinct = {PO.template_id(fn): fn.gates for _, fn in PO.calls(prog)}
    return {"description_bytes": len(text.encode()), "unrolled_gates": prog.root.gates,
            "call_definition_gates": sum(distinct.values()), **best, "population": out.get("population"),
            "strata": len(out.get("strata") or []), "refused": line.get("reject") or ("population" if out and
                                                                                       out.get("population") is None else None)}


def partition(args) -> int:
    from verity.core.protocols.sampled_proofs.one_stage import partition as P
    from verity.ml.circuits import partition_object as PO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(row: dict) -> None:
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("config", "ir", "shape", "query", "n", "description_bytes",
                                                  "unrolled_gates", "call_definition_gates", "seconds", "population",
                                                  "refused", "skipped") if k in row}), flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        for name in args.config or list(CONFIGS):
            for ir in args.ir or ("word", "boolean"):
                if ir == "boolean" and not args.config and name not in BOOLEAN:
                    continue
                t = time.perf_counter()
                prog = serve(name, ir)
                distinct = {PO.template_id(fn): fn for _, fn in PO.calls(prog)}
                head = {"config": name, "ir": ir, "build_seconds": round(time.perf_counter() - t, 3)}
                add(head | {"shape": "serve", "query": "template", "n": None}
                    | _partition_row(args.exe, tmp, prog, PO.template_instances_query(list(distinct.values())),
                                     args.repeat))
                for label, q in (("Q_word v1", PO.query(16, 32)), ("Q_call v1", PO.call_query(16))):
                    row = head | {"shape": "serve", "query": label, "n": None}
                    walked = sum(fn.gates for fn in distinct.values())
                    if walked > args.unrolled_max:
                        add(row | {"call_definition_gates": walked, "unrolled_gates": prog.root.gates,
                                   "skipped": f"more than --unrolled-max {args.unrolled_max} gates in its Calls' Definitions"})
                        continue
                    add(row | _partition_row(args.exe, tmp, prog, q, args.repeat))
                largest = max(distinct.values(), key=lambda fn: fn.gates)
                for n in POPULATION:
                    add(head | {"shape": "population", "query": "template", "n": n, "template": largest.id}
                        | _partition_row(args.exe, tmp, P.population_program(largest, n),
                                         PO.template_instance_query(largest), args.repeat))
    result = {"schema": PARTITION_SCHEMA, "verifier": "sampled-proofs partition", "workload": "Serve_v1",
              "repeat": args.repeat, "unrolled_max": args.unrolled_max, "rows": rows}
    (out / "partition.json").write_text(json.dumps(result, indent=1) + "\n")
    bad = [f"{r['config']}/{r['ir']}/{r['shape']}/{r['n']}" for r in rows
           if r["query"] == "template" and (r.get("rc") != 0 or r.get("refused"))]
    if bad:
        print("template query failures:", ", ".join(bad), file=sys.stderr)
        return 1
    return 0


def store(args) -> int:
    subprocess.run(["research", "fetch", args.run, "--all"], check=True, capture_output=True)
    run = args.runs_dir / args.run
    found = [f for f in (run / "out/results.json", run / "out/partition.json") if f.is_file()]
    if len(found) != 1:
        print(f"{run}/out holds {len(found)} of results.json and partition.json; store takes a run that wrote one",
              file=sys.stderr)
        return 1
    res = json.loads(found[0].read_text())
    meta = {"benchmark": "verifier_cost", "schema": res["schema"], "verifier": res["verifier"], "run": args.run,
            "commit": json.loads((run / "job.json").read_text())["run"]["remote"]["source_sha"]}
    p = subprocess.run(["research", "data", "put", "--kind", "evidence/v1", "--file", str(found[0]),
                        "--preserve", "--meta", json.dumps(meta)], check=True, capture_output=True, text=True)
    print(p.stdout.strip().splitlines()[-1])
    return 0


def plot(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = subprocess.run(["research", "data", "fetch", args.art], check=True, capture_output=True, text=True).stdout.strip()
    res = json.loads(Path(path).read_text())
    nci = res["schema"] == SCHEMA
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, x, label in ((axes[0], "description_bytes", "description (bytes)"), (axes[1], "unrolled_gates", "unrolled circuit (gates)")):
        for ir, marker in (("word", "o"), ("boolean", "s")):
            if not nci:
                for shape, query, style in (("serve", "template", "-"), ("population", "template", ":"),
                                            ("serve", "Q_word v1", "--"), ("serve", "Q_call v1", "-.")):
                    rs = sorted((r for r in res["rows"] if r["ir"] == ir and r["shape"] == shape and r["query"] == query
                                 and "seconds" in r), key=lambda r: r[x])
                    if rs:
                        ax.plot([r[x] for r in rs], [r["seconds"] for r in rs], marker=marker, linestyle=style,
                                label=f"{query}, {'population program' if shape == 'population' else 'Serve_v1'}, {ir}")
                continue
            rs = sorted((r for r in res["rows"] if r["ir"] == ir), key=lambda r: r[x])
            ax.plot([r[x] for r in rs], [r["desc"]["seconds"] for r in rs], marker=marker, label=f"NciCheck.Desc, {ir}")
            us = [r for r in rs if "seconds" in r["unrolled"]]
            if us:
                ax.plot([r[x] for r in us], [r["unrolled"]["seconds"] for r in us], marker=marker, linestyle="--",
                        label=f"NciCheck.check (unrolled), {ir}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("verifier wall time (s)")
        ax.grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{'NCI check' if nci else 'one-stage partition'} on Serve_v1 ({args.art})", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.to, dpi=130)
    print(args.to)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench verifier_cost", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--out", required=True)
    m.add_argument("--config", action="append", choices=list(CONFIGS))
    m.add_argument("--ir", action="append", choices=("word", "boolean"))
    m.add_argument("--unrolled-max", type=int, default=5_000_000)
    m.add_argument("--repeat", type=int, default=3)
    m.add_argument("--exe", type=Path, default=EXE)
    q = sub.add_parser("partition")
    q.add_argument("--out", required=True)
    q.add_argument("--config", action="append", choices=list(CONFIGS))
    q.add_argument("--ir", action="append", choices=("word", "boolean"))
    q.add_argument("--unrolled-max", type=int, default=5_000_000)
    q.add_argument("--repeat", type=int, default=3)
    q.add_argument("--exe", type=Path, default=PIECES)
    s = sub.add_parser("store")
    s.add_argument("run")
    s.add_argument("--runs-dir", type=Path, default=Path.home() / ".research/runs")
    p = sub.add_parser("plot")
    p.add_argument("art")
    p.add_argument("--to", required=True)
    args = ap.parse_args(argv)
    return {"measure": measure, "partition": partition, "store": store, "plot": plot}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
