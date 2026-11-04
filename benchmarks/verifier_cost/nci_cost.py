"""``nci_cost.py CASE``: what `nci-check --policy` (`NciCheck.Desc`) costs under Boolean-first, each check also run through
`nci-phases` (`nci_phases/Main.lean`: the check's steps timed, the slowest Definitions' summaries named), whose verdict
must be `nci-check --policy`'s.

  suite --out DIR --phases EXE --scratch DIR
      the NCI package's own vectors: each `runs` entry of its lean-audit.json generated and its exe timed, the largest
      Program the difftests' references give a verdict on, and each liveness case's big instance (`n` past `WALK`)
  serve --out DIR --phases EXE --scratch DIR [--config NAME ...]
      Serve_v1 (`run.serve`) at each config, word-level and on bits (`lift.on_bits`), θ its registered inputs, one IU
      per Call (`run.instance`)
  glm --out DIR --phases EXE --scratch DIR --workload PATH [--word-only]
      GLM-5.3's rank 0 workload Program of the B1 Build (root: its inputs and one Call, the request) word-level as built,
      then on bits: `verity.ml.definitions.boolean.dry_run`'s gaps, and where there are none the request lifted
      (`lift.on_bits`) and called once by a root of its inputs. θ empty (its weights are bound per run), one IU, the
      request, at `nci_glm.GAMMA`, `nci_glm.ALPHA`

DIR/CASE.jsonl has a row per check: the description's sizes, `nci-check --policy`'s seconds, peak RSS and verdict, and
`nci-phases`'s PHASE lines (DIR/phases/<name>.log keeps them as they come). `--big N` is the body size from which
`nci-phases` times a summary step by step; `--no-policy` runs `nci-phases` alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nci_glm  # noqa: E402
import run  # noqa: E402

NCI = run.NCI


def launch(argv: list[str], log: Path | None = None) -> dict:
    """`argv` timed by `run.LAUNCHER` (its own peak RSS), its stdout kept, and copied to `log` as it comes."""
    with tempfile.NamedTemporaryFile(suffix=".json") as stats, tempfile.TemporaryFile() as err:
        p = subprocess.Popen([sys.executable, "-c", run.LAUNCHER, stats.name, *argv], stdout=subprocess.PIPE, stderr=err,
                             text=True)
        lines = []
        with open(log, "w") if log else open(os.devnull, "w") as f:
            for line in p.stdout:
                lines.append(line.rstrip("\n"))
                f.write(line)
                f.flush()
        p.wait()
        r = json.loads(Path(stats.name).read_text())
        err.seek(0)
        if r["rc"] != 0:
            r["stderr"] = err.read().decode()[-600:]
    return r | {"lines": lines}


def verdict_of(lines: list[str]):
    """The last line's verdict; a last line that isn't JSON (a refusal's message) as `{"unparsed": line}`."""
    last = next((x for x in reversed(lines) if x.strip()), None)
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"unparsed": last}


def small(v):
    """A verdict without its IU list past 8 IUs."""
    if isinstance(v, dict) and len(v.get("ius") or []) > 8:
        return v | {"ius": f"{len(v['ius'])} IUs", "ius_sha256": hashlib.sha256(json.dumps(v["ius"]).encode()).hexdigest()}
    return v


def check(args, name: str, path: Path, row: dict) -> dict:
    """`nci-phases` then `nci-check --policy` on the instance `path`, one row."""
    logs = Path(args.out) / "phases"
    logs.mkdir(parents=True, exist_ok=True)
    ph = launch([str(args.phases), str(path), "20", str(args.big)], logs / f"{name}.log")
    pv = verdict_of(ph.pop("lines"))
    phases = [json.loads(x[len("PHASE "):]) for x in (logs / f"{name}.log").read_text().splitlines() if x.startswith("PHASE ")]
    if args.no_policy:
        d = {"seconds": ph["seconds"], "max_rss_kb": ph["max_rss_kb"], "rc": ph["rc"], "skipped": True}
        dv = pv
    else:
        d = launch([str(args.exe), "--policy", str(path)])
        dv = verdict_of(d.pop("lines"))
    row = row | {"case": name, "desc": d, "verdict": small(dv), "phases_run": ph, "phases": phases, "agrees": pv == dv}
    with open(Path(args.out) / f"{args.cmd}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({"case": name, "seconds": round(d["seconds"], 2), "max_rss_kb": d["max_rss_kb"], "rc": d["rc"],
                      "agrees": row["agrees"], "verdict": str(small(dv))[:160]}), flush=True)
    return row


def suite(args) -> int:
    lock = json.loads((NCI / "lean-audit.json").read_text())
    exes = {e["root"]: e["name"] for e in tomllib.loads((NCI / "lakefile.toml").read_text())["lean_exe"]}
    data = Path(args.scratch) / "suite"
    data.mkdir(parents=True, exist_ok=True)
    bad = 0
    rows = []

    def sub(s: str) -> str:
        return s.replace("{repo}", str(run.ROOT)).replace("{out}", str(data))

    for main, spec in sorted(lock["runs"].items()):
        t = time.perf_counter()
        gen = [sys.executable if a == "python3" else sub(a) for a in spec["generate"]]
        subprocess.run(gen, cwd=NCI, check=True, capture_output=True)
        gen_s = time.perf_counter() - t
        argv = [str(NCI / ".lake/build/bin" / exes[main])] + [str(data / a[1:-1]) if a.startswith("{") else a for a in spec["args"]]
        r = launch(argv)
        tail = r.pop("lines")[-3:]
        files = {a[1:-1]: (data / a[1:-1]).stat().st_size for a in spec["args"] if a.startswith("{")}
        rows.append({"run": main, "generate_seconds": round(gen_s, 2), **r, "data_bytes": files, "tail": tail})
        bad += r["rc"] != 0
        print(json.dumps(rows[-1])[:400], flush=True)
    with open(Path(args.out) / "suite.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    res = [json.loads(x) for x in (data / "desc_difftest/python.jsonl").read_text().splitlines()]
    gates = [v.get("gates", 0) if isinstance(v, dict) else 0 for v in res]
    k = max(range(len(gates)), key=gates.__getitem__)
    inst = (data / "desc_difftest/instances.jsonl").read_text().splitlines()[k]
    path = data / "largest.json"
    path.write_text(inst)
    r = check(args, "largest-desc-difftest", path, {"line": k, "program_bytes": len(json.loads(inst)["program"])})
    bad += r["desc"]["rc"] != 0 or not r["agrees"]
    for line in (data / "desc_liveness/cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        path = data / "big.json"
        path.write_text(json.dumps(c["big"]))
        r = check(args, f"liveness-{c['id']}", path, {"program_bytes": len(c["big"]["program"])})
        bad += r["desc"]["rc"] != 0 or not r["agrees"]
    return 1 if bad else 0


def serve(args) -> int:
    ref = run._reference()
    bad = 0
    for name in args.config or ("tiny", "mid", "b0-1", "b0"):
        for ir in ("word", "boolean"):
            t = time.perf_counter()
            text = run.program(name, ir)
            inst, size = run.instance(ref, text)
            build_s = time.perf_counter() - t
            path = Path(args.scratch) / f"serve-{name}-{ir}.json"
            path.write_text(json.dumps(inst))
            del inst, text
            r = check(args, f"serve-{name}-{ir}", path, {"config": name, "ir": ir, "build_seconds": round(build_s, 1), **size})
            path.unlink()
            bad += r["desc"]["rc"] != 0 or not r["agrees"]
    return 1 if bad else 0


def called_once(fn):
    """A Definition of `fn`'s signature whose body is one Call of `fn`: a Program of it has a root of `fn`'s inputs and that
    Call."""
    from verity.ml.circuits.defs import CompositeDefinition, bind

    params, ret = fn.signature()
    return bind(CompositeDefinition("NciCostOnce", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a),
                                    register=False))


def peak_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def glm(args) -> int:
    from verity.ml.circuits import codec
    from verity.ml.circuits.program import Program
    from verity.ml.definitions import boolean as BB

    import lift

    path = Path(args.workload)
    desc, _ = nci_glm.load(path)
    text, size = nci_glm.described(desc)
    owner = nci_glm.call_owners(desc)
    nodes = sum(len(d.get("body", {}).get("nodes", [])) for d in desc["definitions"].values())
    del desc
    inst = nci_glm.write_instance(Path(args.scratch) / "glm-word.json", json.dumps(text), [], owner, nci_glm.GAMMA,
                                  nci_glm.ALPHA)
    del text
    bad = 0
    r = check(args, "glm-r0-word", inst, {"ir": "word", "body_nodes": nodes, **size})
    inst.unlink()
    bad += r["desc"]["rc"] != 0 or not r["agrees"]
    if args.word_only:
        return 1 if bad else 0

    t = time.perf_counter()
    prog, digest = BB.load_program(path)
    meta = {"word_digest": digest, "decode_seconds": round(time.perf_counter() - t, 1), "decode_peak_gb": peak_gb()}
    t = time.perf_counter()
    dry = BB.dry_run(prog)
    meta |= {"dry_run_seconds": round(time.perf_counter() - t, 1), "gaps": dry["gaps"],
             "non_boolean": {k: dry["non_boolean"][k] for k in ("definitions", "specializations")},
             "missing": [r for r in dry["derived_calls"] if r["gap"]][:40], "dry_run_peak_gb": peak_gb()}
    print(json.dumps({"glm-r0-boolean": "dry run"} | {k: meta[k] for k in ("gaps", "non_boolean", "dry_run_seconds")}),
          flush=True)
    req = prog.root.body.nodes[-1].fn
    if dry["gaps"]:
        meta["lifted"] = None
    else:
        t = time.perf_counter()
        bits = Program(called_once(lift.on_bits(req)))
        meta |= {"lift_seconds": round(time.perf_counter() - t, 1), "lift_peak_gb": peak_gb()}
        t = time.perf_counter()
        bdesc = codec.encode_program(bits)
        btext, bsize = nci_glm.described(bdesc)
        bowner = nci_glm.call_owners(bdesc)
        bnodes = sum(len(d.get("body", {}).get("nodes", [])) for d in bdesc["definitions"].values())
        meta |= {"encode_seconds": round(time.perf_counter() - t, 1), "encode_peak_gb": peak_gb(),
                 "boolean_gates": bits.gates}
        del bdesc, bits, prog, req
        inst = nci_glm.write_instance(Path(args.scratch) / "glm-bits.json", json.dumps(btext), [], bowner, nci_glm.GAMMA,
                                      nci_glm.ALPHA)
        del btext
        r = check(args, "glm-r0-boolean", inst, {"ir": "boolean", "body_nodes": bnodes, **bsize, "lift": meta})
        inst.unlink()
        bad += r["desc"]["rc"] != 0 or not r["agrees"]
    with open(Path(args.out) / "glm-lift.json", "w") as f:
        json.dump(meta, f, indent=1)
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nci_cost.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("suite", "serve", "glm"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", type=Path, required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--exe", type=Path, default=run.EXE)
    ap.add_argument("--config", action="append", choices=list(run.CONFIGS))
    ap.add_argument("--workload")
    ap.add_argument("--big", type=int, default=2**17, help="nci-phases times a body of this many nodes step by step")
    ap.add_argument("--no-policy", action="store_true", help="nci-phases only, its verdict standing for nci-check's")
    ap.add_argument("--word-only", action="store_true", help="glm: the word-level descriptor only, no Boolean lift")
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return {"suite": suite, "serve": serve, "glm": glm}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
