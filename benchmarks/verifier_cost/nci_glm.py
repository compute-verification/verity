"""`nci-check` on a vLLM Build's workload Programs, one per tensor-parallel rank (README.md, "nci_glm").

  python benchmarks/verifier_cost/nci_glm.py --build DIR --out DIR [--exe PATH] [--K BITS] [--token-bits B]

DIR holds the Build's tree (`research data fetch art:<vllm-build/v1> --to DIR`): DIR/build/rank<r>/build_workload/
descriptor.json.gz and workload_program.json. Each rank's workload Program is checked by `nci-check --policy`
(`NciCheck.Desc`) as two instances, both with θ the Program's own `annotations.theta` (the weights' input nodes) and one IU
per request call:

  as-built:    the descriptor's bytes as the Build wrote them;
  registered:  the same descriptor with θ's input nodes calling `RegisteredInput<w>_v1` in place of `Input<w>_v1`, and
               nothing else changed (the Program the Build would write if its components bound the weights at
               registration).

Γ and α are the node's, from the Build's own counts, for inference that reads `--token-bits` per token position: with
`S` the computed gates of every rank's request calls (`population.component_gates`, summed) and `P` the token positions
they run (prompt_len + cap - 1 per request), α = token_bits · P / S bits per gate, so the whole node's computation reading
its prompt alone is at the limit, and Γ = ⌊K / α⌋. DIR/results.json has per rank both verdicts, the non-θ inputs by group
(the prompt and the peers' collective values), and the node's figure with the peers computed inside it, which no Program
yet declares (`joined`: derived from the per-rank counts, not checked).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = HERE.parents[1] / "verity/core/protocols/network/nci/lean/.lake/build/bin/nci-check"
SCHEMA = "verity/nci-glm/v1"
INPUT_ID = re.compile(r"^(Registered)?Input([1-9][0-9]*)_v1$")

#: forks the verifier from a fresh interpreter, which reports its peak RSS (run.py's LAUNCHER)
LAUNCHER = ("import json, os, sys, time\n"
            "t = time.perf_counter()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    try:\n"
            "        os.execvp(sys.argv[2], sys.argv[2:])\n"
            "    finally:\n"
            "        os._exit(127)\n"
            "_, status, ru = os.wait4(pid, 0)\n"
            "with open(sys.argv[1], 'w') as f:\n"
            "    json.dump({'seconds': time.perf_counter() - t, 'max_rss_kb': ru.ru_maxrss,"
            " 'rc': os.waitstatus_to_exitcode(status)}, f)\n")


def timed(argv: list[str]) -> dict:
    """One run of `argv`: wall seconds, peak RSS and its stdout as JSON."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, \
            tempfile.NamedTemporaryFile(suffix=".json") as stats:
        subprocess.run([sys.executable, "-c", LAUNCHER, stats.name, *argv], stdout=out, stderr=err, check=True)
        r = json.loads(Path(stats.name).read_text())
        out.seek(0)
        err.seek(0)
        text, etext = out.read().decode(), err.read().decode()
    if r["rc"] != 0:
        return r | {"stderr": etext[-2000:]}
    return r | {"verdict": json.loads(text)}


def input_width(desc: dict, key: str) -> int | None:
    """The width of an input primitive's value, or None for any other Definition."""
    m = INPUT_ID.match(key)
    d = desc["definitions"].get(key, {})
    return int(m[2]) if m and d.get("form") == "primitive" else None


def declaration(desc: dict, wp: dict) -> tuple[list, list, dict]:
    """θ (the Program's own annotation), the owners (one IU per call node, in order) and the non-θ inputs by group: each
    request's parameter groups but the shared one, in order, over the root's non-θ input nodes."""
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    theta = list(desc["annotations"]["theta"])
    ts = set(theta)
    owner, u, free = [], 0, []
    for k, n in enumerate(nodes):
        w = input_width(desc, desc["callees"][n["fn"]])
        if w is None:
            owner.append(u)
            u += 1
        else:
            owner.append("input")
            if k not in ts:
                free.append((k, w, n.get("n", 1)))
    shared = wp["shared"]["group"]
    groups = [g for r in wp["requests"] for g, c in r["component"]["param_groups"].items() if g != shared for _ in range(c)]
    if len(groups) != len(free):
        raise SystemExit(f"{len(groups)} request-local parameters against {len(free)} non-theta input nodes")
    by: dict[str, dict] = {}
    for g, (_k, w, n) in zip(groups, free):
        key = "prompt" if g == "prompt" else "peers" if g.startswith("peers_") else g
        e = by.setdefault(key, {"nodes": 0, "leaves": 0, "bits": 0})
        e["nodes"] += 1
        e["leaves"] += n
        e["bits"] += n * w
    return theta, owner, by


def register(d: dict, theta: list) -> dict:
    """`d`, changed in place: θ's input nodes call `RegisteredInput<w>_v1` (appended to the callees) in place of
    `Input<w>_v1`."""
    nodes = d["definitions"][d["root"]]["body"]["nodes"]
    index: dict[str, int] = {}
    for k in theta:
        key = d["callees"][nodes[k]["fn"]]
        m = INPUT_ID.match(key)
        if m[1]:
            continue
        reg = f"RegisteredInput{m[2]}_v1"
        if reg not in index:
            d["definitions"].setdefault(reg, {"form": "primitive", "id": reg, "params": [], "ret": ["Value", int(m[2])]})
            if reg in d["callees"]:
                index[reg] = d["callees"].index(reg)
            else:
                index[reg] = len(d["callees"])
                d["callees"].append(reg)
        nodes[k]["fn"] = index[reg]
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exe", type=Path, default=EXE)
    ap.add_argument("--K", type=int, default=20_000_000, help="bits an IU may read besides θ (calibrations.json)")
    ap.add_argument("--token-bits", type=int, default=32, help="bits per token position at the limit")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ranks = sorted((p for p in (args.build / "build").glob("rank*") if (p / "build_workload").is_dir()),
                   key=lambda p: int(p.name[4:]))
    wps = [json.loads((r / "build_workload/workload_program.json").read_text()) for r in ranks]
    S = sum(wp["population"]["component_gates"] for wp in wps)
    P = sum(q["prompt_len"] + q["cap"] - 1 for q in wps[0]["requests"])
    num, den = args.token_bits * P, S
    gamma = args.K * den // num
    params = {"K": args.K, "token_bits": args.token_bits, "positions": P, "node_component_gates": S,
              "gamma": gamma, "alpha": [num, den], "alpha_float": num / den}
    print(json.dumps(params), flush=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for rdir, wp in zip(ranks, wps):
            text = gzip.open(rdir / "build_workload/descriptor.json.gz").read().decode()
            desc = json.loads(text)
            theta, owner, inputs = declaration(desc, wp)
            row = {"rank": int(rdir.name[4:]), "workload_digest": wp["workload_digest"],
                   "descriptor_sha256": hashlib.sha256(text.encode()).hexdigest(), "descriptor_bytes": len(text.encode()),
                   "root_nodes": len(owner), "theta_nodes": len(theta), "ius": max((o for o in owner if o != "input"),
                                                                                   default=-1) + 1,
                   "gates": wp["population"]["gates"], "component_gates": wp["population"]["component_gates"],
                   "non_theta_inputs": inputs}
            for name in ("as_built", "registered"):
                path = Path(tmp) / f"{rdir.name}-{name}.json"
                if name == "registered":
                    text = json.dumps(register(desc, theta), separators=(",", ":"))
                path.write_text(json.dumps({"program": text, "theta": theta, "owner": owner, "gamma": gamma,
                                            "alpha": [num, den]}))
                row[name] = timed([str(args.exe), "--policy", str(path)])
                path.unlink()
            del desc, text
            v = row["registered"].get("verdict") or {}
            if v.get("ius"):
                size, novel = v["ius"][0]
                row["registered"]["size_matches_build"] = size == row["component_gates"]
                row["registered"]["novel_over_limit"] = novel * den / (num * size) if size else None
                row["registered"]["novel_matches_inputs"] = novel == sum(e["bits"] for e in inputs.values())
            rows.append(row)
            print(json.dumps({"rank": row["rank"], "as_built": row["as_built"].get("verdict", row["as_built"]),
                              "registered": {k: v for k, v in row["registered"].items() if k != "verdict"}
                              | {"verdict": v}}), flush=True)
    prompt = rows[0]["non_theta_inputs"].get("prompt", {}).get("bits", 0)
    joined = {"note": "derived from the per-rank counts, not checked: no Program joins the ranks",
              "size": S, "novel_reads": prompt, "limit": num * S // den, "complies": prompt * den <= num * S and S <= gamma}
    result = {"schema": SCHEMA, "verifier": "nci-check --policy", "build": os.environ.get("NCI_GLM_BUILD"),
              "params": params, "ranks": rows, "joined": joined}
    (args.out / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    bad = [r["rank"] for r in rows for k in ("as_built", "registered") if r[k].get("rc") != 0]
    if bad:
        print("nci-check failed to run on ranks", sorted(set(bad)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
