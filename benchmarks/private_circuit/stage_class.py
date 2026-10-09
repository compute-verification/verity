"""Stage one private circuit in a `UniversalUnit_v1{G, N_IN, N_OUT}` class, measuring each phase's wall time and memory.

    python benchmarks/private_circuit/stage_class.py --cls 8090,64,32 --gates F32MulFtz_v3.gates.json --n 2 --out DIR

The circuit's description (``--gates``: a `universal.Netlist` as JSON, ``n_in``, ``gates``, ``outputs``; without it a random
circuit of 3/4 of the class, as `universal.py` builds) is padded to the class (`universal.pad`) and staged as `universal.py`
stages it: a one-Call Program of the unit, `Q_template_instance`, one class statement (`class_statement.stage`, words packed) of
``n`` instances on random ``x``. Agreement before staging is the transcript's outputs against `universal.interpret`; staging
checks the laid-out rows against them. The whole-unit reference evaluations `universal.py` adds (`evaluate`, `evaluate_batch`)
are skipped: at the largest classes they cost more than staging.

``DIR/phases.jsonl`` gets one line per phase as it ends: wall seconds, the process's resident and peak resident GiB, and
``DIR/row.json`` the class, the unit's ANDs, rows and nonzeros and the statement's size. ``--keep-stage`` keeps the staged files; ``--stage-dir`` stages
into another directory, kept (a large class's files belong outside a run dir).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import resource
import secrets
import shutil
import sys
import time
from pathlib import Path

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.defs import bind
from verity.ml.private_circuits import universal as U
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universal import SPEC, _netlist, lanes  # noqa: E402


def _rss_gib() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize() / 2**30


def _peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


class Phases:
    def __init__(self, path: Path):
        self.path, self.t0, self.t = path, time.monotonic(), time.monotonic()
        self.rows: list[dict] = []

    def end(self, name: str, **extra) -> None:
        now = time.monotonic()
        row = {"phase": name, "s": round(now - self.t, 2), "total_s": round(now - self.t0, 2), "rss_gib": round(_rss_gib(), 3),
               "peak_gib": round(_peak_gib(), 3), **extra}
        self.rows.append(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        self.t = now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, help="G,N_IN,N_OUT")
    ap.add_argument("--gates", default=None, help="the circuit's description, a universal.Netlist as JSON")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-stage", action="store_true")
    ap.add_argument("--stage-dir", default=None, help="stage here (kept) instead of DIR/stage")
    a = ap.parse_args()
    G, N_IN, N_OUT = (int(v) for v in a.cls.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ph = Phases(out / "phases.jsonl")
    rng = random.Random(f"pc-stage/{a.seed}/{G}/{N_IN}/{N_OUT}")
    if a.gates:
        j = json.loads(Path(a.gates).read_text())
        small = U.Netlist(j["n_in"], tuple(map(tuple, j["gates"])), tuple(j["outputs"]))
        name = j.get("definition", Path(a.gates).stem)
    else:
        small, name = _netlist(rng, G, N_IN, N_OUT), "random"
    if len(small.outputs) != N_OUT:
        raise SystemExit(f"the circuit has {len(small.outputs)} outputs, the class {N_OUT}")
    net = U.pad(small, G, N_IN)
    xs = [[rng.randrange(2) for _ in range(N_IN)] for _ in range(a.n)]
    want = [U.interpret(net, x) for x in xs]
    ph.end("netlist", circuit=name, gates=len(small.gates), padded=G - len(small.gates),
           flock_plain_leaves=os.environ.get("FLOCK_PLAIN_LEAVES"), leaf_scheme=CS.LEAF_SCHEME)

    fn = bind(U.UniversalUnit, G=G, N_IN=N_IN, N_OUT=N_OUT)
    tr = U.traced(G, N_IN, N_OUT)
    ph.end("trace", ands=tr.counts["and"], xors=tr.counts["xor"], nots=tr.counts["not"], gates=len(tr.gates))
    program = PU.one_call(fn)
    query = PO.template_instance_query(fn)
    units = PU.PartitionUnits(program, query=query)
    [(shape, cl)] = units.classes().items()
    ph.end("units")
    ins, outs, idents = lanes(units, cl, net, xs)
    if [[outs[j][t] for j in range(N_OUT)] for t in range(a.n)] != want:
        raise AssertionError("the transcript's outputs disagree with universal.interpret")
    ph.end("lanes")
    gc.collect()
    d = Path(a.stage_dir) if a.stage_dir else out / "stage"
    salt_key = secrets.token_bytes(32)
    st = CS.stage(units, SPEC, shape, cl, (ins, outs, idents), d, salt_key, packed=True)
    if st["n"] != a.n:
        raise AssertionError(f"staging capped the instances at {st['n']}, not {a.n}")
    ph.end("stage", k_log=st["k_log"], unit_rows=st["unit_rows"])
    row = {"class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT, "W": N_IN + G, "K": U.index_bits(G, N_IN)},
           "circuit": {"name": name, "gates": len(small.gates), "inputs": small.n_in, "outputs": len(small.outputs)},
           "unit": {"ands_closed_form": U.and_count(G, N_IN, N_OUT), "ands_traced": tr.counts["and"], "ands_lowered": st["ands"],
                    "unit_rows": st["unit_rows"], "unit_nnz": st["unit_nnz"]},
           "statement": {k: st[k] for k in ("n", "k_log", "vus_per_block", "w_in", "w_out", "ports", "shared_rows", "circuit_sha512",
                                            "class_sha512", "circuit_bytes")},
           "files": {p.name: p.stat().st_size for p in sorted(d.iterdir()) if p.is_file()},
           "phases": ph.rows, "peak_gib": round(_peak_gib(), 3), "wall_s": round(time.monotonic() - ph.t0, 2)}
    (out / "row.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: row[k] for k in ("class", "unit", "peak_gib", "wall_s")}), flush=True)
    if not (a.keep_stage or a.stage_dir):
        shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
