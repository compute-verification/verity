"""CIR-274 JOINED RECORD: GLM-5.3 at TP8, one decoder layer, a 43-id prompt, one output: the eight ranks' windows built on
their boundary reads (`node_window_reads.rank_window`), joined (`node_window_join.join`), and the joined record written
as its circuit.json document with ``ranks`` (`ranks_of`), ``free`` (`free_rows`) and ``boundary_reads``, beside each
rank's own record. Then held to interface's readers (`check_ranks`, `ranks_of`, `free_of`, `boundary_reads_of`) and to
proofs' round-2 reader (`second_window`: `ranks_of`, `wires_ok`, `reads_ok`, `links_of`, and `failures` on the
`second_window` piece's input, which is written for the Lean's `spot-difftest` to read the same instance).

Usage (from adapters/vllm): joined_record.py OUT. Writes OUT/joined_record/ and OUT/summary.json."""
import argparse
import dataclasses
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

from verity.circuits.codec import canonical_json  # noqa: E402
from verity.commitments.periods import interface as IF  # noqa: E402
from verity.protocols.proofs.spot.verifier import second_window as SW  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_embed as NE  # noqa: E402
from verity_vllm.program.predict import node_window_join as NJ  # noqa: E402
from verity_vllm.program.predict import node_window_multimem as NM  # noqa: E402
from verity_vllm.program.predict import node_window_reads as NR  # noqa: E402
from verity_vllm.program.predict import node_window_select as NSEL  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
a = ap.parse_args()
T0 = time.perf_counter()
out = Path(a.out)
d = out / "joined_record"
d.mkdir(parents=True, exist_ok=True)
FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES, **NE.FAMILIES, **NSEL.FAMILIES}
IDS, LAYERS, WORLD, CAP = 43, 1, 8, 1
summary: dict = {"ids": IDS, "layers": LAYERS, "world": WORLD, "cap": CAP, "ranks": [], "checks": {}}


def H(rec) -> str:
    return hashlib.sha512(canonical_json(IF.circuit_of(rec))).hexdigest()


def save():
    summary["wall_s"] = round(time.perf_counter() - T0, 1)
    summary["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


def check(name, f):
    try:
        v = f()
        summary["checks"][name] = "ok" if v is None or v is True else v
    except Exception as e:  # noqa: BLE001 - a refusal is a result
        summary["checks"][name] = f"{type(e).__name__}: {e}"[:600]
    print("check", name, summary["checks"][name], flush=True)


ranks, records = [], []
for r in range(WORLD):
    t = time.perf_counter()
    x = NR.rank_window(r, glm_request(IDS, LAYERS, rank=r).program.fn, FAMILIES)
    rec = IF.descriptor_record(x.circuit, CAP)
    (d / f"rank{r}.circuit.json").write_bytes(IF.circuit_bytes(rec))
    ranks.append(x)
    records.append(json.loads(IF.circuit_bytes(rec)))
    summary["ranks"].append({"rank": r, "H": H(rec), "units": sum(x.circuit.units.values()), "wires": len(x.circuit.wires),
                             "boundary_reads": [b.doc() for b in x.circuit.boundary_reads], "free": len(x.free),
                             "free_kinds": sorted({f"{f.kind} {f.family}" for f in x.free}), "s": round(time.perf_counter() - t, 1)})
    print(r, {k: v for k, v in summary["ranks"][-1].items() if k != "boundary_reads"}, flush=True)
    save()

t = time.perf_counter()
j = NJ.join(ranks)
summary["join_s"] = round(time.perf_counter() - t, 1)
summary["unwired"] = list(j.unwired)
summary["free_inputs"] = [f"{f.tid} {f.port} [{f.at}, {f.at + f.count}) {f.kind} {f.name} {f.rows}" for f in j.free]
ranks_doc = NJ.ranks_of(ranks, j, CAP)
try:
    free = NJ.free_rows(j)
except Exception as e:  # noqa: BLE001
    free, summary["free_rows_refused"] = [], f"{type(e).__name__}: {e}"[:600]
stated_ranks = IF.Ranks.of_doc(ranks_doc)
stated = dataclasses.replace(j.circuit, ranks=stated_ranks, free=tuple(IF.Free.of(f) for f in free))
joined = json.loads(IF.circuit_bytes(IF.descriptor_record(stated, CAP)))
(d / "joined.circuit.json").write_bytes(IF.circuit_bytes(joined))
own = sum(len(x.circuit.wires) for x in ranks)
summary["joined"] = {"H": H(joined), "units": sum(j.circuit.units.values()), "templates": len(j.circuit.ids),
                     "wires": len(j.circuit.wires), "own_wires": own, "join_wires": len(j.circuit.wires) - own,
                     "boundary_reads": len(j.circuit.boundary_reads), "free": free, "unwired": len(j.unwired),
                     "fields": sorted(joined), "bytes": len(IF.circuit_bytes(joined))}
print("joined", {k: v for k, v in summary["joined"].items() if k != "free"}, flush=True)
save()

check("interface.check_ranks", lambda: IF.check_ranks(j.circuit.layouts, stated_ranks, j.circuit.wires, j.circuit.registered))
check("each rank's window is its record's H", lambda: [r.window for r in stated_ranks.of] == [x["H"] for x in summary["ranks"]])
check("interface.ranks_of(record)", lambda: IF.ranks_of(joined) == stated_ranks)
check("interface.free_of(record)", lambda: [f.doc() for f in IF.free_of(joined)] == free)
check("interface.boundary_reads_of(record)", lambda: IF.boundary_reads_of(joined) == j.circuit.boundary_reads)
sw = SW.ranks_of(joined.get("ranks"))
check("second_window.ranks_of", lambda: sw is not None)
check("second_window.wires_ok", lambda: SW.wires_ok(sw, joined, records))
check("second_window.reads_ok", lambda: SW.reads_ok(sw, joined, records))
check("second_window.links_of", lambda: [str(x) for x in (SW.links_of(joined) or [])])
served = [[x["H"], f"rank{x['rank']}"] for x in summary["ranks"]]
piece = {"allow": True, "given": {"dockets": [], "boundary": None, "verified": [], "refs": [], "ref_reads": []},
         "record": joined, "records": records, "served": served, "registrations": [], "root": "serve", "logged": [],
         "sample": [], "statements": []}
(d / "second_window.input.json").write_bytes(canonical_json({"fn": "second_window", "input": piece}) + b"\n")
check("second_window.failures (no serve: no registrations, sample or statements)",
      lambda: [c for c, _ in SW.failures(True, SW.Given(dockets=()), joined, records, [tuple(s) for s in served], [], "serve",
                                         lambda c: False, [], [])])
summary["files"] = {p.name: p.stat().st_size for p in sorted(d.iterdir())}
save()
print("done", summary["wall_s"], "s", summary["peak_rss_gib"], "GiB", flush=True)
