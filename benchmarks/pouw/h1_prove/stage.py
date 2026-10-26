"""stage.py K WORK: the H1 tile unit's statement circuit, built once, and each requested round staged on it (the prover's).

The circuit (`h1unit.Shape(K).stager`, written once to WORK/circuit.txt) is the long step at K = 4096, so it is built while
the window is served and drawn: once `run.py` writes the window's public shape (WORK/window.json, ``{"k", "slot",
"calls"}``), the statement is bound to the window's Program and template-instance partition (`h1unit.Shape.window`), so
the Lean verifier derives each instance's unit from its own copy of them. `run.py` then drops a request in WORK/requests/
for each statement, ``NAME.json`` ``{"rows": rows.npz, "set": SET, "out": DIR, "indices": [...]}`` (`window.py open`'s
rows, the drawn units' instances in order, and their indices in the Program), and this stages it with C-Flock's
`UnitStager.stage` under the window's own salts as a shared-row statement (`h1unit.statement_tables`): each A port's, B
port's and the record's table the registered rows its instances read, with their value, positions and paths (one call's
units a statement: its A table, z table and b/<K>), and the ok row's at the verifier's registered ``ok``
(`h1unit.register_ok`), so the statement's committed rows are the registered values' and the window's leaves. DIR gets inst.bin, pub.bin (the prover's), outs.npz
(the staged output rows, ``o0`` and ``o1`` per instance) and stage.json, the last one written, which says at which
instances each staged output row is not the one the rows commit (``outputs_differ``: an honest window's none, a changed
record's its ok). WORK/requests/END stops it.

The stager's check of the outputs against the IR reference (`UnitStager.reference_check`, the Boolean Definition's
bit-sliced evaluation, hours at K = 4096) is replaced by `h1zy.reference`, the word reference, on each instance's codes,
scales, record, salt, position and roots: the staged receipt and ok are the lowered circuit's, held to that reference
word for word. `HZ_STAGE` as `hidden_zk/stage.py`'s: ``fast`` (`fast_stage`, the default) or ``library``. ``H1_TMPL``
(`h1unit.layout`): unset or ``flat``, the flat statement, which `job.sh` stages; ``structural``, the templated statement of
the unit's structural lowering (`h1unit.stager`).
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hidden_zk"))

import h1unit as U  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402

NAME = "unit/pouw-h1-tile"


def _rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def _ports(ps) -> list:
    return [(p.name, p.words, p.bits, p.schema) for p in ps]


def check_against_reference(rows: dict, outs: dict, r: int = U.TILE[0], c: int = U.TILE[1]) -> None:
    """Each instance's staged receipt and ok words are `h1zy.reference`'s (over v3's tile under ``H1_TILE=v3``) on its
    rows' codes and scales (`h1unit.split_row`), its record (`h1unit.split_record`), salt, position and roots."""
    from verity.core.protocols.pouw.circuit import h1zy as Z
    salt, units, roots = bytes(rows["salt"]), rows["units"], rows["roots"]
    for i, u in enumerate(units):
        a = [U.split_row(rows[f"w_a{x}"][i]) for x in range(r)]
        b = [U.split_row(rows[f"w_b{x}"][i]) for x in range(c)]
        rc, ok = Z.reference([x for x, _s in a], [x for x, _s in b], [s for _x, s in a], [s for _x, s in b],
                             U.split_record(rows["w_zy"][i], r * c), salt, int(u), (bytes(roots[i][0]), bytes(roots[i][1])),
                             stream=U.tile() == "v3")
        if not (np.array_equal(outs["o0"][i], U.receipt_words(rc)) and np.array_equal(outs["o1"][i], [ok])):
            raise AssertionError(f"{NAME}: instance {i} (unit {int(u)}): the lowered circuit's outputs are not h1zy.reference's")


def main() -> None:
    k, work = int(sys.argv[1]), Path(sys.argv[2])
    stage = os.environ.get("HZ_STAGE", "fast")
    (work / "requests").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if stage == "fast":
        import fast_stage
        fast_stage.install(work / "scratch")
    shape = U.Shape(k)
    while not (work / "window.json").exists():
        if (work / "requests" / "END").exists():
            print("END before the window's shape", flush=True)
            return
        time.sleep(2)
    win = json.loads((work / "window.json").read_text())
    if win["k"] != k:
        raise SystemExit(f"the window's K is {win['k']}, the stager's {k}")
    program, query = shape.window([tuple(c) for c in win["calls"]], win["slot"])
    part = P.partition(program, query)
    t_bind = time.time()
    stager = shape.stager(NAME, circuit=work / "circuit.txt", bind={"partition": P.digest(part), "program": part["program"]})
    if _ports(stager.ports) != _ports(shape.ports()):
        raise SystemExit(f"the statement's ports {_ports(stager.ports)} are not h1unit.Shape's {_ports(shape.ports())}")
    counts = stager.low.layout.counts
    head = {"definition": shape.definition().id, "k": k, "r": shape.r, "c": shape.c, "name": NAME, "subcircuit": stager.sub.id,
            "statement": stager.meta["statement"], "layout": stager.meta.get("layout"), "meta": stager.meta, "pin": stager.pin, "class": stager.class_sha512, "ands": counts["ands"],
            "k_log": stager.meta["k_log"], "vus_per_block": stager.meta["vus_per_block"],
            "circuit_bytes": (work / "circuit.txt").stat().st_size, "staging": stage,
            "build_seconds": round(time.time() - t_bind, 1), "waited_for_window_seconds": round(t_bind - t0, 1),
            "maxrss_gb": _rss_gb()}
    (work / "pin.txt").write_text(stager.pin)
    (work / "circuit.json").write_text(json.dumps(head, indent=1))
    print("BUILT " + json.dumps({x: head[x] for x in head if x != "meta"}), flush=True)
    done: set[str] = set()
    while True:
        reqs = sorted(p for p in (work / "requests").glob("*.json") if p.name not in done)
        if not reqs:
            if (work / "requests" / "END").exists():
                break
            time.sleep(2)
            continue
        req = reqs[0]
        done.add(req.name)
        r = json.loads(req.read_text())
        t1 = time.time()
        out = Path(r["out"])
        z = dict(np.load(r["rows"]))
        rows = {p: z[f"w_{p}"] for p in stager.row_ports}
        stager.reference_check = lambda _rows, outs, z=z: check_against_reference(z, outs, shape.r, shape.c)
        try:
            if [int(u) for u in r["indices"]] != [int(u) for u in z["units"]]:
                raise ValueError(f"the request's indices {r['indices']} are not its rows' units {z['units'].tolist()}")
            st = U.statement_tables(z, stager.row_ports, [p.name for p in stager.ports if p.name not in stager.row_ports])
            got = stager.stage(rows, out, os.urandom(32), salts=st["salts"], indices=r["indices"], set_name=r["set"],
                               tables=st["tables"], refs=st["refs"], registered=st["registered"])
        except Exception as e:  # noqa: BLE001 - a refused round is reported in its stage.json, and the next one staged
            (out / "stage.json").write_text(json.dumps({"request": req.stem, "error": f"{type(e).__name__}: {e}"}))
            print(f"FAILED {req.stem}: {type(e).__name__}: {e}", flush=True)
            continue
        np.savez(out / "outs.npz", **got["outs"])
        differ = {p: [i for i in range(len(z["units"])) if not np.array_equal(got["outs"][p][i], z[f"w_{p}"][i])]
                  for p in ("o0", "o1")}
        rec = {"request": req.stem, "set": r["set"], "units": [int(u) for u in z["units"]], "instances": len(z["units"]),
               "circuit_sha512": got["circuit_sha512"], "ands": got["ands"], "outputs_differ": differ,
               "registered_table_rows": {p: len(e["positions"]) for p, e in st["registered"].items()},
               "seconds": round(time.time() - t1, 1), "maxrss_gb": _rss_gb()}
        (out / "stage.json").write_text(json.dumps(rec, indent=1))
        print("STAGED " + json.dumps(rec), flush=True)
    print("END", flush=True)


if __name__ == "__main__":
    main()
