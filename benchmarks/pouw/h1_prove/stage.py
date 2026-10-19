"""stage.py K WORK: the H1 tile unit's statement circuit, built once, and each requested round staged on it (the prover's).

The circuit (`h1unit.Shape(K).stager`, written once to WORK/circuit.txt) is the long step at K = 4096, so it is built while
the window is committed, served and drawn: `run.py` then drops a request in WORK/requests/ for each statement, ``NAME.json``
``{"rows": rows.npz, "set": SET, "out": DIR}`` (`window.py open`'s rows, the drawn units' instances in order), and this
stages it with C-Flock's `UnitStager.stage` under the window's own salts, so the statement's committed rows are the
window's leaves. DIR gets inst.bin, pub.bin (the prover's) and stage.json, the last one written. WORK/requests/END stops it.

The stager's check of the outputs against the IR reference (`UnitStager.reference_check`, the Boolean Definition's
bit-sliced evaluation, hours at K = 4096) is replaced by `h1tile.reference`, the word reference, on each instance's codes,
salt, position and roots: the staged z and receipt are the lowered circuit's, held to that reference word for word.
`HZ_STAGE` as `hidden_zk/stage.py`'s: ``fast`` (`fast_stage`, the default) or ``library``.
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

NAME = "unit/pouw-h1-tile"


def _rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def _ports(ps) -> list:
    return [(p.name, p.words, p.bits, p.schema) for p in ps]


def check_against_reference(rows: dict, outs: dict) -> None:
    """Each instance's staged z and receipt words are `h1tile.reference`'s on its codes, salt, position and roots."""
    from verity.core.protocols.pouw.circuit import h1tile as H
    salt, units, roots = bytes(rows["salt"]), rows["units"], rows["roots"]
    for i, u in enumerate(units):
        z, rc = H.reference([[int(x) & 0xFF for x in r] for r in rows["a"][i]], [[int(x) & 0xFF for x in r] for r in rows["b"][i]],
                            salt, int(u), (bytes(roots[i][0]), bytes(roots[i][1])))
        if not (np.array_equal(outs["o0"][i], U.z_words(z)) and np.array_equal(outs["o1"][i], U.receipt_words(rc))):
            raise AssertionError(f"{NAME}: instance {i} (unit {int(u)}): the lowered circuit's outputs are not h1tile.reference's")


def main() -> None:
    k, work = int(sys.argv[1]), Path(sys.argv[2])
    stage = os.environ.get("HZ_STAGE", "fast")
    (work / "requests").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if stage == "fast":
        import fast_stage
        fast_stage.install(work / "scratch")
    shape = U.Shape(k)
    stager = shape.stager(NAME, circuit=work / "circuit.txt")
    if _ports(stager.ports) != _ports(shape.ports()):
        raise SystemExit(f"the statement's ports {_ports(stager.ports)} are not h1unit.Shape's {_ports(shape.ports())}")
    counts = stager.low.layout.counts
    head = {"definition": shape.definition().id, "k": k, "r": shape.r, "c": shape.c, "name": NAME, "subcircuit": stager.sub.id,
            "meta": stager.meta, "pin": stager.pin, "class": stager.class_sha512, "ands": counts["ands"],
            "k_log": stager.meta["k_log"], "vus_per_block": stager.meta["vus_per_block"],
            "circuit_bytes": (work / "circuit.txt").stat().st_size, "staging": stage,
            "build_seconds": round(time.time() - t0, 1), "maxrss_gb": _rss_gb()}
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
        salts = {p.name: [bytes(s) for s in z[f"s_{p.name}"]] for p in stager.ports}
        stager.reference_check = lambda _rows, outs, z=z: check_against_reference(z, outs)
        try:
            got = stager.stage(rows, out, os.urandom(32), salts=salts, set_name=r["set"])
        except Exception as e:  # noqa: BLE001 - a refused round is reported in its stage.json, and the next one staged
            (out / "stage.json").write_text(json.dumps({"request": req.stem, "error": f"{type(e).__name__}: {e}"}))
            print(f"FAILED {req.stem}: {type(e).__name__}: {e}", flush=True)
            continue
        same = {p: bool(np.array_equal(got["outs"][p], z[f"w_{p}"])) for p in ("o0", "o1")}
        rec = {"request": req.stem, "set": r["set"], "units": [int(u) for u in z["units"]], "instances": len(z["units"]),
               "circuit_sha512": got["circuit_sha512"], "ands": got["ands"], "outputs_equal_committed": same,
               "seconds": round(time.time() - t1, 1), "maxrss_gb": _rss_gb()}
        (out / "stage.json").write_text(json.dumps(rec, indent=1))
        print("STAGED " + json.dumps(rec), flush=True)
    print("END", flush=True)


if __name__ == "__main__":
    main()
