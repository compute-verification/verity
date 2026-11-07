"""flock_served.py VALUES KIND WORK [N]: one kind of a saved served step (`one_stage/a11.py --save-values`'s directory, or
its `.tar.gz`) staged as a11.py stages it: the kind's unit Definition as a flat statement (`CS.UnitStager` with no layout,
the unit's whole netlist in the circuit file), bound to the population program and partition of that one kind, with the
kind's first N saved units (a11's ``--per-kind``; all of them without N) under a11's salt key. So the files are a11's, byte
for byte while the stager is unchanged, and an a11 session over them verifies against them.

Writes WORK/circuit.txt, WORK/stmt/{inst-N.bin,pub-N.bin}, WORK/partition.json, WORK/program.json and WORK/build.json (the
unit's ANDs and nnz, k_log, the circuit's bytes and lines, the size of the Program descriptor the statement could be
checked on instead, and the build's seconds).
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import tarfile
import time
from pathlib import Path

ONE_STAGE = Path(__file__).resolve().parents[1] / "one_stage"
sys.path.insert(0, str(ONE_STAGE))

import a11  # noqa: E402
import a11_kinds as K  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.ml.circuits import codec  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402


def values_dir(src: Path, work: Path) -> Path:
    if src.is_dir():
        return src
    with tarfile.open(src) as t:
        t.extractall(work / "values", filter="data")
    return next(p.parent for p in (work / "values").rglob("units.json"))


def main() -> None:
    src, kind, work = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _, rows_of_kind, _ = a11.load_values(values_dir(src, work), [kind])
    kv = rows_of_kind[kind]
    defn, n = kv["unit"], min(kv["n"], int(sys.argv[4]) if len(sys.argv) > 4 else kv["n"])
    kv["rows"] = {k: v[:n] for k, v in kv["rows"].items()}
    kv["outs"] = {k: v[:n] for k, v in kv["outs"].items()}
    program = P.mixed_population_program([(defn, n)])
    query = P.template_instances_query([defn])
    part = P.partition(program, query)
    (_, base, _), = P.ranges(program, query)
    (work / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (work / "program.json").write_bytes(canonical_json(encode_program(program)))
    bound = {"partition": P.digest(part), "program": part["program"]}
    with K.no_cyclic_gc():
        st = CS.UnitStager(defn, f"unit/a11-{K.family(defn.id)}", kv["rows_spec"], K.out_rows(len(kv["outs"])),
                           circuit=work / "circuit.txt", bind=bound)
    t1 = time.time()
    salt_key = hashlib.sha256(f"verity/one-stage/a11/salts\0{defn.id}".encode()).digest()
    d = work / "stmt"
    got = st.stage(kv["rows"], d, salt_key, indices=range(base, base + n), set_name=f"{st.iset.name}-{kind}")
    (d / "inst.bin").rename(d / f"inst-{n}.bin")
    (d / "pub.bin").rename(d / f"pub-{n}.bin")
    with open(work / "circuit.txt", "rb") as f:
        lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b""))
    desc = encode_program(program)
    head = {"kind": kind, "definition": defn.id, "n": n, "ands": got["ands"], "unit_nnz": got["unit_nnz"],
            "k_log": st.meta.get("k_log"), "vus_per_block": st.meta.get("vus_per_block"), "pin": st.pin,
            "circuit_bytes": (work / "circuit.txt").stat().st_size, "circuit_lines": lines,
            "descriptor_bytes": codec.descriptor_size(desc), "build_seconds": round(t1 - t0, 1),
            "stage_seconds": round(time.time() - t1, 1),
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    (work / "build.json").write_text(json.dumps(head, indent=1))
    print("BUILT " + json.dumps(head), flush=True)


if __name__ == "__main__":
    main()
