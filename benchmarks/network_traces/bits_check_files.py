"""Scratch: a scan round's Program and its three templates' circuit files, written as scan_zk.py's `program` writes
them for the honest round (`TEMPLATES`, the generator form, a group of one link-window), with nothing staged or proved.

    python benchmarks/network_traces/bits_check_files.py OUT [CLASS ...]      # default: test

For each class, ``OUT/CLASS/joint/program.json`` and ``OUT/CLASS/joint/{first,ingress,egress}/circuit.txt``, and
``OUT/CLASS/files.json``: each file's template, descriptor id, size, pin and lowering time. Lands nowhere."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scan_zk as SZ  # noqa: E402
import warden_zk as WZ  # noqa: E402


def main(argv: list[str]) -> int:
    out, names = Path(argv[0]), argv[1:] or ["test"]
    M = WZ._statement()
    for name in names:
        setup = SZ.SETUPS[name]
        T = setup.params.window
        t = time.time()
        pg = SZ.program(M, setup.cls, SZ.TEMPLATES, 1, T, out / name / "joint")
        files = {}
        for tp, sg in pg.stagers.items():
            path = sg.out / "circuit.txt"
            files[tp] = {"circuit": str(path), "bytes": path.stat().st_size, "pin": sg.st.pin,
                         "definition": sg.defn.id, "template": SZ.PA.template_id(sg.defn), "units": sg.n,
                         "base": sg.base, "k_log": sg.st.meta["k_log"], "vus_per_block": sg.st.meta["vus_per_block"],
                         "lower_s": sg.lower_s}
        rec = {"class": name, "T": T, "group": 1, "program": str(pg.program_file),
               "program_sha256": hashlib.sha256(pg.program_file.read_bytes()).hexdigest(),
               "program_bytes": pg.program_file.stat().st_size, "units": pg.units, "files": files,
               "total_s": round(time.time() - t, 1)}
        (out / name / "files.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        print("BITSFILES " + json.dumps(rec, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
