"""ks_form.py FLOCK_WORK...: whether `verify --zk` takes each statement staged under each FLOCK_WORK/data, before any proving:
`one_stage.proofs.ks_form_why` and `template_why` on the circuit file's head, the Python mirror that
`verity/protocols/flock/prover/tests/test_lean_ks_form.py` holds to the Lean verifier's `ks-form` (seconds a statement). One TSV line a
statement: its directory, type, n, META's ``out_net`` (the net returning the outputs: ``unit``, or a tail stage the unit's
form refuses), whether the public file has shared rows, and the mirror's verdict (``mirror_zk``: ``accepted`` or the
refusal; the Lean verifier's own is its `ks-form`'s). A shared-row public file is Lean's alone to refuse
(`KsScope.sharedWhy`), which the mirror doesn't run, so its line says that check is open."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verity.protocols.spot.verifier.one_stage import proofs as PF

OPEN = "accepted by the mirror; shared rows: Lean's sharedWhy not run"


def form(d: Path) -> list[str]:
    st = json.loads((d / "stage.json").read_text())
    circuit, public = d / "circuit.txt", d / "pub.bin"
    with open(circuit, "rb") as f:
        f.readline()
        meta = f.readline().decode()
    out_net = json.loads(meta[len("META "):]).get("out_net", "unit") if meta.startswith("META ") else "-"
    with open(public, "rb") as f:
        shared = "shared_rows" in json.loads(f.readline())
    why = PF.ks_form_why(circuit, public) or PF.template_why(circuit)
    return [d.name, st.get("census") or st.get("unit", ""), str(st.get("n", "")), str(out_net), str(shared).lower(),
            why or (OPEN if shared else "accepted")]


def main(dirs: list[str]) -> None:
    print("dir\ttype\tn\tout_net\tshared_rows\tmirror_zk")
    for fw in dirs:
        for d in sorted(Path(fw, "data").iterdir()):
            if (d / ".done").exists():
                print("\t".join(form(d)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
