"""Sampled proofs v2's case (`verity/sampled-proofs/one-stage/v2`) of a served window from the files its run kept, for
`composite.py --cases`, and with ``--exe`` its own Lean answer on it.

    v2_case.py --window DIR --out DIR --scratch DIR [--exe PATH]

``--window`` is the window directory `prove_path/run.sh` keeps (``out/sampled-proofs/served/window``): its v1 case
(``case.json``), the verifier's partition (``partition.json``), ``m0/`` with the public file, ``compose.log`` and the
circuit file, and ``session-m0/sessions/*/`` with each session's ``session.json`` and ``*.bin``. The case is a5.py's
`case_v2` of those: v1's params with the partition and ``proofs: {circuit, registered: null}``, v1's commitment, the
evidence with the sessions' proofs, and the blobs' paths.

A run that didn't keep the circuit file gets it rebuilt in ``--scratch`` by the command its ``m0/compose.log`` records
(`compose_circuit.py` over the verifier's synthetic input set, `run.sh`'s ``verity_numerical.bench.generate --n 4 --seed
1``): the circuit is public structure, and the file is used only if its SHA-512 is the log's pin, which the check reads
anyway. The case goes to ``--out`` private as `composite.py` reads it (0700, its files 0600): ``case.json``,
``params.json``, ``commitment.bin``, ``evidence.json`` and, with ``--exe`` (``sampled-proofs-one-stage-zk``),
``answer.json``. A blob with no file leaves only ``missing.json`` saying why, so `composite.py` leaves the protocol out.

Prints one ``V2CASE`` line (the pin, where the circuit file came from, the sessions and blobs) and, with ``--exe``, one
``V2`` line with the answer's verdict only: the whole answer is witness-adjacent (its query is the partition's digest) and
stays in ``--out``. Exits as the check does (0 accept, 1 reject, 2 undecided), 2 for a missing case, 0 without ``--exe``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import a5
from composite import canonical, write_private

from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

ID = "verity/sampled-proofs/one-stage/v2"
TREE = Path(__file__).resolve().parents[2]
CASE_FILES = ("case.json", "params.json", "commitment.bin", "evidence.json", "answer.json")


def compose_command(log: str) -> dict:
    """What ``m0/compose.log`` records: the subcircuit, the pin, the input set's name and the partition and program
    digests `compose_circuit.py` was given."""
    cmd = log.splitlines()[0]
    arg = {k: (m.group(1) if (m := re.search(rf"--{k} ([0-9a-f]+)", cmd)) else None) for k in ("partition", "program")}
    return {"subcircuit": re.search(r'"subcircuit": "([^"]+)"', log).group(1),
            "pin": re.search(r'"pin": "([0-9a-f]{128})"', log).group(1),
            "set": re.search(r"--input-set \S*/sets/([^/\s]+)(\s|$)", cmd).group(1), **arg}


def rebuild(c: dict, scratch: Path) -> tuple[Path | None, str]:
    """The circuit file `compose_command` `c` names, rebuilt in `scratch`, and how; None unless its SHA-512 is the pin."""
    scratch.mkdir(parents=True, exist_ok=True)
    sets, f = scratch / "sets", scratch / "circuit.txt"
    py = sys.executable
    try:
        subprocess.run([py, "-m", "verity_numerical.bench.generate", "--subcircuit", c["subcircuit"], "--n", "4",
                        "--seed", "1", "--out", str(sets)], check=True, capture_output=True, cwd=TREE)
        argv = [py, "benchmarks/one_stage/compose_circuit.py", "--input-set", str(sets / "sets" / c["set"]), "--out", str(f)]
        argv += [x for k in ("partition", "program") if c[k] for x in (f"--{k}", c[k])]
        subprocess.run(argv, check=True, capture_output=True, cwd=TREE)
    except subprocess.CalledProcessError as x:
        return None, f"rebuild failed (exit {x.returncode}): {x.stderr.decode().strip()[-400:]}"
    got = hashlib.sha512(f.read_bytes()).hexdigest()
    if got != c["pin"]:
        return None, f"rebuilt from compose.log's command, SHA-512 {got[:16]}…, not the pin {c['pin'][:16]}…"
    return f, f"rebuilt from compose.log's command, its SHA-512 the pin ({f.stat().st_size} bytes)"


def case(window: Path, scratch: Path) -> tuple[dict, str, int]:
    """`case_v2` of `window`'s kept files: the case, where its circuit file came from, and its sessions' count."""
    c = compose_command((window / "m0/compose.log").read_text())
    v1 = json.loads((window / "case.json").read_text())
    v1["params"]["partition"] = json.loads((window / "partition.json").read_text())
    names = json.loads(bytes.fromhex(v1["evidence"]))["sessions"]
    sessions = tuple(sorted(f.parent for f in (window / "session-m0/sessions").glob("*/session.json")))
    pub = next(iter(sorted((window / "m0").glob("pub-*.bin"))))
    circuit, how = window / "m0/circuit.txt", "kept"
    if not circuit.is_file():
        got, how = rebuild(c, scratch)
        circuit = got or circuit
    v2 = a5.case_v2(v1, names, proof=PF.Proof(circuit, pub, sessions), circuit=c["pin"], registered=None, tables=[])
    return v2, how, len(sessions)


def write(out: Path, v2: dict, how: str) -> list[str]:
    """The case in `out`, or only ``missing.json`` when a blob has no file; the blobs with none."""
    for f in (*CASE_FILES, "missing.json"):
        (out / f).unlink(missing_ok=True)
    gone = [b for b in v2["blobs"] if not Path(b).is_file()]
    if gone:
        why = f"no file for {', '.join(gone)} (circuit: {how})"
        write_private(out, {"missing.json": (json.dumps({"protocol": ID, "why": why}) + "\n").encode()})
        return gone
    write_private(out, {"case.json": json.dumps(v2, sort_keys=True).encode(), "params.json": canonical(v2["params"]),
                        "commitment.bin": bytes.fromhex(v2["commitment"]),
                        "evidence.json": bytes.fromhex(v2["evidence"])})
    return []


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True, help="where a circuit file the run didn't keep is rebuilt")
    ap.add_argument("--exe", help="sampled-proofs-one-stage-zk, for the case's own Lean answer")
    a = ap.parse_args(argv)
    v2, how, n = case(a.window, a.scratch)
    gone = write(a.out, v2, how)
    print("V2CASE " + json.dumps({"circuit": v2["params"]["proofs"]["circuit"], "circuit_file": how, "sessions": n,
                                  "blobs": len(v2["blobs"]), "missing": gone}), flush=True)
    if gone:
        return 2
    if not a.exe:
        return 0
    p = subprocess.run([a.exe, "check", *(str(a.out / f) for f in ("params.json", "commitment.bin", "evidence.json")),
                        *v2["blobs"]], capture_output=True, text=True, check=False)
    line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    if not line:
        print(f"V2 no answer (exit {p.returncode})", flush=True)
        return p.returncode or 1
    write_private(a.out, {"answer.json": (line + "\n").encode()})
    print(f"V2 {ID} {json.loads(line)['verdict']}", flush=True)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
