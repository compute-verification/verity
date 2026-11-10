"""Sampled proofs v2's case (`verity/sampled-proofs/one-stage/v2`) of a served window from the files its run kept, for
`composite.py --cases`, and with ``--exe`` its own Lean answer on it.

    v2_case.py --window DIR --out DIR --scratch DIR [--exe PATH]

``--window`` is the window directory a5 writes and a served run keeps (``out/sampled-proofs/served/window``): its v1
case (``case.json``), the verifier's partition (``partition.json``), a5's ``summary.json`` (its ``members``, each ``k``,
``template``, ``n`` and ``base``, and each template's pin and units a proof block under ``circuits``), each member's
``m<k>/`` with its public file (the first of a template's members, its ``compose.log`` and its circuit file; a member
reading registered values, the verifier's ``registered.json``), each sampled member's ``session-m<k>/sessions/*/`` with
each session's ``session.json`` and ``*.bin``, and the window's registered circuit record ``circuit.json`` where the run
kept it. The case is a5.py's `case_v2` of those, in a5's form (`a5.v2_proof`): a window of one member, v1's params with
the partition and ``proofs: {circuit, registered}``; of several, the per-member form, every member's units, its
template's pin, units a proof block (the summary's, else its circuit file's META) and registered file, and a statement
a sampled member, which v2 refuses until its guarantee is in the lock
(`OneStageZk.MULTI_REFUSED`). A member reading registered values names ``circuit.json`` as its ``interface`` when its
SHA-512 is the summary's ``circuit``, as a5 does. Then v1's commitment, the evidence with the sessions' proofs and the
blobs' paths. A window without ``summary.json`` is ``m0`` alone, its pin its ``compose.log``'s.

A run that didn't keep a circuit file gets it rebuilt in ``--scratch`` by the command its template's ``compose.log``
records (`compose_circuit.py` over the verifier's synthetic input set, `run.sh`'s ``verity_numerical.bench.generate --n 4
--seed 1``): the circuit is public structure, and the file is used only if its SHA-512 is the pin, which the check
reads anyway. The case goes to ``--out`` private as `composite.py` reads it (0700, its files 0600): ``case.json``,
``params.json``, ``commitment.bin``, ``evidence.json`` and, with ``--exe`` (``sampled-proofs-one-stage-zk``),
``answer.json``. A blob with no file leaves only ``missing.json`` saying why, so `composite.py` leaves the protocol out.

Prints one ``V2CASE`` line (the pins, where the circuit files came from, the sessions and blobs) and, with ``--exe``, one
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
import compose_circuit as CM
from composite import canonical, write_private

from verity.core.protocols.sampled_proofs.one_stage import proofs as PF

ID = "verity/sampled-proofs/one-stage/v2"
TREE = Path(__file__).resolve().parents[2]
CASE_FILES = ("case.json", "params.json", "commitment.bin", "evidence.json", "answer.json")


def compose_command(log: str) -> dict:
    """What a member's ``compose.log`` records: the subcircuit, the pin, the input set's name (None for a circuit
    composed from a template, not an input set) and the partition and program digests `compose_circuit.py` was given."""
    cmd = log.splitlines()[0]
    arg = {k: (m.group(1) if (m := re.search(rf"--{k} ([0-9a-f]+)", cmd)) else None) for k in ("partition", "program")}
    sub, s = re.search(r'"subcircuit": "([^"]+)"', log), re.search(r"--input-set \S*/sets/([^/\s]+)(\s|$)", cmd)
    return {"subcircuit": sub and sub.group(1), "pin": re.search(r'"pin": "([0-9a-f]{128})"', log).group(1),
            "set": s and s.group(1), **arg}


def rebuild(c: dict, scratch: Path) -> tuple[Path | None, str]:
    """The circuit file `compose_command` `c` names, rebuilt in `scratch`, and how; None unless its SHA-512 is the pin."""
    if c["set"] is None or c["subcircuit"] is None:
        return None, "not rebuilt: its compose.log names no input set"
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
    """`case_v2` of `window`'s kept files (module docstring): the case, where its circuit files came from, and its
    sessions' count."""
    v1 = json.loads((window / "case.json").read_text())
    v1["params"]["partition"] = json.loads((window / "partition.json").read_text())
    names = json.loads(bytes.fromhex(v1["evidence"]))["sessions"]
    summary = json.loads((window / "summary.json").read_text()) if (window / "summary.json").is_file() else {}
    members = summary.get("members") or [{"k": 0, "template": None, "n": None, "base": 0}]
    home: dict[str | None, int] = {}
    for m in members:
        home.setdefault(m["template"], m["k"])
    circuits: dict[str | None, tuple[Path, str]] = {}

    def circuit(t: str | None) -> Path:
        """Template ``t``'s circuit file, in its first member's directory or rebuilt by its ``compose.log``."""
        if t not in circuits:
            d = window / f"m{home[t]}"
            f, how = d / "circuit.txt", "kept"
            if not f.is_file():
                got, how = rebuild(compose_command((d / "compose.log").read_text()), scratch / d.name)
                f = got or f
            circuits[t] = (f, how)
        return circuits[t][0]

    def pin(t: str | None) -> str:
        """Template ``t``'s pin: the summary's, else its first member's ``compose.log``'s."""
        got = ((summary.get("circuits") or {}).get(t) or {}).get("pin")
        return got or compose_command((window / f"m{home[t]}" / "compose.log").read_text())["pin"]

    def per_block(t: str | None) -> int:
        """Template ``t``'s units a proof block: the summary's, else its circuit file's META ``vus_per_block``."""
        got = ((summary.get("circuits") or {}).get(t) or {}).get("vus_per_block")
        return int(got if got is not None else CM.head(circuit(t))["vus_per_block"])

    one = len(members) == 1
    iface = window / "circuit.json"
    iface = iface if iface.is_file() and hashlib.sha512(iface.read_bytes()).hexdigest() == summary.get("circuit") else None
    own, statements, n, interfaces = [], [], 0, []
    for m in members:
        k, t, d = m["k"], m["template"], window / f"m{m['k']}"
        reads = json.loads((d / "registered.json").read_text()) if (d / "registered.json").is_file() else None
        named = reads is not None and iface is not None
        interfaces += [iface] if named else []
        own.append((k, m["base"], m["n"], None if one else per_block(t), {"circuit": pin(t), "registered": reads}
                    | ({"interface": PF.file_sha512(iface)} if named else {})))
        sessions = tuple(sorted(f.parent for f in (window / f"session-m{k}/sessions").glob("*/session.json")))
        if sessions or one:
            pub = next(iter(sorted(d.glob("pub-*.bin"))))
            statements.append((None if one else f"m{k}", PF.Proof(circuit(t), pub, sessions)))
            n += len(sessions)
    tables = [f for c, _ in circuits.values() if (c.parent / "tables").is_dir()
              for f in sorted((c.parent / "tables").glob("*")) if f.is_file()]
    v2 = a5.case_v2(v1, names, proofs=own[0][4] if one else a5.v2_members(own), statements=statements, tables=tables,
                    interfaces=interfaces)
    hows = [how if one else f"m{home[t]}: {how}" for t, (_, how) in circuits.items()]
    return v2, "; ".join(hows) or "no member drawn", n


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
    pr = v2["params"]["proofs"]
    pins = sorted({m["circuit"] for m in pr["members"].values()}) if "members" in pr else [pr["circuit"]]
    print("V2CASE " + json.dumps({"circuits": pins, "circuit_files": how, "sessions": n, "blobs": len(v2["blobs"]),
                                  "missing": gone}), flush=True)
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
