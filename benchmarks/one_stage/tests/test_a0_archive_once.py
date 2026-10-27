"""a0's Lean verify archives a run's circuit once (`a0.lean_verify`, ``st["archived"]``): every later member's archive
links the first's object instead of putting the circuit again, and the verifier, which holds each object to its key,
still reads it there."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import a0  # noqa: E402

#: a stand-in for flock-verify: ``archive-put`` as Lean's (each file under its SHA-512, logged to ``puts``), and ``verify``
#: accepting when the circuit and public objects it is named hash to their keys, as `archiveGet` holds them
FAKE = '''#!{python}
import hashlib, json, pathlib, sys
args = sys.argv[1:]
if args[0] == "archive-put":
    d = pathlib.Path(args[1]) / "sha512"
    d.mkdir(parents=True, exist_ok=True)
    with open({puts!r}, "a") as f:
        f.write(json.dumps([pathlib.Path(p).name for p in args[2:]]) + "\\n")
    for p in args[2:]:
        b = pathlib.Path(p).read_bytes()
        h = hashlib.sha512(b).hexdigest()
        (d / h).write_bytes(b)
        print(h + "  " + p)
elif args[0] == "verify":
    arc = pathlib.Path(args[args.index("--archive") + 1]) / "sha512"
    ok = all(hashlib.sha512((arc / args[args.index(f) + 1]).read_bytes()).hexdigest() == args[args.index(f) + 1]
             for f in ("--circuit", "--public"))
    print("VERDICT " + json.dumps({{"accepted": ok, "why": None if ok else "archive"}}))
'''


def member(root: Path, k: int, circuit: Path, archived: dict) -> tuple[dict, Path]:
    st = {"dir": root / f"m{k}", "archived": archived}
    st["dir"].mkdir()
    os.link(circuit, st["dir"] / "circuit.txt")
    (st["dir"] / "pub-8.bin").write_bytes(bytes([k]) * 16)
    out = root / f"session-m{k}"
    (out / "sessions" / "s").mkdir(parents=True)
    (out / "sessions" / "s" / "session.json").write_text("{}")
    return st, out


def test_a_runs_circuit_is_archived_once_and_read_from_every_members_archive(tmp_path):
    lean, puts = tmp_path / "flock-verify", tmp_path / "puts.jsonl"
    lean.write_text(FAKE.format(python=sys.executable, puts=str(puts)))
    lean.chmod(0o755)
    circuit = tmp_path / "circuit.txt"
    circuit.write_bytes(b"circuit text " * 1000)
    key = hashlib.sha512(circuit.read_bytes()).hexdigest()
    archived: dict = {}
    verdicts = []
    for k in range(3):
        st, out = member(tmp_path, k, circuit, archived)
        verdicts.append(a0.lean_verify(str(lean), st, 8, out, zk=True))
    assert [v["accepted"] for v in verdicts] == [True, True, True]
    assert [json.loads(x) for x in puts.read_text().splitlines()] == [["circuit.txt", "pub-8.bin"], ["pub-8.bin"], ["pub-8.bin"]]
    objects = [tmp_path / f"session-m{k}" / "archive" / "sha512" / key for k in range(3)]
    assert len({o.stat().st_ino for o in objects}) == 1


def test_without_a_shared_record_each_member_puts_its_circuit(tmp_path):
    lean, puts = tmp_path / "flock-verify", tmp_path / "puts.jsonl"
    lean.write_text(FAKE.format(python=sys.executable, puts=str(puts)))
    lean.chmod(0o755)
    circuit = tmp_path / "circuit.txt"
    circuit.write_bytes(b"circuit text")
    for k in range(2):
        st, out = member(tmp_path, k, circuit, {})
        del st["archived"]
        assert a0.lean_verify(str(lean), st, 8, out, zk=True)["accepted"]
    assert [json.loads(x) for x in puts.read_text().splitlines()] == [["circuit.txt", "pub-8.bin"]] * 2
