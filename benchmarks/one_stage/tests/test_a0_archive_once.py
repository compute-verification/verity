"""a0's Lean verify archives a run's circuit once (`a0.lean_verify`, ``st["archived"]``): every later member's archive
links the first's object instead of putting the circuit again, and the verifier, which holds each object to its key,
still reads it there. Under `a0.ARCHIVE_LINK` a member's own circuit file is linked into its archive and never copied."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pytest  # noqa: E402

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


def own(root: Path, k: int, archived: dict | None) -> tuple[dict, Path, str]:
    """Member `k` with a circuit file of its own, as a window's members have: its statement, session dir and the
    file's SHA-512."""
    circuit = root / f"circuit-{k}.txt"
    circuit.write_bytes(f"circuit {k} ".encode() * 1000)
    st, out = member(root, k, circuit, archived if archived is not None else {})
    if archived is None:
        del st["archived"]
    st[a0.ARCHIVE_LINK] = True
    return st, out, hashlib.sha512(circuit.read_bytes()).hexdigest()


@pytest.mark.parametrize("shared", [True, False])
def test_under_archive_link_a_members_own_circuit_is_linked_into_its_archive_and_never_put(tmp_path, shared):
    """Each member has a circuit file of its own, so none is archived by an earlier one. Under `a0.ARCHIVE_LINK` the
    archive's object is the member's file under a second name (one inode), `archive-put` is given no circuit, and the
    verifier still reads the object under its key and accepts; with or without the members' shared record."""
    lean, puts = tmp_path / "flock-verify", tmp_path / "puts.jsonl"
    lean.write_text(FAKE.format(python=sys.executable, puts=str(puts)))
    lean.chmod(0o755)
    archived: dict | None = {} if shared else None
    for k in range(2):
        st, out, key = own(tmp_path, k, archived)
        assert a0.lean_verify(str(lean), st, 8, out, zk=True)["accepted"]
        assert (out / "archive" / "sha512" / key).stat().st_ino == (st["dir"] / "circuit.txt").stat().st_ino
    assert [json.loads(x) for x in puts.read_text().splitlines()] == [["pub-8.bin"]] * 2


def test_under_archive_link_a_circuit_that_cannot_be_linked_is_put_as_without_it(tmp_path, monkeypatch):
    """Where the link can't be made (the archive on another filesystem), the member's circuit is put, a copy, and the
    verdict is the same; a wrong key would be the verifier's to refuse, since it holds each object to its key."""
    lean, puts = tmp_path / "flock-verify", tmp_path / "puts.jsonl"
    lean.write_text(FAKE.format(python=sys.executable, puts=str(puts)))
    lean.chmod(0o755)
    st, out, key = own(tmp_path, 0, {})

    def refuse(src, dst):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(a0.os, "link", refuse)
    assert a0.lean_verify(str(lean), st, 8, out, zk=True)["accepted"]
    assert [json.loads(x) for x in puts.read_text().splitlines()] == [["circuit.txt", "pub-8.bin"]]
    assert (out / "archive" / "sha512" / key).stat().st_ino != (st["dir"] / "circuit.txt").stat().st_ino
    assert st["archived"] and next(iter(st["archived"].values()))[0] == key
