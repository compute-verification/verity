"""a0's Lean verify archives a run's files once (`a0.lean_verify`, `a0.archive_put`, ``st["archived"]``): every later
member's archive links the first's object instead of putting the file again, and the verifier, which holds each object
to its key, still reads it there. A run that puts its members' files before its sessions (`a0.archive_put` on
`a0.archive_files`) starts one process for them all, and no member's verify puts anything."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import a0  # noqa: E402

#: a stand-in for verity-verify-proof: ``archive-put`` as Lean's (each file under its SHA-512, logged to ``puts``); ``verify``
#: accepting when the circuit and public objects it is named hash to their keys, as `archiveGet` holds them, with
#: `verify`'s lines on standard error; and ``verify-members``, that check for each line of standard input, a member's
#: arguments, then its ``MEMBER`` line (`Flock.VerifyMembers`). A session's ``mode`` file makes the stand-in misbehave on it.
FAKE = '''#!{python}
import hashlib, json, pathlib, sys
args = sys.argv[1:]

def verify(args):
    session = pathlib.Path(args[args.index("--session") + 1])
    mode = (session / "mode").read_text() if (session / "mode").exists() else ""
    if mode == "usage":
        print("usage: verify ...", file=sys.stderr)
        return 2, mode
    arc = pathlib.Path(args[args.index("--archive") + 1]) / "sha512"
    ok = all(hashlib.sha512((arc / args[args.index(f) + 1]).read_bytes()).hexdigest() == args[args.index(f) + 1]
             for f in ("--circuit", "--public"))
    print("inputs: 5 ms\\nsetup: 1 ms\\n" + str(session) + ": 3 ms", file=sys.stderr)
    print("VERDICT " + json.dumps({{"accepted": ok, "why": None if ok else "archive", "session": str(session)}}))
    return 0 if ok else 1, mode

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
    sys.exit(verify(args[1:])[0])
elif args[0] == "verify-members":
    program = pathlib.Path(args[args.index("--archive") + 1]) / "sha512" / args[args.index("--program") + 1]
    if not program.is_file():
        print("program: no such object", file=sys.stderr)
        sys.exit(2)
    for index, line in enumerate(sys.stdin):
        status, mode = verify(json.loads(line))
        if mode == "dies":
            sys.exit(3)
        end = {{"exit": status, "index": index + (mode == "another place")}}
        print("MEMBER " + ("no JSON" if mode == "no JSON" else json.dumps(end, separators=(",", ":"))), flush=True)
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
    lean, puts = tmp_path / "verity-verify-proof", tmp_path / "puts.jsonl"
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


def window(tmp_path: Path) -> tuple[Path, Path, list[tuple[dict, Path]], dict]:
    """Three members of one circuit, each with its own public file and the window's one program; the second and third
    read one registered file and the tables."""
    lean, puts = tmp_path / "verity-verify-proof", tmp_path / "puts.jsonl"
    lean.write_text(FAKE.format(python=sys.executable, puts=str(puts)))
    lean.chmod(0o755)
    circuit, program, table = tmp_path / "circuit.txt", tmp_path / "program.json", tmp_path / "t.u32"
    for f, text in ((circuit, b"circuit text " * 1000), (program, b"{}"), (table, b"table")):
        f.write_bytes(text)
    archived: dict = {}
    members = [member(tmp_path, k, circuit, archived) for k in range(3)]
    for k, (st, _) in enumerate(members):
        st["program_file"] = program
        if k:
            (st["dir"] / "tables").mkdir()
            os.link(table, st["dir"] / "tables" / "t.u32")
            st["registered_file"] = st["dir"] / "registered.json"
            st["registered_file"].write_bytes(b"reads %d" % k)
    return lean, puts, members, archived


def test_a_window_put_before_its_sessions_is_one_process_and_every_member_links_it(tmp_path):
    lean, puts, members, archived = window(tmp_path)
    files = [f for st, _ in members for f in a0.archive_files(st, 8)]
    keys = a0.archive_put(str(lean), tmp_path / "lean-verify" / "archive", files, tmp_path / "archive-put.log", archived)
    assert [json.loads(x) for x in puts.read_text().splitlines()] == [
        ["circuit.txt", "pub-8.bin", "program.json", "pub-8.bin", "t.u32", "registered.json", "pub-8.bin", "registered.json"]]
    assert all(keys[f] == hashlib.sha512(f.read_bytes()).hexdigest() for f in files)
    assert [a0.lean_verify(str(lean), st, 8, out, zk=True)["accepted"] for st, out in members] == [True, True, True]
    assert len(puts.read_text().splitlines()) == 1 and not any((out / "archive-put.log").exists() for _, out in members)
    for st, out in members:
        mine = {keys[f] for f in a0.archive_files(st, 8)}
        assert {f.name for f in (out / "archive" / "sha512").iterdir()} == mine
        assert all((out / "archive" / "sha512" / k).samefile(tmp_path / "lean-verify" / "archive" / "sha512" / k) for k in mine)


def test_the_windows_registered_circuit_record_is_put_once_and_named_only_beside_registered_reads(tmp_path):
    """The window's one record (``st["interface_file"]``) is among a member's files, and `verify`'s ``--interface``,
    exactly where the member has registered reads: the window's put archives it once, each reading member links it and
    names it by its key, and the member with no registered reads is verified without it."""
    lean, puts, members, archived = window(tmp_path)
    record = tmp_path / "circuit.json"
    record.write_bytes(b"the window's registered circuit record")
    for st, _ in members:
        st["interface_file"] = record
    assert [a0.bound_files(st) for st, _ in members] == [["program"], *[["program", "registered", "interface"]] * 2]
    files = [f for st, _ in members for f in a0.archive_files(st, 8)]
    assert [record in a0.archive_files(st, 8) for st, _ in members] == [False, True, True]
    keys = a0.archive_put(str(lean), tmp_path / "lean-verify" / "archive", files, tmp_path / "archive-put.log", archived)
    (put,) = [json.loads(x) for x in puts.read_text().splitlines()]
    assert put.count("circuit.json") == 1 and keys[record] == hashlib.sha512(record.read_bytes()).hexdigest()
    assert [a0.lean_verify(str(lean), st, 8, out, zk=True)["accepted"] for st, out in members] == [True, True, True]
    assert len(puts.read_text().splitlines()) == 1
    for k, (st, out) in enumerate(members):
        said = (out / "lean-verify.log").read_text().splitlines()[0].split()
        assert ("--interface" in said) == bool(k), said
        if k:
            assert said[said.index("--interface") + 1] == keys[record]
            assert said.index("--registered") < said.index("--interface") < said.index("--session")
            assert (out / "archive" / "sha512" / keys[record]).samefile(tmp_path / "lean-verify" / "archive" / "sha512" / keys[record])
    # a negative that drops the reads drops the record with them, and one that drops the record keeps the reads
    st, out = members[1]
    none = {x: y for x, y in st.items() if x != "registered_file"}
    assert a0.bound_files(none) == ["program"] and record not in a0.archive_files(none, 8)
    assert a0.bound_files({x: y for x, y in st.items() if x != "interface_file"}) == ["program", "registered"]


def test_a_file_changed_or_new_since_the_window_put_is_put_by_the_verify_that_reads_it(tmp_path):
    """A negative's altered copy is a new file, and a file rewritten in place is no longer the one archived: each is put
    by its own verify, under its own log, and a file that is gone is left for Lean to refuse."""
    lean, puts, members, archived = window(tmp_path)
    a0.archive_put(str(lean), tmp_path / "lean-verify" / "archive", [f for st, _ in members for f in a0.archive_files(st, 8)],
                   tmp_path / "archive-put.log", archived)
    st, out = members[1]
    changed = out / "changed.json"
    changed.write_bytes(b"other reads")
    assert a0.lean_verify(str(lean), {**st, "registered_file": changed}, 8, out, zk=True, log="negative")["accepted"]
    st["registered_file"].write_bytes(b"rewritten, longer")
    assert a0.lean_verify(str(lean), st, 8, out, zk=True)["accepted"]
    assert [json.loads(x) for x in puts.read_text().splitlines()][1:] == [["changed.json"], ["registered.json"]]
    assert (out / "negative.archive-put.log").exists() and (out / "archive-put.log").exists()
    key = hashlib.sha512(b"rewritten, longer").hexdigest()
    assert (out / "archive" / "sha512" / key).read_bytes() == b"rewritten, longer"
    gone = tmp_path / "gone"
    assert a0.archive_put(str(lean), tmp_path / "other", [gone], tmp_path / "gone.log", archived) == {}


def test_a_held_object_that_is_gone_is_put_again(tmp_path):
    lean, puts, members, archived = window(tmp_path)
    st, out = members[0]
    first = a0.archive_put(str(lean), tmp_path / "first", a0.archive_files(st, 8), tmp_path / "first.log", archived)
    for f in (tmp_path / "first" / "sha512").iterdir():
        f.unlink()
    assert a0.archive_put(str(lean), out / "archive", a0.archive_files(st, 8), out / "archive-put.log", archived) == first
    assert [json.loads(x) for x in puts.read_text().splitlines()] == [["circuit.txt", "pub-8.bin", "program.json"]] * 2
    assert {f.name for f in (out / "archive" / "sha512").iterdir()} == set(first.values())


def test_without_a_shared_record_each_member_puts_its_circuit(tmp_path):
    lean, puts = tmp_path / "lean", tmp_path / "puts.jsonl"
    lean.write_text(FAKE.format(python=sys.executable, puts=str(puts)))
    lean.chmod(0o755)
    circuit = tmp_path / "circuit.txt"
    circuit.write_bytes(b"circuit text")
    for k in range(2):
        st, out = member(tmp_path, k, circuit, {})
        del st["archived"]
        assert a0.lean_verify(str(lean), st, 8, out, zk=True)["accepted"]
    assert [json.loads(x) for x in puts.read_text().splitlines()] == [["circuit.txt", "pub-8.bin"]] * 2
