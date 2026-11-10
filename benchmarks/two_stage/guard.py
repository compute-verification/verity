"""What SPOT[Flock]'s two-stage Lean model (`SpotFlock.stmtOk`) reads of a run of the two-stage toy, and the auditor's
copies it reads them against (`SpotFlock.Held`), as `tests/TwoGuard.lean` reads them.

`export` writes, from a run's ``--out`` (`audit.py`) and its stream:

* ``x.bin`` and ``m1.bin``: the boundary and interior registrations' records, as registered (`registration.json`);
* ``m2.bin``: round 2's message, each template's session statement as `FlockDeploy.Raw.encode` of what Flock's Lean
  verifier read for it (its verify log's command line, the objects of its session's archive, its record's
  ``unit_draw``), in template order, as `FlockDeploy.encList (encBytes ·)`;
* ``verdicts.json``: Flock's Lean verifier's verdict on each of those statements, in template order (`verdict`);
* ``c0.bin`` and ``c1.bin``: the bytes of the challenger's coins each stage's sample read (`service.Stream`'s);
* ``ref-<t>.bin``: template ``t``'s reference statement (`Held.xS₀`), made from what is fixed before the run: the
  circuit and public file of its reference (`toy.references`), the reference window's partition and program as a5
  writes a session's, ``--zk``, one table, and the sample of the second stage's size at its first units
  (`ref_sample`), with no registered reads: Flock's verifier parses a statement's as a ``--registered`` file, roots
  and rows included, and the params' name none of the run's fields;
* ``ref-<t>.reads.json``: template ``t``'s registered reads as the params name them (``proofs.json``'s
  ``templates[t].registered``, canonical JSON as a0 writes a session's ``registered.json``), ``null`` where they don't;
* ``held.json``: the rest of the `Held`, each from the registration and the design: the first stage's `select` input
  (rule, program, query, work table, closure), the second stage's cells ``ns`` and counts ``ks``, the templates'
  names, the design's links (`second_stage.replay_links` at the boundary window's own port names), wires
  (`second_stage.replay_wires`) and the rows their units read (`second_stage.replay_rows`), each derived from the
  replay Definition, each template's units ``vus``, the statement program's descriptor ``desc`` and the check's query.

Nothing in ``held.json`` or a ``ref-<t>.*`` is read from the run's records, sessions or coins; `export` checks the
links, the first stage's input, each reference's partition and program and the run's records against what the run
used, and raises where they differ.

``python3 guard.py pack RUN_OUT FIXTURE`` packs a run's ``<case>/guard/`` for both cases into the guard test's fixture
(`tests/test_guard.py`); ``python3 guard.py flip M2 OUT`` writes the test's flipped round-2 message (`flip_registered`);
``python3 guard.py refs GUARD`` moves the registered reads out of each ``ref-<t>.bin`` of an export that put them there
into its ``ref-<t>.reads.json`` (`split_refs`).
"""
from __future__ import annotations

import hashlib
import json
import lzma
import re
import shlex
import struct
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: `Flock/Lookup.lean`, whose `MUFU_TABLES` Flock's Lean verifier reads from its `--archive` by SHA-512, in its order
LOOKUP = ROOT / "verity" / "core" / "service" / "flock" / "Flock" / "Lookup.lean"
#: `FlockDeploy.FORMAT`, a statement encoding's first field
FORMAT = "flock-verify-statement/v1"
#: the flags of `verify` a statement's encoding carries no field for, or none this export reads
UNREAD = ("--work", "--work-table", "--stratified", "--closure", "--public-inputs", "--comp", "--identifier")


def enc_nat(n: int) -> bytes:
    return struct.pack("<Q", n)


def enc_bytes(b: bytes) -> bytes:
    return enc_nat(len(b)) + b


def enc_str(s: str) -> bytes:
    return enc_bytes(s.encode())


def enc_bool(b: bool) -> bytes:
    return b"\x01" if b else b"\x00"


def enc_opt(enc, v) -> bytes:
    return b"\x00" if v is None else b"\x01" + enc(v)


def enc_list(enc, xs: Sequence) -> bytes:
    return enc_nat(len(xs)) + b"".join(enc(x) for x in xs)


def raw(statement: str, circuit: bytes, public: bytes, tables: Sequence[tuple[str, bytes]], *, zk: bool,
        os_coins: bool = False, count: int = 1, draw: bytes | None = None, partition: bytes | None = None,
        program: bytes | None = None, registered: bytes | None = None) -> bytes:
    """`FlockDeploy.Raw.encode` of a statement with no work sample, stratified count, closure map or public inputs."""
    return (enc_str(FORMAT) + enc_str(statement) + enc_bytes(circuit) + enc_bytes(public)
            + enc_list(lambda t: enc_str(t[0]) + enc_bytes(t[1]), tables) + enc_bool(zk) + enc_bool(os_coins)
            + enc_nat(count) + enc_opt(enc_bytes, draw) + enc_opt(enc_bytes, partition) + enc_opt(enc_bytes, program)
            + enc_opt(enc_nat, None) + enc_opt(enc_bytes, None) + enc_opt(enc_nat, None) + enc_opt(enc_bytes, None)
            + enc_opt(enc_bytes, registered) + enc_opt(enc_bytes, None))


def mufu_tables() -> list[tuple[str, str]]:
    """`MUFU_TABLES`' (name, SHA-512) pairs, in its order."""
    return re.findall(r'⟨"(\w+)",\s*"[0-9a-f]{64}",\s*"([0-9a-f]{128})"', LOOKUP.read_text())


def compact(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()


def verify_args(log: Path) -> list[str]:
    """The arguments after `verify` on the command line a0's verify log opens with."""
    line = next(x for x in log.read_text().splitlines() if x.startswith("$ "))
    argv = shlex.split(line[2:])
    return argv[argv.index("verify") + 1:]


def session_statement(session_dir: Path) -> bytes:
    """The statement Flock's Lean verifier read for a member's session (``audit/session-m<k>``), as
    `FlockDeploy.Raw.encode`: the objects its command line names in its archive (each checked to hash to its name), the
    MUFU tables the archive holds, and its session record's ``unit_draw``."""
    args = verify_args(session_dir / "lean-verify.log")
    opts: dict[str, str] = {}
    flags = set()
    i = 0
    while i < len(args):
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            opts[args[i]], i = args[i + 1], i + 2
        else:
            flags.add(args[i])
            i += 1
    if bad := [f for f in UNREAD if f in opts or f in flags]:
        raise ValueError(f"{session_dir}: verify read {bad}, which this export doesn't encode")
    archive = Path(opts["--archive"])
    if not archive.is_dir():
        archive = session_dir / "archive"

    def named(h: str) -> bytes:
        b = (archive / "sha512" / h).read_bytes()
        if hashlib.sha512(b).hexdigest() != h:
            raise ValueError(f"{archive}: object {h[:16]}… does not hash to its name")
        return b

    def obj(flag: str) -> bytes | None:
        return named(opts[flag]) if flag in opts else None
    record = json.loads((Path(opts["--session"]) if Path(opts["--session"]).is_dir() else
                         next((session_dir / "sessions").iterdir())).joinpath("session.json").read_text())
    sample = record.get("unit_draw")
    return raw(opts["--statement"], obj("--circuit"), obj("--public"),
               [(n, named(h)) for n, h in mufu_tables() if (archive / "sha512" / h).exists()],
               zk="--zk" in flags, os_coins=opts.get("--coins") == "os", count=int(opts.get("--session-tables", "1")),
               draw=None if sample is None else compact(sample), partition=obj("--partition"), program=obj("--program"),
               registered=obj("--registered"))


def verdict(session_dir: Path) -> dict:
    """Flock's Lean verifier's verdict on a member's session: its verify log's ``VERDICT`` line's ``accepted`` and
    ``why``."""
    line = [x for x in (session_dir / "lean-verify.log").read_text().splitlines() if x.startswith("VERDICT ")][-1]
    v = json.loads(line.removeprefix("VERDICT "))
    return {"accepted": v["accepted"], "why": v.get("why", "")}


def ref_sample(q: int, n_t: int, k: int) -> dict:
    """The second stage's sample of template ``t``'s part at its first units: ``q·min(k, n_t)`` of its ``q·n_t``, the size
    of every sample the second stage takes there (`select.second_law`'s subset in each cell)."""
    size = q * min(k, n_t)
    return {"k": size, "law": "subset", "population": q * n_t, "units": list(range(size))}


def export(out: Path, run_out: Path, stream, order: int, n: int, q: int, k: int) -> dict:
    """The guard's inputs for the run at ``run_out`` on ``stream`` (its boundary registration at ``order``), with ``n``
    spans, ``q`` replay blocks sampled and ``k`` proof blocks a cell: written to ``out``; -> ``held.json``'s object."""
    import toy as T
    from verity.core.protocols.spot import second_stage as SS
    from verity.ml.circuits.codec import canonical_json, descriptor_id, encode_program
    from verity.ml.windows.rows import expected_ports

    out.mkdir(parents=True, exist_ok=True)
    entry = stream._entries[order]
    bw, ref = T.boundary_window(n, q), T.interior_window("00" * 64, range(q))
    first = {"rule": stream.spec.selection.to_json(), "program": encode_program(bw.program), "query": dict(bw.query),
             "work_table": None if bw.work_table is None else dict(bw.work_table), "closure": None}
    ran = {"rule": stream.spec.selection.to_json(), "program": encode_program(entry.program), "query": dict(stream.query),
           "work_table": None if entry.work_table is None else dict(entry.work_table), "closure": None}
    if canonical_json(first) != canonical_json(ran):
        raise ValueError("the boundary window's select input is not the one the run's first stage read")
    replay, key = bw.program, descriptor_id(T.REPLAY)
    links = SS.replay_links(replay, key, {"roots": [{"port": p} for p in expected_ports(bw)]})
    brec = json.loads((run_out / "boundary" / "registration.json").read_text())
    if links != SS.replay_links(replay, key, brec):
        raise ValueError("the design's links at the boundary window's ports are not the run's")
    wires = SS.replay_wires(replay, key)
    rows = SS.replay_rows(replay, key)
    part, prog = json.dumps(ref.partition, sort_keys=True).encode(), canonical_json(encode_program(ref.program))
    names, base = [m.template for m in ref.members], 0
    vus = []
    for n_t in T.N_T:
        vus.append(list(range(base, base + n * n_t)))
        base += n * n_t
    query = stream.spec.check.query
    held = {"first": first, "ns": list(stream.cells), "ks": list(stream.counts), "names": names,
            "links": None if links is None else [{"template": x.template, "port": x.port, "value": x.value} for x in links],
            "wires": None if wires is None else [{"template": w.template, "port": w.port, "writer": w.writer,
                                                  "wport": w.wport} for w in wires],
            "rows": [r.to_json() for r in rows],
            "vus": vus, "desc": prog.hex(),
            "query": {"name": query["name"], "version": query["version"], "params": canonical_json(query["params"]).hex()},
            "k": list(stream.counts)}
    audit = run_out / "audit"
    if (audit / "partition.json").read_bytes() != part or (audit / "program.json").read_bytes() != prog:
        raise ValueError("the reference window's partition or program is not the one the sessions were verified under")
    stmts = []
    params = json.loads((run_out / "proofs.json").read_text())["templates"]
    for t, (m, n_t) in enumerate(zip(ref.members, T.N_T)):
        d = run_out / "references" / f"m{t}"
        reads = params[t].get("registered")
        tables = sorted((d / "tables").glob("*.u32")) if (d / "tables").is_dir() else []
        order_of = [name for name, _ in mufu_tables()]
        (out / f"ref-{t}.bin").write_bytes(raw(
            "verity/flock-circuit", (d / "circuit.txt").read_bytes(), (d / "pub.bin").read_bytes(),
            [(f.stem, f.read_bytes()) for f in sorted(tables, key=lambda f: order_of.index(f.stem))], zk=True,
            draw=compact(ref_sample(q, n_t, k)), partition=part, program=prog))
        (out / f"ref-{t}.reads.json").write_bytes(canonical_json(reads))
        stmts.append(session_statement(audit / f"session-m{t}"))
    (out / "m2.bin").write_bytes(enc_list(enc_bytes, stmts))
    (out / "verdicts.json").write_text(json.dumps([verdict(audit / f"session-m{t}") for t in range(len(stmts))]))
    (out / "x.bin").write_bytes((run_out / "boundary" / "registration.json").read_bytes())
    (out / "m1.bin").write_bytes((run_out / "interior" / "registration.json").read_bytes())
    if entry.read is None or entry.second_read is None:
        raise ValueError("a stage's sample read no coins of the challenger's (a sampler drew it)")
    (out / "c0.bin").write_bytes(entry.read)
    (out / "c1.bin").write_bytes(entry.second_read)
    (out / "held.json").write_text(json.dumps(held, indent=1, sort_keys=True))
    return held


def pack(run_out: Path, fixture: Path, cases: Sequence[str] = ("honest", "altered")) -> None:
    """Each case's ``guard/`` under ``run_out`` as ``<case>/`` of one xz tar at ``fixture``. Its window (768 MiB) spans
    the distance to the next copy of a statement's circuit, which is most of every statement, so the tar holds each once."""
    def plain(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
        return ti
    with lzma.open(fixture, "wb", filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": 768 << 20}]) as f, \
            tarfile.open(fileobj=f, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for case in cases:
            for p in sorted((run_out / case / "guard").iterdir()):
                tar.add(p, arcname=f"{case}/{p.name}", filter=plain)


def registered_span(st: bytes) -> tuple[int, int] | None:
    """Where a statement's registered reads sit in its `FlockDeploy.Raw` encoding (`raw`'s field order), or None when it
    carries none."""
    i = 0

    def nat() -> int:
        nonlocal i
        i += 8
        return int.from_bytes(st[i - 8:i], "little")

    def skip() -> None:
        nonlocal i
        n = nat()
        i += n

    def opt(f) -> None:
        nonlocal i
        i += 1
        if st[i - 1]:
            f()

    for _ in range(4):
        skip()
    for _ in range(nat()):
        skip(), skip()
    i += 2
    nat()
    for f in (skip, skip, skip, nat, skip, nat, skip):
        opt(f)
    if st[i] != 1:
        return None
    i += 1
    n = nat()
    return i, i + n


def split_refs(guard: Path) -> None:
    """Each ``ref-<t>.bin`` under ``guard`` that carries registered reads rewritten without them, the reads written to
    its ``ref-<t>.reads.json``, as `export` now writes them; one that carries none gets ``null``."""
    for ref in sorted(guard.glob("ref-*.bin")):
        st = ref.read_bytes()
        span = registered_span(st)
        if span is None:
            reads = b"null"
        else:
            lo, hi = span
            reads, st = st[lo:hi], st[:lo - 9] + b"\x00" + st[hi:]
            json.loads(reads)
        ref.with_suffix(".reads.json").write_bytes(reads)
        ref.write_bytes(st)


def flip_registered(m2: bytes) -> bytes:
    """Round 2's message ``m2`` with one byte of its first statement's registered reads flipped: the first hex digit of
    the first ``"root"`` value, so the reads still parse as JSON and name another root."""
    out = bytearray(m2)
    start = 16
    span = registered_span(m2[start:start + int.from_bytes(m2[8:16], "little")])
    if span is None:
        raise ValueError("round 2's first statement has no registered reads")
    lo, hi = span
    reads = out[start + lo:start + hi]
    at = reads.index(b'"root"')
    at = reads.index(b'"', at + len(b'"root"') + 1) + 1
    out[start + lo + at] = ord("1") if reads[at] == ord("0") else ord("0")
    json.loads(out[start + lo:start + hi])
    return bytes(out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["pack"] and len(sys.argv) == 4:
        pack(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1:2] == ["flip"] and len(sys.argv) == 4:
        Path(sys.argv[3]).write_bytes(flip_registered(Path(sys.argv[2]).read_bytes()))
    elif sys.argv[1:2] == ["refs"] and len(sys.argv) == 3:
        split_refs(Path(sys.argv[2]))
    else:
        sys.exit("usage: python3 guard.py pack RUN_OUT FIXTURE | flip M2 OUT | refs GUARD")
