"""A recursive session's inner statement whose private circuit's description is registered (PROTOCOL.md §16.16): the
description's rows (the class statement's row port of the private circuit's ``op``, ``src`` and ``out`` bits, ``prog`` under
`rec_inner.program_unit`, one table row every instance reads) committed once at registration
(`verity.core.protocols.sampled_proofs.one_stage.registered`), and the statement opening them against the registration's
root, so that a verifier holding the root reads the description from the developer's registration, not from the prover.

`register` runs on a staged statement (``circuit.txt``, ``inst-n.bin``, ``pub-n.bin``, a shared-row file): the
registration's rows and salts are the description table's own (the salts sampled at staging, used for no other row), so its
``b || c`` are the registered ones and the statement's rows, salts and frame-v3 roots are unchanged. It adds the header's
``registered`` (§16.10: each table row's position, its index, and its path to the registered root) to the prover's and the
verifier's files and writes ``registered.json`` beside them, the verifier's own ``--registered`` file
(`verity/registered-reads/v0`): the root, leaf count, row width and domain the registration names, and every unit's
position (the description port's ref of the unit's instance), which the Lean verifier does not read for an inner statement
(`Registered.checkDescription`: no position depends on which unit an instance is). The description's registered value is
its packed row (16 description bits to a u16 word, as the class statement lays out the parameter), so its row width is
the port's words, not the parameter's leaf count.

``claim`` writes another statement's header with a registration's entries: the prover of another description of the class
claiming the registered rows (a statement the verifier refuses: its row does not open the root). ``extra`` registers rows
past the description's, which no instance reads (refused: the registration's rows are not the statement's).

**The circuit registration** (`bind`): the staged statement's one-stage ``verity/registration/v2`` record
(`REGISTRATION`, beside ``circuit.txt``), whose ``registered`` entries are every value registered at staging (``prog``
when `register` is given the statement's program and partition object; candidate 3's stacked table R's top,
`verity_flock.rec_session.stage_stacked`). Its ``program``, ``query`` and ``partition`` are the partition object's that
META names (``program_sha512``, ``partition``), its law one-stage's ``stratified`` over the population, its scheme the
statement's (`circuit.SCHEME`), no served roots (the inner tables' are committed in the session), and its window
``stand-in``: it depends on the circuit and the registration's salts alone. META's ``registration`` (`REGISTRATION_KEY`)
is the record's digest (one-stage ``registration.digest``), so the circuit's SHA-512, the statement digest the prover
absorbs before every inner stream's first coin and Σ all bind the record. The salts of a value whose rows are not the
statement's (R's top) are the prover's alone (`PRIVATE`, beside ``inst-N.bin``): never in the record, META, a transcript or
a verifier's file.

    python -m verity_flock.inner_registered register STAGE [--port PORT=VALUE] [--extra K]
    python -m verity_flock.inner_registered claim STAGE --from STAGE
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import registered as G

from . import circuit as FC

#: the registered value a description port reads (`rec_inner.program_unit`'s parameter)
VALUE = "prog"
OWN = "registered.json"
#: the staged statement's circuit registration (one-stage ``verity/registration/v2``), public, beside ``circuit.txt``
REGISTRATION = "circuit-registration.json"
#: META's key for the circuit registration's digest (one-stage ``registration.digest``)
REGISTRATION_KEY = "registration"
#: the prover's own openings of the registered values whose rows are not the statement's (``{value: {nodes, salts}}``),
#: beside ``inst-N.bin``
PRIVATE = "circuit-registration-private.json"
PRIVATE_KIND = "flock-circuit/registration-private/v0"
WINDOW = {"kind": "stand-in", "of": "the staged statement's circuit registration: its values registered at staging, before any session"}


def _files(stage: Path) -> tuple[Path, Path]:
    pubs = sorted(stage.glob("pub-*.bin"))
    if len(pubs) != 1:
        raise ValueError(f"{stage}: one pub-n.bin, not {len(pubs)}")
    return stage / pubs[0].name.replace("pub-", "inst-"), pubs[0]


def _split(path: Path) -> tuple[dict, bytes]:
    head, body = path.read_bytes().split(b"\n", 1)
    return json.loads(head), body


def _put(path: Path, head: dict, body: bytes) -> None:
    """The prover's file or, ``rows`` false, the verifier's (`circuit.public_file`)."""
    path.write_bytes(json.dumps(head, sort_keys=True).encode() + b"\n" + body if head["rows"] else FC.public_file(head, body))


def tables(stage: Path) -> tuple[dict, dict[str, dict]]:
    """The prover's file's header and each row port's table: its rows (u16, ``(R_p, words)``), salts and ``b || c``, and every
    instance's ref, as `circuit.write` lays out a shared-row file."""
    meta = json.loads((stage / "circuit.txt").read_text().split("\n", 2)[1][5:])
    ports = FC._meta_ports(meta)
    inst, _pub = _files(stage)
    head, body = _split(inst)
    if "shared_rows" not in head:
        raise ValueError(f"{inst}: not a shared-row file, and a description is a table every instance reads")
    counts, n = head["shared_rows"], head["instances"]
    total = sum(counts[p.name] for p in ports)
    if len(body) < sum(2 * p.words * counts[p.name] for p in ports) + (FC.SALT_BYTES + FC.BC_BYTES) * total + 4 * n * len(ports):
        raise ValueError(f"{inst}: shorter than its shared-row tables")
    at, out = 0, {}
    for p in ports:
        size = 2 * p.words * counts[p.name]
        out[p.name] = {"rows": np.frombuffer(body[at:at + size], dtype="<u2").reshape(counts[p.name], p.words)}
        at += size
    for p in ports:
        out[p.name]["salts"] = [body[at + FC.SALT_BYTES * r:at + FC.SALT_BYTES * (r + 1)] for r in range(counts[p.name])]
        at += FC.SALT_BYTES * counts[p.name]
    for p in ports:
        out[p.name]["bcs"] = [body[at + FC.BC_BYTES * r:at + FC.BC_BYTES * (r + 1)] for r in range(counts[p.name])]
        at += FC.BC_BYTES * counts[p.name]
    refs = np.frombuffer(body[at:at + 4 * n * len(ports)], dtype="<u4").reshape(n, len(ports))
    for k, p in enumerate(ports):
        out[p.name]["refs"] = refs[:, k].astype(int).tolist()
    return head | {"meta": meta, "input_ports": [p.name for p in ports[:meta["input_ports"]]]}, out


def description_port(head: dict) -> str:
    """The input port whose one table row every instance reads: the description, when there is exactly one."""
    counts, n = head["shared_rows"], head["instances"]
    one = [p for p in head["input_ports"] if counts[p] == 1]
    if n < 2 or len(one) != 1:
        raise ValueError(f"no one description port among {head['input_ports']} (shared_rows {counts}, {n} instances): name it")
    return one[0]


def _write_header(stage: Path, entries: Mapping[str, dict]) -> dict:
    inst, pub = _files(stage)
    (head, body), (phead, pbody) = _split(inst), _split(pub)
    strip = lambda h: {k: v for k, v in h.items() if k not in ("registered", "rows")}  # noqa: E731
    if strip(head) != strip(phead):
        raise ValueError(f"{pub}: its header is not the prover's file's")
    head = dict(head, registered={k: dict(v) for k, v in entries.items()})
    _put(inst, head, body)
    _put(pub, dict(head, rows=False), pbody)
    return head


def meta(stage: Path) -> dict:
    """The staged statement's META."""
    return json.loads((Path(stage) / "circuit.txt").read_text().split("\n", 2)[1][5:])


def registration_record(program, partition: Mapping, entries: Sequence[Mapping]) -> dict:
    """The circuit registration over ``program`` (a core ``Program``) and its ``verity/partition/v1`` object ``partition``,
    registering ``entries`` (`Registered.entry`): one-stage ``registration.record`` with the law ``stratified`` over the
    population, the statement's scheme, no served roots and the `WINDOW` (the module docstring)."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R

    query = partition["query"]
    return R.record(program, query, {"law": "stratified", "k": P.units(program, query)}, FC.SCHEME, [], WINDOW,
                    partition=partition, registered=list(entries))


def bind_meta(stage: Path, key: str, value) -> None:
    """``circuit.txt`` with META's ``key`` ``value``, and both files' headers' circuit SHA-512 and classes its."""
    stage = Path(stage)
    text = (stage / "circuit.txt").read_text()
    head, line, rest = text.split("\n", 2)
    m = dict(json.loads(line[5:]), **{key: value})
    text = head + "\nMETA " + json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n" + rest
    (stage / "circuit.txt").write_text(text)
    inst, pub = _files(stage)
    for path in (inst, pub):
        h, body = _split(path)
        h = dict(h, circuit_sha512=FC.digest(text), units=dict(h["units"], classes=[FC.class_digest(text)] * h["instances"]))
        _put(path, h, body)


def bind(stage: Path, program, partition: Mapping, entries: Sequence[Mapping], private: Mapping[str, Mapping] | None = None) -> dict:
    """Register ``entries`` with the staged statement in ``stage``: its circuit registration (`registration_record`, with
    every value an earlier `bind` registered; a value is registered once) written to `REGISTRATION`, its digest bound into
    META (`REGISTRATION_KEY`), and ``private`` (``{value: {"nodes", "salts"}}``, the prover's openings) added to `PRIVATE`.
    Refused unless META names ``partition`` and its program. Returns the record."""
    from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R

    stage = Path(stage)
    m = meta(stage)
    if m.get("program_sha512") != partition["program"] or m.get("partition") != {"rule": P.FORMAT, "digest": P.digest(partition)}:
        raise ValueError(f"{stage}: META binds program {m.get('program_sha512')} and partition {m.get('partition')}, not the "
                         f"registration's {partition['program']} and {P.digest(partition)}")
    old = json.loads((stage / REGISTRATION).read_text())["registered"] if (stage / REGISTRATION).exists() else []
    names = [e["value"] for e in [*old, *entries]]
    if len(set(names)) != len(names):
        raise ValueError(f"a value is registered once, and the registration would name {sorted(names)}")
    rec = registration_record(program, partition, [*old, *entries])
    (stage / REGISTRATION).write_text(json.dumps(rec, indent=1))
    bind_meta(stage, REGISTRATION_KEY, R.digest(rec))
    if private:
        own = json.loads((stage / PRIVATE).read_text())["values"] if (stage / PRIVATE).exists() else {}
        own.update({v: {"nodes": list(o["nodes"]), "salts": list(o["salts"])} for v, o in private.items()})
        (stage / PRIVATE).write_text(json.dumps({"kind": PRIVATE_KIND, "values": own}))
    return rec


def bound(stage: Path) -> dict:
    """The staged statement's circuit registration, refused unless its digest is the one META carries."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as R

    stage = Path(stage)
    want = meta(stage).get(REGISTRATION_KEY)
    if want is None:
        raise ValueError(f"{stage}: META binds no circuit registration")
    if not (stage / REGISTRATION).exists():
        raise ValueError(f"{stage}: no {REGISTRATION}, the circuit registration META binds")
    rec = json.loads((stage / REGISTRATION).read_text())
    if R.digest(rec) != want:
        raise ValueError(f"{stage}: {REGISTRATION}'s digest {R.digest(rec)} is not the registration META binds ({want})")
    return rec


def private(stage: Path, value: str) -> dict:
    """The prover's opening (``nodes``, ``salts``) of ``value`` registered at staging (`PRIVATE`)."""
    path = Path(stage) / PRIVATE
    got = json.loads(path.read_text())["values"].get(value) if path.exists() else None
    if got is None:
        raise ValueError(f"{stage}: the prover holds no opening of the registered value {value} ({PRIVATE})")
    return got


def register(stage: Path, ports: Mapping[str, str] | None = None, extra: int = 0, program=None,
             partition: Mapping | None = None) -> dict:
    """Register the description rows of the staged statement in ``stage`` (``ports``: ``{port: value}``, default the one
    `description_port` as `VALUE`) and write the header's ``registered`` and the verifier's ``registered.json``. ``extra``:
    that many more registered rows (zero rows under fresh salts) past the table's, which no instance reads. With
    ``program`` and ``partition`` (the statement's core ``Program`` and ``verity/partition/v1`` object), the entries also
    join the circuit registration (`bind`). Returns ``{port: {"entry", "root", "leaves", "positions"}}``."""
    if (program is None) != (partition is None):
        raise ValueError("the circuit registration takes the statement's program and its partition object together")
    head, tabs = tables(stage)
    meta = head["meta"]
    ports = dict(ports) if ports else {description_port(head): VALUE}
    program_sha512 = meta.get("program_sha512")
    if not program_sha512:
        raise ValueError(f"{stage}: META names no program_sha512, which the registration's domain binds")
    entries, own, out, records = {}, {}, {}, []
    population = max(head["units"]["indices"], default=-1) + 1
    for name, value in ports.items():
        if name not in head["input_ports"]:
            raise ValueError(f"{name} is not an input port of {stage} ({head['input_ports']})")
        t = tabs[name]
        rows = [tuple(int(w) for w in r) for r in t["rows"]]
        salts = list(t["salts"])
        if extra:
            rows += [(0,) * len(rows[0])] * extra
            salts += [os.urandom(FC.SALT_BYTES) for _ in range(extra)]
        reg = G.Registered.commit(program_sha512, value, rows, salts)
        if list(reg.bcs[:len(t["bcs"])]) != t["bcs"]:
            raise AssertionError(f"port {name}: the registration's b || c are not the statement's")
        positions = list(range(len(t["bcs"])))
        _, _, entries[name] = reg.statement(positions)
        reads = [None] * population
        for i, u in enumerate(head["units"]["indices"]):
            reads[u] = t["refs"][i]
        own[name] = {**reg.entry(), "word_bits": G.WORD_BITS, "reads": reads}
        out[name] = {"entry": entries[name], "root": reg.root.hex(), "leaves": reg.leaves, "positions": positions}
        records.append(reg.entry())
    _write_header(stage, entries)
    (stage / OWN).write_text(json.dumps({"format": G.READS, "ports": own}, sort_keys=True))
    if program is not None:
        bind(stage, program, partition, records)
    return out


def claim(stage: Path, source: Path) -> dict:
    """``stage``'s header with ``source``'s registered entries (the verifier's file is ``source``'s ``registered.json``)."""
    src, _ = _split(_files(source)[0])
    if "registered" not in src:
        raise ValueError(f"{source}: no registered entries to claim")
    _write_header(stage, src["registered"])
    return src["registered"]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("stage", type=Path)
    r.add_argument("--port", action="append", default=[], metavar="PORT=VALUE")
    r.add_argument("--extra", type=int, default=0)
    c = sub.add_parser("claim")
    c.add_argument("stage", type=Path)
    c.add_argument("--from", dest="source", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "register":
        got = register(a.stage, dict(p.split("=", 1) for p in a.port) or None, a.extra)
        print(json.dumps({k: {kk: v[kk] for kk in ("root", "leaves", "positions")} for k, v in got.items()}, sort_keys=True))
    else:
        print(json.dumps(sorted(claim(a.stage, a.source)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
