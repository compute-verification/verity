"""A recursive session's inner statement whose private circuit's description is registered (PROTOCOL.md §16.16): the
description's rows (the class statement's row port of the private circuit's ``op``, ``src`` and ``out`` bits, ``prog`` under
`rec_inner.program_unit`, one table row every instance reads) committed once at registration
(`verity.core.protocols.sampled_proofs.one_stage.registered`), and the statement opening them against the registration's
root, so that a verifier holding the root reads the description from the developer's registration, not from the prover.

`register` runs on a staged statement (``circuit.txt``, ``inst-n.bin``, ``pub-n.bin``, a shared-row file): the
registration's rows and salts are the description table's own (the salts drawn at staging, used for no other row), so its
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


def _files(stage: Path) -> tuple[Path, Path]:
    pubs = sorted(stage.glob("pub-*.bin"))
    if len(pubs) != 1:
        raise ValueError(f"{stage}: one pub-n.bin, not {len(pubs)}")
    return stage / pubs[0].name.replace("pub-", "inst-"), pubs[0]


def _split(path: Path) -> tuple[dict, bytes]:
    head, body = path.read_bytes().split(b"\n", 1)
    return json.loads(head), body


def _put(path: Path, head: dict, body: bytes) -> None:
    path.write_bytes(json.dumps(head, sort_keys=True).encode() + b"\n" + body)


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


def register(stage: Path, ports: Mapping[str, str] | None = None, extra: int = 0) -> dict:
    """Register the description rows of the staged statement in ``stage`` (``ports``: ``{port: value}``, default the one
    `description_port` as `VALUE`) and write the header's ``registered`` and the verifier's ``registered.json``. ``extra``:
    that many more registered rows (zero rows under fresh salts) past the table's, which no instance reads. Returns
    ``{port: {"entry", "root", "leaves", "positions"}}``."""
    head, tabs = tables(stage)
    meta = head["meta"]
    ports = dict(ports) if ports else {description_port(head): VALUE}
    program = meta.get("program_sha512")
    if not program:
        raise ValueError(f"{stage}: META names no program_sha512, which the registration's domain binds")
    entries, own, out = {}, {}, {}
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
        reg = G.Registered.commit(program, value, rows, salts)
        if list(reg.bcs[:len(t["bcs"])]) != t["bcs"]:
            raise AssertionError(f"port {name}: the registration's b || c are not the statement's")
        positions = list(range(len(t["bcs"])))
        _, _, entries[name] = reg.statement(positions)
        reads = [None] * population
        for i, u in enumerate(head["units"]["indices"]):
            reads[u] = t["refs"][i]
        own[name] = {**reg.entry(), "word_bits": G.WORD_BITS, "reads": reads}
        out[name] = {"entry": entries[name], "root": reg.root.hex(), "leaves": reg.leaves, "positions": positions}
    _write_header(stage, entries)
    (stage / OWN).write_text(json.dumps({"format": G.READS, "ports": own}, sort_keys=True))
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
