"""Flock's input files for a window's member (``serving_rows.write``'s window): the public file
``m<k>-pub-<n>.bin`` (``rows: false``) and the prover's ``m<k>-inst-<n>.bin``, each the window's bodies under the header
``verity_flock.circuit.write`` gives the circuit the verifier composed for the member's template.  A chunked template's chunks
are all staged under one circuit, whose META, pin and class (`FlockCircuit`) are read off it once."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.commitments.windows.rows import BC_BYTES, SCHEME, copy_private, held_ports, own_entries, registered_header

#: Flock's circuit prover's ``flock-circuit-inputs`` header constants (``verity_flock.circuit``) and the META keys a class leaves out
M0_FORMAT, M0_STATEMENT, M0_BINDING_KEYS = "flock-circuit-inputs", "verity/flock-circuit", ("partition", "program_sha512", "program_digests")
#: a sparse header ``registered`` entry's key: the table rows it gives a position and a path (`read_rows`)
TABLE_ROWS = "table_rows"


@dataclass(frozen=True)
class FlockCircuit:
    """What a member's circuit-prover header reads off the circuit it is staged under (`flock_circuit`): its META, its SHA-512
    pin and its class."""
    meta: dict[str, Any]
    pin: str
    cls: str


def flock_circuit(circuit: str | bytes) -> FlockCircuit:
    """``circuit``'s META, its SHA-512 and its class: the SHA-512 of the circuit with META's binding keys
    (``M0_BINDING_KEYS``) left out, as ``verity_flock.circuit`` writes it."""
    b = circuit.encode() if isinstance(circuit, str) else circuit
    i = b.index(b"\n")
    j = b.index(b"\n", i + 1)
    meta = json.loads(b[i + 6:j])
    cls_meta = {x: v for x, v in meta.items() if x not in M0_BINDING_KEYS}
    cls = hashlib.sha512(b[:i] + b"\nMETA " + json.dumps(cls_meta, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    cls.update(memoryview(b)[j + 1:])
    return FlockCircuit(meta, hashlib.sha512(b).hexdigest(), cls.hexdigest())


def keyed_ports(openings: Sequence[Mapping[str, Any]], win: Mapping[str, Any], k: int) -> frozenset[str]:
    """Member ``k``'s ports a table opening reads at a hidden key: of the circuit record's ``registered.openings``
    (`interface.TableOpening`), each one's ``key``, ``ports`` and ``verdict`` whose reader is the member's template and whose
    units ``[at, at + units)``, from the template's first unit, meet the member's."""
    mem = win["members"][k]
    first = min(int(m["base"]) for m in win["members"] if m["template"] == mem["template"])
    lo, hi = int(mem["base"]) - first, int(mem["base"]) - first + int(mem["n"])
    return frozenset(p for o in openings if o["reader"] == mem["template"] and int(o["at"]) < hi and lo < int(o["at"]) + int(o["units"])
                     for p in (o["key"], *o["ports"], o["verdict"]))


def read_rows(entries: Mapping[str, Mapping[str, Any]], mem: Mapping[str, Any], body: bytes, *,
              keyed: Collection[str]) -> dict[str, dict[str, Any]]:
    """A shared-row member's header ``registered`` ``entries`` (one position and one path per row of a port's table) with
    each port's narrowed to the table rows its instances read, by their refs (the tail of its public ``body``): ``table_rows``,
    those rows in increasing order, and their positions and paths, ``positions[i]`` and ``paths[i]`` table row
    ``table_rows[i]``'s. A port whose instances read every row of its table keeps its entry as given, and so does each port
    of ``keyed`` (`keyed_ports`): a read at a hidden key has no public refs, its row being chosen by a committed value that a
    list would publish. Refused where a ``keyed`` port's entry lists ``table_rows``, an entry is not one position and one
    path per table row, or a ref names no row of its table."""
    if listed := sorted(p for p, e in entries.items() if p in keyed and TABLE_ROWS in e):
        raise ValueError(f"{mem['name']} {listed[0]}: a read at a hidden key lists no table_rows (its row is a committed value's)")
    shared = mem.get("shared_rows")
    if not shared:
        return {p: dict(e) for p, e in entries.items()}
    n, ports = int(mem["n"]), [p for p, _w in mem["ports"]]
    tables = sum(int(shared[p]) for p in ports) * BC_BYTES
    if len(body) != tables + 4 * n * len(ports):
        raise ValueError(f"{mem['name']}: a public body of {len(body)} bytes, not its tables' and refs' {tables + 4 * n * len(ports)}")
    refs = np.frombuffer(body, dtype="<u4", offset=tables).reshape(n, len(ports))
    out = {}
    for p, e in entries.items():
        if p in keyed:
            out[p] = dict(e)
            continue
        rows = int(shared[p])
        if len(e["positions"]) != rows or len(e["paths"]) != rows:
            raise ValueError(f"{mem['name']} {p}: not one position and one path per row of its table's {rows}")
        read = np.unique(refs[:, ports.index(p)]).tolist()
        if read and read[-1] >= rows:
            raise ValueError(f"{mem['name']} {p}: an instance reads table row {read[-1]}, past its table's {rows}")
        out[p] = dict(e) if len(read) == rows else {**e, TABLE_ROWS: read, "positions": [e["positions"][r] for r in read],
                                                    "paths": [e["paths"][r] for r in read]}
    return out


def flock_input_files(window_dir: str | Path, k: int, circuit: str | FlockCircuit, out_dir: str | Path, *,
                      write: Sequence[str] = ("pub", "inst"), files: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
                      sparse: bool = False, keyed: Collection[str] | None = None) -> dict[str, Any]:
    """Member ``k``'s circuit-prover ``m<k>-pub-<n>.bin`` (``rows: false``) and ``m<k>-inst-<n>.bin``, under the header
    ``verity_flock.circuit.write`` gives the circuit it composed: its SHA-512 pin, its META's statement, subcircuit, block shape
    and class, the member's global unit indices, and its ``frame_v3``.  The bodies are the window's bytes, its prover body
    made from its window's files (`rows.copy_private`) and its header's paths read off the window's (`rows.registered_header`).
    A member that reads a registered value, or holds a wired port in its writer's tree (`rows.held_ports`), has every other
    port registered too, as rows of its own tree or its writer's (`rows.own_entries`), the reads the verifier adds at the
    registration's roots (`rows.own_reads`). ``circuit`` is the
    circuit's text or its `flock_circuit`, ``write`` the files to write (a member's ``inst`` is the prover's, wanted only once
    it is sampled), and ``files`` the window's ``window.json`` and ``registration.json`` as read (else read here).
    ``sparse``: each port's entry but the ``keyed`` ones (`keyed_ports` of the circuit record's openings, which ``sparse``
    needs) gives only the table rows the member's instances read (`read_rows`); off until Flock's verifier takes
    ``table_rows``, which an older one refuses."""
    if not set(write) <= {"pub", "inst"}:
        raise ValueError(f"flock_input_files writes pub and inst, not {sorted(set(write) - {'pub', 'inst'})}")
    if sparse and keyed is None:
        raise ValueError("a sparse header needs the member's keyed ports (keyed_ports of the circuit record's openings)")
    d, o = Path(window_dir), Path(out_dir)
    c = circuit if isinstance(circuit, FlockCircuit) else flock_circuit(circuit)
    win, rec = files if files is not None else (json.loads((d / "window.json").read_text()), json.loads((d / "registration.json").read_text()))
    mem = win["members"][k]
    meta = c.meta
    if ([list(p[:2]) for p in meta["ports"]] != mem["ports"] or [list(p) for p in meta["out_ports"]] != mem["out_ports"]
            or meta.get("input_ports") != mem["input_ports"]):
        raise ValueError(f"the circuit's ports {meta['ports']} / {meta['out_ports']} ({meta.get('input_ports')} input rows) are not "
                         f"member {k}'s {mem['ports']} / {mem['out_ports']} ({mem['input_ports']} input rows)")
    if meta.get("row_schemas") != mem.get("row_schemas"):
        raise ValueError(f"the circuit's row schemas {meta.get('row_schemas')} are not member {k}'s {mem.get('row_schemas')}")
    if meta.get("scheme") != SCHEME:
        raise ValueError(f"the circuit commits under {meta.get('scheme')!r}, the window under {SCHEME!r}")
    n, G = int(mem["n"]), int(meta["vus_per_block"])
    idx = list(range(int(mem["base"]), int(mem["base"]) + n))
    units = {"vus_per_block": G, "units_per_vu": meta["units_per_vu"], "indices": idx, "blocks": [idx[b:b + G] for b in range(0, n, G)],
             "classes": [c.cls] * n}
    head = {"format": M0_FORMAT, "statement": meta.get("statement", M0_STATEMENT), "scheme": SCHEME,
            "set": f"served/{rec['window']['run']}/m{k}", "subcircuit": meta["subcircuit"], "range": [0, n], "instances": n,
            "circuit_sha512": c.pin, "units": units, "frame_v3": mem["frame_v3"], "rows": True}
    if mem.get("shared_rows"):
        head["shared_rows"] = mem["shared_rows"]
    pub = (d / f"m{k}-public.bin").read_bytes()
    held = held_ports(win, k)
    if mem.get("registered") or held:
        reads = registered_header(win, k) if mem.get("registered") else {}
        head["registered"] = {**reads, **own_entries(mem, pub, reads, held)}
        if sparse:
            head["registered"] = read_rows(head["registered"], mem, pub, keyed=keyed or ())
    o.mkdir(parents=True, exist_ok=True)
    got = {"circuit_sha512": c.pin, "class": c.cls}
    if "pub" in write:
        got["pub"] = str(o / f"m{k}-pub-{n}.bin")
        # `verity_flock.circuit.public_file`'s header line, so the file's SHA-512 is the prover's record's public_file_sha512
        line = json.dumps(dict(head, rows=False), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        Path(got["pub"]).write_bytes(line.encode() + b"\n" + pub)
    if "inst" in write:
        got["inst"] = str(o / f"m{k}-inst-{n}.bin")
        fd = os.open(got["inst"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(json.dumps(head, sort_keys=True).encode() + b"\n")
            copy_private(d, win, k, f)
            f.write(pub)
    return got
