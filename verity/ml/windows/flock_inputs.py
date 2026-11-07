"""Flock's input files for a window's member (``serving_rows.write``'s window): the public file
``m<k>-pub-<n>.bin`` (``rows: false``) and the prover's ``m<k>-inst-<n>.bin``, each the window's bodies under the header
``verity_flock.circuit.write`` gives the circuit the verifier composed for the member's template.  A chunked template's chunks
are all staged under one circuit, whose META, pin and class (`FlockCircuit`) are read off it once.

A drawn member's files may be completed at audit time (`complete`): every port of its circuit a registered read, each
parameter port at the root the registration names and every other port at its window tree (a read of `window_value`),
the verifier's own ``--registered`` file naming them all (the Lean verifier refuses a file that leaves a port out,
`Flock.Registered.checkComplete`).  A completed file is a shared-row file (`Flock.ProvedScope.sharedRowsOk`), so an instance-major member's
bodies are re-laid port-major under identity refs: the same rows and strings, the same trees, under the same circuit."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.windows.rows import (BC_BYTES, ROW_SCHEMA, SCHEME, Binding, copy_private, port_leaves, registered_header,
                                    string_levels, tree_path)

#: Flock's circuit prover's ``flock-circuit-inputs`` header constants (``verity_flock.circuit``) and the META keys a class leaves out
M0_FORMAT, M0_STATEMENT, M0_BINDING_KEYS = "flock-circuit-inputs", "verity/flock-circuit", ("partition", "program_sha512", "program_digests")


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


def window_value(port: str) -> str:
    """The value a completed member's read of window port ``port`` (``<member>/<port>``) names: never ``port`` itself, as the
    one-stage audit refuses a session's registered read named by one of its registration's roots ports that its
    registered values don't name (`one_stage.audit.session_registered`)."""
    return f"{port}@window"


@dataclass(frozen=True)
class Complete:
    """A member's files with every port a registered read (`complete`): its header's ``shared_rows`` and ``registered``
    entries, its public body (a shared-row body: an instance-major member's re-laid port-major under identity refs,
    ``relaid``), the ports of each kind, and the verifier's ``--registered`` file (``reads``)."""
    shared_rows: dict[str, int]
    registered: dict[str, dict[str, Any]]
    body: bytes
    relaid: bool
    window_ports: tuple[str, ...]
    parameter_ports: tuple[str, ...]
    reads: dict[str, Any]


def identity_refs(n: int, ports: int) -> np.ndarray:
    """A re-laid body's refs: each of ``n`` instances' ref to its own row of each of ``ports`` ports."""
    return np.repeat(np.arange(n, dtype="<u4")[:, None], ports, axis=1)


def relay(body: bytes, n: int, ports: int) -> bytes:
    """An instance-major public body, ``n`` instances of ``ports`` rows' b || c, re-laid port-major under identity refs
    (`complete`'s body for an instance-major member): each port's rows in instance order, then `identity_refs`."""
    if len(body) != n * ports * BC_BYTES:
        raise ValueError(f"a public body of {len(body)} bytes, not {n} instances of {ports} rows' b || c")
    rows = np.frombuffer(body, dtype=np.uint8).reshape(n, ports, BC_BYTES)
    return b"".join(rows[:, j].tobytes() for j in range(ports)) + identity_refs(n, ports).tobytes()


def unrelay(body: bytes, n: int) -> bytes | None:
    """The instance-major body of ``n`` instances that `relay` re-lays to ``body``, else None: `relay` is one to one, so
    its only candidate, read off ``body``'s port-major rows, is it exactly when `relay` gives ``body`` back."""
    if n <= 0 or not body or len(body) % (n * (BC_BYTES + 4)):
        return None
    ports = len(body) // (n * (BC_BYTES + 4))
    rows = np.frombuffer(body, dtype=np.uint8, count=n * ports * BC_BYTES).reshape(ports, n, BC_BYTES)
    staged = rows.transpose(1, 0, 2).tobytes()
    return staged if relay(staged, n, ports) == body else None


def complete(window: Binding, k: int, roots: Mapping[str, Mapping[str, Any]], body: bytes, *, registered: Any,
             params: Mapping[str, Any] | None = None, header: Mapping[str, Mapping[str, Any]] | None = None) -> Complete:
    """Member ``k``'s files completed so the verifier's ``--registered`` file names every port of its circuit: each port the
    registration makes a parameter port at the root it names (``params``: the verifier's own file of the member's parameter
    reads, a pinned value's or a row table's, never the member's), and every other port at its window tree, a read of
    `window_value`. ``roots``
    (``{<member>/<port>: {"root", "domain", "leaves"}}``) are the window's registered roots, fixed before the coins, which
    the auditor's record must tie to the coins before anything completed from them is pinned; ``window`` is the
    verifier's own, its run the root the auditor's record opens. Each window port's tree is folded again from ``body``, the member's public body as served (each row's ``b || c`` a
    leaf, under the domain ``window`` derives for the port), and must be the registered root, so each read opens a root fixed
    before the coins. ``header``: the member's header ``registered`` entries as served (`rows.registered_header`), whose
    paths open its parameter ports. ``registered``: core's ``one_stage.registered`` (its ``READS_V1``, ``SCHEMA_KEY`` and
    ``READS_FROM``), each window port's entry in its form (`served.ServedTree.entry`). ValueError, naming the member and the
    ports, for a port no tree of the registration commits, a tree whose leaves, domain or root are not the window's, a
    parameter port the header doesn't register or a port it registers that is no parameter port, a parameter port named at
    a window tree, and a body not the member's layout."""
    m = window.members[k]
    ports, leaves = list(m.all_rows), port_leaves(m)
    names, n = [p for p, _ in ports], m.n
    tables = sum(leaves[p] for p in names) * BC_BYTES
    if m.tables:
        if len(body) != tables + 4 * n * len(names):
            raise ValueError(f"{m.key}: a public body of {len(body)} bytes, not its tables' {tables} and {n} x {len(names)} refs")
        bcs, at = {}, 0
        for p in names:
            bcs[p], at = body[at:at + leaves[p] * BC_BYTES], at + leaves[p] * BC_BYTES
        refs = np.frombuffer(body, dtype="<u4", offset=at).reshape(n, len(names))
        out = bytes(body)
    else:
        if len(body) != n * len(names) * BC_BYTES:
            raise ValueError(f"{m.key}: a public body of {len(body)} bytes, not {n} instances of {len(names)} rows' b || c")
        rows = np.frombuffer(body, dtype=np.uint8).reshape(n, len(names), BC_BYTES)
        bcs = {p: rows[:, j].tobytes() for j, p in enumerate(names)}
        refs = identity_refs(n, len(names))
        out = relay(body, n, len(names))
    params_ports, head = dict((params or {}).get("ports") or {}), dict(header or {})
    trees = {x for r in roots.values() for x in (r.get("root"), r.get("domain"))}
    why: list[str] = []
    if set(params_ports) != set(head):
        why.append(f"the registration makes {sorted(params_ports)} its parameter ports, and its header registers {sorted(head)}")
    for p, e in params_ports.items():
        if p not in leaves:
            why.append(f"parameter port {p!r} is no port of its circuit")
        elif e.get("root") in trees or e.get("domain") in trees:
            why.append(f"parameter port {p!r} is named at a window tree, not at the registration's root")
        elif p in head and head[p].get("value") != e.get("value"):
            why.append(f"parameter port {p!r} reads {e.get('value')!r}, and its header registers {head[p].get('value')!r}")
    entries, files, missing = {}, {}, []
    for j, (p, words) in enumerate(ports):
        if p in params_ports:
            continue
        r = roots.get(m.port(p))
        if r is None:
            missing.append(p)
            continue
        count, schema = leaves[p], m.schema(p)
        dom = window.domain(m.port(p), schema, count)
        if r.get("leaves") != count or r.get("domain") != dom.domain_id.hex():
            why.append(f"{p}: the registration's tree is {r.get('leaves')} leaves under domain {str(r.get('domain'))[:16]}…, "
                       f"not the window's {count} under {dom.domain_id.hex()[:16]}…")
            continue
        lv = string_levels(dom, schema, bcs[p])
        if lv[-1][0].hex() != r.get("root"):
            why.append(f"{p}: its served rows fold to {lv[-1][0].hex()[:16]}… under the window's domain, not the registered "
                       f"root {str(r.get('root'))[:16]}…")
            continue
        entries[p] = {"value": window_value(m.port(p)), "positions": list(range(count)),
                      "paths": [tree_path(lv, i).hex() for i in range(count)]}
        files[p] = {"value": window_value(m.port(p)), "root": r["root"], "leaves": count, "words": int(words), "domain": r["domain"],
                    **({} if schema == ROW_SCHEMA else {registered.SCHEMA_KEY: schema}), "reads": refs[:, j].tolist(),
                    registered.READS_FROM: m.base}
    if missing:
        why.insert(0, f"ports {missing} have no tree in the window's registration")
    if why:
        raise ValueError(f"{m.key}: " + "; ".join(why))
    return Complete({p: leaves[p] for p in names}, {**head, **entries}, out, not m.tables, tuple(files), tuple(params_ports),
                    {"format": registered.READS_V1, "ports": {**params_ports, **files}})


def flock_input_files(window_dir: str | Path, k: int, circuit: str | FlockCircuit, out_dir: str | Path, *,
                      write: Sequence[str] = ("pub", "inst"), files: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
                      completed: Complete | None = None) -> dict[str, Any]:
    """Member ``k``'s circuit-prover ``m<k>-pub-<n>.bin`` (``rows: false``) and ``m<k>-inst-<n>.bin``, under the header
    ``verity_flock.circuit.write`` gives the circuit it composed: its SHA-512 pin, its META's statement, subcircuit, block shape
    and class, the member's global unit indices, and its ``frame_v3``.  The bodies are the window's bytes, its prover body
    made from its window's files (`rows.copy_private`) and its header's paths read off the window's (`rows.registered_header`).
    ``circuit`` is the
    circuit's text or its `flock_circuit`, ``write`` the files to write (a member's ``inst`` is the prover's, wanted only once
    it is sampled), and ``files`` the window's ``window.json`` and ``registration.json`` as read (else read here).
    ``completed``: the member's files completed (`complete`), their header's ``shared_rows`` and ``registered`` and their
    public body its, the prover body port-major for a re-laid member."""
    if not set(write) <= {"pub", "inst"}:
        raise ValueError(f"flock_input_files writes pub and inst, not {sorted(set(write) - {'pub', 'inst'})}")
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
    if completed is not None:
        if mem.get("shared_rows") and mem["shared_rows"] != completed.shared_rows:
            raise ValueError(f"member {k}'s completed rows {completed.shared_rows} are not its window's {mem['shared_rows']}")
        head["shared_rows"], head["registered"] = completed.shared_rows, completed.registered
    else:
        if mem.get("shared_rows"):
            head["shared_rows"] = mem["shared_rows"]
        if mem.get("registered"):
            head["registered"] = registered_header(win, k)
    o.mkdir(parents=True, exist_ok=True)
    pub = completed.body if completed is not None else (d / f"m{k}-public.bin").read_bytes()
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
            copy_private(d, win, k, f, port_major=completed is not None and completed.relaid)
            f.write(pub)
    return got
