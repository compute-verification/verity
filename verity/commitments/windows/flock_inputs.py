"""Flock's input files for a window's member (``serving_rows.write``'s window): the public file
``m<k>-pub-<n>.bin`` (``rows: false``) and the prover's ``m<k>-inst-<n>.bin``, each the window's bodies under the header
``verity_flock.circuit.write`` gives the circuit the verifier composed for the member's template.  A chunked template's chunks
are all staged under one circuit, whose META, pin and class (`FlockCircuit`) are read off it once."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.commitments.windows.rows import (BC_BYTES, SCHEME, copy_private, held_ports, own_entries,
                                             registered_header)

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


def identity_refs(n: int, ports: int) -> np.ndarray:
    """A re-laid body's refs: each of ``n`` instances' ref to its own row of each of ``ports`` ports."""
    return np.repeat(np.arange(n, dtype="<u4")[:, None], ports, axis=1)


def relay(body: bytes, n: int, ports: int) -> bytes:
    """An instance-major public body, ``n`` instances of ``ports`` rows' b || c, re-laid port-major under identity refs:
    each port's rows in instance order, then `identity_refs`."""
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


def flock_input_files(window_dir: str | Path, k: int, circuit: str | FlockCircuit, out_dir: str | Path, *,
                      write: Sequence[str] = ("pub", "inst"), files: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
                      ) -> dict[str, Any]:
    """Member ``k``'s circuit-prover ``m<k>-pub-<n>.bin`` (``rows: false``) and ``m<k>-inst-<n>.bin``, under the header
    ``verity_flock.circuit.write`` gives the circuit it composed: its SHA-512 pin, its META's statement, subcircuit, block shape
    and class, the member's global unit indices, and its ``frame_v3``.  The bodies are the window's bytes, its prover body
    made from its window's files (`rows.copy_private`) and its header's paths read off the window's (`rows.registered_header`).
    A member that reads a registered value, or holds a wired port in its writer's tree (`rows.held_ports`), has every other
    port registered too, as rows of its own tree or its writer's (`rows.own_entries`), the reads the verifier adds at the
    registration's roots (`rows.own_reads`). ``circuit`` is the
    circuit's text or its `flock_circuit`, ``write`` the files to write (a member's ``inst`` is the prover's, wanted only once
    it is sampled), and ``files`` the window's ``window.json`` and ``registration.json`` as read (else read here)."""
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
    if mem.get("shared_rows"):
        head["shared_rows"] = mem["shared_rows"]
    pub = (d / f"m{k}-public.bin").read_bytes()
    held = held_ports(win, k)
    if mem.get("registered") or held:
        reads = registered_header(win, k) if mem.get("registered") else {}
        head["registered"] = {**reads, **own_entries(mem, pub, reads, held)}
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
