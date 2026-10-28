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

from verity.ml.windows.rows import SCHEME, copy_private, registered_header

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


def flock_input_files(window_dir: str | Path, k: int, circuit: str | FlockCircuit, out_dir: str | Path, *,
                      write: Sequence[str] = ("pub", "inst"), files: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
                      ) -> dict[str, Any]:
    """Member ``k``'s circuit-prover ``m<k>-pub-<n>.bin`` (``rows: false``) and ``m<k>-inst-<n>.bin``, under the header
    ``verity_flock.circuit.write`` gives the circuit it composed: its SHA-512 pin, its META's statement, subcircuit, block shape
    and class, the member's global unit indices, and its ``frame_v3``.  The bodies are the window's bytes, its prover body
    made from its window's files (`rows.copy_private`) and its header's paths read off the window's (`rows.registered_header`).
    ``circuit`` is the
    circuit's text or its `flock_circuit`, ``write`` the files to write (a member's ``inst`` is the prover's, wanted only once
    it is drawn), and ``files`` the window's ``window.json`` and ``registration.json`` as read (else read here)."""
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
    if mem.get("registered"):
        head["registered"] = registered_header(win, k)
    o.mkdir(parents=True, exist_ok=True)
    pub = (d / f"m{k}-public.bin").read_bytes()
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
