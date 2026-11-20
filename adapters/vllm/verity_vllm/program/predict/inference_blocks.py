"""A window circuit's partition into inference blocks, as B emits it beside the circuit: a private table under a hiding root,
with only its budgets and padded counts public.

The partition is an annotation of the circuit (`window_circuit.lower`'s): a public query, the label rule over the root's
template instances, and a private table. Its labels are the requests' slots, one inference block a slot: the slot's
Program, every step of it. The table is the circuit's inference entries (`interface.InferenceEntry`), one row an entry and
never one a unit: unit ``at + i`` of the entry's template is in the block of its first slot plus ``i``'s slot digit. An
entry with no slot digit holds units of no request: they are in no inference block. An entry's step digit is when its
units run, not which block holds them, so the table doesn't carry it.

**Statement** (public; `commit`)::

    {"format": FORMAT, "query": QUERY, "c": c, "k": k, "blocks": grid(m), "rows": grid(R), "root": "<hex>",
     "others": {"gates": g, "fresh": f}}

as `records.inference_blocks.InferenceBlocks` types it.

``m`` is the block count, ``R`` the table's rows, each rounded up to the public grid (`partition_v2.grid`), so the statement
gives a grid level of each and never the counts; the entries, and which units make which block, stay in the table.

``others`` is there only where the circuit has units of no request: their gates and fresh-input bits, which no block
holds.

**Budgets** (`counts`, `budgets`). Every block has exactly ``c`` gates (each template's `Definition.gates` times its units
in the block) and ``k`` fresh-input bits, |in(I) ∖ θ|, θ the weights (`window_builder.WEIGHTS`' parameters) held at their
registered roots: the rows its units read from registered parameters outside θ, the scan state among them. Those reads are
all of the circuit's fresh input (`interface.input_count`'s ``free`` and ``ingress`` with only the weights held), or the
counts are refused; so are blocks that differ in either.

**Table** (private; `rows`, `decode`). `ROW_BYTES`-byte rows, integers little-endian: row 0 the header (`MAGIC`, the
SHA-512 of the circuit's Program, m, the entry count), then one row an entry: its template (`partition_v2.id_digest`,
64 bytes), its first unit (8 bytes), its digit count and which digit is the slot (a byte each, `NO_REQUEST` for an
entry of units of no request), its first slot (4 bytes, then 2 zero bytes) and its radix (4 bytes a digit, at most
`DIGITS`); every unused byte 0. `commit` pads them with zero rows to grid(R) and commits
them with `verity.commitments.advice` before any coin: hm96-sha512/row/v2 hiding leaves under fresh salts in one
frame-v3-sha512 tree, under `BINDING`.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from verity.circuits import codec
from verity.circuits import partition_v2 as PV2
from verity.commitments import advice as AC
from verity.commitments.periods import interface as IF

from verity_vllm.records import inference_blocks as RIB

from .window_builder import WEIGHTS

FORMAT = RIB.SCHEMA
#: the label rule over the root's template instances: each instance's block is its entry's slot digit
QUERY = {"name": "label", "version": 0, "params": {"family": "instances(root)", "by": "slot"}}
ROW_BYTES = PV2.ROW_BYTES
ROW_BITS = 8 * ROW_BYTES
MAGIC = FORMAT.encode() + b"\0"
BINDING = hashlib.sha512(FORMAT.encode() + b"/advice").digest()[:32]
DIGITS = 12
#: the slot digit byte of an entry of units of no request
NO_REQUEST = 0xFF
_ROOT = "00" * 64


def _row(*parts: bytes) -> bytes:
    b = b"".join(parts)
    if len(b) > ROW_BYTES:
        raise AssertionError("a row overflows")
    return b + bytes(ROW_BYTES - len(b))


def _u(x: int, n: int) -> bytes:
    return int(x).to_bytes(n, "little")


def _blocks(circuit: IF.Circuit) -> IF.Blocks:
    if circuit.blocks is None:
        raise ValueError(f"{circuit.name}: no inference entries to partition")
    return circuit.blocks


def program_digest(circuit: IF.Circuit) -> str:
    """The SHA-512 of ``circuit``'s Program (`codec.program_digest`), the ``program`` its record names."""
    return codec.program_digest(circuit.program()[0])


def rows(circuit: IF.Circuit, program: str | None = None) -> list[bytes]:
    """The table's rows (module docstring), unpadded; ``program`` is `program_digest`'s, made here unless given."""
    b = _blocks(circuit)
    program = program_digest(circuit) if program is None else program
    out = [_row(MAGIC, bytes.fromhex(program), _u(b.slots, 8), _u(len(b.inference), 8))]
    for e in b.inference:
        if len(e.radix) > DIGITS:
            raise ValueError(f"inference entry {e.doc()}: {len(e.radix)} digits, past a row's {DIGITS}")
        out.append(_row(bytes.fromhex(PV2.id_digest(circuit.template(e.template))), _u(e.at, 8),
                        _u(len(e.radix), 1), _u(e.axes.index("slot") if e.request else NO_REQUEST, 1), _u(e.slot, 4),
                        bytes(2), *(_u(r, 4) for r in e.radix)))
    return out


def decode(rs: Sequence[bytes]) -> tuple[str, int, list[tuple[str, int, tuple[int, ...], int | None, int]]]:
    """The table `rows` wrote (padding allowed): the Program's SHA-512, m and each entry's (template digest, first unit,
    radix, slot digit, first slot), the slot digit None for units of no request."""
    head = rs[0]
    if head[:len(MAGIC)] != MAGIC:
        raise ValueError(f"not a {FORMAT} table")
    program, m, n = head[32:96].hex(), int.from_bytes(head[96:104], "little"), int.from_bytes(head[104:112], "little")
    entries = []
    for r in rs[1:1 + n]:
        d = r[72]
        entries.append((r[:64].hex(), int.from_bytes(r[64:72], "little"),
                        tuple(int.from_bytes(r[80 + 4 * j:84 + 4 * j], "little") for j in range(d)),
                        None if r[73] == NO_REQUEST else r[73], int.from_bytes(r[74:78], "little")))
    return program, m, entries


def counts(circuit: IF.Circuit) -> tuple[list[int], list[int], int, int]:
    """Each slot's block's gates and fresh-input bits, then the units of no request's, which are in no slot's block: a
    template's `Definition.gates` a unit, and the bits of the rows a unit reads from registered parameters outside θ.
    Refused where those reads are not all of the circuit's fresh input (`interface.input_count`'s ``free`` and
    ``ingress`` with only the weights held)."""
    b = _blocks(circuit)
    gates, none = b.per_slot({t: circuit.template(t).gates for t in circuit.units})
    slots = {t: b.slots_steps(t, 0, n)[0] + 1 for t, n in circuit.units.items()}
    lays, fresh = circuit.layouts, np.zeros(b.slots + 1, dtype=np.int64)
    for r in (circuit.registered.reads if circuit.registered is not None else ()):
        if r.param.split("/", 1)[0] != WEIGHTS and r.port in dict(lays[r.reader][0].rows):
            bits = 16 * dict(lays[r.reader][0].rows)[r.port] // int(circuit.cuts.get(r.reader, {}).get(r.port, 1))
            fresh += np.bincount(slots[r.reader][r.at + np.arange(r.units) // r.cut[1]], minlength=b.slots + 1) * bits
    theta = {p: _ROOT for p in (circuit.registered.params if circuit.registered is not None else {})
             if p.split("/", 1)[0] == WEIGHTS}
    count = IF.input_count(circuit.description(), pins=theta, roots=theta)
    if int(fresh.sum()) != count["free"] + count["ingress"]:
        raise ValueError(f"{circuit.name}: {count['free'] + count['ingress']} fresh input bits, of which the registered reads "
                         f"outside θ give blocks {int(fresh.sum())}: the rest is in no block")
    return gates, [int(f) for f in fresh[1:]], none, int(fresh[0])


def budgets(circuit: IF.Circuit) -> tuple[int, int]:
    """``(c, k)``: every request's block's gates and fresh-input bits (module docstring); refused by name where the
    request blocks differ in either."""
    gates, fresh, _, _ = counts(circuit)
    if len(set(gates)) != 1:
        raise ValueError(f"{circuit.name}: request blocks of unequal gates {gates}: the partition gives every block c gates")
    if len(set(fresh)) != 1:
        raise ValueError(f"{circuit.name}: request blocks of unequal fresh-input bits {fresh}: the partition gives every "
                         "block k fresh-input bits")
    return gates[0], fresh[0]


def commit(circuit: IF.Circuit, program: str | None = None,
           salts: Sequence[bytes] | None = None) -> tuple[dict[str, Any], AC.Committed]:
    """``circuit``'s partition into inference blocks committed (module docstring): its public statement and the
    commitment, the prover's to keep (the rows, their salts, the tree). ``program`` as `rows` takes it."""
    rs = rows(circuit, program)
    padded = rs + [bytes(ROW_BYTES)] * (PV2.grid(len(rs)) - len(rs))
    reg = AC.commit(padded, ROW_BITS, BINDING, salts)
    gates, fresh, g0, f0 = counts(circuit)
    c, k = budgets(circuit)
    st = RIB.InferenceBlocks(query=QUERY, c=c, k=k, blocks=PV2.grid(_blocks(circuit).slots), rows=len(padded),
                             root=reg.root.hex(), others={"gates": g0, "fresh": f0} if g0 or f0 else None)
    return st.dump(), reg


__all__ = ["BINDING", "DIGITS", "FORMAT", "MAGIC", "NO_REQUEST", "QUERY", "ROW_BYTES", "budgets", "commit", "counts",
           "decode", "program_digest", "rows"]
