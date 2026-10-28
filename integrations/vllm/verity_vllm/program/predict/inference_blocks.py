"""A window circuit's partition into inference blocks, as B emits it beside the circuit: a private table under a hiding root,
with only its budgets and padded counts public.

The partition is an annotation of the circuit (`window_circuit.lower`'s): a public query, the label rule over the root's
template instances, and a private table. Its labels are the class's slots, one inference block a slot: the slot's class
Program, every step of it. The table is the circuit's inference entries (`interface.InferenceEntry`), one row an entry and
never one a unit: unit ``at + i`` of the entry's template is in the block of ``i``'s slot digit. An entry's step digit is
when its units run, not which block holds them, so the table doesn't carry it.

**Statement** (public; `commit`)::

    {"format": FORMAT, "query": QUERY, "c": c, "k": k, "blocks": grid(m), "rows": grid(R), "root": "<hex>"}

as `records.inference_blocks.Statement` types it.

``m`` is the block count, ``R`` the table's rows, each rounded up to the public grid (`partition_v2.grid`), so the statement
gives a grid level of each and never the counts; the entries, and which units make which block, stay in the table.

**Budgets** (`budgets`). Every block has exactly ``c`` gates (each template's `Definition.gates` times its units in the
block): an entry's slot digit is of radix m, so it gives every block the same share of its units. ``k`` is a block's fresh
input in bits, |in(I) ∖ θ|, θ the weights (`window_builder.WEIGHTS`' parameters) held at their registered roots: what is
neither wired nor θ's (`interface.input_count`'s ``free`` and ``ingress`` with only the weights held), the scan state among
it, over the m blocks. Every slot is the same class Program lowered side by side and no wire crosses slots, so each block's
fresh input is the circuit's over m.

**Table** (private; `rows`, `decode`). `ROW_BYTES`-byte rows, integers little-endian: row 0 the header (`MAGIC`, the
SHA-512 of the circuit's Program, m, the entry count), then one row an entry: its template (`partition_v2.definition_digest`,
64 bytes), its first unit (8 bytes), its digit count and which digit is the slot (a byte each, then 6 zero bytes) and its
radix (4 bytes a digit, at most `DIGITS`); every unused byte 0. `commit` pads them with zero rows to grid(R) and registers
them with `verity.ml.commitments.advice` before any coin: hm96-sha512/row/v2 hiding leaves under fresh salts in one
frame-v3-sha512 tree, under `BINDING`.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from verity.ml.circuits import partition_object as PO
from verity.ml.circuits import partition_v2 as PV2
from verity.ml.commitments import advice as AC
from verity.ml.windows import interface as IF
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
    """The SHA-512 of ``circuit``'s Program (`partition_object.program_sha512`), the ``program`` its record names."""
    return PO.program_sha512(circuit.program()[0])


def rows(circuit: IF.Circuit, program: str | None = None) -> list[bytes]:
    """The table's rows (module docstring), unpadded; ``program`` is `program_digest`'s, made here unless given."""
    b = _blocks(circuit)
    program = program_digest(circuit) if program is None else program
    out = [_row(MAGIC, bytes.fromhex(program), _u(b.slots, 8), _u(len(b.inference), 8))]
    for e in b.inference:
        if len(e.radix) > DIGITS:
            raise ValueError(f"inference entry {e.doc()}: {len(e.radix)} digits, past a row's {DIGITS}")
        out.append(_row(bytes.fromhex(PV2.definition_digest(circuit.template(e.template))), _u(e.at, 8),
                        _u(len(e.radix), 1), _u(e.axes.index("slot"), 1), bytes(6), *(_u(r, 4) for r in e.radix)))
    return out


def decode(rs: Sequence[bytes]) -> tuple[str, int, list[tuple[str, int, tuple[int, ...], int]]]:
    """The table `rows` wrote (padding allowed): the Program's SHA-512, m and each entry's (template digest, first unit,
    radix, slot digit)."""
    head = rs[0]
    if head[:len(MAGIC)] != MAGIC:
        raise ValueError(f"not a {FORMAT} table")
    program, m, n = head[32:96].hex(), int.from_bytes(head[96:104], "little"), int.from_bytes(head[104:112], "little")
    entries = []
    for r in rs[1:1 + n]:
        d = r[72]
        entries.append((r[:64].hex(), int.from_bytes(r[64:72], "little"),
                        tuple(int.from_bytes(r[80 + 4 * j:84 + 4 * j], "little") for j in range(d)), r[73]))
    return program, m, entries


def budgets(circuit: IF.Circuit) -> tuple[int, int]:
    """``(c, k)``: every block's gates and its fresh-input bits (module docstring)."""
    b = _blocks(circuit)
    c = sum(e.units // b.slots * circuit.template(e.template).gates for e in b.inference)
    theta = {p: _ROOT for p in (circuit.registered.params if circuit.registered is not None else {})
             if p.split("/", 1)[0] == WEIGHTS}
    count = IF.input_count(circuit.description(), pins=theta, roots=theta)
    fresh = count["free"] + count["ingress"]
    if fresh % b.slots:
        raise ValueError(f"{circuit.name}: {fresh} fresh input bits over {b.slots} slots, not a slot's share each")
    return c, fresh // b.slots


def commit(circuit: IF.Circuit, program: str | None = None,
           salts: Sequence[bytes] | None = None) -> tuple[dict[str, Any], AC.Registration]:
    """``circuit``'s partition into inference blocks registered (module docstring): its public statement and the
    registration, the prover's to keep (the rows, their salts, the tree). ``program`` as `rows` takes it."""
    rs = rows(circuit, program)
    padded = rs + [bytes(ROW_BYTES)] * (PV2.grid(len(rs)) - len(rs))
    reg = AC.register(padded, ROW_BITS, BINDING, salts)
    c, k = budgets(circuit)
    st = RIB.Statement(query=QUERY, c=c, k=k, blocks=PV2.grid(_blocks(circuit).slots), rows=len(padded), root=reg.root.hex())
    return st.dump(), reg


__all__ = ["BINDING", "DIGITS", "FORMAT", "MAGIC", "QUERY", "ROW_BYTES", "budgets", "commit", "decode", "program_digest",
           "rows"]
