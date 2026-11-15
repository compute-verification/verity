"""``flock-tables`` files: multi-table statements with private glue (``flock_live::tables``), several tables in one Flock union
commitment, each a pinned ``flock-ir-unit/v2`` template repeated over its blocks, glued by aligned sub-cube relations."""
from __future__ import annotations

import hashlib
import json

FORMAT = "flock-tables"
STATEMENT = "verity/flock-tables"


def region(table: str, fixed: dict[int, int], free: list[int]) -> dict:
    """A sub-cube of ``table``'s local index (in-block bits ``0..k``, then block bits ``k..k+b``): every bit fixed or free."""
    return {"table": table, "fixed": {str(b): int(v) for b, v in sorted(fixed.items())}, "free": [int(b) for b in free]}


def compose(name: str, tables: list[dict], circuits: dict[str, str], glue: list[dict]) -> str:
    """The file: ``tables`` (``{name, circuit, blocks}``, in the union's slot order, ``k_log`` descending), each naming one of
    ``circuits`` (``{name: flock-ir-unit/v2 text}``), and ``glue`` (``{dst, src, map}``: :func:`region` twice and ``"id"``,
    ``{"perm": [...]}`` or ``{"broadcast": [...]}``)."""
    if {t["circuit"] for t in tables} - set(circuits):
        raise ValueError("a table names a circuit the file does not carry")
    meta = {"format": FORMAT, "statement": STATEMENT, "tables": tables, "glue": glue}
    out = [f"{FORMAT} {name}\n", f"META {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n"]
    for cname, text in circuits.items():
        body = text if text.endswith("\n") else text + "\n"
        out.append(f"CIRCUIT {cname} {body.count(chr(10))}\n{body}")
    return "".join(out)


def digest(text: str) -> str:
    """The file's pin: its SHA-512."""
    return hashlib.sha512(text.encode()).hexdigest()
