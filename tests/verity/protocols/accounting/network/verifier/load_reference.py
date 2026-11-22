"""The Python reference of the check of a load's registrations, which the Lean verifier makes
(`NetworkVerifier.Load.loadCheck`, `relay_check.check_load`): `transfer_rows.check_registration` of each of the shape's
values. Nothing on a verifying path reads it; the agreement test holds the Lean check to it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity.protocols.accounting.network.verifier import transfer_rows as TR
from verity.protocols.accounting.network.verifier.certificate import Direction
from verity.protocols.accounting.network.verifier.load_check import FORMAT, KIND, WINDOW, manifest, shape_named


def check_load(doc: Mapping[str, Any], key: Any, session: bytes) -> tuple[dict[str, Any], list[str]]:
    """A ``load.json`` read as the verifier reads it: each of its shape's values a registration that
    `transfer_rows.check_registration` accepts under `key` and `session` for the value it expects, at the document's
    program and link. Returns the count and the problems."""
    try:
        shape = shape_named(doc.get("shape"))
        if doc.get("format") != FORMAT or doc.get("window") != WINDOW or not isinstance(doc.get("link"), str):
            raise ValueError(f"not a load's registrations ({FORMAT}, window {WINDOW})")
    except ValueError as e:
        return {"values": 0, "registered": 0}, [f"load: {e}"]
    values, regs = manifest(shape), doc.get("values")
    if not isinstance(regs, list) or len(regs) != len(values):
        return {"values": len(values), "registered": 0}, [f"load: not one registration of each of {len(values)} values"]
    problems, ok = [], 0
    for v, reg in zip(values, regs):
        try:
            TR.check_registration(reg, key, session, link=doc["link"], direction=Direction.INGRESS, window=WINDOW,
                                  kind=KIND, value=v.name, program=doc.get("program"), words=v.reg_words, rows=v.leaves,
                                  word_bits=v.word_bits, schema=v.schema)
            ok += 1
        except ValueError as e:
            problems.append(f"load {v.name}: {e}")
    return {"values": len(values), "registered": ok}, problems
