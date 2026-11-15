"""The Lean verifier's ``--registered`` port (`verity/registered-reads/v1`) as the workload side writes it: a served
compiler's port over a registered value, and the key of the proof block its reads start at. The workload side's own copy
of the format, loading no verifier module: the verifier reads its ports with its own (`one_stage.produced.port`,
`one_stage.registered`), and `tests/test_records_audit.py` holds the two equal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.commitments.domains import REGISTERED_READS_V1 as READS_V1
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2

__all__ = ["READS_FROM", "READS_V1", "port"]

#: the `READS_V1` port's key: proof block ``reads_from + i`` reads ``reads[i]``, every other proof block nothing
READS_FROM = "reads_from"
#: a registered value's row formats; a port of any but the first names its own under `SCHEMA_KEY`
SCHEMAS = (SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG)
SCHEMA_KEY = "row_schema"
#: the leaf width an entry that names none has
WORD_BITS = 16


def port(value: str, e: Mapping[str, Any], root: str, rows: Sequence[int | None],
         schema: str = SCHEMA_HM96_SHA512_ROW) -> dict[str, Any]:
    """The port reading ``value`` at ``rows`` (one per proof block, None where the proof block doesn't read it): ``e`` its
    entry (``leaves``, ``words``, ``domain``, and ``word_bits`` unless 16) and ``root`` its docket's root.
    Refused (ValueError) for rows of another format than `SCHEMAS`."""
    if schema not in SCHEMAS:
        raise ValueError(f"a registered value's rows are {', '.join(SCHEMAS)} rows, not {schema!r}")
    return {"value": value, "root": root, "leaves": int(e["leaves"]), "words": int(e["words"]),
            "word_bits": int(e.get("word_bits", WORD_BITS)), "domain": e["domain"],
            **({} if schema == SCHEMA_HM96_SHA512_ROW else {SCHEMA_KEY: schema}), "reads": list(rows)}
