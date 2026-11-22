"""The prover's produced tree: a produced value's rows committed under the domain SPOT's `one_stage.produced.domain` names."""

from __future__ import annotations

from collections.abc import Sequence

from verity.commitments import registered_value as RV
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW


def commit(program: str, value: str, rows: Sequence[Sequence[int]], docket: str, salts: Sequence[bytes] | None = None,
           word_bits: int = RV.WORD_BITS, schema: str = SCHEMA_HM96_SHA512_ROW) -> RV.Registered:
    """``rows`` committed as a registered value's with ``"produced": docket`` in its domain's binding, its `entry()` the
    record's and its `statement` the opened reads."""
    return RV.Registered.commit(program, value, rows, salts, word_bits, schema, produced=docket)


__all__ = ["commit"]
