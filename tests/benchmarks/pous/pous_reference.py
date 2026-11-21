"""PoUS's run protocol's Python (`period.answer`) on a served case's bytes, the reference the stand-in Lean of these
tests answers with and the difference tests hold `pous-run-protocol` to. No served road runs it: the verdict there is
the Lean's alone."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from verity.protocols.accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.accounting.memory.pous.verifier import period
from verity.protocols.accounting.memory.pous.verifier import run as R


def pous_answer(case: Mapping[str, Any], *, schemes: Mapping[str, mc.Verify] = mc.SCHEMES,
                protocol: str = period.PROTOCOL) -> dict[str, Any]:
    """`period.answer` on ``case``, decoded from its own bytes; `ValueError` when it doesn't decode."""
    commitment, ev = json.loads(bytes.fromhex(case["commitment"])), json.loads(bytes.fromhex(case["evidence"]))
    p = case["params"]
    try:
        published, session = R.Setup.from_json(commitment), R.ChallengePhaseRecord.from_json(ev["session"])
        point, sid = period.point_of(p["point"]), bytes.fromhex(p["session_id"])
        trusted, digests = [bytes.fromhex(t) for t in p["trusted"]], bytes.fromhex(ev["digests"])
        inputs = period.setup_inputs(ev["setup"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the PoUS case doesn't decode: {type(e).__name__}: {e}") from None
    return period.answer(point, schemes, trusted, published, digests, session, sid, protocol, setup_inputs=inputs)
