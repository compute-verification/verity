"""The in-process engine's request ids, mapped back to *ours*.

``LLM.generate`` numbers requests itself; ``LLMEngine.add_request(request_id=...)`` keeps the
id we assign (vLLM appends ``-<8 random chars>`` internally; ``external_id`` strips it), so the
ids seen by the scheduler / model runner map back to the workload's uuids exactly.  The loop that
adds them and steps the engine is `engine.arrivals.run_requests_arrivals`.
"""

from __future__ import annotations

import re

_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def external_id(internal_req_id: str) -> str:
    """vLLM internal request id -> the id we passed to add_request."""
    return _SUFFIX.sub("", internal_req_id, count=1)
