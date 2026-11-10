"""The driver's served tokens of one run (`capture/tokens.json`), as the global match's chronology check (G6) reads them.

Three writers write it: the one-process capture (`pipeline.m1_capture`), the TP capture (`pipeline.tp.capture`) and the TP
Match arms (`pipeline.tp.match`).  Each writes `requests`, and per request its `request_id` and `generated_token_ids`; those are
required here.  Only the one-process capture writes the realized schedule, the served and executed lengths and the scheduler's
`lag`, so on a TP record they are None and G6 has no driver claim to hold them to.  The prompt and the cap are the Program's,
which both read instead, so they are not typed here.

    t = ServedTokens.load(os.path.join(oracle_dir, "capture", "tokens.json"))
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.records.fields import RecordError, objects, optional, present, required


@dataclass(frozen=True)
class ServedRequest:
    request_id: str
    generated_token_ids: Sequence[int]
    # the one-process capture's alone, and null there too for a step the engine did not report
    arrive_step_realized: int | None
    first_step: int | None
    finish_step: int | None
    finish_reason: str | None
    served_output_len: int | None
    executed_prefix_len: int | None                   # also null on an EOS stop under an unattested lag

    @classmethod
    def of(cls, doc: Mapping[str, Any], where: str, name: str) -> ServedRequest:
        def opt(key: str, kind: type) -> Any:
            return optional(doc, key, where, kind, f"{name}.{key}")
        return cls(request_id=required(doc, "request_id", where, str, f"{name}.request_id"),
                   generated_token_ids=required(doc, "generated_token_ids", where, (list, tuple), f"{name}.generated_token_ids"),
                   arrive_step_realized=opt("arrive_step_realized", int), first_step=opt("first_step", int),
                   finish_step=opt("finish_step", int), finish_reason=opt("finish_reason", str),
                   served_output_len=opt("served_output_len", int), executed_prefix_len=opt("executed_prefix_len", int))


@dataclass(frozen=True)
class ServedTokens:
    where: str                                        # the file it was read from: a refusal names it
    requests: Sequence[ServedRequest]
    # `lag.value`, the LAG the engine's resolved async_scheduling implies; None: a TP record (no `lag` block), or an engine
    # that did not report async_scheduling
    lag: int | None

    @property
    def by_request(self) -> dict[str, ServedRequest]:
        return {r.request_id: r for r in self.requests}

    @classmethod
    def of(cls, doc: ServedTokens | Mapping[str, Any], where: str) -> ServedTokens:
        """The record as its writer wrote it (`doc`, the parsed JSON); a record passes through."""
        if isinstance(doc, cls):
            return doc
        if not isinstance(doc, Mapping):
            raise RecordError(where, "requests", f"cannot be read: the record is a {type(doc).__name__}, not a JSON object")
        reqs = objects(doc, "requests", where)
        lag = optional(doc, "lag", where, Mapping)
        return cls(where=where, requests=tuple(ServedRequest.of(r, where, f"requests[{i}]") for i, r in enumerate(reqs)),
                   lag=None if lag is None else present(lag, "value", where, int, "lag.value"))

    @classmethod
    def load(cls, path: str | Path) -> ServedTokens:
        with open(path) as f:
            return cls.of(json.load(f), str(path))
