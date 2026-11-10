"""The workload Program's declared stop rule and LAG (`stop_rule`, `lag`), as the global match's chronology check (G6) reads them from its
global Program.

`pipeline.global_program` writes both blocks into `workload_program.json`, and the global match's global Program carries them
(`check.match.declaration.from_gprog`); the global match's mock Program (`declaration.mock_global_program`) writes the same blocks.  A
Program built before the stop rule was declared carries no `stop_rule`, and one built before LAG was declared carries no `lag`
block and no LAG in its stop rule; G6 then fails every early stop, or every EOS stop, by name.  Every writer of a stop rule writes
its `eos_token_ids`, and every writer of a `lag` block its `value`.

    d = DeclaredStop.of(gp, "the global Program")
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, optional, present, required


@dataclass(frozen=True)
class StopRule:
    eos_token_ids: Sequence[int] | None               # None: no EOS set declared; empty under `ignore_eos`
    ignore_eos: bool | None                           # written only when the manifest switches EOS off
    source: Mapping[str, Any] | None                  # where `pipeline.global_program` read the EOS ids; the mock Program writes none
    # one LAG under two names, both None on a Program built before LAG was declared
    lag: int | None
    eos_lag_rows: int | None
    eos_lag_source: str | None

    @classmethod
    def of(cls, doc: Mapping[str, Any], where: str) -> StopRule:
        k = "stop_rule"
        r = required(doc, k, where, Mapping)
        return cls(eos_token_ids=present(r, "eos_token_ids", where, (list, tuple), f"{k}.eos_token_ids"),
                   ignore_eos=optional(r, "ignore_eos", where, bool, f"{k}.ignore_eos"),
                   source=optional(r, "source", where, Mapping, f"{k}.source"), lag=optional(r, "lag", where, int, f"{k}.lag"),
                   eos_lag_rows=optional(r, "eos_lag_rows", where, int, f"{k}.eos_lag_rows"),
                   eos_lag_source=optional(r, "eos_lag_source", where, str, f"{k}.eos_lag_source"))


@dataclass(frozen=True)
class LagBlock:
    value: int
    source: str | None                                # the mock Program and `pipeline.global_program` write it; a hand-built Program may not

    @classmethod
    def of(cls, doc: Mapping[str, Any], where: str) -> LagBlock:
        b = required(doc, "lag", where, Mapping)
        return cls(value=required(b, "value", where, int, "lag.value"), source=optional(b, "source", where, str, "lag.source"))


@dataclass(frozen=True)
class DeclaredStop:
    where: str                                        # what handed the Program over: a refusal names it
    stop_rule: StopRule | None                        # None: a Program built before the stop rule was declared
    lag: LagBlock | None                              # None: a Program built before LAG was declared

    @classmethod
    def of(cls, doc: DeclaredStop | Mapping[str, Any], where: str) -> DeclaredStop:
        """The two blocks of a Program as its writer wrote it (`doc`, the parsed JSON); a record passes through."""
        if isinstance(doc, cls):
            return doc
        if not isinstance(doc, Mapping):
            raise RecordError(where, "stop_rule", f"cannot be read: the Program is a {type(doc).__name__}, not a JSON object")
        return cls(where=where, stop_rule=None if doc.get("stop_rule") is None else StopRule.of(doc, where),
                   lag=None if doc.get("lag") is None else LagBlock.of(doc, where))

    @property
    def declared_lag(self) -> int | None:
        """The LAG the Program declares: the `lag` block's, else its stop rule's."""
        if self.lag is not None:
            return self.lag.value
        if self.stop_rule is None:
            return None
        return self.stop_rule.lag if self.stop_rule.lag is not None else self.stop_rule.eos_lag_rows
