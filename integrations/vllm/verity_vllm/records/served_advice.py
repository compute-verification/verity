"""serve-openai's registration of B (`verity-vllm/served-advice-registration/v1`, ``advice_registration.json``), written once
before the first Commit period (`pipeline.serve_advice.register`): B as the verifier recomputes it, the advice's format, ℓ_a
and the commitment's row width, the day the run serves, the deployment's registration log B's entries extend (its composite,
head and committed log) and when it was written, as `serve_advice.log_doc` reads the log back to extend it."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity_vllm.records.fields import RecordError, only, required

SCHEMA = "verity-vllm/served-advice-registration/v1"
FIELDS = ("schema", "compiler", "advice", "cap_bits", "row_bits", "day", "registration", "t")


@dataclass(frozen=True)
class ServedAdviceRegistration:
    compiler: Mapping[str, Any]                       # B's document (`served_compiler.compiler_doc`)
    advice: str                                       # the advice's format (`window_advice.SCHEMA`)
    cap_bits: int                                     # ℓ_a, the length every period's advice is padded to
    row_bits: int
    day: int                                          # the day the log declares B for
    composite: str                                    # the deployment whose log it is
    head: str
    log: Mapping[str, Any]                            # the committed log (`verity.core.protocols.registration.committed`)
    t: float                                          # unix seconds

    @classmethod
    def of(cls, doc: Any, where: str) -> ServedAdviceRegistration:
        """The registration as its writer wrote it, refused by field (`fields.RecordError`) where it isn't one."""
        if not isinstance(doc, Mapping):
            raise RecordError(where, "schema", f"cannot be read: the registration is a {type(doc).__name__}, not a JSON object")
        if doc.get("schema") != SCHEMA:
            raise RecordError(where, "schema", f"is {doc.get('schema')!r}, not {SCHEMA}")
        only(doc, FIELDS, where)
        reg = required(doc, "registration", where, Mapping)
        only(reg, ("composite", "head", "log"), where, "registration.")
        return cls(compiler=required(doc, "compiler", where, Mapping), advice=required(doc, "advice", where, str),
                   cap_bits=required(doc, "cap_bits", where, int), row_bits=required(doc, "row_bits", where, int),
                   day=required(doc, "day", where, int),
                   composite=required(reg, "composite", where, str, "registration.composite"),
                   head=required(reg, "head", where, str, "registration.head"),
                   log=required(reg, "log", where, Mapping, "registration.log"), t=required(doc, "t", where, (int, float)))

    def dump(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "compiler": dict(self.compiler), "advice": self.advice, "cap_bits": self.cap_bits,
                "row_bits": self.row_bits, "day": self.day,
                "registration": {"composite": self.composite, "head": self.head, "log": dict(self.log)}, "t": self.t}
