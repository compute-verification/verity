"""The state one GM-01 run's named checks share: its inputs, the decoded record and fold, and the document they append to."""
from __future__ import annotations

import dataclasses

from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.check.match import instance_sequence as SEQ


class NamedCheck:
    """One named check of the document: its problems and evidence; `done` appends it to `res["checks"]`."""

    def __init__(self, res: dict, name: str, what: str):
        self.res, self.name, self.what = res, name, what
        self.problems: list[str] = []
        self.info: dict = {}

    def fail(self, msg: str) -> None:
        self.problems.append(msg)

    def done(self) -> bool:
        self.res["checks"].append({"id": self.name, "what": self.what, "status": "FAIL" if self.problems else "PASS",
                                   "problems": self.problems[:20], "n_problems": len(self.problems), **self.info})
        return not self.problems


@dataclasses.dataclass(frozen=True)
class MatchRun:
    """One `check` run as its named checks read it: the global Program, the record (renamed to the Program's `r<k>`), the fold and the
    options; `res` is the document every check appends to, `declared` the Program's requests by id."""
    gp: dict
    oracle_dir: str
    tokens_path: str
    within_step: str
    fast: bool
    ops: BD.Ops
    loaded: dict
    tokens: dict | None
    tokens_optional: bool
    rec: dict
    res: dict
    fold: SEQ.Prog
    declared: dict[str, dict]

    @property
    def header(self) -> dict:
        return self.rec["header"]

    @property
    def steps(self) -> dict:
        return self.rec["steps"]

    @property
    def samples(self) -> dict:
        return self.rec["samples"]
