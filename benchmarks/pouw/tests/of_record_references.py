"""`References`: `of_record.Record` with each check's Python reference alone, for the tests of the challenger's own logic,
which build no Lean.  The witness holds no Lean verdict.  The served path has no such mode: `Record` always takes Lean's."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import of_record as OR


class References(OR.Record):
    def _piece(self, check: str, exe: str, package: Path, fn: str, x: Any, reference: Callable[[Any], Any]) -> Any:
        py = OR._value(OR._python(reference, x))
        self._note(check, py, None, None)
        return py

    def prelude(self, python: str | None, lean: str | None) -> str | None:
        self._note("prelude", {"reason": python}, None, None)
        return python or lean

    def answer(self, case: dict) -> dict:
        py = OR.CC.case(copy.deepcopy(case))
        self._note("answer", py, None, None)
        return py
