"""The challenger's checks of record (`served_zk.of_record.Record`) with what verity does not import, `verity_difftest`:
Lean's executables (`build`, `Server`) and the canonical JSON the two sides are compared in (`canonical`)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import verity_difftest as D

from verity.protocols.resource_accounting.compute.pouw.prover.kernels.served_zk import of_record as _of_record


class Record(_of_record.Record):
    def canonical(self, j: Any) -> str:
        return D.canonical(j)

    def start(self, exe: str, package: Path, *args: str) -> D.Server:
        return D.Server([str(self.exes.get(exe) or D.build(package, exe)), *args])


__all__ = ["Record"]
