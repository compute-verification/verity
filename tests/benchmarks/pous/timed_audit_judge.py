"""A stand-in for `pous-timed-audit READY LEAVES AUDIT` in the CPU tests, which build no Lean: the Python reference's
verdict (`timed_audit.verdict`, which `pous_check`'s difftest holds the Lean's to), printed as the Lean prints it.
`exe(directory)` makes an executable running it, as `timed_audit.py --lean-exe` and `bench.py --pous-timed-audit` take
one; `exe(directory, reject=True)` one whose verdict rejects every record."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def exe(directory: Path, reject: bool = False) -> Path:
    path = directory / ("pous-timed-audit-rejecting" if reject else "pous-timed-audit")
    path.write_text(f"#!/bin/sh\nexec {sys.executable} {Path(__file__).resolve()}{' --reject' if reject else ''} \"$@\"\n")
    path.chmod(0o755)
    return path


def main(argv: list[str]) -> int:
    from verity.protocols.accounting.pous.device.timed_audit import verdict

    reject = argv[:1] == ["--reject"]
    ready, leaves, doc = argv[1:] if reject else argv
    v = verdict(json.loads(Path(ready).read_text()), Path(leaves).read_bytes(), json.loads(Path(doc).read_text()))
    if reject:
        v = {"fault": "record"}
    print(json.dumps(v))
    return 0 if v.get("accepted") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
