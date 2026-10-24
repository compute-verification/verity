"""Scratch probe (not for main): m1_unit.py with `circuit._ranges` and `_slot_log` traced, so a unit compose refuses for its
block still prints each range it asked for: PROBE {json} lines, then m1_unit's own."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verity_flock import circuit as FC  # noqa: E402

_ranges, _slot_log = FC._ranges, FC._slot_log


def ranges(items):
    r = _ranges(items)
    print("PROBE " + json.dumps({"k_log": r[1], "items": [[k, lg, n, d.get("net")] for k, lg, n, d in items]}), flush=True)
    return r


def slot_log(useful):
    lg = _slot_log(useful)
    print("PROBE " + json.dumps({"useful": useful, "slot_log": lg}), flush=True)
    return lg


FC._ranges, FC._slot_log = ranges, slot_log
import m1_unit  # noqa: E402

raise SystemExit(m1_unit.main())
