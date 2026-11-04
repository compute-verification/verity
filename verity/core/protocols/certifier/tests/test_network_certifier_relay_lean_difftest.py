"""The difftest of `warden.Warden`'s schedule against the Lean package `certifier_device`'s `Relay.run`: the Python
half.

`verity/core/devices/certifier/lean/scripts/relay_vectors.py` writes 300 runs of the relay form on generated
connections, each with its arrival tick and its lane's `END`-or-`FAIL` index from a run of its own, and what this
package's real `warden.Warden` decides for each: refused, cut queued, or admitted at a tick with its lane's cells.
`check` runs the Lean half in its Lean audit (`runs` in that package's `lean-audit.json`):
`CertifierDevice/RelayDifftestMain.lean` runs `Relay.run` on each instance and fails on any decision that differs. The
audit checks the data against the record's SHA-256, and its pass is cached on the Lean package's files alone, so the test
here is what says the data is still what this package does: a change to the relay's refusal, admission, egress limit or
cut changes the data, fails this test, and `check.py --update` records the new data. The other test checks that the
generator reaches what its docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

LEAN = Path(__file__).resolve().parents[3] / "devices" / "certifier" / "lean"
SCRIPT = LEAN / "scripts" / "relay_vectors.py"
_spec = importlib.util.spec_from_file_location("certifier_relay_vectors", SCRIPT)
vectors = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["CertifierDevice.RelayDifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 300 instances and the relay's decisions on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{relay/instances.jsonl}", "{relay/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_covers_what_it_says():
    """Runs refuse connections, hold them queued past their tick, cap lanes, end them on their own, and cut them queued,
    in their delay and while sending; every answer shape is served; and an admitted lane's cells are its own end's,
    its cap or the cut's."""
    seen: Counter[str] = Counter()
    for seed in range(vectors.COUNT):
        inst = vectors.instance(seed)
        p, decisions = inst["schedule"], vectors.python_results(inst)["decisions"]
        horizon = inst["windows"] * p["window"]
        for (ready, last), (shape, _), d in zip(inst["conns"], inst["answers"], decisions, strict=True):
            if d in ("refused", "cut"):
                seen[d] += 1
                continue
            adm, cells = d
            seen[shape] += 1
            seen["waited"] += adm > ready
            sending = p["slots"] * max(0, horizon - adm - p["delay"])
            assert adm >= ready and cells == min(last + 1, p["egress_cells"], sending), (seed, ready, last, d)
            if cells == 0:
                seen["cut in its delay"] += 1
            elif cells == sending < min(last + 1, p["egress_cells"]):
                seen["cut sending"] += 1
            elif cells == p["egress_cells"] < last + 1:
                seen["capped"] += 1
            else:
                seen["ended"] += 1
    for kind in ["refused", "cut", "waited", "cut in its delay", "cut sending", "capped", "ended", *vectors.SHAPES]:
        assert seen[kind] > 10, (kind, seen)
