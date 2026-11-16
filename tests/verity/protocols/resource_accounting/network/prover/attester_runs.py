"""The runs the chain's tests read (`network_attester_chain`), and the tests of what reads the chain under
`tests/benchmarks/network_traces/` with them: the synthetic network attester run in the gate's layout (`gate_run`), and a
certifier rehearsal's run, which every reader refuses by name (`certifier_rehearsal`). ``G`` is the gate's auditor,
which publishes the run's ``out/``."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6] / "verity/protocols/compose/verifier/live_challenger"))

import auditor as G
from verity.protocols.resource_accounting.network.prover import network_attester_chain as WC
from tests.verity.protocols.resource_accounting.network.prover.test_scan_zk import published


def gate_run(d: Path, seed: bytes = b"warden-run") -> Path:
    """The synthetic network attester run in the gate's layout, under the network attester key ``seed`` makes: its
    files under ``witness/`` with the check the runner writes there, which accepts it, and under ``out/`` what the
    gate publishes of them (`auditor.network_attester_outcome`), its lines among them."""
    run = published(d / "flat", seed=seed)
    (d / "gate" / "out").mkdir(parents=True)
    shutil.copytree(run, d / "gate" / "witness")
    (d / "gate" / "witness" / "check.json").write_text(json.dumps(WC.RC.check(d / "gate" / "witness")))
    outcome = G.network_attester_outcome(d / "gate" / "out", d / "gate" / "witness")
    assert outcome["verdict"]["accepted"] is True, "the gate publishes a run's lines only when its check accepted"
    assert "unruled" not in outcome, "a run's own lines are in the forms the gate publishes them in"
    return d / "gate"


def certifier_rehearsal(root: Path) -> Path:
    """A certifier rehearsal's run as a fetched one holds it: its plan and calibration, and a deployment.json naming the
    certifier's key and links, which no active network attester's does."""
    run = root / "honest"
    run.mkdir()
    docs = {"plan.json": {"windows": 2}, "calibration.json": {"grid": {}},
            "deployment.json": {"certifier_key": "00", "links": {"n2": "served"}, "session": "00" * 32, "run_id": "r"}}
    for name, doc in docs.items():
        (run / name).write_text(json.dumps(doc))
    return run
