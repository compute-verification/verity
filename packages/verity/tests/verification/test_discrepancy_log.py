"""``fixtures/discrepancy_log.json`` is well formed, and nothing leaves ``open`` without a resolution and evidence.

The log records where the paper, the pinned simulator, Verity's reference and the hardware disagree about the
frozen target.  The rule it enforces: an entry becomes ``resolved`` or ``excluded`` only by writing down what was
decided and pointing at evidence that exists -- never merely by changing the reference implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

from verity.verification.target import DISCREPANCY_LOG, FIRST, HAWKEYE_PIN

ROOT = Path(__file__).resolve().parents[4]
REQUIRED = ("id", "profile", "issue", "hawkeye_paper", "hawkeye_simulator", "verity_reference",
            "hardware_evidence", "status", "resolution", "evidence_refs", "owner")
STATUSES = ("open", "resolved", "excluded")


def load() -> dict:
    return json.loads((ROOT / DISCREPANCY_LOG).read_text())


def test_every_entry_has_every_field_and_a_known_status() -> None:
    log = load()
    assert log["schema"] == "verity/discrepancy-log/v0"
    ids = [entry["id"] for entry in log["entries"]]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for entry in log["entries"]:
        missing = [field for field in REQUIRED if field not in entry]
        assert not missing, f"{entry.get('id')}: missing {missing}"
        assert entry["status"] in STATUSES, entry["id"]
        assert entry["profile"] == FIRST.name, f"{entry['id']} is about another profile"
        assert entry["issue"].strip() and entry["owner"].strip()


def test_leaving_open_requires_a_resolution_and_evidence_that_exists() -> None:
    for entry in load()["entries"]:
        refs = entry["evidence_refs"]
        assert isinstance(refs, list)
        for ref in refs:
            # a repository path, a recorded run id, or a URL
            assert (ROOT / ref).exists() or ref.startswith(("r2026", "r2027", "http")), f"{entry['id']}: evidence {ref!r} not found"
        if entry["status"] == "open":
            assert entry["resolution"] is None, f"{entry['id']} is open but carries a resolution; set the status"
            assert entry.get("to_resolve", "").strip(), f"{entry['id']} is open without saying what resolves it"
        else:
            assert isinstance(entry["resolution"], str) and entry["resolution"].strip(), f"{entry['id']} left open without a resolution"
            assert refs, f"{entry['id']} left open without evidence"


def test_the_hawkeye_pin_is_the_one_the_target_names() -> None:
    pin = json.loads((ROOT / HAWKEYE_PIN).read_text())
    assert pin["commit"] == "30703fcb309c943a6df5eee0277cb81815deb8f4"
    assert set(pin["files"]) >= {"src/Ampere_simulator.cpp", "src/gfloat.cpp", "include/gfloat.h", "utils/utils.cpp", "utils/utils.h"}
    assert all(len(sha) == 40 for sha in pin["files"].values())
