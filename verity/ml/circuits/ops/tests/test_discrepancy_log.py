"""``verity/ml/circuits/silicon/discrepancy_log.json`` is well formed, and nothing leaves ``open`` without a resolution and evidence.

The log records where the paper, the pinned simulator, Verity's reference and the hardware disagree about the
frozen target.  The rule it enforces: an entry becomes ``resolved`` or ``excluded`` only by writing down what was
decided and pointing at evidence that exists -- never merely by changing the reference implementation.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.targets.target import DISCREPANCY_LOG, FIRST, HAWKEYE_PIN, TARGETS

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists())
REQUIRED = ("id", "profile", "class", "issue", "hawkeye_paper", "hawkeye_simulator", "verity_reference",
            "hardware_evidence", "status", "resolution", "evidence_refs", "owner")
STATUSES = ("open", "resolved", "excluded")
CLASSES = ("numerics", "binding")


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
        assert entry["class"] in CLASSES and set(log["classes"]) == set(CLASSES), entry["id"]
        # the profile names a target (frozen or draft) or, for instruction classes without a target, an instruction id
        assert entry["profile"] in TARGETS or entry["profile"] in INSTRUCTIONS, f"{entry['id']} is about an unknown profile"
        assert entry["issue"].strip() and entry["owner"].strip()


def test_only_the_kernel_binding_question_is_binding_class() -> None:
    """D6 (which kernels / shapes realise the accumulation chain; owner integration) is a deployment caveat, not a numerics gate;
    every other entry is about words (model vs hardware / paper / simulator)."""
    log = load()
    assert {e["id"] for e in log["entries"] if e["class"] == "binding"} == {"D6"}
    assert all(e["class"] == "numerics" for e in log["entries"] if e["id"] != "D6")


def test_the_frozen_target_keeps_its_entries() -> None:
    """D1..D9 are the frozen target's; the silicon entries (D10..) are about the draft targets or instruction ids."""
    log = load()
    first = [e["id"] for e in log["entries"] if e["profile"] == FIRST.name]
    assert first[:9] == [f"D{i}" for i in range(1, 10)]
    for entry in log["entries"]:
        if entry["profile"] != FIRST.name:
            assert int(entry["id"][1:]) >= 10, f"{entry['id']} is not the frozen target's but numbered among its entries"


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


def test_every_evidence_path_is_an_input_of_this_suite() -> None:
    """The test above checks that each evidence path exists, which the suite's guard can't see: a path outside this suite's
    inputs could be deleted without moving its key, and the check would pass from the cache. List it under
    [tool.verity.tests] inputs in verity/ml/circuits/ops/pyproject.toml."""
    pkg = Path(__file__).resolve().parents[1]
    declared = tomllib.loads((pkg / "pyproject.toml").read_text())["tool"]["verity"]["tests"]["inputs"]
    covered = [pkg.relative_to(ROOT).as_posix(), *declared]
    uncovered = [f"{e['id']}: {ref}" for e in load()["entries"] for ref in e["evidence_refs"]
                 if not ref.startswith(("r2026", "r2027", "http"))
                 and not any(ref == p or ref.startswith(p.rstrip("/") + "/") for p in covered)]
    assert not uncovered, f"evidence outside the catalog's [tool.verity.tests] inputs: {uncovered}"


def test_the_hawkeye_pin_is_the_one_the_target_names() -> None:
    pin = json.loads((ROOT / HAWKEYE_PIN).read_text())
    assert pin["commit"] == "30703fcb309c943a6df5eee0277cb81815deb8f4"
    assert set(pin["files"]) >= {"src/Ampere_simulator.cpp", "src/gfloat.cpp", "include/gfloat.h", "utils/utils.cpp", "utils/utils.h"}
    assert all(len(sha) == 40 for sha in pin["files"].values())
