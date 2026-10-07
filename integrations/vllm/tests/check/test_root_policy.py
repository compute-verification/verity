"""root_policy.py (protected): every root of the M1 capture falls in a frozen class with a passing validator once the
checkpoint identity was verified at build; a config/runtime disagreement about lm_head tying fails; a Root.role outside the
frozen table is a new root class."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from verity_vllm.check import root_policy
from verity_vllm.check.fold_compare import DEFAULT_COS_SIN as COS_SIN, load_profile
from verity_vllm.observe.events import Root, RunHeader, ViewDesc
from verity_vllm.observe.log import read_log

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "data" / "logs" / "m1.jsonl.gz"
SNAPS = REPO / "data" / "logs" / "m1"

pytestmark = pytest.mark.skipif(not (LOG.exists() and SNAPS.exists() and COS_SIN.exists()), reason="M1 capture / fixture not present")


@pytest.fixture(scope="module")
def events():
    return list(read_log(LOG))


def _with_verified(events):
    h = events[0]
    assert isinstance(h, RunHeader)
    return [dataclasses.replace(h, notes={**h.notes, "weights_verified": True})] + events[1:]


def test_m1_roots_all_validated_when_identity_verified(events):
    prof = load_profile("vllm_d9105ea80_sm89_eager")
    rep = root_policy.evaluate(_with_verified(events), SNAPS, prof, cos_sin=COS_SIN, weight_bytes=False)
    assert rep["status"] == "pass", rep["first_failures"]
    assert rep["new_root_classes"] == [] and rep["problems"] == []
    assert set(rep["classes"]) == {"weight", "analytic-table", "ignored", "request-input", "config"}
    assert rep["classes"]["request-input"]["validated"] == 32 and rep["classes"]["ignored"]["validated"] == 120
    assert all(a["status"] == "pass" for a in rep["alias_claims"])
    assert set(rep["ignored_root_consumers"]) <= set(root_policy.IGNORED_ROOT_CONSUMERS)


def test_m1_unverified_identity_fails_weights(events):
    prof = load_profile("vllm_d9105ea80_sm89_eager")
    rep = root_policy.evaluate(events, SNAPS, prof, cos_sin=COS_SIN, weight_bytes=False)
    assert rep["status"] == "fail" and rep["classes"]["weight"]["failed"] == rep["classes"]["weight"]["count"]


def test_untied_config_against_tied_runtime_fails(events):
    from verity_vllm.engine.profiles.vllm_d9105ea80_sm89_eager import build_profile, with_geometry
    from verity.ml.circuits.ops import b1

    prof = build_profile(with_geometry({**b1.B0_CONFIG, "tie_word_embeddings": False}), profile_id="test-untied", case="B0")
    rep = root_policy.evaluate(_with_verified(events), SNAPS, prof, cos_sin=COS_SIN, weight_bytes=False)
    assert rep["status"] == "fail"
    assert any("lm_head tie" in p for p in rep["problems"])


def test_new_root_role_is_a_new_root_class(events):
    prof = load_profile("vllm_d9105ea80_sm89_eager")
    extra = Root(ViewDesc(alloc=999999, offset=0, shape=(4,), strides=(1,), dtype="bfloat16", device="cuda:0"), "const", "some.constant")
    rep = root_policy.evaluate(_with_verified(events) + [extra], SNAPS, prof, cos_sin=COS_SIN, weight_bytes=False)
    assert rep["status"] == "fail" and "role:const" in rep["new_root_classes"]
