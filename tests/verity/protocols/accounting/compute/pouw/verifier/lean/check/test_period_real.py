"""`pouw-period serve` (`PouwVerifier.Period`, ``verity/pouw/period/v1``) alone on one real period: one window whose
circuit is the registry's H1 window at K = 64 over J10-R's ``PouwH1Tile_v5`` (`pouw_credit_cases.registry_window`, a
template `h1_tiles.json` pins), the case `pouw_period_cases.serve` makes. The Lean accepts it with every field of the
outcome but its reason as the composition's product tests compare them, and refuses it by name (``circuit``) once that
window's descriptor is the stand-in window of the same layout, a template the pin doesn't hold.

It takes a minute or more on a node (the real window's template, about 60 MB, is made once). What only it catches: every
other period the Lean sees, the difftests' and the composition's product cases alike, is a stand-in window, which the
pin refuses, so a period the Lean accepts with a credit, its form check, the pin, the credit and the sessions on the
registry's own window, is here alone. It reads no Python verifier's answer."""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import random
from pathlib import Path

import pytest
import verity_difftest as D

import pouw_period_cases as cases

ROOT = Path(__file__).resolve().parents[9]
PACKAGE = ROOT / "verity/protocols/accounting/compute/pouw/verifier/lean/check"
PROTOCOL = "verity/pouw/period/v1"
#: `PouwVerifier.Period.commitmentTag`: a period's digest is SHA-512(tag ‖ 0 ‖ the auditor's ledger's bytes)
COMMITMENT_TAG = b"verity/pouw/period/ledger/v1"


@functools.cache
def _period() -> dict:
    return cases.credit_cases.period_instance(random.Random("real/0"), windows=1, real=True, **cases.COVERED)


@pytest.fixture(scope="module")
def server():
    with D.Server([str(D.build(PACKAGE, "pouw-period")), "serve"]) as s:
        yield s


def test_the_lean_accepts_a_real_pinned_period_and_refuses_it_with_another_descriptor(server):
    d = _period()
    case = cases.serve(d)
    got = server.exchange(case)
    assert (got["protocol"], got["verdict"], got["reason"]) == (PROTOCOL, "accept", None), got.get("reason")
    assert got["commitment"] == hashlib.sha512(COMMITMENT_TAG + b"\0" + bytes.fromhex(case["commitment"])).hexdigest()
    assert got["profile"] == {"period": d["name"], "windows": 1}
    pub = got["public"]
    assert pub["key"] == case["params"]["key"]
    assert len(pub["windows"]) == len(pub["work"]) == len(pub["eps"]) == 1
    assert pub["windows"][0] and pub["work"][0] > 0 and pub["eps"][0] is not None

    changed = copy.deepcopy(d)
    stand_in = cases.credit_cases.descriptor(random.Random("real/stand-in"), json.loads(json.dumps(
        cases.credit_cases.REGISTRY_LAYOUT)), "h1/", j10r=True, tile=5)
    assert stand_in != d["windows"][0]["descriptor"]
    changed["windows"][0]["descriptor"] = stand_in
    got = server.exchange(cases.serve(changed))
    assert (got["protocol"], got["verdict"], got["public"]) == (PROTOCOL, "reject", None)
    assert got["reason"].startswith(f"circuit: window 0 of period {d['name']} "), got["reason"]
