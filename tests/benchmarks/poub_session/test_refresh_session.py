"""A refresh session end to end over the loopback (`session_harness`): the session verifier, its Python books and its
Lean verifier agreeing, against the reference prover over the wire, every append in its slot of the schedule, every
re-key naming its origin, every opening of a re-key's block its origin's position; and with refresh off, a kv session
as before."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.verity.protocols.resource_accounting.memory.poub import session_harness as H  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

N, W, L = 16, 6, 2


@pytest.mark.parametrize("plan", ["honest", "kv:2"])
def test_a_refresh_session_is_accepted_and_its_prover_records_the_rekeys(plan):
    (r,), (p,) = H.run(plan, blocks=N, windows=W, samples=64, lifetime=L, refresh=(True,), kv_every=4, kv_pool=8)
    assert r["accepted"] and r["refresh"] is True, r["failure"]
    rk = p["refresh"]
    assert rk["appends"] >= W - 1 and rk["encoded"] == rk["blocks"] > 0 and rk["encode_ns"] > 0
    if plan == "honest":
        # one registered re-key a window from window 1, each of its class's N / L blocks
        assert rk["appends"] == W - 1 and rk["blocks"] == (W - 1) * N // L and rk["skipped"] == 0


def test_with_refresh_off_a_kv_session_neither_rekeys_by_origin_nor_records_refresh():
    (r,), (p,) = H.run("kv:2", blocks=N, windows=W, samples=64, lifetime=L, kv_every=4, kv_pool=8)
    assert r["accepted"] and "refresh" not in r and "refresh" not in p
