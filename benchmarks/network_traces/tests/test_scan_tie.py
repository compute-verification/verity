"""scan_tie's statements of the egress tie of record (`scan_service.EgressTie`) at the certifier tests' class: one
registration a round, every proof block drawn, the three statements (``first``, ``ingress`` with the phase's T − 1
`IngressScanBucket` units, ``egress``) staged on the verifier's own registered reads, every output the row its read
opens and the ingress statement's S_t chain whole, but the tampered round's egress ``o1`` at the one bucket whose token
the re-signed window changed; and Lean's statement setup (`flock-verify statement`, the verifier's checks before any
proof) accepts every statement but the tampered egress, which it refuses at that port. Nothing here proves
(`scan_tie.py prove` does, on a node)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tests"))
from test_warden_zk import needs_lean, statement  # noqa: E402


def _load():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("network_traces_scan_tie", HERE / "scan_tie.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _load()
SC, SZ, WZ = st.SC, st.SZ, st.WZ


@pytest.fixture(scope="module")
def rounds(tmp_path_factory):
    """Both rounds staged as `scan_tie.prove --dry` stages them (lowering the two scan units takes about 30 s)."""
    out = tmp_path_factory.mktemp("scan-tie")
    t = st.tie_of(out, SZ.TEST)
    M = WZ._statement()
    sgs = st.stagers(M, t.tie, out / "stagers")
    return t, {name: st.stage_round(M, t.tie, t.at, t.tr, sgs, name, out / name, 2.0 ** -20) for name in st.ROUNDS}


@pytest.mark.slow
def test_each_round_is_one_registration_every_block_drawn_each_statement_as_its_verdict_needs(rounds):
    """The Spec's 2T proof blocks in three statements, each drawn; honest's every output the row its read opens and
    tampered's off only at the egress o1 of the bucket `_tamper` changed; both rounds' ingress chains whole."""
    t, got = rounds
    tie, T = t.tie, t.tie.T
    assert [(m.name, m.base, len(m.units)) for m in tie.members] == [("first", 0, 1), ("ingress", 1, T - 1),
                                                                       ("egress", T, T)]
    assert tie.N == 2 * T
    for name, r in got.items():
        assert r.draw.units == list(range(tie.N)), name
        assert all(m["chain"] is None for m in r.staged.values()), name
    assert all(not m["off"] for m in got["honest"].staged.values())
    tampered = got["tampered"].staged
    assert tampered["egress"]["off"] == {"o1": [T + t.at.tampered.bucket]}
    assert not tampered["first"]["off"] and not tampered["ingress"]["off"]
    assert got["honest"].heads.ingress == got["tampered"].heads.ingress
    assert got["honest"].heads.egress.root != got["tampered"].heads.egress.root


@pytest.mark.slow
@needs_lean
def test_lean_accepts_every_statement_but_the_tampered_egress_refused_at_its_o1(rounds, tmp_path):
    """Lean's statement setup, `Registered.checkComplete` with it, on each staged statement under the verifier's own
    registered file: what the host reads above only mirrors Lean, and a staging that drifts from the verifier's reads of
    the network certifier's rows, the trace's trees or the prover's ``scan/view`` and ``scan/next``, or a tamper Lean
    doesn't refuse where the Spec says, fails here and in no cheaper test."""
    t, got = rounds
    for name, r in got.items():
        for k, m in r.staged.items():
            rc, line = statement(m, tmp_path)
            if name == "tampered" and k == "egress":
                assert rc == 1 and line.startswith(f"setup: registered reads: port o1: table row "
                                                   f"{t.at.tampered.bucket} does not open"), line
                continue
            assert rc == 0, (name, k, line)
            reads = json.loads(m["registered_file"].read_text())["ports"]
            assert {p: f["root"] for p, f in json.loads(line)["registered"].items()} == \
                {p: e["root"] for p, e in reads.items()}, (name, k)
