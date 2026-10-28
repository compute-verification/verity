"""N2's sampled period draws on the auditor's challenger's coins (`n2_zk.period_session`, ``--challenger local``): each
member on a session of its own, on the root of its period's certificates, the step and the member, under
live_challenger's stub of the Lean verifier's coins (the fake challenger, `test_round_coins.fake_challenger`), so
nothing builds `sampled-proofs-verifier` and nothing is proved. Each member's draw is `LinkWindows.period_draw`'s on its
session's coins: committed to its period's commitment, then `subset:K` read, then its turn ended (`drawn`); the
period's draw is then held to the session's evidence on the root the verifier recomputes from the certificates it
received (`n2_zk.period_held`). A period that isn't sampled opens no session, and a session the challenger refuses
stops the driver."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.sampled_proofs import service as S

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent)]
import round_coins as RC  # noqa: E402
from test_round_coins import fake_challenger  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("network_traces_n2_zk", HERE / "n2_zk.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


n2 = _load()
N, K = 26, 8


@dataclass(frozen=True)
class Certificate:
    window: int
    signature: bytes


def _period(name: str, cert: str = "cert") -> SimpleNamespace:
    """A period of N windows, each with its salted root and one certificate."""
    return SimpleNamespace(advice=None, windows=[
        SimpleNamespace(root=hashlib.sha512(f"{name}/{w}".encode()).digest(),
                        certificates=[Certificate(w, hashlib.sha512(f"{cert}/{w}".encode()).digest())])
        for w in range(N)])


def _draw(s: RC.Session, name: str, rd: SimpleNamespace) -> dict:
    commitment = hashlib.sha512(("period/" + name).encode()).hexdigest()
    read = bytearray()
    s.rank.commit(bytes.fromhex(commitment))
    rule = S.Law(f"subset:{K}")
    draw = S.select_draw(rule, S.selection_law(rule, N), N, lambda m: read.extend(b := s.rank(m)) or b)
    return {"period": {"roots": [x.root.hex() for x in rd.windows], "k": K, "commitment": commitment,
                       "units": list(draw["units"]), "read": read.hex()},
            "segment": s.rank.drawn()}


def _member(a, name: str, rd: SimpleNamespace, out: Path) -> tuple[RC.Session, dict]:
    s = n2.period_session(a, name, rd, out / name)
    try:
        res = _draw(s, name, rd)
    finally:
        s.close()
    (out / name / "verifier").mkdir(parents=True)
    return s, res


def test_each_members_draw_holds_on_its_own_session_over_its_periods_certificates(tmp_path, monkeypatch):
    """Two members (26 windows, `subset:8`) each open a session of their own on the root of their period's certificates,
    read their coins after their period's commitment, and hold against their session's evidence on the root the
    verifier recomputes from the certificates it received, every `a5_coins.negatives` refused and the coins read the
    ones released. The same coins read as other bytes, held to another commitment, or on a period whose certificates
    changed are refused."""
    a = fake_challenger(tmp_path, monkeypatch, draw=K)
    out = tmp_path / "out"
    rds = {"honest": _period("honest"), "tampered": _period("tampered")}
    got = {n: _member(a, n, rd, out) for n, rd in rds.items()}
    assert (tmp_path / "coins.log").read_text(), "the sessions' coins came from the stub, not a Lean build"
    roots = {n: s.root for n, (s, _) in got.items()}
    assert roots["honest"] != roots["tampered"]
    for n, (s, res) in got.items():
        assert s.root == RC.root(RC.digest(n2.period_certificates(rds[n])), "n2", n)
        held = n2.period_held(s, n, rds[n], res, out / n / "verifier")
        assert held["ok"] and held["root_ok"] and held["coins_reason"] is None and held["session"], held
        assert held["read_is_released"] and held["read_bytes"] == len(res["period"]["read"]) // 2 > 0
        assert held["negatives_refused"] and all(held["negatives_refused"].values()), held["negatives_refused"]
        assert len(res["period"]["units"]) == K and (out / n / "verifier" / "coins.json").is_file()
        assert json.loads((out / n / "verifier" / "period-certificates.json").read_text()) == \
            n2.period_certificates(rds[n])
        assert (out / n / "challenger" / "coins.json").is_file()

    s, res = got["honest"]
    read = res["period"]["read"]
    flipped = {**res, "period": {**res["period"], "read": f"{int(read[:2], 16) ^ 1:02x}" + read[2:]}}
    refused = n2.period_held(s, "honest", rds["honest"], flipped, out / "honest" / "verifier")
    assert not refused["ok"] and not refused["read_is_released"], refused
    other = {**res, "period": {**res["period"], "commitment": "ee" * 64}}
    refused = n2.period_held(s, "honest", rds["honest"], other, out / "honest" / "verifier")
    assert not refused["ok"] and refused["coins_reason"], refused
    moved = n2.period_held(s, "honest", _period("honest", cert="other"), res, out / "honest" / "verifier")
    assert not moved["ok"] and not moved["root_ok"], moved


def test_a_member_that_never_drew_is_refused_and_an_unsampled_period_opens_no_session(tmp_path, monkeypatch):
    """A member whose draw raised before its commitment (its turn never taken) ends its session with no segment, and
    is refused by name; at ``--draw 0`` (Selection All, every window) no session opens and nothing is held."""
    a = fake_challenger(tmp_path, monkeypatch, draw=K)
    rd = _period("honest")
    s = n2.period_session(a, "honest", rd, tmp_path / "honest")
    s.close()
    (tmp_path / "verifier").mkdir()
    held = n2.period_held(s, "honest", rd, {"period": {"commitment": "aa" * 64, "read": ""}}, tmp_path / "verifier")
    assert not held["ok"] and held["coins_reason"] == "the draw released no segment", held
    a.draw = 0
    assert n2.period_session(a, "honest", rd, tmp_path / "all") is None
    assert n2.period_held(None, "honest", rd, {}, tmp_path / "verifier") is None
    assert not (tmp_path / "all").exists()


def test_a_session_the_challenger_refuses_stops_the_driver(tmp_path, monkeypatch):
    a = fake_challenger(tmp_path, monkeypatch, draw=K)
    monkeypatch.setattr(RC.a5_coins(), "reserve_reason", lambda layout, needs: "the challenger refuses the reserve")
    with pytest.raises(SystemExit, match="honest: the challenger's session for the period draw: .*the challenger "
                                         "refuses the reserve"):
        n2.period_session(a, "honest", _period("honest"), tmp_path / "honest")
    assert not (tmp_path / "coins.log").exists()
