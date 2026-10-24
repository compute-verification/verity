"""Served Pearl-C's live coins (`protocol_options.pouw`'s install under the setting ``challenger``): the install has the
gate open the session it reserved on the weights' root it committed, takes that session's salt coin after the root
(`pouw_h1_window.salt_session`, `pouw_pearl_c_host.epoch_of`) and closes the session at release; refused, before any
patch or any coin: no session, a session bound to another commitment or another root, an in-process coin beside the
challenger's, and a mix of the two.

CPU only: the auditor host is `challenger_host.Host` in this process, and the device run a stand-in that starts its epoch
through `pouw_pearl_c_device.epoch_of` (no GPU here).
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import Workload, salt_of, weights_root
from verity.core.protocols.pouw.coins import Challenger
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity_vllm import protocol_options as PO
from verity_vllm.engine import hooks
from verity_vllm.protocol_options import pouw
from verity_vllm.protocol_options import pouw_pearl_c_device as PPD

from tests.protocol_options import challenger_host as CH

SCHEME = "pearl-c-sm120-v1-h3"
ENTRY = f"pouw:{SCHEME}"
AUDITOR = "vy-challenger-1:7811"
ROOTS = {0: b"\x05" * 64}
SID = CH.session_id(1)
#: the gate's reserve answer for the install's session, which the install opens itself on the weights' root
RESERVED = {"session": SID, "seq": 0, "peer": "vy-nebius-1", "by": "vllm", "expires_utc": "2026-10-20T11:18:00Z"}


class Apply:
    def apply(self, layer, x, bias=None):
        return np.asarray(x, np.float32) @ np.asarray(layer.weight, np.float32).T


class Linear:
    def __init__(self, weight):
        self.weight = weight
        self.quant_method = Apply()


class Model:
    def __init__(self, *weights):
        self.layers = [Linear(w) for w in weights]

    def modules(self):
        return iter(self.layers)


RUNS: list = []


class Run:
    """Pearl-C's device run with no device: its epoch from the coins it is handed (`PPD.epoch_of`), and what it records of
    them."""

    def __init__(self, s, entry, kernel, weights, roots, coins, opening, **kw):
        self.epoch = PPD.epoch_of(Workload((Matmul(0, Shape(16, 1024, 128), 0),)), roots, coins, opening)
        self.commits = {"scheme": s.name, "salt": self.epoch.salt.hex(), "coins": coins.to_json(),
                        "salt_opening": opening.to_json()}
        self.details, self.finished = {}, False
        RUNS.append(self)

    def gemm(self, w, x, weight, bias):
        return x

    def end_pass(self):
        return {}

    def finish(self):
        self.finished = True


@pytest.fixture(autouse=True)
def _device_run_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
    monkeypatch.setattr(PPD.Kernel, "load", staticmethod(lambda *a: None))
    monkeypatch.setattr(PPD, "weight_roots", lambda kernel, weights, scheme, salt_key=None: ROOTS)
    monkeypatch.setattr(PPD, "PearlCDeviceRun", Run)
    monkeypatch.setattr(pouw, "_SALT_KEYS", {})
    RUNS.clear()


def _host(tmp_path, monkeypatch, seed=1):
    """The auditor host's challenger over a live epoch, its session `SID` reserved on its ledger and reached at `AUDITOR`."""
    host = CH.Host(tmp_path / "auditor", Wd.live_challenger(0, 1, CH.entropy(seed)), seed, SID, gated=True)
    monkeypatch.setattr(R.RemoteChallenger, "connect",
                        classmethod(lambda cls, at, secret=None: host.client if at == AUDITOR else None))
    return host


def _model():
    return Model(np.random.default_rng(0).standard_normal((128, 1024), np.float32))


def _served(gate, **kw):
    return PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": {
        "device": "cuda", "kernel": "k", "challenger": AUDITOR, "session": RESERVED, "gate": gate, **kw}})


def _untouched(model):
    assert hooks.live(ENTRY) == () and all(m.quant_method.apply.__func__ is Apply.apply for m in model.layers)


def test_a_served_install_takes_the_sessions_salt_coin_after_the_weights_root(tmp_path, monkeypatch):
    """The gate opens the reserved session on the weights' root the install committed, the run's epoch is the session's
    salt coin opened after that root (and no other coin of it), the commitment carries the session without its secret,
    and release closes the session."""
    host = _host(tmp_path, monkeypatch)
    model, root, answers = _model(), weights_root(ROOTS), []

    def gate(sid, r):
        answers.append(host.open(sid, r))
        return {**answers[-1], "secret": "5e" * 32}
    assert PO.refusals(_served(gate)) == ()
    with PO.install(model, _served(gate)) as comp:
        assert hooks.live(ENTRY) != ()
        c = comp.get("pouw").commitment()
    (run,), coins = RUNS, host.challenger.commitment
    assert host.ended() is None and run.finished and run.details["challenger"] == {"closed": True}
    assert host.window == {"root": root.hex()} and c["coin_source"] == "challenger" and c["coins"] == coins.to_json()
    opening = LC.LiveOpening.from_json(c["salt_opening"])
    assert LC.live_reason(coins, opening, Wd.SALT, 0, root) is None
    assert c["salt"] == run.epoch.salt.hex() == salt_of(coins, opening.coin, root).hex()
    assert answers == [{"session": SID, "seq": answers[0]["seq"], "root": root.hex(), "commitment": coins.root.hex()}]
    assert c["session"] == answers[0]
    assert R.released_roots(host.directory / R.RELEASES, coins) == {Wd.SALT: (root,), Wd.DRAW: (), Wd.ROUND: ()}
    _untouched(model)


def test_pearl_c_takes_its_coins_from_one_live_session_or_is_refused_before_any_patch():
    """No session, an open answer in place of the reserve (a session opened before the weights' root was committed), no
    gate, an in-process challenger beside the auditor host's, or a capture: each refused before the model is touched. A
    config naming the auditor host's challenger gets no in-process coins, and the epoch takes a live opening only of a
    live commitment, and an in-process opening only of an in-process one."""
    def refused(gate=lambda sid, r: pytest.fail("refused before the gate"), **kw):
        model = _model()
        ctx = _served(gate, **kw)
        with pytest.raises(PO.Refused) as e:
            PO.install(model, ctx)
        _untouched(model)
        return str(e.value)

    assert "'challenger' needs 'session', the challenger gate's reserve answer" in refused(session=None)
    assert "which it opens on the weights' root" in refused(session=None)
    opened = {"session": SID, "seq": 3, "root": "ca" * 64, "commitment": "c0" * 64}
    assert "opens its session itself, on the weights' root, committed before its salt" in refused(session=opened)
    assert "opens its session itself, on the weights' root" in refused(gate=None)
    assert "the window's coins come from one source" in refused(coins=Challenger(0, random.Random(0).randbytes))
    capture = PO.Context(stage="capture", enabled=(ENTRY,), settings={"pouw": {
        "device": "cuda", "kernel": "k", "challenger": AUDITOR, "session": RESERVED, "gate": lambda *a: None}})
    assert any("opens no session" in r for r in PO.refusals(capture))

    assert "coins" not in pouw.challenged({"pouw": {"challenger": AUDITOR}}, (ENTRY,))["pouw"]
    assert "coins" not in PO.own_auditor({"pouw": {"challenger": AUDITOR}}, SimpleNamespace(protocols=(ENTRY,)))["pouw"]
    assert "coins" in pouw.challenged({"pouw": {}}, (ENTRY,))["pouw"]

    root, workload = weights_root(ROOTS), Workload((Matmul(0, Shape(16, 1024, 128), 0),))
    live = Wd.live_challenger(0, 1, CH.entropy(2))
    live_opening = LC.LiveRecord(live, Wd.SALT).record(root)
    own = Challenger(0, random.Random(0).randbytes)
    own_opening = own.open("salt", root)
    assert PPD.epoch_of(workload, ROOTS, live.commitment, live_opening).salt == salt_of(live.commitment,
                                                                                      live_opening.coin, root)
    for coins, opening in ((live.commitment, own_opening), (own.commitment, live_opening)):
        with pytest.raises(DomainError, match="a live session's salt coin opens against that session's live coin "
                                              "commitment, and nothing else"):
            PPD.epoch_of(workload, ROOTS, coins, opening)


def test_a_session_bound_to_another_commitment_is_closed_and_refused_before_its_first_coin(tmp_path, monkeypatch):
    """A gate answer naming coins other than the challenger's on that port (another process's session) closes the
    session before its salt coin and refuses the install, which patches nothing and starts no run."""
    host, model = _host(tmp_path, monkeypatch), _model()
    with pytest.raises(ValueError, match=f"commits to coins {host.challenger.commitment.root.hex()[:16]}..., not "
                                         f"{'dd' * 8}..., the commitment the gate's answer gives session {SID}"):
        PO.install(model, _served(lambda sid, r: {**host.open(sid, r), "commitment": "dd" * 64}))
    assert host.ended() is None and RUNS == []
    assert R.released_roots(host.directory / R.RELEASES, host.challenger.commitment) == {
        Wd.SALT: (), Wd.DRAW: (), Wd.ROUND: ()}
    _untouched(model)


@pytest.mark.parametrize("answer", [
    {"session": SID, "seq": 0, "root": "ee" * 64, "commitment": "c0" * 64},
    {"session": CH.session_id(2), "seq": 0, "root": None, "commitment": "c0" * 64},
    None,
])
def test_a_gate_answer_not_the_reserved_session_on_the_weights_root_is_refused_before_any_coin(monkeypatch, answer):
    """The gate's answer must be the reserved session opened on the weights' root: another root, another session or no
    open at all is refused before the challenger is reached."""
    root, model = weights_root(ROOTS).hex(), _model()
    given = None if answer is None else {**answer, "root": answer["root"] or root}
    monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(lambda *a, **k: pytest.fail("no challenger reached")))
    with pytest.raises(ValueError, match=f"{ENTRY}: the gate's answer is not session {SID} opened on the weights' root "
                                         f"{root[:16]}"):
        PO.install(model, _served(lambda sid, r: given))
    assert RUNS == []
    _untouched(model)
