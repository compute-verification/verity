"""Served Pearl-C's live coins (`protocol_options.pouw`'s install under the setting ``challenger``): the install has the
gate open the challenge phase it reserved on its salt epoch's root R = `salt.epoch_open_root` of the weights' root W it
committed and the epoch's number k over W (``pass_index``), its coins the salt coin alone (`window.SALT_FAMILIES`), takes
that coin after R (`pouw_h1_window.salt_session`, `pouw_pearl_c_host.epoch_of`) and closes the challenge phase at release, so
the challenge phase ends closed and complete; refused, before any patch or any coin: no challenge phase, a challenge phase of H1's layout, a
challenge phase bound to another commitment or another root, an in-process coin beside the challenger's and a mix of the two,
and a challenge phase reserved while the lane's open period is H1's, which `pouw_h1_window.reserve_salt` refuses before
reserving. A
reinstall is an epoch of its own: a fresh hm96 key, and over the same W the next k. The verifier holds the pass to that
challenge phase (`pouw_h1_challenger.session_reason`'s Pearl-C branch, `salt_session_reason`; Lean `Properties.Pouw.LiveSalt`),
recomputing R from W and ``pass_index``: it counts only reserved salt-only and ended closed and complete, and the pass's
coins and salt are read from the challenge phase, never from the pass.

CPU only: the auditor host is `challenger_host.Host` or `challenger_host.Gate` in this process, and the device run a
stand-in that starts its epoch through `pouw_pearl_c_device.epoch_of` (no GPU here).
"""
from __future__ import annotations

import itertools
import json
import random
import re
import shutil
import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.audit import Workload, salt_of, weights_root
from verity.core.protocols.pouw.coins import Challenger
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.core.protocols.sampled_proofs import challenger as LC
from verity.core.protocols.sampled_proofs import remote as R
from verity.ml.crypto import lms
from verity_vllm import protocols as PO
from verity_vllm.engine import hooks
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw.h1 import challenger as PHC
from verity_vllm.protocols.pouw.h1 import window as PHW
from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import manifest as PPM

from tests.protocols import challenger_host as CH

SCHEME = "pearl-c-sm120-v1-h3"
ENTRY = f"pouw:{SCHEME}"
AUDITOR = "vy-challenger-1:7811"
ROOTS = {0: b"\x05" * 64}
SID = CH.session_id(1)
#: the gate's reserve answer for the install's challenge phase, which the install opens itself on its salt epoch's root
RESERVED = {"session": SID, "seq": 0, "peer": "vy-nebius-1", "by": "vllm", "expires_utc": "2026-10-20T11:18:00Z"}
#: the auditor's challenger key, its ledger's and its challenge phases' (five leaves a challenge phase: reserve, open, record, close, end)
PRIVATE, KEY = CH.challenger_key(3, H=5)


def salt_only(seed: int) -> LC.LiveChallenger:
    """A Pearl-C challenge phase's challenger: the salt coin alone, sampled from `seed`."""
    return LC.LiveChallenger(Wd.LIVE_SCHEME, 0, Wd.SALT_FAMILIES, CH.entropy(seed))


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

    def __init__(self, s, entry, kernel, weights, roots, coins, opening, pass_index=None, **kw):
        self.epoch = PPD.epoch_of(Workload((Matmul(0, Shape(16, 1024, 128), 0),)), roots, coins, opening, pass_index)
        self.commits = {"scheme": s.name, "salt": self.epoch.salt.hex(), "weights_root": self.epoch.weights_root.hex(),
                        "coins": coins.to_json(), "salt_opening": opening.to_json(),
                        **({} if pass_index is None else {"pass_index": pass_index})}
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
    monkeypatch.setattr(PHW, "EPOCHS", {})
    RUNS.clear()


def _host(tmp_path, monkeypatch, seed=1, challenger=None):
    """The auditor host's challenger (by default Pearl-C's, its salt coin alone), its challenge phase `SID` reserved on its
    ledger and reached at `AUDITOR`."""
    host = CH.Host(tmp_path / "auditor", challenger or salt_only(seed), seed, SID, gated=True)
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


def test_a_served_install_takes_the_sessions_salt_coin_after_its_epochs_root(tmp_path, monkeypatch):
    """The gate opens the reserved challenge phase on salt epoch 0's root over the weights' root W the install committed,
    the run's epoch is the challenge phase's salt coin opened after that root (and no other coin of it), its salt over W, the
    commitment carries the challenge phase without its secret and the epoch's ``pass_index``, and release closes the challenge phase."""
    host = _host(tmp_path, monkeypatch)
    model, w, answers = _model(), weights_root(ROOTS), []
    root = PS.epoch_open_root(w, 0)

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
    assert c["salt"] == run.epoch.salt.hex() == salt_of(coins, opening.coin, w).hex()
    assert answers == [{"session": SID, "seq": answers[0]["seq"], "root": root.hex(), "commitment": coins.root.hex()}]
    assert c["session"] == answers[0] and c["pass_index"] == 0 and c["weights_root"] == w.hex()
    assert R.released_roots(host.directory / R.RELEASES, coins) == {Wd.SALT: (root,)}
    _untouched(model)


def test_each_install_is_an_epoch_of_its_own_under_a_fresh_hm96_key_and_over_one_weights_root_the_next_k(
        tmp_path, monkeypatch):
    """Each install's `commit_weights` draws its own hm96 key, so a reinstall's weights' root W is its own. A reinstall
    committing a W this process already had an epoch over (the stand-in's fixed roots: the same key over the same weights)
    is that W's next epoch: k = 1, its challenge phase opened on its own epoch root, never epoch 0's."""
    keys = []
    monkeypatch.setattr(PPD, "weight_roots", lambda kernel, weights, scheme, salt_key=None: keys.append(salt_key) or ROOTS)
    w, model, cs, hosts = weights_root(ROOTS), _model(), [], []
    for k in range(2):
        (tmp_path / str(k)).mkdir()
        hosts.append(_host(tmp_path / str(k), monkeypatch, seed=1 + k))
        with PO.install(model, _served(hosts[-1].open)) as comp:
            cs.append(comp.get("pouw").commitment())
        assert hosts[-1].ended() is None and RUNS[k].finished
    assert [c["pass_index"] for c in cs] == [0, 1] and PHW.EPOCHS == {w.hex(): 2}
    assert [h.window for h in hosts] == [{"root": PS.epoch_open_root(w, k).hex()} for k in range(2)]
    assert [c["session"]["root"] for c in cs] == [PS.epoch_open_root(w, k).hex() for k in range(2)]
    assert len(keys) == 2 and keys[0] != keys[1] and all(len(k) == 32 for k in keys) and pouw._SALT_KEYS[ENTRY] == keys[1]
    _untouched(model)


def test_pearl_c_takes_its_coins_from_one_live_session_or_is_refused_before_any_patch():
    """No challenge phase, an open answer in place of the reserve (a challenge phase opened before the weights' root was committed), no
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
    assert "which it opens on its salt epoch's root (salt.epoch_open_root, over the weights' root)" in refused(
        session=None)
    opened = {"session": SID, "seq": 3, "root": "ca" * 64, "commitment": "c0" * 64}
    assert "opens its session itself, on its salt epoch's root (salt.epoch_open_root, over the weights' root), " \
        "committed before its salt" in refused(session=opened)
    assert "opens its session itself, on its salt epoch's root" in refused(gate=None)
    assert "the window's coins come from one source" in refused(coins=Challenger(0, random.Random(0).randbytes))
    capture = PO.Context(stage="capture", enabled=(ENTRY,), settings={"pouw": {
        "device": "cuda", "kernel": "k", "challenger": AUDITOR, "session": RESERVED, "gate": lambda *a: None}})
    assert any("opens no session" in r for r in PO.refusals(capture))

    assert "coins" not in pouw.challenged({"pouw": {"challenger": AUDITOR}}, (ENTRY,))["pouw"]
    assert "coins" not in PO.own_auditor({"pouw": {"challenger": AUDITOR}}, SimpleNamespace(protocols=(ENTRY,)))["pouw"]
    assert "coins" in pouw.challenged({"pouw": {}}, (ENTRY,))["pouw"]

    root, workload = weights_root(ROOTS), Workload((Matmul(0, Shape(16, 1024, 128), 0),))
    live = salt_only(2)
    live_opening = LC.LiveRecord(live, Wd.SALT).record(PS.epoch_open_root(root, 3))
    own = Challenger(0, random.Random(0).randbytes)
    own_opening = own.open("salt", root)
    assert PPD.epoch_of(workload, ROOTS, live.commitment, live_opening, 3).salt == salt_of(live.commitment,
                                                                                         live_opening.coin, root)
    for coins, opening in ((live.commitment, own_opening), (own.commitment, live_opening)):
        with pytest.raises(DomainError, match="a live session's salt coin opens against that session's live coin "
                                              "commitment, and nothing else"):
            PPD.epoch_of(workload, ROOTS, coins, opening, 3)
    for k in (None, -1, True, 1 << 64, "3"):
        with pytest.raises(DomainError, match="a salt epoch's number is an integer in"):
            PPD.epoch_of(workload, ROOTS, live.commitment, live_opening, k)
    for k in (0, 2, 4):                                   # opened after epoch 3's root, so no other epoch's
        with pytest.raises(DomainError, match="the salt's "):
            PPD.epoch_of(workload, ROOTS, live.commitment, live_opening, k)
    on_w = salt_only(4)
    with pytest.raises(DomainError, match="the salt's "):    # a coin opened after the bare weights' root
        PPD.epoch_of(workload, ROOTS, on_w.commitment, LC.LiveRecord(on_w, Wd.SALT).record(root), 0)


def test_a_session_bound_to_another_commitment_is_closed_and_refused_before_its_first_coin(tmp_path, monkeypatch):
    """A gate answer naming coins other than the challenger's on that port (another process's challenge phase) closes the
    challenge phase before its salt coin and refuses the install, which patches nothing and starts no run."""
    host, model = _host(tmp_path, monkeypatch), _model()
    with pytest.raises(ValueError, match=f"commits to coins {host.challenger.commitment.root.hex()[:16]}..., not "
                                         f"{'dd' * 8}..., the commitment the gate's answer gives session {SID}"):
        PO.install(model, _served(lambda sid, r: {**host.open(sid, r), "commitment": "dd" * 64}))
    assert host.ended() is None and RUNS == []
    assert R.released_roots(host.directory / R.RELEASES, host.challenger.commitment) == {Wd.SALT: ()}
    _untouched(model)


def test_a_session_of_h1s_layout_is_closed_and_refused_before_its_first_coin(tmp_path, monkeypatch):
    """A challenge phase reserved with H1's layout (salt, draw and round coins) would release only its salt coin under a Pearl-C
    pass and end incomplete, which counts nothing: the install closes it before its salt coin and refuses, patching
    nothing and starting no run."""
    host, model = _host(tmp_path, monkeypatch, challenger=Wd.live_challenger(0, 1, CH.entropy(1))), _model()
    with pytest.raises(ValueError, match=f"{ENTRY}: the challenger at {AUDITOR} is not a live window's: the coin "
                                         "commitment's coins are not a Pearl-C salt session's: its salt coin alone"):
        PO.install(model, _served(host.open))
    assert host.ended() is None and RUNS == []
    assert R.released_roots(host.directory / R.RELEASES, host.challenger.commitment) == {
        Wd.SALT: (), Wd.DRAW: (), Wd.ROUND: ()}
    _untouched(model)


@pytest.mark.parametrize("answer", [
    {"session": SID, "seq": 0, "root": "ee" * 64, "commitment": "c0" * 64},
    {"session": CH.session_id(2), "seq": 0, "root": None, "commitment": "c0" * 64},
    None,
])
def test_a_gate_answer_not_the_reserved_session_on_the_epoch_root_is_refused_before_any_coin(monkeypatch, answer):
    """The gate's answer must be the reserved challenge phase opened on salt epoch 0's root: another root, another challenge phase
    or no open at all is refused before the challenger is reached."""
    root, model = PS.epoch_open_root(weights_root(ROOTS), 0).hex(), _model()
    given = None if answer is None else {**answer, "root": answer["root"] or root}
    monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(lambda *a, **k: pytest.fail("no challenger reached")))
    with pytest.raises(ValueError, match=f"{ENTRY}: the gate's answer is not session {SID} opened on salt epoch 0's "
                                         f"root {root[:16]}"):
        PO.install(model, _served(lambda sid, r: given))
    assert RUNS == []
    _untouched(model)


def _reserving(gate):
    """The challenger gate `gate` (`challenger_host.Gate`) as words, for ``reserve`` alone: it reserves challenge phase `SID` with the
    request's layout and answers as the gate's reserve does, naming the period of its lane the challenge phase joined."""
    def call(verb, *words):
        assert verb == "reserve"
        a = dict(zip(words[::2], words[1::2]))
        e = gate.reserve(SID, json.loads(a["--families"]), scheme=a["--scheme"], epoch=int(a["--epoch"]))
        return {**RESERVED, "seq": e["seq"], "period": gate._current.get("vllm")}
    return call


def _kinds(gate):
    return [json.loads(line)["kind"] for line in gate.lines]


H1_OPEN = {"cap": 4, "circuit": "c1" * 64, "draw_k": 27, "rows": 64}
H1_REFUSED = ("the lane's open period p-h1 is H1's (its open commits draw_k, rows): a salt-only session is a window "
              "H1's checks refuse (window.live_epoch_reason)")


def test_a_salt_only_session_is_refused_by_name_while_the_lanes_open_period_is_h1s():
    """While the lane's open period is H1's (its open commits H1's sample K and table rows), a Pearl-C salt-only challenge phase
    would be a window H1's checks refuse, so the period would earn nothing: `reserve_salt` refuses by name before
    reserving, reading the lane's open period from the gate's ledger. A challenge phase reserved in it anyway (here by hand) is
    refused by the install by name, before any patch and before the gate opens it, so no challenge phase is opened; so is a
    reserve that joined a period the install isn't given. A Pearl-C period (its cap and circuit alone), another lane's H1
    period, or none is reserved and installs."""
    gate = CH.Gate(PRIVATE, KEY)
    gate.period_open("p-h1", **H1_OPEN)
    with pytest.raises(ValueError, match=re.escape(f"refused before reserving a salt-only session: {H1_REFUSED}")):
        PHW.reserve_salt(_reserving(gate), lambda: gate.lines, "vllm", "vy-nebius-1", 0)
    assert _kinds(gate) == ["period-open"]

    period = PHW.open_period(gate.lines, "vllm")
    reserved = _reserving(gate)("reserve", "--families", json.dumps(Wd.SALT_FAMILIES), "--scheme", Wd.LIVE_SCHEME,
                                "--epoch", "0")
    assert reserved["period"] == period["period"] == "p-h1"

    def refused(**kw):
        model = _model()
        ctx = _served(lambda sid, r: pytest.fail("the gate opened a session"), **kw)
        assert len(PO.refusals(ctx)) == 1
        with pytest.raises(PO.Refused) as e:
            PO.install(model, ctx)
        _untouched(model)
        return str(e.value)
    assert f"{SCHEME}: {H1_REFUSED}" in refused(session=reserved, period=period)
    assert f"{SCHEME}: the gate's reserve answer joined period p-h1, but the lane's open period read from the auditor's " \
        "ledger (the pouw setting 'period', pouw_h1_window.open_period) is none" in refused(session=reserved)
    assert f"{SCHEME}: pouw setting 'period' is the lane's open period, its period-open body" in refused(
        session=reserved, period={"lane": "vllm"})
    assert _kinds(gate) == ["period-open", "reserve"] and RUNS == []

    late = CH.Gate(PRIVATE, KEY)                          # the period opened between the ledger's read and the reserve
    late.period_open("p-h1", **H1_OPEN)
    with pytest.raises(ValueError, match=re.escape(f"session {SID} is reserved and refused before its open: the gate's "
                                                   "reserve answer joined period p-h1, but the lane's open period")):
        PHW.reserve_salt(_reserving(late), lambda: [], "vllm", "vy-nebius-1", 0)
    assert _kinds(late) == ["period-open", "reserve"]

    for opens in ((), (("p-pc", "vllm", {"cap": 2, "circuit": "c1" * 64}),), (("k-h1", "kernels", H1_OPEN),)):
        g = CH.Gate(PRIVATE, KEY)
        for name, lane, fields in opens:
            g.period_open(name, lane, by=lane, **fields)
        answer, period = PHW.reserve_salt(_reserving(g), lambda: g.lines, "vllm", "vy-nebius-1", 0)
        joined = "p-pc" if opens and opens[0][1] == "vllm" else None
        assert answer["period"] == (period and period["period"]) == joined and _kinds(g)[-1] == "reserve"
        assert PO.refusals(_served(lambda *a: None, session=answer, period=period)) == ()


class _GateSession:
    """Challenge phase `SID` of the challenger gate `gate` (`challenger_host.Gate`, its ledger under `KEY`), reserved with `families`
    and, once the gate opens it (`open`, the install's pouw setting ``gate``), served by `challenger` on a socket pair
    with its record and close signed under `KEY` at the gate's next leaves, as serve.py's are. `end` has the gate end it
    over its ``out/`` once its client has closed it; `tree` is its tree as the gate's ``fetch`` gives it."""

    def __init__(self, tmp_path, monkeypatch, gate, families, challenger):
        self.gate, self.challenger, self.monkeypatch = gate, challenger, monkeypatch
        self.tree = tmp_path / "session"
        self.directory = self.tree / "out"
        self.reason = self.client = self._thread = None
        gate.reserve(SID, families)

    def open(self, sid, root):
        assert sid == SID and self.client is None
        reply = self.gate.open(sid, root)
        self.directory.mkdir(parents=True)
        (self.directory / R.HELLOS).write_text("")
        session = R.open_session(self.challenger, KEY, f"{CH.HOST}.{sid}",
                                 lambda m: lms.sign_at(PRIVATE, self.gate.take(), m).encode(), self.directory, sid)
        ours, theirs = socket.socketpair()
        self._thread = threading.Thread(target=self._serve, args=(ours, session))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)
        self.monkeypatch.setattr(R.RemoteChallenger, "connect", classmethod(lambda cls, at, secret=None: self.client))
        return {**reply, "commitment": self.challenger.commitment.root.hex()}

    def _serve(self, sock, session):
        with sock:
            self.reason = R.serve(sock, session, itertools.count().__next__)

    def end(self):
        self._thread.join()
        self.gate.end(SID, self.directory, self.reason is None)
        return self.reason


def _ended_over(out, families, root, end=True, **override):
    """A ledger of one challenge phase `SID` reserved with `families`, opened on `root` and, with `end`, ended over `out` with its
    end's fields replaced by `override`: what a gate would sign for a challenge phase served otherwise."""
    g = CH.Gate(PRIVATE, KEY)
    g.reserve(SID, families)
    g.open(SID, root)
    if end:
        g.end(SID, out, **override)
    return g.ledger()


def _ended(gate):
    """The body of the last end entry `gate`'s ledger signed."""
    return next(e["body"] for e in map(json.loads, reversed(gate.lines)) if e["kind"] == "end")


def test_a_served_pass_counts_from_its_salt_only_session_ended_closed_and_complete(tmp_path, monkeypatch):
    """The install's challenge phase, reserved with the salt coin alone and opened on salt epoch 0's root R over the
    weights' root W, ends closed and complete with that one coin released, and the verifier holds the pass to it
    (`session_reason`'s Pearl-C branch): the one challenge phase the ledger opened on R, which it recomputes from W and the
    pass's ``pass_index``, read through the gate's fetch. The manifest head names it by its `SESSION_KEYS` alone, with
    ``pass_index`` (`session_head`). The verifier's own weights' root, or the period's registered one, must be the
    pass's; a pass naming no epoch number, another epoch's, its challenge phase by another ledger seq, or
    its hello's secret, is refused."""
    gate = CH.Gate(PRIVATE, KEY)
    s = _GateSession(tmp_path, monkeypatch, gate, Wd.SALT_FAMILIES, salt_only(5))
    with PO.install(_model(), _served(s.open)) as comp:
        c = comp.get("pouw").commitment()
    root, r = weights_root(ROOTS), PS.epoch_open_root(weights_root(ROOTS), 0)
    assert s.end() is None and RUNS[0].finished
    end = _ended(gate)
    assert (end["closed"], end["complete"], end["released"]) == (True, True, {Wd.SALT: 1})
    assert c["weights_root"] == root.hex() and c["pass_index"] == 0 and c["session"]["root"] == r.hex()
    assert c["session"]["commitment"] == end["commitment"]
    ledger, fetch = gate.ledger(), {SID: s.tree}.get
    assert PHC.session_reason(c, ledger, fetch) is None
    assert PHC.session_reason(c, ledger, fetch, registered=root, commit=root.hex()) is None
    assert PHC.session_reason(c, ledger, fetch, commit="cc" * 64) == (
        f"the pass's weights' root {root.hex()[:16]}... is not {'cc' * 8}..., the one the verifier holds")
    assert PHC.session_reason(c, ledger, fetch, registered=bytes(64)) == (
        "the pass's weights' root is not the one the period registered")
    assert PHC.session_reason({**c, "coin_source": "in-process"}, ledger, fetch).startswith(
        "the window's coins were held in the serving process")
    seq = c["session"]["seq"]
    assert PHC.session_reason({**c, "session": {**c["session"], "seq": seq + 1}}, ledger, fetch) == (
        f"the window's session {SID} is ledger entry {seq + 1} in its reply, but entry {seq} opened it")
    assert PHC.session_reason({**c, "session": {**c["session"], "secret": "ab" * 32}}, ledger, fetch) == (
        "the pass's session names its hello's secret, which a pass never records: a refused pass")
    for k in (None, True, -1, 1 << 64, "0", 0.0):
        assert PHC.session_reason({**c, "pass_index": k}, ledger, fetch) == (
            f"the pass's pass_index names no salt epoch: a salt epoch's number is an integer in [0, 2^64), not {k!r}")
    assert PHC.session_reason({k: v for k, v in c.items() if k != "pass_index"}, ledger, fetch).startswith(
        "the pass's pass_index names no salt epoch")
    r1 = PS.epoch_open_root(root, 1).hex()
    assert PHC.session_reason({**c, "pass_index": 1}, ledger, fetch) == (
        f"the window's session is for Commit {r.hex()}, not the window's Commit {r1}")
    assert PHC.session_reason({**c, "pass_index": 1, "session": {**c["session"], "root": r1}}, ledger, fetch) == (
        f"the auditor's ledger has no session for Commit {r1}, so the window's coins are no session's of record")
    assert PPM.SESSION_KEYS == PHC.SESSION_FIELDS and PPM.session_head({}) == {}
    assert PPM.session_head({**c, "session": {**c["session"], "secret": "ab" * 32}}) == {
        "session": {k: c["session"][k] for k in PHC.SESSION_FIELDS}, "pass_index": 0}


def test_a_salt_session_counts_only_reserved_salt_only_and_ended_closed_and_complete(tmp_path, monkeypatch):
    """A challenge phase of H1's layout, as the served Pearl-C passes reserve it, releases only its salt coin and ends
    incomplete: refused, and refused too under an end that claimed it complete, for its reserve's layout. A salt-only
    challenge phase never ended, ended without its client closing it, short of its coin, or having released any other family is
    refused."""
    gate = CH.Gate(PRIVATE, KEY)
    s = _GateSession(tmp_path, monkeypatch, gate, Wd.live_families(1), Wd.live_challenger(0, 1, CH.entropy(6)))
    root = weights_root(ROOTS)
    r = PS.epoch_open_root(root, 0)
    answer = s.open(SID, r.hex())
    opening = LC.LiveRecord(s.client, Wd.SALT).record(r)
    s.client.close()
    assert s.end() is None
    coins = s.challenger.commitment
    h1 = {"scheme": SCHEME, "coin_source": "challenger", "weights_root": root.hex(), "pass_index": 0,
          "coins": coins.to_json(), "salt_opening": opening.to_json(), "salt": salt_of(coins, opening.coin, root).hex(),
          "session": PHC.carried(PHC.session_of(answer))}
    released = {Wd.DRAW: 0, Wd.ROUND: 0, Wd.SALT: 1}
    assert (_ended(gate)["complete"], _ended(gate)["released"]) == (False, released)
    fetch = {SID: s.tree}.get
    assert PHC.session_reason(h1, gate.ledger(), fetch) == (
        f"session {SID} ended without releasing every coin of its families (released {released}): a refused window")
    h1_layout = [list(f) for f in Wd.live_families(1)]
    assert PHC.session_reason(h1, _ended_over(s.directory, Wd.live_families(1), r.hex(), complete=True), fetch) == (
        f"session {SID} was reserved with {h1_layout}, not the salt coin alone [['pouw/salt', 1]]: a refused pass")

    one = tmp_path / "salt-only"
    one.mkdir()
    gate = CH.Gate(PRIVATE, KEY)
    s = _GateSession(one, monkeypatch, gate, Wd.SALT_FAMILIES, salt_only(7))
    with PO.install(_model(), _served(s.open)) as comp:
        c = comp.get("pouw").commitment()
    assert s.end() is None and PHC.session_reason(c, gate.ledger(), {SID: s.tree}.get) is None
    fetch = {SID: s.tree}.get

    def over(**kw):
        return PHC.session_reason(c, _ended_over(s.directory, Wd.SALT_FAMILIES, r.hex(), **kw), fetch)
    assert over(end=False) == (
        f"session {SID} never ended (the auditor's ledger has its open entry and no end): a refused window")
    assert over(closed=False) == f"session {SID} ended without its client closing it (serve.py exited 1): a refused window"
    assert over(complete=False, released={Wd.SALT: 0}) == (
        f"session {SID} ended without releasing every coin of its families (released {{'pouw/salt': 0}}): a refused window")
    assert over(released={Wd.SALT: 1, Wd.DRAW: 0}) == (
        f"session {SID} released {{'pouw/draw': 0, 'pouw/salt': 1}}, not its one salt coin: a refused pass")


def test_a_passs_salt_is_read_from_its_session_never_from_the_pass(tmp_path, monkeypatch):
    """The pass's coins, salt opening and salt agree with one another under another salt-only challenger's commitment
    (`epoch_of` takes them), but they are not the challenge phase's: the commitment is the one the challenge phase's challenger signed,
    read from its files through the ledger, never the pass's word. A pass naming another commitment for its challenge phase, a
    salt other than `audit.salt_of` of the challenge phase's coin, or another challenge phase's tree are refused."""
    gate = CH.Gate(PRIVATE, KEY)
    s = _GateSession(tmp_path, monkeypatch, gate, Wd.SALT_FAMILIES, salt_only(8))
    with PO.install(_model(), _served(s.open)) as comp:
        c = comp.get("pouw").commitment()
    assert s.end() is None
    ledger, fetch, root = gate.ledger(), {SID: s.tree}.get, weights_root(ROOTS)
    assert PHC.session_reason(c, ledger, fetch) is None
    forged = salt_only(9)
    opening = LC.LiveRecord(forged, Wd.SALT).record(PS.epoch_open_root(root, 0))
    workload = Workload((Matmul(0, Shape(16, 1024, 128), 0),))
    salt = PPD.epoch_of(workload, ROOTS, forged.commitment, opening, 0).salt
    own = {**c, "coins": forged.commitment.to_json(), "salt_opening": opening.to_json(), "salt": salt.hex()}
    signed = s.challenger.commitment.root.hex()
    assert PHC.session_reason(own, ledger, fetch) == (
        f"the pass's coins {forged.commitment.root.hex()[:16]}... are not the coin commitment session {SID}'s "
        f"challenger signed ({signed[:16]}...)")
    assert PHC.session_reason({**c, "session": {**c["session"], "commitment": forged.commitment.root.hex()}},
                              ledger, fetch) == (
        f"session {SID}'s end entry records coin commitment {signed[:16]}..., not "
        f"{forged.commitment.root.hex()[:16]}..., the window's (the gate's answer): a refused window")
    assert PHC.session_reason({**c, "salt": "00" * 32}, ledger, fetch) == (
        "the pass's salt is not audit.salt_of of its session's salt coin over its weights' root")
    other = tmp_path / "other"
    shutil.copytree(s.tree, other)
    (other / "out" / R.RECORD).write_text((other / "out" / R.RECORD).read_text().replace(signed, "dd" * 64))
    assert PHC.session_reason(c, ledger, {SID: other}.get).startswith(f"session {SID} is not trusted: ")
