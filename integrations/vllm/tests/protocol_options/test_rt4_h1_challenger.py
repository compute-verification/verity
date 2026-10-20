"""RT4-INT's witnesses on H1's coins from the auditor host's challenger (#2940 @ 79c0a054b1, VLLM-55, stacked on #2916).

Each test asserts what the code does today; one that passes where a finding says the code admits something is that
finding's witness, and flips to the refusal when the code is fixed.

* RT4-INT-17: two sessions under the auditor's one key both pass `trusted_rounds_reason` for one window, and neither
  record names the window, so a prover that drives more than one session keeps the coins it likes.
* RT4-INT-19: a draw coin opened after a root of the prover's, before the window's first round, passes the trusted
  check, which reads only the salt and round families.
* RT4-INT-20: round s's root, the one round s's coin follows, is the same whatever A the window serves.
* RT4-INT-23: the release log, the only evidence that a round's coin followed its root, is an unsigned file; roots
  chosen after every coin are refused against the honest log and pass once the log is rewritten to name them.
* The round rule of record (`rounds_reason`) passes in-process coins; only `trusted_rounds_reason`, which no verifier of
  record calls, refuses them.
"""
from __future__ import annotations

import hashlib
import json
import socket
import threading

import pytest

torch = pytest.importorskip("torch")

from verity.core.protocols.pouw import window as Wd  # noqa: E402
from verity.core.protocols.sampled_proofs import remote as R  # noqa: E402
from verity.core.protocols.sampled_proofs.challenger import LiveCommitment, record_message  # noqa: E402
from verity.ml.crypto import lms  # noqa: E402
from verity_vllm.protocol_options import pouw  # noqa: E402
from verity_vllm.protocol_options import pouw_h1 as PH  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_challenger as PHC  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_requests as RQ  # noqa: E402

from tests.protocol_options import challenger_host as CH  # noqa: E402
from tests.protocol_options import test_pouw_h1 as T  # noqa: E402

PRIVATE, KEY = CH.challenger_key(1)


@pytest.fixture(autouse=True)
def _linear_is_the_stand_in(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (T.Linear,))


class _Session:
    """One session of the auditor's challenger under its one key `KEY`, its record signed at LMS leaf `leaf`, with its own
    coins (`seed`), reached at `T.AUDITOR` until the next session is made."""

    def __init__(self, tmp_path, monkeypatch, name: str, leaf: int, seed: int) -> None:
        self.directory, self.challenger = tmp_path / name, Wd.live_challenger(0, 3, CH.entropy(seed))
        self.directory.mkdir()
        R.open_session(self.challenger, KEY, CH.HOST, lambda m: lms.sign_at(PRIVATE, leaf, m).encode(), self.directory)
        ours, theirs = socket.socketpair()
        self.reason: str | None = None
        self._thread = threading.Thread(target=self._serve, args=(ours,))
        self._thread.start()
        self.client = R.RemoteChallenger(theirs)
        monkeypatch.setattr(R.RemoteChallenger, "connect",
                            classmethod(lambda cls, at: self.client if at == T.AUDITOR else None))

    def _serve(self, sock: socket.socket) -> None:
        with sock:
            self.reason = R.serve(sock, self.challenger, self.directory, iter(range(1 << 20)).__next__)

    def ended(self) -> str | None:
        self._thread.join()
        return self.reason


def _serve(settings, seed: int = 9, prompt: int = 5):
    """`test_pouw_h1._serve_window`'s two served steps of a 3-step window, its activations drawn from `seed` and request
    a's one prompt token `prompt`."""
    model, runner = T.Model(T.Linear(32, 1), T.Linear(40, 2)), settings["pouw"]["runner"]
    comp = T.PO.install(model, T.PO.Context(stage="serve", enabled=(T._entry(),), settings=settings))
    try:
        g = torch.Generator().manual_seed(seed)
        for so, rows in ((T._so(("a", [prompt], 4)), (1, 5)),
                         (T._so(("b", [7], 2), ("c", list(range(8, 39)), 3), running={"a": 1}), (33, 2))):
            runner.execute_model(so)
            for m, layer in zip(rows, model.layers):
                layer.quant_method.apply(layer, (torch.randn(m, T.K, generator=g) * 3).bfloat16())
    finally:
        comp.close()
    return comp.installed[0].commitment()


def _coins(commitment):
    return [r["coin"]["coin"] for r in commitment["rounds"]]


def test_two_sessions_under_the_auditors_key_both_pass_for_one_window_so_the_prover_keeps_the_coins_it_likes(
        tmp_path, monkeypatch):
    a = _Session(tmp_path, monkeypatch, "a", leaf=0, seed=11)
    first = _serve(T._remote())
    b = _Session(tmp_path, monkeypatch, "b", leaf=1, seed=12)
    second = _serve(T._remote())
    assert a.ended() is None and b.ended() is None
    assert PHC.trusted_rounds_reason(first, a.directory, KEY) is None
    assert PHC.trusted_rounds_reason(second, b.directory, KEY) is None
    assert first["weights_root"] == second["weights_root"] and first["calls"] == second["calls"]
    assert _coins(first) != _coins(second)
    for d in (a.directory, b.directory):
        record = json.loads((d / R.RECORD).read_text())
        assert sorted(record) == ["commitment", "host", "signature"]
        assert sorted(record["commitment"]) == ["epoch", "families", "root", "scheme"]
        assert record["commitment"]["epoch"] == 0
        c = LiveCommitment.from_json(record["commitment"])
        assert PHC.lms_verify(KEY, record_message(KEY.public[:16], record["host"], c), bytes.fromhex(record["signature"]))
    assert PHC.trusted_rounds_reason(first, b.directory, KEY) is not None


def test_a_draw_coin_opened_before_the_windows_first_round_passes_the_trusted_check(tmp_path, monkeypatch):
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=13)
    probe = hashlib.sha512(b"a root of the prover's, before any round").digest()
    draw = host.client.open(Wd.DRAW, 0, probe)
    commitment = _serve(T._remote())
    assert host.ended() is None
    assert PHC.trusted_rounds_reason(commitment, host.directory, KEY) is None
    received = R.released_roots(host.directory / R.RELEASES, host.challenger.commitment)
    assert received[Wd.DRAW] == (probe,) and draw.after == probe
    events = [json.loads(line) for line in (host.directory / R.RELEASES).read_text().splitlines()]
    assert [e["label"] for e in events if e.get("event") == "release"][:2] == [Wd.DRAW, Wd.SALT]


def test_round_roots_and_the_coins_after_them_are_the_same_whatever_a_the_window_serves(monkeypatch):
    monkeypatch.setattr(PH.os, "urandom", lambda n: bytes(n))
    served = [_serve(T._settings(), seed) for seed in (9, 10, 9)]
    roots = [[r["root"] for r in c["rounds"]] for c in served]
    assert roots[0] == roots[1] == roots[2] and len(roots[0]) == 3
    assert _coins(served[0]) == _coins(served[1])
    other_request = [r["root"] for r in _serve(T._settings(), 9, prompt=6)["rounds"]]
    assert other_request[0] != roots[0][0]


def test_the_order_of_roots_and_coins_is_an_unsigned_file_and_a_rewrite_of_it_passes(tmp_path, monkeypatch):
    host = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=15)
    weights = hashlib.sha512(b"the weights' root").digest()
    salt = host.client.open(Wd.SALT, 0, weights)
    early = [hashlib.sha512(b"before any round, %d" % s).digest() for s in range(3)]
    coins = [host.client.open(Wd.ROUND, s, early[s]) for s in range(3)]
    host.client.close()
    assert host.ended() is None
    late = [hashlib.sha512(b"a root chosen after every coin, %d" % s + c.coin).digest() for s, c in enumerate(coins)]
    c = host.challenger.commitment.to_json()
    commits = {"coins": c, "round_coins": c, "weights_root": weights.hex(), "salt_opening": salt.to_json(), "steps": 3,
               "rounds": [{"round": s, "root": late[s].hex(), "coin": {**coins[s].to_json(), "after": late[s].hex()}}
                          for s in range(3)]}
    assert PHC.trusted_rounds_reason(commits, host.directory, KEY) == (
        "the window's round roots are not the roots the challenger's log shows it received, in order")
    log = host.directory / R.RELEASES
    text = log.read_text()
    for e, x in zip(early, late):
        text = text.replace(e.hex(), x.hex())
    log.write_text(text)
    assert PHC.trusted_rounds_reason(commits, host.directory, KEY) is None


def test_the_round_rule_of_record_passes_in_process_coins(tmp_path, monkeypatch):
    local = _serve(T._settings())
    assert local["coin_source"] == "in-process"
    assert RQ.rounds_reason(local["rounds"], LiveCommitment.from_json(local["round_coins"])) is None
    auditor = _Session(tmp_path, monkeypatch, "auditor", leaf=0, seed=14)
    auditor.client.close()
    assert auditor.ended() is None
    assert "coins from a challenger the verifier doesn't hold" in PHC.trusted_rounds_reason(local, auditor.directory, KEY)
