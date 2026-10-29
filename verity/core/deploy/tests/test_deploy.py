"""The deployment layer, on the worked example (`deploy_example`): run in memory in simulated time, over TCP to a local
process, against a played-back challenger (mixed), replayed from its own transcript, and through an adversary; with
the refusals the runtime makes by name, each frame's arrival stamp, the record, and the state directory across
restarts."""

from __future__ import annotations

import json
import socket
import struct
import threading
import tomllib

import pytest

import deploy_example as X
from verity.core.deploy import manifest as M
from verity.core.deploy.channels import Playback, dial
from verity.core.deploy.runtime import ARRIVED, RUNTIME, run
from verity.core.deploy.transcript import Transcript
from verity.core.wire import PUBLIC, REFUSALS, Message, Refused, handshake

SEED = b"deploy-core worked example"
BOTH = ["challenger", "responder"]


def simulated(state, protocol=X.PROTOCOL, conf=None, **kw):
    return run(conf or X.config(), BOTH, "simulated", state, protocols={X.ID: protocol}, keys=X.keys(*BOTH),
               verify=X.verify, seed=SEED, **kw)


def stamps(t: Transcript) -> list[int]:
    return [ARRIVED.decode(e.data).fields["ns"] for e in t if e.dir == "arrived"]


def watching(party, seen: list):
    """`party`, with what each message it receives says of where it came from and when it arrived kept in `seen`."""
    class Watching(party):
        def on(self, r):
            seen.append((r.channel, r.message.type, r.at))
            return super().on(r)
    return Watching


def test_simulated_in_memory_takes_the_channels_latency(tmp_path):
    r = simulated(tmp_path)
    assert r.status == "completed", r.outcomes
    # The handshake takes two 1 ms crossings before the challenger starts, then the challenge and its answer one each.
    assert r.outcomes["challenger"] == {"outcome": {"elapsed_ns": 2_000_000}, "refused": None}
    assert r.outcomes["responder"] == {"outcome": {"ok": True, "elapsed_ns": 2_000_000}, "refused": None}
    on_disk = tmp_path / "runs" / "1"
    assert (on_disk / "challenger.transcript").read_bytes() == r.transcripts["challenger"].to_bytes()
    assert (on_disk / "record").read_bytes() == r.record()
    dirs = [e.dir for e in r.transcripts["challenger"]]
    assert dirs[:5] == ["start", "handshake-out", "handshake-in", "handshake-out", "handshake-in"]


def test_rerun_is_byte_identical(tmp_path):
    a, b = simulated(tmp_path / "a"), simulated(tmp_path / "b")
    for p in BOTH:
        assert a.transcripts[p].to_bytes() == b.transcripts[p].to_bytes()
        assert (tmp_path / "a/runs/1" / f"{p}.transcript").read_bytes() == (tmp_path / "b/runs/1" / f"{p}.transcript").read_bytes()
    assert a.record() == b.record()
    other = run(X.config(), BOTH, "simulated", None, protocols=X.PROTOCOLS, keys=X.keys(*BOTH), verify=X.verify,
                seed=SEED + b"'")
    assert other.record() != a.record()


def test_over_tcp_to_a_local_process(tmp_path):
    r = run(X.config(), ["challenger"], "live", tmp_path, protocols=X.PROTOCOLS, keys=X.keys("challenger"),
            verify=X.verify, seed=SEED, clock_step_ns=1000, processes={"responder": X.PROCESS})
    assert r.status == "completed", r.outcomes
    # A virtual clock that steps 1 us a reading: the challenger reads it on sending and on the answer.
    assert r.outcomes["challenger"] == {"outcome": {"elapsed_ns": 1000}, "refused": None}
    assert r.outcomes["responder"] == {"outcome": {"ok": True, "elapsed_ns": 1000}, "refused": None}
    # The answer is stamped on the challenger's own clock, which its first reading stepped and the stamp doesn't.
    assert stamps(r.transcripts["challenger"]) == [1000]
    child = Transcript.read(tmp_path / "runs/1/process-responder/runs/1/responder.transcript")
    assert [e.dir for e in child][-1] == "finish"
    sim = simulated(tmp_path / "sim")
    challenge = [e.data for e in r.transcripts["challenger"].on("responder") if e.dir == "out"][0]
    assert challenge == [e.data for e in sim.transcripts["challenger"].on("responder") if e.dir == "out"][0]


@pytest.mark.parametrize("adversaries", [None, {("challenger", "responder"): {"seed": 0}}], ids=["plain", "adversary"])
def test_a_local_process_that_dies_before_it_connects_fails_the_run(tmp_path, adversaries):
    broken = dict(X.PROCESS, protocols={X.ID: "deploy_example:NO_SUCH_PROTOCOL"})
    r = run(X.config(), ["challenger"], "live", tmp_path, protocols=X.PROTOCOLS, keys=X.keys("challenger"),
            verify=X.verify, seed=SEED, processes={"responder": broken}, adversaries=adversaries)
    assert r.status == "failed"
    assert r.outcomes["responder"] == {"outcome": {"exit": 1}, "refused": "party-died"}
    assert r.outcomes["challenger"]["refused"] == "closed"


def test_a_new_responder_against_the_recorded_challenger(tmp_path):
    recorded = simulated(tmp_path / "first").transcripts["challenger"]
    r = run(X.config(), ["responder"], "mixed", tmp_path / "new", protocols={X.ID: X.protocol(responder=X.NewResponder)},
            playback={"challenger": recorded})
    assert (r.mode, r.status, r.divergence) == ("mixed", "completed", None)
    assert r.outcomes["responder"]["outcome"] == {"ok": True, "elapsed_ns": 2_000_000}
    bad = run(X.config(), ["responder"], "mixed", tmp_path / "bad",
              protocols={X.ID: X.protocol(responder=X.WrongResponder)}, playback={"challenger": recorded})
    assert bad.status == "diverged"
    assert "challenger's transcript has received entry" in bad.divergence


@pytest.mark.parametrize("delay_ns", [0, 1_000_000])
def test_a_peers_frame_reaches_its_party_stamped_with_its_arrival_on_its_clock(tmp_path, delay_ns):
    """The challenger's clock in a simulated run is the run's virtual time: the answer arrives two crossings after the
    2 ms handshake, later by what an adversary holds it back. A source's answer has no stamp, and neither has a frame
    to the responder, which has no clock. The transcript keeps the stamp just before its frame, and a replay feeds it
    back."""
    seen = {p: [] for p in BOTH}
    proto = X.protocol(challenger=watching(X.Challenger, seen["challenger"]),
                       responder=watching(X.Responder, seen["responder"]))
    held = {("responder", "challenger"): {"seed": 7, "after": 2, "delay": 1.0, "delay_ns": delay_ns}}
    r = simulated(tmp_path, proto, adversaries=held if delay_ns else None)
    assert r.status == "completed", r.outcomes
    arrived = 4_000_000 + delay_ns
    assert seen["challenger"] == [("coins", "Coins", None), ("clock", "ClockReading", None),
                                  ("responder", "Answer", arrived), ("clock", "ClockReading", None)]
    assert seen["responder"] == [("challenger", "Challenge", None), ("challenger", "Verdict", None)]
    t = r.transcripts["challenger"]
    answer = next(e.n for e in t if e.dir == "in" and e.channel == "responder")
    assert (t.entries[answer - 1].dir, t.entries[answer - 1].channel, t.entries[answer - 1].kind) == (
        "arrived", "responder", "private")
    assert (stamps(t), stamps(r.transcripts["responder"])) == ([arrived], [])
    live = list(seen["challenger"])
    seen["challenger"].clear()
    again = run(X.config(), ["challenger"], "replay", None, protocols={X.ID: proto}, replay_dir=tmp_path / "runs/1")
    assert (again.status, again.divergence, seen["challenger"]) == ("completed", None, live)
    assert stamps(again.transcripts["challenger"]) == [arrived]


def test_replay_of_the_challenger_from_its_transcript(tmp_path):
    simulated(tmp_path)
    r = run(X.config(), ["challenger"], "replay", None, protocols=X.PROTOCOLS, replay_dir=tmp_path / "runs/1")
    assert (r.status, r.divergence) == ("completed", None)
    assert r.outcomes["challenger"]["outcome"] == {"elapsed_ns": 2_000_000}
    r = run(X.config(), ["challenger"], "replay", None, protocols={X.ID: X.protocol(challenger=X.GreedyChallenger)},
            replay_dir=tmp_path / "runs/1")
    assert r.status == "diverged"
    assert r.divergence == "entry 1: recorded out CoinsDraw on coins, replayed out CoinsDraw on coins"


# The responder's first two frames are its handshake's: an adversary from its third tampers with the answer.
@pytest.mark.parametrize("tamper", [{"drop": 1.0}, {"delay": 1.0, "delay_ns": 10_000_000}])
def test_an_answer_dropped_or_delayed_is_refused_late(tmp_path, tamper):
    r = simulated(tmp_path, adversaries={("responder", "challenger"): {"seed": 7, "after": 2, **tamper}})
    assert r.status == "refused"
    assert r.outcomes["challenger"] == {"outcome": {"elapsed_ns": 5_000_000}, "refused": "late"}


def test_an_answer_corrupted_is_refused_by_name():
    names = set()
    for seed in range(6):
        r = run(X.config(), BOTH, "simulated", None, protocols=X.PROTOCOLS, keys=X.keys(*BOTH), verify=X.verify,
                seed=SEED, adversaries={("responder", "challenger"): {"seed": seed, "after": 2, "corrupt": 1.0}})
        names.add(r.outcomes["challenger"]["refused"])
    assert names <= set(REFUSALS) | {"wrong"} and names & set(REFUSALS), names


def test_a_party_told_of_a_frame_the_wire_refuses_finishes_on_it():
    """The challenger is told `Refused` with the wire's name and detail and finishes on it itself: its finish is its own
    (`output`, where the runtime's would be `private`), and the refused frame is kept as `private`."""
    r = run(X.config(), BOTH, "simulated", None, protocols=X.PROTOCOLS, keys=X.keys(*BOTH), verify=X.verify,
            seed=SEED, adversaries={("responder", "challenger"): {"seed": 2, "after": 2, "corrupt": 1.0}})
    out = r.outcomes["challenger"]
    assert out["refused"] == "malformed" and out["outcome"]["channel"] == "responder" and out["outcome"]["detail"]
    t = r.transcripts["challenger"]
    assert (t.entries[-1].dir, t.entries[-1].kind) == ("finish", "output")
    assert [e.kind for e in t if e.dir == "in" and e.channel == "responder"] == ["private"]


def test_a_peer_that_resets_its_stream_is_recorded_in_the_transcript_and_the_result(tmp_path):
    """The responder's end accepts the challenger's connection, reads the first byte of its Hello and resets it: the
    challenger is told `Closed` with what failed on the stream, which its transcript keeps, and the run's result names
    it. The read makes the reset land on the stream, not on the connect."""
    server = socket.create_server(("127.0.0.1", 0))

    def reset() -> None:
        conn, _ = server.accept()
        conn.recv(1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        conn.close()

    t = threading.Thread(target=reset)
    t.start()
    r = run(X.config(), ["challenger"], "live", tmp_path, protocols=X.PROTOCOLS, keys=X.keys("challenger"),
            verify=X.verify, seed=SEED, dial_to={"responder": "%s:%d" % server.getsockname()[:2]})
    t.join()
    server.close()
    how = r.ended["challenger"]["responder"]
    assert how.startswith(("ConnectionResetError on ", "BrokenPipeError on ")), how
    told = [RUNTIME.decode(e.data) for e in r.transcripts["challenger"] if e.dir == "event"]
    assert told == [Message("Closed", {"how": how})]
    assert r.outcomes["challenger"]["refused"] == "closed" and r.summary()["ended"] == r.ended


def test_a_connect_that_fails_is_recorded_not_raised(tmp_path):
    """A dial whose connect fails (nothing listens at the address) is a channel that ended on connecting: the challenger
    is told `Closed` with it, and the run's result names it, as for a stream that fails later."""
    server = socket.create_server(("127.0.0.1", 0))
    address = "%s:%d" % server.getsockname()[:2]
    server.close()
    r = run(X.config(), ["challenger"], "live", tmp_path, protocols=X.PROTOCOLS, keys=X.keys("challenger"),
            verify=X.verify, seed=SEED, dial_to={"responder": address})
    how = r.ended["challenger"]["responder"]
    assert how.startswith("ConnectionRefusedError on connecting: "), how
    told = [RUNTIME.decode(e.data) for e in r.transcripts["challenger"] if e.dir == "event"]
    assert told == [Message("Closed", {"how": how})]
    assert r.outcomes["challenger"]["refused"] == "closed"


def test_a_connect_the_peer_resets_is_an_ended_pipe(monkeypatch):
    """A peer that accepts and resets before the dialer resumes makes the blocking connect raise
    `ConnectionResetError`; `dial` returns it as an ended pipe."""
    def reset(*_args, **_kwargs):
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(socket, "create_connection", reset)
    pipe = dial("127.0.0.1:1")
    assert pipe.closed and pipe.take() is None and pipe.fileno() is None
    assert pipe.ended == "ConnectionResetError on connecting: [Errno 104] Connection reset by peer"
    with pytest.raises(ValueError, match="HOST:PORT"):
        dial("no-port")


def test_a_corrupted_handshake_is_refused_by_name():
    r = run(X.config(), BOTH, "simulated", None, protocols=X.PROTOCOLS, keys=X.keys(*BOTH), verify=X.verify, seed=SEED,
            adversaries={("responder", "challenger"): {"seed": 1, "corrupt": 1.0}})
    assert r.outcomes["challenger"]["refused"] in set(REFUSALS) | set(handshake.REFUSALS)
    assert r.outcomes["responder"]["refused"] == "closed"


def test_a_key_other_than_the_configurations_is_refused():
    conf = X.config()
    conf["party"][1]["key"] = X.PartyKey("someone").public_key.hex()
    r = simulated(None, conf=conf)
    assert r.outcomes["challenger"]["refused"] == "handshake-key"


def test_unauthenticated_channels_run_when_asked():
    r = run(X.config(), BOTH, "simulated", None, protocols=X.PROTOCOLS, seed=SEED, authenticate=False)
    assert r.outcomes["challenger"] == {"outcome": {"elapsed_ns": 2_000_000}, "refused": None}
    assert not [e for e in r.transcripts["challenger"] if e.dir.startswith("handshake")]
    with pytest.raises(Refused) as e:
        run(X.config(), BOTH, "simulated", None, protocols=X.PROTOCOLS, seed=SEED)
    assert e.value.name == "no-key"


def test_an_undeclared_message_is_refused():
    r = simulated(None, X.protocol(responder=X.RudeResponder))
    assert r.outcomes["responder"]["refused"] == "undeclared"
    assert "Verdict" in r.outcomes["responder"]["outcome"]["detail"]
    assert r.outcomes["challenger"]["refused"] == "closed"
    told = [RUNTIME.decode(e.data) for e in r.transcripts["challenger"] if e.dir == "event"]
    assert told == [Message("Closed", {"how": "end"})] and r.ended == {}
    sent = [e for e in r.transcripts["responder"] if e.dir == "out" and e.channel == "challenger"]
    assert sent == []


def test_a_party_that_raises_fails_the_run():
    r = simulated(None, X.protocol(responder=X.DyingResponder))
    assert (r.status, r.outcomes["responder"]["refused"]) == ("failed", "party-died")


def test_a_run_where_nothing_can_happen_stalls():
    r = simulated(None, X.protocol(responder=X.SilentResponder), X.config(params={**X.PARAMS, "timer": False}))
    assert r.status == "failed"
    assert {p: o["refused"] for p, o in r.outcomes.items()} == {"challenger": "stalled", "responder": "stalled"}


def test_the_record_is_the_public_projection(tmp_path):
    r = simulated(tmp_path)
    t = r.transcripts["challenger"]
    answer = next(e.data for e in t if e.dir == "in" and e.channel == "responder")
    assert answer in t.to_bytes() and answer not in r.record()
    assert {e.kind for e in t.public()} <= PUBLIC
    nonce = next(e.data for e in t if e.dir == "in" and e.channel == "coins")
    assert nonce[-32:] in r.record()


def test_a_frame_past_the_transcripts_bound_is_kept_as_its_digest():
    t = Transcript("challenger", keep=8)
    e = t.append("out", "responder", "private", b"x" * 9)
    assert (e.elided, len(e.data), e.holds(b"x" * 9), e.holds(b"y" * 9)) == (True, 64, True, False)
    assert Transcript.of("challenger", t.to_bytes()).entries == t.entries
    with pytest.raises(Refused) as err:
        Playback(t, "responder")
    assert err.value.name == "elided"


def test_a_refusal_is_public_by_name_and_private_in_detail():
    r = simulated(None, X.protocol(responder=X.DyingResponder))
    public = r.transcripts["responder"].public()
    assert [e.data for e in public if e.dir == "record"] == [b'{"refused":"party-died"}']
    assert b"fell over" not in r.record() and b"fell over" in r.transcripts["responder"].to_bytes()


SITE = {"machine": [{"name": "auditor-box", "owner": "auditor", "provides": ["clock", "coins"]},
                    {"name": "gpu-node", "owner": "developer", "provides": ["gpu"]}],
        "link": [{"ends": ["auditor-box", "gpu-node"], "latency_ms": 1}]}


def test_placement_refuses_by_name_or_marks_the_run_simulated(tmp_path):
    r = simulated(tmp_path / "a", conf=X.config(**SITE))
    assert r.placement.to_json() == {"mode": "simulated", "bindings": {"challenger": "auditor-box",
                                                                        "responder": "gpu-node"}, "unmet": []}
    swapped = X.config(**SITE)
    swapped["party"][0]["machine"], swapped["party"][1]["machine"] = "gpu-node", "auditor-box"
    with pytest.raises(Refused) as e:
        run(swapped, ["challenger"], "live", None, protocols=X.PROTOCOLS)
    assert e.value.name == "principal"
    r = simulated(tmp_path / "b", conf=swapped)
    assert r.status == "completed"
    assert [u["refused"] for u in r.placement.to_json()["unmet"]] == ["principal", "needs", "principal"]
    far = X.config(**SITE)
    far["link"] = [{"ends": ["auditor-box", "gpu-node"], "latency_ms": 9}]
    with pytest.raises(Refused) as e:
        run(far, ["challenger"], "live", None, protocols=X.PROTOCOLS)
    assert e.value.name == "latency"
    lost = X.config(**SITE)
    lost["party"][1]["machine"] = "elsewhere"
    with pytest.raises(Refused) as e:
        simulated(None, conf=lost)
    assert e.value.name == "unknown-machine"


def test_a_workload_provides_what_the_protocol_needs():
    m = M.Manifest("verity/example/needs", 1, "p", X.MANIFEST.parties, X.MANIFEST.channels, X.MANIFEST.messages,
                   needs=("windows",))
    M.workload(m, ["windows", "store"])
    with pytest.raises(Refused) as e:
        M.workload(m, ["store"])
    assert e.value.name == "workload-needs"


def test_the_manifest_renders_as_toml():
    doc = tomllib.loads(M.render(X.MANIFEST))
    assert doc["party"]["challenger"] == {"principal": "auditor", "per": "one", "needs": ["clock", "coins"]}
    assert doc["channel"][0] == {"from": "challenger", "to": "responder", "messages": ["Challenge", "Verdict"],
                                 "latency_ms": 1, "carries_coins": True}
    assert doc["message"]["Answer"] == {"version": 1, "kind": "private"}


def test_a_channels_after_names_the_manifests_messages():
    challenge, answer = X.MANIFEST.channels

    def with_after(*after):
        return M.Manifest("verity/example/after", 1, "p", X.MANIFEST.parties,
                          (challenge, M.Channel(answer.sender, answer.receiver, answer.messages, after=after)),
                          X.MANIFEST.messages)
    assert tomllib.loads(M.render(with_after("Challenge")))["channel"][1]["after"] == ["Challenge"]
    with pytest.raises(ValueError, match=r"\['Nonce'\] on responder -> challenger are not its messages"):
        with_after("Nonce")


def test_a_restart_refuses_the_run_that_never_ended(tmp_path):
    assert simulated(tmp_path).run == 1
    with open(tmp_path / "runs.jsonl", "a") as f:
        f.write('{"event":"started","run":2}\n')
    r = simulated(tmp_path)
    assert (r.run, r.incomplete, r.status) == (3, [2], "completed")
    log = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert {"event": "ended", "refused": "incomplete", "run": 2, "status": "refused"} in log
    assert simulated(tmp_path).incomplete == []
