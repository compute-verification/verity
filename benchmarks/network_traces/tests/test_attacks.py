"""attacks: each attack on the served path that the network certifier's path decides, played on the network certifier
itself at the scan's w63k4 schedule on a `VirtualClock`, the prover's relay (`attacks.Relay`) between its server
connections and an id-form endpoint that records each request it served as serve-openai's requests.jsonl does, and
read as the judge reads a served rehearsal (`attacks.verdict` of the network certifier's check and the piece's steps 1
and 2, `attacks.piece_of`):
- timing and chunking: a relay holding events by secret bits, or writing a byte at a time, leaves every cell and
  bucket the honest relay's, and the check accepts; a response held past its first cell's deadline fails closed
  (late), inside k_fail, and certifies no served answer;
- size padding: a padding cell is a DATA cell that is no output's, which the check refuses and the piece rejects; at
  the egress cells' cap it fails closed (limit) instead; an extra id event is the piece's reject in the clear;
- an altered output token is the piece's reject in the clear;
- an extra connection the relay answers itself is a certified request the endpoint didn't serve; one arriving with the
  queue full is REFUSED and its execution phase certified nowhere;
- a request served straight to the endpoint, under another user or the certified one's, is served traffic the network
  certifier didn't carry.
`bind`'s two reads of the served requests (a certified request served twice, or never) are held on the honest run's."""
from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "live_challenger")]

import attacks as AT  # noqa: E402
import auditor as A  # noqa: E402
import certifier_piece as CP  # noqa: E402
import scan_zk as sz  # noqa: E402
import warden_chain as WC  # noqa: E402

from verity.core.protocols.certifier import warden_rows as WR  # noqa: E402
from verity.core.protocols.certifier.active import DATA, VirtualClock  # noqa: E402
from verity.core.protocols.certifier.release import release_of, with_release  # noqa: E402
from verity.core.protocols.certifier.warden import END, REFUSED, Shim, Warden, cells, read_cell  # noqa: E402

WS = AT.WS
#: the network certifier's schedule on the served path: the scan's, at w63k4's egress cells (4 ids and the trailer)
PARAMS = dataclasses.replace(sz.PARAMS, egress_cells=A.SCAN_CLASSES[CP.V3C_SCAN_CLASS][0])
SESSION = bytes(range(32))
#: the Commit period the endpoint's records put the group in
K = 3


class _Yield:
    def __await__(self):
        yield


class Pipe:
    """One direction of an in-memory connection, written as an `asyncio.StreamWriter` is: `write` feeds `reader` (and
    keeps the bytes in `data`), `close` ends it, `drain` lets the other side run; each pokes the run."""

    def __init__(self, poke) -> None:
        self.reader, self.data, self.closed, self.transport, self.poke = asyncio.StreamReader(), bytearray(), False, \
            self, poke

    def write(self, data: bytes) -> None:
        if not self.closed and data:
            self.reader.feed_data(data)
            self.data += data
            self.poke()

    async def drain(self) -> None:
        await _Yield()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.reader.feed_eof()
            self.poke()

    def is_closing(self) -> bool:
        return self.closed

    def get_write_buffer_size(self) -> int:
        return 0


class _Client:
    """The network certifier's client side of a connection: each write with the tick it came in."""

    def __init__(self, warden: Warden) -> None:
        self.warden, self.writes, self.closed, self.transport = warden, [], False, self

    def write(self, data: bytes) -> None:
        self.writes.append((self.warden.ticks, bytes(data)))

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def get_write_buffer_size(self) -> int:
        return 0


def answer_ids(body: dict) -> list[int]:
    """The endpoint's output: exactly ``max_tokens`` ids, fixed by the prompt."""
    return [(sum(body["prompt"]) * 7919 + 31 * k) % 128_256 for k in range(body["max_tokens"])]


@dataclasses.dataclass
class Played:
    out: Path
    check: dict
    io: dict
    served: list[dict]
    relay: AT.Relay
    warden: Warden

    @property
    def session(self) -> dict:
        """The front's outcome as `served_session.outcome` gives it: the gate's exit and its check's verdict."""
        return {"rc": 0 if self.check["accepted"] else 1, "verdict": {"accepted": self.check["accepted"]}}

    def piece(self) -> dict:
        (self.out / "end.json").write_text(json.dumps(self.session))
        return AT.piece_of(self.out, self.served)


def play(tmp: Path, attack: str, *, tokens: int = 1, prompts: int = 4, nth: int = 1, bits=lambda: 1,
         name: str | None = None) -> Played:
    """`prompts` id requests of `tokens` ids through the network certifier at `PARAMS`, its server connections through
    the relay playing `attack` on connection `nth` to the endpoint, every tick advanced only once each connection waits
    on the network certifier or the clock: its records certified and published as `warden_serve.serve` publishes them
    (in tmp/`name`, the attack's by default), its client's view as `warden_serve.client` writes it, and the endpoint's
    records (requests.jsonl's rows)."""
    name = name or attack
    out = tmp / name
    out.mkdir()
    p, B, P, tau = PARAMS, PARAMS.cell_bytes, PARAMS.payload, PARAMS.bucket_ms / 1000
    requests = [WS.id_request(model="m", user=f"user-{j}", prompt=[j + 1] * 8, max_tokens=tokens, frame_bytes=B,
                              offset=0) for j in range(prompts)]
    signer = WS.cs.signer(tmp / f"{name}.key", 4)
    publisher = WS.Publisher(out, WS.Issuer(signer, SESSION), p, WR.Cells.of)
    windows = WS.group_windows(p, tokens)
    to_endpoint: list[Pipe] = []

    async def main() -> tuple[list[_Client], Warden, AT.Relay]:
        clock = VirtualClock(-tau / 2)
        warden: Warden | None = None
        chains: dict[int, dict] = {}
        held: dict[asyncio.Task, dict] = {}
        tasks: list[asyncio.Task] = []

        def poke() -> None:
            warden._changed.set()

        async def hold(seconds: float) -> None:
            chain = held[asyncio.current_task()]
            chain["holding"] = True
            poke()
            try:
                await clock.sleep(seconds)
            finally:
                chain["holding"] = False

        async def endpoint(reader: asyncio.StreamReader, writer: Pipe) -> None:
            try:
                body = json.loads((await AT.read_request(reader)).partition(b"\r\n\r\n")[2])
                writer.write(WS.id_response(answer_ids(body), P))
            finally:
                writer.close()

        async def open_endpoint():
            up, down = Pipe(poke), Pipe(poke)
            to_endpoint.append(up)
            tasks.append(asyncio.ensure_future(endpoint(up.reader, down)))
            return down.reader, up

        relay = AT.Relay(attack, open_endpoint, cell_bytes=B, bucket_s=tau, delay=p.delay, nth=nth, sleep=hold,
                         bits=bits)

        async def connect():
            to_relay, to_warden = Pipe(poke), Pipe(poke)
            chain = {"to_warden": to_warden, "holding": False}
            task = asyncio.ensure_future(relay.handle(to_relay.reader, to_warden))
            held[task] = chains[id(to_warden.reader)] = chain
            tasks.append(task)
            return to_warden.reader, to_relay

        def waiting(c) -> bool:
            """Connection c moves only with the network certifier or the clock: its request held for its release,
            its response ended or past the cap, or its relay holding with every byte it wrote read."""
            if c.server is None or not c.ingress_ended:
                return False
            chain = chains[id(c.server[0])]
            return (c.release is None or c.egress_ended or c.ended is not None
                    or c.bytes_out > p.egress_cells * p.payload
                    or (chain["holding"] and len(chain["to_warden"].data) == c.bytes_out))

        warden = Warden(p, connect, origin=0.0, windows=windows, clock=clock, sleep=clock.sleep, publish=publisher,
                        withhold=publisher.withhold)
        running = asyncio.create_task(warden.run())
        clients, handled = [], []
        for request in requests:
            reader = asyncio.StreamReader()
            reader.feed_data(cells(request, B))
            reader.feed_eof()
            clients.append(_Client(warden))
            handled.append(asyncio.create_task(warden.handle(reader, clients[-1])))
        await warden.until(lambda w: w.ordinals + w.refused == len(requests))
        for t in range(windows * p.window):
            await warden.until(lambda w: all(waiting(c) for c in w.active))
            clock.advance(t * tau)
            await warden.until(lambda w, t=t: w.ticks > t)
        await running
        await asyncio.gather(*handled)
        for task in tasks:
            task.cancel()
        return clients, warden, relay

    clients, warden, relay = asyncio.run(main())
    publisher.close()
    dep = WS.deployment_record(p, signer.public_key(), SESSION, origin_unix=0.0, windows=windows,
                               listen=("127.0.0.1", 1), upstream=("127.0.0.1", 2), slack=warden.slack,
                               run_id=warden.run_id)
    (out / "deployment.json").write_text(json.dumps(dep))
    log = []
    for index, (request, client) in enumerate(zip(requests, clients)):
        shim = Shim(B)
        for t, data in client.writes:
            for i in range(0, len(data), B):
                kind, shim.ordinal, payload = read_cell(data[i:i + B], B)
                shim.arrivals.append((t * tau + 0.001, kind))
                shim.data += payload if kind in (DATA, END) else b""
        kinds = [k for _, k in shim.arrivals]
        shim.outcome = "refused" if REFUSED in kinds else "end" if END in kinds else "cut"
        log.append(WS.connection_log(index, 0.0, 0.0, request, shim, 0.0, P))
    (out / "client.json").write_text(json.dumps({"connections": log}))
    served = []
    for i, pipe in enumerate(to_endpoint):
        head, _, body = bytes(pipe.data).partition(b"\r\n\r\n")
        doc, rel = json.loads(body), release_of(head + b"\r\n\r\n")
        served.append({"engine_id": f"cmpl-{i}", "user": doc["user"], "prompt_ids": doc["prompt"],
                       "streamed_ids": answer_ids(doc), "windows": [K],
                       "release": None if rel is None else [rel[1].hex(), rel[0]]})
    return Played(out, WS.check(out), WC.io(out), served, relay, warden)


@pytest.fixture(scope="module")
def honest(tmp_path_factory) -> Played:
    return play(tmp_path_factory.mktemp("honest"), "honest")


def _judged(name: str, run: Played) -> dict:
    got = AT.verdict(name, run.session, run.piece())
    assert got["ok"], got
    return got


def test_every_case_has_an_attack_and_every_attack_a_test_a_wait_or_a_named_gap():
    assert {a.case for a in AT.ATTACKS.values()} == set(AT.CASES)
    for a in AT.ATTACKS.values():
        assert a.tests or a.waits or a.outcome == "gap", a.name
        assert all((HERE.parents[1] / t).is_file() for t in a.tests), a.name
        if a.served:
            assert a.relay in AT.RELAYED and a.gate is not None and a.piece in AT.PIECE
    table = AT.table()
    assert all(f"`{name}`" in table for name in AT.ATTACKS) and len(table.splitlines()) == len(AT.ATTACKS) + 2
    assert set(CP.parser().parse_args(["front", "1", "s", "o", "--attack", "late"]).__dict__) >= {"attack",
                                                                                                  "attack_nth"}


def test_the_honest_relay_is_the_endpoint_and_the_check_and_the_piece_pass(honest):
    assert honest.check["accepted"], honest.check["problems"]
    assert honest.io["accepted"] and len(honest.io["records"]) == 4
    assert [r["output"] for r in honest.io["records"]] == [r["streamed_ids"] for r in honest.served]
    assert {r["release"][0] for r in honest.served} == {honest.warden.run_id.hex()}
    assert honest.piece() == {"verdict": "bound", "why": AT.BOUND}
    assert [line["attack"] for line in honest.relay.lines] == ["honest"] * 4


@pytest.mark.parametrize("attack", ["jitter", "chunk"])
def test_holding_events_by_secret_bits_or_chunking_them_moves_no_cell_on_the_virtual_clock(attack, honest, tmp_path):
    bits = itertools.cycle([1, 0, 1, 1]).__next__
    run = play(tmp_path, attack, tokens=4, bits=bits)
    ref = play(tmp_path, "honest", tokens=4)
    assert run.check["accepted"], run.check["problems"]
    assert WS.egress_cells(WS.read_records(run.out, PARAMS)[WS.EGRESS], PARAMS) == \
        WS.egress_cells(WS.read_records(ref.out, PARAMS)[WS.EGRESS], PARAMS)
    assert [(r["output"], r["frames"], r["release"]) for r in run.io["records"]] == \
        [(r["output"], r["frames"], r["release"]) for r in ref.io["records"]]
    assert all(line["attack"] == attack for line in run.relay.lines)
    if attack == "jitter":
        assert sum(line["held_s"] for line in run.relay.lines) > 0 and any(1 in line["bits"] for line in
                                                                          run.relay.lines)
    _judged(attack, run)


def test_a_response_held_past_its_first_cells_deadline_fails_closed_late_and_certifies_no_answer(tmp_path):
    run = play(tmp_path, "late", nth=2)
    assert not run.check["accepted"]
    assert any(AT.LATE in x for x in run.check["problems"]), run.check["problems"]
    assert sorted(map(str, (c["failed"] for c in run.warden.log))) == ["None"] * 3 + ["late"]
    assert {why for _, why in run.io["violations"]} == {WC.FAILED_CELL}
    got = _judged("late", run)
    assert got["piece"] == "missing" and "the network certifier's own check" in got["why"]


def test_a_padding_cell_is_no_outputs_and_at_the_cap_fails_closed_limit(tmp_path):
    run = play(tmp_path, "pad")
    assert not run.check["accepted"]
    assert any(AT.NOT_WHOLE in x for x in run.check["problems"]), run.check["problems"]
    assert {why for _, why in run.io["violations"]} == {WC.NOT_ID_CELL}
    assert _judged("pad", run)["piece"] == "reject"
    cap = play(tmp_path, "pad", tokens=PARAMS.egress_cells - 1, name="pad-cap")
    assert not cap.check["accepted"] and any(AT.LIMIT in x for x in cap.check["problems"]), cap.check["problems"]
    assert [c["failed"] for c in cap.warden.log].count("limit") == 1
    assert {why for _, why in cap.io["violations"]} == {WC.FAILED_CELL}
    assert _judged("pad-cap", cap)["piece"] == "missing"


@pytest.mark.parametrize("attack", ["token", "token-pad"])
def test_an_altered_or_extra_id_reaches_the_client_whole_and_the_piece_rejects_it_in_the_clear(attack, tmp_path):
    run = play(tmp_path, attack)
    assert run.check["accepted"], run.check["problems"]
    (first, *_) = run.io["records"]
    endpoint = run.served[0]["streamed_ids"]
    assert first["output"] == ([endpoint[0] ^ 1, *endpoint[1:]] if attack == "token" else [*endpoint, 0])
    assert _judged(attack, run)["piece"] == "reject"


def test_a_connection_the_relay_answers_itself_is_a_certified_request_the_endpoint_didnt_serve(tmp_path):
    run = play(tmp_path, "extra", nth=3)
    assert run.check["accepted"], run.check["problems"]
    (unserved,) = {r["user"] for r in run.io["records"]} - {r["user"] for r in run.served}
    assert len(run.served) == 3
    assert f"certified request {unserved!r}" in _judged("extra", run)["why"]


def test_a_connection_past_the_queue_is_refused_and_its_window_certified_nowhere(tmp_path):
    """Every request arrives before the first tick, when no lane is taken yet: the queue holds `queue`, one more is
    refused."""
    run = play(tmp_path, "honest", prompts=PARAMS.queue + 1)
    assert (run.warden.refused, run.warden.refused_windows) == (1, {0})
    assert not run.check["accepted"] and any(AT.WITHHELD in x for x in run.check["problems"])
    got = run.piece()
    assert got["verdict"] == "missing" and AT.WITHHELD in got["why"], got


@pytest.mark.parametrize("attack", ["bypass", "bypass-user"])
def test_a_request_served_around_the_network_certifier_is_traffic_it_didnt_carry(attack, tmp_path):
    run = play(tmp_path, attack, nth=2)
    assert run.check["accepted"], run.check["problems"]
    (copy,) = [r for r in run.served if r["release"] is None]
    assert len(run.served) == 5
    (twin,) = [r for r in run.io["records"] if r["prompt"] == copy["prompt_ids"]]
    assert (copy["user"] == twin["user"]) == (attack == "bypass-user")
    assert attack == "bypass-user" or copy["user"] not in {r["user"] for r in run.io["records"]}
    _judged(attack, run)


def test_bind_refuses_a_certified_request_served_twice_or_never(honest):
    run_id = honest.warden.run_id.hex()
    twice = [*honest.served, dict(honest.served[0], engine_id="cmpl-again")]
    with pytest.raises(CP.Rejected, match=f"'user-0' is 2 served requests: {CP.SHARED}"):
        CP.bind(honest.served, honest.io, run_id, twice)
    with pytest.raises(CP.Rejected, match=f"'user-3' is no served request: {CP.UNSERVED}"):
        CP.bind(honest.served[:3], honest.io, run_id, honest.served[:3])
    with pytest.raises(CP.Rejected, match=f"carries no release of the network certifier's run .*{CP.NOT_SEEN}"):
        CP.bind(honest.served, honest.io, "ab" * 16, honest.served)
    assert CP.bind(honest.served, honest.io, run_id, honest.served).keys() == {f"user-{j}" for j in range(4)}


def test_the_relay_counts_completions_so_a_port_probe_or_a_model_listing_is_never_the_attacks():
    """A front's network certifier connects to the endpoint's port and lists its model before its first completion:
    `--nth` counts completions, so the attack plays on the first one and the probes pass as they came."""
    B, P = PARAMS.cell_bytes, PARAMS.payload
    completions = [WS.id_request(model="m", user=f"user-{j}", prompt=[j + 1] * 8, max_tokens=2, frame_bytes=B,
                                 offset=0) for j in range(2)]
    served: list[bytes] = []

    async def endpoint(reader: asyncio.StreamReader, writer: Pipe) -> None:
        request = await AT.read_request(reader)
        served.append(request)
        writer.write(b"HTTP/1.1 200 OK\r\n\r\n{}" if request.startswith(b"GET ")
                     else WS.id_response(answer_ids(json.loads(request.partition(b"\r\n\r\n")[2])), P))
        writer.close()

    async def main() -> tuple[AT.Relay, list[bytes]]:
        async def open_endpoint():
            up, down = Pipe(lambda: None), Pipe(lambda: None)
            asyncio.ensure_future(endpoint(up.reader, down))
            return down.reader, up

        relay = AT.Relay("extra", open_endpoint, cell_bytes=B, bucket_s=0.1, delay=PARAMS.delay, nth=1)
        answers = []
        for request in (b"", b"GET /v1/models HTTP/1.1\r\nhost: s\r\n\r\n", *completions):
            to_relay, back = Pipe(lambda: None), Pipe(lambda: None)
            to_relay.write(request)
            to_relay.close()
            await relay.handle(to_relay.reader, back)
            answers.append(bytes(back.data))
        return relay, answers

    relay, answers = asyncio.run(main())
    assert [(line["attack"], line.get("completion")) for line in relay.lines] == \
        [("honest", None), ("honest", None), ("extra", 1), ("honest", 2)]
    assert served == [b"GET /v1/models HTTP/1.1\r\nhost: s\r\n\r\n", completions[1]]
    body = json.loads(completions[0].partition(b"\r\n\r\n")[2])
    assert answers[1:] == [b"HTTP/1.1 200 OK\r\n\r\n{}", WS.id_response(WS.fake_ids(body), P),
                           WS.id_response(answer_ids(json.loads(completions[1].partition(b"\r\n\r\n")[2])), P)]


def test_the_relays_copy_is_the_requests_own_but_its_release_and_user():
    B = PARAMS.cell_bytes
    request = WS.id_request(model="m", user="user-7", prompt=[5, 6], max_tokens=2, frame_bytes=B, offset=0)
    released = with_release(request, 3, b"\x01" * 16)
    assert release_of(released.partition(b"\r\n\r\n")[0] + b"\r\n\r\n") == (3, b"\x01" * 16)
    assert AT.bypass_request(released, B, same_user=True) == request
    other = AT.bypass_request(released, B, same_user=False)
    assert len(other) == len(request) and b'"user-~' in other and other.replace(b"user-~", b"user-7") == request
    with pytest.raises(ValueError):
        AT.bypass_request(b"POST / HTTP/1.1\r\n\r\n{}", B, same_user=True)
    event = AT.IT.id_event(41, PARAMS.payload)
    assert AT.other_event(event, PARAMS.payload) == AT.IT.id_event(40, PARAMS.payload)


def _kept(serve: Path, run: str, played: Played) -> None:
    """`played`'s network certifier run kept as the front keeps the gate's fetch, under the serve dir's
    private/certifier."""
    import tarfile
    kept = serve / CP.PRIVATE / run
    kept.mkdir(parents=True)
    end = serve.parent / f"{run}-end.json"
    end.write_text(json.dumps(played.session))
    with tarfile.open(kept / "fetch.tar.gz", "w:gz") as tar:
        tar.add(played.out, arcname=f"{run}/witness")
        tar.add(end, arcname=f"{run}/end.json")


def _front(front: Path, played: Played, run: str, span: tuple[float, float]) -> None:
    front.mkdir(parents=True, exist_ok=True)
    (front / "session.json").write_text(json.dumps(played.session | {"run": run}))
    (front / AT.SPAN).write_text(json.dumps({"start_unix": span[0], "end_unix": span[1]}))


def test_the_judge_reads_a_kept_front_and_its_serve_dir(tmp_path, honest):
    """`judge` on the gate's layout: the front's session.json and span, the kept fetch under the serve dir's
    private/certifier, and requests.jsonl, of which another run's requests, those without a release the endpoint
    received outside the front's span, and the warm-ups (``--warmup``'s, ``--stop-prompt``'s, or both) aren't read."""
    run, serve = "r-honest", tmp_path / "serve"
    _kept(serve, run, honest)
    start = 1000.0
    rows = [dict(r, times={"received": start + 1}) for r in honest.served]
    stray = dict(honest.served[0], engine_id="old", user="varied.0", release=None, times={"received": start - 1})
    other = dict(honest.served[0], engine_id="other", release=["cd" * 16, 0], times={"received": start + 2})
    warm = dict(honest.served[0], engine_id="warm", user="warmup-ab-0", release=None, times={"received": start + 1})
    stop = dict(warm, engine_id="stop", user="warmup-cd-0")
    (serve / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [stray, other, warm, stop, *rows]))
    front = tmp_path / "front"
    _front(front, honest, run, (start, start + 60))
    for warmups in ({"requests": [{"user": "warmup-ab-0"}], "stop_prompt": [{"user": "warmup-cd-0"}]},
                    {"stop_prompt": [{"user": "warmup-ab-0"}, {"user": "warmup-cd-0", "finish_reason": "stop"}]}):
        (front / "warmup.json").write_text(json.dumps(warmups))
        got = AT.judge("jitter", front, serve)
        assert (got["piece"], got["ok"]) == ("bound", True), (warmups, got)
    (serve / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [warm | {"user": "x"}, *rows]))
    got = AT.judge("bypass", front, serve)
    assert (got["piece"], got["ok"]) == ("reject", True), got
    assert not list((serve / CP.PRIVATE / run).glob("unpacked-*"))


def test_a_rehearsal_plays_each_attack_in_a_front_of_its_own_and_publishes_its_verdict_alone(tmp_path, honest,
                                                                                               capsys, monkeypatch):
    """`rehearse` with a stand-in front: each attack's front at the rehearsal's flags, the attack's relay and its tokens;
    its span on the node's clock; the verdict judged from what the front kept, its public form without the piece's
    why; and a front that kept nothing is the certifier missing."""
    serve, out, ticks = tmp_path / "serve", tmp_path / "out", itertools.count(1000.0, 10.0)
    serve.mkdir()
    (serve / "requests.jsonl").write_text("".join(json.dumps(dict(r, times={"received": 1001.0})) + "\n"
                                                  for r in honest.served))
    argvs = []

    def front(argv: list[str]) -> int:
        argvs.append(argv)
        if argv[argv.index("--attack") + 1] == "jitter":
            _kept(serve, "r-jitter", honest)
            _front(Path(argv[5]), honest, "r-jitter", (0.0, 0.0))
            return 0
        Path(argv[5]).mkdir(parents=True)
        (Path(argv[5]) / "session.json").write_text(json.dumps({"refused": "the gate is busy"}))
        return 4

    got = AT.rehearse(7, serve, out, ["jitter", "pad", "bypass-user"], ["--gate", "local", "--tokens", "4"],
                      run=front, clock=lambda: next(ticks))
    assert [a[a.index("--attack") + 1] for a in argvs] == ["jitter", "pad", "bypass-user"]
    assert all(a[2:7] == ["front", "7", str(serve), str(out / n), "--gate"] for a, n in
               zip(argvs, ["jitter", "pad", "bypass-user"]))
    assert argvs[1][-2:] == ["--tokens", str(AT.UNDER_CAP)] and "--tokens" not in argvs[0][-2:]
    assert json.loads((out / "jitter" / AT.SPAN).read_text()) == {"start_unix": 1000.0, "end_unix": 1010.0, "rc": 0}
    assert [(v["attack"], v["piece"], v["ok"]) for v in got] == [("jitter", "bound", True), ("pad", "missing", False),
                                                                 ("bypass-user", "missing", False)]
    assert "the gate is busy" in got[1]["why"]
    pub = json.loads((out / "attacks.public.json").read_text())
    assert pub == [AT.public(v) for v in got] and all("why" not in v for v in pub)
    assert json.loads((out / "attacks.json").read_text()) == got
    assert json.loads((out / "pad" / "verdict.json").read_text()) == got[1]
    assert json.loads((out / "session.json").read_text()) == {"rc": 1, "ended": "the rehearsal of 3 attacks ended"}
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("ATTACK ")]
    assert [json.loads(x.removeprefix("ATTACK ")) for x in lines] == pub
    called = []
    monkeypatch.setattr(AT, "rehearse", lambda *a: called.append(a) or [{"ok": True}])
    assert AT.main(["rehearse", "7", "s", "o", "--attacks", "late,extra", "--", "--gate", "local", "--tokens", "3"]) == 0
    assert called == [(7, Path("s").resolve(), Path("o").resolve(), ["late", "extra"],
                       ["--gate", "local", "--tokens", "3"])]
    with pytest.raises(SystemExit):
        AT.main(["rehearse", "7", "s", "o", "--attacks", "swapped-read", "--", "--gate", "local"])
