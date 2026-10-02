"""The POUS verifier (`pipeline/pous_verifier.py`) on `verity_pous.protocol`, against a Python stand-in for the native responder's
wire format, on a small encoding of each scheme: an honest server passes, one that drops blocks fails, the per-segment encoding
is the protocol's, and the responder's report reaches every `Answer`."""
from __future__ import annotations

import socket
import struct
import threading

import pytest
from verity_pous import audit, band, dense, protocol
from verity_pous.params import Params
from verity_pous.verifier import VerifierKey

from verity_vllm.pipeline import pous_verifier as pv

SMALL = {"p3/v1": Params(n=8, wc=64, k=3, segments=2), "dense-chain/v1": dense.DenseParams(8, 64 * 8, segments=2),
         "band-chain/d12/v1": band.BandParams(16, 64 * 8, 12, segments=2)}


def _serve(sock: socket.socket, blocks: list[bytes]) -> None:
    conn, _ = sock.accept()
    with conn:
        while True:
            req = b""
            while len(req) < 16:
                chunk = conn.recv(16 - len(req))
                if not chunk:
                    return
                req += chunk
            op, arg = struct.unpack("<QQ", req)
            if op == pv.OP_CLOSE:
                return
            body = blocks[arg] if op == pv.OP_CHALLENGE else bytes(len(blocks[0]))
            conn.sendall(struct.pack("<QQQ", 0, 1000, 2000) + body)


def _run(blocks: list[bytes], scheme: protocol.Scheme, vk: VerifierKey, **kw) -> dict:
    sock = socket.create_server(("127.0.0.1", 0))
    t = threading.Thread(target=_serve, args=(sock, blocks), daemon=True)
    t.start()
    try:
        return pv.run("127.0.0.1", sock.getsockname()[1], scheme, vk, audits=2, pings=20, k=16, **kw)
    finally:
        t.join()
        sock.close()


def _small(name: str) -> tuple[protocol.Scheme, bytes, bytes, list[list[bytes]], VerifierKey]:
    scheme = pv.scheme_at(name, SMALL[name])
    salt = bytes(range(24))
    n = scheme.params.w_bytes
    w = bytes(range(256)) * (n // 256) + bytes(n % 256)
    segments = pv.encode(scheme, w, salt, workers=2)
    return scheme, w, salt, segments, pv.verifier_key(scheme, protocol.commit(scheme, w), salt, w, segments)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_honest_passes_and_dropped_fails(name):
    scheme, w, salt, segments, vk = _small(name)
    assert segments == protocol.encode(scheme, w, salt)
    assert vk == protocol.verifier_key(scheme, protocol.commit(scheme, w), salt, w=w) == VerifierKey.from_segments(scheme.params, salt, segments)
    blocks = [b for seg in segments for b in seg]
    honest = _run(blocks, scheme, vk)
    assert honest["audit_of_record"]["accepted"] and honest["wrong"] == 0 and honest["rtt"]["n"] == 20
    assert honest["scheme"] == name and honest["k"] == 16 and honest["research_mode"] and honest["profile"]["gaps"]
    assert honest["rtt_allowance_ns"] == int(honest["rtt"]["p99.9"] * 1e9)
    assert honest["device_s"]["max"] == 1e-6 and honest["server_s"]["max"] == 2e-6
    dropped = _run([bytes(len(b)) if i % 2 else b for i, b in enumerate(blocks)], scheme, vk)
    assert not dropped["audit_of_record"]["accepted"] and dropped["wrong"] > 0 and dropped["profile"] is None


def test_a_fixed_allowance_replaces_the_quantile_and_the_tail_covers_every_answer():
    scheme, w, salt, segments, vk = _small("band-chain/d12/v1")
    honest = _run([b for seg in segments for b in seg], scheme, vk, allowance_ns=51_000)
    assert honest["rtt_allowance_ns"] == 51_000 and honest["allowance"] == "fixed" and honest["limit_ns"] == 551_000
    assert honest["rtt"]["n"] == 20 and honest["rtt_after"]["n"] == 20
    t = honest["tail"]
    assert t["audits"] == 2 and t["answers"] == 32 and t["windows_us"] == list(pv.WINDOWS_US)
    assert len(t["slowest"]) == 32 and t["slowest"][0]["server_us"] == pytest.approx(2.0) and t["slowest"][0]["device_us"] == pytest.approx(1.0)
    assert set(honest["sched"]) == {"verifier", "responder"}


def _answers(elapsed_us: list[float], wrong: tuple[int, ...] = ()) -> tuple[audit.Answer, ...]:
    return tuple(audit.Answer(j=j, index=j, correct=j not in wrong, on_time=True, elapsed_ns=int(e * 1000),
                              report={"server_s": e / 2e6, "device_s": e / 4e6}) for j, e in enumerate(elapsed_us))


def test_tail_counts_late_answers_and_the_audits_passing_at_each_window_and_tolerance():
    audits = [_answers([100, 120, 140]),          # on time at every window
              _answers([100, 260, 140]),          # one answer over 250 µs
              _answers([320, 260, 140]),          # two over 250, one over 300
              _answers([100, 120, 900]),          # one over every window
              _answers([100, 120, 140], wrong=(1,))]  # fast but wrong: never passes
    t = pv.tail(audits, windows_us=(150, 250, 300, 750), top=3)
    assert t["late_over"] == {"150": 4, "250": 4, "300": 2, "750": 1}
    assert t["audits_passing"]["150"] == {"0": 1, "1": 3, "2": 4}
    assert t["audits_passing"]["250"] == {"0": 1, "1": 3, "2": 4}
    assert t["audits_passing"]["300"] == {"0": 2, "1": 4, "2": 4}
    assert t["audits_passing"]["750"] == {"0": 3, "1": 4, "2": 4}
    assert [(s["elapsed_us"], s["audit"], s["j"]) for s in t["slowest"]] == [(900.0, 3, 2), (320.0, 2, 0), (260.0, 1, 1)]
    assert t["slowest"][0]["server_us"] == pytest.approx(450.0) and t["slowest"][0]["device_us"] == pytest.approx(225.0)
    assert t["audits"] == 5 and t["answers"] == 15


def test_an_answer_exactly_at_the_window_is_on_time():
    t = pv.tail([_answers([250.0])], windows_us=(250,))
    assert t["late_over"] == {"250": 0} and t["audits_passing"]["250"]["0"] == 1


def test_serving_overlap_is_the_generating_share_of_the_audits_window():
    from verity_vllm.pipeline.pous_run import serving_overlap

    o = serving_overlap([10.0, 20.0], [[0.0, 12.0], [12.5, 19.0], [19.5, 30.0], [31.0, 40.0]])
    assert o == {"window_s": 10.0, "generating_s": 9.0, "fraction": 0.9, "loops": 3}
    assert serving_overlap(None, []) is None


def test_the_reference_decode_is_the_protocols():
    from verity_pous.layout import Layout

    scheme = pv.scheme_at("band-chain/d12/v1", SMALL["band-chain/d12/v1"])
    tensors = {"a": bytes(range(200)), "b": bytes(range(50, 250))}
    lay = Layout.of(band.BandParams(16, 64 * 8, 12), [(n, len(b)) for n, b in tensors.items()])
    scheme = pv.scheme_at(scheme.name, lay.params)
    salt, w = bytes(24), lay.pack(tensors)
    segments = pv.encode(scheme, w, salt, workers=2)
    assert pv.reference_decode(scheme, lay, segments, salt, ["a", "b"], workers=2) == tensors
