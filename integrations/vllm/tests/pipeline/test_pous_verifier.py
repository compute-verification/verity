"""The POUS verifier (`pipeline/pous_verifier.py`) on `verity_pous.protocol`, against a Python stand-in for the native responder's
wire format, on a small encoding of each scheme: an honest server passes, one that drops blocks fails, the per-segment encoding
is the protocol's, and the responder's report reaches every `Answer`."""
from __future__ import annotations

import socket
import struct
import threading

import pytest
from verity_pous import band, dense, protocol
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


def _run(blocks: list[bytes], scheme: protocol.Scheme, vk: VerifierKey) -> dict:
    sock = socket.create_server(("127.0.0.1", 0))
    t = threading.Thread(target=_serve, args=(sock, blocks), daemon=True)
    t.start()
    try:
        return pv.run("127.0.0.1", sock.getsockname()[1], scheme, vk, audits=2, pings=20, k=16)
    finally:
        t.join(timeout=5)
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
    assert honest["scheme"] == name and honest["k"] == 16 and honest["research"] and honest["profile"]["gaps"]
    assert honest["rtt_allowance_ns"] == int(honest["rtt"]["p99.9"] * 1e9)
    assert honest["device_s"]["max"] == 1e-6 and honest["server_s"]["max"] == 2e-6
    dropped = _run([bytes(len(b)) if i % 2 else b for i, b in enumerate(blocks)], scheme, vk)
    assert not dropped["audit_of_record"]["accepted"] and dropped["wrong"] > 0 and dropped["profile"] is None


def test_the_reference_decode_is_the_protocols():
    from verity_pous.layout import Layout

    scheme = pv.scheme_at("band-chain/d12/v1", SMALL["band-chain/d12/v1"])
    tensors = {"a": bytes(range(200)), "b": bytes(range(50, 250))}
    lay = Layout.of(band.BandParams(16, 64 * 8, 12), [(n, len(b)) for n, b in tensors.items()])
    scheme = pv.scheme_at(scheme.name, lay.params)
    salt, w = bytes(24), lay.pack(tensors)
    segments = pv.encode(scheme, w, salt, workers=2)
    assert pv.reference_decode(scheme, lay, segments, salt, ["a", "b"], workers=2) == tensors
