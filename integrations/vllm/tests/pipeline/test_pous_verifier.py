"""The POUS verifier (`pipeline/pous_verifier.py`) against a Python stand-in for the native responder's wire format, on a small
reference encoding: an honest server passes, one that drops blocks fails, and the reference `vk` built per segment equals the
reference's own."""
from __future__ import annotations

import socket
import struct
import threading

from verity_pous import scheme, setup
from verity_pous.params import Params
from verity_pous.primitives import reference
from verity_pous.verifier import VerifierKey

from verity_vllm.pipeline import pous_verifier as pv

SMALL = Params(n=8, wc=64, k=3, segments=2)


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


def _run(blocks: list[bytes], vk: VerifierKey) -> dict:
    sock = socket.create_server(("127.0.0.1", 0))
    t = threading.Thread(target=_serve, args=(sock, blocks), daemon=True)
    t.start()
    try:
        return pv.run("127.0.0.1", sock.getsockname()[1], vk, audits=2, pings=20, k=16)
    finally:
        t.join(timeout=5)
        sock.close()


def test_honest_passes_and_dropped_fails():
    salt = bytes(range(24))
    w = bytes(range(256)) * (SMALL.w_bytes // 256) + bytes(SMALL.w_bytes % 256)
    enc = pv.encode(SMALL, w, salt, workers=2)
    assert enc == scheme.encode(SMALL, reference(SMALL), w, salt)
    vk = VerifierKey.build(enc)
    assert vk == setup.verifier_key(SMALL, reference(SMALL), w, setup.commit(SMALL, w), salt)
    blocks = [enc.block(i) for i in range(SMALL.blocks)]
    honest = _run(blocks, vk)
    assert honest["audit_of_record"]["accepted"] and honest["wrong"] == 0 and honest["rtt"]["n"] == 20
    assert honest["rtt_allowance_ns"] == int(honest["rtt"]["max"] * 1e9)
    dropped = _run([bytes(len(b)) if i % 2 else b for i, b in enumerate(blocks)], vk)
    assert not dropped["audit_of_record"]["accepted"] and dropped["wrong"] > 0
