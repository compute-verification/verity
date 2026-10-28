"""`wipe_stream`, the streamed wipe's native fill on the memory challenger's CPUs: its labels are `streamed_wipe.label`'s
at the gates and past them, its stream is `wipe_party`'s fill frames byte for byte (a short last frame included), a
prover that left raises `ConnectionError` (the session stays unsigned), a streamer deriving other labels is refused at
its gate, and a wipe filled natively is judged as the Python fill's: accepted."""

from __future__ import annotations

import hashlib
import socket
import stat
import sys
import threading

import pytest

from pous_fakes import Clock
import pous_test_signer as ts
import pous_test_site as tsite
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_party as WP
from verity.core.protocols.pous import wipe_period as W
from verity.core.protocols.pous import wipe_store as WS
from verity.core.protocols.pous import wipe_stream as WN

KEY = hashlib.sha512(b"wipe stream test").digest()[:32]
SMALL = sw.WipeParams(1000, W.READS, 1, 1_000, (("gpu-hbm", 1000 * sw.LABEL_BYTES),))


def test_the_streamer_derives_the_labels(wipe_streamer) -> None:
    gate = wipe_streamer.gate()
    assert gate["gates"] == [list(g) for g in WN.GATES] and len(gate["sha256"]) == 64
    assert wipe_streamer.labels(KEY, 0, 300) == sw.labels(KEY, 0, 300)
    assert wipe_streamer.labels(KEY, (1 << 32) - 2, 4) == sw.labels(KEY, (1 << 32) - 2, 4)


def _frames(n: int, chunk: int) -> bytes:
    return b"".join(WP.FILL + s.to_bytes(8, "big") + min(chunk, n - s).to_bytes(4, "big")
                    + sw.labels(KEY, s, min(chunk, n - s)) for s in range(0, n, chunk))


def test_the_stream_is_wipe_partys_fill_frames(wipe_streamer) -> None:
    streamer = WN.Streamer(wipe_streamer.path, 3, 64)
    a, b = socket.socketpair()
    got: dict = {}

    def read() -> None:
        with b, b.makefile("rb") as f:
            got["bytes"] = f.read()

    t = threading.Thread(target=read)
    t.start()
    with a:
        stats = streamer.stream(KEY, 1000, a.fileno())
    t.join()
    assert got["bytes"] == _frames(1000, 64)
    assert stats["labels"] == 1000 and stats["bytes"] == 1000 * sw.LABEL_BYTES and stats["threads"] == 3


def test_a_prover_that_left_raises_connection_error(wipe_streamer) -> None:
    a, b = socket.socketpair()
    b.close()
    with a, pytest.raises(ConnectionError, match="the prover went away during the fill"):
        wipe_streamer.stream(KEY, 1 << 20, a.fileno())


def test_a_streamer_deriving_other_labels_is_refused(tmp_path) -> None:
    fake = tmp_path / "wipe_stream"
    fake.write_text(f"#!{sys.executable}\nimport sys\nsys.stdin.readline()\nprint('{{}}')\n"
                    "start, count = int(sys.argv[3]), int(sys.argv[5])\n"
                    "print(*['00' * 64] * count, sep='\\n')\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(RuntimeError, match=r"derives labels 0\.\.63 unlike streamed_wipe.label"):
        WN.Streamer(fake).gate()


@pytest.mark.parametrize("store", ["python", "host"])
def test_a_native_fill_is_judged_as_the_python_fill(wipe_streamer, store: str) -> None:
    """The challenger's side streams natively over a socket pair to `streamed_wipe.Prover` or to `wipe_store`'s prover
    on host memory, which reads each frame straight into staging; the signed session is accepted."""
    signer = ts.TestSigner(b"native fill " + store.encode(), 1)
    prover = sw.Prover(SMALL) if store == "python" else WS.StoreProver(SMALL, WS.HostMemory(), clock=Clock(10 ** 9, 1))
    a, b = socket.socketpair()
    got: dict = {}

    def native(key: bytes, n: int) -> None:
        got["fill"] = wipe_streamer.stream(key, n, a.fileno())

    def side() -> None:
        with a, a.makefile("rwb") as f:
            got["c"] = WP.serve(f, WP.request_of("n0", SMALL), site=tsite.site_of("n0", SMALL.tiers), signer=signer,
                                log=ts.session_log(), egress_bits=0, clock=Clock(10 ** 9, 1), fill=native)

    t = threading.Thread(target=side)
    t.start()
    with b, b.makefile("rwb") as f:
        answered = WP.answer(f, prover, None)
    t.join()
    c = got["c"]
    a_ = W.answer(b"n0", c.commitment.session, (signer.public_key,), c.commitment, c.signed, ts.SCHEMES,
                  mc.TEST_PREFIX + W.PROTOCOL)
    assert a_["verdict"] == "accept" and c.ended == sw.COMPLETE and answered == W.READS, a_
    assert got["fill"]["labels"] == SMALL.labels and c.filled == SMALL.labels
    if store == "host":
        r = prover.record()
        assert r["fill"]["bytes"] == SMALL.labels * sw.LABEL_BYTES and r["reads"]["n"] == W.READS
