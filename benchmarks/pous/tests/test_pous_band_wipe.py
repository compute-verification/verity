"""The band-fill wipe (`benchmarks/pous/band_wipe/`) on CPU, at a small band geometry: the prover's control loop over
`prover.CpuProver` (the reference encoder's fill and `ControlResponder`s in place of the GPU's kernel and native
responder), the memory challenger's signed sessions on its logical clock (each answer's time the responder's report, so no
outcome depends on the machine), and the verdict against blocks the challenger recomputes from the salt.  The honest wipe
is accepted under the test signer's mapping; the cheat, every 19th block foreign and rebuilt at 13 calls when challenged,
answers correctly but late, so its session ends late and is rejected, and its foreign bytes are still there after."""
from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("verity.core.protocols.pous")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "pous"))

from band_gpu.memory_challenger import LogicalClock  # noqa: E402
from band_wipe import challenger, prover, wire  # noqa: E402
from memory_challenger import live  # noqa: E402
from verity.core.protocols.pous.band import SECURE_FIRST, BandParams  # noqa: E402

BASE = BandParams(32, 4096, 12)
"""32 blocks of 512 B a segment, the band's in-degree 12."""
K = 600
"""Rounds a session: the cheat drops 1/19 of the blocks, so a session misses every one with probability (18/19)^600 < 2^-46."""


def test_limit_is_the_guarantees_deadline():
    assert challenger.limit_ns(SECURE_FIRST) == 6 * 241_500 - 1 == 1_448_999
    assert challenger.ERASE["k"] == challenger.K == 2152


def test_binding_names_the_salt_and_geometry():
    m = SECURE_FIRST.with_segments(3).manifest()
    a = wire.binding(wire.SCHEME, m, bytes(24))
    assert len(a) == 64 and a == wire.binding(wire.SCHEME, m, bytes(24))
    assert a != wire.binding(wire.SCHEME, m, bytes(23) + b"\1")
    assert a != wire.binding(wire.SCHEME, SECURE_FIRST.with_segments(4).manifest(), bytes(24))


def test_honest_wipe_accepted_and_cheat_rejected(tmp_path):
    params = BASE.with_segments(2)
    listener = socket.create_server(("127.0.0.1", 0))
    side = threading.Thread(target=prover.serve, args=(prover.CpuProver(params), listener, tmp_path / "prover"), daemon=True)
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        side.start()
        ch = wire.Channel(socket.create_connection(listener.getsockname()))
        sig = live.test_signer().TestSigner(b"\1" * 32, height=1, directory=tmp_path)
        doc = challenger.run(ch, ch.recv(), "127.0.0.1", None, tmp_path, modes=["honest", "cheat"], sessions=1, signer=sig,
                             log=live.session_log(tmp_path / "sessions", True), k=K, pings=4, clock=LogicalClock(), base=BASE,
                             strict=False)
        side.join()
        ch.close()
    finally:
        sys.setswitchinterval(interval)
        listener.close()
    result = challenger.verify(doc, tmp_path, workers=1, base=BASE)
    [honest], [cheat] = (w["sessions"] for w in result["wipes"])
    assert honest["accepted_under_signer"] and honest["bound_to_salt"]
    assert honest["rounds"] == K and honest["late"] == honest["wrong"] == 0
    assert not honest["accepted"], "a test signer's session fails closed under memory_challenger.SCHEMES"
    assert not cheat["accepted_under_signer"] and cheat["ended"] == "late"
    late = cheat["first_late"]
    assert late["dropped_by_cheat"] and late["correct"] and cheat["rounds"] == late["j"] + 1
    assert cheat["wrong"] == 0 and cheat["late"] == 1
    report = doc["wipes"][1]["prover_report"]
    assert report["foreign_kept"] and report["cheat_setup"]["dropped_blocks"] == len(range(18, params.blocks, 19))
    assert [r["index"] for r in report["recomputed"]] == [late["index"]]
    assert result["passed"]
