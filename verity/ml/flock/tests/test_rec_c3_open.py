"""``flock-circuit stacked``, candidate 3's opening-only prover
(P5): R, the stacked table, committed under plain SHA-512 leaves at fast100(m_R) and opened at two registered claims' points on
the live stream ``stacked/rep0``, against a challenger written here. The opening-only model (`rec_algebra.Shape(...,
opening_only=True)`) accepts its proof at the values R takes at those points, at fast100's least m and at m_R 23 (candidate 3's
first session), and its rounds are the stacked stream's schedule; R with one cell flipped, or a claim value one bit off, is
refused at the claims' ring switches. Skips unless ``FLOCK_CIRCUIT`` names a build of ``flock-circuit`` (feature ``sha512``)."""
from __future__ import annotations

import contextlib
import json
import os
import random
import socket
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from verity.ml.circuits.ops.boolean.gf2k import mul
from verity_flock import rec_algebra as A
from verity_flock import rec_live as L
from verity_flock import rec_sparse as RS

from test_rec_claim import ring_switch

BIN = os.environ.get("FLOCK_CIRCUIT", "")
pytestmark = pytest.mark.skipif(not (BIN and Path(BIN).exists()), reason="FLOCK_CIRCUIT names no flock-circuit build")


def _session(cmd: list[str], g: random.Random, d: Path) -> tuple[list[tuple], list[tuple[bytes, list[int]]], int]:
    """One connection of a coin server, the prover ``cmd`` its client: its requests (rounds' content aside), each round's
    content and coins, the prover's exit code."""
    sock = socket.create_server(("127.0.0.1", 0))
    host, port = sock.getsockname()
    with open(d / "out.txt", "w") as out, open(d / "err.txt", "w") as err:
        proc = subprocess.Popen(cmd + ["--verifier", f"{host}:{port}"], stdout=out, stderr=err)

    def wake():
        proc.wait()
        with contextlib.suppress(OSError):
            socket.create_connection((host, port)).close()

    threading.Thread(target=wake, daemon=True).start()
    reqs, rounds = [], []
    conn, _ = sock.accept()
    with conn, conn.makefile("rwb") as f:
        while (b := L.read_frame(f)) is not None:
            r = L.decode_req(b)
            if r[0] == L.REQ_ROUND:
                coins = [g.getrandbits(128) for _ in range(r[3])]
                rounds.append((r[2], coins))
                reqs.append((r[0], r[1], r[3]))
                L.write_frame(f, L.resp_coins(coins))
            else:
                reqs.append(r)
                L.write_frame(f, L.resp_str(L.RESP_VERDICT, "{}") if r[0] == L.REQ_FINISH else L.resp_ok())
    sock.close()
    return reqs, rounds, proc.wait()


def _points(m: int, g: random.Random) -> list[RS.RegisteredClaim]:
    pts = [[g.getrandbits(128) for _ in range(m)] for _ in range(2)]
    return [RS.RegisteredClaim(k, tuple(A.eq_table(p[:6])), tuple(p[6:])) for k, p in enumerate(pts)]


def _value(bits: np.ndarray, c: RS.RegisteredClaim) -> int:
    """R's multilinear extension at the claim's point: ``Σ_v wv[v] s[v]``, ``s`` the honest ring switch."""
    s = ring_switch(bits, list(c.x[1:]))
    return A.dot([mul(c.skip[v % 64], c.x[0] if v >= 64 else 1 ^ c.x[0]) for v in range(128)], s)


def _open(d: Path, raw: bytes, points: list, seed: int):
    """``flock-circuit stacked`` on R's bytes: R's root (``--root-out``), then the opening at ``points``."""
    d.mkdir()
    (d / "R.bin").write_bytes(raw)
    subprocess.run([BIN, "stacked", "--table", str(d / "R.bin"), "--root-out", str(d / "root.json")], check=True, capture_output=True)
    (d / "points.json").write_text(json.dumps({"x": [[L.f128_hex(v) for v in c.x] for c in points]}))
    reqs, rounds, code = _session([BIN, "stacked", "--table", str(d / "R.bin"), "--points", str(d / "points.json")], random.Random(seed), d)
    assert code == 0, (d / "out.txt").read_text() + (d / "err.txt").read_text()[-2000:]
    return json.loads((d / "root.json").read_text()), reqs, rounds


def _residuals(m: int, reqs: list, rounds: list, points: list, values: list[int]) -> list[int]:
    [proof] = [r for r in reqs if r[0] == L.REQ_PROOF]
    assert proof[1:3] == ("stacked", 0)
    sh = A.Shape(m, 6, 2, salt_len=0, opening_only=True)
    return A.residuals(sh, A.decode(sh, proof[3]), [c for _, c in rounds], [], points, claims=values)[0]


def _nonzero(xs) -> list[int]:
    return [i for i, x in enumerate(xs) if x]


@pytest.mark.parametrize("m", [22, 23])
def test_the_opening_is_accepted_by_the_opening_only_model(m, tmp_path):
    """At m_R 22 and 23 the prover sends only ``Open`` (``stacked/rep0``, its domain), the stream's rounds, ``Proof`` and
    ``Finish``; its first round binds R's digest and the cap ``--root-out`` wrote, and its rounds are ``rep_rounds``' ring
    switches and Ligerito at fast100(m_R) after that binding round. `rec_algebra.decode` reads the proof (the zerocheck and
    lincheck empty) and every residual of the opening-only shape is zero at R's values at the two points."""
    g = random.Random(m)
    raw = g.randbytes(1 << (m - 3))
    points = _points(m, g)
    root, reqs, rounds = _open(tmp_path / "honest", raw, points, m)
    assert root["m"] == m and len(bytes.fromhex(root["cap"])) == 64 << 7
    assert reqs[0] == (L.REQ_OPEN, "stacked/rep0", "", b"flock-circuit/stacked/fast100/rep0", 0)
    assert [r[0] for r in reqs] == [L.REQ_OPEN] + [L.REQ_ROUND] * len(rounds) + [L.REQ_PROOF, L.REQ_FINISH]
    cap = bytes.fromhex(root["cap"])
    got = []
    for k, (content, coins) in enumerate(rounds):
        s = L.split_round(content, k == 0)
        got.append((s.n, [len(c) // 64 for c in s.caps if c != cap], sum(c == cap for c in s.caps), len(s.free)))
        if k == 0:
            assert s.digest.hex() == root["digest"]
    assert got == [(7, [], 1, 128)] + L.rep_rounds(m, 6, 2)[m + 1:]
    bits = np.unpackbits(np.frombuffer(raw, np.uint8), bitorder="little")
    values = [_value(bits, c) for c in points]
    assert _nonzero(_residuals(m, reqs, rounds, points, values)) == []


def test_a_forged_cell_or_claim_value_is_refused(tmp_path):
    """R committed and opened with one cell flipped, the claims R's honest values: both claims' ring-switch residuals fail,
    Ligerito's and T's still hold. The honest R with claim 1's value one bit off: claim 1's residual alone fails."""
    m, g = 22, random.Random(5)
    raw = bytearray(g.randbytes(1 << (m - 3)))
    points = _points(m, g)
    bits = np.unpackbits(np.frombuffer(bytes(raw), np.uint8), bitorder="little")
    values = [_value(bits, c) for c in points]
    _, reqs, rounds = _open(tmp_path / "honest", bytes(raw), points, 1)
    assert _nonzero(_residuals(m, reqs, rounds, points, values)) == []
    assert _nonzero(_residuals(m, reqs, rounds, points, [values[0], values[1] ^ 1])) == [1]
    raw[g.randrange(len(raw))] ^= 1 << g.randrange(8)
    _, reqs, rounds = _open(tmp_path / "forged", bytes(raw), points, 2)
    assert _nonzero(_residuals(m, reqs, rounds, points, values)) == [0, 1]
