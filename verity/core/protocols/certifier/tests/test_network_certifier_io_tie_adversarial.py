"""Adversarial cases against the I/O tie's prototype (`io_tie`) from the advice red team (PROTOCOL.md, "Advice for
arbitrary serving: what it can carry"), each refused or priced:

- No field of a frame but the id is free: another stream, frame kind or payload length, or a finish reason, is
  refused by the reference and by `IoTieEvent`. So concurrent responses can't be told apart by stream and a response's
  last event can't say why it stopped: arbitrary serving needs both fields fixed by the ingress and the advice, not
  left to the prover.
- The port holds F outputs a request (FIXLEN): a variable-length response has no port, a forced refusal until the map
  comes from the advice.
- Attack 8, a covert payload in frames the draws miss: the tie marks exactly the changed positions, so a payload of m
  frames passes K draws with probability (1 − m/N)^K and carries m·log2 V bits: the ε·N·log2 V term at δ.
"""

import math
import random

import numpy as np

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import head_check as HC, io_tie as IT
from verity.core.protocols.certifier.active import CANONICAL_STREAM, DATA, HEADER

from test_network_certifier_io_tie import PARAMS, PORT, Window, coin_for, requests

B = PARAMS.frame_bytes


def _event_circuit(frames, ids):
    fn = bind(IT.IoTieEvent, B=B)
    args = [np.asarray([HC._bytes_bits(f) for f in frames], dtype=np.uint8),
            np.asarray([HC._int_bits(x, 32) for x in ids], dtype=np.uint8)]
    return [int(b) for b in np.asarray(evaluate_bits(fn, *args)).reshape(-1)]


def _reheader(frame: bytes, kind: int = DATA, stream: int = CANONICAL_STREAM, n: int = IT.ID_EVENT_BYTES) -> bytes:
    return HEADER.pack(kind, stream, n) + frame[HEADER.size:]


def test_no_frame_field_but_the_id_is_free():
    x = 151_000
    f = IT.id_frame(x, B)
    event = IT.id_event(x)
    variants = {
        "another stream": _reheader(f, stream=CANONICAL_STREAM + 1),
        "another kind": _reheader(f, kind=DATA ^ 1),
        "a longer payload": _reheader(f, n=IT.ID_EVENT_BYTES + 1),
        "a finish reason": (HEADER.pack(DATA, CANONICAL_STREAM, len(event) + 2)
                            + event.replace(b'"finish_reason":null', b'"finish_reason":"stop"')).ljust(B, b"\0"),
        "a second id in the fill": (f[:HEADER.size + IT.ID_EVENT_BYTES] + b"%10d" % 7).ljust(B, b"\0"),
    }
    assert IT.frame_id(f) == x
    for name, v in variants.items():
        assert len(v) == B and IT.frame_id(v) is None, name
    assert _event_circuit([f, *variants.values()], [x] * (1 + len(variants))) == [1] + [0] * len(variants)


def test_a_variable_length_response_has_no_port():
    reqs = requests(3)
    short = [(reqs[0][0][:PORT.F - 1], reqs[0][1]), *reqs[1:]]
    long = [(reqs[0][0] + [7], reqs[0][1]), *reqs[1:]]
    for bad in (short, long):
        try:
            PORT.rows(bad)
        except ValueError as e:
            assert "outputs" in str(e)
        else:
            raise AssertionError("a request of other than F outputs has a port")


def test_a_payload_in_m_frames_fails_exactly_m_positions_and_is_priced():
    """The tie measures the changed share exactly, so what passes K draws at δ is m < N·ln(1/δ)/K frames, each free
    to carry log2 V bits: at the served shape's busiest window (N = 35,386) and K = 20 a window, δ = 2^−40 lets about
    49,000 frames over a link (Σ_w ε_w·N_w), 844 kbit at V = 151,329."""
    honest = Window(requests(4))
    n = len(honest.frames)
    rng = random.Random(8)
    for m in (1, 3, 17):
        hidden = sorted(rng.sample(range(n), m))
        frames = list(honest.frames)
        for k in hidden:
            frames[k] = IT.id_frame((IT.frame_id(frames[k]) + 1 + rng.randrange(1000)) % 151_329, B)
        got = IT.reference_draws(Window(honest.requests, frames=frames).prove([coin_for(k, n) for k in range(n)]))
        assert [k for k, ok in enumerate(got) if not ok] == hidden
    K, delta, N, V = 20, 2.0 ** -40, 35_386, 151_329
    passing = N * math.log(1 / delta) / K
    assert 49_000 < passing < 49_100
    assert 843_000 < passing * math.log2(V) < 845_000
