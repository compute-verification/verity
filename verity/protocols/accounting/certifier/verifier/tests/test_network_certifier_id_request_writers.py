"""The id request has two writers: the verifier's `id_form.id_request` and the workload's declaration
(`certifier.prover.workload.id_request`), which the adapter imports so it reads nothing of the verifier. They write
the same bytes in the network certifier's cells (``offset=0``) and refuse the same fields, and the link frame's header
is one format on both sides."""

from __future__ import annotations

import random

import pytest

from verity.protocols.accounting.certifier.prover import workload as WL
from verity.protocols.accounting.certifier.verifier import id_form as IDF
from verity.protocols.accounting.certifier.verifier.link_frame import HEADER

U32 = (0, 1, 9, 10, 4095, (1 << 31), (1 << 32) - 1)
RATIOS = (0, 0.0, 0.5, 1, 1.0, 0.000001, 0.7, 2.25, 9.999999)
NAMES = ("m", "", "org/model-name", "x" * 62, "é" * 10, 'q"\\/', "\n\t", "😀" * 4)
FRAMES = (102, 103, 120, 128, 256, 404, 512, 1033, 4096)


def both(**fields) -> tuple[object, object]:
    """Each writer's bytes for `fields`, or its refusal as ``ValueError``."""
    out = []
    for write in (lambda: IDF.id_request(**fields, offset=0), lambda: WL.id_request(**fields)):
        try:
            out.append(write())
        except ValueError:
            out.append(ValueError)
    return out[0], out[1]


def fields(rng: random.Random) -> dict:
    return {"model": rng.choice(NAMES), "user": rng.choice(NAMES), "max_tokens": rng.choice(U32),
            "temperature": rng.choice(RATIOS), "top_p": rng.choice(RATIOS), "seed": rng.choice(U32),
            "prompt": [rng.choice(U32) for _ in range(rng.choice((1, 2, 3, 17, 200)))],
            "frame_bytes": rng.choice(FRAMES)}


def test_the_frame_header_is_one_format():
    assert WL.FRAME_HEADER.format == HEADER.format and WL.FRAME_HEADER.size == HEADER.size == 9


@pytest.mark.parametrize("frame_bytes", FRAMES)
@pytest.mark.parametrize("n", (1, 2, 5, 40, 400))
def test_both_write_the_same_bytes_over_prompt_lengths_and_frames(frame_bytes, n):
    a, b = both(model="org/m", user="u", prompt=[(7919 * k) % (1 << 32) for k in range(n)], max_tokens=16,
                temperature=0.7, top_p=0.95, seed=3, frame_bytes=frame_bytes)
    assert a == b and (a is ValueError or len(a) > 0)


@pytest.mark.parametrize("key,values", [("max_tokens", U32), ("seed", U32), ("temperature", RATIOS),
                                        ("top_p", RATIOS), ("model", NAMES), ("user", NAMES)])
def test_both_write_the_same_bytes_at_each_fields_edges(key, values):
    for v in values:
        f = {"model": "m", "user": "u", "prompt": [1, 2, 3], "max_tokens": 2, "frame_bytes": 256, key: v}
        a, b = both(**f)
        assert a == b and a is not ValueError, (key, v)


@pytest.mark.parametrize("bad", [{"max_tokens": -1}, {"max_tokens": 1 << 32}, {"max_tokens": True},
                                 {"max_tokens": 2.0}, {"seed": 1 << 32}, {"temperature": 10},
                                 {"temperature": 0.1234567}, {"temperature": -0.5}, {"temperature": True},
                                 {"top_p": float("nan")}, {"model": "x" * 63}, {"model": 5}, {"user": "é" * 11},
                                 {"prompt": []}, {"prompt": "abc"}, {"prompt": b"ab"}, {"prompt": [1 << 32]},
                                 {"prompt": [-1]}, {"frame_bytes": 101}, {"frame_bytes": 9}])
def test_both_refuse_the_same_fields(bad):
    f = {"model": "m", "user": "u", "prompt": [1, 2, 3], "max_tokens": 2, "frame_bytes": 256} | bad
    assert both(**f) == (ValueError, ValueError), bad


def test_both_agree_on_a_seeded_sweep_of_every_field():
    rng = random.Random("certifier/id-request-writers")
    wrote = 0
    for _ in range(3000):
        f = fields(rng)
        a, b = both(**f)
        assert a == b, f
        wrote += a is not ValueError
    assert wrote > 1000
