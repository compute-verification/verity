"""served_zk's commitment and its check (`served_zk/served_commit.py`): a served call's fp8-p rows committed as C-Flock commits
a unit's rows, and a tile proof's public file accepted only when its committed rows are the drawn rows of that commitment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from verity.protocols.accounting.work.pouw.circuit import leaves as L
from verity.protocols.accounting.work.pouw.protocol import Context, Matmul, Shape
from verity.protocols.accounting.work.pouw.schemes import pearl_c as C
from verity.protocols.accounting.work.pouw.schemes.pearl_c_device import DEVICES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "served_zk"))
import served_commit as SC  # noqa: E402

K, M, N = 128, 3, 4


def _rows(seed: int, n: int) -> list[bytes]:
    x = np.random.default_rng(seed).standard_normal((n, K)).astype("<f4")
    return [r.tobytes() for r in x]


@pytest.fixture(scope="module")
def served():
    scheme = C.PearlC(k_min=K, forming="v1", device=DEVICES["sm120"], hashing="h2")
    tag = lambda s: hashlib.sha256(s).digest()  # noqa: E731
    ctx = Context(tag(b"salt"), Matmul(5, Shape(16, K, N), 2), tag(b"root_a"), tag(b"root_b"))
    noise = scheme.noise(ctx)
    a = _rows(1, M) + [bytes(4 * K)]                 # a zero row, which forming refuses
    words = {"a": SC.form(scheme, noise, K, "a", a, list(range(M + 1))),
             "b": SC.form(scheme, noise, K, "b", _rows(2, N), list(range(N)))}
    return scheme, ctx, noise, words


def _commit(words, run="r-request"):
    binding = {"run": run, "manifest_sha256": "00" * 32, "scheme": "pearl-c-sm120-v1-h2", "matmul": 5, "shape": [16, K, N],
               "weight": 2, "real_rows": M + 1}
    sides, priv = SC.commit_words(binding, words, K)
    return {"schema": SC.SCHEMA, "binding": binding, "sides": sides}, priv


def _pub(bca: bytes, bcb: bytes, c) -> bytes:
    """A tile proof's public file as `verity_flock.circuit.write` lays it out: its header, then instance 0's b || c in port
    order (the A row, the B row, the tile digest's row)."""
    schemas = {"a0": c["sides"]["a"]["schema"], "b0": c["sides"]["b"]["schema"], "o0": L.ROW_SCHEMA["tile", SC.SIDES["a"]]}
    head = {"instances": 1, "frame_v3": {"schemas": schemas}}
    return json.dumps(head).encode() + b"\n" + bca + bcb + bytes(128)


def _draw(c, a, b):
    return {"commitment_sha256": hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest(), "a": a, "b": b}


def test_rows_are_the_schemes_operands(served):
    """Each formed row is the fp8-p row of the operands `PearlC.encode` forms for a tile holding it, and the refused row is
    excluded."""
    scheme, ctx, noise, words = served
    assert words["a"][M] is None and all(w is not None for w in words["a"][:M] + words["b"])
    from verity.protocols.accounting.work.pouw.protocol import Tile
    ops = scheme.encode(ctx, noise, _rows(1, M)[1:2], _rows(2, N)[3:4], Tile((1,), (3,)))
    assert words["a"][1] == L.words_fp8_p(ops.a.codes[0], ops.pa[0])
    assert words["b"][3] == L.words_fp8_p(ops.b.codes[0], ops.pb[0])


def test_commitment_is_the_statements(served):
    """A row's b || c is C-Flock's commit of the row under its salt (`leaves.commit`), and each side's root the frame over
    those leaves; the refused row is listed as excluded."""
    *_, words = served
    c, priv = _commit(words)
    assert c["sides"]["a"]["excluded"] == [M] and c["sides"]["b"]["excluded"] == []
    assert priv["b"]["bcs"][2] == L.commit("fp8-p", SC.SIDES["b"], words["b"][2], priv["b"]["salts"][2])
    again, _ = SC.commit_words(c["binding"], words, K, {s: p["salts"] for s, p in priv.items()})
    assert again == c["sides"]


def test_check_accepts_the_drawn_rows_only(served, tmp_path):
    *_, words = served
    c, priv = _commit(words)
    SC.save(tmp_path / "p.npz", priv)
    priv = SC.load(tmp_path / "p.npz")
    op = {"a": SC.opening(c, "a", priv["a"]["bcs"], 1), "b": SC.opening(c, "b", priv["b"]["bcs"], 3)}
    honest = _pub(priv["a"]["bcs"][1], priv["b"]["bcs"][3], c)
    assert SC.check(c, _draw(c, 1, 3), honest, op) is None
    # another row of the same commitment, opened at the drawn position
    assert "not the commitment's row" in SC.check(c, _draw(c, 1, 3), _pub(priv["a"]["bcs"][0], priv["b"]["bcs"][3], c), op)
    # the drawn rows of another request's commitment
    other, opriv = _commit(words, run="r-other")
    assert "not the commitment's row" in SC.check(c, _draw(c, 1, 3), _pub(opriv["a"]["bcs"][1], opriv["b"]["bcs"][3], c), op)
    oop = {"a": SC.opening(other, "a", opriv["a"]["bcs"], 1), "b": SC.opening(other, "b", opriv["b"]["bcs"], 3)}
    assert "not the commitment's row" in SC.check(other, _draw(other, 1, 3), honest, oop)
    # a draw against another commitment, an opening of another row, an excluded row
    assert SC.check(c, _draw(other, 1, 3), honest, op) == "the draw names another commitment"
    assert "not the drawn row" in SC.check(c, _draw(c, 2, 3), honest, op)
    assert "excluded" in SC.check(c, _draw(c, M, 3), honest, op)
