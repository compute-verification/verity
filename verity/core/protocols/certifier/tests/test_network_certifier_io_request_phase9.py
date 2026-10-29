"""IoTieRequest's adversarial cases as regressions, on the unit a bucket row of the warden's own ingress: what a
bucket's verdict holds alone, and what rests on the verifier's lanes chaining each connection over its buckets.

  a. The public inputs are the lanes (each lane's ordinal and its request's carry-in and carry-out) and still no
     advice digest or Commit: `tie_service.IoTieVerdict` keys the verdicts to the advice's digest, built only
     from proofs verified on the verifier's own lanes (`io_request.lanes`).
  b. Closed: a carried prompt-field count is no free addend. Checked alone, a bucket still accepts a lane's carry-in's
     fields + Δ with its carry-out's + Δ, but the bucket before holds its carry-out's fields, and the verifier's
     lanes chain each connection from (0, 0) to (0, n), so a link's verdicts accept no Δ.
  c. Open here: the prompt id values are tied to nothing (the ingress-ids tie, `io_ids`, holds them to the Commit's
     `c00.input/ids`).
  d. Changed: the ordinal is read. Each cell's header carries its connection's ordinal, and a lane's cells must be on
     the lane's q, the warden's release order[i] + 1.
  e. The unit reads neither the response's length nor the user: `max_tokens` is parsed as a field and not compared.
"""
from __future__ import annotations

import inspect

import pytest

from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import warden as W
from verity.core.protocols.certifier.active import DATA, HEADER, id_request

B = 512
PARAMS = W.WardenParams(cell_bytes=B, slots=2, lanes=2, window=8, delay=1)
SHAPE = Q.Shape(B, PARAMS.lanes, PARAMS.slots)
#: request 1 (40 tokens, 2 cells) released one cell a bucket, over buckets 1 and 2
SPAN_PL, ARRIVALS, RATES = (5, 40, 5), (0, 0, 1), (10 ** 6, 520, 10 ** 6)


def _ids(i, k):
    return (7919 * i + 104729 * k) % 151329


def _requests(prompt_lens, idfn=_ids):
    return [id_request(model="glm", user=f"u{i}", prompt=[idfn(i, k) for k in range(n)], max_tokens=16,
                       temperature=0.7, top_p=0.95, seed=i, frame_bytes=B, offset=0)
            for i, n in enumerate(prompt_lens)]


def _link(prompt_lens=SPAN_PL, idfn=_ids):
    return Q.serve(PARAMS, _requests(prompt_lens, idfn), ARRIVALS, RATES, buckets=PARAMS.window)


def _ref(link, t, lanes_):
    return Q.reference(SHAPE, Q.Instance(tuple(link.rows[t]), tuple(lanes_)))


def _verdicts(link, prompt_lens):
    try:
        return [_ref(link, t, Q.lanes(SHAPE, prompt_lens, link.releases, t)) for t in range(len(link.rows))]
    except ValueError:
        return None


def test_honest_link_accepts():
    link = _link()
    assert link.releases[1].cells == (0, 1, 1)
    assert _verdicts(link, SPAN_PL) == [1] * PARAMS.window


def test_the_public_inputs_are_the_lanes():
    names = [n for n, _ in Q._sig_request(SHAPE)[0]]
    assert names == ["row", "lanes"]
    assert not [n for n in names if any(s in n.lower() for s in ("advice", "digest", "commit", "ids", "port"))]


def _spanning(link):
    """The bucket in which request 1 (connection 2) is in progress at both ends of its lane: (t, lane index)."""
    for t in range(len(link.rows)):
        for k, ln in enumerate(Q.lanes(SHAPE, SPAN_PL, link.releases, t)):
            if ln.q == 2 and ln.left:
                return t, k
    raise AssertionError("no bucket carries request 1 in")


def test_a_carried_field_count_is_held_by_the_bucket_before():
    """A free addend: bucket t alone accepts its lane's carry-in fields + Δ with its carry-out's + Δ, but
    bucket t − 1 refuses a carry-out of fields + Δ, and the verifier's lanes from an advice of n + Δ aren't the
    cells' either, so either way the link is refused."""
    link = _link()
    t, k = _spanning(link)
    lanes_t, lanes_p = Q.lanes(SHAPE, SPAN_PL, link.releases, t), Q.lanes(SHAPE, SPAN_PL, link.releases, t - 1)
    ln = lanes_t[k]
    kp = next(j for j, p in enumerate(lanes_p) if p.q == ln.q)
    assert (lanes_p[kp].left_out, lanes_p[kp].fields_out) == (ln.left, ln.fields) and ln.fields > 0
    for delta in (1, 5, 100, 1000):
        bumped = ln._replace(fields=ln.fields + delta, fields_out=ln.fields_out + delta)
        assert _ref(link, t, lanes_t[:k] + (bumped,) + lanes_t[k + 1:]) == 1, delta
        out = lanes_p[kp]._replace(fields_out=lanes_p[kp].fields_out + delta)
        assert _ref(link, t - 1, lanes_p[:kp] + (out,) + lanes_p[kp + 1:]) == 0, delta
        forged = (SPAN_PL[0], SPAN_PL[1] + delta, SPAN_PL[2])
        v = _verdicts(link, forged)
        assert v is None or not all(v), delta


def test_prompt_id_values_are_open_here():
    """Open in this unit: the same prompt lengths with any ids are each accepted against their own cells, since it
    reads no Commit (the ingress-ids tie, `io_ids`, holds them to `c00.input/ids`)."""
    for idfn in (lambda i, k: 0, _ids, lambda i, k: (1 << 32) - 1):
        assert _verdicts(_link(SPAN_PL, idfn), SPAN_PL) == [1] * PARAMS.window


def test_the_ordinal_is_read():
    """Each request cell is on its connection's ordinal, which the lane names: a cell moved to another ordinal, or a
    lane read on another, is refused."""
    link = _link()
    t = next(t for t, row in enumerate(link.rows) if any(HEADER.unpack_from(c)[0] == DATA for c in row))
    lanes_ = Q.lanes(SHAPE, SPAN_PL, link.releases, t)
    row = list(link.rows[t])
    i = next(i for i, c in enumerate(row) if HEADER.unpack_from(c)[0] == DATA)
    kind, q, n = HEADER.unpack_from(row[i])
    assert _ref(link, t, lanes_) == 1 and q == lanes_[i // SHAPE.R].q
    for other in (0, q + 1, q - 1):
        moved = row[:i] + [HEADER.pack(kind, other, n) + row[i][HEADER.size:]] + row[i + 1:]
        assert Q.reference(SHAPE, Q.Instance(tuple(moved), lanes_)) == 0, other
    k = i // SHAPE.R
    assert _ref(link, t, lanes_[:k] + (lanes_[k]._replace(q=q + 1),) + lanes_[k + 1:]) == 0


def test_max_tokens_and_user_are_not_checked():
    assert "output_len" not in inspect.getsource(Q)
    names = [n for n, _ in Q._sig_request(SHAPE)[0]]
    assert "max_tokens" not in names and "user" not in names and "output" not in " ".join(names)


@pytest.mark.parametrize("max_tokens", [0, 16, (1 << 32) - 1])
def test_any_max_tokens_is_accepted(max_tokens):
    reqs = [id_request(model="glm", user="u", prompt=[1, 2, 3], max_tokens=max_tokens, frame_bytes=B, offset=0)]
    link = Q.serve(PARAMS, reqs, (0,), (10 ** 6,), buckets=PARAMS.window)
    assert _verdicts(link, (3,)) == [1] * PARAMS.window
