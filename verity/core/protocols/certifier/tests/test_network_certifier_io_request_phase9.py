"""REDTEAM's phase 9, target 1 (`cursor/redteam-phase9-iotiereq-061a` @8d19db0a2b), IoTieRequest's cases as
regressions: what a window's verdict holds alone, and what rests on the statements chaining from window 0.

  a. The public inputs carry the carry-out (`left_out`, `fields_out`) beside the carry-in, and still no advice digest
     or Commit: `advised.IoTieVerdict` (N2-ADVICE) keys the verdicts to the advice's digest, built only from proofs
     verified on the verifier's own statements (`io_request.statements`). The carries' ordinals are no input.
  b. Closed: a carried prompt-field count is no free addend. Checked alone, window 1 still accepts its carry-in's
     fields + Δ with its first ended request's + Δ, but window 0's verdict holds its carry-out's fields, which window
     1's carry-in must equal (`io_request.chained`), so a link's verdicts accept no Δ.
  c. Open: the prompt id values are tied to nothing (PROTOCOL.md, "The ingress tie", the ingress-ids tie to the
     Commit's `c00.input/ids`, not yet built).
  d. A request frame is on `CANONICAL_STREAM`, and no per-request stream number is read: the ordinal is the
     verifier's, carry_in.q + j.
  e. The unit reads neither the response's length nor the user: `max_tokens` is parsed as a field and not compared.
"""
from __future__ import annotations

import hashlib
import inspect

from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier.active import CANONICAL_STREAM, DATA, HEADER, id_request

B = 512
SHAPE = Q.Shape(B, 4, 2)
#: a request that spans the boundary of windows 0 and 1: request 1 (40 tokens) takes 2 frames, one in each
SPAN_PL, SPAN_COUNTS = (5, 40, 5), (1, 0, 0, 1, 1, 1, 0, 0)


def _ids(i, k):
    return (7919 * i + 104729 * k) % 151329


def _requests(prompt_lens, idfn=_ids):
    return [id_request(model="glm", user=f"u{i}", prompt=[idfn(i, k) for k in range(n)], max_tokens=16,
                       temperature=0.7, top_p=0.95, seed=i, frame_bytes=B)
            for i, n in enumerate(prompt_lens)]


def _received(counts):
    return tuple(sum(counts[w * SHAPE.T:(w + 1) * SHAPE.T]) for w in range(len(counts) // SHAPE.T))


def _link(prompt_lens, counts, idfn=_ids):
    return Q.windows_of(SHAPE, _requests(prompt_lens, idfn), counts), _received(counts)


def _ref(cells, view, window):
    return Q.reference(SHAPE, Q.instance(SHAPE, cells, view, window))


def _record(windows):
    insts = [Q.instance(SHAPE, c, Q.RequestView(Q.Carry(0, 0, 0), (), 0, Q.Carry(0, 0, 0)), w)
             for w, c in enumerate(windows)]
    return [Q.Registration("ingress-0", hashlib.sha512(b"ingress-0/%d" % w + i.root).digest(), i.root, i.domain)
            for w, i in enumerate(insts)]


def test_honest_link_accepts():
    windows, received = _link(SPAN_PL, SPAN_COUNTS)
    stmts = Q.statements(SHAPE, SPAN_PL, received, _record(windows))
    assert Q.chained(stmts) and [_ref(windows[s.window], s.view, s.window) for s in stmts] == [1, 1]


def test_the_public_inputs_hold_the_carry_out_and_no_ordinal():
    names = [n for n, _ in Q._sig_request(SHAPE)[0]]
    assert names == ["root", "domain", "cells", "salts", "left", "fields", "ended", "count", "received", "left_out",
                     "fields_out", "switches"]
    assert not [n for n in names if any(s in n.lower() for s in ("advice", "digest", "commit", "ids", "port"))]
    windows, received = _link(SPAN_PL, SPAN_COUNTS)
    view = Q.request_view(SPAN_PL, received, 1, B)
    inst = Q.instance(SHAPE, windows[1], view, 1)
    bits = Q.switches(SHAPE, inst)
    bumped = view._replace(carry_in=view.carry_in._replace(q=view.carry_in.q + 7),
                           carry_out=view.carry_out._replace(q=view.carry_out.q + 7))
    assert Q.arguments(SHAPE, inst, bits) == Q.arguments(SHAPE, inst._replace(view=bumped), bits)


def test_a_carried_field_count_is_held_by_the_previous_windows_carry_out():
    """REDTEAM's free addend: window 1 alone accepts carry_in.fields + Δ with ended[0] + Δ, but window 0 refuses a
    carry-out of fields + Δ, and window 1's carry-in must be window 0's carry-out, so either way the link is
    refused."""
    windows, received = _link(SPAN_PL, SPAN_COUNTS)
    stmts = Q.statements(SHAPE, SPAN_PL, received, _record(windows))
    v0, v1 = stmts[0].view, stmts[1].view
    assert v1.carry_in.left > 0 and v1.ended == (40, 5) and v0.carry_out == v1.carry_in
    for delta in (1, 5, 100, 1000):
        in1 = v1._replace(carry_in=v1.carry_in._replace(fields=v1.carry_in.fields + delta),
                          ended=(v1.ended[0] + delta, v1.ended[1]))
        out0 = v0._replace(carry_out=in1.carry_in)
        assert _ref(windows[1], in1, 1) == 1, delta
        assert _ref(windows[0], out0, 0) == 0, delta
        assert not Q.chained([stmts[0], stmts[1]._replace(view=in1)])
        assert Q.chained([stmts[0]._replace(view=out0), stmts[1]._replace(view=in1)])


def test_a_forged_advice_is_refused_where_its_carry_first_moves():
    """The inflation 40 → 45 through one forged advice keeps every request's frames, and now window 0 refuses it:
    its carry-out's bytes left are the forged request's, not the cells'."""
    windows, received = _link(SPAN_PL, SPAN_COUNTS)
    forged = (5, 45, 5)
    assert Q.ends(forged, B) == Q.ends(SPAN_PL, B)
    stmts = Q.statements(SHAPE, forged, received, _record(windows))
    assert Q.chained(stmts) and [_ref(windows[s.window], s.view, s.window) for s in stmts] == [0, 0]


def test_prompt_id_values_are_open_until_the_ids_tie():
    """Open: the same prompt lengths with any ids are each accepted against their own cells, since the unit reads no
    Commit (PROTOCOL.md, "The ingress tie": the ingress-ids tie to `c00.input/ids`, not yet built). Once it is, this
    test holds the ids to the Commit's."""
    pl, counts = (5, 3, 4), (1, 1, 1, 0, 0, 0, 0, 0)
    received = _received(counts)
    for idfn in (lambda i, k: 0, _ids, lambda i, k: (1 << 32) - 1):
        windows = Q.windows_of(SHAPE, _requests(pl, idfn), counts)
        assert _ref(windows[0], Q.request_view(pl, received, 0, B), 0) == 1


def test_stream_is_canonical_only_no_per_request_ordinal():
    pl, counts = (5, 3, 4), (1, 1, 1, 0, 0, 0, 0, 0)
    cells = list(Q.windows_of(SHAPE, _requests(pl, lambda i, k: 0), counts)[0])
    idx = next(i for i, c in enumerate(cells) if HEADER.unpack(c[:HEADER.size])[:2] == (DATA, CANONICAL_STREAM))
    cells[idx] = HEADER.pack(DATA, 1, HEADER.unpack(cells[idx][:HEADER.size])[2]) + cells[idx][HEADER.size:]
    assert Q.cut(SHAPE, cells, Q.Carry(0, 0, 0), 0) is None
    assert not [n for n, _ in Q._sig_request(SHAPE)[0] if n in ("stream", "ordinal", "q")]


def test_max_tokens_and_user_are_not_checked():
    assert "output_len" not in inspect.getsource(Q)
    names = [n for n, _ in Q._sig_request(SHAPE)[0]]
    assert "max_tokens" not in names and "user" not in names and "output" not in " ".join(names)
