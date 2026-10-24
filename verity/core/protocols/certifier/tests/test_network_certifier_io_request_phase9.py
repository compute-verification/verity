"""Red-team phase 9, target 1 (IoTieRequest, `cursor/io-tie-request-061a`), CPU-only.

IoTieRequest is about to become the ONLY source of N2's advised tie: `advised.ends` takes the prompt-length / order tie
only from a verified IoTieRequest verdict (the lead ejected #3502's prover-supplied tie). So the question is what
IoTieRequest's verdict guarantees ALONE, and what still rests on N2-ADVICE.

Findings (each asserted below), from the reference `io_request.reference` (the Boolean `IoTieRequest` is held to it by
the author's `test_the_unit_is_its_reference`, so attacking the reference is attacking the circuit's spec):

  a. IoTieRequest has NO advice digest, NO Commit/port, and NO carry-out in its public inputs: just
     (root, domain, cells, salts, [window], left, fields, ended, count, received, switches). So the verdict pins
     (certified cells, carry, ended, received) — NOT the advice object. A verdict is valid for ANY advice whose
     `request_view` gives the same (carry, ended, received); binding it to `advised.digest` is N2-ADVICE's job, not the
     circuit's. (`test_no_advice_digest_or_commit_in_public_inputs`)

  b. The carry is a free public input. `carry.left` is pinned by the cell bytes (the END marker's position in a
     continuation frame), but `carry.fields` is an unconstrained ADDEND: for a request that spans a window boundary, a
     single window checked in isolation accepts `carry.fields + Δ` with `ended[0] + Δ`, so the spanning request's
     prompt_len can be inflated by any Δ. The guarantee rests entirely on the cross-window induction — every window
     checked, from window 0, with the carry derived from ONE advice by `request_view`. The full-link check with a
     forged advice IS caught (a later window's `carry.left` no longer matches the real cells).
     (`test_carry_fields_is_a_free_addend_in_isolation`, `test_full_link_induction_closes_the_carry_forgery`)

  c. Live channel under v3: the ingress prompt id VALUES are bound to nothing. `parse_frame` checks only that each
     prompt field is a canonical u32; the id value never enters the verdict, and the unit has no Commit input, so a
     prover may put any prompt content in the certified cells as long as the counts match (~32 bits a prompt field).
     The egress tie v4 binds the Commit's prompt ids to `c00.input/ids`, not to the ingress cells.
     (`test_prompt_id_values_are_unbound`)

  d. The stream-numbering mismatch IO-TIE names (`active.Requests` from 1, io_tie's stream from 0) is moot for
     IoTieRequest: a data cell must be on `CANONICAL_STREAM` (checked), but no per-request stream number is read; the
     ordinal is the verifier's `carry.q + j`, not even a circuit input. It is an egress / CERT-VARLEN concern.
     (`test_stream_is_canonical_only_no_per_request_ordinal`)

  e. `output_len ≤ max_tokens` and user uniqueness are not in it: `max_tokens` is parsed as a field but never compared,
     and IoTieRequest never sees `output_len` or the user. Ranked low — over-run is self-penalizing and bounded by the
     egress output count, and user uniqueness is `certified_io`'s. (`test_max_tokens_and_user_are_not_checked`)
"""

from __future__ import annotations

from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier.active import CANONICAL_STREAM, DATA, HEADER, id_request

B = 512
SHAPE = Q.Shape(B, 4, 2)


def _requests(prompt_lens, idfn=lambda i, k: (7919 * i + 104729 * k) % 151329):
    return [id_request(model="glm", user=f"u{i}", prompt=[idfn(i, k) for k in range(n)], max_tokens=16,
                       temperature=0.7, top_p=0.95, seed=i, frame_bytes=B)
            for i, n in enumerate(prompt_lens)]


def _received(counts):
    T = SHAPE.T
    return tuple(sum(counts[w * T:(w + 1) * T]) for w in range(len(counts) // T))


def _link(prompt_lens, counts, idfn=lambda i, k: (7919 * i + 104729 * k) % 151329):
    windows = Q.windows_of(SHAPE, _requests(prompt_lens, idfn), counts)
    return windows, _received(counts)


def _ref(window_cells, view, window):
    return Q.reference(SHAPE, Q.instance(SHAPE, window_cells, view, window))


# A request that spans the window-0 -> window-1 boundary: request 1 (40 tokens) takes 2 frames, one in each window.
SPAN_PL = (5, 40, 5)
SPAN_COUNTS = (1, 0, 0, 1, 1, 1, 0, 0)


def test_honest_link_accepts():
    windows, received = _link(SPAN_PL, SPAN_COUNTS)
    verdicts = [_ref(windows[w], Q.request_view(SPAN_PL, received, w, B), w) for w in range(len(windows))]
    assert verdicts == [1, 1]


def test_no_advice_digest_or_commit_in_public_inputs():
    """The verdict binds the cells and the view, not the advice: no digest, Commit, port or carry-out input, and not
    even carry.q. Replaying a verdict for another advice with the same view is therefore N2-ADVICE's to prevent."""
    names = [n for n, _ in Q._sig_request(SHAPE)[0]]
    assert names == ["root", "domain", "cells", "salts", "left", "fields", "ended", "count", "received", "switches"]
    assert not [n for n in names if any(s in n.lower() for s in ("advice", "digest", "commit", "ids", "port"))]
    # carry.q is not an argument (arguments uses only carry.left and carry.fields)
    windows, received = _link(SPAN_PL, SPAN_COUNTS)
    view = Q.request_view(SPAN_PL, received, 1, B)
    inst = Q.instance(SHAPE, windows[1], view, 1)
    base = Q.arguments(SHAPE, inst, Q.switches(SHAPE, inst))
    bumped_q = Q.arguments(SHAPE, inst._replace(view=view._replace(carry=view.carry._replace(q=view.carry.q + 7))),
                           Q.switches(SHAPE, inst))
    assert base == bumped_q  # carry.q changes nothing: it only names the verifier's ordinal


def test_carry_fields_is_a_free_addend_in_isolation():
    """A request spanning the boundary: window 1 checked alone accepts carry.fields + Δ with ended[0] + Δ for any Δ,
    while carry.left stays the real continuation. The per-window verdict does not pin the spanning request's
    prompt_len. ONE-LINE FIX: N2-ADVICE must derive every window's carry from one advice via request_view (the
    induction), never accept a per-window verdict in isolation."""
    windows, received = _link(SPAN_PL, SPAN_COUNTS)
    true = Q.request_view(SPAN_PL, received, 1, B)
    assert true.carry.left > 0 and true.ended == (40, 5)  # request 1 is in progress at the window-1 boundary
    assert _ref(windows[1], true, 1) == 1
    for delta in (1, 5, 100, 1000):
        forged = Q.RequestView(Q.Carry(true.carry.q, true.carry.left, true.carry.fields + delta),
                               (true.ended[0] + delta, true.ended[1]), true.received)
        assert _ref(windows[1], forged, 1) == 1, delta  # accepted: ended[0] is carry.fields + the window's fields


def test_full_link_induction_closes_the_carry_forgery():
    """The same inflation, routed through a single forged advice and checked on every window, is caught: window 1's
    carry.left (derived from the forged prompt_len) no longer matches the real cells. So the guarantee holds only when
    every window is checked, from window 0, against one advice."""
    windows, received = _link(SPAN_PL, SPAN_COUNTS)
    forged_pl = (5, 45, 5)  # request 1 inflated 40 -> 45 (still two frames, same ends and received)
    assert Q.ends(forged_pl, B) == Q.ends(SPAN_PL, B)
    verdicts = [_ref(windows[w], Q.request_view(forged_pl, received, w, B), w) for w in range(len(windows))]
    assert verdicts == [1, 0]  # window 0 passes, window 1 refuses -> the link is refused


def test_prompt_id_values_are_unbound():
    """The id VALUES never enter the verdict: the same prompt lengths with any ids (zeros, mixed, the max u32) are
    each accepted against their own cells. IoTieRequest has no Commit, so the ingress prompt content is free given the
    counts. ONE-LINE FIX (owner: PROVE-PATH's compile unit / IO-TIE): tie the ingress prompt ids to c00.input/ids."""
    pl, counts = (5, 3, 4), (1, 1, 1, 0, 0, 0, 0, 0)
    received = _received(counts)
    for idfn in (lambda i, k: 0, lambda i, k: (7919 * i + 104729 * k) % 151329, lambda i, k: (1 << 32) - 1):
        windows = Q.windows_of(SHAPE, _requests(pl, idfn), counts)
        assert _ref(windows[0], Q.request_view(pl, received, 0, B), 0) == 1


def test_stream_is_canonical_only_no_per_request_ordinal():
    """A data cell must be on CANONICAL_STREAM (checked), but no per-request stream number is read: the from-0 / from-1
    numbering IO-TIE names does not reach IoTieRequest's verdict."""
    pl, counts = (5, 3, 4), (1, 1, 1, 0, 0, 0, 0, 0)
    windows = Q.windows_of(SHAPE, _requests(pl, lambda i, k: 0), counts)
    cells = list(windows[0])
    for idx, c in enumerate(cells):
        kind, stream, n = HEADER.unpack(c[:HEADER.size])
        if kind == DATA and stream == CANONICAL_STREAM:
            cells[idx] = HEADER.pack(DATA, 1, n) + c[HEADER.size:]  # a data cell on stream 1
            break
    assert Q.cut(SHAPE, cells, Q.Carry(0, 0, 0), 0) is None  # refused: the stream must be canonical
    # and no "stream"/"ordinal"/"q" input exists at all
    assert not [n for n, _ in Q._sig_request(SHAPE)[0] if n in ("stream", "ordinal", "q")]


def test_max_tokens_and_user_are_not_checked():
    """max_tokens is a parsed field but never compared to output_len (which the unit never sees), and the user is never
    checked for uniqueness. Both are out of IoTieRequest by design. Ranked low: an over-run is self-penalizing and
    bounded by the egress output count; user uniqueness is certified_io's."""
    import inspect

    src = inspect.getsource(Q)
    # the unit takes neither output_len nor a per-request user; it checks the prompt FIELD COUNT, not max_tokens
    assert "output_len" not in src
    names = [n for n, _ in Q._sig_request(SHAPE)[0]]
    assert "max_tokens" not in names and "user" not in names and "output" not in " ".join(names)
