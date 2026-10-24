"""Adversarial cases against the active warden (`warden`), for the v3 red team (REDTEAM-ADVICE of network-accounting).

The warden's own tests show the honest path: the schedule, frame size and timing carry nothing, and nothing of the
server (extra tokens, finish words, raw bytes) reaches a client. These show where a *malicious server* still gets bits
out beyond the stated bound, and confirm which channels the schedule closes.

Each channel the plan note's v3 bound names is one group here:

- **the fail channel.** The server picks the slot a request fails at (the bound charges log2(cells+1) for this) and,
  at that slot, the *code* late vs malformed, which the client sees as two different error types. The bound charges
  nothing for the code, so it is an extra bit a failed connection. The bits reach the client in real time, in a
  window the audit then rejects, so the accepted-runs bound never sees them.
- **the content channel.** The client gets exactly the token ids the server streams; the warden renders every other
  byte from the ids and public rules. So nothing but the all-frames id tie (and batch invariance) pins what the
  client reads: two different id streams of the same length and finish give the client different text in the same
  frames, and the audit accepts both.
- **closed channels.** A failed request still holds its lane to its scheduled end, so the server can't move a later
  request's admission (queue pressure is the clients', not the server's). Every connection is the warden's own
  HTTP/1.1 ``connection: close``.
"""

from __future__ import annotations

import asyncio

import pytest

from verity.core.protocols.certifier.warden import FAIL, TOKEN, Token, audit, read_record_frame
from verity.core.protocols.certifier.tests.test_network_certifier_warden import (
    A, B_, EOS, FB, SCHEDULE, Server, body, chunked, events, http, run, streamed_text)

MAX = 8  # a request's max_tokens here, so S = 8 token slots


def _single(ids, max_tokens=MAX, stream=True):
    """Serve one request whose server yields `ids`, over one window, and return (client, warden, audit verdict)."""
    warden, (client,), _ = run({0: ids}, {0: [http(body(stream=stream, max_tokens=max_tokens))]}, windows=1)
    verdict = audit(SCHEDULE, [r for r in warden.records if r.link == "requests"],
                    [r for r in warden.records if r.link == "responses"])
    return client, warden, verdict


# --- the fail channel ------------------------------------------------------------------------------------------------


def test_the_server_chooses_the_fail_slot_which_the_client_reads_in_a_window_the_audit_rejects():
    """Giving the warden j tokens (0 ≤ j < S) then ending late-fails the request at slot j; j = S is the honest run.
    The S + 1 outcomes are all distinct on the wire the client gets, yet every one is the same B-byte frames in the
    same buckets, and the audit rejects every failing window. So the fail slot is a client-visible channel of
    log2(S + 1) bits a connection, delivered before (and independent of) the audit's verdict."""
    S = SCHEDULE.slots(MAX)
    seen, honest_frames = {}, None
    for j in range(S + 1):
        client, _, verdict = _single([A] * j)  # j tokens, then the stream ends: late fail at slot j (j = S: honest)
        seen[j] = client.data
        if honest_frames is None:
            honest_frames = client.frames()
        assert client.frames() == honest_frames, j  # the layout is the same whatever the server does
        assert verdict["accepted"] == (j == S)  # every fail makes the window not accepted
    assert len(set(seen.values())) == S + 1  # all S + 1 outcomes are distinct bytes to the client
    # the slot is literally in the client's error event, which is the j-th event (after j token events)
    assert "token 3" in str(events(seen[3])[3]["error"]["message"]) and events(seen[3])[3]["error"]["code"] == 504


def test_at_one_fail_slot_the_server_still_picks_the_code_an_extra_bit_the_bound_does_not_charge():
    """At the same slot j the server fails late (withholding token j) or malformed (a token out of position), and the
    client gets two different error types (504 WardenLate vs 502 WardenUpstreamMalformed). The v3 bound charges
    log2(cells + 1) a failed connection, for the slot alone, so this code is an uncharged bit every fail. Cheapest
    fix: one client-visible fail error (keep the reason in the record only)."""
    j = 3
    late_c, late_w, late_v = _single([A] * j)  # withhold token j
    bad_c, bad_w, bad_v = _single([A] * j + [Token(A, j + 1, None)])  # a token out of position: malformed at slot j
    assert events(late_c.data)[j]["error"]["type"] == "WardenLate"
    assert events(bad_c.data)[j]["error"]["type"] == "WardenUpstreamMalformed"
    assert late_c.data != bad_c.data and late_c.frames() == bad_c.frames()  # distinct content, identical frames
    assert not late_v["accepted"] and not bad_v["accepted"]

    def fail_codes(w):  # the record keeps the two apart too, so the choice is the server's, not noise
        return [code for r in w.records if r.link == "responses" for row in r.grid for f in row if f is not None
                for kind, _, _, _, code in [read_record_frame(f)] if kind == FAIL]

    assert fail_codes(late_w) != fail_codes(bad_w)


def test_the_fail_pattern_across_concurrent_connections_is_an_independent_choice_per_connection():
    """Two lanes carry two requests admitted together. Failing the first, or the second, or neither, gives distinct
    combined output, and each connection the server runs honestly is byte-identical to its all-honest self. So the
    channel is one independent fail choice a concurrent connection, not one a window."""
    full = {0: [A, B_, A, EOS], 1: [A, B_, A, EOS]}
    arrivals = {0: [http(body(max_tokens=4)), http(body(max_tokens=4))]}
    _, honest_c, _ = run(full, arrivals, windows=1)
    _, f0_c, _ = run({0: [A, B_], 1: [A, B_, A, EOS]}, arrivals, windows=1)  # fail connection 0 at slot 2
    _, f1_c, _ = run({0: [A, B_, A, EOS], 1: [A, B_]}, arrivals, windows=1)  # fail connection 1 at slot 2
    assert f0_c[1].data == honest_c[1].data  # connection 1 ran honestly: unchanged by connection 0's fail
    assert f1_c[0].data == honest_c[0].data  # and symmetrically
    assert len({(c[0].data, c[1].data) for c in (honest_c, f0_c, f1_c)}) == 3  # three distinct combined outputs


# --- the content channel ---------------------------------------------------------------------------------------------


def test_the_client_reads_whatever_ids_the_server_streams_pinned_only_by_the_id_tie():
    """Two id streams of the same length finishing at EOS give the client different text in exactly the same frames,
    and the audit accepts both (it checks the ids render to their finish, not which ids). So the content is bound only
    by the all-frames id tie: a tie that samples frames leaves the un-sampled ones free, which is why the egress tie is
    selection All. If the kernels aren't batch invariant the server chooses these ids by batching, so the bound's
    content term rests on batch-invariant kernels (a named assumption)."""
    a_client, _, a_v = _single([A, B_, A, EOS], max_tokens=4)
    b_client, _, b_v = _single([B_, A, B_, EOS], max_tokens=4)
    assert streamed_text(a_client.data)[0] == "aba" and streamed_text(b_client.data)[0] == "bab"
    assert a_client.frames() == b_client.frames()  # identical frames
    assert a_v["accepted"] and b_v["accepted"]  # the audit accepts either content
    a_ids = [tok for r in a_v["requests"] for tok in r["ids"]]
    b_ids = [tok for r in b_v["requests"] for tok in r["ids"]]
    assert a_ids != b_ids  # only the token-id frames tell them apart, so the tie must cover every frame


# --- closed channels -------------------------------------------------------------------------------------------------


def test_a_failed_request_still_holds_its_lane_so_the_server_cannot_move_a_later_admission():
    """Two lanes, a queue, three requests at once: two take the lanes and the third waits. Whether the first request
    runs full or fails at slot 1, it holds its lane to the same scheduled end, so the queued request is admitted at the
    same bucket. The server can't use fail-early to free a lane and shift admission: queue pressure is the clients'."""
    # two lanes fill at bucket 0, so the third (arriving at bucket 1) waits for a lane to free
    arrivals = {0: [http(body(max_tokens=4)), http(body(max_tokens=4))], 1: [http(body(max_tokens=4))]}
    full, _, _ = run({q: [A, B_, A, EOS] for q in range(3)}, arrivals, windows=1)
    early, _, _ = run({0: [A], 1: [A, B_, A, EOS], 2: [A, B_, A, EOS]}, arrivals, windows=1)
    admit_full = {e["ordinal"]: e["admitted"] for e in full.log}
    admit_early = {e["ordinal"]: e["admitted"] for e in early.log}
    assert admit_full[2] == admit_early[2]  # the queued request's admission is unchanged by the first's fail
    assert early.log[0]["failed"] == "late"


def test_every_connection_is_the_wardens_own_http_1_1_close_so_no_server_lifecycle_reaches_the_client():
    """The response head the client gets is the warden's HTTP/1.1 200 with ``connection: close`` and a chunked body it
    ends itself; the server's transport never shows through."""
    client, _, _ = _single([A, EOS], max_tokens=4)
    head, _ = chunked(client.data)
    assert head.startswith(b"HTTP/1.1 200 OK")
    assert b"connection: close" in head.lower()
    assert b"transfer-encoding: chunked" in head.lower()
