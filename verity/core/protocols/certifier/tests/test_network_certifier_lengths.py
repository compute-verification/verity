"""The network certifier with lengths rows (`WardenParams.lengths`, `warden`'s "The lengths rows"), driven tick by tick
on `test_network_certifier_warden`'s harness. Each test holds a schedule with lengths rows to one rule of that section
and the same schedule without them to what it did before they were a parameter."""

from __future__ import annotations

import dataclasses

import pytest

from verity.ml.protocols.certifier import ingress_scan as IS
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import DATA, HEADER, PAD, id_request
from verity.core.protocols.certifier.tests.test_network_certifier_warden import EOF, reasons, records, run
from verity.core.protocols.certifier.warden import (END, FAIL, INTEGER_FIELDS, OPTIONAL_FIELDS, Released, WardenParams,
                                                    audit, cells, read_reason, reason_frame)

#: a schedule whose cells hold an id request's head, and the same with lengths rows
PLAIN = WardenParams(cell_bytes=512, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                     ingress_buckets=3, egress_cells=9, every=2)
LEN = dataclasses.replace(PLAIN, lengths=1)
B, P, R, T, D, E = LEN.cell_bytes, LEN.payload, LEN.slots, LEN.window, LEN.delay, LEN.every
#: the default schedule's digest, PLAIN's, and PLAIN's under a response cap of 4, as each was before the lengths rows'
#: form was a parameter
DEFAULT_DIGEST = ("9a727d16f6c1f351d316465c101269c2a4666cbcdf1dbaaef01579fd7e72315a"
                  "ae2be1451412c6328d9f6688f534fb8bbe5416bbaeb5cfc0cae48597c5fd0d50")
PLAIN_DIGEST = ("b064f2c3e90f5059b7393bcbca02b7b94d514041946423a253fa1cca6de11ee4"
                "1898e2e0368e1a03d9e5f8a910fc91ac2f8154924ccd7f669f8b0f4d99964808")
CAPPED_DIGEST = ("6dc8692fb13c37e45ce4d4229a8377931b434ff5c72fd243aeb2d0596dc0226e"
                 "b987b7a37d81a0baeed20ac5eb52ee614b991797a45e85c3fba8916f34361c22")


# --- the mode ----------------------------------------------------------------------------------------------------


def test_the_lengths_rows_form_is_a_parameter_no_other_schedule_hashes():
    """A schedule without lengths rows hashes what it hashed before they were a parameter (the integer fields, the
    egress form, the kinds and the reasons), so its digest, and every head and certificate that binds it, is unchanged;
    the form enters the digest of a schedule that has it, and is no integer field of the schedule the period check
    reads."""
    assert WardenParams().lengths is None and PLAIN.lengths is None
    assert WardenParams().digest.hex() == DEFAULT_DIGEST and PLAIN.digest.hex() == PLAIN_DIGEST
    assert LEN.digest != PLAIN.digest and dataclasses.replace(LEN, lengths=None) == PLAIN
    assert OPTIONAL_FIELDS == ("max_tokens_cap", "lengths") and not set(OPTIONAL_FIELDS) & set(INTEGER_FIELDS)
    assert LEN.reasons == PLAIN.reasons
    capped = dataclasses.replace(PLAIN, max_tokens_cap=4)
    assert capped.digest.hex() == CAPPED_DIGEST
    assert len({PLAIN.digest, LEN.digest, capped.digest, dataclasses.replace(capped, lengths=1).digest}) == 4


def test_a_schedule_with_lengths_rows_is_one_whose_releases_fit_them():
    """The form is 1 or unset; with it the egress form is id, ingress ends by the delay, a connection ends within a
    window of its admission, its egress cells count in a u16 and a cell holds an id request's head and first prompt
    field. K, a row's slots, is the most releases a slot stride can hold."""
    for bad in (0, 2, True, "1", 1.0):
        with pytest.raises(ValueError, match="lengths is 1"):
            dataclasses.replace(PLAIN, lengths=bad)
    for change, why in (({"egress": "bytes"}, "egress form is id"), ({"ingress_buckets": D + 1}, "at most the delay"),
                        ({"window": 8, "every": 1}, "ends within a window"), ({"delay": 13}, "ends within a window"),
                        ({"egress_cells": (1 << 16) + 1, "window": 1 << 16}, "count in a u16"),
                        ({"cell_bytes": 256}, "no room for an id request|doesn't fit its first cell")):
        dataclasses.replace(PLAIN, **change)
        with pytest.raises(ValueError, match=why):
            dataclasses.replace(LEN, **change)
    assert dataclasses.replace(LEN, delay=12, window=16, egress_cells=9).lengths == 1
    assert LEN.stride_releases == 2 == LEN.lanes
    # every = 1: a bucket releases at most a request a lane; long strides: a lane's releases are D + 2 − ingress_buckets
    # apart, and a connection holds its lane D + 1 buckets
    for change, K in (({"every": 1}, 2), ({"every": 4}, 2), ({"every": 8}, 4), ({"every": 16}, 8),
                      ({"every": 16, "ingress_buckets": 1}, 6), ({"every": 8, "lanes": 5, "ingress_buckets": 5}, 15)):
        assert dataclasses.replace(LEN, **change).stride_releases == K, change


# --- a client's failure ------------------------------------------------------------------------------------------


def id_req(i: int, prompt_len: int, **fields) -> bytes:
    kw = {"model": "m", "user": f"user-{i}", "max_tokens": 8, "seed": i, "temperature": 0.7, "top_p": 0.95} | fields
    return id_request(prompt=[1000 * i + k for k in range(prompt_len)], frame_bytes=B, offset=0, **kw)


def response(ids) -> bytes:
    return b"".join(IT.id_event(x, P) for x in ids) + IT.TRAILER


def answer(ids) -> list:
    """A server's script: the response a bucket after the request's, then its end."""
    return [(1, response(ids)), (2, EOF)]


def request_cells(warden, ordinal: int) -> list[tuple[int, int, bytes]]:
    """(bucket, kind, cell) of every ingress cell on `ordinal`, in order."""
    return [(r.window * T + b, f[0], f) for r in records(warden)[0] for b, row in enumerate(r.grid) for f in row
            if f is not None and HEADER.unpack_from(f)[1] == ordinal]


def test_a_connection_whose_client_failed_is_never_released():
    """A client whose last cell is cut short has failed (malformed) with its stream ended. Without lengths rows the
    network certifier still cuts what it holds, releases it to the server with its place in the release order, and
    fails the connection in the same bucket. With them it cuts nothing more: no `END`, no place in the order, nothing
    for the server, and the connection after it takes the place."""
    whole, short = id_req(1, 3), cells(id_req(2, 40), B)[:-10]
    assert len(cells(id_req(2, 40), B)) == 2 * B
    arrivals = {0: [short], 1: [cells(whole, B)]}
    for params, released in ((PLAIN, True), (LEN, False)):
        warden, _, server = run([[], answer((7, 8))], arrivals, params=params, windows=2)
        log = {e["ordinal"]: e for e in warden.log}
        assert (log[1]["failed"], log[1]["failed_at"]) == ("malformed", 0)
        kinds = [k for _, k, _ in request_cells(warden, 1)]
        assert kinds == [PAD, PAD, END, PAD] if released else set(kinds) == {PAD}
        assert (log[1]["release"], log[2]["release"]) == ((0, 1) if released else (None, 0))
        assert bool(server.conns[0].received) == released and warden.requests_released == 1 + released
        assert (log[2]["failed"], log[2]["egress_cells"]) == (None, 3)
        assert [(o, why) for _, o, why in reasons(warden)] == [(1, "malformed")]


# --- the lane ----------------------------------------------------------------------------------------------------


#: the scan that serves the schedule (`ingress_scan.Class.serves`): the ingress proof block's reference over its rows
CLS = IS.Class(B=B, r=2, lanes=2, C=4, EVERY=2, WITHIN=3, D=5, KO=8, KI=100)


def scanned(warden, outputs: dict[int, tuple[int, ...]]):
    """The ingress proof blocks' reference over every bucket of the run's signed ingress rows (`ingress_scan.scan`,
    the Programs' outputs `outputs` by ordinal): None where a bucket's is refused."""
    CLS.serves(warden.params)
    rows = [row for r in records(warden)[0] for row in WR.window_rows(r, warden.params)]
    return IS.scan(CLS, rows, outputs)


def failed_clients() -> dict[str, tuple]:
    """Clients that fail before any release, each as (its bytes, the schedule's other parameters, its reason, the
    bucket it fails in without lengths rows, its request cells' kinds up to there)."""
    two = cells(id_req(2, 40), B)
    assert len(two) == 2 * B
    return {"a cell cut short": (two[:-10], {}, "malformed", 1, [PAD, PAD, END, PAD]),
            "an ingress that never ends": ((two[:B], False), {}, "ingress", 2, [PAD, PAD, DATA, PAD, PAD, PAD]),
            "a request past the cap": (cells(id_req(2, 3, max_tokens=9), B), {"max_tokens_cap": 8}, "cap", 1, [PAD] * 4)}


@pytest.mark.parametrize("name", sorted(failed_clients()))
def test_a_connection_that_is_never_released_holds_its_lane_through_its_delay(name):
    """Both lanes taken at bucket 0, one by a client that fails, and a third connection waiting from bucket 2. With
    lengths rows the failed connection is padding both ways until its first egress slot, bucket s + D, and fails there,
    whenever its client failed and why: its lane frees with the slot the ingress proof block's scan holds for it
    (`ingress_scan.held`, through admitted + D), the third connection is admitted the bucket after, and the scan takes
    every bucket of the run. Without lengths rows the connection ends where it fails and the third takes its lane at
    once, while the scan still holds the slot: it has none for the third, and refuses the run."""
    data, other, why, at, kinds = failed_clients()[name]
    out, first, third = (7, 8), id_req(1, 3), id_req(3, 3)
    arrivals = {0: [cells(first, B), data], 2: [cells(third, B)]}
    answers = [answer(out), [], answer(out)]

    params = dataclasses.replace(LEN, **other)
    warden, clients, server = run(answers, arrivals, params=params, windows=3)
    log = {e["ordinal"]: e for e in warden.log}
    assert (log[2]["release"], log[2]["failed"], log[2]["failed_at"], log[2]["ended"]) == (None, why, 0, D)
    mine = request_cells(warden, 2)
    assert [t for t, _, _ in mine] == [t for t in range(D + 1) for _ in range(R)]
    assert END not in [k for _, k, _ in mine] and [k for _, k, _ in mine][2 * R:] == [PAD] * ((D - 1) * R)
    assert [(t, k) for t, k, _ in clients[1].cells()] == \
        [(t, PAD) for t in range(D) for _ in range(R)] + [(D, FAIL)] + [(D, PAD)] * (R - 1)
    assert reasons(warden) == [(D, 2, why)] and not server.conns[1].received
    assert (log[3]["admitted"], log[3]["release"], log[3]["failed"]) == (D + 1, 1, None)
    assert (log[1]["ended"], log[1]["failed"]) == (D + 1, None)
    assert scanned(warden, {1: out, 3: out}) is not None

    without = dataclasses.replace(PLAIN, **other)
    plain, plain_clients, _ = run(answers, arrivals, params=without, windows=3)
    log = {e["ordinal"]: e for e in plain.log}
    assert (log[2]["failed"], log[2]["ended"]) == (why, at)
    assert [k for _, k, _ in request_cells(plain, 2)] == kinds
    assert [(t, k) for t, k, _ in plain_clients[1].cells()][-R:] == [(at, FAIL)] + [(at, PAD)] * (R - 1)
    assert (log[3]["admitted"], log[3]["failed"]) == (at + 1, None)
    assert scanned(plain, {1: out, 3: out}) is None
    if END not in kinds:
        # the slot, and nothing else of the run: the third arriving once the scan's slot is free, the scan takes it
        later, _, _ = run(answers, {0: arrivals[0], D + 1: arrivals[2]}, params=without, windows=3)
        assert {e["ordinal"]: e["admitted"] for e in later.log}[3] == D + 1
        assert scanned(later, {1: out, 3: out}) is not None


# --- the audit ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(failed_clients()))
def test_a_connection_that_is_never_released_fails_no_window_of_the_audit(name):
    """With lengths rows the read of the records in the clear accepts a window in which a client failed before its
    request's release: no server saw the request, and the connection's entry still names its reason. The same records
    read as a schedule's without lengths rows fail their window, as every `FAIL` does there, and the records of the run
    without lengths rows, whose `FAIL` is before s + D, are not this schedule's."""
    data, other, why, _, _ = failed_clients()[name]
    params, out = dataclasses.replace(LEN, **other), (7, 8)
    arrivals, answers = {0: [cells(id_req(1, 3), B), data]}, [answer(out), []]
    warden, _, _ = run(answers, arrivals, params=params, windows=2)
    verdict = audit(params, *records(warden))
    assert verdict["accepted"] and not any(w["problems"] for w in verdict["windows"])
    entry = verdict["connections"][1]
    assert (entry["ordinal"], entry["ingress_end"], entry["failed"], entry["failed_at"]) == (2, None, why, 0)
    without = dataclasses.replace(params, lengths=None)
    assert audit(without, *records(warden))["windows"][0]["problems"] == \
        [f"connection 2 failed closed ({why}) at egress cell 0"]
    plain, _, _ = run(answers, arrivals, params=without, windows=2)
    assert "connection 2: its egress isn't padding to s + D, data in every slot, one END or FAIL, padding" in \
        audit(params, *records(plain))["windows"][0]["problems"]


def test_the_audit_holds_each_failure_to_the_side_that_can_cause_it():
    """With lengths rows a released connection's `FAIL` is the server's and still fails its window; and records in
    which a client's reason goes with a released request, or a server's with one never released, are no records of
    this schedule."""
    arrivals = {0: [cells(id_req(1, 3), B), cells(id_req(2, 40), B)[:-10]], 8: [cells(id_req(3, 3), B)]}
    warden, _, _ = run([answer((7, 8)), [], []], arrivals, params=LEN, windows=2)
    ingress, egress, fails = records(warden)
    verdict = audit(LEN, ingress, egress, fails)
    assert [(c["ordinal"], c["failed"]) for c in verdict["connections"]] == [(1, None), (2, "malformed"), (3, "late")]
    assert [w["problems"] for w in verdict["windows"]] == [["connection 3 failed closed (late) at egress cell 0"], []]

    def swapped(ordinal: int, why: str):
        """The `fails` records with `ordinal`'s reason replaced."""
        grid = tuple(tuple(reason_frame(ordinal, why, B) if f is not None and read_reason(f, B)[0] == ordinal else f
                           for f in row) for row in fails[0].grid)
        return [dataclasses.replace(fails[0], grid=grid), *fails[1:]]

    assert audit(LEN, ingress, egress, swapped(3, "malformed"))["windows"][0]["problems"] == [
        "connection 3: it failed (malformed) and its request was released",
        "connection 3 failed closed (malformed) at egress cell 0"]
    assert audit(LEN, ingress, egress, swapped(2, "late"))["windows"][0]["problems"] == [
        "connection 2: it failed (late) and its request was never released",
        "connection 2 failed closed (late) at egress cell 0", "connection 3 failed closed (late) at egress cell 0"]


# --- the releases ------------------------------------------------------------------------------------------------


def traffic():
    """Five requests over two windows: two released in one slot stride (as many as its slots), one later in the window,
    then, in the next window's one slot stride, one whose server never answers and one whose server stops after two
    outputs."""
    reqs = [id_req(1, 3), id_req(2, 40), id_req(3, 100), id_req(4, 5), id_req(5, 7)]
    arrivals = {0: [cells(reqs[0], B), cells(reqs[1], B)], 9: [cells(reqs[2], B)], 17: [cells(reqs[3], B)],
                18: [cells(reqs[4], B)]}
    answers = [answer((11, 22, 33)), answer(range(8)), answer((5, 6)), [], [(1, response((1, 2))[:2 * P])]]
    return reqs, arrivals, answers


def test_each_windows_releases_are_handed_by_slot_stride_once_the_next_windows_records_are():
    """Window w's releases, by the slot stride of each one's `END` and in release order: N, the request's bytes, the
    response's `DATA` cells and whether the connection ended in a `FAIL`, as the signed records say. They are handed
    after window w + 1's records, where every one of them has ended; the run's last window's never are."""
    reqs, arrivals, answers = traffic()
    order: list = []
    warden, _, _ = run(answers, arrivals, params=LEN, windows=3,
                       publish=lambda record, grid: order.append((record.link, record.window)),
                       lengths=lambda window, strides: order.append(("lengths", window, strides)))
    assert [e[:2] for e in order] == [
        ("requests", 0), ("responses", 0), ("fails", 0), ("requests", 1), ("responses", 1), ("fails", 1),
        ("lengths", 0), ("requests", 2), ("responses", 2), ("fails", 2), ("lengths", 1)]
    handed = {e[1]: e[2] for e in order if e[0] == "lengths"}
    for strides in handed.values():
        assert len(strides) == T // E and all(len(stride) <= LEN.stride_releases for stride in strides)
    got = {(w, j): stride for w, strides in handed.items() for j, stride in enumerate(strides) if stride}
    assert got == {(0, 0): (Released(0, len(reqs[0]), 3, False), Released(1, len(reqs[1]), 8, False)),
                   (0, 5): (Released(2, len(reqs[2]), 2, False),),
                   (1, 1): (Released(3, len(reqs[3]), 0, True), Released(4, len(reqs[4]), 2, True))}
    # as the records say: a release is an END in the ingress record, N their order, n the DATA cells of its egress
    plain, _, _ = run(answers, arrivals, params=PLAIN, windows=3)
    verdict = audit(PLAIN, *records(plain))
    by_end = sorted(verdict["connections"], key=lambda c: (c["ingress_end"], c["ordinal"]))
    want: dict = {}
    for n, c in enumerate(by_end):
        outputs = c["failed_at"] if c["failed"] else c["egress_cells"] - 1
        want.setdefault((c["ingress_end"] // T, c["ingress_end"] % T // E), []).append(
            Released(n, c["ingress_bytes"], outputs, bool(c["failed"])))
    assert got == {k: tuple(v) for k, v in want.items()}
    assert [c["failed"] for c in verdict["connections"]] == [None, None, None, "late", "late"]
    assert plain.lengths == []


def test_a_window_whose_records_or_the_next_ones_are_not_published_complete_gets_no_lengths_rows():
    """A tick past its slack faults window 2's records, so neither window 2's releases nor window 1's, whose responses
    may reach into it, are handed; a refused connection withholds window 0's records, and with them window 0's
    releases only."""
    reqs, arrivals, answers = traffic()
    late, _, _ = run(answers, arrivals, params=LEN, windows=5, late={2 * T + 3: 0.05})
    assert [r.status.complete for r in records(late)[0]] == [True, True, False, True, True]
    assert [w for w, _ in late.lengths] == [0, 3]
    crowd = {0: [cells(reqs[0], B)] * 3}
    refused, _, _ = run([answer((1,))] * 2, crowd, params=LEN, windows=4)
    assert refused.refused == 1 and refused.refused_windows == {0} and len(refused.withheld) == 3
    assert [w for w, _ in refused.lengths] == [1, 2]
    assert all(not stride for _, strides in refused.lengths for stride in strides)
