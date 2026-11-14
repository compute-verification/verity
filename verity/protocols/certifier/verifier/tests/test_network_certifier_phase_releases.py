"""The bound on an execution phase's releases (`WardenParams.phase_releases`, R), on the device, driven tick by tick on
`test_network_certifier_warden`'s harness: under loads that keep every lane busy, each client sending its request
when the load says, no execution phase releases more than R requests, and a load that fills an execution phase
releases exactly R in it where the schedule lets it. R is the most rows a class's tables have under the lengths
rows' form 2 (`content_rows`' "The classes"), so the device is held to it here and nowhere on its tick."""

from __future__ import annotations

import dataclasses
import random
from collections import Counter

import pytest

from verity.protocols.certifier.verifier.tests.test_network_certifier_lengths import LEN, id_req
from verity.protocols.certifier.verifier.tests.test_network_certifier_warden import Client, Server, run
from verity.protocols.certifier.verifier.warden import WardenParams, cells

#: the lengths tests' schedule at form 2: two lanes, sixteen buckets an execution phase
SMALL = dataclasses.replace(LEN, lengths=2)
#: the schedule `benchmarks/network_traces/connection_rows_run.py` costs by default, at form 2, and the same with
#: ingress half as long
WIDE = WardenParams(cell_bytes=512, slots=8, bucket_ms=100, delay=30, lanes=16, queue=64, window=600, ingress_cells=96,
                    ingress_buckets=30, egress_cells=4097, every=1, lengths=2)
SHORT = dataclasses.replace(WIDE, ingress_buckets=15)


def holding() -> tuple[Server, type[Client]]:
    """A server that never answers, so that every released connection fails at its first egress slot and frees its
    lane then, and clients that each send their request's cells at the bucket they are given, `(cells, bucket)`, or
    at once, `(cells, None)`."""
    waiting: list[tuple[int, Client, bytes]] = []

    class Holding(Client):
        def __init__(self, device, data) -> None:
            made, due = data
            super().__init__(device, made if due is None else (b"", False))
            if due is not None:
                waiting.append((due, self, made))

    class Silent(Server):
        def release(self, t: int) -> None:
            for due, client, made in waiting:
                if due == t:
                    client.reader.feed_data(made)
                    client.reader.feed_eof()
            super().release(t)

    return Silent([]), Holding


def released(params: WardenParams, arrivals: dict[int, list], phases: int) -> tuple[Counter, object]:
    """The releases of each execution phase of a run of `phases` under `arrivals`, by the device's own log, and the
    device. Its records are dropped as they are published: only the count is read."""
    server, client = holding()
    device, _, _ = run(server, arrivals, params=params, windows=phases, client=client,
                       publish=lambda record, grid: None)
    assert device.refused == 0
    return Counter(e["ingress_end"] // params.window for e in device.log if e["release"] is not None), device


def filling(params: WardenParams, phase: int = 1) -> dict[int, list]:
    """A load that fills execution phase `phase`: every lane admits a connection in the earliest bucket whose
    release can be in the execution phase, `ingress_buckets` − 1 before its first, and again in each bucket the lane
    is free, every D + 1. The first connections send their request so that it is released in the execution phase's
    first bucket, and the later ones at once."""
    T, one = params.window, cells(id_req(1, 3), params.cell_bytes)
    assert len(one) == params.cell_bytes
    first = phase * T - (params.ingress_buckets - 1)
    arrivals = {first: [(one, phase * T)] * params.lanes}
    # a request sent at once is released in the bucket after its admission
    for s in range(first + params.delay + 1, (phase + 1) * T - 1, params.delay + 1):
        arrivals[s] = [(one, None)] * params.lanes
    return arrivals


def eager(params: WardenParams, phases: int, hold: int = 0) -> dict[int, list]:
    """A load that keeps every lane busy from the run's first bucket: a connection a lane in each bucket the lane is
    free, its request sent `hold` buckets after its admission, or at once."""
    one = cells(id_req(1, 3), params.cell_bytes)
    return {s: [(one, s + hold if hold else None)] * params.lanes
            for s in range(0, phases * params.window - params.span, params.delay + 1)}


def scattered(params: WardenParams, phases: int, seed: int) -> dict[int, list]:
    """A seeded load of connections that arrive in bursts of a lane's worth at most, each holding its request for a
    time of its own inside the ingress bound: no more than the lanes and the queue hold at once."""
    rng, one = random.Random(seed), cells(id_req(1, 3), params.cell_bytes)
    arrivals: dict[int, list] = {}
    for s in range(0, phases * params.window - params.span, params.delay + 1):
        for _ in range(params.lanes):
            at = s + rng.randrange(params.delay + 1)
            hold = rng.randrange(params.ingress_buckets)
            arrivals.setdefault(at, []).append((one, at + hold + rng.randrange(params.delay) if hold else None))
    return arrivals


def test_the_bound_is_the_schedules_own_number():
    """R = lanes · ⌈(T + ingress_buckets − 1) / (D + 1)⌉: the two schedules' values, and it moves with each of its
    four parameters and with no other."""
    assert (WIDE.phase_releases, SHORT.phase_releases, SMALL.phase_releases) == (16 * 21, 16 * 20, 2 * 3)
    for change, R in (({"lanes": 3}, 9), ({"window": 32, "egress_cells": 9}, 12), ({"ingress_buckets": 1}, 6),
                      ({"ingress_buckets": 2}, 6), ({"delay": 8}, 4), ({"delay": 17, "window": 32}, 4),
                      ({"every": 1}, 6), ({"slots": 4}, 6), ({"queue": 9}, 6), ({"ingress_cells": 2}, 6)):
        assert dataclasses.replace(SMALL, **change).phase_releases == R, change
    assert LEN.phase_releases == SMALL.phase_releases


def busy(params: WardenParams, phases: int, seeds: int) -> list[dict[int, list]]:
    """Loads that keep the lanes busy through a run: sent at once, at each bucket of the ingress bound, and at
    seeded times."""
    return [*(eager(params, phases, hold=hold) for hold in range(params.ingress_buckets)),
            *(scattered(params, phases, seed) for seed in range(seeds))]


def test_no_load_releases_more_than_the_bound_and_the_filling_load_releases_it():
    """At the small schedule: the filling load releases R = 6 in the execution phase it fills, each lane's three,
    and none in the ones beside it, and the execution phase's hand holds them; loads that keep the lanes busy from
    the first bucket, each client sending at once, at any bucket of the ingress bound or at seeded times, release no
    more than R in any execution phase, and one of them R."""
    R, T = SMALL.phase_releases, SMALL.window
    got, device = released(SMALL, filling(SMALL), 3)
    assert got == {1: R} and [w for w, strides in device.lengths if strides is not None] == [0, 1, 2]
    handed = dict(device.lengths)[1]
    assert sum(map(len, handed)) == R and len(handed[0]) == SMALL.lanes
    log = sorted(device.log, key=lambda e: e["ordinal"])
    assert [(e["admitted"], e["ingress_end"]) for e in log[::2]] == [(T - 2, T), (T + 4, T + 5), (T + 10, T + 11)]
    assert all(e["failed"] == "late" and e["ended"] == e["admitted"] + SMALL.delay for e in log)
    most = 0
    for load in busy(SMALL, 4, seeds=40):
        got, _ = released(SMALL, load, 4)
        assert max(got.values(), default=0) <= R, got
        most = max([most, *got.values()])
    assert most == R


def test_the_bound_is_not_the_most_at_every_schedule():
    """R counts a lane's admissions in T + `ingress_buckets` − 1 buckets, and the device releases no request in its
    admission's bucket, so the admissions whose release is in one execution phase are in a bucket fewer. Where that
    bucket is the one a lane's last admission needs, here T = 16, D = 5 and `ingress_buckets` = 4, the filling load
    releases a lane's worth fewer than R, 6 of 8, and no busy load more: R is a bound there and not the most."""
    edge = dataclasses.replace(SMALL, ingress_buckets=4)
    R = edge.phase_releases
    assert R == 8 and (edge.window + edge.ingress_buckets - 2) % (edge.delay + 1) == 0
    got, _ = released(edge, filling(edge), 3)
    assert got == {1: R - edge.lanes}
    for load in busy(edge, 4, seeds=40):
        got, _ = released(edge, load, 4)
        assert max(got.values(), default=0) <= R - edge.lanes, got


@pytest.mark.parametrize("params", [WIDE, SHORT], ids=["ingress-30", "ingress-15"])
def test_at_the_costed_schedule_no_load_releases_more_than_the_bound_and_the_filling_load_releases_it(params):
    """At the schedule the cost runner costs by default and at the same with `ingress_buckets` 15, each a run of
    three execution phases of 600 buckets on 16 lanes: the filling load releases exactly R in the execution phase
    it fills, 336 and 320, so at both schedules R is the most an execution phase releases; its hand holds them; and
    loads that keep every lane busy from the first bucket release no more than R in any execution phase."""
    R = params.phase_releases
    got, device = released(params, filling(params), 3)
    assert got == {1: R} and R == {30: 336, 15: 320}[params.ingress_buckets]
    assert sum(map(len, dict(device.lengths)[1])) == R
    for load in (eager(params, 3), eager(params, 3, hold=params.ingress_buckets - 1), scattered(params, 3, 1)):
        got, _ = released(params, load, 3)
        assert sum(got.values()) > 0 and max(got.values()) <= R, got
