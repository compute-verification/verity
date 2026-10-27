"""tick-load: the network certifier's tick lateness in a served rehearsal on a stub endpoint, beside a CPU hog confined to
this process's own CPUs.

    python tick_load.py --out DIR [--hogs-per-cpu 3] [--requests 400] [--in-flight 4] [--tokens 64]
        [--token-ms 25] [--delta-buckets 52] [--profile planned|served] [--strategies honest[,late,extra]]
        [--secret-bit] [--paced-ms MS]

It starts a stub OpenAI streaming endpoint on loopback (`/v1/models`, `/tokenize`, and `/v1/completions`, each
completion `max_tokens` chunks `--token-ms` apart, side by side, as vLLM decodes the requests in flight), then
`--hogs-per-cpu` busy loops per CPU of this process's affinity (the run's cpuset), each pinned to that set with
`taskset`, then `certifier_serve.py rehearse` against the stub (`--strategies`, honest by default, on N2's
`--profile`), and stops the hogs. It writes
`ticks.json` in `--out`: the CPUs and hogs, the rehearsal's exit and verdict, and per grid the ticks and their lateness
(median, p99, p99.9, max in ms, the ticks past the slack with their window and bucket, and the windows the honest run
was refused on), the requests forwarded to the server past their deadline, by ingress window, with how late in ms
(`forwards_late`, None for one never forwarded), and from a paced proxy its pacer's release starts and write ends and
the buckets it got late or released past the slack, each with its write's end and the previous bucket's (`paced`).
The rehearsal's own outputs are under
`--out`/rehearsal.

**With `--secret-bit`** the stub times its writes by a secret bit per egress bucket, which is J's question
(`NetTiming.JitterBounded`): does an offset within a bucket that the prover chooses reach a receiver outside?
The stub draws a secret of its own, never written, and bucket t's bit and lead (0 … `LEAD_MS` ms) are SHA-512's of it
and t (`SecretBits`). Once `serve` has written the honest run's `deployment.json` (the plan's probes before it are paced
as without), each token waits from its pacing to the first slot after it: bucket t's is its lead before t's boundary
when t's bit is 1, and half a bucket more when 0, so both bits write the same tokens a bucket, and they differ only in
their phase against the egress tick that makes bucket t. `jitter.json` in `--out` holds, per bucket, its bit, its lead,
the tokens written in its slot, the client's offset of the bucket's arrival after its boundary (`peer.json`'s
`bucket_lag_ms`: the proxy writes a bucket's cells in one write, so each cell's offset is its bucket's), the egress
tick's lateness and, from a paced proxy (`certifier_serve.py --paced-ms`), when its pacer started the bucket's write
after the bucket's time; and, for the buckets written in and the idle ones (the control), whether those offsets tell
the bit apart (`distinguish`). `next` holds the same of each bucket's client offset against the next bucket's bit, over
the buckets whose next was written in: that bit's 0 writes half a bucket before the next boundary, so at a pacer
offset of half a bucket (`certifier_serve.PACED_MS`) just before this bucket's write. It holds timing and the bits,
none of the run's records, salts or keys.

Run it inside one job of N CPUs, so the hog loads only the job's own cores.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: the secret-bit stub's lead before its slot's boundary, drawn per bucket from [0, LEAD_MS) ms: the stub's write
#: reaches the proxy through the shim's process, a hop whose latency the stub doesn't know, so the leads span it
LEAD_MS = 2.0
#: how long before its slot the stub stops sleeping on its loop and spins: a loop's sleep ends up to 1 ms late, since
#: epoll's timeout is in ms
SPIN_S = 0.003
#: `distinguish`'s bins of the offset for the mutual information (octiles), and its shuffles of the bits
BINS, SHUFFLES = 8, 200


class SecretBits:
    """The secret-bit stub's slots (`--secret-bit`): a secret of the stub's own, and the honest run's egress grid, read
    from `rehearsal`/honest/deployment.json and the plan's calibration once `serve` has written them. `writes` holds
    each token written in a slot: its bucket and how late after the slot it was written, in ms."""

    def __init__(self, rehearsal: Path, secret: bytes, warden: bool = False) -> None:
        self.rehearsal, self.secret, self.warden = rehearsal, secret, warden
        self.grid: tuple[float, float] | None = None
        self.writes: list[list] = []

    def draw(self, t: int) -> tuple[int, float]:
        """Bucket t's bit and lead, in seconds."""
        h = hashlib.sha512(self.secret + t.to_bytes(8, "big")).digest()
        return h[0] & 1, int.from_bytes(h[1:5], "big") / 2**32 * LEAD_MS / 1000

    def ready(self) -> bool:
        """Whether the grid is known: the origin on this host's monotonic clock as the client takes it, from the
        deployment's `origin_unix`, and the egress bucket in seconds (a warden's, from its own `deployment.json`)."""
        if self.grid is None and self.warden and (path := self.rehearsal / "deployment.json").exists():
            try:
                d = json.loads(path.read_text())
            except ValueError:  # still being written
                return False
            self.grid = (time.monotonic() + d["origin_unix"] - time.time(), d["params"]["bucket_ms"] / 1000)
        elif self.grid is None and (path := self.rehearsal / "honest" / "deployment.json").exists():
            d = json.loads(path.read_text())
            buckets = json.loads((self.rehearsal / "calibration.json").read_text())["grid"]["window_buckets"]
            self.grid = (time.monotonic() + d["origin_unix"] - time.time(), d["window_s"] / buckets)
        return self.grid is not None

    def slot(self, t: int) -> float:
        """Bucket t's slot on the monotonic clock: its lead before its boundary with bit 1, and half a bucket more
        with 0."""
        origin, tau = self.grid
        bit, lead = self.draw(t)
        return origin + t * tau - lead - (0 if bit else tau / 2)

    def bucket(self, due: float) -> int:
        """The bucket of the first slot at or after `due` (slots rise with t, each within half a bucket and `LEAD_MS`
        before its boundary)."""
        origin, tau = self.grid
        t = max(0, math.floor((due - origin) / tau))
        while self.slot(t) < due:
            t += 1
        return t


class Endpoint:
    """The stub endpoint: one model, a tokenizer that makes a token of every byte, and streamed completions paced
    `token_s` apart, every connection on its own; with `bits`, each token written at the first of their slots after its
    pacing, once their grid is known."""

    def __init__(self, token_s: float, bits: SecretBits | None = None) -> None:
        self.token_s, self.bits = token_s, bits

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            n = re.search(rb"content-length: (\d+)", head, re.I)
            body = json.loads(await reader.readexactly(int(n.group(1)))) if n else None
            if head.startswith(b"POST /v1/completions "):
                await self.complete(body, writer)
            else:
                doc = ({"data": [{"id": "stub-model"}]} if head.startswith(b"GET /v1/models ") else
                       {"tokens": list(body["prompt"].encode()), "count": len(body["prompt"].encode())})
                data = json.dumps(doc).encode()
                writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n%s"
                             % (len(data), data))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    async def complete(self, body: dict, writer: asyncio.StreamWriter) -> None:
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        loop = asyncio.get_running_loop()
        start = loop.time()
        bits = self.bits if self.bits is not None and self.bits.ready() else None
        for k in range(body["max_tokens"]):
            due = start + (k + 1) * self.token_s
            if bits is None:
                await asyncio.sleep(max(0.0, due - loop.time()))
            else:
                t = bits.bucket(due)
                due = bits.slot(t)
                await asyncio.sleep(max(0.0, due - SPIN_S - loop.time()))
                while loop.time() < due:
                    pass
            p = b'data: {"choices":[{"text":" t%d"}]}\n\n' % k
            writer.write(b"%x\r\n%s\r\n" % (len(p), p))
            if bits is not None:
                bits.writes.append([t, round((loop.time() - due) * 1000, 3)])
            await writer.drain()
        done = b"data: [DONE]\n\n"
        writer.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(done), done))
        await writer.drain()


def serve_endpoint(token_s: float, bits: SecretBits | None = None) -> int:
    """Start the stub on a loopback port in a thread of its own; its port."""
    ready: list = []
    started = threading.Event()

    def main() -> None:
        async def go() -> None:
            server = await asyncio.start_server(Endpoint(token_s, bits).handle, "127.0.0.1", 0)
            ready.append(server.sockets[0].getsockname()[1])
            started.set()
            await server.serve_forever()

        asyncio.run(go())

    threading.Thread(target=main, daemon=True).start()
    started.wait()
    return ready[0]


def _quantile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def publisher(timing: dict, log: dict, buckets: int) -> dict:
    """The publisher's calls (`publisher_timing.json`, on the proxy's clock): wall and thread CPU seconds of each, and
    per grid the lateness of the ticks whose wait [deadline, deadline + lateness] overlaps a call's [start, end] against
    the rest's."""
    calls = [c for c in timing["calls"] if "end" in c]
    spans = sorted((c["start"], c["end"]) for c in calls)
    def ms(xs: list[float]) -> dict:
        return {"n": len(xs), "median_ms": _quantile(xs, 0.5) if xs else None,
                "p99_ms": _quantile(xs, 0.99) if xs else None, "max_ms": max(xs) if xs else None}

    out = {"switchinterval_s": timing["switchinterval"], "calls": len(calls),
           "wall": ms([(c["end"] - c["start"]) * 1000 for c in calls]),
           "keep_cpu": ms([c["keep_cpu"] * 1000 for c in calls]),
           "certify_cpu": ms([c["certify_cpu"] * 1000 for c in calls]),
           "queued": ms([(c["start"] - c["submit"]) * 1000 for c in calls])}
    for grid, lateness in log["lateness_ms"].items():
        tau = timing["bucket_ms"][grid] / 1000
        over, rest, late = [], [], []
        for k, x in enumerate(lateness):
            t = timing["start"] * buckets + k
            a = timing["origin"] + t * tau
            b = a + max(x, 0.0) / 1000
            hit = [s for s in spans if s[0] < b and s[1] > a]
            (over if hit else rest).append(x)
            if x > 10:
                late.append([t, t // buckets, t % buckets, x, bool(hit)])
        out[grid] = {"overlapping": ms(over), "not_overlapping": ms(rest), "past_10ms": late}
    return out


def summary(run: Path) -> dict:
    """Per grid, the honest run's tick lateness (`proxy.json.gz`), and its refused windows (`check.json`)."""
    log = json.loads(gzip.open(run / "proxy.json.gz").read())
    buckets = json.loads((run / "calibration.json").read_text())["grid"]["window_buckets"]
    slack = log["slack_ms"]
    out = {"slack_ms": slack}
    for grid, lateness in log["lateness_ms"].items():
        late = [[t, t // buckets, t % buckets, x] for t, x in enumerate(lateness) if x > slack]
        out[grid] = {"ticks": len(lateness), "median_ms": _quantile(lateness, 0.5), "p99_ms": _quantile(lateness, 0.99),
                     "p999_ms": _quantile(lateness, 0.999), "max_ms": max(lateness),
                     "past_slack": late, "worst": sorted(enumerate(lateness), key=lambda tx: -tx[1])[:8]}
    for grid, lateness in log["lateness_ms"].items():
        pos = {"first": [x for t, x in enumerate(lateness) if t % buckets == 0],
               "last": [x for t, x in enumerate(lateness) if t % buckets == buckets - 1],
               "other": [x for t, x in enumerate(lateness) if 0 < t % buckets < buckets - 1]}
        out[grid]["by_position"] = {k: {"n": len(v), "median_ms": _quantile(v, 0.5), "p99_ms": _quantile(v, 0.99),
                                        "max_ms": max(v)} for k, v in pos.items() if v}
    if (run / "publisher_timing.json").exists():
        out["publisher"] = publisher(json.loads((run / "publisher_timing.json").read_text()), log, buckets)
    check = json.loads((run / "check.json").read_text()) if (run / "check.json").exists() else {}
    out["forwards_late"] = [[w, ms] for w, ms in log.get("forwards", []) if ms is None or ms > 0]
    if (paced := log.get("paced")) is not None:
        out["paced"] = _paced(paced, buckets, slack)
    out["verdict"] = check.get("verdict")
    out["refused_windows"] = {row["window"]: {d: a["raw"] for d, a in row["audit"].items() if a["raw"]}
                              for row in check.get("rows", []) if not row["n2"]["accepted"]}
    return out


def _paced(paced: dict, buckets: int, slack: float) -> dict:
    """A paced run's pacer reports (`releases`, in ms after each bucket's time): the release starts' and write ends'
    tails, and the buckets it got late, released past the slack or didn't write, each with its write's end and the
    previous bucket's."""
    rows = paced["releases"]
    start, end = [r[2] for r in rows], {r[0]: r[3] for r in rows}
    return {"ms": paced["ms"], "buckets": len(rows), "start_median_ms": _quantile(start, 0.5),
            "start_p99_ms": _quantile(start, 0.99), "start_p999_ms": _quantile(start, 0.999),
            "start_max_ms": max(start), "arrived_max_ms": max(r[1] for r in rows),
            "end_p999_ms": _quantile(list(end.values()), 0.999), "end_max_ms": max(end.values()),
            "late": [[t, t // buckets, t % buckets, a, s, e, end.get(t - 1)] for t, a, s, e, w in rows
                     if a > 0 or s > slack or not w]}


def warden_summary(run: Path) -> dict:
    """A warden rehearsal's (`--warden`) tick lateness (`warden.json`), its pacer's reports (`paced`), and from
    `check.json` its verdict, each refused window's records with their status (`uncertified` for a record without its
    certificate) and its connections that didn't end."""
    log = json.loads((run / "warden.json").read_text())
    dep = json.loads((run / "deployment.json").read_text())
    buckets, slack, ticks = dep["params"]["window"], dep["slack_ms"], log["ticks_ms"]
    out = {"slack_ms": slack, "egress": {
        "ticks": len(ticks), "median_ms": _quantile(ticks, 0.5), "p99_ms": _quantile(ticks, 0.99),
        "p999_ms": _quantile(ticks, 0.999), "max_ms": max(ticks),
        "past_slack": [[t, t // buckets, t % buckets, x] for t, x in enumerate(ticks) if x > slack],
        "worst": sorted(enumerate(ticks), key=lambda tx: -tx[1])[:8]}}
    if (paced := log.get("paced")) is not None:
        out["paced"] = _paced(paced, buckets, slack)
    check = json.loads((run / "check.json").read_text()) if (run / "check.json").exists() else {}
    refused: dict[int, dict[str, str]] = {}
    for w in check.get("windows", []):
        if not w["certified"] or w["status"]:
            refused.setdefault(w["window"], {})[w["record"]] = w["status"] or "uncertified"
    out["verdict"] = None if not check else "accepted" if check["accepted"] else "refused"
    out["refused_windows"] = dict(sorted(refused.items()))
    out["connections_not_ended"] = [[c.get("ordinal"), c["outcome"]] for c in check.get("connections", [])
                                    if c.get("outcome") != "end"]
    out["problems"] = check.get("problems", [])[:20]
    return out


def _ranks(xs: list[float]) -> tuple[list[float], float]:
    """Each value's rank from 1, ties at their mean rank, and Σ (c³ − c) over the ties' counts c."""
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks, ties, i = [0.0] * len(xs), 0.0, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in order[i:j + 1]:
            ranks[k] = (i + j) / 2 + 1
        ties += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    return ranks, ties


def _fit(pairs: list[tuple[float, int]]) -> tuple[float, int]:
    """The threshold and side (1: bit 1 above it, −1: below) that read the most of `pairs`' bits right."""
    xs = sorted(pairs)
    ones, n = sum(b for _, b in xs), len(xs)
    best, below_ones = (max(ones, n - ones), -math.inf, 1 if ones >= n - ones else -1), 0
    for i in range(n - 1):
        below_ones += xs[i][1]
        if xs[i][0] == xs[i + 1][0]:
            continue
        above = (i + 1 - below_ones) + (ones - below_ones)
        cut = (xs[i][0] + xs[i + 1][0]) / 2
        best = max(best, (above, cut, 1), (n - above, cut, -1))
    return best[1], best[2]


def _information(bins: list[int], bits: list[int]) -> float:
    """The plug-in mutual information of `bins` and `bits`, in bits."""
    n, joint = len(bits), Counter(zip(bins, bits))
    pb, px = Counter(bits), Counter(bins)
    return sum(c / n * math.log2(c * n / (px[x] * pb[b])) for (x, b), c in joint.items())


def distinguish(pairs: list[tuple[float, int]], seed: int = 0) -> dict:
    """Whether `pairs`' offsets (ms) tell their bits apart: per bit the count, median and p90, the Mann–Whitney z of bit
    1's offsets against bit 0's with its two-sided p (normal approximation, ties corrected), a threshold decoder fitted
    on the even entries and its accuracy on the odd ones (with its standard error, and the share of the odd entries'
    commoner bit, what guessing gets), and the mutual information of the bit and the offset's bin (`BINS`
    equal-frequency bins) with the share of `SHUFFLES` shuffles of the bits that reach it (its p)."""
    groups = {b: sorted(x for x, c in pairs if c == b) for b in (0, 1)}
    out: dict = {"n": len(pairs), "by_bit": {str(b): {"n": len(xs), "median_ms": _quantile(xs, 0.5) if xs else None,
                                                    "p90_ms": _quantile(xs, 0.9) if xs else None}
                                            for b, xs in groups.items()}}
    n1, n0 = len(groups[1]), len(groups[0])
    if n1 < 2 or n0 < 2:
        return out
    ranks, ties = _ranks([x for x, _ in pairs])
    n = n1 + n0
    u = sum(r for r, (_, b) in zip(ranks, pairs) if b) - n1 * (n1 + 1) / 2
    var = n1 * n0 / 12 * ((n + 1) - ties / (n * (n - 1)))
    z = (u - n1 * n0 / 2) / math.sqrt(var) if var > 0 else 0.0
    out["mann_whitney"] = {"z": round(z, 3), "p": math.erfc(abs(z) / math.sqrt(2))}
    train, test = pairs[0::2], pairs[1::2]
    cut, side = _fit(train)
    right = sum((side * (x - cut) > 0) == bool(b) for x, b in test)
    acc = right / len(test)
    commoner = max(sum(b for _, b in test), len(test) - sum(b for _, b in test)) / len(test)
    out["decoder"] = {"threshold_ms": None if math.isinf(cut) else round(cut, 3), "side": side, "test_n": len(test),
                      "accuracy": round(acc, 4), "stderr": round(math.sqrt(acc * (1 - acc) / len(test)), 4),
                      "guess": round(commoner, 4)}
    xs = sorted(x for x, _ in pairs)
    cuts = [xs[len(xs) * k // BINS] for k in range(1, BINS)]
    bins, bits = [bisect.bisect_right(cuts, x) for x, _ in pairs], [b for _, b in pairs]
    mi, rng, shuffled = _information(bins, bits), random.Random(seed), bits[:]
    reach = 0
    for _ in range(SHUFFLES):
        rng.shuffle(shuffled)
        reach += _information(bins, shuffled) >= mi
    out["information"] = {"bits": round(mi, 5), "p": (1 + reach) / (1 + SHUFFLES)}
    return out


def jitter(run: Path, bits: SecretBits) -> dict:
    """`jitter.json`: per egress bucket of the honest run `run` (the module docstring's `--secret-bit`), and
    `distinguish` of the client's offsets, of the egress ticks' lateness and of the pacer's write starts, for the
    buckets written in and the idle ones, of the client's offsets against the next bucket's bit (`next`), and of the
    written ones' per quarter of the lead's range."""
    lag = json.loads((run / "client" / "peer.json").read_text())["bucket_lag_ms"]
    log = json.loads(gzip.open(run / "proxy.json.gz").read())
    return _jitter(lag, log["lateness_ms"]["egress"], log.get("paced"), bits)


def warden_lags(run: Path) -> list[float | None]:
    """Per bucket of a warden rehearsal `run`, when its first cell reached a client after its boundary, in ms (None: no
    connection's): each cell's bucket is the egress record's (`warden_serve.egress_cells`), its arrival the client's."""
    import warden_serve as ws

    dep = json.loads((run / "deployment.json").read_text())
    params = ws.WardenParams(**dep["params"])
    schedule = ws.egress_cells(ws.read_records(run, params)[ws.EGRESS], params)
    lag: list[float | None] = [None] * (dep["windows"] * params.window)
    for conn in json.loads((run / "client.json").read_text())["connections"]:
        if "error" in conn or conn["outcome"] == "refused":
            continue
        for (t, _), ms in zip(schedule.get(conn["ordinal"], []), conn["arrivals_ms"]):
            x = round(ms - t * params.bucket_ms, 3)
            if t < len(lag) and (lag[t] is None or x < lag[t]):
                lag[t] = x
    return lag


def warden_jitter(run: Path, bits: SecretBits) -> dict:
    """`jitter.json` of a warden rehearsal (`--warden`): `jitter`'s, with each bucket's client offset the first of its
    cells to reach a client (`warden_lags`), the buckets no connection had cells in left out, and each tick's lateness
    and the pacer's release starts from `warden.json`."""
    log = json.loads((run / "warden.json").read_text())
    return _jitter(warden_lags(run), log["ticks_ms"], log.get("paced"), bits)


def _jitter(lag: list[float | None], ticks: list[float], paced: dict | None, bits: SecretBits) -> dict:
    starts = {t: s for t, _, s, _, _ in (paced or {}).get("releases", [])}
    written = Counter(t for t, _ in bits.writes)
    rows = []
    for t, x in enumerate(lag):
        bit, lead = bits.draw(t)
        rows.append([t, bit, round(lead * 1000, 3), written.get(t, 0), x, ticks[t] if t < len(ticks) else None,
                     starts.get(t)])
    late = [x for _, x in bits.writes]
    out = {"lead_ms": LEAD_MS, "bucket_ms": None if bits.grid is None else bits.grid[1] * 1000,
           "paced_ms": (paced or {}).get("ms"),
           "writes": {"n": len(late), "late_median_ms": _quantile(late, 0.5) if late else None,
                      "late_p99_ms": _quantile(late, 0.99) if late else None, "late_max_ms": max(late, default=None)},
           "columns": ["bucket", "bit", "lead_ms", "tokens_written", "client_offset_ms", "tick_lateness_ms",
                       "release_start_ms"],
           "buckets": rows}
    for name, keep in (("written", lambda r: r[3] > 0), ("idle", lambda r: r[3] == 0)):
        sel = [r for r in rows if keep(r) and r[4] is not None]
        out[name] = {"client": distinguish([(r[4], r[1]) for r in sel]),
                     "tick": distinguish([(r[5], r[1]) for r in sel if r[5] is not None]),
                     "release": distinguish([(r[6], r[1]) for r in sel if r[6] is not None])}
    following = [(r, n) for r, n in zip(rows, rows[1:]) if n[3] > 0 and r[4] is not None]
    out["next"] = {"client": distinguish([(r[4], n[1]) for r, n in following]),
                   "release": distinguish([(r[6], n[1]) for r, n in following if r[6] is not None])}
    quarter = LEAD_MS / 4
    out["written_by_lead"] = {
        f"{q * quarter:g}-{(q + 1) * quarter:g}": distinguish([(r[4], r[1]) for r in rows if r[3] > 0 and r[4] is not None
                                                               and q * quarter <= r[2] < (q + 1) * quarter])
        for q in range(4)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tick_load.py", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hogs-per-cpu", type=int, default=3)
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument("--in-flight", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--token-ms", type=float, default=25.0)
    ap.add_argument("--delta-buckets", type=int, default=52)
    ap.add_argument("--profile", default="planned", help="N2's profile, planned or served (`certifier_serve.py plan`)")
    ap.add_argument("--strategies", default="honest", help="the rehearsal's strategies, honest, late, extra")
    ap.add_argument("--secret-bit", action="store_true",
                    help="time the stub's writes by a secret bit per egress bucket and write jitter.json")
    ap.add_argument("--paced-ms", type=float, help="the rehearsal's --paced-ms (certifier_serve.py's or "
                                                   "warden_serve.py's default if not given; 0: the proxy writes each "
                                                   "bucket on its loop, and the warden, which always paces, refuses "
                                                   "it)")
    ap.add_argument("--warden", action="store_true",
                    help="rehearse the warden (warden_serve.py rehearse) instead of the proxy: --requests at --rate a "
                         "second, each of at most --tokens, windows of --window buckets")
    ap.add_argument("--rate", type=float, default=1.0, help="with --warden: the requests' Poisson arrivals a second")
    ap.add_argument("--window", type=int, default=60, help="with --warden: the warden's window, in buckets")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))
    bits = SecretBits(a.out / "rehearsal", os.urandom(64), warden=a.warden) if a.secret_bit else None
    port = serve_endpoint(a.token_ms / 1000, bits)
    cpu_list = ",".join(map(str, cpus))
    paced = [] if a.paced_ms is None else ["--paced-ms", str(a.paced_ms)]
    if a.warden:
        command = [sys.executable, str(HERE / "warden_serve.py"), "rehearse", "--upstream", f"127.0.0.1:{port}",
                   "--model", "stub-model", "--requests", str(a.requests), "--rate", str(a.rate), "--max-tokens",
                   str(a.tokens), "--window", str(a.window), "--tail", str(a.window / 10),
                   "--out", str(a.out / "rehearsal"), *paced]
    else:
        command = [sys.executable, str(HERE / "certifier_serve.py"), "rehearse",
                   "--upstream", f"http://127.0.0.1:{port}", "--requests", str(a.requests),
                   "--in-flight", str(a.in_flight), "--tokens", str(a.tokens),
                   "--delta-buckets", str(a.delta_buckets), "--profile", a.profile,
                   "--strategies", a.strategies, "--out", str(a.out / "rehearsal"), *paced]
    hogs = [subprocess.Popen(["taskset", "-c", cpu_list, sys.executable, "-c", "while True: pass"])
            for _ in range(a.hogs_per_cpu * len(cpus))]
    try:
        rc = subprocess.run(command, env=os.environ | {"CERTIFIER_SERVE_TIMING": "1"}).returncode
    finally:
        for h in hogs:
            h.kill()
        for h in hogs:
            h.wait()
    doc = {"cpus": cpus, "hogs": len(hogs), "rehearse_exit": rc, "token_ms": a.token_ms, "requests": a.requests,
           "in_flight": a.in_flight, "delta_buckets": a.delta_buckets, "profile": a.profile,
           "strategies": a.strategies, "paced_ms": a.paced_ms, "warden": a.warden}
    run = a.out / "rehearsal" if a.warden else a.out / "rehearsal" / "honest"
    if a.warden and (run / "warden.json").exists():
        doc |= {"rate": a.rate, "window": a.window} | warden_summary(run)
    elif (run / "proxy.json.gz").exists():
        doc |= summary(run)
    (a.out / "ticks.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc.get(k) for k in ("verdict", "refused_windows")} |
                     {g: {k: doc[g][k] for k in ("ticks", "p999_ms", "max_ms")} | {"past_slack": len(doc[g]["past_slack"])}
                      for g in ("egress", "ingress") if g in doc}), flush=True)
    if bits is not None and a.warden and (run / "client.json").exists() and (run / "warden.json").exists():
        j = warden_jitter(run, bits) | {"hogs": len(hogs), "cpus": cpus, "verdict": doc.get("verdict")}
    elif bits is not None and (run / "client" / "peer.json").exists() and (run / "proxy.json.gz").exists():
        j = jitter(run, bits) | {"hogs": len(hogs), "cpus": cpus, "verdict": doc.get("verdict")}
    else:
        j = None
    if j is not None:
        (a.out / "jitter.json").write_text(json.dumps(j))
        print(json.dumps({"jitter": {k: {g: {m: d.get(m) for m in ("n", "mann_whitney", "decoder", "information")}
                                         for g, d in j[k].items()} for k in ("written", "idle", "next")}}),
              flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
