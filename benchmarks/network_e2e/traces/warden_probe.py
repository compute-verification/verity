"""`python warden_probe.py ARGS...`: `benchmarks/network_traces/warden_serve.py ARGS...`, with probes the warden doesn't
keep, for telling a late tick's cause: every tick's lateness (`Warden.lateness`, which warden.json keeps only as
quantiles), how long each tick's own work took (`Warden._tick`), when the publisher packed, signed and committed each
record (`Publisher.keep`) and every garbage collection of the warden's process (serve turns the collector off, so any is
an explicit one).

A warden whose publisher is a spawned process (`PublisherProcess`) keeps its records there: that process runs this file
as `__mp_main__`, which times its `Publisher.keep` into `PUBLISHER_KEEPS` in `--out`, and the warden's own process
records when it handed each record's close over (`PublisherProcess.__call__`) and how long serve's `close` waited for
the last one, and writes the process's pid to `PUBLISHER_PID` once it has started (for `host_sample.py`).

When serve ends they go to `ticks.json` in its `--out` directory, times in seconds after the warden's origin:
`lateness_ms` (tick t's lateness past its deadline, origin + t·τ, in ms), `tick_ms` (tick t's `_tick`, in ms), `publish`
([start, ms, link, window] each, an inline publisher's), `gc` ([start, generation, ms, objects collected] of each
collection of generation 1 or 2, or of `GC_MS` or more) and `gc_counts` (all collections, by generation), with `origin`,
`bucket_ms`, `window`, `slack_ms`, `gc_enabled`, `switch_ms` and `cpus` (the warden's affinity, which its publisher
process inherits); and for a spawned publisher `publisher`: its `pid`, `handoff` ([time, link, window] each), `keep`
([start, ms, link, window] each, in its process) and `close` ([start, seconds waited]). The warden's code and schedule
are unchanged; this only wraps its methods and adds a `gc.callbacks` entry.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVE = ROOT / "benchmarks" / "network_traces" / "warden_serve.py"
#: a collection kept in `gc` by its own: any of generation 1 or 2, else one that took at least this long
GC_MS = 1.0
#: in `--out`: the spawned publisher's keeps, one JSON line [monotonic start, seconds, link, window] each
PUBLISHER_KEEPS = "publisher_keeps.jsonl"
#: in `--out`: the spawned publisher's pid
PUBLISHER_PID = "publisher.pid"


class Collections:
    """`gc.callbacks` entry recording each collection's start and length on `clock` (seconds)."""

    def __init__(self, clock=time.monotonic) -> None:
        self.clock, self.events, self.counts, self._start = clock, [], [0, 0, 0], None

    def __call__(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._start = self.clock()
            return
        if self._start is None:
            return
        gen, ms = info.get("generation", 0), (self.clock() - self._start) * 1000
        self.counts[gen] += 1
        if gen > 0 or ms >= GC_MS:
            self.events.append((self._start, gen, ms, info.get("collected", 0)))
        self._start = None

    def doc(self, origin: float) -> dict:
        return {"gc": [[round(t - origin, 4), g, round(ms, 3), n] for t, g, ms, n in self.events], "gc_counts": self.counts}


class Probe:
    """The wrappers' shared record; `install` puts them on the warden's and the publisher's classes. `out` is serve's
    `--out` (where a spawned publisher's keeps are, and its pid goes)."""

    def __init__(self, clock=time.monotonic, out: Path | None = None) -> None:
        self.clock, self.out, self.warden, self.work, self.publish = clock, out, None, [], []
        self.collections = Collections(clock)
        self.publisher_pid, self.handoff, self.close = None, [], None

    def install(self, warden_cls: type, publisher_cls: type, process_cls: type | None = None) -> None:
        run, tick, keep = warden_cls.run, warden_cls._tick, publisher_cls.keep
        probe = self

        async def probed_run(self) -> None:
            probe.warden = self
            await run(self)

        def probed_tick(self, t: int) -> None:
            start = probe.clock()
            try:
                tick(self, t)
            finally:
                probe.work.append(probe.clock() - start)

        def probed_keep(self, record, grid) -> None:
            start = probe.clock()
            try:
                keep(self, record, grid)
            finally:
                probe.publish.append((start, probe.clock() - start, record.link, record.window))

        warden_cls.run, warden_cls._tick, publisher_cls.keep = probed_run, probed_tick, probed_keep
        if process_cls is None:
            return
        init, call, close = process_cls.__init__, process_cls.__call__, process_cls.close

        def probed_init(self, *args, **kwargs) -> None:
            init(self, *args, **kwargs)
            probe.publisher_pid = self.process.pid
            if probe.out is not None:
                (probe.out / PUBLISHER_PID).write_text(str(self.process.pid))

        def probed_call(self, record, grid) -> None:
            probe.handoff.append((probe.clock(), record.link, record.window))
            call(self, record, grid)

        def probed_close(self) -> list:
            start = probe.clock()
            try:
                return close(self)
            finally:
                probe.close = (start, probe.clock() - start)

        process_cls.__init__, process_cls.__call__, process_cls.close = probed_init, probed_call, probed_close

    def publisher(self, origin: float) -> dict | None:
        if self.publisher_pid is None:
            return None
        path = None if self.out is None else self.out / PUBLISHER_KEEPS
        keeps = [json.loads(line) for line in path.read_text().splitlines()] if path and path.exists() else []
        return {"pid": self.publisher_pid,
                "handoff": [[round(t - origin, 4), link, window] for t, link, window in self.handoff],
                "keep": [[round(t - origin, 4), round(s * 1000, 3), link, window] for t, s, link, window in sorted(keeps)],
                "close": None if self.close is None else [round(self.close[0] - origin, 4), round(self.close[1], 3)]}

    def doc(self) -> dict:
        w = self.warden
        return {"origin": w.origin, "bucket_ms": w.params.bucket_ms, "window": w.params.window, "slack_ms": w.slack * 1000,
                "gc_enabled": gc.isenabled(), "switch_ms": sys.getswitchinterval() * 1000,
                "cpus": sorted(os.sched_getaffinity(0)),
                "lateness_ms": [round(x * 1000, 3) for x in w.lateness], "tick_ms": [round(x * 1000, 3) for x in self.work],
                "publish": [[round(t - w.origin, 4), round(s * 1000, 3), link, window]
                            for t, s, link, window in sorted(self.publish)],
                **self.collections.doc(w.origin), "publisher": self.publisher(w.origin)}


def time_keeps(publisher_cls: type, clock=time.monotonic) -> None:
    """In a spawned publisher's process: each `Publisher.keep`'s start and length, appended to `PUBLISHER_KEEPS` in the
    publisher's `out` as it returns."""
    keep = publisher_cls.keep

    def timed_keep(self, record, grid) -> None:
        start = clock()
        try:
            keep(self, record, grid)
        finally:
            with open(self.out / PUBLISHER_KEEPS, "a") as fh:
                fh.write(json.dumps([start, clock() - start, record.link, record.window]) + "\n")

    publisher_cls.keep = timed_keep


def _path() -> None:
    for p in (str(ROOT), str(SERVE.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main(argv: list[str]) -> int:
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    _path()
    import warden_serve

    from verity.core.protocols.certifier import warden

    probe = Probe(out=out)
    gc.callbacks.append(probe.collections)
    probe.install(warden.Warden, warden_serve.Publisher, getattr(warden_serve, "PublisherProcess", None))
    try:
        return warden_serve.main(argv)
    finally:
        if out is not None and probe.warden is not None:
            (out / "ticks.json").write_text(json.dumps(probe.doc(), separators=(",", ":")))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
elif __name__ == "__mp_main__":
    _path()
    import warden_serve

    time_keeps(warden_serve.Publisher)
