"""`python warden_probe.py ARGS...`: `benchmarks/network_traces/warden_serve.py ARGS...`, with probes the warden doesn't
keep, for telling a late tick's cause: every tick's lateness (`Warden.lateness`, which warden.json keeps only as
quantiles), how long each tick's own work took (`Warden._tick`), when the publisher's thread packed and signed each record
(`Publisher.keep`, which holds the interpreter between switches) and every garbage collection of the warden's process
(serve turns the collector off, so any is an explicit one).

When serve ends they go to `ticks.json` in its `--out` directory, times in seconds after the warden's origin:
`lateness_ms` (tick t's lateness past its deadline, origin + t·τ, in ms), `tick_ms` (tick t's `_tick`, in ms), `publish`
([start, ms, link, window] each), `gc` ([start, generation, ms, objects collected] of each collection of generation 1 or
2, or of `GC_MS` or more) and `gc_counts` (all collections, by generation), with `origin`, `bucket_ms`, `window`,
`slack_ms`, `gc_enabled` and `switch_ms`. The warden's code and schedule are unchanged; this only wraps three methods and
adds a `gc.callbacks` entry.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVE = ROOT / "benchmarks" / "network_traces" / "warden_serve.py"
#: a collection kept in `gc` by its own: any of generation 1 or 2, else one that took at least this long
GC_MS = 1.0


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
    """The wrappers' shared record; `install` puts them on the warden's and the publisher's classes."""

    def __init__(self, clock=time.monotonic) -> None:
        self.clock, self.warden, self.work, self.publish = clock, None, [], []
        self.collections = Collections(clock)

    def install(self, warden_cls: type, publisher_cls: type) -> None:
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

    def doc(self) -> dict:
        w = self.warden
        return {"origin": w.origin, "bucket_ms": w.params.bucket_ms, "window": w.params.window, "slack_ms": w.slack * 1000,
                "gc_enabled": gc.isenabled(), "switch_ms": sys.getswitchinterval() * 1000,
                "lateness_ms": [round(x * 1000, 3) for x in w.lateness], "tick_ms": [round(x * 1000, 3) for x in self.work],
                "publish": [[round(t - w.origin, 4), round(s * 1000, 3), link, window]
                            for t, s, link, window in sorted(self.publish)],
                **self.collections.doc(w.origin)}


def main(argv: list[str]) -> int:
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    sys.path[:0] = [str(SERVE.parent), str(ROOT)]
    import warden_serve

    from verity.core.protocols.certifier import warden

    probe = Probe()
    gc.callbacks.append(probe.collections)
    probe.install(warden.Warden, warden_serve.Publisher)
    try:
        return warden_serve.main(argv)
    finally:
        if out is not None and probe.warden is not None:
            (out / "ticks.json").write_text(json.dumps(probe.doc(), separators=(",", ":")))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
