"""Scratch (removed before review): what the stamped walk keeps, by allocation site, when it ends (tracemalloc), for the H1
lane or combine at K over an R x C tile (`stamp_lanes.py`). ``python stamp_probe.py lane|combine K R C``, from
integrations/vllm; ``PROBE_FRAMES`` frames a trace (default 12)."""
import os
import runpy
import sys
import tempfile
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
tracemalloc.start(int(os.environ.get("PROBE_FRAMES", "12")))
sys.path.insert(0, str(HERE))
import stamp  # noqa: E402

orig = stamp.StampLayout.__init__


def init(self, st, *a, **kw):
    snap = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(False, tracemalloc.__file__)])
    tracemalloc.stop()
    out = sys.stderr
    print(f"PROBE traced {sum(s.size for s in snap.statistics('filename')) / 2 ** 30:.2f} GiB", file=out)
    for s in snap.statistics("lineno")[:30]:
        print(f"PROBE line {s.size / 2 ** 20:9.1f} MiB {s.count:>10} {s.traceback[0]}", file=out)
    for s in snap.statistics("traceback")[:12]:
        print(f"PROBE tb {s.size / 2 ** 20:9.1f} MiB {s.count:>10}", file=out)
        for line in s.traceback.format(most_recent_first=True)[:2 * 12]:
            print("PROBE     " + line, file=out)
    out.flush()
    raise SystemExit(0)


stamp.StampLayout.__init__ = init
kind, k, r, c = sys.argv[1:5]
sys.argv = [str(HERE / "stamp_lanes.py"), kind, k, r, c, str(Path(tempfile.mkdtemp()) / "circuit.txt")]
runpy.run_path(str(HERE / "stamp_lanes.py"), run_name="__main__")
