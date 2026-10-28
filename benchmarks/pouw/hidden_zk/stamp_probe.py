"""Scratch (removed before review): what the stamped walk keeps when it ends, for the H1 lane or combine at K over an R x C
tile (`stamp_lanes.py`). ``python stamp_probe.py lane|combine K R C``, from integrations/vllm. ``PROBE_FRAMES`` > 0 traces
allocations (tracemalloc, slow); 0, the default, prints pymalloc's block count, the stamp's own array bytes, the gc-tracked
objects by type, and what a collection frees."""
import collections
import gc
import os
import runpy
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMES = int(os.environ.get("PROBE_FRAMES", "0"))
if FRAMES:
    tracemalloc.start(FRAMES)
sys.path.insert(0, str(HERE))
import stamp  # noqa: E402

orig = stamp.StampLayout.__init__
out = sys.stderr


def say(*a):
    print("PROBE", *a, file=out, flush=True)


def traced():
    snap = tracemalloc.take_snapshot().filter_traces([tracemalloc.Filter(False, tracemalloc.__file__)])
    tracemalloc.stop()
    say(f"traced {sum(s.size for s in snap.statistics('filename')) / 2 ** 30:.2f} GiB")
    for s in snap.statistics("lineno")[:30]:
        say(f"line {s.size / 2 ** 20:9.1f} MiB {s.count:>10} {s.traceback[0]}")


def cheap(st):
    T = st.templates
    say(f"blocks={sys.getallocatedblocks()} {stamp._mem()}")
    arr = sum(t.cols.nbytes + t.ptr.nbytes + t.cst.nbytes + t.vis.nbytes for t in T)
    binds = sum(b.nbytes for b in st.binds)
    say(f"templates={len(T)} arrays_gib={arr / 2 ** 30:.3f} binds={len(st.binds)} binds_gib={binds / 2 ** 30:.3f} "
        f"keys={len(st.keys)} key_items={sum(len(k) for k in st.keys)} outs={sum(len(t.outs) for t in T)}")
    t0 = time.time()
    objs = gc.get_objects()
    c = collections.Counter(type(o).__name__ for o in objs)
    say(f"tracked={len(objs)} in {time.time() - t0:.1f}s: " + ", ".join(f"{k}={v}" for k, v in c.most_common(25)))
    del objs
    t0 = time.time()
    n = gc.collect()
    say(f"collect freed {n} in {time.time() - t0:.1f}s; blocks={sys.getallocatedblocks()} {stamp._mem()}")
    gc.disable()


def init(self, st, *a, **kw):
    if FRAMES:
        traced()
        raise SystemExit(0)
    cheap(st)
    orig(self, st, *a, **kw)
    say(f"after layout blocks={sys.getallocatedblocks()} {stamp._mem()}")


stamp.StampLayout.__init__ = init
kind, k, r, c = sys.argv[1:5]
tmp = Path(tempfile.mkdtemp())
sys.argv = [str(HERE / "stamp_lanes.py"), kind, k, r, c, str(tmp / "circuit.txt")]
say(f"start blocks={sys.getallocatedblocks()} {stamp._mem()}")
try:
    runpy.run_path(str(HERE / "stamp_lanes.py"), run_name="__main__")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
