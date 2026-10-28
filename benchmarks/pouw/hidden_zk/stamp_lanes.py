"""Scratch (removed before review): one H1 leaf statement's circuit file, the lane (`h1zy.PouwH1LaneZy_v1`) or the combine
(`PouwH1Combine_v1`), as `h1_lanes/run.py`'s `stager` writes it, at K over an R x C tile, under ``HZ_STAGE`` (``fast``:
fast_stage installed) and ``H1_TMPL`` unset. ``python stamp_lanes.py lane|combine K R C PATH``; ``chunk K N C PATH``: a 32 x 32
leaf's lane chunk C of N (`h1zy.lane_chunk`), `h1_lanes/j10r.py`'s stager on its statement rows."""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "h1_lanes"))
sys.path.insert(0, str(HERE))

kind, k, r, c, path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
import os  # noqa: E402

if os.environ.get("HZ_STAGE", "fast") == "fast":
    import fast_stage
    fast_stage.install(path.parent / f".scratch-{kind}-k{k}")
import run as LR  # noqa: E402
from verity.core.protocols.pouw import leaf_lanes as LL  # noqa: E402

if kind == "chunk":
    from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
    from verity.core.protocols.pouw.circuit import h1zy as Z  # noqa: E402
    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402

    n, ch = r, c
    d = Z.lane_chunk(HW.h1_tile(k, (32, 32)), n, ch)
    bits = {p: sum(x for _f, _at, x in fs) for p, fs in Z.statement_ports(d)}
    rows = [(p, V2, bits[p], ps) for p, ps in Z.statement_rows(d)]
    work = path.parent / f".work-{kind}-k{k}"
    st = LR.U.stager(d, f"unit/pouw-h1-lane-chunk{ch}of{n}", rows, [(V2, (0,))], None, circuit=work / "circuit.txt")
    st.circuit.replace(path)
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(0)
LR.RC = (r, c)
work = path.parent / f".work-{kind}-k{k}"
ctx = LR.Ctx(work, "stamp-lanes", k, 5)
ctx.wp = 2 * LL.words(k)
win = LR.window(LR.tile(k), k)
setattr(ctx, kind, LR.derived(win, k, {"lane": LL.LANE, "combine": LL.COMBINE}[kind]))
st = LR.stager(ctx, {"lane": LL.LANE, "combine": LL.COMBINE}[kind])
st.circuit.replace(path)
shutil.rmtree(work, ignore_errors=True)
