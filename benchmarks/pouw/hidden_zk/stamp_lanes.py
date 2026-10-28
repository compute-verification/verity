"""Scratch (removed before review): one H1 leaf statement's circuit file, the lane (`h1zy.PouwH1LaneZy_v1`) or the combine
(`PouwH1Combine_v1`), as `h1_lanes/run.py`'s `stager` writes it, at K over an R x C tile, under ``HZ_STAGE`` (``fast``:
fast_stage installed) and ``H1_TMPL`` unset. ``python stamp_lanes.py lane|combine K R C PATH``."""
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

LR.RC = (r, c)
work = path.parent / f".work-{kind}-k{k}"
ctx = LR.Ctx(work, "stamp-lanes", k, 5)
ctx.wp = 2 * LL.words(k)
win = LR.window(LR.tile(k), k)
setattr(ctx, kind, LR.derived(win, k, {"lane": LL.LANE, "combine": LL.COMBINE}[kind]))
st = LR.stager(ctx, {"lane": LL.LANE, "combine": LL.COMBINE}[kind])
st.circuit.replace(path)
shutil.rmtree(work, ignore_errors=True)
