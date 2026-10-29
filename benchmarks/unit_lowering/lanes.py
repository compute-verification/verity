"""`lanes.py KIND K R C PATH TREE STAGING`: an R x C leaf's lane statement (`PouwH1LaneZy_v1`, one circuit for all its lanes,
each 4 rows by 8 columns) or its combine (`PouwH1Combine_v1`) at K, written to PATH as TREE's `benchmarks/pouw/h1_lanes/run.py`
stages it (its `stager`, at its leaf ``RC`` set to R x C), bound to the Program a served window's verifier derives for it
(`derived`), flat unless ``H1_TMPL`` says otherwise. At STAGING ``fast`` `fast_stage` is installed first, as
`pouw_h1_circuits --lower` installs it under ``HZ_STAGE=fast``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def stage(kind: str, k: int, r: int, c: int, scratch: Path, tree: Path):
    """``kind``'s `UnitStager` at K on the R x C leaf, its circuit written under ``scratch``, with whatever staging this process
    has installed."""
    spec = importlib.util.spec_from_file_location("h1_lanes_run", tree / "benchmarks" / "pouw" / "h1_lanes" / "run.py")
    L = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    L.RC = (r, c)
    ctx = L.Ctx(scratch, "unit-lowering", k, 0)
    ctx.wp = 2 * L.LL.words(k)
    setattr(ctx, kind, L.derived(L.window(L.tile(k), k), k, kind))
    return L.stager(ctx, kind)


def main(kind: str, k: int, r: int, c: int, path: Path, tree: Path, staging: str) -> None:
    pouw = tree / "benchmarks" / "pouw"
    sys.path.insert(0, str(pouw / "h1_prove"))
    sys.path.insert(0, str(pouw / "hidden_zk"))
    if staging == "fast":
        import fast_stage
        fast_stage.install(path.parent / f".scratch-{kind}-k{k}")
    stage(kind, k, r, c, path.parent, tree).circuit.rename(path)


if __name__ == "__main__":
    if len(sys.argv) != 8 or sys.argv[1] not in ("lane", "combine") or sys.argv[7] not in ("fast", "library"):
        sys.exit(__doc__.split("\n")[0])
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6]), sys.argv[7])
