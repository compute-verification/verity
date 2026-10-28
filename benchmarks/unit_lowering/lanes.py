"""`lanes.py KIND K PATH TREE`: the 32 x 32 leaf's lane statement (`PouwH1LaneZy_v1`, one circuit for its 32 lanes) or its
combine (`PouwH1Combine_v1`) at K, written to PATH as TREE's `benchmarks/pouw/h1_lanes/run.py` stages it (its `stager`), bound to
the Program a served window's verifier derives for it (`derived`), flat unless ``H1_TMPL`` says otherwise. Under
``HZ_STAGE=fast`` (the default) `fast_stage` is installed first, as `pouw_h1_circuits --lower` installs it.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main(kind: str, k: int, path: Path, tree: Path) -> None:
    pouw = tree / "benchmarks" / "pouw"
    sys.path.insert(0, str(pouw / "h1_prove"))
    sys.path.insert(0, str(pouw / "hidden_zk"))
    if os.environ.get("HZ_STAGE", "fast") == "fast":
        import fast_stage
        fast_stage.install(path.parent / f".scratch-{kind}-k{k}")
    spec = importlib.util.spec_from_file_location("h1_lanes_run", pouw / "h1_lanes" / "run.py")
    L = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    ctx = L.Ctx(path.parent, "unit-lowering", k, 0)
    ctx.wp = 2 * L.LL.words(k)
    setattr(ctx, kind, L.derived(L.window(L.tile(k), k), k, kind))
    L.stager(ctx, kind).circuit.rename(path)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ("lane", "combine"):
        sys.exit(__doc__.split("\n")[0])
    main(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
