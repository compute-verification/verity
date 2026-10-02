"""The hillclimb roll-up (`backends/flock/pod/gemm_hill.py rollup`): every point it appends carries its mode."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

POD = Path(__file__).resolve().parents[1] / "pod"


def _gemm_hill():
    sys.path.insert(0, str(POD))
    try:
        spec = importlib.util.spec_from_file_location("gemm_hill", POD / "gemm_hill.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(POD))


GH = _gemm_hill()
SUB = {"id": "e4m3/GemmCoordinateE4m3_v1/K=2048"}


def _point(run: str, **kw) -> dict:
    return {"schema": GH.POINT_SCHEMA, "subcircuit": SUB, "run_id": run, "commit": "feffeceff" + "0" * 31, "flags": [], **kw}


def test_point_mode_is_the_attempts_zk_setting():
    a = argparse.Namespace(K=2048, dtype="e4m3", step=0, label="x", node="vy-nebius-1", binary_sha=None, question=None,
                           peak_id="rtx-pro-6000-server/e4m3")
    out = {"commit": {"commit": "feffeceff"}, "run_id": "r", "vcpu": 16}
    pt = lambda zk: GH._point(a, {**out, "zk": zk}, {"status": "passed"}, {"flop_per_second": 1.0}, {}, [], [])  # noqa: E731
    assert (pt(True)["mode"], pt(False)["mode"]) == ("zk", "m0")


def test_rollup_writes_each_points_mode(tmp_path):
    f = tmp_path / "rollup.json"
    f.write_text(json.dumps({"schema": GH.ROLLUP_SCHEMA, "subcircuit": SUB, "points": []}))
    for i, pt in enumerate([_point("r-m0", mode="m0"), _point("r-zk", mode="zk", flags=["zk"]),
                            _point("r-old-zk", flags=["packed-statement-unreviewed", "zk"]), _point("r-old-m0")]):
        p = tmp_path / f"p{i}.json"
        p.write_text(json.dumps(pt))
        GH.rollup(argparse.Namespace(point=str(p), file=str(f)))
    got = {p["run_id"]: p["mode"] for p in json.loads(f.read_text())["points"]}
    assert got == {"r-m0": "m0", "r-zk": "zk", "r-old-zk": "zk", "r-old-m0": "m0"}
