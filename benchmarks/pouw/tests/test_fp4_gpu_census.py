"""fp4_gpu_census.py's three stages end to end on small synthetic linears (the census's 7, at k = 256..512, one layer,
two variants, two families), the GPU stage on the CPU: its own checks must agree, each linear's tile judged again from
scratch and one per (variant, layer) through the reference's own code.  Needs #580's tree in PEARL_C4_RULE_TREE and
g++; the stages run in a subprocess (this file as a script), so the tree's verity is the only one it imports."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TREE = os.environ.get("PEARL_C4_RULE_TREE", "")
DIMS = {"self_attn.q_proj": (1024, 256), "self_attn.k_proj": (512, 256), "self_attn.v_proj": (512, 256),
        "self_attn.o_proj": (1024, 256), "mlp.gate_proj": (1024, 384), "mlp.up_proj": (1024, 256),
        "mlp.down_proj": (1024, 512)}


def _stages(out: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import fp4_gpu_census as GC
    import fp4_kt_census as KT
    GC.DIMS = DIMS
    KT.FAMILIES = ("gaussian", "spiky-row")

    def weights(shards, i, variant, prep, blocks):
        got = {}
        for lin in KT.LINEARS:
            n, k = DIMS[lin]
            rng = np.random.default_rng([i, *lin.encode(), *variant.encode()])
            w = 0.02 * rng.standard_normal((len(KT.w_blocks(f"{i}.{lin}", n, blocks)), k))
            side = np.array([5, 77]) if variant.startswith("rot") and lin in KT.IN_NORM else None
            if side is not None:
                w[:, side] = 0.0
            got[lin] = ((GC.bf16_codes(w).astype(np.uint32) << 16).view(np.float32), side)
        return got

    KT.layer_weights, KT.Shards = weights, lambda model: None
    np.savez(out / "prep.npz", unused=np.zeros(1))
    common = ["--out", str(out), "--rule-tree", TREE, "--layers", "3", "--variants", "none|pc,rotb8s|pc",
              "--workers", "2", "--prep", str(out / "prep.npz")]
    for cmd in (["prep"], ["gpu", "--device", "cpu"], ["judge"]):
        rc = GC.main(cmd + common)
        if rc:
            return rc
    return 0


@pytest.mark.slow
@pytest.mark.skipif(not (TREE and shutil.which("g++")), reason="needs PEARL_C4_RULE_TREE (#580's tree) and g++")
def test_the_census_stages_agree_with_the_reference(tmp_path):
    subprocess.run([sys.executable, __file__, str(tmp_path)], check=True,
                   env=os.environ | {"FP4_REPLAY_CACHE": str(tmp_path / "lib")})
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["tiles"] == 2 * 7 * 2 * 16 and s["check_disagreements"] == 0
    assert s["checks"] == {"fresh:agree": 14, "reference:agree": 2}
    assert s["cells"]["rotb8s|pc|q_proj|spiky-row"]["debited"] == 16


if __name__ == "__main__":
    sys.exit(_stages(Path(sys.argv[1])))
