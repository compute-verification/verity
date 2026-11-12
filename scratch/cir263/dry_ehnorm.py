"""CPU dry run of the eh-norm capture's plumbing: the PTX record on the CPU-compiled PTX, the probes, and one compare chunk with
the kernel's output stood in by the twin."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from tests.properties import glm53_mtp_eh_norm_capture_gpu as M
from verity.definitions.numerics import dsa_norm_rows as NR

ptx = Path.home() / ".research/runs/r20261112-134044-8a1d/out/H6144_B8192_w4.ptx"
rec = M.ptx_record(ptx.read_text())
print("ptx", rec["code_sha256"] == M.PTX_CODE_SHA256, rec["counts"] == M.PTX_COUNTS, rec["counts"], rec["reqntid"])
sass = (ptx.with_suffix(".sass")).read_text()
print("sass", {"ffma_rcp_eps": len(M._SASS_FFMA.findall(sass)), "MUFU.RSQ": sass.count("MUFU.RSQ"), "MUFU.RCP": sass.count("MUFU.RCP")})
if "ptx" in sys.argv[1:]:
    raise SystemExit
probes = {h: M.probe(M.EPS, 20261112 + i) for i, h in enumerate(M.HALVES)}
print("probes", {h: (len(p["x"]), len(p["targets"])) for h, p in probes.items()})
o = M.probe_operands(M.served_operands(20261113), probes)
z = M.zeroed(o["e"], o["pos"])
out = np.concatenate([NR.block_rmsnorm_rows(z, o["ew"], M.EPS, M.BLOCK, M.WARPS)["out"],
                      NR.block_rmsnorm_rows(o["p"], o["hw"], M.EPS, M.BLOCK, M.WARPS)["out"]], axis=1)
d = Path(tempfile.mkdtemp())
from tests.properties import glm53_dsa_capture_gpu as DS
paths = DS._save(d, pos=o["pos"], e=o["e"], p=o["p"], ew=o["ew"], hw=o["hw"], out=out)
r = M._compare({"case": "probe", "span": [0, 32], "sample": [0, 5, 8, 20], "paths": paths})
print(json.dumps(r["halves"], indent=0))
