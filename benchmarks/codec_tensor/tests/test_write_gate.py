"""The write gate's reading of a profile (`profiled`) and its rule (`passes`), which the stack's gate
(`benchmarks/vllm_encoded/stack_write_gate.py`) runs on its own launches: a launch at its budget passes, one sector over
it or one local store fails."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import write_gate as WG

HEADER = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size",'
          '"Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"')
FUSED = "void wt::fused_kernel<3, 2, true, 16, 2>" + "(const unsigned char *, float *, int)" * 3


def profile(launches: list[dict] | None, kernels: list[tuple[str, dict]]) -> str:
    """Nsight Compute's CSV (`--csv --print-units base`) of `kernels`, each a name and its metrics' values (0 where
    missing), after the child's line of `launches`."""
    lines = ["==PROF== Connected to process 7 (python)"] + ([WG.MARK + json.dumps(launches)] if launches else [])
    lines.append(HEADER)
    for i, (name, values) in enumerate(kernels):
        lines += [f'"{i}","7","python","host","{name}","1","7","(128, 1, 1)","(188, 1, 1)","0","12.0","metrics",'
                  f'"{m}","sector","{values.get(m, 0)}"' for m in WG.METRICS]
    return "\n".join(lines + ["==PROF== Disconnected from process 7"]) + "\n"


def test_a_profile_reads_its_launches_and_each_kernels_sectors():
    launches = [{"kernel": "fused", "receipt_bytes": 64, "red_bytes": 4}]
    out = profile(launches, [(FUSED, {WG.WRITE: "1,024", WG.ALL: "1,039", "dram__bytes_write.sum": "n/a"}),
                             ("void rb::leaves_kernel(int)", {WG.LOCAL: 2})])
    want, kernels = WG.profiled(out)
    assert want == launches and [k["name"] for k in kernels] == [FUSED[:80], "void rb::leaves_kernel(int)"]
    assert kernels[0][WG.WRITE] == 1024.0 and kernels[0]["dram__bytes_write.sum"] is None
    assert WG.sector_bytes(kernels[0], WG.ALL) == 32 * 1039 and WG.sector_bytes(kernels[0], "dram__bytes_write.sum") == 0
    assert WG.sector_bytes(kernels[1], WG.LOCAL) == 64
    assert WG.profiled(out, 90)[1][0]["name"] == FUSED[:90] and WG.profiled(profile(None, []))[0] is None


def test_a_launch_passes_within_its_budget_and_fails_past_it():
    """A fused launch must write its receipts and add its partials: 6,400 + 4,000 bytes, so 1% and 4 KiB more is
    14,600 bytes, 456 sectors and a quarter."""
    launch = {"kernel": "fused", "receipt_bytes": 6400, "red_bytes": 4000}

    def ok(write: int, red: int, local: int = 0) -> bool:
        res = WG.summarize(profile([launch], [(FUSED, {WG.WRITE: write, WG.RED: red, WG.LOCAL: local})]), 0)
        assert res["ok"] == res["rows"][0]["ok"] and res["rows"][0]["l2_write_bytes"] == 32 * write
        return res["ok"]

    assert ok(200, 256) and not ok(201, 256) and not ok(200, 256, local=1) and ok(0, 0)
    assert WG.passes(32 * 456, 0, 10400) and not WG.passes(32 * 457, 0, 10400)
    assert WG.passes(32 * 584, 0, 10400, kernels=2) and not WG.passes(32 * 585, 0, 10400, kernels=2)
    assert not WG.summarize(profile([launch], []), 0)["ok"]
