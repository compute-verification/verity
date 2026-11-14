"""prof_summary.py's table of parts on a made-up bench.jsonl: each part's difference of arms, their sum, and an arm
the bench didn't time shown as nan."""
from __future__ import annotations

import json

import prof_summary as P

MS = {"plain": 1.0, "cplain": 1.1, "cduo": 1.3, "tK": 1.2, "formA": 0.1, "bsp": 0.2, "genU": 0.05, "td_bfus": 1.75}


def test_the_parts_are_differences_of_arms(tmp_path, capsys):
    row = {"shape": "o_proj", "m": 8192, "n": 4096, "k": 4096, "gpu": "a GPU", "lt_algo": [7],
           "arms": {k: {"ms": v, "mhz": 2000.0} for k, v in MS.items()}}
    (tmp_path / "bench.jsonl").write_text(json.dumps(row) + "\n")
    P.main(str(tmp_path))
    out = capsys.readouterr().out.splitlines()
    assert "shape o_proj m 8192 n 4096 k 4096, a GPU, plain cuBLASLt 1.0000 ms (algo 7)" in out
    assert "| cduo | 1.3000 | 1.300 | 2000 |" in out
    assert "**td_bfus 1.7500 ms = 1.750 x plain**; parts (each timed alone, hot):" in out
    assert "| custom mainloop, no receipts, against cuBLASLt (cplain - plain) | 0.1000 | 0.100 |" in out
    assert "| duo receipts at L = 32 in the mainloop (cduo - cplain) | 0.2000 | 0.200 |" in out
    assert "| the peel as a tail stage (tK - cplain) | 0.1000 | 0.100 |" in out
    assert "| F's expansion alone (genF) | nan | nan |" in out
    assert "| sum of mainloop + receipts + tail + formA + bsp (over plain) | 0.7000 | 0.700 |" in out
    assert "| td_bfus - plain | 0.7500 | 0.750 |" in out
    assert out[-2:] == ["## ncu (one call, --clock-control none)", ""]


def test_the_interleaving_and_ncus_counters_are_read(tmp_path, capsys):
    """The run's other files: il.jsonl's rows, an ncu CSV read from its header line on (its banner left out) with a second
    CSV's metrics joined by kernel, a stall CSV's reasons from the largest, and a CSV with no header shown by its tail."""
    row = {"shape": "o_proj", "m": 8192, "n": 4096, "k": 4096, "gpu": "a GPU", "lt_algo": [7],
           "arms": {k: {"ms": v, "mhz": 2000.0} for k, v in MS.items()}}
    (tmp_path / "bench.jsonl").write_text(json.dumps(row) + "\n\n")
    (tmp_path / "il.jsonl").write_text(json.dumps({"il": "td_bfus", "edited_ms": 1.8, "plain_ms": 1.0, "ratio": 1.8,
                                                   "mhz": 1990.0}) + "\n\n")
    head = '"ID","Kernel Name","Metric Name","Metric Value"\n'
    (tmp_path / "ncu-td.csv").write_text("==PROF== Connected\n" + head + '"0","gemm","tensor","80.5"\n')
    (tmp_path / "ncu2-td.csv").write_text(head + '"0","gemm","issue","3.5"\n')
    (tmp_path / "ncu-stall-td.csv").write_text(
        head + '"0","gemm","smsp__warps_issue_stalled_wait_per_warp_active.pct","1,200.5"\n'
        '"0","gemm","smsp__warps_issue_stalled_branch_per_warp_active.pct","n/a"\n'
        '"0","gemm","smsp__warps_issue_stalled_sleep_per_warp_active.pct","7"\n')
    for name in ("ncu-broken.csv", "ncu2-broken.csv"):          # profile.sh writes both files of an arm, whatever ncu did
        (tmp_path / name).write_text("==ERROR== no kernels were profiled\n")
    P.main(str(tmp_path))
    out = capsys.readouterr().out.splitlines()
    assert "| td_bfus | 1.8000 | 1.0000 | 1.800 | 1990 |" in out
    assert out[out.index("### td") + 2] == "- gemm: tensor 80.5; issue 3.5"
    assert out[out.index("### stall-td") + 2] == "- gemm: wait 1200.50, sleep 7.00"
    assert "broken: no rows ('==ERROR== no kernels were profiled\\n')" in out
