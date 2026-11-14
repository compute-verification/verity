"""summary.py's Markdown on a made-up run directory: each kernel's ratios read from its own shape's rows (hot items
against plain and plainq, the interleaved and cold ratios), the parts at o_proj as differences of arms, the gates' lines
and ncu's counters; a directory with no file prints the headings alone."""
from __future__ import annotations

import json
from pathlib import Path

import summary as S

O_PROJ, QKV, SMALL = (8192, 4096, 4096), (8192, 6144, 4096), (128, 256, 128)
MS = {"plain": 1.0, "plainq": 1.25, "e_plain": 1.1, "e_l7": 1.3, "x13_l2": 1.2, "edited:e_l7": 1.5, "editedf:e_l7": 2.0,
      "wedit": 0.1, "qplain": 0.2, "qdither": 0.25, "qwe": 0.3}


def rows(*xs: dict) -> str:
    return "".join(json.dumps(x) + "\n" for x in xs)


def bench(shape: tuple, scale: float) -> dict:
    m, n, k = shape
    return {"m": m, "n": n, "k": k, "arms": {a: {"ms": scale * v, "mhz": 2000.0} for a, v in MS.items()}}


def il(shape: tuple, arm: str, base: str, ratio: float, **more) -> dict:
    m, n, k = shape
    return {"m": m, "n": n, "k": k, "il": arm, "base": base, "ratio": ratio, **more}


def summary(o: Path, capsys) -> list[str]:
    S.main(o)
    return capsys.readouterr().out.splitlines()


def test_a_kernels_ratios_come_from_its_own_shape(tmp_path, capsys):
    """Two shapes with the same hot ratios and different interleaved rows: each row of the table shows its shape's."""
    (tmp_path / "bench.jsonl").write_text(rows(bench(O_PROJ, 1.0), bench(QKV, 2.0)))
    (tmp_path / "il.jsonl").write_text(rows(
        il(O_PROJ, "e_l7", "plain", 1.31), il(O_PROJ, "e_l7", "plain", 1.4, cold=True),
        il(O_PROJ, "editedf:e_l7", "plainq", 1.61), il(QKV, "e_l7", "plain", 1.33), il(QKV, "edited:e_l7", "plain", 1.52),
        il(QKV, "editedf:e_l7", "plainq", 1.7, cold=True), il(SMALL, "e_l7", "plain", 1.25)))
    out = summary(tmp_path, capsys)
    assert "## o_proj (8192 x 4096 x 4096): hot items (median of the reps, ms)" in out
    assert "| e_l7 | 1.3000 | 1.300 | 1.040 | 2000 |" in out and "| e_l7 | 2.6000 | 1.300 | 1.040 | 2000 |" in out
    assert "| o_proj | e_plain | none | 1.100 |  |  |  |  |  |  |  |" in out
    assert "| o_proj | e_l7 | duo32 | 1.300 | 1.310 | 1.400 | 1.500 |  | 1.600 | 1.610 |  |" in out
    assert "| o_proj | x13_l2 | xin13 | 1.200 |  |  |  |  |  |  |  |" in out
    assert "| qkv | e_l7 | duo32 | 1.300 | 1.330 |  | 1.500 | 1.520 | 1.600 |  | 1.700 |" in out
    assert [x for x in out if x.startswith("| ") and "| plain |" in x and "o_proj" in x] == []
    assert "- IL 128x256x128 e_l7 / plain: 1.2500" in out


def test_the_parts_at_o_proj_are_differences_of_arms(tmp_path, capsys):
    (tmp_path / "bench.jsonl").write_text(rows(bench(QKV, 3.0), bench(O_PROJ, 1.0)))
    out = summary(tmp_path, capsys)
    at = out.index("## o_proj by part (hot items, ms)")
    assert out[at + 2:at + 6] == [
        "- mainloop + 2-atom tail against cuBLASLt (e_plain - plain): +0.1000",
        "- receipts in e_l7 (duo32; e_l7 - e_plain): +0.2000",
        "- receipts in x13_l2 (xin13; x13_l2 - e_plain): +0.1000",
        "- wedit 0.1000; qdither - qplain +0.0500; qwe 0.3000"]


def test_the_gates_and_ncus_counters_are_read(tmp_path, capsys):
    m, n, k = SMALL
    (tmp_path / "gate.jsonl").write_text("a line that is not JSON\n" + rows(
        {"m": m, "n": n, "k": k, "device_ok": True, "control_rejected": True,
         "device": {"e_l7": {"y_bad": 0, "r_bad": 0}, "x_l2": {"y_bad": 0, "r_bad": 3}}}))
    (tmp_path / "ref.jsonl").write_text(rows(
        {"m": m, "n": n, "k": k, "ok": False, "r9_probe": {"n": 5, "mismatch": 0, "r9_rejected": 3}, "r9_seed_skips": 1,
         "tiles": {"0,1": {"X": True, "Y": True, "S_range": [-5, 7], "R": {"e_l7": True, "x_l2": False}}}}))
    (tmp_path / "ncu-o.csv").write_text('==PROF== a banner\n"ID","Metric Name","Metric Value"\n"0","tensor","80.5"\n'
                                        '"0","issue","3.5"\n')
    (tmp_path / "ncu-none.csv").write_text("==PROF== a banner and no table\n")
    out = summary(tmp_path, capsys)
    assert ("- device 128 x 256 x 128: device_ok True, control_rejected True, 2 checks, failing "
            "{'x_l2': {'y_bad': 0, 'r_bad': 3}}") in out
    assert ("- CPU ref 128 x 256 x 128: ok False; receipts 1/2 tile-kernels bit-exact; X, Xq, P_A, W', P_B, peel, Y True; "
            "S range [[-5, 7]]; key_ok_r9 on the device: 0 of 5 words differ from receipt_folds_r9's (3 key_ok words "
            "rejected); the seed's key_ok words r9 skipped: 1") in out
    assert out[-3:] == ["### ncu-o", "", "tensor 80.5; issue 3.5"]


def test_a_directory_with_no_file_prints_the_headings(tmp_path, capsys):
    out = summary(tmp_path, capsys)
    assert out[0] == f"# edited GEMM, cheaper receipts and Llama-8B's linears ({tmp_path})"
    assert [x for x in out if x.startswith("#")][1:] == ["## gates", "## the ratio table"]
    assert out[-1] == "|---|---|---|---|---|---|---|---|---|---|---|"
