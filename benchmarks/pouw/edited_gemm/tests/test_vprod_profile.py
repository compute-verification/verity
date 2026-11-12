"""vprod_profile.py on a made-up run directory: an arm's row from the hot bench and from ncu's one call (executed warp
instructions an entry, issue, DRAM and L2), an arm ncu didn't profile shown with dashes, the sums' marginal (vfold less
vfold_x) as seconds and instructions an entry, and the same numbers in --json and --toml; no bench is exit 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import vprod_profile as V

RAW = ('"ID","gpu__time_duration.sum","smsp__issue_active.avg.pct_of_peak_sustained_active",'
       '"dram__throughput.avg.pct_of_peak_sustained_elapsed","lts__throughput.avg.pct_of_peak_sustained_elapsed"\n'
       '"","msecond","%","%","%"\n')


def write_run(o: Path) -> None:
    """Two shapes: s = 2048 with ncu's call of vfold (400 MMAs and 600 adds) and vfold_x (300 and 500) over 1,000 entries,
    and s = 8192 with the bench alone."""
    def arms(**ms: float) -> dict:
        return {k: {"ms": v} for k, v in ms.items()}

    (o / "vbench.jsonl").write_text("a line that is not JSON\n" + "".join(json.dumps(b) + "\n" for b in (
        {"s": 2048, "kvh": 8, "entries": 1000, "digits": 2, "arms": arms(vedit=1.0, vsum=2.0, vfold=3.0, vfold_x=2.5)},
        {"s": 8192, "kvh": 8, "entries": 4000, "digits": 2, "arms": arms(vsum=8.0)})))
    (o / "ncu_vp").mkdir()
    for arm, mma, add, raw in (("vfold", 400, 600, '"0","2.9","41.5","12.5","30.5"\n'),
                               ("vfold_x", 300, 500, '"0","2.4","40.5","11.5","29.5"\n')):
        base = o / "ncu_vp" / f"vp-2048-{arm}"
        Path(f"{base}.sass.csv").write_text('==PROF== a banner\n"Source","Instructions Executed"\n'
                                            f'"IMMA.16832.S8.S8 R0, R1, R2","{mma}"\n"@P0 IADD3 R1, R2, R3","{add}"\n')
        Path(f"{base}.raw.csv").write_text(RAW + raw)


def run(monkeypatch, capsys, *args: str) -> tuple[int, list[str]]:
    monkeypatch.setattr(sys, "argv", ["vprod_profile.py", *args])
    return V.main(), capsys.readouterr().out.splitlines()


def test_each_arms_row_and_the_sums_marginal(tmp_path, monkeypatch, capsys):
    write_run(tmp_path)
    rc, out = run(monkeypatch, capsys, str(tmp_path))
    assert rc == 0 and out[2:] == [
        "| 2048x8 | vedit | 1.0000 | - | - | - | - | - | - |",
        "| 2048x8 | vsum | 2.0000 | - | - | - | - | - | - |",
        "| 2048x8 | vfold | 3.0000 | 2.9000 | 1.0000 | 32.00 | 41.5 | 12.5 | 30.5 |",
        "| 2048x8 | vfold_x | 2.5000 | 2.4000 | 0.8000 | 25.60 | 40.5 | 11.5 | 29.5 |",
        "| 2048x8 | sums (vfold - vfold_x) | 5.000e-07 s/entry | | 0.2000 | 6.40 | | | |  (r2 vsum 2.000e-06 s/entry)",
        "| 8192x8 | vsum | 8.0000 | - | - | - | - | - | - |",
        "| 8192x8 | sums (vfold - vfold_x) | nan s/entry | | nan | nan | | | |  (r2 vsum 2.000e-06 s/entry)"]


def test_json_and_toml_hold_the_same_numbers(tmp_path, monkeypatch, capsys):
    write_run(tmp_path)
    rc, _ = run(monkeypatch, capsys, str(tmp_path), "--json", str(tmp_path / "v.json"), "--toml", str(tmp_path / "v.toml"))
    got = json.loads((tmp_path / "v.json").read_text())
    assert rc == 0 and list(got) == ["2048x8", "8192x8"] and got["8192x8"]["ncu"] == {}
    vfold = got["2048x8"]["ncu"]["vfold"]
    assert (vfold["warp_inst"], vfold["thread_inst_per_entry"], vfold["ms"]) == (1000, 32.0, 2.9)
    assert vfold["class_per_entry"] == {"ALU": 0.6, "MMA": 0.4}
    assert (tmp_path / "v.toml").read_text().splitlines()[1:] == [
        ("s2048x8 = { entries = 1000, vedit = 1.00000, vsum = 2.00000, vfold = 3.00000, vfold_x = 2.50000, "
         "t_vsum = 5.000e-07, sums_thread_inst = 6.40, vfold_issue_pct = 41.5 }"),
        "s8192x8 = { entries = 4000, vsum = 8.00000, t_vsum = nan, sums_thread_inst = nan }"]


def test_no_bench_is_exit_1(tmp_path, monkeypatch, capsys):
    rc, out = run(monkeypatch, capsys, str(tmp_path))
    assert rc == 1 and len(out) == 2
