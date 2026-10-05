"""sweep_vu: the stopping rule, the plateau stamp and the commit.seconds alias the views read.

    uv run pytest archive/direct/ligero/sweep_vu_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from archive.direct.ligero import sweep_vu as S  # noqa: E402


def test_stops_when_two_doublings_gain_under_two_percent() -> None:
    assert not S.done([100.0, 150.0])
    assert not S.done([100.0, 150.0, 160.0])
    assert S.done([100.0, 150.0, 152.0, 152.9])
    assert S.done([100.0, 101.0, 101.9])


def test_throughput_prefers_end_to_end() -> None:
    doc = {"measurements": [{"name": "t.total", "value": 2.0}, {"name": "vu.count", "value": 4096},
                            {"name": "end_to_end.vu_per_second", "value": 1000.0}]}
    assert S.throughput(doc) == 1000.0
    assert S.throughput({"measurements": doc["measurements"][:2]}) == 2048.0
    e2e = doc["measurements"][:2] + [{"name": "e2e.vu_per_second", "value": 900.0}]
    assert S.throughput({"measurements": e2e}) == 900.0


def test_stamp_adds_the_blocks_views_read() -> None:
    doc = {"contention": {"contended": False},
           "measurements": [{"name": "commitment.seconds", "value": 0.5}, {"name": "end_to_end.seconds", "value": 3.0}]}
    S.stamp(doc, {"id": "s", "point": 2, "plateau": True}, 5)
    assert doc["protocol"]["warm"] is True and doc["protocol"]["runs"] == 5 and doc["protocol"]["contended"] is False
    assert doc["sweep"]["plateau"] is True
    m = S._meas(doc)
    assert m["commit.seconds"] == 0.5 and m["e2e.seconds"] == 3.0


def test_an_input_set_caps_the_sweep_with_a_last_point_at_its_size(tmp_path, monkeypatch) -> None:
    import json

    def fake_call(cmd, stdout=None, stderr=None):
        n = int(cmd[cmd.index("--total-vus") + 1])
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"measurements": [{"name": "e2e.vu_per_second", "value": 1000.0 + n}], "contention": {"contended": False}}))
        return 0

    monkeypatch.setattr(S.subprocess, "call", fake_call)
    for cap, want in ((1920, [1024, 1920]), (6272, [1024, 2048, 4096, 6272]), (4096, [1024, 2048, 4096])):
        d = tmp_path / f"cap{cap}"
        assert S.main(["--relation", "bf16-ampere-x4-k2048+sha256", "--out-dir", str(d), "--cap", str(cap), "--dump", "none", "--",
                       "--reps", "5"]) == 0
        sw = json.loads((d / "sweep.json").read_text())
        assert [p["total_vus"] for p in sw["points"]] == want and sw["plateau_total_vus"] == cap
        assert sw["stopped"].startswith(f"reached the input set's {cap}")
