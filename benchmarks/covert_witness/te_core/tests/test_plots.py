"""te_core.plots draws P1-P4 and A2 and writes the numbers behind them, from ledger rows, with and without a
from-scratch curve."""
import json

import pytest

pytest.importorskip("matplotlib")

from te_core import plots, policy  # noqa: E402

SHAPE = {"d_model": 512, "n_layer": 6, "vocab": 50304, "ctx": 1024, "parameters": 45_029_888, "tokens": 154_206_208,
         "tokens_per_param": 3.42, "forward_macs_per_token": 51_342_080}


def rows():
    out = []
    for i, (direction, macs, bpb, e) in enumerate([("ref-optim", 4e14, 1.17, 9e7), ("ref-optim", 1e15, 1.15, 9e7),
                                                    ("lowrank", 6e14, 1.18, 8e4), ("sparse", 2e15, 1.16, 1e6)]):
        log = {"by": {"fwd/theta": macs / 3, "bwd/grad": 2 * macs / 3, "fwd/updated": macs / 30},
               "ops": {"fwd theta mm [[1]]": [1, macs / 3], "bwd grad mm [[1]]": [1, 2 * macs / 3]}}
        out.append({"id": f"r{i}", "start": "s1", "start_bpb": 1.2, "val_bpb": bpb, "direction": direction,
                    "mechanism": direction, "run_id": f"run{i}", "checkpoint_step": 9, "measurement_commit": "pin",
                    "macs_log": log, "output": {"e_bytes": e}, "annotation": {"steps": 10}, "written_ts": i})
    return out


@pytest.mark.parametrize("scratch", [False, True])
def test_the_four_plots_and_their_numbers(tmp_path, monkeypatch, scratch):
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}}}))
    monkeypatch.setattr(plots, "Ledger", lambda root: type("L", (), {"index": staticmethod(rows)})())
    args = ["--out", str(tmp_path / "out"), "--pin", "pin", "--manifest", str(man), "--drop", "0.02"]
    if scratch:
        c0, _ = policy.c0(SHAPE)
        f = tmp_path / "F.json"
        f.write_text(json.dumps({"s1": [[c0 * 0.5, 1.25], [c0, 1.2], [c0 * 3, 1.15], [c0 * 10, 1.1]]}))
        args += ["--from-scratch", str(f), "--g", "1.5"]
    assert plots.main(args) == 0
    for n in ("P1", "P2", "P3", "P4", "A2"):
        assert (tmp_path / "out" / f"{n}.png").stat().st_size > 10_000
        data = json.loads((tmp_path / "out" / f"{n}.json").read_text())
        assert data["pin"] == ["pin"] and any("stub" in s for s in data["stubs"])
    p2 = json.loads((tmp_path / "out" / "P2.json").read_text())["data"]
    assert p2["x"] == ("g" if scratch else "drop") and p2["starts"]["s1"]["points"]
    assert all(p["S"] >= 1 for p in p2["starts"]["s1"]["points"])
    p4 = json.loads((tmp_path / "out" / "P4.json").read_text())["data"]["s1"]
    assert any(p["full_training"] for p in p4["points"])
    assert min(p["p"] for p in p4["points"]) == pytest.approx(1.0)
    a2 = json.loads((tmp_path / "out" / "A2.json").read_text())["data"]["s1"]
    first = a2["targets"][0]
    assert [s["of"]["id"] for s in first["complete"]] == ["r2", "r0"]
    assert first["complete"][-1]["p"] == pytest.approx(1.0)


def test_a2_steps_are_the_least_compute_up_to_each_delta():
    """A record lowers the curve only if it beats every record of no larger Δ, so the steps never rise, ties at one Δ
    giving one step."""
    def r(i, delta, gates):
        return {"id": i, "gates": gates, "measured": type("M", (), {"delta_bits": delta})()}

    steps = plots.least_by_delta([r("a", 10, 5.0), r("b", 1, 9.0), r("c", 10, 3.0), r("d", 100, 4.0), r("e", 1, 7.0)])
    assert [(s["delta_le"], s["gates"], s["of"]["id"]) for s in steps] == [(1, 7.0, "e"), (10, 3.0, "c")]
