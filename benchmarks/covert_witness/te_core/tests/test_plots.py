"""te_core.plots writes the console's five panels (P1-P4 and A2), each with a `reading` column carrying both readings'
rows, and draws each one's figure from its rows, with and without a from-scratch curve."""
import json

import pytest

pytest.importorskip("matplotlib")

from te_core import formalism, plots, policy

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


def panels(out) -> dict[str, list[dict]]:
    """Each panel's rows, after checking it is the console's panel body with a row per reading."""
    got = {}
    for n in plots.NAMES:
        assert (out / f"{n}.png").stat().st_size > 10_000
        p = json.loads((out / f"{n}.json").read_text())
        assert (p["id"], p["kind"]) == (f"training/{n}", "table") and p["updated_at"] and p["source"]
        assert "reading" in p["columns"] and len(p["columns"]) <= 24
        assert all(len(r) == len(p["columns"]) for r in p["rows"]) and p["rows"]
        assert "stub" in p["note"] and "neither is the headline" in p["note"]
        got[n] = plots.frame(p)
    return got


@pytest.mark.parametrize("scratch", [False, True])
def test_the_five_panels_carry_both_readings(tmp_path, monkeypatch, scratch):
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
    got = panels(tmp_path / "out")
    for n, rs in got.items():
        assert {r["reading"] for r in rs} == set(formalism.READINGS), n
    p1 = got["p1-loss-vs-compute"]
    assert {"honest", "covert"} <= {r["curve"] for r in p1} and ("scratch" in {r["curve"] for r in p1}) == scratch
    p2 = got["p2-slowdown-vs-gain"]
    assert all(r["S"] >= 1 for r in p2 if r["S"] is not None) and any(r["S"] is not None for r in p2)
    assert all((r["g"] is not None) == scratch for r in p2) and {r["basis"] for r in p2} <= {"fit", "target", None}
    p3 = got["p3-policy-plane"]
    assert {r["mark"] for r in p3} >= {"inference block", "c=4096w k=10kB", "server block"}
    assert all(r["S"] >= 1 for r in p3 if r["S"] is not None)
    for name in formalism.READINGS:
        p4 = [r for r in got["p4-decomposition"] if r["reading"] == name and r["target_pct"] == -2]
        assert any(r["algorithm"] == "ref-optim/ref-optim" for r in p4)
        assert min(r["p"] for r in p4) == pytest.approx(1.0)
        assert all(r["S"] == pytest.approx(r["p"] * r["s"]) for r in p4)
    a2 = got["a2-efficiency-floor"]
    by = {name: [{k: v for k, v in r.items() if k != "reading"} for r in a2 if r["reading"] == name]
          for name in formalism.READINGS}
    assert by[formalism.LITERAL] == by[formalism.TAIL_CHARGED]
    first = [r for r in by[formalism.LITERAL] if r["target_pct"] == -0.5]
    assert [r["record"] for r in first] == ["r2", "r0"]
    assert all(r["fit_run_k_bits"] >= r["delta_bits"] for r in first)


def test_one_reading_gives_only_its_rows(tmp_path, monkeypatch):
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}}}))
    monkeypatch.setattr(plots, "Ledger", lambda root: type("L", (), {"index": staticmethod(rows)})())
    assert plots.main(["--out", str(tmp_path / "out"), "--pin", "pin", "--manifest", str(man),
                       "--reading", formalism.TAIL_CHARGED]) == 0
    for p in plots.NAMES:
        rs = plots.frame(json.loads((tmp_path / "out" / f"{p}.json").read_text()))
        assert {r["reading"] for r in rs} == {formalism.TAIL_CHARGED}


def test_a_staircase_is_the_least_compute_reaching_each_loss():
    """A point lowers the curve only if it beats every point of no larger compute, ties at one compute giving one
    step."""
    pts = [(5.0, 1.2, "a"), (1.0, 1.3, "b"), (3.0, 1.1, "c"), (4.0, 1.15, "d"), (1.0, 1.25, "e")]
    assert [p[2] for p in plots.staircase(pts)] == ["e", "c"]
