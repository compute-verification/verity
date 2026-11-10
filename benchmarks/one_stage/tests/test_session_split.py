"""`session_split.py` on a stand-in proof server and device whose sessions' parts are planted as fixed + per-block lines: the
fit recovers each line, the rest is ``e2e_s`` less the other parts, the order is cold at the largest k, one warm-up at each
other k and timed rounds ascending then descending, every draw is a subset of the population of its k, and one refused
session voids the run (no medians or fit, a VOID line, exit 1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import server_rate as SR  # noqa: E402  (scripts beside the suite, importable only once their dir is on the path)
import session_split as SS  # noqa: E402

PLANT = {"witness_s": (2.0, 0.004), "prove_total_s": (0.2, 0.001), "wait_e2e_s": (2.9, 0.0), "rest_s": (0.1, 0.002)}


class Session:
    def __init__(self, label: str, k: int, d: Path, accepted: bool) -> None:
        self.label, self.prover, self.dir, self.accepted = label, "prover-gpu0-0", d, accepted
        self.live = {p: a + b * k for p, (a, b) in PLANT.items() if p != "rest_s"}
        self.live["e2e_s"] = sum(a + b * k for a, b in PLANT.values())
        self.seconds = self.live["e2e_s"] + 0.05

    def result(self) -> None:
        return None


class Server:
    def __init__(self, out: Path, refuse=()) -> None:
        self.out, self.refuse, self.draws = out, set(refuse), []
        (out / "prover-gpu0-0.log").write_text("statement unit/t instances=256 vus_per_block=1 blocks=256 k_log=27 nbl=1 m=35 "
                                               "regions=1 delta=1 built in 40.25s\n")

    @property
    def pids(self) -> dict[str, int]:
        return {}

    def submit(self, st, draw, label: str) -> Session:
        assert len(draw["units"]) == draw["k"] == len(set(draw["units"])) and set(draw["units"]) <= set(range(256))
        self.draws.append(draw["k"])
        i = int(label.rsplit("/", 1)[1])
        d = self.out / "records" / label.replace("/", "-")
        d.mkdir(parents=True)
        return Session(label, draw["k"], d, i not in self.refuse)

    def close(self) -> dict[str, int]:
        return {"prover-gpu0-0": 0, "verifier-t": 0}


class Device:
    def __init__(self, dev: str) -> None:
        self.samples, self.unparsed = [], []

    def stop(self) -> None:
        pass

    def busy(self, t0: float, t1: float) -> None:
        return None

    def peak_mib(self) -> None:
        return None


def run(monkeypatch, tmp_path: Path, **server) -> tuple[int, dict, list[int]]:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "stage.json").write_text(json.dumps({"n": 256}))
    (stage / "pin.txt").write_text("pin")
    made = []
    monkeypatch.setattr(SR, "Device", Device)
    monkeypatch.setattr(SS, "ProofServer", lambda binary, out, **kw: made.append(Server(Path(out), **server)) or made[-1])
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["session_split.py", "--bin", "flock-circuit", "--stage", str(stage), "--out", str(out),
                                      "--device", "0", "--draws", "64,128,256", "--rounds", "2"])
    rc = SS.main()
    return rc, json.loads((out / "split.json").read_text()), made[0].draws


def test_the_fit_recovers_each_planted_part(monkeypatch, tmp_path):
    rc, rec, draws = run(monkeypatch, tmp_path)
    assert rc == 0 and draws == [256, 128, 64, 64, 128, 256, 256, 128, 64]
    for p, (a, b) in PLANT.items():
        assert (round(rec["fit"][p]["fixed_s"], 3), round(rec["fit"][p]["per_block_s"], 5)) == (a, b), p
    assert rec["median"]["128"]["witness_s"] == 2.512 and rec["median"]["128"]["sessions"] == 2
    assert [c["k"] for c in rec["cold"]] == [256] and [w["k"] for w in rec["warmup"]] == [128, 64]
    assert rec["builds"] == [{"log": "prover-gpu0-0", "statement": "unit/t", "instances": 256, "built_s": 40.25}]


def test_one_refused_session_voids_the_run(monkeypatch, tmp_path, capsys):
    rc, rec, _ = run(monkeypatch, tmp_path, refuse={5})
    assert (rc, rec["fit"], rec["median"], rec["refused_n"]) == (1, None, None, 1)
    assert rec["void"] == ["1 of 9 sessions refused"]
    assert "VOID 1 of 9 sessions refused" in capsys.readouterr().out
