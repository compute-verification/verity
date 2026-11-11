"""`server_rate.py`'s rows on a stand-in proof server and device: a row with one refused session, a server process that
exited nonzero, or a prover whose log reports the device out of memory is void (no rate, its reasons in ``void``), and the
run prints a VOID line and exits 1; a clean row keeps its rate and the run exits 0. Each session's record directory is
filed one second after the last (its mtime), so a clean row's rate is exactly one statement a second."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import server_rate as SR  # noqa: E402  (a script beside the suite, importable only once its dir is on the path)

FILED0 = 1_800_000_000  # the first session's filing time, UTC seconds


class Session:
    def __init__(self, label: str, prover: str, d: Path, accepted: bool) -> None:
        self.label, self.prover, self.dir, self.accepted = label, prover, d, accepted
        self.live = {"accepted": True, "e2e_s": 2.0, "prove_total_s": 1.5} if accepted else {"accepted": False,
                                                                                              "why": "R7: no proof"}

    def result(self) -> None:
        return None


class Server:
    """K provers that file session i at FILED0 + i, refuse the sessions in ``refuse``, write ``log`` to the first prover's
    log, close with ``exits`` and report the verifiers placed on ``verifier_cpus``."""

    def __init__(self, out: Path, provers: int, refuse=(), exits=None, log: str = "", verifier_cpus: str | None = None) -> None:
        self.out, self.k, self.refuse, self.verifier_cpus = out, provers, set(refuse), verifier_cpus
        self.exits = exits or {**{f"prover-gpu0-{j}": 0 for j in range(provers)}, "verifier-t": 0}
        (out / "prover-gpu0-0.log").write_text("LIVE\t{}\n" + log)

    @property
    def pids(self) -> dict[str, int]:
        return {}

    def submit(self, st, draw, label: str, verdict=None) -> Session:
        i = int(label.rsplit("/", 1)[1])
        d = self.out / "records" / label.replace("/", "-")
        d.mkdir(parents=True)
        os.utime(d, (FILED0 + i, FILED0 + i))
        return Session(label, f"prover-gpu0-{i % self.k}", d, i not in self.refuse)

    def close(self) -> dict[str, int]:
        return dict(self.exits)


class Device:
    def __init__(self, dev: str) -> None:
        self.samples, self.unparsed = [], []

    def stop(self) -> None:
        pass

    def busy(self, t0: float, t1: float) -> None:
        return None

    def peak_mib(self) -> None:
        return None


def stand_ins(monkeypatch, tmp_path: Path, **server) -> None:
    d = tmp_path / "staged"
    d.mkdir()
    (d / "stage.json").write_text(json.dumps({"n": 1}))
    monkeypatch.setattr(SR, "staged", lambda z, t: (SimpleNamespace(pin="pin"), 1, d))
    monkeypatch.setattr(SR, "Device", Device)
    monkeypatch.setattr(SR, "ProofServer", lambda binary, out, *, provers, **kw: Server(Path(out), provers, **server))


def run(monkeypatch, tmp_path: Path, **server) -> tuple[int, dict]:
    stand_ins(monkeypatch, tmp_path, **server)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["server_rate.py", "--bin", "flock-circuit", "--staged", str(tmp_path), "--types", "t",
                                      "--provers", "1", "--out", str(out), "--device", "0", "--sessions", "8", "--warm", "1",
                                      "--ahead", "1"])
    rc = SR.main()
    [row] = json.loads((out / "rate.json").read_text())
    return rc, row


def test_a_clean_row_keeps_its_rate(monkeypatch, tmp_path):
    rc, row = run(monkeypatch, tmp_path)
    assert (rc, row["statements_per_s"], row["accepted"], "void" in row) == (0, 1.0, 9, False)


def test_a_row_keeps_its_verifier_cpus_and_their_seconds(monkeypatch, tmp_path):
    for d in ("shared", "placed"):
        (tmp_path / d).mkdir()
    _, row = run(monkeypatch, tmp_path / "shared")
    assert (row["verifier_cpus"], row["verifier_cpu_s_per_statement"], row["verifier_cpu_gone"]) == (None, 0.0, [])
    _, row = run(monkeypatch, tmp_path / "placed", verifier_cpus="44-47")
    assert row["verifier_cpus"] == "44-47"


def test_one_refused_session_voids_the_row_and_the_run(monkeypatch, tmp_path, capsys):
    rc, row = run(monkeypatch, tmp_path, refuse={4})
    assert (rc, row["statements_per_s"], row["accepted"], row["refused_n"]) == (1, None, 8, 1)
    assert row["void"] == ["1 of 9 sessions refused"]
    assert [r["label"] for r in row["refused"]] == ["t/k1/4"]
    assert "VOID t K=1: 1 of 9 sessions refused" in capsys.readouterr().out


def test_a_process_that_exited_nonzero_voids_the_row(monkeypatch, tmp_path):
    rc, row = run(monkeypatch, tmp_path, exits={"prover-gpu0-0": 0, "verifier-t": -15})
    assert (rc, row["statements_per_s"], row["void"]) == (1, None, ["exited nonzero: verifier-t rc -15"])


def test_a_prover_out_of_device_memory_voids_the_row(monkeypatch, tmp_path):
    log = "FFI CUDA error out of memory at prove_circuit.cuh:651\n"
    rc, row = run(monkeypatch, tmp_path, log=log)
    assert (rc, row["statements_per_s"], row["void"]) == (1, None, ["out of device memory: prover-gpu0-0"])
