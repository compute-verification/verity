"""`serve.sh`'s refusal of a run directory that already holds a record, before it touches the machine or writes a file:
a share read after a failed run would otherwise be an earlier run's number. And its serve's card under a caller's
gpu-lease: the lease's, with no lease of its own, as gpu-lease isn't re-entrant. And the Lean verdict's limit, which
grows as the cores fall. And PoUW's row in the record of verdicts, which is the window compose's row
(`served_verdicts.piece`) under the local gate's key pin and lane, not one serve.sh writes itself."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SERVE = Path(__file__).resolve().parents[3] / "benchmarks" / "compute" / "serve.sh"


@pytest.mark.parametrize("held", ["steps.txt", "compute-run.json", "logs", "pieces", "record", "registered"])
def test_a_run_dir_holding_a_record_is_refused_before_any_step(tmp_path, held):
    out = tmp_path / "run" / "out"
    out.mkdir(parents=True)
    if "." in held:
        (out / held).write_text("an earlier run's\n")
    else:
        (out / held).mkdir()
    before = sorted(p.name for p in out.iterdir())
    got = subprocess.run(["bash", str(SERVE)], env={**os.environ, "RUN_DIR": str(tmp_path / "run")},
                         capture_output=True, text=True)
    assert got.returncode == 3 and got.stdout == ""
    assert f"{out} already holds a run's record ({out / held})" in got.stderr and "give RUN_DIR a new directory" in got.stderr
    assert sorted(p.name for p in out.iterdir()) == before


def test_a_try_the_machine_checks_refuse_writes_nothing_so_the_next_is_refused_for_the_same_reason(tmp_path):
    """A refused try that left `out/logs` behind made the next try exit 3 for a record it never wrote: the record's
    directories are made only once the checks of the machine pass. Here the card is no H100 or H200, twice."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lake").write_text("#!/bin/sh\nexit 0\n")
    (bin_ / "nvidia-smi").write_text('#!/bin/sh\necho "0, NVIDIA RTX PRO 6000 Blackwell, 12.0"\n')
    for f in ("lake", "nvidia-smi"):
        (bin_ / f).chmod(0o755)
    keep = {k: v for k, v in os.environ.items() if k not in ("GPU", "GPU_LEASE_UUID", "WORK", "VERITY_BENCH_VENV")}
    env = {**keep, "PATH": f"{bin_}:{keep['PATH']}", "HOME": str(tmp_path / "home"), "RUN_DIR": str(tmp_path / "run")}
    for _ in range(2):
        got = subprocess.run(["bash", str(SERVE)], env=env, capture_output=True, text=True)
        assert got.returncode == 3 and got.stdout == ""
        assert "no H100 or H200 (compute capability 9.0) at GPU=0" in got.stderr
        assert not (tmp_path / "run").exists() and not (tmp_path / "home").exists()


def served(tmp_path, **env) -> list[str]:
    """What serve.sh's card choice and its `lease` print of a serve, with a `gpu-lease` on PATH that only says how it was
    called: serve.sh's own lines, run alone."""
    text = SERVE.read_text()
    card = re.search(r'^\[ -z "\$\{GPU_LEASE_UUID:-\}" \] \|\| GPU=.*$', text, re.M)
    lease = re.search(r"^lease\(\) \{.*?^\}$", text, re.M | re.S)
    assert card and lease, "serve.sh's card line under a lease or its lease() moved: this test reads them"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "gpu-lease").write_text('#!/bin/sh\necho "gpu-lease $*"\n')
    (bin_ / "gpu-lease").chmod(0o755)
    script = f'GPU=${{GPU-0}}\n{card[0]}\n{lease[0]}\necho "GPU=$GPU"\nlease echo "served on ${{CUDA_VISIBLE_DEVICES:-none}}"\n'
    keep = {k: v for k, v in os.environ.items() if k not in ("GPU", "GPU_LEASE_UUID", "CUDA_VISIBLE_DEVICES")}
    got = subprocess.run(["bash", "-c", script], env={**keep, "PATH": f"{bin_}:{keep['PATH']}", **env},
                         capture_output=True, text=True, check=True)
    return got.stdout.splitlines()


def test_outside_a_lease_the_serve_takes_one_of_the_card_gpu_names(tmp_path):
    assert served(tmp_path) == ["GPU=0", 'gpu-lease 1 --wait --on 0 --max-min 30 -- echo served on none']


def test_under_a_callers_lease_the_serve_runs_on_its_card_and_takes_no_lease(tmp_path):
    """`research run --gpus 1`, or a caller's own `gpu-lease 1 --cores ...`, holds the card: a second lease of it would wait
    forever, so the serve runs on the lease's card (CUDA_VISIBLE_DEVICES, GPU its UUID for the up-front check)."""
    got = served(tmp_path, GPU_LEASE_UUID="GPU-67cd8250,GPU-099c8642", CUDA_VISIBLE_DEVICES="3,4", GPU="0")
    assert got == ["GPU=GPU-67cd8250", "served on 3,4"]


@pytest.mark.parametrize(("cores", "given", "limit"), [
    (96, None, "3600"), (192, None, "3600"), (36, None, "9600"), (20, None, "17280"), (20, "5000", "5000")])
def test_the_lean_verdicts_limit_grows_as_the_cores_fall(tmp_path, cores, given, limit):
    """On 20 cores the Lean verdict ran past a flat 3,600 s and the run failed with Python's accept in hand: the limit
    is 3,600 s on 96 cores and as much longer as there are fewer, and LEAN_TIMEOUT stands where given."""
    line = re.search(r"^CORES=\$\(nproc\) && LEAN_TIMEOUT=.*$", SERVE.read_text(), re.M)
    assert line, "serve.sh's LEAN_TIMEOUT line moved: this test reads it"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nproc").write_text(f"#!/bin/sh\necho {cores}\n")
    (bin_ / "nproc").chmod(0o755)
    keep = {k: v for k, v in os.environ.items() if k != "LEAN_TIMEOUT"}
    env = {**keep, "PATH": f"{bin_}:{keep['PATH']}", **({"LEAN_TIMEOUT": given} if given else {})}
    got = subprocess.run(["bash", "-c", f'{line[0]}\necho "$LEAN_TIMEOUT"'], env=env, capture_output=True, text=True,
                         check=True)
    assert got.stdout.strip() == limit


#: a stand-in for benchmarks/one_stage/served_verdicts.py whose `piece` returns a row naming what it was asked
PIECE_STUB = '''import json
def piece(key, pieces_dir, run=None, **of_record):
    asked = {"key": key, "pieces": str(pieces_dir), "run": run, **of_record}
    return {"verdict": "missing", "owner": "compute-accounting", "why": "stub: " + json.dumps(asked, sort_keys=True)}, (None, "real")
'''


def verdicts(tmp_path, gate: dict | None, run_id: str = "") -> tuple[dict, str]:
    """serve.sh's `verdicts` run alone in a tree whose served_verdicts.py is `PIECE_STUB`: the record it writes and its
    line."""
    fn = re.search(r"^verdicts\(\) \{.*?^\}$", SERVE.read_text(), re.M | re.S)
    assert fn, "serve.sh's verdicts() moved: this test reads it"
    tree, out = tmp_path / "tree", tmp_path / "out"
    (tree / "benchmarks" / "one_stage").mkdir(parents=True)
    (tree / "benchmarks" / "one_stage" / "served_verdicts.py").write_text(PIECE_STUB)
    (out / "record").mkdir(parents=True)
    (out / "compute-run.json").write_text(json.dumps({"challenger": "local"}))
    if gate is not None:
        (out / "auditor-gate.json").write_text(json.dumps(gate))
    script = f'PY={sys.executable}\nOUT={out}\nREC=$OUT/record\n{fn[0]}\nverdicts\n'
    got = subprocess.run(["bash", "-c", script], cwd=tree, env={**os.environ, "RESEARCH_RUN_ID": run_id},
                         capture_output=True, text=True, check=True)
    return json.loads((out / "record" / "verdicts.json").read_text()), got.stdout


def test_pouws_row_is_the_window_composes_under_the_local_gates_key_pin_and_lane(tmp_path):
    """The row was a fixed `missing` that named no condition of record: it is `served_verdicts.piece`'s on out/pieces,
    with no SPOT window registration (none is made here), the local gate's key pin and lane, and the run's id."""
    doc, line = verdicts(tmp_path, {"key_sha512": "ab" * 64, "lane": "vllm", "pin": "p"}, "r20261116-000000-abcd")
    asked = {"key": "work", "pieces": str(tmp_path / "out" / "pieces"), "run": "r20261116-000000-abcd", "spot": None,
             "key_sha512": "ab" * 64, "lane": "vllm"}
    assert doc["protocols"] == {"work": {"verdict": "missing", "owner": "compute-accounting",
                                         "why": "stub: " + json.dumps(asked, sort_keys=True)}}
    assert doc["format"] == "verity-bench/compute-verdicts/v1" and doc["run"] == {"challenger": "local"}
    assert line.startswith("work: missing: stub: ")


def test_with_in_process_coins_the_row_is_asked_under_no_key_pin_or_lane(tmp_path):
    """`COINS=in-process` writes no auditor-gate.json: the row is asked with no pin and no lane, so it can't be of record."""
    doc, _ = verdicts(tmp_path, None)
    assert '"key_sha512": null, "lane": null' in doc["protocols"]["work"]["why"]
    assert '"run": null' in doc["protocols"]["work"]["why"]


def steps() -> list[str]:
    """serve.sh's step names in the order its lines run them."""
    return re.findall(r"^\s*(?:\[ [^]]* \] \|\| )?step ([a-z-]+) ", SERVE.read_text(), re.M)


def test_the_circuit_record_and_the_period_come_before_the_serve_and_its_close_before_the_sample():
    """PoUW's row of record holds the serve's salt session to a Pearl-C period of the lane's own that committed the
    circuit record, and the sample to the lane's first reserve after that period closed: so the circuit record is built
    before the period opens, the period opens before the serve, and it closes after the serve's passes are retained and
    before python-verify takes the sample."""
    got = steps()
    for name in ("serve-phi", "circuit", "period-open", "serve", "retain", "period-close", "python-verify", "verdicts"):
        assert got.count(name) == 1, (name, got)
    at = got.index
    assert at("serve-phi") < at("circuit") < at("gate") < at("period-open") < at("serve") < at("retain") \
        < at("period-close") < at("python-verify") < at("verdicts")


def period(tmp_path, verb: str) -> tuple[list[str], Path]:
    """serve.sh's `period VERB` run alone with a `$PY` that only prints its arguments: those arguments, and W."""
    fn = re.search(r"^period\(\) \{.*?^\}$", SERVE.read_text(), re.M | re.S)
    assert fn, "serve.sh's period() moved: this test reads it"
    py, w = tmp_path / "py", tmp_path / "w"
    py.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    py.chmod(0o755)
    env = {**os.environ, "PY": str(py), "W": str(w), "PIECE": str(tmp_path / "piece"), "REC": str(tmp_path / "rec"),
           "SCHEME": "pearl-c-h100-v1-h3", "PEARLC_CHALLENGER_GATE": "gate cmd", "PEARLC_CHALLENGER_LANE": "vllm",
           "RESEARCH_RUN_ID": "r20261116-000000-abcd"}
    got = subprocess.run(["bash", "-c", f"{fn[0]}\nperiod {verb}"], env=env, capture_output=True, text=True, check=True)
    return got.stdout.splitlines(), w


def test_the_period_opens_on_the_runs_circuit_record_and_closes_once_the_serves_salt_session_has_ended(tmp_path):
    """The open commits the circuit record's bytes (`pearlplus-period open --circuit-record`), the record
    `pearlplus_of_record` holds the period's circuit to; the close reads the serve's salt session from its served.json
    as one window's and waits for the gate's ledger to end it, since the gate closes no period a window of which runs."""
    opened, _ = period(tmp_path, "open")
    assert opened == ["benchmarks/one_stage/served_verdicts.py", "pearlplus-period", "open", str(tmp_path / "piece"),
                      "--gate", "gate cmd", "--name", "pearlplus-r20261116-000000-abcd", "--scheme", "pearl-c-h100-v1-h3",
                      "--circuit-record", str(tmp_path / "rec" / "circuit-record.json")]
    closed, w = period(tmp_path, "close")
    assert closed == ["benchmarks/one_stage/served_verdicts.py", "pearlplus-period", "close", str(tmp_path / "piece"),
                      "--gate", "gate cmd", "--lane", "vllm", "--windows", str(w / "pearl"), "--wait", "300"]
    assert os.readlink(w / "pearl" / "w0" / "served.json") == str(tmp_path / "piece" / "served.json")
