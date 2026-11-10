"""bench/run_config.py: --dry-run prints the exact stage commands.  The cheap CPU stages on the local m1 capture are
`test_run_config_dry_run_of_record`."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from verity_vllm.pipeline import match as RC

ROOT = Path(__file__).resolve().parents[2]

BASE = ["--config-id", "cid", "--profile", "prof_x", "--workload", "wl.json", "--case", "B1", "--out", "/r/RUN", "--python-gpu", "/pod/py311",
        "--python-cpu", "/pod/py312", "--checkpoint-dir", "/ck/snap"]


def dry(*extra: str) -> dict[str, list[str]]:
    out = subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "match", *BASE, "--dry-run", *extra], capture_output=True, text=True,
                         cwd=ROOT, check=True).stdout
    cmds: dict[str, list[str]] = {}
    for ln in out.splitlines():
        if ln.startswith("[") and "]" in ln:
            st, rest = ln[1:].split("]", 1)
            cmds[st] = shlex.split(rest) if not rest.strip().startswith(("SKIP", "in-process", "write")) else [rest.strip()]
    return cmds


def test_dry_run_gpu_phase():
    c = dry("--phase", "gpu")
    assert list(c) == ["capture", "control", "census", "nonint", "gpu_done"]
    cap = c["capture"]
    assert cap[:4] == ["/pod/py311", "-m", "verity_vllm.pipeline.cli", "m1-capture"]
    assert cap[cap.index("--snapshot-steps") + 1] == "0,1,2,3"        # default, contiguous from 0
    assert cap[cap.index("--out") + 1] == "/r/RUN/capture" and cap[cap.index("--case") + 1] == "B1"
    assert cap[cap.index("--max-num-seqs") + 1] == "8" and "--workload" in cap
    ctl = c["control"]
    assert "--no-observer" in ctl and ctl[ctl.index("--out") + 1] == "/r/RUN/control" and "--snapshot-steps" not in ctl
    cen = c["census"]
    assert cen[1:4] == ["-m", "verity_vllm.pipeline.cli", "census"] and cen[cen.index("--out") + 1] == "/r/RUN/census" and "--no-traces" in cen
    ni = c["nonint"]
    assert ni[1:4] == ["-m", "verity_vllm.pipeline.cli", "noninterference"] and ni[ni.index("--out") + 1] == "/r/RUN/nonint"
    assert ni[ni.index("--case") + 1] == "B1"                        # expected_layers resolves from the case
    assert ni[ni.index("--accepted-ctl") + 1] == "/r/RUN/control/tokens.json"   # the control run, never a workload list
    assert not any("accepted" in a and "control" not in a for a in ni if not a.startswith("--"))   # no workload-authored list anywhere
    assert all(x[0] == "/pod/py311" for k, x in c.items() if k != "gpu_done")


@pytest.mark.slow
def test_dry_run_cpu_phase():
    c = dry("--phase", "cpu", "--laptop", "--full-replay", "--seed", "12345")
    assert list(c) == ["resolve", "census_reconcile", "nonint_compare", "compare", "fold", "fold_2", "golden", "gates"]
    log = "/r/RUN/capture/log.jsonl.gz"
    r = c["resolve"]
    assert r[:5] == ["/pod/py312", "-m", "verity_vllm.pipeline.cli", "resolve-log", log] and r[r.index("--out") + 1] == "/r/RUN/resolution" and "--strict" not in r
    cr = c["census_reconcile"]
    assert cr[cr.index("--reconcile") + 1] == "/r/RUN/census/census.json" and cr[cr.index("--report") + 1] == "/r/RUN/census_reconcile"
    nc = c["nonint_compare"]
    assert "--compare" in nc and nc[nc.index("--case") + 1] == "B1" and nc[nc.index("--accepted-ctl") + 1] == "/r/RUN/control/tokens.json"
    cp = c["compare"]
    assert cp[1:4] == ["-m", "verity_vllm.pipeline.cli", "fold-compare"] and cp[cp.index("--out") + 1] == "/r/RUN/compare.json"
    assert cp[cp.index("--snapshots") + 1] == "/r/RUN/capture" and cp[cp.index("--safetensors") + 1] == "/ck/snap/model*.safetensors"
    for st, out in (("fold", "/r/RUN/fold_summary.json"), ("fold_2", "/r/RUN/fold_summary_2.json")):
        f = c[st]
        assert f[1:6] == ["-m", "verity_vllm.pipeline.cli", "match", "--worker", "fold"] and f[f.index("--out") + 1] == out
    assert c["golden"][1:6] == ["-m", "verity_vllm.pipeline.cli", "golden", "--check", "--out"] and c["golden"][-1] == "/r/RUN/golden.json"
    assert c["gates"][0].startswith("in-process") and "/r/RUN/gates.json" in c["gates"][0] and "out/gen/cards/cid.json" in c["gates"][0]
    assert all(x[0] == "/pod/py312" for k, x in c.items() if k != "gates")


@pytest.mark.slow
def test_dry_run_no_laptop_and_skip():
    c = dry("--phase", "cpu", "--skip", "replay,census,nonint,golden")
    assert "--max-arg-leaves" not in c["fold"]
    for st in ("census_reconcile", "nonint_compare", "golden", "replay_a", "replay_chain"):
        assert c[st][0].startswith("SKIP")
    assert c["resolve"][0] == "/pod/py312" and c["compare"][0] == "/pod/py312"


@pytest.mark.slow
def test_the_stage_names_without_a_command_run_nothing_and_a_skip_naming_them_is_printed():
    retired = ("root_policy", "wiring", "replay_a", "replay_chain")
    c = dry("--phase", "cpu")                                         # no --seed: no stage samples a sample
    assert not set(retired) & set(c)
    c = dry("--phase", "cpu", "--skip", ",".join(retired))
    assert all(c[st][0].startswith("SKIP") for st in retired)


@pytest.mark.slow
def test_dry_run_all_and_snapshot_steps():
    c = dry("--phase", "all", "--snapshot-steps", "0,1,2", "--seed", "7")
    assert list(c)[:4] == ["capture", "control", "census", "nonint"]
    assert c["capture"][c["capture"].index("--snapshot-steps") + 1] == "0,1,2"
    assert "fold" in c and "gates" in c
    c = dry("--phase", "all", "--snapshot-steps", "0,1,2", "--full-replay", "--seed", "7")
    assert not any(k.startswith("replay") for k in c)
    with pytest.raises(subprocess.CalledProcessError):
        dry("--snapshot-steps", "0,2")           # not contiguous from 0


@pytest.mark.slow
def test_dry_run_snapshot_caps_reach_the_capture_only():
    # D87 cap and the cov-moe dispatcher-call cap are capture flags.  The D87 cap is not
    # absent by default: run_config DERIVES it (rows x widest row x 2 B, never below m1_capture's 32 MiB floor) so every prefill MLP row of a
    # >= 1024-prompt row has snapshot evidence; with an unresolvable workload / checkpoint (this dry run) the derivation is the floor.  The
    # dispatcher-call cap stays absent by default, and an explicit CLI value still wins (below).
    c = dry("--phase", "gpu")
    cap = c["capture"]
    assert cap[cap.index("--snapshot-max-bytes") + 1] == str(RC.SNAPSHOT_CAP_FLOOR) and "--snapshot-max-bytes-op" not in cap
    c = dry("--phase", "gpu", "--snapshot-max-bytes", str(256 << 20), "--snapshot-max-bytes-op", str(48 << 20))
    cap = c["capture"]
    assert cap[cap.index("--snapshot-max-bytes") + 1] == str(256 << 20) and cap[cap.index("--snapshot-max-bytes-op") + 1] == str(48 << 20)
    assert all("--snapshot-max-bytes" not in c[st] for st in ("control", "census", "nonint"))


@pytest.mark.slow
def test_dry_run_existing_capture_paths():
    c = dry("--phase", "cpu", "--log", "/data/x/log.jsonl.gz", "--snapshots", "/data/x", "--control-tokens", "/data/ctl.json", "--other-log", "/data/y.jsonl.gz",
            "--seed", "7")
    assert c["resolve"][4] == "/data/x/log.jsonl.gz" and c["compare"][c["compare"].index("--snapshots") + 1] == "/data/x"
    assert c["nonint_compare"][c["nonint_compare"].index("--accepted-ctl") + 1] == "/data/ctl.json"
    assert c["fold_other"][c["fold_other"].index("--log") + 1] == "/data/y.jsonl.gz"


@pytest.mark.slow
def test_dry_run_writes_nothing(tmp_path):
    subprocess.run([sys.executable, "-m", "verity_vllm.pipeline.cli", "match", *BASE[:8], "--out", str(tmp_path / "RUN"), "--dry-run", "--seed", "7"],
                   capture_output=True, text=True, cwd=ROOT, check=True)
    assert not (tmp_path / "RUN").exists()


def test_skip_groups():
    assert RC.Plan._skip_set("replay,fold,resolve") == {"replay_a", "replay_chain", "replay_b", "replay_c", "fold", "fold_2", "fold_other", "resolve"}


def test_run_stage_records_failure_and_tees_log(tmp_path):
    rec = RC.run_stage("boom", [sys.executable, "-c", "import sys; print('hello'); print('err', file=sys.stderr); sys.exit(3)"], tmp_path / "logs/boom.log",
                       max_rss_gb=None, timeout_s=None)
    assert rec["status"] == "failed" and rec["returncode"] == 3 and "exit 3" in rec["reason"]
    text = (tmp_path / "logs/boom.log").read_text()
    assert "hello" in text and "err" in text and text.startswith("$ ")


def test_run_stage_timeout(tmp_path):
    # the stage blocks on a pipe nobody writes to: it ends only when run_stage's wall budget kills it
    rec = RC.run_stage("slow", [sys.executable, "-c", "import os; os.read(os.pipe()[0], 1)"], tmp_path / "slow.log", max_rss_gb=None, timeout_s=1.0)
    assert rec["status"] == "timeout" and rec["reason"] == "wall > 1 s"


# --- the cheap CPU stages on the local m1 capture ------------------------------------------------------------------------
