"""`tick.sh` on a stand-in node: the venv's python is a stub that records each call and writes what the job's next step
reads, so that what the job serves, what it calibrates the tick on and what the encoder charges are checked without GPUs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
CORPUS = "benchmarks/network_e2e/traces/corpus.py"
STUB = """\
#!{python}
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
if args[:1] == ["-c"]:
    if "import vllm" in args[1]:
        print(os.environ["STUB_VLLM"])
        sys.exit(0)
    sys.argv = ["-c", *args[2:]]
    exec(args[1])
    sys.exit(0)
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if Path(args[0]).name == "corpus.py" and args[1] == "run":
    out = Path(args[args.index("--out") + 1])
    (out / args[args.index("--traces") + 1]).mkdir(parents=True, exist_ok=True)
    (out / ".site").mkdir(exist_ok=True)
    (out / "meta.json").write_text(json.dumps({{"eos_token_ids": [154820, 154827]}}))
elif Path(args[0]).name == "corpus.py" and args[1] == "summarize":
    for d in Path(args[2]).iterdir():
        if d.is_dir() and not d.name.startswith("."):
            (d / "steps.jsonl").write_text("")
"""


def job(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess, list[list[str]], Path]:
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    (run / "inputs" / "c4.json.gz").write_bytes(b"")
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "config.json").write_text("{}")
    tools = tmp_path / "venv" / "bin"
    tools.mkdir(parents=True)
    (tools / "python").write_text(STUB.format(python=sys.executable))
    (tools / "nvidia-smi").write_text("#!/bin/sh\necho 0, NVIDIA H200, 0 MiB\n")
    for f in tools.iterdir():
        f.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    p = subprocess.run(["bash", str(HERE / "tick.sh")], cwd=ROOT, capture_output=True, text=True,
                       env={"PATH": f"{tools}:/usr/bin:/bin", "HOME": str(tmp_path), "RESEARCH_RUN_DIR": str(run),
                            "CHECKPOINT": str(tmp_path / "ck"), "VENV": str(tools.parent), "GPU_LEASE_UUID": "GPU-0",
                            "STUB_LOG": str(log), "STUB_VLLM": "0.28.1rc1.dev472+gd9105ea80.cu129", **env})
    calls = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return p, calls, run / "out"


def test_tick_job_serves_arrival_then_the_tick_calibrated_on_it_then_charges_both(tmp_path):
    p, calls, out = job(tmp_path, RATE="2.5")
    assert p.returncode == 0, p.stdout + p.stderr
    common = ["--model-id", "glm53", "--variant", "greedy", "--checkpoint", str(tmp_path / "ck"), "--c4",
              str(tmp_path / "run" / "inputs" / "c4.json.gz"), "--traces", "heavy", "--rate", "2.5", "--requests", "3250",
              "--startup-s", "1800"]
    assert calls == [[CORPUS, "run", *common, "--out", f"{out}/corpus/arrival"],
                     [CORPUS, "summarize", f"{out}/corpus/arrival"],
                     [CORPUS, "run", *common, "--admit", "tick", "--tick-from", f"{out}/corpus/arrival/heavy", "--out",
                      f"{out}/corpus/tick"],
                     [CORPUS, "summarize", f"{out}/corpus/tick"],
                     ["-P", "benchmarks/network_e2e/encodings/v1.py", f"arrival={out}/corpus/arrival/heavy",
                      f"tick={out}/corpus/tick/heavy", "--T", "6000", "--stops", "154820,154827", "--out",
                      f"{out}/advice.json"]]
    assert not list(out.glob("corpus/*/.site")) and (out / "nvidia-smi.txt").exists()


def test_tick_job_calibrates_on_a_given_trace_and_serves_at_least_1000_requests(tmp_path):
    prior = tmp_path / "prior" / "heavy"
    prior.mkdir(parents=True)
    (prior / "steps.jsonl").write_text("")
    p, calls, out = job(tmp_path, RATE="0.5", TICK_FROM=str(prior), TRACE="chunked")
    assert p.returncode == 0, p.stdout + p.stderr
    (run, summarize, v1) = calls
    assert run[:2] == [CORPUS, "run"] and summarize == [CORPUS, "summarize", f"{out}/corpus/tick"]
    assert [run[run.index(k) + 1] for k in ("--requests", "--traces", "--tick-from")] == ["1000", "chunked", str(prior)]
    assert v1[2:4] == [f"arrival={prior}", f"tick={out}/corpus/tick/chunked"]


def test_tick_job_refuses_another_vllm_before_it_takes_the_gpus(tmp_path):
    p, calls, out = job(tmp_path, RATE="1", STUB_VLLM="0.11.0")
    assert p.returncode == 2 and "has vLLM 0.11.0, not d9105ea80" in p.stdout
    assert calls == [] and not out.exists()
