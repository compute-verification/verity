"""`h100_smoke.sh`, dry, over a stand-in tree whose ``PY`` records what the script hands it: the serve, the load and the
window commit it would start. The serve's limits and the fixed load follow the window class ``CLASS``, whose requests
must start in the window's first step together, and the window commit is the applications' launcher's
(`served_verdicts.sh`'s window_commit()), on the sealed window, off the card."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SMOKE = Path("adapters/vllm/tools/served_run/h100_smoke.sh")
VERDICTS = ROOT / "benchmarks/one_stage/served_verdicts.sh"
#: the script's ``PYTHONPATH`` (its ``$TREE`` entries) holds these, linked to the real ones, so its own blocks import the
#: library and the adapter from the tree as on a node, whatever the test's environment installs
PATH_PARTS = ("verity", "adapters/vllm/verity_vllm", "adapters/torch", "tools/config", "tools/research")
ROW = re.search(r"^ROW=(\S+)", (ROOT / SMOKE).read_text(), re.M).group(1)

#: the stand-in ``PY``: each call's argv (and, for the serve, its config) as one JSON line in ``$STUB_LOG``; the
#: script's own Python run for real; every check off a card answered as a card-less machine answers it
STUB = r'''
import json, os, subprocess, sys
a = sys.argv[1:]
rec = {"argv": a}
if a[:1] == ["-"]:
    body = sys.stdin.read()
    if "verity_vllm" not in body and "runpy" not in body:
        sys.exit(subprocess.run([sys.executable, "-c", body, *a[1:]]).returncode)
    if "--protocols-config" in a:
        rec["config"] = json.loads(a[a.index("--protocols-config") + 1])
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(json.dumps(rec) + "\n")
if a[:2] == ["benchmarks/kernels/machine.py", "toolkit"]:
    sys.exit(1)
'''


def run_dry(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess, list[dict]]:
    tree = tmp_path / "tree"
    (tree / SMOKE).parent.mkdir(parents=True)
    shutil.copy(ROOT / SMOKE, tree / SMOKE)
    for part in PATH_PARTS:
        (tree / part).parent.mkdir(parents=True, exist_ok=True)
        (tree / part).symlink_to(ROOT / part)
    wl = tree / "adapters/vllm/workloads"
    wl.mkdir(parents=True)
    (wl / f"{ROW}.json").write_text("{}")
    (wl / "SHA256SUMS").write_text(f"{hashlib.sha256(b'{}').hexdigest()}  {ROW}.json\n")
    py = tmp_path / "py"
    py.write_text(f"#!{sys.executable}\n{STUB}")
    py.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    p = subprocess.run(["bash", str(tree / SMOKE)], cwd=tree, capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PY": str(py), "STUB_LOG": str(log),
                            "DRY": "1", "CARD_SKU": "h100", "TMPDIR": str(tmp_path), **env})
    return p, [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def served(calls: list[dict]) -> tuple[dict, list[str], list[str]]:
    serve = next(c for c in calls if c["argv"][1:2] == ["serve-openai"])
    load = next(c["argv"][2:] for c in calls if c["argv"][2:3] and c["argv"][2].endswith("serve_load.py"))
    commit = next(c["argv"][2:] for c in calls if c["argv"][1:2] == ["benchmarks/one_stage/window_commit.py"])
    return serve, load, commit


def opt(argv: list[str], name: str) -> str:
    assert argv.count(name) == 1, (name, argv)
    return argv[argv.index(name) + 1]


def test_the_serve_and_the_fixed_load_follow_the_window_class(tmp_path):
    p, calls = run_dry(tmp_path)
    assert p.returncode == 0 and "SMOKE DRY OK" in p.stdout, p.stdout[-2000:] + p.stderr[-2000:]
    serve, load, _ = served(calls)
    args = serve["argv"][1:]
    assert [opt(args, k) for k in ("--window-class", "--max-num-seqs", "--max-model-len", "--window-steps",
                                   "--window-requests")] == ["4,27,2", "4", "29", "2", "4"]
    assert "--protocols" not in args and "--scheme" not in args
    assert serve["config"] == {"network-attester": {"order": "release-order"}}
    assert [opt(load, k) for k in ("--fixed", "--prompt-range", "--window-steps")] == ["4", "27:27", "2"]
    assert re.fullmatch(r"[0-9a-f]{32}", opt(load, "--release")), load
    assert "1,24,4" not in (ROOT / SMOKE).read_text() and ".tie" not in (ROOT / SMOKE).read_text()

    p, calls = run_dry(tmp_path / "other", CLASS="8,16,3")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    serve, load, commit = served(calls)
    args = serve["argv"][1:]
    assert [opt(args, k) for k in ("--window-class", "--max-num-seqs", "--max-model-len", "--window-steps")] == \
        ["8,16,3", "8", "19", "3"]
    assert [opt(load, k) for k in ("--fixed", "--prompt-range", "--window-steps")] == ["8", "16:16", "3"]
    assert opt(commit, "--class") == "8,16,3"


def test_a_class_the_limits_do_not_admit_is_refused(tmp_path):
    for i, (cls, why) in enumerate((("64,27,2", "past the engine's 512 a step"), ("4,27", "S,P,O[,LANES,EVERY]"),
                                    ("4,0,2", "S,P,O[,LANES,EVERY]"))):
        p, calls = run_dry(tmp_path / str(i), CLASS=cls)
        assert p.returncode == 3 and "SMOKE REFUSED class" in p.stdout and why in p.stdout, p.stdout + p.stderr
        assert not any(c["argv"][1:2] == ["serve-openai"] for c in calls)
    p, _ = run_dry(tmp_path / "wide", CLASS="64,27,2", MNBT="2048")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]


def test_the_window_commit_is_the_applications_launchers_entry_off_the_card(tmp_path):
    p, calls = run_dry(tmp_path)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    _, _, commit = served(calls)
    assert commit[0] == "window-commit"
    assert [opt(commit, k) for k in ("--window", "--class", "--sampling-design")] == ["0", "4,27,2", "rate:1/3481648"]
    d = opt(commit, "--serve-run")
    assert opt(commit, "--row-dir") == f"{d}/windows/w0"
    assert opt(commit, "--advice") == f"{d}/advice/w0/private/advice.bin"
    assert opt(commit, "--registration") == f"{d}/weights_registration.json"
    assert opt(commit, "--checkpoint-dir").endswith("/hub/models--unsloth--Llama-3.2-1B/snapshots/"
                                                    "9535bd9b1d1dea6acafbdc4813b728796aeb28da")
    assert re.search(r"^LEASE: gpu-lease 1 --wait --max-min 30 --mem-gb 64 -- bash \S+/h100_smoke\.sh card$", p.stdout, re.M), p.stdout
    script = (ROOT / SMOKE).read_text()
    assert script.index('"${LEASE[@]}"') < script.index('timeout -k 1m "$WC_TIMEOUT" "$PY" "${WC[@]}"')
    assert re.search(r"benchmarks/one_stage/window_commit\.py window-commit --serve-run \"\$D\"", VERDICTS.read_text())


def test_the_launcher_refuses_the_retired_tie_and_scheme_by_name(tmp_path):
    for var in ("PEARLPLUS_TIE", "SCHEME"):
        env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "DRY": "1", var: "1" if var == "PEARLPLUS_TIE" else "pearl-plus"}
        p = subprocess.run(["bash", str(ROOT / SMOKE)], env=env, capture_output=True, text=True)
        assert p.returncode == 3 and f"h100_smoke.sh: {var} is retired" in p.stderr, p.stdout + p.stderr
