"""`h100_smoke.sh`'s Pearl+ arm, dry, over a stand-in tree whose ``PY`` records what the script hands it: the serve and
the load it would start, and the config it writes. The arm serves Pearl+'s tie class, the fixed load, whose requests
must start in the window's first step together, so it is held to the applications' launcher (`served_verdicts.sh`):
release order as the network attester's setting in the serve's one config, and the load numbered by a release run."""
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
#: script's own JSON edit run for real; every check off a card answered as a card-less machine answers it
STUB = r'''
import json, os, subprocess, sys
a = sys.argv[1:]
rec = {"argv": a}
if a[:1] == ["-"]:
    body = sys.stdin.read()
    if "providing(" in body or ("verity_vllm" not in body and "runpy" not in body):
        sys.exit(subprocess.run([sys.executable, "-c", body, *a[1:]]).returncode)
    if "--protocols-config" in a:
        rec["config"] = json.load(open(a[a.index("--protocols-config") + 1]))
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(json.dumps(rec) + "\n")
if a[:2] == ["benchmarks/kernels/machine.py", "toolkit"]:
    sys.exit(1)
if a[:1] == ["-c"] and "tie" in a[1]:
    print("4,27,2")
if len(a) > 1 and a[0].endswith("served_passes.py"):
    if a[1] == "verify":
        sys.exit(1)
    json.dump({"pearl-plus": {"device": "cuda", "kernel": "k", "cubin": "c"}}, open(a[a.index("--out") + 1], "w"))
    print("{}")
'''


def run_dry(tmp_path: Path) -> list[dict]:
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
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PY": str(py), "STUB_LOG": str(log), "DRY": "1",
           "SCHEME": "pearl-plus", "PEARLPLUS_TIE": "1", "CARD_SKU": "h100", "TMPDIR": str(tmp_path)}
    p = subprocess.run(["bash", str(tree / SMOKE)], cwd=tree, env=env, capture_output=True, text=True)
    assert p.returncode == 0 and "SMOKE DRY OK" in p.stdout, p.stdout[-2000:] + p.stderr[-2000:]
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_the_pearl_plus_arm_serves_and_sends_the_fixed_load_as_the_applications_launcher_does(tmp_path):
    calls = run_dry(tmp_path)
    serve = next(c for c in calls if c["argv"][1:2] == ["serve-openai"])
    load = next(c["argv"][1:] for c in calls if c["argv"][1:2] and c["argv"][1].endswith("serve_load.py"))
    args = serve["argv"][1:]
    assert args[args.index("--protocols") + 1] == "pearl-plus" and args.count("--protocols") == 1
    assert sorted(serve["config"]) == ["network-attester", "pearl-plus"]
    assert serve["config"]["network-attester"] == {"order": "release-order"}
    assert re.fullmatch(r"[0-9a-f]{32}", load[load.index("--release") + 1]), load
    assert load[load.index("--fixed") + 1] == "4"
    verdicts = VERDICTS.read_text()
    assert "--protocols pearl-plus" in verdicts and '"order": "release-order"' in verdicts and "network-attester" in verdicts
    assert re.search(r'serve_load\.py".*\n.*--release "\$release"', verdicts)
