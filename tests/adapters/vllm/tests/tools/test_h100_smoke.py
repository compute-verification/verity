"""`h100_smoke.sh`, dry, over a stand-in tree whose ``PY`` records what the script hands it: the serve, the load and the
window commit it would start. The serve's limits follow the adapter's statement (`verity_vllm.inputs.serves`), the load is the
input dataset at the timing ``APPLICATION`` names, through the one sender after the door passed it, and the window commit is
the applications' launcher's (`served_verdicts.sh`'s window_commit()), on the sealed window, off the card."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import input_datasets

from verity_vllm.inputs import serves

ROOT = Path(__file__).resolve().parents[5]
S = serves.SERVES
DATASET = serves.application(str(ROOT / "applications/remote.toml"))[0]
SMOKE = Path("adapters/vllm/tools/served_run/h100_smoke.sh")
VERDICTS = ROOT / "benchmarks/one_stage/served_verdicts.sh"
#: the script's ``PYTHONPATH`` (its ``$TREE`` entries) holds these, linked to the real ones, so its own blocks import the
#: library and the adapter from the tree as on a node, whatever the test's environment installs
PATH_PARTS = ("verity", "tools/verity/cli", "adapters/vllm/verity_vllm", "adapters/torch", "tools/config", "tools/research", "library/datasets",
              "applications", "adapters/vllm/tools/served_run/inputs.py")
ROW = re.search(r"^ROW=(\S+)", (ROOT / SMOKE).read_text(), re.M).group(1)

#: the stand-in ``PY``: each call's argv (and, for the serve, its config) as one JSON line in ``$STUB_LOG``; the
#: script's own Python run for real; every check off a card answered as a card-less machine answers it, a toolkit query
#: as `kit.toolkit` answers it from ``$CUDA_HOME`` alone
STUB = r'''
import json, os, subprocess, sys
a = sys.argv[1:]
rec = {"argv": a}
if a[:2] == ["adapters/vllm/tools/served_run/inputs.py", "application"]:
    sys.exit(subprocess.run([sys.executable, *a]).returncode)
if a[:1] == ["-"]:
    body = sys.stdin.read()
    if "verity_cli" in body:
        got = subprocess.run([sys.executable, "-c", body, *a[1:]], capture_output=True, text=True)
        rec["said"] = got.stdout
        with open(os.environ["STUB_LOG"], "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(got.stdout, end="")
        sys.stderr.write(got.stderr)
        sys.exit(got.returncode)
    if "verity_vllm" not in body and "runpy" not in body:
        sys.exit(subprocess.run([sys.executable, "-c", body, *a[1:]]).returncode)
    if "--protocols-config" in a:
        rec["config"] = json.loads(a[a.index("--protocols-config") + 1])
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(json.dumps(rec) + "\n")
if a[:2] == ["benchmarks/kernels/machine.py", "toolkit"]:
    home = os.environ.get("CUDA_HOME")
    if not home:
        sys.exit(1)
    said = json.load(open(os.path.join(home, "version.json")))["cuda"]["version"]
    if said != a[2] and not said.startswith(a[2] + "."):
        sys.exit(1)
    print(home)
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


def served(calls: list[dict], stdout: str) -> tuple[dict, list[str], list[str]]:
    serve = next(c for c in calls if c["argv"][1:2] == ["serve-openai"])
    load = re.search(r"^LOAD: (.*)$", stdout, re.M).group(1).split()
    commit = next(c["argv"][2:] for c in calls if c["argv"][1:2] == ["benchmarks/one_stage/window_commit.py"])
    return serve, load, commit


def opt(argv: list[str], name: str) -> str:
    assert argv.count(name) == 1, (name, argv)
    return argv[argv.index(name) + 1]


def test_the_serve_follows_the_statement_and_the_load_is_the_dataset_through_the_one_sender(tmp_path):
    p, calls = run_dry(tmp_path)
    assert p.returncode == 0 and "SMOKE DRY OK" in p.stdout, p.stdout[-2000:] + p.stderr[-2000:]
    serve, load, commit = served(calls, p.stdout)
    args = serve["argv"][1:]
    assert [opt(args, k) for k in ("--model", "--revision", "--precision", "--window-class", "--max-num-seqs",
                                   "--max-model-len", "--window-steps", "--window-requests")] == \
        [S.model, S.revision, S.precision, S.window_class, str(S.slots), str(S.prompt + S.outputs), str(S.outputs), str(S.slots)]
    assert "--protocols" not in args and "--scheme" not in args
    assert serve["config"] == {"network-attester": {"order": "release-order"}}
    assert load[:2] == ["adapters/vllm/tools/served_run/inputs.py", "send"]
    assert [opt(load, k) for k in ("--dataset", "--timing")] == [DATASET, "at-once"]
    assert re.fullmatch(r"[0-9a-f]{32}", opt(load, "--release")), load
    assert opt(commit, "--class") == S.window_class
    n = len(input_datasets.open(DATASET))
    assert re.search(rf"^DOOR {DATASET}: {n} records the class {S.window_class} takes; identity generated:", p.stdout, re.M)
    assert "1,24,4" not in (ROOT / SMOKE).read_text() and ".tie" not in (ROOT / SMOKE).read_text()


def application(tmp_path: Path, **params: str) -> str:
    text = (ROOT / "applications/remote.toml").read_text()
    for k, v in params.items():
        text = re.sub(rf'^{k} = ".*"$', f'{k} = "{v}"', text, count=1, flags=re.M)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "app.toml").write_text(text)
    return str(tmp_path / "app.toml")


def test_the_dataset_its_timing_and_the_class_are_the_applications(tmp_path):
    p, calls = run_dry(tmp_path / "rate", APPLICATION=application(tmp_path / "rate", timing="rate:2"))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    serve, load, _ = served(calls, p.stdout)
    assert [opt(load, k) for k in ("--dataset", "--timing")] == [DATASET, "rate:2"]
    assert opt(serve["argv"][1:], "--window-class") == S.window_class
    p, calls = run_dry(tmp_path / "wipe", WIPE_CMD="true")
    assert p.returncode == 3 and "SMOKE REFUSED application: WIPE_CMD is set" in p.stdout, p.stdout + p.stderr
    assert not any(c["argv"][1:2] == ["serve-openai"] for c in calls)


def test_a_dataset_the_door_refuses_and_a_class_past_the_engines_step_are_refused_before_the_serve(tmp_path):
    p, calls = run_dry(tmp_path / "unknown", APPLICATION=application(tmp_path / "unknown", dataset="no_such_dataset"))
    assert p.returncode == 3 and "SMOKE REFUSED dataset: REFUSED unknown-dataset" in p.stdout, p.stdout + p.stderr
    assert not any(c["argv"][1:2] == ["serve-openai"] for c in calls)
    narrow = S.slots * S.prompt - 1
    p, calls = run_dry(tmp_path / "narrow", MNBT=str(narrow))
    assert p.returncode == 3 and "SMOKE REFUSED class" in p.stdout and f"past the engine's {narrow} a step" in p.stdout, p.stdout
    assert not any(c["argv"][1:2] == ["serve-openai"] for c in calls)


def test_a_missing_interpreter_is_refused_by_name_with_the_fix_and_an_interpreter_that_errs_is_never_an_empty_refusal(tmp_path):
    """``PY`` that does not exist on the node is refused first, naming it and the fix; one that exists but fails on the
    application's door refuses with what it printed on stderr, not an empty reason."""
    gone = tmp_path / "no-such-venv" / "bin" / "python"
    p, calls = run_dry(tmp_path / "gone", PY=str(gone))
    assert p.returncode == 3 and not calls, p.stdout + p.stderr
    assert f"SMOKE REFUSED interpreter: the interpreter PY={gone} does not exist on this node: pass PY=/path/to/python" in p.stdout, \
        p.stdout + p.stderr
    broken = tmp_path / "broken"
    broken.write_text("#!/bin/sh\necho 'ModuleNotFoundError: no module named verity_x' >&2\nexit 1\n")
    broken.chmod(0o755)
    p, calls = run_dry(tmp_path / "broken", PY=str(broken))
    assert p.returncode == 3, p.stdout + p.stderr
    assert "SMOKE REFUSED dataset: ModuleNotFoundError: no module named verity_x" in p.stdout, p.stdout + p.stderr


def test_the_window_commit_is_the_applications_launchers_entry_off_the_card(tmp_path):
    p, calls = run_dry(tmp_path)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    _, _, commit = served(calls, p.stdout)
    assert commit[0] == "window-commit"
    assert [opt(commit, k) for k in ("--window", "--class", "--sampling-design")] == ["0", S.window_class, "rate:1/3481648"]
    assert "rate:" not in p.stdout and "(the registered sampling design)" in p.stdout, p.stdout
    script = (ROOT / SMOKE).read_text()
    done = next(x for x in script.splitlines() if '"SMOKE DONE' in x)
    assert done.endswith('no sampled verdict, so no δ or ε; the circuit digest is kept private and NO public value fixes it '
                         'before the coins"'), done
    assert "its circuit registered by the prover; no proof is made, and Lean verifies nothing; " in done
    said = [x.lower() for x in script.splitlines() if re.search(r"\b(say|print)\b|said\+=", x)]
    assert not [x for x in said for p in ("compiled by the registered compiler", "charged", "bounded by", "capped at") if p in x]
    assert 'cp "$VW/registration.json" "$VW/circuit.json" "$DREG" "$OUT/window/"' in script
    assert re.search(r'^names = \{.*"window/"', script, re.M)
    d = opt(commit, "--serve-run")
    assert opt(commit, "--row-dir") == f"{d}/windows/w0"
    assert opt(commit, "--advice") == f"{d}/advice/w0/private/advice.bin"
    assert opt(commit, "--registration") == f"{d}/weights_registration.json"
    assert opt(commit, "--checkpoint-dir").endswith("/hub/models--unsloth--Llama-3.2-1B/snapshots/"
                                                    "9535bd9b1d1dea6acafbdc4813b728796aeb28da")
    assert re.search(r"^LEASE: gpu-lease 1 --wait --max-min 30 --mem-gb 64 -- bash \S+/h100_smoke\.sh card$", p.stdout, re.M), p.stdout
    assert script.index('"${LEASE[@]}"') < script.index('timeout -k 1m "$WC_TIMEOUT" "$PY" "${WC[@]}"')
    assert re.search(r"benchmarks/one_stage/window_commit\.py window-commit --serve-run \"\$D\"", VERDICTS.read_text())


def test_the_serve_makes_each_windows_record_as_window_commit_does_unless_record_0_and_its_plan_is_preflighted_first(tmp_path):
    p, calls = run_dry(tmp_path)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    serve, _, commit = served(calls, p.stdout)
    args = serve["argv"][1:]
    assert [opt(args, k) for k in ("--sampling-design", "--checkpoint-dir")] == \
        [opt(commit, k) for k in ("--sampling-design", "--checkpoint-dir")]
    pre = [i for i, c in enumerate(calls) if c["argv"][:3] == ["-", "1", "serve-openai"]]
    assert len(pre) == 1 and calls[pre[0]]["argv"][2:] == serve["argv"][1:], calls
    script = (ROOT / SMOKE).read_text()
    assert script.index("serve.preflight(opts") < script.index('"${LEASE[@]}" ||')
    p, calls = run_dry(tmp_path / "plan", RECORD="0")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "--sampling-design" not in served(calls, p.stdout)[0]["argv"]


def test_the_development_design_is_refused_by_name_on_a_class_other_than_the_one_its_default_law_draws_from(tmp_path):
    p, _ = run_dry(tmp_path / "default", DEVELOPMENT_DESIGN="1")
    if S.window_class == "4,27,2":
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    else:
        assert p.returncode == 3 and "SMOKE REFUSED class: DEV_LAW draws 4 of class 4,27,2's proof blocks" in p.stdout, p.stdout
    p, _ = run_dry(tmp_path / "given", DEVELOPMENT_DESIGN="1", LAW="rate:4/100")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]


def test_the_launcher_refuses_the_retired_tie_and_scheme_by_name(tmp_path):
    for var, value in (("PEARLPLUS_TIE", "1"), ("SCHEME", "pearl-plus"), ("CLASS", S.window_class),
                       ("DATASET", DATASET)):
        env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "DRY": "1", var: value}
        p = subprocess.run(["bash", str(ROOT / SMOKE)], env=env, capture_output=True, text=True)
        assert p.returncode == 3 and f"h100_smoke.sh: {var} is retired" in p.stderr, p.stdout + p.stderr


def test_the_toolkit_is_12_9_unless_toolkit_names_another_and_cuda_home_must_be_that_one(tmp_path):
    """``TOOLKIT`` (default 12.9) is the version the launcher asks machine.py for; where machine.py gives none (a
    ``CUDA_HOME`` of another version, off a machine whose record lists none) the launcher refuses by name, and where it
    gives one (a ``CUDA_HOME`` of that version) the launcher goes on."""
    _, calls = run_dry(tmp_path / "default")
    assert [c["argv"] for c in calls if c["argv"][:2] == ["benchmarks/kernels/machine.py", "toolkit"]] == \
        [["benchmarks/kernels/machine.py", "toolkit", "12.9"]]
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "gpu-lease").write_text("#!/bin/sh\nexit 0\n")
    (bin_ / "gpu-lease").chmod(0o755)
    for i, (home, said) in enumerate((("cuda-12.9", "12.9.1"), ("cuda-13.0", "13.0.0"))):
        kit = tmp_path / home
        kit.mkdir()
        (kit / "version.json").write_text(json.dumps({"cuda": {"version": said}}))
        p, _ = run_dry(tmp_path / f"t{i}", DRY="0", TOOLKIT="13.0", CUDA_HOME=str(kit), HF_HOME=str(tmp_path / "hf"),
                       PATH=f"{bin_}:{os.environ['PATH']}")
        assert p.returncode == 3, p.stdout + p.stderr
        if said.startswith("12.9"):
            assert "SMOKE REFUSED toolkit: neither CUDA_HOME nor the machine gives CUDA 13.0" in p.stdout, p.stdout + p.stderr
        else:
            assert "SMOKE REFUSED toolkit" not in p.stdout and "SMOKE REFUSED model" in p.stdout, p.stdout + p.stderr


def test_the_wipe_gets_the_pins_and_configuration_the_road_hands_a_piece(tmp_path):
    import shlex

    from verity_cli import run, served_table
    app = ROOT / "applications/onsite_full.toml"
    p, calls = run_dry(tmp_path, APPLICATION=str(app), WIPE_CMD="true")
    assert p.returncode == 0 and "SMOKE DRY OK" in p.stdout, p.stdout[-2000:] + p.stderr[-2000:]
    said = next(c["said"] for c in calls if "said" in c)
    want = served_table.pins_and_configuration(run.load(app, run.registry()))
    assert said.splitlines() == [f"export {k}={shlex.quote(v)}" for k, v in want.items()]
    assert json.loads(want["VERITY_PINS"]) and any(x["protocol"] == "wipe" for x in json.loads(want["VERITY_PINS"]))
