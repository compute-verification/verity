"""`run.sh`'s dry runs over a stand-in tree."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest
from verity_cli import run as verity_run
from verity_cli import served_table

HERE = Path(__file__).resolve().parents[5] / "adapters" / "vllm" / "tools" / "workload"
WORKLOAD_RUN = Path("adapters/vllm/tools/workload")
ADDRESS = "192.0.2.21"
ROW_OF_CARDS = {"CARD": "5", "CORES": "0-11", "END": "2026-11-17T03:00Z"}
PLACEMENT = tomllib.loads((HERE / "placement.toml").read_text())
NODE_A = PLACEMENT["node"]["vy-spot-glm53"]
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
TAP = Path("adapters/vllm/verity_vllm/ops/pod_norm_tap.sh")


def git(tree: Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=tree, check=True, capture_output=True, text=True).stdout.strip()


def make_tree(tmp_path: Path, tied_build: bool = True) -> Path:
    """The workload's files in a repository whose origin's main is its one commit, the workloads stand-ins."""
    tree = tmp_path / "tree"
    (tree / WORKLOAD_RUN).mkdir(parents=True)
    shutil.copy(HERE / "run.sh", tree / WORKLOAD_RUN / "run.sh")
    text = (HERE / "placement.toml").read_text().replace(f'TAPS_AT = "{NODE_A["TAPS_AT"]}"', 'TAPS_AT = "HEAD"')
    if not tied_build:
        text = text.replace(f'"{PLACEMENT["tied_row"]}" = "{NODE_A["builds"][PLACEMENT["tied_row"]]}"\n', "")
    (tree / WORKLOAD_RUN / "placement.toml").write_text(text)
    rows = tree / "adapters" / "vllm" / "workloads"
    rows.mkdir(parents=True)
    sums = ""
    for row in (PLACEMENT["row"], PLACEMENT["tied_row"]):
        workload = json.dumps({"model": f"some/{row[:14]}", "case": "SERVED", "engine": {"revision": "ab12"}}).encode()
        (rows / f"{row}.json").write_bytes(workload)
        sums += f"{hashlib.sha256(workload).hexdigest()}  {row}.json\n"
    (rows / "SHA256SUMS").write_text(sums)
    (tree / TAP).parent.mkdir(parents=True)
    (tree / TAP).write_text("# a tap's builder\n")
    for cmd in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "tree"],
                ["clone", "-q", "--bare", ".", str(tmp_path / "origin.git")], ["remote", "add", "origin", str(tmp_path / "origin.git")]):
        git(tree, *cmd)
    return tree


def run(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    """Makes the stand-in tree if it isn't yet; ``env`` is over this environment less placement.toml's variables."""
    tree = tmp_path / "tree"
    if not tree.exists():
        make_tree(tmp_path)
    held = {k: v for k, v in os.environ.items() if k not in {"VY_CHALLENGER_1_HOST", "HANDOFF", *ROW_OF_CARDS}}
    return subprocess.run(["bash", str(tree / WORKLOAD_RUN / "run.sh"), *args], capture_output=True, text=True, check=False,
                          env={**held, "HOME": str(tmp_path), **env})


def flags(got: subprocess.CompletedProcess, flag: str) -> list[str]:
    words = shlex.split(got.stdout)
    return [b for a, b in zip(words, words[1:], strict=False) if a == flag]


def env_of(got: subprocess.CompletedProcess) -> dict[str, str]:
    pairs = [e.split("=", 1) for e in flags(got, "--env")]
    assert len({k for k, _ in pairs}) == len(pairs), f"a variable is given twice: {pairs}"
    return dict(pairs)


def test_the_window_run_serves_on_node_a_on_the_launchs_card_and_cores(tmp_path):
    got = run(tmp_path, "window", "--dry-run", **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    words = shlex.split(got.stdout)
    assert words[words.index("--on") + 1] == "vy-spot-glm53"
    env = env_of(got)
    assert {k: env[k] for k in ("CARD", "CORES", "END", "H1_GATE", "COMPOSITE", "V3C_CLASS", "HIDDEN_SO", "ROW_ID", "ROW")} == {
        **ROW_OF_CARDS, "H1_GATE": "", "COMPOSITE": "", "V3C_CLASS": "", "HIDDEN_SO": NODE_A["HIDDEN_SO"], "ROW_ID": PLACEMENT["row"],
        "ROW": f"{NODE_A['LLAMA_ROOT']}/{NODE_A['builds'][PLACEMENT['row']]}/{PLACEMENT['row']}"}
    assert words[words.index("--") + 1:] == ["taskset", "-c", "0-11", "bash", "adapters/vllm/tools/workload/window.sh"]
    assert f"adapters/vllm/workloads/{PLACEMENT['row']}.json" in flags(got, "--send")
    assert set(re.findall(r"\$\{(\w+)", got.stdout)) <= {"HOME", *NODE_A["env_files"]}


def test_the_window_run_takes_the_gates_coins_by_default_with_mains_line(tmp_path):
    got = run(tmp_path, "window", "--dry-run", **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    env = env_of(got)
    assert {k: env[k] for k in ("PP_CHALLENGER", "PP_PEER", "PP_GATE_KEY", "PP_CHALLENGER_KEY")} == {
        "PP_CHALLENGER": "gate:research@${VY_CHALLENGER_1_HOST}", "PP_PEER": "vy-spot-glm53",
        "PP_GATE_KEY": "${HOME}/.ssh/vllm_lane_ed25519", "PP_CHALLENGER_KEY": "tests/benchmarks/pouw/fixtures/challenger-key.json"}
    assert "DEVELOPMENT" not in env
    main = [Path(s) for s in flags(got, "--send") if s.endswith("/main.txt")]
    assert len(main) == 1 and main[0].read_text().split() == [git(tmp_path / "tree", "rev-parse", "HEAD")]


@pytest.mark.parametrize("args", [("window",), ("window", "--tie"), ("window", "--development"), ("setup",)])
def test_no_address_reaches_a_dry_runs_command(tmp_path, args):
    got = run(tmp_path, *args, "--dry-run", VY_CHALLENGER_1_HOST=ADDRESS, **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    assert ADDRESS not in got.stdout + got.stderr


def test_the_tied_window_runs_with_the_settings_the_tree_writes(tmp_path):
    got = run(tmp_path, "window", "--tie", "--dry-run", **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    env, tie = env_of(got), PLACEMENT["run"]["window"]["tie"]["env"]
    design = next(k for k, v in tie.items() if "stratified-rate:" in v)
    assert {k: env[k] for k in tie} == {**tie, design: "stratified-rate:1/2097152"}
    assert (env["POUW"], env["PEARLPLUS_TIE"], env["V3C_CLASS"], env["PRECISION"]) == ("pearlplus", "1", "4,32,4", "bfloat16")
    assert env["H1_PEER"] == env["PP_PEER"] == "vy-spot-glm53" and "COMPOSITE" not in env
    assert int(env["V3C_AUDIT_MEM_GB"]) > 2015, "the records audit defers off node A, which has 2015 GB"
    piece, lean = 60 * int(env["POUW_TIMEOUT"].removesuffix("m")), int(env["PEARLPLUS_LEAN_TIMEOUT"])
    window = 3600 * int(PLACEMENT["run"]["window"]["research"][-1].removesuffix("h"))
    assert lean + 3600 <= piece <= window - 1800, "Pearl+'s Lean follows its tie and Python check inside its piece, which ends inside the window's"
    assert env["NVDISASM_PATH"] == "/usr/local/cuda-13.0/bin" != f"{env['CUDA_HOME']}/bin", "the Pearl-C cubin's SASS needs nvdisasm"
    assert env["ROW_ID"] == PLACEMENT["tied_row"] and "__bf16__" in env["ROW"] and env["ROW"].endswith(PLACEMENT["tied_row"])
    assert not {"H1_TIE", "ATTESTER_FRONT", "LOAD", "SLOTS", "PROMPTS"} & env.keys()
    assert "--prefix-caching" not in env.get("SERVE_ARGS", "")
    assert [s for s in flags(got, "--send") if s.endswith("/main.txt")], "the gate's Pearl+ salts need main's line too"
    assert f"adapters/vllm/workloads/{PLACEMENT['tied_row']}.json" in flags(got, "--send")


def test_the_tied_windows_sampling_design_and_a_windows_length_are_the_launchs(tmp_path):
    design = next(k for k, v in PLACEMENT["run"]["window"]["tie"]["env"].items() if "stratified-rate:" in v)
    got = run(tmp_path, "window", "--tie", "--dry-run", **ROW_OF_CARDS, **{design: "uniform:3", "LEASE_MIN": "20"})
    assert got.returncode == 0, got.stderr
    assert (env_of(got)[design], env_of(got)["LEASE_MIN"]) == ("uniform:3", "20")
    assert env_of(run(tmp_path, "window", "--dry-run", **ROW_OF_CARDS))["LEASE_MIN"] == ""


def test_a_window_with_no_traced_build_of_its_row_is_refused(tmp_path):
    make_tree(tmp_path, tied_build=False)
    got = run(tmp_path, "window", "--tie", "--dry-run", **ROW_OF_CARDS)
    assert (got.returncode, got.stdout) == (1, "")
    assert "builds name no traced Build of" in got.stderr and "`run.sh setup --tie`" in got.stderr


def test_a_commit_off_mains_line_is_refused_the_gates_coins_and_told_the_cure(tmp_path):
    tree = make_tree(tmp_path)
    git(tree, "commit", "-q", "--allow-empty", "-m", "unlanded")
    got = run(tmp_path, "window", "--dry-run", **ROW_OF_CARDS)
    assert (got.returncode, got.stdout) == (1, "")
    assert "is not on origin/main's first-parent line" in got.stderr and "land it first" in got.stderr and "--development" in got.stderr
    assert run(tmp_path, "window", "--development", "--dry-run", **ROW_OF_CARDS).returncode == 0


def test_a_window_from_a_commit_that_changed_the_taps_is_refused(tmp_path):
    tree = make_tree(tmp_path)
    built = git(tree, "rev-parse", "HEAD")
    placement = tree / WORKLOAD_RUN / "placement.toml"
    placement.write_text(placement.read_text().replace('TAPS_AT = "HEAD"', f'TAPS_AT = "{built}"'))
    (tree / TAP).write_text("# a changed tap's builder\n")
    git(tree, "commit", "-q", "-am", "taps")
    got = run(tmp_path, "window", "--development", "--dry-run", **ROW_OF_CARDS)
    assert (got.returncode, got.stdout) == (1, "")
    assert f"the taps were built at {built[:12]}" in got.stderr and str(TAP) in got.stderr and "TAPS_AT" in got.stderr


@pytest.mark.parametrize("missing", list(ROW_OF_CARDS))
def test_without_a_parameter_the_window_run_starts_nothing_and_names_it(tmp_path, missing):
    got = run(tmp_path, "window", "--dry-run", **{k: v for k, v in ROW_OF_CARDS.items() if k != missing})
    assert (got.returncode, got.stdout) == (1, "")
    assert got.stderr.startswith(f"run.sh: {missing} is unset: placement.toml's ") and f"--env {missing}=VALUE" in got.stderr


def test_a_parameter_a_flag_gives_fills_the_wrap_and_flags_reach_the_recorder(tmp_path):
    got = run(tmp_path, "window", "--dry-run", "--development", "--env", "CORES=12-23", "--env", "WC_TIMEOUT=45m",
              CARD="6", END="2026-11-17T03:00Z")
    assert got.returncode == 0, got.stderr
    words = shlex.split(got.stdout)
    assert words[words.index("--") + 1:words.index("--") + 4] == ["taskset", "-c", "12-23"]
    assert env_of(got)["CORES"] == "12-23" and "WC_TIMEOUT=45m" in flags(got, "--env")


def test_development_coins_are_chosen_and_the_run_says_so(tmp_path):
    got = run(tmp_path, "window", "--development", "--dry-run", **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    env = env_of(got)
    assert (env["PP_CHALLENGER"], env["DEVELOPMENT"]) == ("local", "1")
    assert env["PP_PEER"] == "vy-spot-glm53" and not {"PP_GATE_KEY", "PP_CHALLENGER_KEY"} & env.keys()
    assert not [s for s in flags(got, "--send") if s.endswith("/main.txt")]


@pytest.mark.parametrize("tie", [False, True])
def test_the_setup_run_makes_the_rows_inputs_on_node_a(tmp_path, tie):
    got = run(tmp_path, "setup", *(["--tie"] if tie else []), "--dry-run", **ROW_OF_CARDS)
    assert got.returncode == 0, got.stderr
    words, env = shlex.split(got.stdout), env_of(got)
    row = PLACEMENT["tied_row" if tie else "row"]
    assert words[words.index("--on") + 1] == "vy-spot-glm53" and words[-1] == "adapters/vllm/tools/workload/setup.sh"
    assert env["ROW_ID"] == row and f"adapters/vllm/workloads/{row}.json" in flags(got, "--send")
    assert not {"POUW", "PP_CHALLENGER", "PEARLPLUS_TIE"} & env.keys()


def handoff(tmp_path: Path, served: str) -> Path:
    d = tmp_path / "handoff"
    d.mkdir()
    with tarfile.open(d / "compose.tar", "w") as t:
        body = f"COMPOSE_TREE={served}\n".encode()
        info = tarfile.TarInfo("private/compose/compose.env")
        info.size = len(body)
        t.addfile(info, io.BytesIO(body))
    (d / "compose.tar.sha256").write_text("sums\n")
    return d


def test_the_compose_run_sends_the_windows_hand_off_to_the_compose_node(tmp_path):
    d = handoff(tmp_path, git(make_tree(tmp_path), "rev-parse", "HEAD"))
    got = run(tmp_path, "compose", "--dry-run", HANDOFF=str(d))
    assert got.returncode == 0, got.stderr
    words = shlex.split(got.stdout)
    assert words[words.index("--on") + 1] == "vy-nebius-cpu-14" == PLACEMENT["run"]["window"]["env"]["COMPOSE_NODE"]
    assert flags(got, "--send") == [str(d / "compose.tar"), str(d / "compose.tar.sha256")]
    assert words[words.index("--") + 1:] == ["bash", "adapters/vllm/tools/workload/compose.sh"]
    assert flags(got, "--env") == ["PP_PEER=vy-nebius-cpu-14"]


@pytest.mark.parametrize("args", [("window",), ("window", "--development"), ("window", "--tie"), ("setup",), ("copy",),
                                  ("audit",), ("audit", "--development"), ("compose",)])
def test_every_run_carries_its_nodes_name_in_the_cluster_as_pp_peer_and_a_window_as_h1_peer(tmp_path, args):
    at = {**ROW_OF_CARDS, "WORKLOAD_RUN": "r-window", "AUDIT_NODE": "vy-nebius-cpu-16"}
    if args[0] == "compose":
        at["HANDOFF"] = str(handoff(tmp_path, git(make_tree(tmp_path), "rev-parse", "HEAD")))
    got = run(tmp_path, *args, "--dry-run", **at)
    assert got.returncode == 0, got.stderr
    words, env = shlex.split(got.stdout), env_of(got)
    assert env["PP_PEER"] == words[words.index("--on") + 1]
    assert env.get("H1_PEER") == (env["PP_PEER"] if args[0] == "window" else None)


def test_no_file_of_the_workload_names_the_retired_node():
    assert not [f.name for f in HERE.rglob("*") if f.is_file() and b"vy-dc-h200" in f.read_bytes()]


def test_the_compose_run_refuses_another_commit_than_the_windows(tmp_path):
    tree = make_tree(tmp_path)
    served = git(tree, "rev-parse", "HEAD")
    git(tree, "commit", "-q", "--allow-empty", "-m", "later")
    got = run(tmp_path, "compose", "--dry-run", HANDOFF=str(handoff(tmp_path, served)))
    assert (got.returncode, got.stdout) == (1, "")
    assert f"the window was served at {served} and this checkout is at {git(tree, 'rev-parse', 'HEAD')}" in got.stderr
    assert f"git worktree add DIR {served}" in got.stderr


def test_the_compose_run_without_its_hand_off_starts_nothing(tmp_path):
    got = run(tmp_path, "compose", "--dry-run")
    assert (got.returncode, got.stdout) == (1, "") and got.stderr.startswith("run.sh: HANDOFF is unset")
    got = run(tmp_path, "compose", "--dry-run", HANDOFF=str(tmp_path / "nothing"))
    assert (got.returncode, got.stdout) == (1, "") and "compose.tar, which is not a file" in got.stderr


@pytest.mark.parametrize(("args", "refusal"), [
    (("setup", "--development"), "run.sh: the setup run takes no coins, so --development chooses nothing: drop it"),
    (("compose", "--tie"), "run.sh: the compose run has no tied form: drop --tie")])
def test_a_flag_that_chooses_nothing_is_refused(tmp_path, args, refusal):
    got = run(tmp_path, *args, "--dry-run", HANDOFF=str(tmp_path), **ROW_OF_CARDS)
    assert (got.returncode, got.stdout, got.stderr.strip()) == (1, "", refusal)


@pytest.mark.parametrize("gate", [True, False])
def test_the_deferred_audit_is_copied_from_node_a_then_run_on_the_audit_node(tmp_path, gate):
    at = {"WORKLOAD_RUN": "r-window", "AUDIT_NODE": "vy-nebius-cpu-16"}
    got = run(tmp_path, "copy", "--dry-run", **at)
    assert got.returncode == 0, got.stderr
    words = shlex.split(got.stdout)
    assert words[words.index("--on") + 1] == "vy-spot-glm53" and env_of(got).items() >= at.items()
    assert words[words.index("--") + 1:] == ["bash", "adapters/vllm/tools/workload/copy.sh"]
    got = run(tmp_path, "audit", *([] if gate else ["--development"]), "--dry-run", **at, V3C_WARM_JOBS="8")
    assert got.returncode == 0, got.stderr
    words, env = shlex.split(got.stdout), env_of(got)
    assert words[words.index("--on") + 1] == "vy-nebius-cpu-16" and words[-1] == "adapters/vllm/tools/workload/audit.sh"
    assert (env["WORKLOAD_RUN"], env["V3C_PEER"], env["V3C_WARM_JOBS"], env["V3C_AUDIT_TIMEOUT"]) == ("r-window", "vy-nebius-cpu-16", "8", "")
    assert bool([s for s in flags(got, "--send") if s.endswith("/main.txt")]) == gate == ("PP_GATE_KEY" in env)


def test_the_deferred_audit_names_the_window_it_lacks(tmp_path):
    got = run(tmp_path, "audit", "--dry-run", AUDIT_NODE="vy-nebius-cpu-16")
    assert (got.returncode, got.stdout) == (1, "") and got.stderr.startswith("run.sh: WORKLOAD_RUN is unset")


@pytest.mark.parametrize("tie", [False, True])
def test_the_settings_are_node_as_and_its_rows_and_name_no_address(tmp_path, tie):
    row = PLACEMENT["tied_row" if tie else "row"]
    got = run(tmp_path, "settings", f"some/{row[:14]}", VY_CHALLENGER_1_HOST=ADDRESS)
    assert got.returncode == 0, got.stderr
    env = dict(line.split("=", 1) for line in got.stdout.splitlines())
    assert {k: env[k] for k in ("VENV", "HIDDEN_SO", "DISK_FLOOR_GB", "H1_PEER", "PP_PEER", "ROW_ID")} == {
        "VENV": NODE_A["VENV"], "HIDDEN_SO": NODE_A["HIDDEN_SO"], "DISK_FLOOR_GB": NODE_A["DISK_FLOOR_GB"],
        "H1_PEER": "vy-spot-glm53", "PP_PEER": "vy-spot-glm53", "ROW_ID": row}
    assert env["ROW"] == f"{NODE_A['LLAMA_ROOT']}/{NODE_A['builds'][row]}/{row}" and (env["POUW"] == "pearlplus") == tie
    assert not {"CARD", "CORES", "END", "PRODUCT", "PP_CHALLENGER"} & env.keys() and ADDRESS not in got.stdout


def test_settings_refuse_a_model_no_row_places(tmp_path):
    got = run(tmp_path, "settings", "some/other")
    assert (got.returncode, got.stdout) == (1, "") and "not 'some/other'" in got.stderr


def settings_tree(tmp_path: Path, mode: int = 0o600, sets: str = ADDRESS) -> Path:
    tree = make_tree(tmp_path)
    shutil.copy(HERE / "workload_env.sh", tree / WORKLOAD_RUN / "workload_env.sh")
    (tree / "adapters/vllm/manifests").mkdir(parents=True)
    (tree / "adapters/vllm/manifests/checkpoints-int8.json").write_text('{"checkpoints": []}')
    placement = tree / WORKLOAD_RUN / "placement.toml"
    text = placement.read_text().replace("/workspace/research/vllm-servedproof", str(tmp_path / "node"))
    for var, path in NODE_A["env_files"].items():
        f = tmp_path / Path(path).name
        f.write_text(f"export {var}={sets}\n" if var == "VY_CHALLENGER_1_HOST" else f"{var}='192.0.2.22'\n")
        f.chmod(mode)
        text = text.replace(f'"{path}"', f'"{f}"')
    placement.write_text(text)
    return tree


def test_a_command_runs_under_the_settings_with_the_env_files_filled_in(tmp_path):
    settings_tree(tmp_path)
    show = 'echo "$PP_CHALLENGER|$VY_CHALLENGER_1_HOST|$VENV|$H1_PEER|$ROW_ID"'
    got = run(tmp_path, "settings", f"some/{PLACEMENT['row'][:14]}", "--", "bash", "-c", show,
              PP_CHALLENGER="gate:research@${VY_CHALLENGER_1_HOST}", VENV="/the/callers")
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == f"gate:research@{ADDRESS}|{ADDRESS}|/the/callers|vy-spot-glm53|{PLACEMENT['row']}"
    assert (tmp_path / "node/serve-root").is_dir()


@pytest.mark.parametrize(("mode", "sets", "refusal"), [(0o644, ADDRESS, "is missing or has another mode"),
                                                        (0o600, "", "sets no VY_CHALLENGER_1_HOST")])
def test_a_command_starts_only_under_the_nodes_env_files(tmp_path, mode, sets, refusal):
    settings_tree(tmp_path, mode, sets)
    got = run(tmp_path, "settings", f"some/{PLACEMENT['row'][:14]}", "--", "true")
    assert (got.returncode, got.stdout) == (1, "") and refusal in got.stderr and ADDRESS not in got.stderr


@pytest.mark.parametrize("demo", ["remote", "onsite_network"])
def test_the_bindings_bind_every_party_of_the_demo_by_the_env_files_names(tmp_path, demo):
    path = HERE.parents[3] / "applications" / f"{demo}.toml"
    got = subprocess.run(["bash", str(HERE / "run.sh"), "bindings", str(path)], capture_output=True, text=True, check=False,
                         env={**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"})
    assert got.returncode == 0, got.stderr
    (f := tmp_path / "bindings.toml").write_text(got.stdout)
    cfg = tomllib.loads(path.read_text())
    bound = verity_run.bindings(f, cfg, verity_run.registry())
    assert bound and not [k for p in cfg["party"] for piece in served_table.pieces(cfg, p["name"]) for k in piece.unbound(bound)]
    assert set(re.findall(r"\$\{(\w+)\}", got.stdout)) <= {*NODE_A["env_files"], "HOME"}
    assert not re.search(r"\d+\.\d+\.\d+\.\d+", got.stdout)


@pytest.mark.parametrize("script", ["window.sh", "compose.sh", "setup.sh", "workload_env.sh", "copy.sh", "audit.sh", "handoff.sh"])
def test_the_runs_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(HERE / script)], capture_output=True, check=False).returncode == 0


def window_exit(tmp_path: Path, rc: int, off_node: str, deferred: bool = False) -> tuple[int, str]:
    d, out = tmp_path / "serve", tmp_path / "out"
    out.mkdir(exist_ok=True)
    if deferred:
        (d / "private" / "deferred").mkdir(parents=True)
        (d / "private" / "deferred" / "deferred.env").write_text("K=0\n")
    got = subprocess.run(["bash", "-c", f'log() {{ echo "$*"; }}; . {shlex.quote(str(HERE / "handoff.sh"))}; window_exit {rc}'],
                         capture_output=True, text=True, check=False, cwd=tmp_path,
                         env={**os.environ, "D": str(d), "OUT": str(out), "COMPOSE_OFF_NODE": off_node, "RESEARCH_RUN_ID": "r1"})
    return got.returncode, got.stdout


@pytest.mark.parametrize(("rc", "off_node", "want"), [(0, "0", 0), (1, "0", 1), (2, "0", 2), (0, "1", 4), (1, "1", 1), (2, "1", 2)])
def test_an_on_node_record_with_nothing_to_pack_exits_by_its_verdict_and_an_off_node_one_without_its_hand_off_exits_4(
        tmp_path, rc, off_node, want):
    assert window_exit(tmp_path, rc, off_node)[0] == want


@pytest.mark.parametrize("off_node", ["0", "1"])
def test_a_pending_record_deferred_off_the_node_exits_0_and_says_what_is_left(tmp_path, off_node):
    rc, said = window_exit(tmp_path, 2, off_node, deferred=True)
    assert rc == 0 and "pending: audited and composed off this node: `run.sh copy` and `run.sh audit` with WORKLOAD_RUN=r1" in said
