"""`streamed_wipe/served_wipe.sh`, the wipe's launcher (served_verdicts.sh's WIPE_CMD), on a CPU: the wipe scheme
attached by name with the launcher's settings, one 32 MiB segment in host memory (`wipe_store.HostMemory`), against
`wipe_side.py`, the challenger's stand-in on a host of its own, in place of the side the settings' ``command`` starts;
`nvidia-smi` and `gpu-lease` are stand-ins on PATH, and the launcher's Python (``PY``) is one that puts those stand-ins and
a Lean judge that answers as the Python did in place before it runs the launcher's own code. DIR/answer.json is the
subprotocol's core answer (verity/wipe/v0) in the form `Answer.toJson` writes (what served_verdicts.py's `piece_row` reads
through `pouw.core_check.answer_of`), with served.json and the summary beside it. Every refusal, the attach's or the
launcher's, is made by name before the wipe runs and leaves DIR/refused.txt, one sentence, and no answer.json."""

from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import shlex
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.prover import wipe_served as WS
from verity.protocols.resource_accounting.memory.pous.verifier import streamed_wipe as sw
from verity.protocols.resource_accounting.memory.pous.verifier import wipe_party as WP
from verity.protocols.resource_accounting.memory.pous.verifier import wipe_period as W

WRAPPER = Path(__file__).resolve().parents[3] / "benchmarks" / "pous" / "streamed_wipe" / "served_wipe.sh"
SIDE = Path(__file__).resolve().parents[2] / "verity" / "protocols" / "resource_accounting" / "memory" / "pous" / "verifier" / "wipe_side.py"
ANSWER_KEYS = ["commitment", "profile", "protocol", "public", "reason", "verdict"]
CARD = "GPU-5e7ed000-0000-0000-0000-000000000000"
#: who decided, as the summary and the headline name it: the stand-in judge, on this host
DECIDED = {"program": "wipe-protocol", "host": socket.gethostname()}
#: the SHA-512 of the key the challenger's stand-in (`wipe_side.py`) signs under: the pin of a deployment of it
PINNED = runpy.run_path(str(SIDE))["key_sha512"]()
#: the wipe key the run's configuration pins (VERITY_PINS): the pin of a run that sets none of its own
AUDITOR_PIN = hashlib.sha512(b"the configuration's pin of the memory attester's wipe key").hexdigest()
PINS = [{"key-sha512": hashlib.sha512(b"its PoUS key").hexdigest(), "party": "memory-challenger", "protocol": "pous",
         "scheme": "lms-sha512/n64-w1-h20/v0"},
        {"key-sha512": AUDITOR_PIN, "party": "memory-challenger", "protocol": "wipe", "scheme": "lms-sha512/n64-w1-h20/v0"}]
UNTRUSTED = "the session is signed by a key that is not a trusted memory challenger's"
#: the tree's own lease script, which the launcher takes where the node has no `gpu-lease`
TREE_LEASE = "tools/research/src/research/pods/sh/gpu_lease.sh"
ROOT = Path(__file__).resolve().parents[3]
#: the Lean judge's package, `wipe-protocol`'s
JUDGE = "verity/protocols/resource_accounting/memory/pous/attester/lean"
LAKE = """#!/usr/bin/env bash
echo "$PWD $*" >> "$FAKE_LAKE_LOG"
python3 - << 'EOF' || exit 1
import json, pathlib, sys
for p in json.loads(pathlib.Path("lake-manifest.json").read_text())["packages"]:
    if not pathlib.Path(p["dir"]).is_dir():
        sys.exit(f"error: {p['name']}: package directory not found: {pathlib.Path.cwd() / p['dir']}")
EOF
mkdir -p .lake/build/bin && printf '#!/bin/sh\\nexit 9\\n' > ".lake/build/bin/$2" && chmod +x ".lake/build/bin/$2"
"""
"""A stand-in `lake build EXE` that, like lake, fails where a package the manifest in its directory requires is absent."""

SMI = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_SMI_LOG"
case "$*" in
  *--query-compute-apps*) [ -z "$FAKE_SMI_APPS" ] || echo "$FAKE_SMI_APPS" ;;
  *memory.used*) echo "$FAKE_SMI_USED" ;;
  *name*) echo "$FAKE_SMI_NAME" ;;
esac
"""
IP = "#!/bin/sh\necho '1.1.1.1 via 10.0.0.1 dev eth9 src 10.0.0.2 uid 0'\n"
"""A stand-in `ip`, whose default route leaves by ``eth9``."""
LEASE = """#!/usr/bin/env bash
echo "$GPU_LEASE_WHO $*" >> "$FAKE_LEASE_LOG"
while [ "$1" != -- ]; do shift; done
shift
GPU_LEASE_UUID=GPU-own0 CUDA_VISIBLE_DEVICES=0 exec "$@"
"""
PYTHON = """#!{python}
import json, runpy, sys
from pathlib import Path
if sys.argv[1] != "-":
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
    raise SystemExit
sys.argv = sys.argv[1:]
code = sys.stdin.read()
from verity.runtime.stream import Side
from verity.protocols.resource_accounting.memory.pous.prover import wipe_run, wipe_served as WS
of = Side.of


def stand_in(period, node, key, local, what=None):
    assert (key, what) == ("wipe", "the wipe"), "the wipe's side is period.json's challenger.wipe"
    named = of(period, node, key, local, what)
    Path({party!r}).write_text(json.dumps({{"host": named.host, "command": named.command, "address": named.address}}))
    return Side(named.host, (sys.executable, {side!r}, local[local.index("--device") + 1]))


wipes, run = [], wipe_run.run


def ran(*args, **kw):
    wipes.append(run(*args, **kw))
    return wipes[-1]


def judged(exe, cases):
    assert Path(exe).name == "wipe-protocol" and list(cases) == [wipes[-1].case]
    return [wipes[-1].answer]


Side.of = staticmethod(stand_in)
wipe_run.run, wipe_run.serve = ran, judged
WS.card_of = lambda ordinal: {card!r}
if sys.argv[1] == "run":
{hook}
exec(compile(code, "served_wipe.sh", "exec"), {{"__name__": "__main__"}})
"""
"""The launcher's Python (``PY``): the challenger's side `wipe_side.py` in place of the one the settings' command starts
(the side `Side.of` names written to ``party.json``), a judge that answers each case as the Python answered it, and the
card's UUID; then the launcher's own code, as `python -` runs it, or a script it names (served_case.py)."""
LAUNCH = 'export PATH="$FAKE_BIN:$PATH"; exec bash "$@"'


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


def _python(tmp_path: Path, hook: str = "") -> Path:
    """The launcher's Python for this test, ``hook`` run before the wipe itself."""
    body = "".join(f"    {line}\n" for line in hook.splitlines()) or "    pass\n"
    return _exe(tmp_path / "python", PYTHON.format(python=sys.executable, party=str(tmp_path / "party.json"),
                                                   side=str(SIDE), card=CARD, hook=body.rstrip("\n")))


def _judge(tmp_path: Path) -> str:
    """A Lean judge built already (WIPE_PROTOCOL_EXE), so the launcher builds none; the stand-in judge never runs it."""
    return str(_exe(tmp_path / "wipe-protocol", "#!/bin/sh\nexit 9\n"))


def _settings(key: Path, **more) -> str:
    """The launcher's settings: the auditor's binding through the gate key ``key``, the fill in host memory."""
    return json.dumps({"store": "host", "streamer": "python", "reads": W.READS, "host": "vy-auditor-2",
                       "address": "10.96.0.36:47611",
                       "command": ["ssh", "-i", str(key), "-o", "BatchMode=yes", "research@10.96.0.36"]} | more)


@pytest.fixture
def node(tmp_path) -> dict[str, str]:
    """The environment of a node with a stand-in card and gpu-lease, the launcher's Python, the gate key, and the
    challenger's key pinned (WIPE_KEY_SHA512)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "nvidia-smi", SMI)
    _exe(bin_ / "gpu-lease", LEASE)
    _exe(bin_ / "ip", IP)
    key = tmp_path / "vy-auditor2-wipe"
    key.write_text("not a key\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_LEASE_", "WIPE_", "CUDA_VISIBLE", "VERITY_"))}
    return env | {"FAKE_BIN": str(bin_), "WIPE_NODE": "vy-dc-h200", "WIPE_KEY_SHA512": PINNED,
                  "FAKE_SMI_LOG": str(tmp_path / "smi.log"), "FAKE_SMI_APPS": "", "FAKE_SMI_USED": "0",
                  "FAKE_SMI_NAME": "", "FAKE_LEASE_LOG": str(tmp_path / "lease.log"), "PY": str(_python(tmp_path)),
                  "FAKE_SETTINGS": _settings(key), "VERITY_PINS": "[]"}


def _wipe(env: dict[str, str], d: Path, *args: str) -> subprocess.CompletedProcess:
    """served_wipe.sh SERVE_DIR 3 d SETTINGS, the serve dir ``serve-run`` beside ``d``, the settings the node's unless
    ``args`` gives its own."""
    d.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(["bash", "-c", LAUNCH, "_", str(WRAPPER), str(d.parent / "serve-run"), "3", str(d),
                           *(args or (env["FAKE_SETTINGS"],))], env=env, capture_output=True, text=True, cwd=d.parent)


def _stamps(d: Path) -> list[str]:
    """The events of the piece's timeline, ``d``'s ``stamps.txt``, in order: each line a UTC second, then the event."""
    lines = (d / "stamps.txt").read_text().splitlines()
    assert all(re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", line) for line in lines), lines
    return [line.split(" ", 1)[1] for line in lines]


def _refused(r: subprocess.CompletedProcess, d: Path) -> str:
    """A refusal's shape, which served_verdicts.py's runner reads as a refused row: exit 3, DIR/refused.txt one nonempty
    sentence `WIPE REFUSED: ...`, the piece's last line, and no answer.json; the sentence returned."""
    assert r.returncode == 3, r.stdout + r.stderr
    lines = (d / "refused.txt").read_text().splitlines()
    assert len(lines) == 1 and lines[0].startswith("WIPE REFUSED: ") and lines[0] != "WIPE REFUSED: ", lines
    assert r.stdout.strip().splitlines()[-1] == lines[0] and not (d / "answer.json").exists()
    return lines[0].removeprefix("WIPE REFUSED: ")


def _judged(d: Path) -> str:
    """The judge the period names, as the launcher handed it to the attach."""
    return json.loads((d / "period.json").read_text())["tools"]["wipe-protocol"]


def _is_core_answer(a: dict) -> None:
    """`Core.Answer.ofJson`'s shapes (`pouw.core_check.answer_of`): an accept has no reason and both its public outputs
    and its profile, a refusal a reason and neither, and nothing else."""
    assert sorted(a) == ANSWER_KEYS and isinstance(a["protocol"], str) and len(bytes.fromhex(a["commitment"])) == 64
    if a["verdict"] == "accept":
        assert a["reason"] is None and a["public"] is not None and a["profile"] is not None
    else:
        assert a["verdict"] in ("reject", "undecided") and isinstance(a["reason"], str)
        assert a["public"] is None and a["profile"] is None


def _decided(case: dict) -> dict:
    """The answer `wipe.case.json` decides, recomputed from its bytes."""
    p = case["params"]
    evidence = json.loads(bytes.fromhex(case["evidence"]))
    return W.answer(bytes.fromhex(p["server"]), bytes.fromhex(p["session"]), [bytes.fromhex(t) for t in p["trusted"]],
                    WP.commitment_of(json.loads(bytes.fromhex(case["commitment"]))),
                    None if evidence["signed"] is None else WP.signed_of(evidence["signed"]),
                    egress_mib=evidence["egress_mib"])


def test_the_wipe_on_the_serves_card_writes_its_core_answer(node, tmp_path) -> None:
    """In the serve's lease, before its engine loads: the wipe attached by name, and DIR/answer.json the accepted core
    answer of verity/wipe/v0 that the case decides, the challenger the one the settings' command starts on vy-auditor-2
    (its stand-in's key pinned by the run, not the configuration, so trusted-device's premise reads a rehearsal's), the
    window's serve dir and the card in served.json, and the piece's last line its `WIPE` headline. The case, which holds
    the signed session, is the serve dir's, private, with the composite's piece beside it (served_case.py's `CASE` line
    before the headline), and DIR keeps none of it."""
    d = tmp_path / "pieces" / "wipe"
    judge = _judge(tmp_path)
    r = _wipe(node | {"GPU_LEASE_UUID": f"{CARD},GPU-other", "WIPE_PROTOCOL_EXE": judge}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    answer = json.loads((d / "answer.json").read_text())
    _is_core_answer(answer)
    assert answer["protocol"] == W.PROTOCOL == "verity/wipe/v0" and answer["verdict"] == "accept"
    assert answer == json.loads((d / "wipe.json").read_text())["answer"]
    public = answer["public"]
    assert public["k"] == W.READS and public["memory_bits"] == W.SEGMENT_LABELS * sw.LABEL_BITS
    assert public["certified"] is False and public["foreign_bits"] > 0
    placement = answer["profile"]["placement"]
    assert placement["trusted-device"] == (f"{mc.UNCOMMITTED} (no configuration); the memory attester ran on vy-auditor-2, not the developer's "
                                         "vy-dc-h200")
    profile = {k: v for k, v in answer["profile"].items() if k != "placement"}
    private = tmp_path / "pieces" / "serve-run" / "private" / "wipe"
    case = json.loads((private / "wipe.case.json").read_text())
    assert _decided(case) == answer | {"profile": profile} and not (d / "wipe.case.json").exists()
    assert json.loads((private / "case.json").read_text()) == case
    assert json.loads((private / "answer.json").read_text()) == answer | {"profile": profile}
    assert {f: stat.S_IMODE((private / f).stat().st_mode) for f in ("case.json", "answer.json", "wipe.case.json")} == {
        "case.json": 0o600, "answer.json": 0o600, "wipe.case.json": 0o600} and stat.S_IMODE(private.stat().st_mode) == 0o700
    assert case["evidence"][:64] not in "".join(p.read_text() for p in d.iterdir() if p.is_file())
    party = json.loads((tmp_path / "party.json").read_text())
    assert party == {"host": "vy-auditor-2", "command": json.loads(node["FAKE_SETTINGS"])["command"],
                     "address": ["10.96.0.36", 47611]}
    assert json.loads((d / "served.json").read_text()) == {
        "serve_dir": str(tmp_path / "pieces" / "serve-run"), "window": "3", "server": "vy-dc-h200", "card": CARD,
        "run": node.get("RESEARCH_RUN_ID"), "lease": "the serve's: its engine loads on this card after the wipe",
        "key_pin": "run"}
    summary = json.loads((d / "summary.json").read_text())
    assert summary["verdict"] == "accept" and summary["decided_by"] == DECIDED and summary["k"] == W.READS
    key, = case["params"]["trusted"]
    assert summary["key_sha512"] == hashlib.sha512(bytes.fromhex(key)).hexdigest() == PINNED and summary["key_pinned"]
    assert _judged(d) == judge
    *_, built, last = r.stdout.strip().splitlines()
    assert json.loads(built.removeprefix("CASE ")) == {"piece": str(private), "protocol": W.PROTOCOL, "verdict": "accept"}
    assert last.startswith("WIPE {")
    headline = json.loads(last.removeprefix("WIPE "))
    assert headline | {"fill_GBps": None, "t0": None, "t1": None} == {
        "verdict": "accept", "reason": None, "protocol": W.PROTOCOL, "decided_by": DECIDED, "k": W.READS, "segments": 1,
        "fill_GBps": None, "t0": None, "t1": None, "server": "vy-dc-h200", "card": CARD, "exit": 0, "key_pin": "run"}
    assert not (tmp_path / "lease.log").exists() and not (d / "refused.txt").exists()
    assert _stamps(d) == ["start", "the wipe exit 0", "exit 0"]


def test_a_case_that_cant_reach_the_private_dir_leaves_none_in_dir(node, tmp_path) -> None:
    """The case holds the signed session: where the serve dir's private directory can't be made, DIR, the window's
    output that leaves the node, keeps no copy of it."""
    d = tmp_path / "pieces" / "wipe"
    (tmp_path / "pieces" / "serve-run").mkdir(parents=True)
    (tmp_path / "pieces" / "serve-run" / "private").write_text("not a directory\n")
    r = _wipe(node | {"GPU_LEASE_UUID": f"{CARD},GPU-other", "WIPE_PROTOCOL_EXE": _judge(tmp_path)}, d)
    assert not (d / "wipe.case.json").exists(), r.stdout + r.stderr


def test_outside_a_lease_it_takes_a_card_of_its_own(node, tmp_path) -> None:
    """Without the serve's lease it waits for a card of its own (`gpu-lease 1 --wait` as memory-accounting) and wipes
    that one. A Lean judge built already (WIPE_PROTOCOL_EXE) is the attach's, nothing built here."""
    d = tmp_path / "wipe"
    judge = _judge(tmp_path)
    r = _wipe(node | {"WIPE_PROTOCOL_EXE": judge}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    _is_core_answer(answer := json.loads((d / "answer.json").read_text()))
    assert answer["verdict"] == "accept"
    lease = (tmp_path / "lease.log").read_text().splitlines()
    assert len(lease) == 1 and lease[0].startswith(f"memory-accounting 1 --wait --max-min 30 -- bash {WRAPPER} ")
    served = json.loads((d / "served.json").read_text())
    assert served["card"] == "GPU-own0" and served["lease"].startswith("its own (gpu-lease 1 --wait)")
    assert _judged(d) == judge and not (d / "lake-build.log").exists()
    assert _stamps(d) == ["start", "asked for the card", "leased the card", "the wipe exit 0", "exit 0"]


def test_a_node_with_no_gpu_lease_gets_the_trees_own(node, tmp_path) -> None:
    """Where no `gpu-lease` is installed the wipe takes its card by the tree's own lease script, with the same request
    as memory-accounting, and wipes the card that hands out. The launcher is a copy in a tree whose lease script is a
    stand-in; the tree's own is where the launcher looks for it."""
    tree, d = tmp_path / "tree", tmp_path / "wipe"
    wrapper = tree / "benchmarks" / "pous" / "streamed_wipe" / "served_wipe.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(WRAPPER.read_text())
    (tree / TREE_LEASE).parent.mkdir(parents=True)
    (tree / TREE_LEASE).write_text(LEASE.replace("GPU-own0", "GPU-tree0"))
    assert (ROOT / TREE_LEASE).is_file()
    bin_ = tmp_path / "bin-no-lease"
    bin_.mkdir()
    _exe(bin_ / "nvidia-smi", SMI)
    _exe(bin_ / "ip", IP)
    path = ":".join(p for p in node["PATH"].split(":") if not (Path(p) / "gpu-lease").exists())
    assert shutil.which("gpu-lease", path=f"{bin_}:{path}") is None
    env = node | {"PATH": path, "FAKE_BIN": str(bin_), "WIPE_PROTOCOL_EXE": _judge(tmp_path)}
    r = subprocess.run(["bash", "-c", LAUNCH, "_", str(wrapper), str(tmp_path / "serve-run"), "3", str(d),
                        env["FAKE_SETTINGS"]], env=env, capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    lease = (tmp_path / "lease.log").read_text().splitlines()
    assert len(lease) == 1 and lease[0].startswith(f"memory-accounting 1 --wait --max-min 30 -- bash {wrapper} ")
    assert json.loads((d / "served.json").read_text())["card"] == "GPU-tree0" and not (d / "refused.txt").exists()
    assert _stamps(d) == ["start", "asked for the card", "leased the card", "the wipe exit 0", "exit 0"]


def test_the_lean_judge_is_built_in_a_copy_that_holds_every_package_it_requires(node, tmp_path) -> None:
    """With no judge built already the wrapper builds `wipe-protocol` before it asks for a card: `lake build` in the
    judge's package of a copy of ``verity/`` outside the tree, removed when the piece ends, and the built judge goes to
    the launcher. The stand-in `lake` fails as lake does when the copy lacks a package the judge's manifest requires; each
    one is a path requirement inside ``verity/``, so the copy needs no network and nothing else of the tree."""
    d, home = tmp_path / "wipe", tmp_path / "home"
    home.mkdir()
    _exe(Path(node["FAKE_BIN"]) / "lake", LAKE)
    r = _wipe(node | {"HOME": str(home), "FAKE_LAKE_LOG": str(tmp_path / "lake.log")}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    (built,) = (tmp_path / "lake.log").read_text().splitlines()
    package, command = built.split(" ", 1)
    assert command == "build wipe-protocol" and package.endswith("/" + JUDGE) and ROOT not in Path(package).parents
    assert _judged(d) == f"{package}/.lake/build/bin/wipe-protocol"
    assert len((tmp_path / "lease.log").read_text().splitlines()) == 1
    assert not Path(package).exists() and (d / "lake-build.log").exists()
    for p in json.loads((ROOT / JUDGE / "lake-manifest.json").read_text())["packages"]:
        assert p["type"] == "path", f"the judge requires {p['name']} from {p.get('url')}, which no copy holds"
        assert ROOT / "verity" in (ROOT / JUDGE / p["dir"]).resolve().parents, (p["name"], p["dir"])


def test_a_piece_that_builds_its_judge_stamps_the_build(node, tmp_path) -> None:
    """The build of `wipe-protocol` is on the piece's timeline, beside the lease and the wipe: it starts before the card
    is asked for and ends before the piece does."""
    d, home = tmp_path / "wipe", tmp_path / "home"
    home.mkdir()
    _exe(Path(node["FAKE_BIN"]) / "lake", LAKE)
    r = _wipe(node | {"HOME": str(home), "FAKE_LAKE_LOG": str(tmp_path / "lake.log")}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    stamps = _stamps(d)
    assert [s for s in stamps if s != "built wipe-protocol"] == [
        "start", "the Lean build starts", "asked for the card", "leased the card", "the wipe exit 0", "exit 0"]
    assert stamps.index("the Lean build starts") < stamps.index("built wipe-protocol") < stamps.index("exit 0")


def test_the_wipe_runs_while_its_judge_builds(node, tmp_path) -> None:
    """The stand-in `lake` doesn't end until the wipe has started (a FIFO the launcher's Python writes as it runs the
    wipe and `lake` reads, once the piece has stamped the build's start): the wipe runs before the judge's build ends."""
    d, home, fifo = tmp_path / "wipe", tmp_path / "home", tmp_path / "demo-started"
    home.mkdir()
    os.mkfifo(fifo)
    events = tmp_path / "events"
    _exe(Path(node["FAKE_BIN"]) / "lake", LAKE.replace(
        'python3 - <<', f'echo "lake start" >> {events}\n'
        f'! grep -q "the Lean build starts" "$FAKE_STAMPS" || read -r _ < {fifo}\npython3 - <<', 1)
        + f'echo "lake end" >> {events}\n')
    _python(tmp_path, f'open({str(events)!r}, "a").write("demo start\\n")\n'
                      f'open({str(fifo)!r}, "w").write("started\\n")\n')
    r = _wipe(node | {"HOME": str(home), "FAKE_LAKE_LOG": str(tmp_path / "lake.log"),
                      "FAKE_STAMPS": str(d / "stamps.txt")}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    seen = events.read_text().splitlines()
    assert seen.index("lake start") < seen.index("demo start") < seen.index("lake end"), seen


def test_the_wipe_waits_only_for_a_judge_still_being_built_and_raises_the_builds_own_error(tmp_path) -> None:
    """`wipe_served.built`, the attach's default ``ready``, which `wipe_run` calls before the judge answers: a judge
    with no lock beside it is taken as it is; one with a lock whose build ended (`EXE.ok`) is ready; one whose build
    failed raises that build's error."""
    ready = WS.built
    exe = tmp_path / "bin" / "wipe-protocol"
    exe.parent.mkdir()
    ready(str(exe))
    Path(f"{exe}.lock").touch()
    (exe.parent / "failed").write_text("error: wipe-protocol has a hole\n")
    with pytest.raises(RuntimeError, match="^error: wipe-protocol has a hole$"):
        ready(str(exe))
    Path(f"{exe}.ok").touch()
    ready(str(exe))


def test_a_judge_that_does_not_build_ends_the_piece_after_the_wipe_with_its_own_error(node, tmp_path) -> None:
    """The wipe runs beside the build; the piece then ends with exit 7, the build's own last lines and why, and no
    answer: no headline, no answer.json."""
    d, home = tmp_path / "wipe", tmp_path / "home"
    home.mkdir()
    _exe(Path(node["FAKE_BIN"]) / "lake", '#!/usr/bin/env bash\necho "error: wipe-protocol has a hole"; exit 1\n')
    r = _wipe(node | {"HOME": str(home)}, d)
    assert r.returncode == 7, r.stdout + r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[-1] == f"WIPE FAILED: wipe-protocol doesn't build ({d}/lake-build.log)" and \
        "error: wipe-protocol has a hole" in lines
    assert not (d / "answer.json").exists() and not any(line.startswith("WIPE {") for line in lines)
    stamps = _stamps(d)
    assert "wipe-protocol didn't build" in stamps and stamps[-1] == "the piece stops: wipe-protocol didn't build"


@pytest.mark.parametrize("create", [1, 2])
def test_a_file_that_changes_as_the_copy_reads_it_still_builds_the_judge(node, tmp_path, create) -> None:
    """tar's exit 1 on the copy, a file that changed as it read it (another suite's import writing ``__pycache__``
    beside the tree's Python), still builds `wipe-protocol` and runs the piece; 2, a copy that failed, stops it (exit
    7, saying so) before the judge's build or any card. The stand-in `tar` copies as tar does, then exits so."""
    d, home = tmp_path / "wipe", tmp_path / "home"
    home.mkdir()
    _exe(Path(node["FAKE_BIN"]) / "lake", LAKE)
    _exe(Path(node["FAKE_BIN"]) / "tar", f'#!/usr/bin/env bash\n{shlex.quote(shutil.which("tar"))} "$@" || exit\n'
                                         f'[ "$3" != -c ] || exit {create}\n')
    r = _wipe(node | {"HOME": str(home), "FAKE_LAKE_LOG": str(tmp_path / "lake.log")}, d)
    if create == 1:
        assert r.returncode == 0, r.stdout + r.stderr
        assert (tmp_path / "lake.log").read_text().split(" ", 1)[1] == "build wipe-protocol\n"
        assert _stamps(d)[:3] == ["start", "the Lean build starts", "asked for the card"]
    else:
        assert r.returncode == 7 and "WIPE FAILED: verity/ doesn't copy for wipe-protocol's build" in r.stdout
        assert not (tmp_path / "lake.log").exists() and not (tmp_path / "lease.log").exists()
        assert _stamps(d) == ["start"]


@pytest.mark.parametrize("apps, used, why", [
    ("4242", "70300", "process 4242 on it, and the wipe kills nothing"),
    ("", "2048", "2048 MiB of its memory in use, past idle_mib 1024")])
def test_a_card_that_isnt_idle_is_refused(node, tmp_path, apps, used, why) -> None:
    """A process on the card, or more of its memory in use than ``idle_mib``: the attach refuses before the wipe runs
    (the reason in refused.txt, an earlier answer gone), nothing killed."""
    d = tmp_path / "wipe"
    d.mkdir()
    (d / "answer.json").write_text("{}")
    key = tmp_path / "vy-auditor2-wipe"
    r = _wipe(node | {"GPU_LEASE_UUID": CARD, "FAKE_SMI_APPS": apps, "FAKE_SMI_USED": used, "WIPE_CARD": "any",
                      "WIPE_PROTOCOL_EXE": _judge(tmp_path)}, d, _settings(key, store="cuda"))
    assert _refused(r, d) == f"the card {CARD} isn't idle: {why}"
    assert _stamps(d) == ["start", "refused"] and not any((d / f).exists() for f in ("served.json", "wipe.json"))


def test_a_card_that_is_not_the_one_the_wipe_is_measured_for_is_refused(node, tmp_path) -> None:
    """A cuda store on a card whose SKU isn't WIPE_CARD's (h200-sxm-141gb by default): refused under the lease, before
    the wipe, naming the card, its name and its SKU."""
    d = tmp_path / "wipe"
    name = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    r = _wipe(node | {"GPU_LEASE_UUID": CARD, "FAKE_SMI_NAME": name, "WIPE_PROTOCOL_EXE": _judge(tmp_path)}, d,
              _settings(tmp_path / "vy-auditor2-wipe", store="cuda"))
    why = _refused(r, d)
    assert why.startswith(f"the card {CARD}") and (f"is '{name}', a " in why and why.endswith("not a h200-sxm-141gb")
                                                   or name in why), why
    assert not (tmp_path / "party.json").exists()


def test_a_gate_key_not_on_the_node_is_refused(node, tmp_path) -> None:
    """The settings' command names an ssh key that isn't on the node: the attach refuses before any lease is asked for,
    naming the key and who installs it."""
    d = tmp_path / "wipe"
    missing = tmp_path / "nowhere" / "vy-auditor2-wipe"
    why = _refused(_wipe(node, d, _settings(missing)), d)
    assert why.startswith(f"no wipe gate key {missing} on vy-dc-h200 (") and "infra installs it" in why
    assert not (tmp_path / "lease.log").exists() and _stamps(d) == ["start", "refused"]


def test_a_session_signed_under_another_key_than_the_pinned_one_is_rejected(node, tmp_path) -> None:
    """WIPE_KEY_SHA512 pins another key than the one the challenger signs under: the wipe runs, honest in every read,
    and its answer is a reject (exit 1) naming the key untrusted, decided by a case that trusts no key. The summary
    carries the key the session came with and that it isn't the pinned one."""
    d = tmp_path / "pieces" / "wipe"
    other = hashlib.sha512(b"another key").hexdigest()
    r = _wipe(node | {"GPU_LEASE_UUID": CARD, "WIPE_KEY_SHA512": other, "WIPE_PROTOCOL_EXE": _judge(tmp_path)}, d)
    assert r.returncode == 1, r.stdout + r.stderr
    _is_core_answer(answer := json.loads((d / "answer.json").read_text()))
    assert (answer["verdict"], answer["reason"]) == ("reject", UNTRUSTED)
    case = json.loads((tmp_path / "pieces" / "serve-run" / "private" / "wipe" / "wipe.case.json").read_text())
    assert case["params"]["trusted"] == [] and _decided(case) == answer
    summary = json.loads((d / "summary.json").read_text())
    assert (summary["verdict"], summary["key_pinned"], summary["key_sha512"]) == ("reject", False, PINNED)
    assert json.loads((d / "period.json").read_text())["wipe"]["trusted_sha512"] == other
    headline = json.loads(r.stdout.strip().splitlines()[-1].removeprefix("WIPE "))
    assert (headline["verdict"], headline["reason"], headline["exit"]) == ("reject", UNTRUSTED, 1)


def test_a_run_that_pins_no_key_trusts_the_configurations_wipe_pin_and_no_other(node, tmp_path) -> None:
    """With WIPE_KEY_SHA512 unset the wipe is judged under the key the run's configuration pins for the
    memory attester's wipe (VERITY_PINS), not its PoUS pin, and served.json says the key is the configuration's; the
    stand-in challenger's session, signed under another key, is rejected as any other key's is."""
    d = tmp_path / "pieces" / "wipe"
    env = ({k: v for k, v in node.items() if k != "WIPE_KEY_SHA512"}
           | {"GPU_LEASE_UUID": CARD, "VERITY_PINS": json.dumps(PINS), "WIPE_PROTOCOL_EXE": _judge(tmp_path)})
    r = _wipe(env, d)
    assert r.returncode == 1, r.stdout + r.stderr
    assert json.loads((d / "period.json").read_text())["wipe"]["trusted_sha512"] == AUDITOR_PIN
    assert json.loads((d / "served.json").read_text())["key_pin"] == "configuration"
    _is_core_answer(answer := json.loads((d / "answer.json").read_text()))
    assert (answer["verdict"], answer["reason"]) == ("reject", UNTRUSTED)
    case = json.loads((tmp_path / "pieces" / "serve-run" / "private" / "wipe" / "wipe.case.json").read_text())
    assert case["params"]["trusted"] == [] and _decided(case) == answer


@pytest.mark.parametrize("pin, why", [
    (None, "the pins handed to it give the memory-challenger no key for the wipe"),
    ("", "no challenger key is pinned: WIPE_KEY_SHA512 is set empty, which pins none"),
    ("AB" * 64, "the trusted key is a SHA-512 in 128 lowercase hex digits, not ABABABABABABABAB"),
    ("ab" * 63, "the trusted key is a SHA-512 in 128 lowercase hex digits, not abababababababab")])
def test_with_no_key_pinned_the_wipe_is_refused(node, tmp_path, pin, why) -> None:
    """No key pinned (WIPE_KEY_SHA512 unset and no configuration pin for the wipe, or set empty, whatever the
    configuration pins), or a WIPE_KEY_SHA512 that isn't a
    SHA-512 in lowercase hex: the attach refuses before any lease is asked for and before the wipe, so no session is
    judged under the key it came with."""
    d = tmp_path / "wipe"
    env = {k: v for k, v in node.items() if k != "WIPE_KEY_SHA512"} | ({} if pin is None else {"WIPE_KEY_SHA512": pin})
    assert _refused(_wipe(env | {"VERITY_PINS": json.dumps(PINS)} if pin == "" else env, d), d).startswith(why)
    assert not (tmp_path / "lease.log").exists() and not (tmp_path / "party.json").exists()
    assert not any((d / f).exists() for f in ("served.json", "lake-build.log"))


@pytest.mark.parametrize("given, why", [
    ('{"bogus": 1}', "unknown settings ['bogus']: the wipe's parameters are"),
    ("[1", "the wipe's settings are one JSON object, or a file holding one: "),
    ('{"host": "vy-auditor-2"}', "host, address and command bind the challenger together, not ['host'] alone"),
    ('{"run_keys": true, "simulated": true, "host": "h", "address": "h:1", "command": ["x"]}',
     "run_keys and simulated run the challenger on this node, not on the bound host")])
def test_settings_the_scheme_refuses_are_refused_by_name_before_any_lease(node, tmp_path, given, why) -> None:
    """The launcher's settings are the scheme's (`WIPE.settings`, `check_settings`), one JSON object: one it refuses,
    by name, is refused before the judge's build and any lease."""
    d = tmp_path / "wipe"
    assert _refused(_wipe(node, d, given), d).startswith(why)
    assert not (tmp_path / "lease.log").exists() and _stamps(d) == ["start", "refused"]


def test_a_setting_the_application_states_is_the_applications(node, tmp_path) -> None:
    """A setting the application's configuration states for the wipe (VERITY_PROTOCOLS_CONFIG's ``wipe``) that the
    launcher's settings state otherwise is refused by name, before any lease; stated alike, or only by the
    application, it stands."""
    d = tmp_path / "wipe"
    app = {"private": str(tmp_path / "p"), "wipe": {"reads": W.READS + 1, "idle_mib": 1024}}
    why = _refused(_wipe(node | {"VERITY_PROTOCOLS_CONFIG": json.dumps(app)}, d), d)
    assert why == ("the application's configuration states ['reads'] for the wipe, and the launcher's settings state "
                   "them otherwise: the application's stand")
    assert not (tmp_path / "lease.log").exists()
    app["wipe"]["reads"] = W.READS
    r = _wipe(node | {"VERITY_PROTOCOLS_CONFIG": json.dumps(app), "GPU_LEASE_UUID": CARD,
                      "WIPE_PROTOCOL_EXE": _judge(tmp_path)}, d)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_disk_target_that_is_neither_a_directory_nor_a_block_device_is_refused(node, tmp_path) -> None:
    """A ``disk.paths`` entry that is neither a directory, where the run makes a scratch directory of its own, nor a
    block device, written raw: refused by name before the wipe."""
    d = tmp_path / "wipe"
    nothing = tmp_path / "nothing"
    r = _wipe(node | {"GPU_LEASE_UUID": CARD, "WIPE_PROTOCOL_EXE": _judge(tmp_path)}, d,
              _settings(tmp_path / "vy-auditor2-wipe", disk={"paths": [str(nothing)]}))
    assert _refused(r, d) == f"the disk target {nothing} is neither a directory nor a block device"
    assert not (tmp_path / "party.json").exists()


def test_a_host_with_no_ip_says_what_is_missing(tmp_path) -> None:
    """A host without `ip` (no iproute2), which names the node's control link: refused by name before anything else."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for tool in ("dirname", "basename", "mkdir", "rm", "date", "cat"):
        (bin_ / tool).symlink_to(shutil.which(tool))
    d = tmp_path / "wipe"
    r = subprocess.run(["/bin/bash", str(WRAPPER), str(tmp_path / "serve-run"), "3", str(d)],
                       env={"PATH": str(bin_)}, capture_output=True, text=True)
    assert _refused(r, d) == "this host has no `ip` (iproute2), which names the node's links: apt-get install iproute2"


def test_the_control_link_is_the_nodes_own_unless_the_settings_name_the_links(node, tmp_path) -> None:
    """The record names the node's control link (the default route's NIC) as the launcher reads it; settings that name
    ``links`` are taken as they are."""
    d, e = tmp_path / "pieces" / "wipe", tmp_path / "other" / "wipe"
    env = node | {"GPU_LEASE_UUID": CARD, "WIPE_PROTOCOL_EXE": _judge(tmp_path)}
    assert _wipe(env, d).returncode == 0
    links = json.loads((tmp_path / "pieces" / "serve-run" / "private" / "wipe" / "period.json").read_text())["wipe"]["links"]
    assert links == [{"name": "control", "to": "outside", "mbps": None,
                      "via": "eth9 10.0.0.2: the node's own NIC (ssh, research run)"}]
    named = [{"name": "nic2", "to": "outside", "via": "eth9"}]
    assert _wipe(env, e, _settings(tmp_path / "vy-auditor2-wipe", links=named)).returncode == 0
    assert json.loads((tmp_path / "other" / "serve-run" / "private" / "wipe" / "period.json").read_text())["wipe"]["links"] == named
