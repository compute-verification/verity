"""`streamed_wipe/served_wipe.sh`, the wipe's piece of a served window (served_verdicts.sh's WIPE_CMD), on a
CPU: its demo.sh is `served_wipe_demo.py` (one 32 MiB segment in host memory, `wipe_store.HostMemory`, against
`wipe_side.py`, the challenger's stand-in on a host of its own), and `nvidia-smi` and `gpu-lease` are stand-ins on PATH.
DIR/answer.json is the leg's core answer (verity/wipe/v0) in the form `Answer.toJson` writes (what served_verdicts.py's
`piece_row` reads through `pouw.core_check.answer_of`), the answer `wipe.case.json` decides, with served.json and the
summary beside it; the wipe runs on the serve's leased card, or under a lease of its own. A card that isn't idle and a
gate key that isn't on the node are refused before demo.sh runs, and nothing is killed. The verifier trusts the one
challenger key WIPE_KEY_SHA512 pins, by default vy-auditor-2's wipe party's: a session signed under another is
rejected, and with none pinned the piece refuses before it asks for a card."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shlex
import socket
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from verity.protocols.pous.verifier import streamed_wipe as sw
from verity.protocols.pous.verifier import wipe_party as WP
from verity.protocols.pous.verifier import wipe_period as W

WRAPPER = Path(__file__).resolve().parents[1] / "streamed_wipe" / "served_wipe.sh"
DEMO = WRAPPER.parent / "demo.sh"
FAKE_DEMO = Path(__file__).resolve().parent / "served_wipe_demo.py"
ANSWER_KEYS = ["commitment", "profile", "protocol", "public", "reason", "verdict"]
CARD = "GPU-5e7ed000-0000-0000-0000-000000000000"
#: who decided, as the summary and the headline name it: the stand-in demo.sh's Python, on this host
DECIDED = {"program": "python", "host": socket.gethostname()}
#: the SHA-512 of the key the challenger's stand-in (`wipe_side.py`) signs under: the pin of a deployment of it
PINNED = runpy.run_path(str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "pous" / "verifier" / "tests"
                            / "wipe_side.py"))["key_sha512"]()
#: the SHA-512 of the public key of vy-auditor-2's wipe party: the pin of a run that sets none of its own
AUDITOR_PIN = ("c5f12219aa878038c8e947b0b7c8b4cd90530168603e3eb94f99a1eec793cb41"
               "98fe0cb364d3709ecf8055dc89d86a729d58a046e8871cc21da6cbc9009a7252")
UNTRUSTED = "the session is signed by a key that is not a trusted memory challenger's"
ROOT = Path(__file__).resolve().parents[3]
#: the Lean judge's package, `wipe-protocol`'s
JUDGE = "verity/protocols/pous/device/lean"
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
esac
"""
LEASE = """#!/usr/bin/env bash
echo "$GPU_LEASE_WHO $*" >> "$FAKE_LEASE_LOG"
while [ "$1" != -- ]; do shift; done
shift
GPU_LEASE_UUID=GPU-own0 CUDA_VISIBLE_DEVICES=0 exec "$@"
"""
DEMO_SHIM = """#!/usr/bin/env bash
[ -z "$FAKE_PYTHONPATH" ] || export PYTHONPATH=$FAKE_PYTHONPATH
exec {python} {demo} "$OUT" "${{GPU_LEASE_UUID:-}}" "$@"
"""
LAUNCH = 'export FAKE_PYTHONPATH="${PYTHONPATH:-}" PATH="$FAKE_BIN:$PATH"; exec bash "$@"'
"""The stand-ins first on PATH, and the suite's PYTHONPATH (which the wrapper unsets) kept for the stand-in demo.sh's Python,
so the suite's guard traces what it reads."""


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def node(tmp_path) -> dict[str, str]:
    """The environment of a node with a stand-in card, gpu-lease and demo.sh, WIPE_COMMAND's gate key, and the
    challenger's key pinned (WIPE_KEY_SHA512)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "nvidia-smi", SMI)
    _exe(bin_ / "gpu-lease", LEASE)
    shim = _exe(tmp_path / "demo.sh", DEMO_SHIM.format(python=shlex.quote(sys.executable),
                                                       demo=shlex.quote(str(FAKE_DEMO))))
    key = tmp_path / "vy-auditor2-wipe"
    key.write_text("not a key\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GPU_LEASE_", "WIPE_", "CUDA_VISIBLE"))}
    return env | {"FAKE_BIN": str(bin_), "WIPE_DEMO": str(shim), "WIPE_READS": str(W.READS), "WIPE_NODE": "vy-dc-h200",
                  "WIPE_COMMAND": f"ssh -i {key} -o BatchMode=yes research@10.96.0.36", "WIPE_KEY_SHA512": PINNED,
                  "FAKE_SMI_LOG": str(tmp_path / "smi.log"), "FAKE_SMI_APPS": "", "FAKE_SMI_USED": "0",
                  "FAKE_LEASE_LOG": str(tmp_path / "lease.log"), "PY": sys.executable}


def _wipe(env: dict[str, str], d: Path, *args: str) -> subprocess.CompletedProcess:
    """served_wipe.sh SERVE_DIR 3 d, the serve dir ``serve-run`` beside ``d``."""
    d.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(["bash", "-c", LAUNCH, "_", str(WRAPPER), str(d.parent / "serve-run"), "3", str(d), *args],
                          env=env, capture_output=True, text=True, cwd=d.parent)


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
    """In the serve's lease, before its engine loads: the card checked idle, the wipe of it, and DIR/answer.json the
    accepted core answer of verity/wipe/v0 that the case decides, the challenger the one WIPE_COMMAND starts on
    vy-auditor-2 (trusted-host's premise held), the window's serve dir and the card in served.json, and the
    piece's last line its `WIPE` headline. The case, which holds the signed session, is the serve dir's, private, with the
    composite's piece beside it (served_case.py's `CASE` line before the headline), and DIR keeps none of it."""
    d = tmp_path / "pieces" / "wipe"
    r = _wipe(node | {"GPU_LEASE_UUID": f"{CARD},GPU-other"}, d, "--no-lean")
    assert r.returncode == 0, r.stdout + r.stderr
    answer = json.loads((d / "answer.json").read_text())
    _is_core_answer(answer)
    assert answer["protocol"] == W.PROTOCOL == "verity/wipe/v0" and answer["verdict"] == "accept"
    assert answer == json.loads((d / "wipe.json").read_text())["answer"]
    public = answer["public"]
    assert public["k"] == W.READS and public["memory_bits"] == W.SEGMENT_LABELS * sw.LABEL_BITS
    assert public["certified"] is False and public["foreign_bits"] > 0
    placement = answer["profile"]["placement"]
    assert placement["trusted-host"] == "holds: the memory challenger ran on vy-auditor-2, not the developer's vy-dc-h200"
    profile = {k: v for k, v in answer["profile"].items() if k != "placement"}
    private = tmp_path / "pieces" / "serve-run" / "private" / "wipe"
    case = json.loads((private / "wipe.case.json").read_text())
    assert _decided(case) == answer | {"profile": profile} and not (d / "wipe.case.json").exists()
    assert json.loads((private / "case.json").read_text()) == case
    assert json.loads((private / "answer.json").read_text()) == answer | {"profile": profile}
    assert {f: stat.S_IMODE((private / f).stat().st_mode) for f in ("case.json", "answer.json", "wipe.case.json")} == {
        "case.json": 0o600, "answer.json": 0o600, "wipe.case.json": 0o600} and stat.S_IMODE(private.stat().st_mode) == 0o700
    assert case["evidence"][:64] not in "".join(p.read_text() for p in d.iterdir() if p.is_file())
    party = json.loads((d / "party.json").read_text())
    assert party == {"host": "vy-auditor-2", "command": shlex.split(node["WIPE_COMMAND"]),
                     "address": ["10.96.0.36", 47611]}
    assert json.loads((d / "served.json").read_text()) == {
        "serve_dir": str(tmp_path / "pieces" / "serve-run"), "window": "3", "server": "vy-dc-h200", "card": CARD,
        "run": node.get("RESEARCH_RUN_ID"), "lease": "the serve's: its engine loads on this card after the wipe"}
    summary = json.loads((d / "summary.json").read_text())
    assert summary["verdict"] == "accept" and summary["decided_by"] == DECIDED and summary["k"] == W.READS
    key, = case["params"]["trusted"]
    assert summary["key_sha512"] == hashlib.sha512(bytes.fromhex(key)).hexdigest() == PINNED and summary["key_pinned"]
    demo = json.loads((d / "demo-argv.json").read_text())
    assert demo["lease"] == f"{CARD},GPU-other" and "--wipe-protocol" not in demo["argv"]
    assert demo["argv"] == ["--node", "vy-dc-h200", "--challenger", "vy-auditor-2", "--challenger-address",
                            "10.96.0.36:47611", "--challenger-command", node["WIPE_COMMAND"], "--trusted-sha512", PINNED,
                            "--reads", str(W.READS), "--no-lean"]
    *_, built, last = r.stdout.strip().splitlines()
    assert json.loads(built.removeprefix("CASE ")) == {"piece": str(private), "protocol": W.PROTOCOL, "verdict": "accept"}
    assert last.startswith("WIPE {")
    headline = json.loads(last.removeprefix("WIPE "))
    assert headline | {"fill_GBps": None, "t0": None, "t1": None} == {
        "verdict": "accept", "reason": None, "protocol": W.PROTOCOL, "decided_by": DECIDED, "k": W.READS, "segments": 1,
        "fill_GBps": None, "t0": None, "t1": None, "server": "vy-dc-h200", "card": CARD, "exit": 0}
    assert all(f"-i {CARD}" in line for line in (tmp_path / "smi.log").read_text().splitlines())
    assert not (tmp_path / "lease.log").exists() and not (d / "refused.txt").exists()


def test_outside_a_lease_it_takes_a_card_of_its_own(node, tmp_path) -> None:
    """Without the serve's lease it waits for a card of its own (`gpu-lease 1 --wait` as memory-accounting) and wipes
    that one. A Lean judge built already (WIPE_PROTOCOL_EXE) goes to demo.sh as `--wipe-protocol`, nothing built here."""
    d = tmp_path / "wipe"
    judge = _exe(tmp_path / "wipe-protocol", "#!/bin/sh\nexit 9\n")
    r = _wipe(node | {"WIPE_PROTOCOL_EXE": str(judge)}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    _is_core_answer(answer := json.loads((d / "answer.json").read_text()))
    assert answer["verdict"] == "accept"
    lease = (tmp_path / "lease.log").read_text().splitlines()
    assert len(lease) == 1 and lease[0].startswith(f"memory-accounting 1 --wait --max-min 30 -- bash {WRAPPER} ")
    served = json.loads((d / "served.json").read_text())
    assert served["card"] == "GPU-own0" and served["lease"].startswith("its own (gpu-lease 1 --wait)")
    demo = json.loads((d / "demo-argv.json").read_text())
    assert demo["lease"] == "GPU-own0" and demo["argv"][-2:] == ["--wipe-protocol", str(judge)]
    assert not (d / "lake-build.log").exists()


def test_the_lean_judge_is_built_in_a_copy_that_holds_every_package_it_requires(node, tmp_path) -> None:
    """With no judge built already the wrapper builds `wipe-protocol` before it asks for a card: `lake build` in the
    judge's package of a copy of ``verity/`` outside the tree, removed when the piece ends, and the built judge goes to
    demo.sh. The stand-in `lake` fails as lake does when the copy lacks a package the judge's manifest requires; each
    one is a path requirement inside ``verity/``, so the copy needs no network and nothing else of the tree."""
    d, home = tmp_path / "wipe", tmp_path / "home"
    home.mkdir()
    _exe(Path(node["FAKE_BIN"]) / "lake", LAKE)
    r = _wipe(node | {"HOME": str(home), "FAKE_LAKE_LOG": str(tmp_path / "lake.log")}, d)
    assert r.returncode == 0, r.stdout + r.stderr
    (built,) = (tmp_path / "lake.log").read_text().splitlines()
    package, command = built.split(" ", 1)
    assert command == "build wipe-protocol" and package.endswith("/" + JUDGE) and ROOT not in Path(package).parents
    demo = json.loads((d / "demo-argv.json").read_text())
    assert demo["argv"][-2:] == ["--wipe-protocol", f"{package}/.lake/build/bin/wipe-protocol"]
    assert len((tmp_path / "lease.log").read_text().splitlines()) == 1
    assert not Path(package).exists() and (d / "lake-build.log").exists()
    for p in json.loads((ROOT / JUDGE / "lake-manifest.json").read_text())["packages"]:
        assert p["type"] == "path", f"the judge requires {p['name']} from {p.get('url')}, which no copy holds"
        assert ROOT / "verity" in (ROOT / JUDGE / p["dir"]).resolve().parents, (p["name"], p["dir"])


@pytest.mark.parametrize("apps, used", [("4242, 70212 MiB", "70300"), ("", "2048")])
def test_a_card_that_isnt_idle_is_refused(node, tmp_path, apps, used) -> None:
    """A process on the card, or more of its memory in use than WIPE_IDLE_MIB: refused before demo.sh runs (exit 3, the
    reason in refused.txt, an earlier answer gone), nothing killed."""
    d = tmp_path / "wipe"
    d.mkdir()
    (d / "answer.json").write_text("{}")
    r = _wipe(node | {"GPU_LEASE_UUID": CARD, "FAKE_SMI_APPS": apps, "FAKE_SMI_USED": used}, d, "--no-lean")
    assert r.returncode == 3, r.stdout + r.stderr
    refused = (d / "refused.txt").read_text()
    assert refused.startswith(f"WIPE REFUSED: the card {CARD} isn't idle: ") and "kills nothing" in refused
    assert ("pid 4242, 70212 MiB on it" if apps else "2048 MiB of its memory in use, past WIPE_IDLE_MIB 1024") in refused
    assert r.stdout.strip().splitlines()[-1] == refused.strip()
    assert not any((d / f).exists() for f in ("answer.json", "demo-argv.json", "served.json"))


def test_a_gate_key_not_on_the_node_is_refused(node, tmp_path) -> None:
    """WIPE_COMMAND's ssh key isn't on the node: refused before any lease is asked for, naming the key and who installs
    it."""
    d = tmp_path / "wipe"
    missing = tmp_path / "nowhere" / "vy-auditor2-wipe"
    r = _wipe(node | {"WIPE_COMMAND": f"ssh -i {missing} -o BatchMode=yes research@10.96.0.36"}, d)
    assert r.returncode == 3, r.stdout + r.stderr
    refused = (d / "refused.txt").read_text()
    assert refused.startswith(f"WIPE REFUSED: no wipe gate key {missing} on vy-dc-h200 (") and "infra installs it" in refused
    assert not (tmp_path / "lease.log").exists() and not (d / "demo-argv.json").exists()


def test_a_session_signed_under_another_key_than_the_pinned_one_is_rejected(node, tmp_path) -> None:
    """WIPE_KEY_SHA512 pins another key than the one the challenger signs under: the wipe runs, honest in every read,
    and its answer is a reject (exit 1) naming the key untrusted, decided by a case that trusts no key. The summary
    carries the key the session came with and that it isn't the pinned one."""
    d = tmp_path / "pieces" / "wipe"
    other = hashlib.sha512(b"another key").hexdigest()
    r = _wipe(node | {"GPU_LEASE_UUID": CARD, "WIPE_KEY_SHA512": other}, d, "--no-lean")
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


def test_a_run_that_pins_no_key_trusts_vy_auditor_2s_wipe_party_and_no_other(node, tmp_path) -> None:
    """With WIPE_KEY_SHA512 unset the wipe is judged under the tree's default pin, the digest of the key of
    vy-auditor-2's wipe party: demo.sh gets it as `--trusted-sha512`, and the stand-in challenger's session, signed
    under another key, is rejected as any other key's is, by a case that trusts no key."""
    d = tmp_path / "pieces" / "wipe"
    env = {k: v for k, v in node.items() if k != "WIPE_KEY_SHA512"} | {"GPU_LEASE_UUID": CARD}
    r = _wipe(env, d, "--no-lean")
    assert r.returncode == 1, r.stdout + r.stderr
    argv = json.loads((d / "demo-argv.json").read_text())["argv"]
    assert argv[argv.index("--trusted-sha512") + 1] == AUDITOR_PIN != PINNED
    assert json.loads((d / "period.json").read_text())["wipe"]["trusted_sha512"] == AUDITOR_PIN
    _is_core_answer(answer := json.loads((d / "answer.json").read_text()))
    assert (answer["verdict"], answer["reason"]) == ("reject", UNTRUSTED)
    case = json.loads((tmp_path / "pieces" / "serve-run" / "private" / "wipe" / "wipe.case.json").read_text())
    assert case["params"]["trusted"] == [] and _decided(case) == answer
    summary = json.loads((d / "summary.json").read_text())
    assert (summary["verdict"], summary["key_pinned"], summary["key_sha512"]) == ("reject", False, PINNED)


@pytest.mark.parametrize("pin, why", [
    ("", "no challenger key is pinned: set WIPE_KEY_SHA512, the SHA-512 of the public key of vy-auditor-2's wipe party"),
    ("AB" * 64, "WIPE_KEY_SHA512 is a key's SHA-512 in 128 lowercase hex digits, not ABAB"),
    ("ab" * 63, "WIPE_KEY_SHA512 is a key's SHA-512 in 128 lowercase hex digits, not abab")])
def test_with_no_key_pinned_the_wipe_is_refused(node, tmp_path, pin, why) -> None:
    """WIPE_KEY_SHA512 set empty, which pins no key, or to something that isn't a SHA-512 in lowercase hex: refused
    before any lease is asked for and before demo.sh runs (exit 3, the reason in refused.txt), so no session is judged
    under the key it came with, nor an empty setting taken for the default."""
    d = tmp_path / "wipe"
    r = _wipe(node | {"WIPE_KEY_SHA512": pin}, d)
    assert r.returncode == 3, r.stdout + r.stderr
    refused = (d / "refused.txt").read_text()
    assert refused.startswith(f"WIPE REFUSED: {why}") and r.stdout.strip().splitlines()[-1] == refused.strip()
    assert not (tmp_path / "lease.log").exists()
    assert not any((d / f).exists() for f in ("answer.json", "demo-argv.json", "served.json", "lake-build.log"))


def test_demo_sh_takes_the_wrappers_options(tmp_path) -> None:
    """The wrapper's default demo.sh is beside it and knows every option the wrapper gives it: an unknown one after them
    is the first it refuses, before it writes or builds anything."""
    r = subprocess.run(["bash", str(DEMO), "--node", "n", "--challenger", "c", "--challenger-address", "h:1",
                        "--challenger-command", "x", "--trusted-sha512", PINNED, "--reads", "5500", "--wipe-protocol", "e",
                        "--no-lean", "--bogus"],
                       env=os.environ | {"OUT": str(tmp_path / "out")}, capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout.strip() == "demo.sh: unknown option --bogus (see the header)", r.stdout + r.stderr


@pytest.mark.parametrize("pin, said", [
    ((), "demo.sh: --challenger takes --trusted-sha512: a challenger on its own host is trusted only under its pinned key"),
    (("--trusted-sha512", "AB" * 64), "demo.sh: --trusted-sha512 is a key's SHA-512 in 128 lowercase hex digits, not ABAB")])
def test_demo_sh_takes_a_challenger_on_its_own_host_only_with_its_key_pinned(tmp_path, pin, said) -> None:
    """demo.sh's own rule, whoever calls it: `--challenger` with no `--trusted-sha512`, or with one that isn't a SHA-512
    in lowercase hex, is refused after its options are read, before it writes or builds anything."""
    r = subprocess.run(["bash", str(DEMO), "--challenger", "c", "--challenger-address", "h:1", "--challenger-command", "x",
                        *pin], env=os.environ | {"OUT": str(tmp_path / "out")}, capture_output=True, text=True)
    assert r.returncode == 2 and r.stdout.strip().startswith(said), r.stdout + r.stderr
    assert not any((tmp_path / "out").iterdir())


def test_demo_sh_on_a_host_with_no_ip_says_what_is_missing(tmp_path) -> None:
    """A host without `ip` (no iproute2): demo.sh says so on stderr and exits 3 once its options are read, before it
    writes or builds anything, rather than ending 127 with the error hidden."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for tool in ("dirname", "basename", "mkdir", "hostname"):
        (bin_ / tool).symlink_to(next(Path(d) / tool for d in os.environ["PATH"].split(os.pathsep)
                                      if (Path(d) / tool).is_file()))
    r = subprocess.run(["/bin/bash", str(DEMO), "--no-lean"], env={"PATH": str(bin_), "OUT": str(tmp_path / "out")},
                       capture_output=True, text=True)
    assert r.returncode == 3 and r.stdout == "", r.stdout + r.stderr
    assert r.stderr.strip() == ("demo.sh: this host has no `ip` (iproute2), which names the node's links: apt-get "
                                "install iproute2")
    assert not any((tmp_path / "out").iterdir())
