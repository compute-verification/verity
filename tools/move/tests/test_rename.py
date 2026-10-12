"""`rename.py`, the rename move: where a word is renamed and where it stands, what it moves, the locks' records it leaves for
`audit.py --update --moved`, the pins it rewrites, that a second run changes nothing, and that no tracked path is left to move
on this tree."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("rename_move_under_test", HERE.parent / "rename.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

WORDS = M.Words(tomllib.loads((HERE.parent / "rename_map.toml").read_text()))


def test_a_word_is_renamed_as_a_word_or_a_part_of_an_identifier():
    s = WORDS.sub
    assert s("network_warden, I.warden, witnessWarden, gatewayLeaf") == \
        "network_certifier, I.certifier, witnessCertifier, firewallLeaf"
    assert s("the wardens; Wardens; WARDEN_TAG") == "the certifiers; Certifiers; CERTIFIER_TAG"
    assert s("rewarden, wardenship, Gatewayed") == "rewarden, wardenship, Gatewayed"
    assert s("the prover gateway and the verifier gateway") == "the firewall and the challenger"
    assert s("the verifier's gateway, a gateway") == "the challenger, a firewall"
    assert s("the batch verifier and its GPU workers") == "the verifier and its workers"
    assert s("audit.TimedVerifier, a timed verifier, test_the_timed_verifier") == \
        "audit.MemoryChallenger, a memory challenger, test_the_memory_challenger"
    assert s("the verifier, ContinuousVerifier, VerifierKey, protocol.Verifier, run_timed") == \
        "the verifier, ContinuousMemoryChallenger, VerifierKey, protocol.Verifier, run_timed"


def test_phrases_come_first_and_start_at_a_word():
    s = WORDS.sub
    assert s("Returns the warden's records.") == "Returns the network certificates."
    assert s("An ingress warden's records:") == "An ingress certifier's network certificates:"
    assert s("the warden's clock") == "the certifier's clock"
    assert s("no 2-GPU worker") == "no 2-GPU worker"


def test_what_is_kept_stands():
    s = WORDS.sub
    tags = 'ROW_TAG = b"verity-network-warden/row/v1"; RECORD_TAG = b"verity-network-warden/record/v1"'
    assert s(tags) == tags
    assert s("see note:20261001T0000Z-warden-x and art:0123abcd, cursor/warden-fix-1234") == \
        "see note:20261001T0000Z-warden-x and art:0123abcd, cursor/warden-fix-1234"
    statement = "a recursive session's inner statement: the gateway commits every tree top under a salt, ZK off"
    assert s(f'"use": "{statement}"') == f'"use": "{statement}"'
    assert s("node.gateway.id", "tools/check/guard/x.py") == "node.gateway.id"
    assert s("node.gateway.id", "verity/x.py") == "node.firewall.id"
    assert s("the prover gateway", "tools/check/x.py") == "the firewall"


def test_paths_and_names_are_renamed_part_by_part():
    assert WORDS.path("verity/protocols/accounting/communication/warden/tests/test_network_warden_grid.py") == \
        "verity/protocols/accounting/communication/certifier/tests/test_network_certifier_grid.py"
    assert WORDS.path("verity/Security/Proofs/Warden/Sanity.lean") == "verity/Security/Proofs/NetworkCertifier/Sanity.lean"
    assert WORDS.path("verity/Security/Definitions/Warden.lean") == "verity/Security/Definitions/NetworkCertifier.lean"
    assert WORDS.sub("import Specs.Warden.Guarantees\n#check NetTiming.Warden.obs") == \
        "import Specs.NetworkCertifier.Guarantees\n#check NetTiming.Certifier.obs"
    assert WORDS.name("NetTiming.Interaction.warden") == "NetTiming.Interaction.certifier"
    assert WORDS.path("benchmarks/pous/band_gpu/live.py") == "benchmarks/pous/band_gpu/memory_challenger.py"
    assert WORDS.path("benchmarks/other/band_gpu/live.py") == "benchmarks/other/band_gpu/live.py"


def test_words_in_hold_only_in_their_files():
    text = 'class Verifier:\n    key: VerifierKey\nv = protocol.Verifier(s)\n"""the verifier process"""\n'
    assert WORDS.sub(text, "verity/protocols/accounting/space/pous/protocol.py") == \
        'class MemoryChallenger:\n    key: VerifierKey\nv = protocol.MemoryChallenger(s)\n"""the memory challenger process"""\n'
    assert WORDS.sub(text, "verity/protocols/accounting/work/pouw/protocol.py") == text
    harness = 'LIVE = HERE / "live.py"\nfrom .live import scheme_of\nALIVE = 1\n'
    assert WORDS.sub(harness, "benchmarks/pous/band_gpu/harness.py") == \
        'MEMORY_CHALLENGER = HERE / "memory_challenger.py"\nfrom .memory_challenger import scheme_of\nALIVE = 1\n'
    assert WORDS.sub(harness, "benchmarks/pous/band_gpu/codec.py") == \
        'LIVE = HERE / "memory_challenger.py"\nfrom .memory_challenger import scheme_of\nALIVE = 1\n'
    assert WORDS.sub(harness, "benchmarks/other/harness.py") == harness


def test_the_certifiers_program_is_network_certifier_where_it_stands():
    lean = "verity/protocols/accounting/communication/warden/lean"
    assert WORDS.path(f"{lean}/Warden/Grid.lean") == \
        "verity/protocols/accounting/communication/certifier/lean/NetworkCertifier/Grid.lean"
    assert WORDS.path(f"{lean}/Warden.lean") == "verity/protocols/accounting/communication/certifier/lean/NetworkCertifier.lean"
    assert WORDS.sub('name = "warden_program"\ndefaultTargets = ["Warden", "warden-difftest"]\nroot = "Warden.DifftestMain"\n',
                     f"{lean}/lakefile.toml") == \
        'name = "certifier_program"\ndefaultTargets = ["NetworkCertifier", "certifier-difftest"]\nroot = ' \
        '"NetworkCertifier.DifftestMain"\n'
    code = ("import Definitions.Warden\nimport Warden.Ingress\n(`warden_program`, `Warden`)\n"
            "def f (s : Warden.ClockSync) (w : NetTiming.Warden 1 1 1 Unit Bool (Obs 1 1 1)) := witnessWarden_x\n")
    assert WORDS.sub(code, "verity/Security/Proofs/Warden/Code.lean") == (
        "import Definitions.NetworkCertifier\nimport NetworkCertifier.Ingress\n(`certifier_program`, `NetworkCertifier`)\n"
        "def f (s : NetworkCertifier.ClockSync) (w : NetTiming.Certifier 1 1 1 Unit Bool (Obs 1 1 1)) := witnessCertifier_x\n")
    assert WORDS.sub("The `Warden` type", "verity/Security/Specs/Warden/Assumptions.lean") == "The `Certifier` type"
    assert WORDS.sub("mirror `protocols/network_warden/verity_network_warden/schedule.py` one",
                     "verity/Security/Definitions/Warden/Schedule.lean") == \
        "mirror `verity/protocols/accounting/communication/certifier/schedule.py` one"
    assert WORDS.sub("through `Warden.fault`", "verity/protocols/accounting/communication/warden/active.py") == \
        "through `Certifier.fault`"
    assert WORDS.sub('"meaning": ["Specs.Warden.Guarantees", "Warden", "Flock"]', "verity/Security/lean-audit.json") == \
        '"meaning": ["Specs.NetworkCertifier.Guarantees", "NetworkCertifier", "Flock"]'
    lock = "verity/Security/lean-audit.json"
    assert WORDS.name("Warden.SyncSet.deltas", lock) == "NetworkCertifier.SyncSet.deltas"
    assert WORDS.name("NetTiming.Warden.obs", lock) == "NetTiming.Certifier.obs"


def test_the_locks_records_are_left_for_the_update():
    lock = {"roots": ["Proofs.Warden"], "layers": {"Specs.Warden.Guarantees": {"imports": ["Definitions.Warden"]}},
            "guarantees": {"NetTiming.SecurityProofs.W1": {"signature": "∀ w : NetTiming.Warden, True"}},
            "reads": {"Definitions.Warden.Run": {"definitions": {"NetTiming.Warden": "ab"}}},
            "dependencies": {"Warden": "cd"}, "runs": {"Proofs.Warden.Main": {"why": "the warden's records"}}}
    text = json.dumps(lock, indent=1, ensure_ascii=False)
    out = json.loads(WORDS.sub(text, "x/lean-audit.json", M.record_spans(text)))
    assert out["roots"] == ["Proofs.NetworkCertifier"]
    assert out["layers"] == {"Specs.NetworkCertifier.Guarantees": {"imports": ["Definitions.NetworkCertifier"]}}
    assert out["runs"] == {"Proofs.NetworkCertifier.Main": {"why": "the network certificates"}}
    assert {k: out[k] for k in M.RECORDS} == {k: lock[k] for k in M.RECORDS}


def test_lock_moves_are_the_shortest_renamed_prefixes(tmp_path):
    lock = {"guarantees": {"NetTiming.SecurityProofs.W1": {}},
            "reads": {"M": {"definitions": {n: "00" for n in (
                "NetTiming.Warden", "NetTiming.Warden.obs", "NetTiming.Interaction.warden", "Pous.TimedINC",
                "Warden.Obs", "Warden.Obs.rows", "Warden.expectedRow")}}}}
    (tmp_path / "lean-audit.json").write_text(json.dumps(lock))
    assert M.lock_moves(tmp_path, WORDS, ["lean-audit.json"]) == {
        "NetTiming.Interaction.warden": "NetTiming.Interaction.certifier", "NetTiming.Warden": "NetTiming.Certifier",
        "Warden": "NetworkCertifier"}


def test_the_recorded_paths_are_the_moves_up_to_their_first_renamed_part():
    moves = [("v/warden/a.py", "v/certifier/a.py"), ("v/warden/lean/Warden/B.lean", "v/certifier/lean/NetworkCertifier/B.lean"),
             ("S/Proofs/Warden.lean", "S/Proofs/NetworkCertifier.lean"), ("S/Proofs/Warden/C.lean", "S/Proofs/NetworkCertifier/C.lean")]
    assert M.renamed_paths(moves) == {"S/Proofs/Warden.lean": "S/Proofs/NetworkCertifier.lean",
                                      "S/Proofs/Warden/": "S/Proofs/NetworkCertifier/", "v/warden/": "v/certifier/"}
    assert M.renamed_paths(moves[:1]) == M.renamed_paths(moves[:2])       # a branch's new file under a renamed directory
    with pytest.raises(SystemExit, match="would move to both"):
        M.renamed_paths([("v/warden/a.py", "v/certifier/a.py"), ("v/warden/b.py", "v/guard/b.py")])


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def toy(tmp_path):
    files = {
        "w/warden/grid.py": "class Warden:\n    TAG = b'verity-network-warden/row/v1'\n",
        "w/warden/tests/test_network_warden_grid.py": "from w.warden.grid import Warden\n",
        "Sec/Proofs/lakefile.toml": 'roots = ["Proofs.Warden"]\n',
        "Sec/Proofs/Warden.lean": "import Definitions.Warden\n",
        "Sec/lean-audit.json": json.dumps({"roots": ["Proofs.Warden"],
                                           "reads": {"Definitions.Warden": {"definitions": {"NetTiming.Warden": "00"}}}},
                                          indent=1) + "\n",
        "archive/old.md": "the batch verifier\n",
        "CHANGELOG.md": "- the warden landed\n",
        "README.md": "The prover gateway; `w/warden/grid.py`.\n",
        M.MOVES: json.dumps({"about": "a toy", "moves": [{"paths": {"old/": "w/"}}]}, indent=2) + "\n",
    }
    for p, t in files.items():
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(t)
    lake = (tmp_path / "Sec/Proofs/lakefile.toml").read_bytes()
    (tmp_path / "pous").mkdir()
    (tmp_path / "pous/TRUSTED.sha256").write_text(
        f"{hashlib.sha256(lake).hexdigest()}  ../Sec/Proofs/lakefile.toml\n"
        f"{'0' * 64}  ../README.md\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "toy")
    return tmp_path


def test_the_move_on_a_toy_tree(toy, capsys):
    assert M.main(["--root", str(toy)]) == 0
    tracked = set(git(toy, "ls-files").split())
    assert {"w/certifier/grid.py", "w/certifier/tests/test_network_certifier_grid.py",
            "Sec/Proofs/NetworkCertifier.lean"} <= tracked
    assert not any("warden" in p.lower() for p in tracked)
    assert (toy / "w/certifier/grid.py").read_text() == "class Certifier:\n    TAG = b'verity-network-warden/row/v1'\n"
    assert (toy / "w/certifier/tests/test_network_certifier_grid.py").read_text() == "from w.certifier.grid import Certifier\n"
    assert (toy / "README.md").read_text() == "The firewall; `w/certifier/grid.py`.\n"
    assert (toy / "archive/old.md").read_text() == "the batch verifier\n"
    assert (toy / "CHANGELOG.md").read_text() == "- the warden landed\n"
    lock = json.loads((toy / "Sec/lean-audit.json").read_text())
    assert lock["roots"] == ["Proofs.NetworkCertifier"] and "Definitions.Warden" in lock["reads"]
    assert (toy / "Sec/Proofs/NetworkCertifier.lean").read_text() == "import Definitions.NetworkCertifier\n"
    lake = (toy / "Sec/Proofs/lakefile.toml").read_bytes()
    assert lake == b'roots = ["Proofs.NetworkCertifier"]\n'
    assert (toy / "pous/TRUSTED.sha256").read_text() == (f"{hashlib.sha256(lake).hexdigest()}  ../Sec/Proofs/lakefile.toml\n"
                                                        f"{'0' * 64}  ../README.md\n")   # a stale pin stays stale
    assert json.loads((toy / M.MOVES).read_text())["moves"] == [
        {"paths": {"old/": "w/"}},
        {"paths": {"Sec/Proofs/Warden.lean": "Sec/Proofs/NetworkCertifier.lean", "w/warden/": "w/certifier/"}}]
    status = git(toy, "status", "--porcelain", "--untracked-files=all").splitlines()
    assert status and all(line[1] == " " for line in status)                           # everything staged
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "move")
    head = git(toy, "rev-parse", "HEAD:")
    capsys.readouterr()
    assert M.main(["--root", str(toy)]) == 0
    assert "files rewritten: 0, files moved: 0, TRUSTED.sha256 files repinned: 0" in capsys.readouterr().out
    assert git(toy, "status", "--porcelain") == "" and git(toy, "rev-parse", "HEAD:") == head


def test_the_moves_are_those_of_the_tree_re_recorded(toy, capsys):
    """A branch carried across the move keeps its records under the old names, so the map its `--update --moved` takes is
    `--moves` on its own tree: with a name only its lock records, and none once the lock is re-recorded under the new names."""
    path = toy / "Sec/lean-audit.json"
    lock = json.loads(path.read_text())
    path.write_text(json.dumps(lock | {"guarantees": {"Flock.Guarantees.gatewayLeaf": {}}}, indent=1) + "\n")
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "a branch's guarantee")
    assert M.main(["--root", str(toy)]) == 0
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "move")
    capsys.readouterr()
    assert M.main(["--root", str(toy), "--moves"]) == 0
    assert json.loads(capsys.readouterr().out) == {"Flock.Guarantees.gatewayLeaf": "Flock.Guarantees.firewallLeaf",
                                                   "NetTiming.Warden": "NetTiming.Certifier"}
    path.write_text(json.dumps({"guarantees": {"Flock.Guarantees.firewallLeaf": {}},
                                "reads": {"Definitions.Warden": {"definitions": {"NetTiming.Certifier": "00"}}}}))
    assert M.main(["--root", str(toy), "--moves"]) == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_it_refuses_a_dirty_worktree(toy):
    (toy / "README.md").write_text("edited\n")
    with pytest.raises(SystemExit, match="isn't clean"):
        M.main(["--root", str(toy)])


def test_two_files_landing_at_one_path_fail(toy):
    (toy / "w/certifier").mkdir(parents=True)
    (toy / "w/certifier/grid.py").write_text("")
    git(toy, "add", "-A")
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "clash")
    with pytest.raises(SystemExit, match="two files would land at one path"):
        M.main(["--root", str(toy)])


def test_the_tree_is_renamed():
    files = [p for p in git(ROOT, "ls-files", "-z").split("\0") if p]
    left = [p for p in files if not WORDS.excluded(p) and WORDS.path(p) != p]
    assert not left, f"rename.py would still move: {left[:5]}"
