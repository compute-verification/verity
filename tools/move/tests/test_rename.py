"""`rename.py`, the rename move, with a past move's word maps as fixtures (`fixtures/`): where a word is renamed and where it
stands, what it moves, the locks' records it leaves for the Lean check's `--update --moved`, the pins it rewrites, and that a
second run changes nothing."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rename_move_under_test", HERE.parent / "rename.py")
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M      # `--jobs` hands its workers `M._rewrite`, which pickle finds by its module's name
spec.loader.exec_module(M)

MAP = HERE / "fixtures" / "rename_map.toml"
WORDS = M.Words(tomllib.loads(MAP.read_text()))


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


def test_a_phrase_a_line_break_splits_keeps_the_break():
    s = WORDS.sub
    assert s("reads the warden's\n    records whole") == "reads the network\n    certificates whole"
    assert s("# reads the\n# warden's records") == "# reads the\n# network certificates"
    assert s("/-- the warden's records\n  -- and the prover\n  -- gateway -/") == \
        "/-- the network certificates\n  -- and the firewall -/"
    assert s("the prover\n\ngateway") == "the prover\n\nfirewall"
    assert s('f"{n}: the warden\'s "\n        f"records {x}"') == 'f"{n}: the network "\n        f"certificates {x}"'
    assert s('"a prover "\n    f"gateway {x}"') == '"a prover "\n    f"gateway {x}"'


def test_a_phrase_that_starts_with_an_apostrophe_follows_a_possessive():
    words = M.Words({"words": {"'s firewall": "'s ZK server", "firewall": "zk_server"}})
    assert words.sub("Rust's firewall sends; the audit's\n    firewall; a firewall_state") == \
        "Rust's ZK server sends; the audit's\n    ZK server; a zk_server_state"


def test_what_is_kept_stands():
    s = WORDS.sub
    tags = 'ROW_TAG = b"verity-network-warden/row/v1"; RECORD_TAG = b"verity-network-warden/record/v1"'
    assert s(tags) == tags
    ids = "see note:20261001T0000Z-warden-x and art:0123abcd, cursor/warden-fix-1234"
    assert s(ids) == ids
    statement = "a recursive session's inner statement: the gateway commits every tree top under a salt, ZK off"
    assert s(f'"use": "{statement}"') == f'"use": "{statement}"'
    assert s("node.gateway.id", "tools/check/guard/x.py") == "node.gateway.id"
    assert s("node.gateway.id", "verity/x.py") == "node.firewall.id"
    assert s("the prover gateway", "tools/check/x.py") == "the firewall"


def test_keep_paths_holds_a_path_and_not_the_files_text():
    words = M.Words({"words": {"firewall": "zk_server"}, "keep_paths": {"pkg/firewall/**": ["firewall"]}})
    assert words.path("pkg/firewall/tests/test_firewall.py") == "pkg/firewall/tests/test_firewall.py"
    assert words.path("pkg/other/firewall.py") == "pkg/other/zk_server.py"
    assert words.sub("the firewall serves", "pkg/firewall/tests/test_firewall.py") == "the zk_server serves"


def test_a_held_file_is_named_where_it_stands():
    words = M.Words({"words": {"firewall": "zk_server", "draw": "sample"},
                     "keep_paths": {"pkg/firewall/**": ["firewall"]}, "keep_in": {"a/draw.py": ["draw"]}})
    files = ["pkg/firewall/tests/test_firewall.py", "pkg/firewall/README.md", "a/draw.py", "b/draw.py", "doc.md"]
    assert words.hold(files) == ["pkg/firewall/tests/test_firewall.py", "pkg/firewall/README.md", "a/draw.py"]
    assert words.sub("see pkg/firewall/tests/test_firewall.py, test_firewall.py, a/draw.py and b/draw.py", "doc.md") == \
        "see pkg/firewall/tests/test_firewall.py, test_firewall.py, a/draw.py and b/sample.py"


def test_a_name_in_all_follows_what_it_exports_where_a_quoted_key_stands():
    words = M.Words({"words": {"draw": "sample"}, "keep": {"patterns": ['"draw"']}})
    module = '__all__ = ["draw",\n           "other"]\n\n\ndef draw():\n    return {"draw": 1}\n'
    assert words.sub(module, "m.py") == '__all__ = ["sample",\n           "other"]\n\n\ndef sample():\n    return {"draw": 1}\n'
    assert words.sub(module, "m.md") == module.replace("def draw", "def sample")
    kept = M.Words({"words": {"draw": "sample"}, "keep": {"patterns": ['"draw"', r"\bdraw_salt\b"]}})
    module = '__all__ = ["draw_salt", "draw"]\n\n\ndef draw_salt():\n    pass\n\n\ndef draw():\n    pass\n'
    assert kept.sub(module, "m.py") == '__all__ = ["draw_salt", "sample"]\n\n\ndef draw_salt():\n    pass\n\n\ndef sample():\n    pass\n'
    tag = M.Words({"words": {"DRAW": "SAMPLE"}, "keep": {"patterns": [r"(?<=[\x22])DRAW(?=[\x22 ])"]}})
    module = '__all__ = ["DRAW"]\nDRAW = "pouw/draw"\nlines(p, "DRAW")\n'
    assert tag.sub(module, "m.py") == '__all__ = ["SAMPLE"]\nSAMPLE = "pouw/draw"\nlines(p, "DRAW")\n'


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


def test_a_map_of_words_in_alone_leaves_every_other_file():
    words = M.Words({"words": {}, "words_in": [{"files": ["cli/*.py"], "words": {'"examples"': '"demos"'}}]})
    assert words.sub('ROOT / "examples"\n', "cli/run.py") == 'ROOT / "demos"\n'
    assert words.sub('ROOT / "examples"\n', "other/run.py") == 'ROOT / "examples"\n'
    assert words.path("examples/run.py") == "examples/run.py"


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


def test_a_prefix_whose_names_move_otherwise_than_under_it_is_left_out(tmp_path):
    """A private name holds its module twice (`_private.RemoteAudit.Plan.0.RemoteAudit.spans`): its prefix's move would
    map it back under the wrong old name, which `check.py --moved` refuses, so only the full names move."""
    words = M.Words({"words": {"RemoteAudit": "PoubCheck"}})
    names = ("_private.RemoteAudit.Plan.0.RemoteAudit.spans", "_private.RemoteAudit.Line.0.RemoteAudit.Line.sumOver",
             "RemoteAudit.Plan")
    (tmp_path / "lean-audit.json").write_text(json.dumps({"reads": {"M": {"definitions": {n: "00" for n in names}}}}))
    assert M.lock_moves(tmp_path, words, ["lean-audit.json"]) == {
        "RemoteAudit": "PoubCheck",
        "_private.RemoteAudit.Line.0.RemoteAudit.Line.sumOver": "_private.PoubCheck.Line.0.PoubCheck.Line.sumOver",
        "_private.RemoteAudit.Plan.0.RemoteAudit.spans": "_private.PoubCheck.Plan.0.PoubCheck.spans"}


def test_a_namespace_whose_names_are_renamed_inside_too_is_listed_in_full(tmp_path):
    """`Component` renames a namespace and the names under it that end in it: the prefix alone would give
    `Subprotocol.LineHeldAtComponent`, and the prefix with its override under it is a move `check.py --moved` can't read
    back, so each name is listed in full."""
    words = M.Words({"words": {"Component": "Subprotocol"}})
    lock = {"guarantees": {"R.Component.SecurityProofs.lineHeldAtComponent": {}},
            "reads": {"M": {"definitions": {"R.Component.LineHeldAtComponent": "00", "R.Component.Held": "00"}}}}
    (tmp_path / "lean-audit.json").write_text(json.dumps(lock))
    moves = M.lock_moves(tmp_path, words, ["lean-audit.json"])
    assert moves == {"R.Component.Held": "R.Subprotocol.Held",
                     "R.Component.LineHeldAtComponent": "R.Subprotocol.LineHeldAtSubprotocol",
                     "R.Component.SecurityProofs.lineHeldAtComponent": "R.Subprotocol.SecurityProofs.lineHeldAtSubprotocol"}
    assert not [(o, o2) for o, n in moves.items() for o2, n2 in moves.items()
                if n2.startswith(n + ".") and o2 != o + n2[len(n):]]
    lock["reads"]["M"]["definitions"]["R.Component"] = "00"
    (tmp_path / "lean-audit.json").write_text(json.dumps(lock))
    with pytest.raises(SystemExit, match="rename them in two moves"):
        M.lock_moves(tmp_path, words, ["lean-audit.json"])


def test_a_read_definition_is_renamed_as_its_modules_file_renames_it(tmp_path):
    """A [[words_in]] word holds in its files, so a read definition moves where its module's file is one of them, and stands
    where it is another's, or in a file the map excludes."""
    words = M.Words({"words": {}, "words_in": [{"files": ["S/Definitions/Compose.lean", "S/Old/Gates.lean"],
                                                "words": {"Declared": "RegisteredCircuit"}}],
                     "rewrite": {"exclude": ["S/Old/**"]}})
    lock = {"reads": {"Definitions.Compose": {"definitions": {"Compose.Declared": "00"}},
                      "Definitions.Other": {"definitions": {"Other.Declared": "00"}},
                      "Old.Gates": {"definitions": {"Gates.Declared": "00"}}}}
    (tmp_path / "lean-audit.json").write_text(json.dumps(lock))
    files = ["lean-audit.json", "S/Definitions/Compose.lean", "S/Definitions/Other.lean", "S/Old/Gates.lean"]
    assert M.lock_moves(tmp_path, words, files) == {"Compose.Declared": "Compose.RegisteredCircuit"}


def _tree(root: Path, texts: dict[str, str]) -> list[str]:
    for path, text in texts.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    return sorted(texts)


def test_a_module_is_its_path_below_its_packages_source_directory(tmp_path):
    """`Security/Properties` and `Security/Proofs` both take `Security` as their source directory, so `Properties.X` is
    `Security/Properties/X.lean`, and `Security/Proofs/Properties/X.lean`, whose path ends alike, is `Proofs.Properties.X`:
    a map scoped to the Properties file renames what the Properties lock reads from it."""
    files = _tree(tmp_path, {
        "Security/Properties/lakefile.toml": 'name = "security_properties"\nsrcDir = ".."\n',
        "Security/Proofs/lakefile.toml": 'name = "security_proofs"\nsrcDir = ".."\n',
        "Security/Properties/RemoteAudit.lean": "", "Security/Proofs/Properties/RemoteAudit.lean": "",
        "Security/Properties/lean-audit.json": json.dumps(
            {"reads": {"Properties.RemoteAudit": {"definitions": {"Properties.RemoteAudit.drawnOf": "00"}}}})})
    words = M.Words({"words": {}, "words_in": [{"files": ["Security/Properties/RemoteAudit.lean"],
                                                "words": {"drawnOf": "sampledOf"}}]})
    assert M.module_files(tmp_path, files)["Properties.RemoteAudit"] == [
        ("Security/Properties/RemoteAudit.lean", "Security/Properties")]
    assert M.module_files(tmp_path, files)["Proofs.Properties.RemoteAudit"] == [
        ("Security/Proofs/Properties/RemoteAudit.lean", "Security/Proofs")]
    assert M.lock_moves(tmp_path, words, files) == {"Properties.RemoteAudit.drawnOf": "Properties.RemoteAudit.sampledOf"}


def test_a_module_two_packages_define_is_read_from_the_locks_own_or_renamed_alike(tmp_path):
    """`Check.Main` is `A`'s and `B`'s: `A`'s lock reads `A`'s, so a word held in `A`'s file renames it. `C`'s lock reads it
    from neither of its own: alike in both files it is renamed, and apart it is refused before any records run, naming the
    lock, the definition and each file's rename."""
    texts = {"A/lakefile.toml": 'name = "a"\n', "B/lakefile.toml": 'name = "b"\n', "C/lakefile.toml": 'name = "c"\n',
             "A/Check/Main.lean": "", "B/Check/Main.lean": "",
             "A/lean-audit.json": json.dumps({"reads": {"Check.Main": {"definitions": {"Check.Main.horizon": "00"}}}})}
    files = _tree(tmp_path, texts)
    apart = M.Words({"words": {}, "words_in": [{"files": ["A/Check/Main.lean"], "words": {"horizon": "gridTicks"}}]})
    assert M.lock_moves(tmp_path, apart, files) == {"Check.Main.horizon": "Check.Main.gridTicks"}
    files = _tree(tmp_path, {**texts, "C/lean-audit.json": texts["A/lean-audit.json"]})
    with pytest.raises(SystemExit, match=r"C/lean-audit.json: Check.Main.horizon is read from a module that 2 packages "
                                         r"define.*A/Check/Main.lean: Check.Main.gridTicks; B/Check/Main.lean: Check.Main.horizon"):
        M.lock_moves(tmp_path, apart, files)
    alike = M.Words({"words": {"horizon": "gridTicks"}})
    assert M.lock_moves(tmp_path, alike, files) == {"Check.Main.horizon": "Check.Main.gridTicks"}


def test_a_lock_under_tests_fixtures_is_a_tests_data_and_never_read(tmp_path):
    """A fixture lock holds what its test needs (`unread`'s holds `reads` of lists) and is no package's record: under a map
    that excludes no fixture, the moves are the package lock's alone."""
    words = M.Words({"words": {"Warden": "NetworkCertifier"}})
    locks = {"Security/lean-audit.json": {"guarantees": {"NetTiming.SecurityProofs.W1": {}},
                                          "reads": {"M": {"definitions": {"Warden.Obs": "00"}}}},
             "tools/verity/lean/tests/fixtures/unread/Security/lean-audit.json": {
                 "guarantees": {"NetTiming.Warden.spec": {}}, "reads": {"NetTiming.Warden.spec": ["NetTiming.Warden.helper"]}}}
    for path, lock in locks.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(json.dumps(lock))
    assert M.lock_moves(tmp_path, words, list(locks)) == {"Warden": "NetworkCertifier"}


def test_a_module_that_shares_a_namespaces_name_moves_where_it_is_one():
    """`Properties.RemoteAudit.Line` and `.Line.Record` are modules, a namespace and a structure: a whole import line or
    quoted string names the module, and everywhere else the name stands."""
    areas = M.Words(tomllib.loads((HERE / "fixtures" / "areas_map.toml").read_text()))
    code = ("import Properties.RemoteAudit.Line.Record\nimport Proofs.Properties.RemoteAudit.Line\n"
            "open RemoteAudit\n  Properties.RemoteAudit.Line\n"
            "theorem t (r : Properties.RemoteAudit.Line.Record) : Properties.RemoteAudit.Line.LineHeld r := x\n"
            'roots = ["Properties.RemoteAudit.Line", "Properties.RemoteAudit.Line.LineHeld"]\n')
    assert areas.sub(code, "Security/Proofs/Properties/X.lean") == (
        "import Properties.Accounting.Remote.Line.Record\nimport Proofs.Properties.Accounting.Remote.Line\n"
        "open RemoteAudit\n  Properties.RemoteAudit.Line\n"
        "theorem t (r : Properties.RemoteAudit.Line.Record) : Properties.RemoteAudit.Line.LineHeld r := x\n"
        'roots = ["Properties.Accounting.Remote.Line", "Properties.RemoteAudit.Line.LineHeld"]\n')


def test_a_namespace_that_shares_a_modules_name_stands(tmp_path):
    """The areas move renames module `Properties.CertifierDevice` and keeps namespace `Properties.CertifierDevice`: the
    rewrite and the declaration renames the locks need agree on what stands."""
    areas = M.Words(tomllib.loads((HERE / "fixtures" / "areas_map.toml").read_text()))
    code = ("import Properties.CertifierDevice\nimport Proofs.Properties.CertifierNetwork.Runs\n"
            "namespace Properties.CertifierDevice\nopen Properties.NetworkCertifier Properties.NetworkCertifier.Bytes\n"
            "/-- `Properties.CertifierDevice.failClosed`, stated in `Properties.CertifierNetwork.Runs`. -/\n"
            "end Properties.CertifierDevice\n")
    assert areas.sub(code, "Security/Proofs/Properties/X.lean") == (
        "import Properties.NetworkCertifier.Device\nimport Proofs.Properties.NetworkCertifier.DeviceOutput.Runs\n"
        "namespace Properties.CertifierDevice\nopen Properties.NetworkCertifier Properties.NetworkCertifier.Bytes\n"
        "/-- `Properties.CertifierDevice.failClosed`, stated in `Properties.NetworkCertifier.DeviceOutput.Runs`. -/\n"
        "end Properties.CertifierDevice\n")
    assert areas.path("Security/Proofs/Properties/NetworkCertifier/Check.lean") == \
        "Security/Proofs/Properties/NetworkCertifier/Check/Check.lean"
    lock = {"layers": {"Properties.CertifierDevice": {"imports": ["Properties.Identity"]}},
            "guarantees": {"Properties.CertifierDevice.failClosed": {}},
            "reads": {"Definitions.Population.Exfiltration": {"definitions": {
                "Properties.Identity.TaggedIsFlock": "00", "Population.Exfil.game": "00", "NciCheck.Properties.x": "00"}}}}
    (tmp_path / "lean-audit.json").write_text(json.dumps(lock))
    assert M.lock_moves(tmp_path, areas, ["lean-audit.json"]) == {}
    out = json.loads(areas.sub(json.dumps(lock), "Security/Properties/lean-audit.json", M.record_spans(json.dumps(lock))))
    assert out["layers"] == {"Properties.NetworkCertifier.Device": {"imports": ["Properties.Hash.Identity"]}}


def test_the_recorded_paths_are_the_moves_up_to_the_end_of_each_renamed_span():
    moves = [("v/warden/a.py", "v/certifier/a.py"), ("v/warden/lean/Warden/B.lean", "v/certifier/lean/NetworkCertifier/B.lean"),
             ("S/Proofs/Warden.lean", "S/Proofs/NetworkCertifier.lean"), ("S/Proofs/Warden/C.lean", "S/Proofs/NetworkCertifier/C.lean"),
             ("v/warden/tests/test_warden.py", "v/certifier/tests/test_certifier.py")]
    assert M.renamed_paths(moves) == {"S/Proofs/Warden.lean": "S/Proofs/NetworkCertifier.lean",
                                      "S/Proofs/Warden/": "S/Proofs/NetworkCertifier/", "v/warden/": "v/certifier/",
                                      "v/warden/lean/Warden/": "v/certifier/lean/NetworkCertifier/",
                                      "v/warden/tests/test_warden.py": "v/certifier/tests/test_certifier.py"}
    branch = [*moves, ("v/warden/c.py", "v/certifier/c.py"), ("v/warden/lean/Warden/D.lean", "v/certifier/lean/NetworkCertifier/D.lean")]
    assert M.renamed_paths(moves) == M.renamed_paths(branch)              # a branch's new files under the renamed directories
    swapped = [("S/Properties/Proofs/Nci.lean", "S/Proofs/Properties/Nci.lean"), ("S/Properties/Proofs/Nci/V.lean",
                                                                              "S/Proofs/Properties/Nci/V.lean")]
    assert M.renamed_paths(swapped) == {"S/Properties/Proofs/": "S/Proofs/Properties/"}
    with pytest.raises(SystemExit, match="would move to both"):
        M.renamed_paths([("v/warden/a.py", "v/certifier/a.py"), ("v/warden/b.py", "v/guard/b.py")])


def test_a_directory_is_recorded_only_when_it_moves_whole():
    one = [("v/identity/Identity/Domains.lean", "v/randomness/Randomness/Domains.lean")]
    present = ["v/identity/Identity/Domains.lean", "v/identity/Identity/Tagged.lean", "v/identity/Identity.lean"]
    assert M.renamed_paths(one) == {"v/identity/Identity/": "v/randomness/Randomness/"}
    assert M.renamed_paths(one, present) == dict(one)                    # Tagged.lean stays in Identity/
    moves = [("v/warden/a.py", "v/certifier/a.py"), ("v/warden/tests/test_warden.py", "v/certifier/tests/test_certifier.py")]
    assert M.renamed_paths(moves, [o for o, _ in moves]) == {"v/warden/": "v/certifier/", **dict(moves[1:])}
    excluded = [o for o, _ in moves] + ["v/warden/fixtures/x.json"]           # a file the map excludes stays
    assert M.renamed_paths(moves, excluded) == dict(moves)


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


def toml(data: dict) -> str:
    """A word map as TOML, each table inline: JSON's strings, and its lists and objects of them, are TOML's."""
    def value(v) -> str:
        if isinstance(v, dict):
            return "{" + ", ".join(f"{json.dumps(k, ensure_ascii=False)} = {value(x)}" for k, x in v.items()) + "}"
        if isinstance(v, list):
            return "[" + ", ".join(map(value, v)) + "]"
        return json.dumps(v, ensure_ascii=False)
    return "".join(f"{k} = {value(v)}\n" for k, v in data.items())


def fitted(data: dict, files: list[str]) -> dict:
    """``data`` without the [[words_in]] files, [keep_in] and [keep_paths] globs that match none of ``files``, and without a
    [[words_in]] left with no files: on their tree it renames and keeps just what ``data`` does."""
    def hit(glob: str) -> bool:
        return any(M._match(p, glob) for p in files)
    return {**data, "words_in": [t | {"files": fs} for t in data.get("words_in", []) if (fs := [g for g in t["files"] if hit(g)])],
            **{k: {g: w for g, w in data[k].items() if hit(g)} for k in ("keep_in", "keep_paths") if k in data}}


@pytest.fixture
def toy_map(toy, tmp_path_factory):
    """MAP fitted to the toy: MAP names the files of the tree it moved, of which the toy holds only its lock."""
    path = tmp_path_factory.mktemp("map") / "rename_map.toml"
    path.write_text(toml(fitted(tomllib.loads(MAP.read_text()), git(toy, "ls-files").split())))
    return path


def test_the_move_on_a_toy_tree(toy, toy_map, capsys):
    assert M.main(["--map", str(toy_map), "--root", str(toy)]) == 0
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
        {"paths": {"Sec/Proofs/Warden.lean": "Sec/Proofs/NetworkCertifier.lean", "w/warden/": "w/certifier/",
                   "w/warden/tests/test_network_warden_grid.py": "w/certifier/tests/test_network_certifier_grid.py"}}]
    status = git(toy, "status", "--porcelain", "--untracked-files=all").splitlines()
    assert status and all(line[1] == " " for line in status)                           # everything staged
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "move")
    head = git(toy, "rev-parse", "HEAD:")
    capsys.readouterr()
    assert M.main(["--map", str(toy_map), "--root", str(toy)]) == 0
    assert "files rewritten: 0, files moved: 0, pins repinned: 0" in capsys.readouterr().out
    assert git(toy, "status", "--porcelain") == "" and git(toy, "rev-parse", "HEAD:") == head


def test_the_move_in_many_processes_is_the_move_in_one(toy, toy_map, tmp_path_factory, capsys, monkeypatch):
    """`--jobs` splits the rewriting among forked processes: the staged tree and the report are one process's."""
    monkeypatch.setattr(M, "CHUNK", 2)
    other = tmp_path_factory.mktemp("other")
    git(other, "clone", "-q", str(toy), ".")
    got = {}
    for root, jobs in ((toy, "1"), (other, "4")):
        capsys.readouterr()
        assert M.main(["--map", str(toy_map), "--root", str(root), "--jobs", jobs]) == 0
        got[jobs] = (git(root, "write-tree"), capsys.readouterr().out)
    assert got["1"] == got["4"] and "files rewritten: 0" not in got["1"][1]


def test_the_record_of_the_moves_keeps_their_old_paths_whatever_the_map_excludes(toy, tmp_path_factory):
    """research's `moves.json` records each move's old paths: renamed in it, they'd be lost, and a second run would rename
    the entry the first appended."""
    bare = tmp_path_factory.mktemp("map") / "map.toml"
    bare.write_text('[words]\n"warden" = "certifier"\n')
    assert M.main(["--map", str(bare), "--root", str(toy)]) == 0
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "move")
    assert json.loads((toy / M.MOVES).read_text())["moves"][-1]["paths"]["w/warden/"] == "w/certifier/"
    assert M.main(["--map", str(bare), "--root", str(toy)]) == 0
    assert git(toy, "status", "--porcelain") == ""


def test_the_moves_are_those_of_the_tree_re_recorded(toy, toy_map, capsys):
    """A branch carried across the move keeps its records under the old names, so the map its `--update --moved` takes is
    `--moves` on its own tree: with a name only its lock records, and none once the lock is re-recorded under the new names."""
    path = toy / "Sec/lean-audit.json"
    lock = json.loads(path.read_text())
    path.write_text(json.dumps(lock | {"guarantees": {"Flock.Guarantees.gatewayLeaf": {}}}, indent=1) + "\n")
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "a branch's guarantee")
    assert M.main(["--map", str(toy_map), "--root", str(toy)]) == 0
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "move")
    capsys.readouterr()
    assert M.main(["--map", str(toy_map), "--root", str(toy), "--moves"]) == 0
    assert json.loads(capsys.readouterr().out) == {"Flock.Guarantees.gatewayLeaf": "Flock.Guarantees.firewallLeaf",
                                                   "NetTiming.Warden": "NetTiming.Certifier"}
    path.write_text(json.dumps({"guarantees": {"Flock.Guarantees.firewallLeaf": {}},
                                "reads": {"Definitions.Warden": {"definitions": {"NetTiming.Certifier": "00"}}}}))
    assert M.main(["--map", str(toy_map), "--root", str(toy), "--moves"]) == 0
    assert json.loads(capsys.readouterr().out) == {}


def test_a_file_lands_where_another_leaves(toy):
    """`S/Area.lean` -> `S/Area/Check.lean` while `S/Area/Check.lean` -> `S/Area/Check/Check.lean`: both arrive whole."""
    (toy / "S/Area").mkdir(parents=True)
    (toy / "S/Area.lean").write_text("import S.Area.Check\n")
    (toy / "S/Area/Check.lean").write_text("theorem t : True := trivial\n")
    (toy / "areas.toml").write_text('[words]\n"S/Area" = "S/Area/Check"\n"S.Area" = "S.Area.Check"\n')
    git(toy, "add", "-A")
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "areas")
    assert M.main(["--root", str(toy), "--map", str(toy / "areas.toml")]) == 0
    assert (toy / "S/Area/Check.lean").read_text() == "import S.Area.Check.Check\n"
    assert (toy / "S/Area/Check/Check.lean").read_text() == "theorem t : True := trivial\n"
    assert not (toy / "S/Area.lean").exists()


def test_a_json_pin_follows_a_manifest_the_move_rewrites(toy, toy_map):
    """A JSON pin `pins.py` lists gets a rewritten manifest's new digest, since a move changes where a path package sits,
    which no pinned output reads; any other file it pins and the move rewrites (the README) stays stale, for its command."""
    manifest = '{"packages": [{"type": "path", "dir": "../../w/warden"}]}\n'
    readme = (toy / "README.md").read_text()

    def sha(t: str) -> str:
        return hashlib.sha256(t.encode()).hexdigest()
    pin = {"inputs": {"README.md": sha(readme), "Sec/Proofs/lake-manifest.json": sha(manifest)}, "out": [1]}
    (toy / "tools/verity/check").mkdir(parents=True)
    (toy / "tools/verity/check/pins.py").write_text(
        "from __future__ import annotations\n\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Pin:\n"
        "    path: str\n    key: str | None\n"
        '    under: str\n\n\nPINS = (Pin("fix/vec.json", "inputs", ""), Pin("pous/TRUSTED.sha256", None, "pous"))\n')
    (toy / "Sec/Proofs/lake-manifest.json").write_text(manifest)
    (toy / "fix").mkdir()
    (toy / "fix/vec.json").write_text(json.dumps(pin, indent=2) + "\n")
    git(toy, "add", "-A")
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "a pin")
    assert M.main(["--map", str(toy_map), "--root", str(toy)]) == 0
    moved = (toy / "Sec/Proofs/lake-manifest.json").read_text()
    assert moved == manifest.replace("w/warden", "w/certifier")
    assert json.loads((toy / "fix/vec.json").read_text()) == {
        "inputs": {"README.md": sha(readme), "Sec/Proofs/lake-manifest.json": sha(moved)}, "out": [1]}


def test_a_sorted_json_object_gets_a_renamed_key_in_its_sorted_place(toy, tmp_path_factory, capsys):
    """An object whose keys were sorted comes back sorted, as its writer (`unread.py`'s caps) writes it, its lines otherwise
    as they were; one that wasn't sorted keeps its order, and a file on one line or with a key twice keeps the words alone.
    A second run changes nothing."""
    def caps(areas: list[str]) -> str:
        return json.dumps({"schema": "caps/v1", "areas": {a: [f"{a}.x", f"{a}.y"] for a in areas}}, indent=1) + "\n"
    files = {"Sec/caps.json": caps(["Proofs/Core", "Proofs/Population", "Proofs/Rest"]),
             "Sec/order.json": json.dumps({"Zeta": 1, "Population": 2, "Core": 3}, indent=1) + "\n",
             "Sec/line.json": json.dumps({"Core": 1, "Population": 2, "Rest": 3}) + "\n",
             "Sec/twice.json": '{\n "Core": 1,\n "Core": 2,\n "Population": 3,\n "Rest": 4\n}\n'}
    for p, t in files.items():
        (toy / p).write_text(t)
    git(toy, "add", "-A")
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "json")
    words = tmp_path_factory.mktemp("map") / "map.toml"
    words.write_text('[words]\n"Population" = "SampledProofs"\n')
    assert M.main(["--map", str(words), "--root", str(toy)]) == 0
    assert (toy / "Sec/caps.json").read_text() == caps(["Proofs/Core", "Proofs/Rest", "Proofs/SampledProofs"])
    for p in ("Sec/order.json", "Sec/line.json", "Sec/twice.json"):
        assert (toy / p).read_text() == files[p].replace("Population", "SampledProofs"), p
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "move")
    capsys.readouterr()
    assert M.main(["--map", str(words), "--root", str(toy)]) == 0
    assert "files rewritten: 0" in capsys.readouterr().out


@pytest.mark.parametrize("path", ["Sec/lean-audit.json", "Sec/Proofs/lake-manifest.json", M.MOVES, "Sec/caps.toml"])
def test_a_lock_a_manifest_moves_json_or_a_file_not_json_keeps_the_words_alone(path):
    old = json.dumps({"Core": 1, "Population": 2, "Rest": 3}, indent=1) + "\n"
    new = old.replace("Population", "SampledProofs")
    assert M.resorted(old, new, path) == new
    assert M.resorted(old, new, "Sec/caps.json") == json.dumps({"Core": 1, "Rest": 3, "SampledProofs": 2}, indent=1) + "\n"


def test_it_refuses_a_dirty_worktree(toy, toy_map):
    (toy / "README.md").write_text("edited\n")
    with pytest.raises(SystemExit, match="isn't clean"):
        M.main(["--map", str(toy_map), "--root", str(toy)])


def test_a_text_file_over_the_limit_stops_the_move_and_a_lock_over_it_is_renamed(toy, toy_map, monkeypatch):
    readme, lock = (len((toy / p).read_bytes()) for p in ("README.md", "Sec/lean-audit.json"))
    monkeypatch.setattr(M.L, "TEXT_LIMIT", readme - 1)
    with pytest.raises(SystemExit, match=rf"\n  README\.md \({readme} bytes\)\n"):
        M.main(["--map", str(toy_map), "--root", str(toy)])
    assert git(toy, "status", "--porcelain") == ""
    monkeypatch.setattr(M.L, "TEXT_LIMIT", lock - 1)
    assert M.main(["--map", str(toy_map), "--root", str(toy)]) == 0
    assert json.loads((toy / "Sec/lean-audit.json").read_text())["roots"] == ["Proofs.NetworkCertifier"]


def test_two_files_landing_at_one_path_fail(toy, toy_map):
    (toy / "w/certifier").mkdir(parents=True)
    (toy / "w/certifier/grid.py").write_text("")
    git(toy, "add", "-A")
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "clash")
    with pytest.raises(SystemExit, match="two files would land at one path"):
        M.main(["--map", str(toy_map), "--root", str(toy)])


STALE = ('[words]\n"warden" = "certifier"\n\n[[words_in]]\nfiles = ["w/warden/grid.py", "w/warden/old_grid.py", "w/**/*.lean"]\n'
         'words = { "Warden" = "NetworkCertifier" }\n\n[keep_in]\n"README.md" = ["warden"]\n"docs/**" = ["warden"]\n\n'
         '[keep_paths]\n"w/gone/**" = ["warden"]\n\n[rewrite]\nexclude = ["archive/**", "nowhere/**"]\n')


def test_a_map_naming_a_path_no_file_matches_is_refused(toy, tmp_path_factory, capsys):
    """A [[words_in]] file, a [keep_in] or a [keep_paths] glob that no tracked file matches (a file since moved or deleted)
    holds renames or keeps the move would skip: every mode refuses the map, naming the map and each such path, and changes
    nothing. An exclusion may name what the tree lacks."""
    stale = tmp_path_factory.mktemp("map") / "stale.toml"
    stale.write_text(STALE)
    head = git(toy, "rev-parse", "HEAD:")
    for mode in ([], ["--dry-run"], ["--moves"]):
        with pytest.raises(SystemExit) as e:
            M.main(["--map", str(stale), "--root", str(toy), *mode])
        assert str(e.value) == (f"{stale}: it names paths no tracked file matches, whose renames or keeps the move would skip; "
                                "point each at its file's path on this tree, or drop it:\n"
                                "  [[words_in]] files: w/warden/old_grid.py\n  [[words_in]] files: w/**/*.lean\n"
                                "  [keep_in]: docs/**\n  [keep_paths]: w/gone/**")
    assert capsys.readouterr().out == ""
    assert git(toy, "status", "--porcelain") == "" and git(toy, "rev-parse", "HEAD:") == head
    with pytest.raises(SystemExit, match=r"\n  \[\[words_in\]\] files: verity/Security/Proofs/Warden\.lean\n"):
        M.main(["--map", str(MAP), "--root", str(toy)])               # the past map names the files of the tree it moved


def test_a_map_whose_paths_all_match_runs_as_before(toy, tmp_path_factory, capsys, monkeypatch):
    """The map without what matches nothing is refused nothing and moves the toy just as the whole map did before
    `rename.py` refused it: the entries left rename and keep as they did, and those dropped held nothing on this tree."""
    tmp = tmp_path_factory.mktemp("map")
    (tmp / "stale.toml").write_text(STALE)
    files = git(toy, "ls-files").split()
    (tmp / "fitted.toml").write_text(toml(fitted(tomllib.loads(STALE), files)))
    assert M.Words(tomllib.loads((tmp / "fitted.toml").read_text())).unmatched(files) == []
    other = tmp_path_factory.mktemp("other")
    git(other, "clone", "-q", str(toy), ".")
    capsys.readouterr()
    assert M.main(["--map", str(tmp / "fitted.toml"), "--root", str(toy)]) == 0
    now = git(toy, "write-tree"), capsys.readouterr().out
    monkeypatch.setattr(M.Words, "unmatched", lambda self, files: [])                  # rename.py as it was
    assert M.main(["--map", str(tmp / "stale.toml"), "--root", str(other)]) == 0
    assert (git(other, "write-tree"), capsys.readouterr().out) == now
    assert (toy / "w/certifier/grid.py").read_text().startswith("class NetworkCertifier:\n")      # [[words_in]] renamed
    assert (toy / "README.md").read_text() == "The prover gateway; `w/warden/grid.py`.\n"          # [keep_in] kept


PROSE = M.Words({"scope": "prose", "words": {"draw": "sample", "firewall": "ZK server", "firewalled": "zk_served",
                                             "escape": "acceptance probability", "DrawnUnit": "SampledBlock"}})


@pytest.mark.parametrize("path, old, new", [
    ("a.py",
     '"""Each draw is fresh; the firewall serves it."""\nDRAW = b"draw"\n\n\ndef draw(n):\n'
     '    # a draw of n coins, as `draw` gives; unit_draw, .draw, draw.py, --draw, draw(\n    return f"draw {n}" + draw_one(n)\n',
     '"""Each sample is fresh; the ZK server serves it."""\nDRAW = b"draw"\n\n\ndef draw(n):\n'
     '    # a sample of n coins, as `draw` gives; unit_draw, .draw, draw.py, --draw, draw(\n    return f"draw {n}" + draw_one(n)\n'),
    ("L.lean",
     '/-- A draw of the firewall. -/\ndef draw : Nat := 1 -- the draw\n#eval s!"draw {draw}"\n/- nested /- draw -/ escape -/\n',
     '/-- A sample of the ZK server. -/\ndef draw : Nat := 1 -- the sample\n#eval s!"draw {draw}"\n'
     '/- nested /- sample -/ acceptance probability -/\n'),
    ("n.md",
     "A draw (`draw()`), [draw](draw.md) and https://x/draw; a firewalled escape; a DrawnUnit.\n\n```\ndraw\n```\n",
     "A sample (`draw()`), [sample](draw.md) and https://x/draw; a firewalled acceptance probability; a DrawnUnit.\n\n"
     "```\ndraw\n```\n"),
    ("s.sh",
     "#!/bin/bash\n# a draw\necho \"draw\" 'draw'\ndraw() { :; }\ncat <<X\ndraw\nX\n",
     "#!/bin/bash\n# a sample\necho \"draw\" 'draw'\ndraw() { :; }\ncat <<X\ndraw\nX\n"),
    ("k.cu",
     '// a draw\nconst char *s = "draw"; /* the draw */ int draw = 1;\n',
     '// a sample\nconst char *s = "draw"; /* the sample */ int draw = 1;\n'),
    ("x.json", '{"draw": "a draw"}\n', '{"draw": "a draw"}\n'),
])
def test_a_prose_map_renames_only_words_in_prose(path, old, new):
    """Under `scope = "prose"` a word is renamed in a comment, a docstring or Markdown's text, and nowhere else: not in code,
    a string, backticks, a link target, a fence, a heredoc or a data file, nor as part of a name, a path, a flag, a call or a
    constant; and a word whose new word is a name (`zk_served`) stays."""
    assert PROSE.sub(old, path) == new


def test_a_data_file_under_a_prose_map_is_left_as_it_is():
    """A file with no prose (a data file, a lock) keeps its bytes and its words go uncounted, while a file with prose counts
    a word its literal keeps as left standing."""
    words = M.Words({"scope": "prose", "words": {"draw": "sample", "firewall": "ZK server"},
                     "keep": {"patterns": [r'"draw"']}})
    text = '{"draw": "a draw by the firewall", "unit_draw": 1}\n'
    kept, counts = M.Counter(), M.Counter()
    assert words.sub(text, "x.json", counts=counts, kept=kept) == text
    assert not kept and not counts
    assert words.sub("# a draw\nx = 'a draw by the firewall'\n", "y.py", kept=kept) == "# a sample\nx = 'a draw by the firewall'\n"
    assert kept == {"draw": 1, "firewall": 1}


def test_a_prose_move_moves_nothing_and_leaves_what_a_json_pin_records(toy, capsys):
    """A prose move renames no path and no declaration, and leaves a file a JSON pin records for the pin's command."""
    (toy / "tools/verity/check").mkdir(parents=True)
    (toy / "tools/verity/check/pins.py").write_text(
        "from __future__ import annotations\n\nfrom dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Pin:\n"
        '    path: str\n    key: str | None\n    under: str\n\n\nPINS = (Pin("fix/vec.json", "inputs", ""),)\n')
    (toy / ".gitignore").write_text("__pycache__/\n")
    (toy / "fix").mkdir()
    (toy / "fix/vec.json").write_text(json.dumps({"inputs": {"pinned.md": "00"}}) + "\n")
    (toy / "pinned.md").write_text("The prover gateway.\n")
    (toy / "prose.toml").write_text('scope = "prose"\n\n[words]\n"warden" = "certifier"\n"prover gateway" = "firewall"\n')
    git(toy, "add", "-A")
    git(toy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "a pin")
    lock = (toy / "Sec/lean-audit.json").read_text()
    assert M.main(["--map", str(toy / "prose.toml"), "--root", str(toy), "--moves"]) == 0
    assert json.loads(capsys.readouterr().out) == {}
    assert M.main(["--map", str(toy / "prose.toml"), "--root", str(toy)]) == 0
    assert git(toy, "diff", "--cached", "--name-status").split() == ["M", "CHANGELOG.md", "M", "README.md"]
    assert (toy / "README.md").read_text() == "The firewall; `w/warden/grid.py`.\n"
    assert (toy / "CHANGELOG.md").read_text() == "- the certifier landed\n"
    assert (toy / "pinned.md").read_text() == "The prover gateway.\n"
    assert (toy / "Sec/lean-audit.json").read_text() == lock
