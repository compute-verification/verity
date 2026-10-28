"""The label vocabulary (`research.store.vocab`, `research data vocab`, the `research data label` warning), `research data
refresh-fixtures` and `research data doctor` -- the three additions the store grew after a day of real use.  Every test runs against
its own store root (conftest), its own repo and its own runs directory; nothing here reads ~/.research."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.store import fixtures as F
from research.store import vocab
from research.store.cli import main
from research.store.local import LocalStore

ART = "art:" + "a" * 64
RUN_STALE, RUN_REAL, RUN_FINE = "r20260922-105905-2021", "r20260922-110000-beef", "r20260922-110100-cafe"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A tmp store root, an absent remote config, a tmp repo with an empty fixtures/ directory, a tmp runs root."""
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    monkeypatch.setenv("RESEARCH_REPO", str(tmp_path / "repo"))
    monkeypatch.setenv("RESEARCH_RUNS", str(tmp_path / "runs"))
    (tmp_path / "repo" / "fixtures").mkdir(parents=True)
    (tmp_path / "runs").mkdir()
    return tmp_path


@pytest.fixture
def fs_remote(env):
    (env / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{env / "vault"}"\n')
    return env / "vault"


# ---- vocabulary -----------------------------------------------------------------------------------------------------------------------

RESULT_KEYS = ("candidate", "proof_class", "soundness", "K", "B", "hardware", "authentication", "campaign", "relation", "mode", "zk", "overhead",
               "seconds_per_vu", "track", "scope", "label", "note", "assumptions", "omitted", "domain", "breakthrough",
               "lane", "question", "cited-by", "row", "sweep", "arm", "stage", "source", "superseded_by", "of-record", "goal", "abandoned", "replay_set", "replay_expect", "interaction_attempt_of", "hash", "sharing", "fold",
               "pipeline", "kernel_outcome", "kernel_reason", "memory-line-of-record", "run_of_record")
VERIFICATION_KEYS = ("verified", "verifier", "verifier_seconds", "independently_verified", "verified_by", "finding", "same_device", "below_bar", "outcome",
                     "grant", "ready", "reviewed", "lean_review", "landed", "commit_map", "custody", "source-lost", "challenger-key", "challenger-ledger",
                     "challenger-pins", "challenger-models")
EVIDENCE_KEYS = ("instruction", "model", "model_agrees", "model_status", "model_rejected", "evidence", "family", "instances_dataset", "instances_tier",
                 "mismatches", "target", "device", "instances_manifest_sha256", "component", "jobs", "level", "anchor", "dtype", "staged",
                 "circuits")
APPROACH_KEYS = ("approach", "title", "hypothesis", "approach_type", "attacks", "owner", "approach_status", "reason", "cites", "uncites")


def test_vocabulary_pins_every_key_by_asserter_and_form():
    groups = {k.name: k.group for k in vocab.VOCAB.values()}
    assert {n for n, g in groups.items() if g == vocab.GROUP_RESULT} == set(RESULT_KEYS)
    assert {n for n, g in groups.items() if g == vocab.GROUP_VERIFICATION} == set(VERIFICATION_KEYS)
    assert {n for n, g in groups.items() if g == vocab.GROUP_EVIDENCE} == set(EVIDENCE_KEYS)
    assert {n for n, g in groups.items() if g == vocab.GROUP_APPROACH} == set(APPROACH_KEYS)
    assert vocab.KEYS == tuple(vocab.VOCAB) and len(vocab.KEYS) == len(RESULT_KEYS) + len(VERIFICATION_KEYS) + len(EVIDENCE_KEYS) + len(APPROACH_KEYS)
    assert vocab.key("approach_status").values == ("live", "parked", "killed", "superseded", "merged")
    assert vocab.key("approach_type").values == ("scheme", "route", "attack")
    # the constants are the strings (a consumer imports vocab.PROOF_CLASS, never spells it)
    assert (vocab.CANDIDATE, vocab.PROOF_CLASS, vocab.VERIFIED, vocab.MODEL_AGREES, vocab.INDEPENDENTLY_VERIFIED, vocab.VERIFIED_BY) == \
        ("candidate", "proof_class", "verified", "model_agrees", "independently_verified", "verified_by")
    # the closed sets asked for
    assert vocab.key("verified").values == ("accepted", "rejected", "not-transferable", "no-dumps") == vocab.VERIFIED_VALUES
    assert vocab.key("model_agrees").form == vocab.BOOL and vocab.key("model_agrees").values == ("true", "false")
    assert vocab.key("independently_verified").form == vocab.BOOL and vocab.key("independently_verified").status == vocab.DEPRECATED
    assert vocab.key("verified_by").status == vocab.IGNORED
    assert vocab.key("proof_class").values == vocab.PROOF_CLASSES and "NON_ZK_PROOF" in vocab.PROOF_CLASSES
    assert vocab.key("authentication").values == ("included", "included-hash", "included-hash-shared", "excluded")
    assert vocab.key("mode").values == ("interactive", "fiat-shamir")
    for k in vocab.VOCAB.values():
        assert k.form in (vocab.ENUM, vocab.BOOL, vocab.INT, vocab.NUMBER, vocab.TEXT, vocab.LIST) and k.doc
        assert bool(k.values) == (k.form in (vocab.ENUM, vocab.BOOL)), k.name
        assert k.status in (vocab.CURRENT, vocab.DEPRECATED, vocab.IGNORED)


def test_check_is_enum_and_bool_only_and_reads_values_as_the_catalog_does():
    assert vocab.check("verified", "accepted") is None
    assert "not one of accepted, rejected, not-transferable, no-dumps" in vocab.check("verified", "yes")
    assert vocab.check("model_agrees", True) is None and vocab.check("model_agrees", False) is None
    assert "is a string; the key's form is bool" in vocab.check("model_agrees", "true")                    # the word is not the value (hygiene 2026-09-23)
    assert "is a string; the key's form is bool" in vocab.check("model_agrees", "True")                    # a Python repr: the same complaint
    assert vocab.check("model_agrees", "yes") and vocab.check("zk", 1)
    assert vocab.check("proof_class", "non-zk") and vocab.check("proof_class", "NON_ZK_PROOF") is None
    assert vocab.check("candidate", "A-GKR") is None and "old spelling of 'A-GKR'" in vocab.check("candidate", "A")
    for value in ("C-Flock", "D-SP1", "SP1", "C"):                                   # SP1 and C: the ledger import's, recorded, still valid
        assert vocab.check("candidate", value) is None, value
    assert "is not one of" in vocab.check("candidate", "Flock")
    # free forms are documentation, never checked; int / number forms only object to strings
    for key, value in (("soundness", "2^-128 (heuristic)"), ("K", 1536), ("overhead", 1.5), ("assumptions", ["ROM"]), ("note", "")):
        assert vocab.check(key, value) is None, (key, value)
    assert "is a string; the key's form is int" in vocab.check("K", "1536")
    assert "old spelling of 'verifier'" in vocab.check("checker", "x") and vocab.is_known("checker") is False and vocab.key("checker") is None
    assert "is a ref" in vocab.check("result_artifact", "art:" + "0" * 64) and "not in the vocabulary" in vocab.check("nonesuch", 1)
    assert vocab.is_known("verified")


def test_canonical_spellings_and_coercion():
    assert vocab.canonical_key("notes") == vocab.canonical_key("verify_note") == "note" and vocab.canonical_key("batch") == "B"
    assert vocab.canonical_key("checker") == "verifier" and vocab.canonical_key("verifier") == "verifier"
    assert vocab.canonical_value("candidate", "A") == "A-GKR" and vocab.canonical_value("candidate", "SP1") == "SP1" and vocab.canonical_value("note", "A") == "A"
    assert vocab.coerce("zk", "true") is True and vocab.coerce("zk", "False") is False and vocce("model_agrees", " true ") is True
    assert vocab.coerce("independently_verified", "true") is True and vocab.coerce("zk", "yes") == "yes" and vocab.coerce("zk", True) is True
    assert vocab.coerce("B", "4096") == 4096 and vocab.coerce("batch", "64") == 64 and vocab.coerce("B", "4096.0") == "4096.0" and vocab.coerce("K", 1536) == 1536
    assert vocab.coerce("overhead", "1.5") == 1.5 and vocab.coerce("overhead", "2") == 2 and vocab.coerce("overhead", "1e3") == 1000.0 and vocab.coerce("overhead", "n/a") == "n/a"
    assert vocab.coerce("soundness", "128") == "128" and vocab.coerce("nonesuch", "true") == "true", "free-text and unknown keys are never coerced"
    assert vocab.BACKEND_FAMILIES == ("A-GKR", "B-Ligero", "C-Flock", "D-SP1")
    assert set(vocab.CANDIDATES) == {*vocab.BACKEND_FAMILIES, "B-AIR", "C", "SP1", "shared"} and "A" not in vocab.CANDIDATES
    assert set(vocab.REF_KEYS) == {"path", "result_artifact", "constructed_from_attempt"} and not (set(vocab.REF_KEYS) & set(vocab.KEYS))
    assert vocab.is_known("finding") and vocab.is_known("level") and vocab.is_known("anchor")


def vocce(*a):  # noqa: N802  (a typo-guard: the one call site above means vocab.coerce)
    return vocab.coerce(*a)


def test_render_and_cli_vocab(env, capsys):
    text = vocab.render()
    for name in (*RESULT_KEYS, *VERIFICATION_KEYS, *EVIDENCE_KEYS):
        assert f"\n  {name:<24} " in text or text.startswith(f"  {name:<24} "), name
    assert "accepted | rejected | not-transferable | no-dumps" in text and "[deprecated]" in text and "[ignored]" in text
    assert [g for g in vocab.by_group()] == list(vocab.GROUPS) and sum(len(v) for v in vocab.by_group().values()) == len(vocab.KEYS)
    assert main(["vocab"]) == 0
    assert capsys.readouterr().out.strip() == text
    assert main(["vocab", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == set(vocab.KEYS)
    assert doc["verified"] == {"group": "verification", "form": "enum", "values": list(vocab.VERIFIED_VALUES), "status": "current",
                               "doc": vocab.key("verified").doc}
    assert doc["model_agrees"]["values"] == ["true", "false"] and doc["independently_verified"]["status"] == "deprecated"


def test_label_refuses_what_the_vocabulary_cannot_place_unless_off_vocab(env, capsys):
    """Lanes standardise on the vocabulary's words: an unknown key, a measurement, a ref, an off-enum value or a word in a bool key is
    refused (exit 2, nothing written) with the reason; `--off-vocab` writes it with a warning."""
    st = LocalStore()
    refused = [(["nonesuch", "v1"], "label key 'nonesuch' is not in the vocabulary"),
               (["Relation", "fp8-ada"], "did you mean 'relation'"),
               (["verfied", "accepted"], "did you mean 'verified' or 'verifier'?"),
               (["subbatches", "32"], "label key 'subbatches' is a measurement or run parameter"),
               (["result_artifact", "art:" + "1" * 64], "names another store object: that is a ref"),
               (["model_agrees", "maybe"], "model_agrees='maybe' is not one of true, false"),
               (["verified", "yes"], "not one of accepted, rejected, not-transferable, no-dumps"),
               (["fold", "four"], "fold='four' is not an int"),
               (["overhead", "n/a"], "overhead='n/a' is not a number"),
               (["replay_set", "warden"], "replay_set='warden' does not match"),
               (["replay_set", "warden/rerun"], "replay_set='warden/rerun' does not match"),
               (["replay_expect", "blocked"], "replay_expect='blocked' does not match")]
    for args, why in refused:
        assert main(["label", ART, *args, "--by", "lane-x"]) == 2, args
        out, err = capsys.readouterr()
        assert out == "" and err.startswith("research data label: refused: ") and why in err, (args, err)
        assert "Nothing written (--off-vocab writes it anyway)" in err
    assert st.labels(ART) == []
    assert main(["label", ART, "nonesuch", "v1", "--by", "lane-x", "--off-vocab"]) == 0
    out, err = capsys.readouterr()
    assert Path(out.strip()).is_file() and err.count("\n") == 2 and "local only (write-through off)" in err
    assert "warning: label key 'nonesuch' is not in the vocabulary" in err and "stored anyway (--off-vocab)" in err
    assert main(["label", ART, "model_agrees", "true", "--by", "lane-x", "--value-json"]) == 0
    out, err = capsys.readouterr()
    assert Path(out.strip()).is_file() and err.startswith("research data label: local only (write-through off)") and err.count("\n") == 1
    assert main(["label", ART, "verified", "accepted", "--by", "verifier-1", "--ref", "run:x"]) == 0
    out, err = capsys.readouterr()
    assert Path(out.strip()).is_file() and err.startswith("research data label: local only (write-through off)") and err.count("\n") == 1
    assert {(l.key, l.value) for l in st.labels(ART)} == {("nonesuch", "v1"), ("model_agrees", True), ("verified", "accepted")}
    for key, value in (("replay_set", "warden/re-verify"), ("replay_set", "pouw/re-run"), ("replay_expect", "blocked: V3")):
        assert main(["label", ART, key, value, "--by", "lane-x", "--ref", "run:x"]) == 0, (key, value)
    capsys.readouterr()


def test_lane_shorthand_folds_into_one_spelling(env, capsys):
    """The 2026-09-22/23 lane shorthand: synonyms are written under one key, the adopted concepts are keys of their own."""
    st = LocalStore()
    for key, value, want in (("axis", "l", ("sweep", "l")), ("milestone", "1M-vu", ("arm", "1M-vu")), ("tree", "main@6babe27", ("source", "main@6babe27")),
                             ("release", "lane/x@ec45c0b", ("source", "lane/x@ec45c0b")), ("verifier_note", "Rust", ("note", "Rust")),
                             ("same_device", "false", ("same_device", False)), ("below_bar", "true", ("below_bar", True)), ("fold", "4", ("fold", 4)), ("pipeline", "2", ("pipeline", 2)),
                             ("hash", "poseidon2-babybear-w24", ("hash", "poseidon2-babybear-w24")), ("lane", "b-verify-par", ("lane", "b-verify-par")),
                             ("superseded_by", "r20260922-231034-0693", ("superseded_by", "r20260922-231034-0693"))):
        assert main(["label", ART, key, value, "--by", "lane-x"]) == 0, key
        assert "refused" not in capsys.readouterr().err
        assert (st.labels(ART, key=want[0])[-1].key, st.labels(ART, key=want[0])[-1].value) == want
    assert vocab.suggest("supersededby") == ["superseded_by"] and vocab.suggest("Axis") == ["sweep"] and vocab.suggest("zzzzzz") == []
    assert not set(vocab.MEASUREMENT_KEYS) & set(vocab.KEYS) and not set(vocab.MEASUREMENT_KEYS) & set(vocab.KEY_ALIASES)
    assert set(vocab.KEY_ALIASES.values()) <= set(vocab.KEYS)
    for (key, old), new in vocab.VALUE_ALIASES.items():
        assert vocab.check(key, new) is None and vocab.check(key, old), (key, old, new)


def test_abandoned_is_written_only_with_the_note_that_says_why(env, capsys):
    """`abandoned` marks evidence of a line of work that was given up; the label is only read with its note, so `--ref` must name one
    as note:<lane>/<file name without .md> or the notes README's note:<campaign>/<lane>/<name> (both survive a move into a month folder)."""
    st, note = LocalStore(), "note:flock-backend/20260929T0500Z-report-flock-backend"
    why = ("label key 'abandoned' needs --ref naming the note that says why (note:<lane>/<file name without .md>, "
           "or note:<campaign>/<lane>/<name>)")
    assert vocab.check_ref("abandoned", None) == why and vocab.check_ref("abandoned", note) is None
    for ok in ("note:kb/LANE-CONTRACT", "note:r20-proof/red-team-vu/20260922T1042Z-report-redteam-vu"):
        assert vocab.check_ref("abandoned", ok) is None, ok
    assert vocab.check_ref("abandoned", RUN_REAL) == f"{why}, not {RUN_REAL!r}"
    for path in ("note:lanes/flock-backend/2026-09/x.md", "note:flock-backend/x.md", "note:flock-backend/2026-09/x",
                 "note:flock-backend/28/x", "note:20260929T0500Z-report-flock-backend"):   # paths, and a name without its lane
        assert vocab.check_ref("abandoned", path) is not None, path
    assert vocab.check_ref("superseded_by", None) is None and vocab.check_ref("note", "anything") is None
    for ref in ([], ["--ref", RUN_REAL]):
        assert main(["label", RUN_REAL, "abandoned", "W=31 field, not the protocol's", "--by", "flock-backend", *ref]) == 2
        err = capsys.readouterr().err
        assert err.startswith(f"research data label: refused: {why}") and "Nothing written" in err
    assert st.labels(RUN_REAL) == []
    assert main(["label", RUN_REAL, "abandoned", "W=31 field, not the protocol's", "--by", "flock-backend", "--ref", note]) == 0
    assert "refused" not in capsys.readouterr().err
    (lab,) = st.labels(RUN_REAL)
    assert (lab.key, lab.value, lab.by, lab.ref) == ("abandoned", "W=31 field, not the protocol's", "flock-backend", note)


def _auditor_run(st: LocalStore, tmp: Path, run_id: str, line: dict | None) -> None:
    """Run `run_id`'s Attempt, its run record keeping `line` as out/auditor_line.json (no file with None)."""
    tree = tmp / run_id
    (tree / "out").mkdir(parents=True)
    (tree / "out" / ("auditor_line.json" if line is not None else "credit.md")).write_text(json.dumps(line))
    art = st.put_artifact("run-record/v1", {"run_id": run_id}, tree=tree)
    st.put_attempt({
        "schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": "run", "version": "1"}, "derivation": "drv:" + "0" * 64,
        "reuse_key": None, "argv": ["bash", "auditor.sh"], "params": {}, "key_params": [], "inputs": {}, "conditions": {},
        "key_conditions": [], "source": {"commit": "a" * 40, "dirty": False}, "closure": None, "execution": {}, "cost": None,
        "status": {"state": "done", "rc": 0}, "outputs": {"run_record": art}, "telemetry": None})


def test_a_memory_line_of_record_is_the_leans_and_names_the_auditor_run_that_wrote_it(env, capsys):
    """`memory-line-of-record` on a gate run's verifier run says its line of record is the auditor_line.json of the run its ref
    names, computed by the Lean (that file's `line_by`) and conditional on what that file's `conditional_on` names (RD-21): the
    label reads `lean conditional on NAMES`, and `research data label` reads the file from the ref run's record and refuses a value
    that isn't the file's, a file with no conditional_on naming what statement C rests on or a line.py line, a run whose record
    keeps no file, and a ref that isn't a run id."""
    st = LocalStore()
    why = ("label key 'memory-line-of-record' needs --ref naming the auditor_line.py run whose auditor_line.json is the line (its run "
           "id, r<YYYYMMDD>-<HHMMSS>-<hex>)")
    value = "lean conditional on NoOutsideCompute"
    assert vocab.check("memory-line-of-record", value) is None
    assert vocab.check("memory-line-of-record", "lean conditional on NoOutsideCompute, ProviderLockdown") is None
    for old in ("lean", "python", "credited-pinned-bound", "lean conditional on"):
        assert vocab.check("memory-line-of-record", old).startswith(f"memory-line-of-record={old!r} does not match")
    assert vocab.check_ref("memory-line-of-record", RUN_FINE) is None and vocab.check_ref("memory-line-of-record", None) == why
    for ref in ([], ["--ref", "art:" + "0" * 64], ["--ref", "note:memory-accounting/20261024T0832Z-report-line-lean-record"]):
        assert main(["label", RUN_REAL, "memory-line-of-record", value, "--by", "memory-accounting", *ref]) == 2
        assert capsys.readouterr().err.startswith(f"research data label: refused: {why}")
    runs = {"r20261025-010000-0001": {"line_by": "lean", "conditional_on": {"NoOutsideCompute": "C"}},
            "r20261025-010000-0002": {"line_by": "lean", "conditional_on": {"ProviderLockdown": "L"}},
            "r20261025-010000-0003": {"line_by": "python", "conditional_on": {"NoOutsideCompute": "C"}},
            "r20261025-010000-0004": {"line_by": "lean"},
            "r20261025-010000-0005": {"line_by": "lean", "conditional_on": {"Identity": "I"}},
            "r20261025-010000-0006": None}
    for run_id, line in runs.items():
        _auditor_run(st, env, run_id, line)
    assert main(["label", RUN_REAL, "memory-line-of-record", "python", "--by", "memory-accounting", "--ref", RUN_FINE]) == 2
    assert "does not match" in capsys.readouterr().err
    for run_id, label, err in (
            (RUN_FINE, value, "its ref run's record keeps no out/auditor_line.json"),
            ("r20261025-010000-0006", value, "its ref run's record keeps no out/auditor_line.json"),
            ("r20261025-010000-0001", "lean conditional on ProviderLockdown", "isn't its ref run's line, which is "
             "'lean conditional on NoOutsideCompute'"),
            ("r20261025-010000-0002", value, "isn't its ref run's line, which is 'lean conditional on ProviderLockdown'"),
            ("r20261025-010000-0003", value, "its ref run's line is line_by 'python'"),
            ("r20261025-010000-0004", value, "names no conditional_on with NoOutsideCompute or ProviderLockdown"),
            ("r20261025-010000-0005", "lean conditional on Identity", "names no conditional_on with NoOutsideCompute")):
        assert main(["label", RUN_REAL, "memory-line-of-record", label, "--by", "memory-accounting", "--ref", run_id]) == 2
        assert err in capsys.readouterr().err, (run_id, label)
    assert st.labels(RUN_REAL) == []
    for run_id, label in (("r20261025-010000-0001", value), ("r20261025-010000-0002", "lean conditional on ProviderLockdown")):
        assert main(["label", RUN_REAL, "memory-line-of-record", label, "--by", "memory-accounting", "--ref", run_id]) == 0
    assert {(lab.key, lab.value, lab.ref) for lab in st.labels(RUN_REAL)} == {
        ("memory-line-of-record", value, "r20261025-010000-0001"),
        ("memory-line-of-record", "lean conditional on ProviderLockdown", "r20261025-010000-0002")}


def test_label_writes_the_canonical_key_and_form(env, capsys):
    """Hygiene (2026-09-23): the file holds the vocabulary's spelling and form -- a real boolean for zk / model_agrees /
    independently_verified, an int for B, `note` for `notes` / `verify_note`, `verifier` for `checker`, `A-GKR` for candidate `A` --
    whatever the caller typed; stderr says what was rewritten.  A ref written as a label is refused unless --off-vocab."""
    st = LocalStore()
    assert main(["label", ART, "zk", "true", "--by", "lane-x"]) == 0
    out, err = capsys.readouterr()
    assert "value 'true' written as true" in err and "warning" not in err
    assert main(["label", ART, "model_agrees", "False", "--by", "lane-x"]) == 0
    assert "value 'False' written as false" in capsys.readouterr().err
    assert main(["label", ART, "independently_verified", "true", "--by", "verifier-1"]) == 0
    capsys.readouterr()
    assert main(["label", ART, "batch", "4096", "--by", "lane-x"]) == 0
    err = capsys.readouterr().err
    assert "key 'batch' written as 'B'" in err and "value '4096' written as 4096" in err and "warning" not in err
    assert main(["label", ART, "verify_note", "Rust 12T", "--by", "verify"]) == 0
    assert "key 'verify_note' written as 'note'" in capsys.readouterr().err
    assert main(["label", ART, "checker", "v2", "--by", "lane-x"]) == 0
    assert "key 'checker' written as 'verifier'" in capsys.readouterr().err
    assert main(["label", ART, "candidate", "A", "--by", "a-lane"]) == 0
    err = capsys.readouterr().err
    assert "value 'A' written as \"A-GKR\"" in err and "warning" not in err
    assert main(["label", ART, "result_artifact", "art:" + "1" * 64, "--by", "lane-x"]) == 2
    assert main(["label", ART, "result_artifact", "art:" + "1" * 64, "--by", "lane-x", "--off-vocab"]) == 0
    assert "warning: label key 'result_artifact' names another store object: that is a ref" in capsys.readouterr().err
    assert main(["label", ART, "overhead", "12.5", "--by", "lane-x"]) == 0 and main(["label", ART, "soundness", "128", "--by", "lane-x"]) == 0
    got = {(l.key, l.value) for l in st.labels(ART)}
    assert got == {("zk", True), ("model_agrees", False), ("independently_verified", True), ("B", 4096), ("note", "Rust 12T"), ("verifier", "v2"),
                   ("candidate", "A-GKR"), ("result_artifact", "art:" + "1" * 64), ("overhead", 12.5), ("soundness", "128")}
    raw = [json.loads(p.read_bytes()) for p in sorted(st.labels_dir(ART).glob("*.json"))]
    assert {r["key"]: r["value"] for r in raw}["zk"] is True and {r["key"]: r["value"] for r in raw}["B"] == 4096, "the FILE holds the bool / int"
    for lab in st.labels(ART):
        if lab.key not in ("result_artifact",):
            assert vocab.check(lab.key, lab.value) is None, (lab.key, lab.value)


# ---- refresh-fixtures -----------------------------------------------------------------------------------------------------------------


def _register(st: LocalStore, repo: Path) -> tuple[str, str]:
    """Two repo replicas the way the map has always been written: a file and a tree, bare fixture/v1, meta {} refs {}."""
    f = repo / "fixtures" / "probe" / "a.json"
    d = repo / "fixtures" / "probe" / "battery"
    d.mkdir(parents=True)
    f.write_text('{"x": 1}\n')
    (d / "one.txt").write_text("one\n")
    (d / "sub").mkdir()
    (d / "sub" / "two.txt").write_text("two\n")
    a = st.put_artifact("fixture/v1", {}, None, file=f)
    b = st.put_artifact("fixture/v1", {}, None, tree=d)
    assert F.fixture_id(f) == a and F.fixture_id(d) == b          # the module recomputes exactly what `put` registered
    (repo / F.REPLICA_MAP).write_text(json.dumps({a: "fixtures/probe/a.json", b: "fixtures/probe/battery"}, indent=2) + "\n")
    return a, b


def test_refresh_fixtures_is_idempotent_and_records_history(env, capsys):
    st = LocalStore()
    repo = env / "repo"
    a, b = _register(st, repo)
    before = (repo / F.REPLICA_MAP).read_text()
    assert main(["refresh-fixtures"]) == 0
    out, err = capsys.readouterr()
    assert out == "" and "2 fixture(s)" in err and "0 changed, 2 unchanged, 0 missing; map unchanged" in err
    assert (repo / F.REPLICA_MAP).read_text() == before
    # change the file, add to the tree: dry run reports both, puts nothing, writes nothing
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 2}\n')
    (repo / "fixtures" / "probe" / "battery" / "three.txt").write_text("three\n")
    a2, b2 = F.fixture_id(repo / "fixtures" / "probe" / "a.json"), F.fixture_id(repo / "fixtures" / "probe" / "battery")
    n_manifests = len(list((st.root / "manifests").rglob("*.json")))
    assert main(["refresh-fixtures", "--dry-run"]) == 0
    out, err = capsys.readouterr()
    assert f"{a} -> {a2}  fixtures/probe/a.json" in out and f"{b} -> {b2}  fixtures/probe/battery" in out
    assert "2 changed, 0 unchanged, 0 missing; dry run, nothing written" in err
    assert (repo / F.REPLICA_MAP).read_text() == before and len(list((st.root / "manifests").rglob("*.json"))) == n_manifests
    assert not st.manifest_path(a2[4:]).exists()
    # for real: put under the new ids with the registered shape, map rewritten, old ids kept under `previous`
    assert main(["refresh-fixtures"]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == [f"{a} -> {a2}  fixtures/probe/a.json", f"{b} -> {b2}  fixtures/probe/battery"]
    assert "2 changed, 0 unchanged, 0 missing; map rewritten" in err
    for new in (a2, b2):
        m = st.get_manifest(new)
        assert m.kind == "fixture/v1" and m.meta == {} and m.refs == {}
    assert st.get_manifest(a2).payload.type == "file" and st.get_manifest(b2).payload.type == "tree"
    doc = json.loads((repo / F.REPLICA_MAP).read_text())
    assert list(doc) == [a2, b2, "previous"]                                       # entry order kept, history last
    assert doc[a2] == "fixtures/probe/a.json" and doc[b2] == "fixtures/probe/battery"
    assert doc["previous"] == {a2: [a], b2: [b]}
    assert (repo / F.REPLICA_MAP).read_text().endswith("}\n")
    # the store reads the rewritten map as before: entries only, the `previous` block is invisible
    fresh = LocalStore()
    assert fresh.replica_map() == {a2: "fixtures/probe/a.json", b2: "fixtures/probe/battery"}
    assert fresh.where(a2)[1]["present"] is True                                    # the repo replica is verified at the new id
    # idempotent
    text = (repo / F.REPLICA_MAP).read_text()
    assert main(["refresh-fixtures"]) == 0
    out, err = capsys.readouterr()
    assert out == "" and "0 changed, 2 unchanged, 0 missing; map unchanged" in err and (repo / F.REPLICA_MAP).read_text() == text
    # a second change accumulates history, oldest first; reverting the bytes to an old id lists that id no more
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 3}\n')
    a3 = F.fixture_id(repo / "fixtures" / "probe" / "a.json")
    assert main(["refresh-fixtures"]) == 0
    assert f"{a2} -> {a3}  fixtures/probe/a.json  (previously also {a})" in capsys.readouterr().out
    assert json.loads((repo / F.REPLICA_MAP).read_text())["previous"] == {a3: [a, a2], b2: [b]}
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 1}\n')
    assert main(["refresh-fixtures"]) == 0
    capsys.readouterr()
    doc = json.loads((repo / F.REPLICA_MAP).read_text())
    assert doc[a] == "fixtures/probe/a.json" and doc["previous"] == {a: [a2, a3], b2: [b]}


def test_refresh_fixtures_preserve_pushes_and_reports(env, fs_remote, capsys):
    st = LocalStore()
    repo = env / "repo"
    a, _ = _register(st, repo)
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 9}\n')
    a2 = F.fixture_id(repo / "fixtures" / "probe" / "a.json")
    assert main(["refresh-fixtures", "--preserve", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["preserve"] and rep["written"] and rep["changed"] == 1 and rep["unchanged"] == 1
    e = next(x for x in rep["entries"] if x["status"] == "changed")
    assert e == {"id": a, "path": "fixtures/probe/a.json", "status": "changed", "new_id": a2, "previous": [a], "preserved": True, "errors": []}
    assert st.verify(a2)["preserved"] is True
    assert main(["refresh-fixtures", "--preserve"]) == 0
    assert "0 changed, 2 unchanged" in capsys.readouterr().err
    # a vault that cannot be written is reported per entry and exits 1; the map is still made honest
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 10}\n')
    a3 = F.fixture_id(repo / "fixtures" / "probe" / "a.json")
    (env / "store.toml").write_text('[remote]\ntype = "fs"\npath = "/dev/null/not-a-dir"\n')
    assert main(["refresh-fixtures", "--preserve"]) == 1
    assert f"{a2} -> {a3}  fixtures/probe/a.json  (previously also {a})  NOT preserved: " in capsys.readouterr().out
    assert json.loads((repo / F.REPLICA_MAP).read_text())[a3] == "fixtures/probe/a.json"


def test_refresh_fixtures_missing_conflict_and_no_map(env, tmp_path, capsys):
    st = LocalStore()
    repo = env / "repo"
    a, _ = _register(st, repo)
    # a path that left the working tree is reported, kept, and fails the command; nothing else changes
    (repo / "fixtures" / "probe" / "a.json").unlink()
    text = (repo / F.REPLICA_MAP).read_text()
    assert main(["refresh-fixtures"]) == 1
    out, err = capsys.readouterr()
    assert f"{a}  fixtures/probe/a.json  MISSING: not in the working tree" in out and "0 changed, 1 unchanged, 1 missing" in err
    assert (repo / F.REPLICA_MAP).read_text() == text
    # two entries whose bytes (and bare file names -- identity is content plus name) become identical would collide on one id:
    # reported as a conflict, neither is rewritten
    (repo / "fixtures" / "probe" / "a.json").write_text('{"x": 1}\n')
    (repo / "fixtures" / "other").mkdir()
    (repo / "fixtures" / "other" / "a.json").write_text('{"x": 1}\n')
    doc = json.loads(text)
    doc["art:" + "f" * 64] = "fixtures/other/a.json"
    (repo / F.REPLICA_MAP).write_text(json.dumps(doc, indent=2) + "\n")
    assert main(["refresh-fixtures", "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    bad = [x for x in rep["entries"] if x["status"] == "conflict"]
    assert len(bad) == 1 and bad[0]["new_id"] == a and "already registered" in bad[0]["errors"][0] and not rep["written"]
    # without a map there is nothing to refresh
    assert main(["refresh-fixtures", "--repo", str(tmp_path / "elsewhere")]) == 1
    assert "no fixtures/artifacts.json found" in capsys.readouterr().err


# ---- doctor -----------------------------------------------------------------------------------------------------------------------------


def _attempt(run_id: str) -> dict:
    t = "2026-09-22T10:59:05Z"
    return {"schema": "research/attempt/v0.1", "id": run_id, "tool": {"name": "bench_vu", "version": "1"}, "derivation": "drv:" + "a" * 64,
            "reuse_key": "b" * 64, "argv": ["python", "-m", "x"], "params": {}, "key_params": [], "inputs": {}, "conditions": {}, "key_conditions": [],
            "source": {"commit": "c" * 40, "tree": "d" * 40, "dirty": False, "dirty_digest": None, "lockfiles": {}}, "closure": {},
            "execution": {"start_utc": t, "end_utc": t, "wall_s": 1.5, "host": "vy-g5", "project": "verity", "campaign": "r21"},
            "cost": {"provider": "runpod", "usd": 0.01}, "status": {"state": "exited", "rc": 0, "class": None, "failure": None, "validation": "passed"},
            "outputs": {}, "telemetry": {"events": None, "resources": None}}


def _run_dir(root: Path, name: str, *, error: str | None, state: str | None) -> Path:
    d = root / name
    d.mkdir()
    (d / "job.json").write_text("{}")
    if state:
        (d / "status.json").write_text(json.dumps({"state": state}))
    if error is not None:
        (d / "store_publish_error.json").write_text(json.dumps({"error": error, "at_utc": "2026-09-22T09:00:00Z"}))
    return d


def test_doctor_lists_stale_versus_real_and_cleans_only_the_stale(env, capsys):
    st = LocalStore()
    runs = env / "runs"
    st.put_attempt(_attempt(RUN_STALE))
    stale = _run_dir(runs, RUN_STALE, error="ConfigError: credentials env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set\nmore", state="done")
    real = _run_dir(runs, RUN_REAL, error=f"RuntimeError: run {RUN_REAL} is running: the store only sees completed or failed attempts", state="running")
    fine = _run_dir(runs, RUN_FINE, error=None, state="done")
    odd = _run_dir(runs, "not-a-run-id", error="whatever", state=None)
    (runs / "loose-file.txt").write_text("not a run dir")
    # read-only by default: two rows, two counts, nothing deleted
    assert main(["doctor"]) == 0
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert lines == [f"{RUN_STALE}  stale  attempt in the store  ConfigError: credentials env vars AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set",
                     "not-a-run-id  REAL   state=?  whatever",
                     f"{RUN_REAL}  REAL   state=running  RuntimeError: run {RUN_REAL} is running: the store only sees completed or failed attempts"]
    assert f"3 store_publish_error.json file(s) under {runs}: 1 stale (run is an attempt in {st.root}), 2 real (not in the store); read-only" in err
    assert (stale / "store_publish_error.json").is_file() and (real / "store_publish_error.json").is_file()
    # the JSON shape
    assert main(["doctor", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["cleaned"] is False and [r["run"] for r in rep["stale"]] == [RUN_STALE] and {r["run"] for r in rep["real"]} == {RUN_REAL, "not-a-run-id"}
    assert rep["stale"][0]["state"] == "done" and rep["stale"][0]["at_utc"] == "2026-09-22T09:00:00Z" and rep["stale"][0]["deleted"] is False
    # --clean deletes the stale file and nothing else
    before = sorted(p.relative_to(runs).as_posix() for p in runs.rglob("*"))
    assert main(["doctor", "--clean"]) == 0
    out, err = capsys.readouterr()
    assert f"{RUN_STALE}  stale  deleted  ConfigError" in out and "deleted 1 stale file(s), nothing else touched" in err
    after = sorted(p.relative_to(runs).as_posix() for p in runs.rglob("*"))
    assert set(before) - set(after) == {f"{RUN_STALE}/store_publish_error.json"}
    assert (stale / "job.json").is_file() and (stale / "status.json").is_file()
    assert (real / "store_publish_error.json").is_file() and (odd / "store_publish_error.json").is_file() and (fine / "job.json").is_file()
    # a second pass finds only the real ones; --runs-dir points elsewhere
    assert main(["doctor"]) == 0
    out, err = capsys.readouterr()
    assert RUN_STALE not in out and "0 stale" in err and "2 real" in err
    assert main(["doctor", "--runs-dir", str(env / "nowhere")]) == 0
    assert "0 store_publish_error.json file(s)" in capsys.readouterr().err
