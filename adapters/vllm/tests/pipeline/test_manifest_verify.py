"""`verity-vllm manifest-verify` (`verity_vllm.pipeline.manifest_verify`): the manifest a Commit used against the one rebuilt from
the Build's Programs, the builders stood in.  One request's Program is rebuilt by `manifest.build` and a batch's by
`manifest.build_global`, each under the policies and the query the command line names; the record is printed, written where
`--out` says, and the exit is its `ok`.  Neither a Program nor a fixture with its Programs' directory: refused.  A rebuild
that names no policy runs under none, as rank 0 of 1 and under the query of record."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import cli, manifest
from verity_vllm.pipeline import manifest_verify as MV
from verity_vllm.query.manifest.format import manifest_digest
from verity_vllm.query.manifest.verify import verify
from verity_vllm.query.required import QUERY_OF_RECORD

PROGRAM = "ab" * 64
REBUILT = {"manifest_digest": manifest_digest([]), "program_digest": PROGRAM, "populations": {"identities": 0}, "complete": True}


@pytest.fixture
def built(monkeypatch):
    """What each builder was called with, in order; each says a line to its `log` and returns `REBUILT`."""
    calls: list[tuple] = []

    def builder(name):
        def build(*args, log, **kw):
            log(f"{name} says")
            calls.append((name, args, kw))
            return REBUILT
        return build

    for name in ("build", "build_global"):
        monkeypatch.setattr(manifest, name, builder(name))
    return calls


def _used(tmp_path, **changed):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"identities": [], "manifest_digest": manifest_digest([]), "program_digest": PROGRAM,
                                "query": {"engine": "v2-query"}, **changed}))
    return str(path)


def test_one_request_s_manifest_is_rebuilt_from_its_program_and_workload_under_every_policy_and_the_query_of_record(tmp_path, built, capsys):
    used, out = _used(tmp_path), tmp_path / "commit" / "of" / "manifest_verify.json"
    (tmp_path / "wl.json").write_text(json.dumps({"requests": [{"request_id": "r0"}]}))
    assert cli.main(["manifest-verify", "--manifest", used, "--program", "D/build_request", "--workload", str(tmp_path / "wl.json"), "--out", str(out)]) == 0
    assert built == [("build", ("D/build_request",), {
        "workload": {"requests": [{"request_id": "r0"}]}, "rank": 0, "ranks": 1, "profile": None, "norm_scales": True,
        "taps": ("router_softmax", "vocab_range"), "query": QUERY_OF_RECORD})]
    rec = verify(json.loads((tmp_path / "manifest.json").read_text()), REBUILT, manifest_path=used)
    assert rec["ok"] is True and rec["manifest"] == used
    assert out.read_text() == json.dumps(rec, indent=1)                         # its directory made
    printed = capsys.readouterr()
    assert printed.out == json.dumps(rec, indent=1) + "\n[manifest] VERIFY OK: the manifest of record\n"
    assert printed.err == "build says\n"                                        # the builder's lines, on stderr


def test_a_program_with_no_workload_named_is_rebuilt_with_none(tmp_path, built):
    assert cli.main(["manifest-verify", "--manifest", _used(tmp_path), "--program", "D/build_request", "--fixture", str(tmp_path / "f.json")]) == 0
    assert [(name, args, kw["workload"]) for name, args, kw in built] == [("build", ("D/build_request",), None)]      # the Program wins
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]      # no `--out`: the record is printed alone


def test_a_batch_s_manifest_is_rebuilt_from_its_fixture_and_programs_under_the_policies_and_the_query_named(tmp_path, built):
    (tmp_path / "fixture.json").write_text(json.dumps({"requests": [{"request_id": "r0"}, {"request_id": "r1"}]}))
    given = ["manifest-verify", "--manifest", _used(tmp_path), "--fixture", str(tmp_path / "fixture.json"), "--programs-root", "D"]
    assert cli.main([*given, "--workload-digest", "cd" * 64, "--rank", "1", "--ranks", "2", "--profile", "p", "--no-norm-scales",
                     "--no-router-softmax", "--query", "Q_module_body_v1"]) == 0
    assert cli.main([*given, "--no-vocab-range"]) == 0
    assert cli.main([*given, "--no-router-softmax", "--no-vocab-range"]) == 0
    fixture = {"requests": [{"request_id": "r0"}, {"request_id": "r1"}]}
    assert built == [
        ("build_global", (fixture, "D"), {"workload_digest": "cd" * 64, "rank": 1, "ranks": 2, "profile": "p", "norm_scales": False,
                                          "taps": ("vocab_range",), "query": "Q_module_body_v1"}),
        ("build_global", (fixture, "D"), {"workload_digest": None, "rank": 0, "ranks": 1, "profile": None, "norm_scales": True,
                                          "taps": ("router_softmax",), "query": QUERY_OF_RECORD}),
        ("build_global", (fixture, "D"), {"workload_digest": None, "rank": 0, "ranks": 1, "profile": None, "norm_scales": True,
                                          "taps": (), "query": QUERY_OF_RECORD})]


def test_a_rebuild_that_names_no_policy_is_rank_0_of_1_under_none_and_the_query_of_record(tmp_path, monkeypatch):
    for name in ("build", "build_global"):
        monkeypatch.setattr(manifest, name, lambda *args, **kw: (args, kw))
    unnamed = {"rank": 0, "ranks": 1, "profile": None, "norm_scales": False, "taps": (), "query": QUERY_OF_RECORD, "log": None}
    assert MV.rebuild(program="D/build_request") == (("D/build_request",), {"workload": None, **unnamed})
    (tmp_path / "fixture.json").write_text('{"requests": []}')
    assert MV.rebuild(fixture=str(tmp_path / "fixture.json"), programs_root="D") == (({"requests": []}, "D"), {"workload_digest": None, **unnamed})


def test_a_manifest_that_is_not_the_rebuilt_one_exits_1_and_says_why(tmp_path, built, capsys):
    used, out = _used(tmp_path, program_digest="ef" * 64), tmp_path / "manifest_verify.json"
    assert cli.main(["manifest-verify", "--manifest", used, "--program", "D/build_request", "--out", str(out)]) == 1
    rec = json.loads(out.read_text())
    assert rec["ok"] is False and rec["why"] == f"program_digest {('ef' * 64)[:16]} != the Build's {PROGRAM[:16]}"
    assert capsys.readouterr().out == json.dumps(rec, indent=1) + f"\n[manifest] VERIFY FAIL: {rec['why']}\n"


@pytest.mark.parametrize("given", [[], ["--fixture", "f.json"], ["--programs-root", "D"], ["--workload", "wl.json"]],
                         ids=["neither", "a fixture alone", "the Programs' directory alone", "a workload alone"])
def test_neither_a_program_nor_a_fixture_with_its_programs_directory_is_refused(tmp_path, built, monkeypatch, given):
    monkeypatch.chdir(tmp_path)
    for name in ("f.json", "wl.json"):
        (tmp_path / name).write_text("{}")
    with pytest.raises(SystemExit) as refused:
        MV.main(cli.parse("manifest-verify", ["--manifest", _used(tmp_path), *given], {}))
    assert str(refused.value) == "manifest_verify: give --program [--workload] (B=1) or --fixture --programs-root [--workload-digest] (B>=2)"
    assert built == []
