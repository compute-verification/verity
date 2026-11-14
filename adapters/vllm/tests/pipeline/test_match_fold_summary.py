"""A fold's summary (`verity_vllm.pipeline.match_fold.fold_summary_result`) of a log written here, the fold itself stood in:
what the summary states of a fold's result.  Each root counts once under its class, a weight the profile does not know is named,
the unbound roots are the ones the fold's notes name, a step's instances are counted by spec with the first step the prefill, a
request's sampled tokens and their steps are kept in order, and the requests are the result's or else the header's.  The profile
folds a declared MoE construction through its own hook, a run whose header records a PoUW executor folds PoUW's calls first, and
the Programs' two options reach the profile and the document.  The peak resident size is in GiB on either platform, and
`_jsonable` leaves a document that `json` writes.  The instances' stored form is the one the environment names, a result
whose digest is not its Program's own is refused before a file is written, and the Program written is the fold's own encoding
under its digest, or else the one made here.  The instances' file states each instance whole, its operands' references as runs
or as progressions, and the access journal each entry's view and references the same way, compressed at the level given."""
from __future__ import annotations

import dataclasses
import enum
import gzip
import json
import re
import resource
import sys
import types

import pytest
from verity.circuits import codec
from verity.circuits.refs import Affine, Strided

from verity_vllm.check import gates as G
from verity_vllm.observe import events as EV
from verity_vllm.observe import log as LOG
from verity_vllm.observe.fold import fold as FOLD
from verity_vllm.observe.fold.resolver import Profile
from verity_vllm.pipeline import match_fold as MF
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.records import schema_ids

PROFILES = "verity_vllm.observe.fold.profiles"
VIEW = EV.ViewDesc(alloc=0, offset=0, shape=(1,), strides=(1,), dtype="int64", device="cpu")
REQUESTS = [{"request_id": "r0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4}, {"request_id": "r1", "prompt_token_ids": [], "max_tokens": 2},
            {"request_id": "r2", "max_tokens": 1}]
ROOTS = [("model.w", "weight"), ("model.w", "weight"), ("scale.x", "weight"), ("model.b", "buffer"), ("scale.y", "buffer"), ("prompt[r0]", "prompt_token"),
         ("prompt[r1]", "prompt_token"), ("pos", "position"), ("one", "const")]
SUMMARY = {"spec": {"Add": 2, "Mul": 2}, "error_classes": {"opaque": 1}, "error_examples": ["e"], "params": 9, "ret_type": "Array"}


class MoEFusedExpertsPadded:
    """A pattern by the name the summary reads the scope's pattern from."""


class Linear:
    """Any other pattern."""


def _profile(*patterns):
    return Profile(id="prof-1", patterns=[*patterns], config={}, weight_field=lambda name: ("w",) if name.startswith("model.") else None)


def _log(tmp_path, *, notes=None, requests=REQUESTS):
    header = EV.RunHeader(schema="s", created_utc="", profile_id="prof-1", profile={}, model={}, workload={"requests": requests}, notes=notes or {})
    path = tmp_path / "log.jsonl.gz"
    with LOG.LogWriter(path) as log:
        for event in (header, *(EV.Root(view=VIEW, role=role, name=name) for name, role in ROOTS)):
            log.write(event)
    return path


def _result(**given):
    unbound = "unbound root: Root {!r} (weight, " + repr(VIEW) + ") has no weights field in profile prof-1"
    return types.SimpleNamespace(**{
        "digest": "d1", "unresolved": 2, "memory": {"peak": 5}, "timings": {"fold": 0.5}, "report": types.SimpleNamespace(unsupported=3),
        "notes": [unbound.format("scale.y"), "a note of something else: Root 'model.w' (weight)", unbound.format("scale.x"), unbound.format("scale.y")],
        "instances": [{"step": 4, "spec": "Mul"}, {"step": 4, "spec": "Add"}, {"step": 4, "spec": "Mul"}, {"step": 6, "spec": "Add"}],
        "steps": [{"step": 4, "rows": 3, "computes": 7, "unsupported": 1, "errors": 0, "seconds": 0.25}, {"step": 5, "rows": 1}],
        "sampled": {"r0": [(4, 11), (5, 12)], "r1": [(6, 13)]},
        "errors": [{"class": "Opaque"}, {"class": "prescribed"}, {"class": "other"}, {}], **given})


@pytest.fixture
def profiles(monkeypatch):
    """Puts a profile's module where the fold looks one up by name."""
    def put(name, **attrs):
        module = types.ModuleType(f"{PROFILES}.{name}")
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return put


@pytest.fixture
def folded(monkeypatch):
    """The fold stood in: the result it gives and its summary, and what each fold was called with."""
    calls = []

    def stand_in(result):
        def fold(events, profile, *, strict, with_program):
            calls.append({"events": events, "profile": profile, "strict": strict, "with_program": with_program})
            return result
        monkeypatch.setattr(FOLD, "fold", fold)
        monkeypatch.setattr(FOLD, "summarize", lambda folded_result: {**SUMMARY, "of_the_result": folded_result is result})
        return calls
    return stand_in


def test_the_summary_states_the_fold_s_result_its_roots_by_class_its_steps_and_each_request_s_tokens(tmp_path, profiles, folded, monkeypatch):
    profiles("stand_in", PROFILE=_profile(Linear()))
    monkeypatch.setattr(MF, "_max_rss_gb", lambda: 1.23456)
    result, log = _result(), _log(tmp_path)
    calls = folded(result)
    doc, live = MF.fold_summary_result(log, "stand_in")
    assert live is result and len(calls) == 1
    assert calls[0]["events"] == list(LOG.read_log(log)) and (calls[0]["strict"], calls[0]["with_program"]) == (None, True)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", doc.pop("utc")) and doc.pop("seconds") >= 0 and doc.pop("max_rss_gb") == 1.235
    assert doc == {
        "schema": G.FOLD_SUMMARY_SCHEMA, "log": str(log), "profile": "stand_in", "profile_id": "prof-1", "with_program": True, "digest": "d1",
        "spec": SUMMARY["spec"], "moe_construction": None, "moe_scope_pattern": None, "sampler_construction": None, "registered_values": False,
        "n_instances": 4, "n_steps": 3, "unsupported": 3, "unresolved": 2, "opaque_fallbacks": 2, "errors": 4,
        "error_classes": SUMMARY["error_classes"], "error_examples": SUMMARY["error_examples"],
        "unknown_roots": ["scale.x", "scale.y"],                                # a weight or a buffer the profile has no field for
        "root_classes": {"weight": 2, "unknown": 2, "prompt_token": 2, "position": 1, "const": 1},     # `model.w` once, though the log has it twice
        "unbound_roots": ["scale.x", "scale.y"],
        "per_step": [
            {"step": 4, "phase": "prefill", "rows": 3, "n_instances": 3, "computes": 7, "spec_histogram": {"Add": 1, "Mul": 2}, "unsupported": 1,
             "errors": 0, "seconds": 0.25},
            {"step": 5, "phase": "decode", "rows": 1, "n_instances": 0, "computes": None, "spec_histogram": {}, "unsupported": None, "errors": None,
             "seconds": None},                                                   # a step with no instance
            {"step": 6, "phase": "decode", "rows": None, "n_instances": 1, "computes": None, "spec_histogram": {"Add": 1}, "unsupported": None,
             "errors": None, "seconds": None}],                                  # instances of a step the fold kept no record of
        "tokens": {"r0": [11, 12], "r1": [13]}, "sampled_steps": {"r0": [4, 5], "r1": [6]},
        "requests": [{"request_id": "r0", "prompt_len": 3, "max_tokens": 4, "n_tokens": 2}, {"request_id": "r1", "prompt_len": 0, "max_tokens": 2, "n_tokens": 1},
                     {"request_id": "r2", "prompt_len": 0, "max_tokens": 1, "n_tokens": 0}],
        "params": 9, "ret_type": "Array", "memory": {"peak": 5}, "timings": {"fold": 0.5}, "fold": {**SUMMARY, "of_the_result": True}}
    assert list(doc["per_step"][0]["spec_histogram"]) == ["Add", "Mul"] and list(doc["root_classes"]) == ["weight", "unknown", "prompt_token", "position", "const"]
    assert json.loads(json.dumps(MF._jsonable(doc))) == doc


def test_the_requests_are_the_fold_result_s_before_the_header_s_and_none_when_neither_has_any(tmp_path, profiles, folded):
    profiles("stand_in", PROFILE=_profile())
    kept = [types.SimpleNamespace(request_id="q0", prompt_token_ids=[5, 6], max_tokens=7), {"request_id": "q1", "prompt_token_ids": [8], "max_tokens": 9},
            types.SimpleNamespace()]                                            # a request that states nothing
    folded(_result(requests=kept, sampled={"q0": [(0, 1)]}, instances=[], steps=[], errors=[], notes=[]))
    doc, _ = MF.fold_summary_result(_log(tmp_path), "stand_in", with_program=False)
    assert doc["requests"] == [{"request_id": "q0", "prompt_len": 2, "max_tokens": 7, "n_tokens": 1}, {"request_id": "q1", "prompt_len": 1, "max_tokens": 9, "n_tokens": 0},
                               {"request_id": None, "prompt_len": 0, "max_tokens": None, "n_tokens": 0}]
    assert (doc["with_program"], doc["per_step"], doc["n_steps"], doc["opaque_fallbacks"], doc["errors"], doc["unbound_roots"]) == (False, [], 0, 0, 0, [])
    folded(_result(requests=[]))
    assert [r["request_id"] for r in MF.fold_summary_result(_log(tmp_path), "stand_in")[0]["requests"]] == ["r0", "r1", "r2"]
    assert MF.fold_summary_result(_log(tmp_path, requests=[]), "stand_in")[0]["requests"] == []
    assert next(iter(LOG.read_log(_log(tmp_path)))).workload == {"requests": REQUESTS}


def test_a_profile_s_own_classes_name_its_roots_and_without_them_a_weight_is_one_the_profile_has_a_field_for(tmp_path, profiles, folded):
    folded(_result())
    profiles("classed", PROFILE=_profile(), root_class=lambda name: "scale" if name.startswith("scale.") else "unknown")
    doc, _ = MF.fold_summary_result(_log(tmp_path), "classed")
    assert (doc["root_classes"], doc["unknown_roots"]) == ({"unknown": 2, "scale": 2, "prompt_token": 2, "position": 1, "const": 1}, ["model.w", "model.b"])


def test_a_declared_moe_construction_folds_under_the_profile_s_hook_and_is_refused_by_name_without_one(tmp_path, profiles, folded):
    calls, hooked = folded(_result()), []

    def hook(profile, construction):
        hooked.append((profile.patterns, construction))
        return _profile(MoEFusedExpertsPadded())

    base = _profile(Linear())
    profiles("hooked", PROFILE=base, with_moe_construction=hook)
    profiles("plain", PROFILE=base)
    doc, _ = MF.fold_summary_result(_log(tmp_path), "hooked", moe_construction="padded")
    assert hooked == [(base.patterns, "padded")] and [type(p) for p in calls[-1]["profile"].patterns] == [MoEFusedExpertsPadded]
    assert (doc["moe_construction"], doc["moe_scope_pattern"]) == ("padded", "MoEFusedExpertsPadded")
    for undeclared in (None, TargetProfile.DEFAULT_MOE_CONSTRUCTION):          # the profile as it is: no hook is asked
        for name in ("hooked", "plain"):
            doc, _ = MF.fold_summary_result(_log(tmp_path), name, moe_construction=undeclared)
            assert (doc["moe_construction"], doc["moe_scope_pattern"]) == (undeclared, None) and calls[-1]["profile"].patterns == base.patterns
    assert len(hooked) == 1
    refusal = ("workload declares moe_construction='padded' but profile plain has no with_moe_construction hook: it cannot fold that construction "
               "(refused by name; the MOE-01 fold is NOT a substitute)")
    with pytest.raises(SystemExit, match=f"^{re.escape(refusal)}$"):
        MF.fold_summary_result(tmp_path / "no-log-is-read.jsonl.gz", "plain", moe_construction="padded")


def test_the_programs_options_reach_the_profile_and_a_pouw_run_folds_pouw_s_calls_first(tmp_path, profiles, folded):
    calls = folded(_result())
    profiles("stand_in", PROFILE=_profile(Linear()))
    construction = TargetProfile.SAMPLER_CONSTRUCTIONS[0]
    doc, _ = MF.fold_summary_result(_log(tmp_path, notes={"protocols": ["pouw:duo"]}), "stand_in", sampler_construction=construction, registered_values=True)
    assert (doc["sampler_construction"], doc["registered_values"], doc["moe_scope_pattern"]) == (construction, True, None)
    assert calls[-1]["profile"].registered_values is True and [type(p).__name__ for p in calls[-1]["profile"].patterns] == ["PouwLinear", "Linear"]
    MF.fold_summary_result(_log(tmp_path, notes={"protocols": ["other:x"]}), "stand_in")
    assert calls[-1]["profile"].registered_values is False and [type(p).__name__ for p in calls[-1]["profile"].patterns] == ["Linear"]
    with pytest.raises(SystemExit, match=r"^sampler construction 'no-such' is not one of \["):
        MF.fold_summary_result(tmp_path / "no-log-is-read.jsonl.gz", "stand_in", sampler_construction="no-such")


@pytest.mark.parametrize("platform, peak", [("linux", 3 << 20), ("darwin", 3 << 30)])      # KiB on Linux, bytes on macOS
def test_the_peak_resident_size_is_in_gib_on_either_platform(monkeypatch, platform, peak):
    asked = []

    def getrusage(who):
        asked.append(who)
        return types.SimpleNamespace(ru_maxrss=peak)

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(resource, "getrusage", getrusage)
    assert MF._max_rss_gb() == 3.0 and asked == [resource.RUSAGE_SELF]


class _Colour(enum.IntEnum):
    RED = 2


class _Array:
    def tolist(self):
        return [1, 2]


class _Opaque:
    def __repr__(self):
        return "<opaque>"


def test_jsonable_keeps_json_s_own_values_and_turns_keys_tuples_arrays_and_anything_else_into_them():
    given = {1: (1, [2.5, None, True]), "s": "text", ("a", 1): _Colour.RED, "array": _Array(), "other": _Opaque(), "nested": {2: {3: (_Opaque(),)}}}
    made = MF._jsonable(given)
    assert made == {"1": [1, [2.5, None, True]], "s": "text", "('a', 1)": 2, "array": [1, 2], "other": "<opaque>", "nested": {"2": {"3": ["<opaque>"]}}}
    assert made["('a', 1)"] is _Colour.RED and json.loads(json.dumps(made)) == made


def test_the_instances_stored_form_is_runs_unless_the_environment_names_progressions_and_another_word_is_refused(monkeypatch):
    monkeypatch.delenv(MF.INSTANCES_FORM_ENV, raising=False)
    assert MF.instances_form() == "runs"
    for given, form in (("runs", "runs"), ("progressions", "progressions"), (" Progressions", "progressions"), ("", "runs")):
        monkeypatch.setenv(MF.INSTANCES_FORM_ENV, given)
        assert MF.instances_form() == form
    monkeypatch.setenv(MF.INSTANCES_FORM_ENV, "Neither")                        # the word is read in lower case
    with pytest.raises(SystemExit, match=rf"^{MF.INSTANCES_FORM_ENV}='neither': expected runs \| progressions$"):
        MF.instances_form()


def test_a_result_with_no_encoding_of_its_own_whose_digest_is_not_its_program_s_is_refused_and_nothing_is_written(tmp_path, monkeypatch):
    program, encoded = object(), []

    def encode_program(given):
        encoded.append(given)
        return {"encoded": True}

    monkeypatch.setattr(codec, "encode_program", encode_program)
    monkeypatch.setattr(codec, "program_digest", lambda descriptor: "ab" * 64 if descriptor == {"encoded": True} else "no")
    result = types.SimpleNamespace(program=program, digest="cd" * 64)
    with pytest.raises(SystemExit, match="^program.json digest abababababababab != fold digest cdcdcdcdcdcdcdcd$"):
        MF.persist_program(result, tmp_path, profile_name="stand_in", log_path=tmp_path / "log.jsonl.gz")
    assert encoded == [program] and list(tmp_path.iterdir()) == []


def _nodes(n):
    return types.SimpleNamespace(nodes=[object()] * n)


@pytest.mark.parametrize("own, stated, digest", [
    (True, "cd" * 64, "cd" * 64),                                               # the fold's encoding and digest: nothing is made again
    (True, None, "ab" * 64),                                                    # its encoding and no digest: the encoding's
    (False, "ab" * 64, "ab" * 64),                                              # no encoding of its own: made here, the digest the Program's
    (False, None, "ab" * 64),                                                   # neither
])
def test_the_program_written_is_the_fold_s_own_encoding_under_its_digest_or_else_the_one_made_here(tmp_path, monkeypatch, own, stated, digest):
    program, made = types.SimpleNamespace(root=types.SimpleNamespace(body=_nodes(5))), []

    def encode_program(given):
        made.append(("encoded", given))
        return {"encoded": "here"}

    def program_digest(given):
        made.append(("digested", given))
        return "ab" * 64

    monkeypatch.setattr(codec, "encode_program", encode_program)
    monkeypatch.setattr(codec, "program_digest", program_digest)
    monkeypatch.delenv(MF.INSTANCES_FORM_ENV, raising=False)
    descriptor = {"encoded": "by the fold"} if own else {"encoded": "here"}
    result = types.SimpleNamespace(program=program, body=_nodes(2), digest=stated, instances=[], **({"descriptor": descriptor} if own else {}))
    written = MF.persist_program(result, tmp_path, profile_name="stand_in", log_path=tmp_path / "log.jsonl.gz")
    assert made == [*([] if own else [("encoded", program)]), *([] if own and stated else [("digested", descriptor)])]
    text = (tmp_path / "program.json").read_text()
    assert text == json.dumps({"schema": MF.PROGRAM_SCHEMA, "profile": "stand_in", "log": str(tmp_path / "log.jsonl.gz"), "digest": digest,
                               "n_input_nodes": 3, "n_instances": 2, "program": descriptor, "params": [], "tp": None, "peers_layout": [],
                               "witness_layout": [], "supplied": {}}, separators=(",", ":"), sort_keys=True) + "\n"
    assert written["program"] == {"path": str(tmp_path / "program.json"), "bytes": len(text), "digest": digest, "n_input_nodes": 3}
    assert list(written) == ["program", "instances"] and sorted(p.name for p in tmp_path.iterdir()) == ["instances.jsonl", "program.json"]


def test_the_program_s_document_states_the_result_s_parameters_by_name_its_ranks_its_layouts_and_what_was_supplied(tmp_path, monkeypatch):
    monkeypatch.delenv(MF.INSTANCES_FORM_ENV, raising=False)
    stated = {"params": [("prompt_0", _Opaque()), ("weights", 7)], "tp": {"rank": 1, "world": 2}, "peers_layout": [("peers_0", (0, 1))],
              "witness_layout": ({"name": "schedule_0"},), "supplied": {"table": (1, 2)}}
    result = types.SimpleNamespace(program=types.SimpleNamespace(root=types.SimpleNamespace(body=_nodes(1))), body=_nodes(1), digest="cd" * 64,
                                   instances=[], descriptor={}, **stated)
    MF.persist_program(result, tmp_path, profile_name="stand_in", log_path="log")
    doc = json.loads((tmp_path / "program.json").read_text())
    assert {k: doc[k] for k in stated} == {"params": [["prompt_0", "<opaque>"], ["weights", "7"]], "tp": {"rank": 1, "world": 2},
                                           "peers_layout": [["peers_0", [0, 1]]], "witness_layout": [{"name": "schedule_0"}],
                                           "supplied": {"table": [1, 2]}}
    assert (doc["program"], doc["digest"], doc["n_input_nodes"], doc["n_instances"]) == ({}, "cd" * 64, 0, 1)


@dataclasses.dataclass
class _Window:
    """A view that is no `ViewDesc`."""
    lo: int
    hi: tuple


SEEN = EV.ViewDesc(alloc=3, offset=8, shape=(2, 5), strides=(5, 1), dtype="int64", device="cpu")
SEEN_STORED = {"alloc": 3, "offset": 8, "shape": [2, 5], "strides": [5, 1], "dtype": "int64", "device": "cpu"}
ROWS = Strided("n", 2, 0, 2, 4, 1, 6)                                           # node 2's leaves 0-1, 4-5 and 8-9: three runs, or one progression
ROWS_STORED = {"runs": [["n", 2, 0, 2], ["n", 2, 4, 6], ["n", 2, 8, 10]], "progressions": [["N", 2, 0, 2, 0, 4, 4, 3]]}
META = ("node", "step", "event", "resolution", "pattern", "spec", "row", "rows", "name", "hint", "request", "position")


def _node(*operands):
    return types.SimpleNamespace(args=[types.SimpleNamespace(refs=refs) for refs in operands])


@pytest.mark.parametrize("form", ["runs", "progressions"])
def test_the_instances_file_states_each_instance_whole_and_its_operands_references_in_the_stored_form(tmp_path, monkeypatch, form):
    monkeypatch.setenv(MF.INSTANCES_FORM_ENV, form)
    whole = {"node": 1, "step": 4, "event": 17, "resolution": "exact", "pattern": "Linear", "spec": "Mul", "row": 0, "rows": (0, 2), "name": "h.0",
             "hint": _Opaque(), "request": "r0", "position": 3}
    instances = [{**whole, "operand_roots": ["w", "b"], "collective": {"kind": "gather", "peers": (1,)}, "operand_views": [SEEN, _Window(1, (2, 3))],
                  "output_views": [{"alloc": 9}], "not a field of the record": 1},
                 {"node": 0, "step": 5, "operand_roots": [], "collective": {}},  # node 0 is a node; an empty list of roots or collective is left out
                 {"step": 6}]                                                    # an instance of no node
    result = types.SimpleNamespace(program=None, digest="cd" * 64, instances=instances, body=types.SimpleNamespace(nodes=[
        _node(Affine("p", 0, 3, 2)), _node(ROWS, Affine("n", 1, 5, 1))]))
    written = MF.persist_program(result, tmp_path, profile_name="stand_in", log_path="log")
    text = (tmp_path / "instances.jsonl").read_text()
    assert written == {"instances": {"path": str(tmp_path / "instances.jsonl"), "bytes": len(text), "n": 3, "operand_refs_form": form}}
    assert [p.name for p in tmp_path.iterdir()] == ["instances.jsonl"]         # a result with no Program and no access writes neither file
    absent = dict.fromkeys(META)
    stored = [
        {**whole, "rows": [0, 2], "hint": "<opaque>", "operand_roots": ["w", "b"], "collective": {"kind": "gather", "peers": [1]},
         "operand_views": [SEEN_STORED, {"lo": 1, "hi": [2, 3]}], "output_views": [{"alloc": 9}], "operand_refs": [ROWS_STORED[form], [["n", 1, 5, 6]]]},
        {**absent, "node": 0, "step": 5, "operand_views": [], "output_views": [], "operand_refs": [[["p", 0, 3, 5]]]},
        {**absent, "step": 6, "operand_views": [], "output_views": []}]
    schema = {"runs": schema_ids.REALIZED_INSTANCES_SCHEMA, "progressions": schema_ids.REALIZED_INSTANCES_PROGRESSIONS_SCHEMA}[form]
    head = {"schema": schema, "profile": "stand_in", "digest": "cd" * 64, "n_instances": 3,
            "note": "root-circuit node of instance k = n_input_nodes + k (program.json)", "requests": [], "operand_refs_form": form}
    assert text == json.dumps(head) + "\n" + "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in stored)
    assert list(stored[0]) == [*META, "operand_roots", "collective", "operand_views", "output_views", "operand_refs"]
    assert schema_ids.REALIZED_INSTANCES_SCHEMA != schema_ids.REALIZED_INSTANCES_PROGRESSIONS_SCHEMA
    no_body = types.SimpleNamespace(program=None, digest=None, instances=[{"node": 1, "step": 4}], body=None)
    MF.persist_program(no_body, tmp_path, profile_name="stand_in", log_path="log")
    assert [json.loads(line) for line in (tmp_path / "instances.jsonl").read_text().splitlines()[1:]] == [
        {**absent, "node": 1, "step": 4, "operand_views": [], "output_views": []}]     # a result that holds no body states no references


@pytest.mark.parametrize("form", ["runs", "progressions"])
def test_the_access_journal_states_each_entry_s_view_and_references_in_the_stored_form_at_the_level_given(tmp_path, monkeypatch, form):
    monkeypatch.setenv(MF.INSTANCES_FORM_ENV, form)
    really_open, opened = gzip.open, []

    def gzip_open(path, mode, compresslevel):
        opened.append((path, mode, compresslevel))
        return really_open(path, mode, compresslevel=compresslevel)

    monkeypatch.setattr(gzip, "open", gzip_open)
    accesses = [{"kind": "alloc", "event": 1, "view": SEEN, "refs": None},
                {"kind": "read", "event": 2, "view": None, "refs": [("n", 2, 0, 2), ("p", 0, 3, 5)]},      # references that are runs already
                {"kind": "write", "event": 3, "refs": ROWS, 7: (1, _Opaque())}]                         # the fold's own; a key that is no string
    result = types.SimpleNamespace(program=None, digest="cd" * 64, instances=[], body=None, accesses=accesses)
    journal = tmp_path / "accesses.jsonl.gz"
    for given, level in (({}, 9), ({"journal_level": 1}, 1)):
        written = MF.persist_program(result, tmp_path, profile_name="stand_in", log_path="log", **given)
        assert list(written) == ["instances", "accesses"] and written["accesses"] == {"path": str(journal), "bytes": journal.stat().st_size, "n": 3}
        assert opened.pop() == (journal, "wt", level) and opened == []
        with really_open(journal, "rt") as f:
            text = f.read()
        stored = [{"kind": "alloc", "event": 1, "view": SEEN_STORED, "refs": None},
                  {"kind": "read", "event": 2, "view": None, "refs": [["n", 2, 0, 2], ["p", 0, 3, 5]]},  # runs they stay, in either form
                  {"kind": "write", "event": 3, "refs": ROWS_STORED[form], "7": [1, "<opaque>"]}]
        assert text == "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in stored)
    journal.unlink()
    no_access: tuple[dict[str, object], ...] = ({}, {"accesses": None}, {"accesses": []})
    for none in no_access:
        quiet = types.SimpleNamespace(program=None, digest="cd" * 64, instances=[], body=None, **none)
        assert list(MF.persist_program(quiet, tmp_path, profile_name="stand_in", log_path="log")) == ["instances"] and not journal.exists()
    assert opened == []
