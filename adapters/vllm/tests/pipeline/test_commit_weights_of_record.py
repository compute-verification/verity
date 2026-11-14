"""`verity_vllm.pipeline.commit`: the weights of record, which `main` reads or derives once for each committer after its
weights are registered and holds every pair's served parameters to; and the served row's source, which `main` attaches to
the committer it makes.

Real: all that `tests.pipeline.test_commit_main` runs for real; the digests of the served model's parameters
(`weights_of_record.live_param_digests`); and each pair's pin (`weights_of_record.check`).  The fixture keeps what `main`
handed each (``handed``) and what each gave before `main` wrote on it (``gave``).

Stood in: that module's stand-ins; the four functions that derive a record from a checkpoint's shards and a Program
(`build_record`, `served_record`, `stamp_of_record_set`, `derivation_provenance`), each of which gives what a test states;
the sampled replay a Program's directory would start (`c2_replay.run`); and, for the served row, its source (`Served`), which
ends `main` where the committers' setup has ended.

Held: which of them `main` calls, in order, and with what; the record's file, where `main` writes one; every line `main`
prints between the weights' registration and the instrumented warm-up; and what each pair's row and line say of the pin.
CPU only."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.pipeline import test_commit_main as M
from tests.pipeline.test_commit_main import PLUGIN, RUN, WARMUP, Ended, Record, run
from verity_vllm.check import weights_of_record as WR
from verity_vllm.commit.step_rows import program_field
from verity_vllm.pipeline import c2_replay, commit, gpu_lease, served_match

#: `test_commit_main`'s fixtures, under their names in this module.
stand_ins, before_the_lease = M.stand_ins, M.before_the_lease

MANIFEST, VALUE_CORRESPONDENCE = M.MANIFEST, M.VALUE_CORRESPONDENCE
ACQUIRE = gpu_lease.acquire
CHECKPOINTS = "manifests/checkpoints.json"
ROOT = hashlib.sha256(b"the root of the record").hexdigest()
PROGRAM = hashlib.sha512(b"the Program the record is of").hexdigest()
STAMP: Record = {"program_digests": [PROGRAM], "why": None}
PROVENANCE: Record = {"tool": "the derivation's own", "sources": ["instances.json.gz"]}
TOOL = "verity_vllm.pipeline.commit (in-process --checkpoint-dir fallback)"
REFERENCE = "the pinned checkpoint shards (manifests/checkpoints.json sha256), composed by the implementation's stacking rule (weights_of_record)"
NOT_EVALUATED = "MAN-09 weights pin (not evaluated)"
NO_PROGRAM = "--checkpoint-dir needs the Program of record (--program-dir / --programs-root)"
RECORD_LINE = ("[delta] weights of record ({kind}): root {root} fields {pinned}/{fields} pinned, unpinned {unpinned}, unresolved {unresolved}, "
               "shards pinned {shards}; live digests 1 tensors in {wall} s")
WARNING_LINE = "[delta] WARNING weights of record not derived: {error}"
PIN_LINE = ("[delta] checkpoint_binding (WEIGHTS PIN, pair {pair}): {verdict} root {root} (record for Program {program}) compared {compared} "
            "mismatched {mismatched} missing 0 unpinned {unpinned} shards pinned {shards}{why}")
BASE_FILES = ["declaration_stand_in.json", "runs.jsonl", "steps.jsonl", "summary.json", "summary.md", "verdict.json", "versions.json"]


def record_of(model: M.Model, **more: Any) -> Record:
    """The record of the stand-in model's one parameter as its checkpoint has it: complete, of pinned shards, of `WORKLOAD`'s
    model."""
    digest = hashlib.sha256(model.head.weight.detach().numpy().tobytes()).hexdigest()
    return {"program_digest": PROGRAM, "root_of_record": ROOT, "n_fields": 1, "n_pinned": 1, "unpinned": [], "unresolved": [], "complete": True,
            "why": None, "engine_assumptions": {"kv_cache_dtype": "auto"},
            "fields": [{"field": "head.weight", "served_name": "head.weight", "shape": [M.VOCAB, M.VOCAB], "bits": 32, "digest": digest, "pinned": True}],
            "checkpoint": {"dir": "/checkpoint", "entry": {"repo": "org/model", "revision": "f" * 40}, "all_shards_pinned": True,
                           "shards": {"model.safetensors": {"sha256": "5" * 64, "pinned_sha256": "5" * 64, "pinned": True}}}, **more}


@pytest.fixture
def pinned(stand_ins, monkeypatch):
    """`stand_ins` with the weights of record's functions: the two that read the served model are kept, and the four that
    derive a record are stood in.  ``record`` is what a derivation gives, ``fails`` an exception by the name of the function
    that raises it, ``replayed`` the pairs whose sampled replay was asked for."""
    s = stand_ins
    s.handed, s.gave, s.fails, s.replayed = [], {}, {}, []
    s.record = record_of(M.Model())

    def kept(name: str, real: Any, *, copied: bool = True) -> None:
        def call(*args: Any, **kw: Any) -> Any:
            s.handed.append((name, copy.deepcopy(args) if copied else args, copy.deepcopy(kw)))
            if name in s.fails:
                raise s.fails[name]
            out = real(*args, **kw)
            s.gave[name] = copy.deepcopy(out)
            return out

        monkeypatch.setattr(WR, name, call)

    def stamp(rec: Record, root: str, *, manifest: str | None = None) -> Record:
        rec["of_record_set"] = copy.deepcopy(STAMP)
        return rec

    kept("live_param_digests", WR.live_param_digests, copied=False)
    kept("check", WR.check)
    kept("build_record", lambda *args, **kw: copy.deepcopy(s.record))
    kept("served_record", lambda *args, **kw: copy.deepcopy(s.record))
    kept("stamp_of_record_set", stamp)
    kept("derivation_provenance", lambda *args, **kw: copy.deepcopy(PROVENANCE))
    monkeypatch.setattr(c2_replay, "run", lambda sources, cache: s.replayed.append(sources.pair))
    return s


def names(s: SimpleNamespace) -> list[str]:
    return [name for name, _, _ in s.handed]


def handed(s: SimpleNamespace, name: str) -> tuple[tuple[Any, ...], Record]:
    """The first call of ``name``: what `main` handed it, by position and by keyword."""
    return next((args, kw) for n, args, kw in s.handed if n == name)


def as_json(doc: Any) -> Any:
    return json.loads(json.dumps(doc, default=str))


def record_at(tmp_path: Path, doc: Any) -> Path:
    (tmp_path / "record.json").write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return tmp_path / "record.json"


def setup_lines(ended: Ended) -> list[str]:
    """What `main` printed after the weights' registration and before the instrumented warm-up's line."""
    lines = ended.printed[next(i for i, line in enumerate(ended.printed) if line.startswith("[delta] weight registration")) + 1:]
    return lines[:next(i for i, line in enumerate(lines) if line.startswith("[delta] warm-up instrumented"))]


def record_line(s: SimpleNamespace, **more: Any) -> str:
    return RECORD_LINE.format(**{"kind": PLUGIN, "root": ROOT[:16], "pinned": 1, "fields": 1, "unpinned": 0, "unresolved": 0, "shards": True,
                                 "wall": s.gave["live_param_digests"]["wall_s"], **more})


def pin_lines(ended: Ended) -> list[str]:
    return [line for line in ended.printed if line.startswith("[delta] checkpoint_binding")]


def pin_line(pair: int, verdict: str = "OK [ok]", why: str | None = None, **more: Any) -> str:
    return PIN_LINE.format(**{"pair": pair, "verdict": verdict, "root": ROOT[:16], "program": PROGRAM[:16], "compared": 1, "mismatched": 0,
                              "unpinned": 0, "shards": True, "why": f" -- {why}" if why else "", **more})


def pins(ended: Ended) -> list[Record]:
    """Each instrumented run's pin as its row keeps it, after the row's value correspondence is seen to say the same of it."""
    out = []
    for row in ended.rows():
        if row["arm"] == "control":
            assert "commit" not in row
            continue
        pin = row["commit"]["weights_pin"]
        failed = pin["result"] is False
        assert row["commit"]["value_correspondence"] == {
            **VALUE_CORRESPONDENCE, "result": False if failed else None, "why": f"weights_pin: {pin['why']}" if failed else None,
            "components": {"weights_pin": {
                "result": pin["result"], "why": pin["why"], "root_of_record": pin.get("root_of_record"), "compared": pin.get("compared"),
                "mismatched": pin.get("mismatched"), "missing": pin.get("missing"), "unpinned_n": len(pin.get("unpinned") or []),
                "unresolved": pin.get("unresolved"), "all_shards_pinned": pin.get("all_shards_pinned"), "analytic": [], "reference": REFERENCE}}}
        out.append(pin)
    return out


def pin_of(s: SimpleNamespace, source: str | None) -> Record:
    """A pair's pin as `main` records it: what `check` gave, with the record's source and the root the committer registered."""
    return as_json({**s.gave["check"], "source": source, "committed_weights_root": M.weights_root(s.engine.model)})


def not_evaluated(s: SimpleNamespace, why: str) -> Record:
    return {"result": None, "why": why, "method": NOT_EVALUATED, "source": None, "committed_weights_root": M.weights_root(s.engine.model)}


def error_of(call: Any) -> Exception:
    """What ``call`` raises, whichever it is: the exception `main` records."""
    with pytest.raises(Exception) as raised:
        call()
    return raised.value


def test_a_commit_given_no_record_and_no_checkpoint_takes_no_digest_and_pins_nothing(tmp_path, pinned, capsys):
    pinned.verdicts = [(True, None)]
    ended = run(tmp_path, capsys)
    assert ended.exit is None and pinned.handed == [] and setup_lines(ended) == [] and pin_lines(ended) == []
    assert ended.files() == BASE_FILES and all("weights_pin" not in row.get("commit", {}) for row in ended.rows())


@pytest.mark.parametrize("more", [(), ("--checkpoint-dir", "/checkpoint", "--engine-arg", "kv_cache_dtype=auto")], ids=["alone", "with-a-checkpoint"])
def test_a_record_given_as_a_file_is_read_once_after_the_registration_and_every_pair_is_held_to_it(tmp_path, pinned, capsys, more):
    """``--weights-of-record``: nothing is derived, with a checkpoint's directory or without, and no file is written.  The served
    parameters are digested once, and each pair's pin is `check`'s of the file's record and those digests."""
    pinned.verdicts = [(True, None)]
    file = record_at(tmp_path, record_of(M.Model()))
    ended = run(tmp_path, capsys, "--weights-of-record", str(file), *more)
    [com] = M.MADE
    assert ended.exit is None and com.calls == ["register_weights", *WARMUP, *RUN, *RUN, "close"] and ended.files() == BASE_FILES
    assert names(pinned) == ["live_param_digests", "check", "check"]
    (model,), kw = handed(pinned, "live_param_digests")
    live = pinned.gave["live_param_digests"]
    assert model is pinned.engine.model and kw == {} and live["n"] == 1 and list(live["params"]) == ["head.weight"]
    of_file = {**record_of(M.Model()), "source": str(file)}
    assert [(args, kw) for name, args, kw in pinned.handed if name == "check"] == 2 * [((of_file, live), {
        "program_digests_of_record": None, "checkpoint_of_record": {"repo": "org/model", "revision": None}, "kv_cache_dtype": "auto"})]
    assert setup_lines(ended) == [record_line(pinned)]
    pin = pin_of(pinned, str(file))
    assert pin["result"] is True and pin["compared"] == 1 and pin["why"] is None and pins(ended) == [pin, pin]
    assert pin_lines(ended) == [pin_line(0), pin_line(1)]


def test_what_the_record_leaves_open_is_counted_on_its_line_and_fails_each_pair_by_name(tmp_path, pinned, capsys):
    """A record that is not complete: its line counts what it pins, leaves unpinned and could not resolve, and says what it
    says of its shards; each pair's pin fails with the record's own reason."""
    pinned.verdicts = [(True, None)]
    reason = "1 Program weight field(s) have no checkpoint tensor of record: c.weight"
    left = {"n_fields": 4, "unpinned": ["a.k_scale", "b.k_scale"], "unresolved": [{"field": "c.weight", "why": "no tensor"}], "complete": False,
            "why": reason}
    file = record_at(tmp_path, record_of(M.Model(), **left))
    ended = run(tmp_path, capsys, "--weights-of-record", str(file))
    assert setup_lines(ended) == [record_line(pinned, fields=4, unpinned=2, unresolved=1)]
    why = "weights of record incomplete: " + reason
    pin = pin_of(pinned, str(file))
    assert pin["result"] is False and pin["why"] == why and pins(ended) == [pin, pin]
    assert pin_lines(ended) == [pin_line(pair, "FAIL [mismatch]", why, unpinned=2) for pair in (0, 1)]


def test_a_flipped_weight_is_digested_as_flipped_and_fails_the_pin_and_the_commit_by_its_name(tmp_path, pinned, capsys):
    """``--flip-weight`` with the record of the checkpoint: the digests are of the parameters as the committer registered them,
    so the pin names the changed one, the row's value correspondence fails, and the Commit's own verdict, of a committer that
    declares every required class, ends it with exit 3 for that reason."""
    M.MAKE.update(declare=M.CLASSES)
    file = record_at(tmp_path, record_of(M.Model()))
    ended = run(tmp_path, capsys, "--weights-of-record", str(file), "--flip-weight", "head.weight:0")
    flipped = pinned.gave["live_param_digests"]["params"]["head.weight"]["digest"]
    assert flipped == hashlib.sha256(pinned.engine.model.head.weight.detach().numpy().tobytes()).hexdigest() != record_of(M.Model())["fields"][0]["digest"]
    why = "1 served weight(s) differ from the checkpoint of record: head.weight"
    pin = pin_of(pinned, str(file))
    assert pin["result"] is False and pin["why"] == why and [m["field"] for m in pin["mismatched"]] == ["head.weight"]
    assert pins(ended) == [pin, pin] and pin_lines(ended) == [pin_line(pair, "FAIL [mismatch]", why, mismatched=1) for pair in (0, 1)]
    assert ended.exit == 3 and [row["commit"]["commit_pass"] for row in ended.rows() if row["arm"] != "control"] == [False, False]
    assert ended.doc("verdict.json")["first_fail_reason"] == f"weights_pin: {why}"
    assert ended.printed[-1] == M.FAIL_LINE.format(failed=2, n=2, pair=0, arm="instrumented", why=f"weights_pin: {why}")


@pytest.mark.parametrize(("text", "cut"), [(None, False), (b"not JSON", False), (b"[]", False), (b"\xff" * 400, True)],
                         ids=["absent", "not-json", "not-a-record", "not-text"])
def test_a_record_that_cannot_be_read_is_not_evaluated_and_takes_no_digest(tmp_path, pinned, capsys, text, cut):
    """`main` goes on: the reason is printed once, whole, and recorded in each pair's pin, cut to 300 characters; no pair's pin
    has a result."""
    pinned.verdicts = [(True, None)]
    file = tmp_path / "record.json"
    if text is not None:
        file.write_bytes(text)
    error = error_of(lambda: json.loads(file.read_text()).__setitem__("source", str(file)))
    ended = run(tmp_path, capsys, "--weights-of-record", str(file))
    assert ended.exit is None and pinned.handed == [] and ended.files() == BASE_FILES
    assert setup_lines(ended) == [WARNING_LINE.format(error=repr(error))]
    why = f"weights of record not derived: {error!r}"[:300]
    assert (len(repr(error)) > 300) == cut
    assert pins(ended) == 2 * [not_evaluated(pinned, why)]
    assert pin_lines(ended) == [pin_line(pair, "NOT EVALUATED [not evaluated]", why, root="?", program="?", compared=None, shards=None) for pair in (0, 1)]


def test_a_record_whose_line_cannot_be_written_is_not_evaluated_though_the_digests_were_taken(tmp_path, pinned, capsys):
    """A record that is no record where its line reads it: the digests are taken, the failure is the reason, and no pair is
    held to what was read."""
    pinned.verdicts = [(True, None)]
    file = record_at(tmp_path, record_of(M.Model(), unpinned=5))
    error = error_of(lambda: len(json.loads("5")))
    ended = run(tmp_path, capsys, "--weights-of-record", str(file))
    assert ended.exit is None and names(pinned) == ["live_param_digests"] and setup_lines(ended) == [WARNING_LINE.format(error=repr(error))]
    why = f"weights of record not derived: {error!r}"
    assert pins(ended) == 2 * [not_evaluated(pinned, why)]


def derived(s: SimpleNamespace, source: str, **more: Any) -> Record:
    return {**s.record, "source": source, **more}


def test_a_record_is_derived_from_the_program_of_record_and_the_checkpoint_and_written(tmp_path, pinned, capsys):
    """``--checkpoint-dir`` with ``--program-dir``: the record is built from the two under the engine's checkpoint pins, named for
    where it came from, given its derivation's provenance under this command's name, and written before the digests are taken."""
    pinned.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--checkpoint-dir", "/checkpoint", "--program-dir", "/program")
    assert ended.exit is None and names(pinned) == ["build_record", "derivation_provenance", "live_param_digests", "check", "check"]
    assert handed(pinned, "build_record") == (("/program", "/checkpoint"), {"checkpoints_manifest": CHECKPOINTS, "kv_cache_dtype": "auto"})
    assert handed(pinned, "derivation_provenance") == (("/program", "/checkpoint", CHECKPOINTS, None), {})
    record = derived(pinned, "derived in-process from /program x /checkpoint", derived={**PROVENANCE, "tool": TOOL})
    assert (ended.out / "weights_of_record.json").read_text() == json.dumps(record, indent=1) + "\n"
    assert ended.files() == sorted([*BASE_FILES, "weights_of_record.json"]) and pinned.replayed == [0, 1]
    assert handed(pinned, "check")[0] == (record, pinned.gave["live_param_digests"])
    assert setup_lines(ended) == [record_line(pinned)]
    pin = pin_of(pinned, record["source"])
    assert pin["result"] is True and pins(ended) == [pin, pin] and pin_lines(ended) == [pin_line(0), pin_line(1)]


def test_the_engines_kv_cache_type_is_the_one_a_record_is_derived_under_and_each_pair_is_held_to(tmp_path, pinned, capsys):
    """``--engine-arg kv_cache_dtype=...`` reaches the derivation and each pin; a record that assumes another fails by name."""
    pinned.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--checkpoint-dir", "/checkpoint", "--program-dir", "/program", "--engine-arg", "kv_cache_dtype=fp8")
    assert handed(pinned, "build_record")[1] == {"checkpoints_manifest": CHECKPOINTS, "kv_cache_dtype": "fp8"}
    assert handed(pinned, "check")[1]["kv_cache_dtype"] == "fp8"
    why = "record assumes kv_cache_dtype=auto; the engine runs kv_cache_dtype=fp8 (kv scales not of record)"
    assert [pin["why"] for pin in pins(ended)] == [why, why]


def programs(root: Path, *named: str, empty: tuple[str, ...] = ()) -> str:
    """A root of Programs' directories: each of ``named`` holds its rows' file, each of ``empty`` holds none."""
    for name in (*named, *empty):
        (root / name).mkdir(parents=True)
    for name in named:
        (root / name / "instances.json.gz").write_bytes(b"")
    return str(root)


@pytest.mark.parametrize(("named", "empty", "argv", "program"), [
    (("build_request", "build_request_LP3_T3", "build_request_LP2_T2"), ("build_request_LP1_T1",), (), "{root}/build_request_LP2_T2"),
    (("build_request_c", "build_request_b"), ("build_request", "build_request_LP1_T1"), (), "{root}/build_request_b"),
    (("build_request", "build_request_LP2_T2"), (), ("--program-dir", "/program"), "/program"),
], ids=["a-component", "another-request-program", "the-one-named"])
def test_under_a_programs_root_the_record_is_of_the_first_component_and_stamped_with_the_whole_set(tmp_path, pinned, capsys, named, empty, argv, program):
    """``--checkpoint-dir`` with ``--programs-root``: the Program is the one named, or else the first component that holds its
    rows, or else the first other request Program that does.  The record is stamped over the whole root once it is named and
    before its provenance, and the stamp is in the file."""
    pinned.verdicts = [(True, None)]
    root = programs(tmp_path / "programs", *named, empty=empty)
    program = program.format(root=root)
    ended = run(tmp_path, capsys, "--checkpoint-dir", "/checkpoint", "--programs-root", root, *argv)
    assert ended.exit is None and names(pinned) == ["build_record", "stamp_of_record_set", "derivation_provenance", "live_param_digests", "check", "check"]
    assert handed(pinned, "build_record") == ((program, "/checkpoint"), {"checkpoints_manifest": CHECKPOINTS, "kv_cache_dtype": "auto"})
    source = f"derived in-process from {program} x /checkpoint"
    assert handed(pinned, "stamp_of_record_set") == ((derived(pinned, source), root), {"manifest": None})
    assert handed(pinned, "derivation_provenance") == ((program, "/checkpoint", CHECKPOINTS, None), {})
    record = derived(pinned, source, of_record_set=STAMP, derived={**PROVENANCE, "tool": TOOL})
    assert ended.doc("weights_of_record.json") == record and handed(pinned, "check")[0] == (record, pinned.gave["live_param_digests"])
    assert setup_lines(ended) == [record_line(pinned)] and pins(ended) == 2 * [pin_of(pinned, source)]


@pytest.mark.parametrize(("argv", "fails", "asked", "written"), [
    ((), None, [], False),
    (("--programs-root", "{root}"), None, [], False),
    (("--program-dir", "/program"), ("build_record", ValueError("a shard is not there")), ["build_record"], False),
    (("--program-dir", "/program"), ("live_param_digests", RuntimeError("a parameter could not be read")),
     ["build_record", "derivation_provenance", "live_param_digests"], True),
], ids=["no-program", "a-root-with-no-rows", "the-derivation-fails", "the-digests-fail"])
def test_a_record_that_cannot_be_derived_is_not_evaluated(tmp_path, pinned, capsys, argv, fails, asked, written):
    """``--checkpoint-dir`` with no Program of record, or with a derivation or a digest that raises: `main` goes on with the
    reason in place of a record, and no pair's pin has a result.  A file written before the failure stays."""
    pinned.verdicts = [(True, None)]
    root = programs(tmp_path / "programs", empty=("build_request", "build_request_LP2_T2"))
    error = RuntimeError(NO_PROGRAM)
    if fails is not None:
        error = fails[1]
        pinned.fails[fails[0]] = error
    ended = run(tmp_path, capsys, "--checkpoint-dir", "/checkpoint", *(a.format(root=root) for a in argv))
    assert ended.exit is None and names(pinned) == asked and setup_lines(ended) == [WARNING_LINE.format(error=repr(error))]
    assert ("weights_of_record.json" in ended.files()) == written
    why = f"weights of record not derived: {error!r}"
    assert pins(ended) == 2 * [not_evaluated(pinned, why)]
    assert pin_lines(ended) == [pin_line(pair, "NOT EVALUATED [not evaluated]", why, root="?", program="?", compared=None, shards=None) for pair in (0, 1)]


class AfterTheSetup(Exception):
    """Raised in place of the served Build's rebuild: `main` has set up every committer."""


class Served:
    """`served_match.ServedCommit` as `main` reads it to the end of the committers' setup.  `ORDER` keeps each committer it is
    attached to with the Program digest that committer then had."""

    def __init__(self, build: Record, fault: str | None) -> None:
        self.build, self.src = build, SimpleNamespace(fault=fault, slots=[0], plans=[0, 1], modules=[0, 1, 2])

    def engine(self, eng: SimpleNamespace) -> Record:
        return {"served_engine": "stand-in"}

    def attach(self, com: Any) -> None:
        ORDER.append(("attach", com, getattr(com, "program_digest", None), self))

    def rebuild(self, workload: Record, out: str) -> Record:
        raise AfterTheSetup


#: What the served source and the attachment of record were handed, in order; `serving` empties it.
ORDER: list[tuple[Any, ...]] = []
SERVED_CLASSES = (*M.CLASSES, "compiled_kernel_buffers")


@pytest.fixture
def serving(before_the_lease, pinned, monkeypatch):
    """`pinned` on the served row: the manifest's readers stood in as `before_the_lease` has them, the lease taken as it is,
    the source `Served`, and an attachment of record that keeps what it was handed in `ORDER`."""
    ORDER.clear()

    def attach_of_record(com, eng, args, manifest, out, **kw):
        ORDER.append(("attach_of_record", com, getattr(com, "program_digest", None), kw["served"], manifest, kw["required_classes"]))
        return SimpleNamespace(required_classes=kw["required_classes"], moe_attached=False, hidden_required=None)

    monkeypatch.setattr(gpu_lease, "acquire", ACQUIRE)
    monkeypatch.setattr(served_match, "ServedCommit", Served)
    monkeypatch.setattr(commit, "attach_of_record", attach_of_record)
    return pinned


def served(tmp_path: Path, capsys: Any, *argv: str) -> Ended:
    """`run` on the served row, which `main` leaves after the committers' setup."""
    manifest = M.manifest_at(tmp_path)
    with pytest.raises(AfterTheSetup):
        run(tmp_path, capsys, "--compiled-taps", "served", "--required-manifest", str(manifest), "--program-digest", MANIFEST["program_digest"], *argv)
    return Ended(None, tmp_path / "out", capsys.readouterr().out.splitlines())


def test_the_served_source_is_attached_to_the_committer_before_its_attachment_of_record_and_its_builds_digest_binds_the_roots(tmp_path, serving, capsys):
    """``--compiled-taps served``: the committer `main` makes is handed to the source, takes the served Build's Program digest as
    the roots bind it, and only then gets its attachment of record, which is told of the source.  With no record and no
    checkpoint nothing is digested."""
    ended = served(tmp_path, capsys)
    [com] = M.MADE
    [(_, attached, before, source), (_, of_record, digest, told, manifest, classes)] = ORDER
    assert attached is com and of_record is com and before is None and told is source and isinstance(source, Served)
    assert source.build == MANIFEST and manifest == MANIFEST and classes == SERVED_CLASSES
    assert digest == com.program_digest == program_field(MANIFEST["program_digest"])
    assert com.calls == ["register_weights", *WARMUP] and serving.handed == [] and setup_lines(ended) == []
    assert f"[delta] required classes (served): {SERVED_CLASSES}" in ended.printed
    assert "[delta] served source: slots 1 plans 2 written at graph capture, 3 identities every step" in ended.printed
    assert ended.doc("versions.json")["served_engine"] == "stand-in" and "weights_of_record.json" not in ended.files()


@pytest.mark.parametrize(("argv", "asked", "kv"), [
    ((), "served_record", "auto"), (("--program-dir", "/program"), "served_record", "auto"),
    (("--engine-arg", "kv_cache_dtype=fp8"), "served_record", "fp8"), (("--weights-of-record", "{file}"), None, None),
], ids=["alone", "with-a-program", "with-the-engines-kv-cache-type", "with-a-record"])
def test_the_served_rows_record_is_derived_from_its_builds_weight_reads_unless_one_is_given(tmp_path, serving, capsys, argv, asked, kv):
    """``--compiled-taps served`` with ``--checkpoint-dir``: the record is the served Build's, of the manifest's document and
    under the engine's kv cache type, whether or not a Program's directory is named; it is written, with no stamp and no
    provenance.  A record given as a file is read in its place."""
    file = record_at(tmp_path, record_of(M.Model()))
    ended = served(tmp_path, capsys, "--checkpoint-dir", "/checkpoint", *(a.format(file=file) for a in argv))
    assert names(serving) == [*([asked] if asked else []), "live_param_digests"] and setup_lines(ended) == [record_line(serving)]
    if asked is None:
        assert "weights_of_record.json" not in ended.files()
        return
    assert handed(serving, "served_record") == ((MANIFEST, "/checkpoint"), {"checkpoints_manifest": CHECKPOINTS, "kv_cache_dtype": kv})
    record = derived(serving, "derived in-process from the served Build's weight reads x /checkpoint")
    assert (ended.out / "weights_of_record.json").read_text() == json.dumps(record, indent=1) + "\n"


def test_the_stamp_is_told_of_the_manifest_a_commit_is_given(tmp_path, serving, capsys, monkeypatch):
    """``--checkpoint-dir`` and ``--programs-root`` with ``--required-manifest`` on a row that is not the served one: the record is
    built from the root's Program, and the stamp over the root is handed the manifest's path."""
    def at_commit(*args: Any, **kw: Any) -> None:
        raise AfterTheSetup

    monkeypatch.setattr(commit.PO, "at_commit", at_commit)
    root = programs(tmp_path / "programs", "build_request")
    manifest = M.manifest_at(tmp_path)
    with pytest.raises(AfterTheSetup):
        run(tmp_path, capsys, "--required-manifest", str(manifest), "--program-digest", MANIFEST["program_digest"], "--checkpoint-dir", "/checkpoint",
            "--programs-root", root)
    assert names(serving) == ["build_record", "stamp_of_record_set", "derivation_provenance", "live_param_digests"]
    source = f"derived in-process from {root}/build_request x /checkpoint"
    assert handed(serving, "stamp_of_record_set") == ((derived(serving, source), root), {"manifest": str(manifest)})
    [(_, com, digest, told, of_manifest, _)] = ORDER
    assert com is M.MADE[0] and told is None and of_manifest == MANIFEST and digest is None and not hasattr(com, "program_digest")

