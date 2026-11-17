"""`verity bench compute` (`run.py`): the number is the MACs an accepting Pearl+ answer credits over the MACs the served
model computes at the policy's class, printed as text and as one line of JSON beside its target; an answer that isn't
an accept, or that the Lean verifier's verdict isn't the same as, prints no number and exits 1; a policy that can't
give the model's MACs exits 2. The model's MACs are held to a count made one position, one query-key pair and one row
of logits at a time. PoUW's row in the run's composed record is printed whatever it says, and nothing the developer's
node reports of what it served is read.

Nothing here reaches the evidence store: `FakeStore` stands in for research's local store over the small record under
`fixtures/record` (a run's directory: its `job.json` and its `out/`), which each test copies and edits. A record
withdrawn whole, or in a part that holds a file the command reads, is refused before anything is fetched, and a
withdrawal of other files of it is printed. `--from DIR` reads the same record from a directory, through no store, and
prints what `--from RUN` prints of it."""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import run as C

HERE = Path(__file__).resolve().parent
RUN = "r20000101-000000-aaaa"
COMMIT = "c" * 40
RECORD = HERE / "fixtures" / "record"
LLAMA = C.MODELS["unsloth/Llama-3.2-1B"]
#: the fixture's class, 4 slots of 32 prompt and 4 output tokens, on Llama 3.2 1B
MACS = {"linears": 136_230_993_920, "attention": 165_150_720, "lm_head": 4_202_692_608}
TOTAL = 140_598_837_248
#: the fixture's credit and the MACs its passes hold: every linear of the class, less 3 excluded A rows, at ε = 1/100 and
#: ι = 1/400 of each call
CREDITED = 134_502_518_410
PASSES = 136_201_633_792
SHARE = CREDITED / TOTAL
#: the revision of the benchmark's load (load.toml), which the fixture's policy registers
REVISION = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
#: the fixture's coins and parameters as the kit's line prints them, an in-process run's at development parameters
RAN_WITH = ("ran with: coins in-process; parameters development (sample size 459, delta 1/100, "
            "scheme pearl-c-h100-v1-h3, sample from the serving machine, salt coins from in-process, "
            "Lean verdict development)")
#: the fixture's own statement that it is made up (its `job.json`'s ``made_up``), and the line that prints it
SAYS = json.loads((RECORD / "job.json").read_text())["made_up"]
MADE_UP = [C.kit.kit_report.MADE_UP, C.made_up_text(SAYS)]
#: the edit that makes the fixture a record that doesn't say it is made up, as a run's, whose result.json is written
REAL = {"JOB": lambda d: d.pop("made_up")}
#: the fixture's composed record's reason for PoUW's verdict, and what the printed line says of any verdict but accept
WHY = "a development run: its record names no devices it was served on, which PoUW's verdict needs"
NOTHING = "PoUW credits nothing of this run, and the number above is the Pearl+ verifier's answer alone"


def a_record(tmp_path: Path, **edits) -> Path:
    """A copy of the fixture record, a run's directory (its `job.json`, and its files under `out/`), with `edits`
    applied, each a file's name under `C` (ANSWER, LEAN, LEAN_CHECK, POLICY, VERDICTS, VERIFY, DEVELOPMENT, RAN, JOB) and what becomes of
    it: None deletes it, a string replaces its text, and a function edits its JSON in place."""
    record = tmp_path / "record"
    shutil.copytree(RECORD, record)
    for name, edit in edits.items():
        path = record / getattr(C, name)
        if edit is None:
            path.unlink()
        elif isinstance(edit, str):
            path.write_text(edit)
        else:
            doc = json.loads(path.read_text())
            edit(doc)
            path.write_text(json.dumps(doc))
    return record


def to(*keys, value) -> Callable[[dict], None]:
    """An edit that sets `value` at the path `keys` of a JSON document."""
    def edit(doc: dict) -> None:
        for k in keys[:-1]:
            doc = doc[k]
        doc[keys[-1]] = value
    return edit


class Refused(Exception):
    """What the fake store raises, as research's store raises its own error."""


class FakeStore:
    """Research's local store as `run.EvidenceStore` reads it, over records laid out under a directory: attempts,
    labels, a record's listing, and a fetch that hands back a directory of only the files asked for."""

    def __init__(self, root: Path) -> None:
        self.root, self.attempts, self.records, self.labelled, self.fetched, self.pulled = root, {}, {}, {}, [], []

    def add(self, run: str, record: Path | None, *, rc: int = 0, dirty: bool = False) -> None:
        outputs = {} if record is None else {"run_record": f"art:{run}"}
        self.attempts[run] = {"source": {"commit": COMMIT, "dirty": dirty}, "status": {"rc": rc}, "outputs": outputs}
        self.records[f"art:{run}"] = record

    def withdraw(self, target: str, value: str, by: str = "infra") -> None:
        self.labelled.setdefault(target, []).append(SimpleNamespace(key="withdrawn", value=value, by=by))

    def pull(self, store: FakeStore, targets: list[str], warn: Callable[[str], None]) -> None:
        assert store is self
        self.pulled.append(targets)

    def get_attempt(self, run: str) -> dict | None:
        return self.attempts.get(run)

    def labels(self, target: str, key: str | None = None) -> list[SimpleNamespace]:
        assert any(target in targets for targets in self.pulled), f"{target}'s labels were read before they were pulled"
        return [lab for lab in self.labelled.get(target, []) if key in (None, lab.key)]

    def _held(self, art: str) -> list[str]:
        record = self.records[art]
        return sorted(p.relative_to(record).as_posix() for p in record.rglob("*") if p.is_file())

    def listing(self, art: str, paths: list[str] | None = None) -> list[dict]:
        return [{"path": n} for n in self._held(art) if paths is None or n in paths]

    def fetch(self, art: str, *, paths: list[str]) -> Path:
        self.fetched.append(list(paths))
        wanted = [row["path"] for row in self.listing(art, paths)]
        if not wanted:
            raise Refused(f"{art}: no file matches {paths}")
        to = self.root / f"fetched-{len(self.fetched)}"
        for name in wanted:
            (to / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.records[art] / name, to / name)
        return to


@pytest.fixture
def store(tmp_path: Path, monkeypatch) -> FakeStore:
    """A store with no `research run` around the command."""
    monkeypatch.delenv("OUT", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    return FakeStore(tmp_path / "store")


def bench(store: FakeStore, *argv: str) -> int:
    return C.main(list(argv), store_of=lambda: C.EvidenceStore(store, store.pull, Refused, "withdrawn"))


def no_number(capsys, why: str, said: str | None = None) -> dict:
    """What the command printed with no number: the kit's ``compute: no number: WHY`` (`why` its start), the coins
    the record states (`said`, where given, is that line), that this is no audit, and its document, with nothing on
    stderr. A record that doesn't say who flipped its coins has the line naming what it does state at the end."""
    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert err == "" and lines[0].startswith(f"compute: no number: {why}"), (out, err)
    assert lines[1].startswith("ran with: ") and lines[2] == "a measurement, not an audit"
    assert said is None or lines[1] == said
    doc = json.loads(lines[-1])
    up = [C.kit.kit_report.MADE_UP, C.made_up_text(doc["made_up_says"])] if doc.get("made_up") else []
    rest = lines[3:-1]
    assert rest[:len(up)] == up
    stated = rest[len(up):]
    assert stated == [] if lines[1] != "ran with: not stated" else \
        len(stated) <= 1 and all(x.startswith("  coins as the record states them") for x in stated)
    assert doc["value"] is None and doc["no_number"] == lines[0].removeprefix("compute: no number: ")
    assert doc["audit"] is False
    return doc


def printed(capsys) -> tuple[list[str], dict | None, str]:
    """What the command printed: its text lines, its JSON document (None with nothing printed) and its stderr."""
    out, err = capsys.readouterr()
    lines = out.splitlines()
    return lines[:-1], json.loads(lines[-1]) if lines else None, err


def counted(shape: C.Shape, slots: int, prompt: int, outputs: int) -> dict[str, int]:
    """The MACs of `slots` slots served step by step, counted one position, one query-key pair and one row of logits
    at a time: a slot's first step runs its prompt and each later step the output before it, and every step emits one
    output."""
    d, head_dim = shape.hidden, shape.hidden // shape.heads
    kv = shape.kv_heads * head_dim
    # each layer's q, k, v, o, gate, up and down, as (inputs, outputs)
    matrices = [(d, d), (d, kv), (d, kv), (d, d), (d, shape.mlp), (d, shape.mlp), (shape.mlp, d)]
    macs = {"linears": 0, "attention": 0, "lm_head": 0}
    for _ in range(slots):
        keys = 0
        for step in range(outputs):
            for _ in range(prompt if step == 0 else 1):
                keys += 1
                macs["linears"] += shape.layers * sum(i * o for i, o in matrices)
                for _key in range(keys):
                    # a head's score against the key, then the key's value scaled into the head's output
                    macs["attention"] += shape.layers * shape.heads * (head_dim + head_dim)
            macs["lm_head"] += d * shape.vocab
    return macs


def test_an_accepted_run_prints_the_share_of_the_models_macs_as_text_and_as_json(store, tmp_path, capsys):
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN) == 0
    text, doc, err = printed(capsys)
    assert err == ""
    assert text == [
        ("compute: 95.66% of the served model's MACs (134,502,518,410 of 140,598,837,248), below the 97% target. This "
         "is what the Pearl+ verifier's own answer credits, not PoUW's verdict in the run's record of verdicts"),
        RAN_WITH,
        "a measurement, not an audit",
        *MADE_UP,
        (f"  {RUN}, commit {COMMIT}: Pearl+'s answer accepts (scheme pearl-c-h100-v1-h3) at ε_s 1/100, δ_s 1/100 "
         f"({C.SAMPLING_ALONE}), and the Lean verifier's verdict is the same"),
        C.RESTS_ON,
        (f"  PoUW's verdict in the run's record of verdicts is `missing`, not an accept: {NOTHING} (the record's "
         f"reason for PoUW's verdict: {WHY})"),
        (f"  unsloth/Llama-3.2-1B at revision {REVISION}, 4 slots of 32 prompt and 4 output tokens, the benchmark's load "
         "(load.toml): linears 136,230,993,920, attention 165,150,720 and LM head 4,202,692,608 MACs, which is the "
         "policy's own count of its class"),
        ("  Pearl+ credits linears alone, 96.89% of the model's MACs at this class: its passes hold 136,201,633,792 of "
         "them and its credit is 98.73% of them all"),
        ("  tokens, the class's since the record holds no count of those served: 128 prompt and 16 output, over 140 "
         "positions"),
        f"  capacity share: {C.ABSENT}",
    ]
    assert list(doc)[:4] == ["value", "unit", "commit", "run"]
    assert (doc["value"], doc["unit"], doc["commit"], doc["run"]) == (SHARE, C.UNIT, COMMIT, RUN)
    assert (doc["credited_macs"], doc["model_macs"], doc["macs"], doc["passes_macs"]) == (CREDITED, TOTAL, MACS, PASSES)
    assert doc["audit"] is False and doc["ran_with"]["coins"] == "in-process" and doc["made_up"] is True
    assert doc["made_up_says"] == SAYS
    assert (doc["target"], doc["met"], doc["lean"], doc["policy_agrees"]) == (0.97, False, True, True)
    assert doc["record"] == {"verdict": "missing", "why": WHY}
    assert C.RESTS_ON == ("  the credit rests on Pearl's conjecture TT_OUT rev1, which this repository's red-team rates "
                          "refuted: that a credited MAC was computed is an open assumption, not a proved guarantee")
    # the answer's ε and δ are sampling's, and the document names them so: it has no `epsilon` or `delta` of the credit
    assert (doc["epsilon_s"], doc["delta_s"]) == ("1/100", "1/100") and not {"epsilon", "delta"} & set(doc)
    assert doc["withdrawn"] == [] and store.pulled == [[RUN, f"art:{RUN}"]]
    assert doc["tokens"] == {"prompt": 128, "output": 16, "positions": 140} and doc["tokens_from"] == "class"
    assert doc["wall_s"] is None and doc["capacity_share"] is None


def test_the_models_macs_are_the_formulas(tmp_path):
    """Held to hand counts: a position's linears, a query-key pair and a token's logits on Llama 3.2 1B, the fixture's
    class, and a shape small enough to count on paper."""
    assert C.model_macs(LLAMA, 1, 1, 1) == {"linears": 973_078_528, "attention": 65_536, "lm_head": 262_668_288}
    assert C.model_macs(LLAMA, 4, 32, 4) == MACS and sum(MACS.values()) == TOTAL
    # one layer of q, o (4 x 4), k, v (4 x 2) and gate, up, down (4 x 8) is 144 MACs a position; 3 positions have 6
    # causal pairs of 2 heads x 2 x 2 MACs; 2 outputs have 4 x 10 logits each
    small = C.Shape(layers=1, hidden=4, heads=2, kv_heads=1, mlp=8, vocab=10)
    assert C.model_macs(small, 1, 2, 2) == {"linears": 432, "attention": 48, "lm_head": 80}


@pytest.mark.parametrize("slots, prompt, outputs", [(1, 1, 1), (1, 5, 1), (1, 1, 6), (4, 32, 4), (3, 17, 9)])
def test_the_models_macs_are_a_step_by_step_count_of_the_served_slots(slots, prompt, outputs):
    for shape in (LLAMA, C.Shape(layers=3, hidden=12, heads=4, kv_heads=2, mlp=20, vocab=7)):
        assert C.model_macs(shape, slots, prompt, outputs) == counted(shape, slots, prompt, outputs)


def test_the_number_is_held_to_the_target_exactly(store, tmp_path, capsys, monkeypatch):
    """A share that rounds to the target and is under it is below it."""
    monkeypatch.setattr(C, "TARGET", Fraction(CREDITED, TOTAL))
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert ", at or above the 95.664% target. " in text[0] and doc["met"] is True and doc["target"] == SHARE
    under = tmp_path / "under"
    under.mkdir()
    one_less = a_record(under, ANSWER=to("public", "work", "credited", value=CREDITED - 1),
                        LEAN=to("work", "credited", value=CREDITED - 1))
    store.add(RUN, one_less)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[0].startswith("compute: 95.66% ") and ", below the 95.664% target. " in text[0] and doc["met"] is False


@pytest.mark.parametrize("verdicts, said", [
    (to("protocols", "work", value={"verdict": "accept"}), "  PoUW's verdict in the run's record of verdicts accepts too"),
    (to("protocols", "work", value={"verdict": "reject", "reason": "a sampled tile is wrong"}),
     (f"  PoUW's verdict in the run's record of verdicts is `reject`, not an accept: {NOTHING} (the record's reason "
      "for PoUW's verdict: a sampled tile is wrong)")),
    (to("protocols", value={}), f"  PoUW's verdict in the run's record of verdicts is absent: {NOTHING}"),
    (None, f"  PoUW's verdict in the run's record of verdicts is absent: {NOTHING}"),
])
def test_pouws_row_in_the_composed_record_is_printed_whatever_it_says(store, tmp_path, capsys, verdicts, said):
    store.add(RUN, a_record(tmp_path, VERDICTS=verdicts))
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[7] == said and doc["value"] == SHARE


@pytest.mark.parametrize("edits, why", [
    ({"ANSWER": to("verdict", value="reject"), "LEAN": to("verdict", value="reject")},
     "Pearl+'s answer is reject, not accept (no reason recorded)"),
    ({"ANSWER": lambda d: d.update(verdict="reject", reason="tile 7 of pass 0 is wrong")},
     "Pearl+'s answer is reject, not accept (tile 7 of pass 0 is wrong)"),
    ({"ANSWER": to("protocol", value="verity/pouw/other/v1")},
     "the answer is verity/pouw/other/v1's, not verity/pouw/pearl-plus/v1's"),
    ({"ANSWER": to("public", "work", value={})}, "Pearl+'s answer accepts and holds no credit in MACs"),
    ({"ANSWER": to("public", "work", "credited", value=True)}, "Pearl+'s answer accepts and holds no credit in MACs"),
    ({"ANSWER": to("public", "work", "credited", value=-1)}, "Pearl+'s answer accepts and holds no credit in MACs"),
    ({"ANSWER": "not json"}, "the run's Pearl+ verdicts don't read as verdicts"),
    ({"ANSWER": "[]"}, "the run's Pearl+ verdicts don't read as verdicts"),
    ({"LEAN": to("verdict", value="reject")}, "the Lean verifier's verdict is reject and the answer's accept"),
    ({"LEAN": to("work", "credited", value=CREDITED + 1)},
     f"the Lean verifier's credit is {CREDITED + 1} and the answer's {CREDITED}"),
    ({"LEAN": to("commitment", value="d1" * 64)},
     "the Lean verifier's run commitment is d1d1d1d1d1d1d1d1 and the answer's c0c0c0c0c0c0c0c0"),
    ({"LEAN": "not json"}, "the run's Pearl+ verdicts don't read as verdicts"),
    ({"LEAN_CHECK": to("differ", value=["work"])},
     "the record's comparison of the Lean verifier's verdict with the answer differs on ['work']"),
    ({"LEAN_CHECK": lambda d: d.pop("differ")}, "the run's Pearl+ verdicts don't read as verdicts"),
    ({"ANSWER": lambda d: (d["profile"].update(test_instance=True), d["public"]["work"].update(credited=0)),
      "LEAN": to("work", "credited", value=0)},
     "Pearl+'s answer is a test instance's (its profile's test_instance): its passes are not the served instance's"),
    ({"ANSWER": to("profile", "test_instance", value=True)}, "Pearl+'s answer is a test instance's"),
])
def test_an_answer_that_is_not_an_accept_the_lean_verifier_shares_prints_no_number_and_exits_1(store, tmp_path, capsys,
                                                                                              edits, why):
    store.add(RUN, a_record(tmp_path, **edits))
    assert bench(store, "--from", RUN) == 1
    no_number(capsys, why)


@pytest.mark.parametrize("launcher, said", [(None, RAN_WITH),
                                            ("local", RAN_WITH.replace("coins in-process", "coins local challenger"))])
def test_a_refused_record_still_says_the_coins_it_states(store, tmp_path, capsys, launcher, said):
    """With no number the line under the refusal is still who flipped the coins as the record states it, never
    ``not stated`` where the record says: the fixture's in-process coins, or the launcher's local challenger."""
    edits = {"LEAN": None} | ({} if launcher is None else {"RAN": ran(challenger=launcher, development=True)})
    record = a_record(tmp_path, **edits)
    store.add(RUN, record)
    for argv in (["--from", RUN], ["--from", str(record)]):
        assert bench(store, *argv) == 1
        no_number(capsys, f"the record holds no {C.LEAN}", said)


@pytest.mark.parametrize("edits, missing", [({"LEAN": None}, C.LEAN), ({"LEAN_CHECK": None}, C.LEAN_CHECK),
                                             ({"LEAN": None, "LEAN_CHECK": None}, C.LEAN)])
def test_a_record_with_no_lean_verdict_prints_no_number_and_names_the_file(store, tmp_path, capsys, edits, missing):
    """The answer's credit is no number without the Lean verifier's verdict and the record's comparison of the two."""
    record = a_record(tmp_path, **edits)
    store.add(RUN, record)
    for argv in (["--from", RUN], ["--from", str(record)]):
        assert bench(store, *argv) == 1
        no_number(capsys, f"the record holds no {missing}, so no Lean verdict")


def test_a_dirty_commit_is_printed_as_one(store, tmp_path, capsys):
    store.add(RUN, a_record(tmp_path), dirty=True)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[5] == (f"  {RUN}, commit {COMMIT}-dirty: Pearl+'s answer accepts (scheme pearl-c-h100-v1-h3) at ε_s 1/100, "
                       f"δ_s 1/100 ({C.SAMPLING_ALONE}), and the Lean verifier's verdict is the same")
    assert doc["lean"] is True and doc["value"] == SHARE and doc["commit"] == f"{COMMIT}-dirty"


@pytest.mark.parametrize("edits", [{"ANSWER": None}, dict.fromkeys(("ANSWER", "LEAN", "LEAN_CHECK", "POLICY", "VERDICTS"))])
def test_a_run_that_left_no_answer_prints_no_number_and_exits_1(store, tmp_path, capsys, edits):
    record = a_record(tmp_path, **edits)
    (record / "out/result.json").write_text("{}")
    store.add(RUN, record, rc=3)
    assert bench(store, "--from", RUN) == 1
    assert no_number(capsys, f"{RUN} left no Pearl+ answer (it ended rc 3)")["run"] == RUN


def test_a_run_that_is_not_in_the_store_or_kept_no_record_exits_2(store, capsys):
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"compute: {RUN} is not in the evidence store" in err
    store.add(RUN, None)
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"compute: {RUN} kept no record" in err


def test_a_store_that_cannot_be_read_exits_2(store, tmp_path, capsys, monkeypatch):
    store.add(RUN, a_record(tmp_path))

    def refuse(*args, **kwargs):
        raise Refused("the remote doesn't answer")
    monkeypatch.setattr(store, "fetch", refuse)
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"compute: {RUN}'s record can't be fetched: the remote doesn't answer" in err
    monkeypatch.setattr(store, "get_attempt", refuse)
    assert bench(store, "--from", RUN) == 2
    assert "compute: the evidence store can't be read: the remote doesn't answer" in capsys.readouterr().err

    def absent():
        raise C.Unreadable("the evidence store's client isn't installed here (No module named 'research'): give --from "
                           "the run's record directory, or install research (uv sync --all-packages)")
    assert C.main(["--from", RUN], store_of=absent) == 2
    assert "compute: the evidence store's client isn't installed here" in capsys.readouterr().err


OTHERS = "out/pieces/pouw/coin-put.json, out/pieces/pouw/period.json"


@pytest.mark.parametrize("target", [f"art:{RUN}", RUN])
@pytest.mark.parametrize("value", [
    "private values were published",                                   # the whole record
    f"private values were published (only {C.ANSWER})",                # a file read here
    "private values were published (only out/pieces)",                 # a directory holding files read here
    f"private values were published (only {OTHERS}, out/record/)",     # one of its prefixes holds a file read here
    "private values were published (only the sealed bundle)",          # a reason that ends like a scope and names no file
    "private values were published (only *)",
    f"(only {OTHERS}) and the rest",
])
def test_a_record_withdrawn_whole_or_where_it_is_read_is_refused_before_anything_is_fetched(store, tmp_path, capsys,
                                                                                         target, value):
    record = a_record(tmp_path)
    for name in OTHERS.split(", "):
        (record / name).write_text("{}")
    store.add(RUN, record)
    store.withdraw(target, value)
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"compute: {RUN} was withdrawn ({value!r}, by infra): nothing of it is fetched" in err
    assert store.fetched == []


def test_a_withdrawal_of_other_files_of_the_record_is_printed_beside_the_number(store, tmp_path, capsys):
    record = a_record(tmp_path)
    for name in OTHERS.split(", "):
        (record / name).write_text("{}")
    store.add(RUN, record)
    store.withdraw(f"art:{RUN}", f"the private value of a coin can be recovered (only {OTHERS})")
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    said = f"'the private value of a coin can be recovered (only {OTHERS})', by infra"
    assert doc["withdrawn"] == [said] and doc["value"] == SHARE
    assert text[8] == f"  part of the run's record was withdrawn, none of it a file read here: {said}"
    assert store.fetched == [[*C.RECORDS, C.VERIFY]] and store.pulled == [[RUN, f"art:{RUN}"]]
    # a later withdrawal of the whole record refuses it
    store.withdraw(f"art:{RUN}", "the rest holds private values too")
    assert bench(store, "--from", RUN) == 2
    assert "'the rest holds private values too', by infra" in capsys.readouterr().err


def test_the_command_line_names_a_run(store, capsys):
    with pytest.raises(SystemExit) as e:
        bench(store)
    assert e.value.code == 2 and "--from" in capsys.readouterr().err


@pytest.mark.parametrize("edits, why", [
    ({"POLICY": None}, "the run's record holds no inference-only policy, so no class to count the model's MACs at"),
    ({"POLICY": "not json"}, "the run's inference-only policy doesn't read as one"),
    ({"POLICY": lambda d: d.pop("class")}, "the run's inference-only policy doesn't read as one (KeyError: 'class')"),
    ({"POLICY": to("serve", "model", value="some/other-model")},
     "no shape for the served model 'some/other-model': add it to MODELS in run.py (it holds unsloth/Llama-3.2-1B)"),
    ({"POLICY": to("class", "slots", value=0)},
     "the policy's class is not slots, prompt tokens and output tokens, each at least 1"),
    ({"POLICY": lambda d: d["class"].pop("outputs")},
     "the policy's class is not slots, prompt tokens and output tokens, each at least 1"),
    ({"POLICY": to("class", "positions", value=36)},
     "the policy's class runs 36 positions a slot, not the 35 of 32 prompt and 4 output tokens"),
    ({"POLICY": to("n_inputs", "work", "lm_head", value=1)},
     ("the policy counts its class's MACs as {'linears': 136230993920, 'attention': 165150720, 'lm_head': 4} and "
      "run.py as {'linears': 136230993920, 'attention': 165150720, 'lm_head': 4202692608}")),
    ({"POLICY": lambda d: (d["class"].update(slots=1, prompt=90, outputs=1, positions=90), d.pop("n_inputs"))},
     (f"the record's policy registers unsloth/Llama-3.2-1B at revision {REVISION}, class 1,90,1 and the benchmark's "
      f"load is unsloth/Llama-3.2-1B at revision {REVISION}, class 4,32,4 (load.toml; class is slots, prompt tokens "
      "and output tokens a request): the number is of that load alone")),
    ({"POLICY": to("serve", "revision", value="0" * 40)},
     f"the record's policy registers unsloth/Llama-3.2-1B at revision {'0' * 40}, class 4,32,4 and the benchmark's"),
    ({"POLICY": lambda d: d["serve"].pop("revision")},
     "the record's policy registers unsloth/Llama-3.2-1B at revision None, class 4,32,4 and the benchmark's"),
    ({"VERDICTS": "[]"}, "the run's composed record doesn't read as one"),
    ({"ANSWER": to("public", "work", "credited", value=MACS["linears"] + 1),
      "LEAN": to("work", "credited", value=MACS["linears"] + 1)},
     ("Pearl+'s answer holds 136,230,993,921 MACs, more than the 136,230,993,920 of the linears of the policy's class: "
      "its passes are not one load of the class")),
    ({"ANSWER": to("public", "work", "macs", value=2 * MACS["linears"])},
     "Pearl+'s answer holds 272,461,987,840 MACs, more than the 136,230,993,920 of the linears of the policy's class"),
])
def test_a_record_that_cannot_give_the_models_macs_at_one_load_of_its_class_exits_2(store, tmp_path, capsys, edits, why):
    store.add(RUN, a_record(tmp_path, **edits))
    assert bench(store, "--from", RUN) == 2
    out, err = capsys.readouterr()
    assert out == "" and f"compute: {why}" in err and "no number" not in err


def test_a_policy_that_counts_no_macs_of_its_own_still_gives_the_number(store, tmp_path, capsys):
    store.add(RUN, a_record(tmp_path, POLICY=lambda d: (d.pop("n_inputs"), d["class"].pop("positions"))))
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert doc["policy_agrees"] is None and doc["value"] == SHARE and text[8].endswith("LM head 4,202,692,608 MACs")


def test_nothing_the_developers_node_reports_of_what_it_served_is_read(store, tmp_path, capsys):
    """The run's record also holds the Pearl+ piece's own document, with the tokens and seconds the node says it
    served: the command reads its salt coins' source alone, asks the store for no other file of the piece's, and prints
    no capacity share."""
    record = a_record(tmp_path)
    (record / C.DEVELOPMENT).write_text(json.dumps(
        {"verdict": "accept", "coin_source": "in-process", "served_tokens": 16, "generate_s": 2.0,
         "credit": {"credited": 1}}))
    (record / "out/pieces/pouw/served.json").write_text(json.dumps({"served_tokens": 16, "generate_s": 2.0}))
    store.add(RUN, record)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert store.fetched == [[*C.RECORDS, C.VERIFY]] and not any("served" in g or "*" in g for g in [*C.RECORDS, C.VERIFY])
    assert doc["tokens"] == {"prompt": 128, "output": 16, "positions": 140} and doc["coins"]["salt"] == "in-process"
    assert doc["wall_s"] is None and doc["capacity_share"] is None and text[-1] == f"  capacity share: {C.ABSENT}"
    assert doc["credited_macs"] == CREDITED


#: the bench run's own id and host stamp, which the kit's writer stamps each measurement with
BENCH = "r20000101-000001-bbbb"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}


@pytest.fixture
def stamped(monkeypatch):
    """The kit's writer under a `research run` of its own, stamping with `STAMP` rather than asking the host, and with
    no run directory, so it writes `$OUT/result.json` and never the result of a run the suite itself runs under."""
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", BENCH)
    monkeypatch.setattr(C.kit, "host_stamp", lambda: dict(STAMP))


def test_under_a_research_run_the_printed_document_is_also_its_output(store, tmp_path, capsys, monkeypatch, stamped):
    store.add(RUN, a_record(tmp_path, **REAL))
    assert bench(store, "--from", RUN) == 0
    alone = capsys.readouterr()
    out = tmp_path / "run" / "out"
    monkeypatch.setenv("OUT", str(out))
    assert bench(store, "--from", RUN) == 0
    # what it prints is the same to the byte with the shared result written beside it
    assert capsys.readouterr() == alone
    doc = json.loads(alone.out.splitlines()[-1])
    assert json.loads((out / "compute.json").read_text()) == doc
    assert list(doc) == ["value", "unit", "commit", "run", "target", "met", "credited_macs", "model_macs", "macs",
                         "linears_share", "passes_macs", "scheme", "epsilon_s", "delta_s", "lean", "coins", "record",
                         "withdrawn", "model", "revision", "class", "policy_agrees", "tokens", "tokens_from", "wall_s",
                         "capacity_share", "made_up_says", "ran_with", "audit"]
    # with no number the kit writes the document with no value and why, and the metric alone, with no value and why
    none = tmp_path / "none" / "out"
    monkeypatch.setenv("OUT", str(none))
    store.add(RUN, a_record(tmp_path / "run", **REAL, ANSWER=to("verdict", value="reject")))
    assert bench(store, "--from", RUN) == 1
    doc = no_number(capsys, "Pearl+'s answer is reject")
    assert json.loads((none / "compute.json").read_text()) == doc
    got = json.loads((none / "result.json").read_text())["measurements"]
    assert [(m["name"], m["value"], m["why"], m["audit"]) for m in got] == [(C.METRIC, None, doc["no_number"], False)]


def test_the_shared_result_is_the_metric_by_its_id_verified_and_error_with_capacity_and_free_absent(
        store, tmp_path, capsys, monkeypatch, stamped):
    """The record holds no registration of the run's hardware and no span the verifier holds its passes to, so
    capacity and free are written with no value, each with why, and no reading is named."""
    out = tmp_path / "out"
    monkeypatch.setenv("OUT", str(out))
    store.add(RUN, a_record(tmp_path, **REAL))
    assert bench(store, "--from", RUN) == 0
    result = json.loads((out / "result.json").read_text())
    assert (result["schema"], result["run_id"]) == ("research/result/v0.1", BENCH)
    got = {m["name"]: m for m in result["measurements"]}
    assert list(got) == [C.METRIC, C.CREDITABLE, C.OF_CREDITABLE, "verified", "capacity", "free", "error"]
    assert C.METRIC == "compute-accounting/verified-compute-share"
    # the number is the product of the two shares the text prints beside it
    assert (got[C.CREDITABLE]["value"], got[C.OF_CREDITABLE]["value"]) == (MACS["linears"] / TOTAL, CREDITED / MACS["linears"])
    assert got[C.CREDITABLE]["unit"] == got[C.OF_CREDITABLE]["unit"] == C.SHARE
    assert (got[C.METRIC]["value"], got[C.METRIC]["unit"], got[C.METRIC]["target"], got[C.METRIC]["verdict"]) == (
        SHARE, C.UNIT, 0.97, "below target")
    assert (got["verified"]["value"], got["verified"]["unit"]) == (CREDITED, "MAC")
    why = f"absent: {C.NO_SITE}; and {C.NO_SPAN}"
    for name in ("capacity", "free"):
        assert (got[name]["value"], got[name]["unit"], got[name]["why"]) == (None, "MAC", why)
    # the error is the answer's δ_s, named in its inputs, and its unit says it is sampling's alone
    assert (got["error"]["value"], got["error"]["unit"]) == (0.01, C.DELTA_S) and "why" not in got["error"]
    assert "the answer's δ_s, sampling's alone" in C.DELTA_S
    for m in got.values():
        assert (m["run"], m["commit"], m["host"]) == (BENCH, STAMP["commit"], STAMP)
        assert m["mark"] == "measured" and m["inputs"] == ["measured", RUN, *["delta_s"] * (m["name"] == "error")]
        assert m["audit"] is False
    assert C.NO_SPAN == ("a Pearl+ answer holds its passes to no seconds, and the seconds the run served in stay on the "
                         "developer's machine")


def h200_site(devices: int):
    """A registered site of one host with `devices` H200s, each naming the sheet its card file gives."""
    from verity import device as S
    sheet = S.Sheet.of_file(S.load_card("h200-sxm-141gb"))
    locs = [S.Loc("site", "site"), S.Loc("host", "host", "site")]
    locs += [S.Loc(f"gpu{i}", "device", "host", sheet=sheet.digest()) for i in range(devices)]
    return S.Site((sheet,), tuple(locs))


def test_capacity_is_the_registered_devices_peak_at_each_data_sheet_reading_times_the_span(store, tmp_path):
    """The H200's card file gives two readings, its tensor clock (1830 MHz) and its SM clock (1980 MHz): two devices
    over 600 s are 2 x 132 SMs x 4,096 e4m3 MACs a clock at each, free is capacity less the credit, and a span that
    leaves a fraction of a MAC rounds capacity up."""
    store.add(RUN, a_record(tmp_path))
    doc = C.from_run(C.EvidenceStore(store, store.pull, Refused, "withdrawn"), RUN)
    tensor, sm = 2 * 132 * 4096 * 1_830_000_000 * 600, 2 * 132 * 4096 * 1_980_000_000 * 600
    assert C.capacity(h200_site(2), Fraction(600)) == {"tensor": tensor, "sm": sm}
    assert C.capacity(h200_site(1), Fraction(1, 10 ** 9)) == {"tensor": 989_430, "sm": 1_070_531}
    got = {m["name"]: m for m in C.amounts(doc, h200_site(2), Fraction(600))}
    assert list(got) == [C.METRIC, C.CREDITABLE, C.OF_CREDITABLE, "verified", "capacity at clock.tensor",
                         "free at clock.tensor", "capacity at clock.sm", "free at clock.sm", "error"]
    assert [got[n]["value"] for n in got if n.startswith(("capacity", "free"))] == [
        tensor, tensor - CREDITED, sm, sm - CREDITED]
    assert not any("why" in m for m in got.values())
    # with either missing, capacity and free are absent and say which
    assert {m["name"]: m.get("why") for m in C.amounts(doc, h200_site(2))}["capacity"] == f"absent: {C.NO_SPAN}"
    assert {m["name"]: m.get("why") for m in C.amounts(doc, None, Fraction(600))}["free"] == f"absent: {C.NO_SITE}"


def test_capacity_refuses_hardware_it_cannot_read_a_peak_off(tmp_path):
    from verity import device as S
    with pytest.raises(C.Unreadable, match="the registered hardware holds no device"):
        C.capacity(S.Site(), Fraction(1))
    one = h200_site(1)
    unnamed = S.Site(one.sheets, (*one.locs[:2], S.Loc("gpu0", "device", "host", sheet="00" * 64)))
    with pytest.raises(C.Unreadable, match="the registered device gpu0 names no registered sheet"):
        C.capacity(unnamed, Fraction(1))
    other = S.Sheet(S.DeviceFigures("no-such-card", Fraction(1), Fraction(1)), 1, 1, 1)
    unknown = S.Site((other,), (*one.locs[:2], S.Loc("gpu0", "device", "host", sheet=other.digest())))
    with pytest.raises(C.Unreadable, match="the registered device gpu0's SKU has no card file"):
        C.capacity(unknown, Fraction(1))


def test_an_answer_with_no_delta_writes_the_error_absent_and_says_why(store, tmp_path):
    store.add(RUN, a_record(tmp_path, ANSWER=lambda d: d.pop("profile")))
    doc = C.from_run(C.EvidenceStore(store, store.pull, Refused, "withdrawn"), RUN)
    got = {m["name"]: m for m in C.amounts(doc)}["error"]
    assert (got["value"], got["why"], got["inputs"]) == (None, "the answer states no δ_s", [RUN, "delta_s"])


#: the fixture's give-up, TOTAL less CREDITED, by cause: the verify document's split holds every linear of the class
#: (4 passes, the prefill's and 3 decode steps', of 64 calls each), puts 2 excluded A rows in pass 0's call 0 (qkv)
#: and 1 in pass 1's call 63 (down), and ε's deduction and ι on every call
GIVE_UP = {"lm_head": 4_202_692_608, "attention": 165_150_720, "outside": 0, "excluded_a": 29_360_128,
           "excluded_b": 0, "voluntary": 0, "kept_width": 0, "epsilon_macs": Fraction(34_050_408_448, 25),
           "over_bound_macs": 0, "rho_macs": Fraction(210_686_902_272, 625), "floor": Fraction(278, 625)}


def resum(doc: dict) -> None:
    """Restate a verify document's split totals as its calls' sums, as the verifier states them."""
    got, sums = doc["uncredited"], {}
    for c in got["calls"]:
        for x, v in (("macs", c["macs"]), ("checked_macs", c["checked_macs"]), *c["uncredited_macs"].items(),
                     *((x, Fraction(c[x])) for x in C.TERMS)):
            sums[x] = sums.get(x, 0) + v
    got["totals"] = {**{x: str(v) if isinstance(v, Fraction) else v for x, v in sums.items()},
                     "credited": int(sums["credited_macs"])}


def moved(n: Fraction) -> Callable[[dict], None]:
    """An edit that moves `n` MACs of pass 0's call 3's ε deduction to its credit, ι taking its share, and restates the
    totals."""
    def edit(doc: dict) -> None:
        c = doc["uncredited"]["calls"][3]
        c["epsilon_macs"] = str(Fraction(c["epsilon_macs"]) - n)
        c["rho_macs"] = str(Fraction(c["rho_macs"]) + Fraction(n, 400))
        c["credited_macs"] = str(Fraction(c["credited_macs"]) + Fraction(399 * n, 400))
        resum(doc)
    return edit


def test_per_call_prints_the_give_up_by_cause_and_the_calls_that_give_up_the_most(store, tmp_path, capsys):
    """`--per-call` prints the split in the verify document beside the answer (which the number reads too, for the LM
    head's calls): the causes sum to the model's MACs less the credit exactly, and the calls are ordered by what each
    gives up, named by their linear where the model's shape has one."""
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN) == 0 and "per_call" not in printed(capsys)[1]
    assert store.fetched == [[*C.RECORDS, C.VERIFY]]
    assert bench(store, "--from", RUN, "--per-call", "--top", "2") == 0
    text, doc, err = printed(capsys)
    assert err == "" and store.fetched[-1] == [*C.RECORDS, C.VERIFY] and doc["value"] == SHARE
    got = doc["per_call"]
    assert got["give_up"] == TOTAL - CREDITED == sum(GIVE_UP.values()) and got["calls"] == 256 and got["source"] == C.VERIFY
    assert [(c["cause"], Fraction(c["macs"])) for c in got["causes"]] == list(GIVE_UP.items())
    assert sum(Fraction(c["macs"]) for c in got["causes"]) == got["give_up"]
    assert [(c["pass"], c["call"], c["linear"]) for c in got["top"]] == [(0, 2, "gate_up"), (0, 6, "gate_up")]
    assert got["top"][1]["causes"] == {"epsilon_macs": "1073741824/25", "rho_macs": "6643777536/625"}
    assert text[12:] == [
        ("  per call: the model gives up 6,096,318,838 of its MACs (4.34%); by cause, from the verifier's split of "
         "its 256 calls (out/pieces/pouw/verify.json):"),
        "    LM head, which Pearl+ doesn't credit         4,202,692,608 MACs   68.94%",
        "    attention, which Pearl+ doesn't credit         165,150,720 MACs    2.71%",
        "    linears outside the passes                               0 MACs    0.00%",
        "    excluded A rows                                 29,360,128 MACs    0.48%",
        "    excluded B rows' column blocks                           0 MACs    0.00%",
        "    voluntary rows' bands and blocks                         0 MACs    0.00%",
        "    side channels (k beyond |K|)                             0 MACs    0.00%",
        "    ε's deduction                                1,362,016,338 MACs   22.34%",
        "    sampled tiles over the inert-work bound                  0 MACs    0.00%",
        "    ι, inert work: 1/400 of what ε leaves          337,099,044 MACs    5.53%",
        "    the credit's floor                                       0 MACs    0.00%",
        ("  ε's deduction is the run's, at its sample: the verifier takes it off the records whose cells hold the most "
         "MACs a ticket, so the calls it lands on carry it for the whole run"),
        "  the 2 calls that give up the most:",
        ("    pass 0 call 2, weight 2 gate_up, 128x2048x16384 with 128 real rows: 53,579,717 MACs "
         "(ε's deduction 42,949,673, ι 10,630,044)"),
        ("    pass 0 call 6, weight 6 gate_up, 128x2048x16384 with 128 real rows: 53,579,717 MACs "
         "(ε's deduction 42,949,673, ι 10,630,044)"),
    ]
    assert bench(store, "--from", RUN, "--per-call", "--top", "256") == 0
    top = printed(capsys)[1]["per_call"]["top"]
    assert {(c["pass"], c["call"], c["linear"]): c["causes"]["excluded_a"] for c in top if "excluded_a" in c["causes"]} == {
        (0, 0, "qkv"): str(2 * 2048 * 3072), (1, 63, "down"): str(8192 * 2048)}


def a_call(real_rows: int, n: int, k: int, k_kept: int, cells: int, **uncredited: int) -> dict:
    """A per-call split's call as `uncredited_words` reads it: `cells` credited cells at |K| = `k_kept`, and its
    uncredited MACs by cause, kept_width the real words' side channels."""
    return {"call": 0, "real_rows": real_rows, "n": n, "k": k, "k_kept": k_kept, "checked_macs": cells * k_kept,
            "uncredited_macs": {**dict.fromkeys(C.ROWS, 0), **uncredited,
                                "kept_width": real_rows * n * (k - k_kept)}}


def test_the_uncredited_words_are_the_words_touched_counted_by_hand(store, tmp_path, capsys):
    """The fixture's split excludes 2 A rows of pass 0's call 0 (n = 3,072) and 1 of pass 1's call 63 (n = 2,048):
    2 x 3,072 + 2,048 words. On 4 real rows of 6 words, an excluded B row's block of 2 columns leaves 8 cells, its MACs over k; with one
    of k = 10 channels on the side, every one of the 24 real words holds an uncredited MAC, where the MACs over k are
    (8 x 9 + 24) / 10 = 9.6."""
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN, "--per-call") == 0
    assert printed(capsys)[1]["per_call"]["uncredited_words"] == 2 * 3_072 + 2_048 == 8_192
    assert C.uncredited_words(a_call(4, 6, 10, 10, 16, excluded_b=8 * 10)) == 8
    assert C.uncredited_words(a_call(4, 6, 10, 9, 16, excluded_b=8 * 9)) == 24
    assert C.uncredited_words(a_call(4, 6, 10, 10, 24)) == 0
    assert C.uncredited_words(a_call(4, 6, 10, 9, 24)) == 24
    with pytest.raises(C.Unreadable, match="call 0 holds checked MACs that are no whole number of cells at its kept"):
        C.uncredited_words({**a_call(4, 6, 10, 10, 16), "checked_macs": 161})


def test_the_uncredited_words_are_the_calls_own_sum_over_every_call(store, tmp_path, capsys):
    """The total is each call's words summed over every call of the split, not only the calls printed: with pass 1's
    call 63 given one side channel, in place of as many MACs of its excluded row (the passes hold every linear of the
    class, so there is no room for more), its 4 real rows of 2,048 words all count, and it is not the call printed."""
    def side(doc: dict) -> None:
        c = next(c for c in doc["uncredited"]["calls"] if (c["pass"], c["call"]) == (1, 63))
        width = c["real_rows"] * c["n"]
        c.update(k=c["k"] + 1, uncredited_macs={**c["uncredited_macs"], "kept_width": width,
                                                "excluded_a": c["uncredited_macs"]["excluded_a"] - width})
        resum(doc)
    store.add(RUN, a_record(tmp_path, VERIFY=side))
    assert bench(store, "--from", RUN, "--per-call", "--top", "1") == 0
    got = printed(capsys)[1]["per_call"]
    calls = json.loads((tmp_path / "record" / C.VERIFY).read_text())["uncredited"]["calls"]
    assert [(c["pass"], c["call"]) for c in got["top"]] == [(0, 2)]
    assert got["uncredited_words"] == sum(C.uncredited_words(c) for c in calls) == 2 * 3_072 + 4 * 2_048


@pytest.mark.parametrize("edits, why", [
    ({"VERIFY": None}, ("the record holds no per-call give-up: its verify document has no `uncredited`, which "
                        "`served_passes.py uncredited` writes from the run's passes and verify.json, and --uncredited F "
                        "reads")),
    ({"VERIFY": lambda d: d.pop("uncredited")}, "the record holds no per-call give-up: its verify document has no"),
    ({"VERIFY": lambda d: d["uncredited"].pop("totals")}, "the per-call split doesn't read as one (KeyError: 'totals')"),
    ({"VERIFY": to("commitment", value="c1" * 64)}, "the per-call split is of another run commitment than the answer's"),
    ({"VERIFY": lambda d: (d["uncredited"]["calls"][1].update(macs=d["uncredited"]["calls"][1]["macs"] + 1), resum(d))},
     "the per-call split's call 1 doesn't sum: its MACs are not its checked MACs and its causes"),
    ({"VERIFY": lambda d: d["uncredited"]["calls"][2].update(epsilon_macs="-1")},
     "the per-call split's call 2 holds a count that is no count of MACs"),
    ({"VERIFY": to("uncredited", "totals", "excluded_a", value=0)}, "the per-call split's totals are not its calls' sums"),
    ({"VERIFY": moved(400)}, (f"the per-call split's calls credit {CREDITED + 399:,} of {PASSES:,} MACs and the "
                              f"answer {CREDITED:,} of {PASSES:,}")),
    ({"VERIFY": lambda d: (d["uncredited"]["calls"][3].update(macs=d["uncredited"]["calls"][3]["macs"] + 1,
                                                              uncredited_macs={**d["uncredited"]["calls"][3]["uncredited_macs"],
                                                                               "kept_width": 1}), resum(d))},
     "the per-call split's calls hold 136,230,993,921 real MACs, more than the 136,230,993,920 of the class's linears"),
])
def test_a_per_call_split_that_is_not_the_answers_is_refused(store, tmp_path, capsys, edits, why):
    """The split is the answer's or `--per-call` prints nothing and exits 2; moving ε's MACs to the credit by less than
    the floor's room moves no number and is the answer's still."""
    store.add(RUN, a_record(tmp_path, **edits))
    assert bench(store, "--from", RUN, "--per-call") == 2
    out, err = capsys.readouterr()
    assert out == "" and f"compute: {why}" in err
    store.add(RUN, a_record(tmp_path / "again", VERIFY=moved(Fraction(1, 2))))
    assert bench(store, "--from", RUN, "--per-call") == 0
    floor = Fraction(278, 625) + Fraction(399, 800)
    assert printed(capsys)[1]["per_call"]["causes"][-1] == {"cause": "floor", "macs": str(floor),
                                                            "share": float(floor / (TOTAL - CREDITED))}


def test_per_call_reads_the_grid_form_of_an_answer_held_to_its_classs_fixed_load(store, tmp_path, capsys):
    """An answer held to its class's fixed load publishes the passes' whole MACs (the fixture's checked MACs and its
    excluded A rows') and their credit less the give-up padded up to the window grid. The split is that answer's where
    its credit lies between the published credit and that plus the give-up. The padding past the split's credit is the
    cause ``grid``, and the causes still sum to the give-up exactly. A split whose credit is past the give-up's room,
    or whose whole MACs are not the answer's, is refused."""
    whole, credited = PASSES + 29_360_128, CREDITED - 1_000_000

    def held(give_up: int = 4_194_304, macs: int = whole) -> dict:
        work = {"credited": credited, "macs": macs, "epsilon": "1/100", "give_up": give_up}
        return {"ANSWER": to("public", "work", value=work), "LEAN": to("work", value=work)}

    store.add(RUN, a_record(tmp_path, **held()))
    assert bench(store, "--from", RUN, "--per-call") == 0
    got = printed(capsys)[1]["per_call"]
    causes = {c["cause"]: Fraction(c["macs"]) for c in got["causes"]}
    assert causes == {**GIVE_UP, "grid": 1_000_000} and got["give_up"] == TOTAL - credited == sum(causes.values())
    for i, (give_up, macs) in enumerate([(999_999, whole), (4_194_304, whole - 1)]):
        store.add(RUN, a_record(tmp_path / str(i), **held(give_up, macs)))
        assert bench(store, "--from", RUN, "--per-call") == 2
        assert (f"compute: the per-call split's calls credit {CREDITED:,} of {whole:,} whole MACs and the answer's grid "
                f"form {credited:,} of {macs:,} less a give-up of {give_up:,}") in capsys.readouterr().err


def test_per_call_reads_a_split_in_a_file_of_its_own_for_a_record_that_holds_none(store, tmp_path, capsys):
    """``--uncredited F`` reads a split in a file of its own, with its commitment and its passes' records, in place of
    the verify document, which isn't fetched, and holds those records to the answer's where the answer names them."""
    split = json.loads((RECORD / C.VERIFY).read_text())["uncredited"]
    given = tmp_path / "uncredited.json"
    given.write_text(json.dumps({"schema": split["schema"], "verify": "verify.json", "commitment": "c0" * 64,
                                 "records": ["aa" * 64, "bb" * 64], "uncredited": split}))
    passes = [{"records": "aa" * 64}, {"records": "bb" * 64}]
    store.add(RUN, a_record(tmp_path, VERIFY=None, ANSWER=to("public", "passes", value=passes)))
    assert bench(store, "--from", RUN, "--per-call", "--uncredited", str(given)) == 0
    doc = printed(capsys)[1]
    assert store.fetched == [C.RECORDS] and doc["per_call"]["source"] == str(given)
    assert [Fraction(c["macs"]) for c in doc["per_call"]["causes"]] == list(GIVE_UP.values())
    store.add(RUN, a_record(tmp_path / "other", VERIFY=None, ANSWER=to("public", "passes", value=passes[::-1])))
    assert bench(store, "--from", RUN, "--per-call", "--uncredited", str(given)) == 2
    assert "compute: the per-call split's passes' records are not the answer's" in capsys.readouterr().err
    for argv in (["--uncredited", str(given)], ["--top", "3"]):
        with pytest.raises(SystemExit) as e:
            bench(store, "--from", RUN, *argv)
        assert e.value.code == 2 and "--uncredited and --top go with --per-call" in capsys.readouterr().err


def head_call(real_rows: int = 4, without: int | None = None) -> Callable[[dict], None]:
    """An edit that adds to a verify document's split one call of the LM head (weight 64, k = d, n = V) with `real_rows`
    real rows and one excluded vocabulary row, whose block of 64 columns credits nothing, drops pass `without`'s calls
    where given, and restates the totals."""
    def edit(doc: dict) -> None:
        doc["uncredited"]["calls"] = [c for c in doc["uncredited"]["calls"] if c["pass"] != without]
        k, n = LLAMA.hidden, LLAMA.vocab
        macs, out = real_rows * n * k, real_rows * 64 * k
        checked = macs - out
        doc["uncredited"]["calls"].append({
            "pass": 1, "call": 72, "weight": 64, "m": 16 * -(-real_rows // 16), "k": k, "n": n, "real_rows": real_rows,
            "k_kept": k, "rows": {"excluded_a": 0, "excluded_b": 1, "voluntary_a": 0, "voluntary_b": 0}, "macs": macs,
            "checked_macs": checked, "uncredited_macs": {"excluded_a": 0, "excluded_b": out, "voluntary": 0, "kept_width": 0},
            "epsilon_macs": "0", "over_bound_macs": "0", "rho_macs": str(Fraction(checked, 400)),
            "credited_macs": str(Fraction(399 * checked, 400))})
        resum(doc)
    return edit


def head_record(tmp_path: Path, real_rows: int = 4, without: int | None = None, **edits) -> Path:
    """The fixture record with the LM head's call `head_call` in its passes, the answer's and the Lean verdict's MACs
    and credit restated as the split's."""
    record = a_record(tmp_path, VERIFY=head_call(real_rows, without), **edits)
    totals = json.loads((record / C.VERIFY).read_text())["uncredited"]["totals"]
    for name in (C.ANSWER, C.LEAN):
        doc = json.loads((record / name).read_text())
        work = doc["public"]["work"] if name == C.ANSWER else doc["work"]
        work.update(macs=totals["checked_macs"], credited=totals["credited"])
        (record / name).write_text(json.dumps(doc))
    return record


#: the LM head call's real MACs, and what of them its excluded vocabulary row's block leaves out
HEAD_MACS, HEAD_OUT = 4 * 128_256 * 2048, 4 * 64 * 2048


def test_a_record_whose_passes_hold_the_lm_heads_calls_counts_the_head_and_drops_it_as_a_cause(store, tmp_path, capsys):
    """Where the verifier's split names a call of the LM head's shape, the number's text says Pearl+ credits the linears
    and the LM head, of the MACs both hold, and `--per-call` has no LM head cause: the head's call is a call like the
    linears' (named `lm_head`, its excluded vocabulary row's block among the excluded B rows' column blocks), and what
    the head computes outside the passes is a cause like the linears outside them. The causes still sum exactly."""
    store.add(RUN, head_record(tmp_path))
    assert bench(store, "--from", RUN) == 0
    text, doc, err = printed(capsys)
    credited, can = doc["credited_macs"], MACS["linears"] + MACS["lm_head"]
    assert err == "" and doc["lm_head"] == {"calls": 1, "creditable_macs": can, "creditable_share": can / TOTAL}
    assert text[9] == (f"  Pearl+ credits the linears and the LM head, whose calls the passes hold, {100 * can / TOTAL:.2f}% of "
                       f"the model's MACs at this class: its passes hold {PASSES + HEAD_MACS - HEAD_OUT:,} of them "
                       f"and its credit is {100 * credited / can:.2f}% of them all")
    assert bench(store, "--from", RUN, "--per-call", "--top", "257") == 0
    text, doc, _ = printed(capsys)
    got = doc["per_call"]
    causes = {c["cause"]: Fraction(c["macs"]) for c in got["causes"]}
    assert list(causes)[:3] == ["lm_head_outside", "attention", "outside"] and "lm_head" not in causes
    assert causes["lm_head_outside"] == MACS["lm_head"] - HEAD_MACS and causes["outside"] == GIVE_UP["outside"]
    assert causes["excluded_b"] == HEAD_OUT and sum(causes.values()) == got["give_up"] == TOTAL - credited
    assert not any("LM head, which Pearl+ doesn't credit" in line for line in text)
    assert any(line.startswith("    LM head calls outside the passes ") for line in text)
    head = next(c for c in got["top"] if c["call"] == 72)
    assert (head["linear"], head["n"], head["causes"]["excluded_b"]) == ("lm_head", 128_256, str(HEAD_OUT))


def test_a_credit_above_the_linears_is_refused_unless_the_passes_hold_the_lm_heads_calls(store, tmp_path, capsys):
    """A credit above the class's linears is no load of the class unless the verifier's split, over the answer's run
    commitment, names the LM head's calls; then the bound is the linears and the LM head together."""
    def credit_of(record: Path, credited: int, commitment: str | None = None) -> Path:
        for name in (C.ANSWER, C.LEAN):
            doc = json.loads((record / name).read_text())
            (doc["public"] if name == C.ANSWER else doc)["work"]["credited"] = credited
            (record / name).write_text(json.dumps(doc))
        if commitment is not None:
            doc = json.loads((record / C.VERIFY).read_text())
            (record / C.VERIFY).write_text(json.dumps({**doc, "commitment": commitment}))
        return record

    over, above = MACS["linears"] + 1, MACS["linears"] + MACS["lm_head"] + 1
    store.add(RUN, credit_of(a_record(tmp_path), over))
    assert bench(store, "--from", RUN) == 2
    assert f"more than the {MACS['linears']:,} of the linears of the policy's class" in capsys.readouterr().err
    store.add(RUN, credit_of(head_record(tmp_path / "head"), over))
    assert bench(store, "--from", RUN) == 0 and printed(capsys)[1]["credited_macs"] == over
    store.add(RUN, credit_of(head_record(tmp_path / "other"), over, commitment="c1" * 64))
    assert bench(store, "--from", RUN) == 2
    assert f"more than the {MACS['linears']:,} of the linears of the policy's class" in capsys.readouterr().err
    store.add(RUN, credit_of(head_record(tmp_path / "above"), above))
    assert bench(store, "--from", RUN) == 2
    assert f"more than the {MACS['linears'] + MACS['lm_head']:,} of the linears and the LM head" in capsys.readouterr().err


def test_a_split_whose_lm_head_calls_hold_more_than_the_classs_lm_head_is_refused(store, tmp_path, capsys):
    """The head's calls in the passes hold at most the class's LM head: 17 real rows of it a call is more than its 16.
    The last decode pass is dropped, so the passes are under the class's linears and LM head together and it is the
    split's check that refuses."""
    store.add(RUN, head_record(tmp_path, real_rows=17, without=3))
    assert bench(store, "--from", RUN, "--per-call") == 2
    assert (f"compute: the per-call split's LM head calls hold {17 * 128_256 * 2048:,} real MACs, more than the "
            f"{MACS['lm_head']:,} of the class's LM head") in capsys.readouterr().err


def no_store(*args, **kwargs):
    raise AssertionError("--from DIR touched the evidence store")


@pytest.fixture
def local(monkeypatch) -> None:
    """No `research run` around the command, so it writes no output."""
    monkeypatch.delenv("OUT", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)


@pytest.mark.parametrize("argv", [[], ["--per-call", "--top", "2"]])
def test_a_record_directory_prints_what_the_store_prints_for_the_same_record(store, tmp_path, capsys, argv):
    """The fixture is a run's directory as the store keeps its record: `--from DIR` reads it as it lies, with the run id
    and commit its `job.json` states, and prints byte for byte what `--from RUN` prints of the same record."""
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN, *argv) == 0
    from_store = capsys.readouterr()
    assert C.main(["--from", str(RECORD), *argv], store_of=no_store) == 0
    assert capsys.readouterr() == from_store
    doc = json.loads(from_store.out.splitlines()[-1])
    assert (doc["value"], doc["run"], doc["commit"], doc["credited_macs"]) == (SHARE, RUN, COMMIT, CREDITED)


def test_a_record_directory_is_read_with_no_store_client_at_all(tmp_path, capsys, monkeypatch, local):
    """`--from DIR` imports nothing of research's: with its store client's modules unimportable and the store's opener
    raising, the number is printed, its per-call split too."""
    import sys
    for name in ("research", "research.store", "research.store.local", "research.store.labels_sync",
                 "research.store.vocab"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(C.Unreadable, match="the evidence store's client isn't installed here"):
        C.opened()
    assert C.main(["--from", str(RECORD), "--per-call"], store_of=no_store) == 0
    doc = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert doc["value"] == SHARE and doc["withdrawn"] == [] and doc["per_call"]["give_up"] == TOTAL - CREDITED


@pytest.mark.parametrize("name", ["ANSWER", "POLICY"])
def test_a_record_directory_missing_a_required_file_exits_2_naming_it(tmp_path, capsys, local, name):
    record = a_record(tmp_path, **{name: None})
    assert C.main(["--from", str(record)], store_of=no_store) == 2
    out, err = capsys.readouterr()
    path = getattr(C, name)
    assert out == "" and f"compute: the record directory {record} holds no {path} ({C.REQUIRED[path]})" in err


def test_a_record_directory_that_is_no_run_directory_exits_2_and_says_what_to_name(tmp_path, capsys, local):
    """A run's `out/` named in place of the run's directory is refused with a hint, and a path that is no directory is
    refused as a path, never sent to the store as a run id. `--record` is gone: `--from` takes both."""
    assert C.main(["--from", str(RECORD / "out")], store_of=no_store) == 2
    err = capsys.readouterr().err
    assert f"holds no {C.ANSWER}" in err and "it is a run's out/ directory; name the directory holding it" in err
    assert C.main(["--from", str(tmp_path / RUN)], store_of=no_store) == 2
    assert f"compute: {tmp_path / RUN}: no such directory, and not a run id" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        C.main(["--from", str(RECORD), "--record", str(RECORD)], store_of=no_store)
    assert e.value.code == 2 and "unrecognized arguments: --record" in capsys.readouterr().err


def test_a_run_id_that_names_a_directory_is_read_as_the_directory(tmp_path, capsys, local, monkeypatch):
    """`--from` reads a directory where one is named, even one named like a run id, and goes to the store only for a
    run id that names none."""
    shutil.copytree(RECORD, tmp_path / RUN)
    monkeypatch.chdir(tmp_path)
    assert C.main(["--from", RUN], store_of=no_store) == 0
    assert printed(capsys)[1]["credited_macs"] == CREDITED


def test_a_record_directory_with_no_job_is_the_directorys_run_with_no_commit(tmp_path, capsys, local):
    record = a_record(tmp_path, JOB=None)
    assert C.main(["--from", str(record)], store_of=no_store) == 0
    text, doc, _ = printed(capsys)
    assert (doc["run"], doc["commit"], doc["value"], doc["made_up_says"]) == ("record", None, SHARE, None)
    assert text[3].startswith("  record, commit unknown: Pearl+'s answer accepts")
    dirty = a_record(tmp_path / "dirty", JOB=to("store", "conditions", "source.dirty", value=True))
    assert C.main(["--from", str(dirty)], store_of=no_store) == 0
    assert printed(capsys)[1]["commit"] == f"{COMMIT}-dirty"
    assert C.main(["--from", str(a_record(tmp_path / "bad", JOB="not json"))], store_of=no_store) == 2
    assert f"compute: the record's {C.JOB} doesn't read as the run's" in capsys.readouterr().err


@pytest.mark.parametrize("edits, rc, why", [
    ({"ANSWER": to("verdict", value="reject"), "LEAN": to("verdict", value="reject")}, 1,
     "no number: Pearl+'s answer is reject, not accept (no reason recorded)"),
    ({"LEAN": to("work", "credited", value=CREDITED + 1)}, 1,
     f"no number: the Lean verifier's credit is {CREDITED + 1} and the answer's {CREDITED}"),
    ({"VERIFY": to("commitment", value="c1" * 64)}, 2, "the per-call split is of another run commitment than the answer's"),
])
def test_a_record_directory_is_refused_as_the_store_refuses_the_same_record(store, tmp_path, capsys, edits, rc, why):
    """With no number the kit prints why on stdout (exit 1); what can't be read is on stderr (exit 2)."""
    record = a_record(tmp_path, **edits)
    store.add(RUN, record)
    for argv in (["--from", RUN], ["--from", str(record)]):
        assert C.main([*argv, "--per-call"], store_of=lambda: C.EvidenceStore(store, store.pull, Refused, "withdrawn")) == rc
        out, err = capsys.readouterr()
        assert f"compute: {why}" in (out if rc == 1 else err) and (err if rc == 1 else out) == ""


def test_a_record_directory_writes_the_same_outputs_as_the_store(store, tmp_path, capsys, monkeypatch, stamped):
    real = a_record(tmp_path, **REAL)
    store.add(RUN, real)
    written = {}
    for mode, argv in (("store", ["--from", RUN]), ("dir", ["--from", str(real)])):
        out = tmp_path / mode / "out"
        monkeypatch.setenv("OUT", str(out))
        assert bench(store, *argv) == 0
        written[mode] = [(out / f).read_text() for f in ("compute.json", "result.json")]
    capsys.readouterr()
    assert written["dir"] == written["store"]


@pytest.mark.parametrize("edits, rc", [({}, 0), ({"ANSWER": to("verdict", value="reject")}, 1)])
def test_a_made_up_record_is_written_as_made_up_never_measured(tmp_path, capsys, monkeypatch, stamped, edits, rc):
    """A made-up record goes through the kit as made up (`kit.report`'s ``made_up``), with a number or without: its
    lines under the number, ``made_up`` true in compute.json beside what it says of itself, and each entry of
    result.json marked modeled and made up, never measured and never an audit's."""
    out = tmp_path / "out"
    monkeypatch.setenv("OUT", str(out))
    assert C.main(["--from", str(a_record(tmp_path, **edits))]) == rc
    text, doc, _ = printed(capsys)
    assert text[3:5] == MADE_UP and text[2] == "a measurement, not an audit"
    assert (doc["made_up"], doc["made_up_says"], doc["audit"]) == (True, SAYS, False)
    assert json.loads((out / "compute.json").read_text()) == doc
    got = json.loads((out / "result.json").read_text())["measurements"]
    assert got and all((m["mark"], m["made_up"], m["audit"]) == ("modeled", True, False) for m in got)


def of_record(argv: list[str]) -> list[str]:
    return [*argv[:-1], "--of-record", argv[-1]]


#: the fixture's detail before its coins: its sample size, δ and scheme
DETAIL = "sample size 459, delta 1/100, scheme pearl-c-h100-v1-h3"
#: the edits that make the fixture's coins live: a challenger's sample and salt coins, and the Lean verifier of record
LIVE = {"ANSWER": to("profile", "source_from", value="challenger"), "DEVELOPMENT": to("coin_source", value="challenger"),
        "LEAN_CHECK": lambda d: d.update(argv=of_record(d["argv"]))}


def ran(**said) -> str:
    """The launcher's `out/compute-run.json` saying `said`, as `serve.sh` writes it."""
    return json.dumps({"class": "4,32,4", "scheme": "pearl-c-h100-v1-h3", "draws": 459, "delta": "1/100", **said})


@pytest.mark.parametrize("edits, said", [
    ({}, RAN_WITH),
    ({"DEVELOPMENT": None}, (f"ran with: coins in-process; parameters development ({DETAIL}, sample from the serving machine, salt coins "
                             "not stated, Lean verdict development)")),
    ({"ANSWER": to("profile", "source_from", value="challenger")},
     (f"ran with: coins in-process; parameters development ({DETAIL}, sample from a challenger, salt coins from "
      "in-process, Lean verdict development)")),
    ({"ANSWER": lambda d: (d["profile"].pop("draws"), d["profile"].pop("delta"))},
     ("ran with: coins in-process; parameters development (sample size not stated, delta not stated, scheme "
      "pearl-c-h100-v1-h3, sample from the serving machine, salt coins from in-process, Lean verdict development)")),
    ({"RAN": ran(challenger="local", development=True)}, RAN_WITH.replace("coins in-process", "coins local challenger")),
    ({"RAN": ran(challenger="in-process", development=True)}, RAN_WITH),
    ({**LIVE, "RAN": ran(challenger="local", development=True)},
     (f"ran with: coins local challenger; parameters development ({DETAIL}, sample from a challenger, salt coins from "
      "challenger, Lean verdict of record against the local challenger's key)")),
])
def test_the_ran_with_line_says_who_drew_the_coins_as_the_record_states_it(store, tmp_path, capsys, edits, said):
    """Under the number, the kit's line: who flipped the coins is the launcher's ``challenger`` where the record holds
    `out/compute-run.json`, else ``in-process`` where the sample is the node's or the salt coins in-process; the
    parameters are development; the detail is the record's own words, each `not stated` where it holds none.
    ``a measurement, not an audit`` follows, always. Both ways of reading the record print it."""
    record = a_record(tmp_path, **edits)
    store.add(RUN, record)
    for argv in (["--from", RUN], ["--from", str(record)]):
        assert bench(store, *argv) == 0
        text, doc, _ = printed(capsys)
        assert text[1:3] == [said, "a measurement, not an audit"] and doc["value"] == SHARE and doc["audit"] is False


@pytest.mark.parametrize("edits", [{**LIVE}, {**LIVE, "DEVELOPMENT": None}])
def test_a_challengers_coins_whose_challenger_the_record_does_not_name_are_not_stated(store, tmp_path, capsys, edits):
    """A challenger's sample and salt coins say no more than that some challenger drew them, the auditor's or one the
    developer runs: with no launcher's word the kit's line is ``ran with: not stated``, and the coins line under it
    says what the record does state. Never ``deployed challenger``."""
    record = a_record(tmp_path, **edits)
    store.add(RUN, record)
    for argv in (["--from", RUN], ["--from", str(record)]):
        assert bench(store, *argv) == 0
        text, doc, _ = printed(capsys)
        salt = "salt coins from challenger" if edits["DEVELOPMENT"] is not None else "salt coins: not stated in the record"
        assert text[1:3] == ["ran with: not stated", "a measurement, not an audit"]
        assert text[5] == (f"  coins as the record states them, which doesn't say whose challenger: sample from a challenger, "
                           f"{salt}; Lean verifier run as of record; parameters: sample size 459, delta 1/100, "
                           "development")
        assert doc["ran_with"] is None and doc["audit"] is False and "deployed" not in "\n".join(text)


#: the refusal of a launcher's deployed word, and the word
DEPLOYED = (f"{C.NO_DEPLOYED} (the record's {C.RAN} says the deployed challenger)",
            f"{C.NO_DEPLOYED} (the record's {C.RAN} says development false)")


@pytest.mark.parametrize("edits, why", [
    ({"RAN": "not json"}, f"the record's {C.RAN} doesn't read as JSON"),
    ({"RAN": ran(challenger="the auditor's")}, f"the record's {C.RAN} names the challenger \"the auditor's\", not one of"),
    ({"RAN": ran(challenger="deployed", development=False)}, DEPLOYED[0]),
    ({**{k: v for k, v in LIVE.items() if k != "LEAN_CHECK"}, "RAN": ran(challenger="deployed")}, DEPLOYED[0]),
    ({**LIVE, "DEVELOPMENT": None, "RAN": ran(challenger="deployed")}, DEPLOYED[0]),
    # ci's reader's probe: live coins beside PoUW's development piece, and a launcher saying deployed both
    ({**LIVE, "RAN": ran(challenger="deployed", development=False)}, DEPLOYED[0]),
    # comms' edited record: every file agreeing that the auditor's challenger drew live coins
    ({**LIVE, "RAN": ran(challenger="deployed", development=True)}, DEPLOYED[0]),
    ({**LIVE, "RAN": ran(challenger="local", development=False)}, DEPLOYED[1]),
    ({"DEVELOPMENT": None, "RAN": ran(challenger="in-process", development=False)}, DEPLOYED[1]),
    ({"RAN": ran(development=False)}, DEPLOYED[1]),
])
def test_a_launchers_word_the_command_cannot_check_exits_2(store, tmp_path, capsys, edits, why):
    """A person's own run writes every file of the record, so files that agree with each other prove nothing: a
    launcher's deployed challenger or ``development`` false exits 2, whatever the rest of the record says, as does a
    launcher's word that doesn't read."""
    record = a_record(tmp_path, **edits)
    store.add(RUN, record)
    for argv in (["--from", RUN], ["--from", str(record)]):
        assert bench(store, *argv) == 2
        out, err = capsys.readouterr()
        assert out == "" and f"compute: {why}" in err


PIECES = {"fixture": {}, "live": LIVE, "live, no development piece": {**LIVE, "DEVELOPMENT": None},
          "live, Lean not of record": {k: v for k, v in LIVE.items() if k != "LEAN_CHECK"},
          "no development piece": {"DEVELOPMENT": None}}
LAUNCHERS = {"none": None, **{f"{who} {said}": ran(**({} if who == "unnamed" else {"challenger": who}),
                                                 **({} if said == "unsaid" else {"development": said == "development"}))
                              for who in ("unnamed", "local", "in-process", "deployed")
                              for said in ("development", "deployed", "unsaid")}}


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("pieces", PIECES)
def test_no_record_prints_an_audit(tmp_path, capsys, local, pieces, launcher):
    """Every shape of the record's pieces against every word the launcher may say: a deployed word exits 2, and
    every other prints ``a measurement, not an audit`` with ``audit`` false and development parameters."""
    edits = PIECES[pieces] | ({} if LAUNCHERS[launcher] is None else {"RAN": LAUNCHERS[launcher]})
    rc = C.main(["--from", str(a_record(tmp_path, **edits))])
    text, doc, _ = printed(capsys)
    assert rc == (2 if "deployed" in launcher else 0)
    if rc == 0:
        assert text[1].endswith(("not stated", "Lean verdict development)", "Lean verdict of record)",
                                 "Lean verdict of record against the local challenger's key)"))
        assert text[2] == "a measurement, not an audit" and doc["audit"] is False and "deployed" not in "\n".join(text)


def test_the_benchmarks_load_is_the_fixtures_and_the_harness_class_of_record():
    """`load.toml` is the class the tied path serves (`served_verdicts.sh`'s V3C_CLASS default, 4,32,4) of the
    checkpoint it serves, and the fixture's policy registers it."""
    assert C.fixed_load() == {"model": "unsloth/Llama-3.2-1B", "revision": REVISION, "slots": 4, "prompt": 32,
                              "outputs": 4}
    policy = json.loads((RECORD / C.POLICY).read_text())
    assert {**policy["serve"], **{k: policy["class"][k] for k in ("slots", "prompt", "outputs")}} == C.fixed_load()


def test_a_development_piece_that_does_not_read_as_json_exits_2_naming_it(store, tmp_path, capsys):
    record = a_record(tmp_path, DEVELOPMENT="not json")
    store.add(RUN, record)
    for argv in (["--from", RUN], ["--from", str(record)]):
        assert bench(store, *argv) == 2
        out, err = capsys.readouterr()
        assert out == "" and f"compute: the record's {C.DEVELOPMENT} doesn't read as JSON" in err


def test_a_withdrawn_development_piece_is_read_as_absent_and_a_withdrawn_required_file_still_refuses(store, tmp_path,
                                                                                                   capsys):
    """`development.json` is optional (`OPTIONAL`): withdrawn, it is neither fetched nor read, the kit's line says the
    salt coins are not stated, and the withdrawal is printed beside the number; the number is unchanged."""
    record = a_record(tmp_path)
    (record / "out/pieces/pouw/coin-put.json").write_text("{}")
    store.add(RUN, record)
    why = f"hashes of a coin record (only out/pieces/pouw/coin-put.json, {C.DEVELOPMENT})"
    store.withdraw(f"art:{RUN}", why)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    said = f"{why!r}, by infra"
    assert doc["value"] == SHARE and doc["withdrawn"] == [said] and doc["coins"]["salt"] is None
    assert "sample from the serving machine, salt coins not stated, Lean verdict development" in text[1]
    assert text[8] == f"  part of the run's record was withdrawn, none of it a file read here: {said}"
    assert store.fetched == [[p for p in [*C.RECORDS, C.VERIFY] if p != C.DEVELOPMENT]]
    store.withdraw(f"art:{RUN}", f"private values were published (only {C.LEAN_CHECK}, {C.DEVELOPMENT})")
    assert bench(store, "--from", RUN) == 2
    assert "was withdrawn" in capsys.readouterr().err


def test_every_benchmark_result_reads_the_run_of_record_its_header_names():
    """`metrics.toml` names the served run its rows read once, in its ``RUN OF RECORD`` line: every row's ``--from`` is
    that run, so moving the rows to another run is one edit that this holds complete, and every metric `run.py` writes
    has a row."""
    import re
    import tomllib
    text = (HERE.parents[2] / "benchmarks" / "compute" / "metrics.toml").read_text()
    named = re.search(r"^# RUN OF RECORD: (r\d{8}-\d{6}-[0-9a-f]{4}),", text, re.M)
    assert named, "metrics.toml's header names no run of record"
    rows = tomllib.loads(text)["metric"]
    assert {m["id"]: re.search(r"--from (\S+)", m["command"]).group(1) for m in rows} == dict.fromkeys(
        (m["id"] for m in rows), named.group(1))
    assert {C.METRIC, C.CREDITABLE, C.OF_CREDITABLE} <= {m["id"] for m in rows}
