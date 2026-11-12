"""`verity bench compute` (`run.py`): the number is the MACs an accepting Pearl+ answer credits over the MACs the served
model computes at the policy's class, printed as text and as one line of JSON beside its target; an answer that isn't
an accept, or that the Lean verifier's verdict isn't the same as, prints no number and exits 1; a policy that can't
give the model's MACs exits 2. The model's MACs are held to a count made one position, one query-key pair and one row
of logits at a time. PoUW's row in the run's composed record is printed whatever it says, and nothing the developer's
node reports of what it served is read.

Nothing here reaches the evidence store: `FakeStore` stands in for research's local store over the small record under
`fixtures/record` (a run's `out/`), which each test copies and edits. A record withdrawn whole, or in a part that holds
a file the command reads, is refused before anything is fetched, and a withdrawal of other files of it is printed."""
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
#: the fixture's credit, three quarters of TOTAL
CREDITED = 105_449_127_936


def a_record(tmp_path: Path, **edits) -> Path:
    """A copy of the fixture record, laid out as a run's (its files under `out/`), with `edits` applied, each a file's
    name under `C` (ANSWER, LEAN, LEAN_CHECK, POLICY, VERDICTS) and what becomes of it: None deletes it, a string
    replaces its text, and a function edits its JSON in place."""
    record = tmp_path / "record"
    shutil.copytree(RECORD, record / "out")
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
    return FakeStore(tmp_path / "store")


def bench(store: FakeStore, *argv: str) -> int:
    return C.main(list(argv), store_of=lambda: C.EvidenceStore(store, store.pull, Refused, "withdrawn"))


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
        "compute: 75.00% of the served model's MACs are credited by Pearl+ (105,449,127,936 of 140,598,837,248), below "
        "the 97% target",
        f"  {RUN}, commit {COMMIT}: Pearl+'s answer accepts (scheme pearl-c-h100-v1-h3) at ε_s 1/100, δ_s 1/100 "
        f"(sampling's alone: the assumption's gap and δ, and binding, are not in them), and the Lean verifier's verdict "
        f"is the same",
        "  PoUW's row in the run's composed record is `missing`, not an accept, so the credit is the Pearl+ piece's "
        "alone: development run, credits nothing: the registration can't be read",
        "  unsloth/Llama-3.2-1B at 4 slots of 32 prompt and 4 output tokens: linears 136,230,993,920, attention "
        "165,150,720 and LM head 4,202,692,608 MACs, which is the policy's own count of its class",
        "  Pearl+ credits linears alone, 96.89% of the model's MACs at this class: its passes hold 120,000,000,000 of "
        "them and its credit is 77.40% of them all",
        "  tokens, the class's since the record holds no count of those served: 128 prompt and 16 output, over 140 "
        "positions",
        f"  capacity share: {C.ABSENT}",
    ]
    assert list(doc)[:4] == ["value", "unit", "commit", "run"]
    assert (doc["value"], doc["unit"], doc["commit"], doc["run"]) == (0.75, C.UNIT, COMMIT, RUN)
    assert (doc["credited_macs"], doc["model_macs"], doc["macs"], doc["passes_macs"]) == (CREDITED, TOTAL, MACS, 120_000_000_000)
    assert (doc["target"], doc["met"], doc["lean"], doc["policy_agrees"]) == (0.97, False, True, True)
    assert doc["record"] == {"verdict": "missing", "why": "development run, credits nothing: the registration can't be read"}
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
    monkeypatch.setattr(C, "TARGET", Fraction(3, 4))
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[0].endswith("at or above the 75% target") and doc["met"] is True and doc["target"] == 0.75
    under = tmp_path / "under"
    under.mkdir()
    one_less = a_record(under, ANSWER=to("public", "work", "credited", value=CREDITED - 1),
                        LEAN=to("work", "credited", value=CREDITED - 1))
    store.add(RUN, one_less)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[0].startswith("compute: 75.00% ") and text[0].endswith("below the 75% target") and doc["met"] is False


@pytest.mark.parametrize("verdicts, said", [
    (to("protocols", "work", value={"verdict": "accept"}), "  PoUW's row in the run's composed record accepts too"),
    (to("protocols", "work", value={"verdict": "reject", "reason": "a sampled tile is wrong"}),
     "  PoUW's row in the run's composed record is `reject`, not an accept, so the credit is the Pearl+ piece's alone: "
     "a sampled tile is wrong"),
    (to("protocols", value={}), "  PoUW's row in the run's composed record is absent, so the credit is the Pearl+ "
                                "piece's alone"),
    (None, "  PoUW's row in the run's composed record is absent, so the credit is the Pearl+ piece's alone"),
])
def test_pouws_row_in_the_composed_record_is_printed_whatever_it_says(store, tmp_path, capsys, verdicts, said):
    store.add(RUN, a_record(tmp_path, VERDICTS=verdicts))
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[2] == said and doc["value"] == 0.75


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
])
def test_an_answer_that_is_not_an_accept_the_lean_verifier_shares_prints_no_number_and_exits_1(store, tmp_path, capsys,
                                                                                              edits, why):
    store.add(RUN, a_record(tmp_path, **edits))
    assert bench(store, "--from", RUN) == 1
    out, err = capsys.readouterr()
    assert out == "" and f"compute: no number: {why}" in err


def test_a_record_with_no_lean_verdict_prints_the_answers_number_and_says_so(store, tmp_path, capsys):
    store.add(RUN, a_record(tmp_path, LEAN=None, LEAN_CHECK=None), dirty=True)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert text[1] == (f"  {RUN}, commit {COMMIT}-dirty: Pearl+'s answer accepts (scheme pearl-c-h100-v1-h3) at ε_s 1/100, "
                       f"δ_s 1/100 ({C.SAMPLING_ALONE}), and the record holds no Lean verdict")
    assert doc["lean"] is False and doc["value"] == 0.75 and doc["commit"] == f"{COMMIT}-dirty"


@pytest.mark.parametrize("edits", [{"ANSWER": None}, dict.fromkeys(("ANSWER", "LEAN", "LEAN_CHECK", "POLICY", "VERDICTS"))])
def test_a_run_that_left_no_answer_prints_no_number_and_exits_1(store, tmp_path, capsys, edits):
    record = a_record(tmp_path, **edits)
    (record / "out/result.json").write_text("{}")
    store.add(RUN, record, rc=3)
    assert bench(store, "--from", RUN) == 1
    out, err = capsys.readouterr()
    assert out == "" and f"compute: no number: {RUN} left no Pearl+ answer (it ended rc 3)" in err


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
        raise C.Unreadable("the evidence store's client isn't installed here (No module named 'research')")
    assert C.main(["--from", RUN], store_of=absent) == 2
    assert "compute: the evidence store's client isn't installed here" in capsys.readouterr().err


OTHERS = "out/pieces/pouw/coin-put.json, out/pieces/pouw/development.json"


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
    assert doc["withdrawn"] == [said] and doc["value"] == 0.75
    assert text[3] == f"  part of the run's record was withdrawn, none of it a file read here: {said}"
    assert store.fetched == [C.RECORDS] and store.pulled == [[RUN, f"art:{RUN}"]]
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
     "the policy counts its class's MACs as {'linears': 136230993920, 'attention': 165150720, 'lm_head': 4} and run.py "
     "as {'linears': 136230993920, 'attention': 165150720, 'lm_head': 4202692608}"),
    ({"VERDICTS": "[]"}, "the run's composed record doesn't read as one"),
    ({"ANSWER": to("public", "work", "credited", value=MACS["linears"] + 1), "LEAN": None},
     "Pearl+'s answer holds 136,230,993,921 MACs, more than the 136,230,993,920 of the linears of the policy's class: "
     "its passes are not one load of the class"),
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
    assert doc["policy_agrees"] is None and doc["value"] == 0.75 and text[3].endswith("LM head 4,202,692,608 MACs")


def test_nothing_the_developers_node_reports_of_what_it_served_is_read(store, tmp_path, capsys):
    """The run's record also holds the Pearl+ piece's own document, with the tokens and seconds the node says it
    served: the command asks the store for the verifier's files alone, and prints no capacity share."""
    record = a_record(tmp_path)
    (record / "out/pieces/pouw/development.json").write_text(json.dumps(
        {"verdict": "accept", "served_tokens": 16, "generate_s": 2.0, "credit": {"credited": 1}}))
    store.add(RUN, record)
    assert bench(store, "--from", RUN) == 0
    text, doc, _ = printed(capsys)
    assert store.fetched == [C.RECORDS] and not any("development" in g or "*" in g for g in C.RECORDS)
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
    store.add(RUN, a_record(tmp_path))
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
                         "linears_share", "passes_macs", "scheme", "epsilon_s", "delta_s", "lean", "record", "withdrawn",
                         "model", "class", "policy_agrees", "tokens", "tokens_from", "wall_s", "capacity_share"]
    written = (out / "result.json").read_bytes()
    store.add(RUN, a_record(tmp_path / "run", ANSWER=to("verdict", value="reject")))
    (out / "compute.json").unlink()
    assert bench(store, "--from", RUN) == 1 and not (out / "compute.json").exists()
    assert (out / "result.json").read_bytes() == written


def test_the_shared_result_is_the_metric_verified_and_error_with_capacity_and_free_absent(store, tmp_path, capsys,
                                                                                         monkeypatch, stamped):
    """The record holds no registration of the run's hardware and no span the verifier holds its passes to, so
    capacity and free are written with no value, each with why, and no reading is named."""
    out = tmp_path / "out"
    monkeypatch.setenv("OUT", str(out))
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN) == 0
    result = json.loads((out / "result.json").read_text())
    assert (result["schema"], result["run_id"]) == ("research/result/v0.1", BENCH)
    got = {m["name"]: m for m in result["measurements"]}
    assert list(got) == ["metric", "verified", "capacity", "free", "error"]
    assert (got["metric"]["value"], got["metric"]["unit"], got["metric"]["target"], got["metric"]["verdict"]) == (
        0.75, C.UNIT, 0.97, "below target")
    assert (got["verified"]["value"], got["verified"]["unit"]) == (CREDITED, "MAC")
    why = f"absent: {C.NO_SITE}; and {C.NO_SPAN}"
    for name in ("capacity", "free"):
        assert (got[name]["value"], got[name]["unit"], got[name]["why"]) == (None, "MAC", why)
    # the error is the answer's δ_s, named in its inputs, and its unit says it is sampling's alone
    assert (got["error"]["value"], got["error"]["unit"]) == (0.01, C.DELTA_S) and "why" not in got["error"]
    assert "the answer's δ_s, sampling's alone" in C.DELTA_S
    for m in got.values():
        assert (m["run"], m["commit"], m["host"]) == (BENCH, STAMP["commit"], STAMP)
        assert m["inputs"] == ([RUN, "delta_s"] if m["name"] == "error" else [RUN])
    assert C.NO_SPAN == ("a Pearl+ answer holds its passes to no seconds, and the seconds the run served in stay on the "
                         "developer's node")


def h200_site(devices: int):
    """A registered site of one host with `devices` H200s, each naming the sheet its card file gives."""
    from verity.hardware import site as S
    sheet = S.Sheet.of_file(C.cards.load("h200-sxm-141gb"))
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
    assert list(got) == ["metric", "verified", "capacity at clock.tensor", "free at clock.tensor", "capacity at clock.sm",
                         "free at clock.sm", "error"]
    assert [got[n]["value"] for n in got if n.startswith(("capacity", "free"))] == [
        tensor, tensor - CREDITED, sm, sm - CREDITED]
    assert not any("why" in m for m in got.values())
    # with either missing, capacity and free are absent and say which
    assert {m["name"]: m.get("why") for m in C.amounts(doc, h200_site(2))}["capacity"] == f"absent: {C.NO_SPAN}"
    assert {m["name"]: m.get("why") for m in C.amounts(doc, None, Fraction(600))}["free"] == f"absent: {C.NO_SITE}"


def test_capacity_refuses_hardware_it_cannot_read_a_peak_off(tmp_path):
    from verity.hardware import site as S
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


#: the fixture's give-up, TOTAL less CREDITED, by cause: the verify document's split puts 4 excluded A rows in call 6
#: and 1 in call 70, ε's deduction on calls 2 and 71, and holds 120,150,994,944 of the linears' MACs
GIVE_UP = {"lm_head": 4_202_692_608, "attention": 165_150_720, "outside": 16_079_998_976, "excluded_a": 150_994_944,
           "excluded_b": 0, "voluntary": 0, "kept_width": 0, "epsilon_macs": Fraction(5_700_348_825_400, 399),
           "over_bound_macs": 0, "rho_macs": Fraction(210_898_255_873, 798), "floor": Fraction(1, 2)}


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
    """An edit that moves `n` MACs of call 71's ε deduction to its credit, ι taking its share, and restates the totals."""
    def edit(doc: dict) -> None:
        c = doc["uncredited"]["calls"][3]
        c["epsilon_macs"] = str(Fraction(c["epsilon_macs"]) - n)
        c["rho_macs"] = str(Fraction(c["rho_macs"]) + Fraction(n, 400))
        c["credited_macs"] = str(Fraction(c["credited_macs"]) + Fraction(399 * n, 400))
        resum(doc)
    return edit


def test_per_call_prints_the_give_up_by_cause_and_the_calls_that_give_up_the_most(store, tmp_path, capsys):
    """`--per-call` reads the verify document beside the answer, and only then: the causes sum to the model's MACs less
    the credit exactly, and the calls are ordered by what each gives up, named by their linear where the model's shape
    has one."""
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN) == 0 and "per_call" not in printed(capsys)[1]
    assert store.fetched == [C.RECORDS]
    assert bench(store, "--from", RUN, "--per-call", "--top", "2") == 0
    text, doc, err = printed(capsys)
    assert err == "" and store.fetched[-1] == [*C.RECORDS, C.VERIFY] and doc["value"] == 0.75
    got = doc["per_call"]
    assert got["give_up"] == TOTAL - CREDITED == sum(GIVE_UP.values()) and got["calls"] == 4 and got["source"] == C.VERIFY
    assert [(c["cause"], Fraction(c["macs"])) for c in got["causes"]] == list(GIVE_UP.items())
    assert sum(Fraction(c["macs"]) for c in got["causes"]) == got["give_up"]
    assert [(c["call"], c["linear"]) for c in got["top"]] == [(71, "1x1"), (2, "gate_up")]
    assert got["top"][1]["causes"] == {"epsilon_macs": "3000000001/3", "rho_macs": "9884901887/1200"}
    assert text[7:] == [
        "  per call: the model gives up 35,149,709,312 of its MACs (25.00%); by cause, from the verifier's split of its 4 "
        "calls (out/pieces/pouw/verify.json):",
        "    LM head, which Pearl+ doesn't credit         4,202,692,608 MACs   11.96%",
        "    attention, which Pearl+ doesn't credit         165,150,720 MACs    0.47%",
        "    linears outside the passes                  16,079,998,976 MACs   45.75%",
        "    excluded A rows                                150,994,944 MACs    0.43%",
        "    excluded B rows' column blocks                           0 MACs    0.00%",
        "    voluntary rows' bands and blocks                         0 MACs    0.00%",
        "    side channels (k beyond |K|)                             0 MACs    0.00%",
        "    ε's deduction                               14,286,588,535 MACs   40.64%",
        "    sampled tiles over the inert-work bound                  0 MACs    0.00%",
        "    ι, the inert-work share                        264,283,529 MACs    0.75%",
        "    the credit's floor                                       0 MACs    0.00%",
        "  ε's deduction is the run's, at its sample: the verifier takes it off the records whose cells hold the most MACs "
        "a ticket, so the calls it lands on carry it for the whole run",
        "  the 2 calls that give up the most:",
        "    pass 1 call 71, weight 1000 1x1, 111493951488x1x1 with 111493951488 real rows: 13,532,106,942 MACs "
        "(ε's deduction 13,286,588,535, ι 245,518,407)",
        "    pass 0 call 2, weight 2 gate_up, 128x2048x16384 with 128 real rows: 1,008,237,419 MACs "
        "(ε's deduction 1,000,000,000, ι 8,237,418)",
    ]
    assert bench(store, "--from", RUN, "--per-call") == 0
    top = printed(capsys)[1]["per_call"]["top"]
    assert [(c["call"], c["linear"], c["causes"].get("excluded_a")) for c in top] == [
        (71, "1x1", None), (2, "gate_up", None), (6, "gate_up", "134217728"), (70, "down", "16777216")]


@pytest.mark.parametrize("edits, why", [
    ({"VERIFY": None}, "the record holds no per-call give-up (`uncredited`): the verifier writes it once Pearl+'s "
                       "verifier side of the per-call give-up is on main"),
    ({"VERIFY": lambda d: d.pop("uncredited")}, "the record holds no per-call give-up (`uncredited`)"),
    ({"VERIFY": lambda d: d["uncredited"].pop("totals")}, "the per-call split doesn't read as one (KeyError: 'totals')"),
    ({"VERIFY": to("commitment", value="c1" * 64)}, "the per-call split is of another run commitment than the answer's"),
    ({"VERIFY": lambda d: (d["uncredited"]["calls"][1].update(macs=d["uncredited"]["calls"][1]["macs"] + 1), resum(d))},
     "the per-call split's call 6 doesn't sum: its MACs are not its checked MACs and its causes"),
    ({"VERIFY": lambda d: d["uncredited"]["calls"][2].update(epsilon_macs="-1")},
     "the per-call split's call 70 holds a count that is no count of MACs"),
    ({"VERIFY": to("uncredited", "totals", "excluded_a", value=0)}, "the per-call split's totals are not its calls' sums"),
    ({"VERIFY": moved(400)}, "the per-call split's calls credit 105,449,128,335 of 120,000,000,000 MACs and the answer "
                             "105,449,127,936 of 120,000,000,000"),
    ({"VERIFY": lambda d: (d["uncredited"]["calls"][3].update(macs=d["uncredited"]["calls"][3]["macs"] + 16_079_998_977,
                                                              uncredited_macs={**d["uncredited"]["calls"][3]["uncredited_macs"],
                                                                               "kept_width": 16_079_998_977}), resum(d))},
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
    assert printed(capsys)[1]["per_call"]["causes"][-1] == {"cause": "floor", "macs": "799/800",
                                                            "share": float(Fraction(799, 800) / (TOTAL - CREDITED))}


def test_per_call_reads_a_split_in_a_file_of_its_own_for_a_record_that_holds_none(store, tmp_path, capsys):
    """``--uncredited F`` reads a split in a file of its own, with its commitment and its passes' records, in place of
    the verify document, which isn't fetched, and holds those records to the answer's where the answer names them."""
    split = json.loads((RECORD / "pieces/pouw/verify.json").read_text())["uncredited"]
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
