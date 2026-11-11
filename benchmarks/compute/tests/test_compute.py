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
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import run as C  # noqa: E402

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


def test_under_a_research_run_the_printed_document_is_also_its_output(store, tmp_path, capsys, monkeypatch):
    out = tmp_path / "run" / "out"
    monkeypatch.setenv("OUT", str(out))
    store.add(RUN, a_record(tmp_path))
    assert bench(store, "--from", RUN) == 0
    _, doc, _ = printed(capsys)
    assert json.loads((out / "compute.json").read_text()) == doc
    store.add(RUN, a_record(tmp_path / "run", ANSWER=to("verdict", value="reject")))
    (out / "compute.json").unlink()
    assert bench(store, "--from", RUN) == 1 and not (out / "compute.json").exists()
