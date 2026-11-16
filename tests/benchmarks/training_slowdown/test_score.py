"""training_slowdown.score and the library's `--records` parts, on made records (no run's record): C_e bracketed by
the checkpoints either side of the target, the reference as the least recorded cost at a start, C_e / T_ref, the
paused search's score as the grade gives it, and the rows `--records` prints, each cell a figure or a reason."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from training_slowdown import gates, grade, library, policy, score

E, A, GAMMA = 0.4, 20.0, 0.07
SHAPE = policy.SHAPES[policy.REFERENCE_PARAMS] | {"ctx": 1024, "parameters": policy.REFERENCE_PARAMS}
PER_MAC = gates.per_mac("gemm/bf16")
LOG, STEP_GATES = [["fwd", "theta", "mm", "[[8]]", 1, 80]], 80 * PER_MAC
GAIN = 1.1


def loss_at(c0_macs: float) -> float:
    return E + A * (c0_macs * PER_MAC) ** -GAMMA


#: a made starts registry: four starts on one fit, each at its own loss
REG = {"data": {"d": {}}, "starts": {name: {"data": "d", "shape": SHAPE, "c0_macs": c, "l0_bpb": loss_at(c)}
                                      for name, c in (("s", 1e15), ("m", 1e16), ("l", 1e17), ("x", 1e18))}}
L0 = loss_at(1e15)
TARGET = grade.target(grade.start_fit(REG["starts"], "s")[0], L0, GAIN)
OVER, UNDER = TARGET + 1e-4, TARGET - 1e-4


def made(losses: list, start: str = "s") -> dict:
    """A made record on the search split: one step of 80 MACs between checkpoints, checkpoint i after step i."""
    steps = [{"step": i, "tokens": 512, "struct": "a", "by": {"fwd/theta": 80}} for i in range(len(losses))]
    return {"training_slowdown": "0", "start": {"name": start}, "split": "search", "invalid": [],
            "forward_macs_per_token": 51_342_080, "setup": {"by": {}}, "steps": steps, "tail": {"by": {}},
            "structures": {"a": {"updated": [["w", 0, 80]], "log": LOG}},
            "checkpoints": [{"step": i, "bpb": loss, "source": "output",
                             "score": {"bpb": loss, "start_bpb": L0, "split": "search"}}
                            for i, loss in enumerate(losses)]}


def counted(rec: dict, step: int) -> float:
    return next(c["gates"] for c in grade.checkpoints(score.job(rec), "s") if c["step"] == step)


def test_cost_brackets_the_target_between_the_checkpoints_either_side():
    rec = made([OVER + 0.2, OVER + 0.1, OVER, UNDER, UNDER - 0.1])
    c = score.cost_to_target(rec, "s", GAIN, REG["starts"])
    assert c["reached"] and c["steps"] == [2, 3]
    assert c["gates"] == [counted(rec, 2), counted(rec, 3)] and c["gates"][0] < c["gates"][1]


def test_a_first_checkpoint_at_the_target_opens_the_bracket_at_the_start():
    c = score.cost_to_target(made([UNDER, UNDER - 0.1]), "s", GAIN, REG["starts"])
    assert c["gates"][0] == 0.0 and c["steps"][0] is None


def test_a_record_that_misses_says_how_far_it_got():
    c = score.cost_to_target(made([OVER + 0.1, OVER]), "s", GAIN, REG["starts"])
    assert not c["reached"] and c["why"].startswith("misses the target") and "step 1" in c["why"]


def test_the_reference_is_the_least_recorded_cost_at_each_start():
    costs = {"a": ("s", {"reached": True, "gates": [1.0, 4.0]}), "b": ("s", {"reached": True, "gates": [2.0, 3.0]}),
             "c": ("s", {"reached": False, "why": "misses"}), "d": ("m", {"reached": False, "why": "misses"})}
    refs = score.reference(costs)
    assert refs == {"s": {"gates": 3.0, "entry": "b", "label": score.REFERENCE}}
    assert score.ratio(costs["a"][1], refs["s"]) == {"ratio": [1 / 3, 4 / 3]}
    assert score.ratio(costs["c"][1], refs["s"])["ratio"] is None
    assert score.ratio({"reached": True, "gates": [1.0, 2.0]}, None)["why"]


def test_a_record_with_no_trace_has_the_graders_reason_for_no_search_score():
    config = {"name": "made", "start": "s", "target": {"gain": GAIN}, "policy": "console", "honest": "",
              "seeds": {"search": [0]}}
    s = score.search_score(config, made([OVER, UNDER]), REG)
    assert s["score"] is None and "trace" in s["why"]


def cfg(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, start=SimpleNamespace(name="s"), target={"gain": GAIN}, policy="console",
                           honest="", seeds={"search": [0]})


@pytest.fixture
def records(tmp_path, monkeypatch):
    """Made records of two entries at DIR/RUN/PATH as `of_record.toml` would name them, a third entry held to none,
    and the library's parts from them, with the factor standing in for a traced row's."""
    held = {}
    for name, losses in (("fast", [OVER, UNDER]), ("slow", [OVER + 0.1, OVER, OVER, UNDER])):
        raw = json.dumps(made(losses)).encode()
        (tmp_path / name).mkdir()
        (tmp_path / name / "run.json").write_bytes(raw)
        held[name] = {"run": name, "art": "art:made", "path": "run.json", "sha256": hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(score, "held", lambda: held)
    cfgs = [cfg("fast"), cfg("slow"), cfg("untraced")]
    rows = [{"slowdown": 2.5, "binds": "x"}, {"slowdown": None, "binds": library.CODED},
            {"slowdown": None, "binds": "y"}]
    return library.of_records(cfgs, rows, tmp_path, REG), tmp_path


def test_the_records_parts_are_each_a_figure_or_a_reason(records):
    (parts, refs), root = records
    by = {p["entry"]: p for p in parts}
    assert refs["s"]["entry"] == "fast"
    assert by["fast"]["ratio"]["ratio"][1] == 1.0 and by["slow"]["ratio"]["ratio"][1] > 1.0
    assert by["fast"]["factor"] == {"factor": 2.5}
    assert by["slow"]["factor"] == {"factor": None, "why": "not computed: " + library.CODED}
    assert by["untraced"]["why"] == library.TRACED_ONLY and "cost" not in by["untraced"]
    out = library.records_text(parts, refs, root)
    head, *table = out.split("\n\nwhy")[0].splitlines()[2:]
    assert len(table) == 3 and all(len(line.split()) > 6 for line in table)
    assert "the paused search's score" in head and "slowdown" not in head
    for said in (score.UNIT, score.REFERENCE, library.FACTOR, library.SEARCH_SCORE, library.TRACED_ONLY,
                 library.CODED, "GPU-hours, is not computed"):
        assert said in out


def test_a_held_record_missing_under_the_directory_is_refused_with_its_fetch(records, monkeypatch):
    (_, _), root = records
    (root / "slow" / "run.json").unlink()
    with pytest.raises(SystemExit, match="research data fetch art:made"):
        library.of_records([cfg("slow")], [{"slowdown": None, "binds": "y"}], root, REG)
