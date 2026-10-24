"""The line of record as the Lean verifier computes it (`auditor_line.line_of_record`: `remote-audit-verifier`'s `line`
request), held exactly to `line.py`'s on the same inputs: a disagreement refuses the line, naming both values, whether
the Lean's answer or line.py's is the one that moved, and on gate runs 5-7's credited sessions the Lean's line gives
the line at each T the recorded `auditor_line.json` holds, every share bit for bit (gate run 7's all of them) but for
four of gate runs 5 and 6's (`ULP`), where the record's float arithmetic is a float off the exact share's nearest.

`fixtures/gate_run_<G>.json.gz` holds each credited session's line inputs on the auditor's clock, as this tree's
`verifier.recorded_inputs` reads them from the run's record with no bursts (the reading of gate runs up to 7), its
groups as rows in `line.GROUP`'s order, beside the recorded shares and line (r20261024-090248-249f, which reran each
run's `auditor_line.analyse` through the Lean)."""
from __future__ import annotations

import copy
import gzip
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auditor_line as al  # noqa: E402
import lean_verifier as lv  # noqa: E402
from verity.core.protocols.remote_bandwidth import line  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
OLD = {"4.4e+09": line.TIME_HELD, "5.4e+10": line.LINK_CHARGED}
"""The recorded lines' forms by today's names (`auditor_line.RATES`: disk 4.4e9, and disk 4.4e9 plus link 5e10)."""
GROUPS = [{"released_ns": 10 ** 9 * g, "last_ns": 10 ** 9 * g + 10 ** 8, "passes": 2, "least": 1000, "weights": 1200,
           "hbm": 1600, "host": 400} for g in range(7)]
INPUTS = {"groups": GROUPS, "per": 2, "blocks": 1000, "block_bytes": 4096, "chip_bytes": 4096, "node_bytes": 4096000,
          "windows": [[2000, 2000], [2000, 1500]], "delta": "1/1099511627776",
          "rates": {"squarings_per_s": "30", "encode_squarings": 3, "disk_bytes_per_s": "4096",
                    "link_bytes_per_s": "40960", "rebuild_burst_bytes": "4096", "disk_burst_bytes": "0",
                    "link_burst_bytes": "8192"}}
"""A paced session of seven groups whose line credits weights and filler in both forms, with wrong receipts."""


@pytest.fixture(scope="module")
def lean():
    p = lv.Lean(lv.build(), lambda _: None)
    yield p
    p.close()


class Answers:
    """A Lean verifier whose `line` answers what `change` makes of line.py's line of the request's inputs."""

    def __init__(self, change):
        self.change = change

    def call(self, request: dict, body: bytes = b"") -> dict:
        assert request["op"] == "line"
        return self.change(line.run_json({k: v for k, v in request.items() if k != "op"}))


def moved(answer: dict) -> dict:
    out = copy.deepcopy(answer)
    out[line.TIME_HELD]["time"]["weights_share"] = "1/3"
    return out


def test_the_lean_computes_the_line_line_py_computes(lean):
    """The Lean's answer to `line` is line.py's, every rational in lowest terms and every integer, and a refusal is
    both's, word for word; with no Lean the line is line.py's alone."""
    got = al.line_of_record(INPUTS, lean)
    assert got == line.run_json(INPUTS) == al.line_of_record(INPUTS, None)
    shares = al.shares_of(got)
    assert all(0 < shares[f][x] < 1 for f in line.FORMS for x in al.SHARES)
    assert all(shares[f][x] == float(line.number(shares[f]["exact"][x])) for f in line.FORMS for x in al.SHARES)
    bad = {**INPUTS, "windows": []}
    with pytest.raises(line.Refused, match=f"^{re.escape(line.run_json(bad)['refused'])}$"):
        al.line_of_record(bad, lean)


@pytest.mark.parametrize("side", ["lean", "python"])
def test_a_disagreement_refuses_the_line_naming_both_values(request, monkeypatch, side):
    """A share the Lean and line.py give differently, either way round (the Lean's answer moved, or line.py's beside
    the real Lean's), refuses the line with both values; so does a line one of them refuses and the other doesn't, with
    the refusal and the other's line."""
    right = line.run_json(INPUTS)
    want = right[line.TIME_HELD]["time"]["weights_share"]
    if side == "lean":
        by, lean_says, python_says = Answers(moved), "1/3", want
    else:
        by, lean_says, python_says = request.getfixturevalue("lean"), want, "1/3"
        monkeypatch.setattr(al.line, "run_json", lambda j: moved(line.line(**line.of_json(j)).to_json()))
    with pytest.raises(al.Disagree) as e:
        al.line_of_record(INPUTS, by)
    assert str(e.value) == (f"the Lean's line and line.py's disagree at .{line.TIME_HELD}.time.weights_share: the "
                            f"Lean's \"{lean_says}\", line.py's \"{python_says}\"")

    refusal = {"refused": "the windows are none"}
    said = ("line, weights held " + ", ".join(f"{f} {right[f]['time']['weights_share']}" for f in line.FORMS))
    if side == "lean":
        by, says = Answers(lambda _: refusal), f"the Lean's refusal \"{refusal['refused']}\", line.py's {said}"
    else:
        monkeypatch.setattr(al.line, "run_json", lambda _: refusal)
        says = f"the Lean's {said}, line.py's refusal \"{refusal['refused']}\""
    with pytest.raises(al.Disagree) as e:
        al.line_of_record(INPUTS, by)
    assert str(e.value) == f"the Lean's line and line.py's disagree: {says}"


@pytest.mark.parametrize("change, at", [
    (lambda a: {**a, "extra": 1}, ".extra: the Lean's 1, line.py's nothing"),
    (lambda a: {k: v for k, v in a.items() if k != "wrong"}, ".wrong: the Lean's nothing"),
    (lambda a: {**a, line.LINK_CHARGED: {**a[line.LINK_CHARGED], "stretches": []}},
     f".{line.LINK_CHARGED}.stretches (its length): the Lean's 0"),
    (lambda a: {**a, line.TIME_HELD: {**a[line.TIME_HELD], "time": {**a[line.TIME_HELD]["time"], "stretch": "0"}}},
     f".{line.TIME_HELD}.time.stretch: the Lean's \"0\", line.py's 0"),
])
def test_any_other_difference_refuses_the_line_too(change, at):
    """A key one side lacks, a list of another length, an integer written as a string: each is a disagreement."""
    with pytest.raises(al.Disagree, match="disagree at") as e:
        al.line_of_record(INPUTS, Answers(change))
    assert at in str(e.value)


ULP = {5: {(2, line.TIME_HELD, "time")},
       6: {(0, line.TIME_HELD, "snapshot"), (0, line.LINK_CHARGED, "time"), (2, line.LINK_CHARGED, "snapshot")},
       7: set()}
"""The recorded shares (session, form, share) a float apart from the exact share's nearest float, the Lean's: gate runs
5-7 were analysed in floats, before line.py, and these four came out the next float over, no nearer the exact share."""


@pytest.mark.parametrize("g", [5, 6, 7])
def test_the_leans_line_is_the_record_of_gate_runs_5_to_7(lean, g):
    """Each credited session's shares, both forms, over time and at the passes, weights and filler, with its worst
    stretch, through the Lean's line are the recorded floats bit for bit, but for `ULP`'s, each the float beside the
    Lean's and no nearer the exact share; and the line at each T is the recorded one, its session the record's."""
    run = json.loads(gzip.decompress((FIXTURES / f"gate_run_{g}.json.gz").read_bytes()))
    assert run["gate_run"] == g and run["rates"] == al.rates_at(bursts="none")
    got, off = {}, set()
    for s in run["sessions"]:
        inputs = {**s["inputs"], "groups": [dict(zip(line.GROUP, x)) for x in s["inputs"]["groups"]]}
        got[s["k"]] = al.shares_of(al.line_of_record(inputs, lean))
        for old, form in OLD.items():
            mine, want = got[s["k"]][form], s["recorded"][old]
            for x, w in want.items():
                if mine[x] != w:
                    off.add((s["k"], form, x))
                    exact = line.number(mine["exact"][x])
                    assert mine[x] == float(exact) and w == math.nextafter(mine[x], w), (s["k"], form, x)
                    assert abs(Fraction(w) - exact) > abs(Fraction(mine[x]) - exact), (s["k"], form, x)
    assert off == ULP[g]
    for t, row in run["line"].items():
        at = [s for s in run["sessions"] if s["T"] == t]
        assert row["credited_sessions"] == [s["k"] for s in at]
        for old, form in OLD.items():
            k = min((s["k"] for s in at), key=lambda k: line.number(got[k][form]["exact"]["time"]))
            recorded = next(s["recorded"][old] for s in at if s["k"] == k)
            assert row[old] == {"k": k, **{x: recorded[x] for x in row[old] if x != "k"}}, (t, form)
