"""The line of record as the Lean verifier computes it (`auditor_line.line_of_record`: `remote-audit-verifier`'s
`line_record` request off the record, or its `line` of groups and windows given), held exactly to `line.py`'s on the
same inputs: a disagreement refuses the line, naming both values, whether
the Lean's answer or line.py's is the one that moved, and on gate runs 5-7's credited sessions the Lean's line gives
the line at each T the recorded `auditor_line.json` holds, every share bit for bit once the half chunk the chips'
whole chunks add to the record's S is given back (gate run 7's all of them) but for four of gate runs 5 and 6's
(`ULP`), where the record's float arithmetic is a float off the exact share's nearest.

`fixtures/gate_run_<G>.json.gz` holds each credited session's line inputs on the auditor's clock, of groups and windows
given (`line`), as `verifier.recorded_inputs` read them from the run's record before the line read the Lean's record,
with no bursts (the reading of gate runs up to 7), its groups as rows in `line.GROUP`'s order, beside the recorded
shares and line (from a run that reran each run's `auditor_line.analyse` through the Lean)."""
from __future__ import annotations

import copy
import dataclasses
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
from verity.core.protocols.remote_bandwidth import audit, codec, line, receipts  # noqa: E402

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
REGISTRATION = audit.Registration(b"node-under-test", 1000, bytes(64), bytes(32), codec.DEFAULT, receipts.DEFAULT,
                                  False, receipts.LAYOUT, receipts.sketch_rows(receipts.DEFAULT))
RECORD = {**{k: INPUTS[k] for k in ("per", "block_bytes", "chip_bytes", "node_bytes", "delta", "rates")},
          "registration": lv.registration_json(REGISTRATION), "samples": 1500,
          "seen": [{k: g[k] for k in line.SEEN} for g in GROUPS],
          "record": {"windows": [{"lengths": [1000, 1000], "steps": [2 * t, 2 * t + 1], "post_seed": [0, 0],
                                  "heads": [None, "ab" * 64], "stale": [1000, 1000],
                                  "arrived_ns": [g["last_ns"] - 1, g["last_ns"]], "released_ns": g["released_ns"],
                                  "deadline_ns": g["last_ns"] + 5, "opened_ns": g["last_ns"] + 1, "drawn": 1500}
                                 for t, g in enumerate(GROUPS)],
                     "blocks": 1000, "appended": 0, "released": 0, "lifetime": 2, "pace": {"k": 2, "window_ns": 10 ** 9},
                     "appends": None, "registration": REGISTRATION.digest.hex()}}
"""`INPUTS`' session read off its record of `REGISTRATION`: window t group t, two passes over the 1000 chunks, 1500 of
its 2000 receipts drawn (`line.record_inputs_json`'s form)."""


@pytest.fixture(scope="module")
def lean():
    p = lv.Lean(lv.build(), lambda _: None)
    yield p
    p.close()


class Answers:
    """A Lean verifier whose `line` and `line_record` answer what `change` makes of line.py's line of the request's
    inputs."""

    def __init__(self, change):
        self.change = change

    def call(self, request: dict, body: bytes = b"") -> dict:
        run = {"line": line.run_json, "line_record": line.record_run_json}[request["op"]]
        return self.change(run({k: v for k, v in request.items() if k != "op"}))


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


def test_the_lean_reads_the_line_off_the_record_as_line_py_does(lean):
    """Read off the record (`line_record`), the Lean's line is line.py's, and both are the line of the record's groups
    and windows given; the record's own refusals, a record not of the registration given among them, are both's, word
    for word, and a moved answer is a disagreement."""
    got = al.line_of_record(RECORD, lean)
    assert got == line.record_run_json(RECORD) == al.line_of_record(RECORD, None)
    assert got == line.run_json({**INPUTS, "windows": [[2000, 1500]] * len(GROUPS)})
    assert all(0 < al.shares_of(got)[f]["time"] < 1 for f in line.FORMS)
    other = lv.registration_json(dataclasses.replace(REGISTRATION, payload_root=bytes([1]) * 64))
    for bad, why in (({**RECORD, "seen": RECORD["seen"][:-1]}, "the auditor saw 6 windows, not the record's 7"),
                     ({**RECORD, "samples": 1499}, "window 0 drew 1500 of its 2000 receipts, not min(1499, 2000)"),
                     ({**RECORD, "registration": other}, "the record is not of the registration given"),
                     ({**RECORD, "record": {**RECORD["record"], "blocks": 999}},
                      "the record's 999 registered blocks are not the registration's 1000")):
        with pytest.raises(line.Refused, match=f"^{re.escape(why)}$"):
            al.line_of_record(bad, lean)
    with pytest.raises(al.Disagree, match=f"disagree at .{line.TIME_HELD}.time.weights_share"):
        al.line_of_record(RECORD, Answers(moved))


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
"""The recorded shares (session, form, share) a float apart from the exact share's nearest float: gate runs 5-7 were
analysed in floats, before line.py, and these four came out the next float over, no nearer the exact share."""


@pytest.mark.parametrize("g", [5, 6, 7])
def test_the_leans_line_is_the_record_of_gate_runs_5_to_7(lean, g):
    """Each credited session's shares, both forms, over time and at the passes, weights and filler, with its worst
    stretch, through the Lean's line are the recorded floats bit for bit once the half chunk the chips' rounding up
    adds is given back: the record charged S = chip_bytes/b = 743590.5 chunks, the line charges 743591, so every
    share above zero is the recorded one's exact value less (S·b - chip_bytes)/node_bytes, and the worst stretch is
    the same. That holds but for `ULP`'s, each the float beside the exact share's nearest and no nearer it; and the
    line at each T is the recorded one, its session the record's."""
    run = json.loads(gzip.decompress((FIXTURES / f"gate_run_{g}.json.gz").read_bytes()))
    assert run["gate_run"] == g and run["rates"] == al.rates_at(bursts="none")
    got, off = {}, set()
    for s in run["sessions"]:
        inputs = {**s["inputs"], "groups": [dict(zip(line.GROUP, x)) for x in s["inputs"]["groups"]]}
        b, chip = inputs["block_bytes"], inputs["chip_bytes"]
        half = Fraction(-(-chip // b) * b - chip, inputs["node_bytes"])
        assert half * inputs["node_bytes"] == b // 2, s["k"]
        got[s["k"]] = al.shares_of(al.line_of_record(inputs, lean))
        for old, form in OLD.items():
            mine, want = got[s["k"]][form], s["recorded"][old]
            for x, w in want.items():
                if x not in al.SHARES:
                    assert mine[x] == w, (s["k"], form, x)
                    continue
                share = line.number(mine["exact"][x])
                exact = share + half if share else share
                if float(exact) != w:
                    off.add((s["k"], form, x))
                    assert w == math.nextafter(float(exact), w), (s["k"], form, x)
                    assert abs(Fraction(w) - exact) > abs(Fraction(float(exact)) - exact), (s["k"], form, x)
    assert off == ULP[g]
    for t, row in run["line"].items():
        at = [s for s in run["sessions"] if s["T"] == t]
        assert row["credited_sessions"] == [s["k"] for s in at]
        for old, form in OLD.items():
            k = min((s["k"] for s in at), key=lambda k: line.number(got[k][form]["exact"]["time"]))
            recorded = next(s["recorded"][old] for s in at if s["k"] == k)
            assert row[old] == {"k": k, **{x: recorded[x] for x in row[old] if x != "k"}}, (t, form)
