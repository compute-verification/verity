"""`line.py`'s memory line held to its Lean (`RemoteAudit.Line`, served by `remote-audit-difftest`'s `line`), refusals
as outputs so that both sides' reasons are compared word for word: sessions of zero, one and many groups at every
stretch length, with bursts, the host tier, filler, wrong receipts (the pooled bound of drawn windows, `pooled`) that
fill one group or spill into the next, ties between stretches, and each refusal (the windows', the node's inputs, a
group's, a stretch's empty span), with malformed inputs rejected by both. The same line read off an accepted session's
record (`line_record`, `line.line_of_record`: its groups' passes, its windows and its registered chunks the record's,
what the auditor saw of each window beside it; drawn g > 1 windows a coin, its samples in place of its windows), with
the record's own refusals (a record not of the registration given with it, the auditor saw another count of windows,
a window or a sample drew other than min(s, U), a record drawn g > 1 a coin at one window in flight) and malformed
records and registrations. The line of record is the Lean's
(`auditor_line.py` runs the verifier process's `line_record` and holds it exactly to the Python's), so the Lean is of
record here."""

from __future__ import annotations

import json
import shutil
from fractions import Fraction

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import audit, line, receipts

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

N, B = 1000, 4096


def port(inputs=st.nothing(), reference=line.run_json) -> D.Port:
    return D.Port(M.EXE, "line", reference, inputs, package=M.PACKAGE, of_record="lean", rejects=M.REFUSALS)


def record_port(inputs=st.nothing(), reference=line.record_run_json) -> D.Port:
    return D.Port(M.EXE, "line_record", reference, inputs, package=M.PACKAGE, of_record="lean", rejects=M.REFUSALS)


server = D.server_fixture(port())


def refused(why: str) -> tuple[str, str]:
    return "output", D.canonical({"refused": why})


def rational(top: int) -> st.SearchStrategy:
    """A nonnegative rational in the JSON form: an integer, `n/d`, or a float's decimal."""
    return st.one_of(st.integers(0, top), st.fractions(0, top, max_denominator=10 ** 4).map(line.text),
                     st.floats(0, top, allow_nan=False).map(lambda x: line.text(line.number(x))))


FAULTS = ("least", "read", "weights", "passes", "last", "coin", "rate", "windows", "delta", "encode_squarings", "per",
          "block_bytes", "blocks", "node_bytes")
DELTAS = ("1/1099511627776", line.text(Fraction(1e-9)), "1/3", "999/1000")


@st.composite
def windows(draw, top: int) -> list[list[int]]:
    """A record's windows, each [U, s] with 1 ≤ s ≤ U ≤ `top`: s small enough that some receipts may be wrong, or
    every receipt drawn (none can be)."""
    out = []
    for _ in range(draw(st.integers(1, 3))):
        u = draw(st.integers(1, top))
        out.append([u, draw(st.one_of(st.integers(1, min(u, 30)), st.just(u)))])
    return out


@st.composite
def inputs(draw) -> dict:
    """A paced session's groups and a node, as they are or with one fault (most often neither: Hypothesis leans to
    each choice's first)."""
    per = draw(st.sampled_from([2, 3, 4, 1]))
    n_groups = draw(st.sampled_from([2 * per, 3 * per, 4 * per + 1, 2 * per + 1, 2 * per - 1, per, 1, 0]))
    blocks, b = draw(st.integers(1, 40)), draw(st.sampled_from([B, 7, 1]))
    t, groups = draw(st.integers(0, 10 ** 12)), []
    for _ in range(n_groups):
        n = draw(st.integers(1, 3))
        hbm = draw(st.integers(0, n * blocks))
        groups.append({"released_ns": t, "last_ns": t + draw(st.integers(0, 10 ** 4)), "passes": n, "least": blocks,
                       "weights": draw(st.integers(0, hbm)), "hbm": hbm, "host": n * blocks - hbm})
        t += draw(st.sampled_from([10 ** 4, 10 ** 3, 1, 0])) + draw(st.integers(0, 10 ** 4))
    rates = {"squarings_per_s": draw(rational(10 ** 6)), "encode_squarings": draw(st.integers(1, 10)),
             **{k: draw(rational(10 ** 5)) for k in line.RATES[1:3]},
             **{k: draw(rational(10)) for k in line.RATES[3:]}}
    v = {"groups": groups, "per": per, "blocks": blocks, "block_bytes": b,
         "chip_bytes": draw(st.integers(0, 3 * b)), "node_bytes": draw(st.integers(1, 10 ** 6)),
         "windows": draw(st.one_of(windows(blocks), windows(4 * 3 * blocks * (per + 1)))),
         "delta": draw(st.sampled_from(DELTAS)), "rates": rates}
    fault = {4: draw(st.sampled_from(FAULTS)), 5: "malformed"}.get(draw(st.integers(0, 5)))
    g = draw(st.integers(0, max(n_groups - 1, 0)))
    if fault in ("per", "block_bytes", "blocks", "node_bytes"):
        v[fault] = 0
    elif fault == "encode_squarings":
        rates["encode_squarings"] = 0
    elif fault == "rate":
        rates[draw(st.sampled_from(line.RATES))] = draw(st.sampled_from([-1, "-1/3"]))
    elif fault == "windows":
        t = draw(st.integers(0, len(v["windows"]) - 1))
        v["windows"] = draw(st.sampled_from([[], v["windows"][:t] + [[3, 0]], v["windows"][:t] + [[2, 3]]]))
    elif fault == "delta":
        v["delta"] = draw(st.sampled_from([0, 1, "-1/3", "3/2"]))
    elif fault == "malformed":
        k = draw(st.sampled_from(["per", "windows", "delta", "window", "rates", "groups", "group", "squarings_per_s"]))
        bad = draw(st.sampled_from([-1, "1/0", "1.5", "+3", None, True, [], {}]))
        if k == "window":
            v["windows"] = [*v["windows"][:-1], draw(st.sampled_from([[1], [1, 2, 3], [bad, 1], [2, bad]]))]
        elif k == "group" and groups:
            groups[g] = {**groups[g], draw(st.sampled_from(line.GROUP)): bad}
        elif k == "squarings_per_s":
            rates[k] = bad
        elif k != "group":
            v[k] = bad
        if draw(st.booleans()):
            v.pop(draw(st.sampled_from(sorted(v))))
    elif fault is not None and groups:
        x = groups[g]
        if fault == "passes":
            groups[g] = {**x, "passes": 0}
        elif fault == "last" and x["released_ns"]:
            x["last_ns"] = x["released_ns"] - 1
        elif fault == "coin" and g:
            x["released_ns"] = groups[g - 1]["released_ns"] - 1
            x["last_ns"] = max(x["last_ns"], x["released_ns"])
        elif fault == "least":
            x["least"] = draw(st.integers(0, 2 * blocks).filter(lambda k: k != blocks))
        elif fault == "read":
            x["host"] += 1
        elif fault == "weights":
            x["weights"] = x["hbm"] + 1
    return v


def test_line(server) -> None:
    told = {"line": 0, "credited": 0, "refused": 0}

    def reference(v):
        out = line.run_json(v)
        told["refused" if "refused" in out else "line"] += 1
        told["credited"] += "refused" not in out and line.number(out[line.TIME_HELD]["time"]["weights_share"]) > 0
        return out

    stats = D.run(port(inputs(), reference), examples=500, server=server)
    assert stats.rejects > 20 and told["line"] > 100 and told["credited"] > 15 and told["refused"] > 100, (stats, told)


def session(stretch: int = 1, *, weights: int = 600, host: int = 0, n: int = 12, windows=((1, 1),),
            delta: object = 2.0 ** -40, rates: line.Rates = line.Rates(0, 1), **over) -> dict:
    """`n` groups of two passes over N chunks, 10 ns apart but group 5, `stretch` times that, each one's last root at
    the next coin, its record's `windows` at `delta` (by default one opened whole, so no receipt is wrong)."""
    gs, t = [], 0
    for g in range(n):
        span = 10 * (stretch if g == 5 else 1)
        gs.append(line.Group(t, t + span, 2, N, 2 * weights, 2 * (N - host), 2 * host))
        t += span
    return line.inputs_json(gs, per=4, blocks=N, block_bytes=B, chip_bytes=0, node_bytes=10 * N * B, windows=windows,
                            delta=delta, rates=rates) | over


def test_the_line_is_the_leans_at_its_edges(server) -> None:
    """A burst in one group, bursts and every rate charged, the host tier and filler only, wrong receipts that fill
    the stretch's slowest groups and spill, stretches that tie, and the shortest session with a line."""
    for v in (session(9, rates=line.Rates(10 ** 6, 1)),
              session(9, windows=[(5000, 10)], rates=line.Rates(3 * 10 ** 6, 3, 4.4e9, 5e10, B, 4.4e9, 6291456)),
              session(1, weights=0, host=400, windows=[(10 ** 7, 1)], rates=line.Rates(0, 1, 0, 0, 7, 0, 3)),
              session(1), session(1, n=8, chip_bytes=5 * B)):
        out = D.replay(port(), v, server)
        assert out[0] == "output" and "refused" not in json.loads(out[1]), v
    assert line.run_json(session(9, windows=[(5000, 10)]))["wrong"] == "9375/2"


@pytest.mark.parametrize("over,why", [
    (dict(block_bytes=0), "a block is one or more bytes"),
    (dict(blocks=0), "the store is one or more blocks"),
    (dict(node_bytes=0), "the node is one or more bytes"),
    (dict(rates=line.Rates(1, 0).to_json()), "a block's encode is one or more squarings"),
    (dict(rates=line.Rates(-1, 1).to_json()), "squarings_per_s is nonnegative"),
    (dict(rates=line.Rates(1, 1, disk_bytes_per_s=Fraction(-1, 2)).to_json()), "disk_bytes_per_s is nonnegative"),
    (dict(rates=line.Rates(1, 1, link_bytes_per_s=-1).to_json()), "link_bytes_per_s is nonnegative"),
    (dict(rates=line.Rates(1, 1, rebuild_burst_bytes=-1).to_json()), "rebuild_burst_bytes is nonnegative"),
    (dict(rates=line.Rates(1, 1, disk_burst_bytes=-1).to_json()), "disk_burst_bytes is nonnegative"),
    (dict(rates=line.Rates(1, 1, link_burst_bytes=-1).to_json()), "link_burst_bytes is nonnegative"),
    (dict(windows=[]), "a session has one or more windows"),
    (dict(delta="0"), "δ is above 0 and below 1"),
    (dict(delta=1), "δ is above 0 and below 1"),
    (dict(windows=[[5, 2], [3, 0]]), "window 1 draws no receipt"),
    (dict(windows=[[5, 2], [2, 3]], per=0), "window 1 draws 3 receipts of its 2"),
    (dict(per=0), "a stretch is one or more groups"),
    (dict(per=1), "stretch 0's span is empty: group 1's coin left no later than group 0's end"),
    (dict(per=7), "the session's 12 groups at 7 a stretch leave no stretch before the closing one"),
    (dict(groups=[]), "the session's 0 groups at 4 a stretch leave no stretch before the closing one"),
])
def test_each_refusal_of_the_node_is_the_leans_word_for_word(server, over, why) -> None:
    v = session() | over
    assert line.run_json(v) == {"refused": why}
    assert D.replay(port(), v, server) == refused(why)


@pytest.mark.parametrize("g,over,why", [
    (3, dict(passes=0, hbm=0, weights=0), "group 3 has no pass"),
    (3, dict(last_ns=29), "group 3's last root came before its coin"),
    (3, dict(released_ns=15, last_ns=40), "group 3's coin left before group 2's"),
    (2, dict(least=990), "group 2's least pass read 990 of the 1000 registered blocks: the paced line needs every "
                         "pass to read every registered block, since a block a pass leaves out can be absent across "
                         "it with neither a wrong read nor a rebuild"),
    (2, dict(hbm=1999), "group 2's 2 passes read 1999 registered blocks in all, not 2000"),
    (2, dict(weights=2001, hbm=1999, host=1), "group 2's passes read more weights than HBM, which holds them"),
    (8, dict(released_ns=40, last_ns=80), "stretch 1's span is empty: group 8's coin left no later than group 4's "
                                          "coin"),
])
def test_each_refusal_of_a_group_is_the_leans_word_for_word(server, g, over, why) -> None:
    v = session()
    v["groups"][g] |= over
    if g == 8:
        for x in v["groups"][4:8]:
            x["released_ns"], x["last_ns"] = 40, 40
    assert line.run_json(v) == {"refused": why}
    assert D.replay(port(), v, server) == refused(why)


def test_both_lines_charge_the_chips_in_whole_blocks_rounded_up(server) -> None:
    """One byte over 5 chunks of chip is charged as 6 chunks, in the Python and the Lean alike."""
    over, six = session(1, n=8, chip_bytes=5 * B + 1), session(1, n=8, chip_bytes=6 * B)
    assert line.run_json(over) == line.run_json(six) != line.run_json(session(1, n=8, chip_bytes=5 * B))
    six_blocks = ("output", D.canonical(line.run_json(six)))
    assert D.replay(port(), over, server) == D.replay(port(), six, server) == six_blocks


def registration(blocks: int, root: bytes = bytes(64)) -> audit.Registration:
    """A registration of `blocks` chunks, its codec seed given, as `line_record` reads one (`M.registration_of`)."""
    return audit.Registration(M.NODE, blocks, root, bytes(32), M.P512, receipts.DEFAULT, False, receipts.LAYOUT,
                              receipts.SKETCH_ROWS)


def recorded(v: dict, samples: int = 1) -> dict:
    """`session`'s inputs read off a record of `registration(N)` (`line_record`'s form): window t is group t, each of
    its passes reading the N registered chunks, drawn `min(samples, U_t)`, the rest of the record well formed; what the
    auditor saw of it each group's times and reads."""
    ws = []
    for t, g in enumerate(v["groups"]):
        k = g["passes"]
        ws.append({"lengths": [N] * k, "steps": list(range(3 * t, 3 * t + k)), "post_seed": [0] * k,
                   "heads": [None, "ab" * 64][:k] + [None] * (k - 2), "stale": [N] * k,
                   "arrived_ns": [g["last_ns"]] * k, "released_ns": g["released_ns"], "deadline_ns": g["last_ns"] + 5,
                   "opened_ns": g["last_ns"] + 1, "drawn": min(samples, k * N)})
    reg = registration(v["blocks"])
    record = {"windows": ws, "blocks": v["blocks"], "appended": 0, "released": 0, "lifetime": 2,
              "pace": {"k": 2, "window_ns": 1}, "appends": None, "registration": reg.digest.hex()}
    return {"record": record, "registration": M.registration_json(reg), "samples": samples,
            "seen": [{k: g[k] for k in line.SEEN} for g in v["groups"]],
            **{k: v[k] for k in ("per", "block_bytes", "chip_bytes", "node_bytes", "delta", "rates")}}


INVALID = (("rows", 3), ("layout", "x"), ("bind", True), ("node", ""), ("node", "ab" * 65), ("blocks", 0),
           ("payload_root", "00" * 63), ("codec_seed", "00" * 31), ("receipt", "x"), ("receipt", receipts.BLOCK_SHA512),
           ("codec", {"field": "p256", "masked": False, "rounds": 2}),
           ("codec", {"field": "p512", "masked": False, "rounds": 1}))
"""A registration's fields as JSON gives them, each well formed but one `Registration.make?` refuses."""


@st.composite
def records(draw) -> dict:
    """An accepted session's record of a registration given with it and what the auditor saw of each window, as they
    are or with one fault: the record not of the registration given (another registration, or the record's chunks
    another count), the auditor seeing another count of windows, a window drawn other than min(s, U_t), one of the
    node's inputs `line` refuses, or a malformed record, registration, seen or input (most often none: Hypothesis leans
    to each choice's first)."""
    per = draw(st.sampled_from([2, 3, 1]))
    n_groups = draw(st.sampled_from([2 * per, 3 * per + 1, per, 1, 0]))
    blocks, b, samples = draw(st.integers(1, 40)), draw(st.sampled_from([B, 7])), draw(st.integers(1, 30))
    t, ws, seen = draw(st.integers(0, 10 ** 12)), [], []
    for _ in range(n_groups):
        lengths = draw(st.lists(st.integers(1, 4 * blocks), min_size=1, max_size=3))
        k, last = len(lengths), t + draw(st.integers(0, 10 ** 4))
        hbm = draw(st.one_of(st.just(k * blocks), st.integers(0, k * blocks)))
        seen.append({"released_ns": t, "last_ns": last, "least": blocks,
                     "weights": draw(st.one_of(st.just(hbm), st.integers(0, hbm))), "hbm": hbm,
                     "host": k * blocks - hbm})
        ws.append({"lengths": lengths, "steps": sorted(draw(st.sets(st.integers(0, 10 ** 6), min_size=k, max_size=k))),
                   "post_seed": [draw(st.integers(0, x)) for x in lengths],
                   "heads": draw(st.lists(st.one_of(st.none(), st.binary(min_size=64, max_size=64).map(bytes.hex)),
                                          min_size=k, max_size=k)),
                   "stale": [blocks] * k, "arrived_ns": [last] * k, "released_ns": t, "deadline_ns": last + 10,
                   "opened_ns": last + 1, "drawn": min(samples, sum(lengths))})
        t += draw(st.sampled_from([10 ** 4, 1, 0])) + draw(st.integers(0, 10 ** 4))
    rates = {"squarings_per_s": draw(rational(10 ** 6)), "encode_squarings": draw(st.integers(1, 10)),
             **{k: draw(rational(10 ** 5)) for k in line.RATES[1:3]},
             **{k: draw(rational(10)) for k in line.RATES[3:]}}
    pace = draw(st.one_of(st.none(), st.builds(lambda k, w, one: {"k": k, "window_ns": w} | ({"in_flight": 1} if one
                                                                                             else {}),
                                               st.integers(1, 3), st.integers(1, 10 ** 9), st.booleans())))
    reg = registration(blocks, draw(st.binary(min_size=64, max_size=64)))
    record = {"windows": ws, "blocks": blocks, "appended": draw(st.integers(0, 5)),
              "released": draw(st.integers(0, 5)), "lifetime": draw(st.integers(1, 4)), "pace": pace,
              "appends": draw(st.one_of(st.none(), st.binary(min_size=64, max_size=64).map(bytes.hex))),
              "registration": reg.digest.hex()}
    if (per_coin := draw(st.sampled_from([1, 2, 3]))) > 1:
        record["groups_per_coin"] = per_coin
        for a in range(0, len(ws), per_coin):
            xs = ws[a:a + per_coin]
            total = min(samples, sum(sum(w["lengths"]) for w in xs))
            cuts = sorted(draw(st.lists(st.integers(0, total), min_size=len(xs) - 1, max_size=len(xs) - 1)))
            for w, lo, hi in zip(xs, [0, *cuts], [*cuts, total]):
                w["drawn"] = hi - lo
    rj = M.registration_json(reg)
    v = {"record": record, "registration": rj, "samples": samples, "seen": seen, "per": per, "block_bytes": b,
         "chip_bytes": draw(st.integers(0, 3 * b)), "node_bytes": draw(st.integers(1, 10 ** 6)),
         "delta": draw(st.sampled_from(DELTAS)), "rates": rates}
    if (clock := draw(st.sampled_from([None, *line.CLOCKS]))) is not None:
        v["clock"] = clock
    fault = {3: "seen", 4: "drawn", 5: "malformed", 6: "times", 7: "registration"}.get(draw(st.integers(0, 7)))
    if fault == "registration":
        other = draw(st.sampled_from(["blocks", "root", "record"]))
        if other == "record":
            record["blocks"] = draw(st.integers(0, 2 * blocks).filter(lambda k: k != blocks))
        else:
            v["registration"] = M.registration_json(
                registration(blocks + 1) if other == "blocks" else registration(blocks, bytes([1]) * 64))
    elif fault == "times" and seen:
        draw(st.sampled_from(seen))[draw(st.sampled_from(["released_ns", "last_ns"]))] += 1
    elif fault == "seen":
        v["seen"] = draw(st.sampled_from([seen[:-1], [*seen, seen[-1]]])) if seen else [
            {k: 0 for k in line.SEEN}]
    elif fault == "drawn" and ws:
        w = draw(st.sampled_from(ws))
        w["drawn"] = draw(st.integers(0, sum(w["lengths"]) + 1).filter(lambda d: d != w["drawn"]))
    elif fault == "malformed":
        bad = draw(st.sampled_from([-1, "1", None, True, [], {}, "xyz", "AB" * 64]))
        at = draw(st.sampled_from(["record", "window", "list", "heads", "pace", "appends", "seen", "input",
                                   "registration", "invalid"]))
        if at == "record":
            record[draw(st.sampled_from(["windows", "blocks", "appended", "released", "lifetime",
                                         "groups_per_coin", "registration"]))] = bad
        elif at == "registration":
            k = draw(st.sampled_from(sorted(rj)))
            if k == "codec" and draw(st.booleans()):
                rj["codec"] = {**rj["codec"], draw(st.sampled_from(sorted(rj["codec"]))): bad}
            else:
                rj[k] = bad
        elif at == "invalid":
            k, x = draw(st.sampled_from(INVALID))
            rj[k] = x
        elif at == "window" and ws:
            draw(st.sampled_from(ws))[draw(st.sampled_from(["released_ns", "deadline_ns", "opened_ns", "drawn"]))] = bad
        elif at == "list" and ws:
            w = draw(st.sampled_from(ws))
            k = draw(st.sampled_from(["lengths", "steps", "post_seed", "stale", "arrived_ns", "heads"]))
            w[k] = draw(st.sampled_from([bad, [*w[k], bad]]))
        elif at == "heads" and ws:
            draw(st.sampled_from(ws))["heads"] = [draw(st.sampled_from(["ab", "abc", "AB", 5]))]
        elif at == "pace":
            record["pace"] = draw(st.sampled_from([{"k": bad, "window_ns": 1}, {"k": 1}, bad]))
        elif at == "appends":
            record["appends"] = draw(st.sampled_from(["abc", "AB", 5]))
        elif at == "seen" and seen:
            x = draw(st.sampled_from(seen))
            x[draw(st.sampled_from(line.SEEN))] = bad
        elif at == "input":
            v[draw(st.sampled_from(["samples", "seen", "record", "clock", "registration"]))] = bad
        if draw(st.booleans()):
            target = draw(st.sampled_from([v, record, rj, *ws[:1], *seen[:1]]))
            target.pop(draw(st.sampled_from(sorted(target))))
    return v


def test_line_record(server) -> None:
    told = {"line": 0, "credited": 0, "refused": 0}

    def reference(v):
        out = line.record_run_json(v)
        told["refused" if "refused" in out else "line"] += 1
        told["credited"] += "refused" not in out and line.number(out[line.TIME_HELD]["time"]["weights_share"]) > 0
        return out

    stats = D.run(record_port(records(), reference), examples=600, server=server)
    assert stats.rejects > 20 and told["line"] > 60 and told["credited"] > 10 and told["refused"] > 60, (stats, told)


@pytest.mark.parametrize("samples", [1, 37, 2 * N])
def test_the_records_line_is_the_line_of_its_groups_and_windows(server, samples) -> None:
    """Read off a record, the line is `line`'s at the record's groups (its passes, the auditor's times and reads), its
    windows (Σ lengths_t, drawn_t) and its registered chunks, the Lean's and line.py's alike."""
    for v in (session(9, rates=line.Rates(10 ** 6, 1)), session(1, weights=0, host=400),
              session(9, rates=line.Rates(3 * 10 ** 6, 3, 4.4e9, 5e10, B, 4.4e9, 6291456))):
        r = recorded(v, samples)
        want = line.run_json(v | {"windows": [[2 * N, min(samples, 2 * N)]] * len(v["groups"])})
        assert "refused" not in want and line.record_run_json(r) == want
        assert D.replay(record_port(), r, server) == ("output", D.canonical(want))


def test_a_record_drawn_four_windows_a_coin_reads_its_draws(server) -> None:
    """Drawn four windows a coin (`groups_per_coin`), the record's line is `line`'s at the same groups and its three
    draws, each four windows' receipts and draws summed (`line.draws_of`, the pooled bound's `audit.pooled_windows`),
    the Lean's and line.py's alike; a sample that sampled other than min(s, U) is refused, word for word, and at one window
    in flight the record has no line (`line.SHARED`)."""
    v = session(9, windows=[(5000, 10)], rates=line.Rates(3 * 10 ** 6, 3, 4.4e9, 5e10, B, 4.4e9, 6291456))
    r = recorded(v, 37)
    r["record"]["groups_per_coin"] = 4
    for a in range(0, 12, 4):
        for k, w in enumerate(r["record"]["windows"][a:a + 4]):
            w["drawn"] = [20, 0, 17, 0][k]
    want = line.run_json(v | {"windows": [[8 * N, 37]] * 3})
    assert "refused" not in want and want != line.run_json(v | {"windows": [[2 * N, 37]] * 12})
    assert line.record_run_json(r) == want
    assert D.replay(record_port(), r, server) == ("output", D.canonical(want))
    r["record"]["windows"][5]["drawn"] = 1
    why = f"draw 1 drew 38 of its {8 * N} receipts, not min(37, {8 * N})"
    assert line.record_run_json(r) == {"refused": why} and D.replay(record_port(), r, server) == refused(why)
    r["record"]["pace"]["in_flight"] = 1
    assert line.record_run_json(r) == {"refused": line.SHARED}
    assert D.replay(record_port(), r, server) == refused(line.SHARED)
    r["record"]["groups_per_coin"] = 1
    why = "window 0 drew 20 of its 2000 receipts, not min(37, 2000)"
    assert line.record_run_json(r) == {"refused": why} and D.replay(record_port(), r, server) == refused(why)


def test_a_draw_of_two_to_the_64_receipts_is_refused(server) -> None:
    """Drawn four windows a coin, a sample whose windows' receipts, each below 2^64 as `receive` holds a window's, sum to
    2^64 or more is refused, the Lean's and line.py's word for word: the memory model's sample reads at most 2^64 − 1
    (`Properties.RemoteAudit.Line.receiptMax`). One receipt fewer, the sample is read."""
    v = session(9, windows=[(5000, 10)], rates=line.Rates(3 * 10 ** 6, 3, 4.4e9, 5e10, B, 4.4e9, 6291456))
    r = recorded(v, 37)
    r["record"]["groups_per_coin"] = 4
    for t, w in enumerate(r["record"]["windows"]):
        w["drawn"] = 37 if t % 4 == 0 else 0
    big = r["record"]["windows"][4:8]
    for w in big[:2]:
        w["lengths"][0] = (1 << 63) - N
    u = sum(sum(w["lengths"]) for w in big)
    assert u >= 1 << 64 and all(sum(w["lengths"]) < 1 << 64 for w in big)
    why = f"draw 1 has {u} receipts, not fewer than 2^64"
    assert line.record_run_json(r) == {"refused": why} and D.replay(record_port(), r, server) == refused(why)
    big[0]["lengths"][0] -= u - (1 << 64) + 1
    out = line.record_run_json(r)
    assert "refused" not in out and D.replay(record_port(), r, server) == ("output", D.canonical(out))


def short_seen(r: dict) -> None:
    r["seen"].pop()


def drew_five(r: dict) -> None:
    r["record"]["windows"][3]["drawn"] = 5


def no_blocks(r: dict) -> None:
    r["record"]["blocks"] = 0


def seven_a_stretch(r: dict) -> None:
    r["per"] = 7


def another_registration(r: dict) -> None:
    r["registration"] = M.registration_json(registration(N, bytes([1]) * 64))


def more_blocks(r: dict) -> None:
    r["registration"] = M.registration_json(registration(N + 1))


@pytest.mark.parametrize("change,why", [
    (short_seen, "the auditor saw 11 windows, not the record's 12"),
    (drew_five, "window 3 drew 5 of its 2000 receipts, not min(1, 2000)"),
    (no_blocks, "the record's 0 registered blocks are not the registration's 1000"),
    (seven_a_stretch, "the session's 12 groups at 7 a stretch leave no stretch before the closing one"),
    (another_registration, "the record is not of the registration given"),
    (more_blocks, "the record is not of the registration given"),
])
def test_each_refusal_of_a_record_is_the_leans_word_for_word(server, change, why) -> None:
    r = recorded(session())
    change(r)
    assert line.record_run_json(r) == {"refused": why}
    assert D.replay(record_port(), r, server) == refused(why)


def test_times_seen_are_held_to_the_records_on_its_clock_and_not_on_the_challengers(server) -> None:
    """On the record's clock (`clock` record, the default) a window seen released or last at other times than the
    record's is refused, the Lean's word for word; on the challenger's (its release log's) or a replay's the record
    doesn't hold the times, and the line is read."""
    r = recorded(session())
    w = r["record"]["windows"][2]
    r["seen"][2]["last_ns"] += 1
    why = (f"window 2 was seen released at {w['released_ns']} and last at {w['arrived_ns'][-1] + 1}, not at the "
           f"record's {w['released_ns']} and {w['arrived_ns'][-1]}")
    for x in (r, {**r, "clock": "record"}):
        assert line.record_run_json(x) == {"refused": why}
        assert D.replay(record_port(), x, server) == refused(why)
    for clock in ("challenger", "replay"):
        x = {**r, "clock": clock}
        got = line.record_run_json(x)
        assert "refused" not in got and D.replay(record_port(), x, server) == ("output", D.canonical(got))
