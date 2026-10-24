"""`line.py`'s memory line held to its Lean (`RemoteAudit.Line`, served by `remote-audit-difftest`'s `line`), refusals
as outputs so that both sides' reasons are compared word for word: sessions of zero, one and many groups at every
stretch length, with bursts, the host tier, filler, wrong receipts that fill one group or spill into the next, ties
between stretches, and each refusal (the node's inputs, a group's, a stretch's empty span), with malformed inputs
rejected by both. `auditor_line.py` runs the Python (`README.md`), so the Python is of record here."""

from __future__ import annotations

import json
import shutil
from fractions import Fraction

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import line

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

N, B = 1000, 4096


def port(inputs=st.nothing(), reference=line.run_json) -> D.Port:
    return D.Port(M.EXE, "line", reference, inputs, package=M.PACKAGE, of_record="python", rejects=M.REFUSALS)


server = D.server_fixture(port())


def refused(why: str) -> tuple[str, str]:
    return "output", D.canonical({"refused": why})


def rational(top: int) -> st.SearchStrategy:
    """A nonnegative rational in the JSON form: an integer, `n/d`, or a float's decimal."""
    return st.one_of(st.integers(0, top), st.fractions(0, top, max_denominator=10 ** 4).map(line.text),
                     st.floats(0, top, allow_nan=False).map(lambda x: line.text(line.number(x))))


FAULTS = ("least", "read", "weights", "passes", "last", "coin", "rate", "wrong", "encode_squarings", "per",
          "block_bytes", "blocks", "node_bytes")


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
         "wrong": draw(st.one_of(rational(blocks), rational(4 * 3 * blocks * (per + 1)))), "rates": rates}
    fault = {4: draw(st.sampled_from(FAULTS)), 5: "malformed"}.get(draw(st.integers(0, 5)))
    g = draw(st.integers(0, max(n_groups - 1, 0)))
    if fault in ("per", "block_bytes", "blocks", "node_bytes"):
        v[fault] = 0
    elif fault == "encode_squarings":
        rates["encode_squarings"] = 0
    elif fault == "rate":
        rates[draw(st.sampled_from(line.RATES))] = draw(st.sampled_from([-1, "-1/3"]))
    elif fault == "wrong":
        v["wrong"] = draw(st.sampled_from([-2, "-5/7"]))
    elif fault == "malformed":
        k = draw(st.sampled_from(["per", "wrong", "rates", "groups", "group", "squarings_per_s"]))
        bad = draw(st.sampled_from([-1, "1/0", "1.5", "+3", None, True, [], {}]))
        if k == "group" and groups:
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


def session(stretch: int = 1, *, weights: int = 600, host: int = 0, n: int = 12, wrong: object = 0,
            rates: line.Rates = line.Rates(0, 1), **over) -> dict:
    """`n` groups of two passes over N blocks, 10 ns apart but group 5, `stretch` times that, each one's last root at
    the next coin."""
    gs, t = [], 0
    for g in range(n):
        span = 10 * (stretch if g == 5 else 1)
        gs.append(line.Group(t, t + span, 2, N, 2 * weights, 2 * (N - host), 2 * host))
        t += span
    return line.inputs_json(gs, per=4, blocks=N, block_bytes=B, chip_bytes=0, node_bytes=10 * N * B, wrong=wrong,
                            rates=rates) | over


def test_the_line_is_the_leans_at_its_edges(server) -> None:
    """A burst in one group, bursts and every rate charged, the host tier and filler only, wrong receipts that fill
    the stretch's slowest groups and spill, stretches that tie, and the shortest session with a line."""
    for v in (session(9, rates=line.Rates(10 ** 6, 1)),
              session(9, wrong=Fraction(4001, 1), rates=line.Rates(3 * 10 ** 6, 3, 4.4e9, 5e10, B, 4.4e9, 6291456)),
              session(1, weights=0, host=400, wrong=10 ** 7, rates=line.Rates(0, 1, 0, 0, 7, 0, 3)),
              session(1), session(1, n=8, chip_bytes=5 * B)):
        out = D.replay(port(), v, server)
        assert out[0] == "output" and "refused" not in json.loads(out[1]), v


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
    (dict(wrong="-1/3"), "the wrong receipts are nonnegative"),
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
