"""`edge`'s registration of a window's history: window w commits each edge out-port it returns under its own program and open
root and names it among its roots, window w + 1 registers it under that root and domain and its verifier gives it as a history
port (`{"port": name}`, the previous record's port), and core's one-stage check (`registration.check`) accepts the honest two
windows and refuses tampered history by name."""
from __future__ import annotations

import pytest

from tests.program.test_workload_program import _edge_wl, _groups
from verity.protocols.spot.verifier.one_stage import partition as P, registered as REG, registration as R
from verity.circuits.codec import descriptor_id, program_digest
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.partition_object import calls
from verity.circuits.types import Array, Value, field_starts, field_types
from verity_vllm.pipeline import edge as E
from verity_vllm.program import workload as W

SCHEME = "frame-v3-sha512/hm96-sha512"
LAW = {"law": "subset", "k": 2}
SERVED = {"kind": "served", "run": "r-edge"}
OPEN = ("a1" * 64, "b2" * 64)                    # each window's open root, from the challenger's signed open
WORDS = {p: 1 for p in ("edge.kv", "edge.ik", "edge.iks", "edge.last")}    # the toy's rows: one leaf each
PORTS = ("edge.kv", "edge.ik", "edge.iks", "edge.last")


def _windows() -> tuple[W.WorkloadProgram, W.WorkloadProgram]:
    """Window w leaves r1 inside its prompt (rows [0, 2)) and r2 past it (rows [0, 4) and its last token); window w + 1 carries
    both on, reading their history from its edge ports. Every request reads the weights, so they are bound at registration
    (`theta_registered`): a run Input two proof blocks read is refused (R8-shared-input)."""
    w0 = _edge_wl({0: (3, 0, 0, False), 1: (3, 0, 2, False), 2: (3, 0, 4, False)}, W.Edge(outs=((1, _groups(2)), (2, _groups(4)))),
                  theta_registered=True)
    w1 = _edge_wl({1: (3, 2, 0, False), 2: (3, 4, 0, True)}, W.Edge(ins=((1, _groups(2)), (2, _groups(4)))), theta_registered=True)
    return w0, w1


def _query(wl: W.WorkloadProgram) -> dict:
    """The window's units: each request's call."""
    return P.template_instances_query(list(dict.fromkeys(descriptor_id(fn) for _, fn in calls(wl.program))))


def _own_ports(wl: W.WorkloadProgram, run: str) -> dict:
    """The query's own ports, as the verifier's layout derives them."""
    part, n = P.partition(wl.program, _query(wl)), P.units(wl.program, _query(wl))
    return {p: {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), p, schema, n, run)}
            for p, schema in (("a", "hm96-sha512/row/v1"), ("out", "u16"))}


def _served(wl: W.WorkloadProgram, inputs: list) -> dict[str, list[int]]:
    """Window w's edge out-ports as it computed them: the last fields of its evaluated return, one per port."""
    ret, out = wl.fn.signature()[1], evaluate(wl.fn, *inputs)
    starts, types = field_starts(ret), field_types(ret)
    k = len(types) - len(PORTS)
    return {p: list(out[starts[k + i]:starts[k + i] + types[k + i].leaves]) for i, p in enumerate(PORTS)}


@pytest.fixture(scope="module")
def two():
    w0, w1 = _windows()
    own0, own1 = _own_ports(w0, OPEN[0]), _own_ports(w1, OPEN[1])
    com = E.commit(w0.program, _served(w0, [[1, 2, 3, 4], [10, 11, 12], [20, 21, 22], [30, 31, 32]]), WORDS, OPEN[0])
    ports0 = {**own0, **E.out_ports(w0.program, PORTS, WORDS, OPEN[0])}
    rec0 = R.record(w0.program, _query(w0), LAW, SCHEME, [{"port": p, "root": "ab" * 64, **own0[p]} for p in own0] + E.roots(com),
                    {**SERVED, "row": 0})
    roots1 = [{"port": p, "root": "cd" * 64, **own1[p]} for p in own1]
    rec1 = R.record(w1.program, _query(w1), LAW, SCHEME, roots1, {**SERVED, "row": 1}, previous=R.digest(rec0), registered=E.entries(com))
    kw = {"law": LAW, "scheme": SCHEME, "ports": own1, "window_root": OPEN[1], **E.history(w1.ports, WORDS)}
    return {"w0": w0, "w1": w1, "com": com, "ports0": ports0, "rec0": rec0, "rec1": rec1, "roots1": roots1, "kw": kw}


def _check1(t: dict, rec: dict, log: list) -> R.Verdict:
    return R.check(rec, t["w1"].program, _query(t["w1"]), log=log, **t["kw"])


def test_an_honest_two_window_serve_registers_and_verifies_its_history(two):
    """Window w's record names each edge out-port at the root, leaves and domain its verifier derives from the Program
    (`out_ports`); window w + 1 takes exactly those ports as registered values given as history ports, registers each under
    w's root and domain, and both records pass R1-R7; the history w + 1 reads is the rows w committed."""
    t = two
    w0, w1 = t["w0"], t["w1"]
    assert w1.ports == PORTS
    assert [p for p, _, _, _ in W.edge_ports(w0.edge.static()["out"])] == list(PORTS)
    assert E.history(w1.ports, WORDS)["outputs"] == {p: {"port": p} for p in PORTS}
    assert R.check(t["rec0"], w0.program, _query(w0), LAW, SCHEME, ports=t["ports0"]).ok
    v = _check1(t, t["rec1"], [t["rec0"]])
    assert v.ok, v.codes
    held = {e["value"]: e for e in t["rec1"]["registered"]}
    for r in E.roots(t["com"]):
        assert (held[r["port"]]["root"], held[r["port"]]["domain"], held[r["port"]]["leaves"]) == (r["root"], r["domain"], r["leaves"])
    hist = {p: [x for row in t["com"][p].rows for x in row] for p in PORTS}
    # r1's rows [0, 2) and r2's [0, 4) of its prompt (20.. and 30..) plus each group's weight (2, 3, 4), and r2's sample 30 + 1
    assert hist == {"edge.kv": [22, 23, 32, 33, 34, 32], "edge.ik": [23, 24, 33, 34, 35, 33], "edge.iks": [24, 25, 34, 35, 36, 34],
                    "edge.last": [31]}
    # window w + 1's samples read that history: r1's 11 + 25 (its last carried scale), r2's (31 + 1) + 34
    assert list(evaluate(w1.fn, [1, 2, 3, 4], *(hist[p] for p in PORTS), [10, 11, 12], [20, 21, 22])) == [36, 66]


def test_window_ws_verifier_refuses_an_edge_port_under_another_domain(two):
    """An out-port committed under no window, or under another window's open root, is not the domain window w's verifier
    derives: R3-domain."""
    t = two
    for window in (None, OPEN[1]):
        other = E.roots({"edge.last": REG.Registered.commit(t["com"]["edge.last"].program, "edge.last", t["com"]["edge.last"].rows,
                                                            word_bits=16, window=window)})
        roots = [r if r["port"] != "edge.last" else other[0] for r in t["rec0"]["roots"]]
        assert R.check({**t["rec0"], "roots": roots}, t["w0"].program, _query(t["w0"]), LAW, SCHEME, ports=t["ports0"]).codes == ("R3-domain",)


def _with(t: dict, port: str, entry: dict) -> dict:
    return {**t["rec1"], "registered": [entry if e["value"] == port else e for e in t["rec1"]["registered"]]}


def _relinked(t: dict, rec0: dict) -> tuple[dict, list]:
    """Window w + 1's record chained after another window w ``rec0``, and the log that holds it last."""
    return {**t["rec1"], "previous": R.digest(rec0)}, [rec0]


def test_tampered_history_is_refused_by_name(two):
    """Each way a prover could hand window w + 1 history window w didn't commit is refused: other values committed under w's
    domain, w's values under fresh salts (a fresh root), the value registered afresh under w + 1's own domain, a previous
    record whose port has another leaf count or that lacks the port, and no previous record at all."""
    t = two
    kv = t["com"]["edge.kv"]
    rows = [list(r) for r in kv.rows]
    rows[0][0] ^= 1
    tampered = REG.Registered.commit(kv.program, "edge.kv", rows, word_bits=16, window=OPEN[0]).entry()
    assert tampered["domain"] == kv.entry()["domain"]
    assert _check1(t, _with(t, "edge.kv", tampered), [t["rec0"]]).codes == ("R7-held",)

    fresh = REG.Registered.commit(kv.program, "edge.kv", kv.rows, word_bits=16, window=OPEN[0]).entry()
    assert fresh["root"] != kv.entry()["root"]
    assert _check1(t, _with(t, "edge.kv", fresh), [t["rec0"]]).codes == ("R7-held",)

    own = REG.Registered.commit(program_digest(t["w1"].program), "edge.kv", kv.rows, word_bits=16, window=OPEN[1]).entry()
    assert _check1(t, _with(t, "edge.kv", own), [t["rec0"]]).codes == ("R7-domain", "R7-held")

    # with no previous port to take it from, the value's domain is no history domain either (R7-domain beside R7-output)
    short = {**t["rec0"], "roots": [{**r, "leaves": r["leaves"] - 1} if r["port"] == "edge.kv" else r for r in t["rec0"]["roots"]]}
    lacking = {**t["rec0"], "roots": [r for r in t["rec0"]["roots"] if r["port"] != "edge.last"]}
    for port, (rec, log) in (("edge.kv", _relinked(t, short)), ("edge.last", _relinked(t, lacking)),
                             (None, ({**t["rec1"], "previous": None}, []))):
        v = _check1(t, rec, log)
        assert v.codes == ("R7-output", "R7-domain") and port in (None, v.detail["R7-output"]["value"])


class _Dsa:
    """A stand-in decoder with the DSA history layout: per layer and position a latent row, per indexer layer and position a
    key row and a scale."""

    def prefix_params(self, C: int) -> list:
        return [("prefix", Array(3, Array(C, Array(576, Value(16))))), ("prefix_ik", Array(2, Array(C, Array(128, Value(8))))),
                ("prefix_iks", Array(2, Array(C, Value(32))))]


def test_a_row_is_one_position_of_one_layer_of_its_group():
    assert E.row_words(_Dsa()) == {"edge.kv": 576, "edge.ik": 128, "edge.iks": 1, "edge.last": 1}
