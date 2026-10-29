"""The blocks check (`blocks`) on a TINY class circuit, from the window interface's records: S = 2 slots, each a request
row of NP + 1 = 16 words, and O = 2 output steps. `AmpereBF16TcDot16_v3`'s unit j·O + k reads slot j's request row (its
``a``, the circuit's external input), the loaded weights' row k (``b``) and the loaded bias (``acc``); `F32ToBf16Rn_v2`'s
unit u is wired to its output, and its ``o0`` is response row u (the circuit's external output). The network certifier on
the server's uplink commits and signs the load, the request and the response (`transfer_rows`), and the verifier's pins
come from its certificates alone (`blocks.pins`): with them, held to the window's roots its registrations record, the
circuit's free inputs are 0. The demo's own request and
response rows (`transfer_rows.request_rows`, `response_rows`) bind at the frozen shape on stand-in units (`demo`). Cut in
two servers, each inference block (the record's (slot, step) of each unit) runs on one of them."""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from dataclasses import replace

import pytest

from verity.ml.crypto import lms
from verity.ml.windows import interface as IF
from verity.core.protocols.certifier import blocks as BK
from verity.core.protocols.certifier import transfer_rows as TR
from verity.core.protocols.certifier.certificate import Certificate, Direction, Issuer, LmsKey

import signers

DOT, CAST = "AmpereBF16TcDot16_v3", "F32ToBf16Rn_v2"
S, NP, O, WINDOW = 2, 15, 2, 3
UPLINK, UPLINK1 = "s0/uplink", "s1/uplink"
RESPONSE = {"server": "s0", "dir": "out", "kind": "response", "value": "response", "window": WINDOW,
            "t": {"base": 1, "radix": [S, O], "step": [0, 1]}, "leaves": S * O,
            "writes": [{"writer": CAST, "out": "o0", "at": 0, "radix": [S * O], "base": 0, "step": [1]}]}


def circuit(wired: bool = True, reads: tuple = (), params: dict | None = None, **kw) -> IF.Circuit:
    """The TINY class circuit: its loads (``w``, ``bias``) and request read by the dot's units, the cast wired to them
    (``wired``) or reading ``reads``."""
    base = (IF.Read(DOT, "a", "request", 0, (S, O), 0, (NP + 1, 0)),
            IF.Read(DOT, "b", "w", 0, (S, O), 0, (0, 16)),
            IF.Read(DOT, "acc", "bias", 0, (S * O,), 0, (0,)))
    wires = (IF.Wire(CAST, "a", DOT, "o0", 0, (S * O,), 0, (1,)),) if wired else ()
    reg = IF.Registered({"request": S * (NP + 1), "w": O * 16, "bias": 2, **(params or {})}, reads=base + tuple(reads))
    return IF.Circuit("tiny-class", ((IF.resolve(DOT), S * O), (IF.resolve(CAST), S * O)),
                      external=kw.pop("external", (f"{DOT}/a", f"{CAST}/o0")), wires=wires, registered=reg, **kw)


def record(c: IF.Circuit) -> dict:
    return IF.descriptor_record(c, cap=S * O)


def roots(regs: list[dict]) -> dict[str, str]:
    """The window's roots, by value, as its registrations record them (`interface.input_count` holds a pin to them)."""
    return {r["value"]: r["root"] for r in regs}


def layouts(c: IF.Circuit) -> dict:
    return {t: lay for t, (lay, _n) in c.layouts.items()}


def blocks() -> dict:
    return {"format": BK.FORMAT, "server": [{"name": "s0", "uplink": UPLINK, "units": "all"}],
            "server_time": {"window": WINDOW, "steps": O + 1},
            "transfers": [{"server": "s0", "dir": "in", "kind": "load", "value": "w", "window": 0, "t": None},
                          {"server": "s0", "dir": "in", "kind": "load", "value": "bias", "window": 0, "t": None},
                          {"server": "s0", "dir": "in", "kind": "request", "value": "request", "window": WINDOW,
                           "t": {"base": 0, "radix": [S], "step": [0]}},
                          copy.deepcopy(RESPONSE)]}


WIRE_OUT = {"server": "s0", "dir": "out", "kind": "wire", "value": "h", "window": WINDOW,
            "t": {"base": 1, "radix": [S * O], "step": [0]}, "leaves": 2 * S * O,
            "writes": [{"writer": DOT, "out": "o0", "at": 0, "radix": [S * O], "base": 0, "step": [2]}]}
WIRE_IN = {"server": "s1", "dir": "in", "kind": "wire", "value": "h", "window": WINDOW,
           "t": {"base": 1, "radix": [S * O], "step": [0]}}


def inference(cast: int) -> IF.Blocks:
    """The circuit's inference blocks: the dot's unit j·O + k is slot j's step k, and the cast's unit j·O + k slot j's step
    ``cast`` + k (at 0 the dot's own block, at O blocks of its own after the dot's)."""
    return IF.Blocks(S, cast + O, (IF.InferenceEntry(DOT, 0, (S, O), ("slot", "step"), 0),
                                   IF.InferenceEntry(CAST, 0, (S, O), ("slot", "step"), cast)))


def two_servers(cast: int = O) -> tuple[IF.Circuit, dict]:
    """The circuit cut in two servers, the dot's units s0's and the cast's s1's, the cast in inference blocks of its own
    (`inference`; at ``cast`` 0 in the dot's, so every inference block is on both servers): the dot's outputs cross from
    s0 to s1 as a wire ``h``, s0's egress (transfers[4]) and s1's ingress (transfers[5]), each under its uplink's
    certifier."""
    c = circuit(wired=False, reads=(IF.Read(CAST, "a", "h", 0, (S * O,), 0, (2,)),), params={"h": 2 * S * O},
                blocks=inference(cast))
    b = blocks()
    b["server"] = [{"name": "s0", "uplink": UPLINK, "units": [[DOT, 0, S * O]]},
                   {"name": "s1", "uplink": UPLINK1, "units": [[CAST, 0, S * O]]}]
    b["transfers"][3]["server"] = "s1"
    b["transfers"] += [copy.deepcopy(WIRE_OUT), copy.deepcopy(WIRE_IN)]
    return c, b


def values(seed: int = 7) -> dict[str, list[list[int]]]:
    """Each value's rows as the certifiers saw them cross."""
    rng = random.Random(seed)
    word = lambda n, w: [[rng.randrange(1 << 16) for _ in range(w)] for _ in range(n)]  # noqa: E731
    return {"w": word(O, 16), "bias": word(1, 2), "request": word(S, NP + 1), "response": word(S * O, 1),
            "h": word(S * O, 2)}


def public(seed: bytes) -> LmsKey:
    k = signers.key(seed)
    return LmsKey(k.params, lms.public_key(k))


class Certifiers:
    """One network certifier an uplink, each its own key, signer and session."""

    def __init__(self, tmp_path, uplinks=(UPLINK, UPLINK1)):
        self.issuers = {}
        for i, u in enumerate(uplinks):
            (tmp_path / f"c{i}").mkdir()
            self.issuers[u] = Issuer(signers.signer(tmp_path / f"c{i}", u.encode()), hashlib.sha512(u.encode()).digest()[:32])

    def keys(self) -> dict:
        return {u: (public(u.encode()), i.session) for u, i in self.issuers.items()}

    def register(self, doc: dict, b: dict, data: dict | None = None, *, at: dict | None = None,
                 received: dict | None = None, fresh: bool = False) -> list[dict]:
        """Each transfer of ``b`` committed and signed by its uplink's certifier, a wire's in under its out's salts:
        ``at`` overrides the (link, direction, window) a value's certificate signs for, its head the transfer's own;
        ``received``, the rows a wire's receiving certifier saw, by value (its out's by default); ``fresh``, a wire's in
        under salts of its own."""
        data, servers = data or values(), {s["name"]: s["uplink"] for s in b["server"]}
        out, sent = [], {}
        for t in sorted(b["transfers"], key=lambda t: t["kind"] == "wire" and t["dir"] == "in"):
            link, d = servers[t["server"]], BK.DIRS[t["dir"]]
            wire_in = t["kind"] == "wire" and t["dir"] == "in"
            rows = (received or {}).get(t["value"], data[t["value"]]) if wire_in else data[t["value"]]
            salts = sent[t["value"]].salts if wire_in and not fresh else None
            tr = TR.Transfer.commit(link, d, t["window"], t["kind"], doc["program"], t["value"], rows, salts)
            sent[t["value"]] = tr
            sign = {"link": link, "direction": d, "window": t["window"], **((at or {}).get(t["value"], {}))}
            out.append(TR.registration(tr.head, self.issuers[link].sign(sign["link"], sign["direction"], sign["window"],
                                                                        tr.head.digest())))
        return out


@pytest.fixture(scope="module")
def tiny():
    c = circuit()
    return c, record(c), layouts(c)


def test_a_load_of_32_bit_leaves_is_pinned_at_the_verifiers_width_alone(tiny, tmp_path):
    """The loaded bias, the dot's f32 ``acc`` (2 u16 words a row), registered as a program reading it at 32 bits does,
    one 32-bit leaf a row: pinned when the verifier's ``formats`` name that width, refused at the default 16, and a
    format for no transfer, or a width that doesn't cut a transfer's rows, refused."""
    c, doc, lays = tiny
    certs, b = Certifiers(tmp_path), blocks()
    regs = [r for r in certs.register(doc, b) if r["value"] != "bias"]
    wide = TR.Transfer.commit(UPLINK, Direction.INGRESS, 0, "load", doc["program"], "bias",
                              [[lo | hi << 16] for lo, hi in values()["bias"]], word_bits=32)
    regs.append(TR.registration(wide.head, certs.issuers[UPLINK].sign(UPLINK, Direction.INGRESS, 0, wide.head.digest())))
    pins = BK.pins(doc, b, lays, regs, certs.keys(), {"bias": (32, TR.SCHEMA)})
    assert pins["bias"] == wide.root.hex() and IF.input_count(doc, pins=pins, roots=roots(regs))["free"] == 0
    with pytest.raises(ValueError, match="transfer 'bias' .*registration's words is 1, and the verifier's 2"):
        BK.pins(doc, b, lays, regs, certs.keys())
    with pytest.raises(ValueError, match=r"formats of \['x'\], which no transfer names"):
        BK.pins(doc, b, lays, regs, certs.keys(), {"x": (32, TR.SCHEMA)})
    with pytest.raises(ValueError, match="transfer 'response' .*rows of 1 u16 words are no whole number of 32-bit leaves"):
        BK.pins(doc, b, lays, regs, certs.keys(), {"bias": (32, TR.SCHEMA), "response": (32, TR.SCHEMA)})


def test_free_is_zero_only_with_certified_pins(tiny, tmp_path):
    c, doc, lays = tiny
    certs = Certifiers(tmp_path)
    b = blocks()
    regs = certs.register(doc, b)
    pins = BK.pins(doc, b, lays, regs, certs.keys())
    assert sorted(pins) == ["bias", "request", "w"]
    got = IF.input_count(doc, pins=pins, roots=roots(regs))
    assert got["free"] == 0 and got["registered"] == got["inputs"] - got["wired"] > 0 and got["ingress"] == 0
    assert IF.input_count(doc)["free"] == IF.input_count(doc, pins=pins)["free"] == got["registered"]
    for v in pins:
        assert IF.input_count(doc, pins={p: r for p, r in pins.items() if p != v}, roots=roots(regs))["free"] > 0
    checked = BK.check(doc, b, lays)
    assert [(t.value, t.words, t.rows) for t in checked.transfers] == [("w", 16, O), ("bias", 2, 1), ("request", NP + 1, S),
                                                                        ("response", 1, S * O)]


ADD, R = "I32Add_v2", 2


def demo() -> tuple[IF.Circuit, dict]:
    """The demo's per-request rows at the frozen shape (P = 8 prompt bytes, G = 24 generated) for R = 2 slots, on a
    stand-in for the units that read and write them, which aren't on main (the embedding's token read,
    ``EmbeddingPosition_v1``, and the argmax): `I32Add_v2`'s unit r·G + g reads request row r·P + g mod P as its
    ``a`` (2 words, the token key) and the loaded ``k`` as its ``b``, and its i32 ``o0`` is response row r·G + g, at
    step g + 1 of the server-time window."""
    P, G = TR.PROMPT, TR.GENERATED
    reads = (IF.Read(ADD, "a", "request", 0, (R, G // P, P), 0, (2 * P, 0, 2)), IF.Read(ADD, "b", "k", 0, (R * G,), 0, (0,)))
    c = IF.Circuit("demo-class", ((IF.resolve(ADD), R * G),), external=(f"{ADD}/a", f"{ADD}/o0"),
                   registered=IF.Registered({"request": 2 * R * P, "k": 2}, reads=reads))
    b = {"format": BK.FORMAT, "server": [{"name": "s0", "uplink": UPLINK, "units": "all"}],
         "server_time": {"window": WINDOW, "steps": G + 1},
         "transfers": [{"server": "s0", "dir": "in", "kind": "load", "value": "k", "window": 0, "t": None},
                       {"server": "s0", "dir": "in", "kind": "request", "value": "request", "window": WINDOW,
                        "t": {"base": 0, "radix": [R * P], "step": [0]}},
                       {"server": "s0", "dir": "out", "kind": "response", "value": "response", "window": WINDOW,
                        "t": {"base": 1, "radix": [R, G], "step": [0, 1]}, "leaves": 2 * R * G,
                        "writes": [{"writer": ADD, "out": "o0", "at": 0, "radix": [R * G], "base": 0, "step": [2]}]}]}
    return c, b


def test_the_demos_request_and_response_rows_bind_at_the_frozen_shape(tmp_path):
    c, b = demo()
    doc, lays = record(c), layouts(c)
    req = [bytes(range(i, i + TR.PROMPT)) for i in (0, 100)]
    data = {"k": [[0, 0]], "request": TR.request_rows(req, R),
            "response": TR.response_rows([bytes(range(TR.GENERATED)), b"\xfe" * TR.GENERATED], R)}
    checked = BK.check(doc, b, lays)
    assert [(t.value, t.words, t.rows) for t in checked.transfers] == [
        ("k", 2, 1), ("request", 2, R * TR.PROMPT), ("response", 2, R * TR.GENERATED)]
    certs = Certifiers(tmp_path)
    regs = certs.register(doc, b, data)
    pins = BK.pins(doc, b, lays, regs, certs.keys())
    assert sorted(pins) == ["k", "request"] and IF.input_count(doc, pins=pins, roots=roots(regs))["free"] == 0
    assert IF.input_count(doc, pins={"k": pins["k"]}, roots=roots(regs))["free"] == 16 * 2 * R * TR.GENERATED
    heads = BK.certified(doc, b, lays, regs, certs.keys())
    req, resp = heads["request", Direction.INGRESS], heads["response", Direction.EGRESS]
    assert (req.rows, resp.rows, resp.words) == (16, 48, 2)
    short = edit(b, ("transfers", 2, "leaves"), 2 * R * (TR.GENERATED - 1))
    with pytest.raises(ValueError, match="isn't whole rows of 2 words among the value's 92 leaves"):
        BK.check(doc, short, lays)
    with pytest.raises(ValueError, match="'response'.*registration's rows is 46, and the verifier's 48"):
        BK.pins(doc, b, lays, certs.register(doc, b, data | {"response": data["response"][:-2]}), certs.keys())


def test_two_servers_pin_each_transfer_from_its_own_uplinks_certifier(tmp_path):
    c, b = two_servers()
    doc, lays = record(c), layouts(c)
    certs = Certifiers(tmp_path)
    regs = certs.register(doc, b)
    pins = BK.pins(doc, b, lays, regs, certs.keys())
    assert sorted(pins) == ["bias", "h", "request", "w"] and IF.input_count(doc, pins=pins, roots=roots(regs))["free"] == 0
    heads = BK.certified(doc, b, lays, regs, certs.keys())
    sent, got = heads["h", Direction.EGRESS], heads["h", Direction.INGRESS]
    assert (sent.link, got.link, heads["response", Direction.EGRESS].link) == (UPLINK, UPLINK1, UPLINK1)
    assert sent.root == got.root and sent.domain == got.domain and pins["h"] == got.root.hex()
    keys = certs.keys()
    keys[UPLINK1] = keys[UPLINK]
    with pytest.raises(ValueError, match="'response' .*doesn't verify"):
        BK.pins(doc, b, lays, certs.register(doc, b), keys)


def test_refuses_a_pin_without_a_certificate(tiny, tmp_path):
    c, doc, lays = tiny
    certs = Certifiers(tmp_path)
    b = blocks()
    regs = certs.register(doc, b)
    with pytest.raises(ValueError, match="'w' .*no certificate"):
        BK.pins(doc, b, lays, [r for r in regs if r["value"] != "w"], certs.keys())
    own = TR.Transfer.commit(UPLINK, Direction.INGRESS, 0, "load", doc["program"], "w", values(seed=8)["w"])
    unsigned = [r if r["value"] != "w" else TR.registration(own.head, Certificate(
        bytes.fromhex(r["certificate"]["key"]), UPLINK, Direction.INGRESS, 0, bytes.fromhex(r["certificate"]["digest"]),
        bytes.fromhex(r["certificate"]["signature"]))) for r in regs]
    with pytest.raises(ValueError, match="'w' .*doesn't sign its head"):
        BK.pins(doc, b, lays, unsigned, certs.keys())
    with pytest.raises(ValueError, match="no transfer names"):
        BK.pins(doc, b, lays, [*regs, {**regs[0], "value": "extra"}], certs.keys())
    b2 = blocks()
    del b2["transfers"][1]
    with pytest.raises(ValueError, match="registered read of 'bias', which no transfer brings in"):
        BK.pins(doc, b2, lays, [r for r in regs if r["value"] != "bias"], certs.keys())
    with pytest.raises(ValueError, match="no network certifier on"):
        BK.pins(doc, b, lays, regs, {})


@pytest.mark.parametrize("other,to", [("link", UPLINK1), ("window", WINDOW + 1), ("direction", Direction.EGRESS)])
def test_refuses_a_certificate_for_another_link_window_or_direction(tiny, tmp_path, other, to):
    c, doc, lays = tiny
    certs = Certifiers(tmp_path)
    b = blocks()
    regs = certs.register(doc, b, at={"request": {other: to}})
    with pytest.raises(ValueError, match="'request' .*doesn't verify under the network certifier's key"):
        BK.pins(doc, b, lays, regs, certs.keys())
    later = blocks()
    later["server_time"]["window"] = later["transfers"][2]["window"] = later["transfers"][3]["window"] = WINDOW + 1
    with pytest.raises(ValueError, match="'request' .*registration's window is 3, and the verifier's 4"):
        BK.pins(doc, later, lays, certs.register(doc, b), certs.keys())


def test_refuses_an_uncovered_external_row(tiny):
    c, doc, lays = tiny
    half = circuit(reads=(), external=(f"{DOT}/a", f"{CAST}/o0"))
    half = replace(half, registered=replace(half.registered, reads=(
        IF.Read(DOT, "a", "request", 0, (1, O), 0, (NP + 1, 0)), *half.registered.reads[1:])))
    with pytest.raises(ValueError, match=f"external port {DOT}/a: {O} of its {S * O} unit rows are no transfer's"):
        BK.check(record(half), blocks(), lays)
    b = blocks()
    b["transfers"][3]["writes"][0]["radix"] = [S * O - 1]
    b["transfers"][3]["leaves"], b["transfers"][3]["t"] = S * O - 1, {"base": 1, "radix": [S * O - 1], "step": [0]}
    with pytest.raises(ValueError, match=f"external port {CAST}/o0: 1 of its {S * O} unit rows are no transfer's"):
        BK.check(doc, b, lays)
    more = circuit(external=(f"{DOT}/a", f"{CAST}/o0", f"{DOT}/o0"))
    with pytest.raises(ValueError, match=f"external port {DOT}/o0: {S * O} of its {S * O}"):
        BK.check(record(more), blocks(), lays)


def test_refuses_a_wire_between_servers_and_an_ingress_declaration(tmp_path):
    c, b = two_servers()
    crossed = circuit(blocks=inference(O))
    b2 = blocks()
    b2["server"] = b["server"]
    b2["transfers"][3]["server"] = "s1"
    with pytest.raises(ValueError, match="reader and writer units are not one server's"):
        BK.check(record(crossed), b2, layouts(crossed))
    feed = circuit(reads=(), external=(f"{DOT}/a", f"{CAST}/o0"))
    feed = replace(feed, registered=replace(feed.registered, reads=feed.registered.reads[1:]),
                   ingress=(IF.Ingress(DOT, "a", 0, S * O, 0),))
    with pytest.raises(ValueError, match="ingress declaration"):
        BK.check(record(feed), blocks(), layouts(feed))


def edit(b: dict, path: tuple, value) -> dict:
    b = copy.deepcopy(b)
    at = b
    for k in path[:-1]:
        at = at[k]
    at[path[-1]] = value
    return b


@pytest.mark.parametrize("path,value,why", [
    (("server", 0, "units"), [[DOT, 0, S * O]], "units \\[0, 4\\) are in no server"),
    (("server", 0, "units"), [[DOT, 0, S * O], [CAST, 0, S * O], [CAST, 1, 1]], "in two servers"),
    (("server_time", "steps"), O, "each at a step among the server-time block's 2"),
    (("transfers", 0, "kind"), "request", "a request's window is the server-time block's"),
    (("transfers", 0, "t"), {"base": 0, "radix": [O], "step": [0]}, "a load crosses once a server session"),
    (("transfers", 2, "kind"), "load", "a load's rows are never the circuit's external ports"),
    (("transfers", 2, "window"), WINDOW - 1, "a request's window is the server-time block's"),
    (("transfers", 2, "dir"), "out", "an out transfer's leaves and writes"),
    (("transfers", 3, "kind"), "wire", "a wire's rows are never the circuit's external ports"),
    (("transfers", 3, "writes", 0, "base"), 1, "isn't whole rows of 1 words among the value's 4 leaves"),
    (("transfers", 3, "t"), {"base": 1, "radix": [S * O + 1], "step": [0]}, "its t is over its 4 rows"),
    (("transfers", 3, "server"), "s9", "no server of the circuit's"),
    (("transfers", 1, "value"), "w", "a transfer is of a new value"),
    (("format",), "verity/blocks/v0", "verity/blocks/v1 object"),
])
def test_refuses_a_rule_broken(tiny, path, value, why):
    _c, doc, lays = tiny
    with pytest.raises(ValueError, match=why):
        BK.check(doc, edit(blocks(), path, value), lays)


def test_refuses_a_read_that_is_not_whole_rows_of_the_value():
    c = circuit()
    c = replace(c, registered=replace(c.registered, reads=(
        IF.Read(DOT, "a", "request", 0, (S, O), 0, (NP + 1, 0)), IF.Read(DOT, "b", "w", 0, (S, O), 0, (0, 8)),
        IF.Read(DOT, "acc", "bias", 0, (S * O,), 0, (0,)))))
    with pytest.raises(ValueError, match="'w': read .* isn't whole rows of 16 words"):
        BK.check(record(c), blocks(), layouts(c))


def write(at: int, radix: list, base: int, step: list) -> dict:
    """A write of the cast's output rows into the response."""
    return {"writer": CAST, "out": "o0", "at": at, "radix": radix, "base": base, "step": step}


@pytest.mark.parametrize("writes,leaves,why", [
    ([write(0, [S * O], 0, [0])], S * O, "writes its row 0 from 4 units"),
    ([write(0, [2], 0, [1]), write(2, [2], 1, [1])], S * O, "its row 1 is written twice"),
    ([write(0, [S, O], 0, [1, 1])], S * O, "its row 1 is written twice"),
    ([write(0, [S * O], 0, [1])], S * O + 1, "its row 4 of 5 is written by no unit"),
])
def test_refuses_an_out_value_row_written_by_no_unit_or_two(tiny, writes, leaves, why):
    _c, doc, lays = tiny
    b = edit(edit(blocks(), ("transfers", 3, "writes"), writes), ("transfers", 3, "leaves"), leaves)
    with pytest.raises(ValueError, match=f"'response': .*{why}; an out value's every row is written by exactly one unit"):
        BK.check(doc, b, lays)


@pytest.mark.parametrize("writes", [
    [write(0, [S * O], S * O - 1, [-1])],
    [write(0, [S * O, 1], 0, [1, 7])],
    [write(0, [O, S], 0, [1, O])],
    [write(2, [2], 0, [1]), write(0, [2], 2, [1])],
])
def test_an_out_value_written_once_in_any_order(tiny, writes):
    _c, doc, lays = tiny
    response = BK.check(doc, edit(blocks(), ("transfers", 3, "writes"), writes), lays).transfers[3]
    assert (response.value, response.rows, len(response.gates)) == ("response", S * O, len(writes))


def unpaired(name: str) -> tuple[IF.Circuit, dict]:
    """`two_servers` with its wire ``h`` broken as ``name`` says."""
    c, b = two_servers()
    t = b["transfers"]
    if name == "no out":
        del t[4]
    elif name == "no in":
        del t[5]
    elif name == "two ins":
        t.append(copy.deepcopy(WIRE_IN))
    elif name == "a response of it":
        t.append({**copy.deepcopy(RESPONSE), "value": "h"})
    elif name == "other rows":
        c = circuit(wired=False, reads=(IF.Read(CAST, "a", "h", 0, (S * O,), 0, (2,)),), params={"h": 2 * S * O + 2},
                    blocks=inference(O))
        t[5]["t"]["radix"] = [S * O + 1]
    elif name == "other steps":
        t[5]["t"]["base"] = 2
    return c, b


@pytest.mark.parametrize("name,why", [
    ("no out", "wire 'h': its out and its in, one each"),
    ("no in", "wire 'h': its out and its in, one each"),
    ("two ins", "a transfer is of a new value, named, but a wire's other one"),
    ("a response of it", "a transfer is of a new value, named, but a wire's other one"),
    ("other rows", "wire 'h': its out is 4 rows of 2 words and its in 5 rows of 2"),
    ("other steps", "wire 'h': .*each row of a wire crosses both uplinks at one step"),
])
def test_refuses_a_wire_not_paired(name, why):
    c, b = unpaired(name)
    with pytest.raises(ValueError, match=why):
        BK.check(record(c), b, layouts(c))


def test_a_wires_in_may_come_before_its_out_but_not_on_its_own_server():
    c, b = two_servers()
    b["transfers"][4:] = [b["transfers"][5], b["transfers"][4]]
    assert [(t.value, t.direction) for t in BK.check(record(c), b, layouts(c)).transfers[4:]] == [
        ("h", Direction.INGRESS), ("h", Direction.EGRESS)]
    b = blocks()
    b["transfers"] += [copy.deepcopy(WIRE_OUT), {**copy.deepcopy(WIRE_IN), "server": "s0"}]
    with pytest.raises(ValueError, match="wire 'h': from server s0 to itself"):
        BK.check(record(c), b, layouts(c))


#: the refusal of an inference block on two servers, where both are named
ON_TWO = ("inference block (slot {slot}, step {step}): units of {a} on server {sa} and of {b} on server {sb}; an inference "
          "block runs on one server, the one that may read its inputs")


def test_refuses_an_inference_block_on_two_servers():
    """A server's ingress and egress carry only its inference blocks' declared inputs and outputs,
    and an inference block runs on one server. Cut dot | cast with the cast in the dot's own inference blocks, every
    (slot, step) is on both servers, refused by name; with the cast in blocks of its own (`two_servers`) the same cut is
    accepted. Two servers over a record that states no inference block are refused, nothing then saying which server runs
    each (one server needs none: `tiny`'s record states none)."""
    c, b = two_servers(cast=0)
    with pytest.raises(ValueError, match=re.escape(ON_TWO.format(slot=0, step=0, a=DOT, sa="s0", b=CAST, sb="s1"))):
        BK.check(record(c), b, layouts(c))
    ok, b = two_servers()
    assert [(s.name, s.uplink) for s in BK.check(record(ok), b, layouts(ok)).servers] == [("s0", UPLINK), ("s1", UPLINK1)]
    bare = replace(ok, blocks=None)
    with pytest.raises(ValueError, match=r"2 servers over a circuit record that states no inference block \(blocks.inference\)"):
        BK.check(record(bare), b, layouts(bare))


def entries(doc: dict, edit) -> dict:
    """``doc`` with its inference entries ``edit``ed."""
    doc = copy.deepcopy(doc)
    doc["blocks"]["inference"]["entries"] = edit(doc["blocks"]["inference"]["entries"])
    return doc


@pytest.mark.parametrize("edit,why", [
    (lambda es: es[:1], rf"template {CAST}: inference entries tile units \[0, 0\) of its {S * O}"),
    (lambda es: [es[0], {**es[1], "at": 1, "radix": [S, O - 1]}], rf"template {CAST}: units \[0, 1\) are in no inference entry"),
    (lambda es: [*es, {**es[1], "radix": [1, 1]}], rf"template {CAST}: units \[0, 1\) are in two inference entries"),
    (lambda es: [es[0], {**es[1], "axes": ["step", "step"]}], "one the slot and at most one the step"),
    (lambda es: [es[0], {**es[1], "template": "Gemm_v2{}"}], "a template of the record's"),
    (lambda es: {}, "its entries, a list"),
])
def test_refuses_inference_entries_that_dont_tile_the_units(edit, why):
    c, b = two_servers()
    with pytest.raises(ValueError, match=why):
        BK.check(entries(record(c), edit), b, layouts(c))


#: a record's templates (the circuit's two, at other unit counts) and their inference entries ``(at, radix, axes, step)``:
#: the slot digit fastest, slowest and alone, the step digit slowest, fastest and absent
GRID = {DOT: (60, [(0, [3, 4, 5], ["step", "-", "slot"], 0)]),
        CAST: (25, [(0, [5, 2], ["slot", "step"], 1), (10, [3, 5], ["-", "slot"], 3)])}


def block_of(t: str, u: int) -> tuple[int, int]:
    """Unit ``u`` of template ``t``'s (slot, step), from its digits one by one."""
    for at, radix, axes, step in GRID[t][1]:
        if at <= u < at + math.prod(radix):
            i, digits = u - at, []
            for r in reversed(radix):
                i, d = divmod(i, r)
                digits.insert(0, d)
            return digits[axes.index("slot")], step + (digits[axes.index("step")] if "step" in axes else 0)
    raise AssertionError(f"{t} {u}")


def placed(server_of) -> tuple[dict, list[dict]]:
    """`GRID`'s record, and its servers from each unit's ``server_of(t, u)``, each server's units its runs."""
    runs: dict[str, list[list]] = {}
    for t, (n, _es) in GRID.items():
        for u in range(n):
            r = runs.setdefault(server_of(t, u), [])
            if r and r[-1][0] == t and r[-1][1] + r[-1][2] == u:
                r[-1][2] += 1
            else:
                r.append([t, u, 1])
    doc = {"templates": [[t, n] for t, (n, _es) in GRID.items()],
           "blocks": {"inference": {"entries": [{"template": t, "at": at, "radix": r, "axes": a, "step": st}
                                                for t, (_n, es) in GRID.items() for at, r, a, st in es]}}}
    return doc, [{"name": s, "uplink": f"{s}/uplink", "units": runs[s]} for s in sorted(runs)]


@pytest.mark.parametrize("seed", range(8))
def test_each_inference_block_on_one_server_however_the_runs_cut_the_entries(tiny, seed):
    """`server_blocks` against each unit's (slot, step) worked out digit by digit: units placed on a random server per
    inference block, whatever runs that cuts each entry into, are accepted; moving one unit to another server refuses
    exactly its block, by name (every block has at least 3 units)."""
    _c, _doc, lays = tiny
    rng, names = random.Random(seed), ["s0", "s1", "s2"]
    home = {block_of(t, u): None for t, (n, _es) in GRID.items() for u in range(n)}
    home = {k: rng.choice(names) for k in home}
    doc, servers = placed(lambda t, u: home[block_of(t, u)])
    assert [s.name for s in BK.server_blocks(doc, servers, lays)] == sorted(set(home.values()))
    t = rng.choice(sorted(GRID))
    u = rng.randrange(GRID[t][0])
    (slot, step), other = block_of(t, u), rng.choice([s for s in names if s != home[block_of(t, u)]])
    doc, moved = placed(lambda t2, u2: other if (t2, u2) == (t, u) else home[block_of(t2, u2)])
    with pytest.raises(ValueError, match=re.escape(f"inference block (slot {slot}, step {step}): ")):
        BK.server_blocks(doc, moved, lays)


def test_refuses_a_wire_whose_rows_arrived_other_than_they_left(tmp_path):
    c, b = two_servers()
    doc, lays = record(c), layouts(c)
    certs = Certifiers(tmp_path)
    for regs in (certs.register(doc, b, fresh=True), certs.register(doc, b, received={"h": values(seed=8)["h"]})):
        with pytest.raises(ValueError, match="wire 'h': the root s1/uplink's network certifier signed isn't the one "
                                             "s0/uplink's signed"):
            BK.pins(doc, b, lays, regs, certs.keys())
    regs = certs.register(doc, b)
    with pytest.raises(ValueError, match="names a value and a direction, once: 'h' 'ingress'"):
        BK.pins(doc, b, lays, [*regs, regs[-1]], certs.keys())
    with pytest.raises(ValueError, match="'h' \\(wire ingress on s1/uplink, window 3\\): no certificate"):
        BK.pins(doc, b, lays, [r for r in regs if (r["value"], r["direction"]) != ("h", "ingress")], certs.keys())
