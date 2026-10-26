"""The blocks check (`blocks`) on a TINY class circuit, from the window interface's records: S = 2 slots, each a request
row of NP + 1 = 16 words, and O = 2 output steps. `AmpereBF16TcDot16_v3`'s unit j·O + k reads slot j's request row (its
``a``, the circuit's external input), the loaded weights' row k (``b``) and the loaded bias (``acc``); `F32ToBf16Rn_v2`'s
unit u is wired to its output, and its ``o0`` is response row u (the circuit's external output). The network certifier on
the server's uplink commits and signs the load, the request and the response (`transfer_rows`), and the verifier's pins
come from its certificates alone (`blocks.pins`): with them the circuit's free inputs are 0. The demo's own request and
response rows (`transfer_rows.request_rows`, `response_rows`) bind at the frozen shape on stand-in units (`demo`)."""

from __future__ import annotations

import copy
import hashlib
import random
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


def two_servers() -> tuple[IF.Circuit, dict]:
    """The circuit cut in two servers, the dot's units s0's and the cast's s1's: the dot's outputs cross from s0 to s1
    as a wire, s0's egress (``h_out``) and s1's ingress (``h``), each under its uplink's certifier."""
    c = circuit(wired=False, reads=(IF.Read(CAST, "a", "h", 0, (S * O,), 0, (2,)),), params={"h": 2 * S * O})
    b = blocks()
    b["server"] = [{"name": "s0", "uplink": UPLINK, "units": [[DOT, 0, S * O]]},
                   {"name": "s1", "uplink": UPLINK1, "units": [[CAST, 0, S * O]]}]
    b["transfers"][3]["server"] = "s1"
    b["transfers"] += [{"server": "s0", "dir": "out", "kind": "wire", "value": "h_out", "window": WINDOW,
                        "t": {"base": 1, "radix": [S * O], "step": [0]}, "leaves": 2 * S * O,
                        "writes": [{"writer": DOT, "out": "o0", "at": 0, "radix": [S * O], "base": 0, "step": [2]}]},
                       {"server": "s1", "dir": "in", "kind": "wire", "value": "h", "window": WINDOW,
                        "t": {"base": 1, "radix": [S * O], "step": [0]}}]
    return c, b


def values(seed: int = 7) -> dict[str, list[list[int]]]:
    """Each transfer's rows as the certifier saw them cross."""
    rng = random.Random(seed)
    word = lambda n, w: [[rng.randrange(1 << 16) for _ in range(w)] for _ in range(n)]  # noqa: E731
    return {"w": word(O, 16), "bias": word(1, 2), "request": word(S, NP + 1), "response": word(S * O, 1),
            "h_out": word(S * O, 2), "h": word(S * O, 2)}


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

    def register(self, doc: dict, b: dict, data: dict | None = None, *, at: dict | None = None) -> list[dict]:
        """Each transfer of ``b`` committed and signed by its uplink's certifier: ``at`` overrides the (link, direction,
        window) a value's certificate signs for, its head the transfer's own."""
        data, servers = data or values(), {s["name"]: s["uplink"] for s in b["server"]}
        out = []
        for t in b["transfers"]:
            link, d = servers[t["server"]], BK.DIRS[t["dir"]]
            tr = TR.Transfer.commit(link, d, t["window"], t["kind"], doc["program"], t["value"], data[t["value"]])
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
    assert pins["bias"] == wide.root.hex() and IF.input_count(doc, pins=pins)["free"] == 0
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
    pins = BK.pins(doc, b, lays, certs.register(doc, b), certs.keys())
    assert sorted(pins) == ["bias", "request", "w"]
    got = IF.input_count(doc, pins=pins)
    assert got["free"] == 0 and got["registered"] == got["inputs"] - got["wired"] > 0 and got["ingress"] == 0
    assert IF.input_count(doc)["free"] == got["registered"]
    for v in pins:
        assert IF.input_count(doc, pins=set(pins) - {v})["free"] > 0
    checked = BK.check(doc, b, lays)
    assert [(t.value, t.words, t.rows) for t in checked.transfers] == [("w", 16, O), ("bias", 2, 1), ("request", NP + 1, S),
                                                                        ("response", 1, S * O)]


ADD, R = "I32Add_v2", 2


def demo() -> tuple[IF.Circuit, dict]:
    """The demo's per-request rows at the frozen shape (P = 8 prompt bytes, G = 24 generated) for R = 2 slots, on a
    stand-in for the units that read and write them, which aren't on main (the embedding's token read, CIR-104's
    ``EmbeddingPosition_v1``, and CIR-100's argmax): `I32Add_v2`'s unit r·G + g reads request row r·P + g mod P as its
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
    assert sorted(pins) == ["k", "request"] and IF.input_count(doc, pins=pins)["free"] == 0
    assert IF.input_count(doc, pins={"k"})["free"] == 16 * 2 * R * TR.GENERATED
    heads = BK.certified(doc, b, lays, regs, certs.keys())
    assert (heads["request"].rows, heads["response"].rows, heads["response"].words) == (16, 48, 2)
    short = edit(b, ("transfers", 2, "leaves"), 2 * R * (TR.GENERATED - 1))
    with pytest.raises(ValueError, match="isn't whole rows of 2 words among the value's 92 leaves"):
        BK.check(doc, short, lays)
    with pytest.raises(ValueError, match="'response'.*registration's rows is 46, and the verifier's 48"):
        BK.pins(doc, b, lays, certs.register(doc, b, data | {"response": data["response"][:-2]}), certs.keys())


def test_two_servers_pin_each_transfer_from_its_own_uplinks_certifier(tmp_path):
    c, b = two_servers()
    doc, lays = record(c), layouts(c)
    certs = Certifiers(tmp_path)
    pins = BK.pins(doc, b, lays, certs.register(doc, b), certs.keys())
    assert sorted(pins) == ["bias", "h", "request", "w"] and IF.input_count(doc, pins=pins)["free"] == 0
    heads = BK.certified(doc, b, lays, certs.register(doc, b), certs.keys())
    assert heads["h_out"].link == UPLINK and heads["h_out"].direction is Direction.EGRESS
    assert heads["h"].link == UPLINK1 and heads["response"].link == UPLINK1
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
    with pytest.raises(ValueError, match=f"external port {CAST}/o0: 1 of its {S * O} unit rows are no transfer's"):
        BK.check(doc, b, lays)
    more = circuit(external=(f"{DOT}/a", f"{CAST}/o0", f"{DOT}/o0"))
    with pytest.raises(ValueError, match=f"external port {DOT}/o0: {S * O} of its {S * O}"):
        BK.check(record(more), blocks(), lays)


def test_refuses_a_wire_between_servers_and_an_ingress_declaration(tmp_path):
    c, b = two_servers()
    crossed = circuit()
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
    (("transfers", 3, "leaves"), S * O + 1, "its t is over its 5 rows"),
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
