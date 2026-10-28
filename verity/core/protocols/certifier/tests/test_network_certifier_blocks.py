"""The host's check (`blocks`) on a TINY class circuit, from the window interface's records: S = 2 slots, each a scan state
row of NP + 1 = 16 words, and O = 2 output steps. `AmpereBF16TcDot16_v3`'s unit j·O + k reads slot j's row of the scan's
state (its ``a``, the circuit's external input), the loaded weights' row k (``b``) and the loaded bias (``acc``);
`F32ToBf16Rn_v2`'s unit u is wired to its output, and its ``o0`` is the circuit's external output. The network certifier
on the host's uplink commits and signs each load (`transfer_rows`), and the verifier pins the loads from its certificates
alone (`blocks.pins`) and the scan's state beside them: with both, held to the roots the registrations record, the
circuit's free inputs are 0. A record names no server and no transfer."""

from __future__ import annotations

import copy
import hashlib
import random

import pytest

from verity.ml.crypto import lms
from verity.ml.windows import interface as IF
from verity.core.protocols.certifier import blocks as BK
from verity.core.protocols.certifier import transfer_rows as TR
from verity.core.protocols.certifier.certificate import Certificate, Direction, Issuer, LmsKey
from verity.core.protocols.certifier.ingress_scan import SCAN_STATE

import signers

DOT, CAST = "AmpereBF16TcDot16_v3", "F32ToBf16Rn_v2"
S, NP, O = 2, 15, 2
UPLINK = "s0/uplink"
#: the root the verifier pins the scan's state at (the scan's rows the window reads), a stand-in here
STATE_ROOT = "cd" * 64


def circuit(reads: tuple | None = None, params: dict | None = None, **kw) -> IF.Circuit:
    """The TINY class circuit: its loads (``w``, ``bias``) and the scan's state read by the dot's units, the cast wired to
    them; ``reads`` in place of its registered reads."""
    base = (IF.Read(DOT, "a", SCAN_STATE, 0, (S, O), 0, (NP + 1, 0)),
            IF.Read(DOT, "b", "w", 0, (S, O), 0, (0, 16)),
            IF.Read(DOT, "acc", "bias", 0, (S * O,), 0, (0,)))
    wires = (IF.Wire(CAST, "a", DOT, "o0", 0, (S * O,), 0, (1,)),)
    reg = IF.Registered({SCAN_STATE: S * (NP + 1), "w": O * 16, "bias": 2, **(params or {})},
                        reads=base if reads is None else reads)
    return IF.Circuit("tiny-class", ((IF.resolve(DOT), S * O), (IF.resolve(CAST), S * O)),
                      external=kw.pop("external", (f"{DOT}/a", f"{CAST}/o0")), wires=wires, registered=reg, **kw)


def record(c: IF.Circuit) -> dict:
    return IF.descriptor_record(c, cap=S * O)


def layouts(c: IF.Circuit) -> dict:
    return {t: lay for t, (lay, _n) in c.layouts.items()}


def values(seed: int = 7) -> dict[str, list[list[int]]]:
    """Each load's rows as the network certifier saw them cross."""
    rng = random.Random(seed)
    word = lambda n, w: [[rng.randrange(1 << 16) for _ in range(w)] for _ in range(n)]  # noqa: E731
    return {"w": word(O, 16), "bias": word(1, 2)}


def roots(regs: list[dict]) -> dict[str, str]:
    """The roots the window's registrations record, by value, the scan's state's beside the loads'."""
    return {SCAN_STATE: STATE_ROOT, **{r["value"]: r["root"] for r in regs}}


class Certifier:
    """The network certifier on the host's uplink: its key, signer and session."""

    def __init__(self, tmp_path, link: str = UPLINK):
        (tmp_path / "c").mkdir()
        self.link = link
        self.issuer = Issuer(signers.signer(tmp_path / "c", link.encode()), hashlib.sha512(link.encode()).digest()[:32])

    def key(self) -> tuple[LmsKey, bytes]:
        k = signers.key(self.link.encode())
        return LmsKey(k.params, lms.public_key(k)), self.issuer.session

    def register(self, doc: dict, data: dict | None = None, *, at: dict | None = None) -> list[dict]:
        """Each load committed and signed as it crosses; ``at`` overrides the (link, direction, window) a value's
        certificate signs for, its head the load's own."""
        out = []
        for v, rows in (data or values()).items():
            tr = TR.Transfer.commit(self.link, Direction.INGRESS, BK.SESSION_WINDOW, "load", doc["program"], v, rows)
            sign = {"link": self.link, "direction": Direction.INGRESS, "window": BK.SESSION_WINDOW, **(at or {}).get(v, {})}
            out.append(TR.registration(tr.head, self.issuer.sign(sign["link"], sign["direction"], sign["window"],
                                                                 tr.head.digest())))
        return out


@pytest.fixture(scope="module")
def tiny():
    c = circuit()
    return c, record(c), layouts(c)


def test_free_is_zero_only_with_the_loads_certified_pins_and_the_scans_state(tiny, tmp_path):
    _c, doc, lays = tiny
    cert = Certifier(tmp_path)
    regs = cert.register(doc)
    pins = BK.pins(doc, lays, regs, UPLINK, cert.key())
    assert sorted(pins) == ["bias", "w"]
    held = {**pins, SCAN_STATE: STATE_ROOT}
    got = IF.input_count(doc, pins=held, roots=roots(regs))
    assert got["free"] == 0 and got["registered"] == got["inputs"] - got["wired"] > 0 and got["ingress"] == 0
    for v in held:
        assert IF.input_count(doc, pins={p: r for p, r in held.items() if p != v}, roots=roots(regs))["free"] > 0
    assert [(x.value, x.words, x.rows) for x in BK.check(doc, lays)] == [("bias", 2, 1), ("w", 16, O)]


def test_a_load_of_32_bit_leaves_is_pinned_at_the_verifiers_width_alone(tiny, tmp_path):
    """The loaded bias, the dot's f32 ``acc`` (2 u16 words a row), registered as a program reading it at 32 bits does,
    one 32-bit leaf a row: pinned when the verifier's ``formats`` name that width, refused at the default 16, and a
    format for no load, or a width that doesn't cut a load's rows, refused."""
    _c, doc, lays = tiny
    cert = Certifier(tmp_path)
    regs = [r for r in cert.register(doc) if r["value"] != "bias"]
    wide = TR.Transfer.commit(UPLINK, Direction.INGRESS, BK.SESSION_WINDOW, "load", doc["program"], "bias",
                              [[lo | hi << 16] for lo, hi in values()["bias"]], word_bits=32)
    regs.append(TR.registration(wide.head, cert.issuer.sign(UPLINK, Direction.INGRESS, BK.SESSION_WINDOW,
                                                             wide.head.digest())))
    pins = BK.pins(doc, lays, regs, UPLINK, cert.key(), {"bias": (32, TR.SCHEMA)})
    assert pins["bias"] == wide.root.hex()
    assert IF.input_count(doc, pins={**pins, SCAN_STATE: STATE_ROOT}, roots=roots(regs))["free"] == 0
    with pytest.raises(ValueError, match="load 'bias' .*registration's words is 1, and the verifier's 2"):
        BK.pins(doc, lays, regs, UPLINK, cert.key())
    with pytest.raises(ValueError, match=r"formats of \['x'\], which no load names"):
        BK.pins(doc, lays, regs, UPLINK, cert.key(), {"x": (32, TR.SCHEMA)})
    with pytest.raises(ValueError, match="load 'w' .*rows of 16 u16 words are no whole number of 24-bit leaves"):
        BK.pins(doc, lays, regs, UPLINK, cert.key(), {"bias": (32, TR.SCHEMA), "w": (24, TR.SCHEMA)})


def test_refuses_a_pin_without_a_certificate(tiny, tmp_path):
    _c, doc, lays = tiny
    cert = Certifier(tmp_path)
    regs = cert.register(doc)
    with pytest.raises(ValueError, match="'w' .*no certificate"):
        BK.pins(doc, lays, [r for r in regs if r["value"] != "w"], UPLINK, cert.key())
    own = TR.Transfer.commit(UPLINK, Direction.INGRESS, BK.SESSION_WINDOW, "load", doc["program"], "w", values(seed=8)["w"])
    unsigned = [r if r["value"] != "w" else TR.registration(own.head, Certificate(
        bytes.fromhex(r["certificate"]["key"]), UPLINK, Direction.INGRESS, BK.SESSION_WINDOW,
        bytes.fromhex(r["certificate"]["digest"]), bytes.fromhex(r["certificate"]["signature"]))) for r in regs]
    with pytest.raises(ValueError, match="'w' .*doesn't sign its head"):
        BK.pins(doc, lays, unsigned, UPLINK, cert.key())
    with pytest.raises(ValueError, match="which no load names"):
        BK.pins(doc, lays, [*regs, {**regs[0], "value": "extra"}], UPLINK, cert.key())
    with pytest.raises(ValueError, match="names a value, ingress, once: 'w' 'ingress'"):
        BK.pins(doc, lays, [*regs, regs[0]], UPLINK, cert.key())
    with pytest.raises(ValueError, match="names a value, ingress, once: 'w' 'egress'"):
        BK.pins(doc, lays, [{**regs[0], "direction": "egress"}, *regs[1:]], UPLINK, cert.key())


@pytest.mark.parametrize("other,to", [("link", "s1/uplink"), ("window", 3), ("direction", Direction.EGRESS)])
def test_refuses_a_certificate_for_another_link_window_or_direction(tiny, tmp_path, other, to):
    _c, doc, lays = tiny
    cert = Certifier(tmp_path)
    regs = cert.register(doc, at={"w": {other: to}})
    with pytest.raises(ValueError, match="'w' .*doesn't verify under the network certifier's key"):
        BK.pins(doc, lays, regs, UPLINK, cert.key())
    with pytest.raises(ValueError, match="load 'bias' .*registration's link is 's0/uplink', and the verifier's 's1/uplink'"):
        BK.pins(doc, lays, cert.register(doc), "s1/uplink", cert.key())


def test_refuses_an_ingress_declaration():
    feed = circuit(reads=circuit().registered.reads[1:], ingress=(IF.Ingress(DOT, "a", 0, S * O, 0),))
    with pytest.raises(ValueError, match="an ingress declaration: a host's ingress is the scan's state"):
        BK.check(record(feed), layouts(feed))


def test_refuses_an_external_input_row_the_scans_state_doesnt_fill(tiny):
    """Every row of an external input port is a read of the scan's state, once: a row of none, a port no read fills, and
    a read of the state at a port that isn't external are refused."""
    _c, doc, lays = tiny
    reads = circuit().registered.reads
    half = circuit(reads=(IF.Read(DOT, "a", SCAN_STATE, 0, (1, O), 0, (NP + 1, 0)), *reads[1:]))
    with pytest.raises(ValueError, match=f"external port {DOT}/a: {O} of its {S * O} unit rows are no read of the scan's state"):
        BK.check(record(half), layouts(half))
    twice = copy.deepcopy(doc)
    twice["registered"]["reads"].append({**twice["registered"]["reads"][0], "radix": [1], "step": [0]})
    with pytest.raises(ValueError, match=f"{DOT}/a: reads fill one row of a unit from unit 0 twice"):
        BK.check(twice, lays)
    inner = circuit(external=(f"{CAST}/o0",))
    with pytest.raises(ValueError, match=f"the scan's state: read .* fills {DOT}/a, no external port"):
        BK.check(record(inner), layouts(inner))


def test_refuses_a_load_at_an_external_port():
    c = circuit(reads=(IF.Read(DOT, "a", "w", 0, (S, O), 0, (0, 0)), *circuit().registered.reads[1:]))
    with pytest.raises(ValueError, match="load 'w': read .*: a load's rows are never the circuit's external ports"):
        BK.check(record(c), layouts(c))


def test_refuses_a_read_that_is_not_whole_rows_of_the_value():
    c = circuit(reads=(IF.Read(DOT, "a", SCAN_STATE, 0, (S, O), 0, (NP + 1, 0)), IF.Read(DOT, "b", "w", 0, (S, O), 0, (0, 8)),
                       IF.Read(DOT, "acc", "bias", 0, (S * O,), 0, (0,))))
    with pytest.raises(ValueError, match="'w': read .* isn't whole rows of 16 words"):
        BK.check(record(c), layouts(c))


def test_refuses_a_load_no_unit_reads(tiny):
    c = circuit(params={"spare": 16})
    with pytest.raises(ValueError, match="load 'spare': no unit reads it"):
        BK.check(record(c), layouts(c))
    _c, doc, lays = tiny
    stray = copy.deepcopy(doc)
    stray["registered"]["reads"].append({**stray["registered"]["reads"][1], "param": "elsewhere"})
    with pytest.raises(ValueError, match="a registered read of 'elsewhere', which is no registered parameter"):
        BK.check(stray, lays)


def test_refuses_a_table_over_part_of_a_load(tiny):
    _c, doc, lays = tiny
    part = copy.deepcopy(doc)
    part["registered"]["tables"] = [{"name": "wt", "param": "w", "offset": 0, "words": 16, "rows": 1}]
    with pytest.raises(ValueError, match="load 'w': table wt is a part of the value; a table over a load is the whole value"):
        BK.check(part, lays)
    whole = copy.deepcopy(doc)
    whole["registered"]["tables"] = [{"name": "wt", "param": "w", "offset": 0, "words": 16, "rows": O}]
    assert [(x.value, x.words, x.rows) for x in BK.check(whole, lays)] == [("bias", 2, 1), ("w", 16, O)]


def test_a_loads_reads_may_be_of_one_width_only(tiny):
    _c, doc, lays = tiny
    two = copy.deepcopy(doc)
    two["registered"]["reads"].append({"reader": CAST, "port": "a", "param": "w", "at": 0, "radix": [1], "base": 0,
                                       "step": [0]})
    with pytest.raises(ValueError, match=r"load 'w': its reads are rows of \[2, 16\] words"):
        BK.check(two, lays)
