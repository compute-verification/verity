"""The reference's ties of the counted composite (`reference.WIPE_NODE`, `reference.WIPE_COUNT`, Lean's
`Core.Onsite.wipeNode` and `wipeCount`) and its count's parameters (`reference.wipe_link`, `host_span_params`) on made
pairs: a wipe run between the two marks of a count over a span on its host's one link. Python alone: no case of the
composite reaches a tie at acceptance through `onsite-counted serve`, since SPOT's cases all reject there, so
the Lean ties are checked by `CoreOnsite`'s `#guard`s on the same kinds of pair and the Python ties here."""

from __future__ import annotations

import dataclasses

import pytest

import reference as R
from verity_difftest import canonical

from verity.protocols.accounting.certifier.verifier import span
from verity.protocols.accounting.pous.verifier import streamed_wipe as sw
from verity.protocols.accounting.pous.verifier import wipe_period as WP

C, S = R.WIPE_CASES, R.SPAN_CASES
M, WC = C.M, C.WC
PARAMS = sw.WipeParams(8, 3, 1, 1_000, (("gpu", 3 * sw.LABEL_BYTES), ("host-dram", 5 * sw.LABEL_BYTES)))
TIES = (R.WIPE_NODE, *R.WIPE_COUNT)


@pytest.fixture(autouse=True)
def test_signer(tmp_path, monkeypatch):
    """The wipe's test signer's files under the test's own directory."""
    monkeypatch.setattr(M.ts, "DIR", tmp_path)


def pair(seed: int, *, alone: bool = True, start: bytes | None = None, end: bytes | None = None) -> tuple[dict, dict]:
    """The count's case (`span_case.build("registered", ...)`, its host `node0`) and a wipe of `node0` run between its
    two marks, its evidence giving the count's egress in whole MiB. `start` and `end`: the messages the count's marks
    are on in place of the wipe's. `alone`: the host's one link is the count's."""
    box: dict = {}
    session = bytes([seed]) * 32
    commitment = sw.WipeCommitment(b"node0", session, PARAMS)

    def between(source):
        certificate = source(start or sw.start_message(commitment))
        box["s"] = WC.run(PARAMS, M.deployed_signer(), server=b"node0", session_id=session, certificate=certificate)
        return end or sw.end_message(box["s"].signed)

    count = S.build("registered", seed, between=between, alone=alone)
    s = box["s"]
    named = span.named(span.params(count["params"]), span.commitment(bytes.fromhex(count["commitment"])))
    wipe = C.wipe_case(s.commitment, s.signed, s.trusted, egress_mib=sw.egress_units(named["egress_bits"]))
    return count, wipe


def entry(id: str, case: dict) -> tuple:
    """A case as the composition's evidence holds it: id, canonical params, commitment and evidence bytes."""
    return id, canonical(case["params"]), bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"])


def stray(count: dict, wipe: dict) -> str | None:
    """`holdTies` over the wipe's ties: the first tie whose two sides didn't name one value, else None."""
    ev = [entry(span.ID, count), entry(WP.COUNTED, wipe)]
    return next(filter(None, (t.strays(ev) for t in TIES)), None)


def with_evidence(wipe: dict, **change) -> dict:
    """The wipe's case with its evidence's signed session or count changed."""
    e = C.decode_evidence(bytes.fromhex(wipe["evidence"]))
    return {**wipe, "evidence": canonical(dataclasses.replace(e, **change).to_json()).encode().hex()}


@pytest.mark.parametrize("seed", range(3))
def test_a_wipe_between_the_counts_marks_holds_every_tie(seed):
    count, wipe = pair(seed)
    assert R.HOST_SPAN.answer(count)["verdict"] == R.ACCEPT
    got = R.COUNTED_WIPE.answer(wipe)
    mib = C.decode_evidence(bytes.fromhex(wipe["evidence"])).egress_mib
    assert got["verdict"] == R.ACCEPT and got["public"]["certified"] and got["public"]["egress_mib"] == mib
    ev = [entry(span.ID, count), entry(WP.COUNTED, wipe)]
    assert [len(t.values(ev) or []) for t in TIES] == [2] * len(TIES)
    assert stray(count, wipe) is None


def test_each_tie_finds_its_stray():
    count, wipe = pair(0)
    e = C.decode_evidence(bytes.fromhex(wipe["evidence"]))
    other_server = C.wipe_case(e.signed.commitment, e.signed, (), server=b"node1", egress_mib=e.egress_mib)
    assert stray(count, other_server) == named_apart("node")
    assert stray(count, with_evidence(wipe, signed=dataclasses.replace(e.signed, start_mark=bytes([9]) * 64))) \
        == named_apart("start mark")
    assert stray(*pair(0, start=bytes([3]) * 64)) == named_apart("start message")
    assert stray(*pair(0, end=bytes([6]) * 64)) == named_apart("end message")
    for n in (e.egress_mib - 1, e.egress_mib + 1):
        assert stray(count, with_evidence(wipe, egress_mib=n)) == named_apart("egress")


def named_apart(what: str) -> str:
    """`CoreOnsite`'s `wipeStray`: a stray of the wipe's tie `what`, naming the two subprotocols and neither value."""
    return f"{what}: {WP.COUNTED} and {span.ID} named different values"


def test_a_wipe_that_names_nothing_ties_nothing():
    """An unsigned wipe, or one whose evidence gives no count, names no value, and no tie rejects on it: the wipe's own
    verdict stands (undecided, or `NO_COUNT`)."""
    count, wipe = pair(1)
    for change in ({"signed": None}, {"egress_mib": None}):
        assert stray(count, with_evidence(wipe, **change)) is None
    assert R.COUNTED_WIPE.answer(with_evidence(wipe, egress_mib=None))["reason"].endswith(WP.NO_COUNT)


def test_the_count_is_on_its_hosts_one_link():
    """The count's parameters as the composite takes them refuse a host with a second link besides the memory
    challengers' (the count's standard site gives `node0` a link to `node1`), and parameters with no registered
    hardware; the count's own decoder takes both."""
    count, _ = pair(2, alone=False)
    got = R.HOST_SPAN.answer(count)
    assert got["verdict"] == R.REJECT and "links besides the memory challengers'" in got["reason"]
    assert S.reference(count)["verdict"] == R.ACCEPT
    plain = S.build("honest", 2)
    assert S.reference(plain)["verdict"] == R.ACCEPT
    assert "no registered hardware" in R.HOST_SPAN.answer(plain)["reason"]


KEY0 = "0" * 48


def site_with(*links: tuple[str, str, str]) -> dict:
    """`CoreOnsite`'s `siteWith`: two hosts, a network certifier and a memory challenger, joined by `links`, as
    `registered.site_of` reads a site."""
    locs = [{"id": "node0", "kind": "host", "key": ""}, {"id": "node1", "kind": "host", "key": ""},
            {"id": "cert0", "kind": "certifier", "key": KEY0}, {"id": "chal0", "kind": "challenger", "key": ""}]
    return {"locs": locs, "links": [{"id": i, "ends": (a, b), "beta": 1} for i, a, b in links]}


UPLINK, CHALLENGED = ("eth0", "node0", "cert0"), ("mc0", "chal0", "node0")
BETWEEN, BARE = ("hh0", "node0", "node1"), ("up1", "node0", "outside")


def refusal(site: dict, host: str) -> str | None:
    try:
        R.wipe_link(site, host)
    except span.Refused as e:
        return str(e)
    return None


def test_wipe_link_is_the_leans():
    """The cases of `CoreOnsite`'s `#guard`s on `wipeLink`."""
    assert R.wipe_link(site_with(UPLINK), "node0") == "eth0"
    assert R.wipe_link(site_with(CHALLENGED, ("eth0", "cert0", "node0")), "node0") == "eth0"
    for site, host in [(site_with(UPLINK), "cert0"), (site_with(UPLINK), "node9"), (site_with(UPLINK, CHALLENGED), "node1"),
                       (site_with(CHALLENGED), "node0"), (site_with(BARE), "node0"), (site_with(UPLINK, BETWEEN), "node0"),
                       (site_with(UPLINK, BARE), "node0")]:
        assert refusal(site, host) is not None, (host, site["links"])
    assert refusal(site_with(UPLINK, BETWEEN, CHALLENGED), "node0") == (
        "host node0: it has 2 links besides the memory challengers' (eth0, hh0); a count on one of them is not of what "
        "left on the others")


def test_mib_bytes_is_decimal():
    assert [R.mib_bytes(n) for n in (0, 1, 4096, -3)] == [b"0", b"1", b"4096", b"-3"]
