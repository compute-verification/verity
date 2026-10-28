"""`audit.Registry` held to the verifier's registry in Lean (`RemoteAudit.Registry`, the definition, served by
`remote-audit-difftest`): at most one live registration a node, a second refused and counted, sessions over the live
one as its k-th, their events, `close` counting each session as it stands, R (`drawn`), and `registry/v0` written and
read back. Drawn cases are calls over a few nodes from an empty registry or one read back, with now and then a field
either side refuses; recorded ones run a recorded session (`ra_cases.SCENARIOS`) through the registry, so that `close`
counts the outcome the definition gives it."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import shutil

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from verity.core.protocols.remote_bandwidth import audit, codec, receipts

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PORT = D.Port(M.EXE, "registry", M.registry, st.nothing(), package=M.PACKAGE, of_record=M.OF_RECORD, rejects=M.REFUSALS)
server = D.server_fixture(PORT)

NODES = (M.NODE, b"node-b", b"\x07")
ZERO = "00" * audit.NONCE_BYTES


def expand(seed: bytes, n: int) -> str:
    return b"".join(hashlib.sha512(seed + c.to_bytes(4, "big")).digest() for c in range(-(-n // 64)))[:n].hex()


def rare(draw, n: int = 30) -> bool:
    return draw(st.integers(0, n)) == 0


def node(draw) -> str:
    if rare(draw, 40):
        return draw(st.sampled_from(["", "00" * 65]))
    return draw(st.sampled_from(NODES)).hex()


def register_op(draw) -> dict:
    c = draw(st.sampled_from([M.P512, codec.DEFAULT]))
    kind = draw(st.sampled_from(receipts.KINDS))
    v = {"op": "register", "node": node(draw), "blocks": draw(st.sampled_from([1, 6])),
         "payload_root": expand(draw(st.binary(max_size=4)), 64), "codec": M.codec_json(c), "receipt": kind,
         "bind": False, "layout": receipts.LAYOUT, "rows": 16 if kind == receipts.SKETCH_SHA512 else 0,
         "nonce": expand(draw(st.binary(max_size=4)), audit.NONCE_BYTES)}
    if rare(draw):
        k, bad = draw(st.sampled_from([("bind", True), ("nonce", "00" * 31), ("blocks", 0), ("layout", "fused/v0"),
                                       ("receipt", "x"), ("rows", 32), ("codec", {"field": "p512", "rounds": 1,
                                                                                  "masked": False})]))
        v[k] = bad
    return v


def event_op(draw, n: str) -> dict:
    """An event for session 0 or 1 of node `n`'s live registration: one that releases a coin, takes a root (now and
    then a malformed one), closes a window or verifies it with no openings."""
    t, now = draw(st.integers(0, 1)), draw(st.integers(0, 12))
    root = draw(st.sampled_from([expand(b"root %d" % now, 64), "00" * 63]))
    e = draw(st.sampled_from([
        {"kind": "start", "now": now}, {"kind": "boundary", "now": now},
        {"kind": "receive", "t": t, "i": 0, "root": root, "runs": None, "change": None, "step": 0, "now": now},
        {"kind": "close", "t": t, "now": now}, {"kind": "verify", "t": t, "openings": []}]))
    return {"op": "event", "node": n, "session": draw(st.sampled_from([0, 0, 1])), "event": e}


def session_op(draw, index: int | None) -> dict:
    """A session over the registration drawn `index`-th in the case, or over one given in full that no registry drew;
    now and then its two windows drawn on one openings coin (`groups_per_coin` 2)."""
    windows = draw(st.integers(1, 2))
    pace = {"k": draw(st.integers(1, 2)), "window_ns": 5} if windows == 2 and rare(draw, 4) else None
    g = 2 if windows == 2 and rare(draw, 3) else 1
    v = {"op": "session", "registration": index,
         "params": {"deadline_ns": 100, "deferred": False, "lifetime": 4, "min_passes": 1, "pace": pace,
                    "samples": 4, "window_ns": 1, "windows": windows} | ({"groups_per_coin": g} if g != 1 else {}),
         "coins": [expand(b"coin %d" % u, 32) for u in range(windows + -(-windows // g))], "append_coins": [],
         "not_before_ns": draw(st.integers(0, 2)), "index": draw(st.sampled_from([None, None, 0, 1, -1]))}
    if index is None or rare(draw, 10):
        v["registration"] = M.registration_json(audit.Registration(
            bytes.fromhex(node(draw)) or M.NODE, 6, bytes(64), bytes(32), M.P512, receipts.DEFAULT, False, receipts.LAYOUT,
            receipts.sketch_rows(receipts.DEFAULT)))
    if rare(draw):
        k, bad = draw(st.sampled_from([("coins", []), ("registration", -1), ("index", True)]))
        v[k] = bad
    return v


def read_back(draw) -> dict:
    """A registry/v0 over the nodes, now and then with a field `from_json` refuses."""
    outcomes = st.lists(st.sampled_from(["accepted", audit.UNENDED, "receipt", "late"]), max_size=3)
    live = draw(st.sets(st.sampled_from(NODES)))
    v = {"schema": audit.REGISTRY_SCHEMA,
         "live": {n.hex(): {"registration": expand(n + b" live", 64), "sessions": draw(outcomes)} for n in live},
         "closed": [{"node": n.hex(), "registration": expand(n + b" %d" % u, 64), "sessions": draw(outcomes)}
                    for u, n in enumerate(draw(st.lists(st.sampled_from(NODES), max_size=3)))],
         "refused": {n.hex(): draw(st.integers(1, 3)) for n in draw(st.sets(st.sampled_from(NODES)))}}
    if rare(draw, 6):
        edit = draw(st.integers(0, 6))
        if edit == 0:
            v["schema"] = "verity/remote-bandwidth/registry/v1"
        elif edit == 1:
            v["closed"].append({"node": "07", "registration": "00" * 63, "sessions": []})
        elif edit == 2:
            v["closed"].append({"node": "07", "registration": "00" * 64, "sessions": [""]})
        elif edit == 3:
            v["refused"]["07"] = 0
        elif edit == 4:
            v["refused"]["0A"] = 1
        elif edit == 5:
            v["live"][""] = {"registration": "00" * 64, "sessions": []}
        else:
            del v[draw(st.sampled_from(sorted(v)))]
    return v


def script(draw, n: str, k: int) -> list[dict]:
    """The first few events of session `k` over node `n`'s registration in a window's order, now and then off it: a
    malformed root, a step below the first, a boundary sooner than the window, a close long past it."""
    now = draw(st.integers(0, 3))
    root = draw(st.sampled_from([expand(b"root %d" % now, 64)] * 3 + ["00" * 63]))
    events = [{"kind": "start", "now": now},
              {"kind": "receive", "t": 0, "i": 0, "root": root, "runs": None, "change": None,
               "step": draw(st.sampled_from([0, 0, 0, -1])), "now": now + 1},
              {"kind": "boundary", "now": now + draw(st.sampled_from([2, 2, 0]))},
              {"kind": "close", "t": 0, "now": now + draw(st.sampled_from([3, 3, 200]))},
              {"kind": "verify", "t": 0, "openings": []}]
    return [{"op": "event", "node": n, "session": k, "event": e} for e in events[:draw(st.integers(0, 5))]]


@st.composite
def registry_inputs(draw) -> dict:
    """Calls over the nodes: most often a registration with a session over it and a few of its events, else one call
    alone. The strategy keeps which nodes are live, so that a session names a registration drawn in the case by its
    place among those drawn."""
    registry = read_back(draw) if draw(st.integers(0, 2)) == 0 else None
    live = {n: (None, 0) for n in (registry or {}).get("live", ())}
    ops, drawn = [], []
    for _ in range(draw(st.integers(1, 6))):
        kind = draw(st.sampled_from(["fresh", "fresh", "fresh", "register", "session", "event", "close", "close",
                                     "live", "drawn"]))
        if kind in ("fresh", "register"):
            r = register_op(draw)
            ops.append(r)
            n = r["node"]
            if n not in live:
                live[n] = (len(drawn), 0)
                drawn.append(n)
            if kind == "fresh":
                index, k = live[n]
                ops.append(session_op(draw, index))
                if index is not None and ops[-1]["index"] in (None, k):
                    live[n] = (index, k + 1)
                ops += script(draw, n, k)
        elif kind == "session":
            ops.append(session_op(draw, draw(st.integers(0, len(drawn) - 1)) if drawn else None))
        elif kind == "event":
            ops.append(event_op(draw, draw(st.sampled_from(drawn)) if drawn else node(draw)))
        else:
            n = node(draw)
            ops.append({"op": kind, "node": n})
            if kind == "close":
                live.pop(n, None)
    return {"source": "" if rare(draw, 60) else M.SOURCE.hex(), "registry": registry, "ops": ops}


def test_registry(server) -> None:
    stats = D.run(dataclasses.replace(PORT, inputs=registry_inputs()), examples=300, server=server)
    assert stats.outputs > 150, stats


def recorded(name: str, cut: int | None = None) -> dict:
    """Scenario `name`'s session, its first `cut` events, run through the registry: its registration drawn as the
    scenario's store was (`ra_cases.SOURCE`, a zero nonce), a second refused, the first closed, and a third drawn."""
    case, _ = M.recorded(name)
    reg = case["registration"]
    register = {"op": "register", "nonce": ZERO} | {k: reg[k] for k in (
        "node", "blocks", "payload_root", "codec", "receipt", "bind", "layout", "rows")}
    session = {"op": "session", "registration": 0, "index": 0} | {k: case[k] for k in (
        "params", "coins", "append_coins", "not_before_ns")}
    events = [{"op": "event", "node": reg["node"], "session": 0, "event": e} for e in case["events"][:cut]]
    return {"source": M.SOURCE.hex(), "registry": None, "ops": [
        register, session, *events, register, {"op": "close", "node": reg["node"]},
        {"op": "drawn", "node": reg["node"]}, register | {"nonce": "01" * audit.NONCE_BYTES},
        {"op": "live", "node": reg["node"]}, {"op": "drawn", "node": reg["node"]}]}


@pytest.mark.parametrize("name", ["honest", "honest-appends", "honest-block-sha512", "few", "late", "receipt-path-leaf",
                                  "codec-block", "refusals", "honest-paced-one-flight", "after-boundary-one-flight",
                                  "honest-shared-4", "skipper-shared-4", "few-shared",
                                  "honest-shared-paced-one-flight"])
def test_a_recorded_session_is_counted_at_close_as_the_lean_ends_it(server, name) -> None:
    case, _ = M.recorded(name)
    reg = M.registration_of(case["registration"])
    kind, lean = D.replay(PORT, recorded(name), server)
    out = json.loads(lean)
    assert kind == "output" and out["results"][0] == {"codec_seed": reg.codec_seed.hex(), "digest": reg.digest.hex()}
    assert out["registry"]["closed"] == [{"node": M.NODE.hex(), "registration": reg.digest.hex(),
                                          "sessions": [M.SCENARIOS[name][3]]}]
    assert out["registry"]["refused"] == {M.NODE.hex(): 1}
    assert [out["results"][k] for k in (-4, -1)] == [1, 2]


def test_a_registration_that_binds_tiles_is_refused_as_the_lean_refuses_it(server) -> None:
    """A registration that would bind PoUW's tile outputs: for a node with one live, refused and counted, as any second
    registration is; for one with none, refused by both registries outright (the Lean covers receipts of chunks
    alone), drawing nothing."""
    v = recorded("honest")
    bound = v["ops"][0] | {"bind": True}
    assert D.replay(PORT, {"source": M.SOURCE.hex(), "registry": None, "ops": [bound]}, server)[0] == "reject"
    out = json.loads(D.replay(PORT, {"source": M.SOURCE.hex(), "registry": None,
                                     "ops": [v["ops"][0], bound, {"op": "drawn", "node": M.NODE.hex()}]}, server)[1])
    assert out["results"][1:] == [{"refused": True}, 1]
    assert out["registry"]["refused"] == {M.NODE.hex(): 1}


def test_a_session_that_has_not_ended_is_counted_unended(server) -> None:
    v = recorded("honest", cut=len(M.recorded("honest")[0]["events"]) // 2)
    out = json.loads(D.replay(PORT, v, server)[1])
    assert out["registry"]["closed"][0]["sessions"] == [audit.UNENDED]


def test_a_registry_read_back_closes_its_registration_and_runs_no_session(server) -> None:
    v = recorded("honest")
    v["ops"] = v["ops"][:-6]
    first = json.loads(D.replay(PORT, v, server)[1])["registry"]
    assert first["live"] == {M.NODE.hex(): {"registration": M.registration_of(
        M.recorded("honest")[0]["registration"]).digest.hex(), "sessions": ["accepted"]}}
    session = copy.deepcopy(v["ops"][1]) | {"registration": M.recorded("honest")[0]["registration"], "index": 1}
    again = {"source": M.SOURCE.hex(), "registry": first,
             "ops": [session, v["ops"][0], {"op": "close", "node": M.NODE.hex()}]}
    out = json.loads(D.replay(PORT, again, server)[1])
    assert out["results"][:2] == [{"refused": True}] * 2
    assert out["registry"]["closed"][0]["sessions"] == ["accepted"]
    assert out["registry"]["refused"] == {M.NODE.hex(): 1}
