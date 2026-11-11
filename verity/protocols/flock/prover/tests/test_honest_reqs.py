"""The honest worker's requests in an inner session, held to their Lean (`Flock.Firewall.honestReqs`, the port
``firewall-honest-difftest`` of `verity/difftest`): for what an honest worker brings (`Flock.Firewall.Honest`), the
requests `rec_live`'s encoder writes (`encode_req`, and `test_rec_live`'s frame writers for a round's content) are Lean's,
byte for byte. On a real prover's recorded session the worker's data is read back out of its own requests (`honest_of`)
and Lean writes those requests again, byte for byte, and they finish under `flock-firewall`; Hypothesis samples sessions
around it, malformed ones included. `FirewallHonest` proves they finish for every schedule and every data that fits."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st
from verity_difftest.protocol import unhex
from verity_flock import rec_live as RL

from test_firewall_agreement import FLAT, SCHEDULE2, _bad_schedules, built, lean_run  # noqa: F401 (built: a fixture)
from test_rec_live import SCHEDULE, _bytes, _domain, _label, _obs, _sq

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[4] / "difftest" / "lean"
#: the flat GEMM selftest's shape, whose recorded session `test_firewall_agreement` relays
FLAT_SCHEDULE = RL.schedule_for({"dense_m": 25, "k_log": 22, "nbl": 3, "regions": 3})


# --- the reference: `rec_live`'s encoder on what `honestOf` reads -----------------------------------------------------------

def _field(j, k: str):
    if not isinstance(j, dict) or k not in j:
        raise D.Reject(f"{k}: missing")
    return j[k]


def _arr(j) -> list:
    if not isinstance(j, list):
        raise D.Reject("not an array")
    return j


def _str(j) -> str:
    if not isinstance(j, str):
        raise D.Reject("not a string")
    return j


def _nat(j) -> int:
    if type(j) is not int or j < 0:
        raise D.Reject("not a natural number")
    return j


def _hex(j) -> bytes:
    return unhex(_str(j))


def _at(a: list, i: int):
    if i >= len(a):
        raise D.Reject("an item too short")
    return a[i]


def _item(j) -> bytes:
    """An item's frames, as `Flock.Firewall.Absorb.frames` and `FirewallHonestDifftest.absorbOf` read it."""
    a = _arr(j)
    tag = a[0] if a and isinstance(a[0], str) else ""
    if tag == "domain":
        return _domain(_hex(_at(a, 1)))
    if tag == "label":
        return _label(_hex(_at(a, 1)))
    if tag == "bind":
        return _label(RL.LABEL_R1CS) + _bytes(_hex(_at(a, 1))) + _bytes(_hex(_at(a, 2)))
    if tag in ("cap", "nonce"):
        return _bytes(_hex(_at(a, 1)))
    if tag == "observe":
        kind = _nat(_at(a, 1))
        return _obs([_nat(e) for e in _arr(a[2] if len(a) > 2 else None)], kind)
    if tag == "squeeze":
        return _sq(0, _nat(_at(a, 1)))
    raise D.Reject("not an item")


def _schedule(s) -> dict:
    """`Flock.Firewall.schedule`: `rec_live.check_schedule`, and, since Lean's firewall holds no registered top, a root for
    each table a stream without a parent or a copy of level 0's top names."""
    try:
        RL.check_schedule(s)
    except ValueError as e:
        raise D.Reject(str(e)) from None
    def table(stream: str) -> str:
        return "/".join(stream.split("/")[:-1]) if "/" in stream else stream

    if not (all(r["level0"] == 0 or table(r["stream"]) in s["roots"] for r in s["rounds"])
            and all(v["parent"] != "" or table(n) in s["roots"] for n, v in s["streams"].items())):
        raise D.Reject("a table without a root")
    return s


def honest_reqs(case) -> list[str]:
    """The honest worker's requests, in hex, as `rec_live.encode_req` writes them: its Pings, Register, Hello (the
    schedule's, with Σ in its link, as the worker prints it), the Opens it sends before Commit, Commit, Link, each round
    after the Opens it sends just before it, its proofs, Finish."""
    s = _schedule(_field(case, "schedule"))
    h = _field(case, "honest")
    pings = [_hex(p) for p in _arr(_field(h, "pings"))]
    roots = [(_str(_at(_arr(e), 0)), _hex(_at(e, 1))) for e in _arr(_field(h, "roots"))]
    labels = _field(h, "labels")
    if not isinstance(labels, dict):
        raise D.Reject("labels: not an object")
    labels = {k: _hex(v) for k, v in labels.items()}
    opens = [[_str(x) for x in _arr(o)] for o in _arr(_field(h, "opens"))]
    rounds = [(b"".join(_item(i) for i in _arr(_field(r, "items"))), _nat(_field(r, "kind"))) for r in _arr(_field(h, "rounds"))]
    proofs = [(_str(_at(_arr(e), 0)), _nat(_at(e, 1)), _hex(_at(e, 2))) for e in _arr(_field(h, "proofs"))]
    sigma, root_f = _str(_field(h, "sigma")), _hex(_field(h, "root_f"))

    def opened(j: int) -> list[bytes]:
        return [RL.encode_req(RL.REQ_OPEN, n, s["streams"][n]["parent"], labels.get(n, b""), s["streams"][n]["at"])
                for n in (opens[j] if j < len(opens) else []) if n in s["streams"]]

    hello = {**s["hello"], "link": {**s["hello"]["link"], "sigma": sigma}}
    out = [RL.encode_req(RL.REQ_PING, p) for p in pings]
    out += [RL.encode_req(RL.REQ_REGISTER),
            RL.encode_req(RL.REQ_HELLO, json.dumps(hello, sort_keys=True, separators=(",", ":"), ensure_ascii=False))]
    out += opened(0)
    out += [RL.encode_req(RL.REQ_COMMIT, root_f, roots, []), RL.encode_req(RL.REQ_LINK, bytes(16 * s["link"]["y"]))]
    for i, r in enumerate(s["rounds"]):
        content, kind = rounds[i] if i < len(rounds) else (b"", 0)
        out += opened(i + 1) + [RL.encode_req(RL.REQ_ROUND, r["stream"], content + _sq(r["n"], kind), r["n"])]
    out += [RL.encode_req(RL.REQ_PROOF, t, rep, p) for t, rep, p in proofs]
    out.append(RL.encode_req(RL.REQ_FINISH))
    return [b.hex() for b in out]


# --- a session's requests read back to what the worker brought ----------------------------------------------------------

def items_of(content: bytes, first: bool) -> tuple[list, int]:
    """A round's content as the items its challenger absorbed and its closing squeeze's kind: the statement's binding
    (`LABEL_R1CS`, then the digest and level 0's cap) in a rep's first round, a BYTES frame of 8 bytes a nonce and any other
    a cap."""
    fr = RL.frames(content)
    assert fr and fr[-1][0] == RL.OP_SQUEEZE, "a round without its squeeze"
    items, i = [], 0
    while i < len(fr) - 1:
        op, kind, n, p = fr[i]
        if first and op == RL.OP_LABEL and p == RL.LABEL_R1CS and not any(x[0] == "bind" for x in items):
            assert [f[0] for f in fr[i + 1:i + 3]] == [RL.OP_BYTES] * 2, "a binding without its digest and cap"
            items.append(["bind", fr[i + 1][3].hex(), fr[i + 2][3].hex()])
            i += 3
            continue
        items.append(["domain", p.hex()] if op == RL.OP_DOMAIN else ["label", p.hex()] if op == RL.OP_LABEL
                     else ["nonce" if len(p) == RL.NONCE_BYTES else "cap", p.hex()] if op == RL.OP_BYTES
                     else ["observe", kind, [int.from_bytes(p[16 * j:16 * j + 16], "little") for j in range(n)]]
                     if op == RL.OP_OBSERVE else ["squeeze", kind])
        i += 1
    return items, fr[-1][1]


def honest_of(schedule: dict, reqs: list[bytes]) -> dict:
    """What the worker brought to a session (`honestOf`'s JSON), read from its requests ``reqs``: each Open at its place
    (before Commit, or just before the round after it), each round's content as its items. `honestReqs` of it is ``reqs``
    exactly when the worker sent them in an honest worker's order."""
    h = {"pings": [], "labels": {}, "opens": [[]], "rounds": [], "proofs": []}
    rs = [RL.decode_req(b) for b in reqs]
    i = 0
    while rs[i][0] == RL.REQ_PING:
        h["pings"].append(rs[i][1].hex())
        i += 1
    for t, *f in rs[i:]:
        if t == RL.REQ_HELLO:
            h["sigma"] = json.loads(f[0])["link"]["sigma"]
        elif t == RL.REQ_OPEN:
            h["labels"][f[0]] = f[2].hex()
            h["opens"][-1].append(f[0])
        elif t == RL.REQ_COMMIT:
            h["root_f"], h["roots"] = f[0].hex(), [[n, v.hex()] for n, v in f[1]]
        elif t == RL.REQ_LINK:
            h["opens"].append([])
        elif t == RL.REQ_ROUND:
            r = schedule["rounds"][len(h["rounds"])]
            items, kind = items_of(f[1], r["k"] == 0 and schedule["streams"][r["stream"]]["parent"] == "")
            h["rounds"].append({"items": items, "kind": kind})
            h["opens"].append([])
        elif t == RL.REQ_PROOF:
            h["proofs"].append([f[0], f[1], f[2].hex()])
    return h


def recorded() -> tuple[dict, list[bytes]]:
    """The flat GEMM selftest's honest session (the fetched fixture `test_firewall_agreement` relays) at `schedule_for`'s
    schedule of its shape, as `flock-circuit prove --verifier` sends it and `rec_live` encodes it: 30 Pings of 64 zero bytes
    measuring the link, Register, Hello (the record's text, without its coins), rep 0's Open (its challenger opens before
    level 0's tree is committed), Commit, Link, rep 0's rounds, rep 1's Open and rounds, both proofs, then Finish. The
    rounds are the record's own bytes."""
    rec_dir = FLAT / "records-cpu"
    assert (rec_dir / "honest.json").is_file(), (
        f"{FLAT} is not fetched: `uv run research data fetch-fixtures verity/protocols/flock/prover/tests/fixtures/hidden-outputs`")
    rec = json.loads((rec_dir / "honest.json").read_text())["record"]
    hello = json.loads(rec["hello"])
    assert json.dumps(hello, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == rec["hello"]
    del hello["coins"]
    lk = rec["link"]
    streams = {st["stream"]: st for st in rec["streams"]}
    opens = [RL.encode_req(RL.REQ_OPEN, n, st["parent"], st["label"].encode(), st["at"]) for n, st in streams.items()]
    rounds = {n: [RL.encode_req(RL.REQ_ROUND, n, bytes.fromhex(rd["msg"]), rd["n"]) for rd in st["rounds"]]
              for n, st in streams.items()}
    assert list(streams) == ["circuit/rep0", "circuit/rep1"]
    reqs = [RL.encode_req(RL.REQ_PING, bytes(64))] * 30 + [
        RL.encode_req(RL.REQ_REGISTER),
        RL.encode_req(RL.REQ_HELLO, json.dumps(hello, sort_keys=True, separators=(",", ":"), ensure_ascii=False)),
        opens[0],
        RL.encode_req(RL.REQ_COMMIT, bytes.fromhex(lk["root_f"]), [(t, bytes.fromhex(v)) for t, v in lk["root_b"].items()], []),
        RL.encode_req(RL.REQ_LINK, bytes(16 * len(lk["y"])))]
    reqs += rounds["circuit/rep0"] + [opens[1]] + rounds["circuit/rep1"]
    reqs += [RL.encode_req(RL.REQ_PROOF, "circuit", r, (rec_dir / f"honest.rep{r}.proof").read_bytes()) for r in (0, 1)]
    reqs.append(RL.encode_req(RL.REQ_FINISH))
    return FLAT_SCHEDULE, reqs


# --- sampled sessions ------------------------------------------------------------------------------------------------------

#: a plain string (`plainStr`), its characters in the Basic Multilingual Plane, which the harness's JSON writes as one escape
PLAIN = st.text(st.characters(min_codepoint=0x20, max_codepoint=0xFFFF, exclude_characters='"\\', exclude_categories=["Cs"]),
                max_size=6)
HEX = st.binary(max_size=24).map(bytes.hex)
CAP = st.integers(1, 2).flatmap(lambda k: st.binary(min_size=64 * k, max_size=64 * k)).map(bytes.hex)
KIND = st.integers(0, 255)
ITEM = st.one_of(
    st.tuples(st.just("domain"), HEX), st.tuples(st.just("label"), HEX), st.tuples(st.just("bind"), HEX, CAP),
    st.tuples(st.just("cap"), CAP), st.tuples(st.just("nonce"), st.binary(min_size=8, max_size=8).map(bytes.hex)),
    st.tuples(st.just("observe"), KIND, st.lists(st.integers(0, 2 ** 128 - 1), max_size=4)),
    st.tuples(st.just("squeeze"), KIND)).map(list)


def _honest(s: dict):
    names = sorted(s["streams"])
    return st.fixed_dictionaries({
        "pings": st.lists(HEX, max_size=2), "sigma": PLAIN, "root_f": HEX,
        "roots": st.lists(st.tuples(PLAIN, CAP).map(list), max_size=2),
        "labels": st.dictionaries(st.sampled_from(names), HEX, max_size=len(names)),
        "opens": st.lists(st.lists(st.sampled_from([*names, "elsewhere"]), max_size=2), max_size=len(s["rounds"]) + 2),
        "rounds": st.lists(st.fixed_dictionaries({"items": st.lists(ITEM, max_size=4), "kind": KIND}),
                           max_size=len(s["rounds"]) + 1),
        "proofs": st.lists(st.tuples(PLAIN, st.integers(0, 2 ** 32 - 1), HEX).map(list), max_size=2)})


#: edits that leave a case's `honest` unreadable
SPOILS = [lambda h: h.pop("sigma"), lambda h: h.update(root_f="abc"), lambda h: h.update(root_f="AB"),
          lambda h: h.update(pings=["zz"]), lambda h: h.update(roots=[["t"]]), lambda h: h.update(roots=[[1, "00"]]),
          lambda h: h.update(labels=["x"]), lambda h: h.update(labels={"x": 1}), lambda h: h.update(opens=[[1]]),
          lambda h: h.update(opens=[[], "x"]), lambda h: h.update(rounds=[{"items": [["what", "00"]], "kind": 0}]),
          lambda h: h.update(rounds=[{"items": [], "kind": -1}]), lambda h: h.update(rounds=[{"items": [[]], "kind": 0}]),
          lambda h: h.update(rounds=[{"items": [["observe", 1, [-1]]], "kind": 0}]),
          lambda h: h.update(rounds=[{"items": [["observe", 1]], "kind": 0}]),
          lambda h: h.update(rounds=[{"items": [["squeeze"]], "kind": 0}]), lambda h: h.update(rounds=[{"kind": 0}]),
          lambda h: h.update(rounds=[[]]), lambda h: h.update(proofs=[["t", 0]]),
          lambda h: h.update(proofs=[["t", True, "00"]])]


def _spoiled(case: dict, spoil) -> dict:
    h = json.loads(json.dumps(case["honest"]))
    spoil(h)
    return {**case, "honest": h}


def _json(v) -> bool:
    try:
        D.canonical(v)
    except TypeError:
        return False
    return True


SCHEDULES = [SCHEDULE, SCHEDULE2, FLAT_SCHEDULE]
#: the schedules the ZK server refuses that JSON without floats can write
REFUSED = [s for s in [*_bad_schedules(), {**SCHEDULE, "roots": {}}] if _json(s)]
CASES = st.sampled_from(SCHEDULES).flatmap(lambda s: _honest(s).map(lambda h: {"schedule": s, "honest": h}))
INPUTS = st.one_of(
    CASES, CASES, st.tuples(CASES, st.sampled_from(SPOILS)).map(lambda c: _spoiled(*c)),
    st.tuples(st.sampled_from(REFUSED), _honest(SCHEDULE)).map(lambda c: {"schedule": c[0], "honest": c[1]}))

PORT = D.Port("firewall-honest-difftest", "honest-reqs", reference=honest_reqs, inputs=INPUTS, package=LEAN)
server = D.server_fixture(PORT)


def test_a_real_provers_session_is_honest_reqs_byte_for_byte(server):
    """The recorded session's requests, read back to what the worker brought (`honest_of`): Lean's `honestReqs` of it is
    those requests, byte for byte, and `rec_live`'s encoder agrees."""
    schedule, reqs = recorded()
    case = {"schedule": schedule, "honest": honest_of(schedule, reqs)}
    kind, out = D.replay(PORT, case, server)
    assert kind == "output" and json.loads(out) == [b.hex() for b in reqs]


def test_lean_honest_reqs_of_a_real_session_finish_under_the_firewall(server, built):
    """`flock-firewall` (`Flock.Firewall.run`) finishes on Lean's requests for the recorded session, salting one row for
    each node of a committed top and each round's fresh top and message, as many as the schedule's."""
    schedule, reqs = recorded()
    kind, out = D.replay(PORT, {"schedule": schedule, "honest": honest_of(schedule, reqs)}, server)
    rows = sum(schedule["roots"].values()) + sum(sum(r["tops"]) + r["commits"] for r in schedule["rounds"])
    salts = [hashlib.shake_256(b"honest-salt" + i.to_bytes(8, "little")).digest(192) for i in range(rows)]
    [got] = lean_run([("recorded", schedule, [bytes.fromhex(x) for x in json.loads(out)])], [salts])
    assert got["record"]["end"] == "finish" and len(got["record"]["rounds"]) == len(schedule["rounds"])


def test_honest_reqs_is_rec_lives_encoding(server):
    """Sampled sessions at `test_rec_live`'s schedule, the two-stream one and the flat selftest's: Lean's requests are
    `rec_live`'s, byte for byte, and both refuse a schedule the ZK server refuses or a worker's data that does not read."""
    stats = D.run(PORT, examples=300, server=server)
    assert stats.outputs and stats.rejects
