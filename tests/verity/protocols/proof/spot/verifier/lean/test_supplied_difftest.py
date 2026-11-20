"""A program's supplied inputs (`Spot.Supplied.piece`, `supplied_marks`): the marks of its top-level `inputs`, each
marked input's nodes all `SuppliedInput<w>_v1`, its function one of the registry's with a body, its value under the
verifier's coins read by its layout, held to the Python mirror (`one_stage.registered.marks`, `of_mark`); and the
registry's closed list in Lean equal to the format's (`verity.circuits.supplied.FUNCTIONS`)."""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import re
import shutil
from pathlib import Path

import pytest
from hypothesis import note, strategies as st

import verity_difftest as D
from verity.circuits import supplied as SU
from verity.protocols.proof.spot.verifier.one_stage import registered as G
from verity.protocols.accounting.memory.pous.attester import memory_attester as MA

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proof" / "spot" / "verifier" / "lean")
PIECES = "spot-difftest"
needs_lake = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

SALT = {"supplied": "pous.setup.salt/v1", "args": {}, "layout": "le", "bits": 192}
ONE = {"supplied": "const.u16/v1", "args": {"value": 1}, "layout": "le", "bits": 16}
CALLEES = ["Input16_v1", "RegisteredInput16_v1", "SuppliedInput16_v1", "SuppliedInput1_v1"]


def desc(inputs, fns) -> dict:
    """A descriptor whose root leads with one node a callee index of ``fns`` and whose top-level `inputs` is ``inputs``."""
    return {"root": "R", "callees": CALLEES, "inputs": inputs,
            "definitions": {"R": {"body": {"nodes": [{"fn": f} for f in fns]}}}}


def reference(j):
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    try:
        ms = G.marks(j.get("program"))
        coins = j.get("coins")
        if not isinstance(coins, dict) or not all(isinstance(h, str) and re.fullmatch(r"(?:[0-9a-f]{2})*", h)
                                                  for h in coins.values()):
            raise ValueError("coins")
        commitments = j.get("commitments", {})
        if not isinstance(commitments, dict) or not all(isinstance(h, str) and re.fullmatch(r"(?:[0-9a-f]{2})*", h)
                                                        for h in commitments.values()):
            raise ValueError("commitments")
        widths = j.get("widths")
        if not isinstance(widths, dict):
            raise ValueError("widths")
        log, binding, rec = j.get("log"), j.get("binding"), j.get("record")
        if not isinstance(log, list):
            raise ValueError("log")
        if binding is not None and not (isinstance(binding, dict) and isinstance(binding.get("stream"), str)
                                        and isinstance(binding.get("prover"), str)):
            raise ValueError("binding")
        keys = j.get("attester_keys", [])
        if not (isinstance(keys, list) and all(isinstance(h, str) and re.fullmatch(r"(?:[0-9a-f]{2})*", h) for h in keys)):
            raise ValueError("attester_keys")
        pinned = [bytes.fromhex(h) for h in keys]
        held = {f: bytes.fromhex(h) for f, h in coins.items()}
        commit = {f: bytes.fromhex(h) for f, h in commitments.items()}
        G.order(ms, {v: G.of_mark(held, commit, v, m, pinned) for v, m in ms.items()}, held, commit, j.get("released"),
                len(log))
        out = {}
        for v, m in ms.items():
            w = G._nat(widths.get(v))
            if w is None:
                raise ValueError("width")
            out[v] = [G.row_leaves_of(r, lambda k, n: G._source_bytes(rec, v, k, n, list(ms), log, binding), w,
                                      m["layout"] == "digest")
                      for r in G.of_mark(held, commit, v, m, pinned)]
            c = j["cuts"].get(v) if isinstance(j.get("cuts"), dict) else None
            if c is not None:
                if not (isinstance(c, list) and len(c) == 2 and all(G._nat(x) is not None for x in c)):
                    raise ValueError("cuts")
                out[v] = G.cut(out[v], *c)
        return out
    except ValueError as e:
        raise D.Reject(str(e)) from e


JUNK = st.one_of(st.none(), st.booleans(), st.text(max_size=3), st.integers(-3, 3), st.lists(st.integers(-1, 3), max_size=2))
RECORD = {"population": 3, "window": {"start": 0}, "registered": [{"value": "w", "root": "11" * 64}],
          "roots": [{"port": "p", "root": "22" * 64}]}
BINDING = {"stream": "aa", "prover": "bb"}
ROOTS = [{"supplied": "registered.root/v1", "args": {"value": "w"}, "layout": "digest", "bits": 512},
         {"supplied": "record.root/v1", "args": {"port": "p"}, "layout": "halves", "bits": 512},
         {"supplied": "record.root/v1", "args": {"port": "q"}, "layout": "halves", "bits": 512},
         {"supplied": "window.salt/v1", "args": {}, "layout": "halves", "bits": 512},
         {"supplied": "log.base/v1", "args": {}, "layout": "halves", "bits": 64}]
PEARL = {"supplied": "pouw.pearlplus.salt/v1", "args": {}, "layout": "le", "bits": 256}
MARKS = st.one_of(st.just(SALT), st.just(ONE), st.just(PEARL), st.sampled_from(ROOTS), st.builds(lambda v: {**ONE, "args": {"value": v}}, st.integers(0, 70000)),
                  st.builds(lambda f, lay, b: {"supplied": f, "args": {}, "layout": lay, "bits": b},
                            st.sampled_from([*SU.FUNCTIONS, "no.such/v1"]), st.sampled_from([*SU.LAYOUTS, "be"]),
                            st.sampled_from([0, 16, 192, 1 << 16, (1 << 16) + 8])))


@st.composite
def instances(draw) -> dict:
    """One to three inputs, each run, registered or supplied with a drawn mark, its nodes perhaps of the wrong kind; a
    coin perhaps missing, short or present; widths perhaps missing."""
    inputs, fns = [], []
    for i in range(draw(st.integers(1, 3))):
        n = draw(st.integers(0, 2))
        kind = draw(st.sampled_from(["run", "registered", "supplied", "supplied"]))
        fn = {"run": 0, "registered": 1, "supplied": draw(st.sampled_from([2, 2, 3]))}[kind]
        if draw(st.integers(0, 9)) == 0:
            fn = draw(st.sampled_from([0, 1, 2, 7, -1]))
        fns += [fn] * n
        inputs.append([f"x{i}", n, draw(MARKS)] if kind == "supplied" else [f"x{i}", n])
    if draw(st.integers(0, 9)) == 0:
        inputs.append(draw(JUNK))
    coins = draw(st.sampled_from([{}, {"pous.setup.salt/v1": "ab" * 24}, {"pous.setup.salt/v1": "ab" * 23},
                                  {"pous.setup.salt/v1": 5}, {"window.salt/v1": "07" * 32}, {"window.salt/v1": ""},
                                  {"pouw.pearlplus.salt/v1": "5a" * 32}, {"pouw.pearlplus.salt/v1": "5a" * 31}]))
    rec = draw(st.sampled_from([None, RECORD, {**RECORD, "population": 0}, {"window": 3}]))
    log = draw(st.sampled_from([[], [{"population": 5}], [{"population": -1}], 7]))
    binding = draw(st.sampled_from([None, BINDING, {"stream": 1}]))
    widths = {x[0]: draw(st.sampled_from([16, 16, 1, 3, -1])) for x in inputs if isinstance(x, list) and x
              and isinstance(x[0], str)}
    if draw(st.integers(0, 9)) == 0:
        widths = draw(JUNK)
    cuts = {x[0]: draw(st.one_of(st.tuples(st.integers(0, 16), st.integers(0, 16)).map(list), JUNK))
            for x in inputs if isinstance(x, list) and x and isinstance(x[0], str) and draw(st.booleans())}
    return {"program": desc(inputs, fns), "coins": coins, "widths": widths, "record": rec, "log": log,
            "binding": binding, **({"cuts": cuts} if cuts else {})}


INPUTS = st.one_of(*[instances() for _ in range(6)], JUNK)
MARKED = D.Port(PIECES, "supplied_marks", reference, INPUTS, package=PACKAGE, of_record="lean")

pieces = D.server_fixture(MARKED)


@needs_lake
def test_the_supplied_marks_piece_agrees(pieces):
    stats = D.run(MARKED, examples=1500, server=pieces)
    assert stats.outputs > 100 and stats.rejects > 0, stats


SETUP = {"program": desc([["store", 1], ["salt", 1, SALT], ["one", 1, ONE]], [1, 2, 2]),
         "coins": {"pous.setup.salt/v1": bytes(range(24)).hex()}, "widths": {"salt": 16, "one": 16},
         "record": RECORD, "log": [], "binding": BINDING}


#: the setup's salt read beside the root the record registers for ``w`` (`registered.root/v1`), as input ``rw``
ORDERED = {"program": desc([["salt", 1, SALT], ["rw", 1, ROOTS[0]]], [2, 3]), "widths": {"salt": 16, "rw": 1}}


def test_the_order_rule_names_the_commitments_registered_after_the_coin():
    """`registered.order` refuses by name (E_SUPPLIED_ORDER) a commitment C's marks read whose place is at or after
    the place of a coin they read, and needs a place for such a coin."""
    j = {**SETUP, **ORDERED}
    with pytest.raises(D.Reject, match=r"^E_SUPPLIED_ORDER: \['w'\] registered after pous.setup.salt/v1, which a "
                                       r"supplied value reads$"):
        reference({**j, "released": {"pous.setup.salt/v1": 0}})
    with pytest.raises(D.Reject, match="names no place of pous.setup.salt/v1's coin"):
        reference(j)
    assert reference({**j, "released": {"pous.setup.salt/v1": 1}})["salt"]


@needs_lake
@pytest.mark.parametrize("name, change, want", [
    ("the setup's salt and the held 1", lambda j: j, {"salt": [[0x0100 + 0x0202 * i for i in range(12)]], "one": [[1]]}),
    ("no coin for the salt: no statement", lambda j: {**j, "coins": {}}, None),
    ("a salt of 23 bytes: no statement", lambda j: {**j, "coins": {"pous.setup.salt/v1": "00" * 23}}, None),
    ("a mark on registered nodes", lambda j: {**j, "program": desc([["store", 1, SALT]], [1])}, None),
    ("supplied nodes with no mark", lambda j: {**j, "program": desc([["salt", 1]], [2])}, None),
    ("a function the registry lacks", lambda j: {**j, "program": desc([["one", 1, {**ONE, "supplied": "x/v1"}]], [2])},
     None),
    ("a value that grows with the population", lambda j: {**j, "program": desc(
        [["head", 1, {"supplied": "pous.session.header/v1", "args": {"head": 1 << 20}, "layout": "le",
                      "bits": 8 << 20}]], [2])}, None),
    ("a function with no body yet: no statement", lambda j: {**j, "program": desc(
        [["seed", 1, {"supplied": "poub.read.pass_seed/v1", "args": {"window": 0, "pass": 1},
                      "layout": "halves", "bits": 512}]], [2])}, None),
    ("the log's base after two records", lambda j: {**j, "log": [{"population": 2}, {"population": 3}], "program": desc(
        [["base", 1, ROOTS[4]]], [2]), "widths": {"base": 16}}, {"base": [[0, 0, 5, 0]]}),
    ("Pearl+'s epoch salt, 16 leaves of 16 bits", lambda j: {**j, "coins": {"pouw.pearlplus.salt/v1": bytes(range(32)).hex()},
                                                         "program": desc([["pp", 1, PEARL]], [2]), "widths": {"pp": 16}},
     {"pp": [[0x0100 + 0x0202 * i for i in range(16)]]}),
    ("Pearl+'s salt of 31 bytes: no statement", lambda j: {**j, "coins": {"pouw.pearlplus.salt/v1": "00" * 31},
                                                           "program": desc([["pp", 1, PEARL]], [2]), "widths": {"pp": 16}},
     None),
    ("a port the record doesn't name", lambda j: {**j, "program": desc([["r", 1, ROOTS[2]]], [2]),
                                                  "widths": {"r": 16}}, None),
    ("the setup's salt cut into the three rows of four leaves its reads fix", lambda j: {**j, "cuts": {"salt": [3, 4]}},
     {"salt": [[0x0100 + 0x0202 * i for i in range(4 * r, 4 * r + 4)] for r in range(3)], "one": [[1]]}),
    ("a cut its leaves aren't: the one row", lambda j: {**j, "cuts": {"salt": [5, 4]}},
     {"salt": [[0x0100 + 0x0202 * i for i in range(12)]], "one": [[1]]}),
    ("a cut not [n, k]", lambda j: {**j, "cuts": {"salt": [3]}}, None),
    ("a coin released after the record's root it is read beside", lambda j: {**j, **ORDERED, "released": {
        "pous.setup.salt/v1": 1}}, {"salt": [[0x0100 + 0x0202 * i for i in range(12)]], "rw": [G.digest_leaves(bytes.fromhex("11" * 64))]}),
    ("W's root registered after the setup's salt: E_SUPPLIED_ORDER", lambda j: {**j, **ORDERED, "released": {
        "pous.setup.salt/v1": 0}}, None),
    ("no place of the coin read beside a root", lambda j: {**j, **ORDERED}, None),
    ("a place that is no natural", lambda j: {**j, **ORDERED, "released": {"pous.setup.salt/v1": -1}}, None),
    ("a place of no supplied function", lambda j: {**j, **ORDERED, "released": {"no.such/v1": 1}}, None),
    ("the log's base after the coin of two records", lambda j: {
        **j, "log": [{"population": 2}, {"population": 3}], "program": desc([["salt", 1, SALT], ["base", 1, ROOTS[4]]],
                                                                          [2, 2]),
        "widths": {"salt": 16, "base": 16}, "released": {"pous.setup.salt/v1": 1}}, None),
    ("the log's base before it", lambda j: {
        **j, "log": [{"population": 2}, {"population": 3}], "program": desc([["salt", 1, SALT], ["base", 1, ROOTS[4]]],
                                                                          [2, 2]),
        "widths": {"salt": 16, "base": 16}, "released": {"pous.setup.salt/v1": 2}},
     {"salt": [[0x0100 + 0x0202 * i for i in range(12)]], "base": [[0, 0, 5, 0]]}),
    ("a window salt with no binding", lambda j: {**j, "binding": None, "coins": {"window.salt/v1": "07"},
                                                 "program": desc([["s", 1, ROOTS[3]]], [2]), "widths": {"s": 16}}, None),
], ids=lambda x: x if isinstance(x, str) else "")
def test_named_cases(pieces, name, change, want):
    j = change(SETUP)
    if want is None:
        with pytest.raises(D.Reject):
            reference(j)
        assert pieces.call("supplied_marks", j)[0] != "output"
    else:
        assert reference(j) == want
        assert pieces.call("supplied_marks", j) == ("output", D.canonical(want))


def test_the_registrys_list_is_the_formats():
    text = (PACKAGE / "Spot" / "Supplied.lean").read_text()
    body = re.search(r"def FUNCTIONS : List String :=\s*\[(.*?)\]", text, re.S)
    assert body is not None
    assert re.findall(r'"([^"]+)"', body[1]) == list(SU.FUNCTIONS)


ROUND, HEADER, CONTEXT = "pous.session.round_coin/v1", "pous.session.header/v1", "pous.session.context/v1"
MESSAGE = "pous.session.message/v1"
#: the circuits' own test of the three bodies: the vectors, `held` (the verifier's coins and commitments) and `REFUSALS`, each
#: a refusal with its text, which Lean is held to here
_CASES = Path(__file__).parents[5] / "circuits" / "test_supplied.py"
_spec = importlib.util.spec_from_file_location("session_cases", _CASES)
CASES = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CASES)
VECTORS = CASES.VECTORS
#: the configuration's pin of the memory attester's key, hex
PINNED = [h.hex() for h in CASES.PINNED]


def fixed(label: str, n: int) -> bytes:
    out, i = b"", 0
    while len(out) < n:
        out += hashlib.sha512(label.encode() + i.to_bytes(4, "big")).digest()
        i += 1
    return out[:n]


def mark_of(fn: str, args: dict, head: int | None = None) -> dict:
    if fn == ROUND:
        return {"supplied": fn, "args": args, "layout": "halves", "bits": 96}
    bits = 848 if fn == CONTEXT else 512 if fn == MESSAGE else 8 * (head + head % 2)
    return {"supplied": fn, "args": args, "layout": "le", "bits": bits}


def session_instance(fn: str, args: dict, tables: tuple[dict, dict], bits: int | None = None, released=...) -> dict:
    """One input marked ``fn`` read at 16-bit leaves, the verifier holding ``tables``, its ``(coins, commitments)``, and the
    round coin's, the header's and the message's release (the coin held under each) at place 1 unless ``released`` says otherwise (None: no
    places)."""
    mark = mark_of(fn, args, args.get("head"))
    if bits is not None:
        mark = {**mark, "bits": bits}
    coins, commitments = tables
    released = {fn: 1} if released is ... and fn in (ROUND, HEADER, MESSAGE) else None if released is ... else released
    return {"program": desc([["x0", 1, mark]], [2]), "coins": {f: c.hex() for f, c in coins.items()},
            "commitments": {f: c.hex() for f, c in commitments.items()}, "attester_keys": PINNED, "widths": {"x0": 16},
            "record": RECORD, "log": [], "binding": BINDING, **({} if released is None else {"released": released})}


@functools.cache
def honest(k: int, blocks: int, key_label: int, seed: int, picks: tuple[int, ...]) -> dict:
    """A session the memory attester writes: ``k`` coins over ``blocks`` blocks, each at ``picks[j] % blocks``: the commitment
    the verifier holds before its first coin (the header's fields, the key by its hash, the LMS key and the place), and what
    the session releases after it (the openings and the coin key, how it ended and its rounds, the message and its
    signature). The LMS key, the message and its signature are the vectors', over their session."""
    session = bytes.fromhex(VECTORS["commitment"]["session"])
    key = MA.HM96.key_from_label(f"difftest-key{key_label}".encode())
    salts = [fixed(f"salt{seed}/{j}", MA.HM96.salt_bytes) for j in range(k)]
    tree = MA.CoinTree(session, blocks, [picks[j] % blocks for j in range(k)], key, salts)
    c = MA.SessionCommitment(session, tree.root, key, blocks, k, 854_000 + seed, 2056, fixed(f"binding{seed}", 64))
    return {"commitment": {n: v for n, v in c.to_json().items() if n != "coin_key"}, "coin_key": key.hex(),
            "ended": "complete", "rounds": k, "place": 0,
            "key": VECTORS["key"], "message": VECTORS["message"], "signature": VECTORS["signature"],
            "openings": [tree.opening(j).to_json() for j in range(k)]}


def _flip(h: str, at: int = 0) -> str:
    return h[:at] + ("1" if h[at] == "0" else "0") + h[at + 1:]


def _op(d, j, **kw):
    for o in d["openings"]:
        if o["j"] == j:
            o.update(kw)


def _c(d, **kw):
    d["commitment"].update(kw)


#: each a change to a session that leaves it honest or breaks it in one way a body reads (a part is the commitment's, the
#: coin's openings, or how the session ended)
MUTATIONS = {
    "none": lambda d, j: None,
    "index+1": lambda d, j: _op(d, j, index=d["openings"][j]["index"] + 1) if j < len(d["openings"]) else None,
    "index at blocks": lambda d, j: _op(d, j, index=d["commitment"]["blocks"]),
    "index 2^64": lambda d, j: _op(d, j, index=1 << 64),
    "salt flipped": lambda d, j: _op(d, j, salt=_flip(d["openings"][j]["salt"])) if j < len(d["openings"]) else None,
    "salt short": lambda d, j: _op(d, j, salt=d["openings"][j]["salt"][2:]) if j < len(d["openings"]) else None,
    "path short": lambda d, j: _op(d, j, path=d["openings"][j]["path"][:-1]) if j < len(d["openings"]) else None,
    "path long": lambda d, j: _op(d, j, path=d["openings"][j]["path"] + ["00" * 64]) if j < len(d["openings"]) else None,
    "path flipped": lambda d, j: _op(d, j, path=[_flip(p) for p in d["openings"][j]["path"][:1]] + d["openings"][j]["path"][1:])
    if j < len(d["openings"]) and d["openings"][j]["path"] else None,
    "path no hex": lambda d, j: _op(d, j, path=["zz"]),
    "no path": lambda d, j: [o.pop("path", None) for o in d["openings"]],
    "root flipped": lambda d, j: _c(d, coin_root=_flip(d["commitment"]["coin_root"])),
    "root short": lambda d, j: _c(d, coin_root=d["commitment"]["coin_root"][2:]),
    "k+1": lambda d, j: _c(d, k=d["commitment"]["k"] + 1),
    "k-1": lambda d, j: _c(d, k=max(0, d["commitment"]["k"] - 1)),
    "k 2^32": lambda d, j: _c(d, k=1 << 32),
    "blocks 0": lambda d, j: _c(d, blocks=0),
    "blocks 2^64": lambda d, j: _c(d, blocks=1 << 64),
    "session flipped": lambda d, j: _c(d, session=_flip(d["commitment"]["session"])),
    "session short": lambda d, j: _c(d, session=d["commitment"]["session"][2:]),
    "session upper": lambda d, j: _c(d, session=d["commitment"]["session"].upper()),
    "key flipped": lambda d, j: d.update(coin_key=_flip(d["coin_key"])),
    "key short": lambda d, j: d.update(coin_key=d["coin_key"][2:]),
    "key hash flipped": lambda d, j: _c(d, coin_key_sha512=_flip(d["commitment"]["coin_key_sha512"])),
    "key top bit, its hash held": lambda d, j: (d.update(coin_key=d["coin_key"][:-2] + "ff"),
                                                _c(d, coin_key_sha512=hashlib.sha512(bytes.fromhex(d["coin_key"])).hexdigest())),
    "key a byte short, its hash held": lambda d, j: (d.update(coin_key=d["coin_key"][:-2]),
                                                     _c(d, coin_key_sha512=hashlib.sha512(bytes.fromhex(d["coin_key"])).hexdigest())),
    "no coin key": lambda d, j: d.pop("coin_key"),
    "session 33": lambda d, j: _c(d, session=d["commitment"]["session"] + "00"),
    "limit string": lambda d, j: _c(d, limit_ns="854000"),
    "limit big": lambda d, j: _c(d, limit_ns=10 ** 30),
    "limit 5 digits": lambda d, j: _c(d, limit_ns=85_400),
    "block_bytes bool": lambda d, j: _c(d, block_bytes=True),
    "blocks negative": lambda d, j: _c(d, blocks=-3),
    "binding odd": lambda d, j: _c(d, binding=d["commitment"]["binding"][1:]),
    "no commitment": lambda d, j: d.pop("commitment"),
    "commitment list": lambda d, j: d.update(commitment=[]),
    "no openings": lambda d, j: d.pop("openings"),
    "openings object": lambda d, j: d.update(openings={}),
    "openings dropped": lambda d, j: d.update(openings=[o for o in d["openings"] if o["j"] != j]),
    "wrong opening last": lambda d, j: d["openings"].append({**d["openings"][0], "j": j, "index": 1}) if d["openings"] else None,
    "wrong opening first": lambda d, j: d["openings"].insert(0, {**d["openings"][0], "j": j, "index": 1}) if d["openings"] else None,
    "key short id": lambda d, j: d.update(key="ab" * 15),
    "key 16": lambda d, j: d.update(key="ab" * 16),
    "key upper": lambda d, j: d.update(key=d["key"].upper()),
    "key number": lambda d, j: d.update(key=5),
    "no key": lambda d, j: d.pop("key"),
    "message short": lambda d, j: d.update(message=d["message"][2:]),
    "message upper": lambda d, j: d.update(message=d["message"].upper()),
    "message number": lambda d, j: d.update(message=5),
    "no message": lambda d, j: d.pop("message"),
    "message flipped": lambda d, j: d.update(message=_flip(d["message"])),
    "key unpinned": lambda d, j: d.update(key=_flip(d["key"], 40)),
    "signature flipped": lambda d, j: d.update(signature=_flip(d["signature"], 200)),
    "signature short": lambda d, j: d.update(signature=d["signature"][:-2]),
    "signature leaf": lambda d, j: d.update(signature="00000001" + d["signature"][8:]),
    "no signature": lambda d, j: d.pop("signature"),
    "ended late": lambda d, j: d.update(ended="late"),
    "ended number": lambda d, j: d.update(ended=1),
    "no ended": lambda d, j: d.pop("ended"),
    "rounds+1": lambda d, j: d.update(rounds=d["rounds"] + 1),
    "rounds-1": lambda d, j: d.update(rounds=d["rounds"] - 1),
    "rounds 2^32": lambda d, j: d.update(rounds=1 << 32),
    "rounds string": lambda d, j: d.update(rounds="3"),
    "coin carries a commitment": lambda d, j: d.update(carried=d["commitment"]),
    "place after the coin": lambda d, j: d.update(place=1),
    "place at the coin": lambda d, j: d.update(place=2),
    "place string": lambda d, j: d.update(place="0"),
    "no place": lambda d, j: d.pop("place"),
    "coin carries a root": lambda d, j: d.update(coin_root=_flip(d["commitment"]["coin_root"])),
}
#: the numbers of a text, changed to a form that is no canonical natural: the field and what its value becomes
NUMBER_FORMS = ("{n}.0", "{n}e0", "{n}E0", "0{n}", "+{n}", "-0", "{n}e+0", "{n}.5", "00")
COMMITMENT_NUMBERS = ("blocks", "k", "limit_ns", "block_bytes", "place")
COIN_NUMBERS = ("index", "j")
ENDED_NUMBERS = ("rounds",)


def _formed(t: str, key: str, form: str) -> str:
    return re.sub(rf'("{key}":)(\d+)', lambda m: m[1] + form.format(n=m[2]), t, count=1)


@st.composite
def sessions(draw) -> dict:
    """A session, honest or broken by one to two of `MUTATIONS`, one of the three functions marked at its length, a width or a
    length perhaps off, a text perhaps no JSON or no object, a number of it perhaps in a form that is no canonical natural."""
    k = draw(st.integers(1, 5))
    blocks = draw(st.sampled_from([1, 2, 5, 1 << 20, 1 << 40]))
    doc = copy.deepcopy(honest(k, blocks, draw(st.integers(0, 1)), draw(st.integers(0, 2)),
                               tuple(draw(st.integers(0, (1 << 40) - 1)) for _ in range(k))))
    rnd = draw(st.one_of(*[st.integers(0, k - 1)] * 5, st.integers(0, k + 1), st.sampled_from([1 << 32, (1 << 32) - 1])))
    for name in draw(st.one_of(st.just([]), st.just([]), st.lists(st.sampled_from(sorted(MUTATIONS)), min_size=1, max_size=2))):
        try:
            MUTATIONS[name](doc, rnd if rnd < k else 0)
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            note(f"{name} does not apply to this session: {e!r}")
    commit = CASES.text({x: doc[x] for x in ("commitment", "key", "place") if x in doc})
    coin = CASES.text({x: doc[x] for x in ("openings", "coin_key", "carried", "coin_root") if x in doc})
    ended = CASES.text({x: doc[x] for x in ("ended", "rounds") if x in doc})
    signed = CASES.text({x: doc[x] for x in ("message", "signature") if x in doc})
    if draw(st.integers(0, 6)) == 0:
        commit = _formed(commit, draw(st.sampled_from(COMMITMENT_NUMBERS)), draw(st.sampled_from(NUMBER_FORMS)))
    if draw(st.integers(0, 6)) == 0:
        coin = _formed(coin, draw(st.sampled_from(COIN_NUMBERS)), draw(st.sampled_from(NUMBER_FORMS)))
    if draw(st.integers(0, 9)) == 0:
        ended = _formed(ended, draw(st.sampled_from(ENDED_NUMBERS)), draw(st.sampled_from(NUMBER_FORMS)))
    if draw(st.integers(0, 24)) == 0:
        commit = draw(st.sampled_from(["{", "[]", "null", "", '{"commitment": 5}', "\xff"]))
    if draw(st.integers(0, 24)) == 0:
        coin = draw(st.sampled_from(["{", "[]", "null", "", "\xff"]))
    if draw(st.integers(0, 24)) == 0:
        ended = draw(st.sampled_from(["{", "[]", "null", "", "\xff"]))
    raw = lambda t: t.encode("latin-1") if t == "\xff" else t.encode()  # noqa: E731
    if draw(st.integers(0, 24)) == 0:
        signed = draw(st.sampled_from(["{", "[]", "null", "", "\xff"]))
    coins = {ROUND: raw(coin), HEADER: raw(ended), MESSAGE: raw(signed)}
    commitments = {f: raw(commit) for f in (ROUND, HEADER, CONTEXT, MESSAGE)}
    if draw(st.integers(0, 24)) == 0:
        coins.pop(draw(st.sampled_from([ROUND, HEADER, MESSAGE])))
    if draw(st.integers(0, 24)) == 0:
        commitments.pop(draw(st.sampled_from([ROUND, HEADER, CONTEXT, MESSAGE])))
    fn = draw(st.sampled_from([ROUND, ROUND, HEADER, HEADER, CONTEXT, MESSAGE]))
    tables = (coins, commitments)
    if fn == ROUND:
        placed = draw(st.sampled_from([{ROUND: 1}] * 8 + [{ROUND: 0}, {ROUND: 3}, {}, None]))
        return session_instance(fn, {"round": rnd}, tables, draw(st.sampled_from([96, 96, 96, 64])), placed)
    if fn == CONTEXT:
        return session_instance(fn, {}, tables, draw(st.sampled_from([848, 848, 848, 832])))
    if fn == MESSAGE:
        placed = draw(st.sampled_from([{MESSAGE: 1}] * 8 + [{MESSAGE: 0}, {MESSAGE: 3}, {}, None]))
        return session_instance(fn, {}, tables, draw(st.sampled_from([512, 512, 512, 496])), placed)
    head = max(1, _head_of(tables) + draw(st.sampled_from([0, 0, 0, 0, 0, 0, -1, 1])))
    placed = draw(st.sampled_from([{HEADER: 1}] * 8 + [{HEADER: 0}, {HEADER: 3}, {}, None]))
    return session_instance(fn, {"head": head}, tables, 8 * (head + head % 2) if draw(st.integers(0, 9)) else 8 * head, placed)


def _head_of(tables: tuple[dict, dict]) -> int:
    """The length the header has, which the Python's refusal of a header of length 0 names; 754 where it has none."""
    try:
        G.body(*tables, "x", {"supplied": HEADER, "args": {"head": 0}, "layout": "le", "bits": 8})
    except ValueError as e:
        m = re.search(r"header is (\d+) bytes, not 0$", str(e))
        return int(m[1]) if m else 754
    return 754


@st.composite
def releases_or_programs(draw):
    """Mostly a storage session; one in ten an input of the other supplied functions' cases."""
    return draw(INPUTS) if draw(st.integers(0, 9)) == 0 else draw(sessions())


SESSIONS = D.Port(PIECES, "supplied_marks", reference, releases_or_programs(), package=PACKAGE, of_record="lean")


@needs_lake
def test_the_storage_sessions_bodies_agree(pieces):
    """Lean's three bodies against the Python's on sessions the memory attester's own coin tree writes and on the same
    sessions broken one or two ways each body reads (the opening, the coin tree's root and key, the session, how it ended, the
    numbers in their forms, the key, the shape of the JSON), a few hundred seeded examples."""
    stats = D.run(SESSIONS, examples=400, server=pieces)
    assert stats.outputs > 100 and stats.rejects > 50, stats


VECTOR_CASES = [(f"round {j}", ROUND, {"round": j}, VECTORS["round_coin"][j], "halves", 32) for j in range(3)] + [
    ("the header", HEADER, {"head": VECTORS["head"]}, VECTORS["header"], "le", 16),
    ("the context", CONTEXT, {}, VECTORS["context"], "le", 16),
    ("the message", MESSAGE, {}, VECTORS["message"], "le", 16)]


@needs_lake
@pytest.mark.parametrize("name, fn, args, value, layout, w", VECTOR_CASES, ids=[c[0] for c in VECTOR_CASES])
def test_the_vectors_in_lean_and_python(pieces, name, fn, args, value, layout, w):
    """The values the memory attester's formats give for its released session, in both, as the leaves their layout reads."""
    j = {**session_instance(fn, args, CASES.held(), 4 * len(value)), "widths": {"x0": w}}
    j["program"] = desc([["x0", 1, {**mark_of(fn, args, args.get("head")), "layout": layout, "bits": 4 * len(value)}]], [2])
    want = {"x0": [SU.leaves(layout, bytes.fromhex(value), w)]}
    assert reference(j) == want
    assert pieces.call("supplied_marks", j) == ("output", D.canonical(want))


@needs_lake
@pytest.mark.parametrize("name, tables, fn, args, why", CASES.REFUSALS, ids=[r[0] for r in CASES.REFUSALS])
def test_what_has_no_statement_is_refused_by_the_same_text(pieces, name, tables, fn, args, why):
    """Lean refuses each refusal of the circuits' test, with its text, which is the Python's: `no statement`, naming the
    function and what the held commitment or coin lacks."""
    j = session_instance(fn, args, tables)
    with pytest.raises(D.Reject) as e:
        reference(j)
    kind, text = pieces.call("supplied_marks", j)
    assert kind != "output" and text == str(e.value) == CASES.NO_STATEMENT.replace("x", "x0") + why, text


def test_a_mark_whose_bits_are_not_the_values_is_refused():
    j = session_instance(ROUND, {"round": 0}, CASES.held(), 64)
    with pytest.raises(D.Reject, match="mark says 64"):
        reference(j)


#: the order rule's input: the program reads the root the record registers for ``w`` beside ``x0``
def _ordered(fn: str, args: dict, tables, released=None) -> dict:
    return {**session_instance(fn, args, tables, released=None), "program": desc(
        [["x0", 1, mark_of(fn, args, args.get("head"))], ["rw", 1, ROOTS[0]]], [2, 3]), "widths": {"x0": 16, "rw": 1},
        **({"released": released} if released is not None else {})}


@needs_lake
def test_a_released_commitment_is_not_a_coin_to_the_order_rule(pieces):
    """The order rule (every commitment registered before any coin) counts what the verifier holds as a coin and nothing it
    holds as a commitment: the context, held as a commitment only, needs no place beside the root the record registers; the
    header and the round coin, whose release is held as a coin, need one, whatever is held of them as a commitment."""
    tables = CASES.held()
    j = _ordered(CONTEXT, {}, tables)
    assert reference(j)["x0"] and pieces.call("supplied_marks", j)[0] == "output"
    for fn, args in ((HEADER, {"head": 753}), (ROUND, {"round": 0})):
        j = _ordered(fn, args, tables)
        with pytest.raises(D.Reject, match=f"names no place of {fn}'s coin"):
            reference(j)
        kind, text = pieces.call("supplied_marks", j)
        assert kind != "output" and f"names no place of {fn}'s coin" in text
        for placed in ({fn: 1}, {fn: 1, CONTEXT: 0}):
            j = _ordered(fn, args, tables, placed)
            assert reference(j)["x0"] and pieces.call("supplied_marks", j)[0] == "output"
        j = _ordered(fn, args, tables, {fn: 0})
        with pytest.raises(D.Reject, match="E_SUPPLIED_ORDER"):
            reference(j)
        assert "E_SUPPLIED_ORDER" in pieces.call("supplied_marks", j)[1]


def _programmed(marks, tables, released) -> dict:
    """The marks as inputs of one program, each its own: the verifier's tables and the places its coins are released at."""
    inputs = [[f"x{i}", 1, mark_of(fn, args, args.get("head"))] for i, (fn, args) in enumerate(marks)]
    coins, commitments = tables
    return {"program": desc(inputs, [2] * len(inputs)), "coins": {f: c.hex() for f, c in coins.items()},
            "commitments": {f: c.hex() for f, c in commitments.items()}, "attester_keys": PINNED,
            "widths": {f"x{i}": 16 for i in range(len(marks))},
            "record": RECORD, "log": [], "binding": BINDING, "released": released}


@needs_lake
@pytest.mark.parametrize("name, marks, tables, released, why", CASES.ORDER_REFUSALS, ids=[r[0] for r in CASES.ORDER_REFUSALS])
def test_a_commitment_must_precede_its_coin_in_lean_by_the_same_text(pieces, name, marks, tables, released, why):
    j = _programmed(marks, tables, released)
    with pytest.raises(D.Reject) as e:
        reference(j)
    kind, text = pieces.call("supplied_marks", j)
    assert kind != "output" and text == str(e.value) == why, text


@needs_lake
@pytest.mark.parametrize("name, marks, tables, released", CASES.ORDER_HOLDS, ids=[r[0] for r in CASES.ORDER_HOLDS])
def test_the_order_that_holds_is_accepted_by_both(pieces, name, marks, tables, released):
    j = _programmed(marks, tables, released)
    assert reference(j) and pieces.call("supplied_marks", j)[0] == "output"
