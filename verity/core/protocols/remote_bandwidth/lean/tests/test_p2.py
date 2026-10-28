"""`p2.py`, the remote audit over PoUS's P2 store, held to its Lean (`RemoteAudit.P2`, through `remote-audit-difftest`) on
drawn inputs that include what both sides must refuse: a registration's digest (`p2.Registration.digest`, with
`codec.P2Codec`'s refusals), a unit's sketch from its blocks' partials (`p2.Sketcher`: π a block, σ their sum, and y
from σ against y of the unit's bytes) and an opening's check (`p2.check`) on stores of drawn blocks under PoUS's leaf
and vk tree, honest or with one field changed; a whole session of record (`RemoteAudit.P2Session`, which `p2_party.py`
runs) against `p2_party.Books` on the bytes the verifier reads, the registration file, the request and each root and
openings line, recorded from a prover over a drawn store, honest or with one deviation; and the committed C*
(`tests/fixtures/p2-c-star.bin`, PoUS's setup's prover bytes): its registration's digest, an honest opening of each of
its units and an honest session over it."""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import ra_cases as M
from ra_cases import Entropy
import verity_difftest as D
from verity.core.protocols.pous.verifier import MerkleTree, leaf_hash
from verity.core.protocols.remote_bandwidth import audit, coins, p2, p2_party, receipts
from verity.core.protocols.remote_bandwidth.codec import P2_EXPANDER, P2_POINT, P2Codec
from verity.core.protocols.remote_bandwidth.receipts import Plan

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def port(fn, reference, inputs, **kw) -> D.Port:
    return D.Port(M.EXE, fn, reference, inputs, package=M.PACKAGE, of_record=M.OF_RECORD, rejects=M.REFUSALS, **kw)


server = D.server_fixture(port("p2_sketch", lambda v: v, st.nothing()))


def expand(seed: bytes, n: int) -> bytes:
    return b"".join(hashlib.sha512(seed + c.to_bytes(4, "big")).digest() for c in range(-(-n // 64)))[:n]


def fault(draw, n: int) -> int:
    """Which of `n` faults an input carries, 1 … `n`, or 0 for none, about half the time."""
    f = draw(st.integers(0, 2 * n))
    return f if f <= n else 0


def p2_blocks() -> st.SearchStrategy:
    """A 2,056-byte block: drawn bytes, or one of every byte's extremes as int8."""
    return st.one_of(st.binary(max_size=8).map(lambda s: expand(s, p2.BLOCK_BYTES)),
                     st.sampled_from([b"\x80", b"\x7f", b"\xff", b"\0", b"\x80\x7f"]).map(
                         lambda b: (b * p2.BLOCK_BYTES)[:p2.BLOCK_BYTES]))


# --- decoding, as `RemoteAudit.Difftest` decodes ----------------------------------------------------------------------

def registration_of(v) -> p2.Registration:
    c = M._get(v, "codec")
    codec = P2Codec(M.nat(c, "blocks_per_segment"), M.nat(c, "segments"), M.nat(c, "w"), M.nat(c, "c0"),
                    M._str(c, "expander"))
    return p2.Registration(M.hex_(v, "node"), codec, M.hex_(v, "commitment"), M.hex_(v, "salt"), M.hex_(v, "root"),
                           M.nat(v, "rows"))


def registration_json(r: p2.Registration) -> dict:
    return {"node": r.node.hex(), "codec": r.codec.to_json(), "commitment": r.commitment.hex(), "salt": r.salt.hex(),
            "root": r.root.hex(), "rows": r.rows}


def seed_of(v) -> bytes:
    s = M.hex_(v, "seed")
    if len(s) != receipts.DIGEST_BYTES:
        raise D.Reject("a pass seed is 64 bytes")
    return s


def opening_of(v) -> p2.Opening:
    paths = []
    for x in M._list(v, "paths"):
        if not isinstance(x, list):
            raise D.Reject("paths: a path is a list")
        paths.append(tuple(M._unhex(d, "paths") for d in x))
    return p2.Opening(M.int_(v, "pass"), M.int_(v, "position"), tuple(M.hexes(v, "blocks")), tuple(paths),
                      M.hex_(v, "leaf"), M.hex_(v, "leaf_salt"), tuple(M.hexes(v, "receipt_path")))


def opening_json(o: p2.Opening) -> dict:
    return {"pass": o.pass_index, "position": o.position, "blocks": [b.hex() for b in o.blocks],
            "paths": [[d.hex() for d in x] for x in o.paths], "leaf": o.leaf.hex(), "leaf_salt": o.leaf_salt.hex(),
            "receipt_path": [d.hex() for d in o.receipt_path]}


# --- a registration's digest ------------------------------------------------------------------------------------------

def registration_digest(v) -> str:
    with M.decoding():
        return registration_of(v).digest.hex()


@st.composite
def registration_inputs(draw) -> dict:
    bps, segments = draw(st.sampled_from([(4, 1), (3, 3), (1, 4), (1 << 16, 1038), (1 << 16, (1 << 32) - 1), (2, 2)]))
    v = {"node": expand(draw(st.binary(max_size=8)), draw(st.integers(1, 64))).hex(),
         "codec": {"blocks_per_segment": bps, "c0": P2_POINT[1], "expander": P2_EXPANDER, "segments": segments,
                   "w": P2_POINT[0]},
         "commitment": expand(draw(st.binary(max_size=8)), 64).hex(), "salt": expand(draw(st.binary(max_size=8)), 24).hex(),
         "root": expand(draw(st.binary(max_size=8)), 64).hex(), "rows": draw(st.sampled_from([16, 64]))}
    c = v["codec"]
    edit = fault(draw, 10)
    if edit == 1:
        v["node"] = draw(st.sampled_from(["", "00" * 65]))
    elif edit == 2:
        k, x = draw(st.sampled_from([("w", P2_POINT[0] - 8), ("w", 1024), ("c0", P2_POINT[1] + 2),
                                     ("expander", "chacha8-v2")]))
        c[k] = x
    elif edit == 3:
        c["blocks_per_segment"] = draw(st.sampled_from([0, (1 << 16) + 1]))
    elif edit == 4:
        c["segments"] = draw(st.sampled_from([0, 1 << 32]))
    elif edit == 5:
        c["blocks_per_segment"], c["segments"] = draw(st.sampled_from([(1, 3), (3, 1), (1, 1)]))
    elif edit == 6:
        v["commitment"] = draw(st.sampled_from(["", "00" * 63]))
    elif edit == 7:
        v["salt"] = draw(st.sampled_from(["00" * 23, "00" * 32]))
    elif edit == 8:
        v["root"] = "00" * 63
    elif edit == 9:
        v["rows"] = draw(st.sampled_from([0, 32]))
    elif edit == 10:
        target = draw(st.sampled_from([v, c]))
        del target[draw(st.sampled_from(sorted(target)))]
    return v


def test_p2_registration_digest(server) -> None:
    stats = D.run(port("p2_registration_digest", registration_digest, registration_inputs()), examples=200,
                  server=server)
    assert stats.outputs > 80, stats


# --- a registration's node ----------------------------------------------------------------------------------------------

HOSTS = (b"p2-node", b"node-b", b"\x07", "é".encode())


def register(v) -> str:
    """`p2.register`'s check of its node (`P2.Registration.register`): the digest of the registration
    `registration_of` reads, for a node the Site of `site` registers as a host (`audit.host_of`)."""
    with M.decoding():
        hosts = M.site_of(M._get(v, "site"))
        r = registration_of(v)
    audit.host_of(hosts, r.node)
    return r.digest.hex()


@st.composite
def register_inputs(draw) -> dict:
    """`registration_inputs` under a Site of `HOSTS`, its node most often one of them, else drawn bytes, the site's id
    or bytes that aren't UTF-8; now and then the Site leaves hosts out."""
    v = draw(registration_inputs())
    if v.get("node") not in ("", "00" * 65) and draw(st.integers(0, 7)):
        v["node"] = draw(st.sampled_from(HOSTS * 4 + (b"site0", b"\xff", b"\xc0\xae"))).hex()
    v["site"] = M.site_json(*(HOSTS[:draw(st.integers(0, len(HOSTS)))] if draw(st.integers(0, 5)) == 0 else HOSTS))
    return v


def test_p2_register(server) -> None:
    stats = D.run(port("p2_register", register, register_inputs()), examples=200, server=server)
    assert stats.outputs > 40 and stats.rejects > 40, stats


# --- a unit's sketch ---------------------------------------------------------------------------------------------------

def sketch(v) -> dict:
    """The decode's side, π of each block and σ their sum, and the verifier's, y of the unit's bytes: the Lean's y is
    from σ, so the two sides agree only if σ is the unit's sketch."""
    with M.decoding():
        seed, rows, g, blocks = seed_of(v), M.nat(v, "rows"), M.nat(v, "g"), M.hexes(v, "blocks")
        sk = p2.Sketcher(seed, rows)
        if len(blocks) != p2.UNIT_BLOCKS or any(len(b) != p2.BLOCK_BYTES for b in blocks):
            raise D.Reject("a unit is four 2,056-byte blocks")
        receipts._index(g)
    partials = [sk.block(p2.UNIT_BLOCKS * g + b, x) for b, x in enumerate(blocks)]
    return {"partials": [x.hex() for x in partials], "sigma": sk.total(partials).hex(),
            "receipt": sk(g, b"".join(blocks)).hex()}


@st.composite
def sketch_inputs(draw) -> dict:
    v = {"seed": expand(draw(st.binary(max_size=8)), 64).hex(), "rows": draw(st.sampled_from([16, 16, 64])),
         "g": draw(st.one_of(st.integers(0, 40), st.integers(0, 2 ** 64 - 1))),
         "blocks": [draw(p2_blocks()).hex() for _ in range(p2.UNIT_BLOCKS)]}
    edit = fault(draw, 5)
    if edit == 1:
        v["seed"] = draw(st.sampled_from(["", "00" * 63, "00" * 65]))
    elif edit == 2:
        v["rows"] = draw(st.sampled_from([0, 32]))
    elif edit == 3:
        v["g"] = 2 ** 64
    elif edit == 4:
        bs = v["blocks"]
        v["blocks"] = draw(st.sampled_from([bs[:3], bs + bs[:1], bs[:3] + [bs[3][:-2]], bs[:3] + [bs[3] + "00"]]))
    elif edit == 5:
        del v[draw(st.sampled_from(sorted(v)))]
    return v


def test_p2_sketch(server) -> None:
    stats = D.run(port("p2_sketch", sketch, sketch_inputs()), examples=120, server=server)
    assert stats.outputs > 50, stats


# --- an opening's check -----------------------------------------------------------------------------------------------

def opening(v) -> str | None:
    with M.decoding():
        reg = registration_of(M._get(v, "registration"))
        seed = seed_of(v)
        plan = Plan(M.runs_of(M._get(v, "runs"), "runs"), reg.blocks)
        p = M.nat(v, "p")
        if p >= plan.length:
            raise D.Reject("a position past the plan")
        o = opening_of(M._get(v, "opening"))
        root, i = M.hex_(v, "root"), M.nat(v, "i")
        if len(root) != receipts.DIGEST_BYTES:
            raise D.Reject("a pass's root is 64 bytes")
    return p2.check(reg, seed, plan, root, i, p, o)


#: Stores' geometry (blocks_per_segment, segments): a unit, units with a tail of one, two and three blocks, and
#: seventeen units, so that a plan has a second receipt leaf.
GEOMETRIES = ((4, 1), (2, 3), (3, 3), (5, 3), (17, 4))


@functools.cache
def drawn_store(k: int, rows: int) -> p2.Store:
    """A store of drawn blocks (P2's check never decodes, so they needn't be codewords) under geometry `k`, with
    every byte's extremes among them."""
    bps, segments = GEOMETRIES[k]
    blocks = [expand(b"block %d %d" % (k, j), p2.BLOCK_BYTES) for j in range(bps * segments)]
    blocks[1] = b"\x80" * p2.BLOCK_BYTES
    blocks[-1] = b"\x7f" * p2.BLOCK_BYTES
    codec, salt = P2Codec(bps, segments), expand(b"salt %d" % k, 24)
    root = MerkleTree([leaf_hash(p2.pous_params(codec), salt, j, b) for j, b in enumerate(blocks)]).root
    reg = p2.Registration(b"node %d" % k, codec, expand(b"commitment %d" % k, 64), salt, root, rows)
    return p2.Store(reg, blocks)


def canonical_runs(order: list[int]) -> list[list[int]]:
    """`order` as a plan's runs, consecutive units merged."""
    runs: list[list[int]] = []
    for g in order:
        if runs and runs[-1][0] + runs[-1][1] == g:
            runs[-1][1] += 1
        else:
            runs.append([g, 1])
    return runs


def flip(h: str, k: int = 0) -> str:
    b = bytearray(bytes.fromhex(h))
    b[k % len(b)] ^= 1
    return b.hex()


@st.composite
def opening_inputs(draw) -> dict:
    """An opening of a drawn pass over a drawn store, honest, from a prover that committed a wrong receipt, or with one
    field of the opening, the pass or the registration changed."""
    k, rows = draw(st.integers(0, len(GEOMETRIES) - 1)), draw(st.sampled_from([16, 16, 64]))
    store = drawn_store(k, rows)
    reg = store.registration
    order = draw(st.permutations(range(reg.blocks)))[:draw(st.integers(1, reg.blocks))]
    plan = Plan([tuple(r) for r in canonical_runs(order)], reg.blocks)
    seed, i, p = expand(draw(st.binary(max_size=8)), 64), draw(st.integers(0, 3)), draw(st.integers(0, plan.length - 1))
    prover = p2.Prover(store, entropy=Entropy(b"salts %d" % k))
    edit = fault(draw, 16)
    if edit == 1:
        sk, unit = p2.Sketcher(seed, rows), store.unit(plan.block(p))
        other = bytes([unit[0] ^ 1]) + unit[1:]
        ys = [sk(g, other if q == p else store.unit(g)) for q, g in enumerate(plan)]
        ps = prover.commit(0, i, seed, plan, ys)
    else:
        ps = prover.pass_under(0, i, seed, plan)
    v = {"registration": registration_json(reg), "seed": seed.hex(), "runs": canonical_runs(order),
         "root": ps.tree.commitment.root.hex(), "i": i, "p": p, "opening": opening_json(prover.open(ps, p))}
    o, r = v["opening"], v["registration"]
    b = draw(st.integers(0, p2.UNIT_BLOCKS - 1))
    if edit == 2:
        o[draw(st.sampled_from(["pass", "position"]))] += draw(st.sampled_from([1, -1]))
    elif edit == 3:
        c = draw(st.integers(0, p2.UNIT_BLOCKS - 1))
        o["blocks"][b], o["blocks"][c] = o["blocks"][c], o["blocks"][b]
    elif edit == 4:
        o["blocks"][b] = flip(o["blocks"][b], draw(st.integers(0, p2.BLOCK_BYTES - 1)))
    elif edit == 5:
        x = o["paths"][b]
        o["paths"][b] = draw(st.sampled_from(
            [x[:-1], x + ["00" * 64], [flip(x[0])] + x[1:] if x else ["00" * 64], ["00" * 63] + x[1:]]))
    elif edit == 6:
        o[draw(st.sampled_from(["blocks", "paths"]))].pop()
    elif edit == 7:
        o["blocks"][b] = o["blocks"][b][:draw(st.sampled_from([0, 2 * p2.BLOCK_BYTES - 2]))]
    elif edit == 8:
        o["leaf"] = draw(st.sampled_from([flip(o["leaf"], 64 * (p % receipts.PER_LEAF)), o["leaf"][:-128],
                                          o["leaf"] + "00" * 64]))
    elif edit == 9:
        o["leaf_salt"] = draw(st.sampled_from([flip(o["leaf_salt"]), o["leaf_salt"][:-2]]))
    elif edit == 10:
        o["receipt_path"] = draw(st.sampled_from(
            [[flip(o["receipt_path"][0])] + o["receipt_path"][1:] if o["receipt_path"] else ["00" * 64],
             o["receipt_path"] + ["00" * 64]]))
    elif edit == 11:
        v[draw(st.sampled_from(["root", "seed"]))] = draw(st.sampled_from([flip(v["root"]), "00" * 63]))
    elif edit == 12:
        key = draw(st.sampled_from(["salt", "root"]))
        r[key] = flip(r[key])
    elif edit == 13:
        r["rows"] = 80 - rows
    elif edit == 14:
        v["runs"] = draw(st.sampled_from([[[0, reg.blocks]], [[reg.blocks, 1]], []]))
    elif edit == 15:
        r["node"] = flip(r["node"])
    elif edit == 16:
        target = draw(st.sampled_from([v, o, r]))
        del target[draw(st.sampled_from(sorted(target)))]
    return v


def test_p2_opening(server) -> None:
    stats = D.run(port("p2_opening", opening, opening_inputs()), examples=400, server=server)
    assert stats.outputs > 300, stats


# --- a session of record ----------------------------------------------------------------------------------------------

def p2_event_of(v) -> tuple:
    """`p2SessionFn`'s events: `start` and `boundary` `{"now"}`, `close` `{"t", "now"}`, `root` `{"t", "i", "now",
    "line"}`, `openings` `{"t", "line"}` (each line its bytes in hex) and `fail`."""
    kind = M._str(v, "kind")
    if kind in ("start", "boundary"):
        return kind, M.nat(v, "now")
    if kind == "close":
        return kind, M.int_(v, "t"), M.nat(v, "now")
    if kind == "root":
        return kind, M.int_(v, "t"), M.int_(v, "i"), M.hex_(v, "line"), M.nat(v, "now")
    if kind == "openings":
        return kind, M.int_(v, "t"), M.hex_(v, "line")
    if kind == "fail":
        return (kind,)
    raise D.Reject(f"no P2 event kind {kind}")


def p2_apply(b: p2_party.Books, e: tuple) -> dict:
    kind, *a = e
    return {"start": b.start, "boundary": b.boundary, "close": b.close, "root": b.root, "openings": b.openings,
            "fail": b.fail}[kind](*a)


def p2_session(v) -> dict:
    """`p2SessionFn`: the registration file's bytes and the prover's request line read as `p2_party` reads them
    (`registration_of`, `plan_of`, `message`, `params_of` with the request's own values as the auditor's, as the
    Lean's `paramsOf` reads it), and the events through `p2_party.Books` on a challenger replaying the case's 2W
    coins: the held store, every event's result and the answer, less its reason."""
    with M.decoding():
        data = M.hex_(v, "registration")
    try:
        doc = json.loads(data.decode())
        reg = p2_party.registration_of(doc)
        plan = p2_party.plan_of(doc, reg)
    except (ValueError, KeyError, TypeError, RecursionError):
        return {"refused": "registration"}
    with M.decoding():
        line = M.hex_(v, "request")
    try:
        params = lean_params(reg, line)
    except (ConnectionError, ValueError, KeyError, TypeError):
        return {"refused": "request"}
    with M.decoding():
        committed = M.hexes(v, "coins")
        if len(committed) != 2 * params.windows or any(len(c) != coins.COIN_BYTES for c in committed):
            raise D.Reject("2W coins of 32 bytes")
        M.nat(v, "chunk")
        not_before = M.nat(v, "not_before_ns")
        events = [p2_event_of(e) for e in M._list(v, "events")]
    b = p2_party.Books(params, reg, M.replayed(reg, 0, params.windows, committed, []), plan, not_before)
    results = [p2_apply(b, e) for e in events]
    answer = json.loads(json.dumps(p2_party.answer(b.s, plan=plan)))
    del answer["reason"]
    return {"held": p2_party.held_json(reg, plan), "results": results, "answer": answer}


P2_STEP_NS = 400


def lean_params(reg: p2.Registration, line: bytes):
    """The parameters of the request `line` as the Lean's `P2Session.paramsOf` reads them: the request's own values,
    which `p2_party.params_of` takes as the auditor's (the party holds them to its own, `p2_party.SESSION`)."""
    request = p2_party.message(line, "request")
    return p2_party.params_of(reg, request, request)


def p2_line(key: str, value) -> bytes:
    return json.dumps({key: value}, sort_keys=True).encode() + b"\n"


def p2_bad_root(root: str, pick: int) -> bytes:
    """A root's line near its form, from `pick`: unended, not JSON, another key or two, a root not 64 bytes of
    lowercase hex, not an object or not UTF-8, or ended CRLF, which is in form."""
    lines = (p2_line("root", root)[:-1], b"not json\n", b"\n", p2_line("roots", root),
             json.dumps({"root": root, "t": 0}).encode() + b"\n", p2_line("root", root.upper()),
             p2_line("root", root[:-2]), p2_line("root", root + "00"), p2_line("root", 7), p2_line("root", None),
             b"[" + p2_line("root", root)[:-1] + b"]\n", b'{"root": "\xff"}\n', p2_line("root", root)[:-1] + b"\r\n")
    return lines[pick % len(lines)]


def p2_bad_openings(os: list[dict], pick: int) -> bytes:
    """A window's openings line near its form, from `pick`: unended, another key, an opening fewer, more or out of
    order, not a list, or one opening changed: each of `p2.check`'s failures and `opening_of`'s refusals, or a key
    nothing reads, which is in form."""
    k, n = pick % len(os), pick // len(os)
    o = os[k]
    edits = (lambda: o | {"leaf_salt": o["leaf_salt"][:-2]}, lambda: o | {"pass": str(o["pass"])},
             lambda: o | {"blocks": [flip(o["blocks"][0])] + o["blocks"][1:]}, lambda: o | {"leaf": flip(o["leaf"])},
             lambda: o | {"position": o["position"] + 1}, lambda: {x: y for x, y in o.items() if x != "paths"},
             lambda: o | {"blocks": o["blocks"][:3]}, lambda: o | {"unread": 1})
    lines = (lambda: p2_line("openings", os)[:-1], lambda: p2_line("opening", os),
             lambda: p2_line("openings", os[:-1]), lambda: p2_line("openings", os + os[:1]),
             lambda: p2_line("openings", os[::-1]), lambda: p2_line("openings", {"0": os}),
             *(lambda e=e: p2_line("openings", os[:k] + [e()] + os[k + 1:]) for e in edits))
    return lines[n % len(lines)]()


def p2_events(store: p2.Store, plan: Plan, params, committed: list[bytes], not_before: int, edit: int = 0,
              at: tuple[int, int] = (0, 0), pick: int = 0) -> list[dict]:
    """A session's events as `p2_party.session` drives it, the prover's lines an honest prover's over `store`
    (`p2.Prover`), each step `P2_STEP_NS`, but for deviation `edit` at pass `at` = (t, i), its variant from `pick`:
    3 a unit's reads skipped (its receipt from zeros), 4 window t's openings line near its form (`p2_bad_openings`), 5
    window t's last pass missing, 6 the pass's root line near its form (`p2_bad_root`) and 7 the pass's root after
    window t's deadline. The events end once the session has its outcome."""
    b = p2_party.Books(params, store.registration, M.replayed(store.registration, 0, params.windows, committed, []),
                       plan, not_before)
    prover = p2.Prover(store, entropy=Entropy(b"p2 session"))
    events: list[dict] = []
    passes: dict[tuple[int, int], object] = {}

    def push(e: dict) -> dict:
        """`e` sent and the books' answer, until the session has its outcome; after it, nothing is sent."""
        if b.s.outcome is not None:
            return {}
        events.append(e)
        return p2_apply(b, p2_event_of(e))

    def opened(t: int, now: int) -> None:
        got = push({"kind": "close", "t": t, "now": now})
        if "draws" in got:
            os = [p2_party.opening_json(prover.open(passes[t, i], p)) for i, p in got["draws"]]
            line = p2_bad_openings(os, pick) if edit == 4 and t == at[0] else p2_line("openings", os)
            push({"kind": "openings", "t": t, "line": line.hex()})

    now, previous = not_before, store.registration.digest
    push({"kind": "start", "now": now})
    for t in range(params.windows):
        if t >= len(b.s.windows) or b.s.outcome is not None:
            break
        w = b.s.windows[t]
        for i in range(params.min_passes):
            if edit == 5 and (t, i) == (at[0], params.min_passes - 1):
                continue
            seed = receipts.seed(w.challenge, i, previous)
            if edit == 3 and (t, i) == at:
                sk = p2.Sketcher(seed, store.registration.rows)
                ys = [sk(g, bytes(p2.UNIT_BYTES) if p == pick % plan.length else store.unit(g))
                      for p, g in enumerate(plan)]
                ps = prover.commit(t, i, seed, plan, ys)
            else:
                ps = prover.pass_under(t, i, seed, plan)
            passes[t, i], previous = ps, ps.root
            now += P2_STEP_NS
            if edit == 7 and (t, i) == at:
                push({"kind": "boundary", "now": max(now, w.released_ns + params.window_ns)})
                now = (b.s.deadline(t) or now) + 1
            root = ps.root.hex()
            line = p2_bad_root(root, pick) if edit == 6 and (t, i) == at else p2_line("root", root)
            push({"kind": "root", "t": t, "i": i, "now": now, "line": line.hex()})
            if i == 0 and t > 0:
                opened(t - 1, now)
        now = max(now, w.released_ns + params.window_ns)
        push({"kind": "boundary", "now": now})
    if b.s.outcome is None:
        opened(params.windows - 1, b.s.deadline(params.windows - 1) + 1)
    return events


def p2_held(reg: p2.Registration, data: bytes, line: bytes):
    """The plan and parameters the books hold from the registration file `data` and the request `line`, else the
    error they refuse them with (a session with no events after the request)."""
    try:
        doc = json.loads(data)
        plan = p2_party.plan_of(doc, p2_party.registration_of(doc))
        return plan, lean_params(reg, line)
    except (ConnectionError, ValueError, KeyError, TypeError) as e:
        return e


@st.composite
def p2_session_inputs(draw) -> dict:
    """A session over a drawn store under every unit or a drawn plan, recorded from the prover (`p2_events`), honest
    or with one deviation: the registration or the request refused or near its form (1, 2), one of `p2_events`'s (3
    … 7), or its events changed after: a fail inserted, the events cut short, one repeated, dropped or swapped with
    the next, one's clock changed, a field missing or out of form, or a start too early first (8 … 15)."""
    k, rows = draw(st.integers(0, len(GEOMETRIES) - 1)), draw(st.sampled_from([16, 64]))
    store = drawn_store(k, rows)
    reg = store.registration
    stated = None
    if draw(st.booleans()):
        order = draw(st.permutations(range(reg.blocks)))[:draw(st.integers(1, reg.blocks))]
        stated = Plan([tuple(r) for r in canonical_runs(order)], reg.blocks)
    doc = p2_party.registration_json(reg, stated)
    windows, passes, samples = draw(st.integers(1, 3)), draw(st.integers(1, 2)), draw(st.integers(1, 3))
    window_ns, deadline_ns = draw(st.sampled_from([(2 * P2_STEP_NS, P2_STEP_NS), (P2_STEP_NS, 3 * P2_STEP_NS)]))
    req = p2_party.request_of(reg.node, windows=windows, passes=passes, samples=samples, window_ns=window_ns,
                              deadline_ns=deadline_ns)
    committed = [expand(b"coin %d " % j + draw(st.binary(max_size=4)), coins.COIN_BYTES) for j in range(2 * windows)]
    not_before = draw(st.sampled_from([0, 0, 10 ** 6]))
    edit, pick = fault(draw, 15), draw(st.integers(0, 1 << 16))
    at = (draw(st.integers(0, windows - 1)), draw(st.integers(0, passes - 1)))
    data, line = None, None
    if edit == 1:
        f = pick % 8
        if f == 0:
            doc["digest"] = flip(doc["digest"])
        elif f == 1:
            del doc[sorted(doc)[pick % len(doc)]]
        elif f == 2:
            doc["plan"] = [[[0, reg.blocks + 1]], [], [[0, 1], [1, 1]], "0,1", [[0]], [[0, True]]][pick % 6]
        elif f == 3:
            doc["rows"] = 80 - rows
        elif f == 4:
            doc["plan"] = None
        elif f == 5:
            data = b'{"node": "\xff"}'
        elif f == 6:
            data = b"[]"
        else:
            doc["w"] = float(doc["w"])
    elif edit == 2:
        f = pick % 9
        key = sorted(req)[pick % len(req)]
        if f == 0:
            req["node"] = b"another node".hex()
        elif f == 1:
            req["unread"] = 1
        elif f == 2 and key != "node":
            req[key] = 0
        elif f == 3:
            req[key] = True
        elif f == 4 and key != "node":
            req[key] = float(req[key])
        elif f == 5:
            del req[key]
        elif f == 6:
            line = p2_line("request", req)[:-1]
        elif f == 7:
            line = json.dumps({"request": req, "t": 0}).encode() + b"\n"
        else:
            req["node"] = req["node"].upper()
    data = json.dumps(doc).encode() if data is None else data
    line = p2_line("request", req) if line is None else line
    v = {"registration": data.hex(), "request": line.hex(), "coins": [c.hex() for c in committed],
         "chunk": draw(st.sampled_from([1, 2, 3, 64])), "not_before_ns": not_before, "events": []}
    held = p2_held(reg, data, line)
    if isinstance(held, Exception):
        return v
    plan, params = held
    ev = v["events"] = p2_events(store, plan, params, committed, not_before, edit, at, pick)
    j = pick % (len(ev) + 1)
    if edit == 8:
        ev.insert(j, {"kind": "fail"})
    elif edit == 9:
        del ev[j:]
    elif edit == 10:
        ev.insert(j, dict(ev[min(j, len(ev) - 1)]))
    elif edit == 11:
        del ev[min(j, len(ev) - 1)]
    elif edit == 12 and len(ev) > 1:
        j = pick % (len(ev) - 1)
        ev[j], ev[j + 1] = ev[j + 1], ev[j]
    elif edit == 13 and "now" in ev[min(j, len(ev) - 1)]:
        e = ev[min(j, len(ev) - 1)]
        e["now"] = [0, e["now"] + 10 ** 6, max(0, e["now"] - P2_STEP_NS)][pick % 3]
    elif edit == 14:
        f = pick % 5
        if f == 0:
            v["coins"] = v["coins"][:-1]
        elif f == 1:
            del v[sorted(v)[pick % len(v)]]
        elif f == 2:
            ev.append({"kind": "append", "now": 0})
        elif f == 3:
            ev.append({"kind": "root", "t": 0, "i": 0, "now": 0, "line": "0"})
        else:
            v["coins"][0] = v["coins"][0][:-2]
    elif edit == 15:
        ev.insert(0, {"kind": "start", "now": max(0, not_before - 1)})
    return v


def test_p2_session(server) -> None:
    stats = D.run(port("p2_session", p2_session, p2_session_inputs()), examples=150, server=server)
    assert stats.outputs > 110, stats


# --- the committed C* ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def committed() -> p2.Store:
    """The committed C* (PoUS's setup's prover bytes, `tests/test_p2.py`'s), registered under its vk root."""
    meta = json.loads((FIXTURES / "p2-store.json").read_text())
    data = (FIXTURES / "p2-c-star.bin").read_bytes()
    assert hashlib.sha512(data).hexdigest() == meta["c_star_sha512"]
    m = meta["params"]
    reg = p2.Registration(b"p2-node-under-test", P2Codec(m["blocks_per_segment"], m["segments"]),
                          bytes.fromhex(meta["commitment"]), bytes.fromhex(meta["salt"]), bytes.fromhex(meta["root"]))
    return p2.Store(reg, [data[p2.BLOCK_BYTES * j:p2.BLOCK_BYTES * (j + 1)] for j in range(len(data) // p2.BLOCK_BYTES)])


def test_the_committed_store_s_registration_digest_is_the_leans(server, committed) -> None:
    reg = committed.registration
    assert D.replay(port("p2_registration_digest", registration_digest, st.nothing()), registration_json(reg),
                    server) == ("output", f'"{reg.digest.hex()}"')


def test_the_committed_store_registers_only_for_a_host_the_site_registers(server, committed) -> None:
    """Its node a host of the Site: registered, with its digest; under a Site that doesn't register it, or naming the
    site, refused by both."""
    reg = committed.registration
    v = registration_json(reg) | {"site": M.site_json(reg.node)}
    assert D.replay(port("p2_register", register, st.nothing()), v, server) == ("output", f'"{reg.digest.hex()}"')
    for bad in ({"site": M.site_json()}, {"site": M.site_json(b"node-b")}, {"node": b"site0".hex()}):
        assert D.replay(port("p2_register", register, st.nothing()), v | bad, server)[0] == "reject"


def test_an_honest_opening_of_each_committed_unit_checks_in_lean(server, committed) -> None:
    reg = committed.registration
    plan, seed = Plan.whole(reg.blocks), hashlib.sha512(b"a pass seed").digest()
    prover = p2.Prover(committed, entropy=Entropy(b"committed"))
    ps = prover.pass_under(0, 0, seed, plan)
    for p in range(plan.length):
        v = {"registration": registration_json(reg), "seed": seed.hex(), "runs": [list(r) for r in plan.runs],
             "root": ps.tree.commitment.root.hex(), "i": 0, "p": p, "opening": opening_json(prover.open(ps, p))}
        assert opening(v) is None
        assert D.replay(port("p2_opening", opening, st.nothing()), v, server) == ("output", "null")


def test_an_honest_session_over_the_committed_store_is_accepted_in_lean(server, committed) -> None:
    """Every unit a pass, two windows of two passes, three of each window's four receipts opened, checked two at a
    time: the Lean's answer is the books', an acceptance crediting every receipt the pooled bound leaves."""
    reg = committed.registration
    req = p2_party.request_of(reg.node, windows=2, passes=2, samples=3, window_ns=2 * P2_STEP_NS,
                              deadline_ns=P2_STEP_NS)
    coins_ = [expand(b"committed coin %d" % j, coins.COIN_BYTES) for j in range(4)]
    events = p2_events(committed, Plan.whole(reg.blocks), p2_party.params_of(reg, req, req), coins_, 0)
    v = {"registration": json.dumps(p2_party.registration_json(reg)).encode().hex(),
         "request": p2_line("request", req).hex(), "coins": [c.hex() for c in coins_], "chunk": 2, "not_before_ns": 0,
         "events": events}
    kind, out = D.replay(port("p2_session", p2_session, st.nothing()), v, server)
    answer = json.loads(out)["answer"]
    assert kind == "output" and answer["verdict"] == "accept", answer
    assert reg.blocks == 2 and answer["receipts"] == 2 * 2 * reg.blocks and answer["draws"] == [3, 3]
