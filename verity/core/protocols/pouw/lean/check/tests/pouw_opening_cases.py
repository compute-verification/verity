"""The opening difftest's cases: generated openings of one unit, each ``{"params", "commitment": hex, "units", "unit",
"pub": hex, "opening", "salt": hex or null}``, the arguments of `core_check.opening` (`params` and the registration's
bytes as `answer` reads them, `units` null for a unit opened undrawn, `salt` a certified window's).

A case is an honest opening (a window of P entries, a call's and the filler's each with small side trees, the unit one
of their tiles, its proof's public file a header (over the template's pinned circuit, `core_check.TILES`) and the two
rows' b ‖ c, the registration's entries root over the
entries' leaves), then one mutation (`MUTATIONS`) aimed at one refusal of `core_check.opening`, or at a reading both
sides must share (hex as `bytes.fromhex` reads it, the header's schemas by port name, in any written order).  A header
whose keys sort otherwise than its circuit's ports (`schemas-between`, `schemas-shadowed`, `salted-header-order`) is
refused, the rows never read in key order.  A salted case (`SALTED`) is
the same over a certified window (`CERTIFIED`) and a salted template (`core_check.Salted`): each entry's side trees hold R
A rows per tile row and C B rows per tile column, the opening lists the unit's R and C rows, and the proof's body holds
their b ‖ c and then its public row's, `core_check.public_bc` of the window's context, its salt, the unit and the entry's
roots.  The windows come from a few
fixed contexts, so the side trees' pad nodes, which only the generator computes, are computed once per side domain.
`tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_core_check.py`
holds each mutation to the refusal it aims at.
"""
from __future__ import annotations

import json
import random

import pouw_check_cases as cases
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import row_leaf

#: (context, program, partition, entries): the windows the generator opens
WINDOWS = [(f"served-zk/opening-{i}/{random.Random(i).randbytes(8).hex()}", random.Random(10 + i).randbytes(64).hex(),
            random.Random(20 + i).randbytes(64).hex(), n) for i, n in enumerate((4, 5))]
#: certified windows: each context names its PoUW window (`salt.context`)
CERTIFIED = [(PS.context(f"served-zk/opening-c{i}", i, random.Random(30 + i).randbytes(64).hex()),
              random.Random(40 + i).randbytes(64).hex(), random.Random(50 + i).randbytes(64).hex(), n)
             for i, n in enumerate((4, 5))]


def _leaf(dom, i: int, bc: bytes, schema: str) -> bytes:
    return row_leaf(dom, i, Wd._HM.tree_leaf(Wd._HM.default_key, bc), schema)


def _side(dom, schema: str, bcs: list[bytes]) -> tuple[bytes, list]:
    levels = Wd.side_levels(dom, [_leaf(dom, i, bc, schema) for i, bc in enumerate(bcs)])
    return levels[-1][0], levels


def _header(schemas: dict, circuit: str) -> dict:
    return {"circuit_sha512": circuit, "format": "flock-circuit-inputs", "instances": 1,
            "frame_v3": {"frame": "frame-v3-sha512", "owner": -1, "schemas": schemas}}


def honest(rng: random.Random, salted: bool = False, context: str | None = None, salt: bytes | None = None) -> dict:
    """A window from `WINDOWS`, its entries' tree and side trees, and one unit's opening and proof's public file; with
    `salted`, a certified window from `CERTIFIED` of a salted template, and the window's salt (`context` and `salt`, when
    given, in place of the window's and a random one)."""
    ctx, program, partition, n = rng.choice(CERTIFIED if salted else WINDOWS)
    context = ctx if context is None else context
    base = cases.honest(rng)
    params = {**base["params"], "context": context, "check": {**base["params"]["check"], "entries": n}}
    if (CC.TILES[params["check"]["template"]].salted is not None) != salted:
        template = rng.choice([t for t in sorted(CC.TILES) if (CC.TILES[t].salted is not None) == salted])
        params["check"].update(template=template, tile_work=CC.TILES[template].tile_work)
        params["query"] = CC.TILES[template].query
    lay = CC.TILES[params["check"]["template"]].salted
    per = {"a": 1, "b": 1} if lay is None else {"a": lay.r, "b": lay.c}
    schema_of = dict(CC.OPENED) if lay is None else {"a": CC.ROW_V2, "b": CC.ROW_V2}
    rec = base["record"]
    rec.update(program=program, partition=partition)
    rec["window"]["context"] = context
    dom = CC.window_domain(CC.params(params), rec)
    seg, v2 = (s for _, s in CC.OPENED)
    entries, sides, lo = [], [], rng.randrange(1 << 20)
    for kind in ("call", "filler"):
        n_a, n_b = (rng.randint(1, 3), rng.randint(1, 4)) if lay is None else (rng.randint(1, 2), rng.randint(1, 2))
        bcs = {"a": [rng.randbytes(CC.BC_BYTES) for _ in range(per["a"] * n_a)],
               "b": [rng.randbytes(CC.BC_BYTES) for _ in range(per["b"] * n_b)]}
        trees = {s: _side(Wd.side_domain(dom.domain_id, s), schema_of[s], bcs[s]) for s, _ in CC.OPENED}
        zero = bytes(64)
        call = kind == "call"
        entries.append(Wd.Entry(kind, lo, lo + n_a * n_b, n_a, n_b, trees["a"][0], trees["b"][0],
                                rng.randbytes(64) if call else zero, rng.randbytes(64) if call else zero,
                                rng.randrange(1 << 20) if call else 0, 1 if call else 0, 16 if call else 0))
        sides.append((bcs, trees))
        lo += n_a * n_b
    entries += [Wd.Entry("none")] * (n - len(entries))
    salts = [rng.randbytes(CC.SALT_BYTES) for _ in entries]
    levels = tree_levels(dom, [Wd.entry_leaf(dom, e, Wd.entry_bc(x, s)) for e, (x, s) in enumerate(zip(entries, salts))])
    root = levels[-1][0].hex()
    entries_root = next(r for r in rec["roots"] if r["port"] == "entries")
    entries_root.update(root=root, domain=dom.domain_id.hex(), leaves=n)
    e = rng.randrange(2)
    entry = entries[e]
    unit = rng.randrange(entry.lo, entry.hi)
    i, j = Wd.tile_in(entry, unit)
    bcs, trees = sides[e]
    op = {"unit": unit, "entry": e, "entry_value": entry.to_json(), "entry_salt": salts[e].hex(),
          "window_path": [lv[(e >> k) ^ 1].hex() for k, lv in enumerate(levels[:-1])]}
    rows = {"a": [per["a"] * i + k for k in range(per["a"])], "b": [per["b"] * j + k for k in range(per["b"])]}
    for side, _ in CC.OPENED:
        sdom = Wd.side_domain(dom.domain_id, side)
        opened = [{"index": idx, "path": [x.hex() for x in Wd.side_path(sdom, trees[side][1], idx)]} for idx in rows[side]]
        op[side] = opened[0] if lay is None else opened
    salt = (rng.randbytes(PS.SALT_BYTES) if salt is None else salt) if salted else None
    if lay is None:
        schemas = {"a0": seg, "b0": v2, "o0": v2, "t0": v2}
        body = bcs["a"][i] + bcs["b"][j] + rng.randbytes(3 * CC.BC_BYTES)
    else:
        schemas = {x: CC.ROW_V2 for x in [*(f"a{k}" for k in range(lay.r)), *(f"b{k}" for k in range(lay.c)), "pub", "o0", "o1"]}
        body = (b"".join(bcs["a"][x] for x in rows["a"]) + b"".join(bcs["b"][x] for x in rows["b"])
                + CC.public_bc(context, salt, unit, (entry.root_a, entry.root_b)) + rng.randbytes(2 * CC.BC_BYTES))
    head = _header(schemas, CC.TILES[params["check"]["template"]].circuit)
    pub = json.dumps(head, sort_keys=True).encode() + b"\n" + body
    units = None if rng.random() < 0.3 else sorted({unit, *rng.sample(range(1 << 21), rng.randrange(4))})
    return {"params": params, "record": rec, "units": units, "unit": unit, "pub": pub, "opening": op, "header": head,
            "body": body, "entry": entry, "rows": bcs, "levels": levels, "dom": dom, "salt": salt, "layout": lay}


def _repub(case: dict, head, body: bytes | None = None, raw: bytes | None = None) -> None:
    case["pub"] = (raw if raw is not None else json.dumps(head).encode()) + b"\n" + (case["body"] if body is None else body)


def _spaced(h: str) -> str:
    return " ".join(h[k:k + 2] for k in range(0, len(h), 2))


#: each mutation, by the refusal (or the reading) it aims at
MUTATIONS = [
    "undrawn", "domain", "context", "root-hex", "root-short", "entry-missing", "entry-kind", "entry-root", "entry-negative",
    "entry-huge", "entry-float", "entry-bool", "entry-null", "opening-list", "index-string", "index-bool", "index-missing",
    "salt-short", "salt-type", "salt-upper", "salt-spaced", "salt-bad", "window-path-short", "window-path-type",
    "window-path-digest", "side-path-type", "side-path-digest", "side-missing", "side-index-type", "side-index-bool",
    "entry-other", "entry-negative-index", "entry-past", "forged-entry", "out-of-range", "kind-none", "no-newline",
    "header-not-json", "header-nan", "header-list", "header-utf8", "instances-missing", "instances-bool",
    "instances-two", "instances-float", "instances-string", "circuit-other", "circuit-missing", "circuit-type",
    "schemas-missing", "frame-type", "schemas-type",
    "schemas-swapped", "schemas-one", "schemas-unsorted", "schemas-prefixed", "schemas-between", "schemas-shadowed",
    "body-short", "a-index", "b-index",
    "a-row", "b-row", "b-path", "a-path-short", "rows-swapped", "past-side", "params-bad", "commitment-bad",
    "certified-unsalted", "uncertified-salt",
]
#: a salted case (`honest(…, salted=True)`), honest (``salted``) or with one mutation
SALTED = [
    "salted", "salted-no-salt", "salted-salt-short", "salted-other-salt", "salted-uncertified", "salted-position",
    "salted-roots", "salted-row-salt", "salted-schemas", "salted-schemas-missing", "salted-header-order", "salted-body-short",
    "salted-sides-short", "salted-side-dict", "salted-a-index", "salted-b-row", "salted-pub-swapped", "salted-circuit",
]
NAMES = MUTATIONS + SALTED


def instance(rng: random.Random, name: str = "honest") -> dict:
    """An honest opening from `rng`, then the mutation `name` (none for "honest" and "salted")."""
    case = honest(rng, salted=name in SALTED)
    op, ev, head, rec = case["opening"], case["opening"]["entry_value"], case["header"], case["record"]
    entries_root = next(r for r in rec["roots"] if r["port"] == "entries")
    if name in SALTED:
        _salted(case, rng, name)
    elif name == "undrawn":
        case["units"] = [u for u in (case["units"] or [case["unit"] + 1]) if u != case["unit"]]
    elif name == "domain":
        entries_root["domain"] = cases._hex(rng)
    elif name == "context":
        case["params"]["context"] = next(w[0] for w in WINDOWS if w[0] != case["params"]["context"])
    elif name in ("root-hex", "root-short"):
        entries_root["root"] = "zz" * 64 if name == "root-hex" else entries_root["root"][:-2]
    elif name == "entry-missing":
        del op["entry_value"]
    elif name == "entry-kind":
        ev["kind"] = rng.choice(["tile", 1, ["call"], None])
    elif name == "entry-root":
        ev[rng.choice(["root_a", "root_b", "root_h3", "record"])] = rng.choice(["00", "zz", 7])
    elif name in ("entry-negative", "entry-huge", "entry-float"):
        ev[rng.choice(["lo", "hi", "n_a", "n_b", "tile0", "cols", "m"])] = {"entry-negative": -1, "entry-huge": 1 << 64,
                                                                           "entry-float": "1"}[name]
    elif name == "entry-bool":
        ev["cols"] = True
    elif name == "entry-null":
        op["entry_value"] = rng.choice([None, [], "call", 3])
    elif name == "opening-list":
        case["opening"] = [op]
    elif name in ("index-string", "index-bool", "index-missing"):
        if name == "index-missing":
            del op["entry"]
        else:
            op["entry"] = str(op["entry"]) if name == "index-string" else bool(op["entry"])
    elif name in ("salt-short", "salt-type", "salt-upper", "salt-spaced", "salt-bad"):
        s = op["entry_salt"]
        op["entry_salt"] = {"salt-short": s[:-2], "salt-type": 5, "salt-upper": s.upper(), "salt-spaced": _spaced(s),
                            "salt-bad": s[:-1] + "g"}[name]
    elif name == "window-path-short":
        op["window_path"] = op["window_path"][:-1]
    elif name == "window-path-type":
        op["window_path"] = rng.choice(["ab", [1], None])
    elif name == "window-path-digest":
        op["window_path"][0] = op["window_path"][0][:64]
    elif name == "side-path-type":
        op[rng.choice("ab")]["path"] = rng.choice(["ab", [3], None])
    elif name == "side-path-digest":
        op["b"]["path"][-1] = op["b"]["path"][-1] + "00"
    elif name == "side-missing":
        op[rng.choice("ab")] = rng.choice([None, [], {"index": 0}])
    elif name in ("side-index-type", "side-index-bool"):
        side = rng.choice("ab")
        op[side]["index"] = str(op[side]["index"]) if name == "side-index-type" else bool(op[side]["index"])
    elif name in ("entry-other", "entry-negative-index", "entry-past"):
        op["entry"] = {"entry-other": 1 - op["entry"], "entry-negative-index": -1,
                       "entry-past": case["params"]["check"]["entries"]}[name]
    elif name == "forged-entry":
        ev["hi"] += 1
    elif name in ("out-of-range", "kind-none", "past-side"):
        if name == "kind-none":
            case["entry"] = Wd.Entry("none")
        _reopen(case, rng, name)
    elif name == "no-newline":
        case["pub"] = json.dumps(head).encode()
    elif name == "header-not-json":
        _repub(case, None, raw=b'{"instances": 1,')
    elif name == "header-nan":
        _repub(case, None, raw=json.dumps({**head, "instances": float("nan")}).encode())
    elif name == "header-list":
        _repub(case, [head])
    elif name == "header-utf8":
        _repub(case, None, raw=b"\xff" + json.dumps(head).encode())
    elif name == "instances-missing":
        _repub(case, {k: v for k, v in head.items() if k != "instances"})
    elif name in ("instances-bool", "instances-two", "instances-float", "instances-string"):
        _repub(case, {**head, "instances": {"instances-bool": True, "instances-two": rng.randint(2, 9),
                                            "instances-float": 1.0, "instances-string": "1"}[name]})
    elif name == "circuit-other":
        pinned = head["circuit_sha512"]
        other = next(t.circuit for t in CC.TILES.values() if t.circuit != pinned)
        _repub(case, {**head, "circuit_sha512": rng.choice([cases._hex(rng), pinned.upper(), other])})
    elif name == "circuit-missing":
        _repub(case, {k: v for k, v in head.items() if k != "circuit_sha512"})
    elif name == "circuit-type":
        pinned = head["circuit_sha512"]
        _repub(case, {**head, "circuit_sha512": rng.choice([None, 5, [pinned], {"sha512": pinned}])})
    elif name == "schemas-missing":
        _repub(case, {**head, "frame_v3": {k: v for k, v in head["frame_v3"].items() if k != "schemas"}})
    elif name in ("frame-type", "schemas-type"):
        f = rng.choice([[], "frame", None])
        _repub(case, {**head, "frame_v3": f} if name == "frame-type" else {**head, "frame_v3": {**head["frame_v3"], "schemas": f}})
    elif name in ("schemas-swapped", "schemas-one", "schemas-unsorted", "schemas-prefixed", "schemas-between",
                  "schemas-shadowed"):
        s, x = head["frame_v3"]["schemas"], rng.choice(["a1", "a0x", "b"])
        new = {"schemas-swapped": {"a0": s["b0"], "b0": s["a0"], "o0": s["o0"]}, "schemas-one": {"a0": s["a0"]},
               "schemas-unsorted": dict(reversed(list(s.items()))), "schemas-prefixed": {"0x": s["o0"], **s},
               "schemas-between": {**s, x: s["b0"]}, "schemas-shadowed": {**s, x: s["b0"], "b0": s["a0"]}}[name]
        _repub(case, {**head, "frame_v3": {**head["frame_v3"], "schemas": new}})
    elif name == "body-short":
        _repub(case, head, case["body"][:rng.randrange(2 * CC.BC_BYTES)])
    elif name in ("a-index", "b-index"):
        op[name[0]]["index"] += rng.choice([-1, 1])
    elif name in ("a-row", "b-row"):
        b = bytearray(case["body"])
        k = (0 if name == "a-row" else CC.BC_BYTES) + rng.randrange(CC.BC_BYTES)
        b[k] ^= 1 << rng.randrange(8)
        _repub(case, head, bytes(b))
    elif name == "b-path":
        op["b"]["path"][rng.randrange(len(op["b"]["path"]))] = cases._hex(rng)
    elif name == "a-path-short":
        op["a"]["path"] = op["a"]["path"][1:]
    elif name == "rows-swapped":
        b = case["body"]
        _repub(case, head, b[CC.BC_BYTES:2 * CC.BC_BYTES] + b[:CC.BC_BYTES] + b[2 * CC.BC_BYTES:])
    elif name == "certified-unsalted":
        case["params"]["context"] = rng.choice(CERTIFIED)[0]
    elif name == "uncertified-salt":
        case["salt"] = rng.randbytes(PS.SALT_BYTES)
    elif name not in ("honest", "params-bad", "commitment-bad"):
        raise AssertionError(name)
    params = case["params"]
    if name == "params-bad":
        params = {**params, "delta": "2/4"}
    rec_b = cases._canon(rec).encode()
    if name == "commitment-bad":
        rec_b = json.dumps(rec, sort_keys=True).encode()
    return {"params": params, "commitment": rec_b.hex(), "units": case["units"], "unit": case["unit"],
            "pub": case["pub"].hex(), "opening": case["opening"], "salt": None if case["salt"] is None else case["salt"].hex()}


def _salted(case: dict, rng: random.Random, name: str) -> None:
    """A salted case's mutation `name`: the verifier's salt, the window's context, the public row the proof's body holds,
    the header's rows, or the opened rows."""
    op, head, lay, entry = case["opening"], case["header"], case["layout"], case["entry"]
    at = (lay.r + lay.c) * CC.BC_BYTES
    body, ctx, salt, unit = case["body"], case["params"]["context"], case["salt"], case["unit"]
    roots = (entry.root_a, entry.root_b)
    if name == "salted-no-salt":
        case["salt"] = None
    elif name == "salted-salt-short":
        case["salt"] = salt[:rng.randrange(PS.SALT_BYTES)]
    elif name == "salted-other-salt":
        case["salt"] = rng.randbytes(PS.SALT_BYTES)
    elif name == "salted-uncertified":
        case["params"]["context"], case["salt"] = rng.choice(WINDOWS)[0], None
    elif name in ("salted-position", "salted-roots", "salted-row-salt"):
        pub = {"salted-position": lambda: CC.public_bc(ctx, salt, unit + 1, roots),
               "salted-roots": lambda: CC.public_bc(ctx, salt, unit, roots[::-1]),
               "salted-row-salt": lambda: CC.public_bc(rng.choice(CERTIFIED + WINDOWS)[0] + "/x", salt, unit, roots)}[name]()
        _repub(case, head, body[:at] + pub + body[at + CC.BC_BYTES:])
    elif name in ("salted-schemas", "salted-schemas-missing"):
        s = dict(head["frame_v3"]["schemas"])
        x = rng.choice([*(f"a{k}" for k in range(lay.r)), *(f"b{k}" for k in range(lay.c)), "pub"])
        if name == "salted-schemas":
            s[x] = CC.OPENED[0][1]
        else:
            del s[x]
        _repub(case, {**head, "frame_v3": {**head["frame_v3"], "schemas": s}})
    elif name == "salted-header-order":
        s = {**head["frame_v3"]["schemas"], rng.choice(["a00", f"a{lay.r}", "b", f"a{lay.r - 1}x"]): CC.ROW_V2}
        _repub(case, {**head, "frame_v3": {**head["frame_v3"], "schemas": s}})
    elif name == "salted-body-short":
        _repub(case, head, body[:rng.randrange(at + CC.BC_BYTES)])
    elif name == "salted-sides-short":
        side = rng.choice("ab")
        op[side] = op[side][:-1] if rng.random() < 0.5 else op[side] + op[side][:1]
    elif name == "salted-side-dict":
        op[rng.choice("ab")] = op["a"][0]
    elif name == "salted-a-index":
        op["a"][rng.randrange(lay.r)]["index"] += rng.choice([-1, 1])
    elif name == "salted-b-row":
        b = bytearray(body)
        b[(lay.r + rng.randrange(lay.c)) * CC.BC_BYTES + rng.randrange(CC.BC_BYTES)] ^= 1 << rng.randrange(8)
        _repub(case, head, bytes(b))
    elif name == "salted-pub-swapped":
        last = at - CC.BC_BYTES
        _repub(case, head, body[:last] + body[at:at + CC.BC_BYTES] + body[last:at] + body[at + CC.BC_BYTES:])
    elif name == "salted-circuit":
        other = rng.choice([t.circuit for t in CC.TILES.values() if t.salted is None])
        _repub(case, {**head, "circuit_sha512": other})
    elif name != "salted":
        raise AssertionError(name)


def _reopen(case: dict, rng: random.Random, name: str) -> None:
    """Point the opening at a unit its entry doesn't hold (`out-of-range`, or any unit of an empty entry, `kind-none`), or
    at a row past the side tree (`past-side`: an entry whose range claims more tiles than its sides hold), the entry and
    its window path re-committed so only that rule refuses."""
    entry = case["entry"]
    if name == "past-side":
        entry = Wd.Entry(entry.kind, entry.lo, entry.lo + (1 << 13) * entry.n_b + 1, entry.n_a, entry.n_b, entry.root_a,
                         entry.root_b, entry.root_h3, entry.record, entry.tile0, entry.cols, entry.m)
        unit = entry.hi - 1
    else:
        unit = entry.hi + rng.randrange(3) if entry.kind != "none" else rng.randrange(1 << 20)
    op, dom = case["opening"], case["dom"]
    e = op["entry"]
    leaves = list(case["levels"][0])
    leaves[e] = Wd.entry_leaf(dom, e, Wd.entry_bc(entry, bytes.fromhex(op["entry_salt"])))
    levels = tree_levels(dom, leaves[:dom.count])
    next(r for r in case["record"]["roots"] if r["port"] == "entries")["root"] = levels[-1][0].hex()
    op.update(unit=unit, entry_value=entry.to_json(),
              window_path=[lv[(e >> k) ^ 1].hex() for k, lv in enumerate(levels[:-1])])
    if name == "past-side":
        op["a"]["index"], op["b"]["index"] = divmod(unit - entry.lo, entry.n_b)
    case["unit"] = unit
    case["units"] = None
