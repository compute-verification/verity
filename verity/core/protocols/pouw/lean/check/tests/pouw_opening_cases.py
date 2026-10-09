"""The opening difftest's cases: generated openings of one unit, each ``{"params", "commitment": hex, "units", "unit",
"pub": hex, "opening"}``, the arguments of `core_check.opening` (`params` and the registration's bytes as `answer` reads
them, `units` null for a unit opened undrawn).

A case is an honest opening (a window of P entries, a call's and the filler's each with small side trees, the unit one
of their tiles, its proof's public file a header and the two rows' b ‖ c, the registration's entries root over the
entries' leaves), then one mutation (`MUTATIONS`) aimed at one refusal of `core_check.opening`, or at a reading both
sides must share (hex as `bytes.fromhex` reads it, the header's keys in sorted order).  The windows come from a few
fixed contexts, so the side trees' pad nodes, which only the generator computes, are computed once per side domain.
`tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_core_check.py`
holds each mutation to the refusal it aims at.
"""
from __future__ import annotations

import json
import random

import pouw_check_cases as cases
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import window as Wd
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import row_leaf

#: (context, program, partition, entries): the windows the generator opens
WINDOWS = [(f"served-zk/opening-{i}/{random.Random(i).randbytes(8).hex()}", random.Random(10 + i).randbytes(64).hex(),
            random.Random(20 + i).randbytes(64).hex(), n) for i, n in enumerate((4, 5))]


def _leaf(dom, i: int, bc: bytes, schema: str) -> bytes:
    return row_leaf(dom, i, Wd._HM.tree_leaf(Wd._HM.default_key, bc), schema)


def _side(dom, schema: str, bcs: list[bytes]) -> tuple[bytes, list]:
    levels = Wd.side_levels(dom, [_leaf(dom, i, bc, schema) for i, bc in enumerate(bcs)])
    return levels[-1][0], levels


def _header(schemas: dict, rng: random.Random) -> dict:
    return {"circuit_sha512": cases._hex(rng), "format": "flock-circuit-inputs", "instances": 1,
            "frame_v3": {"frame": "frame-v3-sha512", "owner": -1, "schemas": schemas}}


def honest(rng: random.Random) -> dict:
    """A window from `WINDOWS`, its entries' tree and side trees, and one unit's opening and proof's public file."""
    context, program, partition, n = rng.choice(WINDOWS)
    base = cases.honest(rng)
    params = {**base["params"], "context": context, "check": {**base["params"]["check"], "entries": n}}
    rec = base["record"]
    rec.update(program=program, partition=partition)
    rec["window"]["context"] = context
    dom = CC.window_domain(CC.params(params), rec)
    seg, v2 = (s for _, s in CC.OPENED)
    entries, sides, lo = [], [], rng.randrange(1 << 20)
    for kind in ("call", "filler"):
        n_a, n_b = rng.randint(1, 3), rng.randint(1, 4)
        bcs = {"a": [rng.randbytes(CC.BC_BYTES) for _ in range(n_a)], "b": [rng.randbytes(CC.BC_BYTES) for _ in range(n_b)]}
        trees = {s: _side(Wd.side_domain(dom.domain_id, s), schema, bcs[s]) for s, schema in CC.OPENED}
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
    for (side, _), idx in zip(CC.OPENED, (i, j)):
        sdom = Wd.side_domain(dom.domain_id, side)
        op[side] = {"index": idx, "path": [x.hex() for x in Wd.side_path(sdom, trees[side][1], idx)]}
    schemas = {"a0": seg, "b0": v2, "o0": v2, "t0": v2}
    head = _header(schemas, rng)
    pub = json.dumps(head, sort_keys=True).encode() + b"\n" + bcs["a"][i] + bcs["b"][j] + rng.randbytes(3 * CC.BC_BYTES)
    units = None if rng.random() < 0.3 else sorted({unit, *rng.sample(range(1 << 21), rng.randrange(4))})
    return {"params": params, "record": rec, "units": units, "unit": unit, "pub": pub, "opening": op, "header": head,
            "body": pub.partition(b"\n")[2], "entry": entry, "rows": bcs, "levels": levels, "dom": dom}


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
    "instances-two", "instances-float", "instances-string", "schemas-missing", "frame-type", "schemas-type",
    "schemas-swapped", "schemas-one", "schemas-unsorted", "schemas-prefixed", "body-short", "a-index", "b-index",
    "a-row", "b-row", "b-path", "a-path-short", "rows-swapped", "past-side", "params-bad", "commitment-bad",
]
NAMES = MUTATIONS


def instance(rng: random.Random, name: str = "honest") -> dict:
    """An honest opening from `rng`, then the mutation `name` (none for "honest")."""
    case = honest(rng)
    op, ev, head, rec = case["opening"], case["opening"]["entry_value"], case["header"], case["record"]
    entries_root = next(r for r in rec["roots"] if r["port"] == "entries")
    if name == "undrawn":
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
    elif name == "schemas-missing":
        _repub(case, {**head, "frame_v3": {k: v for k, v in head["frame_v3"].items() if k != "schemas"}})
    elif name in ("frame-type", "schemas-type"):
        f = rng.choice([[], "frame", None])
        _repub(case, {**head, "frame_v3": f} if name == "frame-type" else {**head, "frame_v3": {**head["frame_v3"], "schemas": f}})
    elif name in ("schemas-swapped", "schemas-one", "schemas-unsorted", "schemas-prefixed"):
        s = head["frame_v3"]["schemas"]
        new = {"schemas-swapped": {"a0": s["b0"], "b0": s["a0"], "o0": s["o0"]}, "schemas-one": {"a0": s["a0"]},
               "schemas-unsorted": dict(reversed(list(s.items()))), "schemas-prefixed": {"0x": s["o0"], **s}}[name]
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
    elif name not in ("honest", "params-bad", "commitment-bad"):
        raise AssertionError(name)
    params = case["params"]
    if name == "params-bad":
        params = {**params, "delta": "2/4"}
    rec_b = cases._canon(rec).encode()
    if name == "commitment-bad":
        rec_b = json.dumps(rec, sort_keys=True).encode()
    return {"params": params, "commitment": rec_b.hex(), "units": case["units"], "unit": case["unit"],
            "pub": case["pub"].hex(), "opening": case["opening"]}


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
