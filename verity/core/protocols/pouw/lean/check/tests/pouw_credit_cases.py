"""The credit difftest's cases: the arguments of `credit.window` and `credit.period_credit`, the entries, and of the
steps they run or will, each held to the Lean on its own: `credit._credited`, `credit.drawn`, `credit._outcome`,
`credit._a_link` and `credit._weights`.

A `credited` case is an honest served window read through its challenger's session (`pouw_rounds_cases.Session`): its
manifest rows (the serving process's call rows, each with its credit bit, a credited one with its z table's root), the
round log (each round's root the challenger received, over its inner root and its A segments, `_a_log`, and its coin),
the window's registered entries, and the draw: `credit.draw_root` over them, recorded by the
challenger, then its draw coin opened after it; the auditor's ledger opens and ends the session for the window. Then one
mutation (`CREDIT_NAMES`) aimed at one refusal: a credit bit flipped after the draw coin with the draw root recomputed
(``changed``, RT4-INT-32's withdrawn verdict), a row changed likewise, a missing or malformed bit, a credited row with no
z root, the draw root left stale, no draw coin, a coin of another family, and the forms.

A `drawn` case is a draw coin, the draw's stream, registration and receipt, a window's layout
(`circuit.h1window.program`'s arguments, `layout`): linears at one or two K, called step by step in a shuffled order, so
each K's stratum has ranges apart, and the committed K the draw is sized at (one to the window's units). Its mutations
break each rule `credit.strata` mirrors from `h1window.program`, the coin's, the arguments' and K's forms, give a K past
the window's units, and ``budget`` draws at K = `PLAN_K` from a population larger than it (each stratum's k_s rounded
up, so a unit or two past K).

An `outcome` case is a `drawn` case's arguments and one accepted proof outcome for each unit the coin draws; its
mutations drop one (``missing``), reject one (``failed``), repeat one, add an undrawn unit, prove the units of another
coin the prover chose (``chosen``), and break the forms.

A `window` case is a `credited` case with its circuit's descriptor (`window_desc`: `h1window.program`'s encoding as far
as `credit.layout_of` reads it, each call's tiles a call node each or one batch), a committed K, one call per manifest
row, and an accepted proof of each unit its draw coin draws at K under the served window's arguments; its mutations add
a call the manifest lacks, prove units the prover chose under its own stream, fail or drop a drawn unit, break K's form
or give one past the window's units, pass a `credited` refusal through, and break the descriptor (`DESC_NAMES`): its
form, its program function's record, a template, a K no linear has, the root's nodes, the record's layout, or a record
naming another layout over the same nodes (``alt-layout``). Its mutations of the composed check present a round log the
challenger never received (``never-received``, its coins carried as if opened after its roots and its draw root over
it, so only `core_check.rounds` refuses it), a round coin opened after another root, an A tree the round roots didn't
commit, and no A value.

A `period` case is one to three such windows of one circuit in one period of the auditor's ledger (`period-open`
committing the SHA-512 of the circuit's descriptor and the K every window is drawn at, `credit.K_FIELD`, each window's
session opened and ended, then the `period-close` listing them), presented in its close's order, each with that
descriptor, a window of no period beside it in the ledger. Its mutations (RT4-INT-42) leave the period unclosed or a
window incomplete, drop, reorder or add a window, present the window of no period among the period's, name a period
never opened or no period, forge the ledger, refuse one window, give one window another circuit's descriptor
(``other-circuit``, or the descriptor of another layout, ``alt-layout``) or open the period committing no circuit or
one that is no H1 window's (``bad-circuit``), open it committing no K or a malformed one, commit a K past a window's
units, prove one window's draw at another K (``other-k``), present one window's round log the challenger never received
(``never-received``), and break the forms.

An `a_link` case is a window's registered entries, among them one or two A values (``…/a/<K>``) whose trees are frame-v3
SHA-512 trees (`verity.ml.commitments.merkle.tree_levels`) of `steps` equal blocks, and its round log: each round's
`credit.cover` segments of every A value under `credit.round_root` of a random inner root. Its mutations register an A
tree other than the one the rounds committed (RT4-INT-34), rewrite a segment with or without its round root, leave a gap,
add a segment, misalign one, list an unregistered value, register none or one twice, and break the forms.

A `weights` case is a window's layout, its circuit (its descriptor's SHA-512 and PREFIX), its registered entries (each
K's ``<PREFIX>b/<K>`` under its registered domain, among its A values, z tables and names that are no weights: a
non-canonical K, ``b/032`` or one past Python's ``int`` digit limit), its round-0 request table's root and its session,
and the window's open root over `credit.weights_root` of the ``…/b/<K>`` roots, that root and the session
(`salt.window_open_root`). Its mutations register a B value other than the one the open root binds (another code or
s_w), open the window on another request table's root or session or another root, register a K's B value twice, never,
of a K the layout lacks or under another prefix, give its entry other rows, words, domain (another circuit's, a
window's, other rows') or keys, and break the forms.

`tests/test_difftest.py` draws them for the harness, and `verity/core/protocols/pouw/tests/test_pouw_credit.py` holds each
mutation to the refusal it aims at.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random

import pouw_rounds_cases as rc
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels

MODULES = ["q_proj", "k_proj", "o_proj", "down_proj", "attn·out"]


def _hex(rng: random.Random) -> str:
    return rng.randbytes(64).hex()


def _row(rng: random.Random, i: int) -> dict:
    """A manifest row as the serving process writes it (`Call.summary()` with its tables and bound), its credit bit
    among it."""
    bit = rng.random() < 0.7
    tables = {"a": _hex(rng), **({"z": _hex(rng)} if bit or rng.random() < 0.5 else {})}
    return {"index": i, "step": rng.randrange(4), "layer": rng.randrange(32), "module": rng.choice(MODULES),
            "served": rng.random() < 0.8, "rows": rng.randint(1, 64), "k": rng.choice([64, 4096]), "n": 4096,
            "base": rng.randrange(1 << 20), "units": rng.randint(1, 8), "leaves": [rng.randrange(-3, 1 << 40), None],
            "tables": tables, "bound": {"m": rng.randint(1, 1 << 16), "ok": True}, Cr.CREDIT: bit}


def _draw_root(d: dict) -> str:
    salt, served = bytes.fromhex(d["salt"]), d["served"]
    calls = Cr.calls_root(salt, d["calls"], None if served is None else bytes.fromhex(served))
    return Cr.draw_root(calls, bytes.fromhex(d["draw"]["receipts"]), d["rounds"]).hex()


def _a_log(rng: random.Random, steps: int) -> tuple[list[dict], list[dict]]:
    """A window's registered entries, one or two A values (``…/a/<K>``) of `steps` equal blocks among them, and its round
    log over them (`_log`), its coins not yet opened."""
    window = f"h1/w-{rng.randrange(100)}"
    values, entries = {}, []
    for k in rng.sample([64, 4096, 14336], rng.randint(1, 2)):
        per = rng.choice([1, 2, 3, 4])
        domain, levels = _tree(rng, steps * per)
        values[f"{window}/a/{k}"] = (steps * per, levels)
        entries.append({"value": f"{window}/a/{k}", "root": levels[-1][0].hex(), "leaves": steps * per, "words": 96,
                        "domain": domain})
    if rng.random() < 0.5:
        entries.append({"value": f"{window}/z/{rng.choice([64, 4096])}", "root": _hex(rng), "leaves": 3,
                        "domain": _hex(rng)})
    rng.shuffle(entries)
    return entries, _log(steps, values, [_hex(rng) for _ in range(steps)])


def honest_credit(rng: random.Random, name: str = "honest", rows: int | None = None, sid: str = "s-1",
                  leaf: int = rc.RECORD_LEAF) -> tuple[dict, rc.Session]:
    """An honest served window of `rows` manifest rows (one to five when None) whose draw coin opened after its draw root
    (none for ``no-draw``), its session `sid` signing its record at `leaf`. Its round log is `_a_log`'s, each round's
    root the one the challenger received and its coin the one opened after it; under ``never-received`` the challenger
    received other roots, and the log carries their coins as if opened after its own, its draw root over it (the
    composed check's counterexample: `_credited` and `_a_link` accept it, `core_check.rounds` doesn't)."""
    rounds = rng.choice([1, 2, 3])
    ch = Wd.live_challenger(rng.randrange(4), rounds, rng.randbytes)
    entries, log = _a_log(rng, rounds)
    d = {"salt": _hex(rng), "calls": [_row(rng, i) for i in range(rng.randint(1, 5) if rows is None else rows)],
         "served": rng.choice([None, _hex(rng)]), "draw": {"receipts": _hex(rng)}, "entries": entries}
    received = [bytes.fromhex(x["root"]) for x in log]
    if name == "never-received":
        received = [rng.randbytes(64) for _ in log]

    def draw_after(rec) -> bytes:
        opened = rec.openings if name != "never-received" else [
            dataclasses.replace(o, after=bytes.fromhex(x["root"])) for o, x in zip(rec.openings, log)]
        d["rounds"] = [{**x, "coin": o.to_json()} for x, o in zip(log, opened)]
        d["draw"]["root"] = _draw_root(d)
        return bytes.fromhex(d["draw"]["root"])

    s = rc.Session(rng, ch, received, salt=None, sid=sid, leaf=leaf, draw=None if name == "no-draw" else draw_after)
    if name == "no-draw":
        d["draw"]["coin"] = ch.open(Wd.DRAW, 0, draw_after(s.rounds)).to_json()
    else:
        d["draw"]["coin"] = s.draw[0].to_json()
    return {"key": s.key(), "window": rc._window(rng), "session": s.files(), **d}, s


def _flip_bit(d: dict, rng: random.Random, to: bool | None = None) -> None:
    rows = [r for r in d["calls"] if to is None or r[Cr.CREDIT] != to] or d["calls"]
    row = rng.choice(rows)
    row[Cr.CREDIT] = not row[Cr.CREDIT] if to is None else to
    if row[Cr.CREDIT]:
        row["tables"].setdefault("z", _hex(rng))


def _mutate_credit(d: dict, s: rc.Session, name: str, rng: random.Random) -> None:
    calls, j = d["calls"], rng.randrange(len(d["calls"]))
    match name:
        case "changed":
            _flip_bit(d, rng)
            d["draw"]["root"] = _draw_root(d)
        case "withdrawn":
            _flip_bit(d, rng, to=False)
            d["draw"]["root"] = _draw_root(d)
        case "row-changed":
            calls[j]["tables"]["a"] = _hex(rng)
            d["draw"]["root"] = _draw_root(d)
        case "stale-root":
            _flip_bit(d, rng)
        case "no-bit":
            del calls[j][Cr.CREDIT]
        case "bit-form":
            calls[j][Cr.CREDIT] = rng.choice([1, 0, "true", None, [True]])
        case "no-z":
            calls[j][Cr.CREDIT] = True
            calls[j]["tables"] = rng.choice([{"a": _hex(rng)}, {"a": _hex(rng), "z": _hex(rng)[:-2]}, None,
                                             {"a": _hex(rng), "z": _hex(rng).upper()}])
        case "index":
            calls[j]["index"] = rng.choice([j + 1, str(j), None, True if j == 1 else -1])
        case "row-form":
            deep = 0
            for _ in range(Cr.DEPTH):
                deep = [deep]
            calls[j] = rng.choice([[1], "row", None, {**calls[j], "bound": deep}])
        case "calls-form":
            d["calls"] = rng.choice([None, {"0": calls}, "x"])
        case "salt-form":
            d["salt"] = rng.choice([None, d["salt"][:-2], d["salt"].upper(), 5])
        case "served-form":
            d["served"] = rng.choice(["", "ab", 5, _hex(rng).upper()])
        case "rounds-form":
            k = rng.randrange(len(d["rounds"]))
            d["rounds"] = rng.choice([None, "x", d["rounds"][:k] + [5] + d["rounds"][k + 1:],
                                      d["rounds"][:k] + [{**d["rounds"][k], "root": "zz"}] + d["rounds"][k + 1:],
                                      d["rounds"][:k] + [{**d["rounds"][k], "coin": None}] + d["rounds"][k + 1:]])
        case "draw-form":
            d["draw"] = rng.choice([None, {**d["draw"], "receipts": "zz"}, {k: v for k, v in d["draw"].items()
                                                                            if k != "receipts"}])
        case "no-draw":
            pass
        case "draw-coin":
            coin = d["draw"]["coin"]
            other = Wd.live_challenger(s.ch.commitment.epoch, len(d["rounds"]), rng.randbytes)
            d["draw"]["coin"] = rng.choice([{**coin, "salt": rc._flip(coin["salt"])}, None, {**coin, "index": 1},
                                            {**coin, "after": rc._flip(coin["after"])},
                                            other.open(Wd.DRAW, 0, bytes.fromhex(d["draw"]["root"])).to_json()])
        case "key-form":
            d["key"] = rng.choice([None, {**d["key"], "public": "00"}, {**d["key"], "H": 0}])
        case "forged-ledger":
            d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s), rc._key(1))
        case _:
            raise ValueError(name)


CREDIT_NAMES = ["changed", "withdrawn", "row-changed", "stale-root", "no-bit", "bit-form", "no-z", "index", "row-form",
                "calls-form", "salt-form", "served-form", "rounds-form", "draw-form", "no-draw", "draw-coin", "key-form",
                "forged-ledger"]
CREDIT_KEYS = ("key", "ledger", "window", "session", "salt", "calls", "served", "rounds", "draw")


def credit_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `credited` case: an honest window, then mutation `name`."""
    d, s = honest_credit(rng, name)
    if name != "honest":
        _mutate_credit(d, s, name, rng)
    if "ledger" not in d:
        d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s), rc._key(0))
    return {k: d[k] for k in CREDIT_KEYS}


def credit_args(d: dict) -> tuple:
    """`credit._credited`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return (*(d[k] for k in CREDIT_KEYS), rc.lms_verify)


LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def layout(rng: random.Random, big: bool = False) -> dict:
    """A served window's layout, `h1window.program`'s arguments: one to four linears at one or two K, each n whole blocks
    of the tile's C, called once a step over one to three steps (A block s at step s) in a shuffled order; ``big``, a
    population past `credit.RECORD_K`."""
    r, c = rng.choice([(1, 1), (2, 4), (4, 8), (8, 8), (16, 16)])
    ks = rng.sample([32, 64, 4096, 14336], rng.randint(1, 2))
    names = rng.sample(LINEARS, rng.randint(len(ks), 4))
    linears = [[name, ks[i] if i < len(ks) else rng.choice(ks), c * (rng.randint(1, 4) if not big else 64)]
               for i, name in enumerate(names)]
    calls = [[name, step] for step in range(rng.randint(1, 3)) for name in names]
    rng.shuffle(calls)
    m = r << (rng.randint(0, 2) if not big else 9)
    return {"calls": calls, "linears": linears, "m": m, "tile": [r, c]}


def _canon(x: object) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _template_id(k: int, unit: dict) -> str:
    return f"{Cr.TILE_FN}{{UNIT={_canon(unit)},K={k}}}"


def _template(k: int, r: int, c: int, unit: dict) -> dict:
    """K's template as `h1window.template` encodes it: a composite over the tile unit `unit`, its A and B rows
    ``a_k<K>`` and ``b_k<K>``."""
    word = ["Value", 16]
    params = [[f"a_k{k}", ["Array", r, ["Array", 2, word]]], [f"b_k{k}", ["Array", c, ["Array", 2, word]]],
              ["zy", ["Array", 3 * r * c, word]], ["salt", ["Array", 4, word]], ["position", word],
              ["roots", ["Array", 2, ["Array", 4, word]]]]
    return {"form": "composite", "id": Cr.TILE_FN, "bindings": {"K": k, "UNIT": unit},
            "body": {"params": params, "ret_type": word, "nodes": [{"form": "call", "fn": 0, "args": [], "out": 0}],
                     "ret": {"affine": ["n", 0, 0, 1]}}}


def record(tiles: list, lay: dict, prefix: str = "") -> str:
    """`h1window.program`'s program function: `credit.WINDOW_FN` and its record."""
    vals = {"TILES": tiles, "LINEARS": lay["linears"], "CALLS": lay["calls"], "M": lay["m"], "PREFIX": prefix}
    return Cr.WINDOW_FN + "{" + ",".join(f"{k}={_canon(vals[k])}" for k in Cr.RECORD) + "}"


def window_desc(rng: random.Random, lay: dict, prefix: str = "") -> dict:
    """A served window's circuit as `h1window.program(linears, calls, m, tile)` encodes it (`codec.encode_program`), as
    far as `credit.layout_of` reads it: its program function's record, each K's template, and the root's input batches,
    then each call's (m/R)·(n/C) tiles, each a call of K's template or, at random, a run of them one batch."""
    (r, c), ks = lay["tile"], sorted({k for _name, k, _n in lay["linears"]})
    units = {k: {"fn": f"H1Tile_v1{{K={k},R={r},C={c}}}"} for k in ks}
    tids = {k: _template_id(k, units[k]) for k in ks}
    callees = ["Input16_v1", *(u["fn"] for u in units.values()), *tids.values()]
    defs = {u["fn"]: {"form": "primitive", "id": u["fn"], "params": [], "ret": ["Value", 16]} for u in units.values()}
    defs |= {tids[k]: _template(k, r, c, units[k]) for k in ks}
    defs["Input16_v1"] = {"form": "primitive", "id": "Input16_v1", "params": [], "ret": ["Value", 16]}
    nodes = [{"form": "batch", "fn": 0, "args": [], "out": 0, "axes": [], "n": rng.randint(1, 64)}
             for _ in range(2 * len(ks) + 3)]
    n_of = {name: (k, n) for name, k, n in lay["linears"]}
    for lin, _blk in lay["calls"]:
        k, n = n_of[lin]
        size, fn = (lay["m"] // r) * (n // c), callees.index(tids[k])
        if rng.random() < 0.3:
            nodes.append({"form": "batch", "fn": fn, "args": [], "out": 1, "axes": [0], "n": size})
        else:
            nodes += [{"form": "call", "fn": fn, "args": [], "out": 1} for _ in range(size)]
    fn = record([[k, {"fn": tids[k]}] for k in ks], lay, prefix)
    root = f"Root[{fn}]_v1"
    defs[root] = {"form": "composite", "id": root, "bindings": {},
                  "body": {"params": [], "ret_type": ["Value", 16], "nodes": nodes, "ret": {"concat": []}}}
    return {"callees": callees, "definitions": defs, "program_fn": fn, "root": root, "schema": Cr.SCHEMA,
            "types": [["Value", 16]]}


def desc_bytes(desc: dict) -> bytes:
    """A descriptor's bytes as the verifier holds them: canonical JSON without annotations."""
    return _canon(desc).encode()


def descriptor(rng: random.Random, lay: dict, prefix: str = "") -> str:
    """`window_desc`'s bytes in hex, as a window's evidence carries them."""
    return desc_bytes(window_desc(rng, lay, prefix)).hex()


def _alt_layout(rng: random.Random, lay: dict) -> dict:
    """Another layout of `lay`'s linears: another slot, another n, or the calls in another order, with other units or
    other strata than `lay`'s."""
    (r, c), out = lay["tile"], json.loads(json.dumps(lay))
    while True:
        match rng.choice(["m", "n", "calls", "drop"]):
            case "m":
                out["m"] = lay["m"] * 2
            case "n":
                out["linears"][0][2] = lay["linears"][0][2] + c
            case "calls" if len({k for _n, k, _ in lay["linears"]}) > 1:
                rng.shuffle(out["calls"])
            case "drop" if len(lay["calls"]) > len(lay["linears"]):
                del out["calls"][next(i for i, (lin, _b) in enumerate(lay["calls"])
                                      if sum(x == lin for x, _ in lay["calls"]) > 1
                                      and max(b for x, b in lay["calls"] if x == lin) == lay["calls"][i][1])]
            case _:
                continue
        if (Cr.strata(out)[0] or None) != Cr.strata(lay)[0] and Cr.strata(out)[1] is None:
            return out
        out = json.loads(json.dumps(lay))


def _break_desc(rng: random.Random, lay: dict, name: str) -> str:
    """A descriptor of `lay`'s window broken as `name` says, in hex."""
    desc = window_desc(rng, lay)
    defs, fn = desc["definitions"], desc["program_fn"]
    root = defs[desc["root"]]["body"]["nodes"]
    tiles = json.loads(dict(Cr._record(fn))["TILES"])
    tid = rng.choice(tiles)[1]["fn"]
    match name:
        case "descriptor-form":
            raw = desc_bytes(desc)
            return rng.choice([raw + b" ", raw.replace(b",", b", ", 1), b"\xff" + raw, b"[]",
                               desc_bytes({**desc, "schema": "verity-ir/descriptor/v2"}),
                               desc_bytes({k: v for k, v in desc.items() if k != "callees"}),
                               desc_bytes({**desc, "callees": [*desc["callees"], 5]}),
                               desc_bytes({**desc, "program_fn": None}), b'{"a":NaN}']).hex()
        case "not-window":
            pairs = Cr._record(fn)
            other = rng.choice([
                "PouwH1Window_v2" + fn[len(Cr.WINDOW_FN):],
                Cr.WINDOW_FN + "{" + ",".join(f"{k}={v}" for k, v in pairs[:-1]) + "}",
                Cr.WINDOW_FN + "{" + ",".join(f"{k}={v}" for k, v in [pairs[1], pairs[0], *pairs[2:]]) + "}",
                fn.replace(f"M={lay['m']}", f"M= {lay['m']}"), fn.replace(f"M={lay['m']}", f"M=0{lay['m']}"),
                fn.replace('PREFIX=""', "PREFIX=5"), record(tiles[::-1] if len(tiles) > 1 else [], lay),
                record([[t[0], t[1]["fn"]] for t in tiles], lay), record([[0, tiles[0][1]]] + tiles[1:], lay)])
            desc["program_fn"] = other
        case "template":
            t = defs[tid]
            k = next(k for k, x in tiles if x["fn"] == tid)
            body = t["body"]
            match rng.choice(["k", "id", "form", "params", "rows", "unit", "missing"] + ["rc"] * (len(tiles) > 1)):
                case "k":
                    t["bindings"]["K"] = k + 1
                case "id":
                    t["id"] = "PouwH1WindowTile_v2"
                case "form":
                    t["form"] = "primitive"
                case "params":
                    body["params"] = body["params"][:4]
                case "rows":
                    body["params"][rng.randrange(2)][1] = rng.choice([["Array", 0, ["Value", 16]], ["Value", 16]])
                case "unit":
                    t["bindings"]["UNIT"] = {"fn": "Other_v1"}
                case "missing":
                    del defs[tid]
                case "rc":
                    body["params"][1][1][1] += 1
        case "tile-ks":
            k = max(k for k, _t in tiles) * 2
            unit = {"fn": f"H1Tile_v1{{K={k}}}"}
            extra = _template_id(k, unit)
            defs[extra] = _template(k, *lay["tile"], unit)
            desc["program_fn"] = record(tiles + [[k, {"fn": extra}]], lay)
        case "body":
            calls = [i for i, x in enumerate(root) if not desc["callees"][x["fn"]].startswith("Input")]
            i = rng.choice(calls)
            match rng.choice(["drop", "add", "zero", "scan", "foreign", "fn"]):
                case "drop":
                    del root[i]
                case "add":
                    root.insert(i, dict(root[i]))
                case "zero":
                    root[i] = {**root[i], "form": "batch", "n": 0}
                case "scan":
                    root[i] = {**root[i], "form": "scan", "n": 1}
                case "foreign":
                    root[i] = {**root[i], "fn": 1}
                case "fn":
                    root[i] = {**root[i], "fn": rng.choice([len(desc["callees"]), -1, "2", None])}
        case "layout-form":
            lin = lay["linears"]
            desc["program_fn"] = record(tiles, rng.choice([
                {**lay, "m": -1}, {**lay, "m": True}, {**lay, "m": None}, {**lay, "calls": []}, {**lay, "linears": []},
                {**lay, "linears": lin + [["x", 32]]}, {**lay, "calls": lay["calls"] + [[lin[0][0], -1]]},
                {**lay, "calls": [[5, 0]]}]))
        case "record-layout":
            desc["program_fn"] = record(tiles, {**lay, "calls": lay["calls"] + [["lm_head", 0]]})
        case "alt-layout":
            desc["program_fn"] = record(tiles, _alt_layout(rng, lay))
        case _:
            raise ValueError(name)
    return desc_bytes(desc).hex()


#: each way a window's committed descriptor is no H1 window's, or not the one its record says
DESC_NAMES = ["descriptor-form", "not-window", "template", "tile-ks", "body", "layout-form", "record-layout",
              "alt-layout"]


def units_of(lay: dict) -> int:
    """The window's units N under `lay`."""
    ss, why = Cr.strata(lay)
    assert why is None, why
    return sum(hi - lo for s in ss for lo, hi in s["units"])


def _k(rng: random.Random, lay: dict) -> int:
    """A committed K the window `lay` can be drawn at: one to its N units."""
    n = units_of(lay)
    return rng.choice([1, n, rng.randint(1, n), rng.randint(1, n)])


def _bad_k(rng: random.Random) -> object:
    return rng.choice([0, -1, True, "3", None, [2], Cr.MAX_K + 1])


def _draw_args(rng: random.Random) -> dict:
    stream = rng.choice([_hex(rng), "h1/w-3", ""])
    receipt = rng.choice([{"receipts": _hex(rng)}, {}, {"a": "b", "pass": "x y~"}])
    return {"coin": _hex(rng), "stream": stream, "registration": _hex(rng), "receipt": receipt}


def _break_layout(lay: dict, rng: random.Random, name: str) -> dict:
    lin, (r, c) = lay["linears"], lay["tile"]
    match name:
        case "layout-form":
            return rng.choice([None, [], {**lay, "extra": 1}, {k: v for k, v in lay.items() if k != "m"},
                               {**lay, "tile": [r]}, {**lay, "tile": [0, c]}, {**lay, "m": -1}, {**lay, "m": True},
                               {**lay, "calls": []}, {**lay, "linears": []}, {**lay, "linears": lin + [["x", 32]]},
                               {**lay, "calls": lay["calls"] + [[lin[0][0], -1]]}, {**lay, "calls": [[5, 0]]}])
        case "named-twice":
            return {**lay, "linears": lin + [[lin[0][0], lin[0][1], lin[0][2]]]}
        case "unknown-linear":
            return {**lay, "calls": lay["calls"] + [["lm_head", 0]]}
        case "uncalled":
            return {**lay, "linears": lin + [["lm_head", lin[0][1], c]]}
        case "slot":
            return {**lay, "m": rng.choice([r * 3, r // 2 if r > 1 else 0, r * 2 + 1 if r > 1 else 3])}
        case "kn":
            bad = rng.choice([[lin[0][0], 0, lin[0][2]], [lin[0][0], lin[0][1], lin[0][2] + 1 if c > 1 else 0],
                              [lin[0][0], lin[0][1], 0]])
            return {**lay, "linears": [bad] + lin[1:]}
        case "blocks":
            return {**lay, "calls": lay["calls"] + [[lin[0][0], 1 + max(b for _l, b in lay["calls"]) + 1]]}
    raise ValueError(name)


def drawn_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `drawn` case: a draw coin's arguments and a window's layout, then mutation `name`."""
    d = {**_draw_args(rng), "layout": layout(rng, big=name == "budget")}
    d["k"] = PLAN_K if name == "budget" else _k(rng, d["layout"])
    match name:
        case "honest" | "budget":
            pass
        case "k-form":
            d["k"] = _bad_k(rng)
        case "k-past-units":
            d["k"] = units_of(d["layout"]) + rng.randint(1, 5)
        case "coin-form":
            d["coin"] = rng.choice([None, d["coin"][:-2], d["coin"].upper(), 5])
        case "args-form":
            k, bad = rng.choice([("stream", None), ("stream", 5), ("registration", "zz"),
                                 ("registration", d["registration"].upper()), ("receipt", None), ("receipt", ["a"]),
                                 ("receipt", {"a": 1}), ("receipt", {"a": 'quote"d'}), ("receipt", {"\\": "x"}),
                                 ("receipt", {"a": "é"})])
            d[k] = bad
        case _:
            d["layout"], d["k"] = _break_layout(d["layout"], rng, name), 1
    return d


#: `circuit.plan.RECORD_K`, the K a period commits for ε = 0.1% at δ = 2⁻⁴⁰
PLAN_K = 27_713
DRAWN_NAMES = ["budget", "coin-form", "args-form", "layout-form", "named-twice", "unknown-linear", "uncalled", "slot", "kn",
               "blocks", "k-form", "k-past-units"]
DRAWN_KEYS = ("coin", "stream", "registration", "receipt", "layout", "k")


def _proofs(rng: random.Random, units: list[int]) -> list[dict]:
    proofs = [{"unit": u, "verdict": "accept"} for u in units]
    rng.shuffle(proofs)
    return proofs


def _mutate_proofs(proofs: list[dict], rng: random.Random, name: str, units: list[int]) -> list[dict]:
    j = rng.randrange(len(proofs))
    match name:
        case "missing":
            del proofs[j]
        case "failed":
            proofs[j]["verdict"] = "reject"
        case "twice":
            proofs.insert(rng.randrange(len(proofs) + 1), dict(proofs[j]))
        case "undrawn":
            picked = set(units)
            u = next(u for u in itertools.count(rng.randrange(1 << 20)) if u not in picked)
            proofs.insert(rng.randrange(len(proofs) + 1), {"unit": u, "verdict": "accept"})
        case "proofs-form":
            proofs = rng.choice([None, "x", proofs + [5], proofs[:j] + [{**proofs[j], "verdict": "maybe"}] + proofs[j + 1:],
                                 proofs[:j] + [{**proofs[j], "why": "x"}] + proofs[j + 1:],
                                 proofs[:j] + [{**proofs[j], "unit": -1}] + proofs[j + 1:]])
        case _:
            raise ValueError(name)
    return proofs


def _partial(rng: random.Random) -> tuple[dict, list[int]]:
    """A draw whose coin leaves some of the window's units undrawn, and its units."""
    while True:
        d = {**_draw_args(rng), "layout": layout(rng)}
        d["k"] = _k(rng, d["layout"])
        units, _ = Cr.drawn(*(d[k] for k in DRAWN_KEYS))
        if len(units) < sum(hi - lo for s in Cr.strata(d["layout"])[0] for lo, hi in s["units"]):
            return d, units


def outcome_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `outcome` case: a draw's arguments and an accepted proof of each unit its coin draws, then mutation `name`."""
    if name == "chosen":
        d, units = _partial(rng)
        other = {**d, "coin": _hex(rng)}
        chosen, _ = Cr.drawn(*(other[k] for k in DRAWN_KEYS))
        return {**d, "proofs": _proofs(rng, chosen if chosen != units else units[1:] + [max(units) + 1])}
    d = {**_draw_args(rng), "layout": layout(rng)}
    d["k"] = _k(rng, d["layout"])
    units, _ = Cr.drawn(*(d[k] for k in DRAWN_KEYS))
    proofs = _proofs(rng, units)
    return {**d, "proofs": proofs if name == "honest" else _mutate_proofs(proofs, rng, name, units)}


OUTCOME_NAMES = ["missing", "failed", "twice", "undrawn", "chosen", "proofs-form"]
OUTCOME_KEYS = (*DRAWN_KEYS, "proofs")


def _served_units(d: dict, stream: str | None = None, k: int | None = None) -> list[int]:
    """The units `d`'s draw coin draws at its K (or `k`) under the served window's arguments (`credit.window`'s), or
    under `stream`."""
    coin = d["draw"]["coin"]["coin"]
    got, why = Cr.drawn(coin, d["window"]["root"] if stream is None else stream, d["draw"]["root"],
                        {"receipts": d["draw"]["receipts"]}, d["layout"], d["k"] if k is None else k)
    assert why is None, why
    return got


def honest_window(rng: random.Random, name: str = "honest", sid: str = "s-1", leaf: int = rc.RECORD_LEAF,
                  lay: dict | None = None, desc: str | None = None) -> tuple[dict, rc.Session]:
    """An honest served window (`honest_credit`) of the circuit `desc` (`descriptor` of its layout `lay`, a random one
    when None), a K it can be drawn at, one call per manifest row, and an accepted proof of each unit its draw coin
    draws at that K. ``layout`` is the layout, which the verifier reads from the descriptor."""
    lay = layout(rng) if lay is None else lay
    d, s = honest_credit(rng, name if name in [*CREDIT_NAMES, "never-received"] else "honest", rows=len(lay["calls"]),
                         sid=sid, leaf=leaf)
    d["layout"], d["k"] = lay, _k(rng, lay)
    d["descriptor"] = descriptor(rng, lay) if desc is None else desc
    d["proofs"] = _proofs(rng, _served_units(d))
    return d, s


def window_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `window` case: an honest window, then mutation `name`."""
    credit = name.removeprefix("credit-")
    d, s = honest_window(rng, credit)
    match name:
        case "honest":
            pass
        case "manifest-count":
            lin = d["layout"]["linears"][0][0]
            d["layout"]["calls"].insert(rng.randrange(len(d["layout"]["calls"]) + 1), [lin, 0])
            d["descriptor"] = descriptor(rng, d["layout"])
        case "chosen":
            d, s = honest_window(rng)
            while True:
                chosen = _served_units(d, stream=_hex(rng))
                if chosen != sorted(p["unit"] for p in d["proofs"]):
                    break
                d, s = honest_window(rng)
            d["proofs"] = _proofs(rng, chosen)
        case "failed" | "missing":
            d["proofs"] = _mutate_proofs(d["proofs"], rng, name, [])
        case "descriptor-hex":
            h = d["descriptor"]
            d["descriptor"] = rng.choice([h.upper(), h[:-1], "", "zz", 5, None])
        case _ if name in DESC_NAMES:
            d["descriptor"] = _break_desc(rng, d["layout"], name)
        case "k-form":
            d["k"] = _bad_k(rng)
        case "k-past-units":
            d["k"] = units_of(d["layout"]) + rng.randint(1, 5)
        case "never-received":
            pass
        case "round-coin":
            coin = d["rounds"][rng.randrange(len(d["rounds"]))]["coin"]
            coin["after"] = rc._flip(coin["after"])
        case "a-untied":
            e = rng.choice([e for e in d["entries"] if Cr.is_a(e["value"])])
            e["root"] = _tree(rng, e["leaves"])[1][-1][0].hex()
        case "no-a":
            d["entries"] = [e for e in d["entries"] if not Cr.is_a(e["value"])]
        case _ if credit in CREDIT_NAMES:
            _mutate_credit(d, s, credit, rng)
        case _:
            raise ValueError(name)
    if "ledger" not in d:
        d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s), rc._key(0))
    return {k: d[k] for k in WINDOW_KEYS}


WINDOW_NAMES = ["manifest-count", "chosen", "failed", "missing", "descriptor-hex", *DESC_NAMES,
                "credit-changed", "credit-no-draw", "credit-forged-ledger", "k-form", "k-past-units", "never-received",
                "round-coin", "a-untied", "no-a"]
WINDOW_KEYS = ("key", "ledger", *Cr.WINDOW_KEYS, "k")
#: `credit.window`'s arguments, in order
WINDOW_ARGS = ("key", "ledger", "window", "session", "salt", "calls", "served", "rounds", "draw", "descriptor",
               "entries", "proofs", "k")


def window_args(d: dict) -> tuple:
    """`credit.window`'s arguments for case `d`, the challenger's signatures checked with `lms_verify`."""
    return (*(d[k] for k in WINDOW_ARGS), rc.lms_verify)


def _evidence(d: dict) -> dict:
    return {k: d[k] for k in Cr.WINDOW_KEYS}


def period_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `period` case: one to three honest windows in one period of the auditor's ledger, each proving the circuit
    whose SHA-512 the period's open committed (one window layout's descriptor, `descriptor`), presented in its close's
    order, and a window of no period opened after the close; then mutation `name`."""
    n = rng.randint(2 if name == "order" else 1, 3)
    pname = rng.choice(rc.PERIODS)
    lay = layout(rng)
    desc = descriptor(rng, lay) if name != "bad-circuit" else _break_desc(rng, lay, rng.choice(DESC_NAMES))
    bad = rng.randrange(n) if name == "never-received" else None
    wins = [honest_window(rng, "never-received" if i == bad else "honest", sid=f"s-{i + 1}", leaf=4 * i + 2, lay=lay,
                          desc=desc) for i in range(n)]
    foreign = honest_window(rng, sid="o-1", leaf=4 * n + 3, lay=lay, desc=desc)
    pub = wins[0][1].pub
    descriptor_ = bytes.fromhex(desc)
    least = units_of(lay)
    big_k = rng.choice([1, least, rng.randint(1, least)])
    for d, _s in wins + [foreign]:
        d["proofs"] = _proofs(rng, _served_units(d, k=big_k))
    ends = [{} for _ in wins]
    if name == "incomplete":
        ends[rng.randrange(n)] = {"complete": False}
    committed = {"cap": 4, "circuit": hashlib.sha512(descriptor_).hexdigest(), Cr.K_FIELD: big_k}
    match name:
        case "no-circuit":
            committed = {Cr.K_FIELD: big_k}
        case "no-k":
            del committed[Cr.K_FIELD]
        case "k-form":
            committed[Cr.K_FIELD] = _bad_k(rng)
        case "k-past-units":
            committed[Cr.K_FIELD] = past = min(units_of(d["layout"]) for d, _s in wins) + rng.randint(1, 5)
            for d, _s in wins:
                if units_of(d["layout"]) >= past:
                    d["proofs"] = _proofs(rng, _served_units(d, k=past))
    events = [("period-open", None, {"period": pname, "lane": rc.LANE, "by": rc.LANE, **committed, "pinned": "ab" * 20,
                                     "key_sha512": hashlib.sha512(pub).hexdigest()})]
    for (d, s), end in zip(wins, ends):
        events += [("open", s.sid, rc._open(rng, d["window"], pub)), ("end", s.sid, rc._end(s, **end))]
    outcomes = [{"root": d["window"]["root"], "session": s.sid, "complete": end.get("complete", True)}
                for (d, s), end in zip(wins, ends)]
    events.append(("period-close", None, {"period": pname, "lane": rc.LANE, "by": rc.LANE, "windows": outcomes,
                                          "complete": all(w["complete"] for w in outcomes)}))
    fd, fs = foreign
    events += [("open", fs.sid, rc._open(rng, fd["window"], pub)), ("end", fs.sid, rc._end(fs))]
    leaves = [0] + [x for i in range(n) for x in (4 * i + 1, 4 * i + 4)] + [4 * n + 1, 4 * n + 2, 4 * n + 5]
    key = rc._key(1) if name == "forged-ledger" else rc._key(0)
    ledger = rc.ledger_lines(events if name != "no-close" else [e for e in events if e[0] != "period-close"], key,
                             leaves if name != "no-close" else [q for q, e in zip(leaves, events) if e[0] != "period-close"])
    windows = [_evidence(d) for d, _s in wins]
    out = {"key": wins[0][0]["key"], "ledger": ledger, "name": pname}
    i = rng.randrange(n)
    match name:
        case ("honest" | "incomplete" | "no-close" | "forged-ledger" | "no-circuit" | "no-k" | "k-form" | "k-past-units"
              | "bad-circuit" | "never-received"):
            pass
        case "other-k":
            d = wins[i][0]
            n = units_of(d["layout"])
            mine = _served_units(d, k=big_k)
            other = rng.choice([k for k in (big_k - 1, big_k + 1, rng.randint(1, n)) if 1 <= k <= n and k != big_k
                                and _served_units(d, k=k) != mine] or [None])
            if other is None:
                windows[i]["proofs"] = windows[i]["proofs"][1:]
            else:
                windows[i]["proofs"] = _proofs(rng, _served_units(d, k=other))
        case "missing":
            del windows[i]
        case "order":
            i, j = rng.sample(range(n), 2)
            windows[i], windows[j] = windows[j], windows[i]
        case "extra":
            windows.append(_evidence(fd) if rng.random() < 0.5 else dict(windows[i]))
        case "foreign":
            windows.insert(rng.randrange(n), _evidence(fd))
        case "other-circuit":
            windows[i]["descriptor"] = descriptor(rng, lay, prefix=f"other-{rng.randrange(1 << 20)}/")
        case "alt-layout":
            windows[i]["descriptor"] = descriptor(rng, _alt_layout(rng, lay))
        case "descriptor-form":
            h = windows[i]["descriptor"]
            windows[i]["descriptor"] = rng.choice([h.upper(), h[:-1], "", "zz", 5, None, list(descriptor_)])
        case "unopened":
            out["name"] = next(p for p in rc.PERIODS if p != pname)
        case "name":
            out["name"] = rng.choice([None, 5, [pname]])
        case "window-refused":
            windows[i]["proofs"] = _mutate_proofs(windows[i]["proofs"], rng, rng.choice(["failed", "missing"]), [])
        case "windows-form":
            drop = rng.choice(["proofs", "descriptor"])
            windows = rng.choice([None, {"0": windows[0]}, windows[:i] + [5] + windows[i + 1:],
                                  windows[:i] + [{k: v for k, v in windows[i].items() if k != drop}]
                                  + windows[i + 1:]])
        case "key-form":
            out["key"] = rng.choice([None, {**out["key"], "public": "00"}])
        case _:
            raise ValueError(name)
    return {**out, "windows": windows}


PERIOD_NAMES = ["no-close", "incomplete", "missing", "order", "extra", "foreign", "unopened", "name", "window-refused",
                "windows-form", "forged-ledger", "key-form", "other-circuit", "no-circuit", "descriptor-form", "no-k",
                "k-form", "k-past-units", "other-k", "alt-layout", "bad-circuit", "never-received"]
PERIOD_KEYS = ("key", "ledger", "name", "windows")


def _tree(rng: random.Random, count: int) -> tuple[str, list[tuple[bytes, ...]]]:
    """A frame-v3 SHA-512 tree over `count` leaf digests: its domain id and levels."""
    dom = CommitmentDomain(rng.randbytes(64), rng.randrange(-2, 4), RangeIndexedDomain(0, count), "sha512")
    return dom.domain_id.hex(), tree_levels(dom, [rng.randbytes(64) for _ in range(count)])


def _log(steps: int, values: dict, inners: list[str]) -> list[dict]:
    """The round log: each round's root over its inner root and its `cover` segments of every A value."""
    out = []
    for r in range(steps):
        segs = {}
        for v, (count, levels) in values.items():
            per = count // steps
            segs[v] = [[lv, i, levels[lv][i].hex()] for lv, i in Cr.cover(r * per, (r + 1) * per)]
        out.append({"inner": inners[r], "a": segs})
    return _reroot(out)


def _reroot(log: list[dict]) -> list[dict]:
    for r, x in enumerate(log):
        x["root"] = Cr.round_root(r, bytes.fromhex(x["inner"]), {v: [(lv, i, bytes.fromhex(h)) for lv, i, h in s]
                                                                  for v, s in x["a"].items()}).hex()
    return log


def a_link_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `a_link` case: a window's registered entries and round log, then mutation `name`."""
    steps = rng.choice([2, 3, 4] if name in ("blocks", "swapped") else [1, 2, 3, 4])
    window = f"h1/w-{rng.randrange(100)}"
    values, entries = {}, []
    for k in rng.sample([64, 4096, 14336], rng.randint(1, 2)):
        per = rng.choice([2, 3, 4, 5] if name == "swapped" else [1, 2, 3, 4, 5, 8])
        domain, levels = _tree(rng, steps * per)
        v = f"{window}/a/{k}"
        values[v] = (steps * per, levels)
        entries.append({"value": v, "root": levels[-1][0].hex(), "leaves": steps * per, "words": 96, "domain": domain})
    entries += [{"value": f"{window}/z/{rng.choice([64, 4096])}", "root": _hex(rng), "leaves": 3, "domain": _hex(rng)},
                {"value": f"{window}/a/x", "root": _hex(rng)}][:rng.randint(0, 2)]
    rng.shuffle(entries)
    log = _log(steps, values, [_hex(rng) for _ in range(steps)])
    a = [e for e in entries if Cr.is_a(e["value"])]
    e = rng.choice(a)
    v = e["value"]
    r = rng.randrange(steps)
    segs = log[r]["a"][v]
    match name:
        case "honest":
            pass
        case "untied":
            e["root"] = _tree(rng, e["leaves"])[1][-1][0].hex()
        case "late-rows":
            count, levels = values[v]
            domain, other = _tree(rng, count)
            e["root"], e["domain"] = other[-1][0].hex(), domain
        case "rewritten":
            segs[rng.randrange(len(segs))][2] = _hex(rng)
            _reroot(log)
        case "round-root":
            segs[rng.randrange(len(segs))][2] = _hex(rng)
        case "gap":
            del segs[rng.randrange(len(segs))]
            _reroot(log)
        case "extra":
            segs.append([0, values[v][0], _hex(rng)])
            _reroot(log)
        case "swapped":
            if len(segs) > 1:
                segs[0], segs[1] = segs[1], segs[0]
            else:
                segs[0] = [segs[0][0], segs[0][1] + 1, segs[0][2]]
            _reroot(log)
        case "misaligned":
            lv, i, h = segs[0]
            segs[0] = [lv + 1, i // 2, h] if lv + 1 < Cr.LEVELS else [lv, i + 1, h]
            _reroot(log)
        case "unknown":
            log[r]["a"][f"{window}/a/{rng.choice([1, 2, 3])}"] = [[0, 0, _hex(rng)]]
            _reroot(log)
        case "no-a":
            entries = [x for x in entries if not Cr.is_a(x["value"])]
        case "twice":
            entries.append(dict(e))
        case "entry-form":
            k, bad = rng.choice([("root", None), ("root", "zz"), ("domain", None), ("domain", e["domain"].upper()),
                                 ("leaves", 0), ("leaves", True), ("leaves", Cr.MAX_ROWS + 1), ("leaves", "4")])
            e[k] = bad
        case "blocks":
            e["leaves"] += 1 if (e["leaves"] + 1) % steps else 2 if (e["leaves"] + 2) % steps else 3
        case "log-form":
            log = rng.choice([[], None, "x"])
        case "round-form":
            log[r] = rng.choice([5, {**log[r], "inner": "zz"}, {k: x for k, x in log[r].items() if k != "root"}])
        case "segments-form":
            bad = rng.choice([None, [[Cr.LEVELS, 0, _hex(rng)]], [[0, -1, _hex(rng)]], [[0, 0, "zz"]], [[0, 0]],
                              [[True, 0, _hex(rng)]]])
            log[r]["a"] = rng.choice([{**log[r]["a"], v: bad}, [segs]])
        case "entries-form":
            entries = rng.choice([None, entries + [5], {"0": entries}])
        case _:
            raise ValueError(name)
    return {"rounds": log, "entries": entries}


A_LINK_NAMES = ["untied", "late-rows", "rewritten", "round-root", "gap", "extra", "swapped", "misaligned", "unknown",
                "no-a", "twice", "entry-form", "blocks", "log-form", "round-form", "segments-form", "entries-form"]


def _b_entry(program: str, v: str, k: int, n: int, rng: random.Random) -> dict:
    return {"value": v, "root": _hex(rng), "leaves": n, "words": Cr.b_words(k),
            "domain": G.domain(program, v, Cr.b_words(k), n).domain_id.hex()}


def weights_instance(rng: random.Random, name: str = "honest") -> dict:
    """One `weights` case: a window's registered weights under its open root, then mutation `name`."""
    lay = layout(rng)
    program, prefix = _hex(rng), rng.choice(["", "h1/w-3/"])
    rows: dict[int, int] = {}
    for _name, k, n in lay["linears"]:
        rows[k] = rows.get(k, 0) + n
    entries = [_b_entry(program, f"{prefix}b/{k}", k, n, rng) for k, n in sorted(rows.items())]
    b = list(entries)
    lone = ["a/32@0", "zy@0", "b/x", f"b/0{min(rows)}", "b/" + "0" * 5000 + "1"]
    entries += [{"value": f"{prefix}{v}", "root": _hex(rng)}
                for v in [*(f"a/{k}" for k in rows), *lone][:rng.randint(0, len(rows) + len(lone))]]
    rng.shuffle(entries)
    requests, session = _hex(rng), rng.choice(["s-1", "a.b_c:d-9", "7"])
    roots = {k: bytes.fromhex(e["root"]) for k, e in zip(sorted(rows), b)}
    window = {"root": PS.window_open_root(Cr.weights_root(roots), bytes.fromhex(requests), session).hex()}
    e = rng.choice(b)
    k = next(kk for kk in rows if e["value"] == f"{prefix}b/{kk}")
    match name:
        case "honest":
            pass
        case "other-root":
            e["root"] = _hex(rng)
        case "other-requests":
            requests = _hex(rng)
        case "other-session":
            session = session + "x"
        case "other-window":
            window = {"root": _hex(rng)}
        case "missing":
            entries.remove(e)
        case "twice":
            entries.append({**e, "root": _hex(rng)})
        case "foreign":
            entries.append({"value": f"{prefix}b/{rng.choice([1, 2, 31, 128])}", "root": _hex(rng)})
        case "other-prefix":
            entries.append({**e, "value": rng.choice(["x/", "h1/w-4/", "h1/w-3/h1/w-3/"]) + e["value"]})
        case "other-leaves":
            e["leaves"] = rng.choice([e["leaves"] + 1, e["leaves"] - 1, float(e["leaves"]), True, None])
        case "other-words":
            e["words"] = rng.choice([e["words"] + 1, k // 2, k, str(e["words"])])
        case "other-domain":
            e["domain"] = rng.choice([_hex(rng)[:64], G.domain(_hex(rng), e["value"], e["words"], e["leaves"]).domain_id.hex(),
                                      G.domain(program, e["value"], e["words"], e["leaves"], window=_hex(rng)).domain_id.hex(),
                                      G.domain(program, e["value"], e["words"], e["leaves"] + 1).domain_id.hex(), None])
        case "entry-form":
            got = rng.choice(["extra", "drop", "bits"])
            if got == "extra":
                e["row_schema"] = "x"
            elif got == "bits":
                e["word_bits"] = 16
            else:
                del e[rng.choice(["leaves", "words", "domain"])]
        case "root-form":
            e["root"] = rng.choice([None, "zz", e["root"].upper(), 5])
        case "window-form":
            window = rng.choice([None, {}, {"root": "zz"}, [window]])
        case "requests-form":
            requests = rng.choice([None, "zz", requests.upper(), 5])
        case "session-form":
            session = rng.choice([None, "", "-s", "s/1", "x" * 65, 5, "\u015b"])
        case "circuit-form":
            if rng.random() < 0.5:
                program = rng.choice([None, "zz", program.upper(), program[:64], 5])
            else:
                prefix = rng.choice([None, 5, "a\"b/", "a\\b/", "\u015b/", "tab\t/"])
        case "layout-form":
            lay = rng.choice([None, {**lay, "m": 0}, {k: v for k, v in lay.items() if k != "tile"}])
        case "entries-form":
            entries = rng.choice([None, entries + [5], {"0": entries}])
        case _:
            raise ValueError(name)
    return {"window": window, "layout": lay, "entries": entries, "requests": requests, "session": session,
            "program": program, "prefix": prefix}


WEIGHTS_NAMES = ["other-root", "other-requests", "other-session", "other-window", "missing", "twice", "foreign",
                 "other-prefix", "other-leaves", "other-words", "other-domain", "entry-form", "root-form", "window-form",
                 "requests-form", "session-form", "circuit-form", "layout-form", "entries-form"]
WEIGHTS_KEYS = ("window", "layout", "entries", "requests", "session", "program", "prefix")
