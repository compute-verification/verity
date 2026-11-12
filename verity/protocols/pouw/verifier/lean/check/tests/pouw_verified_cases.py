"""The statements difftest's cases: the arguments of `credit._verified` (the Lean `PouwCheck.Verified`), a served execution phase's
statements, its credit period's committed circuit files and the verifier's answers.

A `verified` case is a served execution phase (`pouw_credit_cases.layout`, its descriptor under a prefix or none, its calls over
one to three rounds), one to six sampled proof blocks, the entries a statement of each sampled call reads (its round's A table of
its K, its z table and the weights' ``b/<K>``, `credit.registered_file`'s, each registered once), one statement a sampled
call in a shuffled order (its circuit and public files' paths and one to three sessions' directories), each K's
committed circuit file, the execution phase's open root, rounds and Program salt, and `runs`: null, the first of the Lean's two
calls (`credit._lean_verified`), or the answers the served path gives: one a statement `credit._asks` names, in its
order, until one the Lean refuses at, each session accepted with the units of its call split among them and its
registered finding the ``--registered`` file's ports at their values and roots.

Its mutations before any run (`PRE_NAMES`, with `runs` null or answered) break the circuit files' form, the statements'
form or name a call past the execution phase's, give a call two statements or one to a call with no sampled proof block, drop a K's
circuit file, register the execution phase's ok rows (``ok-registered``) or none of a value a statement reads (``unregistered``),
and give a descriptor that is no H1 execution phase's. Its mutations of the answers (`RUN_NAMES`) make one statement's verifier
fail to run (``error``) or refuse it (``refused``), reject one session (``rejected``), open a public table or the ok
rows at another port, value or root, open no ok row (``ok-none``), and prove a unit of another call (``unit-other``).

A `lanes` case (`lanes_instance`, `LANES_NAMES`) is a 32 × 32 execution phase with one K in the lanes form (``lane<K>`` and
``combine<K>``, its leaf Programs `leaf_desc`'s, its zy unit's TILE the one they are over) and perhaps one in the tile
form, one or two sampled leaves at the lanes-form K and perhaps tile-form units, and the lanes evidence: each sampled leaf's
lanes and combine statements among the calls' in a shuffled order, and the produced record of their partials under the
execution phase's sample root. Its answers accept each leaf session with every port of its ``--registered`` file, and the rows it
opened of the partials and the known rows (``openings``). Its mutations: a K outside `LeafLanesSound`'s reach, no lanes
evidence, a leaf's statement dropped, a leaf Program other than the committed one or not the execution phase's own, a statement
twice, of an unsampled unit or none of the leaf's, the produced record's P- codes, a table the leaf reads unregistered,
the sample malformed, the evidence's or the circuit files' lanes form broken; then a leaf session's partials or known row
opened at another opening or reported without openings (``P-shared``), its ports at another root, rejected, proving
other units, a run that fails or refuses, and every leaf run reporting no session (``P-opened``).

`tests/test_verified_difftest.py` samples them for the harness, replaying `runs` as the verifier (`replay`).
"""
from __future__ import annotations

import bisect
import dataclasses
import hashlib
import itertools
import random

import pouw_credit_cases as C
from verity.protocols.pouw.verifier import credit as Cr
from verity.protocols.pouw.verifier import leaf_lanes as LL
from verity.protocols.spot.verifier.one_stage import produced as PR
from verity.protocols.spot.verifier.one_stage import proofs as PF

PRE_NAMES = ["files-form", "statements-form", "call-past", "twice", "undrawn", "no-file", "ok-registered",
             "unregistered", "layout"]
RUN_NAMES = ["error", "refused", "rejected", "pub-other", "ok-other", "ok-none", "unit-other"]
LANES_PRE = ["lanes-shape", "lanes-missing", "lanes-no-combine", "lanes-leaf-digest", "lanes-leaf-other",
             "lanes-twice", "lanes-undrawn", "lanes-statement", "lanes-record", "lanes-unregistered", "lanes-draw",
             "lanes-form", "lanes-files"]
LANES_RUN = ["lanes-partial", "lanes-known", "lanes-openings-none", "lanes-root", "lanes-rejected", "lanes-units",
             "lanes-error", "lanes-refused", "lanes-empty"]
LANES_NAMES = ["lanes-honest", *LANES_PRE, *LANES_RUN]
NAMES = PRE_NAMES + RUN_NAMES + LANES_NAMES


def _starts(lay: dict) -> tuple[list[int], list[int]]:
    sizes = [size for size, _work in Cr.call_work(lay)]
    return sizes, list(itertools.accumulate(sizes, initial=0))[:-1]


def _reads(lay: dict, prefix: str, steps: int, c: int) -> list[str]:
    """The values call `c`'s statement reads that the execution phase registers (`credit.registered_file`'s)."""
    k = {lin: k for lin, k, _n in lay["linears"]}[lay["calls"][c][0]]
    return [f"{prefix}a/{k}@{Cr.step_row(lay, steps, c)[0]}", f"{prefix}zy@{c}", f"{prefix}b/{k}"]


def _finding(rng: random.Random, session: str, ask: dict, units: list[int], program: str) -> dict:
    """An accepted session's finding as `credit._finding_json` writes it: its registered ports the ``--registered``
    file's, at their values and roots, some of the A and B ports among them."""
    ports = ask["registered"]["ports"]
    opened = [p for p in ports if p in (Cr.PUB_PORT, Cr.OK_PORT) or rng.random() < 0.5]
    return {"session": session, "accepted": True, "why": "accepted", "program": program,
            "registered": {p: {"value": ports[p]["value"], "root": ports[p]["root"]} for p in opened}, "units": units}


def _answers(rng: random.Random, case: dict, asks: list[dict], name: str, starts: list[int], sizes: list[int]) -> list:
    """The served path's answers to `asks` (`credit._answer`'s form), with `name`'s mutation at one of them, cut after
    the first the Lean refuses at, as `credit._lean_verified` stops."""
    program = hashlib.sha512(bytes.fromhex(case["descriptor"])).hexdigest()
    at = rng.randrange(len(asks)) if asks else None
    out = []
    for j, a in enumerate(asks):
        s = case["statements"][a["statement"]]
        c = s["call"]
        mine = [u for u in case["units"] if starts[c] <= u < starts[c] + sizes[c]]
        parts = [mine[i::len(s["sessions"])] for i in range(len(s["sessions"]))]
        fs = [_finding(rng, x, a, part, program) for x, part in zip(s["sessions"], parts)]
        ans = {"findings": fs}
        if j == at:
            f = rng.choice(fs)
            ports = a["registered"]["ports"]
            pub, ok = ports[Cr.PUB_PORT], ports[Cr.OK_PORT]
            match name:
                case "error":
                    ans = {"error": rng.choice(["flock-verify ended on a signal", "no such file: public.bin"])}
                case "refused":
                    ans = {"refused": [rng.choice(["proof", "circuit", "registration"]),
                                       "the statement's public file is not its circuit's"]}
                case "rejected":
                    f.update(accepted=False, why=rng.choice(["sumcheck round 3", "opening root"]), units=[])
                case "pub-other":
                    other = rng.choice([{"value": pub["value"], "root": C._hex(rng)},
                                        {"value": f"{pub['value'].rsplit('@', 1)[0]}@{c + 1}", "root": pub["root"]}])
                    if rng.random() < 0.5:
                        f["registered"][Cr.PUB_PORT] = other
                    else:
                        f["registered"]["o9"] = other
                case "ok-other":
                    if rng.random() < 0.5:
                        f["registered"][Cr.OK_PORT] = {"value": ok["value"], "root": C._hex(rng)}
                    else:
                        f["registered"]["o9"] = {"value": ok["value"], "root": ok["root"]}
                case "ok-none":
                    del f["registered"][Cr.OK_PORT]
                case "unit-other":
                    f["units"] = sorted({*f["units"], rng.choice(
                        [u for u in range(sum(sizes)) if not starts[c] <= u < starts[c] + sizes[c]] or [sum(sizes)])})
        out.append(ans)
        if "findings" not in ans or not all(f["accepted"] for f in ans["findings"]):
            break
    return out


def verified_instance(rng: random.Random, name: str = "honest") -> dict:
    if name in LANES_NAMES:
        return lanes_instance(rng, name)
    lay = C.layout(rng)
    prefix = rng.choice(["", f"h1/w-{rng.randrange(100)}/"])
    steps = rng.choice([s for s in (1, 2, 3) if len(lay["calls"]) % s == 0])
    sizes, starts = _starts(lay)
    units = sorted(rng.sample(range(sum(sizes)), rng.randint(1, min(6, sum(sizes)))))
    drawn = sorted({bisect.bisect_right(starts, u) - 1 for u in units})
    reads = {v for c in drawn for v in _reads(lay, prefix, steps, c)}
    entries = [{"value": v, "root": C._hex(rng)} for v in sorted(reads)]
    rng.shuffle(entries)
    statements = [{"call": c, "circuit": f"/w/c{c}/circuit.bin", "public": f"/w/c{c}/public.bin",
                   "sessions": [f"/w/c{c}/s{j}" for j in range(rng.randint(1, 3))]} for c in drawn]
    rng.shuffle(statements)
    files = {f"k{k}": C._hex(rng) for k in sorted({k for _lin, k, _n in lay["linears"]})}
    case = {"statements": statements, "files": files, "descriptor": C.descriptor(rng, lay, prefix), "entries": entries,
            "window": C._hex(rng), "units": units, "steps": steps, "salt": C._hex(rng),
            "draw": {"root": C._hex(rng), "coin": C._draw_coin(rng)}, "runs": None}
    s = rng.choice(statements)
    match name:
        case "honest" | "error" | "refused" | "rejected" | "pub-other" | "ok-other" | "ok-none" | "unit-other":
            pass
        case "files-form":
            case["files"] = rng.choice([[], None, {**files, f"k{lay['linears'][0][1]}": C._hex(rng).upper()},
                                        {**files, "k32": C._hex(rng)[:-2]}, {**files, "x": 7}])
        case "statements-form":
            bad = rng.choice([{**s, "sessions": []}, {**s, "call": rng.choice([-1, "1", True, None])},
                              {k: v for k, v in s.items() if k != "public"}, {**s, "extra": 1},
                              {**s, "sessions": ["/w/s", 3]}, {**s, "circuit": None}])
            case["statements"] = rng.choice([[bad], [*statements, bad], {"0": s}])
        case "call-past":
            statements.append({**s, "call": len(lay["calls"]) + rng.randrange(3)})
        case "twice":
            statements.insert(rng.randrange(len(statements) + 1), {**s, "sessions": ["/w/again"]})
        case "undrawn":
            idle = [c for c in range(len(lay["calls"])) if c not in drawn]
            if idle:
                statements.insert(rng.randrange(len(statements) + 1), {**s, "call": rng.choice(idle)})
            else:
                statements.insert(rng.randrange(len(statements) + 1), {**s})
        case "no-file":
            k = {lin: k for lin, k, _n in lay["linears"]}[lay["calls"][s["call"]][0]]
            del files[f"k{k}"]
        case "ok-registered":
            entries.insert(rng.randrange(len(entries) + 1), {"value": f"{prefix}{Cr.OK_VALUE}", "root": C._hex(rng)})
        case "unregistered":
            v = rng.choice(_reads(lay, prefix, steps, s["call"]))
            case["entries"] = [e for e in entries if e["value"] != v] if rng.random() < 0.5 else [
                *entries, {"value": v, "root": C._hex(rng)}]
        case "layout":
            case["descriptor"] = rng.choice([b"{}", b"[1]", C.desc_bytes({"schema": Cr.SCHEMA})]).hex()
        case _:
            raise ValueError(name)
    if name in RUN_NAMES or rng.random() < 0.6:
        raw = bytes.fromhex(case["descriptor"])
        layout, why = Cr.layout_of(raw)
        asks = [] if why is not None else Cr._asks(case["statements"], case["files"], raw, layout, case["entries"],
                                                   case["window"], case["units"], steps, case["salt"], case["draw"])[0]
        case["runs"] = _answers(rng, case, asks, name, starts, sizes)
    return case


# -- the lanes form -------------------------------------------------------------------------------------------------

def _tile_binding(k: int) -> dict:
    """The TILE a lanes-form K's zy unit is over, as a Definition argument encodes: the H1 tile's id."""
    return {"fn": f"PouwH1Tile_v1{{K={k},R=32,C=32}}"}


def zy_unit(k: int) -> str:
    """K's zy unit id, `h1zy.PouwH1TileZy_v1{TILE=…}`, the unit whose TILE `credit.leaf_unit_id` reads."""
    return f"{Cr.ZY_UNIT}{{TILE={C._canon(_tile_binding(k))}}}"


def leaf_desc(kind: str, k: int, at: int | None = None, unit_kind: str | None = None) -> bytes:
    """`h1window.leaf_program`'s descriptor as far as `credit.leaf_of` reads it: the one-call Program of `LEAF_FN`
    [`kind`] at K (`at`, else `k`) over the unit `LEAF_UNIT` [`unit_kind`, else `kind`] bound to K's TILE."""
    tile, uk = _tile_binding(k), unit_kind or kind
    unit = f"{Cr.LEAF_UNIT[uk]}{{TILE={C._canon(tile)}}}"
    fn = f"{Cr.LEAF_FN[kind]}{{UNIT={C._canon({'fn': unit})},K={k if at is None else at}}}"
    root = f"Root[{fn}]_v1"
    defs = {"Input16_v1": {"form": "primitive", "id": "Input16_v1", "params": [], "ret": ["Value", 16]},
            unit: {"form": "composite", "id": Cr.LEAF_UNIT[uk], "bindings": {"TILE": tile},
                   "body": {"params": [], "ret_type": ["Value", 1], "nodes": []}},
            root: {"form": "composite", "id": root, "bindings": {},
                   "body": {"params": [], "ret_type": ["Value", 1],
                            "nodes": [{"form": "batch", "fn": 0, "n": 3}, {"form": "call", "fn": 1}]}}}
    return C.desc_bytes({"callees": ["Input16_v1", unit], "definitions": defs, "program_fn": fn, "root": root,
                         "schema": Cr.SCHEMA})


def _bc(rng: random.Random) -> str:
    return C._hex(rng) + C._hex(rng)


def lanes_instance(rng: random.Random, name: str = "lanes-honest") -> dict:
    """A `lanes` case: `verified_instance`'s arguments over an execution phase with a lanes-form K, `name`'s mutation applied
    (`LANES_NAMES`)."""
    shape_off = name == "lanes-shape" and rng.random() < 0.5
    r, c = (16, 16) if shape_off else (32, 32)
    lk = rng.choice([32, 64, 352, 6144]) if name == "lanes-shape" and not shape_off else rng.choice([384, 448, 4096])
    tk = rng.choice([None, None, 64, 1024])
    ks = [lk] + ([tk] if tk is not None and tk != lk else [])
    names = rng.sample(C.LINEARS, rng.randint(len(ks), 3))
    linears = [[x, ks[i] if i < len(ks) else rng.choice(ks), c * rng.randint(1, 2)] for i, x in enumerate(names)]
    calls = [[x, step] for step in range(rng.randint(1, 2)) for x in names]
    rng.shuffle(calls)
    lay = {"calls": calls, "linears": linears, "m": r << rng.randint(0, 1), "tile": [r, c]}
    prefix = rng.choice(["", f"h1/w-{rng.randrange(100)}/"])
    steps = rng.choice([s for s in (1, 2) if len(calls) % s == 0])
    descriptor = C.descriptor(rng, lay, prefix, unit_fns={lk: zy_unit(lk)})
    raw = bytes.fromhex(descriptor)
    layout, why = Cr.layout_of(raw)
    assert why is None, why
    shape, program = Cr.shape_of(raw), hashlib.sha512(raw).hexdigest()
    sizes, starts = _starts(lay)
    kof = {x: k for x, k, _n in lay["linears"]}
    call_k = [kof[lin] for lin, _blk in lay["calls"]]
    at_lk = [u for u in range(sum(sizes)) if call_k[bisect.bisect_right(starts, u) - 1] == lk]
    at_tk = [u for u in range(sum(sizes)) if u not in at_lk]
    leaves = sorted(rng.sample(at_lk, min(len(at_lk), rng.randint(1, 2))))
    tiled = sorted(rng.sample(at_tk, min(len(at_tk), rng.randint(0, 2))))
    units = sorted(leaves + tiled)
    drawn = sorted({bisect.bisect_right(starts, u) - 1 for u in tiled})
    reads = {v for call in drawn for v in _reads(lay, prefix, steps, call)}
    for u in leaves:
        reads |= {table for v, (table, _row) in Cr.leaf_place(layout, shape, u, steps).items() if v != LL.PARTIALS}
    entries = [{"value": v, "root": C._hex(rng)} for v in sorted(reads)]
    rng.shuffle(entries)
    draw = {"root": C._hex(rng), "coin": C._draw_coin(rng)}
    lanes = (r // LL.LANE_ROWS) * (c // LL.LANE_COLS)
    produced, openings = [], {}
    for u in leaves:
        v = LL.served(prefix, LL.PARTIALS, u)
        produced.append({"value": v, "root": C._hex(rng),
                         **{x: y for x, y in PR.entry(program, v, lanes, 2 * LL.words(lk), draw["root"]).items()
                            if x != "word_bits"}})
        openings[v] = [(row, _bc(rng)) for row in range(lanes)]
    statements = [{"call": call, "circuit": f"/w/c{call}/circuit.bin", "public": f"/w/c{call}/public.bin",
                   "sessions": [f"/w/c{call}/s{j}" for j in range(rng.randint(1, 2))]} for call in drawn]
    for u in leaves:
        for kind, ln in LL.statements(r, c) if not shape_off else [(LL.LANE, 0), (LL.COMBINE, None)]:
            d = f"/w/u{u}/{kind}{'' if ln is None else ln}"
            statements.append({"unit": u, "statement": [kind, ln], "circuit": f"{d}/circuit.bin",
                               "public": f"{d}/public.bin", "sessions": [f"{d}/s{j}" for j in range(rng.randint(1, 2))]})
    rng.shuffle(statements)
    progs = {f"{kind}{lk}": leaf_desc(kind, lk) for kind in (LL.LANE, LL.COMBINE)}
    files = {f"k{k}": C._hex(rng) for k in ks if k != lk}
    files |= {x: C._hex(rng) for x in progs} | {Cr.LEAF_PROGRAMS: {x: hashlib.sha512(b).hexdigest()
                                                                   for x, b in progs.items()}}
    if rng.random() < 0.5:
        files[Cr.LIVE] = C._hex(rng)
    evidence = {"leaf_programs": {x: b.hex() for x, b in progs.items()},
                "produced": PR.record(draw["root"], draw["coin"]["coin"], produced, openings),
                "statements": statements}
    case = {"statements": evidence, "files": files, "descriptor": descriptor, "entries": entries,
            "window": C._hex(rng), "units": units, "steps": steps, "salt": C._hex(rng), "draw": draw, "runs": None}
    leaf_at = [i for i, s in enumerate(statements) if "unit" in s]
    s = statements[rng.choice(leaf_at)]
    match name:
        case "lanes-missing":
            case["statements"] = [x for x in statements if "unit" not in x]
        case "lanes-no-combine":
            u = rng.choice(leaves)
            kind = rng.choice([LL.COMBINE, LL.LANE])
            drop = [i for i, x in enumerate(statements) if x.get("unit") == u and x["statement"][0] == kind]
            del statements[rng.choice(drop)]
        case "lanes-leaf-digest":
            x = rng.choice(sorted(progs))
            evidence["leaf_programs"][x] = rng.choice([b"another".hex(), progs[x].hex().upper(), ""])
        case "lanes-leaf-other":
            x = rng.choice(sorted(progs))
            kind = x[:-len(str(lk))]
            other = rng.choice([leaf_desc(kind, lk, at=lk + 32),
                                leaf_desc(kind, lk, unit_kind=LL.LANE if kind == LL.COMBINE else LL.COMBINE),
                                progs[x] + b" ", progs[x].replace(b'"call"', b'"batch"')])
            evidence["leaf_programs"][x] = other.hex()
            files[Cr.LEAF_PROGRAMS][x] = hashlib.sha512(other).hexdigest()
        case "lanes-twice":
            statements.insert(rng.randrange(len(statements) + 1), {**s, "sessions": ["/w/again"]})
        case "lanes-undrawn":
            idle = [u for u in range(sum(sizes)) if u not in leaves]
            statements.insert(rng.randrange(len(statements) + 1), {**s, "unit": rng.choice(idle) if idle else 0})
        case "lanes-statement":
            statements.insert(rng.randrange(len(statements) + 1), {**s, "statement": rng.choice(
                [[LL.LANE, lanes + rng.randrange(3)], [LL.COMBINE], [LL.CHUNK, 0, 0], [LL.LANE, None], ["tile", 0]])})
        case "lanes-record":
            rec = evidence["produced"]
            evidence["produced"] = rng.choice([{**rec, "draw": C._hex(rng)}, {**rec, "registration": C._hex(rng)},
                                               {**rec, "produced": rec["produced"][1:]},
                                               {**rec, "openings": {v: ops[1:] for v, ops in rec["openings"].items()}},
                                               {**rec, "produced": [{**rec["produced"][0], "root": "zz"},
                                                                    *rec["produced"][1:]]},
                                               {x: y for x, y in rec.items() if x != "openings"}])
        case "lanes-unregistered":
            place = Cr.leaf_place(layout, shape, rng.choice(leaves), steps)
            v = place[rng.choice([LL.A, LL.B, LL.ZY, LL.RECEIPT])][0]
            case["entries"] = [e for e in entries if e["value"] != v] if rng.random() < 0.5 else [
                *entries, {"value": v, "root": C._hex(rng)}]
        case "lanes-draw":
            case["draw"] = rng.choice([None, {"coin": draw["coin"]}, {**draw, "root": draw["root"][:-2]},
                                       {**draw, "coin": {x: y for x, y in draw["coin"].items() if x != "path"}},
                                       {**draw, "coin": None}])
        case "lanes-form":
            bad = rng.choice([{**s, "unit": sum(sizes)}, {**s, "statement": []}, {**s, "statement": [1]},
                              {**s, "statement": [LL.LANE, -1]}, {**s, "sessions": []}, {**s, "extra": 1}])
            if rng.random() < 0.3:
                evidence["leaf_programs"] = rng.choice([[], {**evidence["leaf_programs"], "lane1": 7}])
            else:
                statements.append(bad)
        case "lanes-files":
            case["files"] = rng.choice([{x: y for x, y in files.items() if x != f"{LL.COMBINE}{lk}"},
                                        {**files, f"k{lk}": C._hex(rng)},
                                        {x: y for x, y in files.items() if x != Cr.LEAF_PROGRAMS},
                                        {**files, Cr.LEAF_PROGRAMS: {**files[Cr.LEAF_PROGRAMS], "lane32": C._hex(rng)}},
                                        {**files, Cr.LEAF_PROGRAMS: {x: "ab" for x in files[Cr.LEAF_PROGRAMS]}},
                                        {**files, Cr.LEAF_PROGRAMS: None}])
    if name in LANES_RUN or name == "lanes-honest" or rng.random() < 0.6:
        asks = Cr._asks(case["statements"], case["files"], raw, layout, case["entries"], case["window"], units, steps,
                        case["salt"], case["draw"])[0]
        case["runs"] = _lanes_answers(rng, case, asks, name, starts, sizes)
    return case


def _leaf_finding(session: str, ask: dict, rec: dict) -> dict:
    """An accepted leaf session's finding: every port of its ``--registered`` file at its value and root, and the rows
    it opened of each value the verifier holds openings of (the record's partials, its known rows), each the one
    opening."""
    ports = ask["registered"]["ports"]
    want = PR.wanted(rec, ask["known"])
    return {"session": session, "accepted": True, "why": "accepted", "program": ask["partition"]["program"],
            "registered": {p: {"value": e["value"], "root": e["root"]} for p, e in ports.items()}, "units": [0],
            "openings": {p: [[e["reads"][0], want[e["value"]][e["reads"][0]]]] for p, e in ports.items()
                         if e["value"] in want}}


def _lanes_answers(rng: random.Random, case: dict, asks: list[dict], name: str, starts: list[int],
                   sizes: list[int]) -> list:
    """`_answers` over a lanes case: the call statements' as there, each leaf session `_leaf_finding`'s, with `name`'s
    mutation at one leaf run (every one for ``lanes-empty``), cut after the first the Lean refuses at."""
    lanes = case["statements"]
    listed = lanes["statements"] if isinstance(lanes, dict) else lanes
    calls = [a for a in asks if "leaf" not in a]
    out = _answers(rng, {**case, "statements": listed}, calls, "honest", starts, sizes)
    if len(out) < len(calls) or not all("findings" in a and all(f["accepted"] for f in a["findings"]) for a in out):
        return out
    leafs = [a for a in asks if "leaf" in a]
    at = rng.randrange(len(leafs)) if leafs else None
    for j, a in enumerate(leafs):
        s = listed[a["statement"]]
        fs = [_leaf_finding(x, a, lanes["produced"]) for x in s["sessions"]]
        ans = {"findings": fs}
        if name == "lanes-empty":
            ans = {"findings": []}
        elif j == at:
            f = rng.choice(fs)
            match name:
                case "lanes-partial" | "lanes-known":
                    wanted = [p for p, e in a["registered"]["ports"].items()
                              if (e["value"] in a["known"]) == (name == "lanes-known") and p in f["openings"]]
                    p = rng.choice(wanted)
                    row, bc = f["openings"][p][0]
                    f["openings"][p] = rng.choice([[[row, ("0" if bc[0] != "0" else "1") + bc[1:]]], [],
                                                   [[row + 1, bc]], [[row, bc], [row, bc[::-1]]]])
                case "lanes-openings-none":
                    f["openings"] = rng.choice([None, {}])
                case "lanes-root":
                    p = rng.choice(sorted(f["registered"]))
                    f["registered"] = rng.choice([{**f["registered"], p: {**f["registered"][p], "root": C._hex(rng)}},
                                                  {x: y for x, y in f["registered"].items() if x != p},
                                                  {**f["registered"], "o9": f["registered"][p]}])
                case "lanes-rejected":
                    f.update(accepted=False, why=rng.choice(["sumcheck round 3", "opening root"]), units=[])
                case "lanes-units":
                    f.update(rng.choice([{"units": [0, 1]}, {"units": []}, {"program": C._hex(rng)}]))
                case "lanes-error":
                    ans = {"error": rng.choice(["flock-verify ended on a signal", "no such file: public.bin"])}
                case "lanes-refused":
                    ans = {"refused": [rng.choice(["proof", "circuit", "registration"]),
                                       "the statement's public file is not its circuit's"]}
        out.append(ans)
        if "findings" not in ans or not all(f["accepted"] for f in ans["findings"]):
            break
    return out


@dataclasses.dataclass(frozen=True)
class Opened(PF.Finding):
    """A verdict with the rows its session opened (``{port: [[position, b || c] ...]}``), as `produced.shared` reads."""

    openings: dict | None = None


def replay(runs: list) -> Cr.Run:
    """A verifier that answers `runs` in turn, as `credit._answer` reads its run: a finding with
    ``openings`` (a leaf session's) as `Opened`."""
    answers = iter(runs)

    def run(_proof, *, circuit_sha512, program, partition, registered):
        a = next(answers)
        if "error" in a:
            raise ValueError(a["error"])
        if "refused" in a:
            return None, tuple(a["refused"])
        return [Opened(f["session"], f["accepted"], f["why"], f["program"], f["registered"], tuple(f["units"]),
                       f["openings"]) if "openings" in f else
                PF.Finding(f["session"], f["accepted"], f["why"], f["program"], f["registered"], tuple(f["units"]))
                for f in a["findings"]], None
    return run
