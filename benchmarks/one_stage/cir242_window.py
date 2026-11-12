"""CIR-242 phase 2 scratch, not for main: one sparse-MLA chain row in form (b), each key block's keys with one writer.

A ``SparseMlaSplit_v1`` row at NS = 1 (``TOY``: T = 40 keys, visited in blocks of 8 (the first), 16 and 16, two heads,
DR = 32, DV = 256, so kv_c is eight column blocks of 32 and k_pe a ninth) served as:

* per (head, key block) a scores unit with its keys in column blocks (`SparseMlaHeadScoresCols_v1{G = DR}`, ports ``kv0`` ..
  ``kv7`` and ``k_pe``) and a values unit per column block (`SparseMlaHeadValues_v1{G = DR}`);
* per (head, column block) the chain's end on it (`SparseMlaHeadEnd_v1{DV = G}`);
* per (key block, column block) one writer, `SparseMlaHeadKeys_v1{NVIS, W = DR}`, whose ``o0`` every reader's key port is wired
  to (`chain_rows.wire_keys`), the reader a member with no tree of its own for any wired port (``Member.wired_trees`` False),
  each held port a table of its writer member's rows (``Member.tables``).

Each reader member is one (head, key block), so every port it holds (its keys and its carried state) is one writer member's
rows; each keys member is one column block (``--per column``) or one key block's nine (``--per block``). It reads the row,
checks each unit's Definition on its rows (`UnitStager.reference_check`), commits the window as serving does (`rows.commit`),
and prints one JSON line a fact: what the Binding holds, the rows and bytes the commitment holds against the figures computed
from the Definitions' signatures, the verifier's checks on the public bodies and the record, and each member's published
domain ids (Flock's `PortRoots.dupId`). ``--attack`` runs a prover's tries at other key rows and prints each refusal.
``--wire`` is form (b3): the keys writer `SparseMlaHeadKeysWire_v1`, each word through `Bf16ToF32_v1` and `U32Hi16_v1`, so
its template has gates (at 0 ANDs) where the copy has none.

``uv run python benchmarks/one_stage/cir242_window.py [--per column|block] [--wire] [--attack] [--glm T]``."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "adapters" / "vllm"))

from tests.pipeline import test_serving_rows_splits as TS  # noqa: E402
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.circuits.defs import bind  # noqa: E402
from verity.commitments.windows import rows as CS  # noqa: E402
from verity.commitments.windows import wiring as WI  # noqa: E402
from verity.definitions.registry import boolean_sparse_mla as BM  # noqa: E402
from verity.definitions.registry import sparse_mla as M  # noqa: E402
from verity.protocols.spot.verifier.one_stage import draw as D  # noqa: E402
from verity.protocols.spot.verifier.one_stage import registration as R  # noqa: E402
from verity_vllm.commit import index as IX  # noqa: E402
from verity_vllm.pipeline import chain_rows as CH  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402

TOY = {"T": 40, "NS": 1, "NH": 2, "DR": 32, "DV": 256, "BN": 16, "SCALE": 0.0625}
#: GLM-5.3's sparse-MLA statics (`test_serving_rows_splits.GLM`) at NS = 1 and ``--glm T`` keys
GLM = {"NS": 1, "NH": 8, "DR": 64, "DV": 512, "BN": 64, "SCALE": 0.0625}


def emit(**kw) -> None:
    print(json.dumps(kw, sort_keys=True, default=str), flush=True)


def scores(nvis: int | None, S: dict, boolean: bool = False) -> str:
    st = {x: S[x] for x in ("DR", "DV", "BN", "SCALE")}
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadScoresColsV2, **at, **st, G=S["DR"], DOT=TS.BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadScoresCols, **at, **st, G=S["DR"], DOT=TS.FA3["DOT"]))


def keys(nvis: int, S: dict, boolean: bool = False, wire: bool = False) -> str:
    """The keys writer: the copy (`SparseMlaHeadKeys_v1`, no gate), or with ``wire`` form (b3)'s, each word through
    `Bf16ToF32_v1` and `U32Hi16_v1` (`SparseMlaHeadKeysWire_v1`, gates at 0 ANDs)."""
    if wire:
        return descriptor_id(bind(BM.SparseMlaHeadKeysWireV2 if boolean else M.SparseMlaHeadKeysWire, NVIS=nvis, W=S["DR"]))
    return descriptor_id(bind(BM.SparseMlaHeadKeysV2 if boolean else M.SparseMlaHeadKeys, NVIS=nvis, W=S["DR"]))


def writer(nvis: int | None, S: dict, boolean: bool = False) -> str:
    """Form (b2)'s writer: the scores unit that also returns the keys it read, in column blocks of DR (`SparseMlaHeadScoresKeys_v1`)."""
    st = {x: S[x] for x in ("DR", "DV", "BN", "SCALE")}
    at = {"NVIS": S["BN"], "FIRST": False} if nvis is None else {"NVIS": nvis, "FIRST": True}
    return descriptor_id(bind(BM.SparseMlaHeadScoresKeysV2, **at, **st, G=S["DR"], DOT=TS.BOOL["DOT"]) if boolean
                         else bind(M.SparseMlaHeadScoresKeys, **at, **st, G=S["DR"], DOT=TS.FA3["DOT"]))


def layout_b2(S: dict, each: bool = False, boolean: bool = False) -> tuple[dict[str, str], dict[str, int]]:
    """Form (b2)'s templates by role and caps: per (row, key block) head `chain_rows.KEYS_HEAD`'s scores unit the writer
    (each head's with ``each``), the other heads' `SparseMlaHeadScoresCols_v1`, the values units in column blocks of DR and
    the ends; a later writer member holds every later block of its head, every other member one unit (an end, its head's
    column blocks)."""
    T, BN, DR, DV = (int(S[x]) for x in ("T", "BN", "DR", "DV"))
    first, nc, later = T - (T - 1) // BN * BN, DV // DR, (T - 1) // BN
    tm = {"writer first": writer(first, S, boolean), "writer": writer(None, S, boolean)}
    if not each:
        tm |= {"scores first": scores(first, S, boolean), "scores": scores(None, S, boolean)}
    tm |= {"values first": TS.values(first, S, boolean, G=DR), "values": TS.values(None, S, boolean, G=DR),
           "end": descriptor_id(bind(M.SparseMlaHeadEnd, DV=DR, INV=TS.FA3["INV"]))}
    caps = {t: {"writer": later, "end": nc}.get(r, 1) for r, t in tm.items()}
    return tm, caps


def computed_b2(S: dict, each: bool = False) -> dict[str, int]:
    """The rows form (b2) commits for the row, from the Definitions' signatures alone: per key block the writer's q_nope,
    q_rope, kv, carried m_prev and l_prev past the first (its own trees), m, l, pb, rescale past the first, and its returned
    keys' DV / DR + 1 column blocks; each other head's scores unit's q_nope, q_rope, m, l, pb and rescale (its keys and carried
    state held); each values unit's o; per (head, column block) the end's l, o and output."""
    T, NH, DR, DV, BN = (int(S[x]) for x in ("T", "NH", "DR", "DV", "BN"))
    nc, blocks = DV // DR, [min(T, (b + 1) * BN) - b * BN for b in range(-(-T // BN) - 1, -1, -1)]
    nw = NH if each else 1
    rows: list[int] = []
    for k, nvis in enumerate(blocks):
        w = [2 * DV, 2 * DR, 2 * nvis * (DV + DR)] + (k > 0) * [4, 16] + [4, 16, 2 * nvis] + (k > 0) * [4] + (nc + 1) * [2 * nvis * DR]
        rows += nw * w + (NH - nw) * ([2 * DV, 2 * DR, 4, 16, 2 * nvis] + (k > 0) * [4]) + NH * nc * [4 * DR]
    rows += NH * nc * [16, 4 * DR, 2 * DR]
    return {"rows": len(rows), "bytes": sum(rows), "bytes_whole_blocks": sum(-(-b // 128) * 128 for b in rows)}


def layout(S: dict, per: str, queries: bool = False, wire: bool = False, boolean: bool = False) -> tuple[dict[str, str], dict[str, int]]:
    """The row's templates by role, and each one's chunk cap (`serving_rows.members`); with ``queries`` a head's query one
    writer too, the row's heads one member."""
    T, BN, DR, DV = (int(S[x]) for x in ("T", "BN", "DR", "DV"))
    first, nc, later = T - (T - 1) // BN * BN, DV // DR, (T - 1) // BN
    if first == BN:
        raise SystemExit(f"T = {T} is whole blocks of {BN}: the first block's keys template is the later blocks', give T % BN != 0")
    tm = {"scores first": scores(first, S, boolean), "scores": scores(None, S, boolean),
          "values first": TS.values(first, S, boolean, G=DR), "values": TS.values(None, S, boolean, G=DR),
          "keys first": keys(first, S, boolean, wire), "keys": keys(BN, S, boolean, wire),
          "end": descriptor_id(bind(M.SparseMlaHeadEnd, DV=DR, INV=TS.FA3["INV"]))}
    kf, kl, vc = (1, later, 1) if per == "column" else (nc + 1, later * (nc + 1), nc)
    caps = dict(zip(tm.values(), (1, 1, vc, vc, kf, kl, nc), strict=True))
    if queries:
        tm["queries"] = descriptor_id(bind(M.SparseMlaHeadQuery, DR=DR, DV=DV))
        caps[tm["queries"]] = int(S["NH"])
    return tm, caps


def zk_window(form: str) -> tuple[TS.Served, list[str], dict[str, int], dict]:
    """The toy row's window in form ``form`` (``b3``: form (b) with the wire writer; ``b2``) over Boolean scores, values and
    writers, the templates a window proved under ``--zk`` names (`a5_sparse_mla.py --cir242`): its served row, templates,
    caps and statics."""
    S = TOY
    if form == "b2":
        CH.KEYS_HEAD = 0
        tm, caps = layout_b2(S, boolean=True)
    else:
        tm, caps = layout(S, "column", wire=True, boolean=True)
    return TS.Served(TS.row(S), n=1), list(tm.values()), caps, S


def window(s: TS.Served, ts: list[str], caps: dict[str, int], rewire=None) -> CS.Binding:
    rows = [s.rows] * len(ts)
    ms = (CH.wire_keys if CH.KEYS_HEAD is None else CH.wire_writer)(CH.wire(SRW.members(ts, rows, None, caps), ts, rows), ts, rows)
    ms = rewire(ms) if rewire else ms
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in ts]
    return CS.Binding(*CS.population_program(defs, several=True), D.parse_design("stratified:1"), "ab" * 32, tuple(ms))


def computed(S: dict, queries: bool = False) -> dict[str, int]:
    """The rows form (b) commits for the row, and their bytes, from the Definitions' signatures alone: per (head, key block) the
    scores unit's q_nope and q_rope (unless ``queries``), m, l, pb and (past the first) rescale, and each column block's values
    unit's o; per (key block, column block) the keys unit's in and out; per (head, column block) the end's l, o and output;
    with ``queries``, per head its query's writer's two inputs and two outputs. A wired port held by its writer commits no row:
    the carried state and every reader's keys (and queries)."""
    T, NH, DR, DV, BN = (int(S[x]) for x in ("T", "NH", "DR", "DV", "BN"))
    nc, blocks = DV // DR, [min(T, (b + 1) * BN) - b * BN for b in range(-(-T // BN) - 1, -1, -1)]
    rows: list[int] = []
    for k, nvis in enumerate(blocks):
        rows += NH * ([] if queries else [2 * DV, 2 * DR]) + NH * [4, 16, 2 * nvis] + NH * (k > 0) * [4] + NH * nc * [4 * DR]
        rows += 2 * (nc + 1) * [2 * nvis * DR]
    rows += NH * nc * [16, 4 * DR, 2 * DR] + NH * queries * [2 * DV, 2 * DR, 2 * DV, 2 * DR]
    return {"rows": len(rows), "bytes": sum(rows), "bytes_whole_blocks": sum(-(-b // 128) * 128 for b in rows)}


def frame_dups(c: CS.Committed, k: int) -> dict[str, list[str]]:
    """Member ``k``'s ports that share a published frame-v3 domain id (`Committed.frame_entry`), by id: Flock's `dupId`."""
    by: dict[str, list[str]] = {}
    for p, i in c.frame_entry(k)["domain_ids"].items():
        by.setdefault(i, []).append(p)
    return {i[:16]: ps for i, ps in by.items() if len(ps) > 1}


def record(c: CS.Committed, w: CS.Binding, got) -> dict:
    with tempfile.TemporaryDirectory() as d:
        CS.write(c, Path(d), {"committer": {"module": "cir242_window.py"}}, IX.index_doc(w.members, [e for _g, e in got], w.run),
                 kind="stand-in", registrar=R)
        return json.loads((Path(d) / "registration.json").read_text())


#: the gate-free writers `probe_gateless` lets through
WRITERS = ("SparseMlaHeadKeys_v1", "SparseMlaHeadQuery_v1")


def probe_gateless():
    """A probe, not the verifier: core's partition check (`partition._check_units`) with its refusal of a template of no gate
    set aside for the writers alone (`SparseMlaHeadKeys_v1`, `SparseMlaHeadQuery_v1`), every other refusal as it is, to see
    what else refuses the window."""
    from verity.protocols.spot.verifier.one_stage import partition as P
    from verity.protocols.spot.verifier.one_stage.partition import PO, QueryRefused, UNIT_CLASS_MAX, _template_bits, unit_class

    real = P._check_units

    def check_units(program, part):
        if not isinstance(part, PO.TemplatePartition):
            return real(program, part)
        for t, fn in {t: program.root.body.nodes[k].fn for k, t, _first, n in part.ranges if n}.items():
            if fn.gates == 0 and not t.startswith(WRITERS):
                raise QueryRefused(f"template {t} has no gate", {"template": t})
            if fn.gates and unit_class(bits := _template_bits(fn)) is None:
                raise QueryRefused(f"an instance of {t} commits {bits} bits, more than the widest class, {UNIT_CLASS_MAX}",
                                   {"template": t, "bits": bits})
    P._check_units = check_units
    return real


def publics_of(c: CS.Committed) -> dict[str, bytes]:
    return {m.key: c.public_body(k) for k, m in enumerate(c.window.members)}


def S_of(t: str) -> dict:
    from verity_vllm.query.program_view import spec_statics
    return spec_statics(SRW.template(t).word)


def port(t: str, name: str) -> str:
    """Template ``t``'s row port for parameter ``name``."""
    return f"in{[nm for nm, _ty in SRW.template(t).fn.signature()[0]].index(name)}"


def refusal(f) -> str | None:
    try:
        f()
    except (ValueError, AssertionError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def attack_a(w: CS.Binding, k_val: int, data, publics, port_: str = "in1", what: str = "a: values kv ref names another row "
             "of its writer's table") -> None:
    """A reader's held port's ref another row of its writer's table (a values unit's kv by default)."""
    mine = w.members
    m = mine[k_val]
    i = [p for p, _w in m.rows].index(port_)
    table = dict(m.tables)[port_]
    other = (int(data[k_val]["refs"][0, i]) + 1) % table
    if other == int(data[k_val]["refs"][0, i]):
        emit(attack=what, member=m.key, table=table, note="its table is one row: no ref names another")
        return
    bad = [dict(x) for x in data]
    bad[k_val] = {**data[k_val], "refs": data[k_val]["refs"].copy()}
    bad[k_val]["refs"][0, i] = other
    emit(attack=what, member=m.key, port=port_, table=table, honest_ref=int(data[k_val]["refs"][0, i]), forged_ref=other,
         committer=refusal(lambda: CS.commit(w, bad, salt_key=bytes(32))))
    body = bytearray(publics[m.key])
    at = len(body) - 4 * m.n * len(m.all_rows) + 4 * i
    body[at:at + 4] = np.uint32(other).tobytes()
    forged = {**publics, m.key: bytes(body)}
    others = [u for u in range(sum(x.n for x in mine)) if not m.base <= u < m.base + m.n][:1]
    emit(attack=what[:1], verifier_check=WI.check(mine, forged), check_drawn_unit_drawn=WI.check_drawn(mine, forged, [m.base]),
         check_drawn_unit_not_drawn=WI.check_drawn(mine, forged, others))


def attacks(s: TS.Served, w: CS.Binding, ts: list[str], caps: dict[str, int], role: dict[str, str], got, data, c, rec) -> None:
    publics, mine = publics_of(c), w.members
    vals = [k for k, m in enumerate(mine) if role[m.template] == "values"]
    k_val = next((k for k in vals if dict(mine[k].tables).get(port(mine[k].template, "kv"), 1) > 1), vals[0])
    k_sc = next((k for k, m in enumerate(mine) if role[m.template] == "scores"), None)
    attack_a(w, k_val, data, publics, port(mine[k_val].template, "kv"))
    if "queries" in role.values():
        attack_a(w, k_sc, data, publics, port(mine[k_sc].template, "q_nope"),
                 "q: a scores unit's q_nope ref names another head's row of its writer's table")

    if k_sc is None:
        emit(attack="b, c", note="no head reads another's keys: every scores unit is its own head's writer (--each)")
    else:
        attack_bc(s, w, ts, caps, got, data, rec, publics, k_sc)
    if "keys" in role.values():
        attack_d(w, got, data, mine, next(k for k, m in enumerate(mine) if role[m.template] == "keys"))
    if "writer" in role.values():
        attack_d2(w, got, mine, next(k for k, m in enumerate(mine) if role[m.template] == "writer"))


def attack_bc(s, w, ts, caps, got, data, rec, publics, k_sc) -> None:
    mine, want = w.members, CS.expected_ports(w)
    # (b) a reader given key rows of its own (wired_trees True) that differ from its writer's
    t_sc = mine[k_sc].template
    nc = int(S_of(t_sc)["DV"]) // int(S_of(t_sc)["G"])
    kps = {port(t_sc, f"kv{j}") for j in range(nc)} | {port(t_sc, "k_pe")}
    kv0 = port(t_sc, "kv0")

    def own_trees(ms):
        return [replace(x, wired_trees=True, tables=()) if not x.wired_trees else x for x in ms]
    w2 = window(s, ts, caps, own_trees)
    g2 = [dict(g, rows=dict(g["rows"])) for g, _e in got]
    x = g2[k_sc]["rows"][kv0].copy()
    x[0, 0] ^= np.uint16(1)
    g2[k_sc]["rows"][kv0] = x
    emit(attack="b: a reader's own key rows (wired_trees True) differing from its writer's", member=mine[k_sc].key, port=kv0,
         committer=refusal(lambda: CS.commit(w2, g2, salt_key=bytes(32))))
    w3 = window(s, ts, caps, lambda ms: [replace(x, wired=tuple(q for q in x.wired if q[0] not in kps)) if x.template == t_sc else x
                                         for x in own_trees(ms)])
    c3 = CS.commit(w3, g2, salt_key=bytes(32))
    rec3 = record(c3, w3, got)
    p3 = publics_of(c3)
    emit(attack="b", own_trees_ports=len(CS.expected_ports(w3)), verifier_ports=len(want),
         verifier_registration=R.check(rec3, w.program, w.query, w.law, CS.SCHEME, want, leaf_layer=CS.leaf_layer(p3)).to_json(),
         verifier_wires_on_own_trees=WI.check(w2.members, p3))

    # (c) a scores unit's kv3 wired to column block 4's writer (its writer and return)
    m = mine[k_sc]
    ws = {p: x for p, _o, x in m.wired}
    os_ = {p: o for p, o, _x in m.wired}
    kv3, kv4 = port(t_sc, "kv3"), port(t_sc, "kv4")

    def rewire(ms):
        return [replace(x, wired=tuple((p, os_[kv4], ws[kv4]) if p == kv3 else (p, o, y) for p, o, y in x.wired)) if j == k_sc
                else x for j, x in enumerate(ms)]
    w4 = window(s, ts, caps, rewire)
    c4 = CS.commit(w4, CH.held_rows(w4, [g for g, _e in got]), salt_key=bytes(32))
    p4 = publics_of(c4)
    fv = c4.frame_entry(k_sc)
    roots = {r["port"]: r for r in rec["roots"]}
    emit(attack=f"c: a scores unit's kv3 ({kv3}) wired to column block 4's writer ({kv4}'s)", member=m.key,
         honest_writers=[os_[kv3], ws[kv3]], forged_writers=[os_[kv4], ws[kv4]], verifier_check=WI.check(mine, p4),
         check_drawn_unit_drawn=WI.check_drawn(mine, p4, [m.base]),
         a5_root_check={"held": w.held(m, kv3), "forged_held": w4.held(w4.members[k_sc], kv3),
                        "published_root_is_registered": fv["roots"][kv3] == roots[w.held(m, kv3)]["root"]})
    i, j = ([p for p, _w in m.rows].index(q) for q in (kv3, kv4))
    if w.held(m, kv3) == w.held(m, kv4):
        bad = [dict(x) for x in data]
        bad[k_sc] = {**data[k_sc], "refs": data[k_sc]["refs"].copy()}
        bad[k_sc]["refs"][0, i] = data[k_sc]["refs"][0, j]
        body = bytearray(publics[m.key])
        at = len(body) - 4 * m.n * len(m.all_rows) + 4 * i
        body[at:at + 4] = np.uint32(data[k_sc]["refs"][0, j]).tobytes()
        emit(attack="c by ref: kv3's ref names column block 4's row of the one writer member's table", table=dict(m.tables)[kv3],
             forged_ref=int(data[k_sc]["refs"][0, j]), committer=refusal(lambda: CS.commit(w, bad, salt_key=bytes(32))),
             verifier_check=WI.check(mine, {**publics, m.key: bytes(body)}))
    else:
        emit(attack="c by ref", table=dict(m.tables)[kv3], kv4_ref=int(data[k_sc]["refs"][0, j]), kv3_holds=w.held(m, kv3),
             kv4_holds=w.held(m, kv4), note="kv3's table is the rows of the tree it holds, which holds no row of column block 4")



def attack_d(w, got, data, mine, k_key) -> None:
    """(d) a keys writer's output differing from its input."""
    m = mine[k_key]
    g5 = [dict(g, rows=dict(g["rows"])) for g, _e in got]
    x = g5[k_key]["rows"]["in0"].copy()
    x[0, 0] ^= np.uint16(1)
    g5[k_key]["rows"]["in0"] = x
    c5 = CS.commit(w, CH.held_rows(w, g5), salt_key=bytes(32))
    p5 = publics_of(c5)
    readers = [refusal(lambda q=q, g=g: TS.stager(q.template).reference_check(g["rows"], g["outs"]))
               for q, g in zip(mine, g5) if q.key != m.key]
    emit(attack="d: a keys unit's in0 not its o0 (readers read o0)", member=m.key, committer=None,
         verifier_check=WI.check(mine, p5), check_drawn_all=WI.check_drawn(mine, p5, list(range(sum(q.n for q in mine)))),
         readers_definitions_refused=sum(r is not None for r in readers),
         writer_definition=refusal(lambda: TS.stager(m.template).reference_check(g5[k_key]["rows"], g5[k_key]["outs"])))


def attack_d2(w, got, mine, k_w) -> None:
    """(d2) form (b2)'s writer returning a column block that differs from the keys its own scores were computed from: its
    return c flipped, every reader holding it."""
    m = mine[k_w]
    o = f"o{3 if S_of(m.template)['FIRST'] is True else 4}"
    g = [dict(x, rows=dict(x["rows"]), outs=dict(x["outs"])) for x, _e in got]
    x = g[k_w]["outs"][o].copy()
    x[0, 0] ^= np.uint16(1)
    g[k_w]["outs"][o] = x
    c = CS.commit(w, CH.held_rows(w, g), salt_key=bytes(32))
    p = publics_of(c)
    emit(attack=f"d2: the writer's returned column block 0 ({o}) not the keys its scores read", member=m.key,
         committer=None, verifier_check=WI.check(mine, p), check_drawn_all=WI.check_drawn(mine, p, list(range(sum(q.n for q in mine)))),
         writer_definition=refusal(lambda: TS.stager(m.template).reference_check(g[k_w]["rows"], g[k_w]["outs"])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", choices=("column", "block"), default="column")
    ap.add_argument("--attack", action="store_true")
    ap.add_argument("--glm", type=int, help="GLM-5.3's statics at this many keys (T % 64 != 0)")
    ap.add_argument("--no-check", action="store_true", help="skip the Definitions' checks on the rows")
    ap.add_argument("--probe-gateless", action="store_true", help="past the partition's refusal of the gate-free keys template, "
                    "set it aside for that template alone (`probe_gateless`) and run the rest")
    ap.add_argument("--queries", action="store_true", help="each head's query one writer too (`SparseMlaHeadQuery_v1`), every "
                    "scores unit's q_nope and q_rope wired to it")
    ap.add_argument("--form", choices=("b", "b2"), default="b", help="b2: per key block one head's scores unit the writer "
                    "of its keys (`SparseMlaHeadScoresKeys_v1`), the other heads' scores units and every values unit reading its returns")
    ap.add_argument("--each", action="store_true", help="with --form b2: each head's scores unit the writer for its own values units")
    ap.add_argument("--wire", action="store_true", help="form (b3): the keys writer through wires "
                    "(`SparseMlaHeadKeysWire_v1`, gates at 0 ANDs), not the gate-free copy")
    ap.add_argument("--lean", action="store_true", help="the verifier of record's registration check (Python and Lean, "
                    "`one_stage.verifier.check`, which builds the Lean verifier) on the record, the probe removed")
    a = ap.parse_args()
    t0 = time.perf_counter()
    S = {**GLM, "T": a.glm} if a.glm else TOY
    if a.form == "b2":
        CH.KEYS_HEAD = -1 if a.each else 0
        tm, caps = layout_b2(S, a.each)
    else:
        tm, caps = layout(S, a.per, a.queries, a.wire)
    role = {t: r for r, t in tm.items()}
    ts = list(tm.values())
    s = TS.Served(TS.row(S), n=1)
    emit(fact="statics", S=S, form=a.form, each=a.each, keys_head=CH.KEYS_HEAD, per=a.per, wire=a.wire, templates=tm, caps={role[t]: c for t, c in caps.items()})
    try:
        w = window(s, ts, caps)
    except ValueError as e:
        emit(fact="Binding refused", why=str(e))
        return 1
    for m in w.members:
        held = {} if m.wired_trees else {p: {"tree": w.held(m, p), "leaves": list(w.held_leaves(m, p))} for p, _o, _ws in m.wired}
        emit(fact="member", key=m.key, role=role[m.template], n=m.n, base=m.base, wired_trees=m.wired_trees,
             tables=dict(m.tables), held=held, own_trees=CS.port_leaves(m))
    t1 = time.perf_counter()
    got = TS.read(s, w.members, ts)
    emit(fact="read", members=len(got), units=sum(m.n for m in w.members), s=round(time.perf_counter() - t1, 2))
    if not a.no_check:
        for m, (g, _e) in zip(w.members, got, strict=True):
            TS.stager(m.template).reference_check(g["rows"], g["outs"])
        emit(fact="every unit's Definition holds on its rows", members=len(got))
    data = CH.held_rows(w, [g for g, _e in got])
    c = CS.commit(w, data, salt_key=bytes(32))
    publics = publics_of(c)
    leaves = {p: len(c.bc[p]) // WI.BC_BYTES for p in c.roots}
    words = sum(np.asarray(c.rows[p]).nbytes for p in c.roots)
    refs = sum(4 * r.size for r in c.refs.values())
    emit(fact="committed", trees=len(c.roots), rows=sum(leaves.values()), row_bytes=words,
         public_bc=sum(len(c.bc[p]) for p in c.roots), public_refs=refs, salts=sum(len(c.salts[p]) for p in c.roots),
         public_bodies=sum(map(len, publics.values())), window_bytes=CS.window_bytes(w.members),
         expected_ports=len(CS.expected_ports(w)), expected_rows=sum(v["leaves"] for v in CS.expected_ports(w).values()),
         computed=computed_b2(S, a.each) if a.form == "b2" else computed(S, a.queries))
    emit(fact="frame-v3 domain ids shared within a member (Flock dupId)",
         members={m.key: d for k, m in enumerate(w.members) if (d := frame_dups(c, k))})
    emit(fact="verifier wiring check (wiring.check, every unit)", refused=WI.check(w.members, publics))
    from verity.protocols.spot.verifier.one_stage import partition as P
    try:
        emit(fact="verifier partition (registration R2: partition.units)", population=P.units(w.program, w.query))
    except P.QueryRefused as e:
        emit(fact="verifier partition (registration R2: partition.units)", refused=f"{type(e).__name__}: {e}")
        if not a.probe_gateless:
            return 2
        real = probe_gateless()
        emit(fact=f"PROBE: partition's no-gate refusal set aside for {WRITERS} alone",
             population=P.units(w.program, w.query))
    rec = record(c, w, got)
    if a.lean:
        from verity.protocols.spot.verifier.one_stage import verifier as V
        if a.probe_gateless:
            P._check_units = real
        v = V.check(rec, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=CS.leaf_layer(publics))
        emit(fact="verifier of record (registration: Python and Lean), the record made under the probe, the probe removed",
             verdict=v.to_json())
        if a.probe_gateless:
            probe_gateless()
    v = R.check(rec, w.program, w.query, w.law, CS.SCHEME, CS.expected_ports(w), leaf_layer=CS.leaf_layer(publics))
    emit(fact="verifier", registration=v.to_json(), wiring_check=WI.check(w.members, publics),
         check_drawn_every_unit=WI.check_drawn(w.members, publics, list(range(sum(m.n for m in w.members)))))
    if a.attack:
        attacks(s, w, ts, caps, role, got, data, c, rec)
    emit(fact="resources", s=round(time.perf_counter() - t0, 1), maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
