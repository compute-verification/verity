"""run.py prepare|summarize|cost: one served H1 leaf (32 × 32, at a small K: 448 by default; `h1zy.combine` takes K >= 384,
the least whose partial row compresses before its last SHA-512 block, which Flock's circuit needs of a combine row)
proved as its 32 lanes and its combine (PRF-134, option A:
`h1zy.PouwH1LaneZy_v1` and `PouwH1Combine_v1`, `pouw.leaf_lanes`' statement set), one Flock --zk statement each, every
statement reading the leaf's lane partials, a value committed after the draw (`one_stage.produced`), and the negatives.

* **The statements.** All 32 lanes are one circuit (the lane index is a public input) and the combine another. Each
  statement is the one unit of the Program a served window's verifier derives for it from its own copy of the window
  Program (`h1window.leaf_program`, here of a one-call window at K over this leaf's tile), and the Lean verifier derives
  its unit from its own copy of that Program (``--program``, ``--partition``). Each is flat, or under
  ``H1_TMPL=structural`` the templated statement of its Definition's structural lowering (`h1unit.layout`,
  `h1unit.stager`: struct_lower's template at `h1unit.LAYOUT`, the flat circuit never built; its template, ``unit.tmpl``,
  is what the prover's ``flock-tmpl-unit`` places). The Lean verifier refuses a templated statement once every other
  check has passed (`TMPL_WHY`), until its proof lands (PRF-133). Such a check is labelled `TEMPLATED`, never ACCEPT, and
  counts as passing in place of an expected ACCEPT (summary's ``templated``; ``lean_accepts`` counts the real ones).
* **The registered values** (`registered.Registered.commit`, ``row/v2`` rows of 16-bit leaves): ``a`` (32 rows: int8 codes
  and s_a), ``b`` (32 rows: codes and s_w), ``zy`` (the leaf's z/y record) and ``receipt`` (the leaf's CP-W receipt,
  `h1tile.reference`). STAND-IN: in a served window these are the window's A, z
  and admission tables, the weights' b/<K> and its receipt tree (`h1window`); here a registration record of the lane's
  Program (`registration.record`) stands for the window's, and its digest D is what the partials' domain binds. The roots
  the lanes' seeds read are (SHA-512 of root(a) under `ROOTS_TAG`, root(b)).
* **The partials**, committed after the draw (STAND-IN: the draw of every unit, `draw_digest`): ``partials``, 32 rows, row
  l lane l's (`h1zy.lane_partial`, 2 (KS + m) u32), `produced.commit`'s tree under D, with `produced.record`'s record,
  which `produced.check` holds to the verifier's own entry (``produced_check``). The record keeps every row's opening
  (its ``b || c``): the partials' and the rows of ``a``, ``b``, ``zy`` and ``receipt``.
* **Each statement's rows** are all ports of the verifier's own registered reads (`leaf_lanes.own`): `leaf_lanes.reads`'
  (lane l's ``a0..a3``, ``b0..b7``, ``zy`` and ``partials`` (row l); the combine's ``p0..p31`` and ``receipt``) and its
  known ports (`leaf_lanes.known_rows`, `produced.known_port`): lane l's ``admit`` and ``pub`` (salt, position, roots and the
  lane index l as a u16 word, `h1zy.LANE`, under `h1unit.pub_salt`), and every statement's output ``o0``, the row 1 under
  `h1unit.ok_salt`. The ``--registered`` file is `leaf_lanes.registered_file`'s at the entries and roots of the world's
  registration record and produced record alone (`leaf_lanes.verifier_entries`), at the leaf's placement in the window's
  tables (`h1window.place`: unit 0 of the one-call window, one round), the public file `h1unit.verifier_pub`'s.
  Each check is the Lean verifier's verdict and `pouw-leaf-own-window`'s (PoUW's `PouwCheck.LeafOwn.leafOwn?`): from the
  window's descriptor and round log it places unit 0's reads (`placeOf`), and the file, as flock-verify parses it, holds
  the layout's data ports there at the roots of the records the verifier holds (``holds``), and each of the statement's
  sessions draws every unit of its partition (`PouwWindow.Lanes.checkFull`, the draw policy).
* **The negatives**, each a statement proved and checked like the others:
  ``swap-lane3`` and ``swap-lane4``: a partials tree with rows 3 and 4 swapped (its record well-formed); each lane reads
  its own row, which holds the other lane's partial, so its ok is 0 (REJECT). ``swap-combine``: the combine on that tree,
  whose rows XOR to the receipt in any order (ACCEPT, by design: the swap is refused at the lanes). ``lane3-reads-row4``:
  the honest tree, lane 3 given row 4's leaf at position 3, which opens at no position but 4 (REJECT). ``other-leaf``: the
  combine with p5 another leaf's partial opened in that leaf's tree, whose domain binds that leaf's registration (REJECT).
  ``other-leaf-in-tree-lane5`` and ``other-leaf-in-tree``: the other leaf's partial committed at row 5 of this leaf's tree,
  which opens, and lane 5's and the combine's ok are 0 (REJECT). ``lane5-partial-off-tree``: on that tree, lane 5 given
  its honest partial, so its ok is 1, opened in the honest leaf's tree rather than the registered one: the partials are
  registered reads of every statement, never free rows (RT4-LOGIC-29c), so it opens at no position (REJECT).
  ``receipt-mismatch``: a registered receipt one bit off
  the lanes' XOR, the partials recommitted under its registration: the combine's ok is 0 (REJECT). ``lane3-index4``: lane
  3 whose ``pub`` row holds index 4, which opens no root but lane 4's (REJECT). ``ok-from-public``: the combine whose
  ``o0`` the prover commits afresh, the public file stating it, which the verifier's own reads refuse (REJECT).
  ``lane3-index-from-public``: lane 3 whose ``pub`` row (its index, 3) the prover commits afresh and the public file
  states, checked against a verifier's file that leaves ``pub`` out and says it is complete: refused by the verifier's
  `Registered.checkComplete` by name (REJECT, ``WHY``); ``lane3-index-from-public-open``, the same session against that
  file without ``complete``, which nothing else refuses (ACCEPT, by design: the control).
  ``combine-skips-p5-bit``: the combine over a circuit that reads bit i of ``p5`` in place of its last bit j, two bits the
  honest row holds equal (`lazy_combine`, its own Program), so its ok is 1 on the honest rows and it depends on no bit j:
  the verifier's file marks every A, B and partials port ``read_in_full`` (`leaf_lanes.registered_file`), and Flock's
  `KnownRead.fullRows` refuses the circuit by name (REJECT, ``WHY``); ``combine-skips-p5-bit-unmarked``, the same session
  against that file with ``p5``'s mark removed, which nothing else refuses (ACCEPT, by design: the control).
  ``other-registration-file``: lane 0 of the other leaf, its file and session that leaf's own and consistent, so
  flock-verify accepts it, checked with the verifier holding this leaf's records: `pouw-leaf-own` refuses the file by name
  (REJECT, ``WHY``); ``other-registration-file-held``, the same check holding the other leaf's records (ACCEPT, by
  design: the control).
  Every check registers ok = 1 (``o0``'s known row), so a statement whose ok is 0 is refused at its session's parameters
  (S2/R7: its instance's output commitment is not the verifier's).
* **The shared openings** (``shared_codes``, `produced.shared`): every row the honest sessions report opening (their
  verdicts' ``openings``) is the one the record keeps, or the verifier's known port's; a lane-5 verdict reporting another
  opening of the partials' row 5 (the mixed tree's) is refused (``shared_forged_lane5``: P-shared), with no appeal to
  SHA-512's collision resistance.

``cost`` counts a lane's and the combine's ANDs at a K (4096, the served one) without staging anything. ``prepare`` writes
WORK/plan.tsv (``S tag dir`` a session; ``C tag dir pub registered program partition expect registration produced
descriptor rounds unit statement flock-statement`` a Lean check and `pouw-leaf-own-window`'s), WORK/plan.json
and WORK/prepare.json; ``summarize`` writes OUT/summary.json from the Lean verdicts, and exits 1 unless every check's verdict
is the expected one, the honest statements' reported roots are the produced record's (`produced.opened`) and their
opened rows its openings (`produced.shared`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "h1_prove"))

import h1unit as U  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition, bind  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity.ml.circuits.refs import Coll, concat  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity.core.protocols.pouw import leaf_lanes as LL  # noqa: E402
from verity.core.protocols.pouw.circuit import h1tile as H  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity.core.protocols.pouw.circuit import h1zy as Z  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import partition as P  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import produced as PR  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as R  # noqa: E402

RC = (32, 32)
SCHEME = "frame-v3-sha512/hm96-sha512"
ROOTS_TAG = b"verity/pouw/h1-lanes/roots/v0\0"
DRAW_TAG = b"verity/pouw/h1-lanes/stand-in-draw/v0\0"
SALT_TAG = b"verity/pouw/h1-lanes/stand-in-salt/v0\0"
REGISTERED = (LL.A, LL.B, LL.ZY, LL.RECEIPT)
#: the one-call window's rounds (STAND-IN for the window's round log: one round, its call the leaf's)
STEPS = 1
#: every negative check's expected verdict
EXPECT = {"swap-lane3": "REJECT", "swap-lane4": "REJECT", "swap-combine": "ACCEPT", "lane3-reads-row4": "REJECT",
          "other-leaf": "REJECT", "other-leaf-in-tree-lane5": "REJECT", "other-leaf-in-tree": "REJECT",
          "lane5-partial-off-tree": "REJECT", "receipt-mismatch": "REJECT", "lane3-index4": "REJECT",
          "ok-from-public": "REJECT", "lane3-index-from-public": "REJECT", "lane3-index-from-public-open": "ACCEPT",
          "combine-skips-p5-bit": "REJECT", "combine-skips-p5-bit-unmarked": "ACCEPT", "other-registration-file": "REJECT",
          "other-registration-file-held": "ACCEPT"}
#: the refusal a negative's verdict names
WHY = {"lane3-index-from-public": "registered reads (complete): the statement's port pub is no port of the verifier's",
       "combine-skips-p5-bit": "registered reads: known port p5: no computed row of the circuit reads bit",
       "other-registration-file": "registered reads: port a0 is not the layout's read of "}
#: a check of another check's statement against its registered file without ``complete``
OPEN = {"lane3-index-from-public-open": "lane3-index-from-public"}
#: a check of another check's statement against its registered file with one port's ``read_in_full`` mark removed
UNMARKED = {"combine-skips-p5-bit-unmarked": ("combine-skips-p5-bit", "p5")}
#: the world whose records the verifier holds at a check, where it is not the statement's own
HOLDS = {"other-registration-file": "honest"}
#: a check of another check's statement, the verifier holding that statement's own world's records
HELD = {"other-registration-file-held": "other-registration-file"}
#: the combine row ``combine-skips-p5-bit``'s circuit leaves a bit of unread
LAZY_ROW = 5
#: the templated statement (`circuit.TMPL_STATEMENT`), and the Lean verifier's verdict on one whose every other check
#: passed: it refuses the form last, until the templated statement's proof lands (`Tags.templated`, `ProvedScope.refusal`)
TMPL_STATEMENT = "verity/flock-circuit/tmpl"
TMPL_WHY = f"refused: {TMPL_STATEMENT}, a templated statement: a statement form the soundness proof does not cover"
#: the verdict word of a templated check refused only for its form (`TMPL_WHY`): never ACCEPT
TEMPLATED = "templated: Lean refuses this form until PRF-133"


def tile(k: int):
    return HW.h1_tile(k, RC, H.PouwH1Tile3 if U.tile() == "v3" else None)


def f32(x) -> np.ndarray:
    return np.asarray(x, np.float32).view(np.uint32)


def u32_words(words) -> list[int]:
    return [int(v) for v in np.ascontiguousarray(np.asarray(words, np.uint16).astype("<u2")).view("<u4")]


def code_words(codes: list, scales: list) -> np.ndarray:
    return U.row_words(np.asarray(codes, np.uint8).view(np.int8), np.asarray(scales, np.uint32).view(np.float32))


def draw_digest(registration: str, population: int) -> str:
    """STAND-IN for the draw's digest: the live coin draws the leaf."""
    return hashlib.sha512(DRAW_TAG + canonical_json({"registration": registration, "units": list(range(population))})).hexdigest()


# -- the statements' Programs -----------------------------------------------------------------------------------------------

def _template(name: str, d):
    """A one-call template over ``d``, its parameters and ports (STAND-IN for the window's template, `h1window.template`)."""
    decl = getattr(d.defn, "ports", None)
    ports = {} if decl is None else {"ports": (lambda S: decl(d.S())) if callable(decl) else (lambda S: decl)}
    return bind(CompositeDefinition(name, 1, (), lambda S: d.signature(), lambda B, S, *args: B.call(d, *args),
                                    conformance="protocol-defined", register=False, **ports))


def lazy_combine(d, x: int, j: int, i: int):
    """The combine ``d`` reading bit ``i`` of its partials row ``x`` in place of bit ``j``, so that no wire reads bit
    ``j``: a circuit that leaves a ``read_in_full`` row's bit unread (``combine-skips-p5-bit``)."""
    params, ret = d.signature()
    per = sum(params[0][1].elem.leaf_widths())

    def body(B, S, partials, receipt):
        r = partials.refs
        lo = x * per
        moved = concat([r.slice(0, lo + j), r.slice(lo + i, lo + i + 1), r.slice(lo + j + 1, len(r))])
        return B.call(d, Coll(params[0][1], moved), receipt)

    decl = getattr(d.defn, "ports", None)
    ports = {} if decl is None else {"ports": (lambda S: decl(d.S())) if callable(decl) else (lambda S: decl)}
    return bind(CompositeDefinition("H1LanesLazyCombine", 1, (), lambda S: (params, ret), body,
                                    conformance="protocol-defined", register=False, **ports))


@dataclass
class Bound:
    """One statement kind's Definition, its Program (a one-call template) and partition."""

    defn: object
    program: object
    query: dict
    part: dict

    @property
    def bind(self) -> dict:
        return {"partition": P.digest(self.part), "program": self.part["program"]}


def bound(name: str, d) -> Bound:
    program = Program(_template(name, d))
    query = P.template_instances_query([d])
    part = P.partition(program, query)
    if P.units(program, query) != 1:
        raise AssertionError(f"{name}: the statement's Program is one unit")
    return Bound(d, program, query, part)


def window(t, k: int) -> Program:
    """A window of one call at K over the leaf's tile ``t``: its one unit is the leaf."""
    return HW.program((("w", k, RC[1]),), (("w", 0),), RC[0], RC, unit=lambda _k: Z.unit(t))


def derived(win: Program, k: int, kind: str) -> Bound:
    """``kind``'s statement at K as a served window's verifier derives it from its own copy of the window Program ``win``
    (`h1window.leaf_program`): the one-call Program of its wrapper over the lane or combine of ``win``'s template at K."""
    program, query = HW.leaf_program(win, k, kind)
    part = P.partition(program, query)
    if P.units(program, query) != 1:
        raise AssertionError(f"{kind}: the statement's Program is one unit")
    return Bound(HW.leaf_unit(win, k, kind), program, query, part)


# -- the leaf, its registration and its partials -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Leaf:
    a: list
    b: list
    sa: list
    sw: list
    zy: tuple
    admit: list
    salt: bytes
    position: int


def leaf_data(seed: int, k: int, salt: bytes, position: int) -> Leaf:
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, (RC[0], k)).astype(np.int8).view(np.uint8)
    b = rng.integers(-128, 128, (RC[1], k)).astype(np.int8).view(np.uint8)
    sa = f32(rng.uniform(1e-3, 0.1, RC[0]) / 127)
    sw = f32(rng.uniform(1e-3, 0.5, RC[1]) / 127)
    a_, b_, sa_, sw_ = a.tolist(), b.tolist(), [int(x) for x in sa], [int(x) for x in sw]
    return Leaf(a_, b_, sa_, sw_, Z.records(a_, b_, sa_, sw_), [1] * RC[0], salt, position)


@dataclass(frozen=True)
class World:
    """One registration and the partials after its draw: the verifier's view of a leaf, honest or not."""

    name: str
    leaf: Leaf
    roots: tuple
    regs: dict
    registration: str
    draw: str
    tree: G.Registered
    record: dict
    reg_record: dict


def register(prog: str, leaf: Leaf, place: dict) -> tuple[dict, tuple, list[int]]:
    """The leaf's registered values under ``prog``, each in its window table (``place``, the leaf's from its row 0), the
    roots its seeds read and its receipt."""
    def commit(v, rows):
        return G.Registered.commit(prog, place[v][0], rows, None, G.WORD_BITS, V2)

    regs = {LL.A: commit(LL.A, code_words(leaf.a, leaf.sa).tolist()), LL.B: commit(LL.B, code_words(leaf.b, leaf.sw).tolist()),
            LL.ZY: commit(LL.ZY, [U.record_words(*leaf.zy).tolist()])}
    roots = (hashlib.sha512(ROOTS_TAG + regs[LL.A].root).digest(), regs[LL.B].root)
    z, receipt = H.reference(leaf.a, leaf.b, leaf.salt, leaf.position, roots, stream=U.tile() == "v3")
    assert [v & 0xFFFFFFFF for v in z] == [v & 0xFFFFFFFF for v in leaf.zy[0]], "the record's z is the tile's"
    regs[LL.RECEIPT] = receipt_reg(prog, receipt, place)
    return regs, roots, list(receipt)


def receipt_reg(prog: str, receipt: list[int], place: dict, flip: int | None = None) -> G.Registered:
    """The registered receipt in its window table (``place``), with bit ``flip`` of it flipped when given."""
    rc = list(receipt)
    if flip is not None:
        rc[flip // 32] ^= 1 << (flip % 32)
    return G.Registered.commit(prog, place[LL.RECEIPT][0], [U.receipt_words(rc).tolist()], None, G.WORD_BITS, V2)


def lane_rows(leaf: Leaf, ln: int) -> tuple[list, list, list, list]:
    rows, cols = LL.lane_rows(*RC, ln), LL.lane_cols(*RC, ln)
    return [leaf.a[i] for i in rows], [leaf.b[j] for j in cols], [leaf.sa[i] for i in rows], [leaf.sw[j] for j in cols]


def partials(leaf: Leaf, roots: tuple) -> list[list[int]]:
    out = []
    for ln in range(LL.lanes(*RC)):
        a, b, _sa, _sw = lane_rows(leaf, ln)
        out.append(U.receipt_words(Z.lane_partial(a, b, leaf.admit, leaf.salt, leaf.position, roots, ln, RC)).tolist())
    return out


def world(name: str, ctx, leaf: Leaf, regs: dict, roots: tuple, rows: list) -> World:
    rec = R.record(ctx.reg.program, ctx.reg.query, {"law": "subset", "k": 1}, SCHEME, [],
                   {"kind": "stand-in", "harness": "h1_lanes", "leaf": leaf.position}, partition=ctx.reg.part,
                   registered=[regs[v].entry() for v in REGISTERED])
    d = R.digest(rec)
    draw = draw_digest(d, 1)
    tree = PR.commit(ctx.prog, LL.PARTIALS, rows, d, schema=V2)
    ops = {LL.PARTIALS: PR.rows_of(tree), **{v: PR.rows_of(regs[v]) for v in REGISTERED}}
    return World(name, leaf, roots, regs, d, draw, tree, PR.record(d, draw, [tree.entry()], ops), rec)


def entries(ctx, w: World) -> tuple[dict, dict]:
    """The verifier's entries and roots of ``w``, from its registration record and produced record alone
    (`leaf_lanes.verifier_entries`, Lean's `PouwWindow.Lanes.roots?`, which `pouw-leaf-own` holds each file to)."""
    return LL.verifier_entries(w.reg_record, w.record, LL.lanes(*RC), ctx.wp)


# -- a statement ------------------------------------------------------------------------------------------------------------

def opening(tree: G.Registered, position: int, row: int | None = None) -> dict:
    """What the prover opens for row ``position`` of ``tree``: the row it holds there (or at ``row``, opened in its place,
    with that row's salt and ``b || c``) and the path at ``position``."""
    r = position if row is None else row
    return {"value": tree.value, "row": list(tree.rows[r]), "salt": tree.salts[r], "bc": tree.bcs[r], "position": position,
            "path": tree.path(position).hex()}


def known(words, salt: bytes) -> dict:
    """A row the statement commits afresh, which the public file states: no registered read."""
    return {"row": [int(x) for x in words], "salt": salt}


def known_rows(ctx, w: World, kind: str, ln: int | None, index: int) -> dict:
    """The statement's known ports (`leaf_lanes.known_rows`), each ``(value, row, salt)`` as the verifier computes them:
    lane l's ``pub`` (index l) and ``admit``, and ``o0`` the row 1, under salts anyone derives (`h1unit.pub_salt`,
    `ok_salt`)."""
    lf = w.leaf
    rows = LL.known_rows(*RC, kind, ln, salt=lf.salt, position=lf.position, roots=w.roots, admit=lf.admit)
    salts = {LL.OK: U.ok_salt(f"{ctx.name}#{index}", ctx.position)}
    if kind == LL.LANE:
        salts[LL.PUB] = U.pub_salt(f"{ctx.name}#lane{ln}", lf.position)
        salts[LL.ADMIT] = U.pub_salt(f"{ctx.name}#admit", lf.position)
    return {p: (v, row, salts[p]) for p, (v, row) in rows.items()}


def known_opening(ctx, w: World, value: str, row: list[int], salt: bytes) -> dict:
    """The prover's opening of a known port's row: ``row`` committed under ``salt`` as its one-row tree."""
    return opening(PR.commit(ctx.prog, value, [row], w.registration, [salt], schema=V2), 0)


def stager(ctx, kind: str, variant: str | None = None):
    """``kind``'s stager, or with ``variant`` ``lazy`` the combine over `lazy_combine` (``ctx.lazy``), in `h1unit.layout`'s
    form (`h1unit.stager`). A templated one's template and layout mode go beside its circuit (``unit.tmpl``,
    ``unit.layout``), for the prover's ``flock-tmpl-unit``."""
    from verity_flock import tmpl as TM

    b = ctx.lazy if variant == "lazy" else ctx.lane if kind == LL.LANE else ctx.combine
    circuit = ctx.work / "circuits" / (variant or kind) / "circuit.txt"
    if kind == LL.LANE:
        ab = 16 * (ctx.k // 2 + 2)
        name, rows = "unit/pouw-h1-lane", ([(f"a{s}", V2, ab, [f"a[{s}]"]) for s in range(LL.LANE_ROWS)]
                                           + [(f"b{t}", V2, ab, [f"b[{t}]"]) for t in range(LL.LANE_COLS)]
                                           + [(LL.ADMIT, V2, RC[0], [LL.ADMIT]), (LL.ZY, V2, 48 * RC[0] * RC[1], [LL.ZY]),
                                              (LL.PARTIALS, V2, 16 * ctx.wp, [LL.PARTIALS]),
                                              ("pub", V2, U.PUB_BITS + 16, list(LL.PUBLIC))])
    else:
        name, rows = "unit/pouw-h1-combine", ([(f"p{x}", V2, 16 * ctx.wp, [f"{LL.PARTIALS}[{x}]"]) for x in range(LL.lanes(*RC))]
                                              + [(LL.RECEIPT, V2, 16 * ctx.wp, [LL.RECEIPT])])
    st = U.stager(b.defn, name, rows, [(V2, (0,))], U.layout(), circuit=circuit, bind=b.bind)
    if hasattr(st.low, "template"):
        (circuit.parent / "unit.tmpl").write_text(TM.write(st.low.template))
        (circuit.parent / "unit.layout").write_text(st.meta["layout"])
    return st


def expected(kind: str, ln: int | None, w: World, ins: dict, index: int | None = None) -> list[int]:
    """The statement's ok on its opened rows, on words (`h1zy.lane_reference`, `h1zy.combine_reference`): lane ``ln``'s
    rows, checked as lane ``index`` (default ``ln``), the index its ``pub`` row holds."""
    lf = w.leaf
    if kind == LL.LANE:
        a, b, sa, sw = lane_rows(lf, ln)
        return [Z.lane_reference(a, b, sa, sw, lf.admit, lf.zy, u32_words(ins[LL.PARTIALS]["row"]), lf.salt, lf.position,
                                 w.roots, ln if index is None else index, RC)]
    ps = [u32_words(ins[f"p{x}"]["row"]) for x in range(LL.lanes(*RC))]
    return [Z.combine_reference(ps, u32_words(ins[LL.RECEIPT]["row"]))]


def statement(ctx, kind: str, ln: int | None, tag: str, w: World, ins: dict, index: int, free: tuple = (),
              lane_index: int | None = None, dropped: tuple = (), variant: str | None = None) -> dict:
    """Stage one statement and write the verifier's files: ``ins`` each input port's opening (`opening`,
    `known_opening`); the verifier reads ``w``'s registration and produced roots at `leaf_lanes.reads`' rows and registers
    its known ports (`known_rows`, `produced.known_port`), ``o0`` among them. ``free``: output ports the prover commits
    afresh, which the public file states (``o0`` in ``ok-from-public``). ``dropped``: known input ports committed afresh
    and left out of the verifier's file too, which still says it is complete (``pub`` in ``lane3-index-from-public``),
    written beside it without ``complete`` as well. ``lane_index``: the index the lane's ``pub`` row holds, when not
    ``ln``. ``variant``: the stager and Program of ``combine-skips-p5-bit`` (``lazy``), when not ``kind``'s. The lowered
    ok is held to the word reference (`expected`) in place of `UnitStager.reference_check`, as h1_prove's stage does."""
    st = ctx.stagers[variant or kind]
    ent, roots = entries(ctx, w)
    kr = known_rows(ctx, w, kind, ln, index)
    vk = {p: PR.known_port(ctx.prog, v, row, salt, w.registration, schema=V2) for p, (v, row, salt) in kr.items()}
    reads = LL.registered_file(LL.reads(*RC, kind, ln), ent, roots, {v: V2 for v in ent},
                               {p: kp for p, (kp, _) in vk.items() if p not in dropped}, place=ctx.place)
    want = expected(kind, ln, w, ins, lane_index)
    v_ok, row_ok, salt_ok = kr[LL.OK]
    outs = {LL.OK: known(row_ok, salt_ok) if LL.OK in free else known_opening(ctx, w, v_ok, row_ok, salt_ok)}

    def check(_rows, got):
        if not np.array_equal(np.asarray(got["o0"][0], np.uint16), np.asarray(want, np.uint16)):
            raise AssertionError(f"{tag}: the lowered circuit's ok is not the word reference's")

    st.reference_check = check
    rows = {p: np.asarray([o["row"]], np.uint16) for p, o in ins.items()}
    reg = {p: {"value": o["value"], "positions": [o["position"]], "paths": [o["path"]]}
           for p, o in {**ins, **outs}.items() if "value" in o}
    t0 = time.time()
    d = ctx.work / "st" / tag
    got = st.stage(rows, d, os.urandom(32), salts={p: [o["salt"]] for p, o in {**ins, **outs}.items()}, indices=[0],
                   set_name=f"h1-lanes-{tag}", tables=rows, refs={p: [0] for p in rows}, registered=reg)
    (d / "circuit.txt").unlink(missing_ok=True)
    (d / "circuit.txt").symlink_to(st.circuit)
    if hasattr(st.low, "template"):
        (d / "unit.placed").unlink(missing_ok=True)
        (d / "unit.placed").symlink_to(st.circuit.parent / "unit.placed")
    (d / "pin.txt").write_text(st.pin)
    ports = {p.name: p for p in st.ports}
    committed = {p: U.commit_strings(ports[p], rows[p], [o["salt"]])[0] for p, o in ins.items()}
    mine = {p: o["bc"] if "bc" in o else U.commit_strings(ports[p], np.asarray([o["row"]], np.uint16), [o["salt"]])[0]
            for p, o in {**ins, **outs}.items()}
    vreads = [{p: (o["value"], reads["ports"][p]["reads"][0], o["path"]) for p, o in {**ins, **outs}.items() if "value" in o}]
    if set(vreads[0]) | (set(free) - set(dropped)) != set(reads["ports"]) or set(reads["ports"]) | set(dropped) != LL.own(*RC, kind, ln):
        raise AssertionError(f"{tag}: the opened ports {sorted(vreads[0])} are not the verifier's {sorted(reads['ports'])}")
    pub = U.verifier_pub({"meta": st.meta, "pin": st.pin, "class": st.class_sha512}, f"h1-lanes-{tag}", st.sub.id, [mine],
                         [0], vreads)
    vd = ctx.work / "verifier"
    (vd / f"{tag}.pub.bin").write_bytes(pub)
    (vd / f"{tag}.registered.json").write_text(json.dumps(reads))
    if dropped:
        (vd / f"{tag}.open.registered.json").write_text(json.dumps({x: v for x, v in reads.items() if x != LL.COMPLETE}))
    return {"tag": tag, "kind": kind, "lane": ln, "world": w.name, "stmt": "combine" if ln is None else str(ln),
            "dir": str(d), "pub": str(vd / f"{tag}.pub.bin"),
            "registered": str(vd / f"{tag}.registered.json"), "program": str(vd / f"{variant or kind}.program.json"),
            "partition": str(vd / f"{variant or kind}.partition.json"), "ands": got["ands"], "circuit_sha512": got["circuit_sha512"],
            "statement": st.meta["statement"],
            "ok": np.asarray(got["outs"]["o0"][0]).tolist(), "want": want,
            "known": {kr[p][0]: [list(op)] for p, (_, op) in vk.items()},
            "binding": {p: committed[p] == mine[p] for p in ins}, "seconds": round(time.time() - t0, 1)}


# -- the plan ---------------------------------------------------------------------------------------------------------------

CTX = None


def lane_ins(ctx, w: World, ln: int, partial: dict | None = None, pub_index: int | None = None, pub_free: bool = False) -> dict:
    """Lane ``ln``'s openings; ``pub_index``: the index its ``pub`` row holds, when not ``ln``; ``pub_free``: its ``pub``
    row committed afresh, the public file stating it."""
    t = w.regs
    ins = {f"a{s}": opening(t[LL.A], i) for s, i in enumerate(LL.lane_rows(*RC, ln))}
    ins.update({f"b{s}": opening(t[LL.B], j) for s, j in enumerate(LL.lane_cols(*RC, ln))})
    ins.update({LL.ZY: opening(t[LL.ZY], 0), LL.PARTIALS: partial or opening(w.tree, ln)})
    kr = known_rows(ctx, w, LL.LANE, ln, ln)
    for p in (LL.ADMIT, LL.PUB):
        v, row, salt = kr[p]
        row = LL.pub_words(w.leaf.salt, w.leaf.position, w.roots, pub_index) if p == LL.PUB and pub_index is not None else row
        ins[p] = known(row, salt) if p == LL.PUB and pub_free else known_opening(ctx, w, v, row, salt)
    return ins


def combine_ins(ctx, w: World, p5: dict | None = None) -> dict:
    got = {f"p{x}": opening(w.tree, x) for x in range(LL.lanes(*RC))}
    if p5 is not None:
        got["p5"] = p5
    return {**got, LL.RECEIPT: opening(w.regs[LL.RECEIPT], 0)}


def jobs(ctx) -> list[tuple]:
    """Every statement: ``(tag, kind, lane, world, negative)``."""
    n = LL.lanes(*RC)
    honest = [(f"lane{ln}", LL.LANE, ln, "honest", None) for ln in range(n)] + [("combine", LL.COMBINE, None, "honest", None)]
    return honest + [("swap-lane3", LL.LANE, 3, "swap", None), ("swap-lane4", LL.LANE, 4, "swap", None),
                     ("swap-combine", LL.COMBINE, None, "swap", None), ("lane3-reads-row4", LL.LANE, 3, "honest", "row4"),
                     ("other-leaf", LL.COMBINE, None, "honest", "p5"), ("other-leaf-in-tree-lane5", LL.LANE, 5, "mixed", None),
                     ("other-leaf-in-tree", LL.COMBINE, None, "mixed", None),
                     ("lane5-partial-off-tree", LL.LANE, 5, "mixed", "off-tree"),
                     ("receipt-mismatch", LL.COMBINE, None, "receipt", None),
                     ("lane3-index4", LL.LANE, 3, "honest", "index4"), ("ok-from-public", LL.COMBINE, None, "honest", "ok-free"),
                     ("lane3-index-from-public", LL.LANE, 3, "honest", "pub-free"),
                     ("combine-skips-p5-bit", LL.COMBINE, None, "honest", "lazy"),
                     ("other-registration-file", LL.LANE, 0, "other", None)]


def run_job(job: tuple) -> dict:
    ctx = CTX
    tag, kind, ln, wname, neg = job
    w = ctx.worlds[wname]
    index = LL.statements(*RC).index((kind, ln))
    if kind == LL.LANE:
        partial = {"row4": lambda: opening(w.tree, ln, row=4),
                   "off-tree": lambda: opening(ctx.worlds["honest"].tree, ln)}.get(neg, lambda: None)()
        pub_index = 4 if neg == "index4" else None
        pub_free = neg == "pub-free"
        return statement(ctx, kind, ln, tag, w, lane_ins(ctx, w, ln, partial, pub_index, pub_free), index,
                         lane_index=pub_index, free=(LL.PUB,) if pub_free else (), dropped=(LL.PUB,) if pub_free else ())
    ins = combine_ins(ctx, w, opening(ctx.worlds["other"].tree, 5) if neg == "p5" else None)
    return statement(ctx, kind, ln, tag, w, ins, index, free=(LL.OK,) if neg == "ok-free" else (),
                     variant="lazy" if neg == "lazy" else None)


@dataclass
class Ctx:
    work: Path
    name: str
    k: int
    position: int
    wp: int = 0
    lane: Bound | None = None
    combine: Bound | None = None
    lazy: Bound | None = None
    reg: Bound | None = None
    prog: str = ""
    stagers: dict | None = None
    worlds: dict | None = None
    place: dict | None = None


def prepare(args) -> None:
    global CTX
    t0 = time.time()
    work = Path(args.work)
    for d in ("verifier", "st", "circuits"):
        (work / d).mkdir(parents=True, exist_ok=True)
    ctx = Ctx(work, args.name, args.k, args.position)
    t = tile(args.k)
    ctx.wp = 2 * LL.words(args.k)
    win = window(t, args.k)
    ctx.lane, ctx.combine = derived(win, args.k, LL.LANE), derived(win, args.k, LL.COMBINE)
    ctx.reg = ctx.lane
    ctx.place = HW.place(win, 0, STEPS)
    if any(base for _t, base in ctx.place.values()):
        raise AssertionError(f"the one-call window's unit 0 reads its values from row 0, not {ctx.place}")
    (work / "verifier" / "window.json").write_bytes(canonical_json(encode_program(win)))
    (work / "verifier" / "rounds.json").write_text(json.dumps([{"round": s} for s in range(STEPS)]))
    ctx.prog = P.program_digest(ctx.reg.program)
    for kind, b in ((LL.LANE, ctx.lane), (LL.COMBINE, ctx.combine)):
        (work / "verifier" / f"{kind}.program.json").write_bytes(canonical_json(encode_program(b.program)))
        (work / "verifier" / f"{kind}.partition.json").write_text(json.dumps(b.part))
    print(f"programs: lane {P.program_digest(ctx.lane.program)[:16]}, combine {P.program_digest(ctx.combine.program)[:16]}, "
          f"{time.time() - t0:.0f} s", flush=True)

    salt = hashlib.sha512(SALT_TAG + args.name.encode()).digest()
    la, lb = leaf_data(args.seed, args.k, salt, args.position), leaf_data(args.seed + 1, args.k, salt, args.position + 1)
    regs, roots, receipt = register(ctx.prog, la, ctx.place)
    pa = partials(la, roots)
    xor = [0] * (ctx.wp // 2)
    for p in pa:
        xor = [x ^ v for x, v in zip(xor, u32_words(p))]
    assert xor == [v & 0xFFFFFFFF for v in receipt], "the honest partials XOR to the receipt"
    A = world("honest", ctx, la, regs, roots, pa)
    regs_b, roots_b, _ = register(ctx.prog, lb, ctx.place)
    B = world("other", ctx, lb, regs_b, roots_b, partials(lb, roots_b))
    swapped, mixed = list(pa), list(pa)
    swapped[3], swapped[4] = pa[4], pa[3]
    mixed[5] = list(B.tree.rows[5])
    regs_r = {**regs, LL.RECEIPT: receipt_reg(ctx.prog, receipt, ctx.place, flip=3 * 32 + 17)}
    ctx.worlds = {"honest": A, "other": B, "swap": world("swap", ctx, la, regs, roots, swapped),
                  "mixed": world("mixed", ctx, la, regs, roots, mixed), "receipt": world("receipt", ctx, la, regs_r, roots, pa)}
    for nm in ("swap", "mixed"):
        assert ctx.worlds[nm].registration == A.registration, "only the partials differ"
    assert A.reg_record["program"] == ctx.prog, "the registration's program is the one the partials' entry binds"
    for nm, w in ctx.worlds.items():
        (work / "verifier" / f"{nm}.registration.json").write_text(json.dumps(w.reg_record))
        (work / "verifier" / f"{nm}.produced.json").write_text(json.dumps(w.record))
    mine = {LL.PARTIALS: PR.entry(ctx.prog, LL.PARTIALS, LL.lanes(*RC), ctx.wp, A.registration)}
    produced_check = {nm: list(PR.check(w.record, A.registration, A.draw, mine)) for nm, w in ctx.worlds.items()}
    print(f"registered and produced: {time.time() - t0:.0f} s; produced records' codes against the honest registration: "
          f"{produced_check}", flush=True)
    row = A.tree.rows[LAZY_ROW]
    bit = [(int(row[q // 16]) >> (q % 16)) & 1 for q in range(16 * ctx.wp)]
    j = len(bit) - 1
    i = next(q for q in range(j) if bit[q] == bit[j])
    ctx.lazy = bound("H1LanesLazyCombineStatement", lazy_combine(Z.combine(t), LAZY_ROW, j, i))
    (work / "verifier" / "lazy.program.json").write_bytes(canonical_json(encode_program(ctx.lazy.program)))
    (work / "verifier" / "lazy.partition.json").write_text(json.dumps(ctx.lazy.part))
    print(f"combine-skips-p5-bit: bit {j} of p{LAZY_ROW} unread, bit {i} read in its place", flush=True)
    ctx.stagers = {kind: stager(ctx, kind) for kind in (LL.LANE, LL.COMBINE)}
    ctx.stagers["lazy"] = stager(ctx, LL.COMBINE, "lazy")
    circuits = {kind: {"statement": st.meta["statement"], "layout": st.meta.get("layout"), "pin": st.pin,
                       "class": st.class_sha512, "ands": st.low.layout.counts["ands"], "bytes": st.circuit.stat().st_size}
                for kind, st in ctx.stagers.items()}
    print(f"circuits: {json.dumps(circuits)}, {time.time() - t0:.0f} s", flush=True)
    CTX = ctx

    js = jobs(ctx)
    with mp.get_context("fork").Pool(args.jobs) as pool:
        recs = list(pool.imap_unordered(run_job, js))
    by = {r["tag"]: r for r in recs}
    for tg, of in OPEN.items():
        by[tg] = {**by[of], "tag": tg, "registered": by[of]["registered"].replace(".registered.json", ".open.registered.json")}
    for tg, (of, port) in UNMARKED.items():
        reads = json.loads(Path(by[of]["registered"]).read_text())
        if reads["ports"][port].pop(LL.READ_IN_FULL, None) is not True:
            raise AssertionError(f"{of}: the verifier's file does not mark {port} {LL.READ_IN_FULL}")
        path = work / "verifier" / f"{tg}.registered.json"
        path.write_text(json.dumps(reads))
        by[tg] = {**by[of], "tag": tg, "registered": str(path)}
    for tg, of in HELD.items():
        by[tg] = {**by[of], "tag": tg}
    for tg, r in by.items():
        r["holds"] = HOLDS.get(tg, r["world"])
    honest = [jb[0] for jb in js if jb[0] not in EXPECT]
    proved = [jb[0] for jb in js]
    checks = proved + list(OPEN) + list(UNMARKED) + list(HELD)
    vd = work / "verifier"
    with open(work / "plan.tsv", "w") as f:
        for tg in proved:
            f.write(f"S\t{tg}\t{by[tg]['dir']}\n")
        for tg in checks:
            r = by[tg]
            f.write(f"C\t{tg}\t{r['dir']}\t{r['pub']}\t{r['registered']}\t{r['program']}\t{r['partition']}\t"
                    f"{EXPECT.get(tg, 'ACCEPT')}\t{vd / (r['holds'] + '.registration.json')}\t"
                    f"{vd / (r['holds'] + '.produced.json')}\t{vd / 'window.json'}\t{vd / 'rounds.json'}\t0\t{r['stmt']}\t"
                    f"{r['statement']}\n")
    plan = {"checks": {tg: {**by[tg], "expect": EXPECT.get(tg, "ACCEPT")} for tg in checks}, "honest": honest}
    (work / "plan.json").write_text(json.dumps(plan, indent=1))
    summary = {"k": args.k, "tile": U.tile(), "lanes": LL.lanes(*RC), "words": LL.words(args.k),
               "programs": {kind: P.program_digest(b.program)
                            for kind, b in (("lane", ctx.lane), ("combine", ctx.combine), ("lazy", ctx.lazy))},
               "place": ctx.place, "registration": A.registration, "draw": A.draw, "produced": A.record, "produced_check": produced_check,
               "forged_partial": ctx.worlds["mixed"].tree.bcs[5].hex(),
               "ands": {"lane": by["lane0"]["ands"], "combine": by["combine"]["ands"]}, "circuits": circuits,
               "ok": {tg: by[tg]["ok"] for tg in checks}, "binding": {tg: by[tg]["binding"] for tg in checks},
               "seconds": round(time.time() - t0, 1)}
    (work / "prepare.json").write_text(json.dumps(summary, indent=1))
    print("PREPARED " + json.dumps({x: summary[x] for x in ("k", "tile", "ands", "ok", "produced_check", "seconds")}),
          flush=True)


def _verdict(p: Path) -> dict | None:
    if not p.exists():
        return None
    v = [json.loads(x[8:]) for x in p.read_text(errors="replace").splitlines() if x.startswith("VERDICT ")]
    if not v:
        return None
    return {**v[0], "accepted": all(x.get("accepted") for x in v), "sessions": len(v),
            "why": next((x.get("why", "") for x in v if not x.get("accepted")), "")}


def _own(p: Path) -> dict | None:
    """`pouw-leaf-own`'s verdict on a check's registered file."""
    if not p.exists():
        return None
    v = [json.loads(x[4:]) for x in p.read_text(errors="replace").splitlines() if x.startswith("OWN ")]
    return v[0] if v else None


def _time(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def summarize(args) -> int:
    work, out = Path(args.work), Path(args.out)
    plan, prep = json.loads((work / "plan.json").read_text()), json.loads((work / "prepare.json").read_text())
    rows, opened, verdicts, known = {}, {}, {}, {}
    for tag, c in plan["checks"].items():
        v = _verdict(out / "lean" / f"{tag}.out")
        own = _own(out / "lean" / f"{tag}.own")
        err = out / "lean" / f"{tag}.err"
        templated = v is not None and c.get("statement") == TMPL_STATEMENT and not v["accepted"] and v["why"] == TMPL_WHY
        flock = "NONE" if v is None else "ACCEPT" if v["accepted"] else TEMPLATED if templated else "REJECT"
        passed = flock in ("ACCEPT", TEMPLATED)
        word = "NONE" if v is None or own is None else (flock if passed and own["accepted"] else "REJECT")
        why = (v or {}).get("why") or (err.read_text(errors="replace").strip().splitlines()[-1:] if err.exists() else "")
        if own is not None and not own["accepted"]:
            why = own.get("why", "")
        met = word == c["expect"] or (word == TEMPLATED and c["expect"] == "ACCEPT")
        rows[tag] = {"expect": c["expect"], "lean": word, "as_expected": met and WHY.get(tag, "") in str(why),
                     "why": why, "flock_verify": flock, "statement": c.get("statement"), "leaf_own": own, "holds": c["holds"],
                     "ok": c["ok"], "ands": c["ands"], "prove": _time(out / tag / "prove.time"),
                     "lean_time": _time(out / "lean" / f"{tag}.time"), "registered": (v or {}).get("registered")}
        if tag in plan["honest"] and word in ("ACCEPT", TEMPLATED):
            # `produced.shared` reads only an accepted verdict's openings; a templated one's are those Lean reported
            # before refusing the form, held to the produced record here and never reported as accepted
            verdicts[tag] = {**v, "accepted": True, "of_record": True}
            known.update(c["known"])
            for e in (v.get("registered") or {}).values():
                if opened.setdefault(e["value"], e["root"]) != e["root"]:
                    opened[e["value"]] = None
    honest_record = prep["produced"]
    opened_codes = list(PR.opened(honest_record, opened))
    want = PR.wanted(honest_record, known)
    shared_codes = list(PR.shared(want, verdicts))
    forged = json.loads(json.dumps(verdicts.get("lane5") or {}))
    if forged:
        forged["openings"][LL.PARTIALS] = [[5, prep["forged_partial"]]]
    forged_codes = list(PR.shared(want, {"lane5": forged})) if forged else ["no lane5 verdict"]
    ok = (all(r["as_expected"] for r in rows.values()) and not opened_codes and not prep["produced_check"]["honest"]
          and len(verdicts) == len(plan["honest"]) and not shared_codes and forged_codes == ["P-shared"])
    summary = {"ok": ok, "k": prep["k"], "tile": prep["tile"], "ands": prep["ands"], "circuits": prep.get("circuits"),
               "lean_accepts": sum(r["lean"] == "ACCEPT" for r in rows.values()),
               "templated": sorted(t for t, r in rows.items() if r["lean"] == TEMPLATED),
               "produced_check": prep["produced_check"],
               "opened": opened, "opened_codes": opened_codes, "shared_codes": shared_codes,
               "shared_forged_lane5": forged_codes, "checks": rows, "registration": prep["registration"],
               "draw": prep["draw"], "produced": honest_record, "binding": prep["binding"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for tag, r in rows.items():
        why = "" if r["lean"] == TEMPLATED else str(r["why"])[:160]
        print(f"{tag:26s} {r['lean']:7s} expect {r['expect']:7s} ok={r['ok']} {'ok' if r['as_expected'] else 'MISMATCH'}  {why}")
    print(f"opened roots against the produced record: {opened_codes or 'ok'}; opened rows against its openings: "
          f"{shared_codes or 'ok'}, a forged lane-5 opening: {forged_codes}; ANDs {prep['ands']}; Lean ACCEPTs "
          f"{summary['lean_accepts']}, {TEMPLATED}: {len(summary['templated'])}; all as expected: {ok}")
    return 0 if ok else 1


def cost(args) -> None:
    """A lane's and the combine's ANDs at K (`--k`, 4096 the served one), each its statement's lowering
    (`class_statement.unit_lowering`, structural under ``H1_TMPL=structural``, as `h1unit.stager` lowers it), and a leaf's: 32
    lanes and the combine."""
    from verity_flock import class_statement as CS

    t = tile(args.k)
    n = LL.lanes(*RC)
    lane = ([(f"a{s}", [f"a[{s}]"]) for s in range(LL.LANE_ROWS)] + [(f"b{s}", [f"b[{s}]"]) for s in range(LL.LANE_COLS)]
            + [(LL.ADMIT, [LL.ADMIT]), (LL.ZY, [LL.ZY]), (LL.PARTIALS, [LL.PARTIALS]), ("pub", list(LL.PUBLIC))])
    units = {"combine": (Z.combine(t), [(f"p{x}", [f"{LL.PARTIALS}[{x}]"]) for x in range(n)] + [(LL.RECEIPT, [LL.RECEIPT])]),
             "lane": (Z.lane(t), lane)}
    out = {"k": args.k, "tile": U.tile(), "lanes": n, "words": LL.words(args.k)}
    for name, (d, rows) in units.items():
        t0 = time.time()
        low, _words = CS.unit_lowering(d, f"unit/pouw-h1-{name}", rows, (), structural=U.layout() is not None)
        out[name] = {"ands": low.layout.counts["ands"], "seconds": round(time.time() - t0, 1)}
        print(f"{name}: {out[name]}", flush=True)
    out["leaf_ands"] = n * out["lane"]["ands"] + out["combine"]["ands"]
    print("COST " + json.dumps(out), flush=True)
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "cost.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "summarize", "cost"))
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=448)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--position", type=int, default=5)
    ap.add_argument("--name", default="h1-lanes-local")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a)
    elif a.cmd == "cost":
        cost(a)
    else:
        sys.exit(summarize(a))


if __name__ == "__main__":
    main()
