"""The H1 tile unit as a C-Flock statement, and the window's commitments it is opened against: what the committer, the
prover and the verifier of `run.py` share.

* **The unit** is `circuit.h1zy.PouwH1TileZy` over `circuit.h1tile.PouwH1Tile{EXACT=GemmAccumulatorS8_v2{K, sm120}, R, C,
  PAIRS=the full grid}` (`h1window.tile_unit`): the H1 tile on its rows' codes, with its z/y record checked.
  The statement's Definition is the window's template at K (`Shape.definition`, `h1window.template`: one call of the unit,
  its A and B rows ``a_k<K>`` and ``b_k<K>``). Its statement (`stager`): input rows ``a0..a{R-1}`` (``a_k<K>[i]``, `ports`)
  and ``b0..b{C-1}``, one ``hm96-sha512/row/v2`` bit row per A row and B row (8 K + 32 bits: the int8 codes' bytes, then the
  row's f32 scale, s_a for an A row and s_w for a B row, `h1zy.ROW`); ``zy``, the unit's record (48 R C bits: its cells'
  s32 z, row-major, then their bf16 y, `h1zy.RECORD`, what the served ``scaled_mm_zy`` writes); the public row ``pub``
  (``salt || position || roots``, 1600 bits: the window's salt, the unit's position and the window's roots; the salt is
  PoUW's window salt `pouw.salt.pouw_salt`, `unit_salt`); and two output rows, ``receipt`` (o0: CP-W's 2 (KS + m) words)
  and ``ok`` (o1: one word, 1 exactly when the record is the tile's z and its dequant), each one ``row/v2`` row of exactly
  its words. z is no output: the record, an input, carries it.
* **The public row** stands for public input ports, which the Lean verifier refuses as a form its soundness proof does
  not cover (`ProvedScope.statementRules`). Its hm96 salt is public (`pub_salt`), so the verifier computes the row's
  ``b || c`` from its own salt, position and roots, and its public file binds the proof to them; its content is public, so
  the public salt hides nothing that needed hiding.
* **The ok row** is a row of known content, registered by the verifier: the value ``ok`` (`OK_VALUE`, `register_ok`),
  row u unit u's, is the one word 1 (`OK`) under a public salt (`ok_salt`) at every unit of the window, and the unit's
  ``ok`` port (o1) is a port of the verifier's ``--registered`` file (`registered_reads`), named at its known content
  (`known_ok`: flock-verify computes ``ok``'s root from those rows and refuses the file at another, `check_known`), so
  Lean opens each instance's ok row against that root at the unit's position, and a proof it accepts is one whose unit
  computed 1
  (`Accounting.Guarantees.TS.CheckedSessionSound`'s port named ``ok`` in the circuit and in ``own``). A record that is
  not the tile's product and its dequant makes the unit compute 0, whose row opens no registered root, and a prover that
  commits 1 anyway commits a wrong witness.
* **The window's commitments** are those rows' ``b || c`` (hm96-sha512 under the committer's salts) as frame-v3-sha512
  leaves (`rowleaf.row_leaf`) of the trees a served round commits before its coin (`h1window.table`), the weights', and
  the receipts': each call c's A table ``a/<K>@<c>``, the slot's `slot` rows (the activations every call reads, each
  with its s_a, the token rows padded with zero rows of scale 0) under the call's own salts, and its z table ``zy@<c>``,
  one record a tile of the call in its tile order; ``b/<K>``, every call's B rows (the model's, each with its s_w), call
  after call; and ``receipt`` over the window's tiles. The tables and ``b/<K>`` are registered values (`tables`,
  `register`: `registered.Registered.commit` under the window Program's digest, ``row/v2`` rows of 16-bit leaves,
  `value_words` a row; each table's domain, and ``ok``'s, binds the window's open root, `open_root`, and b/<K>'s none, as
  it is held across windows: `credit.domain_of`), so the Lean verifier opens every row a statement reads
  against their roots; ``receipt`` is the
  window's own tree (`domain`), opened outside the proof. Every unit's ``roots`` input is (the A tables' digest, root(b))
  (`window_roots`). `commit_strings` computes a row's ``b || c`` with C-Flock's own function on the port the statement's
  META declares (`stage.py` holds the two equal), so a proof's committed rows are the window's leaves exactly when its
  rows and salts are the committed ones.
* **The window's public Program** (`h1window.program`, `Shape.window`): every tile of the window as one call of the unit
  (the smoke's `TILE`, 4 × 8), each call its own linear at K, every call reading the one A block (the slot); unit u is
  the tile `window.py` numbers u, tile t of its call c, and its outputs are row u of ``receipt`` and ``ok``. Its
  `Q_template_instances` partition derives the units, and the statement binds it (`Shape.stager`'s ``bind``), so the
  Lean verifier derives each instance's unit from its own copy of the program (``--program``, ``--partition``) and the
  indices the public file states. `unit_reads` is the row each of the unit's ports reads (`h1window.table_reads`): its A
  rows at call c's A table, its record at row t of call c's z table, its B rows at ``b/<K>``. The slot is the window's
  size class and the calls' B rows the model's, so the layout is protocol structure and the real token count stays the
  prover's. A registered port reads one value, so a statement holds one call's units.
* **The verifier's public file** (`verifier_pub`) is built from the opened leaves and the verifier's own public and ok
  rows alone: a shared-row file (`circuit.write`'s), its header the statement's (its circuit pin, its units' global
  indices, each port's frame-v3 root over its table's ``b || c``) with each registered port's table the registered rows its
  units read (``registered``: their positions, the verifier's own, and the opened paths), then the ``b || c`` and each
  unit's refs. Lean ``verify --zk`` holds a session to that file, so it accepts only a proof whose committed rows are the
  window's, whose salt, position and roots are the verifier's and whose ok is 1; with ``--registered``
  (`registered_reads`, the verifier's own reads of one call's units from its Program) it also opens each A row, B row,
  record and ok row against its registered root at the position the verifier's Program reads.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2, row_leaf
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.core.protocols.sampled_proofs.one_stage import registered as G

HM96 = hm96.SHA512
SALT_BYTES = 192
M32 = (1 << 32) - 1
#: the public row's parameters, in its bit order
PUBLIC = HW.PUBLIC
PUB_BITS = CC.PUBLIC_ROW_BITS
#: STAND-INS for what the network certifier and the window's registration give a certified window, which this harness
#: runs without: the certifier's reveal(p) (`stand_in_reveal`), the registered stream, and the registered prover's id
STAND_IN_REVEAL_TAG = b"verity/pouw/h1-prove/stand-in-reveal/v0\0"
STREAM = hashlib.sha512(b"verity/pouw/h1-prove/stand-in-stream/v0").digest()
PROVER = PS.prover_id(b"verity/pouw/h1-prove/stand-in-prover-key/v0")
PUB_SALT_TAG = b"verity/pouw/h1-prove/pub-row-salt/v0\0"
OK_SALT_TAG = CC.OK_SALT_TAG
WINDOW_TAG = "verity/pouw/h1-prove/window/v0"
A_TABLES_TAG = b"verity/pouw/h1-prove/a-tables/v0\0"
STAND_IN_REQUESTS_TAG = b"verity/pouw/h1-prove/stand-in-requests/v0\0"
#: the ok row the verifier accepts: the unit's one output bit, 1, in its one word
OK = np.array(CC.OK_ROW, np.uint16)
#: the value the verifier registers its ok rows as: the window Program's output ``ok`` (`h1window.unit_reads`), row u unit u's
OK_VALUE = "ok"


def tile() -> str:
    """The H1 tile under the unit: ``H1_TILE`` v1 (`h1tile.PouwH1Tile`, z the exact A Bᵀ) or v3 (`h1tile.PouwH1Tile3`, the
    one stream: z = O' - sum_d 128^d D_d of the folded words, with its wrap correction and dither peel; A Bᵀ too)."""
    import os
    t = os.environ.get("H1_TILE", "v1")
    if t not in ("v1", "v3"):
        raise ValueError(f"H1_TILE={t!r}: v1 or v3")
    return t


#: the structural statement's layout mode under ``H1_TMPL=structural`` (struct_lower's costs and the structural verdicts
#: are measured at it)
LAYOUT = "iv/hybrid/4"
#: ``H1_TMPL``'s values for the flat statement (also unset or empty) and for the structural one at `LAYOUT`
FLAT, STRUCTURAL = "flat", "structural"


def layout() -> str | None:
    """The statement's form: ``H1_TMPL`` unset, empty or ``flat``, None: the flat statement (``flock-circuit``, the unit's
    netlist in the file), the one the Lean verifier accepts; ``structural``, the templated statement at `LAYOUT`
    (``flock-circuit/tmpl``, `circuit.compose`'s ``layout``), the unit as `struct_lower`'s template (`stager`); another
    layout mode (`verity_flock.tmpl.mode_of`), the templated statement at it. The Lean verifier refuses a templated
    statement, after every other check, until its proof lands (`Tags.templated`), so it stays opt-in."""
    import os
    from verity_flock import tmpl as TM
    m = os.environ.get("H1_TMPL") or FLAT
    if m == FLAT:
        return None
    if m == STRUCTURAL:
        return LAYOUT
    TM.mode_of(m)
    return m


def stager(definition, name: str, in_rows, out_rows, layout: str | None, **kw):
    """``definition``'s `UnitStager` (``kw`` its ``circuit``, ``bind`` and ``public_inputs``): with ``layout`` the
    templated statement of its structural lowering (`struct_lower`, ``structural``), the flat unit never built; with
    None, or where struct_lower refuses the Definition (its NotImplementedError: assertions or table reads, or a
    specialization of 2^32 columns), the flat statement. The H1 lane, combine and tile have neither at any K a block holds
    (K ≤ 4096 served; past K ≈ 5,041 the lane fits no 2^27 block in either form, and compose refuses it)."""
    import sys
    from verity_flock import class_statement as CS
    if layout is not None:
        try:
            return CS.UnitStager(definition, name, in_rows, out_rows, layout=layout, structural=True, **kw)
        except NotImplementedError as e:
            print(f"{name}: the flat statement: struct_lower refuses {definition.id} ({e})", file=sys.stderr, flush=True)
    return CS.UnitStager(definition, name, in_rows, out_rows, **kw)


def stand_in_reveal(coin: bytes) -> bytes:
    """A STAND-IN for the network certifier's salt for the window, its ``reveal(p)`` (`pouw.salt.revealed`), which this
    harness has no certifier to give: SHA-512 of the epoch's opened salt coin (`pouw.coins`) under this harness's tag.
    Like the certifier's salt it is the auditor's, drawn before the window and opened only after the window's roots are
    recorded; unlike it, no certifier signs it or bounds when it is revealed."""
    return hashlib.sha512(STAND_IN_REVEAL_TAG + bytes(coin)).digest()


def unit_salt(reveal: bytes, window: int, stream: bytes = STREAM, prover: bytes = PROVER) -> bytes:
    """`pouw.salt`, the units' one coin: PoUW's window salt, `pouw.salt.pouw_salt(reveal, window, stream, prover)`, here of
    the stand-in reveal (`stand_in_reveal`), the epoch's number as the window, and the stand-in stream and prover."""
    return PS.pouw_salt(reveal, window, stream, prover)


def pub_salt(window: str, position: int) -> bytes:
    """The public row's hm96 salt, which anyone derives from the window and the unit's position."""
    return hashlib.shake_256(PUB_SALT_TAG + window.encode() + b"\0" + int(position).to_bytes(8, "big")).digest(SALT_BYTES)


def ok_salt(window: str, position: int) -> bytes:
    """The ok row's hm96 salt, which anyone derives from the window and the unit's position: the row's content is the one
    the verifier accepts (`OK`), so the salt hides nothing: `core_check.ok_salt`, which credit registers served ok rows
    under."""
    return CC.ok_salt(window, position)


#: the smoke's R × C tile (the served path's is 32 × 32, `pouw_h1.TILE`)
TILE = (4, 8)


def slot(rows: int, tile: tuple[int, int] = TILE) -> int:
    return HW.slot(rows, tile)


def window_shape(com: dict) -> tuple[tuple[tuple[str, int], ...], int]:
    """A window commitment's public shape: its calls' names and B rows, and its A slot."""
    return tuple((c["name"], int(c["n"])) for c in com["calls"]), int(com["slot"])


def tables(com: dict) -> dict[str, int]:
    """A window commitment's registered values and their rows, in registration order: the weights' ``b/<K>`` (every
    call's B rows), then each call c's A table ``a/<K>@<c>`` (the slot's rows) and z table ``zy@<c>`` (one record a tile
    of the call)."""
    k = com["unit"]["k"]
    out = {f"b/{k}": int(com["b_rows"])}
    for c, call in enumerate(com["calls"]):
        out[f"a/{k}@{c}"], out[f"zy@{c}"] = int(com["slot"]), int(call["tiles"])
    return out


def steps(program) -> int:
    """The window's steps as `h1window.table` counts them: h1_prove serves no rounds, so each call is a step of its own
    and its step's A table ``a/<K>@<c>`` holds its slot alone."""
    return len(HW.call_units(program))


def program_tables(program, k: int) -> dict[str, int]:
    """`tables` as the verifier's own window Program gives them: its B rows, and each call's A table (`h1window.table`
    at `steps`, its slot of M rows) and z table (its tiles)."""
    S, sig = program.fn.S(), dict(program.fn.signature()[0])
    out = {f"b/{k}": sig[f"b/{k}"].n}
    for c, (_u0, n) in enumerate(HW.call_units(program)):
        out[HW.table(program, "a", c, steps(program))], out[HW.table(program, "zy", c)] = S.M, n
    return out


def registered_of(com: dict) -> dict[str, tuple[int, str]]:
    """A commitment's registered values as its entries name them, each its rows and root (`run.check_registration` holds
    the entries to the verifier's own Program)."""
    return {e["value"]: (int(e["leaves"]), e["root"]) for e in com["registered"]}


def open_root(window: str, weights_root: bytes) -> str:
    """The window's open root (`pouw.salt.window_open_root`, hex), which every per-window registered value's domain binds
    (`credit.domain_of`): of the weights' root, root(b/<K>), a STAND-IN for the round-0 request table's root
    (SHA-512 of the window's name under `STAND_IN_REQUESTS_TAG`: this harness serves no requests) and the window's name as
    its session. A served window's is the auditor's ledger's, which opens the challenger's session on it; this harness has
    no ledger, so anyone computes it from the window's name and its b/<K> root, and the commitment that records it is
    fixed before the salt coin."""
    return PS.window_open_root(weights_root, hashlib.sha512(STAND_IN_REQUESTS_TAG + window.encode()).digest(), window).hex()


def window_open_root(com: dict) -> str:
    """`open_root` of a window commitment: its name and its registered b/<K> root."""
    return open_root(com["window"], bytes.fromhex(registered_of(com)[f"b/{com['unit']['k']}"][1]))


def window_roots(com: dict) -> tuple[bytes, bytes]:
    """A window commitment's roots, every unit's ``roots`` input: SHA-512 of its calls' A table roots in call order (under
    `A_TABLES_TAG`), and root(b/<K>). The z tables are in neither, so a changed record moves no receipt."""
    k, reg = com["unit"]["k"], registered_of(com)
    a = hashlib.sha512(A_TABLES_TAG + b"".join(bytes.fromhex(reg[f"a/{k}@{c}"][1]) for c in range(len(com["calls"]))))
    return a.digest(), bytes.fromhex(reg[f"b/{k}"][1])


#: the statement's output rows, by `h1window.unit_reads`' keys
OUT_PORTS = {"receipt": "o0", "ok": "o1"}


def ports(template) -> dict[str, str]:
    """The statement's port of each row a unit of ``template`` (the window's at its K, `h1window.template`) reads, by
    `h1window.unit_reads`' and `registered.reads`' keys (its parameters' ``a_k<K>[i]``, ``b_k<K>[j]`` and ``zy``,
    `h1window.row_key`): A row i's ``a<i>``, B row j's ``b<j>``, the record's ``zy``."""
    (a, at), (b, bt), (zy, _t) = template.signature()[0][:3]
    return {**{HW.row_key(a, at.n, i): f"a{i}" for i in range(at.n)}, **{HW.row_key(b, bt.n, j): f"b{j}" for j in range(bt.n)},
            zy: "zy"}


def window_ports(program) -> dict[str, str]:
    """`ports` of the window ``program``'s template: the statement is one K's circuit, so a window of two K's is refused."""
    tiles = program.fn.bindings["TILES"]
    if len(tiles) != 1:
        raise ValueError(f"h1_prove's statement is one K's circuit; the window has K = {[k for k, _t in tiles]}")
    return ports(tiles[0][1])


def unit_reads(program, tp, u: int) -> dict[str, tuple[str, int]]:
    """`h1window.table_reads` by the statement's ports (`window_ports`, `OUT_PORTS`; ``salt``, ``position`` and ``roots``
    the public row's): each registered port's value (its call c's ``a/<K>@<c>`` or ``zy@<c>``, the weights' ``b/<K>``)
    and its row there, and the window's parameter or output each other port reads."""
    port = {**window_ports(program), **OUT_PORTS}
    return {port.get(key, key): (param, row) for key, (param, row) in HW.table_reads(program, tp, u, steps(program)).items()}


# -- the registered values: each call's tables and the weights' b/<K> -------------------------------------------------------


def value_words(name: str, tile: tuple[int, int] = TILE) -> int:
    """A registered value's row width in 16-bit leaves: an A table ``a/<K>@<c>`` and ``b/<K>`` K/2 + 2 (`h1zy.ROW`), a z
    table ``zy@<c>`` 3 R C (`h1zy.RECORD`), ``ok`` one (`OK`)."""
    if name == OK_VALUE:
        return len(OK)
    return 3 * tile[0] * tile[1] if name.startswith("zy") else int(name.split("/")[1].split("@")[0]) // 2 + 2


def bound_root(name: str, window_root: str) -> str | None:
    """The window root the registered value ``name``'s domain binds: none for the weights' b/<K>, held across windows
    (`credit.is_held`), and ``window_root``, the window's open root (`open_root`), for every other value."""
    return None if CR.is_held(name) else window_root


def register(program: str, name: str, words: np.ndarray, salts, window_root: str) -> G.Registered:
    """A window's rows of the registered value ``name`` (its bit rows' u16 words, `row_words` or `record_words`), committed
    under the window Program's digest ``program`` in the tile's ports' format (``row/v2``, 16-bit leaves) with one of
    ``salts`` a row, its domain binding the window's open root ``window_root`` unless ``name`` is b/<K> (`bound_root`)."""
    return G.Registered.commit(program, name, np.asarray(words).tolist(), [bytes(s) for s in salts], G.WORD_BITS, V2,
                               bound_root(name, window_root))


def register_ok(program: str, window: str, population: int, window_root: str) -> G.Registered:
    """The verifier's registration of the ok rows of ``window``'s ``population`` units (`OK_VALUE`): row u the one word 1
    (`OK`) under unit u's public `ok_salt`, under the window Program's digest ``program`` and the window's open root
    ``window_root`` (`register`: ok is the window's own, one row a unit). Every input is public, so the verifier and the
    prover compute the same root, and the verifier's is the one it opens against."""
    return register(program, OK_VALUE, np.tile(OK, (int(population), 1)), [ok_salt(window, u) for u in range(int(population))],
                    window_root)


def known_ok(window: str, population: int) -> dict:
    """The known content of ``window``'s ``ok`` rows as flock-verify's ``--registered`` file names it under a port's
    ``known`` key (`Registered.KNOWN_KEY`, `Flock/RegKnown.lean`): row u the one word 1 (`OK`) and its public `ok_salt`, so
    the verifier computes ``ok``'s root itself and refuses a file whose ``ok`` root is another (
    `Accounting.Guarantees.TS.HeldSessionSound`)."""
    return {"rows": [OK.tolist()] * int(population), "salts": [ok_salt(window, u).hex() for u in range(int(population))]}


def check_known(program: str, window_root: str, port: str, entry: dict) -> str | None:
    """Why a ``--registered`` port entry named at known content (``entry["known"]``, `known_ok`) is refused, or None: its
    rows are ``row/v2`` rows of 16-bit leaves, as many as its leaves, each of its words under a 192-byte salt, and its root
    is the root of their registration (`register` under the window Program's digest ``program`` and the window's open root
    ``window_root``): `Registered.checkKnown`, flock-verify's check of ``--registered``, which roots them at the entry's
    domain as given, where this also holds that domain to the window's."""
    rows, salts = entry["known"]["rows"], entry["known"]["salts"]
    schema, bits = entry.get("row_schema"), entry.get("word_bits", G.WORD_BITS)
    if schema != V2 or bits != 16:
        return (f"registered reads: port {port}: known content is of {V2} rows of 16-bit leaves, and {entry['value']}'s "
                f"are {schema} rows of {bits}-bit leaves")
    if len(rows) != entry["leaves"] or len(salts) != len(rows):
        return f"registered reads: port {port}: {len(rows)} known rows of {entry['value']}'s {entry['leaves']}"
    if any(len(r) != entry["words"] or any(not 0 <= w < 1 << 16 for w in r) for r in rows) or \
            any(len(bytes.fromhex(s)) != SALT_BYTES for s in salts):
        return f"registered reads: port {port}: a known row is not {entry['words']} u16 words under a {SALT_BYTES}-byte salt"
    reg = register(program, entry["value"], np.array(rows, np.uint16), [bytes.fromhex(s) for s in salts], window_root)
    if reg.root.hex() != entry["root"] or reg.domain.domain_id.hex() != entry["domain"]:
        return (f"registered reads: port {port}: {entry['value']}'s registered root is not the root of its {len(rows)} "
                f"known rows")
    return None


def registered_domain(program: str, name: str, leaves: int, window_root: str) -> CommitmentDomain:
    return G.domain(program, name, value_words(name), int(leaves), window=bound_root(name, window_root))


def registered_levels(program: str, name: str, bcs: list[bytes], window_root: str):
    """The registered tree of ``name``'s rows ``b || c`` (`register`'s, from its ``bcs`` alone)."""
    dom = registered_domain(program, name, len(bcs), window_root)
    return tree_levels(dom, [G.leaf(dom, i, bc, V2) for i, bc in enumerate(bcs)])


def verify_registered(program: str, name: str, leaves: int, root: bytes, index: int, bc: bytes, sibs: list[str],
                      window_root: str) -> bool:
    """``bc`` is row ``index`` of the registered value ``name`` under ``root``, in the window whose open root is
    ``window_root``: the check flock-verify ``--registered`` makes of each table row (`registered.verify_read`)."""
    return G.verify_read(registered_domain(program, name, leaves, window_root), root, index, bc,
                         bytes.fromhex("".join(sibs)), V2)


def registered_reads(program, tp, call: int, roots: dict[str, str], window: str, window_root: str,
                     prog: str | None = None) -> dict:
    """The verifier's ``--registered`` file of a statement over call ``call``'s units (flock-verify ``verify --zk
    --registered``, `h1window.table_reads_file`): for each registered port (`window_ports`), the value it reads (the
    call's A table or z table, or the weights' b/<K>), its root (``roots`` ``{value: root}``, the window's registration),
    rows, width and domain (the tables' binding the window's open root ``window_root``, b/<K>'s none), and the row each
    unit of the call reads in the verifier's own Program (``program``, ``tp`` its template partition; ``prog`` its
    digest), None at every other unit; and the ok port (o1), which reads the verifier's own ``ok`` (`register_ok`, its
    root ``roots[OK_VALUE]``, its domain binding ``window_root`` too) at the unit's row of the Program's output, named at
    its known content (`known_ok` of the window ``window``), which flock-verify holds the root to. The one place h1_prove
    derives it."""
    if OK_VALUE not in roots:
        raise ValueError(f"the verifier registers its ok rows ({OK_VALUE}) beside the window's values, and holds no root of them")
    prog = prog or P.program_digest(program)
    doc = HW.table_reads_file(program, tp, call, roots, ports=window_ports(program), prog=prog, window=window_root,
                              steps=steps(program))
    u0, n, T, o1 = *HW.call_units(program)[call], tp.population, OUT_PORTS["ok"]
    rows = [None] * n
    for u in range(u0, u0 + n):
        value, rows[u - u0] = unit_reads(program, tp, u)[o1]
        if value != OK_VALUE:
            raise ValueError(f"unit {u}'s ok port reads {value}, not the Program's output {OK_VALUE}")
    doc["ports"][o1] = {"value": OK_VALUE, "root": roots[OK_VALUE], "leaves": T, "words": len(OK), "word_bits": G.WORD_BITS,
                        "domain": registered_domain(prog, OK_VALUE, T, window_root).domain_id.hex(),
                        **G.schema_entry(G.SCHEMA_V2),
                        **CC.call_reads(u0, rows), "known": known_ok(window, T)}
    return doc


def table(positions) -> tuple[list[int], list[int], list[int]]:
    """A registered port's shared-row table over its instances' positions (`registered.table`): the table's positions,
    each instance's ref into it, and the first instance that reads each table row."""
    tpos, refs = G.table([int(p) for p in positions])
    first: dict[int, int] = {}
    for i, r in enumerate(refs):
        first.setdefault(r, i)
    return tpos, refs, [first[t] for t in range(len(tpos))]


def one_value(port: str, values) -> str:
    """The one registered value a statement's port reads at all its instances: a port reads one value, so a statement
    holds one call's units."""
    got = sorted({str(v) for v in values})
    if len(got) != 1:
        raise ValueError(f"port {port}: its instances read {got}, and a port reads one registered value: one call's units a "
                         f"statement")
    return got[0]


def statement_tables(z: dict, in_ports: list[str], out_ports: list[str]) -> dict:
    """The prover's shared-row statement of `window.py open`'s rows ``z`` (its rows.npz): each registered port's table
    (``r_<port>``, the rows' positions in the value ``v_<port>`` names, one value a port, `one_value`) the distinct
    positions its instances read in first-read order (`table`), under their registered salts and opened paths
    (``p_<port>``), and every other input port's one row per instance; an output port with positions (the ok row's, at the
    verifier's ``ok``) registered too, its table its instances' own rows; as `UnitStager.stage` takes them: ``tables``,
    ``refs``, ``salts`` (every port's, one per table row) and ``registered``."""
    n = len(z["units"])
    tables, refs, salts, registered = {}, {}, {}, {}
    for p in in_ports:
        w, s = np.asarray(z[f"w_{p}"], np.uint16), [bytes(x) for x in z[f"s_{p}"]]
        if f"r_{p}" not in z:
            tables[p], refs[p], salts[p] = w, list(range(n)), s
            continue
        value = one_value(p, z[f"v_{p}"])
        tpos, refs[p], first = table(z[f"r_{p}"])
        paths = [str(x) for x in z[f"p_{p}"]]
        for i, t in enumerate(refs[p]):
            j = first[t]
            if (s[i], paths[i]) != (s[j], paths[j]) or not np.array_equal(w[i], w[j]):
                raise ValueError(f"port {p}: instances {j} and {i} read position {tpos[t]} as two rows")
        tables[p], salts[p] = w[first], [s[j] for j in first]
        registered[p] = {"value": value, "positions": tpos, "paths": [paths[j] for j in first]}
    for p in out_ports:
        salts[p] = [bytes(x) for x in z[f"s_{p}"]]
        if f"r_{p}" in z:
            registered[p] = {"value": one_value(p, z[f"v_{p}"]), "positions": [int(x) for x in z[f"r_{p}"]],
                             "paths": [str(x) for x in z[f"p_{p}"]]}
    return {"tables": tables, "refs": refs, "salts": salts, "registered": registered}


@dataclass(frozen=True)
class Shape:
    """One H1 tile unit's shape: R A rows and C B rows at contraction K."""

    k: int
    r: int = TILE[0]
    c: int = TILE[1]

    @cached_property
    def layout(self):
        from verity.core.protocols.pouw.circuit import h1tile as H
        return H.Layout(self.r, self.c, self.k)

    @property
    def ab_words(self) -> int:
        """An A or B row's words: K codes and the f32 scale (`h1zy.ROW`)."""
        return self.k // 2 + 2

    @property
    def record_words(self) -> int:
        return 3 * self.r * self.c

    @property
    def receipt_words(self) -> int:
        return 4 * self.layout.sets

    @property
    def in_rows(self) -> list[str]:
        return [f"a{i}" for i in range(self.r)] + [f"b{j}" for j in range(self.c)] + ["zy"]

    def ports(self) -> list:
        """The statement's rows as C-Flock's ports: the A and B rows, the record, the public row, then the receipt's and
        ok's output rows."""
        from verity_flock import circuit as FC
        return ([FC._port(n, self.ab_words, 16 * self.ab_words) for n in self.in_rows[:-1]]
                + [FC._port("zy", self.record_words, 16 * self.record_words), FC._port("pub", PUB_BITS // 16, PUB_BITS)]
                + [FC._port("o0", self.receipt_words, 16 * self.receipt_words), FC._port("o1", len(OK), 16 * len(OK))])

    def unit(self):
        """The tile unit (`h1window.tile_unit`, over the v1 or v3 H1 tile by `tile`)."""
        from verity.core.protocols.pouw.circuit import h1tile as H
        return HW.tile_unit(self.k, (self.r, self.c), H.PouwH1Tile3 if tile() == "v3" else None)

    def definition(self):
        """The statement's Definition: the window's template at K (`h1window.template`, one call of `unit`), the one its
        partition lists."""
        return HW.template(self.k, self.unit())

    def window(self, calls, m: int):
        """The window's public Program (`h1window.program` over this unit: its ``calls`` ``(name, B rows)``, each its own
        linear at K, all reading the one A block of slot ``m``) and its `Q_template_instances` query."""
        program = HW.program([(name, self.k, n) for name, n in calls], [(name, 0) for name, _n in calls], m,
                             (self.r, self.c), unit=lambda _k: self.unit())
        return program, HW.query(program)

    def stager(self, name: str, circuit=None, bind: dict | None = None, flat: bool = False):
        """The unit's `UnitStager` on K's statement rows (`h1window.statement_rows`, under this harness's ports): its circuit,
        built once and written to ``circuit``, bound to ``bind`` (``{"partition", "program"}``: the window's partition and
        Program, `Shape.window`), templated from struct_lower or flat as `layout` says (`stager`), or with ``flat`` the
        flat statement whatever ``H1_TMPL`` says."""
        template = self.definition()
        return stager(template, name, HW.statement_rows(template, ports(template)), HW.STATEMENT_OUT_ROWS,
                      None if flat else layout(), circuit=circuit, bind=bind)


# -- words ------------------------------------------------------------------------------------------------------------------

def row_words(codes: np.ndarray, scales) -> np.ndarray:
    """int8 code rows ``(n, K)`` and their f32 scales ``(n,)`` as their bit rows' u16 words ``(n, K / 2 + 2)``: code t's
    bits are leaves 8 t .. 8 t + 7, then the scale's 32 bits, little-endian (`h1zy.ROW`, the A table leaf)."""
    b = np.ascontiguousarray(np.asarray(codes).astype(np.int8).view(np.uint8))
    b = b.reshape(len(b), -1)
    s = np.ascontiguousarray(np.asarray(scales, "<f4").reshape(len(b))).view(np.uint8).reshape(len(b), 4)
    return np.ascontiguousarray(np.concatenate([b, s], axis=1)).view("<u2").astype(np.uint16)


def split_row(words) -> tuple[list[int], int]:
    """`row_words`' one row back: its codes (u8 encodings) and its scale (the f32 encoding)."""
    b = np.asarray(words, np.uint16).astype("<u2").tobytes()
    return list(b[:-4]), int.from_bytes(b[-4:], "little")


def dequant(z, sa, sw) -> np.ndarray:
    """`h1zy.dequant` on arrays (z s32, s_a and s_w f32, broadcast): RN_bf16(RN32(RN32(RN32(z) · s_a) · s_w)), the served
    epilogue, as bf16 encodings, every NaN 0x7FFF."""
    with np.errstate(over="ignore", invalid="ignore"):
        f = (np.asarray(z, np.int64).astype(np.int32).astype(np.float32) * np.asarray(sa, np.float32)) * np.asarray(sw, np.float32)
    u = np.asarray(f, np.float32).view(np.uint32).astype(np.uint64)
    return np.where(np.isnan(f), 0x7FFF, ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def records(a: np.ndarray, b: np.ndarray, sa, sw) -> tuple[np.ndarray, np.ndarray]:
    """``scaled_mm_zy``'s record of A's rows and B's rows (int8 codes) at their scales (f32): z = A Bᵀ exact, as s32, and
    y = `dequant`(z) at each cell's row s_a and column s_w, each ``(rows of A, rows of B)`` (`h1zy.records` on arrays)."""
    z = (np.asarray(a, np.int64) @ np.asarray(b, np.int64).T).astype(np.int32)
    return z, dequant(z, np.asarray(sa, np.float32)[:, None], np.asarray(sw, np.float32)[None, :])


def record_words(z, y) -> np.ndarray:
    """A unit's record, its cells' z (s32) and y (bf16 encodings), row-major, as its bit row's 3 R C u16 words: z's 32-bit
    words little-endian, then y's (`h1zy.RECORD`, the z table leaf)."""
    zw = np.ascontiguousarray((np.asarray(z, np.int64).reshape(-1) & M32).astype("<u4")).view("<u2")
    return np.concatenate([zw, np.asarray(y, np.uint16).reshape(-1)]).astype(np.uint16)


def split_record(words, cells: int) -> tuple[list[int], list[int]]:
    """`record_words`' one row back: its cells' z (s32 words, two's complement in 32 bits) and y (bf16 encodings)."""
    w = np.asarray(words, np.uint16)
    return [int(v) for v in np.ascontiguousarray(w[:2 * cells].astype("<u2")).view("<u4")], [int(v) for v in w[2 * cells:]]


def pub_words(salt: bytes, position: int, roots: tuple[bytes, bytes]) -> np.ndarray:
    """The public row's 100 u16 words: the unit's salt, position and roots leaves (`seeds.leaves`' bits), in that order,
    bit t of the row bit t % 16 of word t // 16."""
    bits = SD.leaves(salt, True) + SD.position_leaves(position, True) + SD.leaves(roots[0], True) + SD.leaves(roots[1], True)
    assert len(bits) == PUB_BITS
    return np.packbits(np.array(bits, np.uint8), bitorder="little").view("<u2").astype(np.uint16)


def receipt_words(rc: list[int]) -> np.ndarray:
    return np.array([int(v) & M32 for v in rc], dtype="<u4").view("<u2").astype(np.uint16)


def commit_strings(port, rows: np.ndarray, salts: list[bytes]) -> list[bytes]:
    """Each row's ``b || c`` under its salt, by C-Flock's own function (`circuit._commit_strings`) on `port`."""
    from verity_flock import circuit as FC
    return FC._commit_strings([port], {port.name: np.asarray(rows, np.uint16).reshape(len(salts), -1)}, {port.name: salts})[port.name]


# -- the window's own tree --------------------------------------------------------------------------------------------------

def domain(window: str, tree: str, count: int) -> CommitmentDomain:
    b = identity_digest_sha512(WINDOW_TAG, {"window": window, "tree": tree, "leaves": count, "schema": V2})
    return CommitmentDomain(b, -1, RangeIndexedDomain(0, count), hash="sha512")


def leaves(dom: CommitmentDomain, bcs: list[bytes]) -> list[bytes]:
    return [row_leaf(dom, i, HM96.tree_leaf(HM96.default_key, bc), V2) for i, bc in enumerate(bcs)]


def levels(dom: CommitmentDomain, bcs: list[bytes]):
    return tree_levels(dom, leaves(dom, bcs))


def path(lv, index: int) -> list[str]:
    out = []
    for level in lv[:-1]:
        out.append(level[index ^ 1].hex())
        index >>= 1
    return out


def verify(dom: CommitmentDomain, root: bytes, index: int, bc: bytes, sibs: list[str]) -> bool:
    from verity.core.protocols.pouw import core_check as CC
    return CC.verify_path(dom, root, index, leaves_one(dom, index, bc), [bytes.fromhex(s) for s in sibs])


def leaves_one(dom: CommitmentDomain, index: int, bc: bytes) -> bytes:
    return row_leaf(dom, index, HM96.tree_leaf(HM96.default_key, bc), V2)


# -- the verifier's public file ---------------------------------------------------------------------------------------------

def _frame(ports: list, set_name: str, n: int, bcs: dict[str, list[bytes]]) -> dict:
    """`circuit.statement`'s frame-v3 entry over given ``b || c`` (the verifier's: it never holds a row) of ``n`` instances,
    each port's tree over its table's rows."""
    from verity_flock import circuit as FC
    from verity.ml.commitments.frame_v3 import BitRowLeaf, FrameV3
    from verity.ml.commitments.frame_v3 import Port as FPort
    bind, doms = {}, {}
    for p in ports:
        bind[p.name] = identity_digest_sha512(FC.BINDING_TAG, {"set": set_name, "lo": 0, "hi": int(n), "port": p.name, "schema": p.schema})
        doms[p.name] = CommitmentDomain(bind[p.name], FC.OWNER, RangeIndexedDomain(0, len(bcs[p.name])), hash="sha512")
    fv = FrameV3({p.name: FPort(doms[p.name], BitRowLeaf(p.bits) if p.seg else p.leaf()) for p in ports})
    roots = {p.name: fv.root(p.name, [row_leaf(doms[p.name], i, HM96.tree_leaf(HM96.default_key, bc), p.schema)
                                      for i, bc in enumerate(bcs[p.name])]).hex() for p in ports}
    return {"frame": "frame-v3-sha512", "roots": roots, "domain_ids": {k: d.domain_id.hex() for k, d in doms.items()},
            "bindings": {k: b.hex() for k, b in bind.items()}, "owner": FC.OWNER, "schemas": {p.name: p.schema for p in ports},
            "row_key_sha512": hashlib.sha512(HM96.default_key).hexdigest()}


def circuit_meta(circuit: Path) -> dict:
    """A circuit file's META (its second line), without reading the gates."""
    with open(circuit) as f:
        f.readline()
        return json.loads(f.readline()[5:])


def forge_statement(honest: Path, out: Path, pub: bytes, instance: int, port: str, words: np.ndarray, circuit: Path) -> dict:
    """A consistent statement with a wrong witness: ``honest``'s prover file (`circuit.write`'s shared-row file of
    ``circuit``) with instance ``instance``'s row of output ``port`` (its table's row ``instance``) replaced by ``words``,
    and ``pub`` (the verifier's own file of a window that committed those words under the same salt) as its header,
    ``b || c`` and refs. Every committed row is then the opened window's, so binding holds, and the replaced row is not
    what the circuit computes on the instance's inputs. Writes ``out``/inst.bin and pub.bin (``pub`` itself)."""
    from verity_flock import circuit as FC
    data = (honest / "inst.bin").read_bytes()
    nl = data.index(b"\n") + 1
    head = json.loads(data[:nl])
    meta = circuit_meta(circuit)
    ports = FC._meta_ports(meta)
    k = [p.name for p in ports].index(port)
    if "shared_rows" not in head or k < meta["input_ports"]:
        raise ValueError(f"{honest}/inst.bin: an output row of a shared-row file is replaced, not {port} of this one")
    n, counts = head["instances"], [int(head["shared_rows"][p.name]) for p in ports]
    W = sum(c * p.words for c, p in zip(counts, ports))
    rows = bytearray(data[nl:nl + 2 * W])
    salts = data[nl + 2 * W:nl + 2 * W + SALT_BYTES * sum(counts)]
    if len(data) != nl + 2 * W + (SALT_BYTES + 128) * sum(counts) + 4 * n * len(ports):
        raise ValueError(f"{honest}/inst.bin: not a shared-row file of its ports' tables without public words")
    at = 2 * (sum(c * p.words for c, p in zip(counts[:k], ports[:k])) + instance * ports[k].words)
    row = sum(counts[:k]) + instance
    words = np.asarray(words, "<u2").reshape(-1)
    if len(words) != ports[k].words:
        raise ValueError(f"{port}: {len(words)} words, not {ports[k].words}")
    rows[at:at + 2 * len(words)] = words.tobytes()
    pnl = pub.index(b"\n") + 1
    phead, body = json.loads(pub[:pnl]), pub[pnl:]
    if len(body) != 128 * sum(counts) + 4 * n * len(ports) or {x: phead[x] for x in phead if x != "frame_v3"} != dict(
            {x: head[x] for x in head if x != "frame_v3"}, rows=False):
        raise ValueError("the verifier's file is not this statement's but for its frame roots")
    salt, bc = salts[SALT_BYTES * row:SALT_BYTES * (row + 1)], body[128 * row:128 * (row + 1)]
    if commit_strings(ports[k], words[None, :], [salt])[0] != bc:
        raise ValueError(f"{port}: the verifier's b || c of instance {instance} is not these words under the prover's salt")
    out.mkdir(parents=True, exist_ok=True)
    (out / "inst.bin").write_bytes(json.dumps(dict(phead, rows=True), sort_keys=True).encode() + b"\n" + bytes(rows) + salts + body)
    (out / "pub.bin").write_bytes(pub)
    return {"instance": instance, "port": port, "row_words_changed": int(np.count_nonzero(
        np.frombuffer(data[nl + at:nl + at + 2 * len(words)], "<u2") != words))}


def verifier_pub(circuit_head: dict, set_name: str, subcircuit: str, bcs: list[dict[str, bytes]], indices: list[int],
                 reads: list[dict[str, tuple[str, int, str]]]) -> bytes:
    """The verifier's own public file for n units (`circuit.write`'s ``pub_path`` of a shared-row statement):
    ``circuit_head`` the statement's circuit-side fields (its META, pin and class, which the verifier reads off the
    circuit file it holds), each unit's global index in the window's Program (``indices``, ascending: the drawn units),
    each unit's rows' ``b || c`` (port name -> 128 bytes: the window's opened leaves, and its own public and ok rows'), and
    each unit's registered reads (``reads``: port -> (value, position, path), the position the verifier's Program reads
    and the opened path, hex; one value a port, `one_value`). A registered port's table is its units' distinct positions
    in first-read order (`table`), the header's ``registered`` their positions and paths; every other port's is one row
    per unit."""
    from verity_flock import circuit as FC
    meta, pin, cls = circuit_head["meta"], circuit_head["pin"], circuit_head["class"]
    if meta.get("public_inputs"):
        raise ValueError("the unit's statement has public input ports, which the Lean verifier refuses")
    ports = FC._meta_ports(meta)
    n = len(bcs)
    indices = [int(u) for u in indices]
    if len(indices) != n or indices != sorted(set(indices)) or len(reads) != n:
        raise ValueError(f"{n} units need {n} ascending global indices and reads, not {indices}")
    tabs, refs, registered = {}, {}, {}
    for p in ports:
        if p.name not in reads[0]:
            tabs[p.name], refs[p.name] = [u[p.name] for u in bcs], list(range(n))
            continue
        one_value(p.name, [r[p.name][0] for r in reads])
        tpos, refs[p.name], first = table([r[p.name][1] for r in reads])
        for i, t in enumerate(refs[p.name]):
            j = first[t]
            if (bcs[i][p.name], reads[i][p.name]) != (bcs[j][p.name], reads[j][p.name]):
                raise ValueError(f"port {p.name}: units {j} and {i} read position {tpos[t]} as two rows")
        tabs[p.name] = [bcs[j][p.name] for j in first]
        registered[p.name] = {"value": reads[first[0]][p.name][0], "positions": tpos,
                              "paths": [reads[j][p.name][2] for j in first]}
    fv = _frame(ports, set_name, n, tabs)
    vb = meta["vus_per_block"]
    units = {"vus_per_block": vb, "units_per_vu": meta.get("units_per_vu", 1), "indices": indices,
             "blocks": [indices[b:b + vb] for b in range(0, n, vb)], "classes": [cls] * n}
    head = {"format": FC.INST_FORMAT, "statement": meta.get("statement", FC.STATEMENT), "scheme": FC.SCHEME, "set": set_name,
            "subcircuit": subcircuit, "range": [0, n], "instances": n, "circuit_sha512": pin, "units": units, "frame_v3": fv,
            "rows": False, "registered": registered, "shared_rows": {p.name: len(tabs[p.name]) for p in ports}}
    return FC.public_file(head, b"".join(bc for p in ports for bc in tabs[p.name])
                          + np.ascontiguousarray(np.stack([refs[p.name] for p in ports], axis=1).astype("<u4")).tobytes())
