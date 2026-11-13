"""A sampled H1 unit's rows rebuilt from a served window's values, as `drawn` stages them: the word formats both of its forms
read (`split_row`, `split_record`, `receipt_words`), the check that a rebuilt value is the tree the window registered
(`held`), that a statement's every port is a registered read (`complete`) and that its outputs are the rows the window
registered (`outputs_held`); and the lanes form's leaf statements (`leaf_lanes`): their rows (`leaf_rows`), a leaf's
operands at its place (`operands`), its partials (`partials`), its call's receipts (`receipt_value`) and each
statement's rows, salts and ``--registered`` reads on credit's own file (`leaf_statement`)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.protocols.pouw.prover.circuit import h1zy as Z
from verity.protocols.pouw.verifier import credit as CR
from verity.protocols.pouw.verifier import leaf_lanes as LL
from verity.protocols.spot.verifier.one_stage import produced as PR
from verity.protocols.spot.verifier.one_stage import registered as G

from verity_vllm.errors import Refused
from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import served as SV

__all__ = ["LEAF_NAMES", "LEAF_OUT_ROWS", "LEAF_PROGRAM", "held", "leaf_rows", "leaf_statement", "operands", "partials",
           "receipt_value", "receipt_words", "split_record", "split_row"]

M32 = (1 << 32) - 1
#: a lanes-form K's lane and combine statements' names, as `pouw_h1_circuits` lowers them
LEAF_NAMES = {LL.LANE: "unit/pouw-h1-lane", LL.COMBINE: "unit/pouw-h1-combine"}
#: a leaf statement's one output row, ``o0`` (its ok, `leaf_lanes.OK_ROW` when it holds)
LEAF_OUT_ROWS = [(V2, (0,))]
#: beside --circuits' ``<kind><K>.circuit.txt``, its leaf Program's descriptor (`h1window.leaf_program`'s canonical JSON)
LEAF_PROGRAM = "{name}.program.json"


def split_row(words: np.ndarray) -> tuple[list[int], int]:
    """An A or B row's u16 words (`h1zy.ROW`: its codes' bytes, then the f32 scale) back to its codes (u8 encodings) and
    its scale (the f32 encoding)."""
    b = np.asarray(words, np.uint16).astype("<u2").tobytes()
    return list(b[:-4]), int.from_bytes(b[-4:], "little")


def split_record(words: np.ndarray, cells: int) -> tuple[list[int], list[int]]:
    """A unit's record words (`h1zy.RECORD`) back to its cells' z (s32 words) and y (bf16 encodings)."""
    w = np.asarray(words, np.uint16)
    return [int(v) for v in np.ascontiguousarray(w[:2 * cells].astype("<u2")).view("<u4")], [int(v) for v in w[2 * cells:]]


def receipt_words(rc: Sequence[int]) -> np.ndarray:
    return np.array([int(v) & M32 for v in rc], dtype="<u4").view("<u2").astype(np.uint16)


def held(reg: G.Registered, entry: dict, what: str) -> G.Registered:
    """`reg` when it has the root and domain credit's ``--registered`` port `entry` reads; else Refused."""
    if (reg.root.hex(), reg.domain.domain_id.hex()) != (entry["root"], entry["domain"]):
        raise Refused(f"{entry['value']}, rebuilt from {what}, is not the tree the window registered (root "
                      f"{reg.root.hex()[:16]}... domain {reg.domain.domain_id.hex()[:16]}..., registered "
                      f"{entry['root'][:16]}... domain {entry['domain'][:16]}...)")
    return reg


def complete(ports: Sequence[str], reg: dict, call: int) -> None:
    """Every port of call `call`'s statement (`ports`: its circuit's, each input row and output row) is a port of
    credit's ``--registered`` file `reg`, as the Lean verifier requires of every statement it is handed with
    ``--registered`` (`Flock.Registered.checkComplete`, a premise of the accounting soundness proofs); else Refused,
    naming the first port that is not. o0, the unit's receipt, is one only where the window registers the call's
    receipt table (`credit.registered_file`)."""
    missing = [p for p in ports if p not in reg["ports"]]
    if not missing:
        return
    why = (f"call {call}'s statement's port {missing[0]} is no port of credit's registered reads, so the Lean verifier "
           "refuses it (registered reads (complete))")
    if missing[0] == CR.RECEIPT_PORT:
        why += (f": o0 is the unit's receipt, which credit reads at the call's receipt table receipt@{call} only when "
                "the window registers it: serve the window with its receipts (serve-openai --h1-receipts)")
    raise Refused(why)


def outputs_held(z: dict, outs: dict) -> None:
    """Each output row the statement opens at a registered value (o0 at the call's receipt table) is the row the
    window registered there, unit by unit; else Refused: its proof would open at no registered root."""
    for p, rows in outs.items():
        if f"r_{p}" in z:
            bad = np.flatnonzero((np.asarray(rows, np.uint16) != np.asarray(z[f"w_{p}"], np.uint16)).any(axis=1))
            if bad.size:
                raise Refused(f"unit {int(z['units'][bad[0]])}: its {p} is not row {int(z[f'r_{p}'][bad[0]])} of "
                              f"{z[f'v_{p}'][bad[0]]}, which the window registered")


def leaf_rows(kind: str, k: int, tile: Sequence[int]) -> list[tuple]:
    """A sampled leaf's ``kind`` statement's input rows at K over the R × C ``tile``, ``(port, schema, bits, params)`` as
    `UnitStager` takes them: a lane's ``a0`` .. ``a3`` and ``b0`` .. ``b7`` (its A and B rows, each K codes and a scale),
    ``admit`` (the leaf's R admit bits), ``zy`` (the leaf's record), ``partials`` (its row of the leaf's partials) and
    ``pub`` (`leaf_lanes.PUBLIC`); the combine's ``p0`` .. (each lane's partial) and ``receipt``. Its output row is
    `LEAF_OUT_ROWS`."""
    r, c = tile
    wp = 2 * LL.words(k)
    if kind == LL.LANE:
        ab = 16 * (k // 2 + 2)
        return ([(f"a{s}", V2, ab, [f"a[{s}]"]) for s in range(LL.LANE_ROWS)]
                + [(f"b{s}", V2, ab, [f"b[{s}]"]) for s in range(LL.LANE_COLS)]
                + [(LL.ADMIT, V2, r, [LL.ADMIT]), (LL.ZY, V2, 48 * r * c, [LL.ZY]),
                   (LL.PARTIALS, V2, 16 * wp, [LL.PARTIALS]), (LL.PUB, V2, sum(LL.PUBLIC_BITS), list(LL.PUBLIC))])
    if kind == LL.COMBINE:
        return ([(f"p{x}", V2, 16 * wp, [f"{LL.PARTIALS}[{x}]"]) for x in range(LL.lanes(r, c))]
                + [(LL.RECEIPT, V2, 16 * wp, [LL.RECEIPT])])
    raise ValueError(f"a v1 leaf's statement is its {LL.LANE} or its {LL.COMBINE}, not {kind!r}")


def u32_words(words: Any) -> list[int]:
    return [int(v) for v in np.ascontiguousarray(np.asarray(words, np.uint16).astype("<u2")).view("<u4")]


def split_rows(tree: G.Registered, at: int, n: int) -> tuple[list[list[int]], list[int]]:
    """`tree`'s `n` rows from `at` (`split_row` each): their codes, then their scales."""
    got = [split_row(tree.rows[at + i]) for i in range(n)]
    return [v for v, _s in got], [s for _v, s in got]


def operands(win: dict, u: int, trees: dict[str, G.Registered]) -> dict[str, Any]:
    """Unit u's leaf as its statements read it, from the window's rebuilt values ``trees`` at its place
    (`credit.leaf_place`): its R A rows' and C B rows' codes and scales, its record, its admit bits (every row: the tile
    unit masks none, as credit's known ``admit`` row says), the window's Program salt, its position and the roots its seeds
    take, root(``a/<K>@<s>``) and root(``b/<K>``) as the window registered them."""
    lay = win["layout"]
    r, c = lay["tile"]
    at = CR.leaf_place(lay, win["shape"], u, win["steps"])
    a, sa = split_rows(trees[at[LL.A][0]], at[LL.A][1], r)
    b, sw = split_rows(trees[at[LL.B][0]], at[LL.B][1], c)
    zy = split_record(trees[at[LL.ZY][0]].rows[at[LL.ZY][1]], r * c)
    roots = tuple(bytes.fromhex(win["roots"][at[v][0]][0]) for v in (LL.A, LL.B))
    return {"a": a, "sa": sa, "b": b, "sw": sw, "zy": zy, "admit": [1] * r, "salt": bytes.fromhex(win["salt"]),
            "position": u, "roots": roots, "tile": (r, c)}


def lane_operands(x: dict, ln: int) -> tuple[list, list, list, list]:
    """Lane ln's 4 A rows and 8 B rows by slot, codes then scales (`leaf_lanes.lane_rows`, `lane_cols`)."""
    rows, cols = LL.lane_rows(*x["tile"], ln), LL.lane_cols(*x["tile"], ln)
    return [x["a"][i] for i in rows], [x["b"][j] for j in cols], [x["sa"][i] for i in rows], [x["sw"][j] for j in cols]


def partials(x: dict) -> np.ndarray:
    """The leaf's partials, a row a lane: `h1zy.lane_partial` of each lane's rows, its u16 words."""
    out = []
    for ln in range(LL.lanes(*x["tile"])):
        a, b, _sa, _sw = lane_operands(x, ln)
        out.append(receipt_words(Z.lane_partial(a, b, x["admit"], x["salt"], x["position"], x["roots"], ln, x["tile"])))
    return np.array(out, np.uint16)


def receipt_value(win: dict, key: bytes, call: int, trees: dict[str, G.Registered]) -> G.Registered:
    """Call `call`'s ``receipt@<c>`` as `Served.commit` registers it with receipts: one row a unit of the call, its
    receipt's u16 words (`receipt_words`), here the word reference's (`h1zy.reference` of the unit's rows, salt, position
    and roots), at ``RECEIPT_NONCE`` + c under the prover's key, bound to the window's root."""
    sizes = [size for size, _w in CR.call_work(win["layout"])]
    rows = []
    for u in range(sum(sizes[:call]), sum(sizes[:call + 1])):
        x = operands(win, u, trees)
        rc, _ok = Z.reference(x["a"], x["b"], x["sa"], x["sw"], x["zy"], x["salt"], u, x["roots"])
        rows.append(receipt_words(rc))
    return NR.commit(win["program"], f"{win['shape']['prefix']}{LL.RECEIPT}@{call}", np.array(rows, np.uint16), key,
                     SV.RECEIPT_NONCE + call, keep_rows=True, window=win["root"])


def leaf_ok(kind: str, ln: int | None, x: dict, rows: dict[str, np.ndarray]) -> int:
    """The statement's ok on its rows, the word reference's (`h1zy.lane_reference`, `h1zy.combine_reference`)."""
    if kind == LL.LANE:
        if ln is None:
            raise TypeError("a lane's statement names its lane")
        a, b, sa, sw = lane_operands(x, ln)
        return Z.lane_reference(a, b, sa, sw, x["admit"], x["zy"], u32_words(rows[LL.PARTIALS][0]), x["salt"],
                                x["position"], x["roots"], ln, x["tile"])
    return Z.combine_reference([u32_words(rows[f"p{i}"][0]) for i in range(LL.lanes(*x["tile"]))],
                               u32_words(rows[LL.RECEIPT][0]))


def leaf_statement(win: dict, u: int, st: tuple, x: dict, trees: dict[str, G.Registered], tree: G.Registered,
                   produced: dict[str, str], draw_root: str) -> tuple[dict, dict, dict, dict]:
    """Sampled leaf u's statement `st` as `FileStager.stage` takes it, on credit's own ``--registered`` file
    (`credit.leaf_registered_file`): each port's one row, salt and registered read (value, position, path), from the
    window's rebuilt values ``trees`` (each held to the file's root and domain), the leaf's partials ``tree`` (at the
    produced record's root ``produced``) or the known row (`leaf_lanes.known_rows`, committed under
    `leaf_lanes.known_salt` of the sample root), ``o0`` among those; and the file. Refused when a row opens at no root of the
    file, or the statement's ok (`leaf_ok`) is not 1."""
    lay, shape = win["layout"], win["shape"]
    kind, ln = st
    got, why = CR.leaf_registered_file(win["program"], lay, shape, u, st, win["roots"], win["root"], win["steps"],
                                       win["salt"], draw_root, produced)
    if why is not None:
        raise Refused(f"drawn leaf {u}'s statement {list(st)}: credit builds no registered file: {why[1]}")
    reg = got[0]
    known = LL.known_rows(*x["tile"], kind, ln, admit=x["admit"], salt=x["salt"], position=u, roots=x["roots"])
    rows, salts, registered = {}, {}, {}
    for port, e in reg["ports"].items():
        at = e["reads"][0]
        if port in known:
            src = PR.commit(win["program"], e["value"], [known[port][1]], draw_root,
                            [LL.known_salt(draw_root, e["value"])], G.WORD_BITS, G.SCHEMA_V2)
        elif e["value"] == tree.value:
            src = tree
        else:
            src = held(trees[e["value"]], e, "the prover's state and the served weights")
        if src.root.hex() != e["root"]:
            raise Refused(f"drawn leaf {u}'s statement {list(st)}: port {port}'s {e['value']} is not at the root credit's "
                          "file reads it at")
        salts[port] = [bytes(src.salts[at])]
        registered[port] = {"value": e["value"], "positions": [at], "paths": [src.path(at).hex()]}
        if port != LL.OK:
            rows[port] = np.array([src.rows[at]], np.uint16)
    ok = leaf_ok(kind, ln, x, rows)
    if ok != 1:
        raise Refused(f"drawn leaf {u}'s statement {list(st)}: the word reference's ok is {ok}, not 1: its rows are not "
                      "the leaf's honest lane or combine, so it is not proved")
    return rows, salts, registered, reg
