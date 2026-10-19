"""Pinned vectors for the audit window's layout for gates (`verity.core.protocols.pouw.window`): ``uv run python
verity/core/protocols/pouw/tests/window_vectors.py`` rewrites ``tests/vectors/window.json``.  A change to that file
is a change to what the window's tile and entry circuits read (an entry row's bits, a side tree, the window root or the
public inputs file C-Flock's verifier writes), and needs a reason in the PR.

The toy window: P = 4 entries over a toy window domain, the identity ``verity/pouw/window-vectors/v1`` of {"window":
"toy"}, and the static shape `SHAPE`.  Call 0 is an 80 x 32 x 66 call whose tiles start at tile 3 of the pass (so two
tile columns, and a short last tile row 16 rows high): its slots 0 to 70 are admitted and slot 2's forming refused, its
slots 71 to 79 empty, and of its 66 weight rows row 5's forming refused.  Call 1 is 16 x 32 x 4 from pass tile 9: slots 0
to 3 admitted, slot 2 refused, the rest empty.  So N = 80·66 + 16·4 + 1 = 5,345 tiles, of which 70·65 + 3·4 = 4,562 are
credited (level 79, 4,096 tiles); the filler's canonical empty row on each side covers tile 5,344.  Each admitted side
row is the FP8 operand row (k = 32, codes and P SHAKE-256 of a tag) then its index field (`Wd.side_row`); an empty slot
is the canonical empty row; the salts are SHAKE-256 too.  Both calls name one toy pass's -h3 tile root, and F_σ its toy
tile domain; ``tile_reads`` is (p, at) (`Wd.tile_read`) at the indices `READS`.  The draw is K′ = 3 tiles in sessions of
cap 2, so session 1 pads with N − 1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, row_leaf
from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.pouw.circuit import leaves as L

OUT = Path(__file__).with_name("vectors") / "window.json"
TAG = "verity/pouw/window-vectors/v1"
K = 32
P = 4
CAP = 2
DRAWN = (1, 4500, 5344)
ROLES = {"a": ROLE_X, "b": ROLE_W}
SHAPE = (Wd.Call(80, 66, 3), Wd.Call(16, 4, 9))
#: per call, each side's admitted rows (a prefix: the slots holding the batch's requests, every weight row) and the
#: admitted rows whose forming refused
CALLS = ({"admit": {"a": 71, "b": 66}, "refused": {"a": [2], "b": [5]}},
         {"admit": {"a": 4, "b": 4}, "refused": {"a": [2], "b": []}})
#: the drawn indices whose tile reads are pinned: call 0's first tile, a tile past column 64, its short last tile row,
#: an empty slot's, its last; call 1's first and last; the filler
READS = (0, 65, 64 * 66 + 65, 75 * 66 + 1, 5279, 5280, 5343, 5344)


def _bytes(label: str, n: int) -> bytes:
    return hashlib.shake_256(f"{TAG}\0{label}".encode()).digest(n)


def _row(label: str, index: int, ok: int) -> list[int]:
    raw = _bytes(label, K + 64)
    return Wd.side_row(L.words_fp8_p(list(raw[:K]), list(raw[K:K + 64])), index, ok, 1)


def empty_row(index: int) -> list[int]:
    return Wd.side_row([0] * L.ROW_WORDS["fp8-p"](K), index, 0, 0)


def pass_h3() -> tuple[bytes, bytes]:
    """The toy pass's -h3 tile root and its tile tree's domain id."""
    return _bytes("pass/root_h3", 64), identity_digest_sha512(TAG, {"pass": "toy", "tiles": "h3"})


def window_domain() -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(TAG, {"window": "toy"}), Wd.OWNER, RangeIndexedDomain(0, P), hash="sha512")


def _side(dom: CommitmentDomain, side: str, rows: list[list[int]], label: str) -> dict:
    schema = L.ROW_SCHEMA["fp8-p", ROLES[side]]
    bcs = [L.commit("fp8-p", ROLES[side], w, _bytes(f"{label}/salt/{i}", 192)) for i, w in enumerate(rows)]
    leaves = [row_leaf(dom, i, L.tree_leaf(bc), schema) for i, bc in enumerate(bcs)]
    levels = Wd.side_levels(dom, leaves)
    return {"rows": len(rows), "words": rows, "leaves": leaves, "levels": levels, "root": levels[-1][0]}


def side_rows(c: int, call: Wd.Call, sel: dict) -> dict[str, list[list[int]]]:
    count = {"a": call.m, "b": call.n}
    return {s: [_row(f"call{c}/{s}/{i}", i, int(i not in sel["refused"][s])) if i < sel["admit"][s] else empty_row(i)
                for i in range(count[s])] for s in "ab"}


def window() -> dict:
    """The toy window's entries, sides and root, as `vectors` pins them."""
    dom = window_domain()
    sdom = {s: Wd.side_domain(dom.domain_id, s) for s in "ab"}
    root_h3, _ = pass_h3()
    public = Wd.shape_entries(SHAPE, P)
    entries, sides, records = [], [], []
    for c, pub in enumerate(public[:len(SHAPE) + 1]):
        call = c < len(SHAPE)
        record = {"call": c, "shape": [SHAPE[c].m, K, SHAPE[c].n], "tile0": SHAPE[c].tile0} if call else None
        rows = side_rows(c, SHAPE[c], CALLS[c]) if call else {"a": [empty_row(0)], "b": [empty_row(0)]}
        side = {s: _side(sdom[s], s, rows[s], f"entry{c}/{s}") for s in "ab"}
        side["fields"] = {s: [list(Wd.side_field(r)) for r in rows[s]] for s in "ab"}
        entries.append(Wd.Entry(pub.kind, pub.lo, pub.hi, pub.n_a, pub.n_b, side["a"]["root"], side["b"]["root"],
                                root_h3 if call else bytes(64), Wd.record_digest(record) if call else bytes(64),
                                pub.tile0, pub.cols, pub.m))
        sides.append(side)
        records.append(record)
    entries += public[len(entries):]
    records += [None] * (P - len(records))
    salts = [_bytes(f"entry/salt/{e}", 192) for e in range(P)]
    bcs = [Wd.entry_bc(x, s) for x, s in zip(entries, salts)]
    leaves = [Wd.entry_leaf(dom, e, bc) for e, bc in enumerate(bcs)]
    root = tree_levels(dom, leaves)[-1][0]
    real = sum(Wd.selected(x["fields"])["a"] * Wd.selected(x["fields"])["b"] for x in sides[:len(SHAPE)])
    return {"domain": dom, "side_domains": sdom, "entries": entries, "sides": sides, "records": records, "salts": salts,
            "bcs": bcs, "leaves": leaves, "root": root, "real": real, "work": Wd.credit(real, Wd.shape_tiles(SHAPE)),
            "root_h3": root_h3}


def tile_reads(w: dict) -> list[dict]:
    """At each index of `READS`, its (entry, I, J) from the shape (`Wd.tile_at`) and, for a call's tile, the (p, at) its
    unit reads from its entry row and its side rows' indices alone."""
    out = []
    for t in READS:
        e, i, j = Wd.tile_at(SHAPE, t)
        x = {"t": t, "entry": e, "I": i, "J": j}
        if w["entries"][e].kind == "call":
            big_i, big_j = (w["sides"][e]["fields"][s][r][0] for s, r in (("a", i), ("b", j)))
            x["p"], x["at"] = Wd.tile_read(w["entries"][e], big_i, big_j)
        out.append(x)
    return out


def registration(w: dict) -> dict:
    k = len(DRAWN)
    return {"population": w["work"].n, "law": {"law": "subset", "k": k},
            "roots": [{"port": Wd.PORT, "root": w["root"].hex(), "leaves": P, "domain": w["domain"].domain_id.hex()}]}


def vectors() -> dict:
    w = window()
    reg = registration(w)
    _, domain_h3 = pass_h3()
    a = w["sides"][1]["a"]
    sdom = w["side_domains"]["a"]
    return {
        "about": f"python {Path(__file__).name} rewrites this file; its docstring says what is pinned",
        "layout": {"entry_bytes": Wd.ENTRY_BYTES, "entry_words": Wd.ENTRY_WORDS, "depths": Wd.DEPTHS, "t_bits": Wd.T_BITS,
                   "kinds": Wd.KIND_CODE, "schema": Wd.ENTRY_SCHEMA, "side_index_words": Wd.SIDE_INDEX_WORDS,
                   "side_row_words": len(w["sides"][0]["a"]["words"][0])},
        "shape": Wd.shape_json(SHAPE),
        "window": {"domain": w["domain"].domain_id.hex(), "real": w["real"], "work": w["work"].to_json(1),
                   "root": w["root"].hex(), "root_h3": w["root_h3"].hex(), "domain_h3": domain_h3.hex()},
        "records": w["records"],
        "fields": [x["fields"] for x in w["sides"]],
        "tile_reads": tile_reads(w),
        "entries": [{"entry": x.to_json(), "row": x.row().hex(), "digest": Wd.entry_digest(x).hex(), "salt": s.hex(),
                     "bc": bc.hex(), "leaf": leaf.hex()}
                    for x, s, bc, leaf in zip(w["entries"], w["salts"], w["bcs"], w["leaves"])],
        "side": {"entry": 1, "side": "a", "domain": sdom.domain_id.hex(),
                 "domain_b": w["side_domains"]["b"].domain_id.hex(), "rows": a["rows"], "row_1": a["words"][1],
                 "row_15": a["words"][15], "leaves": [x.hex() for x in a["leaves"]], "pad_leaf": sdom.pad(a["rows"]).hex(),
                 "root": a["root"].hex(), "path_of_1": [x.hex() for x in Wd.side_path(sdom, a["levels"], 1)]},
        "public_inputs": {"registration": reg, "drawn": list(DRAWN), "cap": CAP,
                          "sessions": [Wd.public_inputs(reg, DRAWN, s, CAP, domain_h3).hex()
                                       for s in range(Wd.sessions(len(DRAWN), CAP))]},
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
