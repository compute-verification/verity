"""Pinned vectors for the audit window's layout for gates (`verity.core.protocols.pouw.window`): ``uv run python
verity/core/protocols/pouw/tests/window_vectors.py`` rewrites ``tests/vectors/window.json``.  A change to that file
is a change to what the window's tile and entry circuits read (an entry row's bits, a side tree, the window root or the
public inputs file C-Flock's verifier writes), and needs a reason in the PR.

The toy window: P = 4 entries over a toy window domain, the identity ``verity/pouw/window-vectors/v1`` of {"window":
"toy"}.  Call 0 is an 80 x 32 x 128 call whose tiles start at tile 3 of the pass; it keeps A rows 0, 1 and 70 (so a
tile of its short last tile row, 16 rows high) and B rows 5 and 64.  Call 1 is 16 x 32 x 64 from pass tile 9 and keeps
A row 3 and B rows 0 and 1.  So 8 real tiles, level 7, N = 9, and the filler's one zero row on each side covers tile 8.
Each side row is the FP8 operand row (k = 32, codes and P SHAKE-256 of a tag) then its index field (`Wd.side_row`); the
salts are SHAKE-256 too.  Both calls name one toy pass's -h3 tile root, and F_σ its toy tile domain; ``tile_reads`` is
each real tile's (p, at) (`Wd.tile_read`).  The draw is K′ = 3 tiles in sessions of cap 2, so session 1 pads with N − 1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, tree_levels
from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X, row_leaf
from verity.core.protocols.pouw import window as Wd
from verity.ml.protocols.pouw.circuit import leaves as L

OUT = Path(__file__).with_name("vectors") / "window.json"
TAG = "verity/pouw/window-vectors/v1"
K = 32
P = 4
CAP = 2
DRAWN = (1, 4, 6)
ROLES = {"a": ROLE_X, "b": ROLE_W}
#: per call, its shape [m, k, n], its first tile in the pass and the rows each side keeps, by their index in the call
CALLS = ({"shape": [80, K, 128], "tile0": 3, "keep": {"a": [0, 1, 70], "b": [5, 64]}},
         {"shape": [16, K, 64], "tile0": 9, "keep": {"a": [3], "b": [0, 1]}})


def _bytes(label: str, n: int) -> bytes:
    return hashlib.shake_256(f"{TAG}\0{label}".encode()).digest(n)


def _row(label: str, index: int) -> list[int]:
    raw = _bytes(label, K + 64)
    return Wd.side_row(L.words_fp8_p(list(raw[:K]), list(raw[K:K + 64])), index)


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


def window() -> dict:
    """The toy window's entries, sides and root, as `vectors` pins them."""
    dom = window_domain()
    sdom = {s: Wd.side_domain(dom.domain_id, s) for s in "ab"}
    root_h3, _ = pass_h3()
    entries, sides, records, at = [], [], [], 0
    zero = Wd.side_row([0] * L.ROW_WORDS["fp8-p"](K), 0)
    for c, call in enumerate(CALLS + (None,)):
        record = {"call": c, **call} if call else None
        rows = {s: [_row(f"call{c}/{s}/{i}", i) for i in call["keep"][s]] for s in "ab"} if call else \
            {"a": [zero], "b": [zero]}
        side = {s: _side(sdom[s], s, rows[s], f"entry{c}/{s}") for s in "ab"}
        side["indices"] = {s: [Wd.side_index(r) for r in rows[s]] for s in "ab"}
        size = len(rows["a"]) * len(rows["b"]) if call else Wd.bucket(at).n - at
        place = {"tile0": call["tile0"], "cols": -(-call["shape"][2] // Wd.TILE), "m": call["shape"][0]} if call else {}
        entries.append(Wd.Entry("call" if call else "filler", at, at + size, side["a"]["rows"], side["b"]["rows"],
                                side["a"]["root"], side["b"]["root"], root_h3 if call else bytes(64),
                                Wd.record_digest(record) if call else bytes(64), **place))
        sides.append(side)
        records.append(record)
        at += size
    entries += [Wd.Entry("none")] * (P - len(entries))
    records += [None] * (P - len(records))
    salts = [_bytes(f"entry/salt/{e}", 192) for e in range(P)]
    bcs = [Wd.entry_bc(x, s) for x, s in zip(entries, salts)]
    leaves = [Wd.entry_leaf(dom, e, bc) for e, bc in enumerate(bcs)]
    root = tree_levels(dom, leaves)[-1][0]
    return {"domain": dom, "side_domains": sdom, "entries": entries, "sides": sides, "records": records, "salts": salts,
            "bcs": bcs, "leaves": leaves, "root": root, "work": Wd.bucket(entries[len(CALLS)].lo), "root_h3": root_h3}


def tile_reads(w: dict) -> list[dict]:
    """Each real tile t's (I, J) and the (p, at) its unit reads, from its entry row and side rows' indices alone."""
    out = []
    for t in range(w["work"].n):
        e, i, j = Wd.tile_at(w["entries"], t)
        if w["entries"][e].kind != "call":
            continue
        big_i, big_j = (w["sides"][e]["indices"][s][x] for s, x in (("a", i), ("b", j)))
        p, at = Wd.tile_read(w["entries"][e], big_i, big_j)
        out.append({"t": t, "entry": e, "I": big_i, "J": big_j, "p": p, "at": at})
    return out


def registration(w: dict) -> dict:
    k = len(DRAWN)
    return {"population": w["work"].n, "law": {"law": "subset", "k": k},
            "roots": [{"port": Wd.PORT, "root": w["root"].hex(), "leaves": P, "domain": w["domain"].domain_id.hex()}]}


def vectors() -> dict:
    w = window()
    reg = registration(w)
    _, domain_h3 = pass_h3()
    a = w["sides"][0]["a"]
    sdom = w["side_domains"]["a"]
    return {
        "about": f"python {Path(__file__).name} rewrites this file; its docstring says what is pinned",
        "layout": {"entry_bytes": Wd.ENTRY_BYTES, "entry_words": Wd.ENTRY_WORDS, "depths": Wd.DEPTHS, "t_bits": Wd.T_BITS,
                   "kinds": Wd.KIND_CODE, "schema": Wd.ENTRY_SCHEMA, "side_index_words": Wd.SIDE_INDEX_WORDS,
                   "side_row_words": len(w["sides"][0]["a"]["words"][0])},
        "window": {"domain": w["domain"].domain_id.hex(), "work": w["work"].to_json(1), "root": w["root"].hex(),
                   "root_h3": w["root_h3"].hex(), "domain_h3": domain_h3.hex()},
        "records": w["records"],
        "indices": [x["indices"] for x in w["sides"]],
        "tile_reads": tile_reads(w),
        "entries": [{"entry": x.to_json(), "row": x.row().hex(), "digest": Wd.entry_digest(x).hex(), "salt": s.hex(),
                     "bc": bc.hex(), "leaf": leaf.hex()}
                    for x, s, bc, leaf in zip(w["entries"], w["salts"], w["bcs"], w["leaves"])],
        "side": {"entry": 0, "side": "a", "domain": sdom.domain_id.hex(),
                 "domain_b": w["side_domains"]["b"].domain_id.hex(), "rows": a["rows"], "row_1": a["words"][1],
                 "leaves": [x.hex() for x in a["leaves"]], "pad_leaf": sdom.pad(a["rows"]).hex(),
                 "root": a["root"].hex(), "path_of_1": [x.hex() for x in Wd.side_path(sdom, a["levels"], 1)]},
        "public_inputs": {"registration": reg, "drawn": list(DRAWN), "cap": CAP,
                          "sessions": [Wd.public_inputs(reg, DRAWN, s, CAP, domain_h3).hex()
                                       for s in range(Wd.sessions(len(DRAWN), CAP))]},
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
