"""Pinned vectors for the audit window's layout for gates (`verity.protocols.accounting.work.pouw.window`): ``uv run python
verity/protocols/accounting/work/pouw/tests/window_vectors.py`` rewrites ``tests/vectors/window.json``.  A change to that file
is a change to what the window's tile and entry circuits read (an entry row's bits, a side tree, the window root or the
public inputs file C-Flock's verifier writes), and needs a reason in the PR.

The toy window: P = 4 entries over a toy window domain, the identity ``verity/pouw/window-vectors/v1`` of {"window":
"toy"}; call 0 keeps A rows 0, 1, 3 of four (row 2 refused) and both of its B rows, call 1 one A row and two B rows, so 8
real tiles, level 7, N = 9, and the filler's one zero row on each side covers tile 8.  Rows are FP8 rows of k = 32 whose
codes and salts are SHAKE-256 of a tag; the draw is K′ = 3 tiles in sessions of cap 2, so session 1 pads with N − 1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.primitives.commitments.identity import identity_digest_sha512
from verity.primitives.commitments.indexed import RangeIndexedDomain
from verity.primitives.commitments.merkle import CommitmentDomain, tree_levels
from verity.primitives.commitments.rowleaf import ROLE_W, ROLE_X, row_leaf
from verity.protocols.accounting.work.pouw import window as Wd
from verity.protocols.accounting.work.pouw.circuit import leaves as L

OUT = Path(__file__).with_name("vectors") / "window.json"
TAG = "verity/pouw/window-vectors/v1"
K = 32
P = 4
CAP = 2
DRAWN = (1, 4, 6)
ROLES = {"a": ROLE_X, "b": ROLE_W}
#: per call, each side's rows by their index in the call; None where the scheme refused the row
CALLS = ({"a": (0, 1, None, 3), "b": (0, 1)}, {"a": (0,), "b": (0, 1)})


def _bytes(label: str, n: int) -> bytes:
    return hashlib.shake_256(f"{TAG}\0{label}".encode()).digest(n)


def _row(label: str) -> list[int]:
    raw = _bytes(label, K + 64)
    return L.words_fp8_p(list(raw[:K]), list(raw[K:K + 64]))


def window_domain() -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(TAG, {"window": "toy"}), Wd.OWNER, RangeIndexedDomain(0, P), hash="sha512")


def _side(dom: CommitmentDomain, side: str, rows: list[list[int]], label: str) -> dict:
    schema = L.ROW_SCHEMA["fp8-p", ROLES[side]]
    bcs = [L.commit("fp8-p", ROLES[side], w, _bytes(f"{label}/salt/{i}", 192)) for i, w in enumerate(rows)]
    leaves = [row_leaf(dom, i, L.tree_leaf(bc), schema) for i, bc in enumerate(bcs)]
    levels = Wd.side_levels(dom, leaves)
    return {"rows": len(rows), "leaves": leaves, "levels": levels, "root": levels[-1][0]}


def window() -> dict:
    """The toy window's entries, sides and root, as `vectors` pins them."""
    dom = window_domain()
    sdom = {s: Wd.side_domain(dom.domain_id, s) for s in "ab"}
    entries, sides, at = [], [], 0
    zero = [0] * L.ROW_WORDS["fp8-p"](K)
    for c, call in enumerate(CALLS + (None,)):
        keep = {s: [i for i, r in enumerate(call[s]) if r is not None] for s in "ab"} if call else None
        rows = {s: [_row(f"call{c}/{s}/{i}") for i in keep[s]] for s in "ab"} if call else {"a": [zero], "b": [zero]}
        side = {s: _side(sdom[s], s, rows[s], f"entry{c}/{s}") for s in "ab"}
        size = len(rows["a"]) * len(rows["b"]) if call else Wd.bucket(at).n - at
        record = Wd.record_digest({"call": c, "keep": keep}) if call else bytes(64)
        entries.append(Wd.Entry("call" if call else "filler", at, at + size, side["a"]["rows"], side["b"]["rows"],
                                side["a"]["root"], side["b"]["root"], record))
        sides.append(side)
        at += size
    entries += [Wd.Entry("none")] * (P - len(entries))
    salts = [_bytes(f"entry/salt/{e}", 192) for e in range(P)]
    bcs = [Wd.entry_bc(x, s) for x, s in zip(entries, salts)]
    leaves = [Wd.entry_leaf(dom, e, bc) for e, bc in enumerate(bcs)]
    root = tree_levels(dom, leaves)[-1][0]
    return {"domain": dom, "side_domains": sdom, "entries": entries, "sides": sides, "salts": salts, "bcs": bcs,
            "leaves": leaves, "root": root, "work": Wd.bucket(entries[len(CALLS)].lo)}


def registration(w: dict) -> dict:
    k = len(DRAWN)
    return {"population": w["work"].n, "law": {"law": "subset", "k": k},
            "roots": [{"port": Wd.PORT, "root": w["root"].hex(), "leaves": P, "domain": w["domain"].domain_id.hex()}]}


def vectors() -> dict:
    w = window()
    reg = registration(w)
    a = w["sides"][0]["a"]
    sdom = w["side_domains"]["a"]
    return {
        "about": f"python {Path(__file__).name} rewrites this file; its docstring says what is pinned",
        "layout": {"entry_bytes": Wd.ENTRY_BYTES, "entry_words": Wd.ENTRY_WORDS, "depths": Wd.DEPTHS, "t_bits": Wd.T_BITS,
                   "kinds": Wd.KIND_CODE, "schema": Wd.ENTRY_SCHEMA},
        "window": {"domain": w["domain"].domain_id.hex(), "work": w["work"].to_json(1), "root": w["root"].hex()},
        "entries": [{"entry": x.to_json(), "row": x.row().hex(), "digest": Wd.entry_digest(x).hex(), "salt": s.hex(),
                     "bc": bc.hex(), "leaf": leaf.hex()}
                    for x, s, bc, leaf in zip(w["entries"], w["salts"], w["bcs"], w["leaves"])],
        "side": {"entry": 0, "side": "a", "domain": sdom.domain_id.hex(),
                 "domain_b": w["side_domains"]["b"].domain_id.hex(), "rows": a["rows"],
                 "leaves": [x.hex() for x in a["leaves"]], "pad_leaf": sdom.pad(a["rows"]).hex(),
                 "root": a["root"].hex(), "path_of_1": [x.hex() for x in Wd.side_path(sdom, a["levels"], 1)]},
        "public_inputs": {"registration": reg, "drawn": list(DRAWN), "cap": CAP,
                          "sessions": [Wd.public_inputs(reg, DRAWN, s, CAP).hex()
                                       for s in range(Wd.sessions(len(DRAWN), CAP))]},
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
