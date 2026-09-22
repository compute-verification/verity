"""IT's artefact-side negatives against a committed ``run_composed`` record (lane INT3; int3-brief addendum).

    python -m e2e.artefact_negatives --src fixtures/e2e-composed/INT-B0-20260908T0725Z --out /tmp/neg [--cases a b1 b2 b3 c]

Not ``run_composed --negative`` (WA's prover-side mutations, applied to the *live* run and caught by the bundle
verifier): these edit the **committed typed record** -- ``obligations/composed.json``, ``statement.bin``,
``witness.bin`` -- of a copy of the record in a temp directory, re-derive every digest the record carries
so the edited record is internally consistent, and run the second process (``e2e.verify_composed``, recorded
program / plan modes, recorded proofs) over the copy.  The committed record is never touched.

The five cases and their edits are lane IT's (``fixtures/it/int_negatives.py``, ``independent-tests.md`` §5),
imported from there and applied unchanged; only the *choice of leaves* for the swap cases is made from the
record instead of IT's hard-coded ids (which belong to ``INT-B0-20260908T0235Z``):

* ``a``  -- a validly committed wrong result: obligation 0's output word moved one ulp, the leaf, its tree root,
           the sibling paths, the boundary word, h_O, h_L and the witness re-derived.  Intended refusal: the
           obligation's relation (arithmetic).
* ``b1`` -- two leaves of one tree with the *same* layout swap ``(rank, position)``; values and witness untouched.
           Intended refusal: placement (``verify_composed.check_placement``, IT-F2).
* ``b2`` -- two leaves of one tree with *different* layouts swap ``(rank, position)``.
* ``b3`` -- the domain ``position`` alone swapped between the b1 pair.  Intended refusal: placement (IT-F2).
* ``c``  -- obligation 0 (a coordinate unit the challenge selected) and its boundary words removed;
           ``selection.json``, the challenge and the proofs stay.  Intended refusal: coverage
           (``verify_composed.check_coverage``, IT-F1).

"Same layout" is the packed descriptor's (``u16le:<count>``): :func:`leaf_pairs` takes the first tree (interior and
weights trees before boundary trees) with two leaves of equal word count -- in the @2 records the chain row's RU
tree, whose attention row (``[9,64]`` in B0, ``[12,128]`` in B1) and o_proj output row pack the same number of u16
words, as IT's original pair of 576-wide rows did -- and for ``b2`` the first with two of different counts (a
GEMM output row and its ``x``).  Case ``c`` is IT's with the witness trimmed to what the reduced statement opens
(:func:`case_c`; IT's untouched witness fitted because its removed unit shared every tree and leaf with a sibling).
The proofs are not re-made (no prover binary here; SP1 proving is unkeyed, so the proofs row's digest refusal is
not one a re-proving adversary would meet -- ``independent-tests.md`` §5): what this measures is which
*records-row* check refuses each edit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent
for p in (str(POC), str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

IT_NEGATIVES = ROOT / "fixtures" / "it" / "int_negatives.py"
CASES = ("a", "b1", "b2", "b3", "c")
INTENDED = {"a": "relations (arithmetic: check_relations / transparent backend RELATION_REJECTED)",
            "b1": "placement (check_placement: the authenticated leaf at the claimed rank carries another id)",
            "b2": "placement (check_placement), then the backend's INVALID_OPENING on the leaf size",
            "b3": "placement (check_placement: position != rank)",
            "c": "coverage (check_coverage: a challenge-selected unit has no obligation)"}


def it_module() -> Any:
    """Lane IT's ``int_negatives.py`` as a module (``fixtures/it`` is not a package)."""
    spec = importlib.util.spec_from_file_location("it_int_negatives", IT_NEGATIVES)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(IT_NEGATIVES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _words(leaf: dict[str, Any]) -> int:
    layout = leaf["layout"]
    n = 1
    while isinstance(layout, dict) and "array" in layout:
        n *= int(layout["array"][0])
        layout = layout["array"][1]
    return n


def leaf_pairs(obj: dict[str, Any]) -> dict[str, tuple[str, str] | None]:
    """The record's own b1/b3 pair (two leaves of one tree, equal layouts) and b2 pair (different layouts); interior
    and weights trees before boundary trees, lowest ranks first, so the choice is a function of the record."""
    by_root: dict[str, list[dict[str, Any]]] = {}
    for leaf in obj["leaves"]:
        by_root.setdefault(leaf["root"], []).append(leaf)
    same: tuple[str, str] | None = None
    diff: tuple[str, str] | None = None
    for root in sorted(by_root, key=lambda r: (r.startswith("boundary/"), r)):
        leaves = sorted(by_root[root], key=lambda l: l["leaf"])
        for i in range(len(leaves)):
            for k in range(i + 1, len(leaves)):
                if _words(leaves[i]) == _words(leaves[k]):
                    same = same or (leaves[i]["id"], leaves[k]["id"])
                else:
                    diff = diff or (leaves[i]["id"], leaves[k]["id"])
    return {"same": same, "diff": diff}


def case_c(it: Any, d: Path) -> dict[str, Any]:
    """IT's case c (obligation 0 -- a challenge-selected coordinate unit -- and its boundary words removed; leaves,
    references, roots, ``selection.json``, challenge, proofs untouched), with one adaptation the @2 records need:
    when the removed obligation was the only one opening a tree or a leaf, the statement of the reduced object no
    longer names them, so the witness is trimmed to the leaves the reduced statement opens (the same authenticated
    leaves, none re-made).  IT's original left the witness untouched because its removed unit shared every tree and
    leaf with a sibling; here obligation 0 alone opens its weight row leaf (and, in B0, its RU tree)."""
    from veritor.protocol.proofs.packed import PackedWitness, check_packed_openings, decode_packed_witness, encode_packed_witness, packed_table
    from veritor.protocol.proofs.wire import decode_statement

    odir = d / "obligations"
    record = json.loads((odir / "composed.json").read_text())
    o = record["object"]
    old_st = decode_statement(bytes.fromhex(record["statement"])).expand()
    old_table = packed_table(old_st)
    old_witness = decode_packed_witness(bytes.fromhex(record["witness"]))
    leaf_at = {(tree.commitment.root, rank): old_witness.trees[t][k] for t, tree in enumerate(old_table.trees) for k, rank in enumerate(tree.ranks)}
    removed = o["obligations"].pop(0)
    removed_values = record["values"].pop(0)
    it.rederive(record, odir, None)  # statement / h_L / h_O of the reduced object; the witness hex is still the old one
    st2 = decode_statement(bytes.fromhex(record["statement"])).expand()
    table2 = packed_table(st2)
    trimmed = PackedWitness(tuple(tuple(leaf_at[(tree.commitment.root, rank)] for rank in tree.ranks) for tree in table2.trees))
    dropped = len(leaf_at) - sum(len(t) for t in trimmed.trees)
    it.rederive(record, odir, encode_packed_witness(trimmed))
    check_packed_openings(st2, trimmed)  # the trimmed witness authenticates under the reduced statement
    sel = json.loads((odir / "selection.json").read_text())
    return {"case": "c", "edit": "obligation 0 removed from object.obligations and values; leaves/references/roots, selection.json, challenge, proofs untouched; "
                                 f"witness trimmed to the {sum(len(t) for t in trimmed.trees)} leaves the reduced statement opens ({dropped} dropped, none re-made)",
            "removed_obligation": {"subcircuit": next(s["definition"] for s in o["subcircuits"] if s["digest"] == removed["subcircuit"]), "out": removed["boundary"]["out"], "n_words": len(removed_values)},
            "remaining_obligations": len(o["obligations"]), "selection_says": sel["selection"].get("note"), "witness_leaves_dropped": dropped,
            "self_check": "trimmed witness authenticates under the reduced statement (check_packed_openings)"}


def build(src: Path, case: str, dst: Path) -> dict[str, Any]:
    """A copy of the committed record ``src`` at ``dst`` with IT's ``case`` applied and every digest re-derived."""
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; one of {CASES}")
    it = it_module()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("reverify", "reverify-it", "logs", "log.txt"))
    if case == "a":
        edit = it.case_a(dst)
    elif case == "c":
        edit = case_c(it, dst)
    else:
        obj = json.loads((dst / "obligations" / "composed.json").read_text())["object"]
        pairs = leaf_pairs(obj)
        pair = pairs["diff" if case == "b2" else "same"]
        if pair is None:
            raise RuntimeError(f"case {case}: the record has no tree with two leaves of {'different' if case == 'b2' else 'the same'} layout")
        fields = ("position",) if case == "b3" else ("leaf", "position")
        edit = {"case": case, "edit": {"b1": "two leaves of one tree with the same layout swap (rank, position); values and witness untouched",
                                       "b2": "two leaves of one tree with different layouts swap (rank, position)",
                                       "b3": "the domain `position` alone swapped between the b1 pair (ranks kept)"}[case],
                **it._swap(dst, pair[0], pair[1], fields)}
    edit["intended_refusal"] = INTENDED[case]
    edit["source"] = str(src)
    (dst / "edit.json").write_text(json.dumps(edit, indent=1, sort_keys=True) + "\n")
    return edit


def second_process(dst: Path) -> dict[str, Any]:
    """``verify_composed`` in recorded modes over the copy; the rows and the records row's per-check verdicts."""
    from e2e import verify_composed

    t = time.perf_counter()
    rc = verify_composed.main(["--dir", str(dst), "--out", str(dst / "reverify-it"), "--skip-bundle", "--allow-recorded-proofs", "--program", "recorded", "--plan", "recorded"])
    doc = json.loads((dst / "reverify-it" / "reverify.json").read_text())
    rec = doc["rows"]["records"]
    return {"exit": rc, "seconds": round(time.perf_counter() - t, 2), "fresh_verdict": doc["fresh_verdict"], "recorded_verdict": doc["recorded_verdict"],
            "rows": {k: v["status"] for k, v in doc["rows"].items()},
            "records": {"status": rec["status"], "detail": rec["detail"], "problems": rec.get("problems") or [],
                        "coverage": (rec.get("coverage") or {}).get("status"), "coverage_missing": (rec.get("coverage") or {}).get("missing"),
                        "placement": (rec.get("placement") or {}).get("status"), "placement_problems": (rec.get("placement") or {}).get("problems"),
                        "relations": rec.get("relations"), "witness_verdict": rec.get("witness_verdict")},
            "proofs": {"status": doc["rows"]["proofs"]["status"], "detail": doc["rows"]["proofs"]["detail"]}}


def refused_by(records: dict[str, Any]) -> str:
    """The first records-row check that refused, in the order ``check_records`` runs them."""
    for p in records["problems"]:
        if p.startswith("coverage:"):
            return "coverage"
        if p.startswith("placement:"):
            return "placement"
        if p.startswith("roots:"):
            return "roots"
        if p.startswith("relations:"):
            return "relations"
        if p.startswith("transparent backend"):
            return "transparent backend"
        if p.startswith("h_O") or p.startswith("canonical") or p.startswith("re-lowered") or p.startswith("h_L"):
            return "digest"
    return "nothing" if records["status"] == "ok" else "other"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="a committed run_composed record (never modified)")
    ap.add_argument("--out", required=True, help="where the edited copies and the results go")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    a = ap.parse_args(argv)
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"schema": "verity-vllm/e2e-composed/artefact-negatives/v1", "src": str(src), "record": src.name, "cases": {}}
    for case in a.cases:
        dst = out / case
        edit = build(src, case, dst)
        sp = second_process(dst)
        sp["refused_by"] = refused_by(sp["records"])
        results["cases"][case] = {"edit": edit, "second_process": sp}
        print(f"[{case}] {edit['edit']}")
        print(f"   records {sp['records']['status']}: refused by {sp['refused_by']}; intended: {INTENDED[case]}")
        for p in sp["records"]["problems"]:
            print(f"     - {p[:260]}")
        print(f"   proofs {sp['proofs']['status']}; fresh verdict {sp['fresh_verdict']} (recorded {sp['recorded_verdict']}), {sp['seconds']} s")
    (out / "artefact_negatives.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    print(f"-> {out / 'artefact_negatives.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
