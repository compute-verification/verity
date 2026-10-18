"""Identity-set comparison of a v2 (Q_module_body_v1) manifest against a v1 manifest -- every difference classified.

The unit of comparison is the harness's: the IDENTITY_KEY tuple `(program_digest, request_id, step, invocation, op_path, output_member,
element_range, rank)` (`tests/regression/checks/manifest_digest._identity_key`).  Differences are paired by their *address*
`(request_id, step, invocation, op_path)` and classified::

    member_rename     same address and range, different member name         -> a v1 spelling the structure does not give (a residual)
    range             same address and member, different element range      -> the two disagree on the rows / width of one member
    only_v1           an address v1 names and v2 does not                    -> a v1 vocabulary claim the dataflow does not support, or a v2 gap
    only_v2           an address v2 names and v1 does not                    -> a crossing v1's vocabulary called interior (F-r19-int-20), or a v2 excess

`explain` adds the reason the bridge knows for each class (the v2 row's `boundary` annotation: crossing rows / returned rows /
structural member; the v1 row's `spec`), so a report can say for every difference whether it is a named residual, a v1 by-name table
the structure contradicts, or a v2 gap.  Nothing here changes either manifest.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

__all__ = ["identity_key", "compare", "summarize"]

IDENTITY_KEY = ("program_digest", "request_id", "step", "invocation", "op_path", "output_member", "element_range", "rank")


def identity_key(r: dict[str, Any]) -> str:
    return json.dumps([r[k] if k != "element_range" else list(r[k]) for k in IDENTITY_KEY], separators=(",", ":"), sort_keys=True)


def _address(r: dict[str, Any]) -> tuple:
    return (str(r["request_id"]), str(r["step"]), int(r["invocation"]), str(r["op_path"]), int(r["rank"]))


def compare(v1_rows: Iterable[dict[str, Any]], v2_rows: Iterable[dict[str, Any]], *, ignore_program_digest: bool = False) -> dict[str, Any]:
    """The classified difference of two identity populations (see the module docstring)."""
    def key(r):
        if ignore_program_digest:
            return json.dumps([r[k] if k != "element_range" else list(r[k]) for k in IDENTITY_KEY[1:]], separators=(",", ":"))
        return identity_key(r)
    a: dict[str, dict] = {}
    b: dict[str, dict] = {}
    mult_a: Counter = Counter()
    mult_b: Counter = Counter()
    for r in v1_rows:
        k = key(r); a[k] = r; mult_a[k] += 1
    for r in v2_rows:
        k = key(r); b[k] = r; mult_b[k] += 1
    # a global manifest repeats one IDENTITY_KEY per engine step for the request-less hidden-stream rows (key has no engine_step):
    # the harness digests rows WITH multiplicity, so a key present on both sides with different counts is a difference too
    multiplicity = [{"key": k, "v1_rows": mult_a[k], "v2_rows": mult_b[k]} for k in a.keys() & b.keys() if mult_a[k] != mult_b[k]]
    only_a = [a[k] for k in a.keys() - b.keys()]
    only_b = [b[k] for k in b.keys() - a.keys()]
    # pair by address
    by_addr_a: dict[tuple, list] = defaultdict(list)
    by_addr_b: dict[tuple, list] = defaultdict(list)
    for r in only_a:
        by_addr_a[_address(r)].append(r)
    for r in only_b:
        by_addr_b[_address(r)].append(r)
    renames: list[dict] = []
    ranges: list[dict] = []
    unpaired_a: list[dict] = []
    unpaired_b: list[dict] = []
    for addr in set(by_addr_a) | set(by_addr_b):
        ra, rb = list(by_addr_a.get(addr, [])), list(by_addr_b.get(addr, []))
        # same member -> range difference
        for x in list(ra):
            m = next((y for y in rb if y["output_member"] == x["output_member"]), None)
            if m is not None:
                ranges.append({"v1": x, "v2": m})
                ra.remove(x); rb.remove(m)
        # same range -> member rename
        for x in list(ra):
            m = next((y for y in rb if list(y["element_range"]) == list(x["element_range"])), None)
            if m is not None:
                renames.append({"v1": x, "v2": m})
                ra.remove(x); rb.remove(m)
        unpaired_a += ra
        unpaired_b += rb
    return {"v1_identities": len(a), "v2_identities": len(b), "common": len(a.keys() & b.keys()),
            "v1_rows": sum(mult_a.values()), "v2_rows": sum(mult_b.values()), "multiplicity_diff_n": len(multiplicity),
            "only_v1_n": len(only_a), "only_v2_n": len(only_b),
            "member_rename": renames, "range": ranges, "only_v1": unpaired_a, "only_v2": unpaired_b, "multiplicity": multiplicity[:50]}


def _sig(r: dict[str, Any], side: str) -> tuple:
    op = r["op_path"]
    # drop the layer index so identical differences across layers fold into one line
    parts = op.split(".")
    op_fold = ".".join("<k>" if p.isdigit() else p for p in parts)
    if side == "v2":
        bd = r.get("boundary") or {}
        return (op_fold, r["output_member"], r.get("spec"), bd.get("structural_member"), "returned" if bd.get("returned_rows") else "", "crossing" if bd.get("crossing_rows") else "")
    return (op_fold, r["output_member"], r.get("spec"))


def summarize(cmp: dict[str, Any]) -> dict[str, Any]:
    """Fold the classified differences into counted lines with a reason each."""
    out: dict[str, Any] = {k: cmp[k] for k in ("v1_identities", "v2_identities", "common", "v1_rows", "v2_rows", "multiplicity_diff_n", "only_v1_n", "only_v2_n")}
    out["multiplicity"] = cmp.get("multiplicity", [])
    ren: Counter = Counter()
    for d in cmp["member_rename"]:
        ren[(_sig(d["v1"], "v1")[:3], _sig(d["v2"], "v2"))] += 1
    out["member_rename"] = [{"v1": {"op_path": k[0][0], "member": k[0][1], "spec": k[0][2]}, "v2": {"op_path": k[1][0], "member": k[1][1], "spec": k[1][2], "structural": k[1][3]},
                             "n": n, "reason": "v1 member name differs from the structural member: a spelling the structure does not give (named residual)"}
                            for k, n in sorted(ren.items(), key=lambda kv: -kv[1])]
    rng: Counter = Counter()
    rng_ex: dict = {}
    for d in cmp["range"]:
        k = (_sig(d["v1"], "v1"), _sig(d["v2"], "v2"))
        rng[k] += 1
        rng_ex.setdefault(k, (d["v1"].get("rows"), d["v2"].get("rows"), d["v1"]["element_range"], d["v2"]["element_range"]))
    out["range"] = [{"op_path": k[0][0], "member": k[0][1], "spec": k[0][2], "n": n, "example": {"v1_rows": rng_ex[k][0], "v2_rows": rng_ex[k][1], "v1_range": rng_ex[k][2], "v2_range": rng_ex[k][3]},
                     "v2_boundary": {"structural": k[1][3], "returned": k[1][4], "crossing": k[1][5]},
                     "reason": "the two disagree on the rows of one member: v1 counts the vocabulary's rows, v2 the required Values"}
                    for k, n in sorted(rng.items(), key=lambda kv: -kv[1])]
    o1: Counter = Counter()
    for r in cmp["only_v1"]:
        o1[_sig(r, "v1")] += 1
    out["only_v1"] = [{"op_path": k[0], "member": k[1], "spec": k[2], "n": n,
                       "reason": "v1 names this member; under Q_module_body_v1 no required Value has this address/member: either a v1 by-name claim the "
                                 "dataflow does not support (no Call outside the module reads it, the Program does not return it, no return slot recorded) or a v2 gap"}
                      for k, n in sorted(o1.items(), key=lambda kv: -kv[1])]
    o2: Counter = Counter()
    fam2: dict[tuple, str] = {}
    for r in cmp["only_v2"]:
        k = _sig(r, "v2")
        o2[k] += 1
        fam2.setdefault(k, str(r.get("family")))

    def _why2(k: tuple) -> str:
        if fam2.get(k, "").endswith("_stream"):
            return "a kernel-side stream row (hidden / MoE planes): per-component here, unioned per engine step in a global manifest"
        if k[5] == "crossing":
            return "a Value that crosses the module-body boundary in the Program's dataflow which v1's vocabulary called interior (F-r19-int-20 class)"
        if k[4] == "returned":
            return "a recorded return slot the dataflow does not read (RETURN-SLOT policy) that v1 does not name"
        return "a v2 row with no v1 counterpart"

    out["only_v2"] = [{"op_path": k[0], "member": k[1], "spec": k[2], "structural": k[3], "returned": k[4], "crossing": k[5], "family": fam2.get(k), "n": n, "reason": _why2(k)}
                      for k, n in sorted(o2.items(), key=lambda kv: -kv[1])]
    out["unexplained"] = 0    # filled by the caller's classification of only_v1 / only_v2 lines
    return out
