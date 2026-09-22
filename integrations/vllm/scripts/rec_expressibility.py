#!/usr/bin/env python
# ============================================================================================================================
# LEGACY (DIRECTIVE v4) — NOT THE CONTRACT PATH.  This table's classes "exempt (charged)" (a VU class counted into B_ex) and
# "input unit (verifier)" (Input gates as sampled VUs) are not in the contract: an exemption is a selected RU in F under an RU-count
# budget, an inexpressible kind is a completion risk (R18), inputs are not VUs (docs/PROTOCOL.md; PROTOCOL-CONFORMANCE R2/R8/R18).
# Refuses to run without --legacy-directive-v4 (R14); unchanged with it.  Contract path: record_v5/coverage_v5.py (unsupported kinds
# from the coverage table, A5/6's coverage_v1) — the legacy word "exempt" in this file's output is what verifier_v3 reads as
# "no checker" (selective_prover.py ~L402) until coverage_v5 replaces it.  [A10 2026-09-10, WR-8]
# ============================================================================================================================
"""[L1 RECORD 2026-09-10] Expressibility table: every kind of the P1/cu=whole census of B0 -> what the record does with a selected VU of it.

LEGACY (DIRECTIVE v4): see the banner above; requires ``--legacy-directive-v4``.

Classes (brief, DIRECTIVE §1.5.11, orchestrator rulings 2-3, Amendment A1):

* ``provable (approved ELF)`` -- the selected VU IS the unit the approved checker proves: a GEMM output coordinate (``GemmCoordinate_v1{K}``,
  statement ``GemmCoordinate_v2`` over the committed operand row, weight row and output word; K % 16 == 0); ELF 9fd00e60 (approved, archived).
* ``input unit (verifier)`` -- ``Input|batch``: prescribed value (weights: the registered weight commitment; prompt tokens / seed: the request
  commitment); per-RU-copy formulation (SMALL, tonight); the check opens the prover's commitment at the prescribed position and compares; enforced
  by statement validation (every statement references those roots).  Not a zk proof; adds no term to the Bound.
* ``exempt (charged)`` -- every other kind: interior not committed by today's replay (attention / norm / rope / silu / argmax interiors), or committed
  without an approved checker, or an ADAPTER kind (norm output cast via cone, embedding coordinate via row: ruling 3).  Charged as B_ex.

Counts: ``data/rec/p1-b0-lp32s15-validated.json#unit_counts.per_kind_fn_form`` (MEASURED, the validated plan).  Interface status per kind:
``data/rec/predicates-kind-table-tiny-p1-cu-operator.json`` (``e2e.predicates --table`` on TINY, the same operator structure; DERIVED, labelled).
The committed interfaces are those of today's replay: operator boundaries (operand / weight / output rows of every operator call) + the request
boundary (prompt, output tokens, params, seed) + the weights tree; no per-scalar interior of any operator.

    PYTHONPATH=. .venv/bin/python scripts/rec_expressibility.py --out out/record/expressibility.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
ROOT = POC.parent

# [A10 2026-09-10, WR-8] R14 legacy guard.
LEGACY = True
LEGACY_FLAG = "--legacy-directive-v4"
LEGACY_POINTER = (
    f"{Path(__file__).name}: LEGACY (DIRECTIVE v4) — 'exempt (charged)' VU classes / Input gates as VUs; refusing to run without {LEGACY_FLAG}.\n"
    "  The contract has no exempt families and no input VUs (docs/PROTOCOL.md; PROTOCOL-CONFORMANCE R2/R8/R18).\n"
    f"  Contract path: record_v5/coverage_v5.py (A9) over e2e.coverage_v1 (A5/6).  Historical reproduction: re-run with {LEGACY_FLAG}."
)


def legacy_gate(argv: list[str] | None) -> list[str] | None:
    args = list(sys.argv[1:] if argv is None else argv)
    if LEGACY_FLAG in args:
        return [x for x in args if x != LEGACY_FLAG]
    if any(x in ("-h", "--help") for x in args):
        return args
    print(LEGACY_POINTER, file=sys.stderr)
    return None


APPROVED_ELF = "9fd00e60fb6a74762de987cde211aca9de4576de844569da779acac6bf3863db"
APPROVED_VK = "0x0018b8e4e955ad4c348bcf57271643814dfcd68d575323410841c29289bd5797"
INPUT_TEXT = "prescribed; per-RU-copy formulation (SMALL, tonight); enforced by statement validation"
ADAPTER_KINDS = {"F2fpBf16|batch": "norm output cast (cone adapter would prove the cone, not the VU; ruling 3: exempt (charged))",
                 "GatherBf16x|batch": "embedding coordinate (row adapter would compare the whole row; ruling 3: exempt (charged))"}


def base_key(kind_fn_form: str) -> str:
    fn, form = kind_fn_form.rsplit("|", 1)
    base = re.sub(r"\{.*\}$", "", fn)
    base = re.sub(r"_v\d+$", "", base)
    base = re.sub(r"^(GatherBf16x)\d+$", r"\1", base)
    return f"{base}|{form}"


def classify(kind: str, count: int, tiny: dict[str, dict]) -> dict:
    fn, form = kind.rsplit("|", 1)
    bk = base_key(kind)
    t = tiny.get(bk, {})
    operators = sorted(t.keys()) if t else []
    interfaces = sorted({o["interface"] for o in t.values()}) if t else ["unknown (kind not in the TINY table)"]
    row = {"kind": kind, "count_per_request": count, "operators": operators, "interface": interfaces,
           "interface_source": "DERIVED: e2e.predicates --table on TINY P1/cu=operator (same operator structure); today's replay commits operator boundaries only"}
    if bk.startswith("GemmCoordinate|"):
        K = int(re.search(r"K=(\d+)", fn).group(1))
        ok = K % 16 == 0
        row.update({"class": "provable (approved ELF)" if ok else "exempt (charged)",
                    "how": f"statement GemmCoordinate_v2 (format 2: committed x row, weight row W[j,:], output word; K={K}, {K // 16} tc_dot16 steps; per-word "
                           f"Merkle authentication) under ELF {APPROVED_ELF[:12]}... vk {APPROVED_VK[:14]}... (approved, archived)" if ok else f"K={K} not a multiple of 16",
                    "interface_committed": "yes: operand row, weight row and output row are operator boundaries of the Gemm call (committed by the replay)"})
    elif bk == "Input|batch":
        row.update({"class": "input unit (verifier)", "how": INPUT_TEXT + "; check = open the registered weight commitment (weights) or the request commitment "
                    "(prompt tokens, seed) at the prescribed position and compare with the prover's committed word; a verifier check, not a proof; no Bound term",
                    "interface_committed": "yes: weights tree leaf (per row) / request boundary leaf"})
    elif bk in ADAPTER_KINDS:
        row.update({"class": "exempt (charged)", "how": f"ADAPTER kind: {ADAPTER_KINDS[bk]}",
                    "interface_committed": "output committed (operator output row); inputs interior" if bk == "F2fpBf16|batch" else "yes (embedding row committed)",
                    "additional_evidence": "cone / row checks on selected coordinates MAY be reported as additional evidence, not counted in the Bound (STRUCT-B §3 coverage condition)"})
    else:
        committed = interfaces == ["committed"]
        row.update({"class": "exempt (charged)",
                    "how": ("interface committed (operator boundary) but no approved checker takes this kind program" if committed else
                            "interior of its operator: not committed by today's replay (no per-scalar interior capture for attention / norm / rope / silu / argmax)"),
                    "interface_committed": "yes" if committed else ("partly" if "committed" in interfaces or "output-committed" in interfaces else "no")})
    return row


def main(argv: list[str] | None = None) -> int:
    argv = legacy_gate(argv)  # [A10 WR-8]
    if argv is None:
        return 2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(LEGACY_FLAG, action="store_true", help="REQUIRED: acknowledge this is the DIRECTIVE-v4 legacy expressibility table (see banner)")
    ap.add_argument("--validated", default=str(ROOT / "data" / "rec" / "p1-b0-lp32s15-validated.json"))
    ap.add_argument("--tiny-table", default=str(ROOT / "data" / "rec" / "predicates-kind-table-tiny-p1-cu-operator.json"))
    ap.add_argument("--out", required=True, help="expressibility.md (a .json is written beside it)")
    a = ap.parse_args(argv)
    val = json.loads(Path(a.validated).read_text())
    tiny_doc = json.loads(Path(a.tiny_table).read_text())
    tiny: dict[str, dict] = {}
    for key, t in tiny_doc["table"].items():
        bk = base_key(t["kind"])
        status, o = next(iter(t["outcomes"].items()))
        tiny.setdefault(bk, {})[t["operator"]] = {"interface": o["interface"], "status": status, "checker": o.get("checker")}
    counts = val["unit_counts"]["per_kind_fn_form"]
    rows = [classify(k, n, tiny) for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    total = sum(counts.values())
    by_class: dict[str, dict] = {}
    for r in rows:
        c = by_class.setdefault(r["class"], {"kinds": 0, "vus_per_request": 0})
        c["kinds"] += 1
        c["vus_per_request"] += r["count_per_request"]
    doc = {"schema": "veritor/rec/expressibility/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "program": val["program"]["function"], "program_digest": val["program"]["digest"], "plan": val["plan"]["name"], "partition_digest": val["plan"]["partition_digest"],
           "vus_per_request": total, "input_units_per_request": val["unit_counts"]["input_units"], "relation_vus_per_request": val["unit_counts"]["non_input_units"],
           "approved_elf": {"elf_sha256": APPROVED_ELF, "vk_hash": APPROVED_VK, "source": "e2e/approved_guests.json ('reproduced'); archived "
                            "fixtures/results/GUEST-2026-09-09/build-1/veritor-zk-guest.elf"},
           "committed_interfaces": ["request boundary: prompt tokens, output tokens, sampling params, seed (boundary tree per request; committed before the RU challenge)",
                                    "weights: one leaf per parameter row (weights tree; committed before the RU challenge; never recommitted on replay)",
                                    "operator boundaries of every operator call of a replayed request: operand rows, weight rows, output rows (interior trees, per RU)",
                                    "NOT committed: any per-scalar interior of attention / norm / rope / silu / argmax (exempt (charged))"],
           "by_class": by_class, "rows": rows,
           "labels": {"counts": "MEASURED (validated plan, data/rec/p1-b0-lp32s15-validated.json)", "interface": "DERIVED (TINY kind table, same operator structure)"}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    lines = [f"# Expressibility table — {doc['plan']} of `{doc['program']}` (partition `{doc['partition_digest'][:16]}…`), {doc['utc']}", "",
             f"Population per request: {total:,} units = {doc['relation_vus_per_request']:,} relation VUs + {doc['input_units_per_request']:,} input units "
             f"({INPUT_TEXT}). Approved ELF `{APPROVED_ELF[:16]}…` (vk `{APPROVED_VK[:18]}…`), the only checker family. Counts MEASURED (validated plan); "
             "interface status DERIVED from the TINY kind table (same operator structure), labelled.", "",
             "Committed interfaces (today's replay): " + "; ".join(doc["committed_interfaces"]) + ".", "",
             "| class | kinds | units per request |", "|---|---:|---:|"]
    for c, v in sorted(by_class.items()):
        lines.append(f"| {c} | {v['kinds']} | {v['vus_per_request']:,} |")
    lines += ["", "| kind (fn|form) | units / request | class | interface (operators) | how |", "|---|---:|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['kind']}` | {r['count_per_request']:,} | **{r['class']}** | {', '.join(r['interface'])} ({', '.join(r['operators']) or '—'}); committed: "
                     f"{r['interface_committed']} | {r['how']} |")
    lines += ["", "Exceptions by name and width (DIRECTIVE §1.5.9): `ArgmaxStep_v1|scan` — the token-selection scan step as the partition contains it (its outgoing "
              "interface is the scan state, listed here by name; not clamped; exempt (charged) tonight).", "",
              "Rulings applied: 2 (interiors = what the replay captures; REFUSED kinds exempt (charged)), 3 (ADAPTER kinds exempt (charged); cone/row checks only as "
              "additional evidence), A1 (inputs prescribed, not sampled as a class of their own; the per-request input units of P1 stay in the population as validated)."]
    out.write_text("\n".join(lines) + "\n")
    print(f"[rec_expressibility] {len(rows)} kinds -> {out}: " + ", ".join(f"{c}: {v['kinds']} kinds / {v['vus_per_request']:,} units" for c, v in sorted(by_class.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
