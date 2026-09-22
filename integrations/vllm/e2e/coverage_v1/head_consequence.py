"""VU-count and exposure consequence of the whole-head attention VU (S55, PROPOSED partition change) -- A5/6 brief item 4.

Reads ``coverage.json`` (``e2e.coverage_v1.kind_table``) and, per configuration, replaces the attention-body kinds by one
``AttentionHead_v1{T,D}|call`` unit per (position, layer, head) -- ``P x NL x NH`` per RU, which is exactly the census's
``Fa2InvSum_v1|primitive`` count (one epilogue per head call).  Reports, per RU and over 4 RUs, ``n_R`` before/after and the
exposure ``1 - (1 - s)^n`` at s in {2^-14, 2^-19, 2^-30}, under: (a) approved set as-built (before); (b) head VU adopted AND the fa2
guest approved (after; the head has no other checker); (c) after + A5/6 elementwise fallbacks accounted.  ``Const32[0x00000000]``
is shared between attention and the rest of the body and is kept in the unsupported set (conservative).

    python -m e2e.coverage_v1.head_consequence --coverage data/contract/a56-coverage/coverage.json --out data/contract/a56-coverage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .kind_table import S_VALUES, exposure

#: Kinds of the registered attention body (`verity_vllm/registry/b1.py:335-424`) that the head unit absorbs.
ATTENTION_BODY_KINDS = (
    "AmpereBF16TcDot16_v1|batch", "DotBf16_v1{K=64}|call", "F2fpBf16_v1|primitive", "F32MulFtz_v1|primitive", "MufuEx2Ftz_v1|primitive",
    "F32FmaSubFtz_v1|primitive", "F32Max_v1|primitive", "F32AddFtz_v1|primitive", "F32FmaFtz_v1|primitive", "F32MulFtz_v1|batch",
    "GuardNegInfZero_v1|primitive", "F32SubFtz_v1|primitive", "Fa2InvSum_v1|primitive", "Const32[0x3e38aa3b]_v1|primitive", "Const16[0x0000]_v1|primitive",
)
HEAD_KIND = "AttentionHead_v1{T,D}|call"


def _exposures(n: int, n_r: int) -> dict[str, Any]:
    return {"per_RU": {k: exposure(n, s) for k, s in S_VALUES.items()}, "4RU": {k: exposure(n * n_r, s) for k, s in S_VALUES.items()}}


def consequence(doc: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"head_kind": HEAD_KIND, "attention_body_kinds": list(ATTENTION_BODY_KINDS), "configurations": {}}
    for name, cfg in doc["configurations"].items():
        counts: dict[str, int] = cfg["counts_per_RU"]
        n_r = int(cfg.get("N_R", 4))
        total = sum(counts.values())
        attn = {k: counts[k] for k in ATTENTION_BODY_KINDS if k in counts}
        attn_total = sum(attn.values())
        head_units = counts.get("Fa2InvSum_v1|primitive")
        approved = cfg["by_scenario"]["approved"]
        unsupported_before = list(approved["unsupported_kinds"])
        n_before = int(approved["n_R"])
        # after: the attention-body kinds leave the population, one head unit per (position, layer, head) enters; the head is
        # supported only if the fa2 guest is approved (no other checker exists), so "after" presumes that approval.
        removed = sum(counts[k] for k in unsupported_before if k in attn)
        n_after = n_before - removed
        total_after = total - attn_total + (head_units or 0)
        fallbacks = list(approved.get("unsupported_kinds_with_a56_fallbacks", unsupported_before))
        n_after_fb = int(approved.get("n_R_with_a56_fallbacks", n_before)) - sum(counts[k] for k in fallbacks if k in attn)
        rows = []
        for k in ATTENTION_BODY_KINDS:
            if k in counts:
                rows.append({"kind": k, "count_per_RU": counts[k], "share_of_population": f"{counts[k] / total:.4%}", "unsupported_before": k in unsupported_before,
                             "exposure_before_per_RU": {s: exposure(counts[k], v)["probability"][:12] for s, v in S_VALUES.items()} if k in unsupported_before else None,
                             "after": "absorbed into the head unit"})
        out["configurations"][name] = {
            "source": cfg.get("source"), "N_R": n_r, "N_V_per_RU_before": total, "N_V_per_RU_after": total_after,
            "attention_body_VUs_per_RU": attn_total, "attention_share_before": f"{attn_total / total:.4%}",
            "head_units_per_RU": head_units, "head_units_note": "P x NL x NH = the Fa2InvSum count (one epilogue per head call)",
            "reduction_factor_attention": (attn_total / head_units) if head_units else None,
            "per_kind": rows,
            "before": {"scenario": "approved set as-built (9fd00e60), per-primitive attention VUs", "unsupported_kinds": unsupported_before, "n_R": n_before,
                       "N_4RU": n_before * n_r, "exposure": _exposures(n_before, n_r)},
            "after": {"scenario": "head VU adopted (PROPOSED partition change) AND fa2 guest 33cba058 approved (PROPOSED)",
                      "unsupported_kinds": [k for k in unsupported_before if k not in attn], "n_R": n_after, "N_4RU": n_after * n_r,
                      "exposure": _exposures(n_after, n_r)},
            "after_with_a56_fallbacks": {"scenario": "after + SiluMulBf16/Bf16AddF2fp fallbacks (elem_builder, transferred to CK-ELEM) accounted",
                                         "unsupported_kinds": [k for k in fallbacks if k not in attn], "n_R": n_after_fb, "N_4RU": n_after_fb * n_r,
                                         "exposure": _exposures(n_after_fb, n_r)},
            "head_only_if_fa2_not_approved": {"scenario": "head VU adopted, fa2 NOT approved: every head unit is unsupported (no other checker)",
                                              "n_R": n_after + (head_units or 0), "exposure": _exposures(n_after + (head_units or 0), n_r)},
        }
    return out


def to_markdown(doc: dict[str, Any]) -> str:
    L = ["# Whole-head attention VU (S55, PROPOSED) -- count and exposure consequence", "",
         "Probability of selecting an unsupported check -- completion risk, `1 - (1 - s)^n`, exact arithmetic.  `after` presumes BOTH the",
         "partition change (head unit; new digests, counts, certificates -- the user's decision) AND approval of the fa2 guest `33cba058`", ""]
    for name, c in doc["configurations"].items():
        L += [f"## {name}", "", f"source: {c['source']}", "",
              f"* attention-body VUs per RU: **{c['attention_body_VUs_per_RU']:,}** of {c['N_V_per_RU_before']:,} ({c['attention_share_before']}) -> "
              f"**{c['head_units_per_RU']:,}** head units (P x NL x NH); attention VU count / {c['reduction_factor_attention']:,.0f}",
              f"* population per RU: {c['N_V_per_RU_before']:,} -> {c['N_V_per_RU_after']:,}", ""]
        L += ["| scenario | n_R | N (4 RU) | P(unsupported selected) per RU @2^-14 / 2^-19 / 2^-30 | E[#unsupported selected] per RU @2^-14 / 2^-19 / 2^-30 |", "|---|---|---|---|---|"]
        for key in ("before", "after", "after_with_a56_fallbacks", "head_only_if_fa2_not_approved"):
            sc = c[key]
            e = sc["exposure"]["per_RU"]
            p = " / ".join(e[s]["probability"][:10] for s in ("2^-14", "2^-19", "2^-30"))
            ex = " / ".join(e[s]["expected_unsupported_selected_decimal"][:10] for s in ("2^-14", "2^-19", "2^-30"))
            L.append(f"| {sc['scenario']} | {sc['n_R']:,} | {sc.get('N_4RU', sc['n_R'] * c['N_R']):,} | {p} | {ex} |")
        L += ["", "Per attention-body kind (before: each is its own unsupported set; after: absorbed into the head unit):", "",
              "| kind | count per RU | share | unsupported before | P(selected) per RU before @2^-14 / 2^-19 / 2^-30 |", "|---|---|---|---|---|"]
        for r in c["per_kind"]:
            eb = r["exposure_before_per_RU"]
            L.append(f"| `{r['kind']}` | {r['count_per_RU']:,} | {r['share_of_population']} | {r['unsupported_before']} | "
                     + (" / ".join(eb[s] for s in ("2^-14", "2^-19", "2^-30")) if eb else "-") + " |")
        L.append("")
    L += ["What `after` does NOT remove: `Const32[0x00000000]` (shared with the rest of the body; kept), the RMSNorm / rope / sampler / gather kinds",
          "(CK-ELEM's), `ArgmaxStep`.  What `after` requires: (1) the user adopts the head unit as the attention VU (partition change); (2) the fa2",
          "guest is approved (its `tc_dot16` NaN extension is total, `fa2.rs:254-298`, = the registered primitive; the approved guest's is not); (3) the",
          "record commits V rows (`qkv_proj.gemm/port=0`) -- present in the REC profile, absent in `profile-hash`.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = consequence(json.loads(Path(a.coverage).read_text()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "head_vu_consequence.json").write_text(json.dumps(doc, indent=1))
    (out / "head_vu_consequence.md").write_text(to_markdown(doc))
    print(f"wrote {out / 'head_vu_consequence.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
