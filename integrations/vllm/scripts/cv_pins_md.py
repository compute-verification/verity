"""[E6 CURVE 2026-09-13, D54 item 2] PINS.md for E3: the four SmolLM2-135M pins OLD (original program, kept under their own ids for the accepted records)
beside NEW (corrected attention program, E7 dfcea003; revision r2), with `program_revision` = the program digest, N_V, partition / rows digests,
the byte-identity verdicts (population rows; interior rows where the producer ran), the kinds table (32 kinds; per-kind counts old vs new) and the
bound at s = 2^-14, q = 1, delta = 1e-12 (from scale_bounds' JSON when present).  Reads only files under --pinned (the pod chain's layout) and the
old pins in the tree.  Every number is copied from a MEASURED file and labelled by its source path.

    python scripts/cv_pins_md.py --pinned /workspace/cv/e6/pinned --out /workspace/cv/e6/pinned/PINS.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

OLD = {  # size -> (point label, old instance dir in the tree, accepted record on the old pin)
    "4x8": ("4x9", "out/eod/e6/tiny", "n4 (E10, TINY; ACCEPTED)"),
    "32x15": ("32x16", "out/night/e6/b0-32x15", "sc1 (E8; ACCEPTED, T1)"),
    "128x63": ("128x64", "out/scale/e6/pinned/SmolLM2-135M__128x64", "a4-dry (E1; first mismatch t = 64 -> D54); PHASE 2 run pending"),
    "256x63": ("256x64", "out/scale/e6/pinned/SmolLM2-135M__256x64", "none (PHASE 2 run pending)"),
}


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _nv(inst):
    v = inst["partition"].get("vu_count")
    return v.get("value") if isinstance(v, dict) else v


def _rows_v1(inst):
    import re
    m = re.search(r"rows' v1 digest ([0-9a-f]+)", inst["partition"].get("source", "")) if inst else None
    return m.group(1) if m else "?"


def _kinds(exp):
    return {k["kind"]: int(k["count"]) for k in (exp or {}).get("kinds", [])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tree", default=str(ROOT))
    a = ap.parse_args(argv)
    pinned, tree = Path(a.pinned), Path(a.tree)
    pdig = _j(pinned / "program_digests.json") or {}
    geo = _j(pinned.parent / "GEOMETRY.json") or {}
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    L = [f"# PINS — SmolLM2-135M (B0): OLD pins (legacy identity) beside the NEW geometry pins (E6, {utc}; generator `scripts/cv_pins_md.py`; every value copied from the named file)", "",
         "**D57.** The kernel geometry is a declared static of the pinned program identity (`protocol_instance.json#program.identity.attention = {kernel, kBlockM, kBlockN}`); "
         "ABSENT = the LEGACY program (kBlockN 64), which is what the four accepted records (n4, sc1, s4, q3) keep — their ids and program digests are unchanged and the new tree "
         "rebuilds them (the `legacy regression` column: the NEW tree with the LEGACY identity reproduces the old pin's program / plan-rule / rows / partition digests). "
         f"NEW r2 pins declare the MEASURED geometry **{json.dumps(geo.get('attention')) if geo else '(none supplied)'}** — source: {geo.get('source', '—')} — and carry it as `program.attention_provenance`. "
         "`program_revision` = the program digest; the identity's `attention` field is printed beside it.", "",
         "| point | which | instance id | program_revision (program digest) | attention (identity) | partition digest | rows' v1 digest | plan-rule digest | N_V | kinds | request id | legacy regression (new tree, legacy identity) | geometry rows == legacy rows | interior rows (records, legacy) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    kinds_rows = []
    bounds_rows = []
    for size, (pt, olddir, rec) in OLD.items():
        old = _j(tree / olddir / "protocol_instance.json")
        oexp = _j(tree / olddir / "expected_population.json")
        new = _j(pinned / size / "instance" / "protocol_instance.json")
        nexp = _j(pinned / size / "instance" / "expected_population.json")
        ident = _j(pinned / size / "IDENTITY.json") or {}
        legacy = _j(pinned.parent / "legacy" / size / "IDENTITY.json") or {}
        leg_v = "not run" if not legacy else ("**" + legacy.get("verdict", "?") + "**")
        rows = _j(pinned.parent / "rows" / pt / "ROWS_IDENTITY.json") or {}
        rows_v = "not run" if not rows else ("SKIPPED: " + rows["skipped"] if rows.get("skipped") else (f"**IDENTICAL** ({rows['common']} files)" if rows.get("byte_identical") else f"**DIFFER** ({rows.get('differing')} of {rows.get('common')} files; only-in-record {len(rows.get('only_in_record', []))}, only-in-produced {len(rows.get('only_in_produced', []))})"))
        pt_d = pdig.get("points", {}).get(size, {})
        if old:
            L.append(f"| {pt} | OLD (legacy) | `{old['instance_id']}` | `{old['program']['digest'][:16]}…` | legacy (absent) = kBlockN 64 | `{old['partition']['partition_digest'][:16]}…` | `{_rows_v1(old)}` | `{old['partition']['plan_descriptor_digest'][:16]}…` | {_nv(old):,} | {len(_kinds(oexp))} | `{(old['epoch'].get('request_ids') or ['?'])[0]}` | {leg_v} | — | {rows_v} (record {rec}) |")
        if new:
            vl = ident.get("vs_legacy_build", {})
            L.append(f"| {pt} | **NEW r2** | `{new['instance_id']}` | **`{new['program']['digest']}`** | `{json.dumps(new['program']['identity'].get('attention'))}` | `{new['partition']['partition_digest'][:16]}…` | `{_rows_v1(new)}` | `{new['partition']['plan_descriptor_digest'][:16]}…` | {_nv(new):,} | {len(_kinds(nexp))} | `{(new['epoch'].get('request_ids') or ['?'])[0]}` | — | "
                     f"{('**YES**' if vl.get('rows_identical') else '**NO** (expected past one block)') if vl else '?'} | — |")
        elif (pinned / size / "FAILED.json").exists():
            L.append(f"| {pt} | NEW r2 | **FAILED** (`{size}/BUILD.log`) | `{pt_d.get('geometry_program_digest', '?')}` | `{json.dumps(geo.get('attention'))}` | — | — | — | — | — | — | — | — | — |")
        else:
            L.append(f"| {pt} | NEW r2 | not pinned yet | `{pt_d.get('geometry_program_digest', pt_d.get('geometry_program_digest_refused', '?'))}` | `{json.dumps(geo.get('attention'))}` | — | — | — | — | — | — | — | — | — |")
        if old and new:
            ok, nk = _kinds(oexp), _kinds(nexp)
            kinds_rows.append((pt, ok, nk))
        b = _j(pinned / "bounds" / f"SmolLM2-135M__{pt}__r2.json")
        if b:
            bd, sp = b["bound"], b["sampling"]
            bounds_rows.append(f"| {pt} | NEW r2 | {b['population']['N_V']:,} | {sp['s_as_2_pow']} | {sp['q']} | {sp['delta']} | {sp['expected_Q']:.0f} | {bd['j_star']:,} | {bd['beta_pct']:.3f} % | {bd['reading']} |")
    L += ["", "## Kinds table: 32 kinds, per-kind counts OLD vs NEW (`expected_population.json` of each pin; the FULL kind table, not a sample)", ""]
    for pt, ok, nk in kinds_rows:
        same = sorted(set(ok) & set(nk)); changed = [k for k in same if ok[k] != nk[k]]
        L.append(f"**{pt}** — kinds old {len(ok)} / new {len(nk)}; kind SET {'unchanged' if set(ok) == set(nk) else 'CHANGED: only-old ' + str(sorted(set(ok) - set(nk))) + ' only-new ' + str(sorted(set(nk) - set(ok)))}; "
                 f"counts changed in {len(changed)} kinds; N_V old {sum(ok.values()):,} -> new {sum(nk.values()):,} ({sum(nk.values()) - sum(ok.values()):+,}).")
        if changed:
            L += ["", "| kind | old | new | delta |", "|---|---|---|---|"] + [f"| `{k}` | {ok[k]:,} | {nk[k]:,} | {nk[k] - ok[k]:+,} |" for k in changed] + [""]
    L += ["", "## Bound at the declared law (s = 2^-14, q = 1, δ = 1e-12; `scale_bounds.py` on the NEW pins; no record on them yet → |Q| = E|Q|)", "",
          "| point | pin | N_V | s | q | δ | E\\|Q\\| | j* | β | reading |", "|---|---|---|---|---|---|---|---|---|---|"] + (bounds_rows or ["| — | — | bounds not computed on the pod; run `scale_bounds.py --points scripts/cv_points.json` after the pull | | | | | | | |"])
    L += ["", "## What E3's census needs", "",
          "* `program_revision` column = `program.digest`; print `program.identity.attention` beside it (absent = legacy 64) and `program.attention_provenance.source` once per table.",
          "* the pinned PROGRAM is what `can_express` / the census read: `protocol_instance.json` → `program.digest` (new) — the kind ids `AttnBlock_v1{D=64,NVIS=n,FIRST}` now range n ∈ 1..128 and the non-first blocks are `NVIS=128,FIRST=False` (was 64): the shape-key SET changes at 128×64 / 256×64 even where the 32 kinds do not; at ≤ 48 keys (4×9, 32×16) the shape keys are unchanged.",
          "* the 4×9 / 32×16 kinds tables are the old ones (kind-by-kind equal, above) — a FULL kind-table compare is the census statement there; 128×64 / 256×64 need the census re-run (or the kinds-table FULL compare + shape-key check if the kinds are unchanged, per the brief).",
          "* files per pin (`/workspace/cv/e6/pinned/<size>/instance/`): `protocol_instance.json`, `population.v2.json` (the rows), `expected_population.json` (kinds table), `inventory.json`, `coverage_table.json`, `pins.json`; `SHA256SUMS` beside them; `IDENTITY.json` = old-vs-new digests (MEASURED)."]
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"[cv_pins_md] wrote {a.out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
