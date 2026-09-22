"""The paper table from `costlog/v1` files (MEASURED runs) and the night table's MODELED rows (the comparison column).

    python -m e2e.scale.table out/scale/table out/night/table -o out/scale/TABLE.md [--csv out/scale/table/rows.csv] [--fig out/scale/fig/scaling.png]

* Table 1 -- one row per (model, prompt x output, run): the measured stage costs, the MODELED value of the same size beside each measured one
  (from `out/night/table/<model>__<p>x<o>.json` when it carries a MODELED field), peak memory, bytes stored / read, the sampling parameters and
  the bound (s, q, delta, N_V, |Q|, proof-checked / directly checked units, U), the verdicts (live / re-verify / re-verify after repair) and the
  stage reached.  Every number carries its label in the cell (MEASURED numbers plain; MODELED in italics with `~`; PROVISIONAL marked `?`).
* Table 2 -- the cost split per stage in % of the summed wall, so the bottleneck reads off the table.
* Table 3 -- the MODELED rows of the night table that have no measured run (the comparison column on its own).
* Table 4 -- verifier bytes by bucket and mode (E9's accounting: archive check vs `--deployed-openings`).
* Table 5 -- a before/after on the same job (`*.before_after.json` beside the costlogs; E2's reconstruction).
* Table P -- the paper table: one compact row per configuration (accepted record where one exists, MODELED beside, stage reached otherwise).
* CSV -- one line per row and stage for the plot; the figure is log-log stage cost vs tokens (MEASURED solid, MODELED dashed).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from e2e.scale import costlog as CL

STAGE_SHORT = {"serve": "serve", "reconstruction": "reconstruct", "commitment": "commit", "obligations": "obligations", "proving": "prove",
               "verify_live": "verify (live)", "reverify": "re-verify"}

#: night-table field -> (stage, costlog field) for the MODELED comparison column
NIGHT_MAP = {"serve_s": ("serve", "wall_s"), "reconstruction_s": ("reconstruction", "wall_s"), "commit_s": ("commitment", "wall_s"),
             "prove_s": ("proving", "wall_s"), "verify_live_s": ("verify_live", "wall_s"), "reverify_s": ("reverify", "wall_s"),
             "prove_peak_mem_gb": ("proving", "gpu_peak_bytes"), "commit_peak_rss_gb": ("commitment", "peak_rss_bytes")}


def _lab(x: dict[str, Any] | None) -> str:
    if not isinstance(x, dict):
        return "UNKNOWN"
    lab = str(x.get("label") or "UNKNOWN").upper()
    for known in CL.LABELS:  # night-table labels are prose ("MEASURED: serve_v5 ...", "NOT RUN (running since ...)"); keep the schema word
        if lab.startswith(known):
            return known
    return "UNKNOWN"


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):  # e.g. night commit_s {"commit_io_s", "interior_commit_s"}
        parts = [_num(x) for x in v.values()]
        parts = [p for p in parts if p is not None]
        return sum(parts) if parts else None
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def fmt_s(x: float | None, lab: str = "MEASURED") -> str:
    if x is None:
        return "—"
    if x < 1:
        s = f"{x:.3f} s"
    elif x < 100:
        s = f"{x:.1f} s"
    elif x < 3600:
        s = f"{x / 60:.1f} min"
    else:
        s = f"{x / 3600:.2f} h"
    return mark(s, lab)


def fmt_bytes(x: float | None, lab: str = "MEASURED") -> str:
    if x is None:
        return "—"
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if x >= div:
            return mark(f"{x / div:.2f} {unit}", lab)
    return mark(f"{int(x)} B", lab)


def fmt_int(x: Any, lab: str = "MEASURED") -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and not x.is_integer():
        return mark(f"{x:,.1f}", lab)
    return mark(f"{int(x):,}", lab)


def mark(s: str, lab: str) -> str:
    lab = (lab or "UNKNOWN").upper()
    if lab.startswith("MEASURED"):
        return s
    if lab.startswith("MODELED"):
        return f"*~{s}*"
    if lab.startswith("PROVISIONAL"):
        return f"{s}?"
    return f"{s} ({lab.lower()})"


def load_costlogs(d: Path) -> list[dict[str, Any]]:
    docs = []
    for p in sorted(d.glob("*.costlog.json")):
        doc = CL.upgrade(json.load(open(p)))  # rows written by an earlier emitter get the new fields as UNKNOWN
        errs = CL.validate(doc)
        if errs:
            raise SystemExit(f"{p}: costlog does not validate: {errs[:3]}")
        doc["_path"] = str(p)
        docs.append(doc)
    return docs


def load_before_after(d: Path) -> list[dict[str, Any]]:
    """`*.before_after.json` beside the costlogs (schema veritor/scale/reconstruction-before-after/v1): one optimisation measured on the SAME job."""
    out = []
    for p in sorted(d.glob("*.before_after.json")):
        doc = json.load(open(p))
        if str(doc.get("schema", "")).startswith("veritor/scale/reconstruction-before-after/v1"):
            doc["_path"] = str(p)
            out.append(doc)
    return out


def load_night(d: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not d or not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        if p.name.endswith(".e3.json") or p.name.endswith(".merged.json"):
            continue
        m = re.match(r"(.+?)__(\d+)x(\d+)\.json$", p.name)
        if not m:
            continue
        try:
            row = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(row, dict):
            continue
        out[(m.group(1), int(m.group(2)), int(m.group(3)))] = row
    return out


def night_field(row: dict[str, Any] | None, key: str) -> tuple[float | None, str]:
    if not row or key not in row:
        return None, "UNKNOWN"
    f = row[key]
    if isinstance(f, dict) and "value" in f:
        return _num(f["value"]), _lab(f)
    return _num(f), "UNKNOWN"


def night_modeled(row: dict[str, Any] | None, key: str) -> tuple[float | None, str]:
    """The night table's value for the comparison column -- only when it is MODELED (a MEASURED night number is not a model of anything)."""
    v, lab = night_field(row, key)
    if v is None or not lab.startswith("MODELED"):
        return None, lab
    if key.endswith("_gb") and v is not None:
        v = v * 1e9
    return v, lab


def key_of(doc: dict[str, Any]) -> tuple[str, int, int]:
    return (CL.value(doc, "id.model_short"), int(CL.value(doc, "id.prompt_len")), int(CL.value(doc, "id.output_len")))


MODEL_ORDER = {"SmolLM2-135M": 0, "SmolLM2-360M": 1}  # the curve's checkpoint first, its sibling next, other architectures after


def sort_key(doc: dict[str, Any]) -> tuple:
    m = CL.value(doc, "id.model_short")
    return (MODEL_ORDER.get(m, 9), m, int(CL.value(doc, "id.tokens") or 0), str(CL.value(doc, "id.run_id")))


def stage_cell(doc: dict[str, Any], st: str, night: dict[str, Any] | None, night_key: str | None) -> str:
    v = CL.value(doc, f"stages.{st}.wall_s")
    lab = CL.label(doc, f"stages.{st}.wall_s")
    cell = fmt_s(v, lab)
    if night_key:
        mv, mlab = night_modeled(night, night_key)
        if mv is not None:
            cell += f" / {fmt_s(mv, mlab)}"
    return cell


def verdict_cell(doc: dict[str, Any]) -> str:
    st = doc["status"]
    live, rev, rep = st["verdict_live"]["value"], st["verdict_reverify"]["value"], st["verdict_reverify_repaired"]["value"]
    ind = st.get("verdict_independent", {}).get("value")
    reached = st["stage_reached"]["value"]
    if live is None and rev is None and reached in (None, "none", "instance", "serve", "reconstruction", "commitment", "obligations", "proving"):
        h = st["history"]["value"]
        return f"stage reached **{reached or '—'}** — no verdict" + (f"; {str(h)[:220]}" if h else "")
    parts = [f"live **{live or '—'}**", f"shipped re-verify **{rev or '—'}**"]
    if rep:
        parts.append(f"preserved record after verifier repair **{rep}**")
    if ind:
        parts.append(f"independent (T1) **{ind}**")
    ref = st["refusal"]["value"]
    if ref:
        parts.append("refused at " + ", ".join(f"`{c}`" for c in ref))
    return "; ".join(parts)


def verdict_short(doc: dict[str, Any]) -> str:
    """Table P's verdict cell: live / shipped / independent, with the stage reached when nothing ran to a verdict."""
    st = doc["status"]
    live, rev = st["verdict_live"]["value"], st["verdict_reverify"]["value"]
    ind = st.get("verdict_independent", {}).get("value")
    rep = st["verdict_reverify_repaired"]["value"]
    reached = st["stage_reached"]["value"]
    if live is None and rev is None:
        ref = st["refusal"]["value"]
        if ref:
            return f"stage reached: **{reached or '—'}** — **REFUSED**: {'; '.join(re.sub(r'^REFUSED:\s*', '', str(c)) for c in ref)}"
        h = str(st["history"]["value"] or "")
        why = h.split("→")[0].split(";")[0].strip()
        if len(why) > 150:
            why = why[:150].rsplit(" ", 1)[0] + "…"
        return f"stage reached: **{reached or '—'}**" + (f" ({why})" if why else "") + census_suffix(doc)
    def v(x):
        return "—" if x is None else ("**VERIFIED**" if x == "VERIFIED" else f"**{x}**")
    ind_cell = v(ind)
    if ind is None and "PENDING" in str(st.get("verdict_independent", {}).get("note") or ""):
        ind_cell = "T1 independent: PENDING"   # the integrator flips this marker when T1's verdict lands (attach_t1 then fills the field)
    cell = f"{v(live)} / {v(rev)} / {ind_cell}"
    if rep:
        cell += f" (preserved record after repair {v(rep)})"
    ref = st["refusal"]["value"]
    if ref and live == "REFUSED":
        cell += " at " + ", ".join(f"`{str(c).split(' ')[0]}`" for c in ref[:2])
        note = str(st["refusal"].get("note") or "")
        m = re.search(r"([A-Za-z0-9_]+_v\d+\|[a-z-]+: [A-Z_]+)", note)
        if m:
            cell += f" (`{m.group(1)}`, verifier-side)"
    if live == "REFUSED" and not rep:
        pending = [seg.strip() for seg in str(st["history"]["value"] or "").split("→") if "pending" in seg]
        if pending:
            cell += " → " + " → ".join(p_.split("(")[0].strip() for p_ in pending)
    return cell


def _dm(doc: dict[str, Any], mode: str, key: str = "bytes_read") -> tuple[Any, str]:
    """A bucket leaf of the `--deployed-openings` run (transfer.deployed_mode), or (None, UNKNOWN)."""
    dm = doc["transfer"].get("deployed_mode")
    if not isinstance(dm, dict) or not isinstance(dm.get(mode), dict) or not isinstance(dm[mode].get(key), dict):
        return None, "UNKNOWN"
    return dm[mode][key].get("value"), _lab(dm[mode][key])


def transfer_cell(doc: dict[str, Any]) -> str:
    """`archive / deployed-sparse` bytes read in the default entry point (archive check ON); then the same two buckets under --deployed-openings."""
    a, dsp = CL.value(doc, "transfer.archive_check.bytes_read"), CL.value(doc, "transfer.deployed_sparse.bytes_read")
    cell = f"{fmt_bytes(a, CL.label(doc, 'transfer.archive_check.bytes_read'))} / {fmt_bytes(dsp, CL.label(doc, 'transfer.deployed_sparse.bytes_read'))}"
    da, dl = _dm(doc, "archive_check")
    dd, ddl = _dm(doc, "deployed_sparse")
    if da is not None or dd is not None:
        cell += f"; deployed mode: {fmt_bytes(da, dl)} / {fmt_bytes(dd, ddl)}"
    else:
        pb = CL.value(doc, "transfer.deployed_sparse.parts.proofs.bytes_read")
        if pb is not None and a is None:
            cell += f" (proofs {fmt_bytes(pb, CL.label(doc, 'transfer.deployed_sparse.parts.proofs.bytes_read'))})"
    return cell


def program_cell(doc: dict[str, Any]) -> str:
    """`id.program_revision` (D54 §5) · `id.attention` (D57): the bound program's digest, 12 hex chars, then the geometry its identity declares
    ("legacy BN=64" for a pre-D54 identity, "fa2-splitkv-align 64×128" for a D57 pin); "—" for each when the row predates the field or nothing
    is pinned.  The four accepted regressions and the re-pinned 128×64 / 256×64 records are told apart by this cell alone."""
    f = doc["id"].get("program_revision")
    rev = "—" if not isinstance(f, dict) or f.get("value") is None else mark(f"`{str(f['value'])[:12]}`", _lab(f))
    a = doc["id"].get("attention")
    if not isinstance(a, dict) or a.get("value") is None:
        return rev
    return f"{rev} · {mark(str(a['value']), _lab(a))}"


def census_suffix(doc: dict[str, Any]) -> str:
    c = (doc["status"].get("detail") or {}).get("census")
    if isinstance(c, dict) and c.get("value"):
        return f"; census: {c['value']}"
    return ""


def guarantee_cells(doc: dict[str, Any]) -> list[str]:
    g = doc["guarantee"]
    def gv(k):
        return g[k]["value"]
    def gl(k):
        return _lab(g[k])
    s = CL._frac(gv("s")) if gv("s") is not None else "—"
    q = CL._frac(gv("q")) if gv("q") is not None else "—"
    delta = CL._frac(gv("delta")) if gv("delta") is not None else "—"
    U, W = gv("U_bits"), gv("W_bits")
    ub = "—" if U is None else mark(f"U = {U} bits" + (f" (= W)" if W is not None and U == W else (f" of W = {W}" if W is not None else "")), gl("U_bits"))
    if U is None and isinstance((g.get("detail") or {}).get("U_unknown_reason"), dict) and g["detail"]["U_unknown_reason"].get("value"):
        ub = "U UNKNOWN ¶ (certificate not built: A9 `E_COST_CLASS`; Table P footnote)" + (f", W = {W} bits" if W is not None else "")
    if gv("beta") is not None:
        ub += f"; β = {100 * float(gv('beta')):.2f} %" + (f" (j* = {int(gv('j_star')):,})" if gv("j_star") is not None else "")
    checked = "—"
    if gv("proof_checked_units") is not None or gv("directly_checked_units") is not None:
        checked = f"{fmt_int(gv('proof_checked_units'), gl('proof_checked_units'))} proof-checked ({fmt_int(gv('n_proofs'), gl('n_proofs'))} proofs) + {fmt_int(gv('directly_checked_units'), gl('directly_checked_units'))} directly; uncovered {fmt_int(gv('uncovered'), gl('uncovered'))}"
    return [fmt_int(gv("N_V"), gl("N_V")), f"s = {s}, q = {q}, δ = {delta}, B_ex = {gv('B_ex') if gv('B_ex') is not None else '—'}",
            fmt_int(gv("Q"), gl("Q")), checked, ub]


def _accepted_rows(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Table P: one row per configuration -- the accepted record where one exists (live VERIFIED), else the latest run (refused or stage-only)."""
    by_cfg: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for d in docs:
        by_cfg.setdefault(key_of(d), []).append(d)
    out = []
    for k, runs in by_cfg.items():
        acc = [d for d in runs if CL.value(d, "status.verdict_live") == "VERIFIED"]
        pick = sorted(acc or runs, key=lambda d: str(CL.value(d, "status.utc") or ""))[-1]
        pick = dict(pick)
        pick["_history_runs"] = [str(CL.value(d, "id.run_id")) for d in runs if d is not pick and CL.value(d, "id.run_id") != CL.value(pick, "id.run_id")]
        out.append(pick)
    return sorted(out, key=sort_key)


def _s_short(v, lab="MEASURED"):
    return fmt_s(v, lab) if v is not None else "—"


def table_p(docs: list[dict[str, Any]], night: dict[tuple[str, int, int], dict[str, Any]], before_after: list[dict[str, Any]]) -> list[str]:
    L = ["## Table P — the paper table (one row per configuration; the accepted record where one exists; *~italic* = MODELED beside the measured value)\n"]
    hdr = ["model", "P×O (tokens)", "N_V", "\\|Q\\|", "law (s, q, δ)", "U / β", "serve", "reconstruction", "commit + obligations", "prove (n proofs)",
           "verify live / re-verify", "verifier bytes archive / deployed", "record / slim bytes", "verdicts live / shipped / independent"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join(["---"] * len(hdr)) + "|")
    ba_after = {(b.get("model", "").split("/")[-1], int(b["prompt_len"]), int(b["output_len"])): b for b in before_after if b.get("prompt_len")}
    regress_notes: list[tuple] = []     # † rows: T1's REGRESS4 on the D57 tree (status.detail.regression_d57)
    transition_notes: list[tuple] = []  # ‡ rows: E2's six transition records at the row's geometry (status.detail.transition_pass6)
    per_chunk_notes: list[dict] = []    # § the per-chunk prove time across the two large records (stages.proving.detail.per_chunk_note; printed once)
    u_unknown_notes: list[tuple] = []   # ¶ rows whose U is UNKNOWN for a named reason (guarantee.detail.U_unknown_reason)
    for doc in _accepted_rows(docs):
        k = key_of(doc)
        nrow = night.get(k)
        g = doc["guarantee"]
        gv = lambda kk: g[kk]["value"]  # noqa: E731
        gl = lambda kk: _lab(g[kk])  # noqa: E731
        law = "—" if gv("s") is None else f"s = {CL._frac(gv('s'))}, q = {CL._frac(gv('q')) if gv('q') is not None else '—'}, δ = {CL._frac(gv('delta')) if gv('delta') is not None else '—'}"
        U = gv("U_bits")
        ub = "—" if U is None else mark(f"U = {U} b" + (" (= W)" if gv("W_bits") == U else ""), gl("U_bits"))
        ur = (g.get("detail") or {}).get("U_unknown_reason")
        if U is None and isinstance(ur, dict) and ur.get("value"):
            ub = "U UNKNOWN ¶" + (f" (W = {gv('W_bits')} b)" if gv("W_bits") is not None else "")
            u_unknown_notes.append((k, str(CL.value(doc, "id.run_id")), ur))
        if gv("beta") is not None:
            ub += f" / β = {100 * float(gv('beta')):.2f} %"
        def sc(st, nk):
            return stage_cell(doc, st, nrow, nk)
        c, o = CL.value(doc, "stages.commitment.wall_s"), CL.value(doc, "stages.obligations.wall_s")
        if c is None and o is None:
            co = "—"
        elif o is None:  # a commitment measured without obligations (a phase A dry commitment, a run cut before obligations): print it alone, named
            c_note = str((doc["stages"]["commitment"].get("wall_s") or {}).get("note") or "")
            co = _s_short(c, CL.label(doc, "stages.commitment.wall_s")) + (" (dry)" if "DRY" in c_note.upper() else "") + " + —"
        else:
            co = f"{_s_short(c, CL.label(doc, 'stages.commitment.wall_s'))} + {_s_short(o, CL.label(doc, 'stages.obligations.wall_s'))}"
        mv, ml = night_modeled(nrow, "commit_s")
        if mv is not None:
            co += f" / {fmt_s(mv, ml)}"
        rec = sc("reconstruction", "reconstruction_s")
        ba = ba_after.get(k)
        if ba and ba.get("after", {}).get("wall_s", {}).get("value") is not None:
            rec += f"; after `{str(ba['after'].get('tree', '')).split(' ')[0]}`: {fmt_s(ba['after']['wall_s']['value'], _lab(ba['after']['wall_s']))}"
        if k == ("SmolLM2-135M", 512, 128) or k == ("SmolLM2-135M", 128, 64):
            for b in before_after:
                mn = (b.get("modeled_next") or {}).get(f"{k[0]}__{k[1]}x{k[2]}")
                if mn and mn.get("value") is not None and CL.value(doc, "stages.reconstruction.wall_s") is None:
                    rec += f" (after-producer model: {fmt_s(mn['value'], _lab(mn))})"
        npf = gv("n_proofs")
        prove = sc("proving", "prove_s") + (f" ({fmt_int(npf, gl('n_proofs'))})" if npf is not None else "")
        pd = doc["stages"]["proving"].get("detail") or {}
        if isinstance(pd.get("healthy_single_pod_s"), dict) and pd["healthy_single_pod_s"].get("value") is not None:
            # a prove resumed across two pods (128x64 a1: the first pod's GPU was bad; 256x64 a1: the pod's 4 h age ceiling): the two-pod total is the
            # MEASURED wall above; the split (`prove_split`) and the single-pod estimate are printed beside it, labelled
            rt, hs = pd.get("healthy_s_per_chunk") or {}, pd["healthy_single_pod_s"]
            split = (pd.get("prove_split") or {}).get("value")
            if not split:
                dg, hl = pd.get("degraded_pod_s") or {}, pd.get("healthy_pod_s") or {}
                split = f"two-pod total: {fmt_s(dg.get('value'), _lab(dg))} degraded GPU + {fmt_s(hl.get('value'), _lab(hl))} healthy"
            est_min = mark(f"{float(hs['value']) / 60:.0f} min", _lab(hs))
            prove += f" — {split}; healthy single-pod {est_min} at {float(rt.get('value') or 0):.1f} s/chunk"
            if isinstance(pd.get("per_chunk_note"), dict) and pd["per_chunk_note"].get("value"):
                prove += " §"
                if not per_chunk_notes:
                    per_chunk_notes.append(pd["per_chunk_note"])
        ver = f"{sc('verify_live', 'verify_live_s')} / {sc('reverify', 'reverify_s')}"
        src = doc
        a = CL.value(doc, "transfer.archive_check.bytes_read")
        via = ""
        if a is None and _dm(doc, "deployed_sparse")[0] is None:
            for other in docs:  # E9 measured the configuration on another record of the same size (e.g. n3 for 32x16): shown, named
                if key_of(other) == k and other is not doc and CL.value(other, "transfer.archive_check.bytes_read") is not None:
                    src, via = other, f" (record `{CL.value(other, 'id.run_id')}`, same configuration)"
                    break
        doc_ = doc
        doc = src
        a, dsp = CL.value(doc, "transfer.archive_check.bytes_read"), CL.value(doc, "transfer.deployed_sparse.bytes_read")
        da, dl = _dm(doc, "archive_check")
        dd, ddl = _dm(doc, "deployed_sparse")
        ideal = doc["transfer"].get("deployed_ideal") if isinstance(doc["transfer"].get("deployed_ideal"), dict) else None
        if a is None and dd is None:
            vb = "—"
        elif ideal and ideal.get("bytes_transferred_to_verifier_with_stored_openings"):
            # the three byte numbers kept apart (integrator 03:52Z, 08:45Z): the archive check's read / hashed in the default entry point (and the same bucket
            # under --deployed-openings when that run was made), the deployed sparse read / hashed of THIS run (E9's bucket as run: openings rebuilt from
            # whole trees / composed.json re-read, hence the size), and the deployed-ideal transfer with STORED openings, with its breakdown
            ah, dh = CL.value(doc, "transfer.archive_check.bytes_hashed"), CL.value(doc, "transfer.deployed_sparse.bytes_hashed")
            vb = f"archive check {fmt_bytes(a, CL.label(doc, 'transfer.archive_check.bytes_read'))} read / {fmt_bytes(ah, CL.label(doc, 'transfer.archive_check.bytes_hashed'))} hashed (default entry point)"
            if da is not None:
                dah, dahl = _dm(doc, "archive_check", "bytes_hashed")
                vb += f", {fmt_bytes(da, dl)} read / {fmt_bytes(dah, dahl)} hashed under `--deployed-openings`"
            vb += (f"; deployed sparse {fmt_bytes(dsp, CL.label(doc, 'transfer.deployed_sparse.bytes_read'))} read / {fmt_bytes(dh, CL.label(doc, 'transfer.deployed_sparse.bytes_hashed'))} hashed "
                   f"(E9's bucket as run — whole trees / composed.json re-read for {fmt_int(ideal.get('openings'))} openings); "
                   f"deployed-ideal with stored openings {fmt_bytes(ideal['bytes_transferred_to_verifier_with_stored_openings'])} = proofs {fmt_bytes(ideal.get('proof_bytes'))} + "
                   f"statements {fmt_bytes(ideal.get('statement_bytes'))} + typed objects {fmt_bytes(ideal.get('typed_object_bytes'))} + leaves {fmt_bytes(ideal.get('leaf_bytes'))} + "
                   f"paths {fmt_bytes(ideal.get('path_bytes'))}") + via
        else:
            vb = f"{fmt_bytes(a, CL.label(doc, 'transfer.archive_check.bytes_read'))} / {fmt_bytes(dd if dd is not None else dsp, ddl if dd is not None else CL.label(doc, 'transfer.deployed_sparse.bytes_read'))}"
            if da is not None:
                vb += f" (archive bucket in deployed mode {fmt_bytes(da, dl)})"
            vb += via
        doc = doc_
        stor = f"{fmt_bytes(CL.value(doc, 'storage.record_bytes'), CL.label(doc, 'storage.record_bytes'))} / {fmt_bytes(CL.value(doc, 'storage.slim_bytes'), CL.label(doc, 'storage.slim_bytes'))}"
        vd = verdict_short(doc)
        sdet = doc["status"].get("detail") or {}
        if isinstance(sdet.get("regression_d57"), dict) and sdet["regression_d57"].get("value"):
            vd += " †"   # T1's REGRESS4: re-verified on the D57 tree (footnote below the table)
            regress_notes.append((k, str(CL.value(doc, "id.run_id")), sdet["regression_d57"]))
        if isinstance(sdet.get("transition_pass6"), dict) and sdet["transition_pass6"].get("value"):
            vd += " ‡"   # E2's six transition records PASS at this row's geometry (footnote below the table)
            transition_notes.append((k, str(CL.value(doc, "id.run_id")), sdet["transition_pass6"]))
        prog = program_cell(doc)
        prog_s = f", program {prog}" if prog != "—" else ""
        if doc.get("_history_runs"):
            vd += f" (run `{CL.value(doc, 'id.run_id')}`{prog_s}; history: {', '.join('`' + r + '`' for r in doc['_history_runs'])} in Table 1)"
        else:
            vd += f" (run `{CL.value(doc, 'id.run_id')}`{prog_s})"
        nv_cell = fmt_int(gv("N_V"), gl("N_V"))
        for dk, dv in (g.get("detail") or {}).items():  # a second N_V of another FORM (e.g. E2's retained/argmax build at 512x128) is printed beside, named
            if dk.startswith("N_V_") and dk.endswith("_form") and isinstance(dv, dict) and isinstance(dv.get("value"), int):
                nv_cell += f" (protocol form; {dk[4:-5].replace('_', ' ')}: {fmt_int(dv['value'], _lab(dv))} — reconciled in the row's `guarantee.N_V` note)"
        row = [k[0], f"{k[1]}×{k[2]} ({CL.value(doc, 'id.tokens')})", nv_cell, fmt_int(gv("Q"), gl("Q")), law, ub,
               sc("serve", "serve_s"), rec, co, prove, ver, vb, stor, vd]
        L.append("| " + " | ".join(c_.replace("|", "\\|") for c_ in row) + " |")
    # MODELED-only sizes as italic rows
    have = {key_of(d) for d in docs}
    for k, r in sorted(night.items(), key=lambda kr: (kr[0][0], kr[0][1] + kr[0][2])):
        if k in have:
            continue
        def nf(key, f):
            v, lab = night_field(r, key)
            return f(v, lab)
        U = (r.get("certified_U_at_s") or {}) if isinstance(r.get("certified_U_at_s"), dict) else {}
        row = [k[0], f"{k[1]}×{k[2]} ({k[1] + k[2]})", nf("N_V", fmt_int), nf("Q", fmt_int), "s = 2^-14, q = 1, δ = 1e-12 (declared for the curve)",
               mark(f"U = {U.get('U_bits')} b" + (" (= W)" if U.get("U_equals_W") else ""), "MODELED") if U.get("U_bits") else "—",
               nf("serve_s", fmt_s), nf("reconstruction_s", fmt_s), nf("commit_s", fmt_s), nf("prove_s", fmt_s), "—", "—", "—", "no run (night table model only)"]
        L.append("| " + " | ".join(row) + " |")
    L.append("\nColumns: N_V = verification units of the pinned population; |Q| = the sampled units the verifier checked; U in bits is the certified bound on the "
             "output set an accepting adversary can reach (R1's statement per row below); β = j*/N_V, the fraction of faulty units needed to escape with probability δ. "
             "`verifier bytes archive / deployed` = bytes read by the shipped verifier's archive check (recomputing roots from every archived leaf) and by the deployed "
             "verifier's authenticated openings of the sampled units (`--deployed-openings`; E9's instrument), both MEASURED per record where E9 ran it. "
             "Verdicts: live driver on the run pod / shipped re-verifier in a fresh environment / T1's independent re-verification on another machine.\n")
    if regress_notes:
        src = regress_notes[0][2].get("source") or ""
        L.append("† **re-verified on the D57 tree** (T1's REGRESS4, `" + src.split(" + ")[-1] + "`): the record's own frozen tree is unchanged; the SHIPPED verifier of main "
                 "`5dc435ab8` (D57 merged) re-verified the record in both `reverify.py` entry points and rebuilt its ORIGINAL pinned program from the tree's source to the pinned "
                 "digest (`V1.program_rebuilt`) — the legacy geometry as the default when the identity carries none.\n")
        for k, run, f in regress_notes:
            L.append(f"* {k[0]} {k[1]}×{k[2]} `{run}`: {f['value']}")
        L.append("")
    if transition_notes:
        f = transition_notes[0][2]
        rows_ = ", ".join(f"{k[0]} {k[1]}×{k[2]} `{run}`" for k, run, _ in transition_notes)
        L.append(f"‡ **six transition records {f['value']}** at this geometry (E2, `{f.get('source')}`; rows {rows_}): {f.get('note')}\n")
    if per_chunk_notes:
        f = per_chunk_notes[0]
        L.append(f"§ **per-chunk prove time** ({f.get('source')}): {f['value']}\n")
    for k, run, f in u_unknown_notes:
        L.append(f"¶ **U {k[0]} {k[1]}×{k[2]} `{run}`** ({f.get('source')}): {f['value']}\n")
    return L


def table_4(docs: list[dict[str, Any]]) -> list[str]:
    rows = [d for d in docs if CL.value(d, "transfer.archive_check.bytes_read") is not None or _dm(d, "deployed_sparse")[0] is not None]
    if not rows:
        return []
    L = ["## Table 4 — verifier bytes by bucket and mode (E9's `verify_accounting.json`; `bytes read / hashed`, × = ratio to record bytes)\n"]
    hdr = ["model", "size", "run", "record bytes", "mode", "archive check", "deployed sparse", "checkpoint", "unclassified", "peak RSS (deployed bucket)", "wall", "stored openings would be"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for d in sorted(rows, key=sort_key):
        k = key_of(d)
        rb = CL.value(d, "storage.record_bytes") or d["transfer"].get("record_bytes")
        for mode_name, tr in (("archive check (default entry point)", d["transfer"]), ("`--deployed-openings`", d["transfer"].get("deployed_mode"))):
            if not isinstance(tr, dict) or not isinstance(tr.get("archive_check"), dict) or tr["archive_check"].get("bytes_read", {}).get("value") is None:
                continue
            cells = []
            for b in ("archive_check", "deployed_sparse", "checkpoint", "unclassified"):
                br, bh = tr[b]["bytes_read"]["value"], tr[b]["bytes_hashed"]["value"]
                ratio = f" ({br / rb:.2f}×)" if rb and br is not None and b in ("archive_check", "deployed_sparse") else ""
                cells.append(f"{fmt_bytes(br)} / {fmt_bytes(bh)}{ratio}" if br is not None else "—")
            rss = tr["deployed_sparse"].get("peak_rss_bytes", {}).get("value")
            wall = tr.get("wall_s_total")
            ideal = (tr.get("deployed_ideal") or {})
            ideal_cell = "—"
            if ideal.get("path_bytes") is not None:
                ideal_cell = f"paths {fmt_bytes(ideal['path_bytes'])} + leaves {fmt_bytes(ideal.get('leaf_bytes'))} for {ideal.get('openings')} openings; + proofs {fmt_bytes(ideal.get('proof_bytes'))}, statements {fmt_bytes(ideal.get('statement_bytes'))}; typed object (composed.json, the format's) {fmt_bytes(ideal.get('typed_object_bytes'))}"
            L.append("| " + " | ".join([k[0], f"{k[1]}×{k[2]}", str(CL.value(d, "id.run_id")), fmt_bytes(rb), mode_name, *cells, fmt_bytes(rss), fmt_s(wall) if wall else "—", ideal_cell]) + " |")
    L.append("\nThe archive check recomputes every committed root from the archived leaves (`V3.trees_rehashed`): it reads the whole record once (≈ 1.1× record bytes on n4). "
             "In deployed mode the archive bucket collapses to the pins + `V3.trees_from_leaf_hashes` (9 MB on n4) and the verifier's work is the deployed-sparse bucket: "
             "proofs + statements + the openings of the sampled units — dominated today by `composed.json` (the typed object, 405 MB on n4, read twice), which the "
             "`stored openings` column prices as leaf + path bytes (≈ 162 KB for n4's 468 openings; E9's `deployed_ideal`). The checkpoint bucket is the pinned weights (read once, both modes).\n")
    return L


def table_5(before_after: list[dict[str, Any]]) -> list[str]:
    if not before_after:
        return []
    L = []
    for b in before_after:
        L.append(f"## Table 5 — {b.get('title', 'before / after')} (E2)\n")
        L.append(f"Job: `{b.get('job', '—')}`; rows digest {mark('`' + str(b['rows_digest']['value'])[:16] + '…`', _lab(b['rows_digest']))} — {b['rows_digest'].get('note', '')}\n")
        hdr = ["", "tree", "device", "wall", "gates / s", "CPU-s", "peak RSS", "speed-up"]
        L.append("| " + " | ".join(hdr) + " |")
        L.append("|" + "|".join(["---"] * len(hdr)) + "|")
        bw = b["before"]["wall_s"]["value"]
        for name in ("before", "after"):
            x = b[name]
            su = "—" if name == "before" or not bw or not x["wall_s"]["value"] else f"{bw / x['wall_s']['value']:.1f}× wall, {x['gates_per_s']['value'] / b['before']['gates_per_s']['value']:.1f}× gates/s, {b['before']['cpu_s']['value'] / x['cpu_s']['value']:.0f}× less CPU"
            L.append("| " + " | ".join([name, f"`{x.get('tree', '—')}`", x.get("device", "—"), fmt_s(x["wall_s"]["value"], _lab(x["wall_s"])), fmt_int(x["gates_per_s"]["value"], _lab(x["gates_per_s"])),
                                        fmt_int(x["cpu_s"]["value"], _lab(x["cpu_s"])), fmt_bytes(x["peak_rss_bytes"]["value"], _lab(x["peak_rss_bytes"])), su]) + " |")
        bc = b.get("bottleneck_class")
        if bc:
            L.append(f"\nBottleneck class: **{bc['value']}** — {bc.get('note', '')} (source: `{bc.get('source', '')}`).\n")
        mn = b.get("modeled_next") or {}
        if mn:
            L.append("MODELED with the AFTER producer (E2): " + "; ".join(f"{k.replace('__', ' ').replace('x', '×')} ≈ {fmt_s(v['value'], _lab(v))}" for k, v in mn.items()) + ".\n")
    return L


def table_md(docs: list[dict[str, Any]], night: dict[tuple[str, int, int], dict[str, Any]], title_note: str = "",
             before_after: list[dict[str, Any]] | None = None) -> str:
    before_after = before_after or []
    L: list[str] = []
    L.append("# Results table — measured protocol runs with their guarantees (E3 generator `e2e.scale.table`; costlog/v1 rows in `out/scale/table/`)\n")
    L.append("Every number is MEASURED unless marked: *~italic* = MODELED (the night table's model of the same size, shown beside the measured value as "
             "`measured / ~modeled`), `?` = PROVISIONAL, `(not run)` / `(unknown)` = no value (never borrowed). A runtime is never shown without the "
             "guarantee beside it: s, q, δ, N_V, |Q|, the units checked and the bound U in R1's statement (`guarantee.bound_statement` of each row's JSON). "
             "Verification is split into the live driver (step 5 on the run pod) and the shipped re-verifier in a fresh environment; a live REFUSED is "
             "never rewritten by a later re-verification (the row carries both). Bytes read by verification: ARCHIVE check (recomputes roots from archived "
             "leaves) vs the DEPLOYED verifier's authenticated openings of the sampled units (E9's instrument; `transfer.*` of each row).\n")
    if title_note:
        L.append(title_note + "\n")
    L.extend(table_p(docs, night, before_after))
    L.append("## Table 1 — cost per stage, memory, storage, guarantee, verdicts (one row per run; refused runs kept as history)\n")
    hdr = ["model", "prompt×output (tokens)", "run", "program revision · attention", "stage reached", "serve (recorded pass)", "reconstruction", "commitment", "obligations", "proving",
           "verify (live)", "re-verify (shipped)", "peak RSS (commit / re-verify)", "GPU peak (prove)", "stored: record / archive tgz / slim",
           "verifier reads: archive check / deployed sparse (default entry point; then `--deployed-openings`)", "N_V", "sampling law", "\\|Q\\|", "units checked", "bound U; β = j*/N_V", "verdicts", "device"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for doc in sorted(docs, key=sort_key):
        k = key_of(doc)
        nrow = night.get(k)
        st = doc["stages"]
        rss = f"{fmt_bytes(CL.value(doc, 'stages.commitment.peak_rss_bytes'), CL.label(doc, 'stages.commitment.peak_rss_bytes'))} / {fmt_bytes(CL.value(doc, 'stages.reverify.peak_rss_bytes'), CL.label(doc, 'stages.reverify.peak_rss_bytes'))}"
        gpu = fmt_bytes(CL.value(doc, "stages.proving.gpu_peak_bytes"), CL.label(doc, "stages.proving.gpu_peak_bytes"))
        mv, ml = night_modeled(nrow, "prove_peak_mem_gb")
        if mv is not None:
            gpu += f" / {fmt_bytes(mv, ml)}"
        stor = " / ".join(fmt_bytes(CL.value(doc, f"storage.{k_}"), CL.label(doc, f"storage.{k_}")) for k_ in ("record_bytes", "archive_tgz_bytes", "slim_bytes"))
        tr = transfer_cell(doc)
        dev = str(CL.value(doc, "stages.proving.device") or CL.value(doc, "stages.serve.device") or "—")
        rdev = CL.value(doc, "stages.reconstruction.device")
        if rdev:
            dev += f"; rows: {str(rdev)[:40]}"
        row = [str(CL.value(doc, "id.model_short")), f"{k[1]}×{k[2]} ({CL.value(doc, 'id.tokens')})", str(CL.value(doc, "id.run_id")), program_cell(doc),
               str(CL.value(doc, "status.stage_reached") or "—") + census_suffix(doc),
               stage_cell(doc, "serve", nrow, "serve_s"), stage_cell(doc, "reconstruction", nrow, "reconstruction_s"),
               stage_cell(doc, "commitment", nrow, "commit_s"), stage_cell(doc, "obligations", nrow, None), stage_cell(doc, "proving", nrow, "prove_s"),
               stage_cell(doc, "verify_live", nrow, "verify_live_s"), stage_cell(doc, "reverify", nrow, "reverify_s"), rss, gpu, stor, tr,
               *guarantee_cells(doc), verdict_cell(doc), dev]
        L.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
    L.append("")
    # Table 2: cost split
    L.append("## Table 2 — cost split per stage (% of the summed measured wall; the bottleneck reads off the row)\n")
    hdr2 = ["model", "size", "run", "Σ wall"] + [STAGE_SHORT[s] for s in CL.STAGES]
    L.append("| " + " | ".join(hdr2) + " |")
    L.append("|" + "|".join(["---"] * len(hdr2)) + "|")
    for doc in sorted(docs, key=sort_key):
        k = key_of(doc)
        walls = {s: CL.value(doc, f"stages.{s}.wall_s") for s in CL.STAGES}
        tot = sum(v for v in walls.values() if v is not None)
        cells = []
        for s in CL.STAGES:
            v = walls[s]
            cells.append("—" if v is None else f"{100 * v / tot:.1f} %" + (" ←" if tot and v == max(x for x in walls.values() if x is not None) else ""))
        L.append("| " + " | ".join([str(CL.value(doc, "id.model_short")), f"{k[1]}×{k[2]}", str(CL.value(doc, "id.run_id")), fmt_s(tot) if tot else "—"] + cells) + " |")
    L.append("\n`Σ wall` sums the measured stage walls of the row (stages without a measurement are absent from the sum, and the row says so above); "
             "reconstruction runs on a CPU pod, the rest on the run pod — the split is of wall time, not of one machine's time.\n")
    # Table 2b: peak RSS per stage (the integrator's 03:52Z ask for the accepted 128x64 row; printed for every row that measured any)
    rss_rows = []
    for doc in sorted(docs, key=sort_key):
        k = key_of(doc)
        r = {s: CL.value(doc, f"stages.{s}.peak_rss_bytes") for s in CL.STAGES}
        if all(v is None for v in r.values()):
            continue
        cells = [fmt_bytes(r[s], CL.label(doc, f"stages.{s}.peak_rss_bytes")) if r[s] is not None else "—" for s in CL.STAGES]
        gpu = [CL.value(doc, f"stages.{s}.gpu_peak_bytes") for s in ("serve", "commitment")]
        rss_rows.append("| " + " | ".join([str(CL.value(doc, "id.model_short")), f"{k[1]}×{k[2]}", str(CL.value(doc, "id.run_id"))] + cells +
                                          [" / ".join(fmt_bytes(v) if v is not None else "—" for v in gpu)]) + " |")
    if rss_rows:
        L.append("## Table 2b — peak RSS per stage (the runner's process tree; GPU = nvidia-smi memory.used peak at serve / commitment)\n")
        hdr2b = ["model", "size", "run"] + [STAGE_SHORT[s] for s in CL.STAGES] + ["GPU serve / commit"]
        L.append("| " + " | ".join(hdr2b) + " |")
        L.append("|" + "|".join(["---"] * len(hdr2b)) + "|")
        L.extend(rss_rows)
        L.append("\nReconstruction's RSS is the CPU producer's (pool of workers, whole tree); the verify columns are the shipped verifier's own process "
                 "(the archive check re-hashes every leaf — its RSS is the record's trees, not the sampled units').\n")
    # Table 3: MODELED rows without a measured run
    measured_keys = {key_of(d) for d in docs}
    only_model = [(k, r) for k, r in sorted(night.items(), key=lambda kr: (kr[0][0], kr[0][1] + kr[0][2])) if k not in measured_keys]
    if only_model:
        L.append("## Table 3 — MODELED sizes without a measured run (night table `out/night/table/`; the comparison column on its own)\n")
        hdr3 = ["model", "prompt×output (tokens)", "N_V", "E|Q| = s·N_V", "statements", "serve", "reconstruction", "commitment", "proving", "GPU peak", "U (bits)", "labels"]
        L.append("| " + " | ".join(hdr3) + " |")
        L.append("|" + "|".join(["---"] * len(hdr3)) + "|")
        for k, r in only_model:
            def nf(key):
                v, lab = night_field(r, key)
                return v, lab
            cells = [k[0], f"{k[1]}×{k[2]} ({k[1] + k[2]})"]
            for key, f in (("N_V", fmt_int), ("Q", fmt_int), ("statements", fmt_int), ("serve_s", fmt_s), ("reconstruction_s", fmt_s), ("commit_s", fmt_s), ("prove_s", fmt_s)):
                v, lab = nf(key)
                cells.append(f(v, lab))
            v, lab = nf("prove_peak_mem_gb")
            cells.append(fmt_bytes(None if v is None else v * 1e9, lab))
            U = (r.get("certified_U_at_s") or {}) if isinstance(r.get("certified_U_at_s"), dict) else {}
            cells.append(f"{U.get('U_bits')} (= W)" if U.get("U_equals_W") else str(U.get("U_bits", "—")))
            labs = sorted({_lab(r[key]) for key in ("N_V", "serve_s", "commit_s", "prove_s") if isinstance(r.get(key), dict)})
            cells.append(", ".join(labs))
            L.append("| " + " | ".join(str(c) for c in cells) + " |")
        L.append("")
    L.extend(table_4(docs))
    L.extend(table_5(before_after))
    L.append("## Bound statements (R1's words, this row's numbers)\n")
    for doc in sorted(docs, key=sort_key):
        bs = CL.value(doc, "guarantee.bound_statement")
        k = key_of(doc)
        L.append(f"* **{k[0]} {k[1]}×{k[2]} {CL.value(doc, 'id.run_id')}** — " + (bs if bs else "U not supplied for this point (E6) — no bound stated.") + "\n")
    L.append("## Sources\n")
    for doc in sorted(docs, key=sort_key):
        k = key_of(doc)
        srcs = doc.get("sources") or {}
        rp, rsha = CL.value(doc, "status.record_path"), CL.value(doc, "status.record_sha256")
        L.append(f"* {k[0]} {k[1]}×{k[2]} {CL.value(doc, 'id.run_id')}: `{Path(doc.get('_path', '')).name}`; record `{rp or '—'}` sha256 `{(rsha or '—')[:16]}…`; tree `{CL.value(doc, 'status.tree') or '—'}`; "
                 + "; ".join(f"{a} = `{b}`" for a, b in srcs.items()))
    return "\n".join(L) + "\n"


def write_csv(docs: list[dict[str, Any]], night: dict[tuple[str, int, int], dict[str, Any]], path: Path, before_after: list[dict[str, Any]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "prompt_len", "output_len", "tokens", "run_id", "stage", "wall_s", "label", "N_V", "Q", "verdict_live", "source"])
        for doc in sorted(docs, key=sort_key):
            k = key_of(doc)
            for s in CL.STAGES:
                v = CL.value(doc, f"stages.{s}.wall_s")
                if v is None:
                    continue
                w.writerow([k[0], k[1], k[2], CL.value(doc, "id.tokens"), CL.value(doc, "id.run_id"), s, v, CL.label(doc, f"stages.{s}.wall_s"),
                            CL.value(doc, "guarantee.N_V"), CL.value(doc, "guarantee.Q"), CL.value(doc, "status.verdict_live"), doc.get("_path", "")])
        for b in before_after or []:
            aw = (b.get("after") or {}).get("wall_s", {})
            if aw.get("value") is not None:
                w.writerow([b.get("model", "").split("/")[-1], b["prompt_len"], b["output_len"], b.get("tokens", b["prompt_len"] + b["output_len"]), "after:" + str(b["after"].get("tree", "")).split(" ")[0],
                            "reconstruction_after", aw["value"], _lab(aw), "", "", "", b.get("_path", "")])
            for kk, v in (b.get("modeled_next") or {}).items():
                m = re.match(r"(.+?)__(\d+)x(\d+)$", kk)
                if m and v.get("value") is not None:
                    w.writerow([m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(2)) + int(m.group(3)), "after-model", "reconstruction_after", v["value"], _lab(v), "", "", "", b.get("_path", "")])
        for k, r in sorted(night.items()):
            for nk, (s, _f) in NIGHT_MAP.items():
                if not nk.endswith("_s"):
                    continue
                v, lab = night_modeled(r, nk)
                if v is None:
                    continue
                NV, _ = night_field(r, "N_V")
                Q, _ = night_field(r, "Q")
                w.writerow([k[0], k[1], k[2], k[1] + k[2], "MODELED", s, v, lab, NV, Q, "", f"out/night/table/{k[0]}__{k[1]}x{k[2]}.json"])


def write_fig(csv_path: Path, fig_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    rows = list(csv.DictReader(open(csv_path)))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"serve": "C0", "reconstruction": "C1", "commitment": "C2", "obligations": "C6", "proving": "C3", "verify_live": "C4", "reverify": "C5"}
    models = sorted({r["model"] for r in rows})
    markers = {m: mk for m, mk in zip(models, ["o", "s", "^", "D", "v"])}
    for s in CL.STAGES:
        for m in models:
            meas = sorted([(int(r["tokens"]), float(r["wall_s"])) for r in rows if r["stage"] == s and r["model"] == m and r["label"].startswith("MEASURED")])
            mod = sorted([(int(r["tokens"]), float(r["wall_s"])) for r in rows if r["stage"] == s and r["model"] == m and r["label"].startswith("MODELED")])
            if meas:
                ax.plot([t for t, _ in meas], [w for _, w in meas], marker=markers[m], color=colors[s], linestyle="-", label=f"{STAGE_SHORT[s]} ({m}, measured)")
            if mod:
                ax.plot([t for t, _ in mod], [w for _, w in mod], marker=markers[m], color=colors[s], linestyle="--", alpha=0.6, markerfacecolor="none", label=f"{STAGE_SHORT[s]} ({m}, modeled)")
    after = sorted([(int(r["tokens"]), float(r["wall_s"]), r["label"]) for r in rows if r["stage"] == "reconstruction_after"])
    if after:
        meas = [(t, w) for t, w, l in after if l.startswith("MEASURED")]
        mod = [(t, w) for t, w, l in after if l.startswith("MODELED")]
        if meas:
            ax.plot([t for t, _ in meas], [w for _, w in meas], marker="*", markersize=11, color="C1", linestyle="none", label="reconstruct, E2 vecwalk 2129f075 (measured, same rows)")
        if mod:
            ax.plot([t for t, _ in mod], [w for _, w in mod], marker="*", markersize=9, color="C1", linestyle=":", alpha=0.7, markerfacecolor="none", label="reconstruct, E2 vecwalk (modeled)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tokens per request (prompt + output)")
    ax.set_ylabel("wall seconds per stage")
    ax.set_title("Protocol stage cost vs workload size (solid: MEASURED runs; dashed: MODELED)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e2e.scale.table", description=__doc__.split("\n\n")[0])
    ap.add_argument("costlog_dir")
    ap.add_argument("night_dir", nargs="?", default=None)
    ap.add_argument("-o", "--out", default="out/scale/TABLE.md")
    ap.add_argument("--csv", default=None, help="default: <costlog_dir>/rows.csv")
    ap.add_argument("--fig", default=None, help="default: <out dir>/fig/scaling.png; needs matplotlib (skipped otherwise)")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    docs = load_costlogs(Path(a.costlog_dir))
    before_after = load_before_after(Path(a.costlog_dir))
    night = load_night(Path(a.night_dir)) if a.night_dir else {}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table_md(docs, night, a.note, before_after))
    csv_path = Path(a.csv) if a.csv else Path(a.costlog_dir) / "rows.csv"
    write_csv(docs, night, csv_path, before_after)
    fig_path = Path(a.fig) if a.fig else out.parent / "fig" / "scaling.png"
    drew = write_fig(csv_path, fig_path)
    print(f"{out}: {len(docs)} measured row(s), {len(night)} night row(s); csv {csv_path}; figure {'written ' + str(fig_path) if drew else 'SKIPPED (no matplotlib)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
