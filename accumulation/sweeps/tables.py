"""SPEC §6 deliverable tables at the chosen operating points: the **main circuit table** and the **derived security
table** (artifact B), from ``scale.jsonl`` (+ any other sweep file with rows at those points).

Operating points are read from ``frontier_8b.json`` (``operating_points``) unless given explicitly.  Every number
carries a status: ``measured`` (checked plan / certificate at this exact ``(F, G, X)``), ``L@G=None`` (lower bound
computed without the per-RU work cap), ``pending`` (row missing or errored).
"""

from __future__ import annotations

import json
from pathlib import Path

from accumulation.sweeps.frontier import MB, _g, circuit_view, f_label, g_label, pair_points
from accumulation.sweeps.run import finalize_row, is_training, row_key
from accumulation.sweeps.util import RESULTS_DIR, dedupe, read_jsonl, write_json

MODEL_ORDER = ("llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b", "mixtral-8x7b", "deepseek-v3", "qwen3-235b-a22b")
CIRCUIT_ORDER = ("inference-dense", "inference-decode", "inference-moe", "pretrain-dense", "pretrain-moe")
PAIRS = (("inference-dense", "pretrain-dense"), ("inference-moe", "pretrain-moe"))


def _model_rank(m: str) -> tuple:
    return (MODEL_ORDER.index(m), m) if m in MODEL_ORDER else (len(MODEL_ORDER), m)


def load_rows(out_dir: Path, files=("scale.jsonl", "frontier_8b.jsonl", "qscaling.jsonl")) -> list[dict]:
    rows = []
    for f in files:
        rows += read_jsonl(Path(out_dir) / f)
    return dedupe([finalize_row(r) for r in rows], row_key)


def load_operating_points(out_dir: Path) -> dict[str, dict]:
    p = Path(out_dir) / "frontier_8b.json"
    if not p.exists():
        return {}
    ops = json.loads(p.read_text()).get("operating_points", {})
    return {k: v for k, v in ops.items() if v}


def _row_status(r: dict | None) -> str:
    if r is None or r.get("error") or (r.get("L") is None and r.get("U") is None):
        return "pending"
    bits = []
    if r.get("L") is None:
        bits.append("L pending" + (f" ({r['lower_error'][:40]})" if r.get("lower_error") else ""))
    elif (r.get("L_at_G") or "G=None") == "G=None" and r.get("G") is not None:
        bits.append("L@G=None")
    if r.get("U") is None:
        bits.append("U pending" + (f" ({r['upper_error'][:40]})" if r.get("upper_error") else ""))
    return "; ".join(bits) or "measured"


def main_table(rows: list[dict], F: int, G, X: int, tokens: int = 4096) -> list[dict]:
    sel = [r for r in rows if int(r["F"]) == int(F) and (r.get("G") is None) == (G is None) and (G is None or int(r["G"]) == int(G))
           and int(r["X"]) == int(X) and int(r["tokens"]) == tokens and r["alg"] in CIRCUIT_ORDER]
    by = {(r["model"], r["alg"]): r for r in sel}
    out = []
    models = sorted({r["model"] for r in sel}, key=_model_rank)
    for m in models:
        for alg in CIRCUIT_ORDER:
            r = by.get((m, alg))
            if r is None:
                continue
            v = circuit_view(r)
            v.update(status=_row_status(r), ADW_total=r.get("ADW_total"), ADW_matmul=r.get("ADW_matmul"), all_macs=r.get("all_macs"),
                     tokens=r["tokens"], L_source=r.get("L_source"), L_cap=r.get("L_cap"), peak_rss_mb=r.get("peak_rss_mb"),
                     lower_s=r.get("lower_s"), upper_s=r.get("upper_s"), extract_s=r.get("extract_s"))
            out.append(v)
    return out


def security_rows(rows: list[dict], F: int, G, X: int, tokens: int = 4096) -> list[dict]:
    out = []
    for inf_alg, tr_alg in PAIRS:
        for p in pair_points(rows, inf_alg, tr_alg):
            if int(p["F"]) != int(F) or (p["G"] is None) != (G is None) or (G is not None and int(p["G"]) != int(G)) \
               or int(p["X"]) != int(X) or p["Q"] != tokens:
                continue
            inf, tr = p.get("inference") or {}, p.get("training") or {}
            out.append({"model": p["model"], "inference": inf_alg, "training": tr_alg, "Q": p["Q"], "F": F, "G": G, "X": X,
                        "U_inf_MB_per_token": inf.get("U_MB_per_token"), "L_inf_MB_per_token": inf.get("L_MB_per_token"),
                        "L_train_MB_per_token": tr.get("L_MB_per_token"), "U_train_MB_per_token": tr.get("U_MB_per_token"),
                        "M_inf_per_token": inf.get("M_per_token"), "M_train_per_token": tr.get("M_per_token"),
                        "rho_cert": p["rho_cert"], "penalty": p["penalty"], "rho_ach": p["rho_ach"],
                        "status": p["status_cert"] if p["rho_cert"] is not None else "pending",
                        "status_ach": p["status_ach"]})
    out.sort(key=lambda r: _model_rank(r["model"]))
    return out


def main_table_md(tab: list[dict]) -> str:
    hdr = ["model", "circuit", "M/token", "L MB/tok", "U MB/tok", "U/L", "kappa_low=M/U", "kappa_high=M/L", "n_RU/tok",
           "RU work mean", "RU work max", "RU input mean (B)", "U attn MB/tok", "U matmul MB/tok", "ADW MACs/tok", "P (B)",
           "L_source (B)", "recompute", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for v in tab:
        bc = v["U_by_class_per_token"]
        Q = v["Q"] or 1
        lines.append("| " + " | ".join([
            v["model"], v["alg"], _g(v["M_per_token"]), _g(v["L_MB_per_token"]), _g(v["U_MB_per_token"]), _g(v["U_over_L"]),
            _g(v["kappa_low"]), _g(v["kappa_high"]), _g(v["n_RU_per_token"]), _g(v["ru_work_mean"]), _g(v["ru_work_max"]),
            _g(v["ru_input_mean"]), "-" if bc["attention"] is None else f"{bc['attention'] / MB:.3g}",
            "-" if bc["matmul"] is None else f"{bc['matmul'] / MB:.3g}",
            _g((v["ADW_matmul_macs"] or 0) / Q) if v.get("ADW_matmul_macs") is not None else "-",
            _g(v["P"]), _g(v["L_source"]), "-" if v.get("U_recompute") is None else str(v["U_recompute"]), v["status"]]) + " |")
    return "\n".join(lines)


def security_md(tab: list[dict]) -> str:
    hdr = ["model", "pair", "U_inf MB/tok", "L_inf MB/tok", "L_train MB/tok", "U_train MB/tok", "M_inf/tok", "M_train/tok",
           "rho_cert", "penalty 1/rho_cert", "rho_ach", "status"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in tab:
        lines.append("| " + " | ".join([
            r["model"], f"{r['inference']} / {r['training']}", _g(r["U_inf_MB_per_token"]), _g(r["L_inf_MB_per_token"]),
            _g(r["L_train_MB_per_token"]), _g(r["U_train_MB_per_token"]), _g(r["M_inf_per_token"]), _g(r["M_train_per_token"]),
            _g(r["rho_cert"]), _g(r["penalty"]), _g(r["rho_ach"]), r["status"]]) + " |")
    return "\n".join(lines)


def build_tables(out_dir: Path = RESULTS_DIR, ops: dict[str, dict] | None = None, X: int = 5 * MB, tokens: int = 4096,
                 log=print) -> dict:
    out_dir = Path(out_dir)
    rows = load_rows(out_dir)
    ops = ops or load_operating_points(out_dir)
    if not ops:
        log("[tables] no operating points (frontier_8b.json missing): nothing to do")
        return {}
    main_res, sec_res = {"X": X, "tokens": tokens, "points": {}}, {"X": X, "tokens": tokens, "points": {}}
    md_main = ["# Main circuit table (SPEC §6)", "",
               f"X = {X / MB:.0f} MB, Q = seq = {tokens}; one block per operating point (from `frontier_8b.json`). `M` = target "
               "MACs (`credited_macs` for training, `all_macs` for inference); `kappa_low = M/U`, `kappa_high = M/L`; RU statistics "
               "from the checked plan (`plan.ru_stats()`); `U attn` / `U matmul` = the plan's input by op class. Status: `measured`, "
               "`L@G=None` (lower bound without the work cap), `pending`. Full-parameter circuits only (no LoRA in this pass).", ""]
    md_sec = ["# Derived security table (SPEC §6)", "",
              f"Matched inference / training pairs at the same (model, F, G, X = {X / MB:.0f} MB, Q = {tokens}). "
              "`rho_cert = (M_tr/L_tr)/(M_inf/U_inf)` (certified), penalty `1/rho_cert`, `rho_ach = (M_tr/U_tr)/(M_inf/U_inf)`.", ""]
    for name, op in ops.items():
        F, G = int(op["F"]), op.get("G")
        tab = main_table(rows, F, G, X, tokens)
        sec = security_rows(rows, F, G, X, tokens)
        main_res["points"][name] = {"F": F, "G": G, "rows": tab}
        sec_res["points"][name] = {"F": F, "G": G, "rows": sec}
        head = f"## {name}: F = {f_label(F)}, G = {g_label(G)}"
        md_main += [head, "", main_table_md(tab) if tab else "_pending: no rows yet_", ""]
        md_sec += [head, "", security_md(sec) if sec else "_pending: no matched pairs yet_", ""]
        log(f"[tables] {name} (F={f_label(F)}, G={g_label(G)}): {len(tab)} circuit rows, {len(sec)} pairs")
    write_json(out_dir / "main_table.json", main_res)
    write_json(out_dir / "security_table.json", sec_res)
    (out_dir / "main_table.md").write_text("\n".join(md_main))
    (out_dir / "security_table.md").write_text("\n".join(md_sec))
    return {"main": main_res, "security": sec_res}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="SPEC §6 main / security tables at the operating points")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--tokens", type=int, default=4096)
    args = ap.parse_args(argv)
    build_tables(args.out_dir, tokens=args.tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
