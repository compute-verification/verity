"""ROLLOUT_PLAN.md tables P0-P4 + acceptance regressions -> ``results/ROLLOUT_TABLES.md`` and ``plots/figG_amortization.png``.

Driven by ``sweeps/final_report.py`` (``python -m accumulation.sweeps.final_report --rollout`` builds only this file);
never hand-edited.  Rows come sanitized from ``coarse_report.load_rows`` (forward-only circuits are convex by
construction, so every checked U stands; L is withheld only when its stamp predates the root-floor fix and is shown greyed
as ``stale(hash)`` when it predates the current lower module).  Units: exposure in MB/token, state in GB, slowdown =
(bytes/token) / 4 B as ``1.4e4x``.  Analytic numbers appear only in P2 columns 1-3.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

from accumulation.sweeps.analysis import fit_linear
from accumulation.sweeps.analytic import rollout_tiers
from accumulation.sweeps.coarse_report import _find, _index
from accumulation.sweeps.coarse_run import DEFAULT_POLICY, module_versions
from accumulation.sweeps.synthetic import model_cfg

MB = 1_000_000
GB = 1_000_000_000
B_INF = 4.0
Q, FH, GH = DEFAULT_POLICY["Q_inf"], DEFAULT_POLICY["F_hat"], float(DEFAULT_POLICY["G_hat"])
ROLL_Q = (8192, 32768, 131072, 1048576, 4194304)
STEADY_MIN_Q = 131072
DENSE = ("llama3-1b", "llama3-8b", "llama3-70b", "llama3-405b")
MOE = ("mixtral-8x7b", "qwen3-235b-a22b", "deepseek-v3")
SHORT = {"llama3-1b": "1B", "llama3-8b": "8B", "llama3-70b": "70B", "llama3-405b": "405B", "mixtral-8x7b": "Mixtral 8x7B",
         "qwen3-235b-a22b": "Qwen3-235B-A22B", "deepseek-v3": "DeepSeek-V3"}
P4_SETTINGS = [("Q_inf = 2K", 2048, FH, GH), (f"Q_inf = 8K, F_hat = {FH:g} (default)", Q, FH, GH), ("Q_inf = 32K", 32768, FH, GH),
               ("Q_inf = 128K", 131072, FH, GH), ("F_hat = 3", Q, 3.0, GH), ("F_hat = 10", Q, 10.0, GH), ("G_hat = 10", Q, FH, 10.0),
               ("G_hat = 1000", Q, FH, 1000.0)]
ACCEPT = {"8B_1M": {"L": 0.0559, "U": 0.212, "UL": 3.8, "cert": 1.4e4, "att": 5.3e4},
          "8B_UQ": {8192: 1.96, 32768: 0.523, 131072: 0.221, 1048576: 0.212},
          "att_1M": {"llama3-1b": 0.065, "llama3-70b": 0.987, "llama3-405b": 3.64}}
PREV_PLANNER = "48ee653bf3"           # convex family, Up = whole-chunk work (superseded)
NEW_PLANNER = "f4baa2d7be"            # frozen: per-sequence Up + whole-token layer chunks + MoE block family (Agent B)
NEW_PLANNERS = {"9d3fdcc962", NEW_PLANNER}   # intermediate + frozen per-sequence-Up planners
FROZEN_LOWER = "ac6c3bb0e3"
NOT_RERUN_4M = {"llama3-405b", *("mixtral-8x7b", "qwen3-235b-a22b", "deepseek-v3")}   # 4M cells deliberately not redone tonight (preset rollout-4m)
PLANNER_IMPROVEMENT = "planner improvement: per-sequence Up + whole-token layer chunks + MoE block family"
PREV_SNAPSHOT = "rollout_prev_planner.json"


# ------------------------------------------------------------------------------------------------ helpers

def rollout_alg(model: str) -> str:
    return "forward-nonfixed-moe" if model_cfg(model).experts != 1 else "forward-nonfixed"


def inference_alg(model: str) -> str:
    return "inference-moe" if model_cfg(model).experts != 1 else "inference-dense"


def q_label(q: int) -> str:
    return f"{q // 1048576}M" if q >= 1048576 else f"{q // 1024}K"


def mbt(x, nd: int = 3) -> str:
    return "[ ]" if x is None else f"{x / MB:.{nd}g}"


def slow(x, stale: bool = False) -> str:
    if x is None:
        return "[ ]"
    v = f"{x / B_INF:.1e}x"
    return f"<span style=\"color:gray\">*{v}* stale</span>" if stale else v


def gb(x) -> str:
    return "[ ]" if x is None else f"{x / GB:.3g}"


def _detail(r: dict | None) -> dict:
    d = (r or {}).get("U_detail")
    if isinstance(d, str):
        try:
            d = ast.literal_eval(d)
        except (ValueError, SyntaxError):
            d = {}
    return d or {}


FALLBACK_FAMILIES = {"bands", "fine"}


def anatomy(r: dict | None) -> dict | None:
    a = _detail(r).get("anatomy")
    return a if isinstance(a, dict) else None


def u_provisional(r: dict | None) -> str | None:
    """Reason a rollout U is *provisional* (legal partition, not an attack): the planner fell back to a generic family
    (`bands` / `fine`).  `single` (one RU) is admissible while Q_roll <= Q_inf, or when the checked plan's dominant strategy
    is whole-model (a checked plan has Work <= G by construction: with per-sequence Up only G binds a forward RU).
    Every other family name (block / convex-block / Agent B's MoE block family) is admissible."""
    if not r or r.get("U") is None:
        return None
    fam = _detail(r).get("family")
    if fam in FALLBACK_FAMILIES:
        return f"planner fallback: {fam}"
    if fam == "single" and (r.get("Q_step") or 0) > (r.get("Q_inf") or 0):
        an = anatomy(r)
        if not (an and "whole" in str(an.get("dominant", ""))):
            return "planner fallback: single"
    return None


def new_planner(r: dict | None) -> bool:
    return bool(r) and ((r.get("stage_versions") or {}).get("coarse") in NEW_PLANNERS or anatomy(r) is not None)


def frozen_u(r: dict | None) -> bool:
    return bool(r) and (r.get("stage_versions") or {}).get("coarse") == NEW_PLANNER


def not_rerun(r: dict | None) -> bool:
    return bool(r) and r.get("Q_step") == 4194304 and r["model"] in NOT_RERUN_4M and not frozen_u(r)


def l_markers(r: dict) -> str:
    """`gap x%` (master-LP upper bound - certified)/L when > 0; `recheck: <why>` when the independent separation re-check
    did not pass (L is still the proven dual bound).  Only for rows at the current lower hash: older stamps used other
    semantics for these fields."""
    out = ""
    if r.get("L_stale"):
        return out
    if r.get("L_from_wp"):                                # the binding certificate is the wp floor; the coarse MILP's markers are secondary
        return " coarse MILP below the wp floor" if r.get("L_coarse") is not None else ""
    g = r.get("L_gap")
    if isinstance(g, (int, float)) and g > 0:
        out += f" gap {100 * g:.1f}%"
    if r.get("L_certified") is False:
        why = (r.get("L_detail") or {}).get("recheck") if isinstance(r.get("L_detail"), dict) else None
        out += f" recheck: {why or 'not passed'}"
    return out


def l_floor(r: dict | None) -> float | None:
    """Trivially-sound floor P + tokens: every legal partition imports every dynamic root byte at least once."""
    if not r or not r.get("P"):
        return None
    return float(r["P"]) + float(r.get("token_bytes") or 0)


def l_provisional(r: dict | None) -> bool:
    """L below the trivial floor is sound but provably loose (the LP cap under-charges dynamic roots; MoE today)."""
    if not r or r.get("L") is None:
        return False
    f = l_floor(r)
    return f is not None and r["L"] < f * (1 - 1e-9)


def roll_row(idx: dict, model: str, Q_roll: int, Q_inf: int = Q, F_hat: float = FH, G_hat: float = GH) -> dict | None:
    return _find(idx, model, rollout_alg(model), Q_inf, F_hat, G_hat, Q_roll, 1, min(Q_roll, Q_inf))


def l_cell(r: dict | None, cur: dict, nd: int = 3) -> tuple[str, float | None]:
    """(markdown cell for L MB/tok with hash / stale / gap markers, L bytes per token or None)."""
    if r is None:
        return "pod-pending", None
    if r.get("L") is None:
        if r.get("L_withheld") is not None:
            return f"withheld (pre-fix `{r.get('L_withheld_hash')}`)", None
        return ("[ ] (lower error)" if r.get("lower_error") else "pod-pending"), None
    h = (r.get("stage_versions") or {}).get("lower") or (r.get("versions") or {}).get("lower")
    gap = l_markers(r)
    prov = ""
    if l_provisional(r):
        prov = f" provisional (L < P); floor P+tokens = {mbt(l_floor(r) / r['tokens'], nd)}"
    if r.get("L_stale") or h != cur.get("lower"):
        return f"<span style=\"color:gray\">*{mbt(r['L_per_token'], nd)}* stale(`{h}`){gap}{prov}</span>", r["L_per_token"]
    return f"**{mbt(r['L_per_token'], nd)}**{gap}{prov}", r["L_per_token"]


def cert_cell(r: dict | None, L) -> str:
    """Certified slowdown: greyed when stale; `[ ] provisional` + the floor's slowdown when L < P + tokens."""
    if L is None:
        return "[ ]"
    if l_provisional(r):
        return f"[ ] provisional; floor {slow(l_floor(r) / r['tokens'])}"
    return slow(L, bool(r and r.get("L_stale")))


def u_cell(r: dict | None, nd: int = 3) -> tuple[str, float | None]:
    if r is None:
        return "pod-pending", None
    if r.get("U") is None:
        return ("[ ] (upper error)" if r.get("upper_error") else "pod-pending"), None
    pv = u_provisional(r)
    return (f"{mbt(r['U_per_token'], nd)} provisional ({pv})" if pv else mbt(r["U_per_token"], nd)), r["U_per_token"]


def att_cell(r: dict | None, U) -> str:
    if U is None:
        return "[ ]"
    return "[ ] provisional" if u_provisional(r) else slow(U)


def ul_cell(r: dict | None, L, U) -> str:
    if not (L and U):
        return "[ ]"
    return "[ ] provisional" if (u_provisional(r) or l_provisional(r)) else f"{U / L:.2g}"


def mac_per_token(r: dict | None) -> float | None:
    if not r or not r.get("tokens"):
        return None
    m = r.get("credited_macs") or r.get("all_macs")
    return m / r["tokens"] if m else None


def recurring_share(r: dict | None) -> str:
    if not r or r.get("L") is None:
        return "[ ]"
    if r.get("L_stale"):
        return "[ ] (stale L; share follows the redo)"
    share = None
    if isinstance(r.get("L_recurring"), (int, float)):
        share = r["L_recurring"] / r["L"]
    elif isinstance(r.get("L_root_read"), (int, float)):
        share = (r["L"] - r["L_root_read"] - (r.get("token_bytes") or 0)) / r["L"]
    if share is not None:
        return f"{100 * share:.0f}%" if -1e-9 <= share <= 1 + 1e-9 else f"[ ] (inconsistent metadata: L_recurring/L = {share:.2f})"
    if r.get("P") and r["L"] >= r["P"]:
        return f"[ ] (L_root_read pending; (L - P)/L = {100 * (r['L'] - r['P']) / r['L']:.0f}%)"
    return "[ ] (L_root_read pending" + ("; L < P: certificate charges active dynamic roots only)" if r.get("P") else ")")


def fit_r(points: list[tuple[int, float]]) -> dict | None:
    """I(Q) ~ P + r Q over the steady-state points (Q >= STEADY_MIN_Q); r in bytes/token."""
    pts = [(q, v) for q, v in points if q >= STEADY_MIN_Q and v is not None]
    if len(pts) < 2:
        return None
    f = fit_linear([q for q, _ in pts], [v for _, v in pts])
    return {"r": f["slope"], "I0": f["intercept"], "n": f["n"], "Qs": [q for q, _ in pts], "r2": f["r2"]} if f else None


# ------------------------------------------------------------------------------- previous-planner snapshot (history)

def _rkey(r: dict) -> str:
    return "|".join(str(r.get(k)) for k in ("model", "circuit", "Q_inf", "F_hat", "G_hat", "Q_step", "K", "seq"))


def prev_planner_snapshot(rows: list[dict], out_dir: Path) -> dict:
    """U per rollout row as produced by the previous planner, kept in `results/rollout_prev_planner.json` so the redo's
    lower U can be shown next to what it superseded.  Rows not yet at the new planner are (re)recorded; rows already
    redone keep their stored previous value."""
    import json
    path = Path(out_dir) / PREV_SNAPSHOT
    snap = json.loads(path.read_text()) if path.exists() else {"planner": PREV_PLANNER, "rows": {}}
    for r in rows:
        if not r["circuit"].startswith("forward-nonfixed") or r.get("U") is None or new_planner(r):
            continue
        snap["rows"][_rkey(r)] = {"U_per_token": r["U_per_token"], "family": _detail(r).get("family"), "coarse": (r.get("stage_versions") or {}).get("coarse"),
                                  "provisional": u_provisional(r)}
    path.write_text(json.dumps(snap, indent=1, sort_keys=True))
    return snap["rows"]


def prev_u(snap: dict, r: dict | None) -> dict | None:
    return snap.get(_rkey(r)) if r else None


def prev_cell(snap: dict, r: dict | None) -> str:
    pv = prev_u(snap, r)
    if not pv:
        return "--"
    if not new_planner(r):
        return "(current)"
    return f"{mbt(pv['U_per_token'])}" + (" prov." if pv.get("provisional") else "")


# -------------------------------------------------------------------------------------------------- tables

def p0(idx: dict, cur: dict, snap: dict) -> tuple[str, list[dict]]:
    hdr = ["Family", "Model", "# tokens", "MAC/token", "L MB/tok", "U MB/tok", f"U previous planner (`{PREV_PLANNER}`)", "U/L", "Certified slowdown", "Best-attack slowdown", "Status"]
    rows, recs = [], []
    for fam, models in (("Dense", DENSE), ("MoE", MOE)):
        for m in models:
            r = roll_row(idx, m, 1048576)
            lc, L = l_cell(r, cur)
            uc, U = u_cell(r)
            inf = _find(idx, m, inference_alg(m), Q, FH, GH, None, None)
            inf_ok = bool(inf and inf.get("U") is not None and inf.get("U_n_ru") in (None, 1) and abs(inf["U"] - (inf.get("token_bytes") or 0)) <= 1e-9 * inf["U"])
            st = "measured" if (L is not None and U is not None and not r.get("L_stale")) else ("measured (L stale)" if (L is not None and U is not None) else
                                                                                                  ("measured U" if U is not None else "pod-pending"))
            if u_provisional(r):
                st += f"; **U provisional** ({u_provisional(r)}: legal partition, not an attack)"
            if l_provisional(r):
                st += "; **L provisional** (certificate below the P + tokens floor)"
            st += "; inference 4 B/tok one RU" if inf_ok else ("; **inference reference missing**" if not inf else "; **inference not one RU**")
            if r is not None and r.get("U") is not None and not frozen_u(r):
                st += f"; U at {'intermediate' if new_planner(r) else 'previous'} planner (redo at `{NEW_PLANNER}` pending)"
            rows.append([fam, SHORT[m], "1M", f"{mac_per_token(r):.3g}" if mac_per_token(r) else "[ ]", lc, uc, prev_cell(snap, r), ul_cell(r, L, U), cert_cell(r, L), att_cell(r, U), st])
            recs.append({"model": m, "family": fam, "L_b": L, "U_b": U, "cert": (L / B_INF) if L else None, "att": (U / B_INF) if U else None,
                         "mac_per_token": mac_per_token(r), "L_stale": bool(r and r.get("L_stale")), "status": st, "inference_one_ru": inf_ok,
                         "U_provisional": u_provisional(r), "L_provisional": l_provisional(r), "L_floor_b": (l_floor(r) / r["tokens"]) if l_provisional(r) else None,
                         "new_planner": new_planner(r), "U_prev_b": (prev_u(snap, r) or {}).get("U_per_token")})
    md = ["## P0 -- headline table (Q_roll = 1M, default policy point)", "",
          f"Dynamic rollout = the inference forward graph with the weights dynamic (`forward-nonfixed` / `forward-nonfixed-moe`); useful work per token = inference, "
          f"so slowdown = (bytes/token) / 4 B. L in **bold** = current lower module `{cur.get('lower')}`; greyed *italic* = stale (older post-fix hash, shown for "
          f"continuity, never current); `gap` = MILP dual gap when > 0. U from checked plans (forward-only: convex by construction); the 'previous planner' column keeps "
          f"the U the superseded planner `{PREV_PLANNER}` found (whole-chunk Up) once the row has been redone at `{NEW_PLANNER}` (per-sequence Up). No analytic values.", "",
          _md(hdr, rows)]
    return "\n".join(md), recs


def p1(idx: dict, cur: dict, snap: dict) -> tuple[str, dict]:
    hdr = ["Family/model", "# rollout tokens", "Weight-only floor P/Q MB/tok", "L MB/tok", "U MB/tok", "U prev. planner", "Certified slowdown", "Best-attack slowdown", "Recurring share of L"]
    out, fits, curves = [], {}, {}
    for m in (*DENSE, *MOE):
        Ls, Us, curve = [], [], []
        for q in ROLL_Q:
            r = roll_row(idx, m, q)
            lc, L = l_cell(r, cur)
            uc, U = u_cell(r)
            P = (r or {}).get("P") or 2.0 * model_cfg(m).total_params()
            if not_rerun(r):
                uc += " (older planner; not re-run tonight)"
            out.append([SHORT[m], q_label(q), mbt(P / q), lc, uc, prev_cell(snap, r), cert_cell(r, L), att_cell(r, U), recurring_share(r)])
            if L is not None and not l_provisional(r) and not r.get("L_stale"):       # one certificate per fit: current hash only
                Ls.append((q, r["L"]))
            if U is not None and not u_provisional(r) and frozen_u(r):                 # one planner per fit: frozen hash only
                Us.append((q, r["U"]))
            curve.append({"Q_roll": q, "L_b": L, "U_b": U, "P_over_Q": P / q, "L_stale": bool(r and r.get("L_stale")),
                          "U_provisional": u_provisional(r), "L_provisional": l_provisional(r)})
        fits[m] = {"L": fit_r(Ls), "U": fit_r(Us), "P": (roll_row(idx, m, 1048576) or {}).get("P") or 2.0 * model_cfg(m).total_params()}
        curves[m] = curve
    md = ["## P1 -- amortization: does the floor survive Q -> inf?", "",
          "Weight-only floor = P/Q (every dynamic byte read once). Recurring share of L = (L - L_root_read)/L with L_root_read = dynamic-root bytes the certificate "
          "charges once (pending Agent A's field where marked; (L - P)/L given as the P-based value meanwhile). Provisional cells (planner-fallback U, L below the "
          "P + tokens floor) are shown but excluded from the fits and never turned into slowdowns. The 405B and MoE 4M cells were deliberately not re-run at the "
          f"frozen hashes tonight (too slow; preset `rollout-4m` for later): they are shown stale / older-planner and marked as such.", "", _md(hdr, out), "",
          f"Fits over the steady-state points (Q >= {q_label(STEADY_MIN_Q)}): `L(Q) ~ P + r_L Q`, `U(Q) ~ P + r_U Q`, r in MB/token. One certificate / one planner per "
          f"fit: only L at the current lower hash and U at the frozen planner `{NEW_PLANNER}` enter (stale or previous-planner points would mix bounds of different strength).", ""]
    md.append("Reading the curves: U/Q is flat from 128K on (the planner's steady tiling), while L/Q can *rise* with Q over the same range. That is expected, not an "
              "inconsistency: L = root floor + certified block term, and at 128K-1M the one-time root term P still dominates the per-token figure, so L/Q is "
              "pulled down toward P/Q; as Q grows the recurring block term takes over and the certified recurring rate r_L converges toward the attack's r_U "
              "(the 'recurring share of L' column rising toward 100% is the same effect). r_L < r_U is the remaining certificate gap at steady state.")
    md.append("")
    for m in (*DENSE, *MOE):
        f = fits[m]
        rl, ru = f["L"], f["U"]
        s = f"* **{SHORT[m]}** (P = {gb(f['P'])} GB): "
        s += f"r_L = {rl['r'] / MB:.3g} MB/tok (n={rl['n']}, Q in {[q_label(q) for q in rl['Qs']]}, r2 {rl['r2']:.3f})" if rl else "r_L = [ ] (< 2 admissible current-hash steady-state L points)"
        s += "; " + (f"r_U = {ru['r'] / MB:.3g} MB/tok (n={ru['n']}, Q in {[q_label(q) for q in ru['Qs']]}, r2 {ru['r2']:.3f})" if ru else "r_U = [ ] (< 2 admissible frozen-planner steady-state U points)")
        if rl:
            s += f". Result: the one-time dynamic model cost amortizes away, but certified recurring exposure converges to r_L = {rl['r'] / MB:.3g} MB/token " \
                 f"({rl['r'] / B_INF:.1e}x inference); the best attack found converges to r_U = {ru['r'] / MB:.3g} MB/token." if ru else "."
        md.append(s)
    return "\n".join(md), {"fits": fits, "curves": curves}


def p2(idx: dict, cur: dict, p0_recs: list[dict]) -> str:
    hdr = ["Model", "Weights only", "+ activation boundaries", "+ intra-matmul sharding X = 5 MB", "Actual certified L", "Actual best attack U"]
    out, notes = [], []
    for m in (*DENSE, *MOE):
        t = rollout_tiers(m, Q_roll=1048576, Q_inf=Q, F_hat=FH, G_hat=GH)
        rec = next(x for x in p0_recs if x["model"] == m)
        out.append([SHORT[m], f"{t['slow1']:.1e}x *analytic*", f"{t['slow2']:.1e}x *analytic* (m*={t['m_star']:.2g} layers, t*={t['t_star']:.3g} tok)",
                    f"{t['slow3']:.1e}x *analytic/OOM*", ("[ ] provisional" if rec["L_provisional"] else slow(rec["L_b"], rec["L_stale"])),
                    ("[ ] provisional" if rec["U_provisional"] else slow(rec["U_b"]))])
        if rec["U_b"] and not rec["U_provisional"]:
            ratio = rec["U_b"] / t["b2"]
            notes.append(f"* {SHORT[m]}: measured U / tier-2 = {ratio:.2f} (tier-2 {t['b2'] / MB:.3g} MB/tok, {t['plan2']} plan, vs U {rec['U_b'] / MB:.3g}"
                         f"{'' if rec['new_planner'] else ' at the previous planner -- redo pending'}); in+out activation variant {t['b2_inout'] / MB:.3g} MB/tok would give "
                         f"{rec['U_b'] / t['b2_inout']:.2f}.")
    md = ["## P2 -- closing the adversary's escape routes (slowdowns at Q_roll = 1M)", "",
          "Columns 1-3 from `sweeps/analytic.py::rollout_tiers` (convex family, forward-only; derivation in its docstring): (1) weights only = P/Q; "
          "(2) + activation boundaries = optimal m-layer x t-token RU under the Work cap `m t <= G_hat Q_inf Lyr` token-layers -- with Up evaluated per sequence F never "
          "binds a forward RU at F_hat >= 1 -- importing its layers' dynamic weights and the residual stream entering the chunk, (Lyr/m - 1) boundaries per token "
          "(imports only, per SPEC's definition of runtime input; the plan's 'in and out' phrasing would double count -- that variant is listed in the cross-check); "
          "the whole-model x token-chunk plan is a second candidate; (3) + a per-RU input cap X = 5 MB forcing Loomis-Whitney tiling of every dynamic matrix larger "
          "than X, labelled analytic/OOM always (no whole-rollout composition theorem). Columns 4-5 from rows.", "", _md(hdr, out), "",
          "Cross-check of tier (2) against the measured U (should be within ~2x; no tuning):", *notes]
    return "\n".join(md)


def p3(idx: dict, cur: dict) -> str:
    hdr = ["Model", "# tokens", "# RUs", "median/max work per RU", "layers/blocks per RU", "tokens/RU", "dynamic weights imported/RU (GB)",
           "activation input/RU (MB)", "partial-result input/RU (MB)", "dominant strategy", "source", "worst RU, L side (certificate's cheapest shape)"]
    out = []
    for m in ("llama3-8b", "mixtral-8x7b", "qwen3-235b-a22b", "deepseek-v3"):
        for q in ROLL_Q[:4]:
            r = roll_row(idx, m, q)
            if not r or r.get("U") is None:
                out.append([SHORT[m], q_label(q), "pod-pending", "", "", "", "", "", "", "", "", l_shape(r, r["G_hat"] if r else GH)])
                continue
            d = _detail(r)
            an = d.get("anatomy") if isinstance(d.get("anatomy"), dict) else None
            st = r.get("U_ru_stats") or {}
            bc = r.get("U_by_class") or {}
            n = r.get("U_n_ru") or st.get("n_ru")
            if an:
                def mm(key, fmt):
                    v = an.get(key)
                    if isinstance(v, dict):
                        return f"{fmt(v.get('median'))} / {fmt(v.get('max'))}"
                    return fmt(v) if v is not None else "[ ]"
                G_hat = r.get("G_hat") or GH
                work = f"{an.get('work_median_macs', 0):.2e} / {an.get('work_max_macs', 0):.2e} MACs"
                if an.get("work_median_over_G") is not None:
                    work += f" ({an['work_median_over_G'] * G_hat:.2f} / {an['work_max_over_G'] * G_hat:.2f} fwd(Q_inf))"
                dom = str(an.get("dominant", ""))
                if u_provisional(r):
                    dom += f" -- **provisional** ({u_provisional(r)}; legal partition, not an attack)"
                if "active_experts_per_ru" in an:
                    dom += f"; active experts/RU {mm('active_experts_per_ru', lambda x: f'{x:g}')}, active dyn params/RU {mm('active_dyn_param_bytes_per_ru', gb)} GB, " \
                           f"tokens/active expert {mm('tokens_per_active_expert', lambda x: f'{x:,.0f}')}"
                row = [SHORT[m], q_label(q), str(an.get("n_ru", n)), work, mm("layers_per_ru", lambda x: f"{x:g}"), mm("tokens_per_ru", lambda x: f"{x:,.0f}"),
                       mm("dyn_weight_bytes_per_ru", gb), mm("activation_bytes_per_ru", mbt), mm("partial_bytes_per_ru", mbt), dom, "U_detail.anatomy (median / max)"]
            else:                                                     # derived from the checked plan's statistics (averages)
                fam = d.get("family")
                layers = d.get("m") if fam in ("block", "convex-block") else ("band" if fam == "bands" else "[ ]")
                seq = r.get("seq") or 1
                tok = (d.get("s") or d.get("s_f") or 0) * seq if fam in ("block", "convex-block") else (r["tokens"] if fam == "bands" else None)
                dom = {"block": "layer-block x sequence-chunk tiling", "convex-block": "layer-block x sequence-chunk tiling",
                       "bands": "generic root bands (no transformer blocks recognised)", "single": "whole rollout in one RU"}.get(fam, str(fam))
                if q <= Q and n == 1:
                    dom = "whole model, one RU"
                if u_provisional(r):
                    dom += f" -- **provisional** ({u_provisional(r)}; legal partition, not an attack)"
                row = [SHORT[m], q_label(q), str(n), f"{st.get('work_mean', 0):.2e} (mean) / {st.get('work_max', 0):.2e}", str(layers),
                       f"{tok:,.0f}" if tok else "[ ]", gb((bc.get("weights") or 0) / n) if n else "[ ]", mbt((bc.get("activations") or 0) / n) if n else "[ ]",
                       mbt((bc.get("partials") or bc.get("partial") or 0) / n) if n else "[ ]", dom, "derived from U_ru_stats / U_by_class (anatomy pending Agent B)"]
            row.append(l_shape(r, r.get("G_hat") or GH))
            out.append(row)
    return "\n".join(["## P3 -- what the optimal adversary actually does (anatomy of the checked plan)", "",
                      "From `U_detail.anatomy` (Agent B; per-RU cells are median / max, work also as a multiple of fwd(Q_inf)); rows not yet redone at the new planner "
                      "show per-RU averages derived from the plan's `U_ru_stats` / `U_by_class` and family parameters (m layers x s sequences), marked as such. "
                      "MoE: active experts / active dynamic params / tokens per active expert in the strategy column.", "", _md(hdr, out)])


def l_shape(r: dict | None, G_hat: float) -> str:
    """One short cell: the certificate's cheapest RU shape (`L_detail.cheapest_shape`) -- the L-side 'worst RU'."""
    d = (r or {}).get("L_detail")
    sh = d.get("cheapest_shape") if isinstance(d, dict) else None
    if not sh or not sh.get("input_bytes"):
        return "[ ]"
    cells = sh.get("tokens_per_cell") or {}
    toks = sorted(set(round(v) for v in cells.values()))
    tok_s = f"{toks[0]:,}" if len(toks) == 1 else (f"{toks[0]:,}-{toks[-1]:,}" if toks else "?")
    work = sh.get("work")
    fwd = r.get("fwd_work") if r else None
    fwd = fwd if isinstance(fwd, (int, float)) and fwd > 0 else None
    work_s = f", work {work / fwd:.2f} fwd(Q_inf)" if (work and fwd) else (f", work {work:.2e}" if work else "")
    return (f"{len(cells)} layer-cells x {tok_s} tok, input {sh['input_bytes'] / MB:.3g} MB{work_s}, weights present {gb(sh.get('weights_present'))} GB, "
            f"{sh.get('bytes_per_mac', 0):.2e} B/MAC" + (f"; recheck {d.get('recheck')}" if d.get("recheck") not in (None, "passed") else ""))


def p4(idx: dict, cur: dict) -> str:
    hdr = ["Setting", "L MB/tok", "U MB/tok", "Certified slowdown", "Best-attack slowdown", "Status"]
    out, pts = [], {}
    for label, qi, fh, gh in P4_SETTINGS:
        r = roll_row(idx, "llama3-8b", 1048576, qi, fh, gh)
        lc, L = l_cell(r, cur)
        uc, U = u_cell(r)
        out.append([label, lc, uc, cert_cell(r, L), att_cell(r, U), "measured" if (L is not None and U is not None) else ("measured U" if U is not None else
                                                                                                          ("measured L" if L is not None else "pod-pending (preset rollout-p4)"))])
        pts[label] = (r, L, U)
    return "\n".join(["## P4 -- minimal robustness: 8B rollout at Q_roll = 1M", "", _md(hdr, out), "", *p4_commentary(pts),
                      "", "Desired conclusion: large relaxations weaken the bound but do not erase the multi-OOM wedge."])


def p4_commentary(pts: dict) -> list[str]:
    """What the P4 rows show, computed from the rows.  With Up evaluated per sequence only G binds a forward RU, so the
    minimum number of RUs by work is ceil(Q_roll / (G_hat Q_inf)); when that is 1 the whole rollout is one legal RU and
    I* = P + tokens exactly (the L floor is tight)."""
    def get(label_start):
        for k, v in pts.items():
            if k.startswith(label_start):
                return v
        return (None, None, None)
    r0, L0, U0 = get("Q_inf = 8K")
    redone = {lab: new_planner(r) for lab, (r, L, U) in pts.items() if r}
    out = ["What the data show (per-sequence Up: F caps serial depth, not data-parallel tokens, so only G binds a forward RU; the certificate's G area cap is the "
           "*correct* cap for forward circuits -- F matters for training, where dW reductions couple all sequences):"]
    if not any(redone.values()):
        out.append(f"* **pending redo**: no P4 row is at the new planner `{NEW_PLANNER}` yet; U values below are the previous planner's (whole-chunk Up) and are "
                   "expected to move down where G allows fewer, larger RUs.")
    for lab, (r, L, U) in pts.items():
        if not r:
            continue
        qi, gh = r["Q_inf"], r["G_hat"]
        n_min = max(1, math.ceil(r["Q_step"] / (gh * qi)))
        floor = l_floor(r)
        tight = floor and L is not None and abs(r["L"] - floor) <= 1e-6 * floor
        if n_min == 1:
            out.append(f"* {lab}: work = {r['Q_step'] / qi:.0f} fwd(8K) <= G = {gh * qi / Q:.0f} fwd(8K), so the whole 1M rollout is **one legal RU** and "
                       f"I* = P + tokens = {floor / r['tokens'] / MB:.4g} MB/tok exactly" + (f"; L = {L / MB:.4g} sits on it (**tight**)" if tight else (f"; L = {L / MB:.4g}" if L is not None else ""))
                       + (f"; U = {U / MB:.4g}" + (" (matches -- one RU found)" if U is not None and abs(U - floor / r["tokens"]) <= 1e-3 * U else (" (previous planner; the redo should reach the same value)" if not redone.get(lab) else " (planner did not find the single RU)")) if U is not None else "") + ".")
        elif tight:
            out.append(f"* {lab}: at least {n_min} RUs by work (G = {gh * qi / Q:.0f} fwd(8K)); L = P + tokens = {L / MB:.4g} MB/tok is the root + token floor -- the "
                       f"block term is silent here, so L is loose by at most the {n_min}-RU tiling cost (~{n_min * r['P'] / r['tokens'] / MB:.3g} MB/tok if every RU re-imports P)"
                       + (f"; U = {U / MB:.4g}" + ("" if redone.get(lab) else " (previous planner; redo pending)") if U is not None else "") + ".")
    us = [(lab, U) for lab, (r, L, U) in pts.items() if lab.startswith("G_hat") and U is not None]
    if U0 is not None and us:
        same = all(abs(U - U0) <= 1e-6 * U0 for _, U in us)
        out.append(f"* U at {', '.join(lab for lab, _ in us)}: {', '.join(f'{U / MB:.3g}' for _, U in us)} vs {U0 / MB:.3g} MB/tok at the default -- "
                   + ("unchanged: these are previous-planner values (whole-chunk Up let F bind); the redo should lower them." if same and not all(redone.get(lab) for lab, _ in us)
                      else ("unchanged." if same else "lower with larger G, as the per-sequence-Up planner tiles in fewer, larger RUs.")))
    fs = [(lab, L, U) for lab, (r, L, U) in pts.items() if lab.startswith("F_hat") and (L is not None or U is not None)]
    if fs and U0 is not None:
        out.append("* Relaxing F: " + "; ".join(f"{lab}: U {U / MB:.3g} MB/tok" + (f", L {L / MB:.3g}" if L is not None else "") for lab, L, U in fs) +
                   f" (default U {U0 / MB:.3g}" + (f", L {L0 / MB:.3g}" if L0 is not None else "") + ")"
                   + (" -- with per-sequence Up, F should no longer move U at all; a difference means previous-planner rows." if not all(redone.get(lab) for lab, _, _ in fs)
                      else " -- F does not enter forward tiling (per-sequence Up), so U is expected to be F-independent here.")
                   + (" Note the default-point L was stamped by a different lower module than the F_hat rows, so the L values are not directly comparable."
                      if (r0 and any(((pts[lab][0] or {}).get("stage_versions") or {}).get("lower") != (r0.get("stage_versions") or {}).get("lower") for lab, _, _ in fs)) else ""))
    return out


def acceptance(idx: dict, cur: dict, p0_recs: list[dict]) -> str:
    md = ["## Acceptance regressions (ROLLOUT_PLAN.md)", "", f"Reference values are the previous planner's (`{PREV_PLANNER}`, whole-chunk Up). L may only go up as the "
          f"certificate improves; U may only go down as the planner improves -- a decrease at `{NEW_PLANNER}` is reported as such, with its delta, not as a failure.", ""]
    ok_all = True

    def line(name, passed, detail):
        nonlocal ok_all
        ok_all &= bool(passed)
        md.append(f"* {'PASS' if passed else '**FAIL**'} -- {name}: {detail}")

    for m in (*DENSE, *MOE):
        inf = _find(idx, m, inference_alg(m), Q, FH, GH, None, None)
        if not inf or inf.get("U") is None:
            line(f"inference {SHORT[m]} 4 B/token, one RU", False, "row missing / U pending")
        else:
            b = inf["U"] / inf["tokens"]
            line(f"inference {SHORT[m]} 4 B/token, one RU", abs(b - 4.0) < 1e-6 and inf.get("U_n_ru") in (None, 1), f"{b:.4g} B/tok, n_ru = {inf.get('U_n_ru')}")
    rec = next(x for x in p0_recs if x["model"] == "llama3-8b")
    a = ACCEPT["8B_1M"]
    if rec["L_b"] is not None:
        d = rec["L_b"] / MB - a["L"]
        line("8B rollout 1M: L = 0.0559 MB/tok (previous certificate)", d >= -1e-4,
             f"L = {rec['L_b'] / MB:.4g} MB/tok (previous {a['L']}, delta {d:+.4g}; "
             f"{'L increased (certificate improvement: block term + F/G geometry), sound direction' if d > 1e-4 else ('unchanged' if abs(d) <= 1e-4 else 'REGRESSION: L went down')})"
             + (" [stale hash]" if rec["L_stale"] else ""))
    else:
        line("8B rollout 1M: L = 0.0559 MB/tok (previous certificate)", False, "L missing")
    def u_dir(d, tol):
        if d < -tol:
            return f"U decreased ({PLANNER_IMPROVEMENT}), sound direction"
        return "unchanged" if abs(d) <= tol else "REGRESSION: U went up"
    if rec["U_b"] is not None:
        d = rec["U_b"] / MB - a["U"]
        line("8B rollout 1M: U = 0.212 MB/tok", d <= 1e-3, f"U = {rec['U_b'] / MB:.4g} MB/tok (delta {d:+.4g}; {u_dir(d, 1e-3)})")
        if rec["L_b"]:
            line("8B rollout 1M: U/L = 3.8, certified 1.4e4x, attack 5.3e4x (previous)", True,
                 f"U/L = {rec['U_b'] / rec['L_b']:.2g}, certified {rec['cert']:.1e}x, attack {rec['att']:.1e}x (previous 3.8 / 1.4e4x / 5.3e4x)")
    else:
        line("8B rollout 1M: U = 0.212 MB/tok", False, "U missing")
    parts, ok = [], True
    for q, exp in ACCEPT["8B_UQ"].items():
        r = roll_row(idx, "llama3-8b", q)
        v = r["U_per_token"] / MB if (r and r.get("U") is not None) else None
        parts.append(f"{q_label(q)}: {v:.3g} (prev {exp}, {v - exp:+.3g}{', decreased' if v < exp - 1e-3 else ''})" if v is not None else f"{q_label(q)}: [ ]")
        ok &= v is not None and v <= exp + 1e-3
    dec = any(roll_row(idx, "llama3-8b", q) and (roll_row(idx, "llama3-8b", q).get("U_per_token") or 1e99) / MB < exp - 1e-3 for q, exp in ACCEPT["8B_UQ"].items())
    line("8B U(Q): 1.96 -> 0.523 -> 0.221 -> 0.212 MB/tok (previous planner)", ok, "; ".join(parts) + (f" -- U decreased ({PLANNER_IMPROVEMENT}), sound direction" if dec else ""))
    for m, exp in ACCEPT["att_1M"].items():
        r = roll_row(idx, m, 1048576)
        v = r["U_per_token"] / MB if (r and r.get("U") is not None) else None
        line(f"{SHORT[m]} attack at 1M = {exp} MB/tok (previous planner)", v is not None and v <= exp + 1e-3 * max(1, exp),
             f"{v:.3g} MB/tok (delta {v - exp:+.3g}; {u_dir(v - exp, 1e-3 * max(1, exp))})" if v is not None else "[ ]")
    md += ["", f"**Overall: {'all pass' if ok_all else 'failures above -- investigate before merging'}.**"]
    return "\n".join(md)


def _md(hdr: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


# -------------------------------------------------------------------------------------------------- figure

def fig_g(curves: dict, fits: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate((*DENSE, *MOE)):
        c = curves.get(m) or []
        col = f"C{i}"
        U = [(p["Q_roll"], p["U_b"]) for p in c if p["U_b"] is not None and not p.get("U_provisional")]
        Up = [(p["Q_roll"], p["U_b"]) for p in c if p["U_b"] is not None and p.get("U_provisional")]
        L = [(p["Q_roll"], p["L_b"]) for p in c if p["L_b"] is not None]
        if U:
            ax.plot([q for q, _ in U], [v / MB for _, v in U], "s-", color=col, label=f"{SHORT[m]} U/Q")
        if Up:
            ax.plot([q for q, _ in Up], [v / MB for _, v in Up], "x", color=col, alpha=0.6, label=f"{SHORT[m]} U provisional (planner fallback)")
        if L:
            ax.plot([q for q, _ in L], [v / MB for _, v in L], "v-", mfc="none", color=col, label=f"{SHORT[m]} L/Q (certified)")
        if U or L:
            qs = [p["Q_roll"] for p in c]
            ax.plot(qs, [p["P_over_Q"] / MB for p in c], ":", color=col, alpha=0.7, label=f"{SHORT[m]} P/Q floor")
    ax.axhline(B_INF / MB, color="k", ls="-.", lw=1, label="inference 4 B/tok")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"rollout tokens Q_roll (Q_inf = {Q}, F_hat = {FH:g}, G_hat = {GH:g})")
    ax.set_ylabel("runtime input per generated token (MB/token)")
    ax.set_title("G. Dynamic rollout amortization: certified L and best-attack U vs Q_roll; P/Q weight-only floor dotted", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------- build

def provenance(rows: list[dict], cur: dict) -> str:
    """Which lower hash produced each rollout L, which is current, and how stale ones get re-stamped."""
    from collections import Counter
    c = Counter((r.get("stage_versions") or {}).get("lower") or (r.get("versions") or {}).get("lower") or "unstamped"
                for r in rows if r["circuit"].startswith("forward-nonfixed") and r.get("L") is not None)
    parts = ", ".join(f"`{h}` x{n}{' (current)' if h == cur.get('lower') else ''}" for h, n in c.most_common())
    n_stale = sum(n for h, n in c.items() if h != cur.get("lower"))
    roll = [r for r in rows if r["circuit"].startswith("forward-nonfixed") and r.get("L") is not None]
    n_uncert = sum(1 for r in roll if r.get("L_certified") is False)
    n_gap = sum(1 for r in roll if isinstance(r.get("L_gap"), (int, float)) and r["L_gap"] > 0)
    n_frozen_u = sum(1 for r in rows if r["circuit"].startswith("forward-nonfixed") and r.get("U") is not None and frozen_u(r))
    n_u = sum(1 for r in rows if r["circuit"].startswith("forward-nonfixed") and r.get("U") is not None)
    return ("**What is current vs stale.** Frozen hashes: lower `" + FROZEN_LOWER + "`, coarse `" + NEW_PLANNER + "`, upper `eab50c6955`; the pod is re-running every "
            f"rollout row at them (U rows at the frozen planner so far: {n_frozen_u} / {n_u}). Current lower module (`lower.py` + `lower_coarse.py`) hash: `" + str(cur.get("lower")) + "`. "
            "Every L is the *proven dual bound* of the certificate's master LP / MILP (HiGHS, 180 s per MILP), never an incumbent; `L_gap` = (master-LP upper bound - certified)/L "
            f"is shown as `gap x%` when > 0 ({n_gap} rows), and rows whose independent separation re-check did not pass carry `recheck: <why>` ({n_uncert} rows). Rollout L values in this "
            f"file were produced at: {parts or 'none'}. {n_stale} of {sum(c.values())} L cells are stale and are shown greyed in *italic* with their hash -- they are "
            "sound certificates of the module that produced them, never presented as current. A `--redo stale` pass on the pod re-stamps them once Agent A declares "
            "the hash stable (dense rollout L values are expected to be unchanged since A's root-floor fix). U is forward-only here and independent of the lower hash; "
            f"planner hash `{cur.get('coarse')}`; the pod is redoing every rollout row at `{NEW_PLANNER}` (per-sequence Up) -- rows still at the previous planner "
            "are marked in the status column and their U is expected to move down, never up (forward-only partitions are convex by construction).")


def handoff_box(rows: list[dict], idx: dict, cur: dict, p0_recs: list[dict], stamp: str) -> str:
    """Six factual lines: hashes, current vs stale counts, cells still running, dense P0 result, MoE status."""
    import datetime
    roll = [r for r in rows if r["circuit"].startswith("forward-nonfixed")]
    n_u_cur = sum(1 for r in roll if r.get("U") is not None and frozen_u(r))
    n_l_cur = sum(1 for r in roll if r.get("L") is not None and not r.get("L_stale"))
    n_both = sum(1 for r in roll if r.get("U") is not None and frozen_u(r) and r.get("L") is not None and not r.get("L_stale"))
    # cells expected by the plan (P0/P1 grid + P4) that are not yet at the frozen hashes on both sides
    pending = []
    for m in (*DENSE, *MOE):
        for q in ROLL_Q:
            r = roll_row(idx, m, q)
            if not_rerun(r) or (r is None and q == 4194304 and m in NOT_RERUN_4M):
                continue
            if r is None or r.get("U") is None or not frozen_u(r) or r.get("L") is None or r.get("L_stale"):
                pending.append(f"{SHORT[m]} {q_label(q)}")
    for label, qi, fh, gh in P4_SETTINGS:
        r = roll_row(idx, "llama3-8b", 1048576, qi, fh, gh)
        if r is None or r.get("U") is None or not frozen_u(r) or r.get("L") is None or r.get("L_stale"):
            pending.append(f"P4 8B 1M {label.split(' (')[0]}")
    dense = [x for x in p0_recs if x["family"] == "Dense"]
    dense_cur = [x for x in dense if x["L_b"] is not None and x["U_b"] is not None and not x["L_stale"] and x["new_planner"] and frozen_u(roll_row(idx, x["model"], 1048576))]
    if dense_cur:
        uls = [x["U_b"] / x["L_b"] for x in dense_cur]
        dense_s = (f"Dense P0 (Q_roll = 1M, frozen hashes, {len(dense_cur)}/4 models): " +
                   "; ".join(f"{SHORT[x['model']]} L {x['L_b'] / MB:.3g} / U {x['U_b'] / MB:.3g} MB/tok" for x in dense_cur) +
                   f". U/L range {min(uls):.2f}-{max(uls):.2f}; certified slowdown {min(x['cert'] for x in dense_cur):.1e}x-{max(x['cert'] for x in dense_cur):.1e}x" +
                   (f"; not yet at the frozen hashes: {', '.join(SHORT[x['model']] for x in dense if x not in dense_cur)}." if len(dense_cur) < 4 else "."))
    else:
        dense_s = "Dense P0: no model has both L and U at the frozen hashes yet."
    moe_cur = [(m, q) for m in MOE for q in ROLL_Q[:3] if (r := roll_row(idx, m, q)) and r.get("U") is not None and frozen_u(r) and not u_provisional(r) and r.get("L") is not None and not r.get("L_stale")]
    moe_1m = {SHORT[m]: ("current" if (r := roll_row(idx, m, 1048576)) and frozen_u(r) and not u_provisional(r) and r.get("L") is not None and not r.get("L_stale")
                         else ("redoing (U is a planner-fallback value)" if r and u_provisional(r) else "redoing")) for m in MOE}
    moe_s = (f"MoE: block family landed in the planner; {len(moe_cur)}/9 of the 8K-128K MoE cells are current on both sides; 1M rows: " +
             "; ".join(f"{k} {v}" for k, v in moe_1m.items()) + ". 4M MoE cells were not re-run tonight.")
    lines = [f"> **State at hand-off ({stamp})**", f"> 1. Hashes: lower `{cur.get('lower')}`, coarse `{cur.get('coarse')}`, upper `{cur.get('upper')}` (frozen: `{FROZEN_LOWER}` / `{NEW_PLANNER}` / `eab50c6955`).",
             f"> 2. Rollout rows: {len(roll)} total; U at the frozen planner {n_u_cur}, L at the current lower hash {n_l_cur}, both {n_both}; the rest are shown greyed (stale L) or marked previous / intermediate planner (U).",
             f"> 3. Still running on the pod (not at the frozen hashes on both sides): {', '.join(pending) if pending else 'none'}; also the 8B 128K F_hat variants "
             "(Table 2 F axis) and the 405B / MoE 4M cells (preset `rollout-4m`, not run tonight).",
             f"> 4. {dense_s}", f"> 5. {moe_s}",
             f"> 6. Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} from results/coarse.jsonl ({len(rows)} rows); analytic values only in P2 columns 1-3; acceptance section at the bottom. "
             "Refresh after the pod finishes: `python accumulation/sweeps/pod.py pull && python -m accumulation.sweeps.final_report`."]
    return "\n".join(lines)


def build_rollout(rows: list[dict], out_dir: Path, log=print, stamp: str = "8:15 AM") -> dict:
    idx = _index(rows)
    cur = module_versions()
    snap = prev_planner_snapshot(rows, out_dir)
    p0_md, p0_recs = p0(idx, cur, snap)
    p1_md, p1_data = p1(idx, cur, snap)
    parts = ["# Dynamic-rollout tables (ROLLOUT_PLAN.md P0-P4)", "", handoff_box(rows, idx, cur, p0_recs, stamp), "",
             f"Generated by `sweeps/final_report.py` from `results/coarse.jsonl` ({len(rows)} rows). Current module hashes: " +
             ", ".join(f"`{k}={v}`" for k, v in cur.items()) + f". Default policy point F_hat = {FH:g}, G_hat = {GH:g}, Q_inf = {Q}; "
             "rollout rows: tokens = Q_roll, seq = min(Q_roll, Q_inf), K = 1. Never hand-edited.", "", provenance(rows, cur), "",
             "**MoE rows are provisional on both sides.** U: the planner falls back to generic families (`bands` / `fine` / `single`) for the MoE forward -- a legal "
             "partition, not an attack -- so those U are shown marked `provisional` and no best-attack slowdown is derived from them; a MoE U becomes admissible when "
             "`U_detail.family` is a block / expert family (Agent B). L: the MoE certificate charges tokens + an LP cap below the dynamic-root bytes P, although every "
             "legal partition imports all of P at least once -- sound but provably loose -- so it is marked `provisional (L < P)` with the trivially-sound floor "
             "`max(L, P + tokens)` alongside (labelled floor, never certificate). A MoE L becomes admissible when L >= P + tokens at the current lower hash.", "",
             p0_md, "", p1_md, "", "![G](plots/figG_amortization.png)", "", p2(idx, cur, p0_recs), "", p3(idx, cur), "", p4(idx, cur), "",
             acceptance(idx, cur, p0_recs), ""]
    (Path(out_dir) / "ROLLOUT_TABLES.md").write_text("\n".join(parts))
    try:
        fig_g(p1_data["curves"], p1_data["fits"], Path(out_dir) / "plots" / "figG_amortization.png")
    except Exception as e:  # noqa: BLE001
        log(f"[rollout] fig G failed: {type(e).__name__}: {e}")
    log(f"[rollout] ROLLOUT_TABLES.md written ({out_dir})")
    return {"p0": p0_recs, "fits": {m: {k: (v if v is None else {kk: vv for kk, vv in v.items()}) for k, v in f.items() if k in ("L", "U")} for m, f in p1_data["fits"].items()}}
