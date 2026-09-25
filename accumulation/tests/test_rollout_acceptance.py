"""ROLLOUT_PLAN.md acceptance list as regression guards over ``results/coarse.jsonl`` (skipped when the rows are
absent).  Recorded values (2026-09-16): 8B rollout at Q = 1M: L = 0.0559 MB/tok, U = 0.212 MB/tok; 8B U(Q) = 1.96 ->
0.523 -> 0.221 -> 0.212 MB/tok over 8K -> 1M; 1B / 70B / 405B attack at 1M = 0.065 / 0.987 / 3.64 MB/tok.  L may only
rise (certificate improves), U may only fall (planner improves); deltas are printed."""
from __future__ import annotations

import json
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS_PATH = os.path.join(HERE, "results", "coarse.jsonl")
MB = 1e6
DEFAULT = dict(Q_inf=8192, F_hat=1.5, G_hat=4.0)
ROLL_Q = (8192, 32768, 131072, 1048576)
RECORDED_8B_1M = {"L_MB_tok": 0.0559, "U_MB_tok": 0.212}
RECORDED_8B_U = {8192: 1.96, 32768: 0.523, 131072: 0.221, 1048576: 0.212}
RECORDED_ATTACK_1M = {"llama3-1b": 0.065, "llama3-70b": 0.987, "llama3-405b": 3.64}
TOL = 1e-2          # relative: rounding of the recorded 3-significant-figure values


def _rows() -> list[dict]:
    if not os.path.exists(ROWS_PATH):
        pytest.skip("results/coarse.jsonl absent")
    out = []
    with open(ROWS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    if not out:
        pytest.skip("results/coarse.jsonl empty")
    return out


def _default(r: dict) -> bool:
    return all(r.get(k) == v for k, v in DEFAULT.items())


def _rollout(rows, model: str, Q: int, circuit: str = "forward-nonfixed") -> dict | None:
    for r in rows:
        if r.get("model") == model and r.get("circuit") == circuit and r.get("Q_step") == Q and r.get("K") == 1 and _default(r):
            return r
    return None


def _per_tok(r: dict, key: str) -> float | None:
    v = r.get(key)
    return None if v is None else v / r["tokens"]


def test_inference_is_tokens_only_one_ru():
    rows = [r for r in _rows() if r.get("circuit") in ("inference-dense", "inference-moe") and r.get("error") is None]
    if not rows:
        pytest.skip("no inference rows")
    bad = []
    for r in rows:
        tok = r.get("token_bytes")
        if tok is None or r.get("tokens") in (None, 0):
            continue
        if tok != 4 * r["tokens"]:
            bad.append(f"{r['model']} {r['circuit']} Q_inf={r['Q_inf']}: token_bytes {tok} != 4 B/token")
        if r.get("U") is not None and r["U"] != tok:
            bad.append(f"{r['model']} {r['circuit']} Q_inf={r['Q_inf']}: U={r['U']} != token bytes {tok}")
        if r.get("L") is not None and r["L"] != tok:
            bad.append(f"{r['model']} {r['circuit']} Q_inf={r['Q_inf']}: L={r['L']} != token bytes {tok}")
        n_ru = r.get("U_n_ru")
        if n_ru is None and isinstance(r.get("U_detail"), dict):
            n_ru = r["U_detail"].get("n_units")
        if r.get("U") is not None and n_ru not in (None, 1):
            bad.append(f"{r['model']} {r['circuit']} Q_inf={r['Q_inf']}: {n_ru} RUs (must be one)")
    assert not bad, "\n".join(bad)
    print(f"\n{len(rows)} inference rows: 4 B/token, one RU (dense: "
          f"{sum(1 for r in rows if r['circuit'] == 'inference-dense')}, MoE: {sum(1 for r in rows if r['circuit'] == 'inference-moe')})")


def test_8b_rollout_1m_bracket_and_monotone_vs_recorded():
    rows = _rows()
    r = _rollout(rows, "llama3-8b", 1048576)
    if r is None or r.get("U") is None:
        pytest.skip("8B forward-nonfixed @ 1M default-point row absent")
    U = _per_tok(r, "U") / MB
    L = None if r.get("L") is None else _per_tok(r, "L") / MB
    print(f"\n8B rollout 1M: L = {L} MB/tok (recorded {RECORDED_8B_1M['L_MB_tok']}), U = {U:.4f} MB/tok (recorded {RECORDED_8B_1M['U_MB_tok']})")
    assert U <= RECORDED_8B_1M["U_MB_tok"] * (1 + TOL), f"U rose: {U:.4f} > recorded {RECORDED_8B_1M['U_MB_tok']} MB/tok (planner regression)"
    if L is not None:
        assert L >= RECORDED_8B_1M["L_MB_tok"] * (1 - TOL), f"L fell: {L:.4f} < recorded {RECORDED_8B_1M['L_MB_tok']} MB/tok (certificate regression)"
        assert L <= U, f"L={L} > U={U} MB/tok: soundness violation"
        print(f"  deltas: L {L - RECORDED_8B_1M['L_MB_tok']:+.4f} MB/tok, U {U - RECORDED_8B_1M['U_MB_tok']:+.4f} MB/tok, U/L = {U / L:.2f}")
    else:
        pytest.xfail("8B 1M row has no L yet (certificate pending)")


def test_8b_U_per_token_monotone_over_Q():
    rows = _rows()
    pts = []
    for Q in ROLL_Q:
        r = _rollout(rows, "llama3-8b", Q)
        if r is not None and r.get("U") is not None:
            pts.append((Q, _per_tok(r, "U") / MB))
    if len(pts) < 2:
        pytest.skip("fewer than two 8B rollout points")
    print("\n8B U(Q) MB/tok: " + ", ".join(f"{Q}: {u:.4f} (rec {RECORDED_8B_U[Q]})" for Q, u in pts))
    for (q1, u1), (q2, u2) in zip(pts, pts[1:]):
        assert u2 <= u1 * (1 + 1e-9), f"U per token rose from Q={q1} ({u1:.4f}) to Q={q2} ({u2:.4f})"
    for Q, u in pts:
        assert u <= RECORDED_8B_U[Q] * (1 + TOL), f"8B U at Q={Q} rose above the recorded {RECORDED_8B_U[Q]} MB/tok: {u:.4f}"


def test_dense_U_per_token_monotone_each_model():
    rows = _rows()
    models = sorted({r["model"] for r in rows if r.get("circuit") == "forward-nonfixed" and r.get("model", "").startswith("llama3")})
    if not models:
        pytest.skip("no dense rollout rows")
    checked = 0
    for m in models:
        pts = [(Q, _per_tok(r, "U") / MB) for Q in ROLL_Q if (r := _rollout(rows, m, Q)) is not None and r.get("U") is not None]
        for (q1, u1), (q2, u2) in zip(pts, pts[1:]):
            assert u2 <= u1 * (1 + 1e-9), f"{m}: U per token rose from Q={q1} ({u1:.4f}) to Q={q2} ({u2:.4f})"
        if len(pts) >= 2:
            checked += 1
        print(f"\n{m} U(Q) MB/tok: " + ", ".join(f"{Q}: {u:.4f}" for Q, u in pts))
    if checked == 0:
        pytest.skip("no dense model with >= 2 rollout points")


def test_attack_1m_recorded_values_only_fall():
    rows = _rows()
    seen = 0
    for m, rec in RECORDED_ATTACK_1M.items():
        r = _rollout(rows, m, 1048576)
        if r is None or r.get("U") is None:
            continue
        seen += 1
        u = _per_tok(r, "U") / MB
        print(f"\n{m} attack at 1M: {u:.4f} MB/tok (recorded {rec}, delta {u - rec:+.4f})")
        assert u <= rec * (1 + TOL), f"{m}: U at 1M rose to {u:.4f} > recorded {rec} MB/tok"
    if seen == 0:
        pytest.skip("no 1B/70B/405B 1M rows")


def _current_upper_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rows whose recorded planner hashes (``versions.upper`` / ``versions.coarse``) match the modules on disk = the rows
    the tables may show ("stale rows are re-run, never displayed as current"); the rest are stale."""
    from accumulation.sweeps.coarse_run import module_versions
    cur = module_versions()
    current, stale = [], []
    for r in rows:
        v = r.get("versions") or {}
        (current if v.get("upper") == cur.get("upper") and v.get("coarse") == cur.get("coarse") else stale).append(r)
    return current, stale


def test_training_rows_shown_are_acyclic():
    rows = [r for r in _rows() if r.get("is_training") and r.get("U") is not None]
    if not rows:
        pytest.skip("no training rows with U")
    current, stale = _current_upper_rows(rows)
    print(f"\n{len(current)} training rows at the current planner hashes, {len(stale)} stale (must be re-run before display)")
    if not current:
        pytest.skip("no training row at the current planner hashes (all stale: re-run with --redo stale)")
    missing = [f"{r['model']} {r['circuit']} tokens={r['tokens']} K={r.get('K')}" for r in current
               if not (isinstance(r.get("U_detail"), dict) and r["U_detail"].get("acyclic") is True)]
    assert not missing, "current training rows whose U is not certified acyclic (U_detail.acyclic != True):\n" + "\n".join(missing[:40])


def test_no_soundness_violation_rows():
    rows = _rows()
    bad = [r for r in rows if r.get("soundness_violation")]
    bad += [r for r in rows if r.get("L") is not None and r.get("U") is not None and r["L"] > r["U"] and r.get("error") is None]
    assert not bad, "rows flagged soundness_violation or with L > U:\n" + "\n".join(
        f"{r['model']} {r['circuit']} tokens={r.get('tokens')} F_hat={r.get('F_hat')} G_hat={r.get('G_hat')}: L={r.get('L')} U={r.get('U')} "
        f"{r.get('soundness_violation')}" for r in bad[:20])
