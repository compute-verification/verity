"""The recursion bench's table (README.md): a ``recursion-e2e/v1`` result (``rec_profile.read``'s, or ``merge``'s) as
Markdown: a line a row, the outer totals, the verdicts and the release side. Stdlib only, as ``rec_profile``."""
from __future__ import annotations

from typing import Any

import rec_profile as RP


def _fmt(s: dict[str, Any] | None) -> str:
    if not s:
        return "—"
    if len(s["values"]) == 1:
        return f"{s['median']:.3g}"
    return f"{s['median']:.3g} [{s['min']:.3g}–{s['max']:.3g}] (n={len(s['values'])})"


def _bytes(n: int | None) -> str:
    if not n:
        return "—"
    for unit, k in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
        if n >= k:
            return f"{n / k:.3g} {unit}"
    return f"{n} B"


def markdown(doc: dict[str, Any]) -> str:
    p = doc.get("params", {})

    def each(q: dict[str, Any]) -> str:
        return (f"commit {str(q.get('commit'))[:12]}, the inner's {q.get('RUNS')} timed runs after {q.get('WARM')} warm, "
                f"each V* statement's {q.get('OPROVE_RUNS')} after {q.get('OPROVE_WARM')} "
                f"(lease {q.get('GPU_MAX_MIN', '?')} min)"
                + (f", V* statements {q['OPROVE_JOBS']} at a time" if q.get("OPROVE_JOBS") else "")
                + f", {q.get('REPEAT')} repeats per CPU step")
    runs = doc.get("run_params")
    h = doc.get("headline") or {}
    st = doc.get("staged")
    shape = (f"{st['subcircuit']} x {st['n']} (m = {st['m']}, {'served window' if st.get('window') else 'generated values'})"
             if st else f"K = {p.get('K')}, N = {p.get('N')}")
    lines = [f"Recursion bench: {shape}, oprove modes {p.get('modes', '?')}, "
             + ("; ".join(f"{k}: {each(q)}" for k, q in runs.items()) if runs else each(p))
             + f". Seconds: median [min–max] (n). Verdict: {verdict(h, p.get('modes', '?'))}"
             + (" (merged runs)" if h.get("merged") else "") + ".", "",
             "| stage | step | what | time s | run e2e s | pace s | peak RSS GB | GPU peak MiB | proof | dominant phase | ok |"
             + (" run |" if doc.get("runs") else ""),
             "|---|---|---|---|---|---:|---:|---:|---:|---|---|" + ("---|" if doc.get("runs") else "")]
    for r in doc["rows"]:
        run = f" {r.get('extra', {}).get('run')} |" if doc.get("runs") else ""
        dom = r.get("dominant")
        dom_s = (f"{dom['phase']} {dom['s']:.3g} s" + (f" ({dom['share']:.0%} of {dom['of']})"
                                                       if dom.get("share") else "")) if dom else "—"
        pace = r.get("extra", {}).get("session_s")
        pace_s = f"{pace:.3g}" if isinstance(pace, (int, float)) else "—"
        ok = {True: "yes", False: "**no**", None: "—"}[r.get("ok")]
        note = f" {r['note'][:120]}".replace("|", "/") if r.get("note") else ""
        rss = f"{r['peak_rss_gb']:.3g}" if r.get("peak_rss_gb") is not None else "—"
        gpu = f"{r['gpu_peak_mib']:.0f}" if r.get("gpu_peak_mib") else "—"
        lines.append(f"| {r['stage']} | {r['step']} | {r['what']} | {_fmt(r.get('wall_s'))} | {_fmt(r.get('e2e_s'))} | {pace_s} | {rss} | "
                     f"{gpu} | {_bytes(r.get('proof_bytes'))} | {dom_s} | {ok}{note} |{run}")
    s = doc["summary"]
    lines += ["", "Outer (V*'s statements, sum of each statement's median): " + "; ".join(
        f"{m}: prove {v['prove_s']:.3g} s, run e2e {v['e2e_s']:.3g} s, proofs {_bytes(int(v['proof_bytes']))} over "
        f"{v['statements']} statements" for m, v in s["outer"].items()) if s["outer"] else "Outer: not run."]
    if s.get("slowest"):
        lines.append(f"Slowest row (a run's e2e, else a tool's wall): {s['slowest']['stage']} ({s['slowest']['step']}), "
                     f"{s['slowest']['median_s']:.3g} s.")
    if s.get("pipeline_sum_s"):
        lines.append("One recursive run, sum of medians along its path: " + ", ".join(
            f"{k} {v:.3g}" for k, v in s["pipeline"].items() if v) + f"; total {s['pipeline_sum_s']:.3g} s.")
    loop = RP.loop_line(s)
    if loop:
        lines.append(loop)
    if h.get("forged_refused"):
        lines.append("Forgeries (85's forged statements and forge-after's cases), each refused: " + ", ".join(
            f"{k} {'yes' if v else '**no**'}" for k, v in h["forged_refused"].items()) + ".")
    if s.get("not_ok"):
        lines.append("Not accepted or not run: " + ", ".join(s["not_ok"]) + ".")
    lines.append("Steps' exit codes: " + ", ".join(f"{k} {'/'.join(v)}" for k, v in doc.get("steps", {}).items()) + ".")
    rel = doc.get("release")
    if rel:
        lines += ["", (f"**Release side.** Recursive session {rel['session']} on stream {rel['stream']}, inner coins from "
                       f"{rel.get('tree_source') or '?'} (challenger's log holds them: {_yn(rel.get('trees_held'))}); every part "
                       f"released at its time: {_yn(rel['ok'])}. Times are seconds from the inner connection (the schedule's "
                       "start)."), "",
                  "| part | window s | release at s | decided at s | slack s | prove wall s | decided | record |",
                  "|---|---:|---:|---:|---:|---:|---|---|"]
        t0 = rel.get("start_ms") or 0
        for p in rel["parts"]:
            at = {k: f"{(p[k] - t0) / 1000:.1f}" if p.get(k) is not None else "—" for k in ("release_ms", "ready_ms")}
            lines.append(f"| {p['part']} | {_s(p.get('window_ms'))} | {at['release_ms']} | {at['ready_ms']} | {_s(p.get('slack_ms'))} | "
                         f"{_fmt3(p.get('prove_wall_s'))} | {p.get('decided') or '—'} | {'/'.join(p.get('record') or []) or '—'} |")
        if rel.get("stops"):
            lines.append("Stops: " + "; ".join(f"{e.get('part')}: {e.get('why', e.get('event'))}" for e in rel["stops"]) + ".")
    return "\n".join(lines) + "\n"


def verdict(h: dict[str, Any], modes: Any) -> str:
    """The table's verdict from a result's headline. "accepted" only for a whole run whose parts passed and whose recursive
    statement Lean composed and accepted. A run that passed on its parts without that is said to be so, with its oprove
    modes and the reason; so is a headline from before ``composed``, which records no composition. A run whose
    composition Lean refused is not accepted, with Lean's reason: a forced retry's names the retry."""
    if h.get("ok") and h.get("composed") is True:
        return "one whole recursive run, accepted"
    if h.get("parts_ok", h.get("ok")) and h.get("composed") is not True:
        how = "**not composed**" if h.get("composed") is False else "**not recorded as composed**"
        why = str(h.get("not_composed") or "").removeprefix("not composed: ")
        return (f"one whole run, its outer proofs accepted and its recursive statement {how}, so not an accepted recursive "
                f"run (oprove modes {modes}" + (f": {why}" if why else "") + ")")
    if h.get("complete"):
        refused = h.get("refused")
        return "one whole recursive run, **not accepted**" + (
            f": Lean refused its recursive statement ({refused})" if refused else " (see ok and the steps' exits)")
    return "**not a whole recursive run**: some of " + ", ".join(RP.E2E_STEPS) + " did not run"


def _yn(v: Any) -> str:
    return {True: "yes", False: "**no**"}.get(v, "—")


def _s(ms: Any) -> str:
    return f"{ms / 1000:.1f}" if isinstance(ms, (int, float)) else "—"


def _fmt3(x: Any) -> str:
    return f"{x:.3g}" if isinstance(x, (int, float)) else "—"
