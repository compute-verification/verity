"""Front F6's pinned menu: dp4t's buckets, each at the slots per session B that costs least at its public session count,
fixed per model (F2's menu, merge thread 1791629248.978729).

    python -m verity_numerical.bench.unit_costs_menu --inventory unit-costs.json --out DIR [--outer proofs_c3_multiblock.json]
    python -m verity_numerical.bench.unit_costs_menu --inventory unit-costs.json --out DIR --rows VSTAR_DIR

**Buckets** (:func:`buckets`): :mod:`unit_costs_prover`'s dp4t at its CPU cap (``workload``'s cap: the largest power of two
whose 16-slot session at the workload's least ρ is in range and fits one node): the body's buckets by (template class, Z),
each rare type its own bucket, the body's units over the cap one bucket of their pieces at the cap, the tail's samplers each
their own bucket and its other large units one bucket at their largest Z, their units over the cap one bucket of pieces.

**Each bucket at each B** of :data:`B_MENU`: the inner session (``shape``, or ``pieces_shape`` for pieces; F6's law plus the
slot sumcheck), its sessions at the mean and at the public 1 − 2⁻⁴⁰ count, and the inner prover hours. The outer proof of
one inner session is an input (``--outer``): circuits' ``nonzero_cost.py --proofs-c3`` output, ``{"buckets": [...]}``, one
row per (bucket, B) with its ``prove_s`` and ``rss_gib`` on the composed layout, matched by the row id in its
``workload`` field (``WORKLOAD#BUCKET@B``). Its outer hours are sessions x ``prove_s``. ``--rows VSTAR_DIR`` writes those
rows' input instead (``proofs-rows/buckets.json`` beside proofs' ``k2.py`` and ``units.json``): proofs' candidate-3 V* row
for each (bucket, B), its templates by proofs' ``buckets.template`` at the cap and ``inner_m`` the inner session's
log2 D, from proofs' ``vstar_c3.py`` and ``buckets.py`` in VSTAR_DIR (art:4e9ff89a).

**The choice** (:func:`pick`): of the B whose inner session is in range (k_log ≤ 27), fits one node (peak ≤
:data:`~unit_costs_prover.NODE_GIB`), has D within the law's fitted range (≤ 2^:data:`D_FIT_LOG`) and whose outer proof
fits one node, the least public inner + outer hours (ties to the smaller B). Without ``--outer`` it chooses on the public
inner hours alone, and the output says so. Openings are not batched across sessions.

**One session per shape** (:func:`one_session_per_shape`; top's ruling, merge thread 1791664434.225959): a model's
entries that share a shape, (slots, bound), all take the largest inner session among them (a smaller entry's units are
costed in it by the law) and the largest outer proof, and are priced at them. V*'s statements are public, so a shape
whose entries had different sessions would tell which entry a session holds. B is still chosen per bucket, before the
shapes are merged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import unit_costs_padding as P
from . import unit_costs_prover as U

B_MENU = (1, 4, 16, 64, 256, 1024)
#: the law's largest fitted D: 2^35 (a 4096-instance session, r20261003-073948-0e7d)
D_FIT_LOG = 35


def buckets(N: int, K: int, units: Sequence[P.Unit], nets: Mapping[str, Mapping[str, Any]],
            cap: float | None) -> list[dict[str, Any]]:
    """dp4t's buckets at ``cap`` (the module docstring): part, class, Z, units, and whether it holds pieces."""
    body, tail, _thr = P.tail_split(units)
    small = [u for u in body if cap is None or u.z <= cap]
    groups, rares = U.menu_t(N, K, small, nets)
    out = [{"part": "body", "class": f"{c[0]} 2^{c[1]}", "Z": Z, "units": us, "pieces": False} for c, Z, us in groups]
    out += [{"part": "rare", "class": "rare", "Z": u.z, "units": [u], "pieces": False} for u in rares]
    if len(small) < len(body):
        out.append({"part": "body", "class": "cap pieces", "Z": cap, "units": [u for u in body if u not in small],
                    "pieces": True})
    samplers = [u for u in tail if u.template in P.SAMPLERS]
    big = [u for u in tail if u.template not in P.SAMPLERS]
    for mf, us0 in ((P.menu_natural, samplers), (P.menu_max, big)):
        for Z, us in (P.assign(us0, mf(us0)) if us0 else {}).items():
            keep = [u for u in us if cap is None or u.z <= cap]
            if keep:
                out.append({"part": "tail", "class": "tail", "Z": max(u.z for u in keep) if len(keep) < len(us) else Z,
                            "units": keep, "pieces": False})
            if len(keep) < len(us):
                out.append({"part": "tail", "class": "cap pieces", "Z": cap, "units": [u for u in us if u not in keep],
                            "pieces": True})
    return out


def inner(N: int, K: int, b: Mapping[str, Any], B: int, rho: Mapping[str, float], coef: Mapping[str, Sequence[float]],
          **kw: Any) -> dict[str, Any]:
    """Bucket ``b`` at B slots: its inner session and its sessions and prover hours, mean and public."""
    sh = (U.pieces_shape(N, K, b["units"], b["Z"], B, rho, coef, **kw) if b["pieces"]
          else U.shape(N, K, b["units"], b["Z"], B, 1, rho, coef, **kw))
    s = sh["session"]
    return {"slots": B, "k_log": s["k_log"], "in_range": s["in_range"], "D": s["D"], "inner_m": round(math.log2(s["D"])),
            "peak_gib": s["peak_gib"], "session_s": s["total_s"], "sessions_mean": sh["sessions_mean"],
            "sessions_public": sh["sessions_public"], "inner_h_mean": sh["prover_s_mean"] / 3600,
            "inner_h_public": sh["prover_s_public"] / 3600, "rho": s["rho"]}


def excluded(o: Mapping[str, Any], gib: float = U.NODE_GIB) -> str | None:
    """Why option ``o`` (``inner``'s, with its outer row's ``outer_rss_gib`` when there is one) cannot run, or None."""
    if not o["in_range"]:
        return f"k_log {o['k_log']} > {U.K_LOG_MAX}"
    if o["peak_gib"] > gib:
        return f"inner {o['peak_gib']:.0f} GiB > {gib:.0f}"
    if o["inner_m"] > D_FIT_LOG:
        return f"D 2^{o['inner_m']} > 2^{D_FIT_LOG}"
    if o.get("outer_rss_gib", 0.0) > gib:
        return f"outer {o['outer_rss_gib']:.0f} GiB > {gib:.0f}"
    return None


def pick(options: Mapping[int, Mapping[str, Any]], gib: float = U.NODE_GIB) -> int:
    """The B of ``options`` that can run (:func:`excluded`) with the least public inner + outer hours (outer 0 when
    absent), the smaller B on a tie."""
    ok = [B for B, o in options.items() if excluded(o, gib) is None]
    if not ok:
        raise ValueError("no B runs this bucket: " + "; ".join(f"B={B}: {excluded(o, gib)}" for B, o in options.items()))
    return min(ok, key=lambda B: (options[B]["inner_h_public"] + (options[B].get("outer_h_public") or 0.0), B))


def plan(inv: Mapping[str, Any], slot: Mapping[str, Any] | None = U.SLOT_SUMCHECK,
         b_menu: Sequence[int] = B_MENU) -> list[dict[str, Any]]:
    """Every workload's cap, buckets and each bucket's :func:`inner` at every B, with ``at_rho(B, r)``: its :func:`inner`
    at B with every unit at ρ = r (:func:`one_session_per_shape` sizes a smaller entry's session to its shape's)."""
    coef = U.fit()["coef"]
    K = int(inv["K"])
    out = []
    for w in dict.fromkeys(r["workload"] for r in inv["rows"]):
        N, units, _skipped = P.units_of(inv, w)
        rho, nets = U.rhos(inv, w), U.netlists(inv, w)
        r0 = min((rho.get(u.label, U.RHO_TABLE) for u in P.tail_split(units)[0]), default=U.RHO_TABLE)
        cap = U.cap(r0, coef, U.NODE_GIB, nets=[nets[u.label]["bytes"] for u in units if u.label in nets], slot=slot)
        bs = buckets(N, K, units, nets, cap if cap is not None and any(u.z > cap for u in units) else None)
        for b in bs:
            b["by_B"] = {B: inner(N, K, b, B, rho, coef, nets=nets, slot=slot) for B in b_menu}
            b["at_rho"] = lambda B, r, N=N, b=b, nets=nets: inner(N, K, b, B, {u.label: r for u in b["units"]}, coef,
                                                                 nets=nets, slot=slot)
        out.append({"workload": w, "N": N, "cap": cap, "buckets": bs})
    return out


def row_id(w: str, i: int, B: int) -> str:
    return f"{w}#{i}@{B}"


def proofs_rows(inv: Mapping[str, Any], pl: Sequence[Mapping[str, Any]], vstar: str | Path) -> list[dict[str, Any]]:
    """Proofs' candidate-3 row for every (bucket, B), in the form proofs' ``buckets.py`` writes and circuits'
    ``nonzero_cost.py --proofs-c3`` reads, from ``vstar_c3.py`` and ``buckets.py`` in ``vstar``."""
    sys.path.insert(0, str(vstar))
    try:
        V, BK = importlib.import_module("vstar_c3"), importlib.import_module("buckets")
    finally:
        sys.path.remove(str(vstar))
    by = {(r["workload"], r["unit_type"]): r for r in inv["rows"]}
    rows = []
    for W in pl:
        w = W["workload"]
        for i, b in enumerate(W["buckets"]):
            tm: dict[str, tuple[int, int, float]] = {}
            for u in b["units"]:
                key, nu, ell, inst, _src = BK.template(by[(w, u.label)], W["cap"])
                o = tm.get(key, (0, 0, 0))
                tm[key] = (max(nu, o[0]), max(ell, o[1]), max(inst, o[2]))
            nu, ell = max(v[0] for v in tm.values()), max(v[1] for v in tm.values())
            inst = math.ceil(max(v[2] for v in tm.values()))
            T, Tt = len(tm), len(b["units"])
            for B, o in b["by_B"].items():
                m, terms = V.terms(nu, ell, B, 2, T, inst)
                base = V.base(o["inner_m"])
                rows.append({"workload": row_id(w, i, B), "part": b["part"], "Z": b["Z"], "slots": B, "types": Tt,
                             "templates": T, "nu": nu, "ell": ell, "instances_max": inst, "m": m,
                             "m_if_T_types": V.table_m(nu, ell, Tt)[0], "vstar_rows": sum(terms.values()),
                             "vstar_rows_if_T_types": V.rows(nu, ell, B, 2, Tt, inst), "terms": terms,
                             "inner_m": o["inner_m"], "base_rows": base, "total_rows": base + sum(terms.values())})
    return rows


TOTALS = ("sessions_mean", "sessions_public", "inner_h_mean", "inner_h_public", "outer_h_mean", "outer_h_public",
          "total_h_mean", "total_h_public")


SESSION = ("k_log", "inner_m", "inner_peak_gib", "inner_session_s")


def one_session_per_shape(entries: Sequence[dict[str, Any]],
                          own: Sequence[tuple[float, Callable[[float], Mapping[str, Any]]]] | None = None,
                          gib: float = U.NODE_GIB) -> None:
    """Size every entry of a shape (slots, bound) to the shape's largest inner session (largest k_log) and give it the
    shape's largest outer session and RSS, with every outer refusal among them; reprice its hours at them. ``own[i]``:
    entry i's ρ and its :func:`inner` at a given ρ, so a smaller entry is costed as its own units in the larger session
    (the larger entry's ρ gives its k_log); without it, the smaller entry takes the larger's session costs."""
    shapes: dict[tuple[int, float], list[int]] = {}
    for i, e in enumerate(entries):
        shapes.setdefault((e["slots"], e["bound"]), []).append(i)
    for ix in shapes.values():
        es = [entries[i] for i in ix]
        t = max(ix, key=lambda i: (entries[i]["k_log"], entries[i]["inner_session_s"]))
        top = entries[t]
        has = es[0]["outer_session_s"] is not None
        out_s = max(e["outer_session_s"] for e in es) if has else None
        rss = max(e["outer_rss_gib"] for e in es) if has else None
        refused = sorted({r for e in es for r in e["outer_refused"]}) if has else None
        for i in ix:
            e = entries[i]
            if e["k_log"] < top["k_log"]:
                if own is None:
                    e.update({k: top[k] for k in SESSION})
                else:
                    o = own[i][1](own[t][0])
                    assert (o["k_log"], o["inner_m"]) == (top["k_log"], top["inner_m"]), (o["k_log"], top["k_log"])
                    if why := excluded(o, gib):
                        raise ValueError(f"entry {i} cannot run at its shape's session: {why}")
                    e.update(k_log=o["k_log"], inner_m=o["inner_m"], inner_peak_gib=o["peak_gib"],
                             inner_session_s=o["session_s"])
            e.update(outer_session_s=out_s, outer_rss_gib=rss, outer_refused=refused)
            for m in ("mean", "public"):
                e[f"inner_h_{m}"] = e[f"sessions_{m}"] * e["inner_session_s"] / 3600
                e[f"outer_h_{m}"] = e[f"sessions_{m}"] * out_s / 3600 if has else None
                e[f"total_h_{m}"] = e[f"inner_h_{m}"] + (e[f"outer_h_{m}"] or 0.0)


def menu(pl: Sequence[Mapping[str, Any]], K: int, outer: Mapping[str, Any] | None = None,
         outer_sha512: str | None = None) -> dict[str, Any]:
    """Each workload's pinned menu: every bucket at its :func:`pick`, and the totals per audit."""
    rows = {r["workload"]: r for r in (outer or {}).get("buckets", [])}
    models = []
    for W in pl:
        entries = []
        for i, b in enumerate(W["buckets"]):
            opts = {}
            for B, o in b["by_B"].items():
                o = dict(o)
                if outer is not None:
                    r = rows[row_id(W["workload"], i, B)]
                    o.update(outer_s=r["prove_s"], outer_rss_gib=r["rss_gib"], outer_refused=list(r.get("refused") or []),
                             **{f"outer_h_{m}": o[f"sessions_{m}"] * r["prove_s"] / 3600 for m in ("mean", "public")})
                opts[B] = o
            B = pick(opts)
            c = opts[B]
            has = outer is not None
            entries.append({
                "part": b["part"], "class": b["class"], "bound": b["Z"], "slots": B, "pieces": b["pieces"],
                "unit_types": [u.label for u in b["units"]], "k_log": c["k_log"], "inner_m": c["inner_m"],
                "inner_peak_gib": c["peak_gib"], "inner_session_s": c["session_s"],
                "outer_session_s": c["outer_s"] if has else None, "outer_rss_gib": c["outer_rss_gib"] if has else None,
                "outer_refused": c["outer_refused"] if has else None,
                **{k: c[k] for k in ("sessions_mean", "sessions_public", "inner_h_mean", "inner_h_public")},
                **{f"outer_h_{m}": c[f"outer_h_{m}"] if has else None for m in ("mean", "public")},
                **{f"total_h_{m}": c[f"inner_h_{m}"] + (c[f"outer_h_{m}"] if has else 0.0) for m in ("mean", "public")},
                "options": {str(x): excluded(o) or round(o["inner_h_public"] + (o.get("outer_h_public") or 0.0), 6)
                            for x, o in opts.items()}})
        own = [(b["by_B"][e["slots"]]["rho"], lambda r, b=b, B=e["slots"]: b["at_rho"](B, r))
               for b, e in zip(W["buckets"], entries)] if all("at_rho" in b for b in W["buckets"]) else None
        one_session_per_shape(entries, own)
        tot = {k: sum(e[k] or 0.0 for e in entries) for k in TOTALS}
        if outer is None:
            tot["outer_h_mean"] = tot["outer_h_public"] = None
        tot["refused_entries"] = sum(bool(e["outer_refused"]) and e["sessions_public"] > 0 for e in entries)
        models.append({"workload": W["workload"], "N": W["N"], "cap": W["cap"], "entries": entries, "totals": tot})
    return {"format": "verity/f6-menu/v1", "K": K, "b_menu": list(B_MENU), "node_gib": U.NODE_GIB,
            "k_log_max": U.K_LOG_MAX, "d_fit_log": D_FIT_LOG,
            "chosen_on": ("public inner + outer hours (outer per (bucket, B) from the input)" if outer is not None else
                          "public inner hours alone: no outer input, so the outer proof's cost and memory are not counted"),
            "outer_sha512": outer_sha512, "openings_batched": False,
            "inner_per_shape": "the largest among the shape's entries", "models": models}


def _h(x: float | None) -> str:
    return "–" if x is None else f"{x:.1f}"


def render(mn: Mapping[str, Any]) -> str:
    L = [f"# F6 pinned menu: dp4t, slots per session per bucket, K = {mn['K']}", "",
         f"Chosen on {mn['chosen_on']}; B from {mn['b_menu']}; a B runs if k_log ≤ {mn['k_log_max']}, inner and outer peak "
         f"≤ {mn['node_gib']} GiB and D ≤ 2^{mn['d_fit_log']}. Hours per audit, mean / public.", "",
         "| model | cap | entries (body / rare / tail) | inner h | outer h | inner + outer h | refused entries |",
         "|---|---|---|---|---|---|---|"]
    for M in mn["models"]:
        t, parts = M["totals"], [e["part"] for e in M["entries"]]
        L.append(f"| {P._short(M['workload'])} | {U._p2(M['cap'])} | "
                 f"{parts.count('body')} / {parts.count('rare')} / {parts.count('tail')} | "
                 f"{_h(t['inner_h_mean'])} / {_h(t['inner_h_public'])} | {_h(t['outer_h_mean'])} / {_h(t['outer_h_public'])} | "
                 f"{_h(t['total_h_mean'])} / {_h(t['total_h_public'])} | {t['refused_entries']} |")
    L += ["", "| model | part | class | bound | B | types | sessions public | inner h public | outer h public |",
          "|---|---|---|---|---|---|---|---|---|"]
    for M in mn["models"]:
        for e in M["entries"]:
            L.append(f"| {P._short(M['workload'])} | {e['part']} | {e['class']} | {U._p2(e['bound'])} | {e['slots']} | "
                     f"{len(e['unit_types'])} | {e['sessions_public']} | {_h(e['inner_h_public'])} | "
                     f"{_h(e['outer_h_public'])}{' (refused)' if e['outer_refused'] else ''} |")
    return "\n".join(L) + "\n"


def lean(mn: Mapping[str, Any], source: str) -> str:
    """The Lean module `VStar.Menu.Pinned` of the pinned menu ``mn`` (from ``source``): each model's entries as
    `VStar.Menu.Entry` (slots, bound, inner m, k_log, unit types' count)."""
    def entry(e: Mapping[str, Any]) -> str:
        if e["bound"] != int(e["bound"]):
            raise ValueError(f"bound {e['bound']} is not a whole number of nonzeros")
        return f"⟨{e['slots']}, {int(e['bound'])}, {e['inner_m']}, {e['k_log']}, {len(e['unit_types'])}⟩"
    models = ",\n".join(f"  ({json.dumps(M['workload'])},\n    [" + ",\n     ".join(entry(e) for e in M["entries"]) + "])"
                        for M in mn["models"])
    return "\n".join([
        "import VStar.Menu", "",
        "/-! Generated by `benchmarks/numerical/python/verity_numerical/bench/unit_costs_menu.py --lean` from", source,
        f"({mn['format']}, K = {mn['K']}); do not edit. -/", "", "namespace VStar.Menu", "",
        "/-- F6's pinned menu of each model, by workload. -/", "def pinned : List (String × List Entry) := [",
        models + "]", "", "end VStar.Menu", ""])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inventory", help="unit_costs' unit-costs.json")
    ap.add_argument("--out", required=True, help="directory for menu.json and menu.md (or proofs-rows/, or Pinned.lean)")
    ap.add_argument("--lean", nargs=2, metavar=("MENU_JSON", "SOURCE"),
                    help="write MENU_JSON's menus as the Lean module VStar.Menu.Pinned (Pinned.lean), citing SOURCE")
    ap.add_argument("--outer", default=None, help="circuits' nonzero_cost.py --proofs-c3 output on the --rows rows")
    ap.add_argument("--rows", default=None, metavar="VSTAR_DIR",
                    help="write proofs-rows/ (buckets.json, k2.py, units.json) from proofs' vstar_c3.py and buckets.py")
    a = ap.parse_args(argv)
    from .unit_costs import retree

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.lean:
        (out / "Pinned.lean").write_text(lean(json.loads(Path(a.lean[0]).read_text()), a.lean[1]))
        print(f"wrote {out / 'Pinned.lean'}")
        return 0
    if not a.inventory:
        ap.error("--inventory is required unless --lean")
    inv = json.loads(Path(a.inventory).read_text())
    if "token_select" not in inv:
        retree(inv)
    pl = plan(inv)
    if a.rows:
        d = out / "proofs-rows"
        d.mkdir(exist_ok=True)
        for f in ("k2.py", "units.json"):
            shutil.copy(Path(a.rows) / f, d / f)
        (d / "buckets.json").write_text(json.dumps(proofs_rows(inv, pl, a.rows), indent=1) + "\n")
        print(f"wrote {d / 'buckets.json'}")
        return 0
    outer, sha = None, None
    if a.outer:
        raw = Path(a.outer).read_bytes()
        outer, sha = json.loads(raw), hashlib.sha512(raw).hexdigest()
    mn = menu(pl, int(inv["K"]), outer, sha)
    (out / "menu.json").write_text(json.dumps(mn, indent=1) + "\n")
    (out / "menu.md").write_text(render(mn))
    print(f"wrote {out / 'menu.json'} and {out / 'menu.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
