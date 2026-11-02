"""One recursive session's cost per monitoring period: a `recursion_e2e` result's measured parts, one session's (its inner
run, V*'s staging, V*'s firewalled releases one at a time, and the Lean step that checks every release), scaled to every
machine-window of a deployment over a week, a month and a year.

    python benchmarks/sampling_plan/recursion.py --result recursion_e2e.json [--sessions-per-window 1] [--out DIR]

* **Measured**: each part's wall on the run's CPU node, as `recursion_e2e result` reads it (`headline.pipeline`), and
  beside it the same units' non-recursive `--zk` session (the row `baseline --zk`), from the same run.
* **Scaled**: every machine-window of the deployment holds ``sessions-per-window`` sessions of that shape, each at the
  measured cost. A period's sessions are machines x period / window; its node-hours are sessions x a part's wall; the
  nodes the deployment keeps busy are those node-hours over the period.
* **Deployments**: the census's datacenter (`benchmarks/census/deployments.json`, `dc-12500-machines-600s`) and the
  world: `benchmarks/rho/data/reference.toml`'s `world-h200` fleet (HGX servers, derived from the census's
  world-inference-2027), audited in the datacenter's windows (assumed: the census has no world cadence).
"""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from cpu_budget import PERIODS

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = "dc-12500-machines-600s"
WORLD = "world-h200"
#: the result's measured parts (`rec_profile.summary`'s pipeline), by the side that pays them
PROVER = ("inner M0 e2e_s", "V* staging_s (vstage + vcap steps)")
AUDITOR = ("Lean step_s",)


def deployments(root: Path = ROOT) -> list[dict]:
    dc = next(d for d in json.loads((root / "benchmarks/census/deployments.json").read_text()) if d["id"] == DEPLOYMENT)
    world = tomllib.loads((root / "benchmarks/rho/data/reference.toml").read_text())["fleet"][WORLD]["count"]
    return [{"id": dc["id"], "machines": dc["machines"], "window_s": dc["window_s"], "status": "census (stated)"},
            {"id": WORLD, "machines": world["value"], "window_s": dc["window_s"],
             "status": f"benchmarks/rho {WORLD} ({world['status']}: {world['source']}); windows assumed {dc['window_s']} s"}]


def session(result: dict) -> dict:
    """One session's measured walls by part: the prover's (the inner run, V*'s staging, its releases) and the auditor's
    (the Lean step), with the non-recursive `--zk` session over the same units beside them."""
    pipe = result["headline"]["pipeline"]
    outer = next((k for k in pipe if k.startswith("outer prove")), None)
    missing = [k for k in (*PROVER, outer, *AUDITOR) if not pipe.get(k)]
    if missing:
        raise ValueError(f"the result has no measured {missing}: not a whole recursive run")
    zk = next((r for r in result["rows"] if r["stage"] == "baseline --zk"), None)
    prover = {k: pipe[k] for k in (*PROVER, outer)}
    auditor = {k: pipe[k] for k in AUDITOR}
    return {"prover": prover, "auditor": auditor, "prover_s": sum(prover.values()), "auditor_s": sum(auditor.values()),
            "statements": result["summary"]["outer"].get(outer[len("outer prove ("):-len(") e2e_s")], {}).get("statements"),
            "baseline_zk_e2e_s": zk["e2e_s"]["median"] if zk and zk.get("e2e_s") else None,
            "ok": result["headline"].get("ok"), "staged": result.get("staged")}


def periods(s: dict, deps: list[dict], per_window: float = 1.0) -> list[dict]:
    out = []
    for d in deps:
        for name, seconds in PERIODS.items():
            n = d["machines"] * seconds / d["window_s"] * per_window
            row = {"deployment": d["id"], "period": name, "sessions": n}
            for side in ("prover", "auditor"):
                row[f"{side}_node_h"] = n * s[f"{side}_s"] / 3600
                row[f"{side}_nodes"] = n * s[f"{side}_s"] / seconds
            if s["baseline_zk_e2e_s"]:
                row["baseline_zk_node_h"] = n * s["baseline_zk_e2e_s"] / 3600
            out.append(row)
    return out


def render(s: dict, deps: list[dict], rows: list[dict], per_window: float) -> str:
    st = s.get("staged") or {}
    lines = [f"# One recursive session's cost per monitoring period ({per_window:g} a machine-window)", "",
             f"Session: {st.get('n', '?')} x {st.get('unit', '?')} (m = {st.get('m', '?')}), "
             f"{'a served window' if st.get('window') else 'generated values'}; verdict ok = {s['ok']}.", "",
             "Measured, one session on the run's CPU node (s): "
             + ", ".join(f"{k} {v:.4g}" for k, v in {**s["prover"], **s["auditor"]}.items())
             + f"; V* statements {s['statements']}; the non-recursive --zk session over the same units "
             + (f"{s['baseline_zk_e2e_s']:.4g}" if s["baseline_zk_e2e_s"] else "not measured") + ".", "",
             "Scaled (every machine-window at the measured cost):", ""]
    lines += [f"- {d['id']}: {d['machines']:,.0f} machines, {d['window_s']} s windows ({d['status']})" for d in deps]
    lines += ["", "| deployment | period | sessions | prover node-h | prover nodes busy | auditor node-h | auditor nodes busy "
              "| non-recursive --zk node-h |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['deployment']} | {r['period']} | {r['sessions']:.4g} | {r['prover_node_h']:.4g} | "
                     f"{r['prover_nodes']:.4g} | {r['auditor_node_h']:.4g} | {r['auditor_nodes']:.4g} | "
                     f"{r.get('baseline_zk_node_h', float('nan')):.4g} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", type=Path, required=True, help="recursion_e2e result (`run.py result --out`)")
    ap.add_argument("--sessions-per-window", type=float, default=1.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    s, deps = session(json.loads(a.result.read_text())), deployments()
    rows = periods(s, deps, a.sessions_per_window)
    text = render(s, deps, rows, a.sessions_per_window)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "recursion-periods.json").write_text(json.dumps({"session": s, "deployments": deps, "periods": rows}, indent=1))
        (a.out / "recursion-periods.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
