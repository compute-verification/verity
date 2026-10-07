"""The FP8 served table at a scheme's cap: per model, its served pass's per-input measurement (`served_debit.py --per-input`'s
``inputs.json``), the verdicts of its reference verify (`verify_run.py`'s ``verify.json``) where one was run, and the ε that
`HonestTileCapRate`'s served citation takes: the largest of the model's inputs' upper limits, each recomputed at
δ = 0.05/m with m every input the table measures (all its models, and a model's fresh trials beside its pass-salt
measurement), so the limits are 95% simultaneous over the whole table.

  python benchmarks/pouw/pearl_c_vllm/served_table.py --model qwen3-8b INPUTS_JSON [VERIFY_JSON ...] \\
      --model llama-3.1-8b INPUTS_JSON [--fresh qwen3-8b FRESH_INPUTS_JSON] --out DIR

Writes ``<out>/table.json`` and ``<out>/table.md``.  A model's row is refused when its inputs name more than one run, when an
input's openings failed, or when a verify given for it did not accept its pass (a negative control's, ``control*``, must
reject).  ``--fresh`` adds a model's ε from fresh-salt trials (``--fresh-draws``) beside the pass-salt one; ``--pending``
counts trials still running in m, so a table published before them states the limits the finished table will.
``--window`` adds a salt-to-deadline window's replay diagnostic from its audit (refused unless the audit accepted), with its
overhead over cuBLASLt beside the proving cost hidden-zk measures (``--proving-cost``; the cell points to hidden-zk until then).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import served_debit as SD  # noqa: E402


def _cite(rows: list[dict], m: int) -> tuple[list[tuple[float, str]], int]:
    """Each row's upper limit at δ = 0.05/m (`served_debit.input_bound`) and the position of the largest."""
    bounds = [SD.input_bound(r, 0.05 / m) for r in rows]
    return bounds, max(range(len(rows)), key=lambda i: bounds[i][0])


def model_row(name: str, inputs: dict, verifies: list[dict], fresh: dict | None, m: int) -> dict:
    """A model's row, every input bounded at δ = 0.05/m with m the table's measured inputs over all its models."""
    rows = inputs["inputs"]
    runs = sorted({r["run"] for r in rows})
    if len(runs) != 1:
        raise ValueError(f"{name}: inputs from {runs}; a model's ε is over one served pass's inputs")
    if any(r["openings_failed"] for r in rows):
        raise ValueError(f"{name}: an input's openings failed; the measurement is not of the committed pass")
    bounds, worst = _cite(rows, m)
    out = {"model": name, "run": runs[0], "passes": sorted({r["pass"] for r in rows}),
           "manifests": sorted({f"{r['pass']}:{r['manifest_sha256']}" for r in rows}), "m_model": inputs["m"],
           "delta_per_input": 0.05 / m, "tile_draws": sum(r["drawn"] for r in rows),
           "failures": sum(r["failures"] for r in rows), "inputs_with_failures": sum(r["failures"] > 0 for r in rows),
           "share_max": max((r["share_max"] for r in rows if r["share_max"] is not None), default=None),
           "epsilon": bounds[worst][0], "epsilon_method": bounds[worst][1],
           "epsilon_input": {k: rows[worst][k] for k in ("pass", "matmul", "weight", "m", "k", "n") if k in rows[worst]},
           "inputs": [{k: r[k] for k in ("pass", "matmul", "k", "n", "NT", "method", "drawn", "failures", "share_max")}
                      | {"upper": u} for r, (u, _) in zip(rows, bounds)]}
    for v in verifies:
        label = Path(v["manifest"]).parent.name
        want = not label.startswith("control")
        if v.get("cross_check_failed") or bool(v["accepted"]) != want:
            raise ValueError(f"{name}: the verify of {label} says {v['line'][:120]}")
        if v["transcript"].split(":", 1)[0] != runs[0]:
            raise ValueError(f"{name}: the verify of {label} is of {v['transcript'].split(':', 1)[0]}, not {runs[0]}")
    out["verify"] = [{"pass": Path(v["manifest"]).parent.name, "accepted": v["accepted"], "replay": v.get("replay", "scalar"),
                      "tiles": len(v["tiles"]), "cross_checked": len(v.get("cross_checked", [])),
                      "transcript_sha256": v["transcript_sha256"], "line": v["line"]} for v in verifies]
    if fresh is not None:
        if {r["run"] for r in fresh["inputs"]} != {runs[0]} or any(r["openings_failed"] for r in fresh["inputs"]):
            raise ValueError(f"{name}: the fresh trials are not of {runs[0]}'s committed pass")
        if sorted((r["pass"], r["matmul"]) for r in fresh["inputs"]) != sorted((r["pass"], r["matmul"]) for r in rows):
            raise ValueError(f"{name}: the fresh trials are of other inputs than the committed list (served_inputs.sh)")
        fb, fw = _cite(fresh["inputs"], m)
        out["fresh"] = {"draws": fresh["fresh_draws"], "source": fresh["fresh_source"], "m_model": fresh["m"],
                        "trials": sum(r["drawn"] for r in fresh["inputs"]), "failures": sum(r["failures"] for r in fresh["inputs"]),
                        "epsilon": fb[fw][0], "epsilon_input": {k: fresh["inputs"][fw][k] for k in ("pass", "matmul")}}
    return out


def window_row(window_run: str, audit_run: str, audit: dict, proving: dict[str, str]) -> dict:
    """A salt-to-deadline window's row from its audit (`archive/pouw-blake3/exhaustion/audit.py`'s audit.json), refused unless
    every check of the audit held."""
    if not audit.get("ok"):
        bad = [c["check"] for c in audit.get("checks", []) if not c["ok"]]
        raise ValueError(f"window {window_run}: its audit {audit_run} did not accept ({', '.join(bad[:4])})")
    f = audit["fraction"]
    claimed = sum(audit["gemms"].values())
    return {"window_run": window_run, "audit_run": audit_run, "scheme": audit["scheme"], "shape": audit["shape"],
            "t_s": audit["t_s"], "t_d": audit["t_d"], "verifier_commit": audit["verifier_commit"], "claimed_gemms": claimed,
            "credited_gemms": f["credited_gemms"], "capacity_gemms": f["capacity_gemms"],
            "uncertified_fraction": f["uncertified_fraction"], "fraction": f["fraction"], "gamma": f["gamma"], "eps_s": f["eps_s"],
            "slowdown": f["slowdown"], "tiles": len(audit["tiles"]), "tiles_accepted": sum(t["ok"] for t in audit["tiles"]),
            "controls": {c["label"]: "REJECT" if not c["ok"] else "ACCEPT" for c in audit["controls"]},
            "proving_cost": proving.get(audit["scheme"])}


def markdown(table: dict) -> str:
    lines = [f"# FP8 served table at cap {table['cap']}", "",
             "| model | pass | m | tile draws | failures | worst share | ε (95% simultaneous) | bound | fresh-salt ε | verify |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for r in table["models"]:
        if "absent" in r:
            lines.append(f"| {r['model']} | — | — | — | — | — | none | {r['absent']} | — | — |")
            continue
        fresh = (f"{r['fresh']['epsilon']:.4f} ({r['fresh']['trials']} trials, {r['fresh']['failures']} failed)" if "fresh" in r
                 else f"pending (`{r['fresh_pending']['run']}`)" if "fresh_pending" in r else "—")
        verify = ", ".join(f"{v['pass']} {'ACCEPT' if v['accepted'] else 'REJECT'}" for v in r["verify"]) or "—"
        lines.append(f"| {r['model']} | `{r['run']}` | {r['m_model']} | {r['tile_draws']} | {r['failures']} | "
                     f"{r['share_max']:.3g} | {r['epsilon']:.4f} | {r['epsilon_method']} | {fresh} | {verify} |")
    lines += ["", f"ε is the largest of a model's inputs' upper limits on the input's honest tile-rejection rate at the cap, "
              f"each at δ = 0.05/m with m = {table['m']}, every input the table measures over all its models and "
              f"measurements (served_debit.input_bound); an input is one matmul of the pass. Every draw, pass-salt and fresh, "
              f"weighs an input's tiles by their area |rows|·|cols|, which within one input (one shape) is proportional to "
              f"`TR.Wref` (`pearlCTilesDevRev1K`: `tileShare` on the tile's rows, not its passing rows). So ε is a rate "
              f"under Lean's draw, not under the deployed audit's, whose `tile_weight` counts credited cells only and so "
              f"differs on tiles with filler, excluded or voluntary rows (PROTOCOL.md, \"Not here yet\", red team #1014)."]
    quality = [r for r in table["models"] if "quality" in r]
    if quality:
        lines += ["", "| model | quality run | mode | scheme | WikiText-2 perplexity | top-1 vs bf16 | greedy tokens vs bf16 |",
                  "|---|---|---|---|---:|---:|---:|"]
        for r in quality:
            for mode, q in r["quality"]["summary"].items():
                lines.append(f"| {r['model']} | `{r['quality']['run']}` | {mode} | {q.get('scheme') or '—'} | "
                             f"{q['perplexity']:.4f} | {q.get('top1_agreement_vs_bf16', float('nan')):.4f} | "
                             f"{q.get('greedy_token_agreement_vs_bf16', float('nan')):.4f} |")
        lines += ["", "A quality run's scheme may name another hashing format than the served `-h3` (runs before 2026-10-06 "
                  "ran `pearl-c-sm120-v1-h1`): the formats differ only in their hashes. The noise seeds hash A's and B's "
                  "roots, so a run under one format draws other noise than a run under another, as another salt does; the "
                  "forming, the chain and the peel are the same."]
    for r in table["models"]:
        if "forming" in r:
            f = r["forming"]
            lines += ["", f"{r['model']}: the served debit leaves out Lean's forming term fs·(|C|/n)·elems "
                      f"(`pearl_c_debit.forming_elements` is a stub). On its {f['tiles']} recorded tiles, at Prices.sm120Loop, a "
                      f"verdict flips only if at least {f['dead_to_flip_min']:.2%} of a tile's |R|·k elements are salt-dead "
                      f"(forming_headroom.py; the unneeded ones are bounded by the recorded atoms). Lean's ρ_D = 1/64 "
                      f"({1 / 64:.2%}) of a unit's elements (`FormingP.saltDeadP_card_le`, under its domain conditions) "
                      f"{'covers' if f['unit_rows_covered_by_rho_D'] else 'does not cover'} the tiles that hold all their "
                      f"unit's rows (decode); a prefill tile has the threshold only."]
    windows = table.get("windows", [])
    if windows:
        lines += ["", "## Replay diagnostic: salt-to-deadline windows", "",
                  "| window | audit | scheme | t_s | credited / claimed GEMMs | cuBLASLt capacity | replay-diagnostic fraction "
                  "[certified, uncertified] | overhead × cuBLASLt | proving cost | tiles accepted | controls |",
                  "|---|---|---|---|---:|---:|---:|---:|---|---:|---|"]
        for w in windows:
            controls = ", ".join(f"{k} {v}" for k, v in sorted(w["controls"].items()))
            lines.append(f"| `{w['window_run']}` | `{w['audit_run']}` | {w['scheme']} | {w['t_s']} | "
                         f"{w['credited_gemms']:,} / {w['claimed_gemms']:,} | {w['capacity_gemms']:,.0f} | "
                         f"{w['fraction']:.2%} [{w['fraction']:.4f}, {w['uncertified_fraction']:.4f}] | {w['slowdown']:.3f} | "
                         f"{w['proving_cost'] or '— (hidden-zk)'} | {w['tiles_accepted']} / {w['tiles']} | {controls} |")
        fr = [w["fraction"] for w in windows]
        lines += ["", "A replay audit is a diagnostic, not a verification (Daniel, 2026-10-04 4:06 PM PDT): only a sampled "
                  "proof over C-Flock with ZK verifies a computation. Each window's audit recomputes its drawn tiles on the "
                  "CPU (`archive/pouw-blake3/exhaustion/audit.py`), so its fraction is the replay diagnostic: certified = "
                  "(1 − γ)(1 − ε_s)·credited / capacity, the lower end at δ_s = 1/20 over the audit's draws; uncertified = "
                  "credited / capacity, with capacity the cards' fastest cuBLASLt FP8 GEMMs in the window (before and after, "
                  "averaged). The overhead is capacity / credited, the served kernel's time per GEMM over cuBLASLt's. Proving "
                  "cost is the cost of the sampled C-Flock ZK proofs over the same work, which hidden-zk measures; "
                  "'— (hidden-zk)' means it has no number for the scheme yet."]
        if len(fr) > 1:
            lines += ["", f"Across the {len(fr)} windows the replay-diagnostic fraction spans {min(fr):.2%} to {max(fr):.2%} "
                      f"(spread {max(fr) - min(fr):.2%}, {(max(fr) - min(fr)) * 1e4:.1f} basis points)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", nargs="+", action="append", required=True, metavar=("NAME", "JSON"),
                   help="a model's name, its inputs.json, then any verify.json of its pass")
    p.add_argument("--fresh", nargs=2, action="append", default=[], metavar=("NAME", "INPUTS_JSON"))
    p.add_argument("--absent", nargs=2, action="append", default=[], metavar=("NAME", "REASON"),
                   help="a model the table is for but cannot cite, and why")
    p.add_argument("--pending", nargs=3, action="append", default=[], metavar=("NAME", "M", "RUN"),
                   help="a model's fresh trials still running in RUN: their M bounds count in m now, so that no limit moves "
                        "when they land")
    p.add_argument("--forming", nargs=2, action="append", default=[], metavar=("NAME", "HEADROOM_JSON"),
                   help="a model's forming_headroom.py report over its recorded tiles")
    p.add_argument("--quality", nargs=2, action="append", default=[], metavar=("NAME", "QUALITY_JSON"),
                   help="a model's window.sh MODE=perplexity quality.json (every group's gate must have passed)")
    p.add_argument("--window", nargs=3, action="append", default=[], metavar=("WINDOW_RUN", "AUDIT_RUN", "AUDIT_JSON"),
                   help="a salt-to-deadline window's run, its audit's run and the audit's audit.json (its replay diagnostic)")
    p.add_argument("--proving-cost", nargs=2, action="append", default=[], metavar=("SCHEME", "COST"),
                   help="the proving cost hidden-zk measured for a window scheme, as the cell should read")
    p.add_argument("--cap", default="1/1000")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    fresh = {name: json.loads(Path(path).read_text()) for name, path in a.fresh}
    pending = {name: (int(m), run) for name, m, run in a.pending}
    if any(len(spec) < 2 for spec in a.model):
        p.error("--model NAME INPUTS_JSON [VERIFY_JSON ...]")
    if set(pending) & set(fresh) or set(pending) - {spec[0] for spec in a.model}:
        p.error("--pending names a model given by --model and without --fresh")
    specs = [(name, json.loads(Path(inputs).read_text()), verifies) for name, inputs, *verifies in a.model]
    m = (sum(doc["m"] for _, doc, _ in specs) + sum(doc["m"] for doc in fresh.values())
         + sum(n for n, _ in pending.values()))
    models = [model_row(name, doc, [json.loads(Path(v).read_text()) for v in verifies], fresh.get(name), m)
              | ({"fresh_pending": {"m_model": pending[name][0], "run": pending[name][1]}} if name in pending else {})
              for name, doc, verifies in specs]
    for name, path in a.forming:
        doc = json.loads(Path(path).read_text())
        row = next((r for r in models if r["model"] == name), None)
        if row is None:
            p.error(f"--forming {name}: no such --model")
        row["forming"] = {k: doc[k] for k in ("tiles", "dead_to_flip_min", "worst", "unit_rows_covered_by_rho_D", "rho_D")}
    for name, path in a.quality:
        doc = json.loads(Path(path).read_text())
        row = next((r for r in models if r["model"] == name), None)
        if row is None:
            p.error(f"--quality {name}: no such --model")
        if not all(doc["gates"].values()):
            raise ValueError(f"{name}: the quality run's gates {doc['gates']} did not all pass")
        row["quality"] = {k: doc[k] for k in ("run", "model", "dataset", "windows", "window_tokens", "summary")}
    models += [{"model": name, "absent": reason} for name, reason in a.absent]
    table = {"tool": "pouw_pearl_c_served_table", "cap": a.cap, "m": m, "delta_per_input": 0.05 / m, "models": models}
    if a.window:
        proving = dict(a.proving_cost)
        table["windows"] = [window_row(w, r, json.loads(Path(path).read_text()), proving) for w, r, path in a.window]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "table.json").write_text(json.dumps(table, indent=1) + "\n")
    (a.out / "table.md").write_text(markdown(table))
    print(markdown(table), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
