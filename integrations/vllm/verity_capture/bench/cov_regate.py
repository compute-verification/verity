"""[cov-dense 2026-09-17, lane-local] Re-run the two environment-only CPU stages of `run_config` over a finished RUN_DIR:

    golden   `python -m verity_capture.golden --check --out RUN_DIR/golden.json`   (the driver's exact command)
    gates    the in-process `run_config.gates_stage` step (gates.json, card.json, CARD.md, <cards-dir>/<config_id>.json)

Why: the pod checkout shipped by `record_v5/ship.sh` has no `data/logs/`, so `golden --check` reported both corpus
entries "skipped: log not present in this checkout" and G6 failed on every cell for a reason unrelated to the run.  The
golden result is a function of (code, corpus logs) only -- not of the run -- so after shipping the two logs it is re-run
alone and the gates are re-evaluated over the unchanged artifacts.  `stages.json` keeps its records; the `golden` and `gates`
records are updated in place with the new status / seconds / verdict and a `rerun` note saying when and why, so the evidence
trail shows exactly what was redone.  Nothing else in RUN_DIR is touched.

    python -m verity_capture.bench.cov_regate --run-dir RUN_DIR [--cards-dir out/gen/cards] [--why "..."]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cards-dir", default="out/gen/cards")
    ap.add_argument("--why", default="golden corpus logs (data/logs) shipped to the pod after the CPU phase")
    ap.add_argument("--python", default=sys.executable, help="the torch-free python for the golden check")
    a = ap.parse_args(argv)

    from .. import card as CARD
    from .. import gates as G

    run = Path(a.run_dir)
    man = json.loads((run / "run.json").read_text())
    config_id = man["config_id"]
    stages_path = run / "stages.json"
    st = json.loads(stages_path.read_text())
    recs = st["stages"] if isinstance(st, dict) else st
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    note = f"{utc} re-run by cov_regate: {a.why}"

    # --- golden (the driver's command line, appended to the same stage log) --------------------------------------------
    cmd = [a.python, "-m", "verity_capture.golden", "--check", "--out", str(run / "golden.json")]
    log = run / "logs" / "golden.log"
    t0 = time.time()
    with log.open("a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}    # {note}\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    gsec = round(time.time() - t0, 3)
    print(f"[golden] {'ok' if rc == 0 else 'failed'} in {gsec} s", file=sys.stderr)

    # --- gates (run_config.gates_stage, verbatim) -------------------------------------------------------------------------
    t0 = time.time()
    results = G.evaluate_gates(run)
    diag = G.diagnostics(run)
    G.write_gates(run, results, diag)
    card = CARD.build_card(run, results, config_id=config_id)
    CARD.write_card(card, run / G.ARTIFACTS["card"])
    (run / "CARD.md").write_text(CARD.summarize_md(card))
    dest = Path(a.cards_dir) / f"{config_id}.json"
    CARD.write_card(card, dest)
    print(CARD.top_section(card))
    print(G.gates_markdown(results))
    gates_rec = {"stage": "gates", "status": "ok", "returncode": 0, "seconds": round(time.time() - t0, 3), "log": None, "cmd": "in-process",
                 "reason": None, "verdict": card.verdict, "card": str(dest)}

    # --- stages.json: update the two records in place, keep everything else -------------------------------------------
    for r in recs:
        if r.get("stage") == "golden":
            r.update(status="ok" if rc == 0 else "failed", returncode=rc, seconds=gsec, cmd=" ".join(cmd), rerun=note)
        elif r.get("stage") == "gates":
            r.update(**gates_rec, rerun=note)
    stages_path.write_text(json.dumps(st, indent=1))
    print(f"[cov_regate] {run.name}: golden {'ok' if rc == 0 else 'failed'}, verdict {card.verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
