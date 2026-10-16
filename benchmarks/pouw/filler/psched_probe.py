"""Summarize psched_probe.sh's rows (dscore/pscore) with the locked scoring.point_stats and the scorer's per-entry coverage
for a decode point or the prefill step: `python3 benchmarks/pouw/filler/psched_probe.py OUT_DIR ...`."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as S


def main(dirs):
    print("| run | kind | B | L | place | step/fill SMs | phase | slowdown (rounds max) | coverage | filler share | beside/alone "
          "| receipts (bad) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        p = Path(d) / "probe.jsonl"
        for ln in p.read_text().splitlines() if p.is_file() else []:
            if not ln.startswith("{"):
                continue
            r = json.loads(ln)
            if r.get("mode") != "dscore":
                continue
            st = S.point_stats(r)
            e = S.coverage(S.prefill_entry(st) if r["kind"] == "prefill" else S.decode_entry(st, r["B"], r["L"]))
            ph = r.get("phase")
            phs = f"W{ph['W']} NC{ph['NC']} S{ph['S']} t{ph['tiles']}" if ph else "–"
            bo = st["beside_over_alone"]
            print(f"| {Path(d).parent.name[-4:]} | {r['kind']} | {r['B']} | {r['L']} | {r['place']} | "
                  f"{r['step_sms']}/{r['fill_sms']} | {phs} | {st['slowdown']:.4f} ({max(st['slowdown_rounds']):.4f}) | "
                  f"{e['coverage']:.3f} | {e['filler_share']:.3f} | {'–' if bo is None else f'{bo:.3f}'} | "
                  f"{st['rec_checked']} ({st['rec_bad']}) |")


if __name__ == "__main__":
    main(sys.argv[1:])
