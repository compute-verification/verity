"""The v1 final table's score: te_core.rescore, unchanged, on the scorer's ledger with each checkpoint's final value.

    python v1final.py FINAL_DIR [te_core.rescore's arguments]

Run from benchmarks/covert_witness of te_core's checkout (v1cycle's RESCORE_WT). FINAL_DIR is `scorer.py final table`'s
directory: final_submissions.json (each FINAL entry's reruns on the sealed shard: per checkpoint the seeds' mean bpb,
their range and their mean MACs) and table.json (the final and the search starts' bpb). te_core.rescore scores the
ledger's rows (`final_rows`) with these values in place of the search's:

- a FINAL entry's checkpoint takes the rerun row at the nearest MACs (within MACS_MATCH): val_bpb the seeds' mean,
  bpb_min and bpb_max their range; its gates and every other quantity stay the search checkpoint's (rescore_v1's
  block, joined by run and MACs). A checkpoint of an entry with no rerun row near it is left out;
- every other row's val_bpb is carried to the final start at the same drop (val_bpb x final l0 / search l0);
- every row's start_bpb is the final start's, and a row of a start with no final bpb is left out.

Writes te_core.rescore's outputs into its --out, and final_rows.json there: each final row's search value, range and
seeds by ledger id, and per entry its matched and unmatched checkpoints.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from te_core import rescore
from te_core.ledger import Ledger

MACS_MATCH = 0.02


def search_runs(state: Path) -> dict[str, str]:
    """The scorer's record id (an art id) -> its search run id, from the launch records."""
    out = {}
    for line in (state / "index.jsonl").open():
        x = json.loads(line)
        if x.get("source") == "launch.py" and x.get("run"):
            out[x["id"]] = x["run"]
    return out


def final_rows(rows: list[dict], fdir: Path, state: Path) -> tuple[list[dict], dict]:
    """The rows with their final values (module docstring), and what final_rows.json records."""
    fin = json.loads((fdir / "final_submissions.json").read_text())
    tab = json.loads((fdir / "table.json").read_text())
    l0, l0s = tab["final_starts"], tab["search_starts"]
    run_of = search_runs(state)
    reruns: dict[str, list] = defaultdict(list)
    entry_of = {}
    for x in fin:
        run = run_of.get(x["id"])
        reruns[run].append(x)
        entry_of[run] = x["id"]
    out, by_id = [], {}
    entries = {e: {"run": run, "matched": 0, "unmatched": 0, "rerun_rows": len(reruns[run])}
               for run, e in entry_of.items()}
    entries |= {x["id"]: {"run": None, "matched": 0, "unmatched": 0, "rerun_rows": len(xs)}
                for run, xs in reruns.items() if run is None for x in xs}
    for r in rows:
        s, v = r.get("start"), r.get("val_bpb")
        if s not in l0 or s not in l0s:
            continue
        xs = reruns.get(r.get("run_id")) if r.get("run_id") else None
        if xs:
            x = min(xs, key=lambda x: abs(x["macs"] - r["macs"]))
            e = entries[entry_of[r["run_id"]]]
            if abs(x["macs"] - r["macs"]) > MACS_MATCH * max(x["macs"], r["macs"]):
                e["unmatched"] += 1
                continue
            e["matched"] += 1
            by_id[r["id"]] = {"entry": entry_of[r["run_id"]], "split": "final", "search_bpb": v,
                              "final_mean_bpb": x["val_bpb"], "bpb_min": x["bpb_min"], "bpb_max": x["bpb_max"],
                              "seeds": x["seeds"], "final_macs": x["macs"]}
            out.append(r | {"val_bpb": x["val_bpb"], "start_bpb": l0[s]} | by_id[r["id"]])
        else:
            out.append(r | {"val_bpb": v * l0[s] / l0s[s] if v is not None else None, "start_bpb": l0[s],
                            "split": "search", "search_bpb": v})
    return out, {"final_starts": l0, "search_starts": l0s, "rows": by_id, "entries": entries}


def main(argv: list[str]) -> int:
    fdir, args = Path(argv[0]), argv[1:]
    out = Path(args[args.index("--out") + 1])
    state = Path(args[args.index("--scorer") + 1]) if "--scorer" in args else rescore.SCORER
    record = {}

    class FinalLedger(Ledger):
        def index(self) -> list[dict]:
            rows, rec = final_rows(super().index(), fdir, state)
            record.update(rec)
            return rows

    rescore.Ledger = FinalLedger
    rc = rescore.main(args)
    (out / "final_rows.json").write_text(json.dumps(record, indent=1) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
