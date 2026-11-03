"""The sealed side (fixed): what the search never sees, and its one output, a promotion bit per candidate.

    python3 sealed.py decide --in CANDIDATES.json --out BITS.json    # on the node that holds the candidates' values

`SEALED` (`$CW_SEALED`, default /workspace/research/te/sealed, which the run user owns on every node) holds the promotion
and final sets (`heldout/<DATA>/`, `data.py`), the sealed start (`starts/`, `prepare.register_pythia`, loadable only with
`CW_FINAL=1`) and the sealed values (`values/<DATA>/<output_sha256>.json`): for each model a search run's evaluation
scored, its search score and its start's, and its bpb on the promotion set and its start's (`prepare.Scorer.seal`),
written once with a salt. A candidate's training job can't open any of it (candidate.sh's sandbox mounts only the sealed
start, in final mode); any other process on the node can, so the rule that no search agent reads it is the brief's.

`decide` reads CANDIDATES, a list of `{"id": ..., "output_sha256": ...}` (a te_core ledger record's id and its
`output.output_sha256`), and writes BITS: for each id, `promote` (true, false, or null when this node holds no value for
that output) and the value record's sha256 (salted, so it hides the value and binds the bit to it). A candidate is
promoted iff its search gain (its start's search bpb less its own) is positive and its promotion gain (its start's
promotion bpb less its own) is at least PROMOTION_SHARE of it: a gain the search only fitted doesn't carry over to a
shard it never saw. Both gains come from the value record, so a caller chooses which outputs are decided and nothing
else, and each output's bit is the same every time it is asked. Each decision is appended to `decisions.jsonl` beside
the values. Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SEALED = Path(os.environ.get("CW_SEALED", "/workspace/research/te/sealed"))
#: the held-out data's version (`data_v3.py`; v2.2's was `data.py`'s): its sets are under `heldout/<DATA>/`, and the
#: values scored on them under `values/<DATA>/`
DATA = "v3"
VALUES = SEALED / "values" / DATA
#: the share of its search gain a candidate's promotion-set gain must reach
PROMOTION_SHARE = 0.5
KEY = re.compile(r"^[0-9a-f]{64}$")


def value_path(key: str) -> Path:
    if not KEY.match(key):
        raise ValueError(f"sealed: {key!r} is not an output digest (64 hex digits)")
    return VALUES / f"{key}.json"


def decide_one(key: str) -> dict:
    path = value_path(key)
    if not path.is_file():
        return {"promote": None, "value_sha256": None}
    raw = path.read_bytes()
    v = json.loads(raw)
    search_gain = v["start_search_bpb"] - v["search_bpb"]
    promotion_gain = v["start_promotion_bpb"] - v["promotion_bpb"]
    return {"promote": bool(search_gain > 0 and promotion_gain >= PROMOTION_SHARE * search_gain),
            "value_sha256": hashlib.sha256(raw).hexdigest()}


def decide(candidates: list[dict]) -> dict:
    out = {str(c["id"]): {"output_sha256": c["output_sha256"]} | decide_one(c["output_sha256"]) for c in candidates}
    VALUES.mkdir(parents=True, exist_ok=True)
    with open(SEALED / "decisions.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "run": os.environ.get("RESEARCH_RUN_ID", ""),
                            "decided": {k: [v["output_sha256"], v["promote"]] for k, v in out.items()}}) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--in", dest="inp", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    bits = decide(json.loads(a.inp.read_text()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(bits, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v["promote"] for k, v in sorted(bits.items())}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
