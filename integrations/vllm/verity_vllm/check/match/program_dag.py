"""`verity-vllm program-dag --a <derived dir> --b <fold dir> --out dag.json`: the within-step criterion of record over a derived
Program and a fold, both held compact: their live-cone DAG multisets are equal (`global_match_fast.dag_multiset`).  tp-fold-match's
compare leg runs it under `--within-step dag` when the sequence differs, as X-09 does for a request whose step order differs.
Exit 0 equal, 1 not."""
from __future__ import annotations

import json
from dataclasses import dataclass

from verity_vllm.check.match import global_match_fast as GMF
from verity_vllm.check.match import program_loaders as PL
from verity_vllm.config import option


@dataclass(frozen=True, kw_only=True)
class Options:
    a: str = option("--a", help="derived dir (descriptor.json.gz + derive-report.json)")
    b: str = option("--b", help="fold dir (program.json + instances.jsonl)")
    out: str = option("--out")


def main(a: Options) -> int:
    res = GMF.dag_multiset(PL.load_derived(a.a, "derived", compact=True), PL.load_oracle(a.b, "oracle", compact=True))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
    return 0 if res["equal"] else 1
