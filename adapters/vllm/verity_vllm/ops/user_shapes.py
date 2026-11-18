"""Counts stored runs' request users by the shape our load tools give a body's `user`, printing counts only, never a user.

    python3 user_shapes.py LIST    LIST: one path a line, each a run's request_users.jsonl (or requests.jsonl)
"""
import json
import re
import sys
from collections import Counter

SHAPES = {
    "varied.NNNNN (tools/served_run/serve_load.py:139; pipeline/commit_arb.py:231)": r"varied\.\d{5}(\.t\d+)?",
    "fixed.NNNNN (tools/served_run/serve_load.py:148; benchmarks/one_stage/rehearsal_window.py:57)": r"fixed\.\d{5}",
    "bK-I (tools/served_run/serve_load.py:234)": r"b\d+-\d+",
    "user-I (network/prover/scan_zk.py:395; tools/advice_run.py:72)": r"user-\d+",
    "req-NNNNN (benchmarks/network_e2e/traces/corpus.py:265, serve_tax's loads)": r"req-\d{5}",
    "warmup-TAG-I (benchmarks/network_traces/attester_piece.py:300)": r"warmup-.+-\d+",
    "NAME-repR-I (benchmarks/attester_load/run.py:83,280)": r".+-rep\d+-\d+",
    "qI (benchmarks/compile_cost/run.py:100)": r"q\d+",
}
COMPILED = [(name, re.compile(rx)) for name, rx in SHAPES.items()]


def shape(user) -> str:
    if user is None:
        return "none sent"
    return next((name for name, rx in COMPILED if rx.fullmatch(str(user))), "matches no shape")


def main(listing: str) -> int:
    total, files, unreadable = Counter(), 0, 0
    for path in filter(None, map(str.strip, open(listing))):
        try:
            lines = [json.loads(x) for x in open(path) if x.strip()]
        except (OSError, ValueError):
            unreadable += 1
            continue
        files += 1
        total.update(shape(r.get("user")) for r in lines)
    print(json.dumps({"files": files, "unreadable": unreadable, "by_shape": dict(total.most_common())}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
