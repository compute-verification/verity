"""GLM-5.3 as served (vLLM, H200, TP8, FP8): its proof-block census as the planner's candidates. The card
(`glm53_blocks.json`) is `benchmarks/one_stage/block_census.py`'s `card.json` over the Commit Build it names (``build``,
``commit``), the census itself in the store (``source``): per proof-block Definition its count in one request of the served
workload over every rank (``prefill``: the prefill step of ``prompt_len`` tokens; ``decode``: the ``decode_steps`` decode
steps), its output bits, its estimated ANDs (an upper bound: leaves lowered once, nothing shared or folded), its served MACs,
the bytes of the rows it opens and its block's log2 as Flock stages it; and, by row family, the rows no cut fits
(``gaps``: their Definitions, rows per phase, ANDs and why), which no candidate covers.

Each Definition is one candidate (`optimize.Candidate`), its own op type with the one cut the census took: its units the
request's count times ``requests``, its work its own ANDs (the planner credits Boolean work; MACs are informational), its
outputs committed, a statement at most `cap` of them (Flock's m = k_log + nbl <= 35, and at most 2^20), and its replay
grain the request, which holds one request's count of it (``n_rho``).

    uv run python benchmarks/sampling_plan/glm53.py [--card PATH] [--requests N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verity.protocols.spot.optimize import Candidate

CARD = Path(__file__).with_name("glm53_blocks.json")
QUANTITIES = ("blocks", "out_bits", "ands", "macs", "row_bytes")


def load(path: Path = CARD) -> dict:
    return json.loads(Path(path).read_text())


def cap(k_log: int) -> int:
    """The proof blocks one statement holds at a block of 2^k_log."""
    return min(1 << (35 - k_log), 1 << 20)


def candidates(card: dict, requests: int = 1) -> list[Candidate]:
    """One candidate per proof-block Definition over ``requests`` requests of the card's shape."""
    out = []
    for t, d in card["definitions"].items():
        per = d["prefill"] + d["decode"]
        out.append(Candidate(t, t, per * requests, d["ands"], d["out_bits"], out_gates=-(-d["out_bits"] // 16),
                             ands=d["ands"], rows=tuple(d["rows"]), commit_bits=d["out_bits"], cap=cap(d["k_log"]),
                             grain="request", n_rho=per))
    return out


def totals(card: dict) -> dict[str, dict[str, float]]:
    """Proof blocks, output bits, ANDs, served MACs and opened row bytes per prefill token, per decode token and per request."""
    def of(d: dict, n: int) -> dict[str, int]:
        return {"blocks": n, "out_bits": n * d["out_bits"], "ands": n * d["ands"], "macs": n * d["macs"],
                "row_bytes": n * sum(d["rows"])}

    ds = card["definitions"].values()
    phase = {p: {k: sum(of(d, d[p])[k] for d in ds) for k in QUANTITIES} for p in ("prefill", "decode")}
    return {"prefill token": {k: v / card["prompt_len"] for k, v in phase["prefill"].items()},
            "decode token": {k: v / card["decode_steps"] for k, v in phase["decode"].items()},
            "request": {k: phase["prefill"][k] + phase["decode"][k] for k in QUANTITIES}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=Path, default=CARD)
    ap.add_argument("--requests", type=int, default=1)
    a = ap.parse_args()
    card = load(a.card)
    cs = candidates(card, a.requests)
    print(f"{card['model']}: {len(cs)} proof-block Definitions over {a.requests} request(s) of {card['prompt_len']} prompt "
          f"tokens and {card['decode_steps']} decode steps; census {card['source']}")
    for row, q in totals(card).items():
        print(f"  {row:14s} " + "  ".join(f"{k} {v:.4g}" for k, v in q.items()))
    for fam, g in card["gaps"].items():
        print(f"  gap {fam}: {g['definitions']} Definitions, {g['prefill']:,} prefill and {g['decode']:,} decode rows, "
              f"{g['ands']:.4g} ANDs; {g['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
