"""GLM-5.3 as served (vLLM, H200, TP8, FP8): its proof-block census as the planner's candidates. The card
(`glm53_blocks.json`) is `benchmarks/one_stage/block_census.py`'s `card.json` over the Commit Build it names (``build``,
``commit``), the census itself in the store (``source``): per proof-block Definition its count in one request of the served
workload over every rank (``prefill``: the prefill step of ``prompt_len`` tokens; ``decode``: the ``decode_steps`` decode
steps), its output bits, its estimated ANDs (leaves lowered once, table reads counted apart: not a statement's ``ands``
with its reads inline, as `m1_families` says), its served MACs, the bytes of the rows it opens and its block's log2 as
Flock stages it; and, by row family, the rows no cut fits (``gaps``: their Definitions, rows per phase, ANDs and why),
which no candidate covers.

Each Definition is one candidate (`optimize.Candidate`), its own op type with the one cut the census took: its units the
request's count times ``requests``, its work its own ANDs (the planner credits Boolean work; MACs are informational), its
outputs committed, a statement at most `cap` of them (Flock's m = k_log + nbl <= 35, and at most 2^20), and its replay
grain the request, which holds one request's count of it (``n_rho``).

**A stop class's two populations** (`response_blocks`). Under a registered stop rule a response of a stop class adds two
proof blocks beside the card's, one of each a response whatever its length. Its stop block computes where the response
ends and why from its T committed ids: here the rule's Boolean Definition, `StopRule16_v2{T, NSTOP}` at the registered
model's stop ids, taken as a block of its own. That is the leaf holding every gate of the served block but its
``ignore_eos`` choice, and its rows are the rule's five parameters and its word, six rows, where the served block opens
three: its ids, one request row for two of the parameters, and its word, the stop ids and the model's length being its
statics. Its response tie (`ResponseTieStop_v1{O}`, O = T) holds the response row the network certifier released to
those ids and that stop word. Each is a candidate as the card's are, with its ANDs, its rows and its output bits read
from its Definition, and a statement holding as many of it as its block allows, as Flock's composer sizes that block.

Both populations are registered at ``rate:1/1``, every block proved. `charged` says what a sample of a population
leaves: the most of its blocks an accepted sample leaves incorrect at δ (the integrity profile's count bound, in exact
arithmetic), that count over the population (ε), and those blocks' output bits, 16 a stop word. Where every block is
sampled no incorrect block escapes, so all three are 0 at every δ; a smaller sample is charged what its bound gives.
These rows are the plan's: they read no verdict, and say what a sampling design charges, nothing of a served class.

    uv run python benchmarks/sampling_plan/glm53.py [--card PATH] [--requests N] [--stop T [--delta D]]
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from verity.circuits.defs import bind
from verity.circuits.evaluation import bits
from verity.commitments.windows.interface import proof_block_k_log
from verity.definitions.registry.boolean_stop_rule import StopRule16V2
from verity.protocols.accounting.certifier.prover import response_tie
from verity.protocols.proofs.spot.prover.optimize import Candidate
from verity.protocols.proofs.spot.verifier.profile import WINDOW_DELTA, Stratified, Stratum

CARD = Path(__file__).with_name("glm53_blocks.json")
QUANTITIES = ("blocks", "out_bits", "ands", "macs", "row_bytes")
#: GLM-5.3's stop ids, which a stop block reads beside its response's ids
STOP_IDS = 3


def load(path: Path = CARD) -> dict:
    return json.loads(Path(path).read_text())


def cap(k_log: int) -> int:
    """The proof blocks one statement holds at a block of 2^k_log."""
    return min(1 << (35 - k_log), 1 << 20)


def _candidate(t: str, units: int, per: int, ands: int, out_bits: int, rows: Sequence[int], k_log: int) -> Candidate:
    """The proof-block Definition ``t`` as a candidate (the module docstring): ``units`` of it, ``per`` to a request."""
    return Candidate(t, t, units, ands, out_bits, out_gates=-(-out_bits // 16), ands=ands, rows=tuple(rows),
                     commit_bits=out_bits, cap=cap(k_log), grain="request", n_rho=per)


def candidates(card: dict, requests: int = 1) -> list[Candidate]:
    """One candidate per proof-block Definition over ``requests`` requests of the card's shape."""
    out = []
    for t, d in card["definitions"].items():
        per = d["prefill"] + d["decode"]
        out.append(_candidate(t, per * requests, per, d["ands"], d["out_bits"], d["rows"], d["k_log"]))
    return out


def response_blocks(outputs: int, responses: int = 1, stop_ids: int = STOP_IDS) -> list[Candidate]:
    """A stop class's two populations over ``responses`` of its responses, each of at most ``outputs`` ids: the stop
    block and the response tie, one a response, each a candidate whose ANDs are its Definition's Boolean circuit's, whose
    rows are a row a parameter and a row an output, and whose replay grain is the request, which holds one."""
    out = []
    for d in (bind(StopRule16V2, T=outputs, NSTOP=stop_ids), response_tie.block(response_tie.Shape(outputs, True))):
        params, ret = d.signature()
        ins = [-(-sum(t.leaf_widths()) // 8) for _name, t in params]
        outs = [sum(t.leaf_widths()) for t in getattr(ret, "fields", (ret,))]
        rows, ands = (*ins, *(2 * -(-b // 16) for b in outs)), bits.compile(d).counts["and"]
        k_log = proof_block_k_log(ins, rows[len(ins):], ands, packed=False)
        out.append(_candidate(d.id, responses, 1, ands, sum(outs), rows, k_log))
    return out


def charged(c: Candidate, delta: float, sampled: int | None = None) -> dict:
    """What a sample of ``sampled`` of a population's proof blocks, uniform without replacement, leaves at ``delta``;
    every block by default, ``rate:1/1``. ``incorrect``: the most of its blocks an accepted sample leaves incorrect
    except with probability delta (`Stratified.count_bound`); ``eps``: that count over the population; ``bits``: those
    blocks' output bits."""
    k = c.units if sampled is None else sampled
    w = Stratified((Stratum(c.template, c.units, k),)).count_bound(delta)
    return {"population": c.units, "sampled": k, "rate": "{}/{}".format(*Fraction(k, c.units).as_integer_ratio()),
            "incorrect": w, "eps": w / c.units, "bits": w * c.out_bits}


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
    ap.add_argument("--stop", type=int, metavar="T", help="the requests are a stop class's, of at most T ids a response: "
                    "print its stop blocks and its response ties, and what proving every one leaves charged")
    ap.add_argument("--delta", type=float, default=WINDOW_DELTA, help="the delta the charges are read at")
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
    for c in response_blocks(a.stop, a.requests) if a.stop else ():
        x = charged(c, a.delta)
        print(f"  stop class {c.template}: {c.units:,} proof block(s), one a response, {c.ands:,} ANDs and {c.out_bits:,} "
              f"output bits each, {sum(c.rows):,} row bytes opened; rate:{x['rate']} leaves at most {x['incorrect']:,} "
              f"incorrect at delta {a.delta:.3g} (eps {x['eps']:.3g}), {x['bits']:,} bits charged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
