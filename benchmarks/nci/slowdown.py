"""The NCI slowdown bracket for a board row: a proven lower bound and the cost of an explicit compliant construction.

A policy is a calibration: blocks of at most ``Gamma`` gates, each reading at most ``K`` bits besides the committed
parameters, so ``alpha = K / Gamma`` bits per gate. A row (one training method in one configuration) gives the inputs
in ``INPUTS``, each a number or a parameter estimate (``catalog.resolve``).

* ``lower``: the thread theorem, ``Nci.SecurityProofs.ThreadedTraining`` (``Security/Models/Nci/Guarantees/Threads.lean``):
  ``rho >= kappa * b_a / (2 alpha Lambda)`` with ``Lambda = max(e, c K / (4 d b_w))``, and ``rho >= 1`` since a
  containing circuit is no smaller. The weight rate is capped at ``U_bits / n_updated``: a block that uses ``p`` updated
  weights can always read the whole update instead, so no larger rate is a valid charge. A seed-described update
  therefore gets a lower bound near 1, which is the truth about it, not an artifact.
* ``upper``: the cheapest applicable construction, each compliant by construction under its own accounting. Every
  block is padded to ``Gamma`` gates, so a construction's cost is ``Gamma`` over the reference work a block does.
  ``fit``: the state one token touches fits in half a block's reads and its decoding in half a block's work. Each block
  reads that state and spends the rest of its reads on tokens at ``b_tok`` bits, doing ``flops_per_token`` each:
  ``rho = max(1, Gamma b_tok / (flops_per_token (K - U_token))) (1 + U_token / K) + decode / Gamma``, where the
  ``U_token / K`` term (training rows, ``c >= 2``) pays for blocks that sum the gradients, ``K / U_token`` per block.
  ``decode-shard``: the description fits but decoding it does not. A block reads the whole description, decodes the
  slice ``phi = Gamma / (2 decode)`` of the model it covers, and takes ``crossings * w_cut`` bits per token from the
  previous slice: ``rho = Gamma / (phi flops_per_token T)`` with ``T = (K - U_token) / (crossings w_cut)``.
  ``shard``: the state does not fit. Each shard holds ``U_s`` bits of it and receives ``crossings * w_cut`` bits per
  token; at the best split (``U_s = K / 2``) a block's intensity is ``4 crossings w_cut U_token / (K flops_per_token)``,
  padded to ``alpha``. It assumes updated state spread evenly over a token's compute, and that a block's work cap
  never binds (``shard`` checks it).
* ``theta_cap``: a calibration without a cap on the committed parameters' size lets a row decode from a committed table
  (its ``decode_flops_table``) instead of regenerating it, as a seed-based method's noise.
* ``chain``: the earlier chain bound ``kappa * F`` (``Nci.Guarantees.UpdatedMatMul``), for comparison. The lock marks it
  as resting on wiring-form incompressibility of the declared circuit, which Daniel ruled false.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import catalog  # noqa: E402

INPUTS = {
    "kappa": "share of the reference circuit's work in multiplications of an updated weight by an activation",
    "b_w": "bits a block must read per updated weight it uses (a rate)",
    "b_a": "bits a block must read per activation it uses (a rate)",
    "d": "entry fee: activations of a thread a block reads before it can finish a product (inner dimension, or rank)",
    "e": "updated weights one activation multiplies per thread",
    "c": "uses of each updated weight per thread (1 forward only, 2 forward and backward, 3 with recomputation)",
    "n_updated": "number of updated weights",
    "U_bits": "description length of the whole update, bits",
    "U_token_bits": "bits of updated state one token's computation touches",
    "decode_flops": "work to decode the update's description inside a block (0 if stored plainly)",
    "w_cut": "bits per token crossing a shard boundary (residual width times b_a for a standard transformer)",
    "crossings": "boundary crossings per token per shard (1 forward only, 2 with a backward pass)",
    "flops_per_token": "the reference circuit's work per token",
}

THREAD_THEOREM = "Nci.SecurityProofs.ThreadedTraining"
CHAIN_THEOREM = "Nci.SecurityProofs.UpdatedMatMul"


def rates(i: dict, K: float, b_tok: float) -> tuple[float, float, float, float]:
    """The parameters the theorem may use, ``(b_w, b_a, d, e)``, capped where a row's own inputs make its assumptions
    false: no more per updated weight than the whole update spread over them; no more per thread value than a prompt
    token costs, since a token id is a thread value too, or than a boundary crossing costs per value, ``w_cut / d``,
    since a block may take the ``d`` values it needs from a compressed crossing; and when one block can hold all the
    updated state a token touches, it computes that token's every counted multiplication from the token alone, so the
    only valid threading is ``d = 1`` with ``e`` at least the counted multiplications per token,
    ``kappa flops_per_token / 2``."""
    b_w = min(i["b_w"], i["U_bits"] / i["n_updated"]) if i["n_updated"] > 0 else i["b_w"]
    b_a = min(i["b_a"], b_tok, i["w_cut"] / i["d"])
    d, e = i["d"], i["e"]
    if i["U_token_bits"] <= K:
        d, e = 1.0, max(e, i["kappa"] * i["flops_per_token"] / 2)
    return b_w, b_a, d, e


def thread_lower(i: dict, K: float, Gamma: float, b_tok: float) -> float:
    alpha = K / Gamma
    b_w, b_a, d, e = rates(i, K, b_tok)
    lam = max(e, i["c"] * K / (4 * d * b_w))
    return max(1.0, i["kappa"] * b_a / (2 * alpha * lam))


def read_factor(K: float, b_w: float, b_a: float, d: float, steps: int = 4000) -> float:
    """The chain draft's ``f(K) = max_{0<mu<1} min{(1+sqrt mu)^2 b_w b_a d / K, sqrt2 (1+sqrt mu) sqrt(1-mu) b_a sqrt(b_w/K)}``
    (``Nci.readFactor``), maximized on a grid of ``steps`` points."""
    best = 0.0
    for k in range(1, steps):
        mu = k / steps
        a = (1 + math.sqrt(mu)) ** 2 * b_w * b_a * d / K
        b = math.sqrt(2) * (1 + math.sqrt(mu)) * math.sqrt(1 - mu) * b_a * math.sqrt(b_w / K)
        best = max(best, min(a, b))
    return best


def chain_conditional(i: dict, K: float, Gamma: float, b_tok: float) -> float:
    alpha = K / Gamma
    b_w, b_a, d, _ = rates(i, K, b_tok)
    return max(1.0, i["kappa"] * read_factor(K, b_w, b_a, d) / (2 * alpha))


def shard(i: dict, K: float, Gamma: float) -> float:
    U = i["U_token_bits"]
    nci = 4 * i["crossings"] * i["w_cut"] * U / (K * i["flops_per_token"])
    tokens = K / (2 * i["crossings"] * i["w_cut"])
    work = (K / 2) / U * i["flops_per_token"] * tokens
    if work > Gamma:
        raise ValueError(f"a shard's work {work:.3g} exceeds Gamma {Gamma:.3g}: the construction needs fewer tokens")
    return max(1.0, nci * Gamma / K)


def fit(i: dict, K: float, Gamma: float, b_tok: float) -> float:
    U = i["U_token_bits"]
    tokens = (K - U) / b_tok
    aggregate = 1 + (U / K if i["c"] >= 2 else 0.0)
    return max(1.0, Gamma / (i["flops_per_token"] * tokens)) * aggregate + i["decode_flops"] / Gamma


def decode_shard(i: dict, K: float, Gamma: float) -> float:
    phi = Gamma / (2 * i["decode_flops"])
    tokens = min((K - i["U_token_bits"]) / (i["crossings"] * i["w_cut"]), Gamma / (2 * phi * i["flops_per_token"]))
    return max(1.0, Gamma / (phi * i["flops_per_token"] * tokens))


def construction_upper(i: dict, K: float, Gamma: float, b_tok: float) -> tuple[float, str]:
    if i["U_token_bits"] > K / 2:
        return shard(i, K, Gamma), "shard"
    if i["decode_flops"] > Gamma / 2:
        return decode_shard(i, K, Gamma), "decode-shard"
    return fit(i, K, Gamma, b_tok), "fit" if i["U_bits"] <= K / 2 else "fit-per-token"


def bracket(row: dict, calibration: dict, parameters: list[dict]) -> dict:
    i = {name: catalog.resolve(ref, parameters) for name, ref in row["inputs"].items()}
    if not calibration["theta_cap"] and "decode_flops_table" in row:
        i["decode_flops"] = min(i["decode_flops"], row["decode_flops_table"])
    K, Gamma = calibration["K"], calibration["Gamma"]
    b_tok = calibration["b_tok"]
    upper, basis = construction_upper(i, K, Gamma, b_tok)
    return {"row": row["id"], "calibration": calibration["id"], "lower": thread_lower(i, K, Gamma, b_tok),
            "lower_basis": THREAD_THEOREM, "upper": upper, "upper_basis": basis,
            "chain": chain_conditional(i, K, Gamma, b_tok), "inputs": i}


def brackets(rows: list[dict], calibrations: list[dict], parameters: list[dict]) -> list[dict]:
    """Every row under every calibration of its model."""
    return [bracket(r, c, parameters) for c in calibrations for r in rows if r["model"] == c["model"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibration", help="a calibrations.json id (default: every calibration)")
    ap.add_argument("--row", action="append", help="a rows.json id (repeatable; default: every row)")
    args = ap.parse_args(argv)
    parameters, rows = catalog.load("parameters"), catalog.load("rows")
    cals = [c for c in catalog.load("calibrations") if args.calibration in (None, c["id"])]
    rows = [r for r in rows if not args.row or r["id"] in args.row]
    print(f"{'row':44} {'calibration':22} {'lower':>9} {'upper':>9} {'basis':>14} {'chain*':>9}")
    for b in brackets(rows, cals, parameters):
        print(f"{b['row']:44} {b['calibration']:22} {b['lower']:9.3g} {b['upper']:9.3g} {b['upper_basis']:>14} "
              f"{b['chain']:9.3g}")
    print("* chain: the earlier bound, resting on an assumption the lock marks falsified; for comparison only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
