"""The NCI slowdown bracket for a board row: a proven lower bound and the cost of an explicit compliant construction.

A policy is a calibration: blocks of at most ``Gamma`` gates, each reading at most ``K`` bits besides the committed
parameters, so ``alpha = K / Gamma`` bits per gate. A row (one training method in one configuration) gives the inputs
in ``INPUTS``, each a number or a parameter estimate (``catalog.resolve``).

* ``lower``: the thread theorem, ``Nci.SecurityProofs.ThreadedTraining`` (``Security/Models/Nci/Guarantees/Threads.lean``):
  ``rho >= kappa * b_a / (2 alpha Lambda)`` with ``Lambda = max(e, c K / (4 d b_w))``, and ``rho >= 1`` since a
  containing circuit is no smaller. The weight rate is capped at ``U_bits / n_updated``: a block that uses ``p`` updated
  weights can always read the whole update instead, so no larger rate is a valid charge. A seed-described update
  therefore gets a lower bound near 1, which is the truth about it, not an artifact.
* ``upper``: the cost of the cheapest construction ``search`` finds, compliant by construction under its own
  accounting: a block covers a slice of every token's computation, reads its slice's state, the activations crossing
  into it and the tokens' data at ``b_tok`` bits each, and is padded to ``Gamma``. It reports ``fit`` (one block covers
  the whole computation), ``shard`` (slices, the state split among them) or ``decode-shard`` (slices, each reading a
  whole description such as seeds and decoding its own slice). ``shard`` is the closed form of the best split with no
  data and no decoding, kept as a check.
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


def _phis() -> list[float]:
    return [10 ** (-12 + 12 * k / 2400) for k in range(2400)] + [1.0]


def search(U: float, splits: bool, flops: float, w: float, t: float, decode: float, K: float, Gamma: float) -> dict:
    """The cheapest construction in one family. A block covers a slice, a fraction ``phi`` of every token's computation,
    for ``T`` tokens. It reads its slice's state (``phi U``; all of ``U`` when the state is a description that does not
    split, such as seeds), the ``w`` bits per token crossing into the slice (none at ``phi = 1``), and ``t`` bits of
    novel data per token. It does ``phi flops`` of reference work per token and decodes its slice (``phi decode``).
    Padded to ``Gamma``, its cost is ``Gamma`` over its reference work, minimized over ``phi`` and ``T`` under the caps.
    At ``phi = 1`` this is the fit construction; at the best ``phi`` with ``t = 0`` and no decoding it is ``shard``. One
    sequential chain of blocks is assumed, so no gradient aggregation is charged and wall-clock time is ignored."""
    best = None
    for phi in _phis():
        state = phi * U if splits else U
        dec = phi * decode
        if state >= K or dec >= Gamma:
            continue
        w_in = w if phi < 1 else 0.0
        per_token = w_in + t
        tokens = min((K - state) / per_token if per_token > 0 else math.inf, (Gamma - dec) / (phi * flops))
        rho = Gamma / (phi * flops * tokens)
        if best is None or rho < best["rho"]:
            best = {"rho": rho, "phi": phi, "tokens": tokens,
                    "reads": {"state": state, "activations": tokens * w_in, "data": tokens * t},
                    "work": {"reference": phi * flops * tokens, "decode": dec}}
    if best is None:
        raise ValueError("no compliant construction in the family: the state or its decoding exceeds a block")
    best["rho"] = max(1.0, best["rho"])
    best["basis"] = "fit" if best["phi"] == 1.0 else ("shard" if splits else "decode-shard")
    return best


def construction_upper(i: dict, K: float, Gamma: float, b_tok: float, splits: bool = True) -> tuple[float, str]:
    c = search(i["U_token_bits"], splits, i["flops_per_token"], i["crossings"] * i["w_cut"], b_tok, i["decode_flops"],
               K, Gamma)
    return c["rho"], c["basis"]


def bracket(row: dict, calibration: dict, parameters: list[dict]) -> dict:
    i = {name: catalog.resolve(ref, parameters) for name, ref in row["inputs"].items()}
    if not calibration["theta_cap"] and "decode_flops_table" in row:
        i["decode_flops"] = min(i["decode_flops"], row["decode_flops_table"])
    K, Gamma = calibration["K"], calibration["Gamma"]
    b_tok = calibration["b_tok"]
    upper, basis = construction_upper(i, K, Gamma, b_tok, row.get("state_splits", True))
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
