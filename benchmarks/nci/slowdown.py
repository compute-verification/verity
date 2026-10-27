"""The NCI slowdown bracket for a board row: a proven lower bound and the cost of an explicit compliant construction.

A policy is a calibration: blocks of at most ``Gamma`` gates, each reading at most ``K`` bits besides the committed
parameters, so ``alpha = K / Gamma`` bits per gate. A row (one training method in one configuration) gives the inputs
in ``INPUTS``, each a number or a parameter estimate (``catalog.resolve``).

* ``lower``: the thread theorem, ``Nci.SecurityProofs.ThreadedTraining`` (``Security/Definitions/Nci/Guarantees/Threads.lean``):
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
* ``chain``: the partial-sum law, ``Nci.SecurityProofs.ContainedTraining``
  (``Security/Definitions/Nci/Guarantees/Contained.lean``): ``rho >= kappa_chain * f(K) / (2 alpha)``, with ``f`` the read
  factor of ``b_w``, ``b_a`` and the chain's smallest inner dimension. Its assumptions are the support-form
  incompressibility of one chain of the reference circuit at ``b_w`` and ``b_a``, and nothing structural. A chain charges
  each weight once per layer of it, so it is one forward path: ``kappa_chain`` (a row's optional ``chain``) is that
  path's share of the work, below ``kappa``. Rows without a ``chain`` get none.
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

CHAIN = {
    "kappa": "share of the reference circuit's work in one chain's products, 2 flops each: a forward path of updated "
             "matrix multiplications linked by injective maps",
    "d": "the chain's smallest inner dimension",
    "path": "the multiplications the chain runs through, per layer",
}

THREAD_THEOREM = "Nci.SecurityProofs.ThreadedTraining"
CHAIN_THEOREM = "Nci.SecurityProofs.ContainedTraining"


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


def chain_lower(i: dict, chain: dict, K: float, Gamma: float, b_tok: float, d_model: float) -> float:
    """The partial-sum law at ``rates``'s weight rate, the activation rate capped by a token and by a boundary crossing
    spread over the residual's ``d_model`` entries, since every activation the chain charges is computed from one."""
    alpha = K / Gamma
    b_w = rates(i, K, b_tok)[0]
    b_a = min(i["b_a"], b_tok, i["w_cut"] / d_model)
    return max(1.0, chain["kappa"] * read_factor(K, b_w, b_a, chain["d"]) / (2 * alpha))


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
    phis = _phis() + ([K / (2 * U)] if splits and U > K / 2 else [])
    for phi in phis:
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
    d_model = next(m["d_model"] for m in catalog.load("models") if m["id"] == row["model"])
    chain = chain_lower(i, row["chain"], K, Gamma, b_tok, d_model) if "chain" in row else None
    return {"row": row["id"], "calibration": calibration["id"], "lower": thread_lower(i, K, Gamma, b_tok),
            "lower_basis": THREAD_THEOREM, "upper": upper, "upper_basis": basis,
            "chain": chain, "chain_basis": CHAIN_THEOREM, "inputs": i}


def sweep(row: dict, calibration: dict, parameters: list[dict], b_ws: list[float], b_as: list[float]) -> list[dict]:
    """The bracket at each weight and activation rate: how the guarantee and the construction move with how
    compressible training is. The state's size scales with ``b_w`` and the boundary's width with ``b_a``."""
    base = {name: catalog.resolve(ref, parameters) for name, ref in row["inputs"].items()}
    out = []
    for b_w in b_ws:
        for b_a in b_as:
            w, a = b_w / base["b_w"], b_a / base["b_a"]
            inputs = base | {"b_w": b_w, "b_a": b_a, "U_bits": base["U_bits"] * w,
                             "U_token_bits": base["U_token_bits"] * w, "w_cut": base["w_cut"] * a}
            b = bracket(row | {"inputs": inputs}, calibration, parameters)
            out.append({"b_w": b_w, "b_a": b_a, "thread": b["lower"], "chain": b["chain"], "upper": b["upper"]})
    return out


#: training recipes at the rates their matmuls read (weights, activations), each with the compute multiplier of
#: training that way (``train.compute_multiplier``); BF16 is the reference, m = 1
RECIPES = [("bf16", 16, 16, None), ("fp8", 8.25, 8.25, "train.compute_multiplier/w8a8"),
           ("w4-weights", 4.03, 16, "train.compute_multiplier/w4-weights"),
           ("nvfp4", 4.03, 4.5, "train.compute_multiplier/w4a4"),
           ("ternary-weights", 1.58, 8.25, "train.compute_multiplier/ternary-weights"),
           ("w2a2", 2, 2, "train.compute_multiplier/w2a2"), ("w1a1", 1, 1, "train.compute_multiplier/w1a1")]


def efficiency(row: dict, calibration: dict, parameters: list[dict]) -> list[dict]:
    """Pretraining efficiency under the policy, ``m / rho``, for each recipe: ``ceiling`` with the proven chain bound
    (a ceiling when the recipe's chain is incompressible at its own rates) and ``attack`` with the construction's cost.
    The best recipe is the attacker's choice."""
    ests = catalog.estimates(parameters)
    out = []
    for name, b_w, b_a, est in RECIPES:
        m = {"low": 1.0, "central": 1.0, "high": 1.0} if est is None else {k: ests[est][k] for k in catalog.PICKS}
        cell = sweep(row, calibration, parameters, [b_w], [b_a])[0]
        out.append({"recipe": name, "b_w": b_w, "b_a": b_a, "m": m, "chain": cell["chain"], "upper": cell["upper"],
                    "ceiling": m["central"] / cell["chain"] if cell["chain"] else None,
                    "attack": m["central"] / cell["upper"]})
    return out


def brackets(rows: list[dict], calibrations: list[dict], parameters: list[dict]) -> list[dict]:
    """Every row under every calibration of its model."""
    return [bracket(r, c, parameters) for c in calibrations for r in rows if r["model"] == c["model"]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibration", help="a calibrations.json id (default: every calibration)")
    ap.add_argument("--row", action="append", help="a rows.json id (repeatable; default: every row)")
    ap.add_argument("--sweep", action="store_true", help="the proven bounds over a grid of b_w and b_a, per row")
    ap.add_argument("--efficiency", action="store_true", help="m / rho for each training recipe, per row")
    args = ap.parse_args(argv)
    parameters, rows = catalog.load("parameters"), catalog.load("rows")
    cals = [c for c in catalog.load("calibrations") if args.calibration in (None, c["id"])]
    rows = [r for r in rows if not args.row or r["id"] in args.row]
    if args.efficiency:
        for c in cals:
            for r in rows:
                if r["model"] != c["model"] or "chain" not in r:
                    continue
                print(f"{r['id']} under {c['id']}: efficiency m / rho")
                print(f"{'recipe':16} {'b_w':>5} {'b_a':>5} {'m (low-high)':>18} {'chain':>9} {'upper':>9} "
                      f"{'ceiling':>9} {'attack':>9}")
                for x in efficiency(r, c, parameters):
                    m = x["m"]
                    print(f"{x['recipe']:16} {x['b_w']:5.3g} {x['b_a']:5.3g} "
                          f"{m['central']:6.2f} ({m['low']:.2f}-{m['high']:.2f}) {x['chain']:9.2g} {x['upper']:9.2g} "
                          f"{x['ceiling']:9.2g} {x['attack']:9.2g}")
        return 0
    if args.sweep:
        b_ws, b_as = [16, 8, 4, 2.5, 1.58, 1, 0.5], [16, 8, 4, 2, 1, 0.5]
        for c in cals:
            for r in rows:
                if r["model"] != c["model"]:
                    continue
                print(f"{r['id']} under {c['id']}: thread / chain (proven) / construction")
                print(f"{'b_w \\ b_a':>10}" + "".join(f"{a:>25}" for a in b_as))
                cells = sweep(r, c, parameters, b_ws, b_as)
                for b_w in b_ws:
                    print(f"{b_w:>10}" + "".join(
                        f"{x['thread']:>9.2g}/{x['chain'] if x['chain'] is not None else float('nan'):<7.2g}/"
                        f"{x['upper']:<7.2g}" for x in cells if x["b_w"] == b_w))
        return 0
    print(f"{'row':44} {'calibration':22} {'thread':>9} {'chain':>9} {'upper':>9} {'basis':>14}")
    for b in brackets(rows, cals, parameters):
        chain = f"{b['chain']:9.3g}" if b["chain"] is not None else f"{'-':>9}"
        print(f"{b['row']:44} {b['calibration']:22} {b['lower']:9.3g} {chain} {b['upper']:9.3g} {b['upper_basis']:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
