"""A library of training algorithms and what the inference-only policy costs each, with where every block's reads go.

An algorithm (``algorithms.json``) carries its measured scaling (``scaling``: compute-equivalent multipliers over a
dense AdamW Chinchilla-optimal reference, each citing findings, and ``multiplier``, the one at the largest compute) and
its ``convention`` (see ``CONVENTIONS``), and is described by architecture-agnostic traits: the fraction of parameters it
updates, the bits of state a reader needs per updated parameter and whether that state splits across slices of the
model, the width and precision of what crosses a layer boundary and how often per token, the work per token, the
novel data a token brings, and any decoding work. A model (``models.json``) is a shape. A policy is a calibration
(``calibrations.json``), queried with the training data either committed (free to read) or novel (``b_tok`` bits per
token).

``construct`` is ``slowdown.search``: one family of compliant constructions, in which a block covers a slice of every
token's computation, reads its slice's state, the activations crossing into it and the tokens' data, and is padded to
``Gamma``. The board (``rows.json``) and the library use the same search.

``lower`` is the thread theorem (``slowdown.thread_lower``) on the same profile. It is 1 when the algorithm is not one
the theorem covers, and 1 when the data is committed, since its premise ``BaselineRegistered`` then fails (the coverage
gap).

    python benchmarks/nci/library.py table --model kimi-k2 --policy kimi-k2-1m-raw
    python benchmarks/nci/library.py data --policy kimi-k2-1m-raw       # how big a deal is training data, per model
    python benchmarks/nci/library.py query backprop --model dense-7b --data novel
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import catalog  # noqa: E402
import slowdown  # noqa: E402

TRAITS = {
    "theorem_applies": "the thread theorem covers it (end-to-end training of the reference circuit)",
    "updated_fraction": "fraction of the model's parameters updated",
    "state_bits_per_param": "bits of state a reader needs per updated parameter",
    "state_splits": "a slice of the model needs only its own share of the state (false for seeds)",
    "description_bits": "total description of the update when it does not split (seeds and scalars), else null",
    "boundary_width_fraction": "width crossing a layer boundary, as a fraction of d_model",
    "boundary_bits": "bits per entry crossing a boundary",
    "crossings": "boundary crossings per token (1 forward only, 2 with a backward pass)",
    "flops_per_token_per_param": "work per token per active parameter",
    "data_tokens_per_token": "novel data tokens per token of work when the data is not committed",
    "decode_flops_per_param": "work to decode the whole update, per parameter of the model",
    "kappa": "share of the work in multiplications of an updated weight by an activation",
    "uses_per_thread": "uses of each updated weight per token (the theorem's c)",
    "entry_fee_override": "the theorem's d if not the model's entry fee (for example an adapter's rank)",
    "fanout_override": "the theorem's e if not the model's fan-out",
}
MODEL_FIELDS = ("n_total", "n_active", "d_model", "entry_fee", "fanout")
FAMILIES = frozenset({"reference", "architecture", "optimizer", "data", "objective", "parallel-communication",
                      "local-learning", "fine-tuning", "low-rank", "rl-adapter", "seed-based", "zeroth-order",
                      "distillation", "sparsity-moe", "precision", "growth-reuse", "retrieval"})
#: ``attacker``: state at the cheapest rate an attacker could use (weights at the retained-quality rate, no optimizer
#: state); ``as-published``: state at the precision the method is published with (weights and optimizer moments).
CONVENTIONS = frozenset({"attacker", "as-published"})
#: the full-parameter recipe a strategy must beat: ``multiplier_needed`` is the multiplier at which the strategy's
#: construction is as efficient as this one's
REFERENCE = "backprop"


def load_json(name: str) -> list[dict]:
    return json.loads((HERE / f"{name}.json").read_text())


def by_id(name: str) -> dict[str, dict]:
    return {x["id"]: x for x in load_json(name)}


@dataclass(frozen=True)
class Profile:
    """An algorithm on a model under a policy: what one token costs and what a reader must hold."""
    U_slice_total: float   # state a slice covering the whole token's computation reads
    splits: bool
    flops: float           # reference work per token
    w: float               # bits crossing into a slice per token (all crossings)
    t: float               # novel data bits per token
    decode: float          # work to decode the whole update


ATTACKER_RATES = {"state_bits_per_param": "b_w.pretrained/frontier-lossy:central",
                  "boundary_bits": "b_a.activation/boundary-training:central"}


def traits(alg: dict, convention: str = "attacker") -> dict:
    """An algorithm's traits under a convention: ``as-published`` keeps them; ``attacker`` caps its state and boundary
    bits at the literature's central rates (``ATTACKER_RATES``), the cheapest an attacker could use at about the same
    quality, so that every algorithm is compared alike."""
    tr = dict(alg["traits"])
    if convention == "attacker":
        params = catalog.load("parameters")
        for key, ref in ATTACKER_RATES.items():
            tr[key] = min(tr[key], catalog.resolve(ref, params))
    return tr


def profile(alg: dict, model: dict, policy: dict, data: str, convention: str = "attacker") -> Profile:
    tr = traits(alg, convention)
    if tr["state_splits"]:
        U = tr["state_bits_per_param"] * model["n_active"] * tr["updated_fraction"]
    else:
        U = tr["description_bits"]
    w = tr["boundary_width_fraction"] * model["d_model"] * tr["boundary_bits"] * tr["crossings"]
    t = tr["data_tokens_per_token"] * policy["b_tok"] if data == "novel" else 0.0
    return Profile(U_slice_total=U, splits=tr["state_splits"], flops=tr["flops_per_token_per_param"] * model["n_active"],
                   w=w, t=t, decode=tr["decode_flops_per_param"] * model["n_total"])


def construct(p: Profile, K: float, Gamma: float) -> dict:
    """The cheapest construction in ``slowdown.search``'s family, with its reads broken down per block."""
    return slowdown.search(p.U_slice_total, p.splits, p.flops, p.w, p.t, p.decode, K, Gamma)


def _theorem_inputs(alg: dict, model: dict, p: Profile, convention: str) -> dict:
    tr = traits(alg, convention)
    n_updated = tr["updated_fraction"] * model["n_total"]
    U_bits = tr["description_bits"] if not tr["state_splits"] else tr["state_bits_per_param"] * n_updated
    return {"kappa": tr["kappa"], "b_w": tr["state_bits_per_param"], "b_a": tr["boundary_bits"],
            "d": tr["entry_fee_override"] or model["entry_fee"], "e": tr["fanout_override"] or model["fanout"],
            "c": tr["uses_per_thread"], "n_updated": n_updated, "U_bits": U_bits, "U_token_bits": p.U_slice_total,
            "decode_flops": p.decode, "w_cut": p.w / tr["crossings"], "crossings": tr["crossings"],
            "flops_per_token": p.flops}


def lower(alg: dict, model: dict, policy: dict, data: str, convention: str = "attacker") -> tuple[float, str]:
    if not alg["traits"]["theorem_applies"]:
        return 1.0, "the thread theorem does not cover this algorithm"
    if data == "committed":
        return 1.0, "data committed: the theorem's containment hypothesis fails (the coverage gap)"
    p = profile(alg, model, policy, data, convention)
    # A block that holds a token's whole state generates the tokens it does not read (rollouts), so a thread value
    # costs it no more than the novel data per token.
    token_rate = p.t if p.U_slice_total <= policy["K"] else policy["b_tok"]
    return slowdown.thread_lower(_theorem_inputs(alg, model, p, convention), policy["K"], policy["Gamma"], token_rate), \
        slowdown.THREAD_THEOREM


def query(alg_id: str, model_id: str, policy_id: str, data: str = "novel", convention: str = "attacker") -> dict:
    """The bracket and the read breakdown for one algorithm on one model under one policy."""
    alg, model, policy = by_id("algorithms")[alg_id], by_id("models")[model_id], by_id("calibrations")[policy_id]
    p = profile(alg, model, policy, data, convention)
    c = construct(p, policy["K"], policy["Gamma"])
    lo, why = lower(alg, model, policy, data, convention)
    total = sum(c["reads"].values())
    m = (alg.get("multiplier") or {}).get("value")
    ref = construct(profile(by_id("algorithms")[REFERENCE], model, policy, data, convention), policy["K"], policy["Gamma"])
    return {"algorithm": alg_id, "model": model_id, "policy": policy_id, "data": data, "convention": convention,
            "multiplier": m, "multiplier_needed": c["rho"] / ref["rho"],
            "efficiency_achievable": m / c["rho"] if m else None,
            "efficiency_ceiling": m / lo if m else None, "lower": lo, "lower_basis": why,
            "upper": c["rho"], "slices": 1 / c["phi"], "tokens_per_block": c["tokens"],
            "read_shares": {k: v / total for k, v in c["reads"].items()}, "reads": c["reads"], "work": c["work"]}


def data_importance(alg_id: str, model_id: str, policy_id: str, convention: str = "attacker") -> dict:
    """How big a deal training data is: the cost with data novel over the cost with it committed, and where a block's
    reads go when it is novel."""
    novel = query(alg_id, model_id, policy_id, "novel", convention)
    committed = query(alg_id, model_id, policy_id, "committed", convention)
    return {"algorithm": alg_id, "model": model_id, "policy": policy_id, "factor": novel["upper"] / committed["upper"],
            "upper_committed": committed["upper"], "upper_novel": novel["upper"], "read_shares": novel["read_shares"]}


def problems() -> list[str]:
    out = []
    srcs = catalog.load("sources")
    sources = {s["id"] for s in srcs}
    findings = catalog.findings(srcs)
    for m in load_json("models"):
        for f in MODEL_FIELDS:
            if not isinstance(m.get(f), (int, float)) or m[f] <= 0:
                out.append(f"model {m.get('id')}: {f} must be a positive number")
    seen = set()
    for a in load_json("algorithms"):
        aid = a.get("id", "?")
        if aid in seen or not catalog.KEBAB.match(aid):
            out.append(f"algorithm {aid}: id must be unique kebab-case")
        seen.add(aid)
        if a.get("family") not in FAMILIES:
            out.append(f"algorithm {aid}: family {a.get('family')}")
        for key in ("name", "description", "notes"):
            if not a.get(key):
                out.append(f"algorithm {aid}: missing {key}")
        if set(a.get("traits", {})) != set(TRAITS):
            out.append(f"algorithm {aid}: traits must be exactly {sorted(TRAITS)}")
            continue
        t = a["traits"]
        if not t["state_splits"] and not t["description_bits"]:
            out.append(f"algorithm {aid}: a state that does not split needs description_bits")
        for s in a.get("sources", []):
            if s not in sources:
                out.append(f"algorithm {aid}: source {s} is not in sources/")
        if a.get("convention") not in CONVENTIONS:
            out.append(f"algorithm {aid}: convention must be one of {sorted(CONVENTIONS)}")
        for e in a.get("scaling", []):
            for ref in e.get("evidence", []):
                if ref not in findings:
                    out.append(f"algorithm {aid}: scaling evidence {ref} is no finding")
        m = a.get("multiplier")
        if m is not None and not isinstance(m.get("value"), (int, float)):
            out.append(f"algorithm {aid}: multiplier needs a value")
    return out


def _fmt(x: float) -> str:
    return f"{x:.3g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table", help="every algorithm on one model under one policy")
    t.add_argument("--model", default="kimi-k2")
    t.add_argument("--policy", default="kimi-k2-1m-raw")
    t.add_argument("--data", choices=("novel", "committed"), default="novel")
    t.add_argument("--convention", choices=sorted(CONVENTIONS), default="attacker")
    d = sub.add_parser("data", help="how big a deal training data is: every algorithm on every model")
    d.add_argument("--policy", default="kimi-k2-1m-raw")
    d.add_argument("--algorithm", action="append")
    d.add_argument("--convention", choices=sorted(CONVENTIONS), default="attacker")
    q = sub.add_parser("query", help="one algorithm on one model under one policy, as JSON")
    q.add_argument("algorithm")
    q.add_argument("--model", default="kimi-k2")
    q.add_argument("--policy", default="kimi-k2-1m-raw")
    q.add_argument("--data", choices=("novel", "committed"), default="novel")
    q.add_argument("--convention", choices=sorted(CONVENTIONS), default="attacker")
    args = ap.parse_args(argv)
    if args.cmd == "query":
        print(json.dumps(query(args.algorithm, args.model, args.policy, args.data, args.convention), indent=1))
    elif args.cmd == "table":
        print("m: measured multiplier over the reference recipe; m need: the multiplier at which the algorithm's "
              "construction matches full-parameter backprop's; efficiency = m / slowdown (achievable at the "
              "construction, ceiling at the lower bound)")
        print(f"{'algorithm':32} {'conv':>4} {'m':>6} {'m need':>8} {'lower':>8} {'upper':>9} {'eff. achv':>10} "
              f"{'eff. ceil':>10}  reads: state / activations / data")
        for a in load_json("algorithms"):
            try:
                r = query(a["id"], args.model, args.policy, args.data, args.convention)
            except ValueError:
                print(f"{a['id']:32} no compliant construction in the family")
                continue
            s = r["read_shares"]
            m = r["multiplier"]
            eff = (lambda x: _fmt(x) if x else "-")
            print(f"{a['id']:32} {a['convention'][:4]:>4} {(_fmt(m) if m else '-'):>6} {_fmt(r['multiplier_needed']):>8} "
                  f"{_fmt(r['lower']):>8} "
                  f"{_fmt(r['upper']):>9} {eff(r['efficiency_achievable']):>10} {eff(r['efficiency_ceiling']):>10}  "
                  f"{s['state']:6.1%} / {s['activations']:6.1%} / {s['data']:6.1%}")
    else:
        algs = args.algorithm or [a["id"] for a in load_json("algorithms")]
        models = [m["id"] for m in load_json("models")]
        print("upper bound with data novel / with data committed (the factor training data is worth to the policy)")
        print(f"{'algorithm':26} " + " ".join(f"{m:>11}" for m in models))
        for a in algs:
            cells = []
            for m in models:
                try:
                    cells.append(_fmt(data_importance(a, m, args.policy, args.convention)["factor"]))
                except ValueError:
                    cells.append("n/a")
            print(f"{a:26} " + " ".join(f"{c:>11}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
