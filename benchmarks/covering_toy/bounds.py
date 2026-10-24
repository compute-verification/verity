"""What the bounds predict for a `toygraph.Graph`'s minimum `(s, k)`-covering, in words (one gate is one word).

- `lean_two_costs`: `Nci.Guarantees.CoveringBoundTwoCosts` (`Security/Models/Nci/Guarantees/CoveringTwoCosts.lean`,
  proved in `Security/Proofs/Nci/CoveringTwoCosts.lean`): if the graph is `(s, c_w)`-parameter-hard and `c_w D > k`,
  `n >= |M*| / ((k/c_w)(k/c_a) + k)`, `M*` the multiplication blocks chained at `(D, c_a)`. At word level an activation
  is one gate, so it is its own cut and `c_a = 1` exactly; a block is `(D, 1)`-chained iff at least `D` weight slots'
  products at its token reach the activation it reads inside the token's computation (`upstream`). The bound is
  evaluated at the smallest admissible `D`, `floor(k / c_w) + 1`, which maximizes `|M*|`.
- `production`: the finding note's section-4 theorem (note:console/20261024T1810Z-finding-production-bound-red-team),
  `n >= w(V) / k` with `w(V) = beta |V|`, `V` the forward pre-activations (outputs of parameterized maps; the toy's
  maps are square, so `min(d_in, d_out)` per token per map is every output).
- `capacity`: the valid counting bound `n >= |T| / cap(T)`, `cap(T)` the most gates of `T` one `(s, k)`-block computes
  (`cover.block_capacity`, CP-SAT's proven upper bound). With `T` the counted values it is the production bound at the
  instance's measured `beta = k / cap(V)`; with `T = M*` it is the Lean bound's counting step with the true per-block
  cap in place of `(k/c_w)(k/c_a) + k`.
- `trivial`: `n >= |C \\ in(C)| / s`.
"""

from __future__ import annotations

import math

from toygraph import Graph


def upstream(G: Graph) -> dict[tuple, int]:
    """For each multiplication block `(slot, token)`: how many distinct weight slots' products at the same token reach
    the activation it reads, inside the token's computation `T_j`."""
    out = {}
    for tok, gates in G.token_gates.items():
        inT = set(gates)
        mul_here = {g: key[0] for key, g in G.mul.items() if key[1] == tok}
        slots: dict[int, frozenset] = {}
        for g in sorted(gates):
            acc = set()
            for p in G.preds[g]:
                if p in inT:
                    acc |= slots.get(p, frozenset())
            if g in mul_here:
                acc.add(mul_here[g])
            slots[g] = frozenset(acc)
        for g, slot in mul_here.items():
            a = G.preds[g][1]
            out[(slot, tok)] = len(slots.get(a, frozenset())) if a in inT else 0
    return out


def chained(G: Graph, D: int) -> list[int]:
    """`M*` at `(D, c_a = 1)`: the gates of the multiplication blocks with at least `D` upstream slots (D >= 1)."""
    up = upstream(G)
    return [G.mul[key] for key, u in up.items() if u >= max(D, 1)]


def lean_two_costs(G: Graph, k: int, cw: float, ca: float = 1.0) -> dict:
    D = math.floor(k / cw) + 1
    m = len(chained(G, D))
    per_block = (k / cw) * (k / ca) + k
    return {"D": D, "M_star": m, "per_block": per_block, "n_lb": m / per_block if per_block else math.inf}


def production(G: Graph, k: int, beta: float = 1.0, values: list[int] | None = None) -> dict:
    V = G.values if values is None else values
    return {"V": len(V), "beta": beta, "n_lb": beta * len(V) / k}


def trivial(G: Graph, s: int | None) -> float:
    return 0.0 if s is None else len(G.inner()) / s


def values_after_first_step(G: Graph) -> list[int]:
    return [g for g in G.values if G.tag[g]["step"] >= 1]


def honest_block_size(G: Graph, k: int) -> int:
    """`s` matched to traffic (eta = 1): an inference block that reads `k` words of token inputs runs the registered
    model's forward pass on `k / d_in` tokens, and reads the weights (which count toward `s`)."""
    dims = G.meta["dims"]
    fwd = sum(2 * dims[l] * dims[l + 1] - dims[l + 1] for l in range(len(dims) - 1)) + sum(dims[1:-1])
    return G.meta["state_words"] + math.ceil(k / dims[0] * (dims[0] + fwd))
