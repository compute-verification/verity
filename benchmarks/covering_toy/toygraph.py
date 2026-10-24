"""Word-level computation graphs of a few SGD steps of a tiny MLP.

Every multiply, add, subtract, ReLU and ReLU-derivative is one gate, and every gate is one word. The step-1 weights and
the learning rate are registered (`theta`); each step's data (inputs `x`, targets `y`) are unregistered sources; every
updated weight is a computed gate, so a block that needs it either reads it (one unregistered word) or recomputes it.

The recipe is plain SGD on the squared error `1/2 |y_hat - y|^2`, summed over the batch:

    a_0 = x;  z_l = W_l a_{l-1};  a_l = relu(z_l) (l < L);  y_hat = z_L
    delta_L = y_hat - y;  delta_{l-1} = (W_l^T delta_l) * [z_{l-1} > 0]
    W_l <- W_l - lr * sum_b delta_l a_{l-1}^T

Dot products are left-folded add chains, and the batch sum is a left-folded chain over the batch. `reference_sgd` is the
same recipe in numpy; `Graph.evaluate` must agree with it (the tests check it).

The graph carries what the covering bound reads (`Security/Models/Nci/Assumptions/CoveringTwoCosts.lean`):

- `mul`: the multiplication blocks `m_{i,j}`, one gate each: the forward product of updated weight slot `i = (l, r, c)`
  with token `j = (t, b)`'s activation, for every step `t >= 1` (0-based, so every step after the first). Step 0's
  forward products are by registered weights and are not in `P`.
- `token_gates[j]`: the token computation `T_j`, the gates of token `j`'s own forward and backward pass (its data, its
  products, sums, activations, deltas and per-example gradient products), disjoint across tokens.
- `values`: the finding note's counted values (note:console/20261024T1810Z-finding-production-bound-red-team, section
  4): outputs of parameterized maps, here the forward pre-activations `z_l` of every token at every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SOURCE_OPS = ("theta", "lr", "data")


@dataclass
class Graph:
    """A DAG of word gates in topological order (a gate's predecessors have smaller indices)."""

    op: list[str] = field(default_factory=list)
    preds: list[tuple[int, ...]] = field(default_factory=list)
    theta: list[bool] = field(default_factory=list)
    tag: list[dict] = field(default_factory=list)
    mul: dict[tuple, int] = field(default_factory=dict)
    values: list[int] = field(default_factory=list)
    token_gates: dict[tuple, list[int]] = field(default_factory=dict)
    outputs: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, op: str, preds: tuple[int, ...] = (), *, theta: bool = False, **tag) -> int:
        assert all(p < len(self.op) for p in preds)
        assert (op in SOURCE_OPS) == (not preds)
        g = len(self.op)
        self.op.append(op)
        self.preds.append(tuple(preds))
        self.theta.append(theta)
        self.tag.append(tag)
        tok = tag.get("token")
        if tok is not None:
            self.token_gates.setdefault(tok, []).append(g)
        return g

    @property
    def N(self) -> int:
        return len(self.op)

    def is_source(self, g: int) -> bool:
        return not self.preds[g]

    def inner(self) -> list[int]:
        """`C \\ in(C)`: the gates a covering must compute."""
        return [g for g in range(self.N) if self.preds[g]]

    def succs(self) -> list[list[int]]:
        out: list[list[int]] = [[] for _ in range(self.N)]
        for g, ps in enumerate(self.preds):
            for p in set(ps):
                out[p].append(g)
        return out

    def evaluate(self, inputs: dict[int, float]) -> np.ndarray:
        v = np.zeros(self.N)
        for g in range(self.N):
            o, p = self.op[g], self.preds[g]
            if not p:
                v[g] = inputs[g]
            elif o == "mul":
                v[g] = v[p[0]] * v[p[1]]
            elif o == "add":
                v[g] = v[p[0]] + v[p[1]]
            elif o == "sub":
                v[g] = v[p[0]] - v[p[1]]
            elif o == "relu":
                v[g] = max(v[p[0]], 0.0)
            elif o == "drelu":
                v[g] = v[p[0]] if v[p[1]] > 0 else 0.0
            else:
                raise ValueError(o)
        return v

    def summary(self) -> dict:
        n_theta = sum(self.theta)
        n_data = sum(1 for g in range(self.N) if self.op[g] == "data")
        return {
            **self.meta,
            "gates": self.N,
            "inner": self.N - n_theta - n_data,
            "theta": n_theta,
            "data": n_data,
            "mul_blocks": len(self.mul),
            "values": len(self.values),
        }


def _dot(G: Graph, ws: list[int], xs: list[int], *, mul_tags: list[dict], **tag) -> int:
    acc = None
    for w, x, mt in zip(ws, xs, mul_tags):
        m = G.add("mul", (w, x), **tag, **mt)
        acc = m if acc is None else G.add("add", (acc, m), **tag)
    return acc


def mlp(dims: tuple[int, ...] = (2, 2, 2), batch: int = 1, steps: int = 2) -> Graph:
    """`steps` SGD steps of an MLP with layer widths `dims` (input, hidden..., output) on `batch` tokens a step.

    Each step reads a fresh batch. Weight slot `(l, r, c)` is row `r`, column `c` of layer `l` (0-based).
    """
    G = Graph()
    L = len(dims) - 1
    lr = G.add("lr", theta=True, kind="lr")
    W = [
        [[G.add("theta", theta=True, kind="w", slot=(l, r, c), step=0) for c in range(dims[l])] for r in range(dims[l + 1])]
        for l in range(L)
    ]
    G.meta = {
        "model": "mlp",
        "dims": list(dims),
        "batch": batch,
        "steps": steps,
        "state_words": sum(dims[l] * dims[l + 1] for l in range(L)),
        "data_words_per_token": dims[0] + dims[-1],
        "data_words_per_step": batch * (dims[0] + dims[-1]),
    }
    for t in range(steps):
        grads = [[[[] for _ in range(dims[l])] for _ in range(dims[l + 1])] for l in range(L)]
        for b in range(batch):
            j = (t, b)
            tg = dict(step=t, token=j)
            x = [G.add("data", kind="x", i=i, **tg) for i in range(dims[0])]
            y = [G.add("data", kind="y", i=i, **tg) for i in range(dims[-1])]
            acts, pres = [x], []
            for l in range(L):
                zs = []
                for r in range(dims[l + 1]):
                    mt = [dict(slot=(l, r, c)) for c in range(dims[l])]
                    z = _dot(G, W[l][r], acts[-1], mul_tags=mt, phase="fwd", layer=l, **tg)
                    G.tag[z]["value"] = (l, r)
                    G.values.append(z)
                    zs.append(z)
                pres.append(zs)
                acts.append([G.add("relu", (z,), phase="fwd", layer=l, act=True, **tg) for z in zs] if l < L - 1 else zs)
            delta = [G.add("sub", (yh, yy), phase="bwd", layer=L - 1, **tg) for yh, yy in zip(acts[-1], y)]
            for l in reversed(range(L)):
                for r in range(dims[l + 1]):
                    for c in range(dims[l]):
                        grads[l][r][c].append(G.add("mul", (delta[r], acts[l][c]), phase="grad", layer=l, **tg))
                if l:
                    back = []
                    for c in range(dims[l]):
                        col = [W[l][r][c] for r in range(dims[l + 1])]
                        mt = [dict(slot=(l, r, c), bwd=True) for r in range(dims[l + 1])]
                        gsum = _dot(G, col, delta, mul_tags=mt, phase="bwd", layer=l, **tg)
                        back.append(G.add("drelu", (gsum, pres[l - 1][c]), phase="bwd", layer=l - 1, **tg))
                    delta = back
        newW = []
        for l in range(L):
            rows = []
            for r in range(dims[l + 1]):
                row = []
                for c in range(dims[l]):
                    acc = grads[l][r][c][0]
                    for p in grads[l][r][c][1:]:
                        acc = G.add("add", (acc, p), phase="acc", step=t, layer=l)
                    sc = G.add("mul", (lr, acc), phase="upd", step=t, layer=l)
                    row.append(G.add("sub", (W[l][r][c], sc), phase="upd", step=t, layer=l, kind="w", slot=(l, r, c)))
                rows.append(row)
            newW.append(rows)
        W = newW
    G.outputs = [w for layer in W for row in layer for w in row]
    _index_mul_blocks(G)
    return G


def inference(dims: tuple[int, ...] = (2, 2, 2), tokens: int = 4) -> Graph:
    """Honest inference with the same model: the registered weights' forward pass on `tokens` independent tokens, each
    reading its `x`. Its covering at `(s, k)` is the baseline training's slowdown is measured against."""
    G = Graph()
    L = len(dims) - 1
    W = [[[G.add("theta", theta=True, kind="w", slot=(l, r, c), step=0) for c in range(dims[l])] for r in range(dims[l + 1])]
         for l in range(L)]
    G.meta = {"model": "mlp-inference", "dims": list(dims), "batch": tokens, "steps": 1,
              "state_words": sum(dims[l] * dims[l + 1] for l in range(L)), "data_words_per_token": dims[0]}
    for b in range(tokens):
        tg = dict(step=0, token=(0, b))
        a = [G.add("data", kind="x", i=i, **tg) for i in range(dims[0])]
        for l in range(L):
            zs = []
            for r in range(dims[l + 1]):
                z = _dot(G, W[l][r], a, mul_tags=[dict(slot=(l, r, c)) for c in range(dims[l])], phase="fwd", layer=l, **tg)
                G.tag[z]["value"] = (l, r)
                G.values.append(z)
                zs.append(z)
            a = [G.add("relu", (z,), phase="fwd", layer=l, act=True, **tg) for z in zs] if l < L - 1 else zs
        G.outputs += a
    return G


def _index_mul_blocks(G: Graph) -> None:
    """`P`: forward products of updated weights (every step after the first), keyed `(slot, token)`."""
    for g in range(G.N):
        tg = G.tag[g]
        if G.op[g] == "mul" and tg.get("phase") == "fwd" and tg["step"] >= 1:
            key = (tg["slot"], tg["token"])
            assert key not in G.mul
            G.mul[key] = g


def random_inputs(G: Graph, seed: int = 0, lr: float = 0.1) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    vals = {}
    for g in range(G.N):
        if G.op[g] == "lr":
            vals[g] = lr
        elif G.op[g] in ("theta", "data"):
            vals[g] = float(rng.normal())
    return vals


def reference_sgd(G: Graph, inputs: dict[int, float]) -> list[np.ndarray]:
    """The recipe in numpy on `G`'s own sources; returns the final weights by layer."""
    dims, batch, steps = G.meta["dims"], G.meta["batch"], G.meta["steps"]
    L = len(dims) - 1
    lr = next(inputs[g] for g in range(G.N) if G.op[g] == "lr")
    W = [np.zeros((dims[l + 1], dims[l])) for l in range(L)]
    X = np.zeros((steps, batch, dims[0]))
    Y = np.zeros((steps, batch, dims[-1]))
    for g in range(G.N):
        tg = G.tag[g]
        if G.op[g] == "theta":
            l, r, c = tg["slot"]
            W[l][r, c] = inputs[g]
        elif G.op[g] == "data":
            t, b = tg["token"]
            (X if tg["kind"] == "x" else Y)[t, b, tg["i"]] = inputs[g]
    for t in range(steps):
        gr = [np.zeros_like(w) for w in W]
        for b in range(batch):
            acts, pres = [X[t, b]], []
            for l in range(L):
                z = W[l] @ acts[-1]
                pres.append(z)
                acts.append(np.maximum(z, 0) if l < L - 1 else z)
            delta = acts[-1] - Y[t, b]
            for l in reversed(range(L)):
                gr[l] += np.outer(delta, acts[l])
                if l:
                    delta = (W[l].T @ delta) * (pres[l - 1] > 0)
        W = [w - lr * g for w, g in zip(W, gr)]
    return W
