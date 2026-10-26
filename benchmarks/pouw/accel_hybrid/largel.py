"""HYB at large L for v1 (no fold): noise period exactly L, U's columns permuted fresh per segment.

Segment s adds V[:, l] to activation column sL + l and U[:, pi_s(l)] to weight column sL + l. The served product is
recovered exactly from the last word, the sparse wrap adds and three rank-L peels:
O = T_last + 256(wX W'^T + X wW^T) + 65536 wX wW^T - Y_X U^T - V Y_W^T - V (sum_s P_s) U^T,
with Y_X gathered through pi_s and (sum_s P_s)[l, j] the number of segments with pi_s(l) = j.
"""

from __future__ import annotations

import numpy as np

from hyb import M32, s32, stream, uniform, wrap8


def coins(seed: bytes, m: int, n: int, k: int, L: int, u: int):
    S = k // L
    perms = np.stack([np.argsort(stream(seed, f"pi{s}", L)) for s in range(S)])
    return dict(b=stream(seed, "b", m * k).reshape(m, k) & 1, c=stream(seed, "c", n * k).reshape(n, k) & 1,
                V=uniform(seed, "V", (m, L), u), U=uniform(seed, "U", (n, L), u), pi=perms)


def run(A, B, seed, L, u):
    m, k = A.shape
    n, S = B.shape[0], k // L
    cn = coins(seed, m, n, k, L, u)
    Vr = np.tile(cn["V"], (1, S))
    Up = np.concatenate([cn["U"][:, cn["pi"][s]] for s in range(S)], axis=1)
    Xs, Ws = A + cn["b"] + Vr, B + cn["c"] + Up
    X, W = wrap8(Xs), wrap8(Ws)
    wX, wW = (Xs - X) // 256, (Ws - W) // 256
    T_last = sum(X[:, s * L:(s + 1) * L] @ W[:, s * L:(s + 1) * L].T for s in range(S)) & M32
    AB = A + cn["b"]
    YX = np.zeros((m, L), dtype=np.int64)
    P = np.zeros((L, L), dtype=np.int64)
    for s in range(S):
        np.add.at(YX.T, cn["pi"][s], AB[:, s * L:(s + 1) * L].T)
        P[np.arange(L), cn["pi"][s]] += 1
    YW = (B + cn["c"]).reshape(n, S, L).sum(1)
    O = s32(T_last + 256 * (wX @ W.T + X @ wW.T) + 65536 * (wX @ wW.T)
            - YX @ cn["U"].T - cn["V"] @ YW.T - cn["V"] @ P @ cn["U"].T)
    want = s32(AB @ (B + cn["c"]).T)
    return bool(np.array_equal(O, want)), float(wX.astype(bool).mean()), float(wW.astype(bool).mean())


def main() -> None:
    rng = np.random.default_rng(1)
    m, n, k, L = 24, 20, 2048, 256
    for name, A, B in [("uniform", rng.integers(-128, 127, (m, k)), rng.integers(-128, 127, (n, k))),
                       ("zero", np.zeros((m, k), np.int64), np.zeros((n, k), np.int64)),
                       ("A=0, B periodic", np.zeros((m, k), np.int64), np.tile(rng.integers(-64, 64, (n, L)), (1, k // L)))]:
        for u in (80, 127):
            ok, pX, pW = run(A.astype(np.int64), B.astype(np.int64), b"largel" + name.encode(), L, u)
            print(f"{name:16s} u=+-{u}: exact={ok} wrap density X {pX:.4f} W {pW:.4f}")


if __name__ == "__main__":
    main()
